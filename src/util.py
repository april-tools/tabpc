import _thread
import fnmatch
import json
import math
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import GPUtil
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import Tensor
from torchmetrics.functional.clustering import mutual_info_score

INT_IS_NUMERICAL_THRESHOLD = 20


def load_json(file: str | Path) -> dict:
    with open(file) as json_file:
        d = json.load(json_file)
    return d


def store_json(d: dict, *, file: str | Path):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


class edit_json:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.data = load_json(self.filename) if os.path.exists(self.filename) else {}
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        store_json(self.data, file=self.filename)


def file_name(file: str | Path) -> str:
    return str(file).split("/")[-1]


def create_experiment_folder(*, path: Path, postfix: str | None = None) -> Path:
    postfix = f"_{postfix}" if postfix else ""
    folder_name = Path(datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f") + postfix)
    experiment_folder = path / folder_name
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def input_listener():
    def input_thread(a_list):
        try:
            input()
            a_list.append(True)
        except EOFError:
            print("EOFError: No input available.")
            a_list.append(False)

    a_list = []
    _thread.start_new_thread(input_thread, (a_list,))
    return a_list


def pickle_load(file_name: str):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def pickle_store(object, *, file: str):
    with open(file, "wb") as f:
        return pickle.dump(object, f)


def get_available_device(
    mem_required: float = 0.05, verbose: bool = False, stop_if_no_free_gpu: bool = True
):
    if not torch.cuda.is_available():
        return "cpu"

    try:
        devices = GPUtil.getGPUs()

        # Get the results of CUDA_VISIBLE_DEVICES environment variable to make sure
        # we don't try to access a GPU that is not visible

        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            visible_device_ids = [
                int(x) for x in visible_devices.split(",") if x.strip().isdigit()
            ]
            visible_device_ids = sorted(visible_device_ids)

        devices = [
            device
            for device in devices
            if visible_devices is None or device.id in visible_device_ids
        ]

        # Map from global device id to local id if CUDA_VISIBLE_DEVICES is set
        visible_mapping = {vis_idx : local_idx for local_idx, vis_idx in enumerate(visible_device_ids)} if visible_devices is not None else {}

        device_usages = [
            (device.id, device.memoryUsed / device.memoryTotal) for device in devices
        ]

        device_usages.sort(key=lambda x: x[1])

        if device_usages[0][1] > 1 - mem_required:
            if stop_if_no_free_gpu:
                raise RuntimeError("No GPU with sufficient free memory is available.")
            return "cpu"
        
        # Return the correct cuda device string considering CUDA_VISIBLE_DEVICES
        if visible_devices is not None:
            out = "cuda:" + str(visible_mapping[device_usages[0][0]])
        else:
            out = "cuda:" + str(device_usages[0][0])
        
        if verbose:
            print("\033[92m" + f"Using {out}" + "\033[0m")
        return out
    except ValueError as e:
        print(e)
        if stop_if_no_free_gpu:
            raise RuntimeError("Failed to retrieve GPU information.") from e
        return "cpu"


def cross_entropy(*, logits: Tensor, targets: Tensor) -> Tensor:
    """
    Cross entropy loss where the logits dimension is the last one, everything before is batched (differently from base working of torch cross entropy)
    """

    return torch.nn.functional.cross_entropy(
        logits.permute(0, -1, 1) if len(logits.shape) > 2 else logits,
        targets,
        reduction="none",
    )


def is_int(c: pd.Series) -> bool:
    """
    This function is used to determine if an int column is really numerical or used as categorical.
    Obviously it's not perfect, but it's a good heuristic.
    """
    if c.dtype != int:
        return False
    unique_values = c.nunique()
    range_values = c.max() - c.min() + 1
    # TODO: this is probably not possible and wrong, could cause issues
    return (
        unique_values < range_values or range_values > INT_IS_NUMERICAL_THRESHOLD
    ) and (unique_values > INT_IS_NUMERICAL_THRESHOLD)


def is_numerical(c: pd.Series) -> bool:
    return (c.dtype == float) or is_int(c)


def is_categorical(s: pd.Series) -> bool:
    if s.dtype == "object":
        return True
    return not is_numerical(s)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(filter(lambda c: not is_numerical(df[c]), df.columns))


def numerical_columns(df: pd.DataFrame) -> list[str]:
    return list(filter(lambda c: is_numerical(df[c]), df.columns))


def infer_int_series_type(series: pd.Series) -> str:
    unique_values = series.nunique()
    range_values = series.max() - series.min() + 1
    if unique_values < range_values or unique_values > INT_IS_NUMERICAL_THRESHOLD:
        return "numerical_int"
    return "categorical_int"


def is_fake_float(s: pd.Series) -> bool:
    if s.dtype == float and (~s.isna()).all():
        s_int = s.astype(int)
        return (s == s_int).all()
    else:
        return False


def fake_float_to_int(df: pd.DataFrame) -> None:
    for column in df.columns:
        if df[column].dtype == float and (~df[column].isna()).all():
            s_int = df[column].astype(int)
            if (df[column] == s_int).all():
                df[column] = s_int


def split_dataset(
    df: pd.DataFrame, *, train_size: float | int, random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and validation sets.

    :param df: The DataFrame to split.
    :param train_size: The proportion of the dataset to include in the training set.
    :param random_state: Controls the shuffling applied to the data before applying the split.
    :return: A tuple containing the training and validation DataFrames.
    """
    if random_state is not None:
        np.random.seed(random_state)

    shuffled_indices = np.random.permutation(len(df))

    if isinstance(train_size, float):
        train_size = int(len(df) * train_size)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    return df.iloc[train_indices].reset_index(drop=True), df.iloc[
        val_indices
    ].reset_index(drop=True)


def make_json_serializable(d: dict) -> dict:
    allowed_types = (float, int, str, bool, dict, type(None))
    return {k: v if isinstance(v, allowed_types) else str(v) for k, v in d.items()}


def printc(message, color):
    """
    Print a message to the terminal in the specified color.

    color: one of "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    """
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }
    color_code = colors.get(color.lower(), colors["reset"])

    if isinstance(message, dict):
        message = json.dumps(message, indent=4)
    print(f"{color_code}{message}{colors['reset']}")


def type_identification_features(x: pd.Series | np.ndarray | Tensor) -> dict:
    x = x.cpu().numpy() if isinstance(x, Tensor) else x
    s = pd.Series(x) if isinstance(x, np.ndarray) else x
    s = s.dropna()  # Drop NaN values for analysis

    # type
    pandas_type = pd.api.types.infer_dtype(s)

    # unique values analysis
    counts = s.value_counts().sort_index()
    total_unique_values = len(counts)
    fraction_of_unique_values = (counts == 1).sum() / s.size

    features = {
        "pandas_type": pandas_type,
        "total_unique_values": total_unique_values,
        "fraction_of_unique_values": fraction_of_unique_values,
    }

    if pandas_type in ["integer", "mixed-integer", "floating"]:
        # Gaps analysis
        steps = np.diff(counts.index.sort_values())
        min_step = steps.min()
        max_step = steps.max()

        # Counts correlation
        if total_unique_values <= 2:
            corr, pvalue = 0.0, 1.0  # Assume no correlation for very few unique values
        elif fraction_of_unique_values < 1.0 and counts.iloc[1:-1].std() > 0:
            corr, pvalue = stats.pearsonr(counts.iloc[:-1], counts.iloc[1:])  # type: ignore
        else:
            corr, pvalue = None, None

        return features | {
            "min_step": float(min_step),
            "max_step": float(max_step),
            "corr": corr,
            "corr_pvalue": pvalue,
        }
    else:
        return features


def is_numerical(series: pd.Series, verbose: bool = False) -> bool:
    pandas_type = pd.api.types.infer_dtype(series)
    if verbose:
        print(f"\n{series.name}\npandas_type: {pandas_type}")
    if pandas_type not in ["integer", "mixed-integer", "floating"]:
        if verbose:
            print(f"Series {series.name} is not numerical due to dtype: {pandas_type}")
        return False

    counts = series.value_counts().sort_index()
    total_unique_values = len(counts)
    fraction_of_unique_values = total_unique_values / series.size

    # If there are almost no repeated values
    if verbose:
        print(f"Fraction of unique values: {fraction_of_unique_values:.2f}")
    if fraction_of_unique_values > 0.5:
        if verbose:
            print(
                f"Series {series.name} is likely numerical due to the high fraction of unique values: {fraction_of_unique_values:.2f}"
            )
        return True

    # Check if there are holes
    steps = np.diff(counts.index.sort_values())
    min_step = steps.min()
    max_step = steps.max()
    if verbose:
        print(f"Min step: {min_step}, Max step: {max_step}")
    if min_step != max_step:
        if verbose:
            print(
                f"Series {series.name} is likely numerical due to the presence of holes in the unique values (min_step:{min_step}, max_step:{max_step})."
            )
        return True

    # If counts are self correlated, then it's likely numerical
    corr, pvalue = stats.pearsonr(counts.iloc[:-1], counts.iloc[1:])  # type: ignore
    if verbose:
        print(f"Self-correlation: corr={corr}, p-value={pvalue}")
    if pvalue < 0.01:  # type: ignore
        if verbose:
            print(
                f"Series {series.name} is likely numerical due to self-correlation in counts (corr:{corr}, p-value={pvalue})."
            )
        return True

    return False


def infer_type(series: pd.Series) -> str:
    """Infers the type of a pandas Series.

    Args:
        series (pd.Series): _description_

    Returns:
        str: a string describing the type of the series:
            "categorical_int": integer used as categorical
            "categorical_floating": floating used as categorical
            "quantized_floating": floating used as numerical, but quantized (e.g. 0.1, 0.2, 0.3)
            else the output of pd.api.types.infer_dtype
    """
    pandas_type = pd.api.types.infer_dtype(series)

    if pandas_type not in ["integer", "mixed-integer", "floating"]:
        return pandas_type

    type_features = type_identification_features(series)

    if (
        type_features["fraction_of_unique_values"] > 0.5
    ):  # technically collisions are still possible even when sampling float32 noise (apparently it happens)
        # TODO: very permissive threshold, unfortunately it can happen in certain situations because of weird datasets that I don't want to handle
        # If there are very few repeated values, then it's a numerical column (could be both float or int)
        return pandas_type

    # Now we have to check if it is a categorical or numerical column
    numerical = False

    # If a large part of the values is unique, then it's a numerical column
    if type_features["fraction_of_unique_values"] > 0.3:
        numerical = True

    # if there is a huge number of unique values, then it's a numerical column
    if type_features["total_unique_values"] > 200:
        numerical = True

    # if gaps in the sorted values are not constant, and these sorted values are many, then it's a numerical column
    if (
        type_features["min_step"] != type_features["max_step"]
        and type_features["total_unique_values"] > 10
    ):
        numerical = True

    # if the counts are self correlated, then it's a numerical column
    if type_features["corr"] is None or (type_features["corr"] > 0.5 and type_features["corr_pvalue"] < 0.1):  # type: ignore
        numerical = True

    if not numerical:
        return "categorical_" + pandas_type
    if pandas_type == "floating":
        return "quantized_floating"
    return pandas_type


def types_detector(x: pd.DataFrame | torch.Tensor) -> dict[str, str]:
    """
    Detects the types of columns in a DataFrame.

    Args:
        x (pd.DataFrame | torch.Tensor): The input data, either a pandas DataFrame or a PyTorch tensor.
    Returns:
        dict[str, str]: A dictionary mapping column names to their inferred types.
    """
    df = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x.cpu().numpy())
    return {col: infer_type(df[col]) for col in df.columns}


class Dequantizer:
    def __init__(self, train_data: Tensor):
        # find the minimum distance between points in the same column of the training data
        sorted_X = torch.sort(train_data, dim=0)[0]
        differences = torch.diff(sorted_X, dim=0)
        self.min_differences = torch.tensor(
            [
                differences[:, i][differences[:, i] > 0].min(dim=0)[0]
                for i in range(sorted_X.shape[1])
            ],
            device=train_data.device,
        )

        is_category = (train_data.floor() == train_data.long()).all(dim=0)
        self.min_differences[is_category] = 0.0  # don't dequantize categorical columns

    def dequantize(self, x: Tensor) -> Tensor:
        return x + (torch.rand_like(x, device=x.device) - 0.5) * self.min_differences

    def __call__(self, x: Tensor) -> Tensor:
        return self.dequantize(x)


def is_categorical(series: pd.Series) -> bool:
    """
    Check if a pandas Series is categorical based on its dtype and unique values.
    """
    if series.dtype in ["object", "category", "string", "bool"]:
        return True
    else:
        inferred_type = infer_type(series)
        return "categorical" in inferred_type


def number_of_zeros(s: pd.Series) -> int:
    x = np.array(s[~s.isna()])
    max_digits = np.ceil(np.log10(x.max()))
    remainders = np.mod(np.array(x).reshape(-1, 1), (10 ** np.arange(1, max_digits)))
    digits = np.where(np.all(remainders == 0.0, 0))[0]
    if digits.size == 0:
        return 0
    return int(digits.max()) + 1


def decimals_used(x: pd.Series):
    str_vals = x[~x.isna()].astype(str).str.split(".")
    decimal_parts = str_vals.str[1].fillna("")
    if (decimal_parts == "0").all():
        return -number_of_zeros(x)
    decimal_lengths = decimal_parts.str.len()
    return int(decimal_lengths.max())


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def cosine_scheduler_function(t, y_min: float, y_max: float, t_max: int):
    return y_min + 0.5 * (y_max - y_min) * (1 + math.cos(math.pi * t / t_max))


class NoiseCurriculum:
    def __init__(self, train_data: Tensor, max_noise: float, t_max: int):
        self.train_data = train_data
        self.max_noise = max_noise
        self.t_max = t_max
        self.is_category = (train_data.floor() == train_data.long()).all(dim=0)

    def add_noise(self, x: Tensor, *, t: int) -> Tensor:
        std = cosine_scheduler_function(
            t=t, y_min=0.0, y_max=self.max_noise, t_max=self.t_max
        )
        return torch.where(
            self.is_category, x, x + torch.randn_like(x, device=x.device) * std
        )

    def __call__(self, x: Tensor, *, t: int) -> Tensor:
        return self.add_noise(x, t=t)


def find_files(*, starting_folder: str = ".", pattern: str):
    """
    find all files that match the given pattern, starting from the given folder and going down the directory tree
    """
    matches = []
    for root, _, files in os.walk(starting_folder):
        for filename in files:
            full_name = os.path.join(root, filename)
            if fnmatch.fnmatch(full_name, pattern):
                matches.append(full_name)
    return matches


def find_dirs(*, starting_folder: str = ".", pattern: str):
    """
    find all directories that match the given pattern, starting from the given folder and going down the directory tree
    """
    matches = []
    for root, dirs, _ in os.walk(starting_folder):
        for dirname in dirs:
            full_name = os.path.join(root, dirname)
            if fnmatch.fnmatch(full_name, pattern):
                matches.append(full_name)
    return matches


def bin_dataframe(
    df: pd.DataFrame, *, bins: int | dict, labels=None, encode: bool = False
):
    """
    Bins the numerical columns of a DataFrame into n_bins equal-width bins.

    Args:
        df (pd.DataFrame): The input DataFrame.
        n_bins (int): The number of bins to create for each numerical column.
        labels (bool): Whether to use labels for the bins.

    Returns:
        pd.DataFrame: A new DataFrame with numerical columns binned.
    """

    binned_df = df.copy()

    bin_edges = {}
    for col in binned_df.columns:
        if isinstance(bins, dict) and col not in bins:
            continue

        if isinstance(bins, int):
            n_bins = bins
        else:
            bin_edges[col] = bins[col]
            n_bins = len(bin_edges[col])

        if (len(binned_df[col].unique()) > n_bins) and (
            binned_df[col].dtype == float or binned_df[col].dtype == int
        ):
            if col not in bin_edges:
                bin_edges[col] = np.histogram_bin_edges(
                    binned_df[col].dropna(), bins=n_bins
                )
            binned_df[col] = pd.cut(
                binned_df[col],
                bins=bin_edges[col],
                labels=labels,
                include_lowest=True,
            )

    if encode:
        for col in df.columns:
            binned_df[col] = pd.Categorical(binned_df[col]).codes
    return binned_df


def torch_mi_matrix(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[1]
    mi_matrix = torch.zeros((n, n), device=X.device)
    for i in range(n):
        for j in range(i + 1):
            mi = mutual_info_score(X[:, i], X[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    return mi_matrix


def df_mutual_information(
    df: pd.DataFrame, *, n_bins: int = 10, device: str = "cpu", encode: bool = True
) -> dict[str, np.ndarray]:
    """
    Computes the mutual information matrix for a DataFrame with mixed data types.

    Args:
        df (pd.DataFrame): The input DataFrame.
        n_bins (int): The number of bins to use for numerical columns.

    Returns:
        np.ndarray: The mutual information matrix.
    """

    if encode:
        binned_df, _ = bin_dataframe(df, bins=n_bins, labels=False)

        for col in df.columns:
            binned_df[col] = pd.Categorical(binned_df[col]).codes
    else:
        binned_df = df

    X = torch.tensor(binned_df.values, device=device, dtype=torch.long)

    mi = torch_mi_matrix(X).cpu().numpy()
    entropies = np.diagonal(mi).squeeze()
    entropies = np.where(
        entropies > 0, entropies, 1
    )  # where entropy is 0, all the row and column are 0, so it doesn't matter what I put here

    nmi = 2 * mi / (entropies[:, None] + entropies[None, :])
    return {"mi": mi, "nmi": nmi}


def wait_available_gpu(
    min_free_mem: float = 0.5, min_free_usage: float = 0.5, wait: bool = True
) -> str:
    """
    Returns the most free GPU based on memory and load using GPUtil.
    Respects the CUDA_VISIBLE_DEVICES environment variable.

    Args:
        min_free_mem: Minimum fraction of free memory required (0.0 - 1.0)
        min_free_usage: Minimum fraction of free (idle) usage required (0.0 - 1.0)
        wait: If True, waits until a GPU satisfies requirements; else raises.

    Returns:
        str: GPU device string (e.g. "cuda:0")

    Raises:
        RuntimeError: If no suitable GPU and wait=False, or no CUDA available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No NVIDIA GPUs (CUDA) available")

    def pick_gpu():
        gpus = GPUtil.getGPUs()
        best = None
        best_score = -1.0

        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            visible_device_ids = [
                int(x) for x in visible_devices.split(",") if x.strip().isdigit()
            ]
            visible_device_ids = sorted(visible_device_ids)
            gpus = [g for g in gpus if g.id in visible_device_ids]

        visible_mapping = {vis_idx : local_idx for local_idx, vis_idx in enumerate(visible_device_ids)} if visible_devices is not None else {}

        for g in gpus:
            mem_free_fraction = (g.memoryTotal - g.memoryUsed) / g.memoryTotal
            gpu_free_usage_fraction = 1.0 - g.load  # g.load in [0,1]
            if (
                mem_free_fraction >= min_free_mem
                and gpu_free_usage_fraction >= min_free_usage
            ):
                score = (mem_free_fraction + gpu_free_usage_fraction) * 0.5
                if score > best_score:
                    best_score = score
                    best = f"cuda:{visible_mapping[g.id]}" if visible_devices is not None else f"cuda:{g.id}"
        return best

    while True:
        chosen = pick_gpu()
        if chosen:
            return chosen
        if not wait:
            raise RuntimeError(
                f"No GPU meets requirements (free mem>={min_free_mem:.0%}, free usage>={min_free_usage:.0%})"
            )
        print("Waiting for a GPU to become available...")
        time.sleep(3)


def randomly_mask_values(df: pd.DataFrame, *, mask_fraction: float) -> pd.DataFrame:
    """
    Randomly mask a fraction of values in a DataFrame by setting them to NaN.

    This implementation:
    - validates mask_fraction
    - builds a boolean mask using flat indices (works for mixed dtypes)
    - applies pandas.DataFrame.mask to set values to NaN
    """

    df_masked = df.copy()
    if mask_fraction == 0.0:
        return df_masked

    total_values = df_masked.size
    num_masks = int(total_values * mask_fraction)
    if num_masks == 0:
        return df_masked

    # sample flat indices (handle full mask case)
    if num_masks >= total_values:
        flat_indices = np.arange(total_values)
    else:
        flat_indices = np.random.choice(total_values, size=num_masks, replace=False)

    # build boolean mask and apply via DataFrame.mask
    bool_flat = np.zeros(total_values, dtype=bool)
    bool_flat[flat_indices] = True
    bool_mask = bool_flat.reshape(df_masked.shape)
    mask_df = pd.DataFrame(bool_mask, index=df_masked.index, columns=df_masked.columns)

    return df_masked.mask(mask_df)


def mask_values(x: Tensor, *, p: float) -> Tensor:
    """
    Randomly mask a fraction p of values in a Tensor by setting them to NaN.

    Args:
        x (Tensor): The input tensor.
        p (float): The fraction of values to mask (between 0 and 1).

    Returns:
        Tensor: A new tensor with randomly masked values.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0.0 and 1.0")

    if p == 0.0:
        return x

    x_masked = x.clone()
    n_masked = int(x.numel() * p)

    if n_masked == 0:
        return x_masked

    # Sample flat indices (handle full mask case)
    if n_masked >= x.numel():
        flat_indices = torch.arange(x.numel(), device=x.device)
    else:
        flat_indices = torch.randperm(x.numel(), device=x.device)[:n_masked]

    # Flatten with reshape (handles non-contiguous tensors)
    x_flat = x_masked.reshape(-1)
    x_flat[flat_indices] = float("nan")

    return x_masked


def split_torch_dataset(x: Tensor, train_size: float) -> tuple[Tensor, Tensor]:
    """
    Splits a PyTorch tensor into training and validation sets.

    :param x: The input tensor to split.
    :param train_size: The proportion of the dataset to include in the training set.
    :return: A tuple containing the training and validation tensors.
    """
    shuffled_indices = torch.randperm(x.size(0))

    train_size_int = int(x.size(0) * train_size)

    train_indices = shuffled_indices[:train_size_int]
    val_indices = shuffled_indices[train_size_int:]

    return x[train_indices], x[val_indices]
