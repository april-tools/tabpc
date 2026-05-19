import sys
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import LogisticDetection
from sdmetrics.single_table.detection.sklearn import \
    ScikitLearnClassifierDetectionMetric
from sdv.metadata import SingleTableMetadata
from tqdm import tqdm
from xgboost import XGBClassifier

from src.util import (bin_dataframe, df_mutual_information,
                      get_available_device, printc)

sys.path.append(".")
from tabdiff_eval.mle.mle import get_evaluator

DEFAULT_PRECISION_DIGITS = 4


def df_common_binning(
    *,
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    info: dict | None,
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if info is None:
        raise ValueError("info must be provided.")
    numerical_columns = real_data.columns[info["num_col_idx"]]
    if info["task_type"] == "regression":
        numerical_columns = numerical_columns.append(
            real_data.columns[info["target_col_idx"]]
        )
    bin_edges = {
        c: np.histogram_bin_edges(real_data[c], bins=bins) for c in numerical_columns
    }
    binned_real = bin_dataframe(real_data, bins=bin_edges, encode=True)
    binned_synthetic = bin_dataframe(synthetic_data, bins=bin_edges, encode=True)
    return binned_real, binned_synthetic


def float_format(value, precision_digits=DEFAULT_PRECISION_DIGITS) -> float:
    """Format float to a specific number of decimal places."""
    return round(float(value), precision_digits)


def reorder(real_data, syn_data, info):
    """
    Code from TabDiff
    """
    num_col_idx = deepcopy(
        info["num_col_idx"]
    )  # BUG: info will be modified by += in the next few lines
    cat_col_idx = deepcopy(info["cat_col_idx"])
    target_col_idx = deepcopy(info["target_col_idx"])

    task_type = info["task_type"]
    if task_type == "regression":
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]

    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    metadata = info["metadata"]

    columns = metadata["columns"]
    metadata["columns"] = {}

    info["inverse_idx_mapping"]

    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata["columns"][i] = columns[num_col_idx[i]]
        else:
            metadata["columns"][i] = columns[cat_col_idx[i - len(num_col_idx)]]

    return new_real_data, new_syn_data, metadata


def complete_y_only_data(self, syn_data, real_data, target_col_idx):
    syn_target_col = deepcopy(syn_data.iloc[:, 0])
    syn_data = deepcopy(real_data)
    syn_data[target_col_idx] = syn_target_col
    return syn_data


def standard_preprocessing(real_data: pd.DataFrame, syn_data: pd.DataFrame, info: dict):
    info_copy = deepcopy(info)
    syn_data_copy = syn_data.copy()
    real_data_copy = real_data.copy()

    real_data_copy.columns = range(len(real_data_copy.columns))
    syn_data_copy.columns = range(len(syn_data_copy.columns))

    metadata = info_copy["metadata"]
    metadata["columns"] = {
        int(key): value for key, value in metadata["columns"].items()
    }

    return reorder(real_data_copy, syn_data_copy, info_copy)


def dcrs(*, source_num, source_cat, target_num, target_cat, batch_size):
    dcrs_list = []
    device = source_num.device if source_num is not None else source_cat.device

    for i in tqdm(range((source_num.shape[0] // batch_size) + 1)):
        start = i * batch_size
        end = min((i + 1) * batch_size, source_num.shape[0])
        if start >= end:
            continue

        # Numeric distances
        if source_num is not None:
            b_source_num = source_num[start:end]
            diff_target_num = (
                (b_source_num[:, None, :] - target_num[None, :, :]).abs().sum(2)
            )
        else:
            diff_target_num = torch.zeros(
                (end - start, target_cat.shape[0]), device=device
            )

        # Categorical distances (0 if equal else 2)
        if source_cat is not None and source_cat.shape[1] > 0:
            b_source_cat = source_cat[start:end]
            diff_target_cat = (b_source_cat[:, None, :] != target_cat[None, :, :]).to(
                torch.float32
            ).sum(2) * 2.0
        else:
            diff_target_cat = torch.zeros_like(diff_target_num)

        dcrs_list.append((diff_target_num + diff_target_cat).min(1).values)

    return torch.cat(dcrs_list)


def get_reordered_feature_idx(idx: int, info: dict) -> int:
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression' and (target_col_idx[0] not in num_col_idx):
        num_col_idx.append(target_col_idx[0])
    elif task_type == 'binclass' and (target_col_idx[0] not in cat_col_idx):
        cat_col_idx.append(target_col_idx[0])

    idx_is_regression_target = (idx in target_col_idx and task_type == 'regression') # Check if idx is a numerical regression target
    num_target = True if idx in num_col_idx or idx_is_regression_target else False # Check if idx is numerical

    if num_target:
        reordered_idx = num_col_idx.index(idx)
    else:
        reordered_idx = len(num_col_idx) + cat_col_idx.index(idx)

    return reordered_idx


class XGBoostDetection(ScikitLearnClassifierDetectionMetric):
    name = "XGBoost Detection"

    @staticmethod
    def _get_classifier():
        return XGBClassifier(
            eval_metric="logloss", enable_categorical=True, random_state=0
        )


class Metric(ABC):
    @staticmethod
    @abstractmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def requires_gpu() -> bool:
        raise NotImplementedError


class C2ST(Metric):
    @staticmethod
    def name() -> str:
        return "C2ST"

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        metadata = None
        if info is not None:
            real_data, synthetic_data, metadata = standard_preprocessing(
                real_data, synthetic_data, info
            )

        score_logistic_regression = LogisticDetection.compute(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
        )
        score_xgboost = XGBoostDetection.compute(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
        )
        return {
            "logistic_regression (C2ST)": float_format(score_logistic_regression),
            "logistic_regression (AUC)": float_format(
                1 - score_logistic_regression / 2
            ),
            "xgboost (C2ST)": float_format(score_xgboost),
            "xgboost (AUC)": float_format(1 - score_xgboost / 2),
        }

    @staticmethod
    def requires_gpu() -> bool:
        return False


class ShapeTrend(Metric):
    """
    Shape measures the similarity of the distribution of the columns in the real and synthetic data.
    Trend measures the similarity of the correlation between pairs of columns in the real and synthetic data.

    Shape:
        Continuous, Datetimes: average of 1-(Kolmogorov-Smirnov statistic) for every column. Check: https://docs.sdv.dev/sdmetrics/metrics/quality-metrics/kscomplement
        Categorical, Boolean: average of 1-(Total Variation Distance (TVD) ) for every column. Check: https://docs.sdv.dev/sdmetrics/metrics/quality-metrics/tvcomplement
    Trend:
        Continuous, Datetimes: average of 1- | Corr_real(x,y) - Corr_syn(x,y)|/2 for every pair of columns (x,y). Check: https://docs.sdv.dev/sdmetrics/metrics/quality-metrics/correlationsimilarity
        Categorical, Boolean: average of 1- | P_real(x,y) - P_syn(x,y)|/2 for every pair of columns (x,y). Check: https://docs.sdv.dev/sdmetrics/metrics/quality-metrics/contingencysimilarity

    Quality report by SDMetrics: https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included
    """

    @staticmethod
    def name() -> str:
        return "ShapeTrend"

    @staticmethod
    def requires_gpu() -> bool:
        return False

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        # Automatically generate metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_data)
        metadata = metadata.to_dict()

        # Generate SDMetrics quality report, just for Shape and Trend
        qual_report = QualityReport()
        qual_report.generate(real_data, synthetic_data, metadata, verbose=False)
        quality = qual_report.get_properties()

        shapes_details = qual_report.get_details("Column Shapes")

        out_metrics = {
            "Shape": float_format(quality["Score"][0]),
            "Trend": float_format(quality["Score"][1]),
            "Shapes": shapes_details.set_index("Column")["Score"]
            .sort_values()
            .to_dict(),
            "Trends": list(
                qual_report.get_details("Column Pair Trends")
                .sort_values("Score")
                .T.to_dict()
                .values()
            ),
        }

        return out_metrics


class Mle(Metric):
    """
    Code from TabDiff
    """

    @staticmethod
    def name() -> str:
        return "mle"

    @staticmethod
    def requires_gpu() -> bool:
        return True

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        if df_test is None or info is None:
            return {}

        train = synthetic_data.to_numpy()
        test = df_test.to_numpy()

        info = deepcopy(info)

        task_type = info["task_type"]

        evaluator = get_evaluator(task_type)

        if task_type == "regression":
            best_r2_scores, best_rmse_scores = evaluator(train, test, info, val=None)

            overall_scores = {}
            for score_name in ["best_r2_scores", "best_rmse_scores"]:
                overall_scores[score_name] = {}

                scores = eval(score_name)
                for method in scores:
                    name = method["name"]
                    method.pop("name")
                    overall_scores[score_name][name] = method

        else:
            (
                best_f1_scores,
                best_weighted_scores,
                best_auroc_scores,
                best_acc_scores,
                best_avg_scores,
            ) = evaluator(train, test, info, val=None)

            overall_scores = {}
            for score_name in [
                "best_f1_scores",
                "best_weighted_scores",
                "best_auroc_scores",
                "best_acc_scores",
                "best_avg_scores",
            ]:
                overall_scores[score_name] = {}

                scores = eval(score_name)
                for method in scores:
                    name = method["name"]
                    method.pop("name")
                    overall_scores[score_name][name] = method

        mle_score = (
            overall_scores["best_rmse_scores"]["XGBRegressor"]["RMSE"]
            if task_type == "regression"
            else overall_scores["best_auroc_scores"]["XGBClassifier"]["roc_auc"]
        )
        out_metrics = {
            "mle": mle_score,
            # "extras": overall_scores
        }

        return out_metrics


class QuantileDcr(Metric):
    """
    Code from TabDiff
    """

    @staticmethod
    def name() -> str:
        return "quantile_dcr"

    @staticmethod
    def requires_gpu() -> bool:
        return True

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        if df_test is None or info is None:
            return {}

        colnames = real_data.columns.tolist()

        syn_data = synthetic_data.copy()
        real_data_copy = real_data.copy()
        test_data = df_test.copy()
        info = deepcopy(info)

        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]

        task_type = info["task_type"]
        if task_type == "regression":
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        # special treatment for Beijing dataset, categorical columns are mostly an index here
        if (
            "year" in colnames
            and "month" in colnames
            and "day" in colnames
            and "hour" in colnames
        ):
            cat_col_idx = []

        # Reindex columns numerically
        real_data_copy.columns = list(range(len(real_data_copy.columns)))
        syn_data.columns = list(range(len(real_data_copy.columns)))
        test_data.columns = list(range(len(real_data_copy.columns)))

        # Split numeric / categorical
        num_real = real_data_copy[num_col_idx]
        num_syn = syn_data[num_col_idx]
        num_test = test_data[num_col_idx]

        cat_real = real_data_copy[cat_col_idx]
        cat_syn = syn_data[cat_col_idx]
        cat_test = test_data[cat_col_idx]

        # Numeric to numpy
        num_real_np = num_real.to_numpy(dtype=float)
        num_syn_np = num_syn.to_numpy(dtype=float)
        num_test_np = num_test.to_numpy(dtype=float)

        if len(num_col_idx) > 0:

            mean = num_real_np.mean(axis=0)
            std = num_real_np.std(axis=0)

            num_real_np = (num_real_np - mean) / std
            num_syn_np = (num_syn_np - mean) / std
            num_test_np = (num_test_np - mean) / std

        # Label encode categoricals per column (factorize over combined real+test+syn)
        if len(cat_col_idx) > 0:
            n_cat = len(cat_col_idx)
            cat_real_codes = np.empty((len(cat_real), n_cat), dtype=np.int64)
            cat_syn_codes = np.empty((len(cat_syn), n_cat), dtype=np.int64)
            cat_test_codes = np.empty((len(cat_test), n_cat), dtype=np.int64)
            for j, col in enumerate(cat_col_idx):
                all_vals = pd.concat(
                    [cat_real[col], cat_test[col], cat_syn[col]], axis=0
                ).astype(str)
                codes, uniques = pd.factorize(all_vals, sort=True)
                n_real = len(cat_real)
                n_test = len(cat_test)
                cat_real_codes[:, j] = codes[:n_real]
                cat_test_codes[:, j] = codes[n_real : n_real + n_test]
                cat_syn_codes[:, j] = codes[n_real + n_test :]
        else:
            cat_real_codes = np.empty((len(num_real_np), 0), dtype=np.int64)
            cat_syn_codes = np.empty((len(num_syn_np), 0), dtype=np.int64)
            cat_test_codes = np.empty((len(num_test_np), 0), dtype=np.int64)

        # Torch tensors
        device = (
            get_available_device(mem_required=0.50, stop_if_no_free_gpu=False)
            if device is None
            else device
        )

        real_num_th = (
            torch.tensor(num_real_np, dtype=torch.float32, device=device)
            if num_real_np.size
            else None
        )
        syn_num_th = (
            torch.tensor(num_syn_np, dtype=torch.float32, device=device)
            if num_syn_np.size
            else None
        )
        test_num_th = (
            torch.tensor(num_test_np, dtype=torch.float32, device=device)
            if num_test_np.size
            else None
        )

        real_cat_th = (
            torch.tensor(cat_real_codes, dtype=torch.long, device=device)
            if cat_real_codes.size
            else None
        )
        syn_cat_th = (
            torch.tensor(cat_syn_codes, dtype=torch.long, device=device)
            if cat_syn_codes.size
            else None
        )
        test_cat_th = (
            torch.tensor(cat_test_codes, dtype=torch.long, device=device)
            if cat_test_codes.size
            else None
        )

        # Batch size heuristic
        eff_feats = (real_num_th.shape[1] if real_num_th is not None else 0) + (
            real_cat_th.shape[1] if real_cat_th is not None else 0
        )
        batch_size = max(50, 10000 // max(1, eff_feats))

        # n_syn = syn_num_th.shape[0] if syn_num_th is not None else syn_cat_th.shape[0]
        # n_test = (test_num_th.shape[0] if test_num_th is not None else test_cat_th.shape[0])

        dcrs_syn = dcrs(
            source_num=syn_num_th,
            source_cat=syn_cat_th,
            target_num=real_num_th,
            target_cat=real_cat_th,
            batch_size=batch_size,
        )

        dcrs_test = dcrs(
            source_num=test_num_th,
            source_cat=test_cat_th,
            target_num=real_num_th,
            target_cat=real_cat_th,
            batch_size=batch_size,
        )

        # dcrs_test = torch.cat(dcrs_test)
        # score = (dcrs_real < dcrs_test).sum().item() / dcrs_real.shape[0]

        ps = [0.02, 0.05]

        scores = {
            f"dcr_fraction<{p}_test_quantile": (dcrs_syn < torch.quantile(dcrs_test, p))
            .sum()
            .item()
            * 100
            / dcrs_syn.shape[0]
            for p in ps
        }

        return {
            **scores,
            "dcr_syn_mean": float(dcrs_syn.mean().cpu()),
            "dcr_test_mean": float(dcrs_test.mean().cpu()),
        }


class LegacyDensity(Metric):
    """
    Code from TabDiff
    """

    @staticmethod
    def name() -> str:
        return "legacy_density"

    @staticmethod
    def requires_gpu() -> bool:
        return False

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        real_data_copy = real_data.copy()
        syn_data = synthetic_data.copy()

        real_data_copy.columns = range(len(real_data_copy.columns))
        syn_data.columns = range(len(syn_data.columns))

        if info is None:
            printc(
                "Warning: info is None, using ShapeTrend metric instead of LegacyDensity.",
                color="red",
            )
            new_metric = ShapeTrend()
            return new_metric(
                real_data=real_data,
                synthetic_data=synthetic_data,
                df_test=df_test,
                info=info,
                device=device,
            )

        info_copy = deepcopy(info)

        y_only = len(syn_data.columns) == 1
        if y_only:
            target_col_idx = info_copy["target_col_idx"][0]
            syn_data = complete_y_only_data(syn_data, real_data_copy, target_col_idx)

        metadata = info_copy["metadata"]
        metadata["columns"] = {
            int(key): value for key, value in metadata["columns"].items()
        }  # ensure that keys are all integers?

        new_real_data, new_syn_data, metadata = reorder(
            real_data_copy, syn_data, info_copy
        )

        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata, verbose=False)

        # diag_report = DiagnosticReport()
        # diag_report.generate(new_real_data, new_syn_data, metadata)

        quality = qual_report.get_properties()
        # diag_report.get_properties()

        Shape = quality["Score"][0]
        Trend = quality["Score"][1]

        Overall = (Shape + Trend) / 2

        shape_details = qual_report.get_details(property_name="Column Shapes")
        trend_details = qual_report.get_details(property_name="Column Pair Trends")

        if y_only:
            Shape = shape_details["Score"].min()
        out_metrics = {
            "density/Shape": Shape,
            "density/Trend": Trend,
            "density/Overall": Overall,
            "trends": trend_details["Score"].round(4).to_list(),
        }
        # out_extras = {"shapes": shape_details, "trends": trend_details}

        return out_metrics


class MI_l1(Metric):
    @staticmethod
    def name() -> str:
        return "mi_l1"

    @staticmethod
    def requires_gpu() -> bool:
        return True

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        # Binning the DataFrames
        binned_real, binned_synthetic = df_common_binning(
            real_data=real_data, synthetic_data=synthetic_data, info=info, bins=10
        )

        # Compute MI
        real_mis = df_mutual_information(
            binned_real, encode=False, device=device if device is not None else "cpu"
        )
        synthetic_mis = df_mutual_information(
            binned_synthetic,
            encode=False,
            device=device if device is not None else "cpu",
        )

        out_metrics = {}

        for name in real_mis.keys():
            real_mi = real_mis[name]
            synthetic_mi = synthetic_mis[name]

            # Compute L1 error
            error = np.abs(real_mi - synthetic_mi)

            # Select only lower triangle
            tril_indices = np.tril_indices_from(error, k=-1)
            errors = error[tril_indices]

            out_metrics[f"{name}_l1"] = float(errors.mean())

            # Weighted version: weighting by the original (N)MI
            weights = real_mi[tril_indices] + synthetic_mi[tril_indices]
            out_metrics[f"{name}_l1_weighted"] = float(
                (weights * errors).sum() / weights.sum()
            )

        # Complement versions
        for w in ("", "_weighted"):
            out_metrics[f"nmi_l1{w}_complement"] = 1 - out_metrics[f"nmi_l1{w}"]

        return out_metrics


class MI_all(Metric):
    @staticmethod
    def name() -> str:
        return "mi_all"

    @staticmethod
    def requires_gpu() -> bool:
        return True

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
        bins: int | str = "auto",
    ):
        # Binning the DataFrames
        binned_real, binned_synthetic = df_common_binning(
            real_data=real_data, synthetic_data=synthetic_data, info=info, bins=bins
        )

        # Compute MI
        real_mis = df_mutual_information(
            binned_real, encode=False, device=device if device is not None else "cpu"
        )
        synthetic_mis = df_mutual_information(
            binned_synthetic,
            encode=False,
            device=device if device is not None else "cpu",
        )

        out_metrics = {}

        for name in real_mis.keys():
            real_mi = real_mis[name]
            synthetic_mi = synthetic_mis[name]

            # Compute L1 error
            error = np.abs(real_mi - synthetic_mi)

            # Select only lower triangle
            tril_indices = np.tril_indices_from(error, k=-1)
            errors = error[tril_indices]

            out_metrics[f"{name}_l1"] = errors.tolist()
            out_metrics[f"{name}_l1_score"] = float(errors.mean())
            out_metrics[f"{name}_real"] = real_mi[tril_indices].tolist()
            out_metrics[f"{name}_synthetic"] = synthetic_mi[tril_indices].tolist()

            # Weighted version: weighting by the original (N)MI
            weights = real_mi[tril_indices] + synthetic_mi[tril_indices]
            weights /= weights.sum()

            weighted_error = np.multiply(errors, weights)
            out_metrics[f"{name}_l1_weighted"] = weighted_error.tolist()
            out_metrics[f"{name}_l1_weighted_score"] = float(weighted_error.sum())

            out_metrics[f"{name}_l1_similarity"] = (1 - errors).tolist()
            out_metrics[f"{name}_l1_similarity_weighted"] = (weights - weighted_error).tolist()

            out_metrics[f"{name}_real_weighted"] = (real_mi[tril_indices] * weights).tolist()
            out_metrics[f"{name}_synthetic_weighted"] = (synthetic_mi[tril_indices] * weights).tolist()
        # Complement versions
        for w in ("", "_weighted"):
            out_metrics[f"nmi_l1{w}_similarity_score"] = 1 - out_metrics[f"nmi_l1{w}_score"]

        col1_arr, col2_arr = np.tril_indices_from(real_mis["mi"], k=-1)
        col1_arr = col1_arr.tolist()
        col2_arr = col2_arr.tolist()

        # Map the column indices to their reordered versions
        col1_arr = [get_reordered_feature_idx(idx, info) for idx in col1_arr]
        col2_arr = [get_reordered_feature_idx(idx, info) for idx in col2_arr]

        for idx in range(len(col1_arr)):
            if col1_arr[idx] > col2_arr[idx]:
                temp = col1_arr[idx]
                col1_arr[idx] = col2_arr[idx]
                col2_arr[idx] = temp

        out_metrics['col_1_indices'] = col1_arr
        out_metrics['col_2_indices'] = col2_arr

        return out_metrics