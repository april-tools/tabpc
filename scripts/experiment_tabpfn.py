import warnings
import copy
import json
import argparse
import os
import time
import sys

sys.path.append(".")
sys.path.append("cirkit")

import torch
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from tabpfn_extensions import unsupervised
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

from src.preprocessors.compose import Compose
from src.preprocessors.nan_handler import NanHandler
from src.preprocessors.df_type_conversion import TypeConversion
from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
from src.preprocessors.df_dequantization import Dequantization
from src.preprocessors.string_to_int import StringToInt
from src.preprocessors.to_tensor import ToTensor

TABPFN_MAX_CATS = 160

def generate_synthetic_dataset_tabpfn(
        dataset_name: str,
        repeat_idx: int,
        output_data_dir: str,
        save_raw: bool,
        overwrite: bool,
        n_permutations: int,
        use_only_ordinal: bool,
    ):

    # Check if sample already exists
    output_dir = os.path.join(output_data_dir, dataset_name, f"run_{repeat_idx}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tabpfn_{dataset_name}_run_{repeat_idx}_perms_{n_permutations}_new.csv")
    if os.path.exists(output_path) and not overwrite:
        print("Skipping generating existing file:", output_path)
        return

    df_X = pd.read_csv(f"data/{dataset_name}/train.csv")

    # Get dataset metadata
    with open(f"data/{dataset_name}/info.json") as json_f:
        data_info_file = json.load(json_f)

    categorical_features = data_info_file["cat_col_idx"]
    numerical_features = data_info_file["num_col_idx"]

    # Add target label to list of categorical features if required
    task_type = data_info_file["task_type"]
    target_col_idx = data_info_file["target_col_idx"]
    if task_type == "binclass":
        categorical_features += target_col_idx

    # Set numerical features with sufficiently few values to be handled as categorical
    # int_features = data_info_file["int_col_idx"]
    for num_idx in numerical_features:
        if len(df_X.iloc[:, num_idx].value_counts()) <= TABPFN_MAX_CATS:
            categorical_features.append(num_idx)

    # Encode the categorical data as integer values to pass to TabPFN
    if use_only_ordinal:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    else:
        preprocessors = [
            TypeConversion(cat_max_states=160, 
                            int_max_states=None),
            # NanHandler(handle_inflated_values=True, handle_missing_values=False), # TabPFN states you should not preprocess data
            # Dequantization(dequantize=True, dequantize_all_floats=False), # This should not be needed for TabPFN
            # QuantileNormalizer(), # TabPFN states that you should not scale data
            StringToInt(),
            ToTensor(),
        ]
        encoder = Compose(preprocessors)

    df_X_encoded = df_X.copy()

    if use_only_ordinal:
        df_X_encoded.iloc[:, categorical_features] = encoder.fit_transform(df_X_encoded.iloc[:, categorical_features])
        # X_preprocessed = df_X_encoded.to_numpy()
        X_preprocessed = torch.from_numpy(df_X_encoded.values.astype("float32")).float()
        
        feature_names = df_X_encoded.columns
    else:
        X_preprocessed = encoder.fit_transform(df_X_encoded)

        feature_names = encoder.preprocessors[-1].parameters["names"]
    
    feature_indices = [*range(X_preprocessed.shape[1])] # Select all indices for generation
    # Now this can include features added after preprocessing (e.g. for inflated value handling)
    
    y_preprocessed = torch.empty([])

    # Initialize TabPFN models
    model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(), 
        tabpfn_reg=TabPFNRegressor()
    )

    # Run synthetic data generation experiment
    experiment = GenerateSyntheticDataExperiment(
        task_type="unsupervised"
    )

    # import pdb; pdb.set_trace()

    results = experiment.run(
        tabpfn=model_unsupervised,
        X=X_preprocessed,
        y=y_preprocessed,
        attribute_names=feature_names,
        temp=1.0,                     # Temperature parameter for sampling
        n_samples=df_X.shape[0],     # Generate as many samples as original data
        indices=feature_indices,
        categorical_features=categorical_features, # Set categorical features
        should_plot=False,
        n_permutations=n_permutations,
    )

    data = experiment.data
    synthetic_data_raw = data[data["real_or_synthetic"] == "Generated samples"].drop("real_or_synthetic", axis=1)

    # Save a raw copy in case inverting the encoder doesn't work
    if save_raw:
        raw_output_path = os.path.join(output_dir, f"tabpfn_{dataset_name}_run_{repeat_idx}_perms_{n_permutations}_raw.csv")
        synthetic_data_raw.to_csv(raw_output_path)

    # Restore the original categories
    if use_only_ordinal:
        synthetic_data = synthetic_data_raw.copy()
        synthetic_data.iloc[:, categorical_features] = encoder.inverse_transform(synthetic_data.iloc[:, categorical_features])
        synthetic_data_df = synthetic_data.copy()
    else:
        synthetic_data = synthetic_data_raw.copy()
        synthetic_data_tensor = torch.tensor(synthetic_data.values)
        synthetic_data_df = encoder.reverse_transform(synthetic_data_tensor)

    # Save to csv
    synthetic_data_df.to_csv(output_path, index=False)

    return

if __name__ == "__main__":
    # Monkey patch the run method of the GenerateSyntheticDataExperiment class to fix current bug with treating features as categorical
    # GenerateSyntheticDataExperiment.run = run

    parser = argparse.ArgumentParser(description="Generate synthetic data using TabPFN in the format expected by the repo")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["adult", "beijing", "default", "diabetes", "magic", "news", "shoppers", "all"],
                        help="Which dataset to generate a synthetic dataset for.")
    parser.add_argument("--output-data-dir", required=True,
                        help="The directory in which to store the generated data.")
    parser.add_argument("--num-repeats", type=int, default=5,
                        help="Number of times to regenerate a dataset to be able to compute standard deviations. (Default: 5)")
    parser.add_argument("--save-raw", action="store_true", default=False,
                        help="Whether to save the raw generated dataset before inverting the encoding. (Default: False)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Whether to overwrite existing samples with newly generated ones. (Default: False)")
    parser.add_argument("--n-permutations", type=int, default=3,
                        help="The number of permutations to use for TabPFN's imputation generation. (Default: 3)")
    parser.add_argument("--use-only-ordinal", action="store_true", default=False,
                        help="Whether the use TabPC's preprocessing pipeline or just the sklearn OrdinalEncoder. (Default: False)")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="The maximum number of failed generations to allow before skipping to the next sample.")
    args = parser.parse_args()

    datasets = ["adult", "beijing", "default", "diabetes", "magic", "news", "shoppers"] if args.dataset == "all" else [args.dataset]

    generation_times = {}

    for dataset in datasets:
        generation_times[dataset] = {}
        for repeat_idx in range(args.num_repeats):
            print(f"Dataset: {dataset}, Repeat Index: {repeat_idx}")
            failure_count = 0
            while failure_count < args.max_failures:
                print(f"Failure_count: {failure_count}")
                try:
                    start_time = time.time()
                    generate_synthetic_dataset_tabpfn(
                        dataset_name=dataset,
                        repeat_idx=repeat_idx,
                        output_data_dir=args.output_data_dir,
                        save_raw=args.save_raw,
                        overwrite=args.overwrite,
                        n_permutations=args.n_permutations,
                        use_only_ordinal=args.use_only_ordinal
                    )
                    elapsed_time = time.time() - start_time
                    generation_times[dataset][repeat_idx] = elapsed_time
                    break
                except Exception as e:
                    print(f"Error for dataset {dataset} and repeat {repeat_idx}: {e}")
                    failure_count += 1
            if failure_count == args.max_failures:
                generation_times[dataset][repeat_idx] = -1
    
    # Save generation times
    gen_times_pattern = f"generation_times_{args.dataset}_perms_{args.n_permutations}"
    gen_times_path = os.path.join(args.output_data_dir, f"{gen_times_pattern}.json")
    if not args.overwrite:
        while os.path.exists(gen_times_path):
            gen_times_pattern += "_new"
            gen_times_path = os.path.join(args.output_data_dir, f"{gen_times_pattern}.json")
    with open(gen_times_path, "w") as f:
        json.dump(generation_times, f, indent=4)
