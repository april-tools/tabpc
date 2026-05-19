import os
import gc
import sys
import argparse
import pandas as pd
import numpy as np
import torch

sys.path.append(".")
sys.path.append("cirkit")
from train_scripts.pc_sota import main as train_pc
from scripts.array_evaluate import evaluate as evaluate_metrics
from scripts.scrape_conditional_experiment import main as scrape_conditional
from scripts.plot_conditional_experiment import plot_all_conditional_performance
from src.util import Timer, load_json, get_available_device, edit_json, load_json, store_json
from src.metrics.metrics import *
from src.experiment_config import ExperimentConfig, load_experiment
from src.models.probabilistic_circuit import ProbabilisticCircuit
from src.models.shallow_mixture import ShallowMixture

COND_MODELS_FOLDER = 'artifacts/cond_sampling_models'
COND_METRICS_FOLDER = 'artifacts/cond_sampling_metrics_results'
COND_RESULTS_FOLDER = 'article_material/cond_sampling_results'

METRICS = {
    LegacyDensity() : 20,
    MI_l1() : 20,
    C2ST() : 2,
}

TABDIFF_PATHS = [
    f"data/{name}" for name in ["adult", "beijing", "default", "diabetes", "magic",  "news", "shoppers"]
]

COND_PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

DIGITS = 4


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def get(d: dict, keys: tuple):
    try:
        for key in keys:
            d = d[key]
    except KeyError:
        return None
    return d


def randomly_mask_values(df: pd.DataFrame, *, mask_fraction: float) -> pd.DataFrame:
    """
    Randomly mask a fraction of values in a DataFrame by setting them to NaN.

    This implementation:
    - validates mask_fraction
    - builds a boolean mask using flat indices (works for mixed dtypes)
    - applies pandas.DataFrame.mask to set values to NaN
    """

    df_masked = df.copy()
    mask = np.random.rand(*df.shape) < mask_fraction
    return df_masked.mask(mask), mask


def conditional_sample(
        experiment_path: str,
        n_samples: int,
        overwrite: bool = False,
        device = None,
        uncond_batch_size: int | None = None, # Setting None enables adaptive batch size, which we find tends to work better for unconditional sampling
        cond_batch_size: int | None = 10, # Conditional batch size has been chosen to fit on our GPUs, but can be tuned for better performance
        use_train_set_for_conditioning: bool = False # Whether to use the train set for conditioning (if False, use the test set)
        ) -> str:
    """
    Sample conditioned on the test set for an increasing number of features being conditioned, from 0% to 100% with a step of 10%.
     - For 0% conditioning, we sample unconditionally from the model.
     - For 100% conditioning, we simply return the test set as the generated samples, since all features are conditioned.
     - For intermediate percentages, we randomly mask the corresponding percentage of values.
    The generated samples are saved in csv files in a folder named after the experiment, with subfolders for each conditioning percentage.
    The generation times are saved in a json file in the same folder.
    Also, the masks used for conditioning are saved as numpy arrays for later use.
    """
    DEVICE = device if device is not None else get_available_device(mem_required=0.9, verbose=True)

    exp_config: ExperimentConfig = load_experiment(experiment_path)
    
    # create folder for samples
    samples_path = os.path.join(experiment_path, f"new_samples")
    os.makedirs(samples_path, exist_ok=True)

    for conditioning_percentage in COND_PERCENTAGES:

        print(f"Sampling with conditioning percentage: {conditioning_percentage:.2f}")
        
        csv_samples_path = os.path.join(samples_path, f"samples_cond{conditioning_percentage:.2f}")
        os.makedirs(csv_samples_path, exist_ok=True)
        
        print(f"Sampling {n_samples} times from model...")
        
        # load model
        model = exp_config.model
        
        if isinstance(model, ProbabilisticCircuit) or isinstance(model, ShallowMixture):
            model.circuit.to(DEVICE)

        times_file_name = os.path.join(samples_path, "generation_times.json") if conditioning_percentage is None else os.path.join(samples_path, f"generation_times_cond{conditioning_percentage:.2f}.json")

        # generate samples
        if overwrite and os.path.exists(times_file_name):
            os.remove(times_file_name)

        if conditioning_percentage == 1.0:
            n_samples = 1  # only one sample needed, all values are conditioned

        # Initialize df_to_mask
        df_to_mask = None
        if use_train_set_for_conditioning: # Get training set for conditioning if specified
            df_to_mask = exp_config.get_train_data()
        else: # Otherwise get test set
            df_to_mask = exp_config.get_test_data()
        
        # Generate a dataset of the same size as the chosen conditioning set (test or train)
        num_to_generate = len(df_to_mask)

        for i in range(n_samples):

            if use_train_set_for_conditioning:
                name = f"train_sample_{i}"
            else:
                name = f"test_sample_{i}"
            if not overwrite and os.path.exists(os.path.join(csv_samples_path, f"{name}.csv")):
                print(f"Sample {i} already exists, skipping...")
                continue
            
            print(f"Generating sample {i+1}/{n_samples}...")
            with Timer() as generation_timer:
                if conditioning_percentage == 0.0:
                    df = model.generate(device=DEVICE, batch_size=uncond_batch_size, num_to_generate=num_to_generate)
                    mask = np.ones_like(df, dtype=bool)  # all values are masked (i.e., no conditioning)
                elif conditioning_percentage == 1.0:
                    df = df_to_mask.copy()  # all values are conditioned, so we return the test set as the generated samples
                    mask = np.zeros_like(df, dtype=bool)  # no values are masked (i.e., full conditioning)
                else:
                    if not isinstance(model, ProbabilisticCircuit):
                        raise ValueError("Conditional generation is only supported for ProbabilisticCircuit models.")

                    df_with_missing_values, mask = randomly_mask_values(df_to_mask, mask_fraction=1-conditioning_percentage)

                    df = model.conditional_generation(device=DEVICE, batch_size=cond_batch_size, condition=df_with_missing_values, num_to_generate=num_to_generate)
                
                # Save generated samples to csv
                df.to_csv(os.path.join(csv_samples_path, f"{name}.csv"), index=False)
                
                # Save mask for later analysis / use
                np.save(os.path.join(csv_samples_path, f"{name}_mask.npy"), mask)

            with edit_json(filename=times_file_name) as data:
                data[i] = generation_timer.elapsed

        if os.path.exists(times_file_name):
            d = load_json(times_file_name)
            if len(d) == n_samples and conditioning_percentage is None:
                times = np.array(list(d.values()))
                avg_time = np.mean(times)
                std_time = np.std(times)
                store_json({"average": avg_time, "std": std_time}, file=os.path.join(samples_path, "generation_times_summary.json"))


def impute_w_saved_masks(
        experiment_path: str,
        n_samples: int,
        overwrite: bool = False,
        device = None,
        use_train_set_for_conditioning: bool = False # Whether to use the test set for conditioning (if False, use the test set)
        ) -> str:
    """
    Impute the masked values with simple statistics (mean for continuous, mode for categorical) using the saved masks from the conditional sampling step.
    The imputed samples are saved in csv files in a folder named after the experiment, with subfolders for each conditioning percentage.
     - For 0% conditioning, we do not generate any imputed samples since there are no masked values.
     - For 100% conditioning, we simply return the test set as the imputed samples, since all features are conditioned.
     - For intermediate percentages, we impute the masked values with mean for continuous, mode for categorical
    """
    DEVICE = device if device is not None else get_available_device(mem_required=0.9, verbose=True)

    exp_config: ExperimentConfig = load_experiment(experiment_path)
    
    # create folder for imputed samples
    # store these in experiment_path/new_samples/imputed/samples_cond{conditioning_percentage}/sample_{i}.csv
    samples_path = os.path.join(experiment_path, f"new_samples")
    imputed_path = os.path.join(samples_path, "imputed")
    os.makedirs(imputed_path, exist_ok=True)

    for conditioning_percentage in COND_PERCENTAGES[1:]:  # skip 0.0 since we cannot impute with no observed data

        print(f"Imputing with conditioning percentage: {conditioning_percentage:.2f}")
        
        csv_samples_path = os.path.join(imputed_path, f"imputed_cond{conditioning_percentage:.2f}")
        os.makedirs(csv_samples_path, exist_ok=True)

        print(f"Saving imputed samples to {csv_samples_path}")
        
        print(f"Imputing {n_samples} times from model...")
        
        # load model
        model = exp_config.model
        
        if isinstance(model, ProbabilisticCircuit) or isinstance(model, ShallowMixture):
            model.circuit.to(DEVICE)

        df_to_mask = None
        if use_train_set_for_conditioning: # Get training set for conditioning if specified
            df_to_mask = exp_config.get_train_data()
        else: # Otherwise get test set
            df_to_mask = exp_config.get_test_data()

        for i in range(n_samples):

            if use_train_set_for_conditioning:
                name = f"train_imputed_{i}"
            else:
                name = f"test_imputed_{i}"
            
            if not overwrite and os.path.exists(os.path.join(csv_samples_path, f"{name}.csv")):
                print(f"Imputed sample {i} already exists, skipping...")
                continue
            
            print(f"Imputing sample {i+1}/{n_samples}...")
            if use_train_set_for_conditioning:
                mask_name = f"train_sample_{i}_mask.npy"
            else:
                mask_name = f"test_sample_{i}_mask.npy"
            mask = np.load(os.path.join(samples_path, f"samples_cond{conditioning_percentage:.2f}", f"{mask_name}"))
            df_with_missing_values = df_to_mask.mask(mask)

            # Get imputation values (mean for continuous, mode for categorical)
            imputes = df_with_missing_values.convert_dtypes().apply(lambda x: x.mean() if pd.api.types.is_float_dtype(x) else x.mode(), axis=0)
            imputes = imputes.iloc[0] if isinstance(imputes, pd.DataFrame) else imputes
            
            # Impute the missing values in df_with_missing_values with the computed imputes
            df_imputed = df_with_missing_values.fillna(dict(imputes), axis=0)
            df_imputed.to_csv(os.path.join(csv_samples_path, f"{name}.csv"), index=False)


# Experiment flow:
# 1) Train SotA TabPC models so that they are in a separate folder
# 2) Conditionally sample from 0% to 100% of the features being conditioned, with a step of 10%
# 3) Impute the masked values with simple statistics (mean for continuous, mode for categorical)
# 4) Evaluate the samples with the chosen metrics
# 5) Summarize results in csv files

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the conditional experiments for TabPC.')
    parser.add_argument('--models-folder', type=str, help='Target folder for trained models', default=COND_MODELS_FOLDER)
    parser.add_argument('--metrics-folder', type=str, help='Target folder for metrics', default=COND_METRICS_FOLDER)
    parser.add_argument('--results-folder', type=str, help='Target folder for results', default=COND_RESULTS_FOLDER)
    parser.add_argument('--plots-folder', type=str, help='Target folder for plots', default=os.path.join(COND_RESULTS_FOLDER, "plots"))
    parser.add_argument('--seed', help='Random seed (int or None; default: 0)', default=0)
    parser.add_argument('--dataset', type=str, help='Dataset to run the experiment on (default: all)', default="all")
    parser.add_argument('--train', '-t', action='store_true', help='Whether to run the training of models (default: False)')
    parser.add_argument('--sample', '-s', action='store_true', help='Whether to run the conditional sampling (default: False)')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Whether to run the evaluation of the samples (default: False)')
    parser.add_argument('--impute', '-i', action='store_true', help='Whether to run the imputation of the masks (default: False)')
    parser.add_argument('--scrape', '-r', action='store_true', help='Whether to scrape the results and summarize them in csv files (default: False)')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Whether to overwrite existing samples and times (default: False)')
    parser.add_argument('--uncond-batch-size', type=int, default=None, help='Batch size for sampling and imputation (default: None, which enables automatic selection based on GPU memory)')
    parser.add_argument('--cond-batch-size', type=int, default=10, help='Batch size for conditional sampling (default: 10, can be tuned for better performance)')
    parser.add_argument('--use-train-set-for-conditioning', action='store_true', help='Whether to use the train set for conditioning (default: False, which means using the test set for conditioning)')
    parser.add_argument('--do-plots', action='store_true', help='Whether to generate plots from the results after scraping (default: False)')

    args = parser.parse_args()

    # Create specified folders if they don't exist
    os.makedirs(args.models_folder, exist_ok=True)
    os.makedirs(args.metrics_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)
    # 1) Train SotA TabPC models so that they are in a separate folder
    if args.train:
        if args.dataset not in ["all", "adult", "beijing", "default", "diabetes", "magic",  "news", "shoppers"]:
            raise ValueError(f"Invalid dataset name: {args.dataset}. Must be one of: all, adult, beijing, default, diabetes, magic, news, shoppers.")
        
        datasets = TABDIFF_PATHS if args.dataset == "all" else [f"data/{args.dataset}"]
        for dataset_path in datasets:
            print(f"Training TabPC on {dataset_path.split('/')[-1]} dataset")
            train_pc(paths=[dataset_path], seed=args.seed, target_folder=args.models_folder)
            free_memory()

    experiment_paths = os.listdir(args.models_folder)
    experiment_paths = [os.path.join(args.models_folder, path) for path in experiment_paths]

    # 2) Conditionally sample from 0% to 100% of the features being conditioned, with a step of 10%
    if args.sample:
        for experiment_path in experiment_paths:
            try:
                conditional_sample(
                    experiment_path,
                    n_samples=1,
                    overwrite=args.overwrite,
                    uncond_batch_size=args.uncond_batch_size,
                    cond_batch_size=args.cond_batch_size,
                    use_train_set_for_conditioning=args.use_train_set_for_conditioning
                    )
            except Exception as e:
                print(f"Error during conditional sampling for experiment {experiment_path}: {e}")
            free_memory()

    # 3) Impute the masked values with simple statistics (mean for continuous, mode for categorical)
    if args.impute:
        for experiment_path in experiment_paths:
            try:
                impute_w_saved_masks(
                    experiment_path,
                    n_samples=1,
                    overwrite=args.overwrite,
                    use_train_set_for_conditioning=args.use_train_set_for_conditioning
                    )
            except Exception as e:
                print(f"Error during imputation for experiment {experiment_path}: {e}")
            free_memory()

    # 4) Evaluate the samples with the chosen metrics
    if args.evaluate:
        print("Evaluating all generated samples...")
        
        print("Evaluating generated samples conditioned on training set...")
        # Evaluate generated samples conditioned on training set
        evaluate_metrics(generated_data_folder=args.models_folder, original_data_folder="data", metrics=METRICS, target_folder=args.metrics_folder, pattern="*train_sample_0.csv", use_train_set=True)
        print("Evaluating generated samples conditioned on test set...")
        # Evaluate generated samples conditioned on test set
        evaluate_metrics(generated_data_folder=args.models_folder, original_data_folder="data", metrics=METRICS, target_folder=args.metrics_folder, pattern="*test_sample_0.csv", use_train_set=False)

        print("Evaluating all imputed samples...")

        print("Evaluating imputed samples conditioned on training set...")
        # Evaluate imputed samples
        evaluate_metrics(generated_data_folder=args.models_folder, original_data_folder="data", metrics=METRICS, target_folder=args.metrics_folder, pattern="*train_imputed_0.csv", use_train_set=True)
        print("Evaluating imputed samples conditioned on test set...")
        evaluate_metrics(generated_data_folder=args.models_folder, original_data_folder="data", metrics=METRICS, target_folder=args.metrics_folder, pattern="*test_imputed_0.csv", use_train_set=False)

        free_memory()

    # 5) Summarize results in csv files
    if args.scrape:
        scrape_conditional(
            metrics_path=args.metrics_folder,
            output_folder=args.results_folder,
            metrics_dict=METRICS
        )

    # 6) Generate plots from the results after scraping
    if args.do_plots:
        for metric_name in [metric.name() for metric in METRICS.keys()]:                
            # Plot results when conditioning on the test set
            plot_all_conditional_performance(
                results_dir=args.results_folder,
                metric_name=metric_name,
                out_dir=args.plots_folder,
                use_test_results=True
            )

            # Plot results when conditioning on the training set
            plot_all_conditional_performance(
                results_dir=args.results_folder,
                metric_name=metric_name,
                out_dir=args.plots_folder,
                use_test_results=False
            )
