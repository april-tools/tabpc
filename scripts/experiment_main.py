import argparse
import gc
import sys
import os
from functools import lru_cache

sys.path.append(".")
from scripts.sample import main as sample_main
from scripts.array_evaluate import evaluate as evaluate_metrics
from scripts.summarize_metrics_results import summarize_metrics
from scripts.update_article_tables import update_article_tables
from src.util import Timer, load_json
from src.metrics.metrics import *

RESULTS_FOLDER = 'article_material/our_results'
GENERATED_DATA_FOLDER = 'artifacts/generated_data'
METRICS_FOLDER = 'artifacts/metrics_results'
ARTICLE_MODELS_FOLDER = 'artifacts/article_models'
METRICS = {
    LegacyDensity() : 20,
    MI_l1() : 20,
    C2ST() : 2,
    Mle() : 2,
    QuantileDcr() : 2,
}
DIGITS = 4
NUM_RUNS = 5


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


@lru_cache(maxsize=None)
def get_method_name(experiment_folder: str) -> str:
    results = load_json(f"{experiment_folder}/results.json")
    method_name = results['config']['model']['type']
    if method_name == 'ProbabilisticCircuit':
        return 'TabPC'
    if method_name == 'OnlyMarginals':
        q_norm = "QuantileNormalizer" in results['config']['model']['preprocessors']
        nan_handling = results['config']['model']['preprocessors']['NanHandler']['handle_inflated_values']
        if "random_fit" in experiment_folder:
            return "FullyFactorizedRandom"
        elif q_norm and nan_handling:
            return "FullyFactorizedPreprocessed"
        elif q_norm:
            return "FullyFactorizedQuantileNorm"
        elif nan_handling:
            return "FullyFactorizedNanHandling"
        else:
            return "SimpleFactors"
    return method_name


@lru_cache(maxsize=None)
def get_dataset_name(experiment_folder: str) -> str:
    results = load_json(f"{experiment_folder}/results.json")
    dataset_path = results['config']['dataset_path']
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    return dataset_name


def get_run_idx(experiment_folder: str) -> int:
    # Experiment folders are stored as artifacts/article_models/run_{run_idx}/{timestamp}_{method}_{dataset}/
    run_idx_str = experiment_folder.split('/')[2].split('_')[-1]
    try:
        run_idx = int(run_idx_str)
    except ValueError:
        raise ValueError(f"Could not parse run index from folder name: {experiment_folder}")
    return run_idx


def copy_samples_to_generated_data_folder(experiment_folder: str):
    method = get_method_name(experiment_folder)
    dataset = get_dataset_name(experiment_folder)
    run_idx = get_run_idx(experiment_folder)

    src = f"{experiment_folder}/new_samples/samples/*"
    dst = f"{GENERATED_DATA_FOLDER}/{method}/{dataset}/run_{run_idx}/"
    os.makedirs(dst, exist_ok=True)
    os.system(f'rsync -rmv {src} {dst}')

    
def train_fully_factorized_models():
    from train_scripts.fully_factorized import main as train_marginals
    print('Training Fully Factorized models...')
    tabdiff_paths = [
        f"data/{name}" for name in ["adult", "beijing", "default", "shoppers", "magic", "diabetes", "news"]
    ]
    free_memory()
    for run_idx in range(NUM_RUNS):
        train_marginals(paths=tabdiff_paths, target_folder=f"{ARTICLE_MODELS_FOLDER}/run_{run_idx}", do_extras=True)


def train_shallow_mixture_models():
    from train_scripts.shallow_mixture import main as train_shallow_mixtures
    print('Training Shallow Mixture models...')
    tabdiff_paths = [
        f"data/{name}" for name in ["adult", "beijing", "default", "shoppers", "magic", "diabetes", "news"]
    ]
    free_memory()
    for run_idx in range(NUM_RUNS):
        train_shallow_mixtures(paths=tabdiff_paths, target_folder=f"{ARTICLE_MODELS_FOLDER}/run_{run_idx}")


def train_tabpc():
    from train_scripts.pc_sota import main as train_pc_sota
    print('Training TabPC models...')
    tabdiff_paths = [
        f"data/{name}" for name in ["adult", "beijing", "default", "shoppers", "magic", "diabetes", "news"]
    ]
    free_memory()
    for run_idx in range(NUM_RUNS):
        train_pc_sota(paths=tabdiff_paths, target_folder=f"{ARTICLE_MODELS_FOLDER}/run_{run_idx}")


def find_experiments() -> list[str]:
    # Find all experiment folders
    experiment_folders = []
    for root, _, files in os.walk(ARTICLE_MODELS_FOLDER):
        if 'results.json' in files:
            experiment_folders.append(root)
    return experiment_folders


def run_experiments() -> list[str]:
    # Train all models and return list of experiment folders
    train_fully_factorized_models()
    train_shallow_mixture_models()
    train_tabpc()


def update_results(*,
                   experiment_folders: list[str] = [],
                   train_models: str = "none",
                   sample: bool = False,
                   evaluate: bool = False,
                   evaluate_baselines: bool = False,
                   update_tables: bool = False,
                   ) -> None:

    if train_models == "all":
        run_experiments()

    if train_models == "tab_pc":
        train_tabpc()

    if train_models == "shallow_mixture":
        train_shallow_mixture_models()

    if train_models == "fully_factorized":
        train_fully_factorized_models()

    if experiment_folders == []:
        experiment_folders = find_experiments()
            
    # sample all variations
    if sample:
        print('Sampling all variations...')
        sample_main(experiment_paths=experiment_folders, num_samples=1) # 1 sample per model seed (5 models per dataset => 5 samples per dataset)

    # move generated datasets to common folder
    print('Moving generated datasets...')
    for experiment_folder in experiment_folders:
        copy_samples_to_generated_data_folder(experiment_folder)

    # run evaluation metrics
    if evaluate:
        print('Running evaluation metrics...')
        for model in ['TabPC', 'ShallowMixture', 'FullyFactorizedPreprocessed', 'FullyFactorizedRandom']:
            print(f'Evaluating model: {model}...')
            evaluate_metrics(
                generated_data_folder=GENERATED_DATA_FOLDER,
                original_data_folder='data',
                target_folder=METRICS_FOLDER,
                metrics=METRICS,
                overwrite=False,
                randomize_order=True,
                pattern=f"*{model}*.csv",
            )

    if evaluate_baselines:
        # Requires generated data for baseline methods to be stored in GENERATED_DATA_FOLDER
        print('Running evaluation metrics for baselines...')
        for baseline_model in ['CoDi', 'TVAE', 'CTGAN', 'GReaT', 'STaSy', 'TabDiff', 'TabSyn']:
            print(f'Evaluating baseline model: {baseline_model}...')
            evaluate_metrics(
                generated_data_folder=GENERATED_DATA_FOLDER,
                original_data_folder='data',
                target_folder=METRICS_FOLDER,
                metrics=METRICS,
                overwrite=False,
                randomize_order=True,
                pattern=f"*{baseline_model}*.csv",
            )

    # summarize results and update results csv
    print('Summarizing results and updating results csv...')
    try:
        summarize_metrics(
            results_folder='artifacts/metrics_results',
            target_folder='artifacts/metrics_results_summary',
            models_folder=ARTICLE_MODELS_FOLDER,
            new_protocol=True,
        )
    except Exception as e:
        print(f"Error summarizing metrics results: {e}")
        print(f"Check that metric results exist in {METRICS_FOLDER} and are in the correct format.")
        print("Skipping metric summarization.")

    # rank CSVs and use these to and update latex tables
    if update_tables:
        print('Updating latex tables...')
        update_article_tables(
            summary_folder='artifacts/metrics_results_summary',
            result_csvs_folder='article_material/our_results',
            ranked_result_csvs_folder='artifacts/ranked_results',
            tables_folder='article_material/article_tables',
            digits=DIGITS,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Update evaluation results from experiments")
    parser.add_argument(
        "--experiment-folders",
        "-f",
        nargs='*',
        default=[],
        help=("Paths to experiment folders. If omitted, all experiments under"
              f" '{ARTICLE_MODELS_FOLDER}' will be used."),
    )
    parser.add_argument("--train-models", "-t", choices=["all", "tab_pc", "shallow_mixture", "fully_factorized", "none"], default="none", help="Train models before evaluating")
    parser.add_argument("--sample", "-s", action="store_true", help="Run sampling", default=False)
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run evaluation metrics", default=False)
    parser.add_argument("--evaluate-baselines", "-b", action="store_true", help="Run evaluation metrics for baselines", default=False)
    parser.add_argument("--update-tables", action="store_true", help="Update latex tables", default=False)

    args = parser.parse_args()

    # Ensure experiment_folders is a list (argparse gives [] by default)
    exp_folders = args.experiment_folders or []

    with Timer() as generation_timer:
        update_results(
            experiment_folders=exp_folders,
            train_models=args.train_models,
            sample=args.sample,
            evaluate=args.evaluate,
            evaluate_baselines=args.evaluate_baselines,
            update_tables=args.update_tables,
        )

    print(f"Results updated in {generation_timer.elapsed:.2f} seconds.")
