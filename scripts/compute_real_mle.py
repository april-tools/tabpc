import sys
sys.path.append(".")

from src.util import store_json, load_json, find_files, wait_available_gpu, printc
from src.evaluation import generative_performance_metric
import os
from src.metrics.metrics import *
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import torch
import gc

# Add this line to set the start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

METRICS = {
    Mle() : 2,
}

OVERWRITE = False

# GENERATED_DATA_FOLDER = 'artifacts/models'
ORIGINAL_DATA_FOLDER = 'data'
TARGET_FOLDER = 'artifacts/mle_results'
NAMES = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'diabetes']

def find_related_dataset(file_path: str) -> str:
    for name in NAMES:
        if name in file_path:
            return name
    raise ValueError(f"Unknown dataset for file: {file_path}")

def result_exists(generated_data_path, generated_data_folder, target_folder, metric):
    result_dir = os.path.join(os.path.dirname(generated_data_path).replace(generated_data_folder, target_folder), metric.name())
    result_path = os.path.join(result_dir, os.path.basename(generated_data_path).replace('.csv', '.json'))
    return os.path.exists(result_path)

def evaluate_single_file(generated_data_path, generated_data_folder, original_data_folder, target_folder, metric, overwrite):
    printc(f"Evaluating {generated_data_path} with metric {metric.name()}", 'yellow')
    result_dir = os.path.join(os.path.dirname(generated_data_path).replace(generated_data_folder, target_folder), metric.name())
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, os.path.basename(generated_data_path).replace('.csv', '.json'))

    if (not os.path.exists(result_path)) or overwrite:
        original_data_path = os.path.join(original_data_folder, find_related_dataset(generated_data_path))
        
        try:
            metric_result = generative_performance_metric(
                original=original_data_path,
                generated=generated_data_path,
                metric=metric,
                device='cpu' if not metric.requires_gpu() else wait_available_gpu(min_free_mem=0.5, min_free_usage=0.9, wait=True),
                verbose=False,
            )
            store_json(d=metric_result, file=result_path)
        except Exception as e:
            printc(f"Error evaluating {generated_data_path} with metric {metric.name()}: {e}", 'red')
            metric_result = {"error": str(e)}
        finally:
            torch.cuda.empty_cache()
            gc.collect()


def evaluate(*, generated_data_folder, original_data_folder, target_folder, metrics: dict[Metric, int], overwrite=False, pattern="*.csv", randomize_order=False):
    os.makedirs(target_folder, exist_ok=True)
    generated_files = find_files(starting_folder=generated_data_folder, pattern=pattern)
    
    # exclude files in 'old' folders
    generated_files = [f for f in generated_files if 'old' not in f]
    
    # exclude files that already have results
    # generated_files = [f for f in generated_files if not all(result_exists(f, generated_data_folder, target_folder, metric) for metric in metrics.keys())]
    
    print(f"Found {len(generated_files)} generated files in {generated_data_folder}")
    
    if randomize_order:
        random.shuffle(generated_files)

    for metric, max_workers in metrics.items():
        print(f"Processing metric: {metric.name()}")

        partial_func = partial(
            evaluate_single_file,
            generated_data_folder=generated_data_folder,
            original_data_folder=original_data_folder,
            target_folder=target_folder,
            metric=metric,
            overwrite=overwrite
        )

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm.tqdm(executor.map(partial_func, generated_files), total=len(generated_files), desc=f"Evaluating {metric.name()}"))


def get_train_test_info_data(dataset_folder: str):
    info = load_json(f"{dataset_folder}/info.json")
    df_train = pd.read_csv(f"{dataset_folder}/train.csv")
    df_test = pd.read_csv(f"{dataset_folder}/test.csv")
    return df_train, df_test, info


def generative_performance_metric(
    *,
    original: str,
    generated: str,
    device: str = "cpu",
    target_path: str | None = None,
    metric: Metric,
    verbose: bool = False,
) -> dict:
    if os.path.isdir(original):
        df_original, df_test, info = get_train_test_info_data(original)
    else:
        df_original = pd.read_csv(original)
        df_test = None
        info = None
    df_generated = pd.read_csv(generated)

    if verbose:
        printc(f"Computing {metric.name()}...", "yellow")
    return metric(df_original, df_generated, df_test, info, device=device)


if __name__ == "__main__":
    results = []
    for dataset in tqdm(NAMES, desc="Processing datasets"):
        for metric, max_workers in tqdm(METRICS.items(), desc="Processing metrics"):
            for repeat_idx in range(5):
                printc(f"Evaluating dataset: {dataset}, metric: {metric.name()}", 'yellow')

                train_data_path = f'data/{dataset}/train.csv'
                test_data_path = f'data/{dataset}/test.csv'
                try:
                    metric_result = generative_performance_metric(
                        original=f"data/{dataset}",
                        generated=train_data_path,
                        metric=metric,
                        device='cpu' if not metric.requires_gpu() else wait_available_gpu(min_free_mem=0.5, min_free_usage=0.9, wait=True),
                        verbose=True,
                    )
                except Exception as e:
                    printc(f"Error evaluating {train_data_path} with metric {metric.name()}: {e}", 'red')
                    metric_result = {"error": np.nan}
                results.append({
                    'dataset': dataset,
                    f'{metric.name()}': metric_result,
                    'repeat_idx': repeat_idx,
                })

    # Save results to a CSV file
    df = pd.DataFrame(results)
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    result_path = os.path.join(TARGET_FOLDER, f"real_data_{metric.name()}.csv")
    df.to_csv(result_path, index=False)