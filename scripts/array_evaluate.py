import sys
import os
import random
from tqdm import tqdm
import multiprocessing
import torch
import gc
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.append(".")
from src.util import store_json, find_files, wait_available_gpu, printc
from src.evaluation import generative_performance_metric
from src.metrics.metrics import *

# Add this line to set the start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

METRICS = {
    LegacyDensity() : 20,
    MI_l1() : 20,
    C2ST() : 2,
}

ALL_METRICS = {
    LegacyDensity() : 20,
    MI_l1() : 20,
    C2ST() : 2,
    Mle() : 2,
    QuantileDcr() : 2,
}

OVERWRITE = False

GENERATED_DATA_FOLDER = 'artifacts/generated_data'
ORIGINAL_DATA_FOLDER = 'data'
TARGET_FOLDER = 'artifacts/metrics_results'
NAMES = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'diabetes',
        'insurance']

def find_related_dataset(file_path: str) -> str:
    for name in NAMES:
        if name in file_path:
            return name
    raise ValueError(f"Unknown dataset for file: {file_path}")

def evaluate_single_file(generated_data_path, generated_data_folder, original_data_folder, target_folder, metric, overwrite, use_train_set=True):
    #printc(f"Evaluating {generated_data_path} with metric {metric.name()}", 'yellow')
    result_dir = os.path.join(os.path.dirname(generated_data_path).replace(generated_data_folder, target_folder), metric.name())
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, os.path.basename(generated_data_path).replace('.csv', '.json'))

    if (not os.path.exists(result_path)) or overwrite:
        if use_train_set:
            original_data_path = os.path.join(original_data_folder, find_related_dataset(generated_data_path))
        else: # Evaluate against the test set instead of the train set
            original_data_path = os.path.join(original_data_folder, find_related_dataset(generated_data_path), 'test.csv')
        
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


def evaluate(*, generated_data_folder, original_data_folder, target_folder, metrics: dict[Metric, int], overwrite=False, pattern="*.csv", randomize_order=False, use_train_set=True):
    os.makedirs(target_folder, exist_ok=True)
    generated_files = find_files(starting_folder=generated_data_folder, pattern=pattern)
    
    # exclude files in 'old' folders
    # generated_files = [f for f in generated_files if 'old' not in f]
    
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
            overwrite=overwrite,
            use_train_set=use_train_set
        )

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(partial_func, generated_files), total=len(generated_files), desc=f"Evaluating {metric.name()}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated datasets with various metrics.")
    parser.add_argument('--pattern', type=str, default="*.csv", help='Glob pattern to match generated data files (default: "*.csv")')
    parser.add_argument('--target-folder', type=str, default=TARGET_FOLDER, help=f'Target folder to store results (default: {TARGET_FOLDER})')
    parser.add_argument('--original-data-folder', type=str, default=ORIGINAL_DATA_FOLDER, help=f'Folder containing original datasets (default: {ORIGINAL_DATA_FOLDER})')
    parser.add_argument('--generated-data-folder', type=str, default=GENERATED_DATA_FOLDER, help=f'Folder containing generated data (default: {GENERATED_DATA_FOLDER})')
    parser.add_argument('--overwrite', action='store_true', default=OVERWRITE, help='Whether to overwrite existing results (default: False)')
    parser.add_argument('--base-metrics', action='store_true', help='Whether to evaluate only the base metrics instead of all metrics (default: False)')
    args = parser.parse_args()

    pattern = args.pattern
    target_folder = args.target_folder
    generated_data_folder = args.generated_data_folder
    original_data_folder = args.original_data_folder
    overwrite = args.overwrite

    metrics = METRICS if args.base_metrics else ALL_METRICS

    evaluate(
        generated_data_folder=generated_data_folder,
        original_data_folder=original_data_folder,
        target_folder=target_folder,
        metrics=metrics,
        overwrite=overwrite,
        randomize_order=True,
        pattern=pattern,
    )
