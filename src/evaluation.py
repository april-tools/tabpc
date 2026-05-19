import os
from typing import List

import pandas as pd

from src.distribution_plots import (BivariatePlot, DistributionPlot,
                                    MarginalsPlot)
from src.metrics.metrics import *
from src.util import load_json, printc

PLOTS_FOLDER_NAME = "plots"
DEFAULT_METRICS = [
    C2ST(),
    Mle(),
    LegacyDensity(),
    QuantileDcr(),
    MI_l1(),
]
DEFAULT_PLOTS = [MarginalsPlot(), BivariatePlot()]
IMAGES_FORMAT = "png"


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

        if "test.csv" in original:
            # Read info from the parent directory
            parent_dir = os.path.dirname(original)
            info_path = os.path.join(parent_dir, "info.json")
            if os.path.exists(info_path):
                info = load_json(info_path)
            else:
                printc(f"Warning: info.json for {original} not found in {parent_dir}. Some metrics may not work properly.", "red")
    df_generated = pd.read_csv(generated)

    if verbose:
        printc(f"Computing {metric.name()}...", "yellow")
    return metric(df_original, df_generated, df_test, info, device=device)


def generative_performance_metrics(
    original: str,
    generated: str,
    device: str = "cpu",
    target_path: str | None = None,
    metrics: List[Metric] = DEFAULT_METRICS,
    verbose: bool = True,
) -> dict:
    if os.path.isdir(original):
        df_original, df_test, info = get_train_test_info_data(original)
    else:
        df_original = pd.read_csv(original)
        df_test = None
        info = None
    df_generated = pd.read_csv(generated)

    results = {}
    for metric in metrics:
        if verbose:
            printc(f"Computing {metric.name()}...", "yellow")
        results[metric.name()] = metric(
            df_original, df_generated, df_test, info, device=device
        )

    return results


def generative_performance_plots(
    original: str,
    generated: str,
    target_folder: str,
    plots: List[DistributionPlot] = DEFAULT_PLOTS,
    wandb_run=None,
) -> None:
    if os.path.isdir(original):
        df_original = get_train_test_info_data(original)[0]
    else:
        df_original = pd.read_csv(original)
    df_generated = pd.read_csv(generated)

    if len(plots) > 0:
        plots_folder = os.path.join(target_folder, PLOTS_FOLDER_NAME)
        os.makedirs(plots_folder, exist_ok=True)
        for plot_function in plots:
            plot_function(
                df_original,
                df_generated,
                folder=plots_folder,
                format=IMAGES_FORMAT,
                wandb_run=wandb_run,
            )
