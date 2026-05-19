import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

from src.constants import MODELS_FOLDER
from src.evaluation import (generative_performance_metrics,
                            generative_performance_plots)
from src.experiment_config import ExperimentConfig
from src.util import create_experiment_folder, printc, store_json


def run_experiment(
    experiment: ExperimentConfig,
    wandb_run=None,
    do_plots: bool = True,
    do_metrics: bool = True,
    target_folder: str | Path | None = None,
    use_codecarbon: bool = False,
) -> Tuple[str, dict]:
    if target_folder is None:
        target_folder = MODELS_FOLDER
    # Create the experiment folder
    experiment_folder = Path(
        create_experiment_folder(path=Path(target_folder), postfix=experiment.name)
    )

    # Copy the current script to the experiment folder
    shutil.copyfile(os.path.abspath(sys.argv[0]), experiment_folder / "main.py")

    # Generate data
    generated_data_path, train_report = experiment.generate(
        folder=experiment_folder, wandb_run=wandb_run, use_codecarbon=use_codecarbon
    )

    # store configs as a separate json
    store_json(
        experiment.to_dict(),
        file=str(experiment_folder / "config.json"),
    )

    experiment.store(str(experiment_folder))

    # check that the experiment was stored correctly
    from src.experiment_config import load_experiment

    _ = load_experiment(str(experiment_folder))

    results = {}

    # Compute performance metrics
    if do_metrics:
        print("Computing performance metrics...")
        metrics = generative_performance_metrics(
            original=experiment.dataset_path,
            generated=generated_data_path,
            device=experiment.device,
        )

        if "best_validation_loss" in train_report:
            metrics["best_validation_loss"] = round(
                train_report["best_validation_loss"], 4
            )

        # print results
        printc(metrics, "yellow")

        results["metrics"] = metrics

    results["config"] = experiment.to_dict()
    results["train_report"] = train_report

    # Storing results in a single json
    store_json(
        results,
        file=str(experiment_folder / "results.json"),
    )

    # Create performance plots
    if do_plots:
        print("Plotting...")
        _ = generative_performance_plots(
            original=experiment.dataset_path,
            generated=generated_data_path,
            target_folder=str(experiment_folder),
            wandb_run=wandb_run,
        )

    # Log metrics to Weights & Biases if a run is provided
    if wandb_run is not None:
        # wandb_run.log(metrics)
        if do_metrics:
            wandb_run.summary.update(metrics)
            time = {
                k: train_report.get(k, None)
                for k in ["training_time", "generation_time", "max_memory", "train_flops"]
            }
            wandb_run.summary.update(time)
            wandb_run.summary.update(
                {"num_parameters": train_report.get("num_parameters", None)}
            )
            wandb_run.summary.update(
                {"emissions (kgCO2eq)": train_report.get("emissions", None)}
            )

    return str(experiment_folder), results
