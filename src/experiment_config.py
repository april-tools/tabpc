import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.nn.training import loss_plot
from src.util import Timer, load_json, set_seeds, store_json

from .models.load_model import load_model
from .models.model import Model

EXPERIMENT_FILE_NAME = "experiment"
GENERATED_DATA_NAME = "generated.csv"


@dataclass
class ExperimentConfig:
    dataset_path: str
    model: Model
    name: str
    device: str
    seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "name": self.name,
            "model": self.model.to_dict(),
            "device": str(self.device),
            "seed": self.seed,
        }

    def generated_data_path(self, folder: Path) -> Path:
        return folder / GENERATED_DATA_NAME

    def get_train_data(self) -> pd.DataFrame:
        if os.path.isdir(self.dataset_path):
            df_train = pd.read_csv(f"{self.dataset_path}/train.csv")
        else:
            df_train = pd.read_csv(self.dataset_path)
        return df_train

    def get_test_data(self) -> pd.DataFrame:
        if os.path.isdir(self.dataset_path):
            df_test = pd.read_csv(f"{self.dataset_path}/test.csv")
        else:
            raise ValueError(
                "Test data is only available for datasets stored in a directory."
            )
        return df_test

    def generate(self, folder: Path, wandb_run, use_codecarbon: bool = False) -> Tuple[str, dict]:
        if self.seed is not None:
            set_seeds(self.seed)

        df_train = self.get_train_data()

        with Timer() as training_timer:
            train_report = self.model.train(
                df_train=df_train, device=self.device,
                wandb_run=wandb_run, use_codecarbon=use_codecarbon
            )

        if train_report is None:
            train_report = {}

        if (
            train_report is not None
            and "train_losses" in train_report
            and "validation_losses" in train_report
        ):
            loss_plot(
                train_losses=train_report["train_losses"],
                validation_losses=train_report["validation_losses"],
                path=str(folder),
            )

        with Timer() as generation_timer:
            df = self.model.generate()

        path = self.generated_data_path(folder)
        df.to_csv(path, index=False)

        train_report["training_time"] = training_timer.elapsed
        train_report["generation_time"] = generation_timer.elapsed

        return str(path), train_report

    def store(self, experiment_path: str):
        self.model.store(experiment_path=experiment_path)
        store_json(self.to_dict(), file=f"{experiment_path}/experiment_config.json")


def load_experiment(experiment_path: str | Path) -> ExperimentConfig:
    data = load_json(file=f"{experiment_path}/experiment_config.json")
    model = load_model(folder=experiment_path)
    return ExperimentConfig(
        dataset_path=data["dataset_path"],
        name=data["name"],
        model=model,
        device=data["device"],
        seed=data.get("seed", None),
    )
