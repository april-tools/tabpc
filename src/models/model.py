import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch

from src.preprocessors.compose import Compose
from src.preprocessors.preprocessor import Preprocessor
from src.preprocessors.to_tensor import ToTensor
from src.util import (pickle_load, pickle_store, split_dataset,
                      split_torch_dataset, store_json, types_detector)

MODEL_FOLDER_NAME = Path("model")
PARAMS_FILE = Path("params.json")


class Model(ABC):
    def __init__(
        self,
        *,
        preprocessors: List[Preprocessor] | list = [],
        validation_size: float = 0.0,
        fit_preprocessor_only_on_train: bool = False,
    ) -> None:
        super().__init__()
        self.preprocessor = Compose(preprocessors) if preprocessors else ToTensor()
        self.validation_size = validation_size
        self.fit_preprocessor_only_on_train = fit_preprocessor_only_on_train

    def store(self, experiment_path: str | Path):
        model_path = self.get_model_folder(experiment_path)
        os.makedirs(model_path)
        config = self.to_dict()
        config["type"] = self.__class__.__name__
        store_json(config, file=model_path / "model_config.json")
        pickle_store(self.preprocessor, file=str(model_path / "preprocessor.pkl"))
        pickle_store(
            self.original_pandas_types, file=str(model_path / "inferred_types.pkl")
        )
        # self.preprocessor.store(model_path)
        self._store(model_path)

    def load_(self, experiment_path: str | Path):
        model_path = self.get_model_folder(experiment_path)
        self.preprocessor = pickle_load(file_name=str(model_path / "preprocessor.pkl"))
        self.original_pandas_types = pickle_load(
            file_name=str(model_path / "inferred_types.pkl")
        )
        self._load_(model_path)

    def get_model_folder(self, experiment_path: str | Path) -> Path:
        return Path(experiment_path) / MODEL_FOLDER_NAME

    def params_file(self, model_path: str | Path):
        return Path(model_path) / PARAMS_FILE

    def dataset_to_device(self, x: torch.Tensor, device: str | torch.device):
        metadata = getattr(x, "metadata", None)
        x = x.to(device)
        if metadata is not None:
            x.metadata = metadata
        return x

    def get_preprocessed_data(
        self,
        *,
        df: pd.DataFrame,
        train: bool = True,
        fit: bool | None = None,
        device: str | torch.device,
        validation_size: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if fit is None:
            fit = train
        if validation_size is None:
            validation_size = self.validation_size

        if not self.fit_preprocessor_only_on_train:
            x_preprocessed = df_train_processed = (
                self.preprocessor.fit_transform(df)
                if fit
                else self.preprocessor.transform(df)
            )
            metadata = getattr(x_preprocessed, "metadata", None)
            x_train, x_val = (
                split_torch_dataset(x_preprocessed, train_size=1 - validation_size)
                if validation_size > 0
                else (x_preprocessed, None)
            )
            if metadata is not None:
                x_train.metadata = metadata
            if x_val is not None:
                x_val.metadata = metadata
            return (
                self.dataset_to_device(x_train, device=device),
                (
                    self.dataset_to_device(x_val, device=device)
                    if x_val is not None
                    else None
                ),
            )
        else:
            df_train, df_val = (
                split_dataset(df, train_size=1 - validation_size)
                if validation_size > 0
                else (df, None)
            )

            if self.preprocessor is None:
                return df_train, df_val
            else:
                df_train_processed = (
                    self.preprocessor.fit_transform(df_train)
                    if fit
                    else self.preprocessor.transform(df_train)
                )
                df_val_processed = (
                    self.preprocessor.transform(df_val) if df_val is not None else None
                )
                return (
                    self.dataset_to_device(df_train_processed, device),
                    (
                        self.dataset_to_device(df_val_processed, device)
                        if df_val_processed is not None
                        else None
                    ),
                )

    def train(self, df_train: pd.DataFrame, *, device, wandb_run=None, use_codecarbon: bool = False) -> dict:
        self.original_size = len(df_train)
        self.original_pandas_types = df_train.dtypes.to_dict()
        training_data, validation_data = self.get_preprocessed_data(
            df=df_train,
            train=True,
            fit=True,
            device=device,
        )

        self.df_train = df_train
        self.inferred_types = types_detector(df_train)
        return self._train(
            X_train=training_data, X_val=validation_data,
            wandb_run=wandb_run, use_codecarbon=use_codecarbon
        )

    def restore_original_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.original_pandas_types is None:
            raise ValueError(
                "Original pandas types are not set. Train the model first."
            )
        return df.astype(self.original_pandas_types)

    def generate(self, **kwargs) -> pd.DataFrame:
        samples = self._generate(**kwargs)
        df_samples = (
            self.preprocessor.reverse_transform(samples)
            if self.preprocessor is not None
            else samples
        )
        return self.restore_original_types(df_samples)

    @abstractmethod
    def _train(
        self, X_train: torch.Tensor, X_val: torch.Tensor | None = None, wandb_run=None, use_codecarbon: bool = False
    ) -> dict:
        pass

    @abstractmethod
    def _generate(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _store(self, model_path: str | Path):
        pass

    @abstractmethod
    def _load_(self, model_path: str | Path):
        pass

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "preprocessors": self.preprocessor.to_dict() if self.preprocessor else [],
            "validation_size": self.validation_size,
        }

    def prepare_to_store(self):
        pass

    @staticmethod
    def init_from_folder(folder: str | Path):
        pass
