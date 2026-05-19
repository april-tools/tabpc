from pathlib import Path

import pandas as pd
import torch

from src.models.model import Model
from src.preprocessors.preprocessor import Preprocessor
from src.util import pickle_load, pickle_store


class CopyData(Model):
    def __init__(
        self,
        resample: bool = False,
        preprocessors: list[Preprocessor] | list = [],
        fit_preprocessor_only_on_train: bool = False,
    ) -> None:
        super().__init__(
            preprocessors=preprocessors,
            fit_preprocessor_only_on_train=fit_preprocessor_only_on_train,
            validation_size=0.0,
        )
        self.resample = resample

    def train(self, df_train: pd.DataFrame, *, device, wandb_run=None) -> dict:
        self.original_size = len(df_train)
        self.original_pandas_types = df_train.dtypes.to_dict()
        training_data, validation_data = self.get_preprocessed_data(
            df=df_train,
            train=True,
            fit=True,
            device=device,
        )

        self.df_train = df_train
        # inferred_types = types_detector(df_train)

    def _train(
        self, X_train: torch.Tensor, X_val: torch.Tensor | None = None, wandb_run=None, use_codecarbon: bool = False
    ) -> None:
        raise NotImplementedError()

    def _generate(self) -> torch.Tensor:
        raise NotImplementedError()

    def generate(self, **kwargs) -> pd.DataFrame:
        if self.resample:
            return self.df_train.sample(
                n=len(self.df_train), replace=True, random_state=None
            ).reset_index(drop=True)
        return self.df_train

    def _store(self, model_path: str | Path):
        pickle_store(self.df_train, file=str(Path(model_path) / "df_train.pkl"))

    def _load_(self, model_path: str | Path):
        self.df_train = pickle_load(file_name=str(Path(model_path) / "df_train.pkl"))

    @staticmethod
    def init_from_folder(folder: str | Path):
        return CopyData()
