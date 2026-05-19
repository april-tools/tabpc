from abc import ABC, abstractmethod
from pathlib import Path

import torch

from src.util import load_json, store_json


class Preprocessor(ABC):
    def __init__(self):
        self.parameters = {}

    def fit(self, x):
        """
        This deafult fit method is a trick to exploit fit_transform, and then restore the original x if it was modified
        """
        self.parameters["original_types"] = x.dtypes.to_dict()

    def restore_original_types(self, x):
        for col, dtype in self.parameters["original_types"].items():
            try:
                x[col] = x[col].astype(dtype)
            except Exception as e:
                print(
                    f"Warning: could not convert column {col} to type {dtype} due to error: {e}"
                )

    @abstractmethod
    def transform(self, x):
        raise NotImplementedError()

    @abstractmethod
    def reverse_transform(self, x):
        raise NotImplementedError()

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def parameters_file(
        self, model_path: str | Path, *, extension: str = ".json", tag: str = ""
    ):
        # TODO: this works only if every kind of preprocessor is used only once
        return Path(model_path) / Path(str(self.__class__.__name__) + tag + extension)

    def load_(self, model_path: str | Path, tag: str = ""):
        self.parameters = self.deserialize(
            load_json(self.parameters_file(model_path, tag=tag))
        )

    def store(self, model_path: str | Path, tag: str = ""):
        store_json(
            self.serialize(self.parameters),
            file=self.parameters_file(model_path, tag=tag),
        )

    def serialize(self, p: dict):
        return p

    def deserialize(self, p: dict):
        return p

    def to_dict(self) -> dict:
        d = {"class": self.__class__.__name__}
        configs = self.configs_to_dict()
        if len(configs) > 0:
            d["configs"] = configs
        return d

    def configs_to_dict(self) -> dict | None:
        return None

    def inversion_check(self, *, x_original, x_transformed, throw_error=True):
        x_restored = self.reverse_transform(x_transformed)
        self.equality_check(x_original, x_restored, throw_error=throw_error)

    def equality_check(self, df1, df2, throw_error=True):
        if df1.equals(df2):
            return True

        if df1.shape != df2.shape:
            if throw_error:
                raise ValueError("DataFrames have different shapes")
            return False

        if (df1.isna() != df2.isna()).any().any():
            if throw_error:
                raise ValueError("DataFrames have different NaN positions")
            return False

        if df1.dtypes.to_dict() != df2.dtypes.to_dict():
            if throw_error:
                raise ValueError("DataFrames have different dtypes")
            return False

        problematic_columns = (df1 != df2).any().index.tolist()

        for c in problematic_columns:
            s1 = df1[c].dropna()
            s2 = df2[c].dropna()
            if (s1 != s2).any():
                if s1.dtype in ["categorical", "object", str]:
                    if throw_error:
                        raise ValueError(f"DataFrames differ in column {c}")
                    return False
                else:
                    different_elements = s1 != s2
                    reldelta = (s1 - s2) / (1e-6 + (s1 + s2) / 2).abs()[
                        different_elements
                    ]
                    max_reldelta = (reldelta).max()
                    if max_reldelta > 1e-2:
                        if throw_error:
                            raise ValueError(
                                f"DataFrames differ in column {c} with numerical values differing more than 1e-2"
                            )
                        return False

        return True


class TensorPreprocessor(Preprocessor, ABC):
    def fit(self, x: torch.Tensor) -> None:
        pass

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)
