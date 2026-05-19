import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.util import types_detector

from .preprocessor import Preprocessor


def is_numerical(type, include_integers: bool = False) -> bool:
    return (
        "float" in type
        or ("int" in type and include_integers)
        and "categorical" not in type
    )


class Standardizer(Preprocessor):
    def __init__(self, include_integers: bool = False):
        super().__init__()
        self.parameters["include_integers"] = include_integers

    def fit(self, x: pd.DataFrame):
        types = types_detector(x)

        self.parameters["float_columns"] = [
            c
            for c, col_type in types.items()
            if is_numerical(
                col_type, include_integers=self.parameters["include_integers"]
            )
        ]

        if len(self.parameters["float_columns"]) > 0:
            self.parameters["transformer"] = StandardScaler()
            self.parameters["transformer"].fit(x[self.parameters["float_columns"]])

    def transform(self, x):
        df = x.copy()
        if len(self.parameters["float_columns"]) > 0:
            df[self.parameters["float_columns"]] = self.parameters[
                "transformer"
            ].transform(x[self.parameters["float_columns"]])
        return df

    def reverse_transform(self, x):
        df = x.copy()
        if len(self.parameters["float_columns"]) > 0:
            df[self.parameters["float_columns"]] = self.parameters[
                "transformer"
            ].inverse_transform(x[self.parameters["float_columns"]])
        return df

    def load_(self, model_path: str, tag: str = ""):
        pass

    def store(self, model_path: str, tag: str = ""):
        pass

    def to_dict(self) -> dict:
        return {
            "include_integers": self.parameters["include_integers"],
        }

    def serialize(self, p: dict) -> dict:
        # You may want to save scaler.mean_ and scaler.scale_ for serialization
        return {
            "types": self.parameters.get("types", {}),
            "float_columns": self.parameters.get("float_columns", []),
            "mean": getattr(self.parameters["transformer"], "mean_", None),
            "std": getattr(self.parameters["transformer"], "scale_", None),
        }

    def deserialize(self, p: dict) -> dict:
        self.parameters["types"] = p.get("types", {})
        self.parameters["float_columns"] = p.get("float_columns", [])
        # Restore scaler if needed
        return self.parameters
