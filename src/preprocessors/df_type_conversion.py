import pandas as pd

from src.util import is_fake_float, printc, type_identification_features

from .preprocessor import Preprocessor


def is_numerical(type):
    return "float" in type or "int" in type and "categorical" not in type


def is_integer(s: pd.Series):
    """
    Check if a float series is actually an integer series.
    This is done by checking if all values are integers or if the series has no decimal part.
    """
    return (abs(s.round() - s) < 1e-6).all()


def is_categorical_int(features: dict) -> bool:
    return (
        features["pandas_type"] == "integer"
        and features["min_step"] == features["max_step"]
        and features["corr"] < 0.1
    )


def infer_type(s: pd.Series, cat_max_states: int, int_max_states: int | None):
    type_features = type_identification_features(s)

    if "bool" in type_features["pandas_type"]:
        return "boolean"

    if type_features["total_unique_values"] <= cat_max_states:
        return "categorical"

    if type_features["pandas_type"] in [
        "categorical",
        "object",
        "string",
    ] or is_categorical_int(type_features):
        printc(
            f"Warning: column {s.name} is probably a categorical with too many categories ({type_features['total_unique_values']})",
            "red",
        )
        return "categorical"

    if int_max_states:
        if type_features["total_unique_values"] <= int_max_states and is_integer(s):
            return "integer"

    if is_fake_float(s):
        return "integer"

    if type_features["fraction_of_unique_values"] < 0.5:
        printc(
            f"Warning: the input column {s.name} seems quantized: {1-type_features['fraction_of_unique_values']:.2%} are not unique",
            "red",
        )
    return "floating"


class TypeConversion(Preprocessor):
    def __init__(self, cat_max_states: int, int_max_states: int | None = None):
        super().__init__()
        self.parameters["cat_max_states"] = cat_max_states
        self.parameters["int_max_states"] = int_max_states

    def fit(self, x: pd.DataFrame):
        super().fit(x)

        self.parameters["inferred_types"] = {
            col: infer_type(
                x[col],
                self.parameters["cat_max_states"],
                self.parameters["int_max_states"],
            )
            for col in x.columns
        }

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        for col, col_type in self.parameters["inferred_types"].items():
            if col_type in ("categorical", "boolean"):
                # handling the case with nans and ints
                if (
                    self.parameters["original_types"][col] == int
                    and df[col].dtype == float
                ):
                    nan_mask = df[col].isna()
                    df[col] = df[col].fillna(0).astype(int)
                    df[col] = df[col].astype(str)
                    df.loc[nan_mask, col] = "nan"
                else:
                    df[col] = df[col].astype(str)  # .astype("category")

            elif col_type == "integer":
                df[col] = df[col].astype(int) if not any(df[col].isna()) else df[col]
            elif col_type == "floating":
                df[col] = df[col].astype(float)

        # self.inversion_check(x_original=x, x_transformed=df)
        return df

    def reverse_transform(self, x):
        df = x.copy()
        for col, col_type in self.parameters["inferred_types"].items():
            if col_type == "boolean":
                df[col] = df[col] == "True"
            if (df[col] == "nan").any():
                df[col] = df[col].replace("nan", None)

        self.restore_original_types(df)
        return df

    def load_(self, model_path: str, tag: str = ""):
        pass

    def store(self, model_path: str, tag: str = ""):
        pass

    def configs_to_dict(self) -> dict:
        return {
            "cat_max_states": self.parameters["cat_max_states"],
            "int_max_states": self.parameters["int_max_states"],
        }
