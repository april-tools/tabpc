import numpy as np
import pandas as pd

from src.util import fake_float_to_int, infer_type

from .preprocessor import Preprocessor


def nan_handling(table: pd.DataFrame, missing_token: str) -> None:
    for column in table.columns:
        if table[column].isna().any():
            if table[column].dtype in [object, str, bool, "category"]:
                table[column] = table[column].astype("str")
                table[column] = (
                    table[column].fillna(missing_token).replace("nan", missing_token)
                )

            elif table[column].dtype in [int, float]:
                table[f"{column}_is_missing"] = table[column].isna()

    # Pandas doesn't support NaN for integer columns, so we need to convert them back from float to int
    fake_float_to_int(table)


def nan_handling_reverse(
    table: pd.DataFrame, missing_token: str, fake_float_columns
) -> None:
    for column in fake_float_columns:
        table[column] = table[column].astype(float).round()

    for column in table.columns:
        if column[-len("_is_missing") :] == "_is_missing":
            prefix = column[: -len("_is_missing")]
            if table[column].dtype in ["object", "string"]:
                table[prefix] = table[prefix].mask(table[column] == "True", pd.NA)
            else:
                table[prefix] = table[prefix].mask(table[column], pd.NA)
            table.drop(column, axis=1, inplace=True)

        elif table[column].dtype in [object, str, bool, "category"]:
            table[column] = table[column].astype(str).replace(missing_token, pd.NA)


def find_fake_float_columns(table: pd.DataFrame) -> list:
    fake_float_columns = []
    for column in table.columns:
        subset = table[column].dropna()
        if table[column].dtype == float and (subset - subset.astype(int) == 0).all():
            fake_float_columns.append(column)
    return fake_float_columns


def find_inflated_value(s: pd.Series) -> dict | None:
    # Trying to find anomalous oversampled values
    inferred_type = infer_type(s)
    if ("floating" not in inferred_type and "int" not in inferred_type) or (
        "categorical" in inferred_type
    ):
        return None

    out = {}

    normalized_rounded_s = ((s - s.mean()) / s.std()).round(4)
    discretized_counts = normalized_rounded_s.value_counts()

    nr_max = normalized_rounded_s.max()
    nr_min = normalized_rounded_s.min()

    def is_inflated(value: float) -> bool:
        return discretized_counts[value] > (10 * discretized_counts.mean())

    if is_inflated(nr_min):
        out["min"] = s.min()

    if is_inflated(nr_max):
        out["max"] = s.max()

    return None if len(out) == 0 else out


def inflated_value_handling(df: pd.DataFrame, col: str, inflated_values: dict) -> None:
    # Creating a flag column for inflated values
    df[f"{col}_is_inflated"] = "it_is_not"

    if "min" in inflated_values:
        df.loc[df[col] == inflated_values["min"], f"{col}_is_inflated"] = "min"
    if "max" in inflated_values:
        df.loc[df[col] == inflated_values["max"], f"{col}_is_inflated"] = "max"

    df.loc[df[col].isna(), f"{col}_is_inflated"] = None

    # Replacing inflated values with nans
    if "min" in inflated_values:
        df[col] = df[col].replace(inflated_values["min"], np.nan)
    if "max" in inflated_values:
        df[col] = df[col].replace(inflated_values["max"], np.nan)


def inflated_value_handling_reverse(
    df: pd.DataFrame, col: str, inflated_values: dict
) -> None:
    # Replacing extremes
    if "min" in inflated_values:
        iv_min = (
            float(inflated_values["min"])
            if df[col].dtype == float
            else int(inflated_values["min"])
        )
        df.loc[df[col] < inflated_values["min"], col] = iv_min
    if "max" in inflated_values:
        iv_max = (
            float(inflated_values["max"])
            if df[col].dtype == float
            else int(inflated_values["max"])
        )
        df.loc[df[col] > inflated_values["max"], col] = iv_max

    # Using inflated flag column
    if "min" in inflated_values:
        df.loc[df[f"{col}_is_inflated"] == "min", col] = iv_min
    if "max" in inflated_values:
        df.loc[df[f"{col}_is_inflated"] == "max", col] = iv_max

    df.loc[df[f"{col}_is_inflated"].isna(), col] = None
    df.drop(f"{col}_is_inflated", axis=1, inplace=True)


class NanHandler(Preprocessor):
    MISSING_TOKEN = "[MISSING]"

    def __init__(
        self, handle_inflated_values: bool = False, handle_missing_values: bool = False
    ):
        super().__init__()
        self.parameters["handle_inflated_values"] = handle_inflated_values
        self.parameters["handle_missing_values"] = handle_missing_values

    def fit(self, x: pd.DataFrame) -> None:
        super().fit(x)
        self.parameters["has_missing_values"] = False
        if self.parameters["handle_inflated_values"]:
            self.parameters["inflated_values"] = {
                col: find_inflated_value(x[col]) for col in x.columns
            }

            self.parameters["inflated_values"] = {
                k: v
                for k, v in self.parameters["inflated_values"].items()
                if v is not None
            }

            if len(self.parameters["inflated_values"]) > 0:
                print(
                    f"Found inflated values in columns:\n{self.parameters['inflated_values']}"
                )
        else:
            self.parameters["inflated_values"] = {}

        if x.isna().any().any() or len(self.parameters["inflated_values"]) > 0:
            self.parameters["has_missing_values"] = True
            self.parameters["fake_float_columns"] = find_fake_float_columns(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        if self.parameters["has_missing_values"]:
            if self.parameters["handle_missing_values"]:
                nan_handling(df, self.MISSING_TOKEN)
            for col, inflated_values in self.parameters["inflated_values"].items():
                inflated_value_handling(df, col=col, inflated_values=inflated_values)
        self.inversion_check(x_original=x, x_transformed=df)
        return df

    def reverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        if self.parameters["has_missing_values"]:
            for col, inflated_values in self.parameters["inflated_values"].items():
                inflated_value_handling_reverse(
                    df, col=col, inflated_values=inflated_values
                )
            if self.parameters["handle_missing_values"]:
                nan_handling_reverse(
                    df,
                    self.MISSING_TOKEN,
                    fake_float_columns=self.parameters["fake_float_columns"],
                )
        self.restore_original_types(df)
        return df

    def configs_to_dict(self) -> dict:
        return {
            "handle_inflated_values": self.parameters["handle_inflated_values"],
            "handle_missing_values": self.parameters["handle_missing_values"],
        }
