from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from src.util import decimals_used, types_detector

from .preprocessor import Preprocessor


def precise_round(series, precision):
    return series.apply(
        lambda x: float(
            Decimal(str(x)).quantize(
                Decimal("0." + "0" * precision), rounding=ROUND_HALF_UP
            )
        )
    )


class Dequantization(Preprocessor):
    def __init__(self, dequantize: bool = False, dequantize_all_floats: bool = False):
        super().__init__()
        self.parameters["dequantize"] = dequantize
        self.parameters["dequantize_all_floats"] = dequantize_all_floats

    def fit(self, x: pd.DataFrame):
        super().fit(x)
        types = types_detector(x)

        if self.parameters["dequantize_all_floats"]:
            quantized_columns = [
                column for column, dtype in types.items() if "float" in dtype
            ]
        else:
            quantized_columns = [
                column
                for column, dtype in types.items()
                if (("int" in dtype) and ("categorical" not in dtype))
                or ("quantized" in dtype)
            ]

        print("quantized_columns:", quantized_columns)

        self.parameters["precision"] = {
            column: decimals_used(x[column]) for column in quantized_columns
        }

    def transform(self, x: pd.DataFrame):
        df = x.copy()
        if self.parameters["dequantize"]:
            for col, precision in self.parameters["precision"].items():
                df[col] = df[col] + (np.random.uniform(size=len(df[col])) - 0.5) * (
                    10 ** -(precision)
                )
        self.inversion_check(x_original=x, x_transformed=df)
        return df

    def reverse_transform(self, x: pd.DataFrame):
        df = x.copy()
        for c, precision in self.parameters["precision"].items():
            df[c] = precise_round(df[c], precision)  # Avoid rounding errors
            # if precision == 0:
            # If precision is 0, it means the column was originally an integer
            # df[c] = df[c].astype(int)
        self.restore_original_types(df)
        return df

    def configs_to_dict(self) -> dict:
        return {
            "dequantize": self.parameters["dequantize"],
            "dequantize_all_floats": self.parameters["dequantize_all_floats"],
        }
