import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer

from src.util import printc, types_detector

from .preprocessor import Preprocessor


def is_numerical(type, include_integers: bool = False) -> bool:
    return (
        "float" in type
        or ("int" in type and include_integers)
        and "categorical" not in type
    )


def inverse_gaussian_cdf(x):
    # not sure about the derivative
    probit = norm.ppf(x)
    return probit, 1 / norm.pdf(probit)


class QuantileNormalizer(Preprocessor):
    def __init__(self, include_integers: bool = False, invertible: bool = False):
        super().__init__()
        self.parameters["include_integers"] = include_integers
        self.parameters["invertible"] = invertible

    def fit(self, x: pd.DataFrame):
        super().fit(x)
        types = types_detector(x)

        self.parameters["float_columns"] = [
            c
            for c, col_type in types.items()
            if is_numerical(
                col_type, include_integers=self.parameters["include_integers"]
            )
        ]

        if len(self.parameters["float_columns"]) > 0:
            # parameters of QT copied from TabDiff
            self.parameters["transformer"] = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(min(x.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
            )

            self.parameters["transformer"].fit(x[self.parameters["float_columns"]])

            if self.parameters["invertible"]:
                x_float = x[self.parameters["float_columns"]]

                y_float = x_float.copy()
                y_float[self.parameters["float_columns"]] = self.parameters[
                    "transformer"
                ].transform(x_float)

                self.parameters["ranges"] = pd.DataFrame(
                    {
                        "x_max": x_float.max(),
                        "x_min": x_float.min(),
                        "y_max": y_float.max(),
                        "y_min": y_float.min(),
                        "x_std": x_float.std(),
                    }
                )

    def transform(self, x):
        df = x.copy()
        if len(self.parameters["float_columns"]) > 0:
            df[self.parameters["float_columns"]] = self.parameters[
                "transformer"
            ].transform(x[self.parameters["float_columns"]])

            if self.parameters["invertible"]:
                x_float = x[self.parameters["float_columns"]].copy()

                for c in self.parameters["float_columns"]:
                    x_max = self.parameters["ranges"].loc[c, "x_max"]
                    x_min = self.parameters["ranges"].loc[c, "x_min"]
                    right_outliers = x[c] > x_max
                    left_outliers = x[c] < x_min
                    k = 1 / self.parameters["ranges"].loc[c, "x_std"]
                    if right_outliers.any():
                        n_right_outliers = right_outliers.sum()
                        printc(
                            f"QN: Found {n_right_outliers} right outliers in column {c}",
                            "red",
                        )
                        df.loc[right_outliers, c] += k * (
                            x_float.loc[right_outliers, c] - x_max
                        )
                    if left_outliers.any():
                        n_left_outliers = left_outliers.sum()
                        printc(
                            f"QN: Found {n_left_outliers} left outliers in column {c}",
                            "red",
                        )
                        df.loc[left_outliers, c] += k * (
                            x_float.loc[left_outliers, c] - x_min
                        )
        # self.inversion_check(x_original=x, x_transformed=df)
        return df

    def reverse_transform(self, x):
        df = x.copy()
        if len(self.parameters["float_columns"]) > 0:
            df[self.parameters["float_columns"]] = self.parameters[
                "transformer"
            ].inverse_transform(x[self.parameters["float_columns"]])

            if self.parameters["invertible"]:
                x_float = x[self.parameters["float_columns"]].copy()

                for c in self.parameters["float_columns"]:
                    y_max = self.parameters["ranges"].loc[c, "y_max"]
                    y_min = self.parameters["ranges"].loc[c, "y_min"]

                    right_outliers = x[c] > y_max
                    left_outliers = x[c] < y_min
                    k = self.parameters["ranges"].loc[c, "x_std"]
                    if right_outliers.any():
                        n_right_outliers = right_outliers.sum()
                        printc(
                            f"Inverse QN: Found {n_right_outliers} right outliers in column {c}",
                            "red",
                        )
                        df.loc[right_outliers, c] += k * (
                            x_float.loc[right_outliers, c] - y_max
                        )
                    if left_outliers.any():
                        n_left_outliers = left_outliers.sum()
                        printc(
                            f"Inverse QN: Found {n_left_outliers} left outliers in column {c}",
                            "red",
                        )
                        df.loc[left_outliers, c] += k * (
                            x_float.loc[left_outliers, c] - y_min
                        )

        self.restore_original_types(df)
        return df

    def load_(self, model_path: str, tag: str = ""):
        pass

    def store(self, model_path: str, tag: str = ""):
        pass

    def configs_to_dict(self) -> dict:
        return {
            "include_integers": self.parameters["include_integers"],
            "invertible": self.parameters["invertible"],
        }
