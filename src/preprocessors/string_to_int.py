import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.preprocessors.preprocessor import Preprocessor
from src.util import types_detector


class StringToInt(Preprocessor):
    def fit(self, x: pd.DataFrame):
        super().fit(x)

        df = x.copy()  # avoid modifying the original dataframe

        self.parameters["encoders"] = {}
        self.parameters["bool_columns"] = []

        # Convert bool, categorical int and categorical float columns to string, so they will be processed as categorical
        for col, inferred_type in types_detector(df).items():
            if "bool" in inferred_type:
                self.parameters["bool_columns"].append(col)
                df[col] = df[col].astype(str)
            elif "categorical" in inferred_type:
                df[col] = df[col].astype(str)  # TODO: maybe categorical is better

        for col in df.select_dtypes(
            include=["object", "string", "bool", "category"]
        ).columns:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            oe.fit(df[[col]].astype(str))
            self.parameters["encoders"][col] = oe
        self.parameters["names"] = list(df.columns)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        for col, oe in self.parameters.get("encoders", {}).items():
            df[col] = df[col].astype(str)
            encoded_values = oe.transform(df[[col]]).flatten()
            # Replace -1 (unseen categories) with None
            encoded_values[encoded_values == -1] = None
            df[col] = encoded_values
            df[col] = df[col].astype("category")
        return df

    def reverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        for col, oe in self.parameters.get("encoders", {}).items():
            assert (
                df[col].isna().sum() == 0
            ), f"Error: NaN values found in column {col} during reverse_transform in StringToInt"
            df[col] = oe.inverse_transform(df[[col]].astype(int)).flatten()

        # Convert bool columns back to bool
        for col in self.parameters.get("bool_columns", []):
            if col in df.columns:
                # Accept 'True'/'False' as well as '1'/'0'
                df[col] = df[col].map(lambda v: v in [True, "True", "1", 1])

        self.restore_original_types(df)
        return df
