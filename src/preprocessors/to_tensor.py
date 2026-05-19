import pandas as pd
import torch

from .preprocessor import Preprocessor


class ToTensor(Preprocessor):
    def fit(self, x: pd.DataFrame):
        super().fit(x)
        self.verbose = False
        self.parameters["names"] = list(x.columns)

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        metadata = {c: pd.api.types.infer_dtype(df[c]) for c in df.columns}
        x = torch.from_numpy(df.values.astype("float32")).float()
        x.metadata = metadata
        if self.verbose:
            print("Types:")
            print(pd.Series(metadata))
        # self.inversion_check(x_original=df, x_transformed=x)
        return x

    def reverse_transform(self, x: torch.Tensor):
        df = pd.DataFrame(x.cpu().numpy(), columns=self.parameters["names"])
        self.restore_original_types(df)
        return df
