import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

from src.metrics.metrics import Metric

sys.path.append(".")

DEFAULT_PRECISION_DIGITS = 4


class AlphaPrecisionBetaRecall(Metric):
    """
    Code from TabDiff
    """

    @staticmethod
    def name() -> str:
        return "alpha_precision_beta_recall"

    @staticmethod
    def evaluate_quality(real_data, syn_data, info):
        # with open(info_path, 'r') as f:
        #     info = json.load(f)

        # syn_data = pd.read_csv(syn_path)
        # real_data = pd.read_csv(real_path)

        """Special treatment for default dataset and CoDi model"""

        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]
        if info["task_type"] == "regression":
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        num_real_data = real_data[num_col_idx]
        cat_real_data = real_data[cat_col_idx]

        num_real_data_np = num_real_data.to_numpy()
        cat_real_data_np = cat_real_data.to_numpy().astype("str")

        num_syn_data = syn_data[num_col_idx]
        cat_syn_data = syn_data[cat_col_idx]

        num_syn_data_np = num_syn_data.to_numpy()

        # cat_syn_data_np = np.array
        cat_syn_data_np = cat_syn_data.to_numpy().astype("str")

        encoder = OneHotEncoder()
        encoder.fit(cat_real_data_np)

        cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
        cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

        le_real_data = pd.DataFrame(
            np.concatenate((num_real_data_np, cat_real_data_oh), axis=1)
        ).astype(float)
        pd.DataFrame(num_real_data_np).astype(float)
        pd.DataFrame(cat_real_data_oh).astype(float)

        le_syn_data = pd.DataFrame(
            np.concatenate((num_syn_data_np, cat_syn_data_oh), axis=1)
        ).astype(float)
        pd.DataFrame(num_syn_data_np).astype(float)
        pd.DataFrame(cat_syn_data_oh).astype(float)

        # Check for nan
        if le_syn_data.isnull().values.any():
            nan_coordinate = np.isnan(le_syn_data.to_numpy()).nonzero()
            nan_row = np.unique(nan_coordinate[0])
            print(f"Synthetic data contains NaN at row {nan_row}: ")
            print(le_syn_data.iloc[nan_row])
            return None, None

        np.set_printoptions(precision=4)

        print("=========== All Features ===========")
        print("Data shape: ", le_syn_data.shape)

        X_syn_loader = GenericDataLoader(le_syn_data)
        X_real_loader = GenericDataLoader(le_real_data)

        quality_evaluator = eval_statistical.AlphaPrecision()
        qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
        qual_res = {
            k: v for (k, v) in qual_res.items() if "naive" in k
        }  # use the naive implementation of AlphaPrecision
        np.mean(list(qual_res.values()))

        print(
            "alpha precision: {:.6f}, beta recall: {:.6f}".format(
                qual_res["delta_precision_alpha_naive"],
                qual_res["delta_coverage_beta_naive"],
            )
        )

        Alpha_Precision_all = qual_res["delta_precision_alpha_naive"]
        Beta_Recall_all = qual_res["delta_coverage_beta_naive"]

        return Alpha_Precision_all, Beta_Recall_all

    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        info: dict | None = None,
        device: str | None = None,
    ):
        return AlphaPrecisionBetaRecall.evaluate_quality(
            real_data, synthetic_data, info
        )

    @staticmethod
    def requires_gpu() -> bool:
        return False
