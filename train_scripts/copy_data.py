import sys

sys.path.append(".")
sys.path.append("cirkit")

from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
from src.preprocessors.df_dequantization import Dequantization
from src.preprocessors.string_to_int import StringToInt
from src.wandb_util import wandb_init_wrapper
from src.experiment_controller import run_experiment
from src.experiment_config import ExperimentConfig
from src.models.copy_data import CopyData
from src.preprocessors.to_tensor import ToTensor
from src.util import get_available_device
from src.preprocessors.df_type_conversion import TypeConversion
from src.preprocessors.nan_handler import NanHandler
import torch
import gc

def copy_data_experiments(*, paths: list[str], resample: bool,quantile_norm: bool, nan_handling: bool, name: str, target_folder: str | None = None):
    WANDB = False
    DEVICE = get_available_device(verbose=True)

    for path in paths:
        gc.collect()
        torch.cuda.empty_cache()
        data_name = '_'.join(path.split('/')[1:]).split('.')[0]
        NAME = name + '_' + data_name
        print(f"Running experiment with name: {NAME} on dataset: {path}")
        experiment = ExperimentConfig(
            name=NAME,
            dataset_path=path,
            model=CopyData(resample=resample,
                                                preprocessors=[
                                                TypeConversion(cat_max_states=50, int_max_states=None),
                                                NanHandler(handle_inflated_values=nan_handling),
                                                Dequantization(dequantize=True),
                                                QuantileNormalizer() if quantile_norm else None,
                                                StringToInt(),
                                                ToTensor(),
                                                ]),
            seed=1234,
            device=DEVICE,
        )

        if WANDB:
            with wandb_init_wrapper(
                config=experiment.to_dict(), # eventually use CONFIG instead
                name=NAME) as run:
                run_experiment(experiment=experiment, wandb_run=run, do_plots=False, target_folder=target_folder)
        else:
            run_experiment(experiment=experiment, do_plots=False, target_folder=target_folder)

def main(paths, target_folder: str | None = None):
    copy_data_experiments(paths=paths, resample=False, quantile_norm=True, nan_handling=True, name='copy_data', target_folder=target_folder)
    copy_data_experiments(paths=paths, resample=True, quantile_norm=True, nan_handling=True, name='resample_data', target_folder=target_folder)


if __name__ == "__main__":
    tabdiff_paths = [
        f"data/{name}" for name in [
                                    "adult",
                                    "beijing",
                                    "default",
                                    "shoppers",
                                    "magic",
                                    "diabetes",
                                    "news"]
        ]
    paths = sys.argv[1:] if len(sys.argv) > 1 else tabdiff_paths
    main(paths=paths)