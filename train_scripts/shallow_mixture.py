import sys
import argparse

sys.path.append(".")
sys.path.append("cirkit")

from src.preprocessors.nan_handler import NanHandler
from src.preprocessors.df_standardizer import Standardizer
from src.preprocessors.df_type_conversion import TypeConversion
from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
from src.preprocessors.df_dequantization import Dequantization
from src.preprocessors.string_to_int import StringToInt
from src.preprocessors.to_tensor import ToTensor
from src.wandb_util import wandb_init_wrapper
from src.experiment_controller import run_experiment
from src.experiment_config import ExperimentConfig
from src.models.shallow_mixture import ShallowMixture
from src.nn.optimization import Optimization
import torch
from src.util import get_available_device
from train_scripts.tabdiff_datasets_names import PATHS as TABDIFF_PATHS

WANDB = False
DO_PLOTS = False
CATEGORY_THRESHOLD = 50
EPOCHS = 50
USE_QUANTILE_TRANSFORMER = True
HANDLE_INFLATED_VALUES = True
BATCH_SIZE = 1024
LR = 0.05


def units_for_dataset(dataset_path: str) -> int:
    if 'adult' in dataset_path:
        return 20_000
    elif 'beijing' in dataset_path:
        return 50_000
    elif 'diabetes' in dataset_path:
        return 20_000
    elif 'default' in dataset_path:
        return 20_000
    elif 'magic' in dataset_path:
        return 10_000
    elif 'news' in dataset_path:
        return 10_000
    elif 'shoppers' in dataset_path:
        return 20_000
    else:
        raise ValueError(f"Unknown dataset {dataset_path}")


def main(paths, seed: int | None = None, target_folder: str | None = None):
    DEVICE = get_available_device(verbose=True, mem_required=0.90, stop_if_no_free_gpu=True)

    for path in paths:
        NAME = 'shallow_mixture_' + path.split('/')[-1]

        print(f"Running experiment with name: {NAME} on dataset: {path}")
        
        experiment = ExperimentConfig(
            name=NAME,
            dataset_path=path,
            model=ShallowMixture(
                                    optimization=Optimization(epochs=EPOCHS,
                                                                batch_size=BATCH_SIZE,
                                                                compile_nn=False,
                                                                lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                                                lr_scheduler_params={'patience': 1, 'factor': 0.85, 'min_lr': 1e-4},
                                                                optimizer_class=torch.optim.RAdam,
                                                                optimizer_hyperparameters={'lr': LR},
                                                                patience=10,
                                                                restore_best_validation_model=True),
                                    validation_size=0.10,
                                    preprocessors=[ TypeConversion(cat_max_states=CATEGORY_THRESHOLD, int_max_states=None),
                                                    NanHandler(handle_inflated_values=HANDLE_INFLATED_VALUES, handle_missing_values=False), 
                                                    Dequantization(dequantize=True),
                                                    QuantileNormalizer() if USE_QUANTILE_TRANSFORMER else Standardizer(),
                                                    StringToInt(),
                                                    ToTensor(),
                                                    ],
                                    num_components=units_for_dataset(path),
            ),
            seed=seed,
            device=DEVICE
        )

        if WANDB:
            with wandb_init_wrapper(
                config=experiment.to_dict(), # eventually use CONFIG instead
                name=NAME) as run:
                run_experiment(experiment=experiment, wandb_run=run, do_plots=DO_PLOTS, target_folder=target_folder)
        else:
            run_experiment(experiment=experiment, do_plots=DO_PLOTS, target_folder=target_folder)


if __name__ == "__main__":
    # paths = sys.argv[1:] if len(sys.argv) > 1 else TABDIFF_PATHS
    
    parser = argparse.ArgumentParser(description="Train Shallow Mixture models on TabDiff datasets with specified target folder.")
    parser.add_argument('--path', type=str, default=None, help='Path to the dataset. If not specified, defaults to all paths in TABDIFF_PATHS.')
    parser.add_argument('--target-folder', type=str, default=None, help='Target folder for saving results. If not specified, defaults to None.')
    parser.add_argument('--seed', default=None, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # If no path is specified, use all paths from TABDIFF_PATHS
    paths = [args.path] if args.path else TABDIFF_PATHS
    
    # normalize seed: accept None or "None" -> None, otherwise convert to int
    if args.seed is None or (isinstance(args.seed, str) and args.seed.lower() == "none"):
        seed = None
    else:
        seed = int(args.seed)

    main(paths, target_folder=args.target_folder, seed=seed)