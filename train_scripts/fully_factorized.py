import sys
import argparse

sys.path.append(".")
sys.path.append("cirkit")

from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
from src.preprocessors.df_dequantization import Dequantization
from src.preprocessors.string_to_int import StringToInt
from src.wandb_util import wandb_init_wrapper
from src.experiment_controller import run_experiment
from src.experiment_config import ExperimentConfig
from src.models.only_marginals import OnlyMarginals
from src.preprocessors.to_tensor import ToTensor
from src.util import get_available_device
from src.preprocessors.df_type_conversion import TypeConversion
from src.preprocessors.nan_handler import NanHandler


def marginals_experiments(*, paths: list[str], quantile_norm: bool, nan_handling: bool, name: str, target_folder: str | None = None, random_fit: bool = False):
    WANDB = False
    DEVICE = get_available_device(verbose=True)

    for path in paths:
        data_name = '_'.join(path.split('/')[1:]).split('.')[0]
        NAME = name + '_' + data_name
        print(f"Running experiment with name: {NAME} on dataset: {path}")
        experiment = ExperimentConfig(
            name=NAME,
            dataset_path=path,
            model=OnlyMarginals(preprocessors=[
                                                TypeConversion(cat_max_states=50, int_max_states=None),
                                                NanHandler(handle_inflated_values=nan_handling),
                                                Dequantization(dequantize=True),
                                                QuantileNormalizer() if quantile_norm else None,
                                                StringToInt(),
                                                ToTensor(),
                                                ], random_fit=random_fit),
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


def main(paths, target_folder: str | None = None, do_extras=False):

    if do_extras:
        # No preprocessing version
        # marginals_experiments(paths=paths, quantile_norm=False, nan_handling=False, name='FullyFactorized_no_preprocessing', target_folder=target_folder)

        # Partial preprocessing versions
        # marginals_experiments(paths=paths, quantile_norm=False, nan_handling=True, name='FullyFactorized_with_nan_handling', target_folder=target_folder)
        # marginals_experiments(paths=paths, quantile_norm=True, nan_handling=False, name='FullyFactorized_with_quantile_norm', target_folder=target_folder)

        # Random fit version
        marginals_experiments(paths=paths, quantile_norm=True, nan_handling=True, name='FullyFactorized_with_preprocessing_random_fit', target_folder=target_folder, random_fit=True)

    # Full preprocessing version
    marginals_experiments(paths=paths, quantile_norm=True, nan_handling=True, name='FullyFactorized_with_preprocessing', target_folder=target_folder)


if __name__ == "__main__":
    tabdiff_paths = [
        f"data/{name}" for name in ["adult", "beijing", "default", "shoppers", "magic", "diabetes", "news"]
    ]
    # paths = sys.argv[1:] if len(sys.argv) > 1 else tabdiff_paths

    parser = argparse.ArgumentParser(description="Train Fully Factorized models on TabDiff datasets with specified target folder.")
    parser.add_argument('--path', type=str, default=None, help='Path to the dataset. If not specified, defaults to all paths in data/.')
    parser.add_argument('--target-folder', type=str, default=None, help='Target folder for saving results. If not specified, defaults to None.')
    args = parser.parse_args()
    paths = [args.path] if args.path else tabdiff_paths
    main(paths=paths, target_folder=args.target_folder)