import sys

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
from src.models.probabilistic_circuit import ProbabilisticCircuit
from src.nn.optimization import Optimization
import torch
from src.util import get_available_device
from train_scripts.tabdiff_datasets_names import PATHS as TABDIFF_PATHS

WANDB = False
DO_PLOTS = False

DEVICE = get_available_device()

bin_for_mi = 15
dequantize_all_floats = False
handle_inflated_values = True
quantile_normalizer = True
validation_size = 0.10

# Other Hyperparameters
CATEGORY_THRESHOLD = 50
EPOCHS = 150
PATIENCE = 10

def main(paths, target_folder: str | None = None, fit_preprocessor_only_on_train: bool = False, seed: int | None = None):
    for DATASET_PATH in paths:
        # Tuned Hyperparameters
        if 'adult' in DATASET_PATH:
            num_units = 4_096
            batch_size = 512
            lr = 0.25
        elif 'beijing' in DATASET_PATH:
            num_units = 4_096
            batch_size = 512
            lr = 0.25
        elif 'diabetes' in DATASET_PATH:
            num_units = 2_048
            batch_size = 512
            lr = 0.25
        elif 'default' in DATASET_PATH:
            num_units = 512
            batch_size = 512
            lr = 0.1
        elif 'magic' in DATASET_PATH:
            num_units = 2_048
            batch_size = 256
            lr = 0.1
        elif 'news' in DATASET_PATH:
            num_units = 1_024
            batch_size = 512
            lr = 0.1
        elif 'shoppers' in DATASET_PATH:
            num_units = 2_048
            batch_size = 512
            lr = 0.1
        else:
            raise ValueError(f"Unknown dataset {DATASET_PATH}")

        NAME = 'pc_' + DATASET_PATH.split('/')[-1]
        if fit_preprocessor_only_on_train:
            NAME += '_fitpreptrainonly'
        print(f"Running experiment with name: {NAME} on dataset: {DATASET_PATH}")

        # Normalize seed: accept None or "None" -> None, otherwise convert to int
        if seed is None or (isinstance(seed, str) and seed.lower() == "none"):
            seed = None
        else:
            seed = int(seed)

        experiment = ExperimentConfig(
            name=NAME,
            dataset_path=DATASET_PATH,
            model=ProbabilisticCircuit(
                                    fit_preprocessor_only_on_train=fit_preprocessor_only_on_train,
                                    optimization=Optimization(epochs=EPOCHS,
                                                                batch_size=batch_size,
                                                                compile_nn=False,
                                                                lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                                                lr_scheduler_params={'patience': 0, 'factor': 0.85, 'min_lr': 2e-4},
                                                                optimizer_class=torch.optim.RAdam,
                                                                optimizer_hyperparameters={'lr': lr},
                                                                patience=PATIENCE,
                                                                ),
                                    validation_size=validation_size,
                                    preprocessors=[
                                                    TypeConversion(cat_max_states=CATEGORY_THRESHOLD, 
                                                                    int_max_states=None),
                                                    NanHandler(handle_inflated_values=handle_inflated_values, handle_missing_values=False),
                                                    Dequantization(dequantize=True, dequantize_all_floats=dequantize_all_floats),
                                                    QuantileNormalizer(invertible=fit_preprocessor_only_on_train) if quantile_normalizer else Standardizer(),
                                                    StringToInt(),
                                                    ToTensor(),
                                                    ],
                                    num_sum_units=num_units,
                                    num_input_units=num_units,
                                    region_graph='chow-liu-tree',
                                    sum_product_layer='cp',
                                    bin_for_mi=bin_for_mi),
            seed=seed,
            device=DEVICE,
        )

        if WANDB:
            with wandb_init_wrapper(
                config=experiment.to_dict(), # eventually use CONFIG instead
                name=NAME) as run:
                run_experiment(experiment=experiment, wandb_run=run, do_plots=DO_PLOTS, target_folder=target_folder)
        else:
            run_experiment(experiment=experiment, do_plots=DO_PLOTS, target_folder=target_folder)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else TABDIFF_PATHS
    main(paths)