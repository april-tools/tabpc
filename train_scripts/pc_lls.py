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

import argparse

def main(path, num_units, batch_size, lr, seed, target_folder):
    WANDB = False
    DO_PLOTS = False
    DO_METRICS = False
    POSTFIX = '_lls'
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

    # Normalize seed: accept None or "None" -> None, otherwise convert to int
    if seed is None or (isinstance(seed, str) and seed.lower() == "none"):
        seed = None
    else:
        seed = int(seed)

    DATASET_PATH = path

    # PC Hyperparameters
    # num_units = args.num_units
    # batch_size = args.batch_size
    # lr = args.lr

    DATASET_NAME = DATASET_PATH.split('/')[-1].split('.')[0]
    TARGET_FOLDER = f'artifacts/ll_models/{DATASET_NAME}' if target_folder is None else target_folder

    NAME = 'pc_' + DATASET_NAME + '_' + str(num_units) + '_' + str(batch_size) + '_' + str(lr) + POSTFIX
    print(f"Running experiment with name: {NAME} on dataset: {DATASET_PATH}")

    experiment = ExperimentConfig(
        name=NAME,
        dataset_path=DATASET_PATH,
        model=ProbabilisticCircuit(
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
                                                QuantileNormalizer() if quantile_normalizer else Standardizer(),
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
            experiment_folder = run_experiment(experiment=experiment, wandb_run=run, do_plots=DO_PLOTS, do_metrics=DO_METRICS, target_folder=TARGET_FOLDER)
    else:
        experiment_folder = run_experiment(experiment=experiment, do_plots=DO_PLOTS, do_metrics=DO_METRICS, target_folder=TARGET_FOLDER)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the dataset', required=True)
    parser.add_argument('--num-units', type=int, help='Number of units for the PC', required=True)
    parser.add_argument('--batch-size', type=int, help='Batch size for training', default=256)
    parser.add_argument('--lr', type=float, help='Learning rate for training', default=0.05)
    parser.add_argument('--seed', help='Random seed (int or None)', default=None)
    parser.add_argument('--target-folder', type=str, help='Target folder for saving results', default=None)

    args = parser.parse_args()

    path = args.path
    num_units = args.num_units
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed
    target_folder = args.target_folder

    main(path=path, num_units=num_units, batch_size=batch_size, lr=lr, seed=seed, target_folder=target_folder)