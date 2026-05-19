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

WANDB = False
DO_PLOTS = False

path = 'data/adult'
NAME = 'test'
USE_QUANTILE_TRANSFORMER = True
NN_SCALE = 2
CATEGORY_THRESHOLD = 50

experiment = ExperimentConfig(
    name=NAME,
    dataset_path=path,
    model=ProbabilisticCircuit(
                            optimization=Optimization(epochs=1,
                                                        batch_size=128,
                                                        compile_nn=False,
                                                        lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                                        lr_scheduler_params={'patience': 0, 'factor': 0.9, 'min_lr': 1e-4},
                                                        optimizer_class=torch.optim.RAdam,
                                                        optimizer_hyperparameters={'lr': 0.02},
                                                        patience=10,
                                                        restore_best_validation_model=True,
                                                        noise_curriculum=False),
                            validation_size=0.10,
                            preprocessors=[ TypeConversion(cat_max_states=CATEGORY_THRESHOLD, int_max_states=None),
                                            NanHandler(handle_inflated_values=True, handle_missing_values=False),
                                            Dequantization(dequantize=True),
                                            QuantileNormalizer() if USE_QUANTILE_TRANSFORMER else Standardizer(),
                                            StringToInt(),
                                            ToTensor(),
                                            ],
                            num_sum_units=NN_SCALE,
                            num_input_units=NN_SCALE,
                            region_graph='chow-liu-tree',
                            sum_product_layer='cp',
                            features_isolation=()),
    seed=1234,
    device=get_available_device(verbose=True)
)

if WANDB:
    with wandb_init_wrapper(
        config=experiment.to_dict(), # eventually use CONFIG instead
        name=NAME) as run:
        run_experiment(experiment=experiment, wandb_run=run, do_plots=DO_PLOTS)
else:
    run_experiment(experiment=experiment, do_plots=DO_PLOTS)
