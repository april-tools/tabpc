from pathlib import Path

from src.models.copy_data import CopyData
from src.models.model import Model
from src.models.only_marginals import OnlyMarginals
from src.models.probabilistic_circuit import ProbabilisticCircuit
from src.models.shallow_mixture import ShallowMixture
from src.util import load_json


def load_model(folder: str | Path) -> Model:
    _: OnlyMarginals
    _: ProbabilisticCircuit
    _: ShallowMixture
    _: CopyData
    model_folder = Path(folder) / "model"
    type = load_json(file=model_folder / "model_config.json")["type"]
    model = eval(f"{type}.init_from_folder")(model_folder)
    model.load_(folder)
    return model
