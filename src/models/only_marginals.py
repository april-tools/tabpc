from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.distributions import Categorical, Normal
from torch.utils.flop_counter import FlopCounterMode

from src.models.model import Model
from src.preprocessors.preprocessor import Preprocessor
from src.util import pickle_load, pickle_store


class OnlyMarginals(Model):
    def __init__(
        self,
        *,
        preprocessors: List[Preprocessor] | list = [],
        random_fit: bool = False,
    ) -> None:
        super().__init__(preprocessors=preprocessors, validation_size=0.0)
        self.flops = 0
        self.random_fit = random_fit

    def fit_normal(self, x: torch.Tensor):
        nans = torch.isnan(x)

        flop_counter = FlopCounterMode(display=False, depth=None)

        with flop_counter:
            mean = x[~nans].mean()
            std = x[~nans].std()

        flops = flop_counter.get_total_flops()
        self.flops += flops

        if self.random_fit:
            mean = torch.randn_like(mean)
            std = torch.rand_like(std) + 0.1  # avoid zero std
        
        return Normal(loc=mean, scale=std)

    def fit_categorical(self, x: torch.Tensor):
        flop_counter = FlopCounterMode(display=False, depth=None)
        
        with flop_counter:
            probs = torch.bincount(x.long()) / len(x)
        flops = flop_counter.get_total_flops()
        
        self.flops += flops

        if self.random_fit:
            probs = torch.rand_like(probs)
            probs = probs / probs.sum()
        
        return Categorical(probs=probs)

    def fit_marginal(self, x: torch.Tensor, type_name: str):
        if "cat" in type_name:
            return self.fit_categorical(x)
        else:
            return self.fit_normal(x)

    def num_parameters(self) -> int:
        p = 0
        for marginal in self.marginals.values():
            if isinstance(marginal, Normal):
                p += 2  # mean and std
            if isinstance(marginal, Categorical):
                p += len(marginal.probs)
        return p

    def _train(
        self, X_train: torch.Tensor, X_val: torch.Tensor | None = None, wandb_run=None, use_codecarbon: bool = False
    ):
        metadata = pd.Series(X_train.metadata)
        
        torch.cuda.memory.reset_peak_memory_stats()

        self.target_length = X_train.shape[0] + (
            X_val.shape[0] if X_val is not None else 0
        )
        self.marginals = {
            i: self.fit_marginal(X_train[:, i], type_name=metadata.iloc[i])
            for i in range(X_train.shape[1])
        }

        max_memory = torch.cuda.memory.max_memory_allocated(X_train.device)
        print(f"Max memory allocated: {max_memory} bytes")
        print(f"FLOPs: {self.flops}")

        return {"num_parameters": self.num_parameters(), "flops": self.flops, "max_memory": max_memory}

    def _generate(self, **kwargs) -> torch.Tensor:
        return torch.stack(
            [
                marginal.sample((self.target_length,))
                for marginal in self.marginals.values()
            ],
            dim=1,
        )

    def _store(self, model_path: str | Path):
        pickle_store(self.marginals, file=str(Path(model_path) / "marginals.pkl"))
        pickle_store(
            self.target_length, file=str(Path(model_path) / "target_length.pkl")
        )

    def _load_(self, model_path: str | Path):
        self.marginals = pickle_load(file_name=str(Path(model_path) / "marginals.pkl"))
        self.target_length = pickle_load(
            file_name=str(Path(model_path) / "target_length.pkl")
        )

    @staticmethod
    def init_from_folder(folder: str | Path):
        return OnlyMarginals()
