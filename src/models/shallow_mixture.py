import gc
import math
from pathlib import Path
from typing import List

import torch
import tqdm
from torch.utils.flop_counter import FlopCounterMode

from cirkit.backend.torch.queries import IntegrateQuery, SamplingQuery
from cirkit.pipeline import PipelineContext, compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer
from cirkit.templates.utils import (Parameterization,
                                    name_to_input_layer_factory,
                                    parameterization_to_factory)
from cirkit.utils.scope import Scope
from src.circuits.hetero_tabular_data_api import auto_input_layers
from src.models.model import Model
from src.nn.optimization import Optimization
from src.nn.training import nn_training
from src.util import (load_json, pickle_load, pickle_store,
                      printc, store_json)


def shallow_mixture_for_tabular_data(
    input_distributions: List[dict], num_components: int
) -> Circuit:
    weight_factory = parameterization_to_factory(
        Parameterization(
            activation="softmax",  # Parameterize the sum weights by using a softmax activation
            initialization="uniform",  # Initialize the sum weights by sampling from a standard normal distribution
        )
    )

    input_factories = {
        Scope([i]): name_to_input_layer_factory(
            input_layer["name"], **input_layer["args"]
        )
        for i, input_layer in enumerate(input_distributions)
    }

    # Create the input layers
    input_layers = [
        factory(scope=scope, num_output_units=num_components)
        for scope, factory in input_factories.items()
    ]

    # Create the product layer and the sum layer
    prod = HadamardLayer(num_input_units=num_components, arity=len(input_distributions))
    sl = SumLayer(num_components, 1, 1, weight_factory=weight_factory)

    return Circuit(
        layers=input_layers + [prod, sl],  # Layers that appear in the circuit
        in_layers={  # Connections between layers (i.e. edges in the graph as an adjacency list)
            **{layer: [] for layer in input_layers},
            prod: input_layers,
            sl: [prod],
        },
        outputs=[sl],  # Nodes that are returned by the circuit
    )


class ShallowMixture(Model):
    def __init__(
        self,
        *,
        optimization: Optimization,
        num_components: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_components = num_components
        self.optimization = optimization

    def make_circuit(self, *, X_train, num_components):
        metadata = getattr(X_train, "metadata", None)
        if metadata is None:
            raise ValueError(
                "X_train must have metadata attribute for automatic input layer detection."
            )

        input_layer = auto_input_layers(X_train, metadata=metadata)

        symbolic_circuit = shallow_mixture_for_tabular_data(
            input_distributions=input_layer,
            num_components=num_components,
        )

        circuit = compile(
            symbolic_circuit,
            ctx=PipelineContext(
                backend="torch", semiring="lse-sum", fold=True, optimize=False
            ),
        ).to(X_train.device)

        print(f"Structural properties:")
        print(f"  - Smoothness: {symbolic_circuit.is_smooth}")
        print(f"  - Decomposability: {symbolic_circuit.is_decomposable}")
        print(
            f"  - Structured-decomposability: {symbolic_circuit.is_structured_decomposable}"
        )

        return symbolic_circuit, circuit

    def _train(
        self, X_train: torch.Tensor, X_val: torch.Tensor | None = None, wandb_run=None, use_codecarbon: bool = False
    ):
        self.data = torch.vstack([X_train, X_val])
        self.data.metadata = getattr(X_train, "metadata", None)

        if X_val is None:
            raise ValueError("Validation data is required for training.")

        self.target_length = X_train.shape[0] + X_val.shape[0]
        X_train.device

        self.num_features = X_train.shape[1]

        self.symbolic_circuit, self.circuit = self.make_circuit(
            X_train=X_train if self.fit_preprocessor_only_on_train else self.data,
            num_components=self.num_components,
        )

        marginal_query = IntegrateQuery(self.circuit)

        # Redefining the circuit_log_prob function to handle NaN values
        if X_train.isnan().any() or X_val.isnan().any():

            def circuit_log_prob(X: torch.Tensor) -> torch.Tensor:
                return marginal_query(
                    torch.nan_to_num(X, nan=0.0), integrate_vars=X.isnan()
                )

        else:

            def circuit_log_prob(X: torch.Tensor) -> torch.Tensor:
                return self.circuit(X)

        # sanity check
        ll = circuit_log_prob(X_train[:2])

        # Print some statistics
        num_layers = len(list(self.symbolic_circuit.layers))
        print(f"Number of layers: {num_layers}")
        num_parameters = sum(p.numel() for p in self.circuit.parameters())
        print(f"Number of learnable parameters: {num_parameters}")

        if wandb_run:
            wandb_run.summary["num_layers"] = num_layers
            wandb_run.summary["num_parameters"] = num_parameters

        def negative_log_likelihood(X: torch.Tensor) -> torch.Tensor:
            return -circuit_log_prob(X).sum()

        train_losses, validation_losses, epochs_elapsed_times, flops, max_memory, emissions = nn_training(
            train_set=X_train,
            validation_set=X_val,
            optimization=self.optimization,
            loss_function=negative_log_likelihood,
            nn=self.circuit,
            wandb_run=wandb_run,
        )

        train_info = {
            "best_validation_loss": min(validation_losses),
            "num_layers": num_layers,
            "num_parameters": num_parameters,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "flops": flops,
            "max_memory": max_memory,
            "emissions": emissions.final_emissions if emissions else None,
            "epochs_elapsed_times": epochs_elapsed_times,
            "mean_epoch_time": sum(epochs_elapsed_times) / len(epochs_elapsed_times),
            "std_epoch_time": (
                torch.std(torch.tensor(epochs_elapsed_times)).item()
                if len(epochs_elapsed_times) > 1
                else 0.0
            ),
        }

        return train_info

    def optimal_batch_size_for_sampling(self) -> int:
        """
        Determine the optimal batch size for sampling by measuring peak memory
        usage with a small test batch and calculating the max batch size that
        fits in available GPU memory.

        Returns:
            int: Optimal batch size for sampling
        """
        device = next(self.circuit.parameters()).device

        if not torch.cuda.is_available() or device.type == "cpu":
            # If on CPU, use a reasonable default
            return 1000

        # Clear cache and reset stats
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Get baseline memory usage (model + existing tensors)
        baseline_mem = torch.cuda.memory_allocated(device)

        # Test sampling with a small batch
        test_batch_size = 10
        sampling_query = SamplingQuery(self.circuit)

        torch.cuda.reset_peak_memory_stats(device)
        _ = sampling_query(num_samples=test_batch_size)[0]

        # Measure peak memory used during test sampling
        peak_mem = torch.cuda.max_memory_allocated(device)
        sampling_mem_per_sample = (peak_mem - baseline_mem) / test_batch_size

        total_gpu_mem = torch.cuda.get_device_properties(device).total_memory
        available_mem = total_gpu_mem * 0.6 - baseline_mem

        # Calculate optimal batch size
        optimal_batch = max(1, int(available_mem / sampling_mem_per_sample))

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        return optimal_batch

    def compute_log_likelihood(
        self, X: torch.Tensor, device: str, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        marginal_query = IntegrateQuery(self.circuit)

        # Redefining the circuit_log_prob function to handle NaN values
        if X.isnan().any():

            def circuit_log_prob(X: torch.Tensor) -> torch.Tensor:
                return marginal_query(
                    torch.nan_to_num(X, nan=0.0), integrate_vars=X.isnan()
                )

        else:

            def circuit_log_prob(X: torch.Tensor) -> torch.Tensor:
                return self.circuit(X)

        torch.cuda.empty_cache()
        # log_likelihoods = torch.zeros((X.shape[0],), device=device)
        total_log_likelihood = torch.tensor([0.0], device=device)

        num_batches = math.ceil(X.shape[0] / batch_size)

        with torch.no_grad():
            for i in tqdm.tqdm(range(num_batches), desc="Computing log-likelihoods"):
                start = i * batch_size
                end = min((i + 1) * batch_size, X.shape[0])
                num_samples = end - start
                if num_samples > 0:
                    # log_likelihoods[start:end] = circuit_log_prob(X[start:end].to(device))
                    batch_log_likelihood = circuit_log_prob(X[start:end].to(device)).squeeze()
                    total_log_likelihood += batch_log_likelihood.sum()
                torch.cuda.empty_cache()

        mean_NLL = - total_log_likelihood / X.shape[0]

        bpd = mean_NLL / (X.shape[1] * math.log(2))

        torch.cuda.empty_cache()
        return bpd.cpu(), total_log_likelihood.cpu(), mean_NLL.cpu()

    def _generate(
        self,
        batch_size: int | None = None,
        device: str | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if device is not None:
            self.circuit.to(device)
        sampling_query = SamplingQuery(self.circuit)
        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.memory.reset_peak_memory_stats()

        samples = torch.zeros((self.original_size, self.num_features), device=device)
        if condition is not None:
            condition = condition.to(device)
            observed_vars = ~condition.isnan()

        if batch_size is None:
            batch_size = self.optimal_batch_size_for_sampling()
            printc(f"Auto-selected batch size: {batch_size}", "yellow")

        num_batches = math.ceil(self.original_size / batch_size)
        for i in tqdm.tqdm(range(num_batches), desc="Generating samples"):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.original_size)
            num_samples = end - start
            if num_samples > 0:
                with torch.no_grad():
                    if condition is not None:
                        samples[start:end] = sampling_query(num_samples=num_samples, evidence=condition[start:end], ev_mask=observed_vars[start:end])[0]
                    else:
                        samples[start:end] = sampling_query(num_samples=num_samples)[0]

        gc.collect()
        torch.cuda.empty_cache()
        return samples[: self.original_size].cpu()

    def prepare_to_store(self):
        self.circuit.cpu()

    def to_dict(self) -> dict:
        return super().to_dict() | {
            "num_components": self.num_components,
            "optimization": self.optimization.to_dict(),
        }

    def _store(self, model_path: str | Path):
        # pickle_store(self.symbolic_circuit, file=f"{model_path}/symbolic_circuit.pkl")
        pickle_store(self.optimization, file=f"{model_path}/optimization.pkl")
        pickle_store(self.original_size, file=f"{model_path}/original_size.pkl")
        pickle_store(self.num_features, file=f"{model_path}/num_features.pkl")
        pickle_store(self.data.metadata, file=f"{model_path}/x_train_metadata.pkl")
        pickle_store(self.data.cpu(), file=f"{model_path}/x_train.pkl")
        torch.save(self.circuit.cpu().state_dict(), f"{model_path}/torch_circuit.pth")
        # self.circuit.cpu().save(f"{model_path}/torch_circuit.pth")
        store_json(self.to_dict(), file=f"{model_path}/model_config.json")

    def _load_(self, model_path: str | Path):
        # self.symbolic_circuit = pickle_load(f"{model_path}/symbolic_circuit.pkl")
        self.data = pickle_load(file_name=f"{model_path}/x_train.pkl")
        self.data.metadata = pickle_load(file_name=f"{model_path}/x_train_metadata.pkl")
        self.original_size = pickle_load(file_name=f"{model_path}/original_size.pkl")
        self.num_features = pickle_load(file_name=f"{model_path}/num_features.pkl")

        self.symbolic_circuit, self.circuit = self.make_circuit(
            X_train=self.data, num_components=self.num_components
        )
        self.circuit.load_state_dict(torch.load(f"{model_path}/torch_circuit.pth"))

    @staticmethod
    def init_from_folder(folder: str | Path):  # type: ignore
        configs = load_json(file=Path(folder) / "model_config.json")
        configs.pop("type")
        configs["optimization"] = pickle_load(
            file_name=str(Path(folder) / "optimization.pkl")
        )

        # preprocessor = pickle_load(file_name=str(Path(folder) / "preprocessor.pkl"))
        del configs["preprocessors"]

        optimization = pickle_load(file_name=str(Path(folder) / "optimization.pkl"))
        del configs["optimization"]

        return ShallowMixture(preprocessors=None, optimization=optimization, **configs)
