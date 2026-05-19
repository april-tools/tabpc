import gc
import math
from pathlib import Path

import pandas as pd
import torch
import tqdm
from torch.utils.flop_counter import FlopCounterMode

from cirkit.backend.torch.queries import IntegrateQuery, SamplingQuery
from cirkit.pipeline import PipelineContext, compile
from cirkit.templates import utils
from cirkit.templates.data_modalities import tabular_data
from src.circuits.hetero_tabular_data_api import auto_input_layers
from src.models.model import Model
from src.nn.optimization import Optimization
from src.nn.training import nn_training
from src.util import (load_json, mask_values, pickle_load, pickle_store,
                      store_json)


class ProbabilisticCircuit(Model):
    def __init__(
        self,
        *,
        optimization: Optimization,
        num_input_units: int,
        num_sum_units: int,
        pic_net_dim: int | None = None,
        sum_product_layer: str = "cp",
        use_mixing_weights: bool = True,
        region_graph: str = "chow-liu-tree",
        bin_for_mi: int | None = None,
        features_isolation: tuple = (),
        seed: int | None = None,
        missing_fraction: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_input_units = num_input_units
        self.num_sum_units = num_sum_units
        self.sum_product_layer = sum_product_layer
        self.use_mixing_weights = use_mixing_weights
        self.region_graph = region_graph
        self.optimization = optimization
        self.pic_net_dim = pic_net_dim

        self.features_isolation = features_isolation
        self.seed = seed
        self.bin_for_mi = bin_for_mi
        self.missing_fraction = missing_fraction

    def compile_circuit(self, symbolic_circuit):
        return compile(
            symbolic_circuit,
            ctx=PipelineContext(
                backend="torch", semiring="lse-sum", fold=True, optimize=False
            ),
        )

    def make_circuit(self, X_train: torch.Tensor, metadata: dict):
        input_layer = auto_input_layers(X_train, metadata=metadata)

        symbolic_circuit = tabular_data(
            region_graph=(
                self.region_graph if X_train.shape[1] > 2 else "random-binary-tree"
            ),
            bin_for_mi=self.bin_for_mi,
            num_features=X_train.shape[1],
            data=torch.nan_to_num(X_train, nan=0.0),
            input_layers=input_layer,
            num_input_units=self.num_input_units,
            sum_product_layer=self.sum_product_layer,
            num_sum_units=self.num_sum_units,
            sum_weight_param=utils.Parameterization(
                activation="none" if self.pic_net_dim else "softmax",
                initialization="normal",
            ),
            # use_mixing_weights=self.use_mixing_weights,
        )

        circuit = self.compile_circuit(symbolic_circuit).to(X_train.device)

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
        return self._train_sgd(X_train, X_val, wandb_run, use_codecarbon)

    def _train_sgd(
        self, X_train: torch.Tensor, X_val: torch.Tensor | None = None, wandb_run=None, use_codecarbon: bool = False
    ):
        self.data = torch.vstack([X_train, X_val])
        self.metadata = X_train.metadata
        self.data.metadata = self.metadata

        if X_val is None:
            raise ValueError("Validation data is required for training.")

        if self.missing_fraction > 0.0:
            X_train = mask_values(X_train, p=self.missing_fraction)
            X_val = mask_values(X_val, p=self.missing_fraction)

        self.num_features = X_train.shape[1]

        self.symbolic_circuit, self.circuit = self.make_circuit(
            X_train if self.fit_preprocessor_only_on_train else self.data,
            metadata=self.metadata,
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

        train_losses, validation_losses, epochs_elapsed_times, flops, max_memory, emissions_tracker = nn_training(
            train_set=X_train,
            validation_set=X_val,
            optimization=self.optimization,
            loss_function=negative_log_likelihood,
            nn=self.circuit,
            wandb_run=wandb_run,
            seed=self.seed,
        )

        train_info = {
            "best_validation_loss": min(validation_losses),
            "num_layers": num_layers,
            "num_parameters": num_parameters,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "epochs_elapsed_times": epochs_elapsed_times,
            "mean_epoch_time": sum(epochs_elapsed_times) / len(epochs_elapsed_times),
            "std_epoch_time": (
                torch.std(torch.tensor(epochs_elapsed_times)).item()
                if len(epochs_elapsed_times) > 1
                else 0.0
            ),
            "flops": flops,
            "max_memory": max_memory,
            "emissions": emissions_tracker.final_emissions if emissions_tracker else None,
        }

        return train_info


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
                    # log_likelihoods[start:end] = circuit_log_prob(
                    #     X[start:end].to(device)
                    # ).squeeze()
                    batch_log_likelihood = circuit_log_prob(
                        X[start:end].to(device)
                    ).squeeze()
                    total_log_likelihood += batch_log_likelihood.sum()
                torch.cuda.empty_cache()

        # Compute BPD instead of log-likelihoods
        # BPD = NLL / (num_features * log(2))
        # Here, num_features = X.shape[1]
        mean_NLL = - total_log_likelihood / X.shape[0]

        bpd = mean_NLL / (X.shape[1] * math.log(2))

        torch.cuda.empty_cache()
        # return log_likelihoods.cpu()
        # return total_log_likelihood.cpu()
        return bpd.cpu(), total_log_likelihood.cpu(), mean_NLL.cpu()
    

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
        available_mem = total_gpu_mem * 0.8 - baseline_mem

        # Calculate optimal batch size
        optimal_batch = max(1, int(available_mem / sampling_mem_per_sample))

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        return optimal_batch
    

    def conditional_generation(
        self, *, condition: pd.DataFrame, device: str, batch_size: int, num_to_generate: int | None = None
    ) -> pd.DataFrame:
        tensor_condition = (
            self.preprocessor.transform(condition)
            if self.preprocessor is not None
            else condition
        )

        samples = self._generate(
            condition=tensor_condition.to(device), device=device, batch_size=batch_size, num_to_generate=num_to_generate
        )

        df_samples = (
            self.preprocessor.reverse_transform(samples)
            if self.preprocessor is not None
            else samples
        )

        # assert ((df_samples == condition_evidence) | ~masked_evidence.isna()).all().all(), "Conditioned values do not match the generated samples."

        return self.restore_original_types(df_samples)

    def _generate(
        self,
        batch_size: int | None = None,
        device: str | None = None,
        condition: torch.Tensor | None = None,
        num_to_generate: int | None = None,
    ) -> torch.Tensor:
        # if batch_size is None:
        #     batch_size = self.original_size
        if num_to_generate is None:
            num_to_generate = self.original_size
        if batch_size is None:
            batch_size = self.optimal_batch_size_for_sampling()
            print(f"Using optimal batch size for sampling: {batch_size}")
        if device is not None:
            self.circuit.to(device)
        sampling_query = SamplingQuery(self.circuit)
        torch.cuda.empty_cache()
        
        torch.cuda.memory.reset_peak_memory_stats()

        samples = torch.zeros((num_to_generate, self.num_features), device=device)
        if condition is not None:
            condition = condition.to(device)
            observed_vars = ~condition.isnan()
            # mask = mask.to(device)


        BATCHED_CONDITIONAL_SAMPLING = (
            True # TODO: verify that batched conditional sampling indeed works
        )

        num_batches = math.ceil(num_to_generate / batch_size)

        for i in tqdm.tqdm(range(num_batches), desc="Generating samples"):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_to_generate)
            num_samples = end - start
            with torch.no_grad():
                if num_samples > 0:
                    if condition is None:
                        samples[start:end] = sampling_query(num_samples=num_samples)[0]
                    else:
                        if BATCHED_CONDITIONAL_SAMPLING:
                            samples[start:end] = sampling_query(
                                num_samples=1,
                                # x=condition[start:end],
                                # evidence_vars=evidence_vars[start:end],
                                evidence=condition[start:end],
                                ev_mask=observed_vars[start:end]
                            )[0]
                        else:
                            for i in tqdm.tqdm(
                                range(num_to_generate), desc="Generating samples"
                            ):
                                samples[i] = sampling_query(
                                    num_samples=1,
                                    # x=condition[i : i + 1],
                                    # evidence_vars=evidence_vars[i : i + 1],
                                    evidence=condition[i : i + 1],
                                    ev_mask=observed_vars[i : i + 1]
                                )[0]
                            break

        torch.cuda.empty_cache()
        return samples[: num_to_generate].cpu()

    def prepare_to_store(self):
        self.circuit.cpu()

    def _store(self, model_path: str | Path):
        pickle_store(self.symbolic_circuit, file=f"{model_path}/symbolic_circuit.pkl")
        pickle_store(self.optimization, file=f"{model_path}/optimization.pkl")
        pickle_store(self.original_size, file=f"{model_path}/original_size.pkl")
        pickle_store(self.num_features, file=f"{model_path}/num_features.pkl")
        pickle_store(self.data.metadata, file=f"{model_path}/data_metadata.pkl")
        pickle_store(self.data.cpu(), file=f"{model_path}/data.pkl")
        torch.save(self.circuit.cpu().state_dict(), f"{model_path}/torch_circuit.pth")
        store_json(self.to_dict(), file=f"{model_path}/model_config.json")

        # Testing loading to ensure correctness
        self._load_(model_path=model_path)

    def _load_(self, model_path: str | Path):
        self.symbolic_circuit = pickle_load(f"{model_path}/symbolic_circuit.pkl")
        self.data = pickle_load(file_name=f"{model_path}/data.pkl")
        self.metadata = pickle_load(file_name=f"{model_path}/data_metadata.pkl")
        self.data.metadata = self.metadata
        self.original_size = pickle_load(file_name=f"{model_path}/original_size.pkl")
        self.num_features = pickle_load(file_name=f"{model_path}/num_features.pkl")

        # Compile from the stored symbolic circuit instead of creating new one
        self.circuit = self.compile_circuit(self.symbolic_circuit)
        self.circuit.load_state_dict(torch.load(f"{model_path}/torch_circuit.pth"))

    def to_dict(self) -> dict:
        return super().to_dict() | {
            "num_input_units": self.num_input_units,
            "num_sum_units": self.num_sum_units,
            "sum_product_layer": self.sum_product_layer,
            "use_mixing_weights": self.use_mixing_weights,
            "region_graph": self.region_graph,
            "optimization": self.optimization.to_dict(),
            "pic_net_dim": self.pic_net_dim,
            "features_isolation": self.features_isolation,
            "seed": self.seed,
            "bin_for_mi": self.bin_for_mi,
            "missing_fraction": self.missing_fraction,
        }

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

        return ProbabilisticCircuit(
            preprocessors=None, optimization=optimization, **configs
        )
