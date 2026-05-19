import os
import sys
import torch
import gc

sys.path.append(".")
sys.path.append("cirkit")
from src.util import get_available_device, store_json
from src.experiment_config import ExperimentConfig, load_experiment
from src.models.probabilistic_circuit import ProbabilisticCircuit
from src.models.shallow_mixture import ShallowMixture


def compute_likelihoods(experiment_path: str) -> dict:
    
    torch.cuda.empty_cache()
    gc.collect()
    
    device = get_available_device(mem_required=0.9, verbose=True)
    exp_config: ExperimentConfig = load_experiment(experiment_path)
    
    model = exp_config.model
    
    if not (isinstance(model, ProbabilisticCircuit) or isinstance(model, ShallowMixture)):
        raise ValueError("Likelihood computation is only supported for ProbabilisticCircuit and ShallowMixture models.")
    
    model.circuit.to(device)
    
    df_train = exp_config.get_train_data()
    df_test = exp_config.get_test_data()
    
    training_tensor, validation_tensor = model.get_preprocessed_data(
            df=df_train,
            fit=False,
            device=device,
        )
    
    test_tensor, _ = model.get_preprocessed_data(
            df=df_test,
            fit=False,
            device=device,
            validation_size=0.0,
        )
    
    datasets = {
        "train": training_tensor,
        "validation": validation_tensor,
        "test": test_tensor,
    }

    results = {}
    for name, data in datasets.items():
        bpd, total_log_likelihood, mean_NLL = model.compute_log_likelihood(data, device=device, batch_size=1_000)
        results[name] = {
            'size': data.size(0),
            'total_log_likelihood': total_log_likelihood.item(),
            'bpd': bpd.item(),
            'mean_NLL': mean_NLL.item(),
        }
    
    results['n_features'] = training_tensor.size(1)
    
    store_json(results, file=os.path.join(experiment_path, "likelihoods.json"))
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_likelihoods.py <experiment_path1> <experiment_path2> ...")
        sys.exit(1)
    experiment_paths = sys.argv[1:]
    print(f"Found {len(experiment_paths)} experiment paths")
    for experiment_path in experiment_paths:
        # Check if likelihoods.json already exists
        likelihoods_file = os.path.join(experiment_path, "likelihoods.json")
        if os.path.exists(likelihoods_file):
            print(f"Likelihoods already computed for experiment at: {experiment_path}. Skipping.")
            continue
        print(f"Computing likelihoods for experiment at: {experiment_path}")
        try:
            compute_likelihoods(experiment_path=experiment_path)
        except Exception as e:
            print(f"Error computing likelihoods for experiment at: {experiment_path}: {e}")
            continue
