import os
import sys
import gc
import torch
import numpy as np

sys.path.append(".")
sys.path.append("cirkit")
from src.util import Timer, get_available_device, find_dirs, edit_json, load_json, store_json, randomly_mask_values
from src.experiment_config import ExperimentConfig, load_experiment
from src.models.probabilistic_circuit import ProbabilisticCircuit
from src.models.shallow_mixture import ShallowMixture


def get(d: dict, keys: tuple):
    try:
        for key in keys:
            d = d[key]
    except KeyError:
        return None
    return d


def sample(experiment_path: str, n_samples: int, overwrite: bool = False, conditioning_percentage: float | None = None, batch_size: int = 10_000, device = None) -> str:
    DEVICE = device if device is not None else get_available_device(mem_required=0.9, verbose=True)

    exp_config: ExperimentConfig = load_experiment(experiment_path)
    
    # create folder for samples
    samples_path = os.path.join(experiment_path, f"new_samples")
    os.makedirs(samples_path, exist_ok=True)
    
    csv_samples_path = os.path.join(samples_path, f"samples_cond{conditioning_percentage:.2f}") if conditioning_percentage is not None else os.path.join(samples_path, "samples")
    os.makedirs(csv_samples_path, exist_ok=True)
    
    print(f"Sampling {n_samples} times from model...")
    
    # load model
    model = exp_config.model
    
    if isinstance(model, ProbabilisticCircuit) or isinstance(model, ShallowMixture):
        model.circuit.to(DEVICE)
            
    times_file_name = os.path.join(samples_path, "generation_times.json") if conditioning_percentage is None else os.path.join(samples_path, f"generation_times_cond{conditioning_percentage:.2f}.json")

    # generate samples
    if overwrite and os.path.exists(times_file_name):
        os.remove(times_file_name)

    if conditioning_percentage == 1.0:
        n_samples = 1  # only one sample needed, all values are conditioned

    for i in range(n_samples):

        name = f"sample_{i}"
        if not overwrite and os.path.exists(os.path.join(csv_samples_path, f"{name}.csv")):
            print(f"Sample {i} already exists, skipping...")
            continue
        
        print(f"Generating sample {i+1}/{n_samples}...")
        with Timer() as generation_timer:
            if conditioning_percentage is None:
                df = model.generate(device=DEVICE, batch_size=batch_size)
            else:
                if not isinstance(model, ProbabilisticCircuit):
                    raise ValueError("Conditional generation is only supported for ProbabilisticCircuit models.")
                df_train = exp_config.get_train_data()
                df_with_missing_values = randomly_mask_values(df_train, mask_fraction=1-conditioning_percentage)
                
                # This code is used for conditional generation
                df = model.conditional_generation(device=DEVICE, batch_size=batch_size, condition=df_with_missing_values)
            df.to_csv(os.path.join(csv_samples_path, f"{name}.csv"), index=False)

        with edit_json(filename=times_file_name) as data:
            data[i] = generation_timer.elapsed

    if os.path.exists(times_file_name):
        d = load_json(times_file_name)
        if len(d) == n_samples and conditioning_percentage is None:
            times = np.array(list(d.values()))
            avg_time = np.mean(times)
            std_time = np.std(times)
            store_json({"average": avg_time, "std": std_time}, file=os.path.join(samples_path, "generation_times_summary.json"))

    return samples_path

def main(*, experiment_paths: list[str], conditioning_percentages=[None], overwrite: bool = False, num_samples: int = 20):
    for conditioning_percentage in conditioning_percentages:
        if conditioning_percentage is not None:
            print(f"Sampling with conditioning percentage: {conditioning_percentage:.2f}")
        for experiment_path in experiment_paths:
            print(f"Experiment: {experiment_path}")
            # Setting batch_size = None uses the adaptive batch size functionality
            batch_size = None
            try:
                sample(experiment_path, n_samples=num_samples, overwrite=overwrite, conditioning_percentage=conditioning_percentage, batch_size=batch_size)
            except Exception as e:
                print(f"Error sampling from {experiment_path} with conditioning percentage {conditioning_percentage}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
    print("Done.")
            
if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else '*TabPC*_N'
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    experiment_paths = find_dirs(starting_folder='artifacts/', pattern=pattern)
    experiment_paths = [experiment_path for experiment_path in experiment_paths if 'old' not in experiment_path]
    print(f"Found {len(experiment_paths)} experiment paths")
    print(experiment_paths)
    # conditioning_percentages = np.linspace(0,1,11)
    # conditioning_percentages = [None] + list(conditioning_percentages)
    conditioning_percentages = [None]
    main(experiment_paths=experiment_paths, conditioning_percentages=conditioning_percentages, num_samples=num_samples)
