import sys
import argparse
import gc
import torch
import os

sys.path.append(".")
from train_scripts.pc_lls import main as train_pc_lls
from scripts.compute_likelihoods import compute_likelihoods as compute_likelihoods
from scripts.sample import main as sample_main
from scripts.array_evaluate import evaluate as evaluate_metrics
from scripts.scrape_likelihoods_experiment import main as scrape_likelihoods
from scripts.plot_likelihoods_experiment import plot_metric_per_likelihood
from src.metrics.metrics import *

LL_RESULTS_FOLDER = 'article_material/ll_results'
LL_METRICS_FOLDER = 'artifacts/ll_metrics_results'
LL_MODELS_FOLDER = 'artifacts/ll_models'

METRICS = {
    C2ST() : 2,
    # Add these metrics if desired
    # LegacyDensity() : 20,
    # MI_l1() : 20,
    # Mle() : 2,
    # QuantileDcr() : 2,
}

TABDIFF_PATHS = [
    f"data/{name}" for name in ["adult", "beijing", "default", "diabetes", "magic",  "news", "shoppers"]
]

DIGITS = 4

# Hyperparameters to sweep over
LEARNING_RATES = [0.1, 0.25, 0.5]
BATCH_SIZES = [64, 256, 512]
NUM_UNITS = [128, 512, 2048]

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the likelihood experiments for TabPC.')
    parser.add_argument('--train', "-t", action='store_true', help='Whether to not run the training of PCs (default: False)')
    parser.add_argument('--compute-likelihoods', "-l", action='store_true', help='Whether to not compute the likelihoods of the trained PCs (default: False)')
    parser.add_argument('--sample', "-s", action='store_true', help='Whether to not sample from the trained PCs (default: False)')
    parser.add_argument('--evaluate', "-e", action='store_true', help='Whether to not evaluate the samples from the trained PCs (default: False)')
    parser.add_argument('--scrape', "-r", action='store_true', help='Whether to not scrape the likelihoods and metrics results and summarize them in csv files (default: False)')
    parser.add_argument('--models-folder', type=str, help='Folder for trained models', default=LL_MODELS_FOLDER)
    parser.add_argument('--metrics-folder', type=str, help='Target folder for metrics', default=LL_METRICS_FOLDER)
    parser.add_argument('--results-folder', type=str, help='Target folder for results', default=LL_RESULTS_FOLDER)
    parser.add_argument('--plots-folder', type=str, help='Target folder for plots', default=os.path.join(LL_RESULTS_FOLDER, "plots"))
    parser.add_argument('--seed', help='Random seed (int or None; default: 0)', default=0)
    parser.add_argument('--dataset', type=str, help='Dataset to run the experiment on (default: all)', default="all")
    parser.add_argument('--do-plots', action='store_true', help='Whether to also generate the plots after running the experiment (default: False)')

    args = parser.parse_args()

    # Create specified folders if they don't exist
    os.makedirs(args.models_folder, exist_ok=True)
    os.makedirs(args.metrics_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)

    if args.dataset not in ["all", "adult", "beijing", "default", "diabetes", "magic",  "news", "shoppers"]:
        raise ValueError(f"Invalid dataset: {args.dataset}. Must be one of: all, adult, beijing, default, diabetes, magic, news, shoppers")
    datasets = TABDIFF_PATHS if args.dataset == "all" else [f"data/{args.dataset}"]

    if args.train:
        # Step 1: Train PCs with different hyperparameters
        for dataset_path in datasets:
            for num_units in NUM_UNITS:
                for batch_size in BATCH_SIZES:
                    for lr in LEARNING_RATES:
                        if dataset_path == "data/news" and num_units == 2048:
                            num_units = 1024 # reduce number of units for news dataset to avoid OOM
                            print(f"Reducing number of units from 2048 to {num_units} on news dataset to avoid OOM")
                        print(f"Training PC with {num_units} units, batch size {batch_size}, learning rate {lr} on {dataset_path.split('/')[-1]} dataset")
                        # The trained model name follows the pattern {dataset_name}_units_{num_units}_batch_{batch_size}_lr_{lr}
                        train_pc_lls(path=f"{dataset_path}", num_units=num_units, batch_size=batch_size, lr=lr, seed=0, target_folder=args.models_folder)
                        free_memory()
    
    if args.compute_likelihoods:
        # Step 2: Compute the BPD on the validation set for each model and store it in a json file in the model folder
        for model_path in os.listdir(args.models_folder):
            try:
                # Skip if likelihoods have already been computed for this model
                likelihoods_path = os.path.join(args.models_folder, model_path, "likelihoods.json")
                if os.path.exists(likelihoods_path):
                    print(f"Likelihoods already computed for model {model_path}, skipping...")
                    continue
                print(f"Computing likelihoods for model {model_path}")
                compute_likelihoods(experiment_path=os.path.join(args.models_folder, model_path))
            except Exception as e:
                print(f"Error computing likelihoods for model {model_path}: {e}")
            free_memory()

    if args.sample:
        # Step 3: Sample from the trained PCs
        experiment_paths = os.listdir(args.models_folder)
        experiment_paths = [os.path.join(args.models_folder, path) for path in experiment_paths]
        sample_main(experiment_paths=experiment_paths, num_samples=1)

    if args.evaluate:
        # Step 4: Evaluate the samples with the selected metrics (in particular C2ST) and store the results in a json file in the model folder
        evaluate_metrics(generated_data_folder=args.models_folder, original_data_folder="data", metrics=METRICS, target_folder=args.metrics_folder, pattern="*sample_0.csv")
        free_memory()

    if args.scrape:
        # Step 5: Summarize in csvs
        scrape_likelihoods(experiments_path=args.models_folder, metrics_path=args.metrics_folder, output_folder=args.results_folder)

    if args.do_plots:
        # Step 6: Generate plots of metric vs likelihood
        plot_split = 'validation' # whether to plot validation or test likelihoods
        palette = "flare" # color palette for the points in the plot (can be any seaborn palette)
        use_title = True # whether to use dataset names as titles for each subplot
        regression_line = True # whether to fit and plot a regression line (using HuberRegressor for robustness to outliers)
        remove_legend = False # whether to remove the legend from the plot
        
        for dataset in datasets:
            dataset_name = dataset.split('/')[-1]
            plot_filepath = os.path.join(args.plots_folder, f"c2st_vs_{plot_split}_bpd_{dataset_name}.png") # path to save the plot
            plot_metric_per_likelihood(
                ll_results_dir=args.results_folder,
                split=plot_split,
                out_path=plot_filepath,
                dataset_names=[dataset_name],
                palette=palette,
                use_title=use_title,
                regression_line=regression_line,
                remove_legend=remove_legend
            )
