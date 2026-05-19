import pandas as pd
import json
import os
import re
import sys
import argparse

sys.path.append(".")
from src.util import find_files


def value_to_extract(metric_name):
    if metric_name == 'C2ST':
        return ['xgboost (C2ST)']
    elif metric_name == 'legacy_density':
        return ['density/Shape', 'density/Trend']
    elif metric_name == 'mi_l1':
        return ['nmi_l1_weighted_complement']
    else:
        return None
    

def fix_value_key(metric_name, value_key):
    if metric_name == 'legacy_density':
        if value_key == 'density/Shape':
            return 'Shape'
        elif value_key == 'density/Trend':
            return 'Trend'
    elif metric_name == 'mi_l1':
        if value_key == 'nmi_l1_weighted_complement':
            return 'wNMIS'
    elif metric_name == 'C2ST':
        if value_key == 'xgboost (C2ST)':
            return 'XGB-C2ST'
    return value_key
            

def scrape_metric_ll_expts(metrics_path, output_path, metric_name):

    # Desired behaviour:
    # - Scrape metrics for all models in the metrics_path folder that match the pattern
    # - Save these in the output folder in a csv file with columns:
    #   method, dataset, metric, value1, value2, ... where the (number of) values depend(s) on the metric
    #       e.g. for C2ST we would have value1 = xgboost (C2ST) score,
    #       for legacy_density we would have value1 = density/Shape and value2 = density/Trend, etc.

    pattern = f'*_lls/*/{metric_name}/sample_0.json'

    generated_files = find_files(starting_folder=metrics_path, pattern=pattern)
    generated_files = [filepath for filepath in generated_files if metric_name in filepath]
    
    print(f"Found {len(generated_files)} generated files in {metrics_path}")

    all_rows = []
    for filepath in generated_files:
        with open(filepath, 'r') as f:
            data = json.load(f)

            # Directories for this experiment are of the form
            # yyyy-mm-dd_hh:mm:ss_{id}_{model_name}_{dataset_name}_{num_units}_{batch_size}_{lr}_lls/.../
            # We just want to extract the dataset name and number of units from the filename
            ll_pattern = re.search(rf'[\d\-_:]+_[^_]+_[^_]+_(.+)_(\d+)_(\d+)_(.+)_lls/', filepath)
            dataset = ll_pattern.group(1) if ll_pattern else "unknown"
            num_units = ll_pattern.group(2) if ll_pattern else "unknown"
            batch_size = ll_pattern.group(3) if ll_pattern else "unknown"
            lr = ll_pattern.group(4) if ll_pattern else "unknown"
            if "unknown" in [dataset, num_units, batch_size, lr]:
                # print("dataset:", dataset, "num_units:", num_units, "batch_size:", batch_size, "lr:", lr)
                print(f"Could not extract required metadata from filepath: {filepath}")
                continue
            print("Loaded metrics for dataset:", dataset, "with num_units:", num_units, "batch_size:", batch_size, "lr:", lr, "and metric:", metric_name)

            row = {
                'method': f'TabPC_{num_units}_{batch_size}_{lr}',
                'dataset': dataset,
                'metric': metric_name,
            }

            values_to_extract = value_to_extract(metric_name)
            if values_to_extract is None:
                continue

            for value_key in values_to_extract:
                fixed_key = fix_value_key(metric_name, value_key)
                row[fixed_key] = data.get(value_key, None)
            all_rows.append(row)

    df = pd.DataFrame().from_records(all_rows)
    df = df.sort_values(by=['dataset', 'metric', 'method'])
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")


def scrape_likelihood_ll_expts(experiments_path, output_folder):

    # Desired behaviour:
    # - Scrape likelihoods for all models in the experiments_path folder
    # - Save these in the output folder in a csv file with columns:
    #   method, dataset, split, size, total_log_likelihood, bpd, mean_NLL

    pattern = '*_lls/likelihoods.json'

    generated_files = find_files(starting_folder=experiments_path, pattern=pattern)
    
    print(f"Found {len(generated_files)} generated files in {experiments_path}")

    likelihoods = {}
    for filepath in generated_files:
        with open(filepath, 'r') as f:
            data = json.load(f)

            # Directories for this experiment are of the form
            # yyyy-mm-dd_hh:mm:ss_{id}_{model_name}_{dataset}_{num_units}_{batch_size}_{lr}_lls
            # We just want to extract the dataset name and number of units from the filename
            pattern = re.search(rf'[\d\-_:]+_[^_]+_[^_]+_(.+)_(\d+)_(\d+)_(.+)_lls/likelihoods\.json', filepath)
            dataset = pattern.group(1) if pattern else "unknown"
            num_units = pattern.group(2) if pattern else "unknown"
            batch_size = pattern.group(3) if pattern else "unknown"
            lr = pattern.group(4) if pattern else "unknown"
            print("Loaded likelihoods for dataset:", dataset, "with num_units:", num_units, "batch_size:", batch_size, "lr:", lr)

            if "unknown" in [dataset, num_units, batch_size, lr]:
                print(f"Could not extract required metadata from filepath: {filepath}")
                continue

            rows = []
            for split in ['train', 'validation', 'test']:
                if split in data:
                    row = {
                        'method': f'TabPC_{num_units}_{batch_size}_{lr}',
                        'dataset': dataset,
                        'split': split,
                        'size': data[split]['size'],
                        'total_log_likelihood': data[split]['total_log_likelihood'],
                        'bpd': data[split]['bpd'],
                        'mean_NLL': data[split]['mean_NLL'],
                    }
                    rows.append(row)

            likelihoods_key = f'{dataset}_{num_units}_{batch_size}_{lr}'
            likelihoods[likelihoods_key] = rows
    
    # Now save to CSV
    all_rows = []
    for _, rows in likelihoods.items():
        all_rows.extend(rows)
    df = pd.DataFrame().from_records(all_rows)
    df = df.sort_values(by=['dataset', 'split'])
    output_path = os.path.join(output_folder, 'likelihoods.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved likelihoods to {output_path}")


def main(experiments_path, metrics_path, output_folder, scrape_likelihoods=False, scrape_metrics=False):

    if scrape_likelihoods:
        scrape_likelihood_ll_expts(
            experiments_path=experiments_path,
            output_folder=output_folder
        )

    if scrape_metrics:
        for metric_name in ['C2ST']: # FIXME: for now just scrape C2ST, we can add the other metrics later if needed
            scrape_metric_ll_expts(
            metrics_path=metrics_path,
            output_path=os.path.join(output_folder, f'{metric_name}.csv'),
            metric_name=metric_name,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scrape likelihoods and metrics from the likelihood experiments for TabPC.')
    parser.add_argument('--experiments-path', type=str, help='Path to the likelihood experiments results', default='artifacts/ll_experiments')
    parser.add_argument('--metrics-path', type=str, help='Path to the likelihood metrics results', default='artifacts/ll_metrics_results')
    parser.add_argument('--output-folder', type=str, help='Target folder for scraped results', default='artifacts/ll_experiments/scraped_results')
    parser.add_argument('--scrape-likelihoods', action='store_true', help='Whether to scrape the likelihoods results (default: False)')
    parser.add_argument('--scrape-metrics', action='store_true', help='Whether to scrape the metrics results (default: False)')

    args = parser.parse_args()

    main(
        experiments_path=args.experiments_path,
        metrics_path=args.metrics_path,
        output_folder=args.output_folder,
        scrape_likelihoods=args.scrape_likelihoods,
        scrape_metrics=args.scrape_metrics,
    )

    # if len(sys.argv) < 2:
    #     print("Please provide the dataset name as an argument")
    #     sys.exit(1)
    # dataset = sys.argv[1]

    # read_likelihoods_ll_expts(
    #     base_dir=f'artifacts/',
    #     pattern=f'*{dataset}*_lls/likelihoods.json',
    #     output_path=f'artifacts/ll_experiments/{dataset}_likelihoods.csv',
    # )

    # for metric_name in ['C2ST', 'legacy_density', 'mi_l1']:
    #     scrape_metric_ll_expts(
    #         base_dir=f'artifacts/ll_metrics_results/{dataset}',
    #         pattern=f'*{dataset}*_lls/*/sample_0.json',
    #         output_path=f'artifacts/ll_experiments/{dataset}_{metric_name}.csv',
    #         metric_name=metric_name,
    #     )
