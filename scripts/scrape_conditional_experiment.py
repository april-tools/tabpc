import pandas as pd
import json
import os
import re
import sys

sys.path.append(".")
from src.util import find_files


def value_to_extract(metric_name, is_test=True):
    if metric_name == 'C2ST':
        return ['xgboost (C2ST)']
    elif metric_name == 'legacy_density' and is_test:
        return ['Shape', 'Trend']
    elif metric_name == 'legacy_density' and not is_test:
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
            

def scrape_metric_cond_expts(metrics_path, output_path, metric_name):

    # Desired behaviour:
    # - Scrape metrics for all models in the metrics_path folder that match the pattern
    # - Save these in the output folder in a csv file with columns:
    #   method, dataset, metric, value1, value2, ... where the (number of) values depend(s) on the metric
    #       e.g. for C2ST we would have value1 = xgboost (C2ST) score,
    #       for legacy_density we would have value1 = density/Shape and value2 = density/Trend, etc.

    pattern = f'*/{metric_name}/*_0.json'

    generated_files = find_files(starting_folder=metrics_path, pattern=pattern)
    generated_files = [filepath for filepath in generated_files if metric_name in filepath]
    
    print(f"Found {len(generated_files)} generated files in {metrics_path}")

    all_rows = []
    for filepath in generated_files:
        # First handle imputed samples
        is_imputed = False
        if "imputed" in filepath:
            print(f"Found imputed sample file: {filepath}")
            is_imputed = True
        with open(filepath, 'r') as f:
            data = json.load(f)

            # Result directories for this experiment are by default of the following forms:
            # For generated samples:
            #     yyyy-mm-dd_hh:mm:ss_{id}_{model_name}_{dataset_name}/new_samples/samples_cond{x.xx}/{metric_name}/{split}_sample_0.json
            # For imputed samples:
            #     yyyy-mm-dd_hh:mm:ss_{id}_{model_name}_{dataset_name}/new_samples/imputed/imputed_cond{x.xx}/{metric_name}/{split}_imputed_0.json
            # We want to extract the dataset name and conditioning percentage for both types of files
            pattern = None
            if is_imputed:
                print(f"Processing imputed sample file: {filepath}")
                pattern = re.search(rf'[\d\-_:]+_[^_]+_[^_]+_(.+)/new_samples/imputed/imputed_cond(\d\.\d+)/{metric_name}/(.+)_imputed_0\.json', filepath)
            else:
                print(f"Processing generated sample file: {filepath}")
                pattern = re.search(rf'[\d\-_:]+_[^_]+_[^_]+_(.+)/new_samples/samples_cond(\d\.\d+)/{metric_name}/(.+)_sample_0\.json', filepath)
            
            dataset = pattern.group(1) if pattern else "unknown"
            conditioning_percentage = pattern.group(2) if pattern else "unknown"
            split = pattern.group(3) if pattern else "unknown"
            if "unknown" in [dataset, conditioning_percentage, split]:
                print(f"Could not extract required metadata from filepath: {filepath}")
                continue

            print("Loaded metrics for dataset:", dataset, "split:", split, "with conditioning percentage:", conditioning_percentage, "and metric:", metric_name)

            method = ""
            if is_imputed:
                method = f'Imputation'
            else:
                method = f'TabPC'
            
            assert split in ['train', 'test'], f"Unexpected split value extracted from filepath: {split}. Expected 'train' or 'test'."

            row = {
                'method': f'{method}',
                'dataset': dataset,
                'split': split,
                'metric': metric_name,
                'conditioning_percentage': conditioning_percentage
            }

            values_to_extract = value_to_extract(metric_name, split == 'test')
            if values_to_extract is None:
                continue

            for value_key in values_to_extract:
                fixed_key = fix_value_key(metric_name, value_key)
                row[fixed_key] = data.get(value_key, None)
            all_rows.append(row)

    df = pd.DataFrame().from_records(all_rows)
    df = df.sort_values(by=['dataset', 'metric', 'method', 'conditioning_percentage'])
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")


def main(metrics_path, output_folder, metrics_dict):
    for metric_name in [metric.name() for metric in metrics_dict.keys()]:
        scrape_metric_cond_expts(
            metrics_path=metrics_path,
            output_path=os.path.join(output_folder, f'{metric_name}.csv'),
            metric_name=metric_name,
        )


if __name__ == "__main__":
    pass
