import os
import sys
import numpy as np
import pandas as pd
import argparse

sys.path.append(".")

from scripts.summarize_metrics_results import sort_rows
from src.util import find_files

# Take in a CSV file with numerical results containing the mean and std
# Save a new CSV file with a new column containing the rank of each method based on the stored metric for each dataset


METRIC_IS_ASCENDING = {
    # Note: True means that lower is better, False means that higher is better
    'alpha_precision': False,
    'beta_recall': False,
    'mle': None, # Depends on dataset (some are regression, some are classification) 
    'lr_detection': False,
    'xgb_detection': False,
    'nmi_l1_weighted_complement': False,
    'nmi_l1_complement': False,
    'nmi_l1_weighted': True,
    'nmi_l1': True,
    'num_params': True,
    'sampling_time': True,
    'shape': False,
    'trend': False,
    'training_time': True,
    'num_parameters': True,
    'density-Shape': False,
    'density-Trend': False,
    'xgboost(C2ST)': False,
    'logistic_regression(C2ST)': False,
    'dcr': None,
    'dcr002': None,
    'dcr005': None,
}

SORTED_METHODS = [
    'CTGAN',
    'TVAE',
    'GReaT',
    'STaSy',
    'CoDi',
    'TabSyn',
    'TabDiff',
    'FullyFactorizedPreprocessed',
    'ShallowMixture',
    'TabPC',
]

SORTED_DATASETS = [
    'adult',
    'beijing',
    'default',
    'diabetes',
    'magic',
    'news',
    'shoppers',
]

REGRESSION_DATASETS = [
    'beijing',
    'news'
]


def na_option_for_metric(metric_name, metrics_ascending=METRIC_IS_ASCENDING):
    # For metrics where higher is better, we want NaN values to be ranked at the bottom (i.e., treated as the worst possible value)
    if metric_name in ['sampling_time', 'training_time']:
        return 'bottom'
    if metrics_ascending.get(metric_name) is False:
        return 'bottom'
    else:
        return 'top'


def rank_methods(df, metric_col, rank_col, ascending=False, na_option='bottom'):
    # Rank methods for each dataset based on the specified metric column
    df[rank_col] = df.groupby('dataset')[metric_col].rank(ascending=ascending, method='min', na_option=na_option).astype(int)
    return df


def main(
        summary_folder: str,
        output_folder: str,
        pattern: str = "*.csv",
        metrics_ascending: dict = METRIC_IS_ASCENDING,
        digits: int = 4,
        ):
    metric_files = find_files(starting_folder=summary_folder, pattern=pattern)
    print(f"Found {len(metric_files)} metric files to process: {metric_files}")

    os.makedirs(output_folder, exist_ok=True)

    for metric_file in metric_files:
        print(f"Processing metric file: {metric_file}")
        metric_name = os.path.splitext(os.path.basename(metric_file))[0]
        if metric_name not in metrics_ascending.keys():
            print(f"Skipping metric file {metric_file} because metric {metric_name} is not in the provided metrics_ascending dictionary.")
            continue

        is_ascending = metrics_ascending[metric_name]
        na_option = na_option_for_metric(metric_name, metrics_ascending)

        df = pd.read_csv(metric_file)

        for method in SORTED_METHODS:
            for dataset in SORTED_DATASETS:
                # If there is no row for this method and dataset, add a new row with NaN values for mean and std
                # This ensures that all methods and datasets are represented in the final ranked table,
                # even if some combinations are missing from the original metric results
                if not ((df['method'] == method) & (df['dataset'] == dataset)).any():
                    new_row = {'method': method, 'dataset': dataset, 'mean': np.nan, 'std': np.nan}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        if metric_name == 'mle':
            # MLE is a special case because for some datasets (regression) lower is better, while for others (classification) higher is better
            for dataset, group in df.groupby('dataset'): 
                is_ascending = dataset in REGRESSION_DATASETS
                df.loc[group.index, 'rank'] = group['mean'].rank(ascending=is_ascending, method='min', na_option=na_option).astype(int)
            df['rank'] = df['rank'].astype(int)
        elif 'dcr' in metric_name:
            # DCR is a special case because there's not necessarily a ranking
            # Instead, just do nothing
            print(f"Skipping ranking for metric {metric_name} because it is a DCR metric where ranking may not be meaningful.")
            pass
        else:
            # Rank methods based on metric
            is_ascending = metrics_ascending[metric_name]
            df['rank'] = df.groupby('dataset')['mean'].rank(ascending=is_ascending, method='min', na_option=na_option).astype(int)#

        # Drop rows with methods not in SORTED_METHODS
        df = df[df['method'].isin(SORTED_METHODS)]

        df.sort_values(by=['method', 'dataset'], inplace=True)

        # Sort methods according to SORTED_METHODS
        df = sort_rows(df, 'method', SORTED_METHODS)

        output_path = f'{output_folder}/{metric_name}.csv'
        df.to_csv(output_path, index=False, float_format=f'%.{digits}f')

        print(f"Ranked results saved to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rank methods based on metric results and save ranked tables to CSV files.')
    parser.add_argument('--summary-folder', type=str, required=True, help='Folder containing the CSV files with metric results to be ranked.')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder where the ranked CSV files will be saved.')
    parser.add_argument('--pattern', type=str, default='*.csv', help='Pattern to match metric CSV files in the summary folder (default: "*.csv").')
    parser.add_argument('--digits', type=int, default=4, help='Number of decimal places to use when saving ranked results to CSV (default: 4).')

    args = parser.parse_args()

    main(
        summary_folder=args.summary_folder,
        output_folder=args.output_folder,
        pattern=args.pattern,
        digits=args.digits,
    )
