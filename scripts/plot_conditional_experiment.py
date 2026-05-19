import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.append(".")
from src.util import find_files


SORTED_METHODS = [
    'CTGAN',
    'TVAE',
    'GOGGLE',
    'GReaT',
    'STaSy',
    'CoDi',
    'TabDDPM',
    'TabSyn',
    'TabDiff',
    'SimpleFactors',
    'FullyFactorizedPreprocessed',
    'ShallowMixture',
    'TabPC',
    'TabPC_conditional',
]


def plot_all_conditional_performance(results_dir, metric_name, out_dir, out_format='png', use_test_results=True):
    """
    Line plot of metric values vs conditioning percentage for TabPC and Imputation across all datasets.
    
    Parameters:
    -----------
    results_dir: str
        Directory containing the metric CSV files for the conditional performance experiment.
    metric_name: str
        Which metric to plot (e.g. 'C2ST', 'mi_l1', 'legacy_density').
    out_dir: str
        Directory to save the generated plots.
    out_format: str, optional
        Format to save the plots (default: 'png').
    use_test_results: bool, optional
        Whether to plot results on the test set (True) or training set (False) (default: True).
    """
    # This uses the new formatting for conditional results
    # Results are now stored as cond_sampling_results/metric_name.csv
    if metric_name not in ['C2ST', 'mi_l1', 'legacy_density']:
        raise NotImplementedError("plot_all_conditional_performance currently only supports C2ST, mi_l1, and legacy_density metric CSVs.")
    
    metric_pattern = f"*{metric_name}.csv"
    metric_csv_files = find_files(starting_folder=results_dir, pattern=metric_pattern)

    if len(metric_csv_files) != 1:
        raise ValueError(f"Expected exactly one CSV file for metric {metric_name} in {results_dir}, but found {len(metric_csv_files)}: {metric_csv_files}")
    metric_df = pd.read_csv(metric_csv_files[0])

    fixed_names = {"C2ST": "C2ST (XGB)", "mi_l1": "wNMIS"}

    print(metric_df.head())

    if use_test_results:
        metric_df = metric_df[metric_df['split'] == 'test']
    else: # if not using results on the test set, use results on the training set
        metric_df = metric_df[metric_df['split'] == 'train']

    # Sort by dataset and then conditioning_percentage for better plotting
    metric_df = metric_df.sort_values(by=['dataset', 'conditioning_percentage'])

    metric_col_name = ''
    if metric_name == 'C2ST':
        metric_col_names = ['XGB-C2ST']
    elif metric_name == 'mi_l1':
        metric_col_names = ['wNMIS']
    elif metric_name == 'legacy_density':
        metric_col_names = ['Shape', 'Trend']
    else:
        raise NotImplementedError("plot_all_conditional_performance currently only supports C2ST, mi_l1, legacy_density metric CSVs.")
    
    # Plot TabPC and Imputed samples with different line styles, but same marker style and color corresponding to dataset
    imputed_df = metric_df[metric_df['method'] == 'Imputation']
    tabpc_df = metric_df[metric_df['method'] == 'TabPC']

    dashes = {"adult": (2, 2), "beijing": (2, 2), "default": (2, 2), "diabetes": (2, 2), "magic": (2, 2), "news": (2, 2), "shoppers": (2, 2)}

    for metric_col_name in metric_col_names:
        fig, ax = plt.subplots(ncols=1)
        # Make marker style and color correspond to dataset
        # Make line style also correspond to method, with TabPC as solid and MeanImputation as dashed
        sns.lineplot(ax=ax, data=tabpc_df, x='conditioning_percentage', y=metric_col_name, hue='dataset', markers=True, style='dataset', dashes=False)
        sns.lineplot(ax=ax, data=imputed_df, x='conditioning_percentage', y=metric_col_name, hue='dataset', markers=True, style='dataset', dashes=dashes, alpha=0.5, legend=False)
        
        handles, labels = ax.get_legend_handles_labels()
        # Add another heading for method in the legend below datasets
        # This should have no line or marker, just be a text label
        handles.append(plt.Line2D([0], [0], color='black', label='', linestyle='None', markersize=0))
        labels.append('Method')

        # Add a single legend for TabPC with solid line
        tabpc_handle = plt.Line2D([0], [0], color='grey', label='TabPC', linestyle='-', markersize=6)
        handles.append(tabpc_handle)
        labels.append('TabPC')
        # Add a single legend for MeanImputation with dashed line alpha 0.5
        mean_imp_handle = plt.Line2D([0], [0], color='grey', label='Imputation', linestyle='--', markersize=6, alpha=0.5)
        handles.append(mean_imp_handle)
        labels.append('Imputation')
        
        handles.insert(0, plt.Line2D([0], [0], color='black', label='Dataset', linestyle='None', markersize=0))
        labels.insert(0, 'Dataset')

        ax.legend(handles=handles, labels=labels, title='', framealpha=0.65, ncol=2, loc='center')

        if metric_name == 'legacy_density':
            fixed_metric_name = metric_col_name
        else:
            fixed_metric_name = fixed_names.get(metric_name, metric_name)

        out_path = os.path.join(out_dir, f"{fixed_metric_name}_all_datasets_conditional_performance_{'test' if use_test_results else 'train'}.{out_format}")

        ax.set(xlabel='Conditional Percentage', ylabel=fixed_metric_name)
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved as: {out_path}")
        plt.close()
