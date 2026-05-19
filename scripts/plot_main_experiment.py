import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
C2ST_XGB_CSV_PATH = 'article_material/our_results/xgb_detection.csv'
C2ST_LR_CSV_PATH = 'article_material/our_results/lr_detection.csv'
ALPHA_PRECISION_CSV_PATH = 'article_material/our_results/alpha_precision.csv'
TRAINING_TIME_CSV_PATH = 'article_material/our_results/training_time.csv'


def fix_metric_name(name: str) -> str:
    """
    Fix metric names for better readability in plots.
    """
    name_mappings = {
        'alpha_precision': 'Alpha Precision',
        'beta_recall': 'Beta Recall',
        'dcr': 'Distance to Closest Record (DCR)',
        'lr_detection': 'C2ST (LR)',
        'xgb_detection': 'C2ST (XGB)',
        'mle': 'Machine Learning Efficiency (MLE)',
        'nmi_l1_weighted': 'wNORMIE',
        'nmi_l1': 'NORMIE',
        'trend': 'Trend',
        'nmi_l1_weighted_complement': 'wNMIS',
        'nmi_l1_complement': 'NMIS',
        'logistic_regression (C2ST)': 'C2ST (LR)',
        'xgboost (C2ST)': 'C2ST (XGB)',
        'xgboost(C2ST)': 'C2ST (XGB)',
        'logistic_regression(C2ST)': 'C2ST (LR)',
    }
    return name_mappings.get(name, name.replace('_', ' ').title())


def fix_method_name(name: str) -> str:
    """
    Fix method names for better readability in plots.
    """
    name_mappings = {
        'CTGAN': 'CTGAN',
        'TVAE': 'TVAE',
        'GOGGLE': 'GOGGLE',
        'GReaT': 'GReaT',
        'STaSy': 'STaSy',
        'CoDi': 'CoDi',
        'TabDDPM': 'TabDDPM',
        'TabSyn': 'TabSyn',
        'TabDiff': 'TabDiff',
        'SimpleFactors': r'SimpleFactors $\it{(ours)}$',
        'FullyFactorizedPreprocessed': r'FF $\it{(ours)}$',
        'ShallowMixture': r'SM $\it{(ours)}$',
        'TabPC': r'TabPC $\it{(ours)}$',
        'TabPC_conditional': r'TabPC Conditional $\it{(ours)}$',
    }
    return name_mappings.get(name, name.replace('_', ' ').title())


def plot_metric_vs_time(metric_1_csv, metric_2_csv, train_time_csv, methods, out_path, figsize=(10, 6), palette="Set2"):
    """
    Plot metric vs training time for different methods.
    
    Parameters:
    -----------
    metric_1_csv : str
        Path to the CSV file containing the first metric data
    metric_2_csv : str
        Path to the CSV file containing the second metric data
    train_time_csv : str
        Path to the CSV file containing training time data
    methods : list
        List of methods to include in the plot
    figsize : tuple
        Figure size (width, height)
    palette : str
        Color palette for the plot
    """
    # Read the first metric data
    metric_1_df = pd.read_csv(metric_1_csv)
    # Get only specified methods
    metric_1_df = metric_1_df[metric_1_df['method'].isin(methods)]
    # Rename mean to metric_mean, std to metric_std for clarity
    metric_1_df = metric_1_df.rename(columns={'mean': 'metric_1_mean', 'std': 'metric_1_std'})

    # Repeat for the second metric
    metric_2_df = pd.read_csv(metric_2_csv)
    metric_2_df = metric_2_df[metric_2_df['method'].isin(methods)]
    metric_2_df = metric_2_df.rename(columns={'mean': 'metric_2_mean', 'std': 'metric_2_std'})
    
    # Read the training time data
    train_time_df = pd.read_csv(train_time_csv)
    # Get only specified methods
    train_time_df = train_time_df[train_time_df['method'].isin(methods)]
    # Rename mean to train_time
    train_time_df = train_time_df.rename(columns={'mean': 'train_time'})

    # Merge the metric data with training time data on method and dataset
    merged_df_1 = pd.merge(metric_1_df, train_time_df, on=['method', 'dataset'])
    merged_df_2 = pd.merge(metric_2_df, train_time_df, on=['method', 'dataset'])

    # Use fixed method names for better readability
    merged_df_1['method'] = merged_df_1['method'].apply(fix_method_name)
    merged_df_2['method'] = merged_df_2['method'].apply(fix_method_name)
    methods = [fix_method_name(m) for m in methods]

    # Ensure that TabPC is always the first method in the legend for better visibility
    # Sort the rest of methods by name for consistent coloring
    tabpc_methodname = fix_method_name('TabPC')
    if tabpc_methodname in methods:
        merged_without_tabpc_df_1 = merged_df_1[merged_df_1['method'] != tabpc_methodname]
        # Sort remaining methods by method for consistent coloring
        merged_without_tabpc_df_1 = merged_without_tabpc_df_1.sort_values(by='method')
        tabpc_df_1 = merged_df_1[merged_df_1['method'] == tabpc_methodname]
        # Place TabPC at the start for better visibility
        merged_df_1 = pd.concat([tabpc_df_1, merged_without_tabpc_df_1])

        merged_without_tabpc_df_2 = merged_df_2[merged_df_2['method'] != tabpc_methodname]
        # Sort remaining methods by method for consistent coloring
        merged_without_tabpc_df_2 = merged_without_tabpc_df_2.sort_values(by='method')
        tabpc_df_2 = merged_df_2[merged_df_2['method'] == tabpc_methodname]
        # Place TabPC at the start for better visibility
        merged_df_2 = pd.concat([tabpc_df_2, merged_without_tabpc_df_2])
    else:
        # Sort by method for consistent coloring
        merged_df_1 = merged_df_1.sort_values(by='method')
        merged_df_2 = merged_df_2.sort_values(by='method')

    # Capitalize column names for better readability in the plot
    method_name = r"$\bf{Method}$"
    dataset_name = r"$\bf{Dataset}$"
    merged_df_1 = merged_df_1.rename(columns={'method': method_name, 'dataset': dataset_name})
    merged_df_2 = merged_df_2.rename(columns={'method': method_name, 'dataset': dataset_name})

    fig, axs = plt.subplots(ncols=2, sharey=False)

    axs[0].grid(axis='y', alpha=0.3, zorder=0.1)
    axs[1].grid(axis='y', alpha=0.3, zorder=0.1)
    sns.scatterplot(ax=axs[0], data=merged_df_1, x='train_time', y='metric_1_mean', hue=method_name, style=dataset_name, palette=palette)
    sns.scatterplot(ax=axs[1], data=merged_df_2, x='train_time', y='metric_2_mean', hue=method_name, style=dataset_name, palette=palette)

    # Set x axis as log scale for better visibility
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    # Customize the plot
    metric_1_name = os.path.splitext(os.path.basename(metric_1_csv))[0]
    fixed_name_1 = fix_metric_name(metric_1_name)
    metric_2_name = os.path.splitext(os.path.basename(metric_2_csv))[0]
    fixed_name_2 = fix_metric_name(metric_2_name)

    axs[0].set_xlabel('training time (seconds)')
    axs[1].set_xlabel('training time (seconds)')
    
    # Remove y label from individual subplots and set a common y label for both
    axs[0].set_ylabel(fixed_name_1)
    axs[1].set_ylabel(fixed_name_2)

    # Move legend to above plots and split into two columns
    axs[0].legend_.remove()
    axs[1].legend_.remove()

    # Add only the hue in the legend, not the style
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Remove style entries (datasets) from legend
    by_label = {k: v for k, v in by_label.items() if k in merged_df_1[method_name].unique()}
    # Put legend in the center above the two plots
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=1, bbox_to_anchor=(0.775, 0.8))
    
    # Save the plot instead of showing it
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved as: {out_path}")
    plt.close()


def scatterplot_metric_vs_metric(metric_1_csv, metric_2_csv, methods, out_path, figsize=(10, 6), palette="Set2"):
    """
    Plot one metric vs another metric across all datasets for a given method.
    
    Parameters:
    -----------
    metric_1_csv : str
        Path to the CSV file containing the first metric data
    metric_2_csv : str
        Path to the CSV file containing the second metric data
    method : str
        Method to include in the plot
    figsize : tuple
        Figure size (width, height)
    palette : str
        Color palette for the plot
    """

    # Read the first metric data
    metric_1_df = pd.read_csv(metric_1_csv)
    metric_1_df = metric_1_df[metric_1_df['method'].isin(methods)]
    metric_1_name = os.path.splitext(os.path.basename(metric_1_csv))[0]
    fixed_metric_1_name = fix_metric_name(metric_1_name)
    metric_1_df = metric_1_df.rename(columns={'mean': f'{fixed_metric_1_name}_mean', 'std': f'{fixed_metric_1_name}_std'})
    
    # Read the second metric data
    metric_2_df = pd.read_csv(metric_2_csv)
    metric_2_df = metric_2_df[metric_2_df['method'].isin(methods)]
    metric_2_name = os.path.splitext(os.path.basename(metric_2_csv))[0]
    fixed_metric_2_name = fix_metric_name(metric_2_name)
    metric_2_df = metric_2_df.rename(columns={'mean': f'{fixed_metric_2_name}_mean', 'std': f'{fixed_metric_2_name}_std'})

    merged_df = pd.merge(metric_1_df, metric_2_df, on=['method', 'dataset'])

    # Make sure that TabPC is first in the method list for better visibility
    # tabpc_methodname = fix_method_name('TabPC')
    if 'TabPC' in methods:
        merged_without_tabpc_df = merged_df[merged_df['method'] != 'TabPC']
        # Sort remaining methods by method for consistent coloring
        merged_without_tabpc_df = merged_without_tabpc_df.sort_values(by='method')
        tabpc_df = merged_df[merged_df['method'] == 'TabPC']
        # Place TabPC at the start for better visibility
        merged_df = pd.concat([tabpc_df, merged_without_tabpc_df])
    else:
        # Sort by method for consistent coloring
        merged_df = merged_df.sort_values(by='method')

    # Place FF model last if it is present
    if 'FullyFactorizedPreprocessed' in methods:
        merged_without_ff_df = merged_df[merged_df['method'] != 'FullyFactorizedPreprocessed']
        ff_df = merged_df[merged_df['method'] == 'FullyFactorizedPreprocessed']
        # Place FF at the end for better visibility
        merged_df = pd.concat([merged_without_ff_df, ff_df])

    # Rename method to Method and dataset to Dataset for better readability
    method_name = r"\textbf{Method}"
    dataset_name = r"\textbf{Dataset}"
    merged_df = merged_df.rename(columns={'method': method_name, 'dataset': dataset_name})

    print(merged_df)

    # Fix method names for better readability
    merged_df[method_name] = merged_df[method_name].apply(fix_method_name)

    # Create a scatter plot of metric vs training time

    fig, ax = plt.subplots(ncols=1)

    ax = sns.scatterplot(data=merged_df, x=f'{fixed_metric_1_name}_mean', y=f'{fixed_metric_2_name}_mean', hue=method_name, style=dataset_name, palette=palette, s=60)

    plt.xlabel(fixed_metric_1_name)
    plt.ylabel(fixed_metric_2_name)
    # Set legend to two columns
    plt.legend(ncol=2, loc='best')
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot instead of showing it
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved as: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replicate article plots from main experiment results')
    parser.add_argument('--time-plot-metric_1', type=str, default=C2ST_XGB_CSV_PATH,
                        help='Path to the first metric CSV file (defaults to "article_material/our_results/xgb_detection.csv")')
    parser.add_argument('--time-plot-metric_2', type=str, default=ALPHA_PRECISION_CSV_PATH,
                        help='Path to the second metric CSV file (defaults to "article_material/our_results/alpha_precision.csv")')
    parser.add_argument('--train-time-csv', type=str, default=TRAINING_TIME_CSV_PATH,
                        help='Path to the training time CSV file (defaults to "article_material/our_results/training_time.csv")')
    parser.add_argument('--methods-time-plot', type=str, nargs='+', required=True, help='List of methods to include in the plot')
    parser.add_argument('--out-path', type=str, required=True, help='Path to save the generated plot')
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 6), help='Figure size (width, height)')
    parser.add_argument('--palette', type=str, default='Set2', help='Color palette for the plot')
    parser.add_argument('--scatter-plot-metric_1', type=str, default=C2ST_XGB_CSV_PATH,
                        help='Path to the first metric CSV file for scatter plot (defaults to "article_material/our_results/xgb_detection.csv")')
    parser.add_argument('--scatter-plot-metric_2', type=str, default=C2ST_LR_CSV_PATH,
                        help='Path to the second metric CSV file for scatter plot (defaults to "article_material/our_results/lr_detection.csv")')
    parser.add_argument('--methods-scatter-plot', type=str, nargs='+', required=True, help='List of methods to include in the scatter plot')
    args = parser.parse_args()

    plot_metric_vs_time(
        metric_1_csv=args.time_plot_metric_1,
        metric_2_csv=args.time_plot_metric_2,
        train_time_csv=args.train_time_csv,
        methods=args.methods_time_plot,
        out_path=args.out_path,
        figsize=tuple(args.figsize),
        palette=args.palette,
    )

    scatterplot_metric_vs_metric(
        metric_1_csv=args.scatter_plot_metric_1,
        metric_2_csv=args.scatter_plot_metric_2,
        methods=args.methods_scatter_plot,
        out_path=args.out_path.replace('.png', '_scatter.png'),
        figsize=tuple(args.figsize),
        palette=args.palette,
    )
