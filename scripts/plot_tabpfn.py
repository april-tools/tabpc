import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import sys
import tqdm 
import json
import numpy as np
from sklearn.linear_model import HuberRegressor

from tueplots import bundles, figsizes, fontsizes


SORTED_METHODS = [
    # 'CTGAN',
    # 'TVAE',
    # 'GOGGLE',
    # 'GReaT',
    # 'STaSy',
    # 'CoDi',
    # 'TabDDPM',
    # 'SimpleFactors',
    # 'FullyFactorizedPreprocessed',
    # 'ShallowMixture',
    'TabPC',
    'TabDiff',
    'TabSyn',
    'TabPFN',
]


def sort_rows(df: pd.DataFrame, col: str, names: list) -> pd.DataFrame:
    """
    Sort the rows based on the values of a given column.
    If the value do not appear in the list of given names, the values are put at the end.
    The rest of the rows are sorted according to the given list of names.
    """
    # Create a categorical type with the specified order
    cat_type = pd.CategoricalDtype(categories=names, ordered=True)
    df[col] = df[col].astype(cat_type)
    return df.sort_values([col, 'dataset'])


def find_files(*, starting_folder: str = ".", pattern: str):
    """
    find all files that match the given pattern, starting from the given folder and going down the directory tree
    """
    matches = []
    for root, _, files in os.walk(starting_folder):
        for filename in files:
            full_name = os.path.join(root, filename)
            if fnmatch.fnmatch(full_name, pattern):
                matches.append(full_name)
    return matches


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
        # 'TabPFN': r'TabPFN$\textsuperscript{*}$',
        'TabPFN': 'TabPFN',
    }
    return name_mappings.get(name, name.replace('_', ' ').title())



def fix_metric_name(name: str) -> str:
    """
    Fix metric names for better readability in plots.
    """
    name_mappings = {
        'alpha_precision': 'alpha precision (naive)',
        'beta_recall': 'beta recall (naive)',
        'alpha_precision_oc': 'alpha precision',
        'beta_recall_oc': 'beta recall',
        'dcr': 'distance to closest record (DCR)',
        'lr_detection': 'C2ST (LR)',
        'xgb_detection': 'C2ST (XGB)',
        'mle': 'machine learning efficacy (MLE)',
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


def plot_metric_vs_time(metric_1_csv, metric_2_csv, train_time_csv, sampling_time_csv, methods, out_path, figsize=(10, 6), palette="Set2", train_n_sample=True):
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
    sampling_time_csv : str
        Path to the CSV file containing sampling time data
    methods : list
        List of methods to include in the plot
    figsize : tuple
        Figure size (width, height)
    palette : str
        Color palette for the plot
    """
    # Read the metric data
    metric_1_df = pd.read_csv(metric_1_csv)
    # Get only specified methods
    metric_1_df = metric_1_df[metric_1_df['method'].isin(methods)]
    # Rename mean to metric_mean, std to metric_std for clarity
    metric_1_df = metric_1_df.rename(columns={'mean': 'metric_1_mean', 'std': 'metric_1_std'})

    metric_2_df = pd.read_csv(metric_2_csv)
    # Get only specified methods
    metric_2_df = metric_2_df[metric_2_df['method'].isin(methods)]
    # Rename mean to metric_mean, std to metric_std for clarity
    metric_2_df = metric_2_df.rename(columns={'mean': 'metric_2_mean', 'std': 'metric_2_std'})
    
    # Read the training time data
    train_time_df = pd.read_csv(train_time_csv)
    # Get only specified methods
    train_time_df = train_time_df[train_time_df['method'].isin(methods)]
    # Rename mean to train_time
    train_time_df = train_time_df.rename(columns={'mean': 'train_time'})

    sample_time_df = pd.read_csv(sampling_time_csv)
    sample_time_df = sample_time_df[sample_time_df['method'].isin(methods)]
    sample_time_df = sample_time_df.rename(columns={'mean': 'sampling_time'})

    # print(train_time_df)
    # print(sample_time_df)

    for method in methods:
        if method not in train_time_df['method'].unique():
            for dataset in train_time_df['dataset'].unique():
                new_row = {
                    'method': [method],
                    'dataset': [dataset],
                    'train_time': [0.0],
                    'std': [0.0],
                }
                df_new_row = pd.DataFrame(new_row)
                train_time_df = pd.concat([train_time_df, df_new_row])

    print(train_time_df)

    merged_times = pd.merge(train_time_df, sample_time_df, on=['method', 'dataset'])
    if train_n_sample:
        merged_times['total_time'] = merged_times['train_time'] + merged_times['sampling_time']
    else:
        merged_times['total_time'] = merged_times['train_time']

    print(merged_times)

    # Add together training and sampling times

    merged_df_1 = pd.merge(metric_1_df, merged_times, on=['method', 'dataset'])
    merged_df_2 = pd.merge(metric_2_df, merged_times, on=['method', 'dataset'])

    # Use fixed method names for better readability
    merged_df_1['method'] = merged_df_1['method'].apply(fix_method_name)
    merged_df_2['method'] = merged_df_2['method'].apply(fix_method_name)
    methods = [fix_method_name(m) for m in methods]

    tabpc_methodname = fix_method_name('TabPC')

    FIXED_SORTED_METHOD_NAMES = []
    for sorted_method_name in SORTED_METHODS:
        FIXED_SORTED_METHOD_NAMES.append(fix_method_name(sorted_method_name))

    if tabpc_methodname in methods:
        merged_without_tabpc_df_1 = merged_df_1[merged_df_1['method'] != tabpc_methodname]
        # Sort remaining methods by method for consistent coloring
        merged_without_tabpc_df_1 = merged_without_tabpc_df_1.sort_values(by='method')
        merged_without_tabpc_df_1 = sort_rows(merged_without_tabpc_df_1, 'method', FIXED_SORTED_METHOD_NAMES)
        tabpc_df_1 = merged_df_1[merged_df_1['method'] == tabpc_methodname]
        # Place TabPC at the start for better visibility
        merged_df_1 = pd.concat([tabpc_df_1, merged_without_tabpc_df_1])

        merged_without_tabpc_df_2 = merged_df_2[merged_df_2['method'] != tabpc_methodname]
        # Sort remaining methods by method for consistent coloring
        merged_without_tabpc_df_2 = merged_without_tabpc_df_2.sort_values(by='method')
        merged_without_tabpc_df_2 = sort_rows(merged_without_tabpc_df_2, 'method', FIXED_SORTED_METHOD_NAMES)
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
    
    # Create a scatter plot of metric vs training time
    style_rc = bundles.icml2024()
    plt.rcParams.update(style_rc)
    plt.rcParams.update(figsizes.icml2024_half(nrows=1, ncols=2, height_to_width_ratio=0.8))
    # figsize_rc = figsizes.icml2022_full()["figure.figsize"]

    fig, axs = plt.subplots(ncols=2, sharey=False)

    # axs = axs[::-1]

    axs[0].grid(axis='y', alpha=0.3, zorder=0.1)
    axs[1].grid(axis='y', alpha=0.3, zorder=0.1)
    # Uncomment the following line to use different markers for datasets
    # ax = sns.scatterplot(data=merged_df, x='train_time', y='metric_mean', hue=method_name, style=dataset_name, s=200, palette=palette)
    sns.scatterplot(ax=axs[0], data=merged_df_1, x='total_time', y='metric_1_mean', hue=method_name, style=dataset_name, palette=palette)
    sns.scatterplot(ax=axs[1], data=merged_df_2, x='total_time', y='metric_2_mean', hue=method_name, style=dataset_name, palette=palette)

    # Add y error bars
    # for _, row in merged_df_1.iterrows():
    #     axs[0].errorbar(row['train_time'], row['metric_1_mean'], yerr=row['metric_1_std'], 
    #                fmt='none', color='black', capsize=3, capthick=1, zorder=0.2)
    # for _, row in merged_df_2.iterrows():
    #     axs[1].errorbar(row['train_time'], row['metric_2_mean'], yerr=row['metric_2_std'], 
    #                fmt='none', color='black', capsize=3, capthick=1, zorder=0.2)

    # Set x axis as log scale for better visibility
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    # ax.tick_params(axis='both', which='major', labelsize=12)

    # Customize the plot
    metric_1_name = os.path.splitext(os.path.basename(metric_1_csv))[0]
    fixed_name_1 = fix_metric_name(metric_1_name)
    metric_2_name = os.path.splitext(os.path.basename(metric_2_csv))[0]
    fixed_name_2 = fix_metric_name(metric_2_name)

    if train_n_sample:
        axs[0].set_xlabel(r'total time to first generated dataset (seconds)')
        axs[1].set_xlabel(r'total time to first generated dataset (seconds)')
    else:
        axs[0].set_xlabel(r'training time (seconds)')
        axs[1].set_xlabel(r'training time (seconds)')
    
    # Remove y label from individual subplots and set a common y label for both
    axs[0].set_ylabel(fixed_name_1)
    axs[1].set_ylabel(fixed_name_2)
    # axs[0].set_ylabel(f'C2ST')
    # fig.supylabel('C2ST')

    # Set the title of each subplot as the classifier it uses
    # axs[0].set_title(fixed_name_1)
    # axs[1].set_title(fixed_name_2)

    # Move legend to above plots and split into two columns
    axs[0].legend_.remove()
    axs[1].legend_.remove()
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05))

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


def plot_one_metric_vs_time(metric_1_csv, train_time_csv, sampling_time_csv, methods, out_path, figsize=(10, 6), palette="Set2"):
    """
    Plot metric vs training time for different methods.
    
    Parameters:
    -----------
    metric_1_csv : str
        Path to the CSV file containing the first metric data
    train_time_csv : str
        Path to the CSV file containing training time data
    sampling_time_csv : str
        Path to the CSV file containing sampling time data
    methods : list
        List of methods to include in the plot
    figsize : tuple
        Figure size (width, height)
    palette : str
        Color palette for the plot
    """
    # Read the metric data
    metric_1_df = pd.read_csv(metric_1_csv)
    # Get only specified methods
    metric_1_df = metric_1_df[metric_1_df['method'].isin(methods)]
    # Rename mean to metric_mean, std to metric_std for clarity
    metric_1_df = metric_1_df.rename(columns={'mean': 'metric_1_mean', 'std': 'metric_1_std'})

    # Read the training time data
    train_time_df = pd.read_csv(train_time_csv)
    # Get only specified methods
    train_time_df = train_time_df[train_time_df['method'].isin(methods)]
    # Rename mean to train_time
    train_time_df = train_time_df.rename(columns={'mean': 'train_time'})

    sample_time_df = pd.read_csv(sampling_time_csv)
    sample_time_df = sample_time_df[sample_time_df['method'].isin(methods)]
    sample_time_df = sample_time_df.rename(columns={'mean': 'sampling_time'})

    # print(train_time_df)
    # print(sample_time_df)

    # Fix missing training times
    for method in methods:
        if method not in train_time_df['method'].unique():
            for dataset in train_time_df['dataset'].unique():
                new_row = {
                    'method': [method],
                    'dataset': [dataset],
                    'train_time': [0.0],
                    'std': [0.0],
                }
                df_new_row = pd.DataFrame(new_row)
                train_time_df = pd.concat([train_time_df, df_new_row])

    # print(train_time_df)

    merged_times = pd.merge(train_time_df, sample_time_df, on=['method', 'dataset'])
    merged_times['total_time'] = merged_times['train_time'] + merged_times['sampling_time']

    print(merged_times)

    # Add together training and sampling times

    merged_df_1 = pd.merge(metric_1_df, merged_times, on=['method', 'dataset'])

    print(merged_df_1)

    # Use fixed method names for better readability
    merged_df_1['method'] = merged_df_1['method'].apply(fix_method_name)
    methods = [fix_method_name(m) for m in methods]

    tabpc_methodname = fix_method_name('TabPC')

    FIXED_SORTED_METHOD_NAMES = []
    for sorted_method_name in SORTED_METHODS:
        FIXED_SORTED_METHOD_NAMES.append(fix_method_name(sorted_method_name))

    if tabpc_methodname in methods:
        merged_without_tabpc_df_1 = merged_df_1[merged_df_1['method'] != tabpc_methodname]
        # Sort remaining methods by method for consistent coloring
        merged_without_tabpc_df_1 = merged_without_tabpc_df_1.sort_values(by='method')
        merged_without_tabpc_df_1 = sort_rows(merged_without_tabpc_df_1, 'method', FIXED_SORTED_METHOD_NAMES)
        tabpc_df_1 = merged_df_1[merged_df_1['method'] == tabpc_methodname]
        # Place TabPC at the start for better visibility
        merged_df_1 = pd.concat([tabpc_df_1, merged_without_tabpc_df_1])
    else:
        # Sort by method for consistent coloring
        merged_df_1 = merged_df_1.sort_values(by='method')

    # Capitalize column names for better readability in the plot
    method_name = r"$\bf{Method}$"
    dataset_name = r"$\bf{Dataset}$"
    merged_df_1 = merged_df_1.rename(columns={'method': method_name, 'dataset': dataset_name})
    
    # Create a scatter plot of metric vs training time
    style_rc = bundles.icml2024()
    plt.rcParams.update(style_rc)
    plt.rcParams.update(figsizes.icml2024_half(nrows=1, ncols=2, height_to_width_ratio=0.8))
    # figsize_rc = figsizes.icml2022_full()["figure.figsize"]

    fig, axs = plt.subplots(sharey=False)

    # axs = axs[::-1]

    axs.grid(axis='y', alpha=0.3, zorder=0.1)
    # Uncomment the following line to use different markers for datasets
    # ax = sns.scatterplot(data=merged_df, x='train_time', y='metric_mean', hue=method_name, style=dataset_name, s=200, palette=palette)
    sns.scatterplot(ax=axs, data=merged_df_1, x='total_time', y='metric_1_mean', hue=method_name, style=dataset_name, palette=palette)

    # Add y error bars
    # for _, row in merged_df_1.iterrows():
    #     axs[0].errorbar(row['train_time'], row['metric_1_mean'], yerr=row['metric_1_std'], 
    #                fmt='none', color='black', capsize=3, capthick=1, zorder=0.2)
    # for _, row in merged_df_2.iterrows():
    #     axs[1].errorbar(row['train_time'], row['metric_2_mean'], yerr=row['metric_2_std'], 
    #                fmt='none', color='black', capsize=3, capthick=1, zorder=0.2)

    # Set x axis as log scale for better visibility
    axs.set_xscale('log')
    # ax.tick_params(axis='both', which='major', labelsize=12)

    # Customize the plot
    metric_1_name = os.path.splitext(os.path.basename(metric_1_csv))[0]
    fixed_name_1 = fix_metric_name(metric_1_name)

    axs.set_xlabel(r'total time to first generated dataset (seconds)')
    
    # Remove y label from individual subplots and set a common y label for both
    axs.set_ylabel(fixed_name_1)
    # axs[0].set_ylabel(f'C2ST')
    # fig.supylabel('C2ST')

    # Set the title of each subplot as the classifier it uses
    # axs[0].set_title(fixed_name_1)
    # axs[1].set_title(fixed_name_2)

    # Move legend to above plots and split into two columns
    axs.legend_.remove()
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05))

    # Add only the hue in the legend, not the style
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Remove style entries (datasets) from legend
    by_label = {k: v for k, v in by_label.items() if k in merged_df_1[method_name].unique()}
    # Put legend in the center above the plot
    fig.legend(by_label.values(), by_label.keys(), loc='center left', ncol=1,
                bbox_to_anchor=(0.15, 0.6)
                )
    
    # Save the plot instead of showing it
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved as: {out_path}")
    plt.close()


if __name__ == "__main__":
    metric_csv_1 = "article_material/our_results/xgb_detection.csv"
    metric_csv_2 = "artifacts/AP_BR_Results_OC_investigation/Summary/alpha_precision_oc/alpha_precision_oc.csv"
    train_time_csv = "article_material/our_results/training_time.csv"
    sampling_time_csv = "article_material/our_results/sampling_time.csv"
    methods = [
        "TabPC",
        "TabDiff",
        "TabSyn",
        # "TabPFN"
    ]

    plot_metric_vs_time(
        metric_csv_1,
        metric_csv_2,
        train_time_csv,
        sampling_time_csv,
        methods,
        "C2ST_XGB_AP_train_n_sample_original_implementation.pdf",
        train_n_sample=False
    )

    # plot_one_metric_vs_time(
    #     metric_csv_1,
    #     train_time_csv,
    #     sampling_time_csv,
    #     methods,
    #     "C2ST_XGB_train_n_sample.pdf"
    # )

    # plot_one_metric_vs_time(
    #     "artifacts/AP_BR_Results_OC_investigation/Summary/alpha_precision_oc/alpha_precision_oc.csv",
    #     train_time_csv,
    #     sampling_time_csv,
    #     methods,
    #     "C2ST_XGB_AP_updated.pdf"
    # )

    # plot_one_metric_vs_time(
    #     "artifacts/AP_BR_Results_OC_investigation/Summary/alpha_precision/alpha_precision.csv",
    #     train_time_csv,
    #     sampling_time_csv,
    #     methods,
    #     "C2ST_XGB_AP_naive.pdf"
    # )