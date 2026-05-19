import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.linear_model import HuberRegressor


def plot_metric_per_likelihood(
        ll_results_dir,
        split,
        out_path,
        dataset_names,
        palette="flare",
        use_title=True,
        regression_line=False,
        remove_legend=False,
        ) -> None:
    """
    Scatter plot of metric values vs likelihood for different methods across all datasets.
    
    Parameters:
    -----------
    ll_results_dir: str
        Directory containing the likelihoods.csv and metric CSV files.
    split: str
        Which split to plot ('validation', 'test').
    out_path: str
        Path to save the generated plot.
    dataset_names: list of str
        List of dataset names to include in the plot.
    palette: str or list of colors, optional
        Color palette for the points (default: "flare").
    use_title: bool, optional
        Whether to use dataset names as titles for each subplot (default: True).
    regression_line: bool, optional
        Whether to fit and plot a regression line (default: False).
    remove_legend: bool, optional
        Whether to remove the legend from the plot (default: False).
    """

    num_plots = len(dataset_names)

    fig, axs = plt.subplots(ncols=num_plots)

    likelihoods_csv_path = os.path.join(ll_results_dir, "likelihoods.csv")
    metric_csv_path = os.path.join(ll_results_dir, "C2ST.csv")
    try:
        likelihoods_df = pd.read_csv(likelihoods_csv_path)
        metric_df = pd.read_csv(metric_csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    for i, dataset_name in enumerate(dataset_names):
        # Filter for the current dataset and required values
        likelihoods_df_dataset = likelihoods_df[likelihoods_df['dataset'] == dataset_name]
        metric_df_dataset = metric_df[metric_df['dataset'] == dataset_name]

        # Merge the likelihoods and metric dataframes on method and dataset
        merged_df = pd.merge(metric_df_dataset, likelihoods_df_dataset, left_on=['method', 'dataset'], right_on=['method', 'dataset'])

        # Sort by model for consistent coloring
        # Models are named TabPC_{num_units}_{batch_size}_{lr}, so we can sort by num_units, then batch_size, then lr
        # Extract num_units, batch_size, and lr from the method name and add them as separate columns for sorting
        merged_df['num_units'] = merged_df['method'].apply(lambda x: int(x.split('_')[1]))
        merged_df['batch_size'] = merged_df['method'].apply(lambda x: int(x.split('_')[2]))
        merged_df['lr'] = merged_df['method'].apply(lambda x: float(x.split('_')[3]))
        merged_df = merged_df.sort_values(by=['num_units', 'batch_size', 'lr'], ascending=[False, False, False])

        # Plot split LL vs metric
        ax = axs[i] if num_plots > 1 else axs
        sns.scatterplot(ax=ax, data=merged_df[merged_df['split'] == split], x='bpd', y='XGB-C2ST', size='batch_size', hue='num_units', style='lr', palette=palette)
        
        if regression_line:
            X = merged_df[merged_df['split'] == split]['bpd'].values.reshape(-1, 1)
            y = merged_df[merged_df['split'] == split]['XGB-C2ST'].values
            huber = HuberRegressor().fit(X, y)
            # Plot regression line
            x_range = np.linspace(X.min(), X.max(), 100)
            y_range = huber.predict(x_range.reshape(-1, 1))
            ax.plot(x_range, y_range, color='grey', linestyle='--', linewidth=2)

            # Add R^2 score to the plot
            r2 = huber.score(X, y)
            if dataset_name == 'adult':
                ax.text(0.1, 0.25, f'$R^2$: {r2:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
            else:
                ax.text(0.4, 0.95, f'$R^2$: {r2:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        ax.set_xlabel(f'{split.title()} BPD')
        ax.set_ylabel('C2ST (XGB)')
        
        if use_title:
            ax.set_title(f'{dataset_name}')

        # Remove legend if requested
        if remove_legend:
            ax.legend_.remove()

    plt.savefig(out_path, dpi=300)
    print(f"Plot saved as: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metric values vs likelihood for different methods across all datasets.")
    parser.add_argument("--ll-results-dir", type=str, required=True, help="Directory containing the likelihoods.csv and metric CSV files.")
    parser.add_argument("--split", type=str, choices=['validation', 'test'], default='validation', help="Which split to plot (default: 'validation').")
    parser.add_argument("--out-path", type=str, required=True, help="Path to save the generated plot.")
    parser.add_argument("--dataset-names", nargs='+', required=True, help="List of dataset names to include in the plot.")
    parser.add_argument("--palette", type=str, default="flare", help="Color palette for the points (default: 'flare').")
    parser.add_argument("--use-title", action='store_true', help="Whether to use dataset names as titles for each subplot (default: False).")
    parser.add_argument("--regression-line", action='store_true', help="Whether to fit and plot a regression line (default: False).")
    parser.add_argument("--remove-legend", action='store_true', help="Whether to remove the legend from the plot (default: False).")

    args = parser.parse_args()

    plot_metric_per_likelihood(
        ll_results_dir=args.ll_results_dir,
        split=args.split,
        out_path=args.out_path,
        dataset_names=args.dataset_names,
        palette=args.palette,
        use_title=args.use_title,
        regression_line=args.regression_line,
        remove_legend=args.remove_legend,
    )
