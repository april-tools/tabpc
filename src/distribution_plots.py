import itertools
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

import wandb
from src.util import is_categorical


def plot_hist_comparison(s_original, s_generated, title: str, path: str):
    # Drop NaNs
    s1 = s_original.dropna()
    s2 = s_generated.dropna()

    fig, ax = plt.subplots(figsize=(6, 4))

    total_unique_values = len(set(s1.unique()) | set(s2.unique()))
    if is_categorical(s_original) or total_unique_values < 100:
        categories = sorted(set(s1.unique()) | set(s2.unique()))
        real_counts = s1.value_counts(normalize=True).reindex(categories, fill_value=0)
        synth_counts = s2.value_counts(normalize=True).reindex(categories, fill_value=0)
        x = np.arange(len(categories))
        width = 0.6 if total_unique_values < 20 else 1
        ax.bar(
            x, real_counts, width=width, label="real", alpha=0.5, color="C0", zorder=2
        )
        ax.bar(
            x,
            synth_counts,
            width=(width * 0.6) if total_unique_values < 20 else width,
            label="synthetic",
            alpha=0.7,
            color="salmon" if total_unique_values < 20 else "C1",
            zorder=2,
        )
        if total_unique_values < 20:
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("Density")
    else:
        bins = int(len(s1) ** 0.5)
        data_min = min(s1.min(), s2.min())
        data_max = max(s1.max(), s2.max())
        bin_edges = np.linspace(data_min, data_max, bins + 1)
        ax.hist(
            s1,
            bins=bin_edges,
            alpha=0.8,
            label="real",
            density=True,
            color="C0",
            zorder=2,
        )
        ax.hist(
            s2,
            bins=bin_edges,
            alpha=0.5,
            label="synthetic",
            density=True,
            color="C1",
            zorder=2,
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.8, zorder=1)
    fig.tight_layout()
    fig.savefig(path)

    return fig


def count_matrix(
    s_discr: pd.Series,
    s_real: pd.Series,
    bin_edges: np.ndarray,
    categories,
    probability: bool = False,
):
    """
    Returns a matrix of counts for the discrete values in s_discr and s_real.
    """
    assert len(s_discr) == len(s_real), "s_discr and s_real must have the same length"
    counts_matrix = np.zeros((len(categories), len(bin_edges) - 1))
    for i, category in enumerate(categories):
        real_cat_data = s_real[s_discr == category].dropna()
        counts_matrix[i] = np.histogram(real_cat_data, bins=bin_edges, density=False)[0]
        if probability:
            counts_matrix[i] /= len(s_discr)
    return counts_matrix


def hist_diff_hybrid_plot(
    df_original: pd.DataFrame,
    df_generated: pd.DataFrame,
    ax,
    cat_column: str,
    num_column: str,
):
    """
    Creates a hybrid histogram difference plot for categorical vs numerical data.
    The categorical variable is always on the y-axis and the numerical variable on the x-axis.
    For each category, a histogram is built for both real and synthetic variables. The plot shows the difference between the synthetic and real histograms.
    """
    # Get all unique categories
    all_categories = sorted(
        set(df_original[cat_column].unique()) | set(df_generated[cat_column].unique())
    )

    # Determine bins based on real data across all categories
    real_num_data = df_original[num_column].dropna()
    n_bins = 50
    bin_edges = np.linspace(real_num_data.min(), real_num_data.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    real_prob_matrix = count_matrix(
        s_discr=df_original[cat_column],
        s_real=df_original[num_column],
        bin_edges=bin_edges,
        categories=all_categories,
        probability=True,
    )

    synth_prob_matrix = count_matrix(
        s_discr=df_generated[cat_column],
        s_real=df_generated[num_column],
        bin_edges=bin_edges,
        categories=all_categories,
        probability=True,
    )

    max_bar_heights = (
        np.maximum(real_prob_matrix.max(axis=1), synth_prob_matrix.max(axis=1)) * 1.05
    )
    hist_heights = np.maximum(max_bar_heights, max_bar_heights.max() / 10)
    hist_bottoms = hist_heights.cumsum(axis=0) - hist_heights

    for i, category in enumerate(all_categories):
        # Plot as line at the category level
        ax.bar(
            bin_centers,
            real_prob_matrix[i],
            bottom=hist_bottoms[i],
            width=bin_centers[1] - bin_centers[0],
            alpha=0.7,
            color="C0",
            label="Real" if i == 0 else "",
        )

        ax.bar(
            bin_centers,
            synth_prob_matrix[i],
            bottom=hist_bottoms[i],
            width=bin_centers[1] - bin_centers[0],
            alpha=0.5,
            color="C1",
            label="Synthetic" if i == 0 else "",
        )

    # Set proper labels
    ax.set_xlabel(num_column)
    ax.set_ylabel(cat_column)
    ax.set_yticks(hist_bottoms)
    ax.set_yticklabels(all_categories)

    # Add a horizontal line at each category level for reference
    for i in range(len(all_categories)):
        ax.axhline(y=hist_bottoms[i], color="gray", linestyle="--", alpha=0.3)

    ax.set_title("Histogram Difference: Synthetic vs Real")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def bivariate_plot(df_original: pd.DataFrame, df_generated: pd.DataFrame, path: str):
    """
    Plots a scatter plot of the two features for both original and generated data.
    """
    CAT_THRESHOLD = 20

    if df_original.shape[1] != 2 or df_generated.shape[1] != 2:
        raise ValueError(
            "Both dataframes must have exactly two columns for bivariate plot."
        )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    n_unique_values_col_1 = len(df_original.iloc[:, 0].unique())
    n_unique_values_col_2 = len(df_original.iloc[:, 1].unique())

    c1_is_cat = n_unique_values_col_1 <= CAT_THRESHOLD
    c2_is_cat = n_unique_values_col_2 <= CAT_THRESHOLD

    if (not c1_is_cat) and (not c2_is_cat):
        # Prepare data for random z-order
        real_x = np.array(df_original.iloc[:, 0])
        real_y = np.array(df_original.iloc[:, 1])

        synth_x = np.array(df_generated.iloc[:, 0])
        synth_y = np.array(df_generated.iloc[:, 1])

        # Combine all data
        all_x = np.concatenate([real_x, synth_x])
        all_y = np.concatenate([real_y, synth_y])
        all_colors = np.concatenate(
            [np.zeros(len(real_x)), np.ones(len(synth_x))]
        )  # 0 for real, 1 for synthetic

        # Create random order for plotting
        random_indices = np.random.permutation(len(all_x))

        # Apply random ordering to all arrays
        all_x = all_x[random_indices]
        all_y = all_y[random_indices]
        all_colors = all_colors[random_indices]

        # Plot with efficient scatter using color mapping
        scatter = ax.scatter(
            all_x,
            all_y,
            c=all_colors,
            alpha=0.3,
            marker=".",
            s=20,
            cmap="tab10",
            vmin=0,
            vmax=9,  # Use discrete colormap
        )

        # Add legend manually with dummy plots
        ax.scatter([], [], alpha=0.5, color="C0", marker=".", s=20, label="Real")
        ax.scatter([], [], alpha=0.5, color="C1", marker=".", s=20, label="Synthetic")

        ax.set_xlabel(df_original.columns[0])
        ax.set_ylabel(df_original.columns[1])
        ax.set_title("Bivariate Scatter: Real vs Synthetic")
        ax.legend()
        ax.grid(True, alpha=0.7, zorder=1)
        fig.tight_layout()
        fig.savefig(path)
    elif c1_is_cat and (not c2_is_cat) or (not c1_is_cat) and c2_is_cat:
        cat_column = df_original.columns[0] if c1_is_cat else df_original.columns[1]
        num_column = df_original.columns[1] if c1_is_cat else df_original.columns[0]

        hist_diff_hybrid_plot(df_original, df_generated, ax, cat_column, num_column)
        fig.tight_layout()
        fig.savefig(path)

    return fig


class DistributionPlot(ABC):
    @staticmethod
    @abstractmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        folder: str,
        format: str,
        wandb_run=None,
    ):
        raise NotImplementedError


class MarginalsPlot(DistributionPlot):
    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        folder: str,
        format: str,
        wandb_run=None,
    ):
        marginals_folder = f"{folder}/marginals"
        os.makedirs(marginals_folder, exist_ok=True)
        for column in real_data.columns:
            fig = plot_hist_comparison(
                s_original=real_data[column],
                s_generated=synthetic_data[column],
                title=f"Marginal distribution of {column}",
                path=f"{marginals_folder}/{column}.{format}",
            )
            if wandb_run is not None:
                wandb_run.log({f"marginal_{column}": wandb.Image(fig)})
            plt.close(fig)


class BivariatePlot(DistributionPlot):
    @staticmethod
    def __call__(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        folder: str,
        format: str,
        wandb_run=None,
    ):
        bivariate_folder = f"{folder}/bivariate"
        os.makedirs(bivariate_folder, exist_ok=True)

        lunique = {col: len(real_data[col].unique()) for col in real_data.columns}
        num_cols = [
            col
            for col in real_data.columns
            if lunique[col] > 50 and real_data[col].dtype in [np.float64, np.int64]
        ]
        rc = real_data[num_cols].corr().abs().round(2)

        all_pairs = list(itertools.combinations(real_data.columns, 2))

        PLOTS_LIMIT = 1000
        plots_count = 0

        for c1, c2 in tqdm.tqdm(all_pairs, desc="Generating bivariate plots"):
            tag = ""
            if c1 in num_cols or c2 in num_cols:
                if c1 in num_cols and c2 in num_cols:
                    tag = f"{rc[c1][c2]:.2f}"[-2:] + "_"
                else:
                    tag = "h_"

                fig = bivariate_plot(
                    df_original=real_data[[c1, c2]],
                    df_generated=synthetic_data[[c1, c2]],
                    path=f"{folder}/bivariate/{tag}{c1}_vs_{c2}.{format}",
                )
                plots_count += 1
                if wandb_run is not None:
                    wandb_run.log({f"{c1}_vs_{c2}": wandb.Image(fig)})
                plt.close(fig)
                if plots_count >= PLOTS_LIMIT:
                    break
