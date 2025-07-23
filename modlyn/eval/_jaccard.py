from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class CompareScores:
    """Class for comparing feature importance methods using Jaccard index."""

    def __init__(self, dataframes, n_top_values=None):
        """Initialize with dataframes and n_top values to compare.

        Parameters:
        -----------
        dataframes : list of pd.DataFrame
            List of dataframes with method results. Each should have df.attrs["method_name"]
        n_top_values : list of int
            List of top-N values to compare across
        """
        if n_top_values is None:
            n_top_values = [25, 50, 100, 200]
        self.dataframes = dataframes
        self.n_top_values = n_top_values
        self.results_df = None

    def compute_jaccard_comparison(self):
        """Compute Jaccard comparison for n methods across different n_top values."""
        method_names = [df.attrs["method_name"] for df in self.dataframes]

        # Find common features and samples
        common_genes = set.intersection(*[set(df.columns) for df in self.dataframes])
        common_cells = set.intersection(*[set(df.index) for df in self.dataframes])
        common_genes, common_cells = sorted(common_genes), sorted(common_cells)
        n_genes = len(common_genes)

        # Align dataframes
        dfs_aligned = [df.loc[common_cells, common_genes] for df in self.dataframes]

        results = []

        for n_top in self.n_top_values:
            # Compute method pairs
            for cell_line in common_cells:
                scores = {
                    name: df.loc[cell_line]
                    for df, name in zip(dfs_aligned, method_names)
                }
                top_features = {
                    name: set(scores[name].abs().nlargest(n_top).index)
                    for name in method_names
                }

                for method1, method2 in combinations(method_names, 2):
                    overlap = len(top_features[method1] & top_features[method2])
                    union = len(top_features[method1] | top_features[method2])
                    jaccard = overlap / union if union > 0 else 0

                    results.append(
                        {
                            "cell_line": cell_line,
                            "n_top": n_top,
                            "method_pair": f"{method1}↔{method2}",
                            "jaccard": jaccard,
                        }
                    )

        # Add random baselines after all method pairs
        for n_top in self.n_top_values:
            if n_top >= n_genes:
                random_jaccard = 1.0
            else:
                random_jaccard = (2 * n_top) / (2 * n_genes - n_top)

            results.append(
                {
                    "n_top": n_top,
                    "method_pair": "Random baseline",
                    "jaccard": random_jaccard,
                }
            )

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def plot_jaccard_comparison(self):
        """Plot Jaccard indices as grouped bar plot."""
        if self.results_df is None:
            raise ValueError("Must run compute_jaccard_comparison() first")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Get mean Jaccard per method pair and n_top
        bar_data = (
            self.results_df.groupby(["n_top", "method_pair"])["jaccard"]
            .mean()
            .unstack()
        )
        n_top_values = sorted(self.results_df["n_top"].unique())

        # Set up grouped bar plot
        n_pairs = len(bar_data.columns)
        x = np.arange(len(n_top_values))
        width = 0.15

        # Use seaborn color palette
        colors = sns.color_palette("tab10", n_pairs)

        # Plot bars
        for i, method_pair in enumerate(bar_data.columns):
            values = [bar_data.loc[n_top, method_pair] for n_top in n_top_values]
            bars = ax.bar(
                x + i * width,
                values,
                width,
                color=colors[i],
                label=method_pair,
                alpha=0.8,
                edgecolor="black",
            )

            # Add value labels
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        # Formatting
        ax.set_xlabel("Number of Top Features (n_top)")
        ax.set_ylabel("Jaccard Index")
        ax.set_title("Jaccard Index vs Top-N Features")
        ax.set_xticks(x + width * (n_pairs - 1) / 2)
        ax.set_xticklabels(n_top_values)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, None)

        plt.tight_layout()

    def plot_heatmaps(self):
        """Plot heatmaps for all methods side by side."""
        # Sort columns and index alphabetically for each dataframe
        dfs_sorted = [df.sort_index().sort_index(axis=1) for df in self.dataframes]
        method_names = [df.attrs["method_name"] for df in self.dataframes]

        # Find global min and max across all dataframes
        vmin = min(df.min().min() for df in dfs_sorted)
        vmax = max(df.max().max() for df in dfs_sorted)

        # Create subplots
        n_methods = len(dfs_sorted)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))

        # Handle single method case
        if n_methods == 1:
            axes = [axes]

        # Plot heatmaps
        for i, (df, method_name) in enumerate(zip(dfs_sorted, method_names)):
            sns.heatmap(df, ax=axes[i], cmap="viridis", vmin=vmin, vmax=vmax, cbar=True)
            axes[i].set_title(method_name)

        plt.tight_layout()
        plt.close()
        return fig
