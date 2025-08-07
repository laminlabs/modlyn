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

    def plot_weight_correlation(self, figsize=(10, 6)):
        """Plot weight correlation between methods.

        Creates a correlation plot showing how well different methods' weights
        correlate across all features for each class/cell line.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if len(self.dataframes) < 2:
            raise ValueError("Need at least 2 methods to compute correlations")

        method_names = [df.attrs["method_name"] for df in self.dataframes]

        # Find common features and samples
        common_genes = set.intersection(*[set(df.columns) for df in self.dataframes])
        common_cells = set.intersection(*[set(df.index) for df in self.dataframes])
        common_genes, common_cells = sorted(common_genes), sorted(common_cells)

        # Align dataframes
        dfs_aligned = [df.loc[common_cells, common_genes] for df in self.dataframes]

        # Compute correlations for each cell line and method pair
        correlations = []
        for cell_line in common_cells:
            for method1, method2 in combinations(range(len(method_names)), 2):
                weights1 = dfs_aligned[method1].loc[cell_line].values
                weights2 = dfs_aligned[method2].loc[cell_line].values

                # Calculate Pearson correlation
                corr = np.corrcoef(weights1, weights2)[0, 1]

                correlations.append(
                    {
                        "cell_line": cell_line,
                        "method_pair": f"{method_names[method1]} vs {method_names[method2]}",
                        "correlation": corr,
                    }
                )

        corr_df = pd.DataFrame(correlations)

        # Create the plot with 3 subplots to include the scatter plot
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))

        # 1. Box plot of correlations by method pair (Left)
        if len(corr_df["method_pair"].unique()) == 1:
            # Single method pair - use histogram
            axes[0].hist(corr_df["correlation"], bins=20, alpha=0.7, edgecolor="black")
            axes[0].set_xlabel("Correlation")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Weight Correlation Distribution")
        else:
            # Multiple method pairs - use box plot
            sns.boxplot(data=corr_df, x="method_pair", y="correlation", ax=axes[0])
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
            axes[0].set_title("Weight Correlation by Method Pair")

        axes[0].grid(True, alpha=0.3)

        # 2. Weight Scatter Plot (Middle) - This matches your image!
        if len(method_names) >= 2:
            # Use first cell line for scatter plot demonstration
            first_cell_line = common_cells[0]
            weights1 = dfs_aligned[0].loc[first_cell_line].values
            weights2 = dfs_aligned[1].loc[first_cell_line].values

            # Create scatter plot
            axes[1].scatter(weights1, weights2, alpha=0.6, s=20)

            # Add correlation line (red dashed)
            z = np.polyfit(weights1, weights2, 1)
            p = np.poly1d(z)
            axes[1].plot(weights1, p(weights1), "r--", alpha=0.8, linewidth=2)

            # Calculate correlation for this cell line
            cell_corr = np.corrcoef(weights1, weights2)[0, 1]

            axes[1].set_xlabel(f"{method_names[0]} Weights")
            axes[1].set_ylabel(f"{method_names[1]} Weights")
            axes[1].set_title(
                f"Weight Comparison: {first_cell_line}\nCorrelation: {cell_corr:.3f}"
            )
            axes[1].grid(True, alpha=0.3)

        # 3. Correlation by cell line (Right)
        if len(corr_df["method_pair"].unique()) == 1:
            corr_by_line = (
                corr_df.groupby("cell_line")["correlation"]
                .mean()
                .sort_values(ascending=True)
            )
            axes[2].barh(range(len(corr_by_line)), corr_by_line.values)
            axes[2].set_yticks(range(len(corr_by_line)))
            axes[2].set_yticklabels(corr_by_line.index)
            axes[2].set_xlabel("Correlation")
            axes[2].set_title("Correlation by Cell Line")
            axes[2].grid(True, alpha=0.3)

            # Add correlation value annotations
            for i, v in enumerate(corr_by_line.values):
                axes[2].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        else:
            # Multiple method pairs - show correlation matrix heatmap
            pivot_corr = corr_df.pivot_table(
                index="cell_line", columns="method_pair", values="correlation"
            )
            sns.heatmap(
                pivot_corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[2]
            )
            axes[2].set_title("Correlation Matrix by Cell Line")

        plt.tight_layout()

        # Print summary statistics
        overall_corr = corr_df["correlation"].mean()
        print(f"Overall mean correlation: {overall_corr:.4f}")
        print(
            f"Correlation range: [{corr_df['correlation'].min():.4f}, {corr_df['correlation'].max():.4f}]"
        )
        print(f"Methods are {overall_corr*100:.1f}% correlated on average!")

        return fig, corr_df

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
