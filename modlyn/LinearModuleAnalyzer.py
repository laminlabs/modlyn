import warnings

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import norm

warnings.filterwarnings("ignore")


class LinearModelAnalyzer:
    """Comprehensive analysis for linear models with uncertainty estimation."""

    def __init__(self, model, adata, datamodule=None):
        self.model = model
        self.adata = adata
        self.datamodule = datamodule
        self.weights = model.linear.weight.detach().cpu().numpy()
        self.bias = (
            model.linear.bias.detach().cpu().numpy()
            if model.linear.bias is not None
            else None
        )

        # Get class and gene names
        if "y" in adata.obs.columns:
            if hasattr(adata.obs["y"], "cat"):
                self.class_names = adata.obs["y"].cat.categories.tolist()
            else:
                self.class_names = sorted(adata.obs["y"].unique())
        else:
            self.class_names = [f"Class_{i}" for i in range(model.linear.out_features)]

        # Get proper gene names from adata.var
        self.gene_names = self._extract_gene_names(adata)
        self.n_classes, self.n_genes = self.weights.shape

    def _extract_gene_names(self, adata):
        """Extract proper gene names from adata.var."""
        print("Extracting gene names from adata.var...")
        print(f"adata.var columns: {adata.var.columns.tolist()}")
        print(f"adata.var shape: {adata.var.shape}")

        if adata.var.shape[1] > 0:
            print("adata.var preview:")
            print(adata.var.head())

            # Check for gene name columns in order of preference
            gene_columns = [
                "feature_name",
                "gene_name",
                "symbol",
                "gene_symbol",
                "gene_id",
            ]

            for col in gene_columns:
                if col in adata.var.columns:
                    gene_names = adata.var[col].astype(str).tolist()
                    print(f"Using '{col}': {gene_names[:5]}")
                    return gene_names

            # If no standard column found, use first column if it looks like gene names
            if len(adata.var.columns) > 0:
                first_col = adata.var.columns[0]
                sample_vals = adata.var[first_col][:5].astype(str).tolist()
                if any(
                    any(c.isalpha() for c in val) and len(val) > 1
                    for val in sample_vals
                ):
                    gene_names = adata.var[first_col].astype(str).tolist()
                    print(f"Using first column '{first_col}': {gene_names[:5]}")
                    return gene_names

        # Check if var_names contain gene symbols (not just numbers)
        var_names = adata.var_names.astype(str).tolist()
        if any(any(c.isalpha() for c in name) for name in var_names[:10]):
            print(f"Using adata.var_names: {var_names[:5]}")
            return var_names

        # Fallback to gene indices
        gene_names = [f"Gene_{i:05d}" for i in range(adata.n_vars)]
        print(f"Using gene indices: {gene_names[:5]}")
        return gene_names

    def _simple_weight_uncertainty(self):
        """Simple uncertainty based on weight magnitude and class separation."""
        # Estimate uncertainty based on weight statistics
        weight_stds = np.abs(self.weights) * 0.1  # Simple heuristic

        # Higher uncertainty for smaller weights
        weight_stds += 0.05 / (np.abs(self.weights) + 0.01)

        return self.weights, weight_stds

    def create_volcano_plot(self, class1_idx=0, class2_idx=1, uncertainty=None):
        """Create volcano plot comparing two classes.

        X-axis: log fold change (weight difference)
        Y-axis: -log10(p-value) or significance metric.
        """
        class1_name = self.class_names[class1_idx]
        class2_name = self.class_names[class2_idx]

        # Calculate log fold change (weight difference)
        log_fc = self.weights[class1_idx] - self.weights[class2_idx]

        # Calculate p-values or significance metric
        if uncertainty is not None:
            _, weight_stds = uncertainty
            # T-test like statistic
            se_diff = np.sqrt(
                weight_stds[class1_idx] ** 2 + weight_stds[class2_idx] ** 2
            )
            t_stats = np.abs(log_fc) / (se_diff + 1e-8)
            p_values = 2 * (1 - norm.cdf(t_stats))  # Two-tailed test
            neg_log_p = -np.log10(p_values + 1e-10)
        else:
            # Use weight magnitude as significance proxy
            neg_log_p = np.log10(np.abs(log_fc) + 0.01)

        # Create volcano plot
        plt.figure(figsize=(12, 8))

        # Color points by significance and effect size
        colors = [
            "gray" if (abs(fc) < 0.5 or nlp < 2) else "red" if fc > 0 else "blue"
            for fc, nlp in zip(log_fc, neg_log_p)
        ]

        plt.scatter(log_fc, neg_log_p, c=colors, alpha=0.6, s=20)

        # Add significance thresholds
        plt.axhline(y=2, color="black", linestyle="--", alpha=0.5, label="p=0.01")
        plt.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
        plt.axvline(x=-0.5, color="black", linestyle="--", alpha=0.5)

        plt.xlabel(f"Weight Difference ({class1_name} - {class2_name})")
        plt.ylabel("-log10(p-value)" if uncertainty else "log10(|Weight Difference|)")
        plt.title(f"Volcano Plot: {class1_name} vs {class2_name}")
        plt.grid(True, alpha=0.3)

        # Annotate top genes
        top_genes_idx = np.argsort(neg_log_p)[-10:]
        for idx in top_genes_idx:
            if abs(log_fc[idx]) > 0.3:  # Only annotate if effect size is meaningful
                plt.annotate(
                    self.gene_names[idx],
                    (log_fc[idx], neg_log_p[idx]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                )

        plt.tight_layout()
        plt.savefig(
            f"volcano_plot_{class1_name}_vs_{class2_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        return log_fc, neg_log_p

    def create_heatmap_analysis(self, top_k=30):
        """Create comprehensive heatmap analysis."""
        # 1. Gene importance heatmap
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_gene_indices = np.argsort(gene_importance)[-top_k:][::-1]

        # Select subset of classes for readability
        n_classes_show = min(20, len(self.class_names))
        class_subset = range(
            0, len(self.class_names), max(1, len(self.class_names) // n_classes_show)
        )[:n_classes_show]

        weights_subset = self.weights[np.ix_(class_subset, top_gene_indices)]

        plt.figure(figsize=(15, 10))

        # Create heatmap
        sns.heatmap(
            weights_subset,
            xticklabels=[self.gene_names[i] for i in top_gene_indices],
            yticklabels=[self.class_names[i] for i in class_subset],
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Gene Weight"},
        )

        plt.title(f"Heatmap: Top {top_k} Genes vs Classes")
        plt.xlabel("Genes")
        plt.ylabel("Classes")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("heatmap_genes_vs_classes.png", dpi=300, bbox_inches="tight")
        plt.show()

        return top_gene_indices

    def create_weight_adata_for_scanpy(self, top_k=20):
        """Create a pseudo-AnnData object.

        Here, 'expression' values are linear model weights
        This allows us to use scanpy.pl.dotplot with model weights.
        """
        # Extract weights and info
        weights = self.weights  # Shape: (n_classes, n_genes)

        # Get top genes across all classes
        gene_importance = np.mean(np.abs(weights), axis=0)
        top_gene_indices = np.argsort(gene_importance)[-top_k:][::-1]
        top_gene_names = [self.gene_names[i] for i in top_gene_indices]

        print(f"Top {top_k} genes: {top_gene_names[:5]}...")

        # Create expression matrix where each "cell" represents a class
        # and each "gene" has expression = weight for that class
        # Shape: (n_classes, n_top_genes)
        expression_matrix = weights[:, top_gene_indices]

        # Normalize weights to make them look like expression values
        # Shift to make all positive (scanpy expects positive expression)
        min_weight = np.min(expression_matrix)
        if min_weight < 0:
            expression_matrix = expression_matrix - min_weight + 0.1

        # Scale to reasonable expression range (0-10)
        max_weight = np.max(expression_matrix)
        if max_weight > 0:
            expression_matrix = (expression_matrix / max_weight) * 10

        # Create obs (one row per class)
        obs_df = pd.DataFrame(
            {
                "class": self.class_names,
                "group": self.class_names,  # This will be our groupby variable
            }
        )
        obs_df.index = [f"class_{i}" for i in range(len(self.class_names))]

        # Create var (one row per top gene)
        var_df = pd.DataFrame(
            {"gene_name": top_gene_names, "original_index": top_gene_indices}
        )
        var_df.index = top_gene_names

        # Create the pseudo-AnnData object
        weight_adata = anndata.AnnData(
            X=expression_matrix,  # Shape: (n_classes, n_top_genes)
            obs=obs_df,
            var=var_df,
        )

        print(f"Created weight AnnData: {weight_adata}")

        return weight_adata, top_gene_names

    def create_scanpy_dotplot(self, top_k=20, **kwargs):
        """Use real scanpy.pl.dotplot with model weights."""
        print(" Creating scanpy dotplot with model weights...")

        # Create the weight-based AnnData
        weight_adata, top_gene_names = self.create_weight_adata_for_scanpy(top_k)

        # Use scanpy dotplot
        # Here, each "class" is treated as a group, and "expression" is the weight
        fig = sc.pl.dotplot(
            weight_adata,
            var_names=top_gene_names,  # Genes to show
            groupby="group",  # Group by class
            standard_scale="var",  # Standardize across genes
            colorbar_title="Standardized\nWeight",
            size_title="|Weight|",
            figsize=(
                max(12, len(top_gene_names) * 0.4),
                max(6, len(weight_adata.obs) * 0.3),
            ),
            return_fig=True,
            **kwargs,
        )

        return fig, weight_adata

    def create_custom_scanpy_dotplot(self, top_k=20, use_abs_size=True):
        """Create custom scanpy-style dotplot."""
        weights = self.weights

        # Get top genes
        gene_importance = np.mean(np.abs(weights), axis=0)
        top_gene_indices = np.argsort(gene_importance)[-top_k:][::-1]
        top_gene_names = [self.gene_names[i] for i in top_gene_indices]

        # Create the plot data
        plot_data = []
        for class_idx, class_name in enumerate(self.class_names):
            for gene_idx in top_gene_indices:
                weight = weights[class_idx, gene_idx]
                gene_name = self.gene_names[gene_idx]

                plot_data.append(
                    {
                        "Class": class_name,
                        "Gene": gene_name,
                        "Weight": weight,
                        "AbsWeight": abs(weight),
                        "Sign": "Positive" if weight > 0 else "Negative",
                    }
                )

        df = pd.DataFrame(plot_data)

        # Pivot for plotting
        weight_pivot = df.pivot(index="Class", columns="Gene", values="Weight")
        abs_weight_pivot = df.pivot(index="Class", columns="Gene", values="AbsWeight")

        # Limit classes for visualization
        max_classes = min(20, len(self.class_names))
        weight_pivot = weight_pivot.iloc[:max_classes]
        abs_weight_pivot = abs_weight_pivot.iloc[:max_classes]

        fig, ax = plt.subplots(
            figsize=(max(12, len(top_gene_names) * 0.4), max(8, max_classes * 0.3))
        )

        # Create dots
        for i, class_name in enumerate(weight_pivot.index):
            for j, gene_name in enumerate(weight_pivot.columns):
                weight = weight_pivot.loc[class_name, gene_name]
                abs_weight = abs_weight_pivot.loc[class_name, gene_name]

                # Size based on absolute weight
                if use_abs_size:
                    size = (abs_weight / abs_weight_pivot.max().max()) * 300 + 20
                else:
                    size = 100

                # Color based on weight direction and magnitude
                ax.scatter(
                    j,
                    i,
                    s=size,
                    c=weight,
                    cmap="RdBu_r",
                    vmin=-abs(weight_pivot.values).max(),
                    vmax=abs(weight_pivot.values).max(),
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.5,
                )

        # Customize
        ax.set_xticks(range(len(weight_pivot.columns)))
        ax.set_xticklabels(weight_pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(weight_pivot.index)))
        ax.set_yticklabels(weight_pivot.index)

        ax.set_xlabel("Genes")
        ax.set_ylabel("Classes/Perturbations")
        ax.set_title(f"Model Weights: Top {top_k} Genes per Class")

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=plt.Normalize(
                vmin=-abs(weight_pivot.values).max(),
                vmax=abs(weight_pivot.values).max(),
            ),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Gene Weight")

        # Add size legend
        sizes = [50, 150, 300]
        labels = ["Low |Weight|", "Medium |Weight|", "High |Weight|"]

        for size, label in zip(sizes, labels):
            ax.scatter(
                [],
                [],
                s=size,
                c="gray",
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
                label=label,
            )

        ax.legend(title="Weight Magnitude", bbox_to_anchor=(1.15, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig("scanpy_style_weights_dotplot.png", dpi=300, bbox_inches="tight")
        plt.show()

        return fig, df

    def analyze_confounders_vs_biology(self):
        """Analyze confounders (plate effects) vs biological variables."""
        print("\n" + "=" * 60)
        print("CONFOUNDER vs BIOLOGICAL ANALYSIS")
        print("=" * 60)

        # Identify potential confounders and biological variables
        obs_columns = self.adata.obs.columns.tolist()

        # Confounders (technical variables)
        confounders = [
            col
            for col in obs_columns
            if any(
                x in col.lower()
                for x in ["plate", "batch", "barcode", "sublibrary", "sample"]
            )
        ]

        # Biological variables
        biological = [
            col
            for col in obs_columns
            if any(
                x in col.lower()
                for x in ["drug", "cell_line", "cell_type", "tissue", "treatment"]
            )
        ]

        print(f"Potential confounders: {confounders}")
        print(f"Biological variables: {biological}")

        # Analyze variance explained by each
        variance_analysis = {}

        for var_type, variables in [
            ("Confounders", confounders),
            ("Biological", biological),
        ]:
            print(f"\n{var_type}:")
            for var in variables:
                if var in self.adata.obs.columns:
                    unique_vals = self.adata.obs[var].nunique()
                    print(f"  {var}: {unique_vals} unique values")
                    variance_analysis[var] = {
                        "type": var_type,
                        "unique_values": unique_vals,
                    }

        return variance_analysis

    def _print_summary_stats(self):
        """Print summary statistics."""
        print(f"Model has {self.n_classes} classes and {self.n_genes} genes")
        print(f"Average weight magnitude: {np.mean(np.abs(self.weights)):.4f}")
        print(
            f"Most variable class: {self.class_names[np.argmax(np.var(self.weights, axis=1))]}"
        )

        # Show top genes with real names
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_gene_indices = np.argsort(gene_importance)[-10:][::-1]

        print("\nTop 10 most important genes (with real names):")
        for i, idx in enumerate(top_gene_indices):
            print(f"  {i+1:2d}. {self.gene_names[idx]}: {gene_importance[idx]:.4f}")


# Main analysis functions
def quick_analysis_with_scanpy_dotplot(model, adata, datamodule=None):
    """Quick analysis with real scanpy dotplot."""
    analyzer = LinearModelAnalyzer(model, adata, datamodule)

    print("\n volcano plot...")
    uncertainty = analyzer._simple_weight_uncertainty()
    analyzer.create_volcano_plot(0, 1, (uncertainty[0], uncertainty[1]))

    print("\n scanpy dotplot...")
    fig1, weight_adata = analyzer.create_scanpy_dotplot(top_k=15)

    print("\n custom scanpy-style dotplot...")
    fig2, df = analyzer.create_custom_scanpy_dotplot(top_k=15)

    print("\n heatmap...")
    analyzer.create_heatmap_analysis(top_k=20)

    print("\n variance analysis...")
    analyzer.analyze_confounders_vs_biology()

    print("\n summary statistics...")
    analyzer._print_summary_stats()

    return analyzer, weight_adata, df


def full_analysis(model, adata, datamodule=None):
    """Run complete analysis."""
    analyzer = LinearModelAnalyzer(model, adata, datamodule)

    # Run all analyses
    uncertainty = analyzer._simple_weight_uncertainty()

    # Create all plots
    analyzer.create_volcano_plot(0, 1, uncertainty)
    fig1, weight_adata = analyzer.create_scanpy_dotplot(top_k=20)
    fig2, df = analyzer.create_custom_scanpy_dotplot(top_k=20)
    top_genes = analyzer.create_heatmap_analysis(top_k=30)
    variance_analysis = analyzer.analyze_confounders_vs_biology()
    analyzer._print_summary_stats()

    return {
        "analyzer": analyzer,
        "weight_adata": weight_adata,
        "df": df,
        "top_genes": top_genes,
        "variance_analysis": variance_analysis,
    }
