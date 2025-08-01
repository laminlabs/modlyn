"""COMPLETE LINEAR MODEL ANALYSIS - SINGLE FILE VERSION.

All-in-one script for linear model analysis with publication-ready figures
and biological insights.
"""

import warnings
from datetime import datetime

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm, pearsonr
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Set publication style
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


class CompleteAnalyzer:
    """All-in-one analyzer for linear models."""

    def __init__(self, model, adata):
        self.model = model
        self.adata = adata
        self.weights = model.linear.weight.detach().cpu().numpy()
        self.class_names = self._get_class_names()
        self.gene_names = self._get_gene_names()
        self.results = {}

    def _get_class_names(self):
        if "y" in self.adata.obs.columns:
            if hasattr(self.adata.obs["y"], "cat"):
                return self.adata.obs["y"].cat.categories.tolist()
            return sorted(self.adata.obs["y"].unique())
        return [f"Class_{i}" for i in range(self.weights.shape[0])]

    def _get_gene_names(self):
        for col in ["feature_name", "gene_name", "symbol"]:
            if col in self.adata.var.columns:
                return self.adata.var[col].astype(str).tolist()
        return self.adata.var_names.astype(str).tolist()

    def figure_1_model_overview(self):
        """Figure 1: Model performance and weight distribution."""
        print("📊 Creating Figure 1: Model Overview...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # A: Weight distribution
        weights_flat = self.weights.flatten()
        axes[0, 0].hist(weights_flat, bins=50, alpha=0.7, color="#2E86AB")
        axes[0, 0].set_xlabel("Weight value")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("A. Weight Distribution")

        # B: Class separability
        class_var = np.var(self.weights, axis=1)
        axes[0, 1].bar(range(min(20, len(class_var))), class_var[:20], color="#A23B72")
        axes[0, 1].set_xlabel("Class index")
        axes[0, 1].set_ylabel("Weight variance")
        axes[0, 1].set_title("B. Class Separability (Top 20)")

        # C: Gene importance
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_20_idx = np.argsort(gene_importance)[-20:]
        axes[1, 0].barh(range(20), gene_importance[top_20_idx], color="#F18F01")
        axes[1, 0].set_yticks(range(20))
        axes[1, 0].set_yticklabels([self.gene_names[i] for i in top_20_idx], fontsize=8)
        axes[1, 0].set_xlabel("Mean |weight|")
        axes[1, 0].set_title("C. Top 20 Important Genes")

        # D: Weight correlation (subset for visualization)
        n_show = min(20, len(self.class_names))
        weight_subset = self.weights[:n_show, :]
        weight_corr = np.corrcoef(weight_subset)
        im = axes[1, 1].imshow(weight_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, 1].set_title(f"D. Class Correlation (Top {n_show})")
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

        plt.tight_layout()
        plt.savefig("Figure1_ModelOverview.png")
        plt.savefig("Figure1_ModelOverview.pdf")
        plt.show()

        return gene_importance

    def create_weight_adata_for_scanpy(self, top_k=20):
        """Create a pseudo-AnnData object where 'expression' values are linear model weights."""
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

    def figure_2_scanpy_dotplot(self, top_k=25, **kwargs):
        """Figure 2: Professional scanpy dotplot using real scanpy.pl.dotplot."""
        print("🔴 Creating Figure 2: Scanpy Dotplot with model weights...")

        # Create the weight-based AnnData
        weight_adata, top_gene_names = self.create_weight_adata_for_scanpy(top_k)

        # Use scanpy dotplot
        # Here, each "class" is treated as a group, and "expression" is the weight
        try:
            # Set scanpy settings for better display
            sc.settings.set_figure_params(dpi=300, facecolor="white")

            # Create the dotplot - scanpy handles the figure creation
            sc.pl.dotplot(
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
                show=False,  # Don't show immediately
                **kwargs,
            )

            # Get the current figure and save it
            fig = plt.gcf()
            fig.suptitle("Model Weights: Scanpy Dotplot", fontsize=16, y=0.98)

            plt.tight_layout()
            plt.savefig("Figure2_ScanpyDotplot.png", dpi=300, bbox_inches="tight")
            plt.savefig("Figure2_ScanpyDotplot.pdf", bbox_inches="tight")
            plt.show()

            print("✅ Scanpy dotplot created successfully!")

        except Exception as e:
            print(f"⚠️  Scanpy dotplot failed: {e}")
            print("Creating custom dotplot instead...")

            # Fallback to custom dotplot
            gene_importance = np.mean(np.abs(self.weights), axis=0)
            top_genes_idx = np.argsort(gene_importance)[-top_k:][::-1]
            top_genes = [self.gene_names[i] for i in top_genes_idx]

            n_classes_show = min(30, len(self.class_names))
            weights_subset = self.weights[:n_classes_show, top_genes_idx]

            self._create_custom_dotplot(
                weights_subset, top_genes, self.class_names[:n_classes_show]
            )
            fig = plt.gcf()

        return fig, weight_adata

    def _create_custom_dotplot(self, weights_subset, gene_names, class_names):
        """Create custom dotplot if scanpy fails."""
        fig, ax = plt.subplots(
            figsize=(max(12, len(gene_names) * 0.4), max(8, len(class_names) * 0.3))
        )

        # Normalize for visualization
        weights_norm = (weights_subset - weights_subset.mean()) / weights_subset.std()

        for i, _class_name in enumerate(class_names):
            for j, _gene_name in enumerate(gene_names):
                weight = weights_subset[i, j]
                norm_weight = weights_norm[i, j]

                # Size based on absolute weight
                size = (abs(weight) / abs(weights_subset).max()) * 300 + 20

                ax.scatter(
                    j,
                    i,
                    s=size,
                    c=norm_weight,
                    cmap="RdBu_r",
                    vmin=-2,
                    vmax=2,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.5,
                )

        ax.set_xticks(range(len(gene_names)))
        ax.set_xticklabels(gene_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Genes")
        ax.set_ylabel("Perturbations")
        ax.set_title("Model Weights: Custom Dotplot")

        plt.colorbar(
            plt.cm.ScalarMappable(cmap="RdBu_r"), ax=ax, label="Normalized Weight"
        )
        plt.tight_layout()
        plt.savefig("Figure2_CustomDotplot.png")
        plt.show()

    def figure_3_volcano_plots(self):
        """Figure 3: Volcano plots for key comparisons."""
        print("🌋 Creating Figure 3: Volcano Plots...")

        # Select interesting class pairs
        n_plots = min(3, len(self.class_names) - 1)
        class_pairs = [(0, i + 1) for i in range(n_plots)]

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        for i, (c1, c2) in enumerate(class_pairs):
            # Calculate log fold change
            log_fc = self.weights[c1] - self.weights[c2]
            significance = np.log10(np.abs(log_fc) + 0.01)

            # Color points
            colors = [
                "#FF6B6B"
                if fc > 0.5 and sig > 1
                else "#4ECDC4"
                if fc < -0.5 and sig > 1
                else "#95A5A6"
                for fc, sig in zip(log_fc, significance)
            ]

            axes[i].scatter(log_fc, significance, c=colors, alpha=0.7, s=20)

            # Add thresholds
            axes[i].axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
            axes[i].axvline(x=-0.5, color="black", linestyle="--", alpha=0.5)
            axes[i].axhline(y=1, color="black", linestyle="--", alpha=0.5)

            # Annotate top genes
            top_idx = np.argsort(significance)[-5:]
            for idx in top_idx:
                if abs(log_fc[idx]) > 0.3:
                    axes[i].annotate(
                        self.gene_names[idx],
                        (log_fc[idx], significance[idx]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

            axes[i].set_xlabel(
                f"Weight difference ({self.class_names[c1]} - {self.class_names[c2]})"
            )
            axes[i].set_ylabel("log10(|Effect size|)")
            axes[i].set_title(f"{self.class_names[c1]} vs {self.class_names[c2]}")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("Figure3_VolcanoPlots.png")
        plt.savefig("Figure3_VolcanoPlots.pdf")
        plt.show()

    def explore_perturbation_mechanisms(self):
        """Analyze perturbation mechanisms and similarity."""
        print("💊 Analyzing perturbation mechanisms...")

        # Calculate perturbation similarity based on gene weight patterns
        perturbation_similarity = np.corrcoef(self.weights)

        # Create similarity heatmap (subset for visualization)
        n_show = min(30, len(self.class_names))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            perturbation_similarity[:n_show, :n_show],
            cmap="RdBu_r",
            center=0,
            square=True,
            xticklabels=self.class_names[:n_show],
            yticklabels=self.class_names[:n_show],
            cbar_kws={"label": "Gene Signature Correlation"},
        )
        plt.title("Perturbation Similarity Matrix\n(Based on Gene Weight Patterns)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("Perturbation_Similarity_Matrix.png")
        plt.show()

        # Find most similar perturbation pairs
        similarity_pairs = []
        for i in range(len(perturbation_similarity)):
            for j in range(i + 1, len(perturbation_similarity)):
                if perturbation_similarity[i, j] > 0.7:  # High similarity threshold
                    similarity_pairs.append((i, j, perturbation_similarity[i, j]))

        print(
            f"🔍 Found {len(similarity_pairs)} highly similar perturbation pairs (correlation > 0.7)"
        )
        for i, j, corr in sorted(similarity_pairs, key=lambda x: x[2], reverse=True)[
            :5
        ]:
            print(f"  {self.class_names[i]} ↔ {self.class_names[j]}: {corr:.3f}")

        self.results["perturbation_similarity"] = perturbation_similarity
        return perturbation_similarity

    def explore_gene_networks(self, top_k=50):
        """Analyze gene co-expression networks."""
        print("🧬 Analyzing gene networks...")

        # Get top genes
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_genes_idx = np.argsort(gene_importance)[-top_k:][::-1]

        # Calculate gene-gene correlations
        gene_corr = np.corrcoef(self.weights[:, top_genes_idx].T)

        # Find gene modules using hierarchical clustering
        distance_matrix = 1 - np.abs(gene_corr)
        linkage_matrix = linkage(squareform(distance_matrix), method="ward")
        clusters = fcluster(linkage_matrix, t=0.7, criterion="distance")

        # Analyze modules
        unique_clusters = np.unique(clusters)
        gene_modules = {}

        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_genes = [
                self.gene_names[top_genes_idx[i]] for i in np.where(mask)[0]
            ]
            if len(cluster_genes) >= 3:
                gene_modules[f"Module_{cluster_id}"] = cluster_genes

        print(f"🔍 Found {len(gene_modules)} gene modules:")
        for module, genes in list(gene_modules.items())[:5]:
            print(f"  {module}: {genes[:3]}... ({len(genes)} genes)")

        # Plot gene correlation network
        plt.figure(figsize=(12, 10))
        plt.imshow(gene_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(label="Gene Correlation")
        plt.title(f"Gene Co-regulation Network (Top {top_k} Genes)")
        plt.xlabel("Genes")
        plt.ylabel("Genes")
        plt.savefig("Gene_Network_Analysis.png")
        plt.show()

        self.results["gene_modules"] = gene_modules
        return gene_modules

    def analyze_confounders(self):
        """Identify confounding factors."""
        print("🔍 Analyzing confounding factors...")

        obs_cols = self.adata.obs.columns

        # Identify technical vs biological variables
        technical_vars = [
            col
            for col in obs_cols
            if any(
                x in col.lower()
                for x in ["plate", "batch", "barcode", "sample", "well"]
            )
        ]

        biological_vars = [
            col
            for col in obs_cols
            if any(
                x in col.lower()
                for x in ["drug", "treatment", "cell_line", "tissue", "condition"]
            )
        ]

        print(f"📊 Technical variables found: {technical_vars}")
        print(f"🧬 Biological variables found: {biological_vars}")

        # Analyze distribution of technical variables
        if technical_vars:
            n_vars = len(technical_vars)
            fig, axes = plt.subplots(1, min(3, n_vars), figsize=(5 * min(3, n_vars), 4))
            if min(3, n_vars) == 1:
                axes = [axes]

            for i, var in enumerate(technical_vars[:3]):
                if var in self.adata.obs.columns:
                    counts = self.adata.obs[var].value_counts()
                    axes[i].bar(range(len(counts)), counts.values)
                    axes[i].set_title(f"{var}\n({len(counts)} categories)")
                    axes[i].set_xlabel("Category")
                    axes[i].set_ylabel("Count")

            plt.tight_layout()
            plt.savefig("Confounders_Analysis.png")
            plt.show()

        self.results["confounders"] = {
            "technical": technical_vars,
            "biological": biological_vars,
        }

        return technical_vars, biological_vars

    def generate_summary_report(self):
        """Generate comprehensive summary."""
        print("📋 Generating summary report...")

        n_classes, n_genes = self.weights.shape

        report = f"""
LINEAR MODEL ANALYSIS SUMMARY
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
---------------
• Observations: {self.adata.n_obs:,}
• Genes: {n_genes:,}
• Classes: {n_classes}
• Weight range: [{self.weights.min():.3f}, {self.weights.max():.3f}]

KEY FINDINGS
-----------
"""

        # Add specific findings based on results
        if "drug_similarity" in self.results:
            max_sim = np.max(
                self.results["drug_similarity"][self.results["drug_similarity"] < 0.99]
            )
            report += f"• Maximum drug similarity: {max_sim:.3f}\n"

        if "gene_modules" in self.results:
            n_modules = len(self.results["gene_modules"])
            report += f"• Gene modules identified: {n_modules}\n"

        if "confounders" in self.results:
            n_tech = len(self.results["confounders"]["technical"])
            n_bio = len(self.results["confounders"]["biological"])
            report += f"• Technical variables: {n_tech}\n"
            report += f"• Biological variables: {n_bio}\n"

        # Top genes
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_genes_idx = np.argsort(gene_importance)[-10:][::-1]

        report += "\nTOP 10 PREDICTIVE GENES\n"
        report += "-----------------------\n"
        for i, idx in enumerate(top_genes_idx):
            report += f"{i+1:2d}. {self.gene_names[idx]}: {gene_importance[idx]:.4f}\n"

        report += """
RECOMMENDATIONS
--------------
1. Validate gene signatures with independent data
2. Perform pathway enrichment on gene modules
3. Test drug combinations based on similarity
4. Investigate cell line-specific responses
5. Control for identified confounding factors

FILES GENERATED
--------------
• Figure1_ModelOverview.png/pdf
• Figure2_ScanpyDotplot.png (or CustomDotplot.png)
• Figure3_VolcanoPlots.png/pdf
• Drug_Similarity_Matrix.png
• Gene_Network_Analysis.png
• Confounders_Analysis.png (if applicable)
"""

        # Save report
        with open("Analysis_Summary_Report.txt", "w") as f:
            f.write(report)

        print("✅ Summary report saved as 'Analysis_Summary_Report.txt'")
        return report


def run_complete_analysis(model, adata, save_prefix="analysis"):
    """Run complete analysis pipeline.

    Parameters:
    -----------
    model : torch model with linear layer
    adata : AnnData object
    save_prefix : str, prefix for saved files

    Returns:
    --------
    analyzer : CompleteAnalyzer object with all results
    """
    print("🚀 STARTING COMPLETE LINEAR MODEL ANALYSIS")
    print("=" * 60)
    print(f"⏰ Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dataset: {adata.n_obs:,} observations × {adata.n_vars:,} genes")
    print(f"🧮 Model: {model.linear.weight.shape[0]} classes")
    print("=" * 60)

    # Initialize analyzer
    analyzer = CompleteAnalyzer(model, adata)

    try:
        # Create all figures and analyses
        print("\n📸 CREATING PUBLICATION FIGURES")
        print("-" * 40)

        analyzer.figure_1_model_overview()
        weight_adata, top_genes = analyzer.figure_2_scanpy_dotplot()
        analyzer.figure_3_volcano_plots()

        print("\n🔬 BIOLOGICAL EXPLORATION")
        print("-" * 40)

        analyzer.explore_perturbation_mechanisms()
        analyzer.explore_gene_networks()
        technical_vars, biological_vars = analyzer.analyze_confounders()

        print("\n📋 GENERATING SUMMARY")
        print("-" * 40)

        analyzer.generate_summary_report()

        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Partial results may still be available in analyzer.results")

    return analyzer


def quick_analysis(model, adata):
    """Quick 5-minute analysis."""
    print("⚡ QUICK ANALYSIS")
    print("=" * 30)

    weights = model.linear.weight.detach().cpu().numpy()

    print(f"📊 Dataset: {adata.n_obs:,} obs × {adata.n_vars:,} genes")
    print(f"🧮 Model: {weights.shape[0]} classes")
    print(f"📈 Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Get gene names
    gene_names = adata.var_names.astype(str).tolist()
    if "feature_name" in adata.var.columns:
        gene_names = adata.var["feature_name"].astype(str).tolist()

    # Top genes
    gene_importance = np.mean(np.abs(weights), axis=0)
    top_genes_idx = np.argsort(gene_importance)[-10:][::-1]

    print("\n🔥 Top 10 predictive genes:")
    for i, idx in enumerate(top_genes_idx):
        print(f"  {i+1:2d}. {gene_names[idx]}: {gene_importance[idx]:.4f}")

    # Quick plots
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(weights.flatten(), bins=50, alpha=0.7, color="skyblue")
    plt.title("Weight Distribution")
    plt.xlabel("Weight value")

    plt.subplot(1, 3, 2)
    plt.bar(range(10), gene_importance[top_genes_idx], color="orange")
    plt.title("Top 10 Gene Importance")
    plt.ylabel("Mean |weight|")
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    class_var = np.var(weights, axis=1)
    plt.bar(range(min(20, len(class_var))), class_var[:20], color="lightcoral")
    plt.title("Class Separability (Top 20)")
    plt.ylabel("Weight variance")

    plt.tight_layout()
    plt.savefig("Quick_Analysis.png")
    plt.show()

    print("✅ Quick analysis complete!")
    return gene_importance, top_genes_idx


# Example usage:
"""
# Quick exploration (5 minutes)
gene_importance, top_genes = quick_analysis(model, adata)

# Full analysis (20-30 minutes)
analyzer = run_complete_analysis(model, adata)

# Access results
summary = analyzer.results
"""
