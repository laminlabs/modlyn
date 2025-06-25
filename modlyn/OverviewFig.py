"""Overview figure.

Creates a comprehensive Nature-style figure showing the MODLYN framework
for analyzing massive single-cell datasets with linear models.
"""

import warnings
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

warnings.filterwarnings("ignore")

# Set Nature journal style
plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)


class MODLYNFigure:
    """Create comprehensive MODLYN framework figure."""

    def __init__(self):
        self.colors = {
            "data": "#2E86AB",  # Blue for data
            "model": "#A23B72",  # Purple for model
            "analysis": "#F18F01",  # Orange for analysis
            "biology": "#C73E1D",  # Red for biology
            "accent": "#87CEEB",  # Light blue accent
            "text": "#2F4F4F",  # Dark gray for text
        }

    def create_comprehensive_figure(self, model=None, adata=None):
        """Create the main MODLYN framework figure."""
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(
            4,
            4,
            figure=fig,
            hspace=0.3,
            wspace=0.3,
            height_ratios=[1, 1.2, 1, 1],
            width_ratios=[1, 1, 1, 1],
        )

        # A: Framework Overview (spans top)
        ax_framework = fig.add_subplot(gs[0, :])
        self._draw_framework_overview(ax_framework)

        # B: Data Scale Visualization
        ax_scale = fig.add_subplot(gs[1, 0])
        self._draw_scale_comparison(ax_scale)

        # C: Linear Model Architecture
        ax_model = fig.add_subplot(gs[1, 1])
        self._draw_model_architecture(ax_model)

        # D: Weight Analysis Workflow
        ax_workflow = fig.add_subplot(gs[1, 2:])
        self._draw_analysis_workflow(ax_workflow)

        # E: Example Results (if data provided)
        if model is not None and adata is not None:
            ax_results1 = fig.add_subplot(gs[2, :2])
            ax_results2 = fig.add_subplot(gs[2, 2:])
            self._draw_example_results(ax_results1, ax_results2, model, adata)
        else:
            ax_results = fig.add_subplot(gs[2, :])
            self._draw_mock_results(ax_results)

        # F: Applications and Impact
        ax_applications = fig.add_subplot(gs[3, :])
        self._draw_applications(ax_applications)

        # Add panel labels
        self._add_panel_labels(fig, gs)

        # Add main title
        fig.suptitle(
            "MODLYN: Linear Models for Massive Single-Cell Analysis",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.savefig("Figure_MODLYN_Framework.png", dpi=300, bbox_inches="tight")
        plt.savefig("Figure_MODLYN_Framework.pdf", bbox_inches="tight")
        plt.show()

        return fig

    def _draw_framework_overview(self, ax):
        """Draw the overall framework flowchart."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis("off")

        # Draw workflow boxes
        boxes = [
            {
                "xy": (0.5, 1),
                "width": 1.5,
                "height": 0.8,
                "label": "Tahoe-100M\nDataset\n(200M cells)",
                "color": self.colors["data"],
            },
            {
                "xy": (2.5, 1),
                "width": 1.5,
                "height": 0.8,
                "label": "MODLYN\nData Loader",
                "color": self.colors["model"],
            },
            {
                "xy": (4.5, 1),
                "width": 1.5,
                "height": 0.8,
                "label": "Linear\nClassifier",
                "color": self.colors["model"],
            },
            {
                "xy": (6.5, 1),
                "width": 1.5,
                "height": 0.8,
                "label": "Weight\nAnalysis",
                "color": self.colors["analysis"],
            },
            {
                "xy": (8.5, 1),
                "width": 1.2,
                "height": 0.8,
                "label": "Biological\nInsights",
                "color": self.colors["biology"],
            },
        ]

        for box in boxes:
            rect = FancyBboxPatch(
                box["xy"],
                box["width"],
                box["height"],
                boxstyle="round,pad=0.05",
                facecolor=box["color"],
                alpha=0.7,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            ax.text(
                box["xy"][0] + box["width"] / 2,
                box["xy"][1] + box["height"] / 2,
                box["label"],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

        # Draw arrows
        arrow_positions = [(2, 1.4), (4, 1.4), (6, 1.4), (8, 1.4)]
        for x, y in arrow_positions:
            ax.annotate(
                "",
                xy=(x + 0.4, y),
                xytext=(x, y),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "black"},
            )

        # Add subtitle
        ax.text(
            5,
            0.2,
            '"Linear models in the age of AI"',
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": self.colors["accent"],
                "alpha": 0.5,
            },
        )

    def _draw_scale_comparison(self, ax):
        """Draw scale comparison chart."""
        ax.set_title("B. Dataset Scale", fontweight="bold", pad=10)

        # Data for comparison
        datasets = [
            "10X Genomics\n(~10K)",
            "Mouse Atlas\n(~1M)",
            "Human Atlas\n(~10M)",
            "Tahoe-100M\n(~200M)",
        ]
        cell_counts = [1e4, 1e6, 1e7, 2e8]
        colors_scale = [
            self.colors["accent"],
            self.colors["data"],
            self.colors["model"],
            self.colors["biology"],
        ]

        bars = ax.bar(
            datasets, cell_counts, color=colors_scale, alpha=0.8, edgecolor="black"
        )
        ax.set_yscale("log")
        ax.set_ylabel("Number of Cells")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, cell_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 1.1,
                f"{count:.0e}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        # Highlight Tahoe-100M
        bars[-1].set_edgecolor(self.colors["biology"])
        bars[-1].set_linewidth(3)

    def _draw_model_architecture(self, ax):
        """Draw linear model architecture."""
        ax.set_title("C. Linear Model Architecture", fontweight="bold", pad=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Input layer (genes)
        gene_y_positions = np.linspace(1, 9, 5)
        for i, y in enumerate(gene_y_positions):
            circle = plt.Circle((2, y), 0.3, color=self.colors["data"], alpha=0.7)
            ax.add_patch(circle)
            if i == 2:  # Middle gene
                ax.text(
                    2,
                    y,
                    "Gene\n19K",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                )

        # Add "..." to indicate more genes
        ax.text(2, 5.5, "...", ha="center", va="center", fontsize=12, fontweight="bold")

        # Weight matrix
        ax.add_patch(
            plt.Rectangle(
                (4, 3),
                2,
                4,
                facecolor=self.colors["model"],
                alpha=0.7,
                edgecolor="black",
            )
        )
        ax.text(
            5,
            5,
            "Weight\nMatrix\n19K×50",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

        # Output layer (classes)
        class_y_positions = np.linspace(2, 8, 4)
        for i, y in enumerate(class_y_positions):
            circle = plt.Circle((8, y), 0.3, color=self.colors["analysis"], alpha=0.7)
            ax.add_patch(circle)
            if i == 1:  # Second class
                ax.text(
                    8,
                    y,
                    "Class\n50",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                )

        # Add "..." for more classes
        ax.text(8, 5.5, "...", ha="center", va="center", fontsize=12, fontweight="bold")

        # Draw connections
        for gene_y in gene_y_positions:
            for class_y in class_y_positions:
                ax.plot([2.3, 4], [gene_y, 5], "k-", alpha=0.3, linewidth=0.5)
                ax.plot([6, 7.7], [5, class_y], "k-", alpha=0.3, linewidth=0.5)

        # Labels
        ax.text(
            2,
            0.5,
            "Gene Expression\n(Input)",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            8,
            0.5,
            "Perturbation\n(Output)",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    def _draw_analysis_workflow(self, ax):
        """Draw the analysis workflow."""
        ax.set_title("D. Weight Analysis Workflow", fontweight="bold", pad=10)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)
        ax.axis("off")

        # Analysis steps
        steps = [
            {
                "xy": (1, 4),
                "size": (1.8, 1),
                "label": "Extract\nWeights",
                "color": self.colors["model"],
            },
            {
                "xy": (4, 4),
                "size": (1.8, 1),
                "label": "Gene\nImportance",
                "color": self.colors["analysis"],
            },
            {
                "xy": (7, 4),
                "size": (1.8, 1),
                "label": "Perturbation\nSimilarity",
                "color": self.colors["analysis"],
            },
            {
                "xy": (10, 4),
                "size": (1.8, 1),
                "label": "Pathway\nEnrichment",
                "color": self.colors["biology"],
            },
            {
                "xy": (2.5, 1.5),
                "size": (1.8, 1),
                "label": "Volcano\nPlots",
                "color": self.colors["accent"],
            },
            {
                "xy": (5.5, 1.5),
                "size": (1.8, 1),
                "label": "Dotplots",
                "color": self.colors["accent"],
            },
            {
                "xy": (8.5, 1.5),
                "size": (1.8, 1),
                "label": "Heatmaps",
                "color": self.colors["accent"],
            },
        ]

        for step in steps:
            rect = FancyBboxPatch(
                step["xy"],
                step["size"][0],
                step["size"][1],
                boxstyle="round,pad=0.05",
                facecolor=step["color"],
                alpha=0.7,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            ax.text(
                step["xy"][0] + step["size"][0] / 2,
                step["xy"][1] + step["size"][1] / 2,
                step["label"],
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )

        # Draw connections
        arrow_positions = [(2.8, 4.5), (5.8, 4.5), (8.8, 4.5)]
        for x, y in arrow_positions:
            ax.annotate(
                "",
                xy=(x + 1, y),
                xytext=(x, y),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "black"},
            )

        # Vertical arrows to visualizations
        for x in [3.4, 6.4, 9.4]:
            ax.annotate(
                "",
                xy=(x, 2.5),
                xytext=(x, 3.8),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "gray"},
            )

    def _draw_example_results(self, ax1, ax2, model, adata):
        """Draw example results if model and data are provided."""
        weights = model.linear.weight.detach().cpu().numpy()

        # Left plot: Gene importance
        ax1.set_title("E. Example Results: Gene Importance", fontweight="bold", pad=10)
        gene_importance = np.mean(np.abs(weights), axis=0)
        top_20_idx = np.argsort(gene_importance)[-20:]

        # Get gene names
        gene_names = adata.var_names.astype(str).tolist()
        if "feature_name" in adata.var.columns:
            gene_names = adata.var["feature_name"].astype(str).tolist()

        y_pos = np.arange(20)
        ax1.barh(
            y_pos, gene_importance[top_20_idx], color=self.colors["analysis"], alpha=0.7
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([gene_names[i] for i in top_20_idx], fontsize=6)
        ax1.set_xlabel("Mean |Weight|")

        # Right plot: Perturbation similarity heatmap
        ax2.set_title("Perturbation Similarity Matrix", fontweight="bold", pad=10)
        n_show = min(20, weights.shape[0])
        similarity_matrix = np.corrcoef(weights[:n_show])

        im = ax2.imshow(
            similarity_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )
        ax2.set_xlabel("Perturbations")
        ax2.set_ylabel("Perturbations")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.6)
        cbar.set_label("Correlation", fontsize=7)

    def _draw_mock_results(self, ax):
        """Draw mock results when no data is provided."""
        ax.set_title("E. Example Results: Key Outputs", fontweight="bold", pad=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")

        # Mock result boxes
        results = [
            {
                "xy": (0.5, 2.5),
                "size": (2, 1),
                "label": "Gene\nImportance\nRanking",
                "color": self.colors["analysis"],
            },
            {
                "xy": (3, 2.5),
                "size": (2, 1),
                "label": "Perturbation\nSimilarity\nMatrix",
                "color": self.colors["analysis"],
            },
            {
                "xy": (5.5, 2.5),
                "size": (2, 1),
                "label": "Pathway\nEnrichment\nScores",
                "color": self.colors["biology"],
            },
            {
                "xy": (8, 2.5),
                "size": (1.5, 1),
                "label": "Cell Type\nSignatures",
                "color": self.colors["biology"],
            },
            {
                "xy": (1.5, 0.5),
                "size": (2, 1),
                "label": "Differential\nGene Analysis",
                "color": self.colors["accent"],
            },
            {
                "xy": (4, 0.5),
                "size": (2, 1),
                "label": "Drug MoA\nPrediction",
                "color": self.colors["accent"],
            },
            {
                "xy": (6.5, 0.5),
                "size": (2, 1),
                "label": "Biomarker\nDiscovery",
                "color": self.colors["accent"],
            },
        ]

        for result in results:
            rect = FancyBboxPatch(
                result["xy"],
                result["size"][0],
                result["size"][1],
                boxstyle="round,pad=0.05",
                facecolor=result["color"],
                alpha=0.6,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            ax.text(
                result["xy"][0] + result["size"][0] / 2,
                result["xy"][1] + result["size"][1] / 2,
                result["label"],
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )

    def _draw_applications(self, ax):
        """Draw applications and impact."""
        ax.set_title("F. Applications and Impact", fontweight="bold", pad=10)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3)
        ax.axis("off")

        # Application categories
        applications = [
            {
                "xy": (0.5, 1.5),
                "size": (2.5, 1),
                "label": "Drug Discovery\n• MoA prediction\n• Target identification",
                "color": self.colors["biology"],
            },
            {
                "xy": (3.5, 1.5),
                "size": (2.5, 1),
                "label": "Clinical Research\n• Biomarker discovery\n• Patient stratification",
                "color": self.colors["analysis"],
            },
            {
                "xy": (6.5, 1.5),
                "size": (2.5, 1),
                "label": "Basic Biology\n• Pathway analysis\n• Gene regulation",
                "color": self.colors["model"],
            },
            {
                "xy": (9.5, 1.5),
                "size": (2, 1),
                "label": "Method Development\n• Scalable analysis\n• Interpretability",
                "color": self.colors["data"],
            },
        ]

        for app in applications:
            rect = FancyBboxPatch(
                app["xy"],
                app["size"][0],
                app["size"][1],
                boxstyle="round,pad=0.05",
                facecolor=app["color"],
                alpha=0.6,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            ax.text(
                app["xy"][0] + app["size"][0] / 2,
                app["xy"][1] + app["size"][1] / 2,
                app["label"],
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )

        # Add key advantages
        ax.text(
            6,
            0.3,
            "Key Advantages: Scalable • Interpretable • Fast • Memory-efficient",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": self.colors["accent"],
                "alpha": 0.7,
            },
        )

    def _add_panel_labels(self, fig, gs):
        """Add panel labels A, B, C, etc."""
        labels = ["A", "B", "C", "D", "E", "F"]
        positions = [
            (0.02, 0.95),  # A
            (0.02, 0.75),  # B
            (0.27, 0.75),  # C
            (0.52, 0.75),  # D
            (0.02, 0.48),  # E
            (0.02, 0.25),  # F
        ]

        for label, pos in zip(labels, positions):
            fig.text(
                pos[0],
                pos[1],
                label,
                fontsize=12,
                fontweight="bold",
                bbox={
                    "boxstyle": "circle,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "black",
                },
            )


def create_modlyn_figure(model=None, adata=None):
    """Create the main MODLYN framework figure."""
    modlyn_fig = MODLYNFigure()
    fig = modlyn_fig.create_comprehensive_figure(model, adata)

    # Generate figure caption
    caption = """
Figure: MODLYN Framework for Scalable Single-Cell Analysis

(A) Overall workflow from Tahoe-100M dataset through MODLYN data loader to biological insights.
(B) Scale comparison showing Tahoe-100M's unprecedented size in the single-cell landscape.
(C) Linear model architecture: 19K genes as input, weight matrix (19K×50), 50 perturbation classes as output.
(D) Weight analysis workflow from extraction to biological interpretation and visualization.
(E) Example results showing gene importance ranking and perturbation similarity patterns.
(F) Applications spanning drug discovery, clinical research, basic biology, and method development.

The MODLYN framework enables rapid, interpretable analysis of massive single-cell datasets
through linear models, providing scalable solutions for understanding cellular responses
to perturbations at unprecedented scale.
"""

    print("✅ MODLYN figure created successfully!")
    print("\n📝 Figure Caption:")
    print(caption)

    # Save caption
    with open("MODLYN_Figure_Caption.txt", "w") as f:
        f.write(caption)

    return fig, caption


# Usage examples:
"""
# Create figure without data (uses mock results)
fig, caption = create_modlyn_figure()

# Create figure with actual model and data
fig, caption = create_modlyn_figure(model=your_model, adata=your_adata)
"""
