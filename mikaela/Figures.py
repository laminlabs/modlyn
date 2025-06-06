import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class NatureFigures:
    """Create publication-ready figures for linear model analysis"""
    
    def __init__(self, model, adata):
        self.model = model
        self.adata = adata
        self.weights = model.linear.weight.detach().cpu().numpy()
        self.class_names = self._get_class_names()
        self.gene_names = self._get_gene_names()
        
    def _get_class_names(self):
        if 'y' in self.adata.obs.columns:
            if hasattr(self.adata.obs['y'], 'cat'):
                return self.adata.obs['y'].cat.categories.tolist()
            return sorted(self.adata.obs['y'].unique())
        return [f"Class_{i}" for i in range(self.weights.shape[0])]
    
    def _get_gene_names(self):
        for col in ['feature_name', 'gene_name', 'symbol']:
            if col in self.adata.var.columns:
                return self.adata.var[col].astype(str).tolist()
        return self.adata.var_names.astype(str).tolist()
    
    def figure_1_model_overview(self):
        """Figure 1: Model performance and weight distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # A: Weight distribution
        weights_flat = self.weights.flatten()
        axes[0,0].hist(weights_flat, bins=50, alpha=0.7, color='#2E86AB')
        axes[0,0].set_xlabel('Weight value')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('A. Weight Distribution')
        
        # B: Class separability (weight variance)
        class_var = np.var(self.weights, axis=1)
        bars = axes[0,1].bar(range(len(class_var)), class_var, color='#A23B72')
        axes[0,1].set_xlabel('Class index')
        axes[0,1].set_ylabel('Weight variance')
        axes[0,1].set_title('B. Class Separability')
        
        # C: Gene importance
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_20_idx = np.argsort(gene_importance)[-20:]
        axes[1,0].barh(range(20), gene_importance[top_20_idx], color='#F18F01')
        axes[1,0].set_yticks(range(20))
        axes[1,0].set_yticklabels([self.gene_names[i] for i in top_20_idx])
        axes[1,0].set_xlabel('Mean |weight|')
        axes[1,0].set_title('C. Top 20 Important Genes')
        
        # D: Weight correlation between classes
        weight_corr = np.corrcoef(self.weights)
        im = axes[1,1].imshow(weight_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1,1].set_title('D. Class Weight Correlation')
        plt.colorbar(im, ax=axes[1,1], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('Figure1_ModelOverview.png')
        plt.savefig('Figure1_ModelOverview.pdf')
        plt.show()
        
        return gene_importance
    
    def figure_2_scanpy_dotplot(self, top_k=25):
        """Figure 2: Professional scanpy dotplot"""
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_genes_idx = np.argsort(gene_importance)[-top_k:][::-1]
        top_genes = [self.gene_names[i] for i in top_genes_idx]
        
        # Create weight-based AnnData
        weights_subset = self.weights[:, top_genes_idx]
        weights_scaled = (weights_subset - weights_subset.min()) + 0.1
        
        obs_df = pd.DataFrame({
            'perturbation': self.class_names,
            'group': self.class_names
        })
        obs_df.index = [f"pert_{i}" for i in range(len(self.class_names))]
        
        var_df = pd.DataFrame({'gene_name': top_genes})
        var_df.index = top_genes
        
        weight_adata = sc.AnnData(X=weights_scaled, obs=obs_df, var=var_df)
        
        # Create the dotplot
        fig = sc.pl.dotplot(
            weight_adata,
            var_names=top_genes,
            groupby='group',
            standard_scale='var',
            colorbar_title='Standardized\nWeight',
            size_title='|Weight|',
            figsize=(min(20, len(top_genes) * 0.6), min(15, len(self.class_names) * 0.4)),
            return_fig=True
        )
        
        plt.savefig('Figure2_ScanpyDotplot.png')
        plt.savefig('Figure2_ScanpyDotplot.pdf')
        plt.show()
        
        return weight_adata, top_genes
    
    def figure_3_volcano_plots(self, class_pairs=None):
        """Figure 3: Volcano plots for key comparisons"""
        if class_pairs is None:
            # Auto-select interesting pairs
            class_pairs = [(0, 1), (0, 2)] if len(self.class_names) > 2 else [(0, 1)]
        
        n_plots = len(class_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        for i, (c1, c2) in enumerate(class_pairs):
            # Calculate log fold change
            log_fc = self.weights[c1] - self.weights[c2]
            
            # Significance based on effect size
            significance = np.log10(np.abs(log_fc) + 0.01)
            
            # Color points
            colors = ['#FF6B6B' if fc > 0.5 and sig > 1 else 
                     '#4ECDC4' if fc < -0.5 and sig > 1 else '#95A5A6'
                     for fc, sig in zip(log_fc, significance)]
            
            axes[i].scatter(log_fc, significance, c=colors, alpha=0.7, s=20)
            
            # Add thresholds
            axes[i].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
            axes[i].axvline(x=-0.5, color='black', linestyle='--', alpha=0.5)
            axes[i].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            
            # Annotate top genes
            top_idx = np.argsort(significance)[-10:]
            for idx in top_idx:
                if abs(log_fc[idx]) > 0.3:
                    axes[i].annotate(self.gene_names[idx], 
                                   (log_fc[idx], significance[idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
            
            axes[i].set_xlabel(f'Weight difference ({self.class_names[c1]} - {self.class_names[c2]})')
            axes[i].set_ylabel('log10(|Effect size|)')
            axes[i].set_title(f'{self.class_names[c1]} vs {self.class_names[c2]}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Figure3_VolcanoPlots.png')
        plt.savefig('Figure3_VolcanoPlots.pdf')
        plt.show()
    
    def figure_4_pathway_analysis(self, top_k=20):
        """Figure 4: Pathway/functional analysis visualization"""
        gene_importance = np.mean(np.abs(self.weights), axis=0)
        top_genes_idx = np.argsort(gene_importance)[-top_k:][::-1]
        
        # Create a clustered heatmap
        weights_subset = self.weights[:, top_genes_idx]
        
        # Cluster both genes and classes
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Clustered heatmap
        gene_linkage = linkage(pdist(weights_subset.T), method='ward')
        class_linkage = linkage(pdist(weights_subset), method='ward')
        
        # Reorder based on clustering
        from scipy.cluster.hierarchy import leaves_list
        gene_order = leaves_list(gene_linkage)
        class_order = leaves_list(class_linkage)
        
        weights_ordered = weights_subset[np.ix_(class_order, gene_order)]
        
        im = axes[0].imshow(weights_ordered, cmap='RdBu_r', aspect='auto')
        axes[0].set_xticks(range(len(gene_order)))
        axes[0].set_xticklabels([self.gene_names[top_genes_idx[i]] for i in gene_order], 
                              rotation=45, ha='right')
        axes[0].set_yticks(range(len(class_order)))
        axes[0].set_yticklabels([self.class_names[i] for i in class_order])
        axes[0].set_title('Clustered Weight Heatmap')
        plt.colorbar(im, ax=axes[0])
        
        # Right: Gene module analysis
        # Simple correlation-based modules
        gene_corr = np.corrcoef(weights_subset.T)
        
        # Find highly correlated gene pairs
        corr_pairs = []
        for i in range(len(gene_corr)):
            for j in range(i+1, len(gene_corr)):
                if abs(gene_corr[i,j]) > 0.7:
                    corr_pairs.append((i, j, gene_corr[i,j]))
        
        # Plot correlation network (simplified)
        axes[1].imshow(gene_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Gene Co-regulation Network')
        axes[1].set_xticks(range(len(top_genes_idx)))
        axes[1].set_xticklabels([self.gene_names[i] for i in top_genes_idx], 
                              rotation=45, ha='right')
        axes[1].set_yticks(range(len(top_genes_idx)))
        axes[1].set_yticklabels([self.gene_names[i] for i in top_genes_idx])
        
        plt.tight_layout()
        plt.savefig('Figure4_PathwayAnalysis.png')
        plt.savefig('Figure4_PathwayAnalysis.pdf')
        plt.show()
        
        return corr_pairs

def create_publication_figures(model, adata):
    """Main function to create all publication figures"""    
    nf = NatureFigures(model, adata)
    
    # Create all figures
    gene_importance = nf.figure_1_model_overview()
    weight_adata, top_genes = nf.figure_2_scanpy_dotplot()
    # nf.figure_3_volcano_plots()
    corr_pairs = nf.figure_4_pathway_analysis()
    
    # Generate figure legends
    legends = {
        'Figure 1': """Model Overview and Performance Metrics. 
        (A) Distribution of linear model weights across all gene-class pairs. 
        (B) Class separability measured by weight variance per perturbation. 
        (C) Top 20 most important genes ranked by mean absolute weight. 
        (D) Correlation matrix of weight patterns between different perturbations.""",
        
        'Figure 2': f"""Gene Expression Signature Analysis. 
        Dot plot showing standardized weights for top {len(top_genes)} genes across all perturbations. 
        Dot size represents weight magnitude, color represents standardized weight value. 
        Genes and perturbations are ordered by hierarchical clustering.""",
        
        # 'Figure 3': """Differential Gene Analysis. 
        # Volcano plots comparing perturbation effects. X-axis shows weight differences, 
        # Y-axis shows effect size significance. Red dots: upregulated genes, 
        # Blue dots: downregulated genes, Gray dots: non-significant changes.""",
        
        'Figure 4': """Pathway Co-regulation Analysis. 
        (Left) Hierarchically clustered heatmap of gene weights. 
        (Right) Gene co-regulation network based on weight correlations across perturbations."""
    }
    
    print("\n📝 Figure Legends:")
    for fig, legend in legends.items():
        print(f"\n{fig}: {legend}")
    
    return nf, legends

# Usage example:
# nf, legends = create_publication_figures(model, adata)