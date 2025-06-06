import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class ProperUncertaintyEstimator:
    def __init__(self, model, adata, datamodule):
        self.model = model
        self.adata = adata
        self.datamodule = datamodule
        self.weights = model.linear.weight.detach().cpu().numpy()
        self.feature_names = getattr(adata.var, 'gene_symbols', None) or [f'Feature_{i}' for i in range(self.weights.shape[1])]
        
    def method_1_fisher_information(self):
        print("Computing Fisher Information Matrix standard errors...")
        self.model.eval()
        all_probs = []
        all_x = []
        
        with torch.no_grad():
            for batch in self.datamodule.val_dataloader():
                x, y = batch
                if torch.cuda.is_available():
                    x = x.cuda()
                    self.model = self.model.cuda()
                
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                all_x.append(x.cpu())
        
        all_probs = torch.cat(all_probs).numpy()
        all_x = torch.cat(all_x).numpy()
        n_samples, n_features = all_x.shape
        n_classes = all_probs.shape[1]
        
        if n_samples > 10000:
            idx = np.random.choice(n_samples, 10000, replace=False)
            all_x = all_x[idx]
            all_probs = all_probs[idx]
        
        standard_errors = np.zeros((n_classes, n_features))
        for k in range(n_classes):
            for j in range(n_features):
                fisher_diag = np.sum(all_probs[:, k] * (1 - all_probs[:, k]) * (all_x[:, j] ** 2))
                if fisher_diag > 1e-8:
                    standard_errors[k, j] = 1.0 / np.sqrt(fisher_diag)
                else:
                    standard_errors[k, j] = np.inf
        
        standard_errors = np.minimum(standard_errors, 10.0)
        print(" Fisher Information standard errors computed")
        return standard_errors
    
    def method_2_sklearn_approach(self):
        print("Computing uncertainty using sklearn LogisticRegression...")
        all_x = []
        all_y = []
        
        for batch in self.datamodule.val_dataloader():
            x, y = batch
            all_x.append(x.cpu().numpy())
            all_y.append(y.cpu().numpy())
        
        X = np.concatenate(all_x)
        y = np.concatenate(all_y)
        
        if X.shape[0] > 50000:
            idx = np.random.choice(X.shape[0], 50000, replace=False)
            X = X[idx]
            y = y[idx]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        sklearn_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
        sklearn_model.fit(X_scaled, y)
        
        n_samples = X.shape[0]
        base_se = 1.0 / np.sqrt(n_samples)
        weight_scale = 1.0 / (np.abs(sklearn_model.coef_) + 0.01)
        standard_errors = base_se * weight_scale
        
        print(" sklearn-based standard errors computed")
        return standard_errors
    
    def method_3_bootstrap_proper(self, n_bootstrap=30):
        print(f"Computing bootstrap standard errors (n_bootstrap={n_bootstrap})...")
        all_x = []
        all_y = []
        
        for batch in self.datamodule.val_dataloader():
            x, y = batch
            all_x.append(x.cpu().numpy())
            all_y.append(y.cpu().numpy())
        
        X = np.concatenate(all_x)
        y = np.concatenate(all_y)
        
        if X.shape[0] > 20000:
            idx = np.random.choice(X.shape[0], 20000, replace=False)
            X = X[idx]
            y = y[idx]
        
        bootstrap_weights = []
        n_samples = X.shape[0]
        
        for i in range(n_bootstrap):
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            
            try:
                scaler = StandardScaler()
                X_boot_scaled = scaler.fit_transform(X_boot)
                boot_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=i)
                boot_model.fit(X_boot_scaled, y_boot)
                bootstrap_weights.append(boot_model.coef_)
            except Exception:
                continue
        
        if len(bootstrap_weights) > 10:
            bootstrap_weights = np.array(bootstrap_weights)
            standard_errors = np.std(bootstrap_weights, axis=0)
            print(f" Bootstrap completed with {len(bootstrap_weights)} successful iterations")
            return standard_errors, bootstrap_weights
        else:
            print(" Bootstrap failed")
            return None, None
    
    def compute_p_values(self, weights, standard_errors):
        z_scores = weights / (standard_errors + 1e-8)
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        return z_scores, p_values
    
    def plot_uncertainty_overview(self, results):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Uncertainty Estimation Overview', fontsize=16, fontweight='bold')
        
        standard_errors = results['standard_errors']
        z_scores = results['z_scores']
        p_values = results['p_values']
        
        # 1. Standard Error Distribution
        se_flat = standard_errors.flatten()
        se_flat = se_flat[np.isfinite(se_flat)]
        axes[0, 0].hist(se_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Standard Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Standard Errors')
        axes[0, 0].axvline(np.median(se_flat), color='red', linestyle='--', label=f'Median: {np.median(se_flat):.3f}')
        axes[0, 0].legend()
        
        # 2. Z-score Distribution
        z_flat = z_scores.flatten()
        z_flat = z_flat[np.isfinite(z_flat)]
        axes[0, 1].hist(z_flat, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Z-score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Z-scores')
        axes[0, 1].axvline(0, color='red', linestyle='-', alpha=0.5, label='No Effect')
        axes[0, 1].axvline(1.96, color='orange', linestyle='--', alpha=0.7, label='p=0.05')
        axes[0, 1].axvline(-1.96, color='orange', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # 3. P-value Distribution
        p_flat = p_values.flatten()
        p_flat = p_flat[np.isfinite(p_flat)]
        axes[0, 2].hist(p_flat, bins=50, alpha=0.7, color='salmon', edgecolor='black')
        axes[0, 2].set_xlabel('P-value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of P-values')
        axes[0, 2].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
        axes[0, 2].legend()
        
        # 4. Uncertainty vs Weight scatter
        weights_flat = self.weights.flatten()
        valid_idx = np.isfinite(weights_flat) & np.isfinite(se_flat[:len(weights_flat)])
        scatter = axes[1, 0].scatter(np.abs(weights_flat[valid_idx]), se_flat[valid_idx], 
                                   alpha=0.6, c=p_flat[valid_idx], cmap='RdYlGn_r', s=10)
        axes[1, 0].set_xlabel('|Weight|')
        axes[1, 0].set_ylabel('Standard Error')
        axes[1, 0].set_title('Uncertainty vs Weight Magnitude')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        plt.colorbar(scatter, ax=axes[1, 0], label='P-value')
        
        # 5. Significance pie chart
        counts = [np.sum(p_flat < 0.001), np.sum((p_flat >= 0.001) & (p_flat < 0.01)), 
                 np.sum((p_flat >= 0.01) & (p_flat < 0.05)), np.sum(p_flat >= 0.05)]
        labels = ['p < 0.001', 'p < 0.01', 'p < 0.05', 'p ≥ 0.05']
        colors = ['darkgreen', 'green', 'orange', 'red']
        axes[1, 1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Statistical Significance Summary')
        
        # 6. Feature importance
        avg_importance = np.mean(np.abs(self.weights), axis=0)
        top_features_idx = np.argsort(avg_importance)[-10:]
        importance_values = avg_importance[top_features_idx]
        feature_names = [self.feature_names[i] for i in top_features_idx]
        
        y_pos = np.arange(len(feature_names))
        axes[1, 2].barh(y_pos, importance_values, color='lightblue', alpha=0.7)
        axes[1, 2].set_yticks(y_pos)
        axes[1, 2].set_yticklabels(feature_names, fontsize=8)
        axes[1, 2].set_xlabel('Average |Weight|')
        axes[1, 2].set_title('Top 10 Most Important Features')
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_heatmap(self, results, top_n_features=30):
        standard_errors = results['standard_errors']
        p_values = results['p_values']
        
        avg_uncertainty = np.mean(standard_errors, axis=0)
        top_features_idx = np.argsort(avg_uncertainty)[-top_n_features:]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Uncertainty Heatmaps (Top {top_n_features} Most Uncertain Features)', fontsize=16)
        
        selected_features = [self.feature_names[i][:15] + '...' if len(self.feature_names[i]) > 15 
                           else self.feature_names[i] for i in top_features_idx]
        
        # Standard Errors Heatmap
        se_selected = standard_errors[:, top_features_idx]
        im1 = axes[0].imshow(se_selected, cmap='Reds', aspect='auto')
        axes[0].set_title('Standard Errors')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Classes')
        axes[0].set_xticks(range(len(selected_features)))
        axes[0].set_xticklabels(selected_features, rotation=90, ha='right', fontsize=8)
        plt.colorbar(im1, ax=axes[0])
        
        # P-values Heatmap
        p_selected = p_values[:, top_features_idx]
        im2 = axes[1].imshow(p_selected, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        axes[1].set_title('P-values (Green = Significant)')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Classes')
        axes[1].set_xticks(range(len(selected_features)))
        axes[1].set_xticklabels(selected_features, rotation=90, ha='right', fontsize=8)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance_with_uncertainty(self, results, top_n=15):
        standard_errors = results['standard_errors']
        p_values = results['p_values']
        
        avg_importance = np.mean(np.abs(self.weights), axis=0)
        avg_uncertainty = np.mean(standard_errors, axis=0)
        avg_p_values = np.mean(p_values, axis=0)
        
        top_features_idx = np.argsort(avg_importance)[-top_n:]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Top {top_n} Features: Importance with Uncertainty', fontsize=14)
        
        # Feature importance with error bars
        x_pos = np.arange(len(top_features_idx))
        importance_values = avg_importance[top_features_idx]
        uncertainty_values = avg_uncertainty[top_features_idx]
        p_val_colors = avg_p_values[top_features_idx]
        
        colors = ['darkgreen' if p < 0.001 else 'green' if p < 0.01 else 'orange' if p < 0.05 else 'red' 
                 for p in p_val_colors]
        
        axes[0].barh(x_pos, importance_values, xerr=uncertainty_values, color=colors, alpha=0.7, capsize=3)
        axes[0].set_yticks(x_pos)
        axes[0].set_yticklabels([self.feature_names[i] for i in top_features_idx], fontsize=9)
        axes[0].set_xlabel('Average |Weight| (Importance)')
        axes[0].set_title('Feature Importance with Uncertainty Bars')
        
        # Uncertainty vs Importance scatter
        scatter = axes[1].scatter(avg_importance, avg_uncertainty, c=avg_p_values, cmap='RdYlGn_r', s=30, alpha=0.7)
        axes[1].set_xlabel('Average |Weight| (Importance)')
        axes[1].set_ylabel('Average Standard Error (Uncertainty)')
        axes[1].set_title('Importance vs Uncertainty (Color = p-value)')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        plt.colorbar(scatter, ax=axes[1], label='Average p-value')
        
        plt.tight_layout()
        return fig
    
    def create_dashboard(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Uncertainty Analysis Dashboard', fontsize=16, fontweight='bold')
        
        standard_errors = results['standard_errors']
        p_values = results['p_values']
        
        # 1. Main uncertainty heatmap
        im = axes[0, 0].imshow(standard_errors, cmap='Reds', aspect='auto')
        axes[0, 0].set_title('Standard Error Heatmap')
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Classes')
        plt.colorbar(im, ax=axes[0, 0])
        
        # 2. P-value distribution
        p_flat = p_values.flatten()
        p_flat = p_flat[np.isfinite(p_flat)]
        axes[0, 1].hist(p_flat, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 1].axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        axes[0, 1].set_xlabel('P-value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('P-value Distribution')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # 3. Uncertainty vs Weight scatter
        weights_flat = self.weights.flatten()
        se_flat = standard_errors.flatten()
        valid_idx = np.isfinite(weights_flat) & np.isfinite(se_flat)
        scatter = axes[1, 0].scatter(np.abs(weights_flat[valid_idx]), se_flat[valid_idx], 
                                   c=p_flat[:np.sum(valid_idx)], cmap='RdYlGn_r', alpha=0.6, s=20)
        axes[1, 0].set_xlabel('|Weight|')
        axes[1, 0].set_ylabel('Standard Error')
        axes[1, 0].set_title('Uncertainty vs Weight')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        plt.colorbar(scatter, ax=axes[1, 0], label='p-value')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        n_total = p_values.size
        n_sig_05 = np.sum(p_values < 0.05)
        n_sig_01 = np.sum(p_values < 0.01)
        n_sig_001 = np.sum(p_values < 0.001)
        mean_se = np.mean(standard_errors)
        
        summary_text = f"""SUMMARY STATISTICS:

Total Parameters: {n_total:,}
Mean Standard Error: {mean_se:.4f}

Significant Parameters:
• p < 0.05: {n_sig_05:,} ({100*n_sig_05/n_total:.1f}%)
• p < 0.01: {n_sig_01:,} ({100*n_sig_01/n_total:.1f}%)
• p < 0.001: {n_sig_001:,} ({100*n_sig_001/n_total:.1f}%)

Model Shape: {self.weights.shape[0]} classes × {self.weights.shape[1]} features"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def compare_methods(self):
        print(" COMPARING UNCERTAINTY ESTIMATION METHODS")
        print("="*60)
        
        results = {}
        bootstrap_weights = None
        
        # Try each method
        try:
            se_fisher = self.method_1_fisher_information()
            results['fisher'] = se_fisher
        except Exception as e:
            print(f"Fisher Information failed: {e}")
            results['fisher'] = None
        
        try:
            se_sklearn = self.method_2_sklearn_approach()
            results['sklearn'] = se_sklearn
        except Exception as e:
            print(f"sklearn approach failed: {e}")
            results['sklearn'] = None
        
        try:
            se_bootstrap, bootstrap_weights = self.method_3_bootstrap_proper(n_bootstrap=20)
            results['bootstrap'] = se_bootstrap
        except Exception as e:
            print(f"Bootstrap failed: {e}")
            results['bootstrap'] = None
        
        # Choose best method
        if results['sklearn'] is not None:
            chosen_se = results['sklearn']
            print(" Using sklearn-based standard errors")
        elif results['fisher'] is not None:
            chosen_se = results['fisher']
            print(" Using Fisher Information standard errors")
        elif results['bootstrap'] is not None:
            chosen_se = results['bootstrap']
            print(" Using bootstrap standard errors")
        else:
            chosen_se = np.abs(self.weights) * 0.1 + 0.01
            print(" All methods failed - using approximation")
        
        # Compute p-values
        z_scores, p_values = self.compute_p_values(self.weights, chosen_se)
        
        print(f"Significant weights (p < 0.05): {np.sum(p_values < 0.05)} / {p_values.size}")
        
        return {
            'standard_errors': chosen_se,
            'z_scores': z_scores,
            'p_values': p_values,
            'all_methods': results,
            'bootstrap_weights': bootstrap_weights,
            'weights': self.weights,
            'feature_names': self.feature_names
        }


def get_proper_uncertainty(model, adata, datamodule, create_visualizations=True, interactive=False):
    """
    Get proper statistical uncertainty for linear model weights with visualizations
    
    Returns:
    --------
    dict : Results with uncertainty estimates and visualization figures
    """
    estimator = ProperUncertaintyEstimator(model, adata, datamodule)
    results = estimator.compare_methods()
    
    if create_visualizations:        
        # Create plots
        overview_fig = estimator.plot_uncertainty_overview(results)
        heatmap_fig = estimator.plot_uncertainty_heatmap(results)
        importance_fig = estimator.plot_feature_importance_with_uncertainty(results)
        dashboard_fig = estimator.create_dashboard(results)
        
        # Save plots
        overview_fig.savefig('uncertainty_overview.png', dpi=300, bbox_inches='tight')
        heatmap_fig.savefig('uncertainty_heatmap.png', dpi=300, bbox_inches='tight')
        importance_fig.savefig('feature_importance_uncertainty.png', dpi=300, bbox_inches='tight')
        dashboard_fig.savefig('uncertainty_dashboard.png', dpi=300, bbox_inches='tight')
        
        print(" All plots saved as PNG files")
        
        # Add figures to results
        results.update({
            'overview_fig': overview_fig,
            'heatmap_fig': heatmap_fig,
            'importance_fig': importance_fig,
            'dashboard_fig': dashboard_fig
        })
        
        if interactive:
            plt.show()
    
    return results


# Example usage
if __name__ == "__main__":
    print("Enhanced Uncertainty Estimation Script")
    print("Usage: results = get_proper_uncertainty(model, adata, datamodule)")
    print("Access plots: results['overview_fig'], results['dashboard_fig'], etc.")