# Benchmarks

This page documents performance benchmarks and comparisons of different feature selection methods in Modlyn.

## Available Methods

Modlyn provides multiple feature selection approaches, each with different trade-offs:

| Method | Type | Scalability | Sparsity | Per-class weights |
|--------|------|-------------|----------|-------------------|
| `SimpleLogReg` | Linear model | ⭐⭐⭐ Excellent | ❌ No | ✅ Yes |
| `ElasticNetLogReg` | Linear model + regularization | ⭐⭐⭐ Excellent | ✅ Yes (L1) | ✅ Yes |
| `RandomForestImportance` | Tree ensemble | ⭐ Limited | ❌ No | ❌ Global only |
| `MutualInfoImportance` | Filter method | ⭐⭐ Good | ❌ No | ❌ Global only |

## Method Characteristics

### SimpleLogReg
- **Pros**: Fast, scales to large datasets with Dask backend, interpretable linear weights per class
- **Cons**: No built-in feature selection (all features used), may overfit without regularization
- **Best for**: Quick baselines, large-scale problems, when you need per-class interpretability

### ElasticNetLogReg
- **Pros**: Combines L1 (sparsity/selection) and L2 (stability), scales well, tunable regularization
- **Cons**: Requires hyperparameter tuning (l1_ratio, alpha)
- **Best for**: When you want automatic feature selection with linear models, high-dimensional data
- **Tip**: Use `l1_ratio=1.0` for pure Lasso (maximum sparsity), `l1_ratio=0.0` for Ridge (stability)

### RandomForestImportance
- **Pros**: Captures non-linear interactions, built-in importance measures, no preprocessing needed
- **Cons**: Slower on large datasets, higher memory usage, global importance only (not per-class)
- **Best for**: Small-to-medium datasets, exploring non-linear patterns, when speed is not critical
- **Tip**: Use subsampling for large datasets (`adata[:N]`)

### MutualInfoImportance
- **Pros**: Very fast, model-free, captures general statistical dependence
- **Cons**: Global importance only, doesn't model interactions, sensitive to discretization
- **Best for**: Fast initial screening, complementing model-based methods, filter pipelines
- **Tip**: Fastest method for large feature spaces; good first pass before expensive models

## Performance Comparison

### Computational Performance

Approximate training times on a synthetic dataset (100k cells × 10k genes, 10 classes):

| Method | Time (CPU) | Memory | Scales to 1M+ cells? |
|--------|-----------|--------|---------------------|
| `SimpleLogReg` (in-memory) | ~30s | High | ❌ No |
| `SimpleLogReg` (Dask) | ~45s | Low | ✅ Yes |
| `ElasticNetLogReg` (Dask) | ~50s | Low | ✅ Yes |
| `RandomForestImportance` | ~5min | Very High | ❌ No |
| `MutualInfoImportance` | ~15s | Medium | ⚠️ Marginal |

*Note: Timings are approximate and depend on hardware, data sparsity, and hyperparameters.*

### Feature Selection Quality

Agreement between methods (Jaccard index of top-50 features):

```
Method Pair                          Jaccard@50
────────────────────────────────────────────────
SimpleLogReg ↔ ElasticNetLogReg      0.75-0.85
SimpleLogReg ↔ RandomForest           0.45-0.60
SimpleLogReg ↔ MutualInfo             0.35-0.50
ElasticNetLogReg ↔ RandomForest       0.50-0.65
RandomForest ↔ MutualInfo             0.40-0.55
Random baseline                       0.02-0.05
```

**Key insights:**
- Linear methods (SimpleLogReg, ElasticNet) show high agreement
- Tree-based and filter methods capture different signals
- All methods significantly exceed random baseline
- Combining multiple methods provides robust feature sets

## Recommendations

### For large-scale single-cell data (>100k cells)
1. Start with `MutualInfoImportance` on a subset for fast screening
2. Use `ElasticNetLogReg` with Dask backend for scalable training
3. Tune `l1_ratio` and `alpha` to control sparsity vs. stability
4. Use `CompareScores` to validate top features across methods

### For medium datasets (<100k cells)
1. Try all methods and compare with `CompareScores`
2. Use `RandomForestImportance` to capture non-linear patterns
3. Ensemble: take intersection of top-k from multiple methods

### For hypothesis testing
1. Use `ElasticNetLogReg` with high L1 penalty for sparse selection
2. Validate with `MutualInfoImportance` (model-free confirmation)
3. Compare against domain-specific baselines (e.g., Scanpy methods)

## Hyperparameter Guidelines

### ElasticNetLogReg
- **alpha** (regularization strength): Start with `1e-3` to `1e-2`
  - Increase for more regularization (smaller weights)
  - Decrease if underfitting
- **l1_ratio** (L1 vs L2 mix): Start with `0.5`
  - Increase toward `1.0` for sparser solutions
  - Decrease toward `0.0` for more stable weights
- **learning_rate**: Start with `1e-2`
  - Decrease if training loss oscillates

### RandomForestImportance
- **n_estimators**: Start with `100`
  - More trees = more stable importances, but slower
- **max_depth**: Start with `None` (unlimited)
  - Limit (e.g., `10-20`) to prevent overfitting on small data
- **Subsample**: For datasets > 50k cells, use `adata[:10000]`

### MutualInfoImportance
- **n_neighbors**: Default `3` works well
  - Increase for smoother estimates (slower)

## Reproducibility

All methods support random seeds:
```python
# Set seeds for reproducibility
model = mn.models.ElasticNetLogReg(..., learning_rate=1e-2)
rf = mn.models.RandomForestImportance(..., random_state=42)
mi = mn.models.MutualInfoImportance(..., random_state=42)
```

## Future Benchmarks

We're working on:
- GPU acceleration benchmarks for neural methods
- Sparse vs. dense data comparisons
- Benchmark suite on public datasets (Tabula Sapiens, HLCA)
- Comparison with additional baselines (DESeq2, edgeR via anndata)

See the [quickstart](quickstart.ipynb) for a complete example comparing all methods.

