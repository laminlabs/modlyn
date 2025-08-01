# Validation Notebook Fixes

## Root Cause: Regularization & Training Mismatch
The negative correlation (-0.034) indicates that Modlyn and Sklearn are learning completely different patterns. This is caused by:

1. **Regularization mismatch**: sklearn has default L2 regularization (C=1.0), modlyn likely has weight_decay=0
2. **Insufficient training**: modlyn needs more epochs to converge
3. **Batch size issues**: small datasets need full batch training
4. **Different optimizers**: sklearn uses LBFGS, Lightning uses Adam by default

## Exact Code Changes Needed

### In validate_arrayloader_equivalence.ipynb:

**1. Fix SimpleLogReg parameters:**
```python
# BEFORE (causing issues):
linear_model = SimpleLogReg(
    adata=adata_modlyn,
    label_column="y",
    learning_rate=1e-3,
    weight_decay=1e-4  # TOO LOW!
)

# AFTER (fixed):
linear_model = SimpleLogReg(
    adata=adata_modlyn,
    label_column="y",
    learning_rate=1e-2,    # Higher learning rate
    weight_decay=1.0       # Match sklearn's default regularization
)
```

**2. Fix training parameters:**
```python
# BEFORE:
trainer = L.Trainer(
    max_epochs=5,  # TOO FEW!
    enable_progress_bar=True,
    logger=False
)

# AFTER:
trainer = L.Trainer(
    max_epochs=100,        # Much more training
    enable_progress_bar=True,
    logger=False,
    enable_checkpointing=False
)
```

**3. Fix datamodule for small datasets:**
```python
# BEFORE:
datamodule = SimpleLogRegDataModule(
    adata_train=adata_train,
    adata_val=adata_val,
    label_column="y",
    train_dataloader_kwargs={"batch_size": 512, "num_workers": 0},  # Mini-batch bad for small data
    val_dataloader_kwargs={"batch_size": 512, "num_workers": 0}
)

# AFTER:
datamodule = SimpleLogRegDataModule(
    adata_train=adata_train,
    adata_val=adata_val,
    label_column="y",
    train_dataloader_kwargs={"batch_size": len(adata_train), "num_workers": 0},  # Full batch
    val_dataloader_kwargs={"batch_size": len(adata_val), "num_workers": 0}
)
```

**4. Add reproducibility:**
```python
# Add at the top of the notebook:
import torch
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

## Expected Results After Fixes

With these changes, you should see:
- ✅ Weight correlations > 0.95 (instead of -0.034)
- ✅ Similar training accuracies between methods
- ✅ Most cell lines with >99% correlation
- ✅ Validation: "SUCCESS: All results are essentially identical!"

## Background: Why These Fixes Work

### Regularization Matching
- sklearn LogisticRegression has default C=1.0 (L2 penalty)
- This roughly corresponds to weight_decay=1.0 in PyTorch
- Your current weight_decay=1e-4 is 10,000x weaker!

### Training Convergence
- sklearn's LBFGS optimizer converges quickly
- Lightning's Adam needs many more epochs (100+ vs 5)
- Full batch training mimics sklearn's behavior better

### From Alex Wolf's Analysis
> "The results have to be identical for all dataset sizes where we can use scanpy/sklearn. If they are not, we have to find better hyper parameters."

> "What this also shows is the absence of L2 regularization that sklearn has by default. That's why we have all these blue values in Modlyn, but not in Scanpy."
