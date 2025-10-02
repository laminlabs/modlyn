from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import IterableDataset


def test_dataset_type_alias_normalizes_and_trains():
    # Inject a fake DaskDataset into the expected import path
    fake_pkg = types.ModuleType("arrayloaders")
    fake_io = types.ModuleType("arrayloaders.io")
    fake_dl = types.ModuleType("arrayloaders.io.dask_loader")

    class FakeDaskDataset(IterableDataset):
        def __init__(self, adata, label_column: str, shuffle: bool, n_chunks: int, dask_scheduler: str):
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            self.X = X.astype("float32")
            self.y = pd.Categorical(adata.obs[label_column]).codes.astype("int64")

        def __iter__(self):
            for i in range(self.X.shape[0]):
                yield self.X[i], int(self.y[i])

    fake_dl.DaskDataset = FakeDaskDataset
    sys.modules["arrayloaders"] = fake_pkg
    sys.modules["arrayloaders.io"] = fake_io
    sys.modules["arrayloaders.io.dask_loader"] = fake_dl

    # Small synthetic dataset
    X = np.random.rand(64, 8).astype("float32")
    obs = pd.DataFrame({"cell_line": np.random.choice(["A", "B", "C"], size=64)})
    adata = ad.AnnData(X=X, obs=obs)

    from modlyn.models import SimpleLogReg

    model = SimpleLogReg(adata=adata, label_column="cell_line")
    model.fit(
        adata_train=adata,
        adata_val=None,
        train_dataloader_kwargs={"batch_size": 16, "num_workers": 0},
        dataset_type="arrayloaders-dasd",  # alias to be normalized
        n_chunks=2,
        dask_scheduler="threads",
        max_epochs=1,
        num_sanity_val_steps=0,
        max_steps=5,
    )

    assert model.datamodule is not None
    assert model.datamodule.dataset_type == "dask-arrayloader"

