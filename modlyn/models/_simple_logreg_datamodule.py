from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from typing import Literal

    import anndata as ad


class SimpleLogRegDataModule(L.LightningDataModule):
    """A configurable LightningDataModule for classification tasks.

    Supports three backends:
    - TensorDataset (in-memory)
    - DaskDataset (arrayloaders)
    - annbatch (on-disk Zarr via annbatch)

    Args:
        adata_train: `AnnData` object containing the training data.
        adata_val: `AnnData` object containing the validation data.
        label_column: Name of the column in `obs` that contains the target values.
        dataset_type: Type of dataset to use. One of: "in-memory", "dask-arrayloader", or "annbatch".
        train_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the training dataset.
        val_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the validation dataset.
        n_chunks: Number of chunks of the underlying dask.array to load at a time (only used when dataset_type="dask-arrayloader").
        dask_scheduler: The Dask scheduler to use for parallel computation (only used when dataset_type="dask-arrayloader").
        Note on annbatch: pass annbatch configuration inside train_dataloader_kwargs under key
            "annbatch_config" (dict with keys like batch_size, chunk_size, preload_nchunks, preload_to_gpu).

    Examples:
        >>> # For small datasets (in-memory)
        >>> datamodule = ConfigurableDataModule(
        ...     adata_train=train_data,
        ...     adata_val=val_data,
        ...     label_column="cell_type",
        ...     dataset_type="in-memory"
        ... )

        >>> # For large datasets (dask-backed)
        >>> from arrayloaders.io.dask_loader import read_lazy_store
        >>> adata_train = read_lazy_store("path/to/train/store", obs_columns=["label"])
        >>> datamodule = ConfigurableDataModule(
        ...     adata_train=adata_train,
        ...     adata_val=None,
        ...     label_column="label",
        ...     dataset_type="dask-arrayloader",
        ...     train_dataloader_kwargs={
        ...         "batch_size": 2048,
        ...         "drop_last": True,
        ...         "num_workers": 4
        ...     },
        ...     n_chunks=16,
        ...     dask_scheduler="threads"
        ... )
        >>> # annbatch (on-disk Zarr)
        >>> # Ensure adata_train.uns["_annbatch_paths"] = ["/path/to/collection/dataset_0.zarr", ...]
        >>> datamodule = ConfigurableDataModule(
        ...     adata_train=adata_train,
        ...     adata_val=None,
        ...     label_column="label",
        ...     dataset_type="annbatch",
        ...     train_dataloader_kwargs={
        ...         "annbatch_config": {"batch_size": 2048, "chunk_size": 32, "preload_nchunks": 64, "preload_to_gpu": False}
        ...     },
        ... )
    """

    def __init__(
        self,
        adata_train: ad.AnnData | None,
        adata_val: ad.AnnData | None,
        label_column: str,
        dataset_type: Literal["in-memory", "dask-arrayloader", "annbatch"] = "in-memory",
        train_dataloader_kwargs=None,
        val_dataloader_kwargs=None,
        n_chunks: int = 8,
        dask_scheduler: Literal["synchronous", "threads"] = "threads",
    ):
        super().__init__()
        if train_dataloader_kwargs is None:
            train_dataloader_kwargs = {}
        if val_dataloader_kwargs is None:
            val_dataloader_kwargs = {}

        self.adata_train = adata_train
        self.adata_val = adata_val
        self.label_col = label_column
        self.dataset_type = dataset_type
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs
        self.n_chunks = n_chunks
        self.dask_scheduler = dask_scheduler
        # Extract annbatch configuration if present to avoid leaking into torch DataLoader kwargs
        self.annbatch_config = {}
        if "annbatch_config" in self.train_dataloader_kwargs:
            try:
                self.annbatch_config = dict(self.train_dataloader_kwargs.pop("annbatch_config"))
            except Exception:
                self.annbatch_config = {}

        # Fit label encoder on training data (used by both backends)
        self.label_encoder = None
        if self.adata_train is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.adata_train.obs[self.label_col])

    def _prepare_tensor_data(self, adata):
        """Convert AnnData to tensors for TensorDataset."""
        # Get features
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X_tensor = torch.FloatTensor(X)

        # Get labels and encode them
        y = adata.obs[self.label_col]
        y_encoded = self.label_encoder.transform(y)
        y_tensor = torch.LongTensor(y_encoded)

        return X_tensor, y_tensor

    def _create_tensor_dataset(self, adata):
        """Create a TensorDataset from AnnData."""
        X_tensor, y_tensor = self._prepare_tensor_data(adata)
        return TensorDataset(X_tensor, y_tensor)

    def _create_dask_dataset(self, adata, shuffle=True):
        """Create a DaskDataset from AnnData."""
        try:
            from arrayloaders.io.dask_loader import DaskDataset  # lazy import
        except Exception as e:
            raise ImportError(
                "arrayloaders is required for dataset_type='dask-arrayloader'. Install with `pip install arrayloaders`."
            ) from e

        return DaskDataset(
            adata,
            label_column=self.label_col,
            shuffle=shuffle,
            n_chunks=self.n_chunks,
            dask_scheduler=self.dask_scheduler,
        )

    def _create_annbatch_iterable(self, adata, shuffle=True):
        """Create an annbatch-backed IterableDataset from zarr paths.

        Expects a list of zarr dataset paths in `adata.uns["_annbatch_paths"]`.
        """
        try:
            from annbatch import ZarrSparseDataset
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "annbatch is required for dataset_type='annbatch'. Install with `pip install annbatch[zarrs]`."
            ) from e

        import anndata as ad
        import zarr
        from pathlib import Path
        import numpy as np
        import torch
        from torch.utils.data import IterableDataset

        paths = adata.uns.get("_annbatch_paths")
        if not paths:
            raise ValueError(
                "AnnData.uns['_annbatch_paths'] missing; provide list of zarr dataset paths"
            )
        # Build lightweight AnnData handles for each path
        ann_batches: list[ad.AnnData] = []
        for p in paths:
            store = zarr.open(Path(p))
            if not ("X" in store and "obs" in store):  # skip invalid
                continue
            ann_batches.append(
                ad.AnnData(
                    X=ad.io.sparse_dataset(store["X"]),
                    obs=ad.io.read_elem(store["obs"]),
                )
            )
        if not ann_batches:
            raise ValueError("No valid zarr datasets found for annbatch")

        cfg = {
            "batch_size": 2048,
            "chunk_size": 32,
            "preload_nchunks": 64,
            "preload_to_gpu": False,
        }
        cfg.update(self.annbatch_config or {})

        ds = (
            ZarrSparseDataset(
                batch_size=cfg.get("batch_size", 2048),
                chunk_size=cfg.get("chunk_size", 32),
                preload_nchunks=cfg.get("preload_nchunks", 64),
                preload_to_gpu=cfg.get("preload_to_gpu", False),
            ).add_anndatas(ann_batches, obs_keys=self.label_col)
        )

        class _AnnBatchTorchIterable(IterableDataset):
            """Wrap annbatch iterator to yield torch tensors per batch."""

            def __init__(self, annbatch_ds, label_encoder):
                self._ds = annbatch_ds
                self._le = label_encoder

            def __iter__(self):
                for batch in self._ds:
                    # Expect batch as (X, y) arrays
                    if isinstance(batch, tuple) and len(batch) == 2:
                        xb, yb = batch
                    elif isinstance(batch, dict) and "X" in batch and "y" in batch:
                        xb, yb = batch["X"], batch["y"]
                    else:
                        # Fallback: skip unknown batch format
                        continue
                    # Encode labels if non-integer
                    try:
                        yb_enc = self._le.transform(yb) if yb.dtype.kind not in ("i", "u") else yb
                    except Exception:
                        yb_enc = np.array([int(v) for v in yb], dtype=np.int64)
                    x_tensor = torch.as_tensor(xb, dtype=torch.float32)
                    y_tensor = torch.as_tensor(yb_enc, dtype=torch.long)
                    yield x_tensor, y_tensor

        return _AnnBatchTorchIterable(ds, self.label_encoder)

    def _collate_dask_batch(self, batch):
        """Collate function for DaskDataset batches -> (x_tensor, y_tensor)."""
        import numpy as np
        import torch

        try:
            import scipy.sparse as sp
        except Exception:  # pragma: no cover - optional
            sp = None

        if not batch:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        first = batch[0]
        if isinstance(first, tuple) and len(first) == 3:
            xs, ys, _ = zip(*batch, strict=False)
        else:
            xs, ys = zip(*batch, strict=False)
        if self.label_encoder is None:
            raise RuntimeError("label_encoder not initialized")
        # Encode labels; fallback to ints if encoder mismatch occurs
        try:
            y_enc = self.label_encoder.transform(list(ys))
        except Exception:
            y_enc = np.array([int(y) for y in ys], dtype=np.int64)
        # ensure each row is a contiguous 1D float32 array; handle sparse and object types
        xs_arr = []
        for x in xs:
            # densify sparse rows
            if sp is not None and getattr(sp, "issparse", None) and sp.issparse(x):
                arr = x.toarray()
            else:
                arr = np.asarray(x)
            # flatten any 2D shapes (e.g., 1 x n_vars)
            if arr.ndim > 1:
                arr = arr.ravel()
            # robust dtype conversion
            if arr.dtype == object:
                # last-resort element-wise float coercion
                try:
                    arr = arr.astype(np.float32, copy=False)
                except Exception:
                    arr = np.array([float(v) for v in arr], dtype=np.float32)
            else:
                arr = arr.astype(np.float32, copy=False)
            xs_arr.append(arr)
        x_tensor = torch.as_tensor(np.stack(xs_arr, axis=0), dtype=torch.float32)
        y_tensor = torch.as_tensor(y_enc, dtype=torch.long)
        return x_tensor, y_tensor

    def train_dataloader(self):
        if self.adata_train is None:
            raise ValueError("adata_train is None")

        kwargs = dict(self.train_dataloader_kwargs)
        if self.dataset_type == "in-memory":
            train_dataset = self._create_tensor_dataset(self.adata_train)
        elif self.dataset_type == "dask-arrayloader":
            train_dataset = self._create_dask_dataset(self.adata_train, shuffle=True)
            kwargs.setdefault("collate_fn", self._collate_dask_batch)
        elif self.dataset_type == "annbatch":
            # annbatch iterator yields already-batched tensors; wrap with DataLoader using batch_size=1
            # and identity collate so PyTorch doesn't re-batch and interfere with annbatch's batching
            train_dataset = self._create_annbatch_iterable(self.adata_train, shuffle=True)
            # Remove user-provided batching args (irrelevant for annbatch)
            for k in ("batch_size", "drop_last", "shuffle", "num_workers"):
                kwargs.pop(k, None)
            kwargs["batch_size"] = 1
            kwargs["num_workers"] = 0
            kwargs["collate_fn"] = (lambda batch: batch[0])
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        return DataLoader(train_dataset, **kwargs)

    def val_dataloader(self):
        if self.adata_val is None:
            return []

        kwargs = dict(self.val_dataloader_kwargs)
        if self.dataset_type == "in-memory":
            val_dataset = self._create_tensor_dataset(self.adata_val)
        elif self.dataset_type == "dask-arrayloader":
            val_dataset = self._create_dask_dataset(self.adata_val, shuffle=False)
            kwargs.setdefault("collate_fn", self._collate_dask_batch)
        elif self.dataset_type == "annbatch":
            val_dataset = self._create_annbatch_iterable(self.adata_val, shuffle=False)
            for k in ("batch_size", "drop_last", "shuffle", "num_workers"):
                kwargs.pop(k, None)
            kwargs["batch_size"] = 1
            kwargs["num_workers"] = 0
            kwargs["collate_fn"] = (lambda batch: batch[0])
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        return DataLoader(val_dataset, **kwargs)
