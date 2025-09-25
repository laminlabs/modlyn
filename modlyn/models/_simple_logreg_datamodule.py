from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from arrayloaders.io.dask_loader import DaskDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from typing import Literal

    import anndata as ad


class SimpleLogRegDataModule(L.LightningDataModule):
    """A configurable LightningDataModule for classification tasks.

    Supports both TensorDataset (for in-memory data) and DaskDataset (for large datasets).

    Args:
        adata_train: `AnnData` object containing the training data.
        adata_val: `AnnData` object containing the validation data.
        label_column: Name of the column in `obs` that contains the target values.
        dataset_type: Type of dataset to use. Either "in-memory" or "dask-arrayloader".
        train_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the training dataset.
        val_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the validation dataset.
        n_chunks: Number of chunks of the underlying dask.array to load at a time (only used when dataset_type="dask-arrayloader").
        dask_scheduler: The Dask scheduler to use for parallel computation (only used when dataset_type="dask-arrayloader").

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
    """

    def __init__(
        self,
        adata_train: ad.AnnData | None,
        adata_val: ad.AnnData | None,
        label_column: str,
        dataset_type: Literal["in-memory", "dask-arrayloader"] = "in-memory",
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

        # Fit label encoder on training data (only needed for tensor datasets)
        if self.dataset_type == "in-memory" and self.adata_train is not None:
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
        return DaskDataset(
            adata,
            label_column=self.label_col,
            shuffle=shuffle,
            n_chunks=self.n_chunks,
            dask_scheduler=self.dask_scheduler,
        )

    def train_dataloader(self):
        if self.adata_train is None:
            raise ValueError("adata_train is None")

        if self.dataset_type == "in-memory":
            train_dataset = self._create_tensor_dataset(self.adata_train)
        elif self.dataset_type == "dask-arrayloader":
            train_dataset = self._create_dask_dataset(self.adata_train, shuffle=True)
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        return DataLoader(train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self):
        if self.adata_val is None:
            return None

        if self.dataset_type == "in-memory":
            val_dataset = self._create_tensor_dataset(self.adata_val)
        elif self.dataset_type == "dask-arrayloader":
            val_dataset = self._create_dask_dataset(self.adata_val, shuffle=False)
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        return DataLoader(val_dataset, **self.val_dataloader_kwargs)
