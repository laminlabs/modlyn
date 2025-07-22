from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    import anndata as ad


class SimpleLogRegDataModule(L.LightningDataModule):
    """A simple LightningDataModule for classification tasks using TensorDataset.

    Args:
        adata_train: anndata.AnnData object containing the training data.
        adata_val: anndata.AnnData object containing the validation data.
        label_column: Name of the column in `obs` that contains the target values.
        train_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the training dataset.
        val_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the validation dataset.
    """

    def __init__(
        self,
        adata_train: ad.AnnData | None,
        adata_val: ad.AnnData | None,
        label_column: str,
        train_dataloader_kwargs=None,
        val_dataloader_kwargs=None,
    ):
        super().__init__()
        if train_dataloader_kwargs is None:
            train_dataloader_kwargs = {}
        if val_dataloader_kwargs is None:
            val_dataloader_kwargs = {}

        self.adata_train = adata_train
        self.adata_val = adata_val
        self.label_col = label_column
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs

        # Fit label encoder on training data
        if self.adata_train is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.adata_train.obs[self.label_col])

    def _prepare_data(self, adata):
        """Convert AnnData to tensors."""
        # Get features
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X_tensor = torch.FloatTensor(X)

        # Get labels and encode them
        y = adata.obs[self.label_col]
        y_encoded = self.label_encoder.transform(y)
        y_tensor = torch.LongTensor(y_encoded)

        return X_tensor, y_tensor

    def train_dataloader(self):
        if self.adata_train is None:
            raise ValueError("adata_train is None")

        X_tensor, y_tensor = self._prepare_data(self.adata_train)
        train_dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self):
        if self.adata_val is None:
            return None

        X_tensor, y_tensor = self._prepare_data(self.adata_val)
        val_dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(val_dataset, **self.val_dataloader_kwargs)
