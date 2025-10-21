from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    import anndata as ad


class MutualInfoImportance:
    """Feature importance using Mutual Information.

    Computes the mutual information between each feature and the target labels.
    This is a filter method that doesn't require training a model, making it
    very fast for large datasets.

    Args:
        adata: An `AnnData` to infer dimensions from.
        label_column: Name of the column in `obs` that contains the target values.
        random_state: Random seed for reproducibility.
        n_neighbors: Number of neighbors to use for MI estimation.

    """

    def __init__(
        self,
        adata: ad.AnnData,
        label_column: str,
        random_state: int = 42,
        n_neighbors: int = 3,
    ):
        self._adata = adata
        self.label_column = label_column
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.label_encoder = LabelEncoder()
        self.mi_scores: np.ndarray | None = None

    def fit(self, adata: ad.AnnData | None = None):
        """Compute mutual information scores.

        Args:
            adata: `AnnData` object containing the data. If None, uses the adata from __init__.

        """
        if adata is None:
            adata = self._adata

        # Prepare features
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # Prepare labels
        y = adata.obs[self.label_column]
        y_encoded = self.label_encoder.fit_transform(y)

        # Compute mutual information
        self.mi_scores = mutual_info_classif(
            X, y_encoded, random_state=self.random_state, n_neighbors=self.n_neighbors
        )

    def get_weights(self) -> pd.DataFrame:
        """Get mutual information scores as a DataFrame.

        Returns a DataFrame where each row corresponds to a class, and values
        represent the MI score. MI is computed globally (not per-class), so we
        broadcast the same scores across all classes.

        """
        if self.mi_scores is None:
            raise RuntimeError("Model must be fitted before calling get_weights()")

        n_classes = len(self.label_encoder.classes_)

        # Broadcast MI scores across all classes (MI is global, not per-class)
        weights = np.tile(self.mi_scores, (n_classes, 1))

        df = pd.DataFrame(
            weights,
            columns=self._adata.var_names,
            index=self.label_encoder.classes_,
        )
        df.attrs["method_name"] = "mutual_info"
        return df

