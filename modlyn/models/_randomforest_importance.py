from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    import anndata as ad


class RandomForestImportance:
    """Feature importance using RandomForest's built-in Gini importance.

    This is a non-neural baseline that's fast and interpretable. It uses
    scikit-learn's RandomForestClassifier and extracts feature importances
    after fitting.

    Args:
        adata: An `AnnData` to infer dimensions from.
        label_column: Name of the column in `obs` that contains the target values.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees (None = unlimited).
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 = all cores).

    """

    def __init__(
        self,
        adata: ad.AnnData,
        label_column: str,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self._adata = adata
        self.label_column = label_column
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.label_encoder = LabelEncoder()

    def fit(self, adata: ad.AnnData | None = None):
        """Fit the RandomForest model.

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

        # Fit model
        self.model.fit(X, y_encoded)

    def get_weights(self) -> pd.DataFrame:
        """Get feature importances as a DataFrame.

        Returns a DataFrame where each row corresponds to a class, and values
        represent the feature importance. For RandomForest, we broadcast the
        same importance across all classes since RF gives global importance.

        """
        if not hasattr(self.model, "feature_importances_"):
            raise RuntimeError("Model must be fitted before calling get_weights()")

        importances = self.model.feature_importances_
        n_classes = len(self.label_encoder.classes_)

        # Broadcast importances across all classes (RF doesn't give per-class importance)
        weights = np.tile(importances, (n_classes, 1))

        df = pd.DataFrame(
            weights,
            columns=self._adata.var_names,
            index=self.label_encoder.classes_,
        )
        df.attrs["method_name"] = "randomforest_importance"
        return df

    def score(self, adata: ad.AnnData) -> float:
        """Return accuracy on the given dataset.

        Args:
            adata: `AnnData` object to evaluate on.

        Returns:
            Accuracy score (0.0 to 1.0).

        """
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        y = adata.obs[self.label_column]
        y_encoded = self.label_encoder.transform(y)
        return self.model.score(X, y_encoded)

