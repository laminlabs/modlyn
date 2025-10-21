from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def small_adata():
    """Create a small synthetic AnnData for testing."""
    rng = np.random.default_rng(42)
    n_obs, n_vars = 100, 50
    X = rng.random((n_obs, n_vars)).astype("float32")
    obs = pd.DataFrame({"cell_type": rng.choice(["A", "B", "C"], size=n_obs)})
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return AnnData(X=X, obs=obs, var=var)


class TestElasticNetLogReg:
    def test_initialization(self, small_adata):
        from modlyn.models import ElasticNetLogReg

        model = ElasticNetLogReg(
            adata=small_adata,
            label_column="cell_type",
            l1_ratio=0.7,
            alpha=0.1,
        )
        assert model.l1_ratio == 0.7
        assert model.alpha == 0.1
        assert model.linear.in_features == 50
        assert model.linear.out_features == 3

    def test_fit_in_memory(self, small_adata):
        from modlyn.models import ElasticNetLogReg

        model = ElasticNetLogReg(
            adata=small_adata, label_column="cell_type", learning_rate=1e-2
        )
        model.fit(
            adata_train=small_adata,
            adata_val=None,
            dataset_type="in-memory",
            train_dataloader_kwargs={"batch_size": 16},
            max_epochs=1,
            max_steps=5,
        )
        assert model.datamodule is not None
        assert len(model.train_losses) > 0

    def test_get_weights(self, small_adata):
        from modlyn.models import ElasticNetLogReg

        model = ElasticNetLogReg(adata=small_adata, label_column="cell_type")
        model.fit(
            adata_train=small_adata,
            adata_val=None,
            dataset_type="in-memory",
            train_dataloader_kwargs={"batch_size": 16},
            max_epochs=1,
            max_steps=5,
        )
        weights_df = model.get_weights()
        assert weights_df.shape == (3, 50)  # 3 classes, 50 genes
        assert "method_name" in weights_df.attrs
        assert weights_df.attrs["method_name"].startswith("elasticnet")

    def test_elasticnet_penalty(self, small_adata):
        from modlyn.models import ElasticNetLogReg

        model = ElasticNetLogReg(
            adata=small_adata, label_column="cell_type", l1_ratio=0.5, alpha=1.0
        )
        penalty = model.compute_elasticnet_penalty()
        assert penalty.item() > 0  # Penalty should be positive

    def test_l1_only(self, small_adata):
        """Test pure L1 (Lasso) regularization."""
        from modlyn.models import ElasticNetLogReg

        model = ElasticNetLogReg(
            adata=small_adata, label_column="cell_type", l1_ratio=1.0, alpha=1.0
        )
        model.fit(
            adata_train=small_adata,
            adata_val=None,
            dataset_type="in-memory",
            train_dataloader_kwargs={"batch_size": 16},
            max_epochs=2,
            max_steps=10,
        )
        # With high L1, we expect some weights to be driven toward zero
        weights = model.linear.weight.detach().numpy()
        assert np.sum(np.abs(weights) < 0.1) > 0

    def test_l2_only(self, small_adata):
        """Test pure L2 (Ridge) regularization."""
        from modlyn.models import ElasticNetLogReg

        model = ElasticNetLogReg(
            adata=small_adata, label_column="cell_type", l1_ratio=0.0, alpha=1.0
        )
        model.fit(
            adata_train=small_adata,
            adata_val=None,
            dataset_type="in-memory",
            train_dataloader_kwargs={"batch_size": 16},
            max_epochs=1,
            max_steps=5,
        )
        weights = model.linear.weight.detach().numpy()
        # L2 doesn't zero out weights, just shrinks them
        assert np.all(np.abs(weights) > 0)


class TestRandomForestImportance:
    def test_initialization(self, small_adata):
        from modlyn.models import RandomForestImportance

        model = RandomForestImportance(
            adata=small_adata, label_column="cell_type", n_estimators=50
        )
        assert model.n_estimators == 50
        assert model.label_column == "cell_type"

    def test_fit(self, small_adata):
        from modlyn.models import RandomForestImportance

        model = RandomForestImportance(
            adata=small_adata, label_column="cell_type", n_estimators=10
        )
        model.fit()
        assert hasattr(model.model, "feature_importances_")
        assert len(model.model.feature_importances_) == 50

    def test_get_weights(self, small_adata):
        from modlyn.models import RandomForestImportance

        model = RandomForestImportance(
            adata=small_adata, label_column="cell_type", n_estimators=10
        )
        model.fit()
        weights_df = model.get_weights()
        assert weights_df.shape == (3, 50)  # 3 classes, 50 genes
        assert "method_name" in weights_df.attrs
        assert weights_df.attrs["method_name"] == "randomforest_importance"
        # All rows should be identical (RF gives global importance)
        assert np.allclose(weights_df.iloc[0], weights_df.iloc[1])

    def test_score(self, small_adata):
        from modlyn.models import RandomForestImportance

        model = RandomForestImportance(
            adata=small_adata, label_column="cell_type", n_estimators=10
        )
        model.fit()
        accuracy = model.score(small_adata)
        assert 0.0 <= accuracy <= 1.0
        # On training data with RF, should get decent accuracy
        assert accuracy > 0.5

    def test_fit_with_custom_adata(self, small_adata):
        """Test fitting with a different adata than initialization."""
        from modlyn.models import RandomForestImportance

        # Create a second adata
        rng = np.random.default_rng(43)
        X2 = rng.random((80, 50)).astype("float32")
        obs2 = pd.DataFrame({"cell_type": rng.choice(["A", "B", "C"], size=80)})
        var2 = pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
        adata2 = AnnData(X=X2, obs=obs2, var=var2)

        model = RandomForestImportance(adata=small_adata, label_column="cell_type")
        model.fit(adata=adata2)
        assert len(model.model.feature_importances_) == 50


class TestMutualInfoImportance:
    def test_initialization(self, small_adata):
        from modlyn.models import MutualInfoImportance

        model = MutualInfoImportance(
            adata=small_adata, label_column="cell_type", n_neighbors=5
        )
        assert model.n_neighbors == 5
        assert model.label_column == "cell_type"

    def test_fit(self, small_adata):
        from modlyn.models import MutualInfoImportance

        model = MutualInfoImportance(adata=small_adata, label_column="cell_type")
        model.fit()
        assert model.mi_scores is not None
        assert len(model.mi_scores) == 50
        # MI scores should be non-negative
        assert np.all(model.mi_scores >= 0)

    def test_get_weights(self, small_adata):
        from modlyn.models import MutualInfoImportance

        model = MutualInfoImportance(adata=small_adata, label_column="cell_type")
        model.fit()
        weights_df = model.get_weights()
        assert weights_df.shape == (3, 50)  # 3 classes, 50 genes
        assert "method_name" in weights_df.attrs
        assert weights_df.attrs["method_name"] == "mutual_info"
        # All rows should be identical (MI is global, not per-class)
        assert np.allclose(weights_df.iloc[0], weights_df.iloc[1])

    def test_get_weights_before_fit_raises(self, small_adata):
        from modlyn.models import MutualInfoImportance

        model = MutualInfoImportance(adata=small_adata, label_column="cell_type")
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.get_weights()

    def test_fit_with_custom_adata(self, small_adata):
        """Test fitting with a different adata than initialization."""
        from modlyn.models import MutualInfoImportance

        # Create a second adata
        rng = np.random.default_rng(44)
        X2 = rng.random((80, 50)).astype("float32")
        obs2 = pd.DataFrame({"cell_type": rng.choice(["A", "B", "C"], size=80)})
        var2 = pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
        adata2 = AnnData(X=X2, obs=obs2, var=var2)

        model = MutualInfoImportance(adata=small_adata, label_column="cell_type")
        model.fit(adata=adata2)
        assert len(model.mi_scores) == 50


class TestCrossMethodComparison:
    """Test that all methods produce comparable outputs."""

    def test_all_methods_return_consistent_format(self, small_adata):
        from modlyn.models import (
            ElasticNetLogReg,
            MutualInfoImportance,
            RandomForestImportance,
        )

        # Train all methods
        elasticnet = ElasticNetLogReg(adata=small_adata, label_column="cell_type")
        elasticnet.fit(
            adata_train=small_adata,
            adata_val=None,
            dataset_type="in-memory",
            train_dataloader_kwargs={"batch_size": 32},
            max_epochs=1,
            max_steps=5,
        )

        rf = RandomForestImportance(
            adata=small_adata, label_column="cell_type", n_estimators=10
        )
        rf.fit()

        mi = MutualInfoImportance(adata=small_adata, label_column="cell_type")
        mi.fit()

        # Get weights from all
        en_weights = elasticnet.get_weights()
        rf_weights = rf.get_weights()
        mi_weights = mi.get_weights()

        # All should have same shape
        assert en_weights.shape == rf_weights.shape == mi_weights.shape

        # All should have method_name attr
        assert "method_name" in en_weights.attrs
        assert "method_name" in rf_weights.attrs
        assert "method_name" in mi_weights.attrs

        # All should have same columns and index
        assert list(en_weights.columns) == list(rf_weights.columns)
        assert list(en_weights.index) == list(rf_weights.index)

    def test_methods_work_with_compare_scores(self, small_adata):
        """Ensure new methods work with the existing CompareScores evaluation."""
        from modlyn.eval import CompareScores
        from modlyn.models import (
            MutualInfoImportance,
            RandomForestImportance,
            SimpleLogReg,
        )

        # Train multiple methods
        simple = SimpleLogReg(adata=small_adata, label_column="cell_type")
        simple.fit(
            adata_train=small_adata,
            adata_val=None,
            dataset_type="in-memory",
            train_dataloader_kwargs={"batch_size": 32},
            max_epochs=1,
            max_steps=5,
        )

        rf = RandomForestImportance(
            adata=small_adata, label_column="cell_type", n_estimators=10
        )
        rf.fit()

        mi = MutualInfoImportance(adata=small_adata, label_column="cell_type")
        mi.fit()

        # Get weights
        dataframes = [simple.get_weights(), rf.get_weights(), mi.get_weights()]

        # Run comparison
        comparison = CompareScores(dataframes, n_top_values=[10, 20])
        results = comparison.compute_jaccard_comparison()

        assert results is not None
        assert len(results) > 0
        assert "jaccard" in results.columns
        assert "method_pair" in results.columns

