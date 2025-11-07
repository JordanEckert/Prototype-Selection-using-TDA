"""
Unit tests for the Bifiltration Prototype Selector
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons
from bifiltration_prototype_selector import BifiltrationPrototypeSelector


class TestBifiltrationPrototypeSelector:
    """Test suite for BifiltrationPrototypeSelector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=2,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def selector(self):
        """Create a default selector instance."""
        return BifiltrationPrototypeSelector(
            k_neighbors=3,
            homology_dimension=0,
            min_persistence=0.001,
            neighbor_quantile=0.2,
            radius_statistic='mean'
        )
    
    def test_initialization(self):
        """Test selector initialization with various parameters."""
        selector = BifiltrationPrototypeSelector()
        assert selector.k_neighbors == 5
        assert selector.homology_dimension == 0
        assert selector.min_persistence == 0.01
        assert selector.neighbor_quantile == 0.15
        assert selector.radius_statistic == 'mean'
        assert selector.metric == 'euclidean'
        
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = BifiltrationPrototypeSelector(
            k_neighbors=10,
            homology_dimension=1,
            min_persistence=0.1,
            neighbor_quantile=0.5,
            radius_statistic='median',
            metric='manhattan'
        )
        assert selector.k_neighbors == 10
        assert selector.homology_dimension == 1
        assert selector.min_persistence == 0.1
        assert selector.neighbor_quantile == 0.5
        assert selector.radius_statistic == 'median'
        assert selector.metric == 'manhattan'
    
    def test_fit_basic(self, selector, sample_data):
        """Test basic fitting functionality."""
        X, y = sample_data
        selector.fit(X, y, target_class=1)
        assert selector.prototypes_ is not None
        assert len(selector.prototypes_) > 0
        assert len(selector.prototypes_) < np.sum(y == 1)
    
    def test_fit_all_classes(self, selector, sample_data):
        """Test fitting for all classes."""
        X, y = sample_data
        all_prototypes = []
        
        for class_label in np.unique(y):
            selector.fit(X, y, target_class=class_label)
            assert selector.prototypes_ is not None
            all_prototypes.extend(selector.prototypes_)
        
        # Check that we have prototypes from both classes
        assert len(all_prototypes) > 0
        assert len(np.unique(all_prototypes)) <= len(all_prototypes)
    
    def test_get_prototypes(self, selector, sample_data):
        """Test prototype retrieval."""
        X, y = sample_data
        selector.fit(X, y, target_class=0)
        X_prototypes, indices = selector.get_prototypes(X)
        
        assert X_prototypes.shape[1] == X.shape[1]
        assert len(X_prototypes) == len(indices)
        assert np.array_equal(X_prototypes, X[indices])
    
    def test_get_prototypes_before_fit(self, selector, sample_data):
        """Test that get_prototypes raises error before fitting."""
        X, y = sample_data
        with pytest.raises(ValueError, match="Model has not been fitted"):
            selector.get_prototypes(X)
    
    def test_different_metrics(self, sample_data):
        """Test selector with different distance metrics."""
        X, y = sample_data
        metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
        
        for metric in metrics:
            selector = BifiltrationPrototypeSelector(
                k_neighbors=3,
                metric=metric
            )
            selector.fit(X, y, target_class=0)
            assert selector.prototypes_ is not None
            assert len(selector.prototypes_) > 0
    
    def test_summary_statistics(self, selector, sample_data):
        """Test summary statistics generation."""
        X, y = sample_data
        
        # Before fitting
        stats = selector.get_summary_statistics()
        assert stats is None
        
        # After fitting
        selector.fit(X, y, target_class=1)
        stats = selector.get_summary_statistics()
        
        assert stats is not None
        assert 'n_prototypes' in stats
        assert 'min_persistence' in stats
        assert 'homology_dimension' in stats
        assert 'k_neighbors' in stats
        assert 'neighbor_quantile' in stats
        assert 'radius_statistic' in stats
        assert stats['n_prototypes'] == len(selector.prototypes_)
    
    def test_sparse_data(self):
        """Test with sparse data."""
        from scipy.sparse import csr_matrix
        
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_redundant=0,
            n_informative=10,
            random_state=42
        )
        
        X_sparse = csr_matrix(X)
        selector = BifiltrationPrototypeSelector(k_neighbors=3)
        selector.fit(X_sparse, y, target_class=0)
        
        assert selector.prototypes_ is not None
        assert len(selector.prototypes_) > 0
    
    def test_imbalanced_data(self):
        """Test with imbalanced dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            weights=[0.9, 0.1],  # 90-10 split
            random_state=42
        )
        
        selector = BifiltrationPrototypeSelector(k_neighbors=3)
        
        # Get prototypes for minority class
        selector.fit(X, y, target_class=1)
        minority_prototypes = selector.prototypes_
        
        # Get prototypes for majority class  
        selector.fit(X, y, target_class=0)
        majority_prototypes = selector.prototypes_
        
        # Check that we have prototypes for both
        assert len(minority_prototypes) > 0
        assert len(majority_prototypes) > 0
        
        # Typically, minority class should have fewer prototypes
        # (though not always guaranteed)
        assert len(minority_prototypes) <= len(majority_prototypes) * 2
    
    def test_single_class_error(self):
        """Test error handling when only one class is present."""
        X = np.random.randn(100, 2)
        y = np.ones(100)  # Only one class
        
        selector = BifiltrationPrototypeSelector(k_neighbors=3)
        
        with pytest.raises(ValueError, match="No samples found for non-target"):
            selector.fit(X, y, target_class=1)
    
    def test_custom_metric_function(self, sample_data):
        """Test with custom metric function."""
        X, y = sample_data
        
        def custom_metric(X1, X2):
            # Simple custom metric: sum of absolute differences
            return np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]).sum(axis=2)
        
        selector = BifiltrationPrototypeSelector(
            k_neighbors=3,
            metric=custom_metric
        )
        selector.fit(X, y, target_class=0)
        
        assert selector.prototypes_ is not None
        assert len(selector.prototypes_) > 0
    
    def test_reduction_percentage(self, selector, sample_data):
        """Test that reduction percentage is reasonable."""
        X, y = sample_data
        target_class = 0
        n_target = np.sum(y == target_class)
        
        selector.fit(X, y, target_class=target_class)
        n_prototypes = len(selector.prototypes_)
        reduction_pct = (1 - n_prototypes / n_target) * 100
        
        # Should achieve some reduction
        assert reduction_pct > 0
        # But not eliminate everything
        assert reduction_pct < 100
        # Typically between 50-90% reduction
        assert 20 <= reduction_pct <= 95
    
    def test_deterministic_results(self, sample_data):
        """Test that results are deterministic for same input."""
        X, y = sample_data
        
        selector1 = BifiltrationPrototypeSelector(k_neighbors=3)
        selector2 = BifiltrationPrototypeSelector(k_neighbors=3)
        
        selector1.fit(X, y, target_class=0)
        selector2.fit(X, y, target_class=0)
        
        # Results should be identical
        assert np.array_equal(selector1.prototypes_, selector2.prototypes_)
    
    def test_different_homology_dimensions(self, sample_data):
        """Test with different homology dimensions."""
        X, y = sample_data
        
        # H0 (connected components)
        selector_h0 = BifiltrationPrototypeSelector(
            homology_dimension=0,
            k_neighbors=3
        )
        selector_h0.fit(X, y, target_class=0)
        n_proto_h0 = len(selector_h0.prototypes_)
        
        # H1 (loops) - typically selects more points
        selector_h1 = BifiltrationPrototypeSelector(
            homology_dimension=1,
            k_neighbors=3
        )
        selector_h1.fit(X, y, target_class=0)
        n_proto_h1 = len(selector_h1.prototypes_)
        
        assert n_proto_h0 > 0
        assert n_proto_h1 > 0
        # H1 might select different number of points
        # (not necessarily more, depends on topology)
    
    def test_quantile_effect(self, sample_data):
        """Test effect of neighbor_quantile parameter."""
        X, y = sample_data
        
        # Lower quantile = more aggressive reduction
        selector_low = BifiltrationPrototypeSelector(
            neighbor_quantile=0.1,
            k_neighbors=3
        )
        selector_low.fit(X, y, target_class=0)
        n_proto_low = len(selector_low.prototypes_)
        
        # Higher quantile = less aggressive reduction
        selector_high = BifiltrationPrototypeSelector(
            neighbor_quantile=0.5,
            k_neighbors=3
        )
        selector_high.fit(X, y, target_class=0)
        n_proto_high = len(selector_high.prototypes_)
        
        # Generally, higher quantile should give more prototypes
        # (though not strictly guaranteed due to topology)
        assert n_proto_low > 0
        assert n_proto_high > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
