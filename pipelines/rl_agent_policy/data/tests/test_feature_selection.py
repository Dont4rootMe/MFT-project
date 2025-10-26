"""
Tests for FeatureSelector class (feature_selection.py).

Tests cover:
- Feature selection methods validation
- Variance threshold filtering
- Correlation filtering
- PCA reduction
- Top-K feature selection
- Per-currency and post-merge selection
- Feature count validation
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.rl_agent_policy.data.feature_selection import FeatureSelector
from pipelines.rl_agent_policy.data.tests.conftest import validate_dataframe_structure


class TestFeatureSelectorInitialization:
    """Test FeatureSelector initialization."""
    
    def test_init_with_empty_config(self):
        """Test initialization with empty config."""
        selector = FeatureSelector(None)
        
        assert selector.config == {}
        assert selector.per_cfg == {}
        assert selector.post_cfg == {}
    
    def test_init_with_config(self, feature_selection_config):
        """Test initialization with full config."""
        selector = FeatureSelector(feature_selection_config)
        
        assert selector.config == feature_selection_config
        assert 'per_currency' in selector.per_cfg or selector.per_cfg == feature_selection_config.get('per_currency', {})
        assert 'post_merge' in selector.post_cfg or selector.post_cfg == feature_selection_config.get('post_merge', {})


class TestPerCurrencySelection:
    """Test per-currency feature selection."""
    
    def test_per_currency_disabled(self, sample_processed_data):
        """Test that disabled selector returns unchanged data."""
        config = {
            'per_currency': {
                'enabled': False
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(sample_processed_data)
        
        # Should return same number of columns
        assert result.shape[1] == sample_processed_data.shape[1]
    
    def test_per_currency_enabled_reduces_features(self, sample_processed_data):
        """Test that enabled selector reduces features."""
        # Add many correlated features
        df = sample_processed_data.copy()
        for i in range(20):
            df[f'feature_{i}'] = df['close'] + np.random.randn(len(df)) * 0.01
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'variance_corr',
                'variance_threshold': 0.01,
                'correlation_threshold': 0.95,
                'max_features': 10
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should have fewer columns (base columns + at most max_features)
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        core_cols = [c for c in df.columns if any(k in c.lower() for k in ['open', 'high', 'low', 'close', 'volume'])]
        max_expected = len(date_cols) + len(core_cols) + 10
        
        assert result.shape[1] <= max_expected, \
            f"Should reduce features to at most {max_expected}, got {result.shape[1]}"


class TestVarianceCorrelationFilter:
    """Test variance and correlation filtering."""
    
    def test_correlation_filtering(self):
        """Test that highly correlated features are removed."""
        # Create data with correlated features
        np.random.seed(42)
        base = np.random.randn(100)
        
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': base * 100,
            'feature_1': base + np.random.randn(100) * 0.01,  # Highly correlated
            'feature_2': base + np.random.randn(100) * 0.01,  # Highly correlated
            'feature_3': np.random.randn(100) * 50  # Uncorrelated
        })
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'variance_corr',
                'variance_threshold': 0.0,
                'correlation_threshold': 0.95
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should remove one of the correlated features
        feature_count = sum(1 for col in result.columns if col.startswith('feature_'))
        assert feature_count < 3, "Should remove at least one correlated feature"
    
    def test_max_features_limit(self):
        """Test that max_features limit is respected."""
        # Create data with many features
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100)
        })
        
        # Add 50 uncorrelated features
        for i in range(50):
            df[f'feature_{i}'] = np.random.randn(100) * (i + 1)
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'variance_corr',
                'variance_threshold': 0.0,
                'correlation_threshold': 1.0,  # Don't filter by correlation
                'max_features': 10
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Count feature columns (excluding date and core columns)
        feature_cols = [c for c in result.columns if c.startswith('feature_')]
        assert len(feature_cols) <= 10, \
            f"Should have at most 10 features, got {len(feature_cols)}"


class TestPCAReduction:
    """Test PCA dimensionality reduction."""
    
    def test_pca_reduction_enabled(self):
        """Test PCA reduction when enabled."""
        # Create data with many features
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100) * 100
        })
        
        # Add 20 features
        for i in range(20):
            df[f'feature_{i}'] = np.random.randn(100) * (i + 1)
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'pca',
                'n_components': 5
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should have PCA components
        pca_cols = [c for c in result.columns if c.startswith('pca_')]
        assert len(pca_cols) == 5, f"Should have 5 PCA components, got {len(pca_cols)}"
    
    def test_pca_preserves_base_columns(self):
        """Test that PCA preserves date and core columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'open': np.random.randn(100) * 100,
            'high': np.random.randn(100) * 100,
            'low': np.random.randn(100) * 100,
            'close': np.random.randn(100) * 100,
            'volume': np.random.randint(1000, 10000, 100).astype(float)
        })
        
        # Add features
        for i in range(10):
            df[f'feature_{i}'] = np.random.randn(100)
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'pca',
                'n_components': 3
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should preserve base columns
        assert 'date' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
    
    def test_pca_components_limit(self):
        """Test that PCA respects component limit when fewer features exist."""
        # Create data with few features
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100),
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'pca',
                'n_components': 10  # More than available features
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should have at most 2 components (number of features)
        pca_cols = [c for c in result.columns if c.startswith('pca_')]
        assert len(pca_cols) <= 2, \
            f"Should have at most 2 PCA components, got {len(pca_cols)}"


class TestTopKSelection:
    """Test top-K feature selection."""
    
    def test_topk_selection(self):
        """Test top-K feature selection by variance."""
        # Create features with different variances
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100) * 100
        })
        
        # Add features with controlled variance
        for i in range(10):
            variance = (i + 1) * 10
            df[f'feature_{i}'] = np.random.randn(100) * variance
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'topk',
                'k': 5
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should have exactly 5 features (excluding date and close)
        feature_cols = [c for c in result.columns if c.startswith('feature_')]
        assert len(feature_cols) == 5, \
            f"Should have exactly 5 features, got {len(feature_cols)}"

    
    def test_topk_with_max_features_alias(self):
        """Test that max_features works as alias for k."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100)
        })
        
        for i in range(10):
            df[f'feature_{i}'] = np.random.randn(100) * (i + 1)
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'topk',
                'max_features': 3  # Using max_features instead of k
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        feature_cols = [c for c in result.columns if c.startswith('feature_')]
        assert len(feature_cols) == 3, "Should respect max_features parameter"


class TestPostMergeSelection:
    """Test post-merge feature selection."""
    
    def test_post_merge_disabled(self, sample_processed_data):
        """Test that disabled post-merge returns unchanged data."""
        config = {
            'post_merge': {
                'enabled': False
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_post_merge(sample_processed_data)
        
        # Should return same data
        assert result.shape == sample_processed_data.shape
        pd.testing.assert_frame_equal(result, sample_processed_data)
    
    def test_post_merge_not_implemented(self, sample_processed_data):
        """Test that post-merge methods are not yet implemented."""
        config = {
            'post_merge': {
                'enabled': True,
                'method': 'some_method'
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_post_merge(sample_processed_data)
        
        # Should return unchanged data with warning
        assert result.shape == sample_processed_data.shape


class TestBaseColumnsIdentification:
    """Test identification of base columns."""
    
    def test_get_base_columns_date(self):
        """Test identification of date columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'Date': pd.date_range('2023-01-01', periods=10),
            'close': np.random.randn(10)
        })
        
        selector = FeatureSelector({})
        date_cols, core_cols, feature_cols = selector._get_base_columns(df)
        
        assert len(date_cols) >= 1, "Should identify date columns"
        assert 'date' in date_cols or 'Date' in date_cols
    
    def test_get_base_columns_core(self):
        """Test identification of core OHLCV columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': np.random.randn(10),
            'high': np.random.randn(10),
            'low': np.random.randn(10),
            'close': np.random.randn(10),
            'volume': np.random.randint(1000, 10000, 10).astype(float)
        })
        
        selector = FeatureSelector({})
        date_cols, core_cols, feature_cols = selector._get_base_columns(df)
        
        assert 'open' in core_cols
        assert 'high' in core_cols
        assert 'low' in core_cols
        assert 'close' in core_cols
        assert 'volume' in core_cols
    
    def test_get_base_columns_features(self):
        """Test identification of feature columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'close': np.random.randn(10),
            'feature_1': np.random.randn(10),
            'feature_2': np.random.randn(10),
            'ma_7': np.random.randn(10)
        })
        
        selector = FeatureSelector({})
        date_cols, core_cols, feature_cols = selector._get_base_columns(df)
        
        assert 'feature_1' in feature_cols
        assert 'feature_2' in feature_cols
        assert 'ma_7' in feature_cols
        assert 'close' not in feature_cols  # Should be in core_cols


class TestDataIntegrity:
    """Test data integrity after selection."""
    
    def test_selection_preserves_row_count(self):
        """Test that selection preserves number of rows."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100)
        })
        
        for i in range(20):
            df[f'feature_{i}'] = np.random.randn(100)
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'topk',
                'k': 5
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        assert len(result) == len(df), "Row count should be preserved"
    
    def test_selection_preserves_date_column(self):
        """Test that date column is preserved."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100)
        })
        
        for i in range(10):
            df[f'feature_{i}'] = np.random.randn(100)
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'variance_corr',
                'variance_threshold': 0.0,
                'correlation_threshold': 0.95
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        assert 'date' in result.columns, "Date column should be preserved"
        pd.testing.assert_series_equal(result['date'], df['date'])
    
    def test_selection_no_invalid_values(self):
        """Test that selection doesn't introduce invalid values."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100) * 100
        })
        
        for i in range(10):
            df[f'feature_{i}'] = np.random.randn(100) * 10
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'variance_corr',
                'variance_threshold': 0.01,
                'correlation_threshold': 0.95
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Check for inf values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(result[col]).any(), \
                f"Column {col} should not contain infinite values"


class TestUnknownMethods:
    """Test handling of unknown selection methods."""
    
    def test_unknown_per_currency_method(self, sample_processed_data):
        """Test that unknown method returns unchanged data with warning."""
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'unknown_method'
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(sample_processed_data)
        
        # Should return unchanged data
        pd.testing.assert_frame_equal(result, sample_processed_data)


class TestEdgeCases:
    """Test edge cases in feature selection."""
    
    def test_selection_with_no_features(self):
        """Test selection when there are no feature columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'close': np.random.randn(10)
        })
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'variance_corr',
                'variance_threshold': 0.0
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should return base columns unchanged
        assert 'date' in result.columns
        assert 'close' in result.columns
    
    
    def test_topk_with_k_larger_than_features(self):
        """Test top-K when K is larger than number of features."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100),
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        
        config = {
            'per_currency': {
                'enabled': True,
                'method': 'topk',
                'k': 100  # More than available features
            }
        }
        selector = FeatureSelector(config)
        
        result = selector.select_per_currency(df)
        
        # Should keep all features
        feature_cols = [c for c in result.columns if c.startswith('feature_')]
        assert len(feature_cols) == 2, "Should keep all available features"

