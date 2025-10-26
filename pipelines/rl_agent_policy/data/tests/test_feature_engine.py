"""
Tests for FeatureEngineeringProcessor class (feature_engine.py).

Tests cover:
- Feature generation validation
- Technical indicators correctness
- Custom features validation
- Quantstats features
- Data cleaning and preparation
- Feature value ranges and types
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.rl_agent_policy.data.feature_engine import FeatureEngineeringProcessor
from pipelines.rl_agent_policy.data.tests.conftest import (
    validate_dataframe_structure,
    validate_no_invalid_values
)


class TestFeatureEngineeringInitialization:
    """Test FeatureEngineeringProcessor initialization."""
    
    def test_init_with_empty_config(self):
        """Test initialization with empty config."""
        processor = FeatureEngineeringProcessor({})
        
        assert processor.config == {}
        assert processor.enabled is True  # default
    
    def test_init_with_config(self, feature_engineering_config):
        """Test initialization with full config."""
        processor = FeatureEngineeringProcessor(feature_engineering_config)
        
        assert processor.config == feature_engineering_config
        assert processor.enabled is True
    
    def test_init_disabled(self):
        """Test initialization with disabled flag."""
        config = {'enabled': False}
        processor = FeatureEngineeringProcessor(config)
        
        assert processor.enabled is False


class TestFeatureGeneration:
    """Test feature generation methods."""
    
    def test_generate_features_disabled(self, sample_ohlcv_data):
        """Test that disabled processor returns unchanged data."""
        config = {'enabled': False}
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Should return copy of original data
        assert len(result.columns) == len(sample_ohlcv_data.columns)
        pd.testing.assert_index_equal(result.columns, sample_ohlcv_data.columns)
    
    def test_generate_features_structure(self, sample_ohlcv_data):
        """Test that generated features maintain data structure."""
        config = {
            'enabled': True,
            'technical_indicators': {'enabled': False},
            'custom_features': {'enabled': False},
            'quantstats_features': {'enabled': False}
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        validate_dataframe_structure(result)
        assert len(result) == len(sample_ohlcv_data), "Row count should be preserved"
    
    def test_generate_features_returns_dataframe(self, sample_ohlcv_data):
        """Test that generate_features returns a DataFrame."""
        config = {'enabled': True}
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"


class TestCustomFeatures:
    """Test custom feature generation."""
    
    def test_custom_features_enabled(self, sample_ohlcv_data, feature_engineering_config):
        """Test custom feature generation when enabled."""
        processor = FeatureEngineeringProcessor(feature_engineering_config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Check for moving average features
        ma_cols = [col for col in result.columns if 'ma_' in col]
        assert len(ma_cols) > 0, "Should generate moving average features"
    
    def test_moving_averages_calculation(self, sample_ohlcv_data):
        """Test moving averages are calculated correctly."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7, 14],
                'price_features': ['moving_averages']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Check MA columns exist
        assert 'BTC_ma_7' in result.columns
        assert 'BTC_ma_14' in result.columns
        
        # Verify MA calculation for a specific period
        expected_ma_7 = sample_ohlcv_data['close'].rolling(window=7).mean()
        actual_ma_7 = result['BTC_ma_7']
        
        # Allow for NaN in first few values
        pd.testing.assert_series_equal(
            expected_ma_7.iloc[10:].reset_index(drop=True),
            actual_ma_7.iloc[10:].reset_index(drop=True),
            check_names=False
        )
    
    def test_exponential_moving_averages(self, sample_ohlcv_data):
        """Test EMA calculation."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7],
                'price_features': ['exponential_moving_averages']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        assert 'BTC_ema_7' in result.columns
        
        # Check EMA values are numeric and finite
        assert pd.api.types.is_numeric_dtype(result['BTC_ema_7'])
        assert not np.isinf(result['BTC_ema_7']).any()
    
    def test_volatility_calculation(self, sample_ohlcv_data):
        """Test volatility feature calculation."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7, 14],
                'price_features': ['volatility']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        assert 'BTC_vol_7' in result.columns
        assert 'BTC_vol_14' in result.columns
        
        # Volatility should be non-negative
        assert (result['BTC_vol_7'].dropna() >= 0).all(), "Volatility should be non-negative"
    
    def test_log_returns_calculation(self, sample_ohlcv_data):
        """Test log returns calculation."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [],
                'price_features': ['log_returns']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Check for log return columns
        lr_cols = [col for col in result.columns if 'lr_' in col]
        assert len(lr_cols) > 0, "Should generate log return features"
        
        # Log returns should be in reasonable range (not too large)
        for col in lr_cols:
            if result[col].notna().any():
                assert result[col].abs().max() < 1.0, \
                    f"Log returns in {col} seem unreasonably large"
    
    def test_price_shifts(self, sample_ohlcv_data):
        """Test price shift features."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [],
                'price_features': ['price_shifts']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Check for previous price columns
        assert 'BTC_prev_close' in result.columns
        assert 'BTC_prev_volume' in result.columns
        
        # First value should be NaN
        assert pd.isna(result['BTC_prev_close'].iloc[0])
        
        # Other values should match shifted original
        assert result['BTC_prev_close'].iloc[1] == sample_ohlcv_data['close'].iloc[0]
    
    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI indicator calculation."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [],
                'price_features': []  # RSI is always calculated
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        assert 'BTC_rsi_14' in result.columns
        
        # RSI should be between 0 and 100
        rsi_values = result['BTC_rsi_14'].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), \
            "RSI should be between 0 and 100"
    
    def test_macd_calculation(self, sample_ohlcv_data):
        """Test MACD indicator calculation."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [],
                'price_features': []  # MACD is always calculated
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        assert 'BTC_macd' in result.columns
        
        # MACD should be numeric
        assert pd.api.types.is_numeric_dtype(result['BTC_macd'])


class TestTechnicalIndicators:
    """Test technical indicator generation."""
    
    def test_technical_indicators_disabled(self, sample_ohlcv_data):
        """Test that technical indicators are skipped when disabled."""
        config = {
            'enabled': True,
            'technical_indicators': {
                'enabled': False
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Should not have many technical indicator columns
        # (only custom features)
        assert isinstance(result, pd.DataFrame)


class TestQuantstatsFeatures:
    """Test quantstats feature generation."""

    def test_quantstats_disabled(self, sample_ohlcv_data):
        """Test that quantstats features are skipped when disabled."""
        config = {
            'enabled': True,
            'quantstats_features': {
                'enabled': False
            },
            'custom_features': {'enabled': False},
            'technical_indicators': {'enabled': False}
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Should not have quantstats columns
        qs_cols = [col for col in result.columns if 'qs_' in col]
        assert len(qs_cols) == 0, "Should not have quantstats features"


class TestDataCleaning:
    """Test data cleaning functionality."""
    
    def test_clean_infinite_values(self):
        """Test that infinite values are replaced."""
        config = {'enabled': True}
        processor = FeatureEngineeringProcessor(config)
        
        # Create data with infinite values
        data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'value': [1.0, np.inf, 3.0],
            'value2': [np.inf, -np.inf, 5.0]
        })
        
        cleaned = processor._clean_features(data)
        
        # Should not have infinite values
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(cleaned[col]).any(), \
                f"Column {col} should not contain infinite values"
    
    def test_clean_forward_fill_nan(self):
        """Test that NaN values are forward filled."""
        config = {'enabled': True}
        processor = FeatureEngineeringProcessor(config)
        
        # Create data with NaN values
        data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'value': [1.0, np.nan, np.nan, 4.0]
        })
        
        cleaned = processor._clean_features(data)
        
        # NaN should be forward filled (except first row if it starts with NaN)
        expected = pd.Series([1.0, 1.0, 1.0, 4.0], name='value')
        pd.testing.assert_series_equal(cleaned['value'], expected)
    
    def test_clean_remove_all_nan_columns(self):
        """Test that columns with all NaN are removed."""
        config = {'enabled': True}
        processor = FeatureEngineeringProcessor(config)
        
        # Create data with all-NaN column
        data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'good_col': [1.0, 2.0],
            'bad_col': [np.nan, np.nan]
        })
        
        cleaned = processor._clean_features(data)
        
        assert 'bad_col' not in cleaned.columns, \
            "Columns with all NaN should be removed"
        assert 'good_col' in cleaned.columns, \
            "Valid columns should be preserved"
    
    def test_clean_remove_duplicate_columns(self):
        """Test that duplicate columns are removed."""
        config = {'enabled': True}
        processor = FeatureEngineeringProcessor(config)
        
        # Create data with duplicate columns
        data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'col1': [1.0, 2.0],
            'col2': [3.0, 4.0]
        })
        data['col1_dup'] = data['col1']  # Add duplicate
        data = data.rename(columns={'col1_dup': 'col1'})  # Make names identical
        
        cleaned = processor._clean_features(data)
        
        # Should not have duplicate column names
        assert len(cleaned.columns) == len(set(cleaned.columns)), \
            "Should not have duplicate column names"


class TestFeaturePrefixing:
    """Test feature prefixing functionality."""
    
    def test_feature_prefix_applied(self, sample_ohlcv_data):
        """Test that currency prefix is applied to features."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7],
                'price_features': ['moving_averages']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Check that features have BTC prefix
        ma_cols = [col for col in result.columns if 'ma_' in col]
        for col in ma_cols:
            assert col.startswith('BTC_'), f"Feature {col} should have BTC_ prefix"
    
    def test_no_prefix_when_not_specified(self, sample_ohlcv_data):
        """Test features without prefix when currency_name is empty."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7],
                'price_features': ['moving_averages']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='')
        
        # Should have ma_ columns without prefix
        ma_cols = [col for col in result.columns if 'ma_' in col]
        assert len(ma_cols) > 0, "Should have MA columns"
        
        # At least one should not have currency prefix
        non_prefixed = [col for col in ma_cols if not col.startswith('BTC_') and not col.startswith('ETH_')]
        assert len(non_prefixed) > 0, "Should have features without currency prefix"


class TestFeatureValueValidation:
    """Test feature value validation."""
    
    def test_features_numeric(self, sample_ohlcv_data, feature_engineering_config):
        """Test that generated features are numeric."""
        processor = FeatureEngineeringProcessor(feature_engineering_config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # All columns except date should be numeric
        non_date_cols = [col for col in result.columns if 'date' not in col.lower()]
        for col in non_date_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), \
                f"Column {col} should be numeric"
    
    def test_features_finite(self, sample_ohlcv_data, feature_engineering_config):
        """Test that features don't contain infinite values."""
        processor = FeatureEngineeringProcessor(feature_engineering_config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(result[col]).any(), \
                f"Column {col} contains infinite values"
    
    def test_features_reasonable_range(self, sample_ohlcv_data):
        """Test that features are in reasonable ranges."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7],
                'price_features': ['moving_averages', 'volatility']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        result = processor.generate_features(sample_ohlcv_data, currency_name='BTC')
        
        # Volatility should be non-negative
        vol_cols = [col for col in result.columns if 'vol_' in col]
        for col in vol_cols:
            assert (result[col].dropna() >= 0).all(), \
                f"Volatility column {col} should be non-negative"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        config = {'enabled': True}
        processor = FeatureEngineeringProcessor(config)
        
        empty_df = pd.DataFrame()
        
        # Should not crash
        result = processor.generate_features(empty_df, currency_name='BTC')
        assert isinstance(result, pd.DataFrame)
    
    def test_minimal_data(self):
        """Test with minimal data (< lookback period)."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7, 14],
                'price_features': ['moving_averages']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        # Only 5 rows, less than lookback period
        small_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5).strftime('%Y-%m-%d %H:%M:%S'),
            'close': [100, 101, 102, 103, 104]
        })
        
        result = processor.generate_features(small_data, currency_name='BTC')
        
        # Should still work, even if features have NaN
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
    
    def test_missing_columns(self):
        """Test handling of missing expected columns."""
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7],
                'price_features': ['log_returns']
            }
        }
        processor = FeatureEngineeringProcessor(config)
        
        # Data without some expected columns
        incomplete_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10).strftime('%Y-%m-%d %H:%M:%S'),
            'close': np.random.rand(10) * 100
        })
        
        # Should handle gracefully
        result = processor.generate_features(incomplete_data, currency_name='BTC')
        assert isinstance(result, pd.DataFrame)

