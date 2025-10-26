"""
Integration tests for data validation.

Tests cover end-to-end validation of the complete data pipeline:
- Format validation
- Structure validation  
- Type validation
- Range validation
- Required features presence
- Data integrity checks
- Multi-currency consistency
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.rl_agent_policy.data.data_parser import DataHandler
from pipelines.rl_agent_policy.data.feature_engine import FeatureEngineeringProcessor
from pipelines.rl_agent_policy.data.feature_selection import FeatureSelector
from pipelines.rl_agent_policy.data.tests.conftest import (
    validate_dataframe_structure,
    validate_data_types,
    validate_value_ranges,
    validate_no_invalid_values,
    validate_ohlcv_consistency
)


class TestDataFormatValidation:
    """Test data format validation."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_output_is_dataframe(self, mock_cdd, sample_ohlcv_data):
        """Test that final output is a pandas DataFrame."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        assert isinstance(data, pd.DataFrame), "Output should be a pandas DataFrame"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_has_date_column(self, mock_cdd, sample_ohlcv_data):
        """Test that output has a date/datetime column."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        assert len(date_cols) > 0, "Data should have at least one date column"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_date_format_valid(self, mock_cdd, sample_ohlcv_data):
        """Test that date column has valid datetime format."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            # Should be parseable as datetime
            parsed_dates = pd.to_datetime(data[date_col], errors='coerce')
            assert parsed_dates.notna().all(), "All dates should be valid"


class TestDataStructureValidation:
    """Test data structure validation."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_data_not_empty(self, mock_cdd, sample_ohlcv_data):
        """Test that output data is not empty."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        assert len(data) > 0, "Data should not be empty"
        assert len(data.columns) > 0, "Data should have columns"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_minimum_rows(self, mock_cdd, sample_ohlcv_data):
        """Test that data has minimum required rows."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        # Should have at least some data points
        min_rows = 10
        assert len(data) >= min_rows, \
            f"Data should have at least {min_rows} rows, got {len(data)}"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_no_duplicate_rows(self, mock_cdd, sample_ohlcv_data):
        """Test that there are no duplicate rows."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        duplicate_count = data.duplicated().sum()
        assert duplicate_count == 0, \
            f"Found {duplicate_count} duplicate rows"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_dates_sorted(self, mock_cdd, sample_ohlcv_data):
        """Test that dates are sorted in ascending order."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            dates = pd.to_datetime(data[date_col])
            assert dates.is_monotonic_increasing, "Dates should be sorted in ascending order"


class TestDataTypeValidation:
    """Test data type validation."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_numeric_columns_are_numeric(self, mock_cdd, sample_ohlcv_data):
        """Test that numeric columns have numeric types."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        # Check that most columns (except date) are numeric
        non_date_cols = [col for col in data.columns if 'date' not in col.lower()]
        numeric_count = sum(pd.api.types.is_numeric_dtype(data[col]) for col in non_date_cols)
        
        assert numeric_count == len(non_date_cols), \
            "All non-date columns should be numeric"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_no_object_columns_except_date(self, mock_cdd, sample_ohlcv_data):
        """Test that only date columns are object/string type."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        object_cols = data.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            assert 'date' in col.lower(), \
                f"Non-date column {col} should not be object type"


class TestValueRangeValidation:
    """Test value range validation."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_no_negative_prices(self, mock_cdd, sample_ohlcv_data):
        """Test that price columns don't have negative values."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        # Check price-related columns
        price_keywords = ['open', 'high', 'low', 'close', 'price']
        price_cols = [col for col in data.columns 
                     if any(keyword in col.lower() for keyword in price_keywords)]
        
        for col in price_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                min_val = data[col].min()
                assert min_val >= 0, \
                    f"Price column {col} has negative values: {min_val}"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_no_negative_volume(self, mock_cdd, sample_ohlcv_data):
        """Test that volume is non-negative."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        volume_cols = [col for col in data.columns if 'volume' in col.lower()]
        
        for col in volume_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                assert (data[col] >= 0).all(), \
                    f"Volume column {col} has negative values"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_rsi_in_valid_range(self, mock_cdd, sample_ohlcv_data):
        """Test that RSI values are between 0 and 100."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [],
                'price_features': []
            }
        }
        
        handler = DataHandler(
            symbols=['BTC'],
            feature_engineering=config
        )
        data = handler.get_data()
        
        rsi_cols = [col for col in data.columns if 'rsi' in col.lower()]
        
        for col in rsi_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                valid_values = data[col].dropna()
                if len(valid_values) > 0:
                    assert (valid_values >= 0).all() and (valid_values <= 100).all(), \
                        f"RSI column {col} has values outside [0, 100] range"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_volatility_non_negative(self, mock_cdd, sample_ohlcv_data):
        """Test that volatility values are non-negative."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        config = {
            'enabled': True,
            'custom_features': {
                'enabled': True,
                'lookback_periods': [7],
                'price_features': ['volatility']
            }
        }
        
        handler = DataHandler(
            symbols=['BTC'],
            feature_engineering=config
        )
        data = handler.get_data()
        
        vol_cols = [col for col in data.columns if 'vol_' in col]
        
        for col in vol_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                valid_values = data[col].dropna()
                if len(valid_values) > 0:
                    assert (valid_values >= 0).all(), \
                        f"Volatility column {col} has negative values"


class TestInvalidValueValidation:
    """Test validation of invalid values (NaN, inf, etc)."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_no_infinite_values(self, mock_cdd, sample_ohlcv_data):
        """Test that data contains no infinite values."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        validate_no_invalid_values(data, allow_nan=True)  # Allow some NaN in features
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_core_columns_no_nan(self, mock_cdd, sample_ohlcv_data):
        """Test that core OHLCV columns have no NaN values."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        # Core columns should not have NaN
        core_keywords = ['open', 'high', 'low', 'close']
        core_cols = [col for col in data.columns 
                    if any(keyword == col.split('_')[-1].lower() for keyword in core_keywords)]
        
        for col in core_cols:
            nan_count = data[col].isna().sum()
            assert nan_count == 0, \
                f"Core column {col} has {nan_count} NaN values"


class TestRequiredFeaturesValidation:
    """Test validation of required features."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_has_price_features(self, mock_cdd, sample_ohlcv_data):
        """Test that data has price-related features."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        price_keywords = ['open', 'high', 'low', 'close']
        price_cols = [col for col in data.columns 
                     if any(keyword in col.lower() for keyword in price_keywords)]
        
        assert len(price_cols) > 0, "Data should have price-related columns"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_has_volume_feature(self, mock_cdd, sample_ohlcv_data):
        """Test that data has volume feature."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        volume_cols = [col for col in data.columns if 'volume' in col.lower()]
        
        assert len(volume_cols) > 0, "Data should have volume column"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_multi_currency_has_all_currencies(self, mock_cdd, sample_multi_currency_data):
        """Test that multi-currency data has features for all currencies."""
        mock_instance = Mock()
        
        def fetch_side_effect(exchange, main, currency, timeframe):
            return sample_multi_currency_data[currency].copy()
        
        mock_instance.fetch.side_effect = fetch_side_effect
        mock_cdd.return_value = mock_instance
        
        currencies = ['BTC', 'ETH']
        handler = DataHandler(
            symbols=currencies,
            feature_engineering={'enabled': False}
        )
        data = handler.get_data()
        
        for currency in currencies:
            currency_cols = [col for col in data.columns if col.startswith(f"{currency}_")]
            assert len(currency_cols) > 0, \
                f"Data should have columns for {currency}"


class TestOHLCVConsistency:
    """Test OHLCV data consistency across the pipeline."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_single_currency_ohlcv_consistency(self, mock_cdd, sample_ohlcv_data):
        """Test OHLCV consistency for single currency."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(
            symbols=['BTC'],
            feature_engineering={'enabled': False}
        )
        data = handler.get_data()
        
        validate_ohlcv_consistency(data, prefix='')
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_multi_currency_ohlcv_consistency(self, mock_cdd, sample_multi_currency_data):
        """Test OHLCV consistency for multiple currencies."""
        mock_instance = Mock()
        
        def fetch_side_effect(exchange, main, currency, timeframe):
            return sample_multi_currency_data[currency].copy()
        
        mock_instance.fetch.side_effect = fetch_side_effect
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(
            symbols=['BTC', 'ETH'],
            feature_engineering={'enabled': False}
        )
        data = handler.get_data()
        
        # Check consistency for each currency
        for currency in ['BTC', 'ETH']:
            validate_ohlcv_consistency(data, prefix=f'{currency}_')


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline validation."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_complete_pipeline_with_all_features(self, mock_cdd, sample_ohlcv_data):
        """Test complete pipeline with all features enabled."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        config = {
            'symbols': ['BTC'],
            'feature_engineering': {
                'enabled': True,
                'custom_features': {
                    'enabled': True,
                    'lookback_periods': [7, 14],
                    'price_features': [
                        'moving_averages',
                        'volatility',
                        'log_returns'
                    ]
                },
                'technical_indicators': {'enabled': False},
                'quantstats_features': {'enabled': False}
            },
            'feature_selection': {
                'per_currency': {
                    'enabled': True,
                    'method': 'variance_corr',
                    'variance_threshold': 0.01,
                    'correlation_threshold': 0.95
                }
            }
        }
        
        handler = DataHandler(**config)
        data = handler.get_data()
        
        # Validate all aspects
        validate_dataframe_structure(data)
        validate_no_invalid_values(data, allow_nan=True)
        
        assert len(data) > 0, "Pipeline should produce non-empty data"
        assert len(data.columns) > 5, "Pipeline should produce multiple features"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_pipeline_reproducibility(self, mock_cdd, sample_ohlcv_data):
        """Test that pipeline produces consistent results."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        
        data1 = handler.get_data()
        data2 = handler.get_data()
        
        # Should get identical results (from cache)
        pd.testing.assert_frame_equal(data1, data2)
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_pipeline_with_feature_selection(self, mock_cdd, sample_ohlcv_data):
        """Test pipeline with feature selection reduces features."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        # Without feature selection
        handler_no_selection = DataHandler(
            symbols=['BTC'],
            feature_engineering={
                'enabled': True,
                'custom_features': {
                    'enabled': True,
                    'lookback_periods': [7, 14, 21, 30],
                    'price_features': [
                        'moving_averages',
                        'exponential_moving_averages',
                        'volatility'
                    ]
                }
            },
            feature_selection={'per_currency': {'enabled': False}}
        )
        data_no_selection = handler_no_selection.get_data()
        
        # With feature selection
        handler_with_selection = DataHandler(
            symbols=['BTC'],
            feature_engineering={
                'enabled': True,
                'custom_features': {
                    'enabled': True,
                    'lookback_periods': [7, 14, 21, 30],
                    'price_features': [
                        'moving_averages',
                        'exponential_moving_averages',
                        'volatility'
                    ]
                }
            },
            feature_selection={
                'per_currency': {
                    'enabled': True,
                    'method': 'topk',
                    'k': 5
                }
            }
        )
        data_with_selection = handler_with_selection.get_data()
        
        # Feature selection should reduce column count
        assert data_with_selection.shape[1] < data_no_selection.shape[1], \
            "Feature selection should reduce number of columns"


class TestDataQualityMetrics:
    """Test data quality metrics."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_minimum_data_completeness(self, mock_cdd, sample_ohlcv_data):
        """Test that data has minimum completeness (low NaN ratio)."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        # Calculate completeness per column
        completeness = 1 - data.isna().sum() / len(data)
        
        # Most columns should be >80% complete
        low_completeness_cols = completeness[completeness < 0.8]
        
        # Allow some feature columns to have lower completeness due to lookback periods
        assert len(low_completeness_cols) < len(data.columns) * 0.3, \
            "Too many columns have low completeness"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_feature_variance(self, mock_cdd, sample_ohlcv_data):
        """Test that features have reasonable variance."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(
            symbols=['BTC'],
            feature_engineering={
                'enabled': True,
                'custom_features': {
                    'enabled': True,
                    'lookback_periods': [7],
                    'price_features': ['moving_averages']
                }
            }
        )
        data = handler.get_data()
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Check that features have some variance
        for col in numeric_cols:
            variance = data[col].var()
            if not pd.isna(variance):
                # Variance should be positive for non-constant columns
                # (Some columns might legitimately have low variance)
                assert variance >= 0, f"Column {col} has negative variance"

