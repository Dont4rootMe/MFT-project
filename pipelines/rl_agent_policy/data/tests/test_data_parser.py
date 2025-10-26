"""
Tests for DataHandler class (data_parser.py).

Tests cover:
- Data structure and format validation
- Data type checking
- Value range validation
- Required columns presence
- Multi-currency data combination
- Cache functionality
- Pipeline integrity
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.rl_agent_policy.data.data_parser import DataHandler
from pipelines.rl_agent_policy.data.tests.conftest import (
    validate_dataframe_structure,
    validate_data_types,
    validate_no_invalid_values,
    validate_ohlcv_consistency
)


class TestDataHandlerInitialization:
    """Test DataHandler initialization."""
    
    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        handler = DataHandler(
            symbols=['BTC', 'ETH'],
            time_freq='1d'
        )
        
        assert handler.symbols == ['BTC', 'ETH']
        assert handler.time_freq == '1d'
        assert handler.exchange == 'Binance'
        assert handler.main_currency == 'USDT'
        assert handler.feature_processor is not None
        assert handler.feature_selector is not None
    
    def test_init_with_full_config(self, data_handler_config):
        """Test initialization with full configuration."""
        handler = DataHandler(**data_handler_config)
        
        assert handler.symbols == data_handler_config['symbols']
        assert handler.time_freq == data_handler_config['time_freq']
        assert handler.exchange == data_handler_config['exchange']
        assert handler.main_currency == data_handler_config['main_currency']
    
    def test_init_cache_configuration(self):
        """Test cache configuration initialization."""
        cache_config = {
            'enabled': True,
            'dir': 'test_cache_dir',
            'checkpoint_name': 'test_checkpoint'
        }
        
        handler = DataHandler(
            symbols=['BTC'],
            cache=cache_config
        )
        
        assert handler.cache_enabled is True
        assert handler.cache_dir == 'test_cache_dir'
        assert handler.checkpoint_name == 'test_checkpoint'
        assert 'test_checkpoint' in handler.cache_path


class TestDataFetching:
    """Test data fetching and preparation."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_fetch_raw_data_structure(self, mock_cdd, sample_ohlcv_data):
        """Test that fetched raw data has correct structure."""
        # Mock the CryptoDataDownload.fetch method
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        raw_data = handler._fetch_raw_data()
        
        # Validate structure
        assert isinstance(raw_data, dict)
        assert 'BTC' in raw_data
        
        btc_data = raw_data['BTC']
        validate_dataframe_structure(
            btc_data,
            required_columns=['date', 'open', 'high', 'low', 'close', 'volume']
        )
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_fetch_raw_data_types(self, mock_cdd, sample_ohlcv_data):
        """Test that fetched data has correct data types."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        raw_data = handler._fetch_raw_data()
        
        btc_data = raw_data['BTC']
        
        # Check numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert pd.api.types.is_numeric_dtype(btc_data[col]), \
                f"Column {col} should be numeric"
        
        # Check date column is string format
        assert btc_data['date'].dtype == object or pd.api.types.is_string_dtype(btc_data['date'])
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_prepare_raw_data_sorting(self, mock_cdd, sample_ohlcv_data):
        """Test that raw data is properly sorted by date."""
        # Shuffle the data
        shuffled_data = sample_ohlcv_data.sample(frac=1).reset_index(drop=True)
        
        mock_instance = Mock()
        mock_instance.fetch.return_value = shuffled_data
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        raw_data = handler._fetch_raw_data()
        
        btc_data = raw_data['BTC']
        dates = pd.to_datetime(btc_data['date'])
        
        # Check if sorted
        assert dates.is_monotonic_increasing, "Data should be sorted by date"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_prepare_raw_data_no_duplicates(self, mock_cdd, sample_ohlcv_data):
        """Test that prepared data has no duplicate dates."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        raw_data = handler._fetch_raw_data()
        
        btc_data = raw_data['BTC']
        date_counts = btc_data['date'].value_counts()
        
        assert (date_counts == 1).all(), "Should not have duplicate dates"


class TestOHLCVValidation:
    """Test OHLCV data consistency."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_ohlcv_high_low_consistency(self, mock_cdd, sample_ohlcv_data):
        """Test that High >= Low for all records."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        raw_data = handler._fetch_raw_data()
        
        btc_data = raw_data['BTC']
        assert (btc_data['high'] >= btc_data['low']).all(), \
            "High should be >= Low for all records"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_ohlcv_price_in_range(self, mock_cdd, sample_ohlcv_data):
        """Test that Close is within High-Low range."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        raw_data = handler._fetch_raw_data()
        
        btc_data = raw_data['BTC']
        
        # Allow small floating point errors
        tolerance = 1e-6
        assert (btc_data['close'] <= btc_data['high'] + tolerance).all(), \
            "Close should be <= High"
        assert (btc_data['close'] >= btc_data['low'] - tolerance).all(), \
            "Close should be >= Low"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_volume_non_negative(self, mock_cdd, sample_ohlcv_data):
        """Test that volume is non-negative."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        raw_data = handler._fetch_raw_data()
        
        btc_data = raw_data['BTC']
        assert (btc_data['volume'] >= 0).all(), "Volume should be non-negative"


class TestMultiCurrencyCombination:
    """Test multi-currency data combination."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_combine_currencies_structure(self, mock_cdd, sample_multi_currency_data):
        """Test that combined data has correct structure."""
        mock_instance = Mock()
        
        def fetch_side_effect(exchange, main, currency, timeframe):
            return sample_multi_currency_data[currency].copy()
        
        mock_instance.fetch.side_effect = fetch_side_effect
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(
            symbols=['BTC', 'ETH'],
            feature_engineering={'enabled': False},
            feature_selection={'per_currency': {'enabled': False}, 'post_merge': {'enabled': False}}
        )
        
        # Mock the feature engineering to pass through
        with patch.object(handler, '_apply_feature_engineering') as mock_fe:
            mock_fe.return_value = sample_multi_currency_data
            combined = handler._combine_currencies(sample_multi_currency_data)
        
        # Should have date column
        date_col = [col for col in combined.columns if 'date' in col.lower()][0]
        assert date_col in combined.columns
        
        # Should have prefixed columns for each currency
        for currency in ['BTC', 'ETH']:
            currency_cols = [col for col in combined.columns if col.startswith(f"{currency}_")]
            assert len(currency_cols) > 0, f"Should have {currency} prefixed columns"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_combine_currencies_common_dates(self, mock_cdd, sample_multi_currency_data):
        """Test that combined data only includes common dates."""
        # Create data with different date ranges
        btc_data = sample_multi_currency_data['BTC'].iloc[:90].copy()  # 90 days
        eth_data = sample_multi_currency_data['ETH'].iloc[10:].copy()  # days 10-100
        
        modified_data = {'BTC': btc_data, 'ETH': eth_data}
        
        handler = DataHandler(symbols=['BTC', 'ETH'])
        
        with patch.object(handler, '_apply_feature_engineering') as mock_fe:
            mock_fe.return_value = modified_data
            combined = handler._combine_currencies(modified_data)
        
        # Should only have dates that exist in both datasets
        # That would be days 10-90, so 80 days
        expected_length = 80
        assert len(combined) == expected_length, \
            f"Combined data should have {expected_length} rows (common dates only)"
    
    def test_combine_currencies_no_duplicate_columns(self, sample_multi_currency_data):
        """Test that combined data has no duplicate column names."""
        handler = DataHandler(symbols=['BTC', 'ETH'])
        
        with patch.object(handler, '_apply_feature_engineering') as mock_fe:
            mock_fe.return_value = sample_multi_currency_data
            combined = handler._combine_currencies(sample_multi_currency_data)
        
        # Check for duplicate column names
        assert len(combined.columns) == len(set(combined.columns)), \
            "Combined data should not have duplicate column names"


class TestDataValidation:
    """Test data validation and integrity."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_no_infinite_values(self, mock_cdd, sample_ohlcv_data):
        """Test that processed data contains no infinite values."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(
            symbols=['BTC'],
            feature_engineering={'enabled': False},
            feature_selection={'per_currency': {'enabled': False}, 'post_merge': {'enabled': False}}
        )
        
        data = handler.get_data()
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(data[col]).any(), \
                f"Column {col} should not contain infinite values"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_data_not_empty(self, mock_cdd, sample_ohlcv_data):
        """Test that final data is not empty."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        data = handler.get_data()
        
        assert len(data) > 0, "Final data should not be empty"
        assert len(data.columns) > 0, "Final data should have columns"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_data_shape_consistency(self, mock_cdd, sample_ohlcv_data):
        """Test that data shape is consistent across calls."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC'])
        
        data1 = handler.get_data()
        data2 = handler.get_data()  # Should return cached version
        
        assert data1.shape == data2.shape, "Data shape should be consistent"
        assert list(data1.columns) == list(data2.columns), "Columns should be consistent"


class TestCaching:
    """Test caching functionality."""
    
    def test_cache_save_and_load(self, sample_ohlcv_data):
        """Test that cache saves and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = {
                'enabled': True,
                'dir': tmpdir,
                'checkpoint_name': 'test_cache'
            }
            
            with patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload') as mock_cdd:
                mock_instance = Mock()
                mock_instance.fetch.return_value = sample_ohlcv_data.copy()
                mock_cdd.return_value = mock_instance
                
                # First handler - should fetch and cache
                handler1 = DataHandler(
                    symbols=['BTC'],
                    cache=cache_config,
                    feature_engineering={'enabled': False},
                    feature_selection={'per_currency': {'enabled': False}, 'post_merge': {'enabled': False}}
                )
                data1 = handler1.get_data()
                
                # Check cache file exists
                assert os.path.exists(handler1.cache_path), "Cache file should be created"
                assert os.path.exists(handler1.metadata_path), "Metadata file should be created"
                
                # Second handler - should load from cache
                handler2 = DataHandler(
                    symbols=['BTC'],
                    cache=cache_config,
                    feature_engineering={'enabled': False},
                    feature_selection={'per_currency': {'enabled': False}, 'post_merge': {'enabled': False}}
                )
                data2 = handler2.get_data()
                
                # Data should be identical
                pd.testing.assert_frame_equal(data1, data2)
    
    def test_cache_metadata_saved(self, sample_ohlcv_data):
        """Test that cache metadata is saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = {
                'enabled': True,
                'dir': tmpdir,
                'checkpoint_name': 'test_metadata'
            }
            
            with patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload') as mock_cdd:
                mock_instance = Mock()
                mock_instance.fetch.return_value = sample_ohlcv_data.copy()
                mock_cdd.return_value = mock_instance
                
                handler = DataHandler(
                    symbols=['BTC'],
                    time_freq='1d',
                    exchange='Binance',
                    cache=cache_config,
                    feature_engineering={'enabled': False},
                    feature_selection={'per_currency': {'enabled': False}, 'post_merge': {'enabled': False}}
                )
                handler.get_data()
                
                # Check metadata file
                import yaml
                with open(handler.metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                assert 'symbols' in metadata
                assert 'time_freq' in metadata
                assert metadata['time_freq'] == '1d'
                assert metadata['exchange'] == 'Binance'


class TestErrorHandling:
    """Test error handling in DataHandler."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_fetch_failure_handling(self, mock_cdd):
        """Test handling of fetch failures."""
        mock_instance = Mock()
        mock_instance.fetch.side_effect = Exception("Network error")
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(symbols=['BTC', 'ETH'])
        
        with pytest.raises(ValueError, match="Failed to fetch data for any currency"):
            handler._fetch_raw_data()
    
    def test_empty_symbols_list(self):
        """Test handling of empty symbols list."""
        with pytest.raises(Exception):
            handler = DataHandler(symbols=[])
            handler.get_data()
    
    def test_combine_currencies_no_common_dates(self):
        """Test error when no common dates exist."""
        # Create datasets with completely different date ranges
        data1 = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30).strftime('%Y-%m-%d %H:%M:%S'),
            'close': np.random.rand(30)
        })
        
        data2 = pd.DataFrame({
            'date': pd.date_range('2023-02-01', periods=30).strftime('%Y-%m-%d %H:%M:%S'),
            'close': np.random.rand(30)
        })
        
        processed_data = {'BTC': data1, 'ETH': data2}
        
        handler = DataHandler(symbols=['BTC', 'ETH'])
        
        with pytest.raises(ValueError, match="No common dates"):
            handler._combine_currencies(processed_data)


class TestFeatureColumns:
    """Test feature column generation and presence."""
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_base_columns_present(self, mock_cdd, sample_ohlcv_data):
        """Test that base OHLCV columns are present after processing."""
        mock_instance = Mock()
        mock_instance.fetch.return_value = sample_ohlcv_data.copy()
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(
            symbols=['BTC'],
            feature_engineering={'enabled': False}
        )
        data = handler.get_data()
        
        # Should have at least date column
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        assert len(date_cols) > 0, "Should have date column"
    
    @patch('pipelines.rl_agent_policy.data.data_parser.CryptoDataDownload')
    def test_multi_currency_column_prefixes(self, mock_cdd, sample_multi_currency_data):
        """Test that multi-currency data has proper column prefixes."""
        mock_instance = Mock()
        
        def fetch_side_effect(exchange, main, currency, timeframe):
            return sample_multi_currency_data[currency].copy()
        
        mock_instance.fetch.side_effect = fetch_side_effect
        mock_cdd.return_value = mock_instance
        
        handler = DataHandler(
            symbols=['BTC', 'ETH'],
            feature_engineering={'enabled': False},
            feature_selection={'per_currency': {'enabled': False}, 'post_merge': {'enabled': False}}
        )
        data = handler.get_data()
        
        # Check for BTC and ETH prefixes
        btc_cols = [col for col in data.columns if col.startswith('BTC_')]
        eth_cols = [col for col in data.columns if col.startswith('ETH_')]
        
        assert len(btc_cols) > 0, "Should have BTC prefixed columns"
        assert len(eth_cols) > 0, "Should have ETH prefixed columns"

