"""
Pytest configuration and shared fixtures for data tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    n_days = 100
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(n_days).cumsum() * 0.02)
    
    data = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d %H:%M:%S'),
        'open': close_prices * (1 + np.random.randn(n_days) * 0.01),
        'high': close_prices * (1 + np.abs(np.random.randn(n_days)) * 0.02),
        'low': close_prices * (1 - np.abs(np.random.randn(n_days)) * 0.02),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n_days).astype(float),
    })
    
    # Ensure high >= close >= low and high >= open >= low
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def sample_multi_currency_data():
    """Generate sample multi-currency data."""
    n_days = 100
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    currencies = ['BTC', 'ETH', 'BNB']
    data = {}
    
    np.random.seed(42)
    for i, currency in enumerate(currencies):
        close_prices = (100 + i * 50) * (1 + np.random.randn(n_days).cumsum() * 0.02)
        
        data[currency] = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d %H:%M:%S'),
            'open': close_prices * (1 + np.random.randn(n_days) * 0.01),
            'high': close_prices * (1 + np.abs(np.random.randn(n_days)) * 0.02),
            'low': close_prices * (1 - np.abs(np.random.randn(n_days)) * 0.02),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, n_days).astype(float),
        })
        
        # Ensure high >= close >= low
        data[currency]['high'] = data[currency][['open', 'high', 'close']].max(axis=1)
        data[currency]['low'] = data[currency][['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def feature_engineering_config():
    """Sample feature engineering configuration."""
    return {
        'enabled': True,
        'technical_indicators': {
            'enabled': True,
            'strategies': ['momentum', 'trend'],
            'exclude': []
        },
        'custom_features': {
            'enabled': True,
            'lookback_periods': [7, 14, 30],
            'price_features': [
                'moving_averages',
                'exponential_moving_averages',
                'volatility',
                'log_returns',
                'price_shifts'
            ]
        },
        'quantstats_features': {
            'enabled': False,
            'categories': ['risk', 'returns'],
            'exclude': []
        },
        'performance': {
            'n_cores': 1
        }
    }


@pytest.fixture
def feature_selection_config():
    """Sample feature selection configuration."""
    return {
        'per_currency': {
            'enabled': True,
            'method': 'variance_corr',
            'variance_threshold': 0.01,
            'correlation_threshold': 0.95,
            'max_features': 50
        },
        'post_merge': {
            'enabled': False,
            'method': 'none'
        }
    }


@pytest.fixture
def data_handler_config(feature_engineering_config, feature_selection_config):
    """Complete DataHandler configuration."""
    return {
        'symbols': ['BTC', 'ETH'],
        'time_freq': '1d',
        'exchange': 'Binance',
        'main_currency': 'USDT',
        'feature_engineering': feature_engineering_config,
        'feature_selection': feature_selection_config,
        'cache': {
            'enabled': False,
            'dir': 'test_cache',
            'checkpoint_name': None
        }
    }


@pytest.fixture
def sample_processed_data():
    """Generate sample processed data with features."""
    n_days = 100
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    np.random.seed(42)
    data = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d %H:%M:%S'),
        'open': 100 + np.random.randn(n_days) * 10,
        'high': 105 + np.random.randn(n_days) * 10,
        'low': 95 + np.random.randn(n_days) * 10,
        'close': 100 + np.random.randn(n_days) * 10,
        'volume': np.random.randint(1000000, 10000000, n_days).astype(float),
        'ma_7': 100 + np.random.randn(n_days) * 5,
        'ma_14': 100 + np.random.randn(n_days) * 5,
        'ema_7': 100 + np.random.randn(n_days) * 5,
        'vol_7': np.abs(np.random.randn(n_days) * 2),
        'rsi_14': np.random.uniform(20, 80, n_days),
    })
    
    return data


# Validation helper functions
def validate_dataframe_structure(df: pd.DataFrame, required_columns: list = None):
    """Validate basic dataframe structure."""
    assert isinstance(df, pd.DataFrame), "Data must be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) > 0, "DataFrame should have at least one row"
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        assert not missing_cols, f"Missing required columns: {missing_cols}"


def validate_data_types(df: pd.DataFrame, type_specs: Dict[str, type] = None):
    """Validate column data types."""
    if type_specs:
        for col, expected_type in type_specs.items():
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]) if expected_type in [int, float] else True, \
                    f"Column {col} should be numeric"


def validate_value_ranges(df: pd.DataFrame, range_specs: Dict[str, tuple] = None):
    """Validate value ranges for numeric columns."""
    if range_specs:
        for col, (min_val, max_val) in range_specs.items():
            if col in df.columns:
                actual_min = df[col].min()
                actual_max = df[col].max()
                if min_val is not None:
                    assert actual_min >= min_val, \
                        f"Column {col} has values below minimum: {actual_min} < {min_val}"
                if max_val is not None:
                    assert actual_max <= max_val, \
                        f"Column {col} has values above maximum: {actual_max} > {max_val}"


def validate_no_invalid_values(df: pd.DataFrame, allow_nan: bool = False):
    """Check for invalid values (NaN, inf, -inf)."""
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert not np.isinf(df[col]).any(), f"Column {col} contains infinite values"
    
    if not allow_nan:
        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        assert not nan_cols, f"Columns contain NaN values: {nan_cols}"


def validate_ohlcv_consistency(df: pd.DataFrame, prefix: str = ""):
    """Validate OHLCV data consistency rules."""
    open_col = f"{prefix}open" if prefix else "open"
    high_col = f"{prefix}high" if prefix else "high"
    low_col = f"{prefix}low" if prefix else "low"
    close_col = f"{prefix}close" if prefix else "close"
    volume_col = f"{prefix}volume" if prefix else "volume"
    
    if all(col in df.columns for col in [high_col, low_col, close_col]):
        # High should be >= Close
        assert (df[high_col] >= df[close_col]).all() or (df[high_col] - df[close_col]).abs().max() < 1e-6, \
            f"High should be >= Close for {prefix or 'base currency'}"
        
        # Low should be <= Close
        assert (df[low_col] <= df[close_col]).all() or (df[close_col] - df[low_col]).abs().max() < 1e-6, \
            f"Low should be <= Close for {prefix or 'base currency'}"
        
    if volume_col in df.columns:
        # Volume should be non-negative
        assert (df[volume_col] >= 0).all(), f"Volume should be non-negative for {prefix or 'base currency'}"

