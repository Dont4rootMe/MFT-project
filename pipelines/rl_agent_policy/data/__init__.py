"""
Data processing module for cryptocurrency data.

This module provides:
- DataHandler: Main interface for data fetching and processing
- FeatureEngineeringProcessor: Feature generation and engineering
- FeatureSelector: Feature selection and dimensionality reduction
"""

__all__ = [
    'DataHandler',
    'FeatureEngineeringProcessor',
    'FeatureSelector',
]

# Lazy imports to avoid loading heavy dependencies
def __getattr__(name):
    if name == 'DataHandler':
        from .data_parser import DataHandler
        return DataHandler
    elif name == 'FeatureEngineeringProcessor':
        from .feature_engine import FeatureEngineeringProcessor
        return FeatureEngineeringProcessor
    elif name == 'FeatureSelector':
        from .feature_selection import FeatureSelector
        return FeatureSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

