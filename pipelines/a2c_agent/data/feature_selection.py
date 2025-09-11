import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Clean feature selection processor for cryptocurrency data.
    
    This class handles feature selection using various methods including
    variance-based selection and ML-based selection.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature selector.
        
        Args:
            config: Feature selection configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.method = self.config.get('method', 'variance')
        self.max_features = self.config.get('max_features', None)
        
        logger.info(f"Initialized FeatureSelector with method: {self.method}")
    
    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from the input data based on configuration.
        
        Args:
            data: Input dataframe with features
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        if not self.enabled:
            logger.info("Feature selection disabled")
            return data
        
        if self.max_features is None or self.max_features == -1:
            logger.info("Keeping all features (max_features not set)")
            return data
        
        logger.info(f"Applying {self.method} feature selection...")
        
        if self.method == 'variance':
            return self._variance_selection(data)
        elif self.method == 'ml_based':
            return self._ml_based_selection(data)
        else:
            logger.warning(f"Unknown selection method: {self.method}, keeping all features")
            return data
    
    def _variance_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on variance threshold.
        
        Args:
            data: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        try:
            threshold = self.config.get('threshold', 0.16)
            
            # Identify different column types
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in date_cols]
            
            if not feature_cols:
                logger.warning("No numeric features found for variance selection")
                return data
            
            # Fill NaN values for variance calculation
            feature_data = data[feature_cols].fillna(0)
            
            # Apply variance threshold
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(feature_data)
            
            # Get selected features
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            # If max_features is set, limit the number of features
            if self.max_features and len(selected_features) > self.max_features:
                # Calculate variances and select top features
                variances = feature_data.var()
                top_features = variances.nlargest(self.max_features).index.tolist()
                selected_features = [f for f in selected_features if f in top_features]
            
            result = data[date_cols + selected_features].copy()
            
            logger.info(f"Variance selection: {len(feature_cols)} -> {len(selected_features)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error in variance feature selection: {e}")
            return data
    
    def _ml_based_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select features using machine learning-based approach.
        
        Args:
            data: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        try:
            # Identify different column types
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            close_cols = [col for col in data.columns if 'close' in col.lower()]
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in date_cols and col not in close_cols]
            
            if not feature_cols:
                logger.warning("No features available for ML-based selection")
                return data
            
            # Use the first close column as target
            if not close_cols:
                logger.warning("No close price column found for target creation")
                return data
            
            target_col = close_cols[0]
            
            # Prepare features and target
            X = data[feature_cols].fillna(0)
            y = (data[target_col].pct_change() > 0).astype(int)
            
            # Remove NaN values from target
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) == 0 or y.sum() == 0 or (y == 0).sum() == 0:
                logger.warning("Insufficient valid data for ML-based selection")
                return data
            
            # Determine number of features to select
            k = min(self.max_features or len(feature_cols), len(feature_cols))
            
            # Use SelectKBest with f_classif
            selector = SelectKBest(score_func=f_classif, k=k)
            
            try:
                selector.fit(X, y)
                selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            except Exception as e:
                logger.warning(f"SelectKBest failed, using top variance features: {e}")
                # Fallback to variance-based selection
                variances = X.var()
                selected_features = variances.nlargest(k).index.tolist()
            
            result = data[date_cols + close_cols + selected_features].copy()
            
            logger.info(f"ML-based selection: {len(feature_cols)} -> {len(selected_features)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error in ML-based feature selection: {e}")
            return data
