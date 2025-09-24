import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineeringProcessor:
    """
    Clean feature engineering processor for cryptocurrency data.
    
    This class handles the generation of technical indicators, custom features,
    and quantstats-based features for cryptocurrency OHLCV data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature engineering processor.
        
        Args:
            config: Feature engineering configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        logger.info("Initialized FeatureEngineeringProcessor")
        
    def generate_features(self, data: pd.DataFrame, currency_name: str = "") -> pd.DataFrame:
        """
        Generate all configured features for the data.
        
        Args:
            data: Raw OHLCV dataframe
            currency_name: Name of the currency for feature prefixing
            
        Returns:
            pd.DataFrame: Dataframe with generated features
        """
        if not self.enabled:
            logger.info("Feature engineering disabled")
            return data.copy()
        
        df = data.copy()
        prefix = f"{currency_name}_" if currency_name else ""
        
        # Generate different types of features
        if self.config.get('technical_indicators', {}).get('enabled', False):
            df = self._generate_technical_indicators(df, prefix)
        
        if self.config.get('custom_features', {}).get('enabled', False):
            df = self._generate_custom_features(df, prefix)
        
        if self.config.get('quantstats_features', {}).get('enabled', False):
            df = self._generate_quantstats_features(df, prefix)
        
        # Clean and prepare final dataset
        df = self._clean_features(df)
        
        logger.info(f"Generated {len(df.columns)} total features for {currency_name or 'data'}")
        return df
    
    def _generate_technical_indicators(self, data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """Generate technical analysis indicators using pandas_ta."""
        try:
            import pandas_ta as ta
            
            df = data.copy()
            
            # Ensure we have a proper index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Set cores for parallel processing
            performance_config = self.config.get('performance', {})
            n_cores = performance_config.get('n_cores') or os.cpu_count()
            df.ta.cores = n_cores
            
            # Generate features by strategy
            strategies = self.config['technical_indicators']['strategies']
            exclude = self.config['technical_indicators']['exclude']
            
            original_columns = set(df.columns)
            
            for strategy in strategies:
                try:
                    df.ta.study(strategy, exclude=exclude)
                    logger.info(f"Generated {strategy} indicators for {prefix}")
                except Exception as e:
                    logger.warning(f"Failed to generate {strategy} indicators: {e}")
            
            # Reset index to get date back as column
            df.reset_index(inplace=True)
            
            # Add prefix to new columns
            if prefix:
                new_columns = set(df.columns) - original_columns - {'date'}
                rename_dict = {col: f"{prefix}{col}" for col in new_columns}
                df.rename(columns=rename_dict, inplace=True)
            
            return df
            
        except ImportError:
            logger.warning("pandas_ta not available. Skipping technical indicators.")
            return data
        except Exception as e:
            logger.error(f"Error generating technical indicators: {e}")
            return data
    
    def _generate_custom_features(self, data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """Generate custom features like moving averages, volatility, etc."""
        df = data.copy()
        
        lookback_periods = self.config['custom_features']['lookback_periods']
        price_features = self.config['custom_features']['price_features']
        
        custom_features = {}
        
        try:
            # Moving averages
            if "moving_averages" in price_features:
                for period in lookback_periods:
                    custom_features[f'{prefix}ma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential moving averages
            if "exponential_moving_averages" in price_features:
                for period in lookback_periods:
                    custom_features[f'{prefix}ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Volatility
            if "volatility" in price_features:
                for period in lookback_periods:
                    custom_features[f'{prefix}vol_{period}'] = df['close'].rolling(window=period).std()
            
            # Log returns
            if "log_returns" in price_features:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        custom_features[f'{prefix}lr_{col}'] = np.log(df[col]).diff()
            
            # Price shifts (previous values)
            if "price_shifts" in price_features:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        custom_features[f'{prefix}prev_{col}'] = df[col].shift(1)
            
            # Add custom RSI and MACD
            custom_features[f'{prefix}rsi_14'] = self._calculate_rsi(df['close'], 14)
            custom_features[f'{prefix}macd'] = self._calculate_macd(df['close'])
            
            # Convert to DataFrame and concatenate
            custom_df = pd.DataFrame(custom_features, index=df.index)
            result = pd.concat([df, custom_df], axis=1)
            
            logger.info(f"Generated {len(custom_features)} custom features for {prefix}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating custom features: {e}")
            return df
    
    def _generate_quantstats_features(self, data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """Generate quantstats-based features with subset selection capability."""
        try:
            import quantstats as qs
            qs.extend_pandas()
            
            df = data.copy()
            qs_config = self.config['quantstats_features']
            
            # Get feature categories/subsets to include
            feature_categories = qs_config.get('categories', ['all'])
            exclude = qs_config.get('exclude', [])
            
            # Set date index for quantstats
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Define quantstats feature categories
            quantstats_categories = {
                'risk': [
                    'max_drawdown', 'avg_drawdown', 'avg_drawdown_days', 'recovery_factor',
                    'ulcer_index', 'serenity_index', 'var', 'cvar', 'expected_shortfall'
                ],
                'returns': [
                    'sharpe', 'sortino', 'calmar', 'omega', 'information_ratio',
                    'modigliani', 'adjusted_sortino', 'skew', 'kurtosis'
                ],
                'performance': [
                    'win_rate', 'avg_win', 'avg_loss', 'profit_factor', 'payoff_ratio',
                    'profit_ratio', 'cpc_index', 'tail_ratio', 'outlier_win_ratio',
                    'outlier_loss_ratio', 'common_sense_ratio', 'cagr', 'rar'
                ],
                'volatility': [
                    'volatility', 'rolling_volatility', 'volatility_of_volatility'
                ],
                'drawdown': [
                    'to_drawdown_series', 'drawdown_details', 'max_drawdown',
                    'avg_drawdown', 'avg_drawdown_days'
                ],
                'distribution': [
                    'skew', 'kurtosis', 'outliers', 'kelly', 'risk_of_ruin'
                ]
            }
            
            # Determine which indicators to use
            if 'all' in feature_categories:
                # Use all available quantstats functions
                all_indicators = [f for f in dir(qs.stats) if f[0] != '_']
                indicators_to_use = set(all_indicators)
            else:
                # Use only specified categories
                indicators_to_use = set()
                for category in feature_categories:
                    if category in quantstats_categories:
                        indicators_to_use.update(quantstats_categories[category])
                    else:
                        logger.warning(f"Unknown quantstats category: {category}")
            
            # Apply exclusions
            indicators_list = [indicator for indicator in indicators_to_use if indicator not in exclude]
            
            # Filter to only include functions that actually exist in quantstats
            available_indicators = [f for f in dir(qs.stats) if f[0] != '_']
            indicators_list = [indicator for indicator in indicators_list if indicator in available_indicators]
            
            qs_features = {}
            successful_indicators = []
            
            for indicator_name in indicators_list:
                try:
                    indicator_func = getattr(qs.stats, indicator_name)
                    indicator = indicator_func(df['close'])
                    
                    # Handle different return types
                    if isinstance(indicator, pd.Series):
                        qs_features[f'{prefix}qs_{indicator_name}'] = indicator
                        successful_indicators.append(indicator_name)
                    elif isinstance(indicator, (int, float)) and not pd.isna(indicator):
                        # Convert scalar values to series
                        qs_features[f'{prefix}qs_{indicator_name}'] = pd.Series(
                            [indicator] * len(df), index=df.index
                        )
                        successful_indicators.append(indicator_name)
                except Exception as e:
                    logger.debug(f"Failed to generate {indicator_name}: {e}")
                    continue
            
            if qs_features:
                qs_df = pd.DataFrame(qs_features, index=df.index)
                result = pd.concat([df, qs_df], axis=1)
                logger.info(f"Generated {len(qs_features)} quantstats features for {prefix}")
                logger.debug(f"Successful quantstats indicators: {successful_indicators}")
            else:
                result = df
                logger.warning(f"No quantstats features were successfully generated for {prefix}")
            
            return result.reset_index()
            
        except ImportError:
            logger.warning("quantstats not available. Skipping quantstats features.")
            return data
        except Exception as e:
            logger.error(f"Error generating quantstats features: {e}")
            return data
    
    def _calculate_rsi(self, price: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=price.index, dtype=float)
    
    def _calculate_macd(self, price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = price.ewm(span=fast).mean()
            ema_slow = price.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            return macd - signal_line
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return pd.Series(index=price.index, dtype=float)
    
    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features."""
        df = data.copy()
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df = df.fillna(method='ffill')
        
        # Drop columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    