import pandas as pd
import logging
import os
import json
import hashlib
from typing import Dict, List
from tensortrade.data.cdd import CryptoDataDownload
import yaml
from omegaconf import OmegaConf

from .feature_engine import FeatureEngineeringProcessor
from .feature_selection import FeatureSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Clean DataHandler for cryptocurrency data processing.
    
    This class provides a simple interface for data fetching, feature engineering,
    and feature selection. End users should only interact with the get_data() method.
    """
    
    def __init__(
        self,
        symbols: List[str],
        time_freq: str = "1d",
        exchange: str = "Binance",
        main_currency: str = "USDT",
        feature_engineering: Dict = None,
        feature_selection: Dict = None,
        cache: Dict = None,
    ):
        """
        Initialize DataHandler with configuration.
        
        Args:
            symbols: List of symbols to fetch (e.g., ["BTC", "ETH"])
            time_freq: Time frequency for data (e.g., "1h", "5m", "1d")
            exchange: Exchange name (default: "Binance")
            main_currency: Main currency (default: "USDT")
            feature_engineering: Feature engineering configuration
            feature_selection: Feature selection configuration
        """
        self.symbols = list(symbols)
        self.time_freq = time_freq
        self.exchange = exchange
        self.main_currency = main_currency
        
        # Convert Hydra configs to plain dictionaries
        fe_cfg = OmegaConf.to_container(feature_engineering, resolve=True) if feature_engineering else {}
        fs_cfg = OmegaConf.to_container(feature_selection, resolve=True) if feature_selection else {}

        # Initialize processors
        self.feature_processor = FeatureEngineeringProcessor(fe_cfg)
        self.feature_selector = FeatureSelector(fs_cfg)

        # Cache configuration
        self.cache_config = OmegaConf.to_container(cache, resolve=True) if cache else {}
        self.cache_enabled = self.cache_config.get("enabled", False)
        self.cache_dir = self.cache_config.get("dir", "cache")
        self.checkpoint_name = self.cache_config.get("checkpoint_name")
        self.cache_id = self.checkpoint_name or self._config_hash()
        self.cache_path = os.path.join(self.cache_dir, self.cache_id, "data.pkl")
        self.metadata_path = os.path.join(self.cache_dir, self.cache_id, "config.yaml")
        
        # Internal data storage
        self._processed_data = None
        
        logger.info(f"Initialized DataHandler for {len(self.symbols)} currencies")

    def get_data(self) -> pd.DataFrame:
        """
        Public method to get processed data.
        
        This is the only method end users should interact with.
        Returns a clean pandas DataFrame with all processing applied.
        
        Returns:
            pd.DataFrame: Processed and ready-to-use data
        """
        if self._processed_data is None:
            if self.cache_enabled and os.path.exists(self.cache_path):
                logger.info(f"Loading data from cache: {self.cache_path}")
                self._processed_data = pd.read_pickle(self.cache_path)
                # Synchronize symbol list with metadata if available
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "r") as f:
                        meta = yaml.safe_load(f) or {}
                    self.symbols = meta.get("symbols", self.symbols)
                else:
                    self.symbols = sorted({
                        col.split("_")[0] for col in self._processed_data.columns
                        if "_" in col
                    })
            else:
                logger.info("Processing data...")
                self._processed_data = self._run_pipeline()
                if self.cache_enabled:
                    os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                    self._processed_data.to_pickle(self.cache_path)
                    with open(self.metadata_path, "w") as f:
                        yaml.safe_dump(self._metadata(), f)
                    logger.info(f"Cached processed data to {self.cache_path}")
        
        return self._processed_data
    
    def _run_pipeline(self) -> pd.DataFrame:
        """
        Internal method to run the complete data processing pipeline.
        
        Returns:
            pd.DataFrame: Processed data
        """
        logger.info("Starting data processing pipeline...")
        
        # Step 1: Fetch raw data
        raw_data = self._fetch_raw_data()
        
        # Step 2: Generate features for all currencies
        processed_data = self._apply_feature_engineering(raw_data)
        
        # Step 3: Combine multi-currency data
        combined_data = self._combine_currencies(processed_data)
        
        # Step 4: Apply feature selection
        final_data = self._apply_feature_selection(combined_data)
        
        logger.info("Data processing pipeline completed successfully")
        return final_data
    
    def _fetch_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch raw OHLCV data for all configured currencies.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping currency symbols to raw dataframes
        """
        logger.info(f"Fetching data for currencies: {self.symbols}")
        logger.info(f"Exchange: {self.exchange}, Quote: {self.main_currency}, Timeframe: {self.time_freq}")
        
        cdd = CryptoDataDownload()
        raw_data = {}

        for currency in self.symbols:
            try:
                logger.info(f"Fetching {currency}/{self.main_currency} data...")
                
                # Fetch raw data from exchange
                timeframe = 'd' if self.time_freq == '1d' else self.time_freq
                data = cdd.fetch(
                    self.exchange,
                    self.main_currency,
                    currency,
                    timeframe
                )
                
                # Prepare and clean data
                if 'date' not in data.columns:
                    data = data.reset_index()
                
                # Select OHLCV columns and clean
                data = data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                data = self._prepare_raw_data(data)
                
                raw_data[currency] = data
                logger.info(f"✓ Successfully fetched {len(data)} records for {currency}")
                
            except Exception as e:
                logger.error(f"✗ Failed to fetch data for {currency}: {e}")
                continue
        
        if not raw_data:
            raise ValueError("Failed to fetch data for any currency")

        # Update the symbol list to reflect successfully fetched currencies.
        self.symbols = list(raw_data.keys())

        return raw_data
    
    def _prepare_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare raw OHLCV data.
        
        Args:
            df: Raw OHLCV dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df = df.copy()
        
        # Convert data types
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        
        # Sort by date and reset index
        df.sort_values(by='date', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Convert date to string format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    
    def _apply_feature_engineering(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply feature engineering to all currency data.
        
        Args:
            raw_data: Dictionary of raw currency dataframes
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of processed currency dataframes
        """
        logger.info("Applying feature engineering...")

        processed_data = {}

        for currency, data in raw_data.items():
            logger.info(f"Processing features for {currency}...")

            features = self.feature_processor.generate_features(
                data, currency_name=currency
            )

            # Per-currency feature selection
            features = self.feature_selector.select_per_currency(features)

            processed_data[currency] = features

            logger.info(
                f"✓ Generated {len(features.columns)} features for {currency} after selection"
            )

        return processed_data
    
    def _combine_currencies(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine data from multiple currencies into a single dataframe.
        
        Args:
            processed_data: Dictionary of processed currency dataframes
            
        Returns:
            pd.DataFrame: Combined dataframe
        """
        logger.info("Combining multi-currency data...")
        
        # Determine common date range
        date_sets = []
        prefixed_dfs = []
        for currency, df in processed_data.items():
            df_copy = df.copy()

            date_col = [col for col in df_copy.columns if 'date' in col.lower()][0]
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            date_sets.append(set(df_copy[date_col]))

            # Add currency prefix to columns that are not already prefixed
            for col in df_copy.columns:
                if 'date' in col.lower():
                    continue
                if not col.startswith(f"{currency}_"):
                    df_copy.rename(columns={col: f"{currency}_{col}"}, inplace=True)

            prefixed_dfs.append((currency, df_copy, date_col))

        if not prefixed_dfs:
            raise ValueError("No data available to combine")

        # Find intersection of dates and limit to common timeframe
        common_dates = sorted(set.intersection(*date_sets))
        if not common_dates:
            raise ValueError("No common dates across currencies")

        # Trim each dataframe to the common date range
        trimmed_dfs = []
        for currency, df_copy, date_col in prefixed_dfs:
            df_trimmed = df_copy[df_copy[date_col].isin(common_dates)]
            trimmed_dfs.append(df_trimmed)

        # Merge dataframes on the date column using inner joins
        result = trimmed_dfs[0]
        base_date_col = [col for col in result.columns if 'date' in col.lower()][0]

        for df in trimmed_dfs[1:]:
            df_date_col = [col for col in df.columns if 'date' in col.lower()][0]
            result = pd.merge(result, df, left_on=base_date_col, right_on=df_date_col, how='inner')

            if df_date_col != base_date_col and df_date_col in result.columns:
                result.drop(columns=[df_date_col], inplace=True)

        # Sort by date and reset index
        result = result.sort_values(base_date_col).reset_index(drop=True)

        logger.info(
            f"✓ Combined {len(processed_data)} currency datasets into {result.shape} with common timeframe"
        )

        # Convert date column back to string for consistency
        result[base_date_col] = result[base_date_col].dt.strftime('%Y-%m-%d %H:%M:%S')

        return result
    
    def _apply_feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection to the combined data.
        
        Args:
            data: Combined dataframe
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        logger.info("Applying post-merge feature selection...")

        selected_data = self.feature_selector.select_post_merge(data)

        logger.info(
            f"✓ Selected {len(selected_data.columns)} features from {len(data.columns)}"
        )

        return selected_data

    def _metadata(self) -> Dict:
        """Generate metadata describing the dataset and processing steps."""
        return {
            "symbols": self.symbols,
            "time_freq": self.time_freq,
            "exchange": self.exchange,
            "main_currency": self.main_currency,
            "feature_engineering": self.feature_processor.config,
            "feature_selection": self.feature_selector.config,
        }

    def _config_hash(self) -> str:
        """Compute a stable hash of the configuration for cache identification."""
        config = self._metadata()
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
