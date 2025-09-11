import pandas as pd
import logging
from typing import Dict, List
from tensortrade.data.cdd import CryptoDataDownload

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
        self.symbols = symbols
        self.time_freq = time_freq
        self.exchange = exchange
        self.main_currency = main_currency
        
        # Initialize processors
        self.feature_processor = FeatureEngineeringProcessor(feature_engineering or {})
        self.feature_selector = FeatureSelector(feature_selection or {})
        
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
            logger.info("Processing data...")
            self._processed_data = self._run_pipeline()
        
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
                data = cdd.fetch(
                    self.exchange,
                    self.main_currency,
                    currency,
                    self.time_freq
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
        df['date'] = pd.to_datetime(df['date'])
        
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
            
            # Generate features using the feature processor
            processed_data[currency] = self.feature_processor.generate_features(
                data, currency_name=currency
            )
            
            logger.info(f"✓ Generated {len(processed_data[currency].columns)} features for {currency}")
        
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
        
        combined_dfs = []
        
        for currency, df in processed_data.items():
            df_copy = df.copy()
            
            # Add currency prefix to all columns except date
            for col in df_copy.columns:
                if 'date' not in col.lower():
                    df_copy.rename(columns={col: f"{currency}_{col}"}, inplace=True)
            
            combined_dfs.append(df_copy)
        
        # Merge dataframes on date column
        result = combined_dfs[0]
        date_col = [col for col in result.columns if 'date' in col.lower()][0]
        
        for df in combined_dfs[1:]:
            df_date_col = [col for col in df.columns if 'date' in col.lower()][0]
            result = pd.merge(result, df, left_on=date_col, right_on=df_date_col, how='outer')
            
            # Drop duplicate date column
            if df_date_col != date_col and df_date_col in result.columns:
                result.drop(columns=[df_date_col], inplace=True)
        
        # Sort by date and reset index
        result = result.sort_values(date_col).reset_index(drop=True)
        
        logger.info(f"✓ Combined {len(processed_data)} currency datasets into {result.shape}")
        
        return result
    
    def _apply_feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection to the combined data.
        
        Args:
            data: Combined dataframe
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        logger.info("Applying feature selection...")
        
        # Apply feature selection using the feature selector
        selected_data = self.feature_selector.select_features(data)
        
        logger.info(f"✓ Selected {len(selected_data.columns)} features from {len(data.columns)}")
        
        return selected_data
