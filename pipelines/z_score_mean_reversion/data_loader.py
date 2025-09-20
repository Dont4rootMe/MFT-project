"""Data loading utilities for the z-score mean reversion baseline."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from tensortrade.data.cdd import CryptoDataDownload

LOGGER = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration required to download a single asset price series."""

    currency: str
    main_currency: str = "USDT"
    exchange: str = "Binance"
    time_frame: str = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SingleAssetDataLoader:
    """Load a single cryptocurrency close price time series in-memory."""

    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self._data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Return the processed close price series."""

        if self._data is None:
            LOGGER.info(
                "Fetching data for %s/%s on %s with timeframe %s",
                self.config.currency,
                self.config.main_currency,
                self.config.exchange,
                self.config.time_frame,
            )
            raw = self._fetch_from_source()
            self._data = self._post_process(raw)

        return self._data.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_from_source(self) -> pd.DataFrame:
        cdd = CryptoDataDownload()
        timeframe = "d" if self.config.time_frame == "1d" else self.config.time_frame
        data = cdd.fetch(
            self.config.exchange,
            self.config.main_currency,
            self.config.currency,
            timeframe,
        )
        if "date" not in data.columns:
            raise ValueError("Downloaded data must include a 'date' column")

        data["date"] = pd.to_datetime(data["date"], utc=True)
        data = data.sort_values("date").reset_index(drop=True)

        if self.config.start_date:
            start_ts = pd.Timestamp(self.config.start_date, tz="UTC")
            data = data.loc[data["date"] >= start_ts]
        if self.config.end_date:
            end_ts = pd.Timestamp(self.config.end_date, tz="UTC")
            data = data.loc[data["date"] <= end_ts]

        return data

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()
        processed["date"] = pd.to_datetime(processed["date"], utc=True)
        processed = processed.sort_values("date").reset_index(drop=True)
        processed = processed.drop(columns=["unix"])
        
        date_column = ['date']
        value_columns = ["close", "open", "high", "low", "volume"]
        
        processed = processed[date_column + value_columns]
        processed[value_columns] = processed[value_columns].astype(float)
        processed.rename(columns={
            "close": f"{self.config.currency}_close", 
            "open": f"{self.config.currency}_open", 
            "high": f"{self.config.currency}_high", 
            "low": f"{self.config.currency}_low",
            "volume": f"{self.config.currency}_volume"}, 
            inplace=True
        )
        return processed


def build_data_loader(config: dict) -> SingleAssetDataLoader:
    """Factory to instantiate the data loader from a dictionary config."""

    loader_cfg = DataLoaderConfig(
        currency=config["currency"],
        main_currency=config.get("main_currency", "USDT"),
        exchange=config.get("exchange", "Binance"),
        time_frame=config.get("time_frame", "1d"),
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
    )
    return SingleAssetDataLoader(loader_cfg)
