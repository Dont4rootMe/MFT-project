"""Utilities for loading and preparing single-asset OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tensortrade.data.cdd import CryptoDataDownload


@dataclass
class DataConfig:
    """Configuration describing how to download and resample the dataset."""

    currency: str
    main_currency: str = "USDT"
    exchange: str = "Binance"
    time_frame: str = "1h"
    start: Optional[str] = None
    end: Optional[str] = None
    timezone: Optional[str] = "UTC"


class PriceDataLoader:
    """Load OHLCV data and prepare returns for modelling."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Return the cached dataset or download it on first access."""

        if self._data is None:
            raw = self._fetch_from_source()
            prepared = self._prepare_frame(raw)
            self._data = prepared
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
            raise ValueError("Fetched data must contain a 'date' column")

        data["date"] = pd.to_datetime(data["date"], utc=True)
        data = data.sort_values("date")

        if self.config.start:
            start_ts = pd.Timestamp(self.config.start, tz="UTC")
            data = data[data["date"] >= start_ts]
        if self.config.end:
            end_ts = pd.Timestamp(self.config.end, tz="UTC")
            data = data[data["date"] <= end_ts]

        return data.reset_index(drop=True)

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df = df.rename(columns={"date": "timestamp"})

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Fetched data missing required columns: {missing}")

        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        if self.config.timezone:
            df["timestamp"] = df["timestamp"].dt.tz_convert(self.config.timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df.set_index("timestamp")
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        resampled = df.resample(self.config.time_frame).agg(agg)
        resampled = resampled.dropna()
        resampled = resampled[resampled["close"] > 0]

        resampled["log_close"] = np.log(resampled["close"])
        resampled["log_return"] = resampled["log_close"].diff().fillna(0.0)

        prepared = resampled.reset_index()
        return prepared


def build_data_loader(config: dict) -> PriceDataLoader:
    data_cfg = DataConfig(
        currency=str(config["currency"]),
        main_currency=str(config.get("main_currency", "USDT")),
        exchange=str(config.get("exchange", "Binance")),
        time_frame=str(config.get("time_frame", "1h")),
        start=config.get("start"),
        end=config.get("end"),
        timezone=config.get("timezone", "UTC"),
    )
    return PriceDataLoader(data_cfg)
