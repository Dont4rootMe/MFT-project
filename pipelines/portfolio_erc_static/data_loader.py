"""Utilities for loading multi-asset OHLC data in-memory.

This module mirrors the data access patterns used across the other
pipelines but strips any feature engineering or caching. It downloads
raw OHLCV data for the configured spot pairs and aligns them on the
intersection of trading dates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tensortrade.data.cdd import CryptoDataDownload

LOGGER = logging.getLogger(__name__)


@dataclass
class PriceDataConfig:
    """Configuration describing the desired OHLC panel."""

    symbols: Iterable[str]
    exchange: str = "Binance"
    main_currency: str = "USDT"
    time_frame: str = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class MultiAssetOHLCFetcher:
    """Fetch OHLC time series for multiple cryptocurrencies.

    The class keeps everything in-memory (no caching) to satisfy the
    reproducibility requirement of the pipeline.
    """

    def __init__(self, config: PriceDataConfig):
        self.config = config
        self._panel: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        if self._panel is None:
            raw = self._download_all()
            self._panel = self._align_panel(raw)
        return self._panel.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _download_all(self) -> Dict[str, pd.DataFrame]:
        downloader = CryptoDataDownload()
        result: Dict[str, pd.DataFrame] = {}

        timeframe = "d" if self.config.time_frame == "1d" else self.config.time_frame

        for symbol in self.config.symbols:
            LOGGER.info(
                "Fetching %s/%s from %s (%s)",
                symbol,
                self.config.main_currency,
                self.config.exchange,
                timeframe,
            )
            data = downloader.fetch(
                self.config.exchange,
                self.config.main_currency,
                symbol,
                timeframe,
            )
            if "date" not in data.columns:
                raise ValueError(f"Fetched frame for {symbol} is missing a 'date' column")

            prepared = self._prepare_single(data, symbol)
            if prepared.empty:
                LOGGER.warning("No rows left for %s after preprocessing", symbol)
                continue
            result[symbol] = prepared

        if not result:
            raise ValueError("Failed to download data for any of the requested symbols")

        return result

    def _prepare_single(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        processed = df.copy()
        processed["date"] = pd.to_datetime(processed["date"], utc=True)
        processed = processed.sort_values("date").reset_index(drop=True)

        if self.config.start_date:
            processed = processed.loc[
                processed["date"] >= pd.Timestamp(self.config.start_date, tz="UTC")
            ]
        if self.config.end_date:
            processed = processed.loc[
                processed["date"] <= pd.Timestamp(self.config.end_date, tz="UTC")
            ]

        processed = processed[["date", "open", "high", "low", "close", "volume"]]
        processed = processed.rename(
            columns={
                "open": f"{symbol}_open",
                "high": f"{symbol}_high",
                "low": f"{symbol}_low",
                "close": f"{symbol}_close",
                "volume": f"{symbol}_volume",
            }
        )

        processed = processed.dropna(subset=[f"{symbol}_close"])
        processed = processed.reset_index(drop=True)
        return processed

    def _align_panel(self, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        common_dates = None
        for frame in frames.values():
            dates = set(frame["date"])
            common_dates = dates if common_dates is None else common_dates & dates
        if not common_dates:
            raise ValueError("No common trading dates across the requested symbols")

        common_index = sorted(common_dates)

        aligned: List[pd.DataFrame] = []
        for symbol, frame in frames.items():
            subset = frame[frame["date"].isin(common_index)].copy()
            subset = subset.sort_values("date").reset_index(drop=True)
            aligned.append(subset)

        panel = aligned[0]
        for extra in aligned[1:]:
            panel = panel.merge(extra, on="date", how="inner")

        panel = panel.sort_values("date").reset_index(drop=True)
        panel["date"] = pd.to_datetime(panel["date"], utc=True)
        panel.set_index("date", inplace=True)
        panel = panel.astype(float)
        return panel


def load_price_panel(config_dict: Dict) -> pd.DataFrame:
    """Convenience wrapper for YAML configs."""

    cfg = PriceDataConfig(
        symbols=config_dict["symbols"],
        exchange=config_dict.get("exchange", "Binance"),
        main_currency=config_dict.get("main_currency", "USDT"),
        time_frame=config_dict.get("time_frame", "1d"),
        start_date=config_dict.get("start_date"),
        end_date=config_dict.get("end_date"),
    )
    loader = MultiAssetOHLCFetcher(cfg)
    return loader.load()
