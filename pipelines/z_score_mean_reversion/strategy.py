"""Streaming implementation of the z-score mean reversion strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Hyperparameters of the mean reversion strategy."""

    window: int
    entry_threshold: float
    epsilon: float
    sensitivity: float
    max_weight: float
    ewma_lambda: float
    target_volatility: float
    volatility_floor: float


@dataclass
class SimulationConfig:
    """Auxiliary knobs for the deterministic simulation."""

    tolerance: float = 1e-6
    include_discrete_signal: bool = True
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")


class ZScoreMeanReversionStrategy:
    """Generates actions sequentially from streaming market observations."""

    def __init__(
        self,
        strategy_config: StrategyConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        if strategy_config.window <= 1:
            raise ValueError("window must be greater than 1")
        if strategy_config.entry_threshold <= 0:
            raise ValueError("entry_threshold must be positive")
        if strategy_config.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if strategy_config.entry_threshold <= strategy_config.epsilon:
            raise ValueError("entry_threshold must exceed epsilon")
        if not 0 < strategy_config.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 < strategy_config.ewma_lambda < 1:
            raise ValueError("ewma_lambda must be in (0, 1)")
        if strategy_config.target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if strategy_config.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")
        if strategy_config.sensitivity <= 0:
            raise ValueError("sensitivity must be positive")

        self.config = strategy_config
        self.sim_config = simulation_config

        self._log_prices: Deque[float] = deque(maxlen=self.config.window)
        self._last_log_price: Optional[float] = None
        self._ewma_vol: Optional[float] = None
        self._discrete_side: int = 0
        self._step: int = 0
        self._history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all running statistics before starting a new episode."""

        self._log_prices.clear()
        self._last_log_price = None
        self._ewma_vol = None
        self._discrete_side = 0
        self._step = 0
        self._history = []

    def get_action(self, observation: np.ndarray) -> int:
        """Return the next discrete action given the latest observation."""

        price = self._extract_price(observation)
        z_score = 0.0
        raw_weight = 0.0
        target_weight = 0.0
        scaling = 0.0
        volatility = self._ewma_vol or 0.0
        discrete_side = self._discrete_side
        log_price: Optional[float] = None

        if price is not None and price > 0:
            log_price = float(np.log(price))
            self._log_prices.append(log_price)

            z_score = self._compute_z_score()
            volatility = self._update_ewma(log_price)
            scaling = self._compute_scaling(volatility)
            discrete_side = self._update_discrete_side(z_score)
            raw_weight = self._compute_raw_weight(z_score, discrete_side)
            target_weight = raw_weight * scaling

        action = self._decide_action(target_weight)

        self._history.append(
            {
                "step": self._step,
                "price": price,
                "log_price": log_price,
                "z_score": z_score,
                "raw_weight": raw_weight,
                "target_weight": target_weight,
                "volatility": volatility,
                "scaling": scaling,
                "discrete_side": discrete_side,
                "action": action,
            }
        )

        self._discrete_side = discrete_side
        self._step += 1
        return action

    def history_frame(self) -> pd.DataFrame:
        """Return the recorded indicator history as a DataFrame."""

        if not self._history:
            columns = [
                "step",
                "price",
                "log_price",
                "z_score",
                "raw_weight",
                "target_weight",
                "volatility",
                "scaling",
                "discrete_side",
                "action",
            ]
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_price(self, observation: np.ndarray) -> Optional[float]:
        if observation is None:
            return None
        array = np.asarray(observation, dtype=float)
        if array.size == 0:
            return None
        price = float(array.reshape(-1)[-1])
        if price <= 0:
            return None
        return price

    def _compute_z_score(self) -> float:
        if len(self._log_prices) < self.config.window:
            return 0.0
        window_array = np.fromiter(self._log_prices, dtype=float)
        mean = window_array.mean()
        std = window_array.std(ddof=1)
        if std <= 0:
            return 0.0
        return float((window_array[-1] - mean) / std)

    def _update_ewma(self, log_price: float) -> float:
        if self._last_log_price is None:
            self._last_log_price = log_price
            if self._ewma_vol is None:
                if len(self._log_prices) > 1:
                    returns = np.diff(np.fromiter(self._log_prices, dtype=float))
                    std = returns.std(ddof=1)
                    if np.isnan(std) or std <= 0:
                        std = self.config.volatility_floor
                    self._ewma_vol = float(std)
                else:
                    self._ewma_vol = float(self.config.volatility_floor)
            return float(self._ewma_vol)

        r_t = float(log_price - self._last_log_price)
        self._last_log_price = log_price

        prev = float(self._ewma_vol) if self._ewma_vol is not None else abs(r_t)
        lam = self.config.ewma_lambda
        updated = np.sqrt(max(lam * prev ** 2 + (1 - lam) * r_t ** 2, 0.0))
        self._ewma_vol = float(updated)
        return self._ewma_vol

    def _compute_scaling(self, volatility: float) -> float:
        denom = float(volatility) + float(self.config.volatility_floor)
        if denom <= 0:
            return 1.0
        alpha = self.config.target_volatility / denom
        return float(min(1.0, max(alpha, 0.0)))

    def _update_discrete_side(self, z_score: float) -> int:
        side = self._discrete_side
        if z_score <= -self.config.entry_threshold:
            side = 1
        elif z_score >= self.config.entry_threshold:
            side = -1
        elif abs(z_score) < self.config.epsilon:
            side = 0
        elif side == 1 and z_score >= -self.config.epsilon:
            side = 0
        elif side == -1 and z_score <= self.config.epsilon:
            side = 0
        return side

    def _compute_raw_weight(self, z_score: float, discrete_side: int) -> float:
        weight = -self.config.sensitivity * z_score
        weight = float(np.clip(weight, -self.config.max_weight, self.config.max_weight))
        if self.sim_config.include_discrete_signal and discrete_side == 0:
            return 0.0
        if not self.sim_config.include_discrete_signal and abs(z_score) < self.config.epsilon:
            return 0.0
        return weight

    def _decide_action(self, target_weight: float) -> int:
        if target_weight > self.sim_config.tolerance:
            return -1
        if target_weight < -self.sim_config.tolerance:
            return 1
        return 0


def build_strategy(config: Dict, simulation: Dict) -> ZScoreMeanReversionStrategy:
    """Factory helper that constructs the strategy from dictionaries."""

    strategy_cfg = StrategyConfig(
        window=int(config["window"]),
        entry_threshold=float(config["entry_threshold"]),
        epsilon=float(config["epsilon"]),
        sensitivity=float(config["sensitivity"]),
        max_weight=float(config["max_weight"]),
        ewma_lambda=float(config["ewma_lambda"]),
        target_volatility=float(config["target_volatility"]),
        volatility_floor=float(config["volatility_floor"]),
    )

    sim_cfg = SimulationConfig(
        tolerance=float(simulation.get("tolerance", 1e-6)),
        include_discrete_signal=bool(simulation.get("include_discrete_signal", True)),
        warmup_steps=int(simulation.get("warmup_steps", 0)),
    )

    return ZScoreMeanReversionStrategy(strategy_cfg, sim_cfg)
