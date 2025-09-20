"""Streaming implementation of a KNN-based trading policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


_VALID_WINDOWS = {24, 36, 48, 72}
_VALID_NEIGHBORS = {3, 5, 7, 11, 15}
_VALID_HORIZONS = {1, 4, 24}
_VALID_METRICS = {"euclidean", "cosine"}
_VALID_DECAYS = {1.0, 0.99, 0.97}


@dataclass
class StrategyConfig:
    mode: str
    window: int
    horizon: int
    neighbors: int
    metric: str
    gaussian_bandwidth: float
    time_decay: float
    beta: float
    max_weight: float
    ewma_lambda: float
    target_volatility: float
    volatility_floor: float
    fee_rate: float
    slippage_rate: float

    def __post_init__(self) -> None:
        mode_normalized = self.mode.lower()
        if mode_normalized not in {"classification", "regression"}:
            raise ValueError("mode must be either 'classification' or 'regression'")
        self.mode = mode_normalized

        if self.window not in _VALID_WINDOWS:
            raise ValueError(f"window must be one of {_VALID_WINDOWS}")
        if self.horizon not in _VALID_HORIZONS:
            raise ValueError(f"horizon must be one of {_VALID_HORIZONS}")
        if self.neighbors not in _VALID_NEIGHBORS:
            raise ValueError(f"neighbors must be one of {_VALID_NEIGHBORS}")
        if self.metric not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {_VALID_METRICS}")
        if self.gaussian_bandwidth <= 0:
            raise ValueError("gaussian_bandwidth must be positive")
        if self.time_decay not in _VALID_DECAYS:
            raise ValueError(f"time_decay must be one of {_VALID_DECAYS}")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 < self.ewma_lambda < 1:
            raise ValueError("ewma_lambda must be in (0, 1)")
        if self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")
        if self.fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        if self.slippage_rate < 0:
            raise ValueError("slippage_rate must be non-negative")


@dataclass
class SimulationConfig:
    tolerance: float = 1e-6
    warmup_steps: int = 0
    observation_start: Optional[int] = None

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.observation_start is not None and self.observation_start < 0:
            raise ValueError("observation_start must be non-negative when provided")


class KNNStrategy:
    """Sequential KNN learner operating on rolling log-return windows."""

    def __init__(
        self,
        strategy_config: StrategyConfig,
        simulation_config: SimulationConfig,
        price_frame: pd.DataFrame,
    ) -> None:
        self.config = strategy_config
        self.sim_config = simulation_config

        if "close" not in price_frame.columns:
            raise ValueError("price_frame must contain a 'close' column")
        if "log_return" not in price_frame.columns:
            raise ValueError("price_frame must contain a 'log_return' column")

        self._timestamps = pd.to_datetime(price_frame["timestamp"].to_numpy())
        self._prices = price_frame["close"].to_numpy(dtype=float)
        self._returns = price_frame["log_return"].to_numpy(dtype=float)

        if np.any(~np.isfinite(self._prices)):
            raise ValueError("price data contains non-finite values")

        self._feature_matrix = self._build_feature_matrix(self._returns, self.config.window)
        self._future_sum = self._build_future_returns(self._returns, self.config.horizon)
        self._classification_target = np.sign(self._future_sum)

        valid_features = ~np.isnan(self._feature_matrix).any(axis=1)
        valid_targets = ~np.isnan(self._future_sum)
        self._train_mask = valid_features & valid_targets
        self._train_indices = np.where(self._train_mask)[0]

        feature_start = max(self.config.window - 1, 0)
        obs_start = self.sim_config.observation_start or 0
        self._start_index = max(feature_start, obs_start)

        self._history: List[Dict[str, float]] = []
        self._current_action: int = 0
        self._current_index: int = self._start_index
        self._ewma_vol: Optional[float] = None
        self._step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history = []
        self._current_action = 0
        self._current_index = self._start_index
        self._ewma_vol = None
        self._step = 0

    def get_action(self, observation: np.ndarray) -> int:
        price = self._extract_price(observation)
        timestamp = self._get_timestamp(self._current_index)
        expected_return = 0.0
        side = 0
        neighbors = 0
        neighbor_weight = 0.0
        volatility = self._update_volatility(self._current_index)
        scaling = self._volatility_scaling(volatility)
        raw_weight = 0.0
        target_weight = 0.0

        idx = self._current_index
        if idx < len(self._prices) and idx >= self.config.window - 1:
            knn_result = self._evaluate_knn(idx)
            if knn_result is not None:
                neighbors = knn_result["neighbors"]
                neighbor_weight = knn_result["weight_sum"]
                if self.config.mode == "classification":
                    side = self._decide_side(knn_result["weighted_side"])
                else:
                    expected_return = knn_result["expected_return"]
                    raw_weight = self._compute_raw_weight(expected_return, volatility)
                    target_weight = raw_weight * scaling

        action = self._decide_action(side, target_weight)

        self._history.append(
            {
                "step": self._step,
                "timestamp": timestamp,
                "price": price,
                "index": float(idx),
                "side": float(side),
                "expected_return": float(expected_return),
                "neighbors": float(neighbors),
                "neighbor_weight": float(neighbor_weight),
                "volatility": float(volatility),
                "vol_target": float(scaling),
                "raw_weight": float(raw_weight),
                "target_weight": float(target_weight),
                "action": float(action),
            }
        )

        self._current_action = action
        self._current_index += 1
        self._step += 1
        return action

    def history_frame(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame(
                columns=[
                    "step",
                    "timestamp",
                    "price",
                    "index",
                    "side",
                    "expected_return",
                    "neighbors",
                    "neighbor_weight",
                    "volatility",
                    "vol_target",
                    "raw_weight",
                    "target_weight",
                    "action",
                ]
            )
        return pd.DataFrame(self._history)

    # ------------------------------------------------------------------
    # KNN helpers
    # ------------------------------------------------------------------
    def _evaluate_knn(self, index: int) -> Optional[Dict[str, float]]:
        if index >= len(self._prices):
            return None
        if np.isnan(self._feature_matrix[index]).any():
            return None

        train_indices = self._training_indices_up_to(index - self.config.horizon)
        if len(train_indices) == 0:
            return None

        query = self._feature_matrix[index]
        train_x = self._feature_matrix[train_indices]

        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0, ddof=0)
        std[std < 1e-12] = 1.0

        train_z = (train_x - mean) / std
        query_z = (query - mean) / std

        if not np.all(np.isfinite(query_z)):
            return None

        distances = self._compute_distances(train_z, query_z)
        finite_mask = np.isfinite(distances)
        if not np.any(finite_mask):
            return None

        train_indices = train_indices[finite_mask]
        distances = distances[finite_mask]
        train_z = train_z[finite_mask]

        order = np.argsort(distances)
        k = min(self.config.neighbors, len(order))
        selected = order[:k]

        distances = distances[selected]
        train_indices = train_indices[selected]

        gaussian_weights = np.exp(-0.5 * (distances / self.config.gaussian_bandwidth) ** 2)
        decay = np.power(self.config.time_decay, index - train_indices)
        weights = gaussian_weights * decay

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return None

        future_sum = self._future_sum[train_indices]
        classification_target = self._classification_target[train_indices]

        weighted_side = float(np.dot(weights, classification_target))
        expected_return = float(np.dot(weights, future_sum) / weight_sum)

        return {
            "neighbors": float(k),
            "weight_sum": weight_sum,
            "weighted_side": weighted_side,
            "expected_return": expected_return,
        }

    def _compute_distances(self, train_z: np.ndarray, query_z: np.ndarray) -> np.ndarray:
        if self.config.metric == "euclidean":
            diff = train_z - query_z
            return np.linalg.norm(diff, axis=1)

        # cosine distance
        query_norm = np.linalg.norm(query_z)
        train_norms = np.linalg.norm(train_z, axis=1)
        denom = train_norms * query_norm
        denom[denom == 0] = np.inf
        cosine_similarity = np.einsum("ij,j->i", train_z, query_z) / denom
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        return 1.0 - cosine_similarity

    def _training_indices_up_to(self, limit: int) -> np.ndarray:
        if limit < 0 or len(self._train_indices) == 0:
            return np.array([], dtype=int)
        pos = np.searchsorted(self._train_indices, limit, side="right")
        return self._train_indices[:pos]

    # ------------------------------------------------------------------
    # Risk management
    # ------------------------------------------------------------------
    def _update_volatility(self, index: int) -> float:
        if index <= 0 or index >= len(self._returns):
            if self._ewma_vol is None:
                self._ewma_vol = self.config.volatility_floor
            return float(self._ewma_vol)

        r_t = float(self._returns[index])
        if not np.isfinite(r_t):
            r_t = 0.0

        if self._ewma_vol is None:
            self._ewma_vol = max(abs(r_t), self.config.volatility_floor)
        else:
            lam = self.config.ewma_lambda
            self._ewma_vol = float(np.sqrt(lam * self._ewma_vol**2 + (1 - lam) * r_t**2))
            if self._ewma_vol < self.config.volatility_floor:
                self._ewma_vol = self.config.volatility_floor
        return float(self._ewma_vol)

    def _volatility_scaling(self, volatility: float) -> float:
        if volatility <= 0:
            return 1.0
        return float(min(1.0, self.config.target_volatility / (volatility + self.config.volatility_floor)))

    def _compute_raw_weight(self, expected_return: float, volatility: float) -> float:
        denominator = max(volatility, self.config.volatility_floor)
        weight = self.config.beta * expected_return / denominator
        return float(np.clip(weight, -self.config.max_weight, self.config.max_weight))

    def _decide_side(self, weighted_side: float) -> int:
        if weighted_side > self.sim_config.tolerance:
            return 1
        if weighted_side < -self.sim_config.tolerance:
            return -1
        return 0

    def _decide_action(self, side: int, target_weight: float) -> int:
        if self.config.mode == "classification":
            if side > 0:
                return 1
            if side < 0:
                return -1
            return 0

        if target_weight > self.sim_config.tolerance:
            return 1
        if target_weight < -self.sim_config.tolerance:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Utility methods
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

    def _get_timestamp(self, index: int) -> Optional[pd.Timestamp]:
        if index < len(self._timestamps):
            return pd.Timestamp(self._timestamps[index])
        return None

    @staticmethod
    def _build_feature_matrix(returns: np.ndarray, window: int) -> np.ndarray:
        n = len(returns)
        features = np.full((n, window), np.nan, dtype=float)
        if n >= window:
            windows = np.lib.stride_tricks.sliding_window_view(returns, window)
            features[window - 1 :, :] = windows
        return features

    @staticmethod
    def _build_future_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
        n = len(returns)
        future = np.full(n, np.nan, dtype=float)
        if horizon <= 0 or n <= horizon:
            return future
        windows = np.lib.stride_tricks.sliding_window_view(returns[1:], horizon)
        sums = windows.sum(axis=1)
        future[: len(sums)] = sums
        return future


def build_strategy(
    strategy_cfg: Dict,
    simulation_cfg: Dict,
    price_frame: pd.DataFrame,
    observation_start: int,
) -> KNNStrategy:
    strategy = StrategyConfig(
        mode=strategy_cfg["mode"],
        window=int(strategy_cfg["window"]),
        horizon=int(strategy_cfg["horizon"]),
        neighbors=int(strategy_cfg["neighbors"]),
        metric=str(strategy_cfg.get("metric", "euclidean").lower()),
        gaussian_bandwidth=float(strategy_cfg.get("gaussian_bandwidth", 1.0)),
        time_decay=float(strategy_cfg.get("time_decay", 1.0)),
        beta=float(strategy_cfg.get("beta", 1.0)),
        max_weight=float(strategy_cfg.get("max_weight", 1.0)),
        ewma_lambda=float(strategy_cfg.get("ewma_lambda", 0.94)),
        target_volatility=float(strategy_cfg.get("target_volatility", 0.02)),
        volatility_floor=float(strategy_cfg.get("volatility_floor", 1e-6)),
        fee_rate=float(strategy_cfg.get("fee_rate", 0.0)),
        slippage_rate=float(strategy_cfg.get("slippage_rate", 0.0)),
    )

    simulation = SimulationConfig(
        tolerance=float(simulation_cfg.get("tolerance", 1e-6)),
        warmup_steps=int(simulation_cfg.get("warmup_steps", 0)),
        observation_start=int(simulation_cfg.get("observation_start", observation_start)),
    )

    return KNNStrategy(strategy, simulation, price_frame)
