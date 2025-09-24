"""Streaming implementation of a KNN-based trading agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Deque

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
    """Streaming KNN agent that learns from incremental observations."""

    def __init__(
        self,
        strategy_config: StrategyConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        self.config = strategy_config
        self.sim_config = simulation_config

        # Streaming data buffers
        self._prices: Deque[float] = deque()
        self._returns: Deque[float] = deque()
        self._timestamps: Deque[pd.Timestamp] = deque()
        
        # Historical feature matrix and targets for training
        self._historical_features: List[np.ndarray] = []
        self._historical_targets: List[float] = []
        self._historical_classifications: List[int] = []
        
        # Current state
        self._last_price: Optional[float] = None
        self._current_features: Optional[np.ndarray] = None
        self._ewma_vol: Optional[float] = None
        self._step: int = 0
        self._history: List[Dict[str, float]] = []
        
        # Future return calculation buffer for training data
        self._future_return_buffer: Deque[float] = deque(maxlen=self.config.horizon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all streaming state for a new episode."""
        self._prices.clear()
        self._returns.clear()
        self._timestamps.clear()
        self._historical_features.clear()
        self._historical_targets.clear()
        self._historical_classifications.clear()
        self._future_return_buffer.clear()
        
        self._last_price = None
        self._current_features = None
        self._ewma_vol = None
        self._step = 0
        self._history = []

    def get_action(self, observation: np.ndarray) -> int:
        """Get action from streaming observation without future data access."""
        price = self._extract_price(observation)
        timestamp = pd.Timestamp.now()
        
        # Initialize tracking variables
        expected_return = 0.0
        side = 0
        neighbors = 0
        neighbor_weight = 0.0
        volatility = 0.0
        scaling = 1.0
        raw_weight = 0.0
        target_weight = 0.0
        
        if price is not None and price > 0:
            # Update streaming buffers
            self._update_streaming_data(price, timestamp)
            
            # Update volatility
            volatility = self._update_streaming_volatility()
            scaling = self._volatility_scaling(volatility)
            
            # Build current features if we have enough data
            if len(self._returns) >= self.config.window:
                self._current_features = self._build_current_features()
                
                # Evaluate KNN if we have training data and current features
                if len(self._historical_features) > 0 and self._current_features is not None:
                    knn_result = self._evaluate_knn()
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
        
        # Record history
        self._history.append({
            "step": self._step,
            "timestamp": timestamp,
            "price": price,
            "side": float(side),
            "expected_return": float(expected_return),
            "neighbors": float(neighbors),
            "neighbor_weight": float(neighbor_weight),
            "volatility": float(volatility),
            "vol_target": float(scaling),
            "raw_weight": float(raw_weight),
            "target_weight": float(target_weight),
            "action": float(action),
        })
        
        # Update training data after we have enough future returns
        self._update_training_data()
        
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
    # Streaming data management
    # ------------------------------------------------------------------
    def _update_streaming_data(self, price: float, timestamp: pd.Timestamp) -> None:
        """Update streaming buffers with new price observation."""
        self._prices.append(price)
        self._timestamps.append(timestamp)
        
        # Calculate log return if we have previous price
        if self._last_price is not None:
            log_return = np.log(price / self._last_price)
            self._returns.append(log_return)
            self._future_return_buffer.append(log_return)
        
        self._last_price = price
        
        # Keep only necessary history (window + horizon for training)
        max_history = self.config.window + self.config.horizon + 100  # Extra buffer
        while len(self._prices) > max_history:
            self._prices.popleft()
            self._timestamps.popleft()
        while len(self._returns) > max_history - 1:  # Returns are one less than prices
            self._returns.popleft()

    def _build_current_features(self) -> Optional[np.ndarray]:
        """Build feature vector from current return window."""
        if len(self._returns) < self.config.window:
            return None
        
        # Get the most recent window of returns
        recent_returns = list(self._returns)[-self.config.window:]
        features = np.array(recent_returns, dtype=float)
        
        if np.any(~np.isfinite(features)):
            return None
            
        return features

    def _update_training_data(self) -> None:
        """Update training data with completed feature-target pairs."""
        # We can only create training data if we have enough returns for both 
        # features and future targets
        min_required = self.config.window + self.config.horizon
        
        if len(self._returns) < min_required:
            return
            
        # Calculate how many new training samples we can create
        current_training_samples = len(self._historical_features)
        max_possible_samples = len(self._returns) - self.config.window - self.config.horizon + 1
        
        if max_possible_samples <= current_training_samples:
            return
            
        # Create new training samples
        returns_array = np.array(self._returns)
        
        for i in range(current_training_samples, max_possible_samples):
            # Feature window starts at position i
            feature_start = i
            feature_end = i + self.config.window
            
            # Target window starts after feature window
            target_start = feature_end
            target_end = target_start + self.config.horizon
            
            if target_end <= len(returns_array):
                features = returns_array[feature_start:feature_end].copy()
                future_returns = returns_array[target_start:target_end]
                
                if np.all(np.isfinite(features)) and np.all(np.isfinite(future_returns)):
                    target = float(np.sum(future_returns))
                    classification = int(np.sign(target))
                    
                    self._historical_features.append(features)
                    self._historical_targets.append(target)
                    self._historical_classifications.append(classification)

    # ------------------------------------------------------------------
    # KNN evaluation
    # ------------------------------------------------------------------
    def _evaluate_knn(self) -> Optional[Dict[str, float]]:
        """Evaluate KNN on current features using historical training data."""
        if self._current_features is None or len(self._historical_features) == 0:
            return None

        # Convert historical features to matrix
        train_x = np.array(self._historical_features)
        query = self._current_features

        # Standardize features
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0, ddof=0)
        std[std < 1e-12] = 1.0

        train_z = (train_x - mean) / std
        query_z = (query - mean) / std

        if not np.all(np.isfinite(query_z)):
            return None

        # Compute distances
        distances = self._compute_distances(train_z, query_z)
        finite_mask = np.isfinite(distances)
        
        if not np.any(finite_mask):
            return None

        # Filter valid distances
        valid_distances = distances[finite_mask]
        valid_indices = np.where(finite_mask)[0]

        # Select k nearest neighbors
        order = np.argsort(valid_distances)
        k = min(self.config.neighbors, len(order))
        selected = order[:k]

        selected_distances = valid_distances[selected]
        selected_indices = valid_indices[selected]

        # Calculate weights (no time decay in streaming version since all training data is historical)
        gaussian_weights = np.exp(-0.5 * (selected_distances / self.config.gaussian_bandwidth) ** 2)
        weights = gaussian_weights  # Could add time decay based on when training samples were added

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return None

        # Get targets for selected neighbors
        selected_targets = [self._historical_targets[i] for i in selected_indices]
        selected_classifications = [self._historical_classifications[i] for i in selected_indices]

        # Compute weighted predictions
        weighted_side = float(np.dot(weights, selected_classifications))
        expected_return = float(np.dot(weights, selected_targets) / weight_sum)

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


    # ------------------------------------------------------------------
    # Risk management
    # ------------------------------------------------------------------
    def _update_streaming_volatility(self) -> float:
        """Update EWMA volatility with the most recent return."""
        if len(self._returns) == 0:
            if self._ewma_vol is None:
                self._ewma_vol = self.config.volatility_floor
            return float(self._ewma_vol)

        # Get most recent return
        r_t = float(self._returns[-1])
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
        """Extract price from observation array."""
        if observation is None:
            return None
        array = np.asarray(observation, dtype=float)
        if array.size == 0:
            return None
        price = float(array.reshape(-1)[-1])
        if price <= 0:
            return None
        return price
        
    # ------------------------------------------------------------------
    # Agent interface methods (for compatibility with TensorTrade Agent)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the strategy state to a file."""
        import pickle
        state = {
            'config': self.config,
            'sim_config': self.sim_config,
            'historical_features': self._historical_features,
            'historical_targets': self._historical_targets,
            'historical_classifications': self._historical_classifications,
            'ewma_vol': self._ewma_vol,
            'step': self._step
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def restore(self, path: str) -> None:
        """Restore the strategy state from a file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.sim_config = state['sim_config']
        self._historical_features = state['historical_features']
        self._historical_targets = state['historical_targets']
        self._historical_classifications = state['historical_classifications']
        self._ewma_vol = state['ewma_vol']
        self._step = state['step']


def build_strategy(
    strategy_cfg: Dict,
    simulation_cfg: Dict,
) -> KNNStrategy:
    """Build streaming KNN strategy without pre-loaded data."""
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
    )

    simulation = SimulationConfig(
        tolerance=float(simulation_cfg.get("tolerance", 1e-6)),
        warmup_steps=int(simulation_cfg.get("warmup_steps", 0)),
        observation_start=simulation_cfg.get("observation_start"),
    )

    return KNNStrategy(strategy, simulation)
