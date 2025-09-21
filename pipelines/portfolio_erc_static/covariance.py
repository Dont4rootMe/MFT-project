"""Covariance estimation utilities for the ERC pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class LedoitWolfConfig:
    window: int
    min_history: int


class LedoitWolfShrinkage:
    """Estimate a Ledoit–Wolf shrunk covariance matrix.

    The implementation targets the diagonal of the sample covariance matrix
    as described in the original paper, which coincides with shrinking the
    off-diagonal entries towards zero while retaining individual asset
    variances.
    """

    def __init__(self, config: LedoitWolfConfig):
        if config.window <= 1:
            raise ValueError("window must be greater than 1")
        if config.min_history < config.window:
            LOGGER.warning(
                "min_history (%s) is smaller than window (%s); upgrading",
                config.min_history,
                config.window,
            )
            config.min_history = config.window
        self.config = config

    def estimate(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        if len(returns) < self.config.min_history:
            raise ValueError(
                f"Need at least {self.config.min_history} observations for covariance estimation"
            )

        windowed = returns.tail(self.config.window)
        X = windowed.to_numpy(dtype=float)
        X = np.asarray(X)
        X = X - X.mean(axis=0, keepdims=True)
        T = X.shape[0]
        if T <= 1:
            raise ValueError("Not enough observations in window after centering")

        sample_cov = np.cov(X, rowvar=False, ddof=1)
        if not np.all(np.isfinite(sample_cov)):
            raise ValueError("Sample covariance contains NaNs or infs")

        target = np.diag(np.diag(sample_cov))

        # Ledoit-Wolf shrinkage intensity towards the diagonal target.
        centered_outer = np.einsum("ti,tj->tij", X, X)
        centered_outer -= sample_cov
        phi_matrix = np.mean(centered_outer**2, axis=0)
        phi_off = phi_matrix - np.diag(np.diag(phi_matrix))
        phi = float(phi_off.sum())

        diff = sample_cov - target
        rho = float(np.sum(diff**2))
        if rho == 0:
            shrinkage = 0.0
        else:
            shrinkage = max(0.0, min(1.0, phi / rho))

        shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * target
        return shrunk, shrinkage
