"""Equal Risk Contribution solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

LOGGER = logging.getLogger(__name__)


@dataclass
class ERCSolverConfig:
    tolerance: float = 1e-5
    max_iterations: int = 500
    min_weight: float = 0.0
    max_weight: float = 1.0


class ERCSolver:
    """Iterative multiplicative solver for long-only ERC portfolios."""

    def __init__(self, config: ERCSolverConfig):
        self.config = config

    def solve(
        self,
        covariance: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        n_assets = covariance.shape[0]
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance matrix must be square")

        w = (
            np.ones(n_assets) / n_assets
            if initial_weights is None
            else np.array(initial_weights, dtype=float)
        )
        w = self._project(w)

        info = {
            "converged": False,
            "iterations": 0,
            "tolerance": self.config.tolerance,
        }

        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets

        def objective(weights: np.ndarray) -> float:
            cov_w = covariance @ weights
            port_var = float(weights @ cov_w)
            rc = weights * cov_w
            target = port_var / n_assets
            return float(np.sum((rc - target) ** 2))

        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
        result = minimize(
            objective,
            w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.config.max_iterations, "ftol": self.config.tolerance},
        )

        if not result.success:
            LOGGER.warning("SLSQP failed to converge: %s", result.message)

        w = self._project(result.x)
        portfolio_var = float(w @ covariance @ w)
        marginal_contrib = covariance @ w
        risk_contrib = w * marginal_contrib
        target = portfolio_var / n_assets
        max_err = float(np.max(np.abs(risk_contrib - target)) / portfolio_var) if portfolio_var > 0 else float("inf")

        info["converged"] = bool(result.success and max_err < self.config.tolerance)
        info["iterations"] = result.nit if hasattr(result, "nit") else self.config.max_iterations
        info["risk_contrib"] = risk_contrib
        info["portfolio_variance"] = portfolio_var
        info["max_error"] = max_err
        return w, info

    def _project(self, weights: np.ndarray) -> np.ndarray:
        w = np.array(weights, dtype=float)
        w = np.clip(w, self.config.min_weight, self.config.max_weight)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        return w
