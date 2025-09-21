"""Rebalance scheduling utilities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RebalanceScheduler:
    warm_start_days: int
    rebalance_period_days: int
    enable_online_rebalance: bool = True

    _has_rebalanced: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.warm_start_days < 1:
            raise ValueError("warm_start_days must be positive")
        if self.rebalance_period_days < 1:
            raise ValueError("rebalance_period_days must be positive")

    def should_rebalance(self, day_index: int) -> bool:
        """Return True if a rebalance decision should be generated.

        Parameters
        ----------
        day_index : int
            Index of the day that has just been observed (0-based). The
            scheduler assumes that trades decided on day ``day_index`` will be
            executed on ``day_index + 1``.
        """

        observed_days = day_index + 1
        if observed_days < self.warm_start_days:
            return False

        if not self.enable_online_rebalance and self._has_rebalanced:
            return False

        relative = observed_days - self.warm_start_days
        should_rebalance = relative % self.rebalance_period_days == 0

        if should_rebalance and not self.enable_online_rebalance:
            self._has_rebalanced = True

        return should_rebalance
