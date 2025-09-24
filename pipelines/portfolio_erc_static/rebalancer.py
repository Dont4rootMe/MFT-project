"""Rebalancing logic for translating target weights into TensorTrade orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from tensortrade.oms.orders import TradeSide
from tensortrade.oms.instruments import registry
from tensortrade.oms.wallets import Portfolio, Wallet

LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    base_symbol: str
    symbols: Iterable[str]
    commission_buy: float = 0.0
    commission_sell: float = 0.0
    slippage_bps: float = 0.0
    delta: Mapping[str, float] = field(default_factory=dict)
    lot_size: Mapping[str, float] = field(default_factory=dict)
    tick_size: Mapping[str, float] = field(default_factory=dict)

    def delta_for(self, symbol: str) -> float:
        return float(self.delta.get(symbol, self.delta.get("default", 0.0)))

    def lot_for(self, symbol: str) -> Optional[float]:
        return self.lot_size.get(symbol, self.lot_size.get("default"))

    def tick_for(self, symbol: str) -> Optional[float]:
        return self.tick_size.get(symbol, self.tick_size.get("default"))


@dataclass
class TradeInstruction:
    symbol: str
    side: TradeSide
    amount: float  # Trade quantity in units of the asset.
    price: float


class PortfolioRebalancer:
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.symbols = list(config.symbols)

    # ------------------------------------------------------------------
    def compute_orders(
        self,
        portfolio: Portfolio,
        close_prices: pd.Series,
        next_open_prices: pd.Series,
        target_weights: np.ndarray,
    ) -> List[TradeInstruction]:
        if len(target_weights) != len(self.symbols):
            raise ValueError("Target weights dimension mismatch")

        cash_wallet = self._get_wallet(portfolio, self.config.base_symbol)
        cash_balance = float(cash_wallet.balance.as_float())

        # Gather holdings and current weights
        quantities = {}
        current_values = {}
        for symbol in self.symbols:
            wallet = self._get_wallet(portfolio, symbol)
            qty = float(wallet.balance.as_float())
            price = float(close_prices[f"{symbol}_close"])
            quantities[symbol] = qty
            current_values[symbol] = qty * price

        net_worth = cash_balance + sum(current_values.values())
        if net_worth <= 0:
            LOGGER.warning("Net worth is non-positive; skipping rebalance")
            return []

        current_weights = {sym: (current_values[sym] / net_worth if net_worth else 0.0) for sym in self.symbols}

        target = self._apply_no_trade_bands(target_weights, current_weights)
        target_values = {sym: float(target[i]) * net_worth for i, sym in enumerate(self.symbols)}

        # Compute desired quantity changes using next open prices
        deltas_value = {sym: target_values[sym] - current_values[sym] for sym in self.symbols}

        prices_open = {
            sym: self._round_price(float(next_open_prices[f"{sym}_open"]), sym)
            for sym in self.symbols
        }

        buy_plan: Dict[str, float] = {}
        sell_plan: Dict[str, float] = {}

        for sym in self.symbols:
            price = prices_open[sym]
            if price <= 0:
                LOGGER.warning("Price for %s is non-positive; skipping", sym)
                continue
            diff = deltas_value[sym]
            lot = self.config.lot_for(sym)
            if diff > 0:
                qty = diff / price
                qty = self._round_quantity(qty, lot)
                if qty > 0:
                    buy_plan[sym] = qty
            elif diff < 0:
                qty = min(quantities[sym], abs(diff) / price)
                qty = self._round_quantity(qty, lot)
                if qty > 0:
                    sell_plan[sym] = qty

        # Compute budgets
        slip = self.config.slippage_bps / 10000.0
        total_buy_cost = sum(
            buy_plan[sym]
            * prices_open[sym]
            * (1.0 + self.config.commission_buy + slip)
            for sym in buy_plan
        )
        total_sell_proceeds = sum(
            sell_plan[sym]
            * prices_open[sym]
            * (1.0 - self.config.commission_sell - slip)
            for sym in sell_plan
        )
        available_cash = cash_balance + total_sell_proceeds

        if total_buy_cost > 0 and available_cash <= 0:
            LOGGER.warning("Insufficient cash to finance purchases; skipping buys")
            buy_plan = {}
            total_buy_cost = 0.0

        scale = 1.0
        if total_buy_cost > available_cash > 0:
            scale = available_cash / total_buy_cost
            # LOGGER.info("Scaling buy orders by %.4f to respect budget", scale)

        instructions: List[TradeInstruction] = []

        realised_proceeds = 0.0
        for sym, qty in sell_plan.items():
            if qty <= 0:
                continue
            price = prices_open[sym]
            instructions.append(TradeInstruction(sym, TradeSide.SELL, qty, price))
            realised_proceeds += qty * price * (1.0 - self.config.commission_sell - slip)

        remaining_cash = cash_balance + realised_proceeds

        for sym, qty in buy_plan.items():
            if qty <= 0:
                continue
            price = prices_open[sym]
            scaled_qty = qty * scale
            scaled_qty = self._round_quantity(
                scaled_qty, self.config.lot_for(sym)
            )
            if scaled_qty <= 0:
                continue

            gross_cost = scaled_qty * price * (1.0 + self.config.commission_buy + slip)
            if gross_cost > remaining_cash:
                affordable_qty = self._round_quantity(
                    remaining_cash
                    / (price * (1.0 + self.config.commission_buy + slip)),
                    self.config.lot_for(sym),
                )
                if affordable_qty <= 0:
                    LOGGER.debug(
                        "Skipping %s buy: insufficient cash after rounding", sym
                    )
                    continue
                scaled_qty = affordable_qty
                gross_cost = scaled_qty * price * (
                    1.0 + self.config.commission_buy + slip
                )

            remaining_cash = max(0.0, remaining_cash - gross_cost)
            instructions.append(TradeInstruction(sym, TradeSide.BUY, scaled_qty, price))

        return instructions

    # ------------------------------------------------------------------
    def _apply_no_trade_bands(self, target_weights: np.ndarray, current_weights: Dict[str, float]) -> np.ndarray:
        target = np.array(target_weights, dtype=float)
        fixed = []
        for i, sym in enumerate(self.symbols):
            band = self.config.delta_for(sym)
            if abs(target[i] - current_weights.get(sym, 0.0)) < band:
                target[i] = current_weights.get(sym, 0.0)
                fixed.append(i)

        fixed_sum = float(target[fixed].sum()) if fixed else 0.0
        remaining = [i for i in range(len(self.symbols)) if i not in fixed]
        remaining_target_sum = float(target[remaining].sum()) if remaining else 0.0
        remaining_budget = max(0.0, 1.0 - fixed_sum)

        if remaining and remaining_target_sum > 0:
            for i in remaining:
                target[i] = target[i] / remaining_target_sum * remaining_budget
        elif remaining:
            share = remaining_budget / len(remaining)
            for i in remaining:
                target[i] = share

        if target.sum() == 0:
            target = np.ones_like(target) / len(target)
        else:
            target = np.maximum(target, 0)
            target = target / target.sum()
        return target

    def _round_quantity(self, quantity: float, lot: Optional[float]) -> float:
        if lot is None or lot <= 0:
            return float(max(quantity, 0.0))
        if quantity <= 0:
            return 0.0
        return float(np.floor(quantity / lot) * lot)

    def _round_price(self, price: float, symbol: str) -> float:
        tick = self.config.tick_for(symbol)
        if tick is None or tick <= 0:
            return price
        return float(np.round(price / tick) * tick)

    def _get_wallet(self, portfolio: Portfolio, symbol: str) -> Wallet:
        instrument = registry.get(symbol)
        if instrument is None:
            raise ValueError(f"Instrument {symbol} is not registered")
        exchange = portfolio.exchange_pairs[0].exchange
        return portfolio.get_wallet(exchange.id, instrument)
