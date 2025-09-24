"""Custom action scheme for scheduled portfolio rebalancing."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import List, Optional

from gymnasium.spaces import Discrete

from tensortrade.env.default.actions import TensorTradeActionScheme
from tensortrade.oms.orders import TradeSide
from tensortrade.oms.orders.create import market_order
from tensortrade.oms.wallets import Portfolio

from pipelines.portfolio_erc_static.rebalancer import TradeInstruction
LOGGER = logging.getLogger(__name__)


class ScheduledOrdersActionScheme(TensorTradeActionScheme):
    """Action scheme that executes pre-built TensorTrade orders on demand."""

    registered_name = "scheduled_orders"

    def __init__(self, portfolio: Portfolio):
        super().__init__()
        self.portfolio = portfolio
        self._instructions: List[TradeInstruction] = []
        self._pairs = {p.pair.quote.symbol: p for p in portfolio.exchange_pairs}
        self._cash_wallet = portfolio.get_wallet(
            portfolio.exchange_pairs[0].exchange.id, portfolio.base_instrument
        )

    @property
    def action_space(self):  # type: ignore[override]
        return Discrete(1)

    def schedule(self, instructions: Optional[List[TradeInstruction]] = None) -> None:
        self._instructions = list(instructions or [])

    def clear(self) -> None:
        self._instructions = []

    def perform(self, env, action):  # type: ignore[override]
        executed = []
        for instr in self._instructions:
            pair = self._pairs.get(instr.symbol)
            if pair is None:
                LOGGER.warning("No exchange pair found for %s", instr.symbol)
                continue
            price = float(instr.price)
            if instr.side == TradeSide.SELL:
                asset_wallet = self.portfolio.get_wallet(
                    pair.exchange.id, pair.pair.base
                )
                available_qty = float(asset_wallet.balance.as_float())
                quantity = min(instr.amount, available_qty)
                if quantity <= 0:
                    continue
                order = market_order(
                    TradeSide.SELL, pair, price, quantity, self.portfolio
                )
                order.price = Decimal(str(price))
                self.broker.submit(order)
                result = self.broker.update()
                if result:
                    executed.extend(result)
            else:
                cash_wallet = self._cash_wallet
                available_cash = float(cash_wallet.balance.as_float())
                if price <= 0:
                    LOGGER.warning("Non-positive price for %s", instr.symbol)
                    continue
                max_affordable = available_cash / price if price > 0 else 0.0
                quantity = min(instr.amount, max_affordable)
                if quantity <= 0:
                    continue
                notional = quantity * price
                order = market_order(
                    TradeSide.BUY, pair, price, notional, self.portfolio
                )
                order.price = Decimal(str(price))
                self.broker.submit(order)
                result = self.broker.update()
                if result:
                    executed.extend(result)

        self._instructions = []
        return executed

    def get_orders(self, action, portfolio: Portfolio):  # pragma: no cover - unused hook
        return []
