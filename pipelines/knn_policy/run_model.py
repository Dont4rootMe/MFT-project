"""Entrypoint for executing the streaming KNN baseline."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

import tensortrade.env.default as default
from tensortrade.env.default import actions as action_api, rewards as reward_api
from tensortrade.env.default.renderers import construct_renderers
from tensortrade.feed.core import DataFeed, NameSpace, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import Instrument, registry
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

from pipelines.knn_policy.data_loader import build_data_loader
from pipelines.knn_policy.strategy import KNNStrategy, build_strategy

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional.
    torch = None

LOGGER = logging.getLogger("knn_baseline")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).with_name("config.yaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_execution_service(slippage_rate: float):
    if slippage_rate <= 0:
        return execute_order

    def _service(order, base_wallet, quote_wallet, current_price, options, clock):
        price = current_price

        def _apply_factor(value, factor):
            if isinstance(value, Decimal):
                return value * (Decimal(1) + Decimal(str(factor)))
            return value * (1 + factor)

        if order.is_buy:
            price = _apply_factor(current_price, slippage_rate)
        elif order.is_sell:
            price = _apply_factor(current_price, -slippage_rate)
        return execute_order(
            order=order,
            base_wallet=base_wallet,
            quote_wallet=quote_wallet,
            current_price=price,
            options=options,
            clock=clock,
        )

    return _service


def build_environment(
    data: pd.DataFrame,
    env_config: Dict[str, Any],
    currency: str,
    main_currency: str,
):
    exchange_name = "simulator"
    commission = env_config.get("fee_rate")

    # prepare two instruments: USDT and currency
    if main_currency not in registry:
        Instrument(main_currency, 2, main_currency)
    base_instrument = registry[main_currency]

    if currency not in registry:
        registry[currency] = Instrument(currency, 8, currency)
    asset_instrument = registry[currency]

    # we do trade on close prices of previous day
    price_column = f"{currency}_close"
    price_stream = Stream.source(list(data[price_column]), dtype="float").rename(
        f"{main_currency}-{currency}"
    )

    # create exchange manager, simulating currency swapping between us and platform
    options = ExchangeOptions(commission=commission)
    exchange = Exchange(exchange_name, service=execute_order, options=options)(price_stream)

    # creation of wallets for cash and asset
    initial_cash = float(env_config.get("initial_cash", 0.0))
    initial_amount = float(env_config.get("initial_amount", 0.0))

    cash_wallet = Wallet(exchange, initial_cash * base_instrument)
    asset_wallet = Wallet(exchange, initial_amount * asset_instrument)
    portfolio = Portfolio(base_instrument, [cash_wallet, asset_wallet])

    # adding stream from price data frame
    with NameSpace(exchange_name):
        feature_streams = [
            Stream.source(list(data[price_column]), dtype="float").rename(price_column)
        ]

    renderer_streams = [
        Stream.source(data[f"{currency}_close"], dtype="float").rename("close"),
        Stream.source(data[f"{currency}_open"], dtype="float").rename("open"),
        Stream.source(data[f"{currency}_high"], dtype="float").rename("high"),
        Stream.source(data[f"{currency}_low"], dtype="float").rename("low"),
        Stream.source(data[f"{currency}_volume"], dtype="float").rename("volume"),
    ]
    if "date" in data.columns:
        renderer_streams.append(Stream.source(list(data["date"])).rename("date"))

    feed = DataFeed(feature_streams)
    feed.compile()

    renderer_feed = DataFeed(renderer_streams)
    renderer_feed.compile()

    renderer_cfg = env_config.get("renderers", "all")
    renderer_formats = env_config.get("renderer_formats", ["png", "html"])
    renderers = None
    if renderer_cfg:
        renderers = construct_renderers(renderer_cfg, display=True, save_formats=renderer_formats)


    action_scheme = action_api.get('bsh', cash=cash_wallet, asset=asset_wallet, proportion=0.1) # always trade 10% of source wallet
    reward_scheme = reward_api.get('simple')

    env_kwargs: Dict[str, Any] = {
        "portfolio": portfolio,
        "action_scheme": action_scheme,
        "reward_scheme": reward_scheme,
        "feed": feed,
        "renderer_feed": renderer_feed,
        "window_size": int(env_config.get("window_size", 30)),
        "max_episode_length": env_config.get("max_episode_length"),
        "enable_logger": False,
    }
    if renderers:
        env_kwargs["renderer"] = renderers

    env = default.create(**env_kwargs)
    return env


def run_strategy(env, strategy: KNNStrategy) -> float:
    state, _ = env.reset(start_from_start=True)
    strategy.reset()

    done = False
    total_reward = 0.0
    step_index = 0
    warmup_steps = max(0, int(getattr(strategy.sim_config, "warmup_steps", 0)))

    while not done:
        action = strategy.get_action(state)
        skip_decision = step_index < warmup_steps
        state, reward, terminated, truncated, _ = env.step(action, skip_decision=skip_decision)
        done = terminated or truncated
        total_reward += reward
        step_index += 1

    return total_reward


def main(config_path: Optional[str] = None) -> None:
    config = load_config(Path(config_path) if config_path else None)

    logging.basicConfig(level=getattr(logging, config.get("log_level", "INFO")))
    LOGGER.info("Loaded configuration from %s", config_path or "default config")

    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    LOGGER.info("Global seed set to %d", seed)

    data_loader = build_data_loader(config["data"])
    price_frame = data_loader.load()
    LOGGER.info("Fetched %d rows of price data", len(price_frame))

    env_window = int(config.get("environment", {}).get("window_size", 30))
    observation_start = max(env_window - 1, 0)

    strategy = build_strategy(
        config["strategy"],
        config.get("simulation", {}),
        price_frame,
        observation_start,
    )

    env = build_environment(
        price_frame,
        config.get("environment", {}),
        config["data"].get("currency", "ASSET"),
        config["data"].get("main_currency", "USD"),
        strategy,
    )

    final_reward = run_strategy(env, strategy)

    env.render()

    output_dir = Path(config.get("output_dir", "pipelines/knn_policy/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    render_dir = output_dir / "renderers"
    render_dir.mkdir(parents=True, exist_ok=True)
    env.save(str(render_dir))

    performance_dict = env.action_scheme.portfolio.performance
    performance = pd.DataFrame.from_dict(performance_dict, orient="index")
    performance.reset_index(drop=True, inplace=True)

    history = strategy.history_frame()

    evaluation_path = output_dir / "evaluation.csv"
    performance.to_csv(evaluation_path, index=False)
    LOGGER.info("Saved evaluation metrics to %s", evaluation_path)

    history_path = output_dir / "strategy_history.csv"
    history.to_csv(history_path, index=False)
    LOGGER.info("Saved strategy diagnostics to %s", history_path)

    final_reward_path = output_dir / "final_reward.txt"
    final_reward_path.write_text(f"{final_reward}\n")
    LOGGER.info("Final reward (environment cumulative reward): %.6f", final_reward)

    config_copy_path = output_dir / "config_used.yaml"
    with config_copy_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the KNN baseline strategy")
    parser.add_argument("--config", type=str, default=None, help="Optional path to a YAML configuration file")
    args = parser.parse_args()
    main(args.config)
