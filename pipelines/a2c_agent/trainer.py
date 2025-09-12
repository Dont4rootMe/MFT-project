import sys
from pathlib import Path

# Ensure project root is on the path so local modules can be imported
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, to_absolute_path
from typing import Union, List

import tensortrade.env.default as default
from tensortrade.env.default.actions import BSH, TensorTradeActionScheme
from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.feed.core import DataFeed, Stream, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument, registry
from tensortrade.oms.orders import proportion_order
from gymnasium.spaces import MultiDiscrete

from pipelines.a2c_agent.train.a2c import A2CTrainer


class MultiBSH(TensorTradeActionScheme):
    """A simple multi-asset extension of the BSH action scheme.

    Each asset wallet toggles between holding base currency and the asset
    whenever its corresponding action switches between 0 and 1.
    """

    def __init__(self, cash: Wallet, assets: List[Wallet]):
        super().__init__()
        self.cash = cash
        self.assets = assets
        self.action = [0] * len(assets)
        self.listeners: List = []

    @property
    def action_space(self):
        return MultiDiscrete([2] * len(self.assets))

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, actions, portfolio: Portfolio):
        orders = []
        for i, a in enumerate(actions):
            if abs(a - self.action[i]) > 0:
                src = self.cash if self.action[i] == 0 else self.assets[i]
                tgt = self.assets[i] if self.action[i] == 0 else self.cash
                if src.balance == 0:
                    self.action[i] = a
                    continue
                orders.append(proportion_order(portfolio, src, tgt, 1.0))
                self.action[i] = a
        for listener in self.listeners:
            listener.on_action(actions)
        return orders

    def reset(self):
        super().reset()
        self.action = [0] * len(self.assets)


@hydra.main(config_path="../../conf", config_name="a2c_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entrypoint for training the A2C agent."""
    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    data_handler = instantiate(cfg.data)
    data = data_handler.get_data()

    # determine validation split either by ratio or absolute size
    validation_size: Union[int, float] = cfg.get("validation_size", 0.2)
    if isinstance(validation_size, int):
        if validation_size < 0:
            raise ValueError("validation_size must be non-negative")
        split_idx = max(0, len(data) - validation_size)
    elif isinstance(validation_size, float):
        if not 0 <= validation_size <= 1:
            raise ValueError("validation_size ratio must be between 0 and 1")
        split_idx = int(len(data) * (1 - validation_size))
    else:
        raise TypeError("validation_size must be int or float")

    train_df = data.iloc[:split_idx].reset_index(drop=True)
    valid_df = data.iloc[split_idx:].reset_index(drop=True)

    assets = cfg.get("assets") or data_handler.symbols
    main_currency = data_handler.main_currency

    if main_currency not in registry:
        Instrument(main_currency, 2, main_currency)
    base_instrument = registry[main_currency]

    asset_instruments = []
    for sym in assets:
        if sym not in registry:
            Instrument(sym, 8, sym)
        asset_instruments.append(registry[sym])

    def build_env(df: pd.DataFrame):
        price_streams = [
            Stream.source(list(df[f"{sym}_close"]), dtype="float").rename(
                f"{main_currency}-{sym}"
            )
            for sym in assets
        ]
        exchange = Exchange(cfg.env.exchange, service=execute_order)(*price_streams)

        cash = Wallet(exchange, cfg.env.initial_cash * base_instrument)
        asset_wallets = [Wallet(exchange, 0 * inst) for inst in asset_instruments]
        portfolio = Portfolio(base_instrument, [cash, *asset_wallets])

        with NameSpace(cfg.env.exchange):
            feature_streams = [
                Stream.source(list(df[c]), dtype="float").rename(c)
                for c in df.columns
                if c != "date"
            ]
        feed = DataFeed(feature_streams)
        feed.compile()

        # renderer feed for plotting or further analysis
        renderer_streams = []
        if "date" in df.columns:
            renderer_streams.append(Stream.source(list(df["date"])).rename("date"))
        for sym in assets:
            for field in ["open", "high", "low", "close", "volume"]:
                column = f"{sym}_{field}"
                if column in df.columns:
                    renderer_streams.append(
                        Stream.source(list(df[column]), dtype="float").rename(column)
                    )
        renderer_feed = DataFeed(renderer_streams)
        renderer_feed.compile()

        if len(asset_wallets) == 1:
            action_scheme = BSH(cash=cash, asset=asset_wallets[0])
        else:
            action_scheme = MultiBSH(cash=cash, assets=asset_wallets)
        reward_scheme = SimpleProfit()

        env = default.create(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            feed=feed,
            renderer_feed=renderer_feed,
            window_size=cfg.env.window_size,
            enable_logger=False,
        )
        return env

    train_env = build_env(train_df)
    valid_env = build_env(valid_df)

    # ------------------------------------------------------------------
    # Agent and trainer
    # ------------------------------------------------------------------
    agent = instantiate(cfg.model, env=train_env)
    train_config = instantiate(cfg.train.approach)
    trainer = A2CTrainer(agent=agent, train_env=train_env, valid_env=valid_env, config=train_config)
    trainer.train()

    # ------------------------------------------------------------------
    # Final evaluation and result handling
    # ------------------------------------------------------------------
    state = valid_env.reset(start_from_time=True)
    done = False
    total_reward = 0.0
    while not done:
        action = agent.get_action(state)
        state, reward, done, _ = valid_env.step(action)
        total_reward += reward

    results_dir = Path(to_absolute_path(train_config.output_dir))
    results_dir.mkdir(parents=True, exist_ok=True)
    perf = pd.DataFrame.from_dict(valid_env.action_scheme.portfolio.performance, orient="index")
    perf.to_csv(results_dir / "evaluation.csv")
    with open(results_dir / "final_reward.txt", "w") as f:
        f.write(str(total_reward))

    print(f"Final evaluation reward: {total_reward:.4f}")


if __name__ == "__main__":
    main()

