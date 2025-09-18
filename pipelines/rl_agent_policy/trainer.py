import sys
from pathlib import Path

# Ensure project root is on the path so local modules can be imported
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from typing import Union
import logging

import tensortrade.env.default as default
from tensortrade.env.default import actions as action_api, rewards as reward_api
from tensortrade.env.default.renderers import construct_renderers
from tensortrade.feed.core import DataFeed, Stream, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument, registry
from gymnasium.spaces import MultiDiscrete

from pipelines.rl_agent_policy.train.a2c import A2CTrainer


@hydra.main(config_path="../../conf", config_name="a2c_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entrypoint for training the A2C agent."""
    
    if cfg.debug_logging:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    data_handler = instantiate(cfg.data)
    data = data_handler.get_data()

    # determine validation split either by ratio or absolute size
    validation_size: Union[int, float] = cfg.validation.get("validation_size", 365)
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

    # Ensure train and validation data have identical feature dimensions
    train_features = train_df.drop(columns=["date"], errors="ignore").shape[1]
    valid_features = valid_df.drop(columns=["date"], errors="ignore").shape[1]
    if train_features != valid_features:
        raise ValueError(
            f"Train and validation feature counts differ: {train_features} != {valid_features}"
        )

    # Propagate inferred input shape to the model configuration
    input_shape = [train_features, cfg.env.window_size]
    OmegaConf.set_struct(cfg.model.shared_network, False)
    cfg.model.shared_network.input_shape = input_shape

    assets = cfg.get("assets") or data_handler.symbols
    main_currency = data_handler.main_currency

    if main_currency not in registry:
        Instrument(main_currency, 2, main_currency)
    base_instrument = registry[main_currency]

    asset_instruments = []
    
    assert isinstance(cfg.env.assets_initial, int) or len(assets) == len(cfg.env.assets_initial), \
        "assets and assets_initial must have the same length or be constant"
    
    if isinstance(cfg.env.assets_initial, int):
        cfg.env.assets_initial = cfg.env.assets_initial * len(assets)

    for sym, init_amount in zip(assets, cfg.env.assets_initial):
        if sym not in registry:
            registry[sym] = Instrument(sym, 8, sym)
        asset_instruments.append((registry[sym], init_amount))

    # ------------------------------------------------------------------
    # Environment building
    # ------------------------------------------------------------------
    def build_env(df: pd.DataFrame):
        
        # we do trade on close prices for previous day
        price_streams = [
            Stream.source(list(df[f"{sym}_close"]), dtype="float").rename(
                f"{main_currency}-{sym}"
            )
            for sym in assets
        ]
        exchange = Exchange(cfg.env.exchange, service=execute_order)(*price_streams)

        cash = Wallet(exchange, cfg.env.initial_cash * base_instrument)
        asset_wallets = [Wallet(exchange, init_amount * inst) for inst, init_amount in asset_instruments]
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

        action_cfg = cfg.get("action_scheme")
        try:
            action_scheme = action_api.create(action_cfg)
        except TypeError:
            params = {"cash": cash}
            if len(asset_wallets) == 1:
                params["asset"] = asset_wallets[0]
            else:
                params["assets"] = asset_wallets
            action_scheme = action_api.create(action_cfg, **params)

        reward_cfg = cfg.get("reward_scheme")
        reward_scheme = reward_api.create(reward_cfg)
        
        # create renderers
        renderer_list = cfg.validation.get('renderers', 'all')
        renderers = construct_renderers(renderer_list, display=False)

        env = default.create(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            feed=feed,
            renderer_feed=renderer_feed,
            window_size=cfg.env.window_size,
            enable_logger=False,
            renderer=renderers
        )
        return env

    train_env = build_env(train_df)
    valid_env = build_env(valid_df)

    # ------------------------------------------------------------------
    # Agent and trainer
    # ------------------------------------------------------------------
    output_dir = Path(to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_accelerate = cfg.get('use_accelerate', None)
    
    agent = instantiate(cfg.model, env=train_env)
    train_config = instantiate(cfg.train.approach)
    trainer = A2CTrainer(
        agent=agent, 
        train_env=train_env, 
        valid_env=valid_env, 
        output_dir=output_dir, 
        config=train_config,
        use_accelerate=use_accelerate
    )
    trainer.train()

    # ------------------------------------------------------------------
    # Final evaluation and result handling
    # ------------------------------------------------------------------
    state, _ = valid_env.reset(start_from_time=True)
    done = False
    total_reward = 0.0
    while not done:
        action_idx = agent.get_action(state)
        env_action = action_idx
        if isinstance(valid_env.action_space, MultiDiscrete):
            env_action = np.array(
                np.unravel_index(action_idx, valid_env.action_space.nvec)
            ).astype(int)
        state, reward, terminated, truncated, _ = valid_env.step(env_action)
        done = terminated or truncated
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

