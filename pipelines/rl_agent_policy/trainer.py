import sys
from pathlib import Path

# Ensure project root is on the path so local modules can be imported
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
import hydra
import numpy as np
import random
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from typing import Union, Optional
import logging
import copy

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

# Try to import accelerate to detect distributed training
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None


def _get_process_info():
    """Get process information for distributed training."""
    if ACCELERATE_AVAILABLE:
        try:
            # Try to create accelerator to get process info
            accelerator = Accelerator()
            return {
                'process_index': accelerator.process_index,
                'num_processes': accelerator.num_processes,
                'is_main_process': accelerator.is_main_process
            }
        except:
            pass
    
    # Fallback to single process
    return {
        'process_index': 0,
        'num_processes': 1,
        'is_main_process': True
    }


def _create_process_specific_data_split(train_df, valid_df, process_index, num_processes):
    """
    Create process-specific data splits to ensure different episodes for each process.
    
    Args:
        train_df: Training dataframe
        valid_df: Validation dataframe  
        process_index: Current process index
        num_processes: Total number of processes
        
    Returns:
        tuple: (process_train_df, process_valid_df)
    """
    if num_processes == 1:
        return train_df, valid_df
    
    # Split training data across processes with offset to ensure different episodes
    train_len = len(train_df)
    chunk_size = train_len // num_processes
    
    # Create overlapping chunks with offset to ensure variety
    start_idx = (process_index * chunk_size) % train_len
    end_idx = ((process_index + 1) * chunk_size) % train_len
    
    if end_idx <= start_idx:
        # Handle wrap-around
        process_train_df = pd.concat([
            train_df.iloc[start_idx:],
            train_df.iloc[:end_idx]
        ]).reset_index(drop=True)
    else:
        process_train_df = train_df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # For validation, use the same data but with different random seeds
    # This ensures consistent evaluation while allowing different exploration
    process_valid_df = valid_df.copy()
    
    return process_train_df, process_valid_df


@hydra.main(config_path="../../conf", config_name="a2c_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entrypoint for training the A2C agent."""
    
    # Get process information for distributed training
    process_info = _get_process_info()
    process_index = process_info['process_index']
    num_processes = process_info['num_processes']
    is_main_process = process_info['is_main_process']
    
    if cfg.debug_logging:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)  # Only show errors, suppress warnings
        
        # Suppress warnings from specific modules
        logging.getLogger('root').setLevel(logging.ERROR)
        logging.getLogger('tensortrade').setLevel(logging.ERROR)
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        logging.getLogger('plotly').setLevel(logging.ERROR)
    
    # Log process information
    if is_main_process or num_processes == 1:
        print(f"🚀 Process Information:")
        print(f"   Process index: {process_index}")
        print(f"   Total processes: {num_processes}")
        print(f"   Is main process: {is_main_process}")
    
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

    base_train_df = data.iloc[:split_idx].reset_index(drop=True).copy()
    base_valid_df = data.iloc[split_idx:].reset_index(drop=True).copy()
    
    # Create process-specific data splits for parallel training
    train_df, valid_df = _create_process_specific_data_split(
        base_train_df, base_valid_df, 0, 1
    )
    
    if is_main_process or num_processes == 1:
        print(f"📊 Data Split Information:")
        print(f"   Total data points: {len(data)}")
        print(f"   Base training data: {len(base_train_df)}")
        print(f"   Base validation data: {len(base_valid_df)}")
        print(f"   Process training data: {len(train_df)}")
        print(f"   Process validation data: {len(valid_df)}")

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
    def build_env(df: pd.DataFrame, env_rng: Optional[random.Random] = None):
        
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

        action_scheme = action_api.get(
            'simple',
            portfolio=portfolio,
            criteria=[None],
            trade_sizes=[0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
            min_order_abs=0,
            min_order_pct=0,
        )
        
        reward_scheme = reward_api.get('risk-adjusted')

        reward_cfg = cfg.get("reward_scheme")
        reward_scheme = reward_api.create(reward_cfg)
        
        # create renderers
        renderer_list = cfg.validation.get('renderers', 'all')
        renderer_formats = cfg.validation.get('renderer_formats', ["png", "html"])
        renderers = construct_renderers(renderer_list, display=False, save_formats=renderer_formats)

        env = default.create(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            feed=feed,
            renderer_feed=renderer_feed,
            window_size=cfg.env.window_size,
            max_episode_length=cfg.env.get('max_episode_length', None),
            enable_logger=False,
            renderer=renderers,
            rng=env_rng
        )
        
        return env

    # Create process-specific environment seeds and RNG instances
    base_seed = cfg.get('seed', 42)
    train_env_seed = (base_seed + process_index * 2003) % (2**32)  # Different prime for env seeding
    valid_env_seed = (base_seed + process_index * 2003 + 1009) % (2**32)  # Offset for validation
    
    # Create separate RNG instances for each environment
    train_rng = random.Random(train_env_seed)
    valid_rng = random.Random(valid_env_seed)
    
    train_env = build_env(train_df, train_rng)
    valid_env = build_env(valid_df, valid_rng)
    
    if is_main_process or num_processes == 1:
        print(f"🌱 Environment Configuration:")
        print(f"   Base seed: {base_seed}")
        print(f"   Train env seed: {train_env_seed}")
        print(f"   Valid env seed: {valid_env_seed}")
        print(f"   Max episode length: {cfg.env.get('max_episode_length', 'No limit')}")
        print(f"   Window size: {cfg.env.window_size}")

    # ------------------------------------------------------------------
    # Agent and trainer
    # ------------------------------------------------------------------
    output_dir = Path(to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_accelerate = cfg.get('use_accelerate', None)
    
    agent = instantiate(cfg.model, env=train_env)
    train_config = instantiate(cfg.train.approach)
    
    # Set the seed in the training configuration
    if hasattr(cfg, 'seed') and cfg.seed is not None:
        train_config.seed = cfg.seed
    
    trainer = A2CTrainer(
        agent=agent, 
        train_env=train_env, 
        valid_env=valid_env, 
        output_dir=output_dir, 
        config=train_config,
        max_episode_length=cfg.env.get('max_episode_length', None),
        use_accelerate=use_accelerate
    )
    trainer.train()

    # ------------------------------------------------------------------
    # Final evaluation and result handling
    # ------------------------------------------------------------------
    state, _ = valid_env.reset(begin_from_start=True)
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

