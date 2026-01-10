"""
DVC Pipeline Stage: Model Evaluation

This script evaluates a trained model on validation data and outputs metrics
in a format suitable for DVC metrics tracking.
"""

import sys
from pathlib import Path
import argparse
import json
import yaml
import os

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

import tensortrade.env.default as default
from tensortrade.env.default import actions as action_api, rewards as reward_api
from tensortrade.feed.core import DataFeed, Stream, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument, registry
from gymnasium.spaces import MultiDiscrete


def calculate_metrics(performance_df: pd.DataFrame) -> dict:
    """
    Calculate trading performance metrics from portfolio performance.
    
    Args:
        performance_df: DataFrame with portfolio performance over time
        
    Returns:
        Dictionary of calculated metrics
    """
    if 'net_worth' not in performance_df.columns:
        return {'error': 'net_worth column not found'}
    
    net_worth = performance_df['net_worth'].values
    
    # Calculate returns
    returns = np.diff(net_worth) / net_worth[:-1]
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return {'error': 'No valid returns to calculate'}
    
    # Basic metrics
    total_return = (net_worth[-1] - net_worth[0]) / net_worth[0] if net_worth[0] > 0 else 0
    
    # Sharpe Ratio (annualized, assuming daily data)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Max Drawdown
    peak = np.maximum.accumulate(net_worth)
    drawdown = (peak - net_worth) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Win Rate
    positive_returns = np.sum(returns > 0)
    win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
    
    # Profit Factor
    gains = np.sum(returns[returns > 0])
    losses = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gains / losses if losses > 0 else float('inf')
    
    # Calmar Ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'profit_factor': float(min(profit_factor, 100)),  # Cap for JSON serialization
        'calmar_ratio': float(min(calmar_ratio, 100)),
        'sortino_ratio': float(sortino_ratio),
        'final_net_worth': float(net_worth[-1]),
        'initial_net_worth': float(net_worth[0]),
        'num_periods': len(net_worth),
        'mean_daily_return': float(mean_return),
        'std_daily_return': float(std_return)
    }


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    metrics_file: str = 'metrics.json'
) -> dict:
    """
    Evaluate a trained model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to hydra configuration
        output_dir: Directory to save evaluation results
        metrics_file: Name of metrics output file
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"📊 Evaluating model...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Config: {config_path}")
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Load data
    data_handler = instantiate(cfg.data)
    data = data_handler.get_data()
    
    # Get validation split
    validation_size = cfg.validation.get("validation_size", 365)
    if isinstance(validation_size, int):
        split_idx = max(0, len(data) - validation_size)
    else:
        split_idx = int(len(data) * (1 - validation_size))
    
    valid_df = data.iloc[split_idx:].reset_index(drop=True).copy()
    
    print(f"   Validation size: {len(valid_df)} periods")
    
    # Build environment
    assets = cfg.get("assets") or data_handler.symbols
    main_currency = data_handler.main_currency
    
    if main_currency not in registry:
        Instrument(main_currency, 2, main_currency)
    base_instrument = registry[main_currency]
    
    asset_instruments = []
    assets_initial = cfg.env.assets_initial
    if isinstance(assets_initial, int):
        assets_initial = [assets_initial] * len(assets)
    
    for sym, init_amount in zip(assets, assets_initial):
        if sym not in registry:
            registry[sym] = Instrument(sym, 8, sym)
        asset_instruments.append((registry[sym], init_amount))
    
    # Build validation environment
    price_streams = [
        Stream.source(list(valid_df[f"{sym}_close"]), dtype="float").rename(
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
            Stream.source(list(valid_df[c]), dtype="float").rename(c)
            for c in valid_df.columns
            if c != "date"
        ]
    
    feed = DataFeed(feature_streams)
    feed.compile()
    
    action_scheme = action_api.get(
        'simple',
        portfolio=portfolio,
        criteria=[None],
        trade_sizes=[0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
        min_order_abs=0,
        min_order_pct=0,
    )
    
    reward_cfg = cfg.get("reward_scheme")
    reward_scheme = reward_api.create(reward_cfg)
    
    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=DataFeed([]),
        window_size=cfg.env.window_size,
        max_episode_length=cfg.env.get('max_episode_length', None),
        enable_logger=False,
    )
    
    # Load model
    print(f"   Loading model from checkpoint...")
    agent = instantiate(cfg.model, env=env)
    
    state_path = os.path.join(checkpoint_path, 'model', 'state.pt')
    if os.path.exists(state_path):
        state_dict = torch.load(state_path, map_location='cpu')
        agent.network.load_state_dict(state_dict)
        print(f"   ✓ Model loaded successfully")
    else:
        print(f"   ⚠ No checkpoint found, using random weights")
    
    # Run evaluation
    state, _ = env.reset(begin_from_start=True)
    done = False
    total_reward = 0.0
    step_count = 0
    
    print(f"   Running evaluation...")
    
    while not done:
        action_idx = agent.get_action(state)
        env_action = action_idx
        if isinstance(env.action_space, MultiDiscrete):
            env_action = np.array(
                np.unravel_index(action_idx, env.action_space.nvec)
            ).astype(int)
        state, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
    
    print(f"   ✓ Evaluation complete: {step_count} steps, total reward: {total_reward:.4f}")
    
    # Calculate metrics
    perf = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient="index")
    metrics = calculate_metrics(perf)
    metrics['total_reward'] = float(total_reward)
    metrics['steps'] = step_count
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON (for DVC metrics)
    metrics_path = os.path.join(output_dir, metrics_file)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics saved to {metrics_path}")
    
    # Save detailed performance
    perf_path = os.path.join(output_dir, 'performance.csv')
    perf.to_csv(perf_path)
    print(f"   ✓ Performance saved to {perf_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model and output DVC metrics"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/model/checkpoints/latest',
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='conf/a2c_trainer.yaml',
        help='Path to hydra configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/evaluation',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--metrics-file',
        type=str,
        default='metrics.json',
        help='Name of metrics output file'
    )
    
    args = parser.parse_args()
    
    metrics = evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output,
        metrics_file=args.metrics_file
    )
    
    print(f"\n📈 Evaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


if __name__ == '__main__':
    main()

