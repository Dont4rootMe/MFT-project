"""
Inference script for A2C Trading Agent.

The model operates on a sliding window of the last N periods (default: 30).
For each window, it outputs a single trading decision.

Two modes are available:
  - single: Get ONE prediction based on the last N periods (real-time trading)
  - batch:  Process entire file with sliding window (backtesting)

Usage:
    # Single prediction (requires at least window_size rows)
    python -m src.predict single --input_path window.csv --output_path pred.json
    
    # Batch predictions for backtesting
    python -m src.predict batch --input_path history.csv --output_path preds.csv
    
    # Docker usage
    docker run -v ./data:/data ml-app:v1 single \
        --input_path /data/window.csv --output_path /data/pred.json
"""

import sys
from pathlib import Path
import argparse
import json
import os

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration loaded from checkpoint for inference."""
    window_size: int
    trade_sizes: List[float]
    action_scheme_name: str
    n_features: int = 39
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> 'InferenceConfig':
        """
        Load inference config from checkpoint meta.json.
        
        This is the preferred method - ensures inference uses exact same
        config as training.
        """
        checkpoint_path = Path(checkpoint_path)
        meta_path = checkpoint_path / 'meta.json'
        
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Checkpoint meta.json not found at {meta_path}. "
                f"Please use a checkpoint that contains full config."
            )
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check if checkpoint has full config
        if 'config' not in meta:
            # Fallback to Hydra config for old checkpoints
            print("   ⚠️ Checkpoint doesn't contain full config, using Hydra config")
            return cls.from_hydra_config()
        
        config = meta['config']
        
        return cls(
            window_size=config.get('window_size', 30),
            trade_sizes=config.get('trade_sizes', [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]),
            action_scheme_name=config.get('action_scheme_name', 'simple'),
            n_features=config.get('n_features', 39)
        )
    
    @classmethod
    def from_hydra_config(cls, config_dir: str = None) -> 'InferenceConfig':
        """
        Fallback: Load inference config from Hydra configuration.
        Use from_checkpoint() when possible for exact reproducibility.
        """
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        
        if config_dir is None:
            config_dir = str(project_root / 'conf')
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='a2c_trainer')
        
        window_size = cfg.env.get('window_size', 30)
        
        action_cfg = cfg.get('action_scheme', {})
        trade_sizes = list(action_cfg.get('trade_sizes', [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]))
        action_scheme_name = action_cfg.get('name', 'simple')
        
        return cls(
            window_size=window_size,
            trade_sizes=trade_sizes,
            action_scheme_name=action_scheme_name
        )
    
    @property
    def n_actions(self) -> int:
        """Calculate number of actions: 1 (hold) + 2 * len(trade_sizes)"""
        return 1 + 2 * len(self.trade_sizes)


def load_model(checkpoint_path: str):
    """
    Load the trained A2C model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        
    Returns:
        tuple: (model_state, metadata)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load model state
    state_path = checkpoint_path / 'model' / 'state.pt'
    if not state_path.exists():
        raise FileNotFoundError(f"Model state not found at {state_path}")
    
    model_state = torch.load(state_path, map_location='cpu')
    
    # Load metadata if available
    meta_path = checkpoint_path / 'meta.json'
    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    
    return model_state, meta


def preprocess_data(
    df: pd.DataFrame, 
    window_size: int = 30,
    verbose: bool = True
) -> tuple:
    """
    Preprocess input data using the same pipeline as training.
    
    Input format (raw OHLCV):
        Required columns: open, high, low, close, volume
        Optional column: date/open_time (timestamp)
        
    Args:
        df: Input DataFrame with market data
        window_size: Size of observation window
        verbose: Print progress messages
        
    Returns:
        tuple: (feature_matrix, dates)
    """
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    
    df = df.copy()
    
    # Rename columns to standard format
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ('open_time', 'timestamp', 'time'):
            col_mapping[col] = 'date'
        elif col_lower in ('open', 'high', 'low', 'close', 'volume'):
            col_mapping[col] = col_lower
    
    if col_mapping:
        df = df.rename(columns=col_mapping)
    
    # Extract dates
    dates = None
    if 'date' in df.columns:
        dates = df['date'].copy()
    
    # Select only OHLCV + date columns
    keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep_cols if c in df.columns]]
    
    if verbose:
        print("   Applying feature engineering pipeline...")
    
    # Load configuration using Hydra
    config_dir = str(project_root / 'conf')
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='a2c_trainer')
    
    # Get feature engineering config
    fe_cfg = OmegaConf.to_container(cfg.data.feature_engineering, resolve=True)
    fs_cfg = OmegaConf.to_container(cfg.data.feature_selection, resolve=True)
    
    from pipelines.rl_agent_policy.data.feature_engine import FeatureEngineeringProcessor
    from pipelines.rl_agent_policy.data.feature_selection import FeatureSelector
    
    feature_processor = FeatureEngineeringProcessor(fe_cfg)
    feature_selector = FeatureSelector(fs_cfg)
    
    # Process features
    processed = feature_processor.generate_features(df, currency_name='ASSET')
    processed = feature_selector.select_per_currency(processed)
    
    # Update dates
    if 'date' in processed.columns:
        dates = processed['date'].copy()
        processed = processed.drop(columns=['date'])
    
    if verbose:
        print(f"   Generated {len(processed.columns)} features")
    
    # Convert to numeric and fill NaN
    for col in processed.columns:
        processed[col] = pd.to_numeric(processed[col], errors='coerce')
    
    processed = processed.ffill().bfill()
    
    return processed.values, dates


def build_model(
    model_state: dict, 
    input_shape: tuple, 
    device: torch.device,
    expected_n_actions: Optional[int] = None
):
    """
    Build model architecture and load weights.
    
    Args:
        model_state: Loaded state dict
        input_shape: (num_features, window_size)
        device: torch device
        expected_n_actions: Expected number of actions from config (for validation)
        
    Returns:
        tuple: (shared_network, actor_head)
    """
    from pipelines.rl_agent_policy.models.cnn import CNNBackbone
    from pipelines.rl_agent_policy.models.heads import ActorHead
    
    # Determine number of actions from actor head state (ground truth from model)
    actor_state = model_state['actor']
    n_actions = None
    for key in actor_state.keys():
        if 'weight' in key:
            n_actions = actor_state[key].shape[0]
    
    if n_actions is None:
        raise ValueError(
            "Could not determine n_actions from model checkpoint. "
            "The actor head state dict does not contain expected weight tensors."
        )
    
    # Validate against expected n_actions from config
    if expected_n_actions is not None and n_actions != expected_n_actions:
        raise ValueError(
            f"Model n_actions ({n_actions}) does not match config n_actions ({expected_n_actions}). "
            f"This indicates a mismatch between the model checkpoint and current config. "
            f"Make sure the config matches the training configuration."
        )
    
    # Build model
    shared_network = CNNBackbone(input_shape).to(device)
    actor_head = ActorHead(shared_network.output_dim, n_actions).to(device)
    
    # Load weights
    shared_network.load_state_dict(model_state['shared'])
    actor_head.load_state_dict(model_state['actor'])
    
    shared_network.eval()
    actor_head.eval()
    
    return shared_network, actor_head


def predict_single(
    shared_network,
    actor_head,
    window_features: np.ndarray,
    device: torch.device,
    trade_sizes: List[float]
) -> dict:
    """
    Generate a single prediction from a window of features.
    
    Args:
        shared_network: Loaded CNN backbone
        actor_head: Loaded actor head
        window_features: Features array of shape (window_size, num_features)
        device: torch device
        trade_sizes: List of position sizes from config (must match training config!)
        
    Returns:
        dict: Prediction with action_id, signal, position_size, confidence, probabilities
    """
    n_sizes = len(trade_sizes)
    
    with torch.no_grad():
        # Reshape for CNN: (batch=1, channels=num_features, length=window_size)
        x = torch.FloatTensor(window_features.T).unsqueeze(0).to(device)
        
        # Forward pass
        shared_out = shared_network(x)
        logits = actor_head(shared_out)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get action and confidence
        action = int(probs.argmax())
        confidence = float(probs.max())
    
    # Decode action
    if action == 0:
        signal = 'HOLD'
        size = 0.0
    elif 1 <= action <= n_sizes:
        signal = 'BUY'
        size = trade_sizes[action - 1]
    elif n_sizes + 1 <= action <= 2 * n_sizes:
        signal = 'SELL'
        size = trade_sizes[action - n_sizes - 1]
    else:
        signal = 'HOLD'
        size = 0.0
    
    return {
        'action_id': action,
        'signal': signal,
        'position_size': size,
        'confidence': confidence,
        'probabilities': {
            'hold': float(probs[0]),
            'buy': float(probs[1:n_sizes+1].sum()),
            'sell': float(probs[n_sizes+1:].sum())
        }
    }


def run_single_prediction(
    input_path: str,
    output_path: str,
    checkpoint_path: str = 'outputs/model/checkpoints/latest',
    window_size: Optional[int] = None
) -> dict:
    """
    Run single prediction mode.
    
    Input file must contain at least `window_size` rows of OHLCV data.
    For accurate feature engineering, we recommend providing more history
    (e.g., 100+ rows) - only the last `window_size` features will be used.
    
    Args:
        input_path: Path to input CSV file (minimum window_size rows, recommended 100+)
        output_path: Path to save prediction JSON
        checkpoint_path: Path to model checkpoint
        window_size: Observation window size (if None, loaded from config)
        
    Returns:
        dict: Prediction result
    """
    # Minimum rows needed for stable feature engineering (rolling windows, etc.)
    MIN_ROWS_FOR_FE = 100
    
    # Load configuration from checkpoint (preferred) or fallback to Hydra
    print(f"⚙️  Loading configuration from checkpoint...")
    try:
        inference_cfg = InferenceConfig.from_checkpoint(checkpoint_path)
        print(f"   ✓ Loaded config from checkpoint")
    except (FileNotFoundError, KeyError) as e:
        print(f"   ⚠️ Checkpoint config not found, using Hydra config: {e}")
        inference_cfg = InferenceConfig.from_hydra_config()
    
    # Use config window_size if not specified
    if window_size is None:
        window_size = inference_cfg.window_size
    
    print(f"   Window size: {window_size}")
    print(f"   Trade sizes: {inference_cfg.trade_sizes}")
    print(f"   Action scheme: {inference_cfg.action_scheme_name}")
    print(f"   N actions: {inference_cfg.n_actions}")
    
    print(f"\n🎯 Single Prediction Mode")
    print(f"   Input: {input_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model_state, meta = load_model(checkpoint_path)
    print(f"   ✓ Model loaded (trained for {meta.get('episode', '?')} episodes)")
    
    # Load input data
    print(f"\n📊 Loading data...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} rows")
    
    if len(df) < window_size:
        raise ValueError(
            f"Input file has {len(df)} rows, but model requires at least "
            f"{window_size} rows (window_size). Please provide more historical data."
        )
    
    if len(df) < MIN_ROWS_FOR_FE:
        print(f"   ⚠ Warning: Only {len(df)} rows provided. Recommend {MIN_ROWS_FOR_FE}+ "
              f"rows for stable feature engineering.")
    
    # Preprocess entire file for stable feature computation
    # (rolling statistics, etc. need history)
    print(f"\n🔧 Preprocessing (using all {len(df)} rows for feature engineering)...")
    features, dates = preprocess_data(df, window_size)
    
    if len(features) < window_size:
        raise ValueError(
            f"After preprocessing, only {len(features)} rows remain. "
            f"Need at least {window_size}. Some rows may have been dropped due to NaN values."
        )
    
    # Take only the LAST window_size rows for prediction
    window_features = features[-window_size:]
    print(f"   Using last {window_size} rows of processed features")
    print(f"   Window shape: ({window_size}, {window_features.shape[1]}) = (periods, features)")
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🤖 Building model on {device}...")
    
    input_shape = (window_features.shape[1], window_size)
    shared_network, actor_head = build_model(
        model_state, input_shape, device,
        expected_n_actions=inference_cfg.n_actions
    )
    
    # Generate prediction using trade_sizes from config
    print(f"\n🎯 Generating prediction...")
    prediction = predict_single(
        shared_network, actor_head, window_features, device,
        trade_sizes=inference_cfg.trade_sizes
    )
    
    # Add metadata (including config parameters for reproducibility)
    result = {
        'prediction': prediction,
        'metadata': {
            'window_size': window_size,
            'num_features': window_features.shape[1],
            'input_file': input_path,
            'checkpoint': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'config': {
                'trade_sizes': inference_cfg.trade_sizes,
                'action_scheme': inference_cfg.action_scheme_name,
                'n_actions': inference_cfg.n_actions
            }
        }
    }
    
    # Save result
    print(f"\n💾 Saving prediction...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"   ✓ Saved to {output_path}")
    
    # Display result
    print(f"\n" + "="*50)
    print(f"📈 PREDICTION RESULT")
    print(f"="*50)
    print(f"   Signal:        {prediction['signal']}")
    print(f"   Position Size: {prediction['position_size']*100:.1f}%")
    print(f"   Confidence:    {prediction['confidence']*100:.1f}%")
    print(f"   Action ID:     {prediction['action_id']}")
    print(f"\n   Probabilities:")
    print(f"     HOLD: {prediction['probabilities']['hold']*100:.1f}%")
    print(f"     BUY:  {prediction['probabilities']['buy']*100:.1f}%")
    print(f"     SELL: {prediction['probabilities']['sell']*100:.1f}%")
    print(f"="*50)
    
    return result


def run_batch_prediction(
    input_path: str,
    output_path: str,
    checkpoint_path: str = 'outputs/model/checkpoints/latest',
    window_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Run batch prediction mode for backtesting.
    
    Processes the entire input file with a sliding window,
    generating one prediction per window position.
    
    Args:
        input_path: Path to input CSV file with historical data
        output_path: Path to save predictions CSV
        checkpoint_path: Path to model checkpoint
        window_size: Observation window size (if None, loaded from config)
        
    Returns:
        pd.DataFrame: All predictions
    """
    # Load configuration from checkpoint (preferred) or fallback to Hydra
    print(f"⚙️  Loading configuration from checkpoint...")
    try:
        inference_cfg = InferenceConfig.from_checkpoint(checkpoint_path)
        print(f"   ✓ Loaded config from checkpoint")
    except (FileNotFoundError, KeyError) as e:
        print(f"   ⚠️ Checkpoint config not found, using Hydra config: {e}")
        inference_cfg = InferenceConfig.from_hydra_config()
    
    # Use config window_size if not specified
    if window_size is None:
        window_size = inference_cfg.window_size
    
    print(f"   Window size: {window_size}")
    print(f"   Trade sizes: {inference_cfg.trade_sizes}")
    print(f"   N actions: {inference_cfg.n_actions}")
    
    print(f"\n📊 Batch Prediction Mode (Backtesting)")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model_state, meta = load_model(checkpoint_path)
    print(f"   ✓ Model loaded (trained for {meta.get('episode', '?')} episodes)")
    
    # Load input data
    print(f"\n📊 Loading data...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if len(df) < window_size:
        raise ValueError(
            f"Input file has {len(df)} rows, but model requires at least "
            f"{window_size} rows (window_size)."
        )
    
    # Preprocess entire file
    print(f"\n🔧 Preprocessing...")
    features, dates = preprocess_data(df, window_size)
    num_samples = len(features)
    num_features = features.shape[1]
    print(f"   Features shape: ({num_samples}, {num_features})")
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🤖 Building model on {device}...")
    
    input_shape = (num_features, window_size)
    shared_network, actor_head = build_model(
        model_state, input_shape, device,
        expected_n_actions=inference_cfg.n_actions
    )
    
    # Generate predictions with sliding window
    print(f"\n🎯 Generating predictions with sliding window...")
    num_predictions = num_samples - window_size + 1
    print(f"   Will generate {num_predictions} predictions")
    
    predictions = []
    for i in range(window_size - 1, num_samples):
        window_start = i - window_size + 1
        window_features = features[window_start:i + 1, :]
        
        pred = predict_single(
            shared_network, actor_head, window_features, device,
            trade_sizes=inference_cfg.trade_sizes
        )
        
        # Add window info
        pred['window_start'] = window_start
        pred['window_end'] = i
        pred['step'] = len(predictions)
        
        predictions.append(pred)
        
        if (len(predictions) % 200 == 0) or (len(predictions) == num_predictions):
            print(f"   Processed {len(predictions)}/{num_predictions} windows...")
    
    # Convert to DataFrame
    results = pd.DataFrame([
        {
            'step': p['step'],
            'window_start': p['window_start'],
            'window_end': p['window_end'],
            'action_id': p['action_id'],
            'signal': p['signal'],
            'position_size': p['position_size'],
            'confidence': p['confidence'],
            'prob_hold': p['probabilities']['hold'],
            'prob_buy': p['probabilities']['buy'],
            'prob_sell': p['probabilities']['sell']
        }
        for p in predictions
    ])
    
    # Add dates if available
    if dates is not None and len(dates) >= num_samples:
        results['date'] = dates.iloc[window_size - 1:].values
        cols = ['step', 'date'] + [c for c in results.columns if c not in ['step', 'date']]
        results = results[cols]
    
    # Save results
    print(f"\n💾 Saving predictions...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"   ✓ Saved to {output_path}")
    
    # Summary
    print(f"\n📈 Summary:")
    signal_counts = results['signal'].value_counts()
    for signal, count in signal_counts.items():
        pct = count / len(results) * 100
        print(f"   {signal}: {count} ({pct:.1f}%)")
    print(f"   Average confidence: {results['confidence'].mean():.4f}")
    
    print(f"\n✅ Batch prediction complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="A2C Trading Model Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The model operates on a sliding window of the last N periods (default: 30).
Each window produces exactly ONE trading decision.

MODES:
  single  Get ONE prediction from the last N periods (for real-time trading)
  batch   Process entire file with sliding window (for backtesting)

EXAMPLES:
  # Single prediction - input must have at least 30 rows
  python -m src.predict single --input_path window.csv --output_path pred.json
  
  # Batch backtesting
  python -m src.predict batch --input_path history.csv --output_path preds.csv
  
  # Docker: single prediction
  docker run -v ./data:/data ml-app:v1 single \\
      --input_path /data/window.csv --output_path /data/pred.json
  
  # Docker: batch prediction
  docker run -v ./data:/data ml-app:v1 batch \\
      --input_path /data/history.csv --output_path /data/preds.csv

INPUT FORMAT (CSV with OHLCV data):
  open_time,open,high,low,close,volume
  1609459200000,0.35,0.38,0.34,0.37,1000000
  1609545600000,0.37,0.40,0.36,0.39,1200000
  ...
  (minimum 30 rows for single prediction)

OUTPUT FORMAT:
  Single mode: JSON with prediction and metadata
  Batch mode:  CSV with step, signal, position_size, confidence per window
"""
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Prediction mode')
    
    # Single prediction mode
    single_parser = subparsers.add_parser(
        'single',
        help='Single prediction from last N periods (real-time trading)'
    )
    single_parser.add_argument(
        '--input_path', type=str, required=True,
        help='CSV file with at least window_size rows of OHLCV data'
    )
    single_parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output JSON file for prediction'
    )
    single_parser.add_argument(
        '--checkpoint', type=str, default='outputs/model/checkpoints/latest',
        help='Path to model checkpoint (default: outputs/model/checkpoints/latest)'
    )
    single_parser.add_argument(
        '--window_size', type=int, default=None,
        help='Observation window size (default: from config)'
    )
    
    # Batch prediction mode
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch predictions with sliding window (backtesting)'
    )
    batch_parser.add_argument(
        '--input_path', type=str, required=True,
        help='CSV file with historical OHLCV data'
    )
    batch_parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output CSV file for predictions'
    )
    batch_parser.add_argument(
        '--checkpoint', type=str, default='outputs/model/checkpoints/latest',
        help='Path to model checkpoint (default: outputs/model/checkpoints/latest)'
    )
    batch_parser.add_argument(
        '--window_size', type=int, default=None,
        help='Observation window size (default: from config)'
    )
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        print("\nError: Please specify a mode: 'single' or 'batch'")
        sys.exit(1)
    
    if args.mode == 'single':
        run_single_prediction(
            input_path=args.input_path,
            output_path=args.output_path,
            checkpoint_path=args.checkpoint,
            window_size=args.window_size
        )
    elif args.mode == 'batch':
        run_batch_prediction(
            input_path=args.input_path,
            output_path=args.output_path,
            checkpoint_path=args.checkpoint,
            window_size=args.window_size
        )


if __name__ == '__main__':
    main()
