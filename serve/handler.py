"""
TorchServe Custom Handler for A2C Trading Model.

This handler uses the original model architecture from pipelines/
and loads configuration from the saved experiment in outputs/.

Expected input format:
{
    "window": [
        {"open": 0.35, "high": 0.38, "low": 0.34, "close": 0.37, "volume": 1000000},
        ...  (30 rows for window_size=30)
    ]
}

Output format:
{
    "action_id": 11,
    "signal": "SELL",
    "position_size": 0.1,
    "confidence": 0.72,
    "probabilities": {"hold": 0.15, "buy": 0.13, "sell": 0.72}
}
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class A2CTradingHandler(BaseHandler):
    """
    Custom TorchServe handler for A2C Trading Model.
    
    Uses:
    - Original model architecture from pipelines/rl_agent_policy/models/
    - Configuration from the experiment's saved config
    """
    
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.shared_network = None
        self.actor_head = None
        self.config = None
        self.project_root = None
        
    def initialize(self, context):
        """
        Initialize model by loading from the experiment checkpoint.
        
        Args:
            context: TorchServe context with model dir and properties
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        logger.info(f"Initializing A2C Trading Handler from {model_dir}")
        
        # Find project root and add to path
        # In Docker, the project is mounted at /app
        possible_roots = [
            Path("/app"),  # Docker mount point
            Path(model_dir).parent.parent,  # Relative to model-store
            Path.cwd(),
        ]
        
        for root in possible_roots:
            if (root / "pipelines" / "rl_agent_policy" / "models").exists():
                self.project_root = root
                break
        
        if self.project_root is None:
            # Fallback: use bundled model definitions
            logger.warning("Project root not found, using bundled model")
            self.project_root = Path(model_dir)
        
        # Add project root to path for imports
        sys.path.insert(0, str(self.project_root))
        logger.info(f"Project root: {self.project_root}")
        
        # Load configuration from bundled config.json (exported from checkpoint)
        # Priority: config.json (from checkpoint) > Hydra config > defaults
        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from checkpoint: window_size={self.config.get('window_size')}, "
                       f"n_actions={self.config.get('n_actions')}, n_features={self.config.get('n_features')}")
        else:
            # Try loading from Hydra config
            try:
                self.config = self._load_hydra_config()
                logger.info(f"Loaded config from Hydra: {self.config}")
            except Exception as e:
                logger.error(f"Could not load config: {e}")
                raise RuntimeError(
                    "No config found! The model archive should contain config.json "
                    "exported from the training checkpoint."
                )
        
        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        # Load model from checkpoint
        model_path = Path(model_dir) / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model_state = torch.load(model_path, map_location=self.device)
        
        # Import and build model architecture from original source
        try:
            from pipelines.rl_agent_policy.models.cnn import CNNBackbone
            from pipelines.rl_agent_policy.models.heads import ActorHead
            logger.info("Loaded model architecture from pipelines/")
        except ImportError:
            # Fallback to bundled model (for standalone deployment)
            logger.warning("Using bundled model architecture")
            from model_arch import CNNBackbone, ActorHead
        
        input_shape = (self.config["n_features"], self.config["window_size"])
        self.shared_network = CNNBackbone(input_shape).to(self.device)
        self.actor_head = ActorHead(
            self.shared_network.output_dim, 
            self.config["n_actions"]
        ).to(self.device)
        
        # Load weights
        self.shared_network.load_state_dict(model_state['shared'])
        self.actor_head.load_state_dict(model_state['actor'])
        
        self.shared_network.eval()
        self.actor_head.eval()
        
        logger.info("Model loaded successfully")
        self.initialized = True
    
    def _load_hydra_config(self) -> dict:
        """Load configuration from Hydra config files."""
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        
        config_dir = str(self.project_root / 'conf')
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='a2c_trainer')
        
        action_cfg = cfg.get('action_scheme', {})
        trade_sizes = list(action_cfg.get('trade_sizes', [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]))
        
        return {
            "window_size": cfg.env.get('window_size', 30),
            "trade_sizes": trade_sizes,
            "n_actions": 1 + 2 * len(trade_sizes),
            "n_features": 39
        }
        
    def preprocess(self, data: List[Dict]) -> torch.Tensor:
        """
        Preprocess input data into model-ready tensor.
        
        Args:
            data: List of request data dictionaries
            
        Returns:
            torch.Tensor: Preprocessed input tensor (batch, features, window)
        """
        batch_tensors = []
        
        for request in data:
            body = request.get("body")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body.decode('utf-8'))
            elif isinstance(body, str):
                body = json.loads(body)
            
            # Extract window data
            window = body.get("window", body.get("data", body))
            
            if isinstance(window, list):
                # Convert list of dicts to numpy array
                if isinstance(window[0], dict):
                    # OHLCV format
                    features = []
                    for row in window:
                        features.append([
                            float(row.get("open", 0)),
                            float(row.get("high", 0)),
                            float(row.get("low", 0)),
                            float(row.get("close", 0)),
                            float(row.get("volume", 0))
                        ])
                    window_array = np.array(features, dtype=np.float32)
                else:
                    # Already a 2D array
                    window_array = np.array(window, dtype=np.float32)
            else:
                raise ValueError(f"Invalid window format: {type(window)}")
            
            # Validate shape
            expected_window = self.config["window_size"]
            if len(window_array) < expected_window:
                raise ValueError(
                    f"Window has {len(window_array)} rows, "
                    f"need at least {expected_window}"
                )
            
            # Take last window_size rows
            window_array = window_array[-expected_window:]
            
            # Apply feature engineering if only 5 features (raw OHLCV)
            if window_array.shape[1] == 5:
                window_array = self._engineer_features(window_array)
            
            # Validate feature count
            if window_array.shape[1] != self.config["n_features"]:
                if window_array.shape[1] < self.config["n_features"]:
                    padding = np.zeros((
                        expected_window, 
                        self.config["n_features"] - window_array.shape[1]
                    ))
                    window_array = np.hstack([window_array, padding])
                else:
                    window_array = window_array[:, :self.config["n_features"]]
            
            # Normalize
            window_array = self._normalize(window_array)
            
            # Transpose for CNN: (features, window_size)
            tensor = torch.FloatTensor(window_array.T)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)
        return batch
    
    def _engineer_features(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        Apply basic feature engineering to raw OHLCV data.
        
        For production, use the same pipeline as training by importing
        from pipelines.rl_agent_policy.data.feature_engine
        """
        o, h, l, c, v = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        
        features = [o, h, l, c, v]
        
        # Returns
        returns = np.zeros_like(c)
        returns[1:] = (c[1:] - c[:-1]) / (c[:-1] + 1e-8)
        features.append(returns)
        
        # Log returns
        log_returns = np.zeros_like(c)
        log_returns[1:] = np.log(c[1:] / (c[:-1] + 1e-8) + 1e-8)
        features.append(log_returns)
        
        # Price range
        price_range = (h - l) / (c + 1e-8)
        features.append(price_range)
        
        # Body
        body = (c - o) / (h - l + 1e-8)
        features.append(body)
        
        # Volume change
        vol_change = np.zeros_like(v)
        vol_change[1:] = (v[1:] - v[:-1]) / (v[:-1] + 1e-8)
        features.append(vol_change)
        
        # Moving averages
        for window in [5, 10, 20]:
            if len(c) >= window:
                sma = np.convolve(c, np.ones(window)/window, mode='same')
                features.append(sma)
                features.append((c - sma) / (sma + 1e-8))
        
        # Volatility
        for window in [5, 10]:
            if len(returns) >= window:
                vol = np.zeros_like(returns)
                for i in range(window, len(returns)):
                    vol[i] = np.std(returns[i-window:i])
                features.append(vol)
        
        result = np.column_stack(features)
        
        # Pad to expected features
        if result.shape[1] < self.config["n_features"]:
            padding = np.zeros((
                result.shape[0], 
                self.config["n_features"] - result.shape[1]
            ))
            result = np.hstack([result, padding])
        
        return result[:, :self.config["n_features"]]
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalization per feature."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std
    
    def inference(self, data: torch.Tensor) -> torch.Tensor:
        """Run model inference."""
        with torch.no_grad():
            shared_out = self.shared_network(data)
            logits = self.actor_head(shared_out)
            probs = torch.softmax(logits, dim=-1)
        return probs
    
    def postprocess(self, inference_output: torch.Tensor) -> List[Dict[str, Any]]:
        """Convert model output to trading signals."""
        trade_sizes = self.config["trade_sizes"]
        n_sizes = len(trade_sizes)
        
        results = []
        probs = inference_output.cpu().numpy()
        
        for batch_probs in probs:
            action = int(batch_probs.argmax())
            confidence = float(batch_probs.max())
            
            # Decode action
            if action == 0:
                signal = "HOLD"
                size = 0.0
            elif 1 <= action <= n_sizes:
                signal = "BUY"
                size = trade_sizes[action - 1]
            elif n_sizes + 1 <= action <= 2 * n_sizes:
                signal = "SELL"
                size = trade_sizes[action - n_sizes - 1]
            else:
                signal = "HOLD"
                size = 0.0
            
            result = {
                "action_id": action,
                "signal": signal,
                "position_size": size,
                "confidence": confidence,
                "probabilities": {
                    "hold": float(batch_probs[0]),
                    "buy": float(batch_probs[1:n_sizes+1].sum()),
                    "sell": float(batch_probs[n_sizes+1:].sum())
                }
            }
            results.append(result)
        
        return results
    
    def handle(self, data: List[Dict], context) -> List[Dict]:
        """Entry point for TorchServe requests."""
        if not self.initialized:
            self.initialize(context)
        
        if data is None:
            return None
        
        preprocessed = self.preprocess(data)
        output = self.inference(preprocessed)
        result = self.postprocess(output)
        
        return result
