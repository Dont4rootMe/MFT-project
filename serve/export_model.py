#!/usr/bin/env python
"""
Export trained A2C model for TorchServe.

This script:
1. Loads the trained model checkpoint from outputs/
2. Extracts configuration from the experiment
3. Creates the model archive (.mar file)

The handler imports models from pipelines/ - no code duplication!

Usage:
    python serve/export_model.py --checkpoint outputs/model/checkpoints/latest
"""

import sys
import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch


def load_config_from_checkpoint(checkpoint_path: Path) -> dict:
    """
    Load configuration from checkpoint meta.json.
    This is the preferred method - uses exact config from training.
    """
    meta_path = checkpoint_path / 'meta.json'
    
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    if 'config' in meta:
        config = meta['config']
        print(f"   ✓ Loaded config from checkpoint")
        return {
            "window_size": config.get('window_size', 30),
            "trade_sizes": config.get('trade_sizes', [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]),
            "n_actions": config.get('n_actions', 15),
            "n_features": config.get('n_features', 39),
            "action_scheme_name": config.get('action_scheme_name', 'simple'),
            "assets": config.get('assets', []),
            "seed": config.get('seed', 42),
        }
    else:
        # Fallback for old checkpoints
        print(f"   ⚠️ Checkpoint doesn't contain config, using Hydra")
        return load_config_from_hydra()


def load_config_from_hydra():
    """Fallback: Load configuration from Hydra config."""
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    
    config_dir = str(project_root / 'conf')
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='a2c_trainer')
    
    action_cfg = cfg.get('action_scheme', {})
    trade_sizes = list(action_cfg.get('trade_sizes', [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]))
    
    return {
        "window_size": cfg.env.get('window_size', 30),
        "trade_sizes": trade_sizes,
        "n_actions": 1 + 2 * len(trade_sizes),
        "n_features": 39  # Will be updated from actual model
    }


def export_model(
    checkpoint_path: str,
    output_dir: str = "serve/model-store",
    model_name: str = "a2c-trading"
):
    """
    Export model for TorchServe.
    
    Args:
        checkpoint_path: Path to model checkpoint (e.g., outputs/model/checkpoints/latest)
        output_dir: Output directory for .mar file
        model_name: Name of the model archive
    """
    print(f"🚀 Exporting model for TorchServe...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_dir}")
    print(f"   Model name: {model_name}")
    
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    serve_dir = project_root / "serve"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration from checkpoint (preferred) or fallback to Hydra
    print(f"\n⚙️  Loading configuration from checkpoint...")
    try:
        config = load_config_from_checkpoint(checkpoint_path)
    except (FileNotFoundError, KeyError) as e:
        print(f"   ⚠️ {e}, falling back to Hydra config")
        config = load_config_from_hydra()
    
    print(f"   Window size: {config['window_size']}")
    print(f"   Trade sizes: {config['trade_sizes']}")
    print(f"   N actions: {config['n_actions']}")
    
    # Load model state from checkpoint
    print(f"\n📦 Loading model checkpoint from {checkpoint_path}...")
    state_path = checkpoint_path / 'model' / 'state.pt'
    if not state_path.exists():
        raise FileNotFoundError(f"Model state not found at {state_path}")
    
    model_state = torch.load(state_path, map_location='cpu')
    print(f"   ✓ Loaded model state")
    
    # Extract n_features from model weights
    shared_state = model_state['shared']
    for key in shared_state.keys():
        if 'conv1.weight' in key:
            n_features = shared_state[key].shape[1]
            config['n_features'] = n_features
            print(f"   ✓ Detected n_features: {n_features}")
            break
    
    # Load experiment metadata if available
    meta_path = checkpoint_path / 'meta.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        config['experiment'] = {
            'episode': meta.get('episode'),
            'checkpoint_path': str(checkpoint_path)
        }
        print(f"   ✓ Loaded experiment metadata (episode {meta.get('episode')})")
    
    # Create temporary export directory
    export_tmp = serve_dir / "export_tmp"
    if export_tmp.exists():
        shutil.rmtree(export_tmp)
    export_tmp.mkdir(parents=True)
    
    # Save model state
    print(f"\n💾 Saving model artifacts...")
    model_path = export_tmp / "model.pt"
    torch.save(model_state, model_path)
    print(f"   ✓ Saved model.pt ({model_path.stat().st_size / 1024:.1f} KB)")
    
    # Save configuration (extracted from conf/ and checkpoint)
    config_path = export_tmp / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   ✓ Saved config.json (from conf/a2c_trainer.yaml)")
    
    # Copy handler (uses imports from pipelines/)
    shutil.copy(serve_dir / "handler.py", export_tmp / "handler.py")
    print(f"   ✓ Copied handler.py (imports from pipelines/)")
    
    # Create model archive
    print(f"\n📦 Creating model archive...")
    mar_path = output_dir / f"{model_name}.mar"
    
    # Remove existing .mar if present
    if mar_path.exists():
        mar_path.unlink()
    
    # Try torch-model-archiver first
    cmd = [
        "torch-model-archiver",
        "--model-name", model_name,
        "--version", "1.0",
        "--handler", str(export_tmp / "handler.py"),
        "--extra-files", f"{export_tmp}/model.pt,{export_tmp}/config.json",
        "--export-path", str(output_dir),
        "--force"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Created {mar_path} (torch-model-archiver)")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"   ⚠️ torch-model-archiver not available, creating manual archive...")
        
        # Manual archive creation
        import zipfile
        with zipfile.ZipFile(mar_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(export_tmp / "model.pt", "model.pt")
            zf.write(export_tmp / "config.json", "config.json")
            zf.write(export_tmp / "handler.py", "handler.py")
            
            # Create MAR-MANIFEST.json
            manifest = {
                "createdOn": "01/10/2026 00:00:00",
                "runtime": "python",
                "model": {
                    "modelName": model_name,
                    "serializedFile": "model.pt",
                    "handler": "handler.py",
                    "modelVersion": "1.0"
                },
                "archiverVersion": "0.9.0"
            }
            zf.writestr("MAR-INF/MANIFEST.json", json.dumps(manifest, indent=2))
        
        print(f"   ✓ Created {mar_path} (manual archive)")
    
    # Cleanup
    shutil.rmtree(export_tmp)
    
    print(f"\n✅ Model export complete!")
    print(f"   Archive: {mar_path}")
    print(f"   Size: {mar_path.stat().st_size / 1024:.1f} KB")
    print(f"\n📝 Note: The handler imports models from pipelines/rl_agent_policy/models/")
    print(f"   Make sure to include pipelines/ and conf/ when building Docker image.")
    
    return str(mar_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export A2C model for TorchServe",
        epilog="""
Example:
    python serve/export_model.py --checkpoint outputs/model/checkpoints/latest
    
The exported .mar file contains:
    - model.pt: Model weights from the checkpoint
    - config.json: Configuration extracted from conf/a2c_trainer.yaml
    - handler.py: Custom handler (imports from pipelines/)
"""
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/model/checkpoints/latest",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="serve/model-store",
        help="Output directory for .mar file"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="a2c-trading",
        help="Model name for the archive"
    )
    
    args = parser.parse_args()
    
    export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        model_name=args.name
    )


if __name__ == "__main__":
    main()
