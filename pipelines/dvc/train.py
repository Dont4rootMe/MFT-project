"""
DVC Pipeline Stage: Model Training

This script wraps the main trainer to provide a DVC-compatible interface.
It trains the A2C agent and saves the model checkpoint.
"""

import sys
from pathlib import Path
import argparse
import os
import shutil

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import subprocess


def train_model(
    config_path: str,
    output_dir: str,
    use_accelerate: bool = False
) -> None:
    """
    Train the model using the main trainer.
    
    Args:
        config_path: Path to training configuration (or hydra config name)
        output_dir: Directory to save model outputs
        use_accelerate: Whether to use accelerate for distributed training
    """
    print(f"🚀 Starting model training...")
    print(f"   Config: {config_path}")
    print(f"   Output: {output_dir}")
    print(f"   Use Accelerate: {use_accelerate}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the training command
    trainer_script = project_root / 'pipelines' / 'rl_agent_policy' / 'trainer.py'
    
    # Override output directory for DVC pipeline
    hydra_overrides = [
        f"output_dir={output_dir}",
        f"use_accelerate={str(use_accelerate).lower()}",
    ]
    
    if use_accelerate:
        cmd = [
            'accelerate', 'launch',
            str(trainer_script),
            *hydra_overrides
        ]
    else:
        cmd = [
            'python',
            str(trainer_script),
            *hydra_overrides
        ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")
    
    # Copy the latest checkpoint to a standard location for DVC
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    latest_checkpoint = os.path.join(checkpoints_dir, 'latest')
    
    if os.path.exists(latest_checkpoint):
        # Copy to outputs/model for DVC tracking
        dvc_model_dir = os.path.join(project_root, 'outputs', 'model')
        os.makedirs(dvc_model_dir, exist_ok=True)
        
        # Copy checkpoint
        dvc_checkpoint_dir = os.path.join(dvc_model_dir, 'checkpoints', 'latest')
        if os.path.exists(dvc_checkpoint_dir):
            shutil.rmtree(dvc_checkpoint_dir)
        
        os.makedirs(os.path.dirname(dvc_checkpoint_dir), exist_ok=True)
        shutil.copytree(latest_checkpoint, dvc_checkpoint_dir)
        
        print(f"   ✓ Model checkpoint copied to {dvc_checkpoint_dir}")
    
    print(f"✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train the A2C model (DVC pipeline wrapper)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='a2c_trainer',
        help='Hydra configuration name or path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/model',
        help='Output directory for model and checkpoints'
    )
    parser.add_argument(
        '--accelerate',
        action='store_true',
        help='Use accelerate for distributed training'
    )
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        output_dir=args.output,
        use_accelerate=args.accelerate
    )


if __name__ == '__main__':
    main()

