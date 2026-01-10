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
    
    # Find the latest checkpoint and create a 'latest' symlink for evaluate stage
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    
    if os.path.exists(checkpoints_dir):
        # Find all numbered checkpoint directories
        checkpoint_dirs = []
        for d in os.listdir(checkpoints_dir):
            dir_path = os.path.join(checkpoints_dir, d)
            if os.path.isdir(dir_path) and d.isdigit():
                checkpoint_dirs.append((int(d), dir_path))
        
        if checkpoint_dirs:
            # Sort by number and get the latest
            checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
            latest_num, latest_path = checkpoint_dirs[0]
            
            # Create 'latest' symlink
            latest_link = os.path.join(checkpoints_dir, 'latest')
            if os.path.islink(latest_link):
                os.remove(latest_link)
            elif os.path.isdir(latest_link):
                shutil.rmtree(latest_link)
            elif os.path.exists(latest_link):
                os.remove(latest_link)
            
            os.symlink(str(latest_num), latest_link)
            print(f"   ✓ Created 'latest' symlink -> {latest_num}")
    
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

