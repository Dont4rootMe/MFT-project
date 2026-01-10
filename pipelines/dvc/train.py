"""
DVC Pipeline Stage: Model Training with MLflow Integration

This script wraps the main trainer to provide a DVC-compatible interface
and integrates MLflow for experiment tracking.
"""

import sys
from pathlib import Path
import argparse
import os
import shutil
import json
import hashlib
from datetime import datetime

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import subprocess
import yaml
import mlflow
from mlflow.models import infer_signature


def get_dvc_hash(dvc_file_path: str) -> str:
    """Extract the MD5 hash from a .dvc file."""
    if not os.path.exists(dvc_file_path):
        return "unknown"
    
    with open(dvc_file_path, 'r') as f:
        dvc_content = yaml.safe_load(f)
    
    if dvc_content and 'outs' in dvc_content:
        for out in dvc_content['outs']:
            if 'md5' in out:
                return out['md5']
    return "unknown"


def load_config_params(config_path: str) -> dict:
    """Load parameters from a YAML config file."""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    return config


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dictionary for MLflow parameter logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        elif v is not None:
            items.append((new_key, v))
    return dict(items)


def train_model(
    config_path: str,
    output_dir: str,
    use_accelerate: bool = False,
    experiment_name: str = "mft-a2c-training",
    run_name: str = None
) -> None:
    """
    Train the model using the main trainer with MLflow tracking.
    
    Args:
        config_path: Path to training configuration (or hydra config name)
        output_dir: Directory to save model outputs
        use_accelerate: Whether to use accelerate for distributed training
        experiment_name: MLflow experiment name
        run_name: Optional run name (auto-generated if not provided)
    """
    print(f"🚀 Starting model training with MLflow tracking...")
    print(f"   Config: {config_path}")
    print(f"   Output: {output_dir}")
    print(f"   Use Accelerate: {use_accelerate}")
    print(f"   Experiment: {experiment_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"a2c_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load configuration files for parameter logging
    config_dir = project_root / 'conf'
    main_config = load_config_params(config_dir / 'a2c_trainer.yaml')
    train_config = load_config_params(config_dir / 'train' / 'a2c_train.yaml')
    data_config = load_config_params(config_dir / 'data' / 'default_data.yaml')
    model_config = load_config_params(config_dir / 'model' / 'a2c_model.yaml')
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        print(f"   MLflow Run ID: {run.info.run_id}")
        print(f"   MLflow Run Name: {run_name}")
        
        # ========== LOG PARAMETERS ==========
        # Log main training parameters
        if main_config:
            flat_main = flatten_dict(main_config, 'config')
            # Filter out _target_ and other non-parameter keys
            flat_main = {k: v for k, v in flat_main.items() 
                        if not k.endswith('_target_') and v is not None}
            mlflow.log_params({k: v for k, v in list(flat_main.items())[:100]})  # MLflow limit
        
        # Log training approach parameters
        if train_config and 'approach' in train_config:
            approach_params = train_config['approach']
            flat_approach = flatten_dict(approach_params, 'train')
            flat_approach = {k: v for k, v in flat_approach.items() 
                           if not k.endswith('_target_') and v is not None}
            mlflow.log_params({k: v for k, v in list(flat_approach.items())[:50]})
        
        # Log data configuration
        if data_config:
            mlflow.log_param("data.symbols", str(data_config.get('symbols', [])))
            mlflow.log_param("data.time_freq", data_config.get('time_freq', '1d'))
            mlflow.log_param("data.exchange", data_config.get('exchange', 'Binance'))
        
        # ========== LOG DVC TAGS ==========
        # Get DVC data hash for reproducibility tracking
        dvc_raw_hash = get_dvc_hash(project_root / 'data' / 'raw.dvc')
        mlflow.set_tag("dvc_data_hash", dvc_raw_hash)
        mlflow.set_tag("dvc_version", "tracked")
        mlflow.set_tag("pipeline", "dvc")
        
        # Log git info if available
        try:
            git_hash = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=str(project_root), 
                capture_output=True, text=True
            ).stdout.strip()
            if git_hash:
                mlflow.set_tag("git_commit", git_hash[:8])
        except:
            pass
        
        # ========== RUN TRAINING ==========
        trainer_script = project_root / 'pipelines' / 'rl_agent_policy' / 'trainer.py'
        
        hydra_overrides = [
            f"output_dir={output_dir}",
            f"use_accelerate={str(use_accelerate).lower()}",
        ]
        
        if use_accelerate:
            cmd = ['accelerate', 'launch', str(trainer_script), *hydra_overrides]
        else:
            cmd = ['python', str(trainer_script), *hydra_overrides]
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Run training
        start_time = datetime.now()
        result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Log training duration
        mlflow.log_metric("training_duration_seconds", training_duration)
        
        # Print training output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_code", result.returncode)
            raise RuntimeError(f"Training failed with return code {result.returncode}")
        
        mlflow.set_tag("status", "success")
        
        # ========== PROCESS CHECKPOINTS ==========
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        
        if os.path.exists(checkpoints_dir):
            checkpoint_dirs = []
            for d in os.listdir(checkpoints_dir):
                dir_path = os.path.join(checkpoints_dir, d)
                if os.path.isdir(dir_path) and d.isdigit():
                    checkpoint_dirs.append((int(d), dir_path))
            
            if checkpoint_dirs:
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
                
                # Log checkpoint step
                mlflow.log_metric("final_checkpoint_step", latest_num)
        
        # ========== LOG ARTIFACTS ==========
        # Log dvc.lock as artifact for reproducibility
        dvc_lock_path = project_root / 'dvc.lock'
        if dvc_lock_path.exists():
            mlflow.log_artifact(str(dvc_lock_path), "dvc")
            print(f"   ✓ Logged dvc.lock as artifact")
        
        # Log model checkpoint as artifact
        latest_checkpoint = os.path.join(checkpoints_dir, 'latest') if os.path.exists(checkpoints_dir) else None
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            mlflow.log_artifacts(latest_checkpoint, "model/checkpoint")
            print(f"   ✓ Logged model checkpoint as artifact")
        
        # Log configuration files as artifacts
        for config_file in ['a2c_trainer.yaml']:
            config_file_path = config_dir / config_file
            if config_file_path.exists():
                mlflow.log_artifact(str(config_file_path), "config")
        
        # Log training logs if available
        train_log_path = os.path.join(output_dir, 'a2c_train.log')
        if os.path.exists(train_log_path):
            mlflow.log_artifact(train_log_path, "logs")
        
        # Log validation plots if available
        validation_dir = os.path.join(output_dir, 'validation_logging')
        if os.path.exists(validation_dir):
            for step_dir in os.listdir(validation_dir):
                step_path = os.path.join(validation_dir, step_dir)
                if os.path.isdir(step_path):
                    for plot_file in os.listdir(step_path):
                        if plot_file.endswith(('.html', '.png')):
                            mlflow.log_artifact(
                                os.path.join(step_path, plot_file),
                                f"plots/{step_dir}"
                            )
            print(f"   ✓ Logged validation plots as artifacts")
        
        # ========== LOG METRICS FROM EVALUATION ==========
        # Check if final_reward.txt exists
        final_reward_path = os.path.join(output_dir, 'final_reward.txt')
        if os.path.exists(final_reward_path):
            with open(final_reward_path, 'r') as f:
                final_reward = float(f.read().strip())
            mlflow.log_metric("final_reward", final_reward)
            print(f"   ✓ Final reward: {final_reward:.4f}")
        
        # Log evaluation CSV if available
        eval_csv_path = os.path.join(output_dir, 'evaluation.csv')
        if os.path.exists(eval_csv_path):
            mlflow.log_artifact(eval_csv_path, "evaluation")
        
        print(f"✅ Training complete!")
        print(f"   MLflow Run ID: {run.info.run_id}")
        print(f"   View results: mlflow ui --port 5000")


def main():
    parser = argparse.ArgumentParser(
        description="Train the A2C model with MLflow tracking (DVC pipeline wrapper)"
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
    parser.add_argument(
        '--experiment',
        type=str,
        default='mft-a2c-training',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name (auto-generated if not provided)'
    )
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        output_dir=args.output,
        use_accelerate=args.accelerate,
        experiment_name=args.experiment,
        run_name=args.run_name
    )


if __name__ == '__main__':
    main()
