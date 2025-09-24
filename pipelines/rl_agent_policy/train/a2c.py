"""Training utilities for A2C agents."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import shutil

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import MultiDiscrete
from tqdm import tqdm

from tensortrade.agents import ReplayMemory
from tensortrade.agents.a2c_agent import A2CTransition

# Try to import accelerate, but make it optional
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None


def _set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_process_seed(base_seed: Optional[int], process_index: int = 0) -> int:
    """
    Generate a process-specific seed to ensure different processes don't use the same seed.
    
    Args:
        base_seed: Base seed from configuration
        process_index: Index of the current process (0 for single process)
        
    Returns:
        int: Process-specific seed
    """
    if base_seed is None:
        # Generate a random base seed if none provided
        base_seed = random.randint(0, 2**32 - 1)
    
    # Create process-specific seed by combining base seed with process index
    # Use a prime number to ensure good distribution
    process_seed = (base_seed + process_index * 1009) % (2**32)
    
    return process_seed


def _get_auto_device() -> torch.device:
    """
    Automatically detect the best available device, respecting CUDA_VISIBLE_DEVICES.
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible is not None:
            visible_devices = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
            if visible_devices:
                # Use the first visible device
                device_id = visible_devices[0]
                device = torch.device(f'cuda:{device_id}')
                print(f"🎯 Auto-detected device: {device} (from CUDA_VISIBLE_DEVICES={cuda_visible})")
                return device
        
        # Default to cuda:0 if CUDA_VISIBLE_DEVICES is not set or invalid
        device = torch.device('cuda:0')
        print(f"🎯 Auto-detected device: {device} (CUDA available)")
        return device
    else:
        device = torch.device('cpu')
        print(f"🎯 Auto-detected device: {device} (CUDA not available)")
        return device


@dataclass
class A2CConfig:
    """Configuration parameters for A2C training."""

    batch_size: int = 128
    discount_factor: float = 0.9999
    learning_rate: float = 0.0001
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay_steps: int = 200
    entropy_c: float = 0.0001
    memory_capacity: int = 1000
    n_steps: Optional[int] = None
    n_episodes: int = 100_000
    validate_every_episodes: int = 500
    checkpoint_every_episodes: int = 1000
    resume_path: Optional[str] = None
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients before updating
    seed: Optional[int] = None  # Base seed for reproducibility


class A2CTrainer:
    """Simple trainer implementing the A2C algorithm with validation."""

    def __init__(self, 
                 agent, 
                 train_env, 
                 valid_env, 
                 output_dir, 
                 max_episode_length,
                 config: Optional[A2CConfig] = None,
                 use_accelerate: Optional[bool | None] = None 
    ):
        self.agent = agent
        self.train_env = train_env
        self.valid_env = valid_env
        self.cfg = config or A2CConfig()
        self.output_dir = output_dir
        self.max_episode_length = max_episode_length
        
        # Determine whether to use accelerate or manual device management
        self.use_accelerate = self._should_use_accelerate(use_accelerate)
        
        # Initialize seed management
        if self.use_accelerate:
            # Initialize accelerator with gradient accumulation
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.cfg.gradient_accumulation_steps
            )
            self.device = self.accelerator.device
            process_index = self.accelerator.process_index
        else:
            # Use manual device management
            self.accelerator = None
            self.device = _get_auto_device()
            process_index = 0
            # Move agent to device
            self.agent.shared_network.to(self.device)
            self.agent.actor_head.to(self.device)
            self.agent.critic_head.to(self.device)
        
        # Set process-specific seed for reproducibility
        self.process_seed = _get_process_seed(self.cfg.seed, process_index)
        _set_seed(self.process_seed)
        
        # Log seed information
        is_main_process = not self.use_accelerate or self.accelerator.is_main_process
        if is_main_process:
            print(f"🌱 Seed Configuration:")
            print(f"   Base seed: {self.cfg.seed}")
            print(f"   Process seed: {self.process_seed}")
            if self.use_accelerate:
                print(f"   Process index: {process_index}")
                print(f"   Total processes: {self.accelerator.num_processes}")

        self.output_dir = Path(output_dir)
        self.ckpt_root = self.output_dir / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        
        self.valid_output_dir = self.output_dir / "validation_logging"
        self.valid_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode = 0
        self._start_episode = 0
        self._start_step = 0
        self._accumulation_step = 0  # Track gradient accumulation steps

        # Prepare models and optimizer (only if using accelerate)
        if self.use_accelerate:
            self.agent.shared_network, self.agent.actor_head, self.agent.critic_head, self.agent.optimizer = self.accelerator.prepare(
                self.agent.shared_network, 
                self.agent.actor_head, 
                self.agent.critic_head, 
                self.agent.optimizer
            )

        resume_path = self.cfg.resume_path
        if resume_path:
            self._load_checkpoint(resume_path)
        elif (self.ckpt_root / "latest").exists():
            self._load_checkpoint(self.ckpt_root / "latest")

    def _should_use_accelerate(self, use_accelerate) -> bool:
        """
        Determine whether to use accelerate or manual device management.
        
        Returns:
            bool: True if should use accelerate, False for manual device management
        """
        # If explicitly set in config, use that
        if use_accelerate is not None:
            if use_accelerate and not ACCELERATE_AVAILABLE:
                print("⚠️  Warning: use_accelerate=True but accelerate not available. Falling back to manual device management.")
                return False
            return use_accelerate
        
        # Auto-detect: use accelerate if available and we're in a distributed context
        if ACCELERATE_AVAILABLE:
            # Check if we're running with accelerate launch (common environment variables)
            distributed_env_vars = [
                'LOCAL_RANK', 'RANK', 'WORLD_SIZE', 
                'MASTER_ADDR', 'MASTER_PORT',
                'ACCELERATE_USE_FSDP', 'ACCELERATE_USE_DEEPSPEED'
            ]
            
            if any(var in os.environ for var in distributed_env_vars):
                print("🚀 Detected distributed training environment. Using Accelerate.")
                return True
            
            # Check if mixed precision is requested via environment
            if os.environ.get('ACCELERATE_MIXED_PRECISION', '').lower() in ['fp16', 'bf16']:
                print("🚀 Detected mixed precision request. Using Accelerate.")
                return True
        
        # Default to manual device management
        print("🎯 Using manual device management (no distributed training detected).")
        return False

    # ------------------------------------------------------------------
    # Core optimisation helpers (adapted from the original agent)
    # ------------------------------------------------------------------
    def _compute_returns(self, rewards, dones):
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.cfg.discount_factor * G * (1 - int(d))
            returns.append(G)
        return torch.tensor(list(reversed(returns)), dtype=torch.float32, device=self.agent.device)

    def _apply_gradient_descent(self, memory: ReplayMemory):
        transitions = memory.tail(self.cfg.batch_size)
        batch = A2CTransition(*zip(*transitions))

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        if states.ndim == 3:
            states = states.permute(0, 2, 1)

        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        values = torch.stack(batch.value).to(self.device)

        returns = self._compute_returns(rewards, dones)
        advantages = returns - values.detach()

        features = self.agent.shared_network(states)
        logits = self.agent.actor_head(features)
        new_values = self.agent.critic_head(features).squeeze(-1)

        critic_loss = F.huber_loss(new_values, returns)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -(selected_log_probs * advantages).mean()
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        loss = actor_loss + 0.5 * critic_loss - self.cfg.entropy_c * entropy

        # Handle gradient accumulation
        if self.use_accelerate:
            # Use accelerate with gradient accumulation
            with self.accelerator.accumulate(self.agent.shared_network):
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        list(self.agent.shared_network.parameters())
                        + list(self.agent.actor_head.parameters())
                        + list(self.agent.critic_head.parameters()),
                        max_norm=0.5,
                    )
                self.agent.optimizer.step()
                self.agent.optimizer.zero_grad()
        else:
            # Manual gradient accumulation
            loss = loss / self.cfg.gradient_accumulation_steps  # Scale loss for accumulation
            loss.backward()
            
            self._accumulation_step += 1
            
            # Only update weights after accumulating enough gradients
            if self._accumulation_step % self.cfg.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    list(self.agent.shared_network.parameters())
                    + list(self.agent.actor_head.parameters())
                    + list(self.agent.critic_head.parameters()),
                    max_norm=0.5,
                )
                self.agent.optimizer.step()
                self.agent.optimizer.zero_grad()
                self._accumulation_step = 0

    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int, episode: int):
        # Only save on main process in distributed training
        if self.use_accelerate and not self.accelerator.is_main_process:
            return
            
        ckpt_dir = self.ckpt_root / str(step)
        model_dir = ckpt_dir / "model"
        trainer_dir = ckpt_dir / "trainer"
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer_dir.mkdir(parents=True, exist_ok=True)

        if self.use_accelerate:
            # Use accelerator to unwrap models for saving
            self.accelerator.save(
                {
                    "shared": self.accelerator.unwrap_model(self.agent.shared_network).state_dict(),
                    "actor": self.accelerator.unwrap_model(self.agent.actor_head).state_dict(),
                    "critic": self.accelerator.unwrap_model(self.agent.critic_head).state_dict(),
                },
                model_dir / "state.pt",
            )
            # Save optimizer state
            self.accelerator.save(self.agent.optimizer.state_dict(), trainer_dir / "optimizer.pt")
            
            if hasattr(self.agent, "scheduler"):
                self.accelerator.save(self.agent.scheduler.state_dict(), trainer_dir / "scheduler.pt")
        else:
            # Manual saving without accelerator
            torch.save(
                {
                    "shared": self.agent.shared_network.state_dict(),
                    "actor": self.agent.actor_head.state_dict(),
                    "critic": self.agent.critic_head.state_dict(),
                },
                model_dir / "state.pt",
            )
            torch.save(self.agent.optimizer.state_dict(), trainer_dir / "optimizer.pt")
            
            if hasattr(self.agent, "scheduler"):
                torch.save(self.agent.scheduler.state_dict(), trainer_dir / "scheduler.pt")
            
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump({"step": step, "episode": episode}, f)

        latest_dir = self.ckpt_root / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(ckpt_dir, latest_dir)

    def _load_checkpoint(self, path):
        ckpt_dir = Path(path)
        
        # Load model states with proper device handling
        model_state = torch.load(ckpt_dir / "model" / "state.pt", map_location=self.device)
        
        if self.use_accelerate:
            # Load with accelerator unwrapping
            self.accelerator.unwrap_model(self.agent.shared_network).load_state_dict(model_state["shared"])
            self.accelerator.unwrap_model(self.agent.actor_head).load_state_dict(model_state["actor"])
            self.accelerator.unwrap_model(self.agent.critic_head).load_state_dict(model_state["critic"])
        else:
            # Direct loading without accelerator
            self.agent.shared_network.load_state_dict(model_state["shared"])
            self.agent.actor_head.load_state_dict(model_state["actor"])
            self.agent.critic_head.load_state_dict(model_state["critic"])
        
        # Load optimizer state
        opt_state = torch.load(ckpt_dir / "trainer" / "optimizer.pt", map_location=self.device)
        self.agent.optimizer.load_state_dict(opt_state)
        
        # Load scheduler state if exists
        sched_path = ckpt_dir / "trainer" / "scheduler.pt"
        if hasattr(self.agent, "scheduler") and sched_path.exists():
            sched_state = torch.load(sched_path, map_location=self.device)
            self.agent.scheduler.load_state_dict(sched_state)
            
        # Load metadata
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._start_step = meta.get("step", 0)
            self._start_episode = meta.get("episode", 0)

    def train(self):
        cfg = self.cfg
        
        # Print training configuration
        is_main_process = not self.use_accelerate or self.accelerator.is_main_process
        if is_main_process:
            if self.use_accelerate:
                print(f"🚀 Accelerate Training Configuration:")
                print(f"   Device: {self.accelerator.device}")
                print(f"   Distributed: {self.accelerator.distributed_type}")
                print(f"   Mixed Precision: {self.accelerator.mixed_precision}")
                print(f"   Number of processes: {self.accelerator.num_processes}")
                print(f"   Process index: {self.accelerator.process_index}")
                print(f"   Gradient Accumulation Steps: {cfg.gradient_accumulation_steps}")
                if cfg.gradient_accumulation_steps > 1:
                    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
                    print(f"   Effective Batch Size: {effective_batch_size} ({cfg.batch_size} × {cfg.gradient_accumulation_steps})")
            else:
                print(f"🎯 Manual Device Training Configuration:")
                print(f"   Device: {self.device}")
                print(f"   CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   CUDA Device Count: {torch.cuda.device_count()}")
                    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
                print(f"   Gradient Accumulation Steps: {cfg.gradient_accumulation_steps}")
                if cfg.gradient_accumulation_steps > 1:
                    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
                    print(f"   Effective Batch Size: {effective_batch_size} ({cfg.batch_size} × {cfg.gradient_accumulation_steps})")
        
        # update learning rate
        for pg in self.agent.optimizer.param_groups:
            pg["lr"] = cfg.learning_rate

        memory = ReplayMemory(cfg.memory_capacity, transition_type=A2CTransition)
        
        self.episode = self._start_episode
        steps_done = self._start_step
        if cfg.n_steps and not cfg.n_episodes:
            cfg.n_episodes = np.iinfo(np.int32).max
        
        # Track last rewards for tqdm display
        last_train_reward = 0.0
        last_val_reward = 0.0
        
        # Create tqdm iterator for episodes (only on main process)
        episode_range = range(self.episode, cfg.n_episodes)
        if is_main_process:
            pbar = tqdm(episode_range, desc="", initial=self.episode, total=cfg.n_episodes)
        else:
            pbar = None
        
        for self.episode in episode_range:
            inner_index = 0
            state, _ = self.train_env.reset()
            done = False
            
            train_reward = 0
            while not done and (inner_index < self.max_episode_length or self.max_episode_length is None):
                inner_index += 1
                
                threshold = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(
                    -steps_done / cfg.eps_decay_steps
                )
                action_idx = self.agent.get_action(state, threshold=threshold)
                env_action = action_idx
                if isinstance(self.train_env.action_space, MultiDiscrete):
                    env_action = np.array(
                        np.unravel_index(action_idx, self.train_env.action_space.nvec)
                    ).astype(int)
                next_state, reward, terminated, truncated, _ = self.train_env.step(env_action)
                done = terminated or truncated

                with torch.no_grad():
                    # Move state to device for inference
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                    _, value = self.agent._infer(state_tensor.cpu().numpy())
                memory.push(state, action_idx, reward, done, value)

                state = next_state
                steps_done += 1
                train_reward += reward

                # Apply gradient descent with consideration for accumulation
                if len(memory) >= cfg.batch_size:
                    self._apply_gradient_descent(memory)

                if cfg.n_steps and steps_done >= cfg.n_steps:
                    done = True
            
            # Update last training reward and tqdm description
            last_train_reward = train_reward
            
            if (
                cfg.checkpoint_every_episodes
                and self.episode % cfg.checkpoint_every_episodes == 0
            ):
                self._save_checkpoint(steps_done, self.episode)
            
            # run validation once per episode (only on main process)
            if self.episode % cfg.validate_every_episodes == 0 and is_main_process:
                last_val_reward = self._validate()
            
            # Update tqdm description with current rewards (only on main process)
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Train: {last_train_reward:.2f} | Val: {last_val_reward:.2f}")
        
        # Close the progress bar and save final checkpoint (only on main process)
        if pbar is not None:
            pbar.close()
        
        # Wait for all processes to finish before saving final checkpoint (only if using accelerate)
        if self.use_accelerate:
            self.accelerator.wait_for_everyone()
        
        self._save_checkpoint(steps_done, self.episode)

    def _validate(self, save_validation_output: bool = True):
        # create output directory for current validation run
        validation_output_dir = self.valid_output_dir / f"step_{self.episode}"
        validation_output_dir.mkdir(parents=True, exist_ok=True)
        
        state, _ = self.valid_env.reset(begin_from_start=True)
        done = False
        total_reward = 0.0

        while not done:
            action_idx = self.agent.get_action(state)
            env_action = action_idx
            if isinstance(self.valid_env.action_space, MultiDiscrete):
                env_action = np.array(
                    np.unravel_index(action_idx, self.valid_env.action_space.nvec)
                ).astype(int)
            state, reward, terminated, truncated, _ = self.valid_env.step(env_action)
            done = terminated or truncated
            total_reward += reward
        
        # render and save information on the run
        self.valid_env.render()
        if save_validation_output:
            self.valid_env.save(validation_output_dir)

        return total_reward