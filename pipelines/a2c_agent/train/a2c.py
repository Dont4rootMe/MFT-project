"""Training utilities for A2C agents."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import MultiDiscrete

from tensortrade.agents import ReplayMemory
from tensortrade.agents.a2c_agent import A2CTransition


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
    n_episodes: int = 1
    output_dir: str = "runs/a2c"
    checkpoint_every_steps: int = 1000
    resume_path: Optional[str] = None


class A2CTrainer:
    """Simple trainer implementing the A2C algorithm with validation."""

    def __init__(self, agent, train_env, valid_env, config: Optional[A2CConfig] = None):
        self.agent = agent
        self.train_env = train_env
        self.valid_env = valid_env
        self.cfg = config or A2CConfig()

        self.output_dir = Path(self.cfg.output_dir)
        self.ckpt_root = self.output_dir / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        self._start_episode = 0
        self._start_step = 0

        resume_path = self.cfg.resume_path
        if resume_path:
            self._load_checkpoint(resume_path)
        elif (self.ckpt_root / "latest").exists():
            self._load_checkpoint(self.ckpt_root / "latest")

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

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.agent.device)
        if states.ndim == 3:
            states = states.permute(0, 2, 1)

        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.agent.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.agent.device)
        dones = torch.tensor(batch.done, dtype=torch.bool, device=self.agent.device)
        values = torch.stack(batch.value).to(self.agent.device)

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

        self.agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent.shared_network.parameters())
            + list(self.agent.actor_head.parameters())
            + list(self.agent.critic_head.parameters()),
            max_norm=0.5,
        )
        self.agent.optimizer.step()

    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int, episode: int):
        ckpt_dir = self.ckpt_root / str(step)
        model_dir = ckpt_dir / "model"
        trainer_dir = ckpt_dir / "trainer"
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer_dir.mkdir(parents=True, exist_ok=True)

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
        model_state = torch.load(ckpt_dir / "model" / "state.pt", map_location=self.agent.device)
        self.agent.shared_network.load_state_dict(model_state["shared"])
        self.agent.actor_head.load_state_dict(model_state["actor"])
        self.agent.critic_head.load_state_dict(model_state["critic"])
        opt_state = torch.load(ckpt_dir / "trainer" / "optimizer.pt", map_location=self.agent.device)
        self.agent.optimizer.load_state_dict(opt_state)
        sched_path = ckpt_dir / "trainer" / "scheduler.pt"
        if hasattr(self.agent, "scheduler") and sched_path.exists():
            self.agent.scheduler.load_state_dict(torch.load(sched_path, map_location=self.agent.device))
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._start_step = meta.get("step", 0)
            self._start_episode = meta.get("episode", 0)

    def train(self):
        cfg = self.cfg
        # update learning rate
        for pg in self.agent.optimizer.param_groups:
            pg["lr"] = cfg.learning_rate

        memory = ReplayMemory(cfg.memory_capacity, transition_type=A2CTransition)
        episode = self._start_episode
        steps_done = self._start_step
        if cfg.n_steps and not cfg.n_episodes:
            cfg.n_episodes = np.iinfo(np.int32).max

        while episode < cfg.n_episodes:
            state, _ = self.train_env.reset()
            done = False
            while not done:
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
                    _, value = self.agent._infer(state)
                memory.push(state, action_idx, reward, done, value)

                state = next_state
                steps_done += 1

                if len(memory) >= cfg.batch_size:
                    self._apply_gradient_descent(memory)

                if (
                    cfg.checkpoint_every_steps
                    and steps_done % cfg.checkpoint_every_steps == 0
                ):
                    self._save_checkpoint(steps_done, episode)

                if cfg.n_steps and steps_done >= cfg.n_steps:
                    done = True
            # run validation once per episode
            self._validate()
            episode += 1

        self._save_checkpoint(steps_done, episode)

    def _validate(self):
        state, _ = self.valid_env.reset(start_from_time=True)
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
        print(f"Validation reward: {total_reward:.4f}")
