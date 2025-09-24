"""Training utilities for PPO agents."""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import json
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensortrade.agents.a2c_agent import A2CTransition  # for type reference


@dataclass
class PPOConfig:
    """Configuration parameters for PPO training."""

    batch_size: int = 2048  # number of timesteps per rollout
    mini_batch_size: int = 64
    epochs: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: Optional[int] = None
    n_episodes: int = 1
    output_dir: str = "runs/ppo"
    checkpoint_every_steps: int = 1000
    resume_path: Optional[str] = None


class RolloutBuffer:
    """Simple buffer to collect rollouts for PPO."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """Trainer implementing the PPO algorithm with validation."""

    def __init__(self, agent, train_env, valid_env, config: Optional[PPOConfig] = None):
        self.agent = agent
        self.train_env = train_env
        self.valid_env = valid_env
        self.cfg = config or PPOConfig()

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
    def _compute_gae(self, rewards, dones, values, last_value):
        cfg = self.cfg
        values = np.append(values, last_value)
        gae = 0.0
        advantages = np.zeros_like(rewards, dtype=np.float32)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def _update(self, buffer: RolloutBuffer, last_value: float):
        cfg = self.cfg
        states = torch.tensor(np.stack(buffer.states), dtype=torch.float32, device=self.agent.device)
        if states.ndim == 3:
            states = states.permute(0, 2, 1)
        actions = torch.tensor(buffer.actions, dtype=torch.int64, device=self.agent.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=self.agent.device)
        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.bool_)
        values = np.array(buffer.values, dtype=np.float32)
        advantages, returns = self._compute_gae(rewards, dones, values, last_value)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.agent.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.agent.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = len(buffer)
        inds = np.arange(batch_size)
        for _ in range(cfg.epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                mb_inds = inds[start:end]

                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]

                features = self.agent.shared_network(mb_states)
                logits = self.agent.actor_head(features)
                new_values = self.agent.critic_head(features).squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(new_values, mb_returns)

                loss = actor_loss + cfg.vf_coef * critic_loss - cfg.ent_coef * entropy

                self.agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.agent.shared_network.parameters())
                    + list(self.agent.actor_head.parameters())
                    + list(self.agent.critic_head.parameters()),
                    cfg.max_grad_norm,
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
        for pg in self.agent.optimizer.param_groups:
            pg["lr"] = cfg.learning_rate

        buffer = RolloutBuffer()
        episode = self._start_episode
        total_steps = self._start_step
        if cfg.n_steps and not cfg.n_episodes:
            cfg.n_episodes = np.iinfo(np.int32).max

        state = self.train_env.reset()
        done = False
        episode_steps = 0
        while episode < cfg.n_episodes:
            logits, value = self.agent._infer(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            next_state, reward, done, _ = self.train_env.step(action.item())

            buffer.add(state, action.item(), log_prob, reward, done, value.item())

            state = next_state
            total_steps += 1
            episode_steps += 1

            if len(buffer) >= cfg.batch_size:
                with torch.no_grad():
                    last_value = 0.0 if done else self.agent._infer(state)[1].item()
                self._update(buffer, last_value)
                buffer.clear()

            if (
                cfg.checkpoint_every_steps
                and total_steps % cfg.checkpoint_every_steps == 0
            ):
                with torch.no_grad():
                    last_value = 0.0 if done else self.agent._infer(state)[1].item()
                if len(buffer) > 0:
                    self._update(buffer, last_value)
                    buffer.clear()
                self._save_checkpoint(total_steps, episode)

            if done or (cfg.n_steps and episode_steps >= cfg.n_steps):
                with torch.no_grad():
                    last_value = 0.0 if done else self.agent._infer(state)[1].item()
                if len(buffer) > 0:
                    self._update(buffer, last_value)
                    buffer.clear()
                self._validate()
                state = self.train_env.reset()
                done = False
                episode += 1
                episode_steps = 0

        self._save_checkpoint(total_steps, episode)

    def _validate(self):
        state = self.valid_env.reset(begin_from_start=True)
        done = False
        total_reward = 0.0
        while not done:
            action = self.agent.get_action(state)
            state, reward, done, _ = self.valid_env.step(action)
            total_reward += reward
        print(f"Validation reward: {total_reward:.4f}")
