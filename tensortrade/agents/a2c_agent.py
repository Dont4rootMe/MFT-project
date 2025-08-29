# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
# PyTorch re‑implementation of the original TensorFlow 2 A2C agent.
# ------------------------------------------------------------------------
# References:
#   - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/#agent-interface
# ------------------------------------------------------------------------

from deprecated import deprecated
import random
from datetime import datetime
from collections import namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensortrade.agents import Agent, ReplayMemory

A2CTransition = namedtuple(
    "A2CTransition", ["state", "action", "reward", "done", "value"]
)


# ────────────────────────────────────────────────────────────────────────────────
# Helper networks
# ────────────────────────────────────────────────────────────────────────────────


class SharedNet(nn.Module):
    """
    Convolutional feature extractor shared by both actor and critic.
    """

    def __init__(self, input_shape: tuple):
        """
        Args:
            input_shape: Shape of observation without batch dimension
                         (C, L) for Conv1D.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[0],
            out_channels=64,
            kernel_size=6,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding="same",
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Determine the size of the flattened feature vector dynamically.
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self._forward_conv(dummy)
            self.output_dim = x.shape[1]

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """
        Forward pass through shared conv layers.

        Args:
            x: Tensor with shape (B, C, L)
        """
        return self._forward_conv(x)


class ActorHead(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits


class CriticHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # state‑value estimate


# ────────────────────────────────────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────────────────────────────────────


@deprecated(
    version="1.0.4",
    reason="Builtin agents are being deprecated in favor of external implementations (ie: Ray)",
)
class A2CAgent(Agent):
    """
    Advantage Actor‑Critic agent implemented in PyTorch.
    """

    def __init__(
        self,
        env: "TradingEnvironment",
        shared_network: Optional[nn.Module] = None,
        actor_network: Optional[nn.Module] = None,
        critic_network: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.env = env
        self.n_actions = env.action_space.n
        # Convert TF style (L, C) to PyTorch (C, L) for Conv1d.
        self.observation_shape = (env.observation_space.shape[1], env.observation_space.shape[0])  # (C, L)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Networks
        self.shared_network = (shared_network or SharedNet(self.observation_shape)).to(
            self.device
        )
        if actor_network and critic_network:
            self.actor_head = actor_network.to(self.device)
            self.critic_head = critic_network.to(self.device)
        else:
            self.actor_head = ActorHead(self.shared_network.output_dim, self.n_actions).to(
                self.device
            )
            self.critic_head = CriticHead(self.shared_network.output_dim).to(self.device)

        # Optimizer (shared parameters get one optimizer to ensure synchronous update)
        self.optimizer = optim.Adam(
            list(self.shared_network.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters())
        )

        self.env.agent_id = self.id

    # ────────────────────────────────────────────────────────────────────────
    # Model I/O
    # ────────────────────────────────────────────────────────────────────────

    def save(self, path: str, **kwargs):
        """
        Save state‑dicts of actor and critic heads plus shared conv layers.
        """
        episode = kwargs.get("episode", None)
        suffix = (
            f"{self.id[:7]}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            if episode
            else f"{self.id[:7]}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        )
        state = {
            "shared": self.shared_network.state_dict(),
            "actor": self.actor_head.state_dict(),
            "critic": self.critic_head.state_dict(),
            "episode": episode,
        }
        torch.save(state, f"{path}a2c_agent__{suffix}")

    def restore(self, path: str):
        """
        Restore agent from a checkpoint produced by ``save``.
        """
        state = torch.load(path, map_location=self.device)
        self.shared_network.load_state_dict(state["shared"])
        self.actor_head.load_state_dict(state["actor"])
        self.critic_head.load_state_dict(state["critic"])

    # ────────────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer(self, state: np.ndarray):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Convert to (B, C, L)
        if state_t.ndim == 3:  # original shape (B, L, C)
            state_t = state_t.permute(0, 2, 1)
        features = self.shared_network(state_t)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)  # (B,)
        return logits, value

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """
        ε‑greedy sampling from policy.
        """
        threshold: float = kwargs.get("threshold", 0.0)
        if random.random() < threshold:
            return np.random.choice(self.n_actions)

        logits, _ = self._infer(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    # ────────────────────────────────────────────────────────────────────────
    # Training helpers
    # ────────────────────────────────────────────────────────────────────────

    def _compute_returns(self, rewards, dones, discount_factor):
        """
        Compute discounted returns in reverse (vectorized).
        """
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + discount_factor * G * (1 - int(d))
            returns.append(G)
        return torch.tensor(list(reversed(returns)), dtype=torch.float32, device=self.device)

    def _apply_gradient_descent(
        self,
        memory: ReplayMemory,
        batch_size: int,
        learning_rate: float,
        discount_factor: float,
        entropy_c: float,
    ):
        transitions = memory.tail(batch_size)
        batch = A2CTransition(*zip(*transitions))

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        if states.ndim == 3:  # (B, L, C) -> (B, C, L)
            states = states.permute(0, 2, 1)

        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        values = torch.stack(batch.value).to(self.device)

        returns = self._compute_returns(rewards, dones, discount_factor)
        advantages = returns - values.detach()

        # Forward pass
        features = self.shared_network(states)
        logits = self.actor_head(features)
        new_values = self.critic_head(features).squeeze(-1)

        # Critic loss (Huber)
        critic_loss = F.huber_loss(new_values, returns)

        # Actor loss (policy gradient with advantage weighting)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -(selected_log_probs * advantages).mean()

        # Entropy regularization
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        loss = actor_loss + 0.5 * critic_loss - entropy_c * entropy

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.shared_network.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters()),
            max_norm=0.5,
        )
        self.optimizer.step()

    # ────────────────────────────────────────────────────────────────────────
    # Training loop
    # ────────────────────────────────────────────────────────────────────────

    def train(
        self,
        n_steps: int = None,
        n_episodes: int = None,
        save_every: int = None,
        save_path: str = None,
        callback: callable = None,
        **kwargs,
    ) -> float:
        batch_size: int = kwargs.get("batch_size", 128)
        discount_factor: float = kwargs.get("discount_factor", 0.9999)
        learning_rate: float = kwargs.get("learning_rate", 0.0001)
        eps_start: float = kwargs.get("eps_start", 0.9)
        eps_end: float = kwargs.get("eps_end", 0.05)
        eps_decay_steps: int = kwargs.get("eps_decay_steps", 200)
        entropy_c: float = kwargs.get("entropy_c", 0.0001)
        memory_capacity: int = kwargs.get("memory_capacity", 1000)

        # update learning rate if provided (allows runtime adjustment)
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate

        memory = ReplayMemory(memory_capacity, transition_type=A2CTransition)
        episode = 0
        steps_done = 0
        total_reward = 0.0
        stop_training = False

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print(f"====      AGENT ID: {self.id}      ====")

        while episode < n_episodes and not stop_training:
            state = self.env.reset()
            done = False

            print(
                f"====      EPISODE ID ({episode + 1}/{n_episodes}): {self.env.episode_id}      ===="
            )

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(
                    -steps_done / eps_decay_steps
                )
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                # Compute value of current state for advantage
                with torch.no_grad():
                    _, value = self._infer(state)

                memory.push(state, action, reward, done, value)

                state = next_state
                total_reward += reward
                steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(
                    memory,
                    batch_size,
                    learning_rate,
                    discount_factor,
                    entropy_c,
                )

                if n_steps and steps_done >= n_steps:
                    done = True
                    stop_training = True

            is_checkpoint = save_every and episode % save_every == 0
            if save_path and (is_checkpoint or episode + 1 == n_episodes):
                self.save(save_path, episode=episode)

            episode += 1

        mean_reward = total_reward / max(1, steps_done)
        return mean_reward