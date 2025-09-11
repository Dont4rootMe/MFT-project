
# Copyright 2020 The TensorTrade Authors.
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

from __future__ import annotations

import os
import random
from copy import deepcopy
from collections import namedtuple
from datetime import datetime
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deprecated import deprecated

from tensortrade.agents import Agent, ReplayMemory


DQNTransition = namedtuple("DQNTransition", ["state", "action", "reward", "next_state", "done"])


# ---- Small utility modules -------------------------------------------------
class CausalConv1d(nn.Module):
    """Causal 1D convolution via left padding.

    Equivalent to Keras Conv1D(padding="causal").
    Pads only on the left by (kernel_size-1)*dilation so that output[t]
    never depends on input at time > t.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


# ---- Policy network (ported from the original TF/Keras architecture) ------
class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape: Iterable[int], n_actions: int, dropout: float = 0.9) -> None:
        super().__init__()
        # Keras used channels-last (T, F). We'll operate in channels-first (B, F, T).
        self.n_actions = n_actions
        feat_dim = observation_shape[-1] if len(observation_shape) > 1 else 1

        # First block: three parallel causal convs on the same padded input
        self.pad_1 = nn.ConstantPad1d((1, 0), 0.0)  # mimics ZeroPadding1D(padding=stride) with stride==1
        self.conv1 = CausalConv1d(feat_dim, 16, kernel_size=4, stride=2, bias=True)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = CausalConv1d(feat_dim, 32, kernel_size=4, stride=2, bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = CausalConv1d(feat_dim, 64, kernel_size=4, stride=2, bias=True)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(p=dropout)

        # Second block: again three parallel convs on previous concat
        in_ch_2 = 16 + 32 + 64
        self.conv4 = CausalConv1d(in_ch_2, 16, kernel_size=4, stride=2, bias=True)
        self.bn4 = nn.BatchNorm1d(16)
        self.conv5 = CausalConv1d(in_ch_2, 32, kernel_size=4, stride=2, bias=True)
        self.bn5 = nn.BatchNorm1d(32)
        self.conv6 = CausalConv1d(in_ch_2, 64, kernel_size=4, stride=2, bias=True)
        self.bn6 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(p=dropout)

        # Third block: three parallel convs on previous concat
        in_ch_3 = 16 + 32 + 64
        self.conv7 = CausalConv1d(in_ch_3, 16, kernel_size=4, stride=2, bias=True)
        self.bn7 = nn.BatchNorm1d(16)
        self.conv8 = CausalConv1d(in_ch_3, 32, kernel_size=4, stride=2, bias=True)
        self.bn8 = nn.BatchNorm1d(32)
        self.conv9 = CausalConv1d(in_ch_3, 64, kernel_size=4, stride=2, bias=True)
        self.bn9 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(p=dropout)

        # Pool + stacked GRUs
        self.pool = nn.AvgPool1d(kernel_size=3, stride=2)
        # We'll use batch_first=True so tensors are [B, T, C].
        self.gru1 = nn.GRU(input_size=in_ch_3, hidden_size=64, num_layers=1, batch_first=True)
        self.drop4 = nn.Dropout(p=dropout)
        self.gru2 = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.drop5 = nn.Dropout(p=dropout)
        # concat_rnn_1 = [drop4, drop5] => size 128
        self.gru3 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.drop6 = nn.Dropout(p=dropout)
        # concat_rnn_2 = [drop4, drop5, drop6] => size 192
        self.gru4 = nn.GRU(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        # Last GRU in Keras returned final state only; we'll take the last output step.

        # Dense head
        self.drop7 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(64, 32)
        self.sm1 = nn.Softmax(dim=-1)
        self.fc2 = nn.Linear(32, 16)
        self.sm2 = nn.Softmax(dim=-1)
        self.fc3 = nn.Linear(16, 16)
        self.prelu = nn.PReLU()
        self.drop8 = nn.Dropout(p=dropout)
        self.fc_pre_out = nn.Linear(16, n_actions)
        self.sigmoid = nn.Sigmoid()
        self.fc_out = nn.Linear(n_actions, n_actions)
        self.softmax_out = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x as [B, F, T]. If given [B, T, F], transpose.
        if x.dim() == 3 and x.size(1) < x.size(2):
            # Heuristic: if channels < time, assume shape is [B, T, F]
            x = x.transpose(1, 2)

        # First branch group (each branch sees the same padded input)
        x_pad = self.pad_1(x)
        b1 = self.bn1(F.prelu(self.conv1(x_pad), torch.tensor(0.25, device=x.device)))
        b2 = self.bn2(F.prelu(self.conv2(x_pad), torch.tensor(0.25, device=x.device)))
        b3 = self.bn3(F.prelu(self.conv3(x_pad), torch.tensor(0.25, device=x.device)))
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.drop1(x)

        # Second branch group
        b4 = self.bn4(F.prelu(self.conv4(x), torch.tensor(0.25, device=x.device)))
        b5 = self.bn5(F.prelu(self.conv5(x), torch.tensor(0.25, device=x.device)))
        b6 = self.bn6(F.prelu(self.conv6(x), torch.tensor(0.25, device=x.device)))
        x = torch.cat([b4, b5, b6], dim=1)
        x = self.drop2(x)

        # Third branch group
        b7 = self.bn7(F.prelu(self.conv7(x), torch.tensor(0.25, device=x.device)))
        b8 = self.bn8(F.prelu(self.conv8(x), torch.tensor(0.25, device=x.device)))
        b9 = self.bn9(F.prelu(self.conv9(x), torch.tensor(0.25, device=x.device)))
        x = torch.cat([b7, b8, b9], dim=1)  # [B, in_ch_3, L]
        x = self.drop3(x)

        # Pool then GRUs (switch to [B, T, C])
        x = self.pool(x)
        x = x.transpose(1, 2)

        x1, _ = self.gru1(x)
        x1 = self.drop4(x1)
        x2, _ = self.gru2(x1)
        x2 = self.drop5(x2)
        cat1 = torch.cat([x1, x2], dim=-1)
        x3, _ = self.gru3(cat1)
        x3 = self.drop6(x3)
        cat2 = torch.cat([x1, x2, x3], dim=-1)
        x4, _ = self.gru4(cat2)
        # Take last time step
        x4 = x4[:, -1, :]

        # Dense head
        x = self.drop7(x4)
        x = self.sm1(self.fc1(x))
        x = self.sm2(self.fc2(x))
        x = self.prelu(self.fc3(x))
        x = self.drop8(x)
        x = self.sigmoid(self.fc_pre_out(x))
        x = self.softmax_out(self.fc_out(x))  # shape [B, n_actions]
        return x


@deprecated(
    version="1.0.4",
    reason="Builtin agents are being deprecated in favor of external implementations (e.g., Ray). Ported to PyTorch.",
)
class DQNAgent(Agent):
    """
    A PyTorch reimplementation of the original TensorFlow-based DQNAgent.

    Notes
    -----
    * Uses Smooth L1 (Huber) loss for temporal-difference errors.
    * Maintains a target network updated at a fixed step interval.
    * Retains the original model topology and activations (softmax head) to
      mirror behavior, even though classic DQN uses linear Q-value outputs.
    """

    def __init__(self, env: "TradingEnv", policy_network: nn.Module | None = None):
        self.env = env
        self.n_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network: nn.Module = policy_network or PolicyNetwork(
            observation_shape=self.observation_shape, n_actions=self.n_actions
        )
        self.policy_network.to(self.device)

        self.target_network: nn.Module = deepcopy(self.policy_network)
        for p in self.target_network.parameters():
            p.requires_grad = False
        self.target_network.to(self.device)

        self.env.agent_id = self.id

    # -------------------------- I/O helpers ---------------------------------
    def _state_to_tensor(self, s: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert a single state to tensor [1, F, T] on the correct device."""
        if isinstance(s, np.ndarray):
            x = torch.from_numpy(s).float()
        else:
            x = s.float()
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif x.dim() == 2:
            # Original Keras expected (T, F). Switch to (F, T) and add batch
            x = x.transpose(0, 1).unsqueeze(0)
        elif x.dim() == 3:
            # assume already batched; reorder to [B, F, T] if [B, T, F]
            if x.size(1) > x.size(2):
                pass  # [B, F, T]
            else:
                x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unsupported state shape: {tuple(x.shape)}")
        return x.to(self.device)

    def _batch_to_tensors(self, batch: DQNTransition) -> tuple[torch.Tensor, ...]:
        state_batch = torch.stack([self._state_to_tensor(s).squeeze(0) for s in batch.state], dim=0)
        next_state_batch = torch.stack([self._state_to_tensor(s).squeeze(0) for s in batch.next_state], dim=0)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    # ----------------------------- API --------------------------------------
    def restore(self, path: str, **kwargs) -> None:
        """Load model weights from a .pt/.pth file.

        Expects a file produced by :meth:`save` (state_dict).
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_network.load_state_dict(state_dict)
        self.target_network = deepcopy(self.policy_network)
        for p in self.target_network.parameters():
            p.requires_grad = False
        self.target_network.to(self.device)

    def save(self, path: str, **kwargs) -> None:
        episode: int | None = kwargs.get("episode", None)
        os.makedirs(path, exist_ok=True)
        filename = (
            f"policy_network__{self.id[:7]}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        torch.save(self.policy_network.state_dict(), os.path.join(path, filename))

    @torch.inference_mode()
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        threshold: float = kwargs.get("threshold", 0.0)
        if random.random() < threshold:
            return np.random.choice(self.n_actions)
        x = self._state_to_tensor(state)  # [1, F, T]
        self.policy_network.eval()
        q = self.policy_network(x)  # [1, A]
        return int(q.argmax(dim=1).item())

    def _apply_gradient_descent(
        self, memory: ReplayMemory, batch_size: int, learning_rate: float, discount_factor: float
    ) -> None:
        # Optimizer & loss
        if not hasattr(self, "_optimizer"):
            # NAdam is available in torch.optim (PyTorch 2.x)
            self._optimizer = torch.optim.NAdam(self.policy_network.parameters(), lr=learning_rate)
        loss_fn = F.smooth_l1_loss  # Huber-like loss

        transitions = memory.sample(batch_size)
        batch = DQNTransition(*zip(*transitions))
        state_b, action_b, reward_b, next_state_b, done_b = self._batch_to_tensors(batch)

        self.policy_network.train()
        self._optimizer.zero_grad(set_to_none=True)

        # Current Q(s,a)
        q_values = self.policy_network(state_b)  # [B, A]
        state_action_values = q_values.gather(1, action_b.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            # Target: r + gamma * max_a' Q_target(s', a') for non-terminal s'
            next_q_values = self.target_network(next_state_b)  # [B, A]
            next_state_values = next_q_values.max(dim=1).values
            next_state_values = torch.where(done_b, torch.zeros_like(next_state_values), next_state_values)
            expected_state_action_values = reward_b + discount_factor * next_state_values

        loss = loss_fn(state_action_values, expected_state_action_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        self._optimizer.step()

    def train(
        self,
        n_steps: int = 1000,
        n_episodes: int = 10,
        save_every: int | None = None,
        save_path: str = "agent/",
        callback: callable | None = None,
        **kwargs,
    ) -> float:
        batch_size: int = kwargs.get("batch_size", 256)
        memory_capacity: int = kwargs.get("memory_capacity", n_steps * 10)
        discount_factor: float = kwargs.get("discount_factor", 0.95)
        learning_rate: float = kwargs.get("learning_rate", 0.01)
        eps_start: float = kwargs.get("eps_start", 0.9)
        eps_end: float = kwargs.get("eps_end", 0.05)
        eps_decay_steps: int = kwargs.get("eps_decay_steps", n_steps)
        update_target_every: int = kwargs.get("update_target_every", 1000)
        render_interval: int | None = kwargs.get("render_interval", n_steps // 10)

        memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        episode = 0
        total_steps_done = 0
        total_reward = 0.0

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print(f"====      AGENT ID: {self.id}      ====")

        while episode < n_episodes:
            state = self.env.reset()
            done = False
            steps_done = 0

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(-total_steps_done / max(1, eps_decay_steps))
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                memory.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps_done += 1
                total_steps_done += 1

                if len(memory) >= batch_size:
                    self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor)

                if n_steps and steps_done >= n_steps:
                    done = True

                if render_interval is not None and steps_done % render_interval == 0:
                    self.env.render(episode=episode, max_episodes=n_episodes, max_steps=n_steps)

                if steps_done % update_target_every == 0:
                    # Hard update
                    self.target_network.load_state_dict(self.policy_network.state_dict())

            is_checkpoint = save_every and (episode % save_every == 0)
            if save_path and (is_checkpoint or episode == n_episodes - 1):
                self.save(save_path, episode=episode)

            if not render_interval or steps_done < n_steps:
                self.env.render(episode=episode, max_episodes=n_episodes, max_steps=n_steps)

            self.env.save()
            episode += 1

        mean_reward = total_reward / max(1, steps_done)
        return float(mean_reward)