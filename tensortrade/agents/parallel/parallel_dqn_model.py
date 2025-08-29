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
# ---------------------------------------------------------------------------
# PyTorch port of the original TensorFlow Parallel DQN model.
# ---------------------------------------------------------------------------

from deprecated import deprecated
import random
import copy
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ParallelDQNModel"]


# ────────────────────────────────────────────────────────────────────────────────
# Network definition
# ────────────────────────────────────────────────────────────────────────────────


class PolicyNet(nn.Module):
    """
    Simple 1‑D convolutional network that ends in a probability distribution
    over actions (via softmax). Mirrors the layer layout of the original TF
    version but adapts the tensor layout to PyTorch (B, C, L).
    """

    def __init__(self, input_shape: tuple, n_actions: int):
        """
        Args:
            input_shape: observation shape as (L, C) or (C, L).
            n_actions:   size of the discrete action space.
        """
        super().__init__()
        # Ensure (C, L) for Conv1d
        if len(input_shape) != 2:
            raise ValueError("Expected 1‑D observation space, e.g. (L, C).")

        channels_first = (input_shape[0] <= 10)  # heuristic
        C, L = (input_shape[1], input_shape[0]) if channels_first else input_shape[::-1]

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C, out_channels=64, kernel_size=6, padding="same"),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(self._get_flat_dim(C, L), n_actions),
            nn.Sigmoid(),  # matches TF layout (Dense + sigmoid + Dense + softmax)
            nn.Linear(n_actions, n_actions),
        )

    def _get_flat_dim(self, C, L):
        # feed a dummy tensor to compute flatten size
        with torch.no_grad():
            x = torch.zeros(1, C, L)
            x = self.net[:-3](x)  # forward up to flatten
            return x.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped (B, C, L). Convert first if needed.
        Returns:
            Action probabilities (softmax output).
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


# ────────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ────────────────────────────────────────────────────────────────────────────────


@deprecated(
    version="1.0.4",
    reason="Builtin agents are being deprecated in favor of external implementations (ie: Ray)",
)
class ParallelDQNModel:
    """
    Lightweight wrapper that keeps a policy network and its target clone in
    sync. Does not itself implement any learning logic.
    """

    def __init__(
        self,
        create_env: Callable[[], "TradingEnvironment"],
        policy_network: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        # Create a temporary env only to read the spaces; never used for trading.
        temp_env = create_env()
        self.n_actions = temp_env.action_space.n
        self.observation_shape = temp_env.observation_space.shape

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Networks
        self.policy_network = (policy_network or self._build_policy_network()).to(self.device)
        self.target_network = copy.deepcopy(self.policy_network).to(self.device)
        self.target_network.eval()  # target network is frozen

    # ---------------------------------------------------------------------
    # Network constructors / I/O
    # ---------------------------------------------------------------------

    def _build_policy_network(self) -> nn.Module:
        """
        Create the default convolutional policy network.
        """
        return PolicyNet(self.observation_shape, self.n_actions)

    # ---------------------------------------------------------------------
    # Serialization helpers
    # ---------------------------------------------------------------------

    def restore(self, path: str, **kwargs):
        """
        Load weights from a `.pth` checkpoint and clone them into target net.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(state_dict)
        self.target_network = copy.deepcopy(self.policy_network).to(self.device)
        self.target_network.eval()

    def save(self, path: str, **kwargs):
        agent_id: str = kwargs.get("agent_id", "No_ID")
        episode: int = kwargs.get("episode")

        if episode is not None:
            filename = f"policy_network__{agent_id}__{str(episode).zfill(3)}.pth"
        else:
            filename = f"policy_network__{agent_id}.pth"

        torch.save(self.policy_network.state_dict(), f"{path}{filename}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """
        ε‑greedy action selection.
        """
        threshold: float = kwargs.get("threshold", 0.0)
        if random.random() < threshold:
            return np.random.randint(self.n_actions)

        # Prepare tensor (1, C, L)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 2:  # (L, C) -> (C, L)
            state_t = state_t.permute(1, 0)
        state_t = state_t.unsqueeze(0)

        probs = self.policy_network(state_t)  # (1, n_actions)
        return probs.argmax(dim=-1).item()

    # ---------------------------------------------------------------------
    # Synchronisation helpers
    # ---------------------------------------------------------------------

    def update_networks(self, model: "ParallelDQNModel"):
        """
        Hard copy weights from another model (typically the learner thread).
        """
        self.policy_network.load_state_dict(model.policy_network.state_dict())
        self.target_network.load_state_dict(model.target_network.state_dict())

    def update_target_network(self):
        """
        Hard update of target <- online.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())
