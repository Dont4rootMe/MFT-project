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

from tensortrade.agents import Agent
from pipelines.rl_agent_policy.models import CNNBackbone, ActorHead, CriticHead

A2CTransition = namedtuple(
    "A2CTransition", ["state", "action", "reward", "done", "value"]
)


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
        action_space = env.action_space
        if hasattr(action_space, "n"):
            self.n_actions = action_space.n
            self._action_nvec = None
        elif hasattr(action_space, "nvec"):
            self.n_actions = int(np.prod(action_space.nvec))
            self._action_nvec = action_space.nvec
        else:
            raise AttributeError("Unsupported action space type")
        # Convert TF style (L, C) to PyTorch (C, L) for Conv1d.
        self.observation_shape = (
            env.observation_space.shape[1],
            env.observation_space.shape[0],
        )  # (C, L)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Networks
        backbone = shared_network or CNNBackbone(self.observation_shape)
        self.shared_network = backbone.to(self.device)
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

    # Training utilities removed. Training is now handled by dedicated
    # trainer classes located under ``pipelines.rl_agent_policy.train``.

    def train(self, *args, **kwargs):
        """Stub to satisfy abstract base class requirements.

        Training is managed externally via the pipeline trainers under
        ``pipelines.rl_agent_policy.train``. Calling this method directly will
        raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            "Use A2CTrainer or other trainer classes for optimization."
        )
