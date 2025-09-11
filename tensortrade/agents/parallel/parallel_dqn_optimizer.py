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
# PyTorch port of the original TensorFlow-based ParallelDQNOptimizer.
# ---------------------------------------------------------------------------

from deprecated import deprecated
from multiprocessing import Process, Queue
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tensortrade.agents import ReplayMemory, DQNTransition


@deprecated(
    version="1.0.4",
    reason="Builtin agents are being deprecated in favor of external implementations (ie: Ray)",
)
class ParallelDQNOptimizer(Process):
    """
    Optimizer process that samples experiences pushed by multiple environment
    workers, performs gradient descent on the shared policy network, and sends
    the updated model back to the workers via a multiprocessing queue.
    """

    def __init__(
        self,
        model: "ParallelDQNModel",
        n_envs: int,
        memory_queue: Queue,
        model_update_queue: Queue,
        done_queue: Queue,
        discount_factor: float = 0.9999,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        memory_capacity: int = 10_000,
        max_grad_norm: Optional[float] = 1.0,
    ):
        super().__init__()
        self.model = model
        self.n_envs = n_envs
        self.memory_queue = memory_queue
        self.model_update_queue = model_update_queue
        self.done_queue = done_queue

        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.max_grad_norm = max_grad_norm

        # Convenience
        self.device = self.model.device

    # ------------------------------------------------------------------ #
    # Process main loop
    # ------------------------------------------------------------------ #

    def run(self):
        memory = ReplayMemory(self.memory_capacity, transition_type=DQNTransition)

        # Optimizer (Torch NAdam available since v1.8)
        optimizer = optim.NAdam(
            self.model.policy_network.parameters(), lr=self.learning_rate
        )

        while self.done_queue.qsize() < self.n_envs:
            # Drain memory queue
            while self.memory_queue.qsize() > 0:
                sample = self.memory_queue.get()
                # `sample` is expected to be an iterable matching DQNTransition.
                memory.push(*sample)

            if len(memory) < self.batch_size:
                continue

            # Sample a minibatch
            transitions = memory.sample(self.batch_size)
            batch = DQNTransition(*zip(*transitions))

            # ------------------------------------------------------------------
            # Prepare tensors
            # ------------------------------------------------------------------
            state_batch = torch.as_tensor(
                np.stack(batch.state), dtype=torch.float32, device=self.device
            )
            if state_batch.ndim == 3:  # (B, L, C) -> (B, C, L)
                state_batch = state_batch.permute(0, 2, 1)

            next_state_batch = torch.as_tensor(
                np.stack(batch.next_state), dtype=torch.float32, device=self.device
            )
            if next_state_batch.ndim == 3:
                next_state_batch = next_state_batch.permute(0, 2, 1)

            action_batch = torch.as_tensor(
                batch.action, dtype=torch.int64, device=self.device
            )
            reward_batch = torch.as_tensor(
                batch.reward, dtype=torch.float32, device=self.device
            )
            done_batch = torch.as_tensor(batch.done, dtype=torch.bool, device=self.device)

            # ------------------------------------------------------------------
            # Compute Q(s, a) and expected Q targets
            # ------------------------------------------------------------------
            # Current Q values for the actions taken
            q_pred = self.model.policy_network(state_batch)
            state_action_values = q_pred.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = self.model.target_network(next_state_batch)
                next_state_values, _ = torch.max(q_next, dim=1)
                next_state_values = torch.where(
                    done_batch, torch.zeros_like(next_state_values), next_state_values
                )
                expected_state_action_values = reward_batch + (
                    self.discount_factor * next_state_values
                )

            # ------------------------------------------------------------------
            # Loss, back‑prop, optimizer step
            # ------------------------------------------------------------------
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.policy_network.parameters(), self.max_grad_norm
                )
            optimizer.step()

            # Send the updated model back to env workers
            self.model_update_queue.put(self.model)
