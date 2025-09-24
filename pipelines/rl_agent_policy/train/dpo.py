"""Skeleton of a Direct Preference Optimisation trainer.

The implementation here is intentionally lightweight and serves as a
placeholder for more sophisticated preference‑based learning methods.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    n_episodes: int = 1


class DPOTrainer:
    """Minimal trainer demonstrating the required interface."""

    def __init__(self, agent, train_env, valid_env, config: Optional[DPOConfig] = None):
        self.agent = agent
        self.train_env = train_env
        self.valid_env = valid_env
        self.cfg = config or DPOConfig()

    def train(self):
        """Run preference‑based optimisation.

        The full algorithm is outside the scope of this repository; the method
        simply iterates over the validation environment to showcase the
        interface.
        """
        for _ in range(self.cfg.n_episodes):
            state = self.valid_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.agent.get_action(state)
                state, reward, done, _ = self.valid_env.step(action)
                total_reward += reward
            print(f"DPO validation reward: {total_reward:.4f}")
