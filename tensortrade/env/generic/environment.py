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
# limitations under the License

import uuid
import logging

from typing import Dict, Any, Tuple, Optional
from random import randint
import random

import gymnasium
import numpy as np

from tensortrade.core import TimeIndexed, Clock, Component
from tensortrade.env.generic import (
    ActionScheme,
    RewardScheme,
    Observer,
    Stopper,
    Informer,
    Renderer
)


class TradingEnv(gymnasium.Env, TimeIndexed):
    """A trading environment made for use with Gym-compatible reinforcement
    learning algorithms.

    Parameters
    ----------
    action_scheme : `ActionScheme`
        A component for generating an action to perform at each step of the
        environment.
    reward_scheme : `RewardScheme`
        A component for computing reward after each step of the environment.
    observer : `Observer`
        A component for generating observations after each step of the
        environment.
    informer : `Informer`
        A component for providing information after each step of the
        environment.
    renderer : `Renderer`
        A component for rendering the environment.
    kwargs : keyword arguments
        Additional keyword arguments needed to create the environment.
    """

    agent_id: str = None
    episode_id: str = None

    def __init__(self,
                 action_scheme: ActionScheme,
                 reward_scheme: RewardScheme,
                 observer: Observer,
                 stopper: Stopper,
                 informer: Informer,
                 renderer: Renderer,
                 min_periods: int = None,
                 max_episode_steps: int = None,
                 max_episode_length: Optional[int] = None,
                 rng: Optional[random.Random] = None,
                 **kwargs) -> None:
        super().__init__()
        self.clock = Clock()

        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.observer = observer
        self.stopper = stopper
        self.informer = informer
        self.renderer = renderer
        self.min_periods = min_periods
        self.max_episode_length = max_episode_length
        
        # Initialize random number generator
        self.rng = rng if rng is not None else random.Random()

        for c in self.components.values():
            c.clock = self.clock

        self.action_space = action_scheme.action_space
        self.observation_space = observer.observation_space

        self._enable_logger = kwargs.get('enable_logger', False)
        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

    @property
    def components(self) -> 'Dict[str, Component]':
        """The components of the environment. (`Dict[str,Component]`, read-only)"""
        return {
            "action_scheme": self.action_scheme,
            "reward_scheme": self.reward_scheme,
            "observer": self.observer,
            "stopper": self.stopper,
            "informer": self.informer,
            "renderer": self.renderer
        }

    def step(self, action: Any, skip_decision: bool = False) -> 'Tuple[np.array, float, bool, dict]':
        """Makes one step through the environment.

        Parameters
        ----------
        action : Any
            An action to perform on the environment.
            
        skip_decision : bool
            Whether to skip decision making and just observe the environment.

        Returns
        -------
        `np.array`
            The observation of the environment after the action being
            performed.
        float
            The computed reward for performing the action.
        bool
            Whether or not the episode is complete.
        dict
            The information gathered after completing the step.
        """
        # if skip_decision is True, we skip the entering of trading segment
        if not skip_decision:
            self.action_scheme.perform(self, action)
        
        # morning –> night
        obs = self.observer.observe(self)
        reward = self.reward_scheme.reward(self)
        terminated = self.stopper.stop(self)
        truncated = False
        info = self.informer.info(self)

        self.clock.increment()

        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, begin_from_start = False) -> tuple["np.array", dict[str, Any]]:
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If provided, it will seed the environment's RNG.
        begin_from_start : bool, optional
            Whether to start from the beginning of the time series (default: False).

        Returns
        -------
        obs : `np.array`
            The first observation of the environment.
        info : dict
            Additional information from the environment.
        """
        # Set the seed if provided
        if seed is not None:
            self.rng.seed(seed)
        
        if begin_from_start or self.max_episode_length is None:
            random_start = 0
        else:
            size = len(self.observer.feed.process[-1].inputs[0].iterable)
            
            # Calculate the valid range for episode start considering max_episode_length
            # Ensure episode can fit within data: start from [0, size - max_episode_length - 1]
            max_start_idx = max(0, size - self.max_episode_length - 1)
            
            # Use the environment's RNG instead of global randint
            random_start = self.rng.randint(0, max_start_idx) if max_start_idx > 0 else 0

        self.episode_id = str(uuid.uuid4())
        self.clock.reset()

        for c in self.components.values():
            if hasattr(c, "reset"):
                if isinstance(c, Observer):
                    c.reset(random_start=random_start)
                else:
                    c.reset()

        obs = self.observer.observe(self)
        info = self.informer.info(self)

        self.clock.increment()

        return obs, info

    def seed(self, seed: Optional[int] = None) -> list:
        """Seed the environment's random number generator.
        
        Parameters
        ----------
        seed : int, optional
            The seed to use for the random number generator.
            
        Returns
        -------
        list
            List containing the seed used.
        """
        if seed is not None:
            self.rng.seed(seed)
        return [seed]
    
    def get_price_history(self) -> np.array:
        return self.observer.renderer_history

    def get_order_history(self) -> list[dict]:
        """Gets the order history."""
        return self.action_scheme.broker.trades
    
    def get_pnl_history(self) -> np.array:
        return self.action_scheme.portfolio.performance['net_worth']

    def render(self, **kwargs) -> None:
        """Renders the environment."""
        self.renderer.render(self, **kwargs)

    def save(self, root_path: str | None = None) -> None:
        """Saves the rendered view of the environment."""
        self.renderer.save(root_path)

    def close(self) -> None:
        """Closes the environment."""
        self.renderer.close()
