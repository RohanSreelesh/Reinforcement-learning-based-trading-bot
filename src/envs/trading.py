from enum import Enum
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd

class Actions (Enum):
    Sell = -1
    Hold = 0
    Buy = 1

class TradingEnv (gym.Env):
    # Properties (public)
    render_mode: str
    prices: float
    data_frames: pd.DataFrame
    window_size: int
    shape: tuple[int, int]

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box

    metadata = {
        "render_modes": ["human"],
        "render_fps": 4
    }

    # Properties (private)
    _start_tick: int
    _end_tick: int
    _current_tick: int = None
    _last_trade_tick: int = None
    _total_reward: float
    _total_profit: float

    # Constructor 
    def __init__ (self, data_frames: pd.DataFrame, window_size: int, render_mode: str = None):
        self.render_mode = render_mode

        self.data_frames = data_frames
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(low=-1e10, high=1e10, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._total_reward = None
        self._total_profit = None

    # Business logics
    def _process_data (self):
        raise NotImplementedError

    def _update_history (self, info):
        raise NotImplementedError

    def _calculate_reward (self, action):
        raise NotImplementedError

    def _update_profit (self, action):
        raise NotImplementedError

    def max_possible_profit (self): 
        raise NotImplementedError

    # Lifecycle
    def reset (self, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.action_space.seed(
            int((self.np_random.uniform(0, seed if seed is not None else 1)))
        )

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0
        self._total_profit = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step (self, action):
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        self._last_trade_tick = self._current_tick

        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info
    
    def close (self):
        plt.close()

    def _get_observation (self):
        raise NotImplementedError
    
    def _get_info (self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
        )

    # Render
    def render (self, mode='human'):
        raise NotImplementedError
    
    def _render_frame (self):
        self.render()
