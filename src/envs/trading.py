import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
from typing import List, Dict
from models import Account
from enums import Action as ActionType
from models import Action
import time


class TradingEnv(gym.Env):
    # Properties (public)
    render_mode: str
    prices: float
    data_frames: pd.DataFrame
    window_size: int
    shape: tuple[int, int]
    max_shares_per_trade: int

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box

    account: Account
    history: Dict[str, List[float] | Dict]

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Properties (private)
    _start_tick: int
    _end_tick: int
    _current_tick: int = None
    _last_trade_tick: int = None
    _total_reward: float
    _total_profit: float
    _fig: plt.Figure
    _graphs: List[plt.Axes]

    # Constructor
    def __init__(
        self,
        data_frames: pd.DataFrame,
        window_size: int,
        render_mode: str = None,
        start: float = 0,
        goal: float = 0,
        stop_loss_limit: float = 0,
        max_shares_per_trade: int = 10,
    ):
        self.account = Account(start, goal, stop_loss_limit)

        self.render_mode = render_mode

        self.data_frames = data_frames
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self.max_shares_per_trade = max_shares_per_trade
        self.observation_space = gym.spaces.Box(low=-1e10, high=1e10, shape=self.shape, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(2 * self.max_shares_per_trade + 1)

        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._total_reward = None
        self._total_profit = None

        self.history = {}

    # Business logics
    def _process_data(self):
        prices = self.data_frames.loc[:, "Close"].to_numpy()

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _update_history(self, info: dict[str, float], action):
        type = Action.get_action_type(action)

        self.history.setdefault("actions", {})
        self.history["actions"].update({self._current_tick: type})

        for key, value in info.items():
            self.history.setdefault(key, [])
            self.history[key].append(value)

        self.history["account_total"].append(self.account.get_total_value(self.prices[self._current_tick]))
        self.history.setdefault("shares", [])
        self.history["shares"].append(self.account.holdings)

    def _fulfill_order(self, action):
        previous_price = self.prices[self._current_tick - 1]
        current_price = self.prices[self._current_tick]
        order_quantity = action
        available_funds = self.account.available_funds
        current_holding = self.account.holdings

        if (action > 0 and order_quantity * current_price <= available_funds) or (action < 0 and order_quantity <= current_holding):
            self.account.update_holding(action, current_price)
            self._last_trade_tick = self._current_tick

        delta = self.account.get_total_value(current_price) - self.account.get_total_value(previous_price)

        return delta

    # Lifecycle
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if seed is not None:
            self.action_space.seed(seed)
        else:
            self.action_space.seed(int(time.time()))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0
        self._total_profit = 0
        self.account.reset()
        self.history = {
            "reward": [0.0],
            "profit": [0.0],
            "actions": {},
            "account_total": [self.account.available_funds],
            "shares": [0],
        }

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        terminated = False
        self._truncated = False
        self._current_tick += self.window_size
        action_modifier = self.max_shares_per_trade
        trade = action - action_modifier

        if self._current_tick > self._end_tick:
            self._current_tick = self._end_tick

        step_reward = 0
        current_stock_price = self.prices[self._current_tick]

        if self._current_tick == self._end_tick:
            self._truncated = True
            terminated = True
            self._current_tick = self._end_tick
            trade = -self.account.holdings

        elif self.account.should_exit(current_stock_price):
            self._truncated = True
            terminated = True
            self._end_tick = self._current_tick
            trade = -self.account.holdings

        step_reward = self._fulfill_order(trade)

        self._total_reward += step_reward

        self._total_profit = self.account.calculate_profit(current_stock_price)

        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info, trade)

        if self.render_mode == "human":
            self.render()

        return observation, step_reward, terminated, self._truncated, info

    def close(self):
        plt.close()

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size + 1) : self._current_tick + 1]

    def _get_info(self):
        return dict(
            reward=self._total_reward,
            profit=self._total_profit,
        )

    # Render
    def render(self):
        if self.render_mode != "human":
            return

        if not hasattr(self, "_fig") or not hasattr(self, "_graphs"):
            self._fig, self._graphs = plt.subplots(3, 1, figsize=(10, 9))
            plt.ion()

        for graph in self._graphs:
            graph.clear()

        self._fig.canvas.manager.set_window_title("Trading History (Live)")
        self._plot_action_history(self._current_tick)
        self._plot_total_value_history()
        self._plot_shares_vs_time()

        plt.draw()
        plt.pause(0.01)

    def _plot_action_history(self, tick):
        trading_graph = self._graphs[1]
        trading_graph.plot(self.prices[:tick], label="Price", color="blue")

        action_history: Dict[int, ActionType] = self.history["actions"]

        for i in range(tick):
            action = action_history.get(i)
            if action != None:
                if action == ActionType.Buy:
                    trading_graph.plot(i, self.prices[i], "g^")
                elif action == ActionType.Sell:
                    trading_graph.plot(i, self.prices[i], "rv")
                elif action == ActionType.Hold:
                    trading_graph.plot(i, self.prices[i], "yo")

        trading_graph.set_title("Price and Actions")
        trading_graph.legend()

    def _plot_total_value_history(self):
        # Plot account balance on the second axis
        account_graph = self._graphs[0]

        total_values = []
        for total_value in self.history["account_total"]:
            total_values.append(total_value)

        account_graph.axes.get_xaxis().set_visible(False)
        account_graph.axhline(y=self.account.goal, label="Goal", color="green")
        account_graph.plot(total_values, label="Total value", color="black")
        account_graph.axhline(y=self.account.stop_loss_limit, label="Stop Loss", color="orange")
        account_graph.set_title("Total Account Value Over Time")
        account_graph.legend()

    def _plot_shares_vs_time(self):
        shares_graph = self._graphs[2]

        ticks = list(range(self._start_tick, self._current_tick + 1, self.window_size))

        shares_history = self.history["shares"]

        ticks = ticks[: len(shares_history)]
        shares_history = shares_history[: len(ticks)]

        # Plotting
        shares_graph.plot(ticks, shares_history, label="Shares Over Time", color="purple")
        shares_graph.set_title("Currently Owned Shares vs Time Tick")
        shares_graph.set_xlabel("Time Tick")
        shares_graph.set_ylabel("Number of Shares")
        shares_graph.legend()

    def render_final_result(self):
        self._fig, self._graphs = plt.subplots(3, 1, figsize=(16, 9))
        self._fig.canvas.manager.set_window_title("Trading History")

        # Plot the action history
        self._plot_action_history(self._end_tick + 1)

        # Plot the total value history
        self._plot_total_value_history()

        # Plot the shares vs time history
        self._plot_shares_vs_time()
        print(f"Final account balance is: {self.history['account_total'][-1]}")

        # Turn off interactive mode so that the plot stays up
        plt.ioff()
        plt.show()
