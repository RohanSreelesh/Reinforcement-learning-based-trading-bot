import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
from typing import List
from models import Action, Account
from enums import Reward, Action as ActionType


class TradingEnv(gym.Env):
    # Properties (public)
    render_mode: str
    prices: float
    data_frames: pd.DataFrame
    window_size: int
    shape: tuple[int, int]

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box

    account: Account
    history: dict[str, List[float]]

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
    ):
        self.account = Account(start, goal, stop_loss_limit)

        self.render_mode = render_mode

        self.data_frames = data_frames
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(n=len(ActionType), start=ActionType.Sell.value)
        self.observation_space = gym.spaces.Box(
            low=-1e10, high=1e10, shape=self.shape, dtype=np.float32
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._total_reward = None
        self._total_profit = None

        # tracking
        self.history = {}

    # Business logics
    def _process_data(self):
        prices = self.data_frames.loc[:, "Close"].to_numpy()

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _update_history(self, info: dict[str, float], action: Action):
        self.history.setdefault("action", [])
        self.history["action"].append(action.type)
        for key, value in info.items():
            self.history.setdefault(key, [])
            self.history[key].append(value)
        self.history["account_total"].append(self.account.get_total_value(self.prices[self._current_tick]))

    def _fulfill_order(self, action: Action):
        current_price = self.prices[self._current_tick]
        type = action.type
        order_quantity = action.quantity
        available_funds = self.account.available_funds
        current_holding = self.account.holdings

        match type:
            case ActionType.Buy:
                if order_quantity * current_price > available_funds:
                    return Reward.INVALID.value

                self._last_trade_tick = self._current_tick
                return self.account.update_holding(action.quantity, current_price)

            case ActionType.Sell:
                if order_quantity > current_holding:
                    return Reward.INVALID.value
                self._last_trade_tick = self._current_tick
                return self.account.update_holding(-action.quantity, current_price)

            case _:
                return Reward.NO_TRADE.value

    # Lifecycle
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0
        self._total_profit = 0
        self.account.reset()
        self.history = {"reward": [0.0], "profit": [0.0], "action": [ActionType.Hold], "account_total": [self.account.get_current_account_value()]}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: Action):
        self._truncated = False
        self._current_tick += 1
        step_reward = 0
        if self._current_tick == self._end_tick or self.account.should_exit():
            self._truncated = True

        if self.account.should_exit() or self._current_tick == self._end_tick:
            step_reward = self._fulfill_order(Action(ActionType.Sell, self.account.holdings))
        else:
            step_reward = self._fulfill_order(action)

        self._total_reward += step_reward

        self._total_profit = self.account.calculate_profit()

        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info, action)

        if self.render_mode == "human":
            self.render()
        return observation, step_reward, False, self._truncated, info

    def close(self):
        plt.close()

    def _get_observation(self):
        return self.signal_features[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]

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
            self._fig, self._graphs = plt.subplots(2, 1, figsize=(10, 6))
            plt.ion()

        for graph in self._graphs:
            graph.clear()

        self._fig.canvas.manager.set_window_title("Action and Balance History (Live)")
        self._plot_action_history_live()
        self._plot_total_value_history()
        plt.draw()
        plt.pause(0.01)

    def _plot_action_history_live(self):
        trading_graph = self._graphs[0]
        trading_graph.plot(self.prices[: self._current_tick], label="Price", color="blue")

        for tick in range(self._start_tick, len(self.history["action"])):
            if self.history["action"][tick] == ActionType.Buy:
                trading_graph.plot(tick, self.prices[tick], "g^")
            elif self.history["action"][tick] == ActionType.Sell:
                trading_graph.plot(tick, self.prices[tick], "rv")
            elif self.history["action"][tick] == ActionType.Hold:
                trading_graph.plot(tick, self.prices[tick], "yo")

        trading_graph.set_title("Price and Actions")
        trading_graph.legend()

    def _plot_action_history_final(self):
        # Plot prices and actions on the first axis
        trading_graph = self._graphs[0]
        trading_graph.plot(
            self.prices[: len(self.history["action"]) + self.window_size],
            label="Price",
            color="blue",
        )

        for tick in range(len(self.history["action"])):
            if self.history["action"][tick] == ActionType.Buy:
                trading_graph.plot(tick, self.prices[tick], "g^")
            elif self.history["action"][tick] == ActionType.Sell:
                trading_graph.plot(tick, self.prices[tick], "rv")
            elif self.history["action"][tick] == ActionType.Hold:
                trading_graph.plot(tick, self.prices[tick], "yo")

        trading_graph.set_title("Price and Actions")
        trading_graph.legend()

    def _plot_total_value_history(self):
        # Plot account balance on the second axis
        account_graph = self._graphs[1]

        total_values = []
        for reward in self.history["account_total"]:
            total_value = reward
            total_values.append(total_value)

        account_graph.plot(total_values, label="Total value", color="black")
        account_graph.set_title("Total Account Value Over Time")
        account_graph.legend()


    def render_final_result(self):
        self._fig, self._graphs = plt.subplots(2, 1, figsize=(16, 6))
        self._fig.canvas.manager.set_window_title("Action and Balance History")
        self._plot_action_history_final()
        self._plot_total_value_history()

        # Turn off interactive mode so that the plot stays up
        plt.ioff()
        plt.show()
