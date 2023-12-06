from enums import Action as ActionType
from models import Account
import math
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs import TradingEnv

class Action:
    @staticmethod
    def get_action_type (action: int) -> ActionType:
        if action > 0:
            return ActionType.Buy

        if action < 0:
            return ActionType.Sell

        return ActionType.Hold

    @staticmethod
    def get_action_mask (env: 'TradingEnv') -> np.ndarray:
        trading_env: 'TradingEnv' = env.unwrapped
        account: Account = trading_env.account
        if not trading_env._current_tick:
            trading_env._current_tick = 0
        stock_price: float = trading_env.prices[trading_env._current_tick]
        limit = trading_env.max_shares_per_trade
        low = -account.holdings
        
        high = int(math.floor(account.available_funds / stock_price))
        mask = [1 if (value >= low and value <= high) else 0 for value in range(-limit, limit + 1)]
        # print(f"account available funds {account.available_funds}, stock price {stock_price}, low {low}, high {high}, mask {mask}")
        # print(len(mask))
        # print(f"account available funds {account.available_funds}, stock price {stock_price}, low {low}, high {high}")
        return np.array(mask, dtype=np.int8)
