from enums import Action as ActionType
from models import Account
import math
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs import TradingEnv

VALID = 1
INVALID = 0


class Action:
    @staticmethod
    def get_action_type(action: int) -> ActionType:
        if action > 0:
            return ActionType.Buy

        if action < 0:
            return ActionType.Sell

        return ActionType.Hold

    @staticmethod
    def get_action_mask(env: "TradingEnv") -> np.ndarray:
        trading_env: "TradingEnv" = env.unwrapped
        account: Account = trading_env.account
        stock_price: float = trading_env.prices[trading_env._current_tick]

        limit = trading_env.max_shares_per_trade
        low = -account.holdings
        high = math.floor(account.available_funds / stock_price)

        mask = [VALID if (value >= low and value <= high) else INVALID for value in range(-limit, limit + 1)]

        return np.array(mask, dtype=np.int8)

    @staticmethod
    def is_action_valid(mask: np.ndarray, action: int):
        return mask[action] == VALID
