from envs import TradingEnv
from models import Account
import math
import numpy as np


def get_valid_action_masking(env: TradingEnv):
    account: Account = env.unwrapped.account
    stock_price: float = env.unwrapped.prices[env.unwrapped._current_tick]

    limit = env.unwrapped.max_shares_per_trade
    low = -account.holdings
    high = math.floor(account.available_funds / stock_price)

    mask = [1 if (value >= low and value <= high) else 0 for value in range(-limit, limit + 1)]

    return np.array(mask, dtype=np.int8)
