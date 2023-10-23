from gymnasium.envs.registration import register
from .trading import *

register(
    id='trading-v1',
    entry_point='envs.trading:TradingEnv',
    kwargs={
        'data_frames': None,
        'window_size': 30
    }
)
