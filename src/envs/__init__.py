from gymnasium.envs.registration import register
from .trading import *
from .trading2 import *

register(id="trading-v1", entry_point="envs.trading:TradingEnv")
register(id="trading-v2", entry_point="envs.trading2:TradingEnv1")
