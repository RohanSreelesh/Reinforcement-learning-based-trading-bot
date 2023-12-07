from envs import TradingEnv
import data as STOCKS
import gymnasium as gym

def setup_env_for_testing ():
    env = gym.make(
        "trading-v1",
        data_frames=STOCKS.NASDAQ_TEST,
        window_size=1,
        render_mode=None,
        start=10000,
        goal=20000,
        stop_loss_limit=5000,
        max_shares_per_trade=1,
    )

    trading_env: TradingEnv = env.unwrapped
    
    return trading_env
