from envs import TradingEnv
import data as STOCKS
import gymnasium as gym


def setup_env_for_testing():
    env_train = gym.make(
        "trading-v1",
        data_frames=STOCKS.NASDAQ_TRAIN,
        window_size=5,
        render_mode=None,
        start=100000,
        goal=200000,
        stop_loss_limit=50000,
        max_shares_per_trade=10,
    )

    env_test = gym.make(
        "trading-v1",
        data_frames=STOCKS.NASDAQ_TEST,
        window_size=5,
        render_mode=None,
        start=100000,
        goal=200000,
        stop_loss_limit=50000,
        max_shares_per_trade=10,
    )

    trading_env_train: TradingEnv = env_train.unwrapped
    trading_env_test: TradingEnv = env_test.unwrapped

    return trading_env_train, trading_env_test
