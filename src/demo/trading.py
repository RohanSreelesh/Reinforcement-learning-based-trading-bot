import gymnasium as gym
from envs import TradingEnv
import data as STOCKS
from models import ActionType, Action


def demo():
    env = gym.make(
        "trading-v1",
        data_frames=STOCKS.COCA_COLA,
        window_size=30,
        render_mode="human",
        start=1000,
        goal=1150,
        stop_loss_limit=900,
        max_shares_per_trade=10
    )

    trading_env: TradingEnv = env.unwrapped

    max_episodes = 1

    for _ in range(max_episodes):
        env.reset()

        while True:
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    env.close()
    trading_env.render_final_result()
