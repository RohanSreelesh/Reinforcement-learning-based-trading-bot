import gymnasium as gym
from envs import TradingEnv
import data as STOCKS
from models import Action


def demo():
    env = gym.make(
        "trading-v1",
        data_frames=STOCKS.COCA_COLA,
        window_size=30,
        render_mode="human",
        start=1000,
        goal=2000,
        stop_loss_limit=500,
        max_shares_per_trade=1000,
    )

    trading_env: TradingEnv = env.unwrapped

    max_episodes = 1

    for _ in range(max_episodes):
        env.reset()

        while True:
            action_mask = Action.get_action_mask(env)
            action = env.action_space.sample(mask=action_mask)

            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

    env.close()
    trading_env.render_final_result()
