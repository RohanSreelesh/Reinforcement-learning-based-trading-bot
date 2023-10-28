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
        goal=2000,
        stop_loss_limit=50,
    )

    trading_env: TradingEnv = env.unwrapped

    max_episodes = 1

    for _ in range(max_episodes):
        env.reset()

        while True:
            action_type: ActionType = env.action_space.sample()

            # TODO: should this be part of the action space as well or it's part of the learning steps for agent
            action = Action(action_type, 100)

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    env.close()
    trading_env.render_final_result()
