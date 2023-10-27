import gymnasium as gym
from envs import TradingEnv
from data import COCA_COLA, TD
from models import ActionType, Action, Account
import matplotlib.pyplot as plt


def demo():
    env: TradingEnv = gym.make(
        "trading-v1",
        data_frames=COCA_COLA,
        window_size=30,
        render_mode="human",
        start=1000,
        goal=2000,
        stop_loss_limit=50,
    )

    max_episodes = 250

    env.reset()

    for _ in range(max_episodes):
        action_type: ActionType = env.action_space.sample()

        # TODO: should this be part of the action space as well or it's part of the learning steps for agent
        action = Action(action_type, 100)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    env.close()
    env.unwrapped.render_all()
