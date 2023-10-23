import gymnasium as gym
from envs import TradingEnv, Actions

def demo ():
    env = gym.make('trading-v1', render_mode="human")
    max_episodes = 1000

    for _ in range(max_episodes):
        action = env.action_space.sample()

        _, _, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            env.reset()

    env.close()
