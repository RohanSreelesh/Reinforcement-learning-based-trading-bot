import gymnasium as gym
from envs import TradingEnv
import data as STOCKS
from models import Action
from tqdm import tqdm


def demo():
    env = gym.make(
        "trading-v1",
        data_frames=STOCKS.NASDAQ_TEST,
        window_size=30,
        render_mode="human",
        start=40000,
        goal=200000,
        stop_loss_limit=0,
        max_shares_per_trade=1000,
    )

    trading_env: TradingEnv = env.unwrapped

    max_episodes = 100
    avilable_funds_ending = []
    for i in tqdm(range(max_episodes), desc="Evaluating Model"):
        env.reset()
        print(i)
        while True:
            action_mask = Action.get_action_mask(env)
            action = env.action_space.sample(mask=action_mask)

            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break
        avilable_funds_ending.append(env.history["account_total"][-1])
    #env.close()
    #trading_env.render_final_result()
    with open("basic.txt", "w") as f:
        for item in avilable_funds_ending:
            f.write("%s\n" % item)
        # write average
        f.write("Average:\n%s\n" % (sum(avilable_funds_ending) / len(avilable_funds_ending)))
