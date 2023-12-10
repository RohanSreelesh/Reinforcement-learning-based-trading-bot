import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs import TradingEnv
import data as STOCKS
from stable_baselines3.common.callbacks import BaseCallback  # Import BaseCallback
from tqdm import tqdm
import numpy as np
from models import Action
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import ProgressBarCallback

def create_trading_env(STOCK):
    return TradingEnv(
        data_frames=STOCK,
        window_size=30,
        render_mode="human",
        start=100000,
        goal=200000,
        stop_loss_limit=50000,
        max_shares_per_trade=1000
    )
def action_mask_fn(env):
    valid_actions_mask = Action.get_action_mask(env)  # Returns mask 0 - 2001
    return valid_actions_mask
def demo():
    env = create_trading_env(STOCKS.NASDAQ_TRAIN)
    env = ActionMasker(env, action_mask_fn)

    model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=0.1, gamma=0.99, verbose=1, ent_coef=0.1, n_steps=2048)
    model.learn(total_timesteps=2048, use_masking=True, log_interval=1, progress_bar=True)
    model.save("ppo_trading_model")
    # Load the model
    model = MaskablePPO.load("ppo_trading_model")

    # Evaluate the trained model
    env = create_trading_env(STOCKS.NASDAQ_NEW)
    env = ActionMasker(env, action_mask_fn)
    obs = env.reset()[0]
    avilable_funds_ending = []
    for i in tqdm(range(1), desc="Evaluating Model"):
        obs = env.reset()[0]
        for _ in range(2000):
            action_masks = Action.get_action_mask(env)
            obs = np.array(obs).reshape(env.observation_space.shape)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, rewards, dones, truncated, infos = env.step(action)
            if dones:
                break
        avilable_funds_ending.append(env.history["account_total"][-1])
    env.render_final_result()
    with open("ppo_trading_model.txt", "w") as f:
        for item in avilable_funds_ending:
            f.write("%s\n" % item)
        f.write("Average:\n%s\n" % (sum(avilable_funds_ending) / len(avilable_funds_ending)))