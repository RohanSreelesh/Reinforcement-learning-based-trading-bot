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


class VerboseCallback(BaseCallback):
    def _on_step(self) -> bool:
        print("Step number:", self.num_timesteps)
        return True
def create_trading_env(STOCK):
    # Assuming you have a function or way to initialize your custom TradingEnv
    return TradingEnv(
        data_frames=STOCK,
        window_size=30,
        render_mode="human",
        start=40000,  # Starting value, you should define it
        goal=200000,  # Goal value, you should define it
        stop_loss_limit=0,  # Stop loss limit, define as needed
        max_shares_per_trade=1000  # Maximum shares per trade, define as needed
    )
def train_for_one_episode(env, model):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
def action_mask_fn(env):
    # Calculate the valid action mask here
    valid_actions_mask = Action.get_action_mask(env)  # Returns mask 0 - 2001
    return valid_actions_mask
def demo():
    # Create vectorized environments
    # vec_env = make_vec_env(create_trading_env, n_envs=1)
    # # Initialize PPO model
    #model = PPO("MlpPolicy", env, verbose=1)

    env = create_trading_env(STOCKS.NASDAQ_TRAIN)
    env = ActionMasker(env, action_mask_fn)
    # callback = VerboseCallback()

    # # Train the model
    # policy_kwargs = dict(ent_coef=0.01)  # Adjust this value as needed

    print("learning")
    model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=0.1, gamma=0.99, verbose=1, ent_coef=0.1, n_steps=5000)
    model.learn(total_timesteps=5000, use_masking=True, log_interval=1, progress_bar=True)
    model.save("ppo_trading_model")

    # Load the model
    model = MaskablePPO.load("ppo_trading_model")

    # Evaluate the trained model
    env = create_trading_env(STOCKS.NASDAQ_TEST)
    env = ActionMasker(env, action_mask_fn)
    obs = env.reset()
    obs = obs[0]
    avilable_funds_ending = []
    for _ in tqdm(range(2000), desc="Evaluating Model"):
        action_masks = Action.get_action_mask(env)
        obs = np.array(obs).reshape(env.observation_space.shape)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, rewards, dones, truncated, infos = env.step(action)
        if dones:
            break
    #env.render()
    print(env.history["account_total"])
    print(env.history["account_total"][-1])
    avilable_funds_ending.append(env.history["account_total"][-1])
    #env.close()
    #env.render_final_result()
    # Push it to a text file
    with open("ppo_trading_model.txt", "w") as f:
        for item in avilable_funds_ending:
            f.write("%s\n" % item)