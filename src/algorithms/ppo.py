from envs import TradingEnv
from tqdm import tqdm
import numpy as np
from models import Action
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from algorithms.classes.helper import setup_env_for_testing


def action_mask_fn(env: TradingEnv):
    return Action.get_action_mask(env)  # Returns mask 0 - 2001


def demo():
    env, env_test = setup_env_for_testing()
    env = ActionMasker(env, action_mask_fn)

    model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=0.1, gamma=0.99, verbose=1, ent_coef=0.1, n_steps=2048)
    model.learn(total_timesteps=2048, use_masking=True, log_interval=1, progress_bar=True)
    model.save("assets/ppo_trading_model")

    # Load the model
    model = MaskablePPO.load("assets/ppo_trading_model")

    # Evaluate the trained model
    env = ActionMasker(env_test, action_mask_fn)
    obs = env.reset()[0]
    available_funds_ending = []

    for _ in tqdm(range(1), desc="Evaluating Model"):
        obs = env.reset()[0]

        for _ in range(2000):
            action_masks = Action.get_action_mask(env)
            obs = np.array(obs).reshape(env.observation_space.shape)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, _, terminated, _, _ = env.step(action)

            if terminated:
                break

        available_funds_ending.append(env.unwrapped.history["account_total"][-1])

    env.unwrapped.render_final_result()

    with open("logs/ppo_trading_model.txt", "w") as f:
        for item in available_funds_ending:
            f.write("%s\n" % item)
        f.write("Average:\n%s\n" % (sum(available_funds_ending) / len(available_funds_ending)))
