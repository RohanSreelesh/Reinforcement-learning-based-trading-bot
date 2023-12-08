from models import Action
from .classes.helper import setup_env_for_testing


def demo():
    _, env = setup_env_for_testing()

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
    env.render_final_result()
