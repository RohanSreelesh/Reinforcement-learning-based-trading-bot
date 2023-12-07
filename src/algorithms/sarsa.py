import numpy as np
from models import Action
from envs import TradingEnv
from .classes.helper import setup_env_for_testing
from .classes.sarsa_linear_learner import SarsaLinearLearner
from .classes.state_featurizer import StateFeaturizer


def epsilon_greedy_policy(agent: SarsaLinearLearner, epsilon: float, env: TradingEnv, state: np.ndarray):
    if np.random.rand() < epsilon:
        action_mask = Action.get_action_mask(env)
        # Random action
        return env.action_space.sample(mask=action_mask)

    # Greedy action
    return np.argmax(agent.predict(state))


def demo():
    env, _ = setup_env_for_testing() # choose the training environment

    num_features = np.prod(env.observation_space.shape)
    num_actions = env.action_space.n

    agent = SarsaLinearLearner(num_features, num_actions)

    featurizer = StateFeaturizer()
    featurizer.fit(env.signal_features)

    max_episodes = 1000

    # Training loop
    for _ in range(max_episodes):
        state, _ = env.reset()
        state = featurizer.transform(state)
        done = False

        # Choose initial action using epsilon-greedy policy
        action = epsilon_greedy_policy(agent=agent, env=env, state=state, epsilon=0.1)

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = featurizer.transform(next_state)

            done = terminated or truncated

            # Choose next action using epsilon-greedy policy
            next_action = epsilon_greedy_policy(agent=agent, env=env, state=next_state, epsilon=0.1)

            # Update Q-values using SARSA update rule
            agent.update(state=state, next_state=next_state, action=action, reward=reward, next_action=next_action, done=done)

            state = next_state
            action = next_action  # Update action to next action

    env.close()
    env.render_final_result()
    print(env.history["account_total"][-1])

    _, env = setup_env_for_testing() # choose the testing environment
    # Run a single episode for testing
    max_episodes = 1
    for _ in range(max_episodes):
        state, _ = env.reset()
        state = featurizer.transform(state)
        done = False

        # Choose initial action using epsilon-greedy policy
        action = epsilon_greedy_policy(agent=agent, env=env, state=state, epsilon=0.1)

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = featurizer.transform(next_state)

            done = terminated or truncated

            # Choose next action using epsilon-greedy policy
            next_action = epsilon_greedy_policy(agent=agent, env=env, state=next_state, epsilon=0.1)

            # Update Q-values using SARSA update rule
            # agent.update(state=state, next_state=next_state, action=action, reward=reward, next_action=next_action, done=done)

            state = next_state
            action = next_action  # Update action to next action

    # Close the testing environment and render the test result
    env.close()
    env.render_final_result()
    print(env.history["account_total"][-1])
