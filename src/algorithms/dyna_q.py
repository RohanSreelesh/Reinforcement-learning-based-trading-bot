import numpy as np
import random
from envs import TradingEnv
from models import Action
from algorithms.classes.helper import setup_env_for_testing
from algorithms.classes.state_featurizer import StateFeaturizer
from algorithms.classes.linear_dyna_q_learner import LinearDynaQLearner


class AgentActionHistory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def greedy_policy(agent: LinearDynaQLearner, env: TradingEnv, state: np.ndarray):
    predictions = agent.predict(state)
    action_mask = Action.get_action_mask(env)

    predictions_with_valid_action = np.array(
        [predictions[action] if Action.is_action_valid(action_mask, action) else -np.Infinity for action in np.arange(len(predictions))]
    )

    return np.argmax(predictions_with_valid_action)


def epsilon_greedy_policy(agent: LinearDynaQLearner, epsilon: float, env: TradingEnv, state: np.ndarray):
    if np.random.rand() < epsilon:
        action_mask = Action.get_action_mask(env)
        # Random action
        return env.action_space.sample(mask=action_mask)

    # Greedy action
    return greedy_policy(agent, env, state)


def demo():
    env, env_test = setup_env_for_testing()

    num_features = np.prod(env.observation_space.shape)
    num_actions = env.action_space.n

    agent = LinearDynaQLearner(num_features, num_actions)
    featurizer = StateFeaturizer()
    featurizer.fit(env.signal_features)

    action_history = AgentActionHistory(capacity=10000)

    max_episodes = 50
    planning_steps = 30
    max_steps_per_episode = 1000  # set a limit to the number of steps per episode

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = featurizer.transform(state)

        for step in range(max_steps_per_episode):
            action = epsilon_greedy_policy(agent, 0.1, env, state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = featurizer.transform(next_state)

            agent.update(state, next_state, action, reward, terminated or truncated)
            agent.update_model(state, action, next_state, reward)

            action_history.push(state, action, next_state, reward)

            for _ in range(planning_steps):
                sim_state, sim_action, _, _ = action_history.sample(1)[0]
                sim_next_state, sim_reward = agent.simulate_step(sim_state, sim_action)
                agent.update(sim_state, sim_next_state, sim_action, sim_reward, False)

            state = next_state

            if terminated or truncated:
                break

    # Test loop
    state, _ = env_test.reset()
    state = featurizer.transform(state)
    done = False

    while not done:
        action = greedy_policy(
            agent=agent,
            env=env_test,
            state=state,
        )

        next_state, reward, terminated, truncated, _ = env_test.step(action)
        next_state = state = featurizer.transform(next_state)

        done = terminated or truncated

        state = next_state

    env_test.close()
    env_test.render_final_result()
