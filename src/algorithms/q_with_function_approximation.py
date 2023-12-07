import numpy as np
from models import Action
from envs import TradingEnv
from .classes.helper import setup_env_for_testing
from .classes.linear_learner import LinearQLearner
from .classes.state_featurizer import StateFeaturizer

def epsilon_greedy_policy(agent: LinearQLearner, epsilon: float, env: TradingEnv, state: np.ndarray):
    if np.random.rand() < epsilon:
        action_mask = Action.get_action_mask(env)
        # Random action
        return env.action_space.sample(mask=action_mask) 

    # Greedy action
    return np.argmax(agent.predict(state))

def demo ():
    env = setup_env_for_testing()

    num_features = np.prod(env.observation_space.shape)
    num_actions = env.action_space.n

    agent = LinearQLearner(num_features, num_actions)

    featurizer = StateFeaturizer()
    featurizer.fit(env.signal_features)
    
    max_episodes = 1000

    # Training loop
    for _ in range(max_episodes):
        state, _ = env.reset()
        state = featurizer.transform(state)
        done = False

        while not done:
            action = epsilon_greedy_policy(
                agent=agent,
                env=env,
                state=state,
                epsilon=0.1
            )

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = state = featurizer.transform(next_state)
            
            done = terminated or truncated

            agent.update(
                state=state,
                next_state=next_state,
                action=action,
                reward=reward,
                done=done
            )

            state = next_state
            
    env.close()
    env.render_final_result()
