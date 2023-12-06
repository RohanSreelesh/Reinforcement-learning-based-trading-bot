import numpy as np
import gymnasium as gym
from envs import TradingEnv
import data as STOCKS
from models import Action

class LinearApproximator:
    def __init__(self, feature_size, action_size):
        # Initialize weights for each action
        self.weights = np.random.randn(feature_size, action_size)

    def predict(self, features):
        # Compute Q-values for all actions
        return np.dot(features, self.weights)

    def update(self, features, action, target, learning_rate):
        # Compute the prediction
        predictions = self.predict(features)
        error = target - predictions[action]
        # Update weights for the chosen action
        self.weights[:, action] += learning_rate * error * features

class QLearningAgent:
    def __init__(self, approximator, num_actions, learning_rate, discount_factor, epsilon):
        self.approximator = approximator
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def select_action(self, env, state, action_mask):
        if np.random.rand() < self.epsilon:
            # Exploration: Random action based on action mask
            valid_actions = [i - env.max_shares_per_trade for i, valid in enumerate(action_mask) if valid]
            action = np.random.choice(valid_actions)
        else:
            q_values = self.approximator.predict(state)
            q_values[~action_mask] = -np.inf

            best_action_index = np.argmax(q_values)
            action = best_action_index - env.max_shares_per_trade

            # Debugging: Log Q-values and action selection details
            print("Q-values:", q_values)
            print("Best Action Index:", best_action_index)
            print("Chosen Action:", action)
            print("Action Mask:", action_mask)
        print(f"Action: {action}")
        return action

    def update(self, state, action, reward, next_state, env):
        # Convert action back to original indexing
        action_index = action + env.max_shares_per_trade

        # Predict Q-values for the next state
        next_q_values = self.approximator.predict(next_state)
        
        # Calculate the target value
        target = reward + self.discount_factor * np.max(next_q_values)
        
        # Update the approximator
        self.approximator.update(state, action_index, target, self.learning_rate)
def create_trading_env():
    # Assuming you have a function or way to initialize your custom TradingEnv
    return TradingEnv(
        data_frames=STOCKS.COCA_COLA,
        window_size=30,
        render_mode="human",
        start=1000,  # Starting value, you should define it
        goal=2000,  # Goal value, you should define it
        stop_loss_limit=500,  # Stop loss limit, define as needed
        max_shares_per_trade=1000  # Maximum shares per trade, define as needed
    )

def demo():
    # Assume the environment (env) is defined and initialized here
    num_episodes = 100 # Define the number of episodes for training
    env = create_trading_env()
    feature_size = 60
    action_size = env.action_space.n
    # Initialize the linear function approximator and the Q-learning agent
    approximator = LinearApproximator(feature_size, action_size)
    agent = QLearningAgent(approximator, action_size, learning_rate=0.01, discount_factor=0.99, epsilon=0.1)

    # Main training loop
    for episode in range(num_episodes):
        # Reset the environment and extract the state
        observation, info = env.reset()
        print("Shape of observation:", observation.shape)  # This should help you understand the shape
        state = observation.flatten()  # Flatten the state

        done = False
        while not done:
            # Get the action mask from the environment
            action_mask = Action.get_action_mask(env)

            # Pass the state and action mask to the agent
            action = agent.select_action(env, state, action_mask)
            # Step the environment and extract the next state
            observation, reward, done, info = env.step(action)
            next_state = observation.flatten()  # Flatten the next state

            agent.update(state, action, reward, next_state, env)
            state = next_state

