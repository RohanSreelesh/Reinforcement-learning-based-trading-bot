import numpy as np

class LinearModel:
    def __init__(self, num_features, num_actions):
        self.state_weights = np.random.randn(num_features, num_features)
        self.reward_weights = np.random.randn(num_features, 1)
        self.learning_rate = 0.01

    def update(self, state, action, next_state, reward):
        predicted_next_state = np.dot(state.flatten(), self.state_weights)
        predicted_reward = np.dot(state.flatten(), self.reward_weights).item()

        state_error = next_state.flatten() - predicted_next_state
        reward_error = reward - predicted_reward

        self.state_weights += self.learning_rate * np.outer(
            state.flatten(), state_error)
        self.reward_weights += self.learning_rate * reward_error * state.flatten(
        )[:, np.newaxis]

    def predict(self, state, action):
        simulated_next_state = np.dot(state.flatten(), self.state_weights)
        simulated_reward = np.dot(state.flatten(), self.reward_weights).item()
        return simulated_next_state, simulated_reward