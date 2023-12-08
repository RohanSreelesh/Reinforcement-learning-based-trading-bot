import numpy as np


class SarsaLinearLearner:
    def __init__(self, num_features, num_actions, learning_rate=0.01, gamma=0.99):
        self.num_actions = num_actions
        self.weights = np.random.randn(num_features, num_actions)
        self.learning_rate = learning_rate
        self.gamma = gamma

    def predict(self, state: np.ndarray):
        """Predict Q-values for all actions for a given state"""
        return np.dot(state.flatten(), self.weights)

    def update(self, state: np.ndarray, next_state: np.ndarray, action: int, next_action: int, reward: float, done: bool):
        """Update the weights using the SARSA update rule"""
        Q_values_current = self.predict(state)
        Q_values_next = self.predict(next_state)

        # SARSA update
        td_target = reward + (0 if done else self.gamma * Q_values_next[next_action])
        td_error = td_target - Q_values_current[action]

        # Update weights
        self.weights[:, action] += self.learning_rate * td_error * state.flatten()
