import numpy as np
from .linear_model import LinearModel


class LinearDynaQLearner:
    def __init__(self, num_features, num_actions, learning_rate=0.01, gamma=0.99):
        self.num_actions = num_actions
        self.weights = np.random.randn(num_features, num_actions)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = LinearModel(num_features, num_actions)

    def predict(self, state: np.ndarray):
        return np.dot(state.flatten(), self.weights)

    def update(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float, done: bool):
        Q_values_current = self.predict(state)
        Q_values_next = self.predict(next_state)
        best_next_action = np.max(Q_values_next)
        td_target = reward + (0 if done else self.gamma * best_next_action)
        td_error = td_target - Q_values_current[action]
        self.weights[:, action] += self.learning_rate * td_error * state.flatten()

    def update_model(self, state, action, next_state, reward):
        self.model.update(state, action, next_state, reward)

    def simulate_step(self, state, action):
        return self.model.predict(state, action)
