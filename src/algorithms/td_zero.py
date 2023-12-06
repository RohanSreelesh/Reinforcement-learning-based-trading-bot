import gymnasium as gym
import numpy as np
from envs import TradingEnv
import data as STOCKS
from models import Action

class TDAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))  # Action-value function initialization

    def choose_action(self, state, action_mask):
        # Extract price difference from the state
        price_diff = int(state[-1][-1])

        # Rest of the state without the price difference
        state_without_diff = int(state[-1][-1])

        # Applying the action mask to get valid actions
        adjusted_action_mask = np.roll(action_mask, self.n_actions // 2)
        valid_actions = np.where(adjusted_action_mask == 1)[0]
        best_action_index = valid_actions[np.argmax(self.Q[state_without_diff, valid_actions])]
        best_action = best_action_index - self.n_actions // 2

        return best_action

    def update(self, state, action, reward, next_state):
        # Extract price difference from both current and next state
        print(f"State: {state}, Next State: {next_state}")
        current_price_diff = int(state[-1][-1])
        next_price_diff = int(next_state[-1][-1])
        print(f"Current Price Diff: {current_price_diff}, Next Price Diff: {next_price_diff}")
        # Rest of the state without the price difference

        # Update Q-value
        next_best_action = np.argmax(self.Q[int(next_price_diff)])
        self.Q[current_price_diff, action] += self.alpha * (reward + self.gamma * self.Q[next_price_diff, int(next_price_diff)] - self.Q[current_price_diff, action])

def demo():
    goal = 2000
    stop_loss_limit = 0
    start = 1000  # Assuming this is the starting account value
    env = gym.make(
        "trading-v1",
        data_frames=STOCKS.COCA_COLA,
        window_size=30,
        render_mode="human",
        start=start,
        goal=goal,
        stop_loss_limit=stop_loss_limit,
        max_shares_per_trade=1000,
    )

    trading_env: TradingEnv = env.unwrapped
    n_states = goal - stop_loss_limit + 10000
    n_actions = 2 * trading_env.max_shares_per_trade + 1  # Total number of possible actions
    agent = TDAgent(n_states, n_actions)

    max_episodes = 10  # Number of episodes for training

    for episode in range(max_episodes):
        state_tuple, _ = env.reset()
        current_state_array = state_tuple[0]  # Extracting the array part of the state
        current_state = np.reshape(current_state_array, [1, -1])  # Reshape to fit the model
        total_reward = 0
        done = False

        print(f"Starting episode {episode+1}")

        while not done:
            action_mask = Action.get_action_mask(env)
            action = 0
            # use episolon greedy to choose action
            if np.random.uniform() < 0.9:
                action = env.action_space.sample(mask=action_mask)
            else:
                action = agent.choose_action(current_state, action_mask)
            
            # Execute the chosen action in the environment
            next_state_tuple, reward, terminated, truncated, info = env.step(action)
            next_state_array = next_state_tuple  # Extracting the array part of the next state
            next_state = np.reshape(next_state_array, [1, -1])

            total_reward += reward

            # Update the agent
            agent.update(current_state, action, reward, next_state)

            # Logging information
            print(f"Current State: {current_state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Total Reward: {total_reward}")

            # Transition to the next state
            current_state = next_state

            if terminated or truncated:
                print("Episode terminated.")
                break
            
        trading_env.render_final_result()
        print(f"Episode {episode+1} finished with total reward: {total_reward}")
    env.close()
