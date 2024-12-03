import numpy as np
import pickle

from confs.path_conf import tsp_agents_data_dir
from rl.tsp.agents.agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.5,
                 reward_strategy="negative_distance", model_path=tsp_agents_data_dir + "best_q-table.pkl"):
        """
        Initialize the Q-Learning Agent.

        :param num_cities: Number of cities
        :param num_actions: Number of possible actions
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration rate
        :param reward_strategy: Reward strategy to be used
        :param model_path: Path to save/load the Q-table model
        """
        super().__init__(num_cities, num_actions, alpha, gamma, epsilon, reward_strategy, model_path)

    def initialize_model(self):;
        """
        Initialize the Q-learning model by creating a Q-table with zeros.
        """
        self.q_table = np.zeros((self.num_cities, self.num_actions))  # Initialize Q-table
        print("Initialized new Q-learning model.")

    def load_model(self):
        """
        Load the Q-learning model from the specified path.
        """
        try:
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                if saved_data["num_cities"] == self.num_cities:
                    self.q_table = saved_data["q_table"]
                    self.history_best_distance = saved_data["history_best_distance"]
                    print("Loaded existing Q-learning model.")
                else:
                    print(
                        f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_data['num_cities']}. Initializing new model.")
                    self.initialize_model()
        except FileNotFoundError:
            print(f"Model path {self.model_path} does not exist. Initializing new Q-learning model.")
            self.initialize_model()
        except Exception as e:
            print(f"Error loading model: {e}. Initializing new Q-learning model.")
            self.initialize_model()

    def save_model(self):
        """
        Save the current state of the Q-learning model to a file.
        """
        saved_data = {
            "num_cities": self.num_cities,
            "q_table": self.q_table,
            "history_best_distance": self.history_best_distance
        }
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(saved_data, f)
            print(f"Model saved with distance: {self.history_best_distance}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def get_greedy_action(self, state):
        """
        Select the action with the highest Q-value (greedy action).

        :param state: Current state (dictionary)
        :return: Greedy action
        """
        state_index = self.state_to_index(state)
        return np.argmax(self.q_table[state_index])  # Greedy action

    def state_to_index(self, state):
        """
        Map the state dictionary to the Q-table index.

        :param state: State dictionary
        :return: State index (assuming it contains 'current_city')
        """
        return state['current_city']

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table based on the state-action-reward sequence.

        :param state: Current state (dictionary)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (dictionary)
        :param done: Whether the episode has ended
        """
        current_index = self.state_to_index(state)
        next_index = self.state_to_index(next_state)

        # Select the best action for the next state
        best_next_action = np.argmax(self.q_table[next_index])

        # Compute the TD target using the Bellman equation
        td_target = reward + self.gamma * self.q_table[next_index, best_next_action] * (1 - done)

        # Compute the TD error
        td_error = td_target - self.q_table[current_index, action]

        # Update the Q-table using the TD error
        self.q_table[current_index, action] += self.alpha * td_error

    def train(self, env, num_episodes):
        """
        Train the Q-Learning agent in the given environment.

        :param env: Environment instance
        :param num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state
            total_distance = 0
            done = False
            visited = []
            print(f"Episode {episode + 1}/{num_episodes} - Training")

            while not done:
                # Choose an action using the parent class's choose_action method
                action = self.choose_action(state)

                # Take action in the environment
                next_state, base_reward, done = env.step(action)
                # Assuming env.step returns (next_state, base_reward, done)

                # Prepare reward parameters for adjustment
                reward_params = {
                    "distance": next_state['step_distance'],  # Distance for the current step
                    "done": done,
                    "total_distance": next_state['total_distance'],
                    "visited": next_state['current_path'],
                    "env": env
                }

                # Adjust the reward based on the strategy
                reward = self.adjust_reward(reward_params)

                # Update the Q-table with the transition
                self.update(state, action, reward, next_state, done)

                # Update state and track total distance
                state = next_state
                visited.append(state['current_city'])  # Track visited cities (example)
                total_distance = env.total_distance  # Use environment's total_distance attribute

            # Save the model if a new best distance is achieved
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()
