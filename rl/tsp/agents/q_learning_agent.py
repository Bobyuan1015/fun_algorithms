import numpy as np
import os
import random
import pickle

from confs.path_conf import system_model_dir
from rl.tsp.common.data_processor import adjust_reward


class QLearningAgent:
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 reward_strategy="negative_distance", model_path=system_model_dir + "best_q-table.pkl"):
        self.num_cities = num_cities
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_best_distance = float('inf')  # Track the best distance for bonus calculation
        self.reward_strategy = reward_strategy  # Reward strategy used
        self.model_path = model_path  # Path to save/load the model

        # Check if a saved model exists and load it
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.q_table = np.zeros((num_cities, num_actions))  # Initialize Q-table
            print("Initialized new Q-learning model.")

    def load_model(self):
        """Load only the key data from the saved model."""
        with open(self.model_path, 'rb') as f:
            saved_data = pickle.load(f)
            if saved_data["num_cities"] == self.num_cities:
                self.q_table = saved_data["q_table"]
                self.history_best_distance = saved_data["history_best_distance"]
                print("Loaded existing Q-learning model.")
            else:
                print(f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_data['num_cities']}. Initializing new model.")
                self.q_table = np.zeros((self.num_cities, self.num_actions))  # Initialize Q-table

    def save_model(self):
        """Save only the necessary data."""
        saved_data = {
            "num_cities": self.num_cities,
            "q_table": self.q_table,
            "history_best_distance": self.history_best_distance
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(saved_data, f)
        print(f"Model saved with distance: {self.history_best_distance}")

    def choose_action(self, state):
        """
        Choose action based on epsilon-greedy policy
        :param state: Current state (a dictionary)
        :return: Chosen action
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Random action
        else:
            state_index = self.state_to_index(state)
            return np.argmax(self.q_table[state_index])  # Greedy action

    def state_to_index(self, state):
        """
        Map the state dictionary to an index in the Q-table.
        :param state: State dictionary
        :return: Index for the state (assuming 'current_city' is part of the dictionary)
        """
        return state['current_city']

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value based on the state-action-reward sequence.
        :param state: Current state (dict)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (dict)
        """
        current_index = self.state_to_index(state)
        next_index = self.state_to_index(next_state)

        best_next_action = np.argmax(self.q_table[next_index])
        td_target = reward + self.gamma * self.q_table[next_index, best_next_action]
        td_error = td_target - self.q_table[current_index, action]
        self.q_table[current_index, action] += self.alpha * td_error

    def train(self, env, num_episodes):
        """
        Train the Q-learning agent in the TSP environment
        :param env: TSP environment instance
        :param num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state (now a dictionary)
            total_distance = 0
            visited = []
            print(f"{episode}/{num_episodes}  train")
            done = False
            while not done:
                action = self.choose_action(state)  # Choose action based on current state
                next_state, base_reward, done = env.step(action)  # Get next state, reward, and done flag

                # Prepare reward parameters
                reward_params = {
                    "distance": next_state['step_distance'],  # Current step's distance
                    "done": done,
                    "total_distance": next_state['total_distance'],
                    "visited": next_state['current_path'],
                    "env": env
                }

                # Adjust reward using the chosen strategy
                reward = adjust_reward(reward_params, self.reward_strategy, self.history_best_distance)
                # Update Q-table with new state-action-reward info
                self.update_q_value(state, action, reward, next_state)

                # Update state and other variables
                state = next_state
                visited.append(state['current_city'])  # Track visited cities (example)
                total_distance = env.total_distance

            # Update best path distance after episode
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()  # Save the model using pickle after each episode

