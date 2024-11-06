import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pickle

from confs.path_conf import system_model_dir
from rl.tsp.common.data_processor import adjust_reward, state_to_vector


class DQNAgent:
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 model_path=system_model_dir+"best_dqn_model.pkl",reward_strategy="negative_distance"):
        self.num_cities = num_cities
        self.num_actions = num_actions
        self.reward_strategy = reward_strategy
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_best_distance = float('inf')  # Track the best distance for bonus calculation
        self.model_path = model_path  # Path to save/load the model
        self.device = torch.device( "cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the neural network for Q-value approximation
        self.q_network = self._build_model(self.num_cities * 2 + 2)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        # Check if a saved model exists and load it
        if os.path.exists(self.model_path):
            saved_agent = torch.load(self.model_path)
            if saved_agent['num_cities'] == self.num_cities:
                self.q_network.load_state_dict(saved_agent['q_network_state_dict'])
                self.history_best_distance = saved_agent['history_best_distance']
                print("Loaded existing DQN model.")
            else:
                print(f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_agent['num_cities']}. Initializing new model.")
        else:
            print("Initialized new DQN model.")

    def _build_model(self,input_dim):
        # Build a simple fully connected neural network to approximate Q-values
        model = nn.Sequential(
            nn.Linear(input_dim, 128),  # Adjust input size based on the state representation
            nn.ReLU(),
            nn.Linear(128, 128),  # Hidden layer
            nn.ReLU(),
            nn.Linear(128, self.num_actions)  # Output layer (Q-values for each action)
        )
        return model.to(self.device)


    def choose_action(self, state):
        state_vector = state_to_vector(state, self.num_cities)  # Convert state to vector
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Random action
        else:
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Greedy action

    def update_q_value(self, state, action, reward, next_state, done):
        state_vector = state_to_vector(state, self.num_cities)
        next_state_vector = state_to_vector(state, self.num_cities)

        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        best_next_action = torch.argmax(next_q_values)
        td_target = reward + self.gamma * next_q_values[0, best_next_action] * (1 - done)
        loss = self.loss_fn(q_values[0, action], td_target)

        # Backpropagate the loss and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state (a dictionary)
            done = False
            total_distance = 0

            while not done:
                action = self.choose_action(state)  # Choose action
                next_state, reward, done = env.step(action)  # Take action and get next state
                reward_params = {
                    "distance": next_state['step_distance'],  # Current step's distance
                    "done": done,
                    "total_distance": next_state['total_distance'],
                    "visited": next_state['current_path'],
                    "env": env
                }
                # Adjust reward using the chosen strategy
                reward = adjust_reward(reward_params, self.reward_strategy, self.history_best_distance)
                # Update Q-value(network) using DQN
                self.update_q_value(state, action, reward, next_state, done)

                state = next_state
                total_distance = next_state['total_distance']

            # Save the model if the total distance improves
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                torch.save({
                    'num_cities': self.num_cities,
                    'q_network_state_dict': self.q_network.state_dict(),
                    'history_best_distance': self.history_best_distance
                }, self.model_path)
                print(f"Model saved at episode {episode}, distance: {total_distance}")
