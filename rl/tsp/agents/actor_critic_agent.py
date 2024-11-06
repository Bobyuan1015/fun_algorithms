import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pickle

from confs.path_conf import system_model_dir
from rl.tsp.common.data_processor import adjust_reward, state_to_vector

class ActorCriticAgent:
    def __init__(self, num_cities, num_actions, alpha=0.001, gamma=0.99, epsilon=0.1,
                 reward_strategy="negative_distance", model_path=system_model_dir + "best_actor_critic_model.pkl"):
        self.num_cities = num_cities
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.reward_strategy = reward_strategy  # Reward strategy used
        self.model_path = model_path  # Path to save/load the model

        # Neural networks for actor and critic
        self.actor = self.build_actor(self.num_cities * 2 + 2)
        self.critic = self.build_critic(self.num_cities * 2 + 2)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.alpha)

        # Initialize the best distance with a large number
        self.history_best_distance = float('inf')

        # Check if a saved model exists and load it
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print("Initialized new Actor-Critic model.")


    def build_actor(self, input_dim):
        """Builds the Actor neural network."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=-1)  # Output a probability distribution
        )

    def build_critic(self, input_dim):
        """Builds the Critic neural network."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def load_model(self):
        """Load only the key data from the saved model."""
        with open(self.model_path, 'rb') as f:
            saved_data = pickle.load(f)
            if saved_data["num_cities"] == self.num_cities:
                self.actor.load_state_dict(saved_data["actor_state_dict"])
                self.critic.load_state_dict(saved_data["critic_state_dict"])
                self.history_best_distance = saved_data["history_best_distance"]
                print("Loaded existing Actor-Critic model.")
            else:
                print(f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_data['num_cities']}. Initializing new model.")

    def save_model(self):
        """Save only the necessary data."""
        saved_data = {
            "num_cities": self.num_cities,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
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
        state_vector = state_to_vector(state, self.num_cities)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)
        action_probs = self.actor(state_tensor)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Random action
        else:
            return torch.argmax(action_probs).item()  # Greedy action

    def update(self, state, action, reward, next_state, done):
        """
        Update the actor and critic using the state-action-reward sequence.
        :param state: Current state (tensor)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (tensor)
        :param done: Whether the episode is done
        """
        state_vector = state_to_vector(state, self.num_cities)
        next_state_vector = state_to_vector(next_state, self.num_cities)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # Compute the value of the state
        value = self.critic(torch.tensor(state_vector, dtype=torch.float32))

        # Compute the target value
        with torch.no_grad():
            next_value = self.critic(torch.tensor(next_state_vector, dtype=torch.float32))
            target = reward_tensor + (1 - done) * self.gamma * next_value

        # Compute critic loss (mean squared error)
        critic_loss = nn.MSELoss()(value, target)

        # Compute actor loss (policy gradient)
        action_probs = self.actor(torch.tensor(state_vector, dtype=torch.float32))
        action_log_prob = torch.log(action_probs[action])
        advantage = target - value
        actor_loss = -action_log_prob * advantage

        # Total loss
        total_loss = actor_loss + critic_loss

        # Update networks
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def train(self, env, num_episodes):
        """
        Train the Actor-Critic agent in the TSP environment
        :param env: TSP environment instance
        :param num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state (now a dictionary)
            total_distance = 0
            done = False
            visited = []
            print(f"{episode}/{num_episodes}  train")
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

                # Update Actor-Critic model
                self.update(state, action, reward, next_state, done)

                # Update state and other variables
                state = next_state
                visited.append(state['current_city'])  # Track visited cities (example)
                total_distance = next_state['total_distance']  # Use next_state's total_distance

            # Save the model if the best distance is improved
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()  # Save only the key data
