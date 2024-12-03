import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import random

from confs.path_conf import system_model_dir
from rl.tsp.agents.agent import BaseAgent
from rl.tsp.common.data_processor import state_to_vector

class PPOAgent(BaseAgent):
    def __init__(self, num_cities, num_actions, alpha=0.001, gamma=0.99, epsilon=0.1,
                 reward_strategy="negative_distance", model_path=system_model_dir + "best_ppo_model.pkl",
                 use_shared_network=True):
        """
        Initialize the PPO Agent.

        :param num_cities: Number of cities
        :param num_actions: Number of possible actions
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration rate
        :param reward_strategy: Reward strategy to be used
        :param model_path: Path to save/load the model
        :param use_shared_network: Whether to use a shared network for actor and critic
        """
        self.use_shared_network = use_shared_network  # Whether to use a shared network
        # Call the parent class constructor
        super().__init__(num_cities, num_actions, alpha, gamma, epsilon, reward_strategy, model_path)

    def initialize_model(self):
        """
        Initialize the PPO model, including Actor and Critic networks.
        """
        # Define the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize networks and optimizer
        if self.use_shared_network:
            # Build shared network
            self.shared_network = self.build_shared_network().to(self.device)
            # Build actor and critic heads based on the shared network
            self.actor = self.build_actor_head(self.shared_network).to(self.device)
            self.critic = self.build_critic_head(self.shared_network).to(self.device)
            # Use a single optimizer for the shared network
            self.optimizer = optim.Adam(self.shared_network.parameters(), lr=self.alpha)
        else:
            # Build separate actor and critic networks
            self.actor = self.build_actor(self.num_cities * 2 + 2).to(self.device)
            self.critic = self.build_critic(self.num_cities * 2 + 2).to(self.device)
            # Use separate optimizers for actor and critic
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.alpha)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.alpha)

        print("Initialized new PPO model.")

    def build_shared_network(self):
        """
        Build a shared neural network for both Actor and Critic.

        :return: Shared neural network model
        """
        return nn.Sequential(
            nn.Linear(self.num_cities * 2 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

    def build_actor(self, input_dim):
        """
        Build the Actor neural network.

        :param input_dim: Dimension of the input layer
        :return: Actor neural network model
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=-1)  # Output a probability distribution over actions
        )

    def build_critic(self, input_dim):
        """
        Build the Critic neural network.

        :param input_dim: Dimension of the input layer
        :return: Critic neural network model
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output the state value estimate
        )

    def build_actor_head(self, shared_network):
        """
        Build the Actor head using the shared network.

        :param shared_network: Shared neural network
        :return: Actor head model
        """
        return nn.Sequential(
            shared_network,
            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=-1)  # Output a probability distribution over actions
        )

    def build_critic_head(self, shared_network):
        """
        Build the Critic head using the shared network.

        :param shared_network: Shared neural network
        :return: Critic head model
        """
        return nn.Sequential(
            shared_network,
            nn.Linear(128, 1)  # Output the state value estimate
        )

    def load_model(self):
        """
        Load the PPO model from the specified path.
        """
        with open(self.model_path, 'rb') as f:
            saved_data = pickle.load(f)
            if saved_data["num_cities"] == self.num_cities:
                if self.use_shared_network:
                    # Load shared network and heads
                    self.shared_network.load_state_dict(saved_data["shared_network_state_dict"])
                    self.actor.load_state_dict(saved_data["actor_state_dict"])
                    self.critic.load_state_dict(saved_data["critic_state_dict"])
                else:
                    # Load separate actor and critic networks
                    self.actor.load_state_dict(saved_data["actor_state_dict"])
                    self.critic.load_state_dict(saved_data["critic_state_dict"])
                self.history_best_distance = saved_data["history_best_distance"]
                # Ensure models are on the correct device
                if self.use_shared_network:
                    self.shared_network.to(self.device)
                self.actor.to(self.device)
                self.critic.to(self.device)
                print("Loaded existing PPO model.")
            else:
                print(f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_data['num_cities']}. Initializing new model.")
                self.initialize_model()

    def save_model(self):
        """
        Save the current state of the PPO model to a file.
        """
        saved_data = {
            "num_cities": self.num_cities,
            "history_best_distance": self.history_best_distance
        }
        if self.use_shared_network:
            # Save shared network and heads
            saved_data["shared_network_state_dict"] = self.shared_network.state_dict()
            saved_data["actor_state_dict"] = self.actor.state_dict()
            saved_data["critic_state_dict"] = self.critic.state_dict()
        else:
            # Save separate actor and critic networks
            saved_data["actor_state_dict"] = self.actor.state_dict()
            saved_data["critic_state_dict"] = self.critic.state_dict()
        with open(self.model_path, 'wb') as f:
            pickle.dump(saved_data, f)
        print(f"Model saved with best distance: {self.history_best_distance}")

    def get_greedy_action(self, state):
        """
        Select the action with the highest probability (greedy action).

        For PPO, the greedy action is typically the action with the highest probability.

        :param state: Current state (dictionary)
        :return: Greedy action
        """
        state_vector = state_to_vector(state, self.num_cities)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        return torch.argmax(action_probs).item()  # Return the index of the highest probability action

    def update(self, state, action, reward, next_state, done):
        """
        Update the Actor and Critic using the state-action-reward sequence.

        :param state: Current state (dictionary)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (dictionary)
        :param done: Whether the episode has ended
        """
        state_vector = state_to_vector(state, self.num_cities)
        next_state_vector = state_to_vector(next_state, self.num_cities)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # Convert state vectors to tensors and move to device
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state_vector, dtype=torch.float32, device=self.device)

        # Compute the value of the current state
        value = self.critic(state_tensor)

        # Compute the target value using the reward and the value of the next state
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)
            target = reward_tensor + (1 - done) * self.gamma * next_value

        # Calculate Critic loss (Mean Squared Error)
        critic_loss = nn.MSELoss()(value, target)

        # Calculate Actor loss (Policy Gradient)
        action_probs = self.actor(state_tensor)
        action_log_prob = torch.log(action_probs[action])
        advantage = target - value
        actor_loss = -action_log_prob * advantage.detach()  # Detach advantage to prevent backpropagation through critic

        # Combine losses
        total_loss = actor_loss + critic_loss

        # Backpropagation and optimization
        if self.use_shared_network:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        else:
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

    def train(self, env, num_episodes):
        """
        Train the PPO agent in the TSP environment.

        :param env: Environment instance
        :param num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state (dictionary)
            total_distance = 0
            done = False
            visited = []
            print(f"Episode {episode + 1}/{num_episodes} - Training")
            while not done:
                # Choose an action using the parent class's choose_action method
                action = self.choose_action(state)
                next_state, base_reward, done = env.step(action)  # Get next state, reward, and done flag

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

                # Update the PPO model with the transition
                self.update(state, action, reward, next_state, done)

                # Update state and track total distance
                state = next_state
                visited.append(state['current_city'])  # Track visited cities (example)
                total_distance = next_state['total_distance']  # Use next_state's total_distance

            # Save the model if a new best distance is achieved
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()  # Save only the key data
