from abc import ABC, abstractmethod
import os
import random
from rl.tsp.common.reward_policy import adjust_reward

class BaseAgent(ABC):
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 reward_strategy="negative_distance", model_path=None):
        """
        Base Agent class. All other Agent classes should inherit from this class.

        :param num_cities: Number of cities
        :param num_actions: Number of possible actions
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration rate
        :param reward_strategy: Reward strategy to be used
        :param model_path: Path to save/load the model
        """
        self.num_cities = num_cities
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.reward_strategy = reward_strategy  # Reward strategy
        self.model_path = model_path  # Model save/load path
        self.history_best_distance = float('inf')  # Record of the best distance achieved

        # Initialize the model structure (to be implemented by subclasses)
        self.initialize_model()

        # Load existing model parameters if the model path exists
        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
        else:
            print("Initialized new model.")

    @abstractmethod
    def initialize_model(self):
        """
        Initialize the model. Subclasses must implement this method to set up their specific model architecture.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Load a saved model. Subclasses must implement this method to load their specific model architecture.
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Save the current model. Subclasses must implement this method to save their specific model architecture.
        """
        pass

    @abstractmethod
    def get_greedy_action(self, state):
        """
        Get the greedy action. Subclasses must implement this method to define their specific greedy action selection strategy.

        :param state: Current state (dictionary)
        :return: Greedy action
        """
        pass

    def choose_action(self, state, use_sample=True):
        """
        Choose an action based on the ε-greedy policy.

        :param state: Current state (dictionary)
        :param use_sample: Whether to use exploration strategy
        :return: Selected action
        """
        if use_sample and random.uniform(0, 1) < self.epsilon:
            # Explore: select a random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: select the greedy action
            return self.get_greedy_action(state)

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update the model parameters. Subclasses must implement this method to define their specific update mechanisms.

        :param state: Current state (dictionary)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (dictionary)
        :param done: Whether the episode has ended
        """
        pass

    @abstractmethod
    def train(self, env, num_episodes):
        """
        Train the Agent. Subclasses must implement this method to define their specific training processes.

        :param env: Environment instance
        :param num_episodes: Number of training episodes
        """
        pass

    def adjust_reward(self, reward_params):
        """
        Adjust the reward based on the strategy.

        :param reward_params: Dictionary of reward parameters
        :return: Adjusted reward
        """
        return adjust_reward(reward_params, self.reward_strategy, self.history_best_distance)
