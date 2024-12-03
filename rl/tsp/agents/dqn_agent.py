import torch
import torch.nn as nn
import torch.optim as optim
from confs.path_conf import system_model_dir
from rl.tsp.agents.agent import BaseAgent
from rl.tsp.common.data_processor import state_to_vector


class DQNAgent(BaseAgent):
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 model_path=system_model_dir + "best_dqn_model.pkl", reward_strategy="negative_distance"):
        """
        Initialize the DQN Agent.

        :param num_cities: Number of cities
        :param num_actions: Number of possible actions
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration rate
        :param model_path: Path to save/load the model
        :param reward_strategy: Reward strategy to be used
        """
        super().__init__(num_cities, num_actions, alpha, gamma, epsilon, reward_strategy, model_path)

    def initialize_model(self):
        """
        Initialize the DQN model, including the Q-network and optimizer.
        """
        # Define the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Define the Q-network
        self.q_network = self._build_model(self.num_cities * 2 + 2).to(self.device)

        # Define the optimizer for the Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

        # Define the loss function (Mean Squared Error)
        self.loss_fn = nn.MSELoss()

        print("Initialized new DQN model.")

    def load_model(self):
        """
        Load the DQN model from the specified path.
        """
        # Load the saved agent state
        saved_agent = torch.load(self.model_path, map_location=self.device)

        # Check if the number of cities matches
        if saved_agent['num_cities'] == self.num_cities:
            # Load the Q-network state dictionary
            self.q_network.load_state_dict(saved_agent['q_network_state_dict'])
            self.history_best_distance = saved_agent['history_best_distance']
            print("Loaded existing DQN model.")
        else:
            print(
                f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_agent['num_cities']}. Initializing new model.")
            self.initialize_model()

    def save_model(self):
        """
        Save the current state of the DQN model.
        """
        torch.save({
            'num_cities': self.num_cities,
            'q_network_state_dict': self.q_network.state_dict(),
            'history_best_distance': self.history_best_distance
        }, self.model_path)
        print(f"Model saved with distance: {self.history_best_distance}")

    def _build_model(self, input_dim):
        """
        Build a simple fully connected neural network to approximate Q-values.

        :param input_dim: Dimension of the input layer
        :return: Constructed neural network model
        """
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        return model.to(self.device)

    def get_greedy_action(self, state):
        """
        Select the action with the highest Q-value (greedy action).

        :param state: Current state (dictionary)
        :return: Greedy action
        """
        state_vector = state_to_vector(state, self.num_cities)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()  # Return the index of the highest Q-value action

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update the Q-value using the Bellman equation.

        :param state: Current state (dictionary)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (dictionary)
        :param done: Whether the episode has ended
        """
        # Convert states to vectors and then to tensors
        state_vector = state_to_vector(state, self.num_cities)
        next_state_vector = state_to_vector(next_state, self.num_cities)

        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Compute current Q-values
        q_values = self.q_network(state_tensor)

        # Compute Q-values for the next state
        next_q_values = self.q_network(next_state_tensor)

        # Select the best action for the next state
        best_next_action = torch.argmax(next_q_values)

        # Compute the target Q-value using the Bellman equation
        td_target = reward + self.gamma * next_q_values[0, best_next_action] * (1 - done)

        # Compute the loss between current Q-value and target Q-value
        loss = self.loss_fn(q_values[0, action], td_target)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-network based on the transition.

        :param state: Current state (dictionary)
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state (dictionary)
        :param done: Whether the episode has ended
        """
        self.update_q_value(state, action, reward, next_state, done)

    def train(self, env, num_episodes):
        """
        Train the DQN agent in the given environment.

        :param env: Environment instance
        :param num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state
            done = False
            total_distance = 0

            print(f"Episode {episode + 1}/{num_episodes} - Training")
            while not done:
                # Choose an action using the parent class's choose_action method
                action = self.choose_action(state)

                # Take action in the environment
                next_state, reward, done = env.step(action)
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

                # Update the Q-network with the transition
                self.update(state, action, reward, next_state, done)

                # Update state and track total distance
                state = next_state
                total_distance = next_state['total_distance']

            # Save the model if a new best distance is achieved
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()
