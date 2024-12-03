from rl.tsp.agents.agent import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from confs.path_conf import system_model_dir
from rl.tsp.common.data_processor import state_to_vector


class ActorCriticAgent(BaseAgent):
    def __init__(self, num_cities, num_actions, alpha=0.001, gamma=0.99, epsilon=0.1,
                 reward_strategy="negative_distance", model_path=system_model_dir + "best_actor_critic_model.pkl"):
        super().__init__(num_cities, num_actions, alpha, gamma, epsilon, reward_strategy, model_path)

    def initialize_model(self):
        """Initialize the Actor-Critic model with actor and critic networks."""
        # Define the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize actor and critic networks
        self.actor = self.build_actor(self.num_cities * 2 + 2).to(self.device)
        self.critic = self.build_critic(self.num_cities * 2 + 2).to(self.device)

        # Define optimizers for actor and critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.alpha)

        print("Initialized new Actor-Critic model.")

    def load_model(self):
        """Load the Actor-Critic model from the specified path."""
        with open(self.model_path, 'rb') as f:
            saved_data = pickle.load(f)
            if saved_data["num_cities"] == self.num_cities:
                # Load state dictionaries for actor and critic
                self.actor.load_state_dict(saved_data["actor_state_dict"])
                self.critic.load_state_dict(saved_data["critic_state_dict"])
                self.history_best_distance = saved_data["history_best_distance"]

                # Move models to the appropriate device
                self.actor.to(self.device)
                self.critic.to(self.device)
                print("Loaded existing Actor-Critic model.")
            else:
                print(
                    f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_data['num_cities']}. Initializing new model.")
                self.initialize_model()

    def save_model(self):
        """Save the current state of the Actor-Critic model."""
        saved_data = {
            "num_cities": self.num_cities,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "history_best_distance": self.history_best_distance
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(saved_data, f)
        print(f"Model saved with distance: {self.history_best_distance}")

    def build_actor(self, input_dim):
        """Build the Actor neural network."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=-1)  # Output a probability distribution over actions
        )

    def build_critic(self, input_dim):
        """Build the Critic neural network."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_greedy_action(self, state):
        """Select the action with the highest probability (greedy action)."""
        state_vector = state_to_vector(state, self.num_cities)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        return torch.argmax(action_probs).item()  # Return the index of the highest probability action

    def update(self, state, action, reward, next_state, done):
        """Update the Actor and Critic networks based on the transition."""
        # Convert states to vectors and then to tensors
        state_vector = state_to_vector(state, self.num_cities)
        next_state_vector = state_to_vector(next_state, self.num_cities)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state_vector, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)

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
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def train(self, env, num_episodes):
        """Train the Actor-Critic agent in the given environment."""
        for episode in range(num_episodes):
            state = env.reset()
            total_distance = 0
            print(f"Episode {episode + 1}/{num_episodes} - Training")
            done = False

            while not done:
                # Select an action using the parent class's choose_action method
                action = self.choose_action(state)
                next_state, base_reward, done = env.step(action)

                # Prepare reward parameters for adjustment
                reward_params = {
                    "distance": next_state['step_distance'],
                    "done": done,
                    "total_distance": next_state['total_distance'],
                    "visited": next_state['current_path'],
                    "env": env
                }

                # Adjust the reward based on the strategy
                reward = self.adjust_reward(reward_params)

                # Update the Actor-Critic model with the transition
                self.update(state, action, reward, next_state, done)

                # Update state and track total distance
                state = next_state
                total_distance = next_state['total_distance']

            # Save the model if a new best distance is achieved
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()
