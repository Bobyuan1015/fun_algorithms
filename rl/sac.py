import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import wandb
import os
from torch.distributions import Categorical, Normal

from confs.path_conf import system_rl_model_dir

# Initialize wandb for experiment tracking
wandb.init(project="SAC-Implementation")


class ContinuousActor(nn.Module):
    """
    Actor network for continuous action spaces.

    This network outputs the mean and log standard deviation for each action dimension.
    Actions are sampled from a Gaussian distribution parameterized by these outputs.
    The sampled actions are then passed through a Tanh function to ensure they lie within
    the valid action range of the environment.
    """

    def __init__(self, env, hidden_dim=256):
        """
        Initializes the ContinuousActor network.

        Args:
            env (gym.Env): The environment with a continuous action space.
            hidden_dim (int, optional): Number of units in the hidden layers. Defaults to 256.
        """
        super(ContinuousActor, self).__init__()
        self.action_dim = env.action_space.shape[0]
        self.fc1 = nn.Linear(env.observation_space.shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            action (torch.Tensor): The sampled action after applying Tanh.
            log_prob (torch.Tensor): The log probability of the action.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # Clamp to maintain numerical stability
        std = log_std.exp()

        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterization trick for backpropagation
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Correction for Tanh squashing
        log_prob -= (2 * (np.log(2) - action - nn.functional.softplus(-2 * action))).sum(-1, keepdim=True)

        action = torch.tanh(action)
        return action, log_prob


class DiscreteActor(nn.Module):
    """
    Actor network for discrete action spaces.

    This network outputs logits for each possible discrete action.
    Actions are sampled from a categorical distribution based on these logits.
    """

    def __init__(self, env, hidden_dim=256):
        """
        Initializes the DiscreteActor network.

        Args:
            env (gym.Env): The environment with a discrete action space.
            hidden_dim (int, optional): Number of units in the hidden layers. Defaults to 256.
        """
        super(DiscreteActor, self).__init__()
        self.action_dim = env.action_space.n
        self.fc1 = nn.Linear(env.observation_space.shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            action (torch.Tensor): The sampled discrete action.
            log_prob (torch.Tensor): The log probability of the action.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)  # Maintain consistent shape
        return action, log_prob


class Critic(nn.Module):
    """
    Critic network for estimating Q-values.

    This network estimates the Q-value for a given state-action pair.
    Two separate critic networks are used in SAC to mitigate overestimation bias.
    """

    def __init__(self, env, hidden_dim=256):
        """
        Initializes the Critic network.

        Args:
            env (gym.Env): The environment.
            hidden_dim (int, optional): Number of units in the hidden layers. Defaults to 256.
        """
        super(Critic, self).__init__()
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            action_dim = 1
        else:
            raise ValueError("Unsupported action space type.")

        self.input_dim = env.observation_space.shape[0] + action_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.
            action (torch.Tensor): The input action.

        Returns:
            q_value (torch.Tensor): The estimated Q-value.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=state.device, dtype=torch.float32)

        if action.dim() == 0:
            action = action.unsqueeze(0)
        # elif action.dim() == 1:
        #     action = action.unsqueeze(1)

        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent.

    Implements the SAC algorithm for both continuous and discrete action spaces.
    This includes network initialization, action selection, training loop, and model
    saving/loading mechanisms. SAC is an off-policy actor-critic algorithm that
    optimizes a stochastic policy in an entropy-regularized framework to balance
    exploration and exploitation.
    """

    def __init__(self, env_name='Pendulum-v1', action_type='continuous', gamma=0.99, tau=0.005, lr=3e-4, alpha=0.2):
        """
        Initializes the SACAgent.

        Args:
            env_name (str, optional): Name of the Gym environment. Defaults to 'Pendulum-v1'.
            action_type (str, optional): Type of action space ('continuous' or 'discrete'). Defaults to 'continuous'.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            tau (float, optional): Soft update coefficient for target networks. Defaults to 0.005.
            lr (float, optional): Learning rate for optimizers. Defaults to 3e-4.
            alpha (float, optional): Entropy regularization coefficient. Defaults to 0.2.
        """
        self.env = gym.make(env_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.action_type = action_type

        # Initialize networks and load existing models if available
        self.create_networks()
        self.load_model()

    def create_networks(self):
        """
        Creates the actor and critic networks based on the action type.

        For continuous actions, a ContinuousActor is used.
        For discrete actions, a DiscreteActor is used.
        Additionally, two Critic networks and their corresponding target networks are initialized.
        """
        if self.action_type == 'continuous':
            self.actor = ContinuousActor(self.env).to(self.device)
        else:
            self.actor = DiscreteActor(self.env).to(self.device)

        self.critic_1 = Critic(self.env).to(self.device)
        self.critic_2 = Critic(self.env).to(self.device)
        self.target_critic_1 = Critic(self.env).to(self.device)
        self.target_critic_2 = Critic(self.env).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr)

        # Initialize target networks with the same weights as the original critics
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def load_model(self):
        """
        Loads saved model weights if they exist.

        Model filenames are determined based on the action type to ensure that
        continuous and discrete models are stored separately.
        """
        actor_file = "sac_actor_discrete.pkl" if self.action_type == "discrete" else "sac_actor_continuous.pkl"
        critic_1_file = "sac_critic_1_discrete.pkl" if self.action_type == "discrete" else "sac_critic_1_continuous.pkl"
        critic_2_file = "sac_critic_2_discrete.pkl" if self.action_type == "discrete" else "sac_critic_2_continuous.pkl"

        actor_path = os.path.join(system_rl_model_dir, actor_file)
        critic_1_path = os.path.join(system_rl_model_dir, critic_1_file)
        critic_2_path = os.path.join(system_rl_model_dir, critic_2_file)

        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
            print(f"Loaded existing {self.action_type} actor model from {actor_path}.")

        if os.path.exists(critic_1_path):
            self.critic_1.load_state_dict(torch.load(critic_1_path))
            print(f"Loaded existing critic model 1 from {critic_1_path}.")

        if os.path.exists(critic_2_path):
            self.critic_2.load_state_dict(torch.load(critic_2_path))
            print(f"Loaded existing critic model 2 from {critic_2_path}.")

    def select_action(self, state):
        """
        Selects an action based on the current policy.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            action (np.ndarray or int): The selected action.
            log_prob (float): The log probability of the selected action.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor(state)
        return action.cpu().numpy()[0], log_prob.cpu().item()

    def train(self, num_episodes=1000, max_timesteps=200):
        """
        Executes the training loop for the SAC agent.

        For each episode, the agent interacts with the environment, collects experiences,
        updates the networks, and logs progress. The best performing model based on
        cumulative rewards is saved.

        Args:
            num_episodes (int, optional): Number of training episodes. Defaults to 1000.
            max_timesteps (int, optional): Maximum timesteps per episode. Defaults to 200.
        """
        best_reward = -np.inf
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            for t in range(max_timesteps):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update(state, action, reward, next_state, log_prob, done)

                state = next_state
                episode_reward += reward
                if done:
                    break

            # Log episode reward to wandb and console
            wandb.log({"episode_reward": episode_reward})
            print(f"Episode: {episode}, Reward: {episode_reward}")

            # Save the model if current episode's reward is the best so far
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model()

    def save_model(self):
        """
        Saves the actor and critic models to disk.

        The actor and critic model filenames are determined based on the action type to ensure
        that continuous and discrete models are stored separately.
        """
        actor_file = "sac_actor_discrete.pkl" if self.action_type == "discrete" else "sac_actor_continuous.pkl"
        critic_1_file = "sac_critic_1_discrete.pkl" if self.action_type == "discrete" else "sac_critic_1_continuous.pkl"
        critic_2_file = "sac_critic_2_discrete.pkl" if self.action_type == "discrete" else "sac_critic_2_continuous.pkl"

        actor_path = os.path.join(system_rl_model_dir, actor_file)
        critic_1_path = os.path.join(system_rl_model_dir, critic_1_file)
        critic_2_path = os.path.join(system_rl_model_dir, critic_2_file)

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)
        print(f"Saved best {self.action_type} actor model to {actor_path}.")
        print(f"Saved best {self.action_type} critic model 1 to {critic_1_path}.")
        print(f"Saved best {self.action_type} critic model 2 to {critic_2_path}.")

    def update(self, state, action, reward, next_state, log_prob, done):
        """
        Updates the actor and critic networks based on the collected experience.

        This involves computing the loss for each network, performing backpropagation,
        and updating the target networks using a soft update rule.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray or int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            log_prob (float): Log probability of the action.
            done (bool): Whether the episode has ended.
        """
        # Convert inputs to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # Compute target Q-values using target networks
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * target_q

        # Estimate current Q-values using critic networks
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        # Compute critic losses as Mean Squared Error between estimated and target Q-values
        critic_1_loss = nn.functional.mse_loss(q1, q_target)
        critic_2_loss = nn.functional.mse_loss(q2, q_target)

        # Update Critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Update Critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Compute actor loss
        new_action, log_prob = self.actor(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks to slowly track the main networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Log losses to wandb and console
        wandb.log({
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item()
        })
        print(
            f"Critic 1 Loss: {critic_1_loss.item():.4f}, "
            f"Critic 2 Loss: {critic_2_loss.item():.4f}, "
            f"Actor Loss: {actor_loss.item():.4f}"
        )


if __name__ == "__main__":
    """
    Entry point for training the SAC agent.

    Selects the environment based on the action type and initializes the SACAgent.
    Begins the training process for a specified number of episodes and timesteps.
    """
    env_type = "continuous"  # Set to "continuous" or "discrete" to choose environment type

    if env_type == "continuous":
        env_name = "Pendulum-v1"
    elif env_type == "discrete":
        env_name = "CartPole-v1"
    else:
        raise ValueError("Invalid environment type. Choose either 'continuous' or 'discrete'.")

    agent = SACAgent(env_name=env_name, action_type=env_type)

    num_episodes = 500
    max_timesteps = 200

    print(f"Starting training on {env_name} with {env_type} action space...")
    agent.train(num_episodes=num_episodes, max_timesteps=max_timesteps)
    print("Training completed.")
