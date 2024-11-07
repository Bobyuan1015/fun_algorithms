import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

# Initialize wandb
wandb.init(project="DuelingDQN-CartPole", config={"episodes": 500, "max_timesteps": 200})

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Dueling DQN network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # Common feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        feature = self.feature(state)
        advantage = self.advantage(feature)
        value = self.value(feature)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# Replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr % self.max_size] = data
            self.ptr += 1
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_dones.append(done)

        return (
            torch.FloatTensor(batch_states).to(device),
            torch.FloatTensor(batch_next_states).to(device),
            torch.LongTensor(batch_actions).to(device),
            torch.FloatTensor(batch_rewards).to(device),
            torch.FloatTensor(batch_dones).to(device)
        )


# Dueling DQN algorithm implementation
class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 load_model=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.best_reward = -np.inf

        # Model paths
        self.model_path = "dueling_dqn_model.pth"

        # Define Dueling DQN networks
        self.q_network = DuelingDQN(state_dim, action_dim).to(device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Try loading the saved model if exists
        if load_model and os.path.exists(self.model_path):
            self.q_network.load_state_dict(torch.load(self.model_path))
            print("Loaded saved model")

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def train(self, replay_buffer, batch_size=64):
        # Sample from replay buffer
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)

        # Compute Q target
        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0]
            target_q = reward + (1 - done) * self.gamma * next_q_values

        # Compute Q values for actions taken
        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Calculate loss
        loss = nn.MSELoss()(q_values, target_q)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log loss to wandb
        wandb.log({"loss": loss.item()})

        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save(self.q_network.state_dict(), self.model_path)
            print(f"New best model saved with reward: {reward}")


# Main function
def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQNAgent(state_dim, action_dim, load_model=True)
    replay_buffer = ReplayBuffer()
    total_episodes = 500
    max_timesteps = 200

    for episode in range(total_episodes):
        state = env.reset()[0]
        episode_reward = 0

        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _ ,_ = env.step(action)
            replay_buffer.add((state, next_state, action, reward, float(done)))
            state = next_state
            episode_reward += reward

            # Start training after storing enough samples
            if len(replay_buffer.storage) > 1000:
                agent.train(replay_buffer)

            if done:
                break

        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()

        # Log episode reward to wandb
        wandb.log({"episode_reward": episode_reward})

        # Save best model
        agent.save_model(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")


if __name__ == "__main__":
    main()
