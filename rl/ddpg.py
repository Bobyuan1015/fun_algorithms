import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

# Initialize wandb
wandb.init(project="DDPG-Pendulum", config={"episodes": 100, "max_timesteps": 200})

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a

# Define Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        q = self.l3(q)
        return q

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
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(reward)
            batch_dones.append(done)

        return (
            torch.FloatTensor(batch_states).to(device),
            torch.FloatTensor(batch_next_states).to(device),
            torch.FloatTensor(batch_actions).to(device),
            torch.FloatTensor(batch_rewards).unsqueeze(1).to(device),
            torch.FloatTensor(batch_dones).unsqueeze(1).to(device)
        )

# DDPG algorithm implementation
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, load_model=False):
        # Model paths
        self.actor_model_path = "actor.pth"
        self.critic_model_path = "critic.pth"

        # Actor network and target network
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic network and target network
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Try loading the saved model if exists
        if load_model and os.path.exists(self.actor_model_path) and os.path.exists(self.critic_model_path):
            self.actor.load_state_dict(torch.load(self.actor_model_path))
            self.critic.load_state_dict(torch.load(self.critic_model_path))
            print("Loaded saved model")

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.best_reward = -np.inf

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action

    def train(self, replay_buffer, iterations, batch_size=64, gamma=0.99, tau=0.005):
        for _ in range(iterations):
            # Sample from replay buffer
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)

            # Update Critic network
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()
            current_Q = self.critic(state, action)
            critic_loss = nn.MSELoss()(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update Actor network
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Log losses to wandb
            wandb.log({"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()})

    def save_model(self, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save(self.actor.state_dict(), self.actor_model_path)
            torch.save(self.critic.state_dict(), self.critic_model_path)
            print(f"New best model saved with reward: {reward}")

# Main function
def main():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    ddpg = DDPG(state_dim, action_dim, max_action, load_model=True)
    replay_buffer = ReplayBuffer()
    total_episodes = 100
    max_timesteps = 200
    exploration_noise = 0.1

    for episode in range(total_episodes):
        state = env.reset()[0]
        episode_reward = 0

        for t in range(max_timesteps):
            action = ddpg.select_action(state)
            # Add exploration noise
            action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(
                env.action_space.low, env.action_space.high
            )

            next_state, reward, done, _ ,_ = env.step(action)
            done_bool = float(done) if t + 1 < max_timesteps else 0

            replay_buffer.add((state, next_state, action, reward, done_bool))
            state = next_state
            episode_reward += reward

            if len(replay_buffer.storage) > 1000:
                ddpg.train(replay_buffer, iterations=5)

            if done:
                break

        # Log episode reward to wandb
        wandb.log({"episode_reward": episode_reward})

        # Save best model
        ddpg.save_model(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

if __name__ == "__main__":
    main()
