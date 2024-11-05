import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.memory import Memory
from rl.agents.base_agent import Agent


class ActorCriticAgent(Agent):
    def __init__(self, states_size, actions_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 gamma=0.95, lr=0.001, max_memory=2000):
        """
        Initialize the ActorCriticAgent.

        Parameters:
            states_size (int): Dimension of the state space.
            actions_size (int): Number of possible discrete actions.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate for exploration probability.
            gamma (float): Discount factor for future rewards.
            lr (float): Learning rate for the optimizers.
            max_memory (int): Maximum size of the replay memory.
        """
        self.states_size = states_size
        self.actions_size = actions_size
        self.memory = Memory(max_memory=max_memory)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr

        # Initialize the best average reward
        self.best_avg_reward = -float('inf')  # Initialize to negative infinity

        # Define the path to save the model
        self.model_path = "actor_critic_agent.pth"

        # Build the actor and critic models
        self.actor_model, self.critic_model = self.build_models(states_size, actions_size)

        # Define the optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.lr)

        # Load existing model parameters if available
        self.load_model()

    def build_models(self, states_size, actions_size):
        """Build the Actor and Critic neural network models."""
        actor_model = nn.Sequential(
            nn.Linear(states_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, actions_size)  # Outputs action logits
        )

        critic_model = nn.Sequential(
            nn.Linear(states_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1)  # Outputs state value
        )

        # Initialize weights
        for layer in actor_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in critic_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        return actor_model, critic_model

    def save_model(self):
        """Save the current model parameters."""
        torch.save({
            'actor_state_dict': self.actor_model.state_dict(),
            'critic_state_dict': self.critic_model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'best_avg_reward': self.best_avg_reward
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model1(self):
        """Load the model parameters if a saved model exists."""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_model.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            print(f"Loaded model from {self.model_path} with best_avg_reward = {self.best_avg_reward}")
        else:
            print("No saved model found. Starting fresh.")

    import os
    import torch

    def load_model(self):
        """Load the model parameters if a saved model exists and the state_dicts match."""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)

            # Helper function to check state_dict compatibility
            def check_state_dict(current_state_dict, loaded_state_dict, component_name="component"):
                current_keys = set(current_state_dict.keys())
                loaded_keys = set(loaded_state_dict.keys())

                if current_keys != loaded_keys:
                    print(f"[{component_name}] 键不匹配。")
                    missing = current_keys - loaded_keys
                    unexpected = loaded_keys - current_keys
                    if missing:
                        print(f"缺失的键: {missing}")
                    if unexpected:
                        print(f"意外的键: {unexpected}")
                    return False

                for key in current_keys:
                    if current_state_dict[key].shape != loaded_state_dict[key].shape:
                        print(f"[{component_name}] 键 '{key}' 的形状不匹配。")
                        print(f"当前模型形状: {current_state_dict[key].shape}，加载的模型形状: {loaded_state_dict[key].shape}")
                        return False
                return True

            # 检查 actor_model
            if 'actor_state_dict' in checkpoint:
                if not check_state_dict(self.actor_model.state_dict(), checkpoint['actor_state_dict'], "actor_model"):
                    print("Actor 模型的 state_dict 不匹配。放弃加载权重。")
                    return
            else:
                print("Checkpoint 中缺少 'actor_state_dict'。放弃加载权重。")
                return

            # 检查 critic_model
            if 'critic_state_dict' in checkpoint:
                if not check_state_dict(self.critic_model.state_dict(), checkpoint['critic_state_dict'],
                                        "critic_model"):
                    print("Critic 模型的 state_dict 不匹配。放弃加载权重。")
                    return
            else:
                print("Checkpoint 中缺少 'critic_state_dict'。放弃加载权重。")
                return

            # 检查 actor_optimizer
            if 'actor_optimizer_state_dict' in checkpoint:
                # Optimizer 的 state_dict 结构更复杂，通常包括 'state' 和 'param_groups'
                # 这里主要检查 'param_groups' 的结构是否匹配
                current_opt_state = self.actor_optimizer.state_dict()
                loaded_opt_state = checkpoint['actor_optimizer_state_dict']

                if 'param_groups' not in loaded_opt_state or 'state' not in loaded_opt_state:
                    print("Actor 优化器的 state_dict 缺少必要的键。放弃加载优化器权重。")
                    return

                # 检查 'param_groups'
                if len(current_opt_state['param_groups']) != len(loaded_opt_state['param_groups']):
                    print("Actor 优化器的 'param_groups' 长度不匹配。放弃加载优化器权重。")
                    return

                # 可以进一步检查每个 param_group 的参数是否一致（可选）
                # 这里只进行简单的长度检查
            else:
                print("Checkpoint 中缺少 'actor_optimizer_state_dict'。放弃加载优化器权重。")
                return

            # 检查 critic_optimizer
            if 'critic_optimizer_state_dict' in checkpoint:
                current_opt_state = self.critic_optimizer.state_dict()
                loaded_opt_state = checkpoint['critic_optimizer_state_dict']

                if 'param_groups' not in loaded_opt_state or 'state' not in loaded_opt_state:
                    print("Critic 优化器的 state_dict 缺少必要的键。放弃加载优化器权重。")
                    return

                if len(current_opt_state['param_groups']) != len(loaded_opt_state['param_groups']):
                    print("Critic 优化器的 'param_groups' 长度不匹配。放弃加载优化器权重。")
                    return
            else:
                print("Checkpoint 中缺少 'critic_optimizer_state_dict'。放弃加载优化器权重。")
                return

            # 如果所有检查都通过，则加载 state_dict
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_model.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            print(f"成功从 {self.model_path} 加载模型，best_avg_reward = {self.best_avg_reward}")
        else:
            print("未找到保存的模型。开始新的训练。")

    def update_best_reward(self, avg_reward):
        """
        Update the best average reward and save the model if improved.

        Parameters:
            avg_reward (float): The current average reward to compare against the best.
        """
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            self.save_model()
            print(f"New best average reward: {self.best_avg_reward:.2f}. Model saved.")

    def train(self, batch_size=32, n_updates=1000):
        """Train the Actor-Critic agent using a batch of experiences from replay memory."""
        if len(self.memory.cache) < batch_size:
            return
        for _ in range(n_updates):
            # Randomly sample a batch from memory
            batch = random.sample(self.memory.cache, batch_size)
            states, actions, _, rewards, next_states, dones = zip(*batch)

            # Convert to torch tensors
            states = torch.tensor(self.expand_state_vector(np.array(states)), dtype=torch.float32)
            next_states = torch.tensor(self.expand_state_vector(np.array(next_states)), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)  # Shape: (batch_size, 1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)

            # Critic: calculate value of the current states
            values = self.critic_model(states)  # Shape: (batch_size, 1)
            next_values = self.critic_model(next_states)  # Shape: (batch_size, 1)
            target_values = rewards + self.gamma * next_values.detach() * (1 - dones)  # Shape: (batch_size, 1)

            # Calculate critic loss
            critic_loss = F.mse_loss(values, target_values)

            # Actor: calculate advantage
            advantages = target_values - values.detach()  # Shape: (batch_size, 1)
            action_probs = F.softmax(self.actor_model(states), dim=-1)  # Shape: (batch_size, action_size)
            action_log_probs = torch.log(action_probs.gather(1, actions) + 1e-8)  # Shape: (batch_size, 1)

            # Calculate actor loss (policy gradient)
            actor_loss = -action_log_probs * advantages  # Shape: (batch_size, 1)
            actor_loss = actor_loss.mean()  # Scalar

            # Update the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=1.0)  # Clip gradients
            self.critic_optimizer.step()

            # Update the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)  # Clip gradients
            self.actor_optimizer.step()

        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def act(self, state_, test=False):
        """Select an action using the epsilon-greedy policy."""
        state = torch.tensor(self.expand_state_vector(state_), dtype=torch.float32)
        if test:
            with torch.no_grad():
                action_logits = self.actor_model(state)
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, num_samples=1).item()
        else:
            if np.random.rand() > self.epsilon:
                with torch.no_grad():
                    action_logits = self.actor_model(state)
                    action_probs = F.softmax(action_logits, dim=-1)
                    action = torch.multinomial(action_probs, num_samples=1).item()
            else:
                action = np.random.randint(self.actions_size)

        return action, None  # The second value can be used for additional information if needed
