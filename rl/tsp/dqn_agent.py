import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.memory import Memory
from rl.agents.base_agent import Agent


class DQNAgent(Agent):
    def __init__(self, states_size, actions_size, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.9995,
                 gamma=0.8, lr=0.002, max_memory=1000, target_update_freq=10):
        """
        Initialize the DQNAgent.

        Parameters:
            states_size (int): Dimension of the state space.
            actions_size (int): Number of possible actions.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate for exploration probability.
            gamma (float): Discount factor for future rewards.
            lr (float): Learning rate for the optimizer.
            max_memory (int): Maximum size of the replay memory.
            target_update_freq (int): Frequency (in steps) to update the target network.
        """
        self.states_size = states_size
        self.actions_size = actions_size
        self.memory = Memory(max_memory=max_memory)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.target_update_freq = target_update_freq
        self.step_count = 0  # To track when to update the target network

        # Initialize the best average reward
        self.best_avg_reward = -float('inf')  # Initialize to negative infinity

        # Define the path to save the model
        self.model_path = "dqn_agent.pth"

        # Build the model and optimizer
        self.model = self.build_model(states_size, actions_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Build the target network and copy weights from the main network
        self.target_model = self.build_model(states_size, actions_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set target network to evaluation mode

        # Load existing model parameters if available
        self.load_model()

    def build_model(self, states_size, actions_size):
        """Build the DQN model."""
        model = nn.Sequential(
            nn.Linear(states_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, actions_size)
        )
        return model

    def save_model(self):
        """Save the current model parameters."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_avg_reward': self.best_avg_reward
        }, self.model_path)
        print(f"Model saved to {self.model_path}  best_avg_reward={self.best_avg_reward} ")

    def load_model1(self):
        """Load the model parameters if a saved model exists."""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

            # 辅助函数：检查 state_dict 的兼容性
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
                        print(
                            f"当前模型形状: {current_state_dict[key].shape}，加载的模型形状: {loaded_state_dict[key].shape}")
                        return False
                return True

            # 检查主模型
            if 'model_state_dict' in checkpoint:
                if not check_state_dict(self.model.state_dict(), checkpoint['model_state_dict'], "model"):
                    print("主模型的 state_dict 不匹配。放弃加载权重。")
                    return
            else:
                print("Checkpoint 中缺少 'model_state_dict'。放弃加载权重。")
                return

            # 检查目标模型
            if 'target_model_state_dict' in checkpoint:
                if not check_state_dict(self.target_model.state_dict(), checkpoint['target_model_state_dict'],
                                        "target_model"):
                    print("目标模型的 state_dict 不匹配。放弃加载权重。")
                    return
            else:
                print("Checkpoint 中缺少 'target_model_state_dict'。放弃加载权重。")
                return

            # 检查优化器
            if 'optimizer_state_dict' in checkpoint:
                current_opt_state = self.optimizer.state_dict()
                loaded_opt_state = checkpoint['optimizer_state_dict']

                # 检查 'param_groups' 是否存在
                if 'param_groups' not in loaded_opt_state or 'state' not in loaded_opt_state:
                    print("优化器的 state_dict 缺少必要的键。放弃加载优化器权重。")
                    return

                # 检查 'param_groups' 的长度是否一致
                if len(current_opt_state['param_groups']) != len(loaded_opt_state['param_groups']):
                    print("优化器的 'param_groups' 长度不匹配。放弃加载优化器权重。")
                    return

                # 可选：进一步检查每个 param_group 的具体参数（如学习率等）
                # 这里只进行简单的长度检查
            else:
                print("Checkpoint 中缺少 'optimizer_state_dict'。放弃加载优化器权重。")
                return

            # 如果所有检查都通过，则加载 state_dict
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
                print(f"成功从 {self.model_path} 加载模型，best_avg_reward = {self.best_avg_reward}")
            except Exception as e:
                print(f"加载 state_dict 时发生错误: {e}")
                print("放弃加载权重。")
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

    def train(self, batch_size=32, n_updates=100):
        """Train the DQN agent using a batch of experiences from replay memory."""
        if len(self.memory.cache) < batch_size:
            return
        for _ in range(n_updates):

            # Randomly sample a batch from memory

            batch = random.sample(self.memory.cache, batch_size)

            # Unzip the batch
            states, actions, _, rewards, next_states, dones = zip(*batch)

            # Convert to torch tensors
            states = torch.tensor(self.expand_state_vector(np.array(states)), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)  # Shape: (batch_size, 1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)
            next_states = torch.tensor(self.expand_state_vector(np.array(next_states)), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)

            # Compute current Q values
            q_values = self.model(states).gather(1, actions)  # Shape: (batch_size, 1)

            # Compute target Q values using the target network
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)  # Shape: (batch_size, 1)
                target = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute loss
            loss = F.mse_loss(q_values, target)

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update the target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("Target network updated.")

    def act(self, state, test=False):
        """Select an action using the epsilon-greedy policy."""
        state = torch.tensor(self.expand_state_vector(state), dtype=torch.float32)

        if test:
            with torch.no_grad():
                q_values = self.model(state)
                action = torch.argmax(q_values, dim=1).item()
        else:
            if np.random.rand() > self.epsilon:
                with torch.no_grad():
                    q_values = self.model(state)
                    action = torch.argmax(q_values, dim=1).item()
            else:
                action = np.random.randint(self.actions_size)

        return action, None  # The second value can be used for additional information if needed
