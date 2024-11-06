import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from confs.path_conf import system_model_dir
from rl.tsp.base_agent import Agent
from rl.tsp.memory import Memory

USE_SHARED_Q_V_NETS = True
USE_TOTAL_LOSS = False


class PPOAgent(Agent):
    def __init__(self, states_size, actions_size, epsilon_clip=0.2, gamma=0.99, lam=0.95, lr=0.001, entropy_coef=0.01,
                 value_loss_coef=0.5, epsilon=0.):

        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon_clip = epsilon_clip
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon  # PPO 属于策略优化算法，它与基于值函数的算法（如 DQN）有所不同。所以 PPO 中动作选择的基本机制，以及为什么不使用 epsilon-greedy 采样。
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.use_total_loss = USE_TOTAL_LOSS
        self.use_shared_network = USE_SHARED_Q_V_NETS
        # 最佳奖励初始化
        self.best_avg_reward = -float('inf')  # Initialize best average reward

        # Define model saving path
        self.model_path = f"{system_model_dir}ppo_agent.pth"

        if self.use_shared_network:
            self.policy_value_network = self.build_shared_model(states_size, actions_size)
            self.optimizer = optim.Adam(self.policy_value_network.parameters(), lr=self.lr)
        else:
            self.policy_network = self.build_policy_model(states_size, actions_size)
            self.value_network = self.build_value_model(states_size)
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)

        self.memory = Memory(max_memory=2000)

        # 检查并加载本地已有的模型
        self.load_model()

    def build_shared_model(self, states_size, actions_size):
        model = nn.Sequential(
            nn.Linear(states_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, actions_size + 1)  # actions_size for policy, 1 for value
        )
        return model

    def build_policy_model(self, states_size, actions_size):
        model = nn.Sequential(
            nn.Linear(states_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, actions_size)
        )
        return model

    def build_value_model(self, states_size):
        model = nn.Sequential(
            nn.Linear(states_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return model

    def save_model(self):
        """ Save the current model parameters """
        if self.use_shared_network:
            torch.save({
                'policy_value_network_state_dict': self.policy_value_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_avg_reward': self.best_avg_reward
            }, self.model_path)
            print(f"Model saved to {self.model_path}")
        else:
            torch.save({
                'policy_network_state_dict': self.policy_network.state_dict(),
                'value_network_state_dict': self.value_network.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                'best_avg_reward': self.best_avg_reward
            }, self.model_path)
            print(f"Model saved to {self.model_path}")

    def load_model1(self):
        """ Load the model parameters if the saved model exists """
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            if self.use_shared_network:
                self.policy_value_network.load_state_dict(checkpoint['policy_value_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
                self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            print(f"Loaded model from {self.model_path} with best_avg_reward = {self.best_avg_reward}")
        else:
            print("No saved model found. Starting fresh.")

    def load_model(self):
        """Load the model parameters if the saved model exists and matches the current architecture."""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)

            # Helper function to check state_dict compatibility
            def check_state_dict(current_state_dict, loaded_state_dict, component_name="component"):
                current_keys = set(current_state_dict.keys())
                loaded_keys = set(loaded_state_dict.keys())

                if current_keys != loaded_keys:
                    print(f"[{component_name}] Keys do not match.")
                    missing_keys = current_keys - loaded_keys
                    unexpected_keys = loaded_keys - current_keys
                    if missing_keys:
                        print(f"  Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"  Unexpected keys: {unexpected_keys}")
                    return False

                for key in current_keys:
                    if current_state_dict[key].shape != loaded_state_dict[key].shape:
                        print(f"[{component_name}] Shape mismatch for key '{key}'.")
                        print(
                            f"  Current shape: {current_state_dict[key].shape} | Loaded shape: {loaded_state_dict[key].shape}")
                        return False
                return True

            # Load parameters based on whether a shared network is used
            if self.use_shared_network:
                # Check policy_value_network
                if 'policy_value_network_state_dict' in checkpoint:
                    if check_state_dict(self.policy_value_network.state_dict(),
                                        checkpoint['policy_value_network_state_dict'],
                                        "policy_value_network"):
                        # Load state_dict if checks pass
                        self.policy_value_network.load_state_dict(checkpoint['policy_value_network_state_dict'])
                        print("Successfully loaded policy_value_network state_dict.")
                    else:
                        print("Failed to load policy_value_network state_dict due to mismatched keys or shapes.")
                        return
                else:
                    print("Checkpoint missing 'policy_value_network_state_dict'. Skipping loading.")
                    return

                # Check optimizer
                if 'optimizer_state_dict' in checkpoint:
                    loaded_optimizer_state = checkpoint['optimizer_state_dict']
                    current_optimizer_state = self.optimizer.state_dict()

                    # Simple check: compare param_groups length
                    if len(current_optimizer_state['param_groups']) == len(loaded_optimizer_state['param_groups']):
                        self.optimizer.load_state_dict(loaded_optimizer_state)
                        print("Successfully loaded optimizer state_dict.")
                    else:
                        print("Optimizer param_groups length mismatch. Skipping optimizer loading.")
                        return
                else:
                    print("Checkpoint missing 'optimizer_state_dict'. Skipping optimizer loading.")
                    return
            else:
                # Check policy_network
                if 'policy_network_state_dict' in checkpoint:
                    if check_state_dict(self.policy_network.state_dict(),
                                        checkpoint['policy_network_state_dict'],
                                        "policy_network"):
                        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
                        print("Successfully loaded policy_network state_dict.")
                    else:
                        print("Failed to load policy_network state_dict due to mismatched keys or shapes.")
                        return
                else:
                    print("Checkpoint missing 'policy_network_state_dict'. Skipping loading.")
                    return

                # Check value_network
                if 'value_network_state_dict' in checkpoint:
                    if check_state_dict(self.value_network.state_dict(),
                                        checkpoint['value_network_state_dict'],
                                        "value_network"):
                        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
                        print("Successfully loaded value_network state_dict.")
                    else:
                        print("Failed to load value_network state_dict due to mismatched keys or shapes.")
                        return
                else:
                    print("Checkpoint missing 'value_network_state_dict'. Skipping loading.")
                    return

                # Check policy_optimizer
                if 'policy_optimizer_state_dict' in checkpoint:
                    loaded_policy_optimizer_state = checkpoint['policy_optimizer_state_dict']
                    current_policy_optimizer_state = self.policy_optimizer.state_dict()

                    if len(current_policy_optimizer_state['param_groups']) == len(
                            loaded_policy_optimizer_state['param_groups']):
                        self.policy_optimizer.load_state_dict(loaded_policy_optimizer_state)
                        print("Successfully loaded policy_optimizer state_dict.")
                    else:
                        print("Policy optimizer param_groups length mismatch. Skipping optimizer loading.")
                        return
                else:
                    print("Checkpoint missing 'policy_optimizer_state_dict'. Skipping policy optimizer loading.")
                    return

                # Check value_optimizer
                if 'value_optimizer_state_dict' in checkpoint:
                    loaded_value_optimizer_state = checkpoint['value_optimizer_state_dict']
                    current_value_optimizer_state = self.value_optimizer.state_dict()

                    if len(current_value_optimizer_state['param_groups']) == len(
                            loaded_value_optimizer_state['param_groups']):
                        self.value_optimizer.load_state_dict(loaded_value_optimizer_state)
                        print("Successfully loaded value_optimizer state_dict.")
                    else:
                        print("Value optimizer param_groups length mismatch. Skipping optimizer loading.")
                        return
                else:
                    print("Checkpoint missing 'value_optimizer_state_dict'. Skipping value optimizer loading.")
                    return

            # Load best_avg_reward
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            print(f"Loaded model from {self.model_path} with best_avg_reward = {self.best_avg_reward}")
        else:
            print("No saved model found. Starting fresh.")

    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        """ GAE计算，使用done标志处理每个episode的结束 """
        gae = 0
        advantages = []
        for i in reversed(range(len(rewards))):
            if dones[i]:  # 如果当前时间步是episode的终止状态
                next_value = 0  # 终止状态后的下一步价值为0
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return advantages

    def compute_loss(self, states, actions, old_log_probs, returns, advantages):
        """ 损失计算，基于策略损失、价值损失、熵奖励 """
        if self.use_shared_network:
            logits_values = self.policy_value_network(states)
            logits, values = logits_values[:, :-1], logits_values[:, -1]
        else:
            logits = self.policy_network(states)
            values = self.value_network(states).squeeze()

        # 策略分布
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 计算新旧策略的概率比 r_t
        ratio = torch.exp(new_log_probs - old_log_probs)  # r_t = exp(new - old)

        # 剪辑策略损失
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # 价值损失
        value_loss = F.mse_loss(values, returns)

        # 总损失
        if self.use_total_loss:
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        else:
            loss = policy_loss

        return loss, policy_loss, value_loss, entropy

    def act(self, state,test=False):
        """ 根据当前策略选择动作 """
        state = torch.tensor(self.expand_state_vector(state), dtype=torch.float32)

        if self.use_shared_network:
            logits_values = self.policy_value_network(state)
            logits = logits_values[:, :-1]  # 策略输出是 logits
        else:
            logits = self.policy_network(state)

        # 创建策略分布a1:0.1    a2:0.3     a3:0.6
        dist = torch.distributions.Categorical(logits=logits)  # 来表示动作的概率分布，然后从该分布中采样动作。

        # 从策略分布中采样动作
        action = dist.sample().item()

        # 返回动作及其对应的对数概率，用于后续训练
        log_prob = dist.log_prob(torch.tensor(action))

        return action, log_prob


    def train(self, batch_size=32, epochs=10):
        """ 基于完整batch进行训练，支持多个episode和不完整的episode，不能打乱顺序 """
        if len(self.memory.cache) < batch_size:
            return

        # 从 Memory 中提取批次数据，可能来自多个 episode，保持顺序
        batch = self.memory.cache
        states, actions, old_log_probs, rewards, next_states, dones = zip(*batch)

        # 转换为 torch 张量
        states = torch.tensor(self.expand_state_vector(np.array(states)), dtype=torch.float32)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(old_log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(self.expand_state_vector(np.array(next_states)), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 计算当前的状态值
        with torch.no_grad():
            if self.use_shared_network:
                values = self.policy_value_network(states)[:, -1]
                next_values = self.policy_value_network(next_states)[:, -1]
            else:
                values = self.value_network(states).squeeze()
                next_values = self.value_network(next_states).squeeze()

        # 使用 GAE 计算优势，注意处理不完整的 episode 和 done 标志
        advantages = self.compute_gae(rewards=rewards, values=values, next_value=next_values[-1], dones=dones,
                                      gamma=self.gamma, lam=self.lam)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # 计算目标回报
        target = rewards + (1 - dones) * self.gamma * next_values

        # 打乱数据顺序（在 GAE 计算完成之后）
        indices = np.arange(len(states))
        np.random.shuffle(indices)

        states = states[indices]
        actions = actions[indices]
        old_log_probs = old_log_probs[indices]
        advantages = advantages[indices]
        target = target[indices]

        # 多次更新（K epochs）
        for _ in range(epochs):
            # 计算损失
            loss, policy_loss, value_loss, entropy = self.compute_loss(states, actions, old_log_probs, target,
                                                                       advantages)

            if self.use_shared_network:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        # 清空 memory 缓存
        self.memory.empty_cache()

    def update_best_reward(self, avg_reward):
        """ Update the best average reward and save the model if it's improved """
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            self.save_model()
            print(f"New best average reward: {self.best_avg_reward:.2f}. Model saved.")

