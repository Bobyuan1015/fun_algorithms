import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
import pickle
import os
import matplotlib.pyplot as plt

from confs.path_conf import system_ex_model_dir


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)


# 生成随机城市坐标
def generate_cities(num_cities=20, seed=42):
    np.random.seed(seed)
    cities = np.random.rand(num_cities, 2) * 100  # 100x100的平面
    return cities


# 层次聚类
def hierarchical_clustering(cities, n_clusters=4):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(cities)
    return labels


# 保存结果
def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


# 简单的TSP环境
class TSPEnvironment:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.reset()

    def reset(self):
        self.current_city = 0
        self.visited = [False] * self.num_cities
        self.visited[0] = True
        self.path = [0]
        self.total_distance = 0
        return self.get_state()

    def step(self, action):
        if self.visited[action]:
            reward = -1  # 惩罚重复访问
            done = False
            return self.get_state(), reward, done
        distance = np.linalg.norm(self.cities[self.current_city] - self.cities[action])
        self.total_distance += distance
        self.current_city = action
        self.visited[action] = True
        self.path.append(action)
        done = all(self.visited)
        if done:
            # 回到起点
            distance = np.linalg.norm(self.cities[self.current_city] - self.cities[0])
            self.total_distance += distance
            self.path.append(0)
            reward = -self.total_distance
        else:
            reward = -distance
        return self.get_state(), reward, done

    def get_state(self):
        return np.array(self.visited + [self.current_city], dtype=np.float32)


# DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out


# 经验回放
from collections import deque


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-3, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory()
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, available_actions):
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * \
                       np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state)
                q_values = q_values.cpu().numpy()
                # Mask invalid actions
                q_values_filtered = [q_values[a] if a in available_actions else -np.inf for a in range(len(q_values))]
                return np.argmax(q_values_filtered)

    def optimize(self, batch_size=64, target_update=10):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # Compute loss
        loss = nn.MSELoss()(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 训练DQN
def train_dqn(env, agent, num_episodes=500, target_update=10, verbose=True):
    best_model_path = 'best_dqn_model.pth'
    best_total_distance = float('inf')
    training_results = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            available_actions = [i for i, visited in enumerate(env.visited) if not visited]
            action = agent.select_action(state, available_actions)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.optimize()
        # Update target network
        if episode % target_update == 0:
            agent.update_target_network()
        # 检查是否为最佳模型
        if env.total_distance < best_total_distance:
            best_total_distance = env.total_distance
            torch.save(agent.policy_net.state_dict(), best_model_path)
        # 记录训练结果
        training_results.append({
            'episode': episode,
            'total_distance': env.total_distance,
            'total_reward': total_reward,
            'epsilon': agent.epsilon
        })
        # 打印训练进度
        if verbose and (episode % 10 == 0 or episode == 1):
            print(
                f"Episode {episode}/{num_episodes} | Total Distance: {env.total_distance:.2f} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
    return training_results, best_model_path


# 主流程
def main():
    # 参数设置
    NUM_CITIES = 20
    SEED = 42
    N_CLUSTERS = 4
    NUM_EPISODES = 500
    TARGET_UPDATE = 10

    # 生成城市
    cities = generate_cities(num_cities=NUM_CITIES, seed=SEED)

    # 层次聚类
    labels = hierarchical_clustering(cities, n_clusters=N_CLUSTERS)

    # 统计聚类信息
    clustering_stats = {}
    for cluster_id in range(N_CLUSTERS):
        cluster_size = np.sum(labels == cluster_id)
        clustering_stats[cluster_id] = {
            'num_cities': cluster_size,
            'city_indices': np.where(labels == cluster_id)[0].tolist()
        }
    # 保存聚类结果和统计信息
    save_pickle(labels, system_ex_model_dir+'clustering_results.pkl')
    save_pickle(clustering_stats, system_ex_model_dir+'clustering_stats.pkl')
    print("Clustering results saved to 'clustering_results.pkl'")
    print("Clustering statistics saved to 'clustering_stats.pkl'")

    # 为每个簇训练DQN
    all_results = {}
    for cluster_id in range(N_CLUSTERS):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_cities = cities[cluster_indices]
        print(f"\nTraining DQN for Cluster {cluster_id + 1} with {len(cluster_indices)} cities.")

        # 初始化环境和代理
        env = TSPEnvironment(cluster_cities)
        state_size = len(env.get_state())
        action_size = len(cluster_cities)
        agent = DQNAgent(state_size, action_size)

        # 训练
        training_results, best_model_path = train_dqn(env, agent, num_episodes=NUM_EPISODES,
                                                      target_update=TARGET_UPDATE, verbose=True)
        # 统计强化学习结果
        best_distance = min([res['total_distance'] for res in training_results])
        average_distance = np.mean([res['total_distance'] for res in training_results])
        all_results[cluster_id] = {
            'training_results': training_results,
            'best_model_path': best_model_path,
            'best_distance': best_distance,
            'average_distance': average_distance
        }
        print(f"Best Total Distance for Cluster {cluster_id + 1}: {best_distance:.2f}")
        print(f"Average Total Distance for Cluster {cluster_id + 1}: {average_distance:.2f}")

    # 保存所有结果
    save_pickle(all_results, system_ex_model_dir+'rl_results.pkl')
    print("\nReinforcement Learning results saved to 'rl_results.pkl'")

    # 可视化训练结果（可选）
    plt.figure(figsize=(12, 8))
    for cluster_id, result in all_results.items():
        distances = [res['total_distance'] for res in result['training_results']]
        plt.plot(distances, label=f'Cluster {cluster_id + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Total Distance')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(system_ex_model_dir+'training_progress.png')
    plt.show()
    print("Training progress plot saved to 'training_progress.png'")

    # 返回统计信息
    return {
        'clustering_stats': clustering_stats,
        'rl_stats': all_results
    }


if __name__ == "__main__":
    stats = main()
    # 保存统计信息为JSON格式（可选）
    import json

    with open(system_ex_model_dir+'clustering_stats.json', 'w') as f:
        json.dump(stats['clustering_stats'], f, indent=4)
    with open(system_ex_model_dir+'rl_stats.json', 'w') as f:
        # 将numpy数据类型转换为原生Python类型
        rl_stats_serializable = {}
        for cluster_id, data in stats['rl_stats'].items():
            rl_stats_serializable[cluster_id] = {
                'training_results': data['training_results'],
                'best_model_path': data['best_model_path'],
                'best_distance': data['best_distance'],
                'average_distance': data['average_distance']
            }
            # 处理可能的非序列化数据
            for key in ['training_results']:
                for episode_data in rl_stats_serializable[cluster_id][key]:
                    for k, v in episode_data.items():
                        if isinstance(v, np.float32) or isinstance(v, np.float64):
                            episode_data[k] = float(v)
        json.dump(rl_stats_serializable, f, indent=4)
    print("Clustering and Reinforcement Learning statistics saved to 'clustering_stats.json' and 'rl_stats.json'")

