# 1. Create the TSP environment. The state consists of all points,
# where visited locations are set to 1 and unvisited locations are set to 0.
# Each environment state is represented as a vector, for example,
# [0, 0, 1, 0, 0, 0] indicates that there are 6 locations,
# the third location has been visited, while the others have not.

# 2. The action corresponds to the index of the maximum value returned
# by the Q-network, allowing the selection of the location with the
# highest Q-value among the 6 points.

# 3. The reward is calculated as the cumulative distance based on the
# order of visited locations, summing the distances between each pair of points.

# 4. The Q-value for visited locations is set to negative infinity,
# preventing revisits to those points.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random, pickle, os
from utils.logger import logging
import matplotlib.pyplot as plt
from numpy import mean
from itertools import permutations


class TSPEnvironment:
    def __init__(self, num_locations=10):
        self.num_locations = num_locations
        self.file_path = 'tsp_data.pkl'

        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                # Check if the number of locations matches
                if data['num_locations'] == self.num_locations:
                    self.locations = data['locations']
                    self.distances = data['distances']
                    print("Loaded data from file.")
                else:
                    print("Number of locations mismatch. Generating new data.")
                    self.locations = self.generate_random_locations()
                    self.distances = self.calculate_distances()
                    self.save_data()
        else:
            print("No existing data file found. Generating new data.")
            self.locations = self.generate_random_locations()
            self.distances = self.calculate_distances()
            self.save_data()
        self.locations = self.generate_random_locations()
        self.distances = self.calculate_distances()
        self.reset()

    def save_data(self):
        data = {
            'num_locations': self.num_locations,
            'locations': self.locations,
            'distances': self.distances
        }
        with open(self.file_path, 'wb') as f:
            pickle.dump(data, f)

    def calculate_path_distance(self, index_array):
        # Clean the index_array #[0, 3, 7, 8, 8, 1, 8, 10, 2, 5, 4, 6, 9, 0]  这个路径要清洗
        cleaned_array = []
        for i, elem in enumerate(index_array):
            if elem not in cleaned_array or (i == len(index_array) - 1 and elem == 0):
                cleaned_array.append(elem)

        # Calculate the total distance
        total_distance = 0
        if len(cleaned_array) != len(self.locations) + 1:
            print(f'-----生成的路径不对，cleaned_array={cleaned_array}  原始路径={index_array}')
        for i in range(len(cleaned_array) - 1):
            total_distance += self.distances[cleaned_array[i], cleaned_array[i + 1]]

        return total_distance

    def calculate_total_distance_from_first(self):
        n = self.distances.shape[0]  # 获取坐标的数量
        # 固定第一个点，生成剩余点的排列
        remaining_points = range(1, n)  # 剩余的点从1到n-1
        all_paths = permutations(remaining_points)  # 生成所有剩余点的排列
        path_distances = []  # 存储每条路径的总距离

        for path in all_paths:
            # 添加固定的第一个点
            full_path = (0,) + path + (0,)  # 形成完整路径，起点是0，终点也返回0
            total_distance = 0
            # 计算路径的总距离
            for i in range(len(full_path) - 1):
                total_distance += self.distances[full_path[i], full_path[i + 1]]  # 计算相邻点之间的距离
            path_distances.append((full_path, total_distance))  # 存储路径及其总距离

        # 按照总距离从小到大排序
        sorted_path_distances = sorted(path_distances, key=lambda x: x[1], reverse=False)

        for i in range(len(sorted_path_distances)):
            path, distance = sorted_path_distances[i]
            # 将计算结果拼接
            sorted_path_distances[i] = (path, distance)  # 只保留路径和距离

        return sorted_path_distances

    def generate_random_locations(self):
        return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(self.num_locations)]

    def calculate_distances(self):
        self.distances = np.zeros((self.num_locations, self.num_locations))
        for i in range(self.num_locations):
            for j in range(self.num_locations):
                self.distances[i][j] = np.sqrt((self.locations[i][0] - self.locations[j][0]) ** 2 +
                                               (self.locations[i][1] - self.locations[j][1]) ** 2)
        self.all_paths = self.calculate_total_distance_from_first()
        self.best_path = self.all_paths[0]
        return self.distances

    def reset(self):
        self.current_location = 0
        self.unvisited_locations = set(range(1, self.num_locations))
        self.visited_locations = [self.current_location]
        self.total_distance = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros(self.num_locations)
        state[self.visited_locations] = 1
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def step(self, action):
        big_penalty = -100  # -np.inf
        if action not in self.unvisited_locations:
            return self.get_state(), big_penalty, False

        distance_to_next = self.distances[self.current_location][action]
        self.unvisited_locations.remove(action)
        self.visited_locations.append(action)
        self.total_distance += distance_to_next
        self.current_location = action

        done = len(self.unvisited_locations) == 0
        if done:  # 要加上 终点到起点的距离
            distance_to_next += self.distances[action][0]

        reward = -distance_to_next
        return self.get_state(), reward, done

    def render(self, distance, path, true_distance, save=False):
        x, y = zip(*self.locations)
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, color="red", label="Locations")
        plt.plot(x, y, "o")

        # Add arrows to show visit order and display location values

        for i in range(1, len(path)):
            start_idx = path[i - 1]
            end_idx = path[i]
            plt.annotate(
                '', xy=self.locations[end_idx], xytext=self.locations[start_idx],
                arrowprops=dict(arrowstyle="->", color="blue", lw=1)
            )
            # Display the coordinates for each location
            plt.text(self.locations[start_idx][0], self.locations[start_idx][1],
                     f"{path[i - 1]}", fontsize=12, ha="right")

        # Display the last point's coordinates as well
        plt.text(self.locations[path[-1]][0], self.locations[path[-1]][1],
                 f"{path[-1]}", fontsize=12, ha="right")

        # Label the start and end points
        plt.annotate("Start", xy=self.locations[path[0]],
                     xytext=(self.locations[path[0]][0] - 0.2,
                             self.locations[path[0]][1] - 0.2),
                     color="green", weight="bold", fontsize=9)
        plt.annotate("End", xy=self.locations[path[-1]],
                     xytext=(self.locations[path[-1]][0] + 0.2,
                             self.locations[path[-1]][1] + 0.2),
                     color="red", weight="bold", fontsize=9)

        plt.legend()
        plt.title(f"TSP Path, distance={distance} true={true_distance}\n {path}", fontsize=9)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        # Save the figure if requested
        if save:
            plt.savefig("TSP_path.png")
        else:
            plt.show()

def select_action(model,state,epsilon,method):

    if method =="qlearning&sample":
        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(env.unvisited_locations))
        else:
            with torch.no_grad():
                q_values = model(state)
                q_values[:, list(env.visited_locations)] = -float('inf')
                action = torch.argmax(q_values).item()
    elif method =="qlearning":
        with torch.no_grad():
            q_values = model(state)
            q_values[:, list(env.visited_locations)] = -float('inf')
            action = torch.argmax(q_values).item()
    elif method == "qtable&sample":
        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(env.unvisited_locations))
        else:
            q_table = model
            action = np.argmax(q_table[state, :])
    elif method == "q-table":
        q_table = model
        action = np.argmax(q_table[state, :])
    else:
        action = random.choice(list(env.unvisited_locations))

        print(f'no random')
    return action


def play(env, model, experience=None, epsilon=0.3, num_play_episodes=50, show=False, test=False):
    """收集初始经验，用于离线训练"""
    total_rewards = []
    performance = []
    # print(f'paly epsilon={epsilon}')
    global best_performance
    for _ in range(num_play_episodes):
        next_state = state = env.reset()
        done = False
        episode_rewards = []
        episode_actions = [0]
        episode_states = []
        count_step = 0
        episode_states.append(state)
        use_sample = False
        while not done:
            count_step += 1
            if test:
                select_action(model,state,epsilon,method='')
                with torch.no_grad():
                    q_values = model(state)
                    q_values[:, list(env.visited_locations)] = -float('inf')
                    action = torch.argmax(q_values).item()

            else:
                if random.uniform(0, 1) < epsilon or use_sample:
                    action = random.choice(list(env.unvisited_locations))
                    use_sample = False
                else:
                    with torch.no_grad():
                        q_values = model(state)
                        if test:
                            q_values[:, list(env.visited_locations)] = -float('inf')
                        action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            if next_state.equal(state) and count_step != 1:
                if test:
                    logging.info(f'选的有问题 episode={_} {state} a={action}  episode_actions={episode_actions}  count_step={count_step}')
                use_sample = True

            episode_actions.append(action)
            episode_states.append(next_state)

            experience.append((state, action, reward, next_state, done))
            state = next_state

            episode_rewards.append(reward)
        episode_actions.append(0)
        test_result = env.calculate_path_distance(episode_actions)
        if test:
            if test_result <= best_performance[0]:
                best_performance[0] = test_result
                best_performance[1] = episode_actions
                env.render(distance=test_result, path=episode_actions, true_distance=env.best_path[1],
                           save=True)
                logging.info(
                    f'{_} episode, epsilon={epsilon}  distance={test_result}/最优={env.best_path}, a={episode_actions},  r={["{:.4f}".format(r) for r in episode_rewards]}')
        performance.append((episode_actions, test_result))
        total_rewards.append(sum(episode_rewards))
        # logging.info(f'{_} episode, distance={test_result}/最优={env.best_path}, a={episode_actions},  r={["{:.4f}".format(r) for r in episode_rewards]}')
    # logging.info(f'最优路径={best_performance} 真实最优={env.best_path} R={["{:.4f}".format(r) for r in total_rewards]}   ')
    # if show or test:
    #
    #     plt.plot(total_rewards)
    #     plt.title("play Rewards")
    #     plt.show()
    return performance, total_rewards


class DQN(nn.Module):
    def __init__(self, num_locations):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_locations, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, num_locations)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def custom_loss(q_values, targets, visited_masks):
    # 确保 q_values 和 targets 具有相同的形状：[batch_size, num_locations]
    mse_loss = F.mse_loss(q_values, targets)

    # 初始化惩罚项
    penalty = 0.0
    batch_size = q_values.size(0)

    # 遍历批次中的每个样本
    for i in range(batch_size):
        # 使用 `masked_select` 选择已访问和未访问位置的 Q 值
        visited_q_values = q_values[i].masked_select(visited_masks[i])
        unvisited_q_values = q_values[i].masked_select(~visited_masks[i])

        # 如果存在未访问的 Q 值，则计算最小未访问 Q 值的惩罚项
        if unvisited_q_values.numel() > 0:
            min_unvisited_q = unvisited_q_values.min()
            penalty += torch.sum(F.relu(visited_q_values - min_unvisited_q) ** 2)

    # 合并 MSE 损失和平均惩罚项
    total_loss = mse_loss + penalty / batch_size
    return total_loss


def train_dqn(env, model, experience, epoch=100, gamma=0.95, batch_size=32, epsilon=1.0, epsilon_decay=0.996,
              epsilon_min=0.01):
    optimizer = optim.Adam(model.parameters(), lr=0.008)
    loss_ = []
    train_mask = True
    pool_size = batch_size*10
    for episode in range(epoch):
        # print(f'train epsilon={epsilon}')

        experience=[]
        play(env, model, experience, epsilon=epsilon, num_play_episodes=pool_size, show=False)

        if len(experience) >= pool_size:
            for i in range(5000):
                minibatch = random.sample(experience, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = torch.cat(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards).float()
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones).float()

                q_values = model(states).gather(1, actions)
                if train_mask:
                    next_q_values = model(next_states).max(1)[0].detach()
                else:
                    pred_q = model(next_states)
                    # 创建一个掩码，将 next_states 中值为1的位置标记出来
                    mask = (next_states == 1)  # 布尔掩码
                    # 将 pred_q 中对应 mask 的位置赋值为负无穷
                    pred_q[mask] = float('-inf')
                    # 计算每一行的最大Q值
                    next_q_values = pred_q.max(1)[0].detach()

                targets = rewards + gamma * next_q_values * (1 - dones)

                if train_mask:
                    targets = targets.unsqueeze(1)
                    visited_masks = torch.zeros_like(states, dtype=torch.bool)
                    for idx, s in enumerate(states):
                        visited_indices = torch.where(s == 1)[0]
                        visited_masks[idx, visited_indices] = True
                    loss = custom_loss(q_values, targets, visited_masks)
                else:
                    loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    logging.info(
                        f"Episode {episode}/{epoch}  train {i} ,testing------------> play  epsilon={epsilon}, result:"
                        f"{play(env, model, experience, epsilon=epsilon, num_play_episodes=1, test=True)}  best_path={env.best_path}")
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        # Store reward for visualization
        total_reward = sum(rewards)
        loss_.append(loss)
        # if episode % logging.info_interval == 0:
        # logging.info(f"Episode {episode}/{epoch}, Loss: {loss.item():.4f}, Total Reward: {total_reward:.2f}")
        # logging.info(f"Episode {episode}/{epoch}, Loss: {loss.item():.4f}")
        if episode % 2 == 0:
            num_play_episodes = 2
            logging.info(
                f"Episode {episode}/{epoch} ,testing------------> play {num_play_episodes}  epsilon={epsilon}, result:"
                f"{play(env, model, experience, epsilon=epsilon, num_play_episodes=num_play_episodes, test=True)}  best_path={env.best_path}")

    plt.plot([item.detach().cpu().numpy() for item in loss_])
    plt.title("Training loss")
    # plt.show()
    plt.savefig("Training_los.png")


def test_tsp_dqn(env, model):
    state = env.reset()
    done = False
    path = [0]

    while not done:
        with torch.no_grad():
            q_values = model(state)
            q_values[:, list(env.visited_locations)] = -float('inf')
            action = torch.argmax(q_values).item()
            path.append(action)

        state, _, done = env.step(action)

    path.append(0)  # Return to start
    logging.info("Optimal Path:", path)
    env.render()


if __name__ == "__main__":
    num_locations = 11
    env = TSPEnvironment(num_locations)
    model = DQN(num_locations)
    experience = []
    best_performance = [1000000, []]  # 路径距离，路径顺序
    # 使用 play 收集初始经验
    train_dqn(env, model, experience)

    # # 测试 DQN
    # logging.info("Testing DQN model...")
    # test_tsp_dqn(env, model)

