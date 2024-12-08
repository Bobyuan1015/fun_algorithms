from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime
from confs.path_conf import system_experience_dir
from rl.tsp.common.reward_policy import adjust_reward

import pickle
from confs.path_conf import tsp_agents_data_dir
from rl.tsp.agents.agent import BaseAgent
from rl.tsp.agents.agent import BaseAgent对应源码如下：
class BaseAgent(ABC):
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 reward_strategy="negative_distance", model_path=None):
        self.num_cities = num_cities
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_strategy = reward_strategy
        self.model_path = model_path
        self.history_best_distance = float('inf')
        self.initialize_model()
        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
        else:
            print("Initialized new model.")

    @abstractmethod
    def initialize_model(self):
        pass
    @abstractmethod
    def load_model(self):
        pass
    @abstractmethod
    def save_model(self):
        pass
    @abstractmethod
    def get_greedy_action(self, state):
        pass
    def choose_action(self, state, use_sample=True):
        if use_sample and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.get_greedy_action(state)
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass
    @abstractmethod
    def train(self, env, num_episodes):
        pass
    def adjust_reward(self, reward_params):
        return adjust_reward(reward_params, self.reward_strategy, self.history_best_distance)


class QLearningAgent(BaseAgent):
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.5,
                 reward_strategy="negative_distance", model_path=tsp_agents_data_dir + "best_q-table.pkl"):
        date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = f'{system_experience_dir}/{date_string}/'
        os.makedirs(self.base_dir , exist_ok=True)

        self.model_path = self.base_dir  +'model_QLearningAgent.pkl'
        super().__init__(num_cities, num_actions, alpha, gamma, epsilon, reward_strategy, model_path)
    def initialize_model(self):
        self.q_table = np.zeros((self.num_cities, self.num_actions))  # Initialize Q-table
        print("Initialized new Q-learning model.")
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                if saved_data["num_cities"] == self.num_cities:
                    self.q_table = saved_data["q_table"]
                    self.history_best_distance = saved_data["history_best_distance"]
                    print("Loaded existing Q-learning model.")
                else:
                    print(f"Model file found, but num_cities mismatch: expected {self.num_cities}, got {saved_data['num_cities']}. Initializing new model.")
                    self.initialize_model()
        except FileNotFoundError:
            print(f"Model path {self.model_path} does not exist. Initializing new Q-learning model.")
            self.initialize_model()
        except Exception as e:
            print(f"Error loading model: {e}. Initializing new Q-learning model.")
            self.initialize_model()
    def save_model(self):
        saved_data = {
            "num_cities": self.num_cities,
            "q_table": self.q_table,
            "history_best_distance": self.history_best_distance
        }
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(saved_data, f)
            print(f"Model saved with distance: {self.history_best_distance}")
        except Exception as e:
            print(f"Error saving model: {e}")
    def get_greedy_action(self, state):
        state_index = self.state_to_index(state)
        return np.argmax(self.q_table[state_index])
    def state_to_index(self, state):
        return state['current_city']
    def update(self, state, action, reward, next_state, done):
        current_index = self.state_to_index(state)
        next_index = self.state_to_index(next_state)
        best_next_action = np.argmax(self.q_table[next_index])
        td_target = reward + self.gamma * self.q_table[next_index, best_next_action] * (1 - done)
        td_error = td_target - self.q_table[current_index, action]
        self.q_table[current_index, action] += self.alpha * td_error
    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            total_distance = 0
            done = False
            visited = []
            print(f"Episode {episode + 1}/{num_episodes} - Training")
            while not done:
                action = self.choose_action(state)
                next_state, base_reward, done = env.step(action)
                reward_params = {
                    "distance": next_state['step_distance'],
                    "done": done,
                    "total_distance": next_state['total_distance'],
                    "visited": next_state['current_path'],
                    "env": env
                }
                reward = self.adjust_reward(reward_params)
                self.update(state, action, reward, next_state, done)
                state = next_state
                visited.append(state['current_city'])
                total_distance = env.total_distance

            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()

import numpy as np
import os
from collections import defaultdict
import itertools
from utils.logger import Logger
log = Logger("grid_search_q_learning", '1').get_logger()
class QLearningTSP:
    def __init__(self, cities, state_space_config="step", alpha=0.1, gamma=0.9, epsilon=0.2,
                 episodes=1000, save_q_every=100):
        self.cities = cities
        self.num_cities = len(cities)
        self.state_space_config = state_space_config
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.save_q_every = save_q_every
        self.states = list(range(self.num_cities))
        if self.state_space_config == "step":
            # Simple state: current city
            self.q_table = np.zeros((self.num_cities, self.num_cities))
            self.strategy_matrix = np.zeros((self.num_cities, self.num_cities))
        elif self.state_space_config == "visits":
            # Detailed state: visited cities as a tuple
            self.q_table = defaultdict(lambda: np.zeros(self.num_cities))
            self.strategy_matrix = dict()
        else:
            raise ValueError("Invalid state_space_config. Choose 'simple' or 'path'.")
        self.episode_rewards = []
        self.average_rewards = []
        self.cumulative_rewards = []
        self.q_table_snapshots = []
        self.q_value_changes = defaultdict(list) if self.state_space_config == "visits" else {state: [] for state in self.states}
        self.action_frequency = np.zeros((self.num_cities, self.num_cities))
        self.action_frequencies = []
        self.iteration_strategies = []
        self.action_counts = {a:0 for a in self.states}
    def update_strategy(self, state, action):
        key  = tuple([str(i) for i in state])
        if key not in self.strategy_matrix:
            self.strategy_matrix[key] = {action:0}
        elif action not in self.strategy_matrix[key]:
            self.strategy_matrix[key][action] = 0
        self.strategy_matrix[key][action] += 1
    def get_strategy_matrix(self):
        if self.state_space_config == "step":
            return self.strategy_matrix
        elif self.state_space_config == "visits":
            strategy_matrix = np.zeros((self.num_cities, self.num_cities))
            for state, actions in self.strategy_matrix.items():
                current_city = int(state[-1])
                for action, count in actions.items():
                    strategy_matrix[current_city][action] = count
            return strategy_matrix
    def _get_current_city(self, visited):
        return visited[-1]

    def _choose_action(self, visited):
        current_city = self._get_current_city(visited)
        available_actions = [a for a in range(self.num_cities) if a not in visited]
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            if self.state_space_config == "step":
                q_values = self.q_table[current_city]
            elif self.state_space_config == "visits":
                state = tuple(visited)
                q_values = self.q_table[state]
            available_q_values = {a: q_values[a] for a in available_actions}
            return max(available_q_values, key=available_q_values.get)
    def _update_q_table(self, visited, action, reward, next_visited):
        if self.state_space_config == "step":
            state = self._get_current_city(visited)
            next_state = self._get_current_city(next_visited)
            current_q = self.q_table[state][action]
            next_max_q = np.max(self.q_table[next_state])
            updated_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
            self.q_table[state][action] = updated_q
            self.q_value_changes[state].append(updated_q)
        elif self.state_space_config == "visits":
            state = tuple(visited)
            next_state = tuple(next_visited)
            current_q = self.q_table[state][action]
            next_max_q = np.max(self.q_table[next_state])
            updated_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
            self.q_table[state][action] = updated_q
            self.q_value_changes[state].append(updated_q)
    def update_counts(self, action):
        self.action_counts[int(action)] += 1
    def train(self):
        stable_episode = None
        for episode in range(self.episodes):
            start_city = np.random.choice(self.states)
            visited = [start_city]
            episode_reward = 0
            while len(visited) < self.num_cities:
                action = self._choose_action(visited)
                self.update_counts(action)
                reward = -self.cities[self._get_current_city(visited)][action]
                episode_reward += reward
                next_visited = visited + [action]
                self._update_q_table(visited, action, reward, next_visited)
                self.action_frequency[self._get_current_city(visited)][action] += 1
                self.update_strategy(visited, action)
                visited = next_visited
            self.episode_rewards.append(episode_reward)
            self.cumulative_rewards.append(sum(self.episode_rewards))
            self.average_rewards.append(np.mean(self.episode_rewards))
            if len(self.episode_rewards) > 10:
                reward_diff = np.abs(self.episode_rewards[-1] - np.mean(self.episode_rewards[-10:]))
                if reward_diff < 1e-3 and stable_episode is None:
                    stable_episode = episode
            if (episode + 1) % self.save_q_every == 0:
                if self.state_space_config == "step":
                    self.q_table_snapshots.append(self.q_table.copy())
                elif self.state_space_config == "visits":
                    sampled_states = list(itertools.islice(self.q_table.items(), 100))  # 采样前100个状态
                    sampled_q_table = {state: q.copy() for state, q in sampled_states}
                    self.q_table_snapshots.append(sampled_q_table)
                self.save_q_table(episode)
            if episode % 100 == 0:
                frequencies = [count / (episode + 1) for count in self.action_counts.values()]
                self.action_frequencies.append(frequencies)
                strategy_matrix = self.get_strategy_matrix()
                self.iteration_strategies.append(strategy_matrix)

        log.info(f"Training converged at episode: {stable_episode}")

    def save_results(self):
        os.makedirs("results", exist_ok=True)
        pd.DataFrame({"Episode": range(1, len(self.episode_rewards) + 1),
                      "Reward": self.episode_rewards}).to_csv("results/episode_rewards.csv", index=False)
        pd.DataFrame({"Episode": range(1, len(self.cumulative_rewards) + 1),
                      "Cumulative Reward": self.cumulative_rewards}).to_csv("results/cumulative_rewards.csv",
                                                                            index=False)
        pd.DataFrame({"Episode": range(1, len(self.average_rewards) + 1),
                      "Average Reward": self.average_rewards}).to_csv("results/average_rewards.csv", index=False)

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards, label="Episode Rewards")
        plt.plot(range(1, len(self.average_rewards) + 1), self.average_rewards, label="Average Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.title("Training Progress")
        plt.grid(True)
        plt.show()

    def plot_q_value_changes(self):
        plt.figure(figsize=(12, 6))
        if self.state_space_config == "step":
            for state, q_values in self.q_value_changes.items():
                plt.plot(q_values, label=f"State {state}")
        elif self.state_space_config == "visits":
            # 绘制部分状态的 Q-value 变化
            sampled_states = list(itertools.islice(self.q_value_changes.items(), 10))  # 采样前10个状态
            for state, q_values in sampled_states:
                plt.plot(q_values, label=f"State {state}")
        plt.xlabel("Updates")
        plt.ylabel("Q-value")
        plt.legend()
        plt.title("Q-value Changes Over Time")
        plt.show()

    def plot_policy_evolution(self):
        if not self.q_table_snapshots:
            print("No Q-table snapshots to plot.")
            return

        for i, q_table in enumerate(self.q_table_snapshots):
            plt.figure(figsize=(10, 8))
            if self.state_space_config == "step":
                plt.imshow(q_table, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Q-table at Episode {i * self.save_q_every + self.save_q_every}")
            elif self.state_space_config == "visits":
                # 对于高维 Q-table，只绘制采样状态
                sampled_states = list(itertools.islice(q_table.items(), 100))
                sampled_q = np.array([q for _, q in sampled_states])
                plt.imshow(sampled_q, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Q-table Snapshot at Episode {i * self.save_q_every + self.save_q_every} (Sampled States)")
            plt.xlabel("Action")
            plt.ylabel("State")
            plt.show()

    def plot_action_frequencies(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.action_frequency, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title("Action Frequency Heatmap")
        plt.xlabel("Action")
        plt.ylabel("State")
        plt.show()

    def plot_q_value_trends(self):
        if self.state_space_config == "step":
            x, y = np.meshgrid(range(self.num_cities), range(self.num_cities))
            q_values = self.q_table
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, q_values, cmap='viridis')
            ax.set_title("Q-value Trends")
            ax.set_xlabel("State")
            ax.set_ylabel("Action")
            ax.set_zlabel("Q-value")
            plt.show()
        elif self.state_space_config == "visits":
            # 聚合 Q-values 进行可视化
            # 示例：对采样状态的 Q-values 取平均
            sampled_q_tables = self.q_table_snapshots
            if not sampled_q_tables:
                print("No Q-table snapshots available for plotting.")
                return
            # 取最后一个快照
            last_snapshot = sampled_q_tables[-1]
            # 按动作聚合 Q-values
            aggregate_q = np.zeros(self.num_cities)
            count = 0
            for q_values in last_snapshot.values():
                aggregate_q += q_values
                count += 1
            aggregate_q /= count if count > 0 else 1
            plt.figure(figsize=(10, 6))
            plt.bar(range(self.num_cities), aggregate_q, color='skyblue')
            plt.xlabel("Action")
            plt.ylabel("Average Q-value")
            plt.title("Average Q-value per Action (Aggregated)")
            plt.show()

    def get_policy(self):
        policy = {}
        if self.state_space_config == "step":
            for state in self.states:
                best_action = np.argmax(self.q_table[state])
                policy[state] = best_action
        elif self.state_space_config == "visits":
            for state, q_values in self.q_table.items():
                best_action = np.argmax(q_values)
                policy[state] = best_action
        return policy

    def plot_stategy(self):
        plt.figure(figsize=(15, 6))
        num_plots = len(self.iteration_strategies)
        for i, strategy in enumerate(self.iteration_strategies):
            plt.subplot(1, num_plots, i + 1)
            cax = plt.imshow(strategy, cmap='hot', interpolation='nearest')
            plt.title(f'Iteration {i * 100}')
            plt.xticks(ticks=np.arange(self.num_cities), labels=np.arange(self.num_cities))
            plt.yticks(ticks=np.arange(self.num_cities), labels=np.arange(self.num_cities))
            plt.xlabel('Destination City')
            plt.ylabel('Current City')
            plt.colorbar(cax, fraction=0.046, pad=0.04)  # 添加颜色条
        plt.tight_layout()
        plt.show()

你是一个python写深度学习、强化学习的高手
要求把QLearningTSP的严格代码适配进QLearningAgent中，要求不省略功能，特别是区分state创建q-table的逻辑 要求保留QLearningTS中的两种情况。
QLearningAgent里面特性：
1.有专门计算reward的功能，不需要使用QLearningTSP的-self.cities[self._get_current_city(visited)][action]来获取reward：
reward_params = {
    "distance": next_state['step_distance'],  # Distance for the current step
    "done": done,
    "total_distance": next_state['total_distance'],
    "visited": next_state['current_path'],
    "env": env
}
reward = self.adjust_reward(reward_params)
2.模型保存成pkl的后缀，所以类变量不能使用lambda初始化，比如defaultdict(lambda: defaultdict(int))类似这种；保存模型需要增加state_space_config，对应加载模型的时候需要匹配state_space_config。
3.QLearningAgent要求适配所有QLearningTSP的plot函数，绘图的时候图保存到self.base_dir目录的下（不要嵌套result子文件夹），并且要求单独绘制图(不要多张图放到一起)。保存后，show图。另外绘图的时候，颜色不要使用黑色，可以参照yolov5绘图选择的颜色
4.适配QLearningTSP的strategy_matrix，在train过程中赋值 和plot_strategy 绘制 要求不报错，特别是key、value的值要兼容, 容易出错函数plot_strategy，update_strategy函数；
5.要对应save_results函数
4.train函数中total_distance = float('inf') 为初始值 而不是0

最后只需要给出QLearningAgent的完整代码即可，不需要BaseAgent，但是要from rl.tsp.agents.agent import BaseAgent