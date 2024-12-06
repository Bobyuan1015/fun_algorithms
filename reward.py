import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict
import itertools
from tqdm import tqdm
from utils.logger import Logger
from utils.logger import Logger

log = Logger("reward", '1').get_logger()

class QLearningTSP:
    def __init__(self, cities, state_space_config="step", alpha=0.1, gamma=0.9, epsilon=0.2,
                 episodes=1000, save_q_every=100):
        """
        Initialize the Q-Learning TSP solver.

        Parameters:
        - cities: 2D numpy array representing distances between cities.
        - state_space_config: "step" for current city only, "visits" for visited cities.
        - alpha: Learning rate.
        - gamma: Discount factor.
        - epsilon: Exploration rate.
        - episodes: Number of training episodes.
        - save_q_every: Interval of episodes to save Q-table snapshots.
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.state_space_config = state_space_config
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.save_q_every = save_q_every

        # Initialize states and Q-table
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

        # Data recording
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
        """
        获取当前城市，即访问过的最后一个城市。
        """
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

            if episode % 100 == 0:  # Record frequencies every 100 steps
                frequencies = [count / (episode + 1) for count in self.action_counts.values()]
                self.action_frequencies.append(frequencies)
                strategy_matrix = self.get_strategy_matrix()
                self.iteration_strategies.append(strategy_matrix)

        log.info(f"Training converged at episode: {stable_episode}")

    def save_q_table(self, episode):
        os.makedirs("q_tables", exist_ok=True)
        if self.state_space_config == "step":
            file_path = f"q_tables/q_table_simple_{episode + 1}.npy"
            np.save(file_path, self.q_table)
        elif self.state_space_config == "visits":
            # 保存为字典形式
            file_path = f"q_tables/q_table_path_{episode + 1}.npy"
            # 由于 defaultdict 不能直接保存，需要转换为普通字典
            q_table_dict = {state: q for state, q in self.q_table.items()}
            np.save(file_path, q_table_dict)
