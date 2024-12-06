import os
import random
import pickle
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools

from matplotlib import pyplot as plt

from rl.tsp.agents.agent import BaseAgent
from rl.tsp.common.reward_policy import adjust_reward
from confs.path_conf import tsp_agents_data_dir
from utils.logger import Logger

log = Logger("reward", '1').get_logger()

class QLearningAgent(BaseAgent):
    def __init__(self, num_cities, num_actions, alpha=0.1, gamma=0.99, epsilon=0.5,
                 reward_strategy="negative_distance", model_path=tsp_agents_data_dir + "best_q-table.pkl",
                 state_space_config="step", episodes=1000, save_q_every=100):
        self.state_space_config = state_space_config
        self.episodes = episodes
        self.save_q_every = save_q_every
        self.q_value_changes = defaultdict(list) if self.state_space_config == "visits" else {state: [] for state in range(num_cities)}
        self.action_frequency = np.zeros((num_cities, num_actions))
        self.action_counts = {a: 0 for a in range(num_actions)}
        self.episode_rewards = []
        self.average_rewards = []
        self.cumulative_rewards = []
        self.q_table_snapshots = []
        self.action_frequencies = []
        self.iteration_strategies = []
        self.strategy_matrix = defaultdict(lambda: defaultdict(int))
        super().__init__(num_cities, num_actions, alpha, gamma, epsilon, reward_strategy, model_path)

    def initialize_model(self):
        if self.state_space_config == "step":
            self.q_table = np.zeros((self.num_cities, self.num_actions))
        elif self.state_space_config == "visits":
            self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
        else:
            raise ValueError("Invalid state_space_config. Choose 'step' or 'visits'.")
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
        if self.state_space_config == "step":
            return state['current_city']
        elif self.state_space_config == "visits":
            return tuple(state['current_path'])
        else:
            raise ValueError("Invalid state_space_config. Choose 'step' or 'visits'.")

    def update(self, state, action, reward, next_state, done):
        current_index = self.state_to_index(state)
        next_index = self.state_to_index(next_state)
        best_next_action = np.argmax(self.q_table[next_index])
        td_target = reward + self.gamma * self.q_table[next_index, best_next_action] * (1 - done)
        td_error = td_target - self.q_table[current_index, action]
        self.q_table[current_index, action] += self.alpha * td_error
        self.q_value_changes[current_index].append(self.q_table[current_index, action])

    def update_strategy(self, state, action):
        key = tuple(state['current_path'])
        self.strategy_matrix[key][action] += 1

    def get_strategy_matrix(self):
        if self.state_space_config == "step":
            return self.strategy_matrix
        elif self.state_space_config == "visits":
            strategy_matrix = np.zeros((self.num_cities, self.num_actions))
            for state, actions in self.strategy_matrix.items():
                current_city = state[-1]
                for action, count in actions.items():
                    strategy_matrix[current_city][action] = count
            return strategy_matrix

    def update_counts(self, action):
        self.action_counts[action] += 1

    def train(self, env, num_episodes):
        stable_episode = None
        for episode in range(num_episodes):
            state = env.reset()
            total_distance = 0
            done = False
            visited = []
            print(f"Episode {episode + 1}/{num_episodes} - Training")
            while not done:
                action = self.choose_action(state)
                self.update_counts(action)
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
                self.action_frequency[state['current_city']][action] += 1
                self.update_strategy(state, action)

            self.episode_rewards.append(-total_distance)
            self.cumulative_rewards.append(sum(self.episode_rewards))
            self.average_rewards.append(np.mean(self.episode_rewards))

            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()

            if len(self.episode_rewards) > 10:
                reward_diff = np.abs(self.episode_rewards[-1] - np.mean(self.episode_rewards[-10:]))
                if reward_diff < 1e-3 and stable_episode is None:
                    stable_episode = episode

            if (episode + 1) % self.save_q_every == 0:
                if self.state_space_config == "step":
                    self.q_table_snapshots.append(self.q_table.copy())
                elif self.state_space_config == "visits":
                    sampled_states = list(itertools.islice(self.q_table.items(), 100))
                    sampled_q_table = {state: q.copy() for state, q in sampled_states}
                    self.q_table_snapshots.append(sampled_q_table)
                self.save_q_table(episode)

            if episode % 100 == 0:
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
            file_path = f"q_tables/q_table_path_{episode + 1}.npy"
            q_table_dict = {state: q for state, q in self.q_table.items()}
            np.save(file_path, q_table_dict)

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
        """
        Extract the policy from the Q-table.

        Returns:
        - policy: Dictionary mapping states to the best action.
        """
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