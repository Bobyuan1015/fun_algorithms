import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict
import itertools

class QLearningTSP:
    def __init__(self, cities, state_space_config="simple", alpha=0.1, gamma=0.9, epsilon=0.2,
                 episodes=1000, save_q_every=100):
        """
        Initialize the Q-Learning TSP solver.

        Parameters:
        - cities: 2D numpy array representing distances between cities.
        - state_space_config: "simple" for current city only, "path" for visited cities.
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

        if self.state_space_config == "simple":
            # Simple state: current city
            self.q_table = np.zeros((self.num_cities, self.num_cities))
        elif self.state_space_config == "path":
            # Detailed state: visited cities as a tuple
            self.q_table = defaultdict(lambda: np.zeros(self.num_cities))
        else:
            raise ValueError("Invalid state_space_config. Choose 'simple' or 'path'.")

        # Data recording
        self.episode_rewards = []
        self.average_rewards = []
        self.cumulative_rewards = []
        self.q_table_snapshots = []
        self.q_value_changes = defaultdict(list) if self.state_space_config == "path" else {state: [] for state in self.states}
        self.action_frequency = np.zeros((self.num_cities, self.num_cities))

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
            if self.state_space_config == "simple":
                q_values = self.q_table[current_city]
            elif self.state_space_config == "path":
                state = tuple(visited)
                q_values = self.q_table[state]
            available_q_values = {a: q_values[a] for a in available_actions}
            return max(available_q_values, key=available_q_values.get)

    def _update_q_table(self, visited, action, reward, next_visited):
        if self.state_space_config == "simple":
            state = self._get_current_city(visited)
            next_state = self._get_current_city(next_visited)
            current_q = self.q_table[state][action]
            next_max_q = np.max(self.q_table[next_state])
            updated_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
            self.q_table[state][action] = updated_q
            self.q_value_changes[state].append(updated_q)
        elif self.state_space_config == "path":
            state = tuple(visited)
            next_state = tuple(next_visited)
            current_q = self.q_table[state][action]
            next_max_q = np.max(self.q_table[next_state])
            updated_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
            self.q_table[state][action] = updated_q
            self.q_value_changes[state].append(updated_q)

    def train(self):
        stable_episode = None
        for episode in range(self.episodes):
            start_city = np.random.choice(self.states)
            visited = [start_city]

            episode_reward = 0

            while len(visited) < self.num_cities:
                action = self._choose_action(visited)
                reward = -self.cities[self._get_current_city(visited)][action]
                episode_reward += reward

                next_visited = visited + [action]

                self._update_q_table(visited, action, reward, next_visited)

                self.action_frequency[self._get_current_city(visited)][action] += 1
                visited = next_visited

            self.episode_rewards.append(episode_reward)
            self.cumulative_rewards.append(sum(self.episode_rewards))
            self.average_rewards.append(np.mean(self.episode_rewards))

            if len(self.episode_rewards) > 10:
                reward_diff = np.abs(self.episode_rewards[-1] - np.mean(self.episode_rewards[-10:]))
                if reward_diff < 1e-3 and stable_episode is None:
                    stable_episode = episode

            if (episode + 1) % self.save_q_every == 0:
                if self.state_space_config == "simple":
                    self.q_table_snapshots.append(self.q_table.copy())
                elif self.state_space_config == "path":
                    # 对于高维状态空间，可以采样一部分状态
                    sampled_states = list(itertools.islice(self.q_table.items(), 100))  # 采样前100个状态
                    sampled_q_table = {state: q.copy() for state, q in sampled_states}
                    self.q_table_snapshots.append(sampled_q_table)
                self.save_q_table(episode)

        print(f"Training converged at episode: {stable_episode}")

    def save_q_table(self, episode):
        os.makedirs("q_tables", exist_ok=True)
        if self.state_space_config == "simple":
            file_path = f"q_tables/q_table_simple_{episode + 1}.npy"
            np.save(file_path, self.q_table)
        elif self.state_space_config == "path":
            # 保存为字典形式
            file_path = f"q_tables/q_table_path_{episode + 1}.npy"
            # 由于 defaultdict 不能直接保存，需要转换为普通字典
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
        if self.state_space_config == "simple":
            for state, q_values in self.q_value_changes.items():
                plt.plot(q_values, label=f"State {state}")
        elif self.state_space_config == "path":
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
            if self.state_space_config == "simple":
                plt.imshow(q_table, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Q-table at Episode {i * self.save_q_every + self.save_q_every}")
            elif self.state_space_config == "path":
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
        if self.state_space_config == "simple":
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
        elif self.state_space_config == "path":
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
        if self.state_space_config == "simple":
            for state in self.states:
                best_action = np.argmax(self.q_table[state])
                policy[state] = best_action
        elif self.state_space_config == "path":
            for state, q_values in self.q_table.items():
                best_action = np.argmax(q_values)
                policy[state] = best_action
        return policy

if __name__ == "__main__":
    num_cities = 5
    state_type = "simple"
    # state_type = "path"
    cities = np.random.randint(10, 100, size=(num_cities, num_cities))

    tsp_solver_detailed = QLearningTSP(cities, state_space_config=state_type, episodes=50, save_q_every=50)
    tsp_solver_detailed.train()
    tsp_solver_detailed.save_results()
    tsp_solver_detailed.plot_results()
    tsp_solver_detailed.plot_q_value_changes()
    tsp_solver_detailed.plot_policy_evolution()
    tsp_solver_detailed.plot_action_frequencies()
    tsp_solver_detailed.plot_q_value_trends()

todo list
1.grid search最优参数
2.动态配置reward
3.运行数据