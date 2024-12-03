import numpy as np
import random
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_dir)
from confs.path_conf import tsp_data_dir
from rl.env.tsp_env import TSPEnv
from utils.logger import Logger

logger = Logger('QLearningTSP').logger


class QLearningTSP:
    def __init__(self, env: TSPEnv, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.99,
                 min_epsilon=0.1, alpha_decay=0.995, min_alpha=0.1):
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.min_epsilon = min_epsilon
        self.num_cities = env.num_cities
        self.distance_matrix = env.distance_matrix

        # Q-table dimensions: (num_cities * 2^num_cities, num_cities)
        # First dimension is the composite state (current_city * 2^num_cities + visited_mask)
        # Second dimension is the next city to visit (action)
        self.q_table = np.zeros((self.num_cities * 2 ** self.num_cities, self.num_cities))

        self.best_solution = None
        self.best_distance = float('inf')
        self.fitness_history = []
        self.exploration_history = []
        self.total_rewards = []
        self.convergence_episode = None

        # Set the random seed for reproducibility
        seed = 2
        random.seed(seed)
        np.random.seed(seed)

    def get_state_id(self, current_city, visited_mask):
        """
        Combine the current city and visited mask into a unique state id.
        - current_city: 当前城市的编号 (0 到 num_cities-1)
        - visited_mask: 一个整数，表示已经访问过的城市。通过二进制位来表示城市是否访问。

        返回一个唯一的状态编号
        """
        return current_city * 2 ** self.num_cities + visited_mask

    def choose_action(self, current_city, visited_mask):
        """
        选择一个动作（下一个城市）基于当前状态和epsilon-greedy策略。
        - current_city: 当前城市编号
        - visited_mask: 一个整数，表示已经访问过的城市
        """
        state_id = self.get_state_id(current_city, visited_mask)

        if bin(visited_mask).count('1') == self.num_cities:
            return int(list(bin(visited_mask))[0])  # All cities visited, return to start city

        if random.random() < self.epsilon:
            # Exploration: randomly choose a city that has not been visited
            unvisited = [city for city in range(self.num_cities) if not (visited_mask & (1 << city))]
            return random.choice(unvisited) if unvisited else None
        else:
            # Exploitation: choose the city with the highest Q value
            q_values = self.q_table[state_id]
            unvisited_q_values = [(city, q_values[city]) for city in range(self.num_cities) if
                                  not (visited_mask & (1 << city))]
            return max(unvisited_q_values, key=lambda x: x[1])[0] if unvisited_q_values else None

    def update_q_table(self, current_city, visited_mask, next_city, reward):
        """
        使用Q-learning更新规则更新Q表。
        - current_city: 当前城市编号
        - visited_mask: 已访问城市的二进制掩码
        - next_city: 下一个城市
        - reward: 当前动作的奖励
        """
        state_id = self.get_state_id(current_city, visited_mask)
        max_future_q = max(self.q_table[state_id, :]) if next_city is not None else 0
        current_q = self.q_table[state_id, next_city]
        self.q_table[state_id, next_city] += self.alpha * (reward + self.gamma * max_future_q - current_q)

    def run_episode(self):
        """
        执行一次TSP问题的Q-learning训练。
        """
        start = current_city = 0
        visited_mask = 1 << current_city  # Mark the starting city as visited
        tour = [current_city]
        total_distance = 0
        rewards = []

        while len(tour) <= self.num_cities:  # Not all cities have been visited
            next_city = self.choose_action(current_city, visited_mask)
            if next_city is None:
                break

            distance = self.distance_matrix[current_city, next_city]
            reward = 1 / distance  # Reward is the inverse of the distance
            # if next_city == start:
            #     reward += 10  # Add a reward for completing the tour

            # Update Q-table
            self.update_q_table(current_city, visited_mask, next_city, reward)

            total_distance += distance
            visited_mask |= (1 << next_city)  # Mark the next city as visited
            tour.append(next_city)
            current_city = next_city
            rewards.append(reward)

        return tour, total_distance, sum(rewards)

    def run(self, param_combo):
        """
        运行Q-learning进行训练并返回最优结果。
        """
        true_distance = self.env.opt_tour_distance
        for episode in range(self.episodes):
            tour, total_distance, episode_reward = self.run_episode()
            self.total_rewards.append(episode_reward)

            if total_distance < self.best_distance:
                self.best_distance = total_distance
                self.best_solution = tour.copy()
                self.convergence_episode = episode  # Update convergence episode

                # Save the Q-table when best_distance is updated
                q_table_filename = self.generate_q_table_filename(param_combo)
                np.save(q_table_filename, self.q_table)  # Save the Q-table

            self.fitness_history.append(self.best_distance)
            self.exploration_history.append(self.epsilon)

            if episode % 9 == 0:
                # Decay epsilon and alpha with a minimum threshold
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
            if episode % 200 == 0:
                logger.info(
                    f"Num_cities={self.num_cities} Episode {episode} | "
                    f"Current Distance: {total_distance:.2f} vs Best Distance: {self.best_distance:.2f} vs True distance:{true_distance} | "
                    f"Tour: {tour} vs True:{self.env.opt_tour} | "
                    f"Alpha: {self.alpha:.4f}, Gamma: {self.gamma:.4f}, Epsilon: {self.epsilon:.4f}, "
                    f"Min Alpha: {self.min_alpha}, Min Epsilon: {self.min_epsilon}"
                )

        return {
            'true_distance': true_distance,
            'true_solution': [i - 1 for i in self.env.opt_tour] + [0],
            'best_solution': self.best_solution,
            'best_distance': self.best_distance,
            'fitness_history': self.fitness_history,
            'exploration_history': self.exploration_history,
            'total_rewards': self.total_rewards,
            'convergence_episode': self.convergence_episode
        }

    def generate_q_table_filename(self, param_combo):
        """
        生成Q表保存文件的路径
        """
        dir_ = root_dir+f'q/{param_combo[0]}_{param_combo[1]}_{param_combo[2]}_{param_combo[3]}_{param_combo[4]}'
        os.makedirs(dir_, exist_ok=True)
        filepath = dir_ + f"/Q.npy"
        return filepath


def parameter_search(env, param_grid):
    """
    使用给定的参数网格执行参数搜索并返回结果。
    """
    results = []
    total_combinations = len(param_grid['alpha']) * len(param_grid['gamma']) * len(
        param_grid['epsilon_decay']) * len(param_grid['min_alpha']) * len(param_grid['min_epsilon'])

    # Create header if file does not exist
    csv_file_name=root_dir + f'{env.num_cities}-exp.csv'
    header = ['alpha', 'gamma', 'epsilon_decay', 'min_alpha', 'min_epsilon', 'best_distance'
              ,'true_distance','best_solution','true_solution','total_rewards','convergence_episode']
    df = pd.DataFrame(columns=header)
    df.to_csv(csv_file_name, index=False)

    param_combinations = [
        (alpha, gamma, epsilon_decay, min_alpha, min_epsilon)
        for alpha in param_grid['alpha']
        for gamma in param_grid['gamma']
        for epsilon_decay in param_grid['epsilon_decay']
        for min_alpha in param_grid['min_alpha']
        for min_epsilon in param_grid['min_epsilon']
    ]


    for param_combo in tqdm(param_combinations, total=total_combinations, desc="Parameter Search Progress",
                            unit="combo"):
        alpha, gamma, epsilon_decay, min_alpha, min_epsilon = param_combo

        params = {
            'episodes': 100000,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': 0.5,
            'epsilon_decay': epsilon_decay,
            'alpha_decay': 0.995,
            'min_alpha': min_alpha,
            'min_epsilon': min_epsilon
        }

        qlearning_tsp = QLearningTSP(env, **params)
        metrics = qlearning_tsp.run(param_combo)

        # 保存结果
        result = {
            'alpha': alpha,
            'gamma': gamma,
            'epsilon_decay': epsilon_decay,
            'min_alpha': min_alpha,
            'min_epsilon': min_epsilon,
            'best_distance': metrics['best_distance'],
            'true_distance': metrics['true_distance'],
            'best_solution': metrics['best_solution'],
            'true_solution': metrics['true_solution'],

            'total_rewards': sum(metrics['total_rewards']),
            'convergence_episode': metrics['convergence_episode']
        }
        # results.append(result)
        # df_results = pd.DataFrame(results)
        df_result = pd.DataFrame([result])
        df_result.to_csv(csv_file_name, mode='a', header=False, index=False)

    # return df_results


if __name__ == '__main__':
    date_string = datetime.now().strftime("%Y-%m-%d")
    root_dir = tsp_data_dir + f'ex_qlearning/{date_string}/'
    os.makedirs(root_dir, exist_ok=True)

    # 测试所有的城市数量
    result = {k: v for k, v in TSPEnv.map_cities.items() if k > 0}

    for i in result:
        num_cities = i
        env = TSPEnv(num_cities=num_cities, use_tsplib=True)
        param_grid = {
            'alpha': np.arange(0, 1.1, 0.1).tolist(),
            'gamma': np.arange(0, 1.1, 0.1).tolist(),
            'epsilon_decay': [0.9, 0.95, 0.99, 0.995, 0.9995],
            'min_alpha': [0.01, 0.1, 0.2],
            'min_epsilon': [0.01, 0.1, 0.2]
        }
        results_df = parameter_search(env, param_grid)
        print(results_df)
