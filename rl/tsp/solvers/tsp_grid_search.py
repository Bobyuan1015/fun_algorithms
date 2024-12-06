import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os,sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_dir)
from tqdm import tqdm

from confs.path_conf import  tsp_data_dir
from rl.env.tsp_env import TSPEnv
from utils.logger import Logger

logger = Logger(pre_dir='ex_tsp',pre_file='QLearningTSP').logger

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
        # self.q_table = np.zeros((self.num_cities, self.num_cities))
        self.q_table = np.zeros((self.num_cities, 2 ** self.num_cities))  # Q-table updated for city and visited states

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

    def choose_action(self, current_city, visited):
        if len(visited) == self.num_cities:
            return list(visited)[0]
        if random.random() < self.epsilon:
            unvisited = [city for city in range(self.num_cities) if city not in visited]
            return random.choice(unvisited) if unvisited else None
        else:
            q_values = [(city, self.q_table[current_city, city]) for city in range(self.num_cities) if
                        city not in visited]
            return max(q_values, key=lambda x: x[1])[0] if q_values else None

    def update_q_table(self, current_city, next_city, reward):
        max_future_q = max(self.q_table[next_city, :]) if next_city is not None else 0
        current_q = self.q_table[current_city, next_city]
        self.q_table[current_city, next_city] += self.alpha * (reward + self.gamma * max_future_q - current_q)

    def run_episode(self):
        start = current_city = 0
        visited = set([current_city])
        tour = [current_city]
        total_distance = 0
        rewards = []

        while len(tour) <= self.num_cities:
            next_city = self.choose_action(current_city, visited)
            if next_city is None:
                break

            distance = self.distance_matrix[current_city, next_city]
            reward = 1 / distance
            if next_city == start:
                reward += 10

            self.update_q_table(current_city, next_city, reward)

            total_distance += distance
            visited.add(next_city)
            tour.append(next_city)
            current_city = next_city
            rewards.append(reward)

        return tour, total_distance, sum(rewards)

    def run(self, param_combo):
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

            # Log current parameters and best distance comparison
            if episode % 10 == 0:
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
            'true_distance':true_distance,
            'true_solution':[i-1 for i in self.env.opt_tour] +[0],
            'best_solution': self.best_solution,
            'best_distance': self.best_distance,
            'fitness_history': self.fitness_history,
            'exploration_history': self.exploration_history,
            'total_rewards': self.total_rewards,
            'convergence_episode': self.convergence_episode
        }

    def generate_q_table_filename(self, param_combo):
        # Create a filename based on the parameter combination
        dir_ = root_dir+'q/'
        os.makedirs(dir_, exist_ok=True)
        filepath = dir_+f"/Q_{param_combo[0]}_{param_combo[1]}_{param_combo[2]}_{param_combo[3]}_{param_combo[4]}.npy"
        return filepath


def parameter_search(env, param_grid):
    results = []
    total_combinations = len(param_grid['alpha']) * len(param_grid['gamma']) * len(
        param_grid['epsilon_decay']) * len(param_grid['min_alpha']) * len(param_grid['min_epsilon'])

    # 使用 tqdm 来显示进度条
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
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(root_dir + f'{num_cities}-exp.csv', index=False)
    return df_results

if __name__ == '__main__':
    root_dir = tsp_data_dir + 'ex_qlearning/'
    os.makedirs(root_dir, exist_ok=True)
    result = {k: v for k, v in TSPEnv.map_cities.items() if k > 0 }

    for i in result:
        num_cities = i
        env = TSPEnv(num_cities=num_cities, use_tsplib=True)
        param_grid = {
            'alpha': np.arange(0, 1.1, 0.1).tolist(),
            'gamma': np.arange(0,1.1,0.1).tolist(),
            'epsilon_decay': [0.9,0.95,0.99,0.995,0.9995],
            'min_alpha': [0.01, 0.1, 0.2],
            'min_epsilon': [0.01, 0.1, 0.2]
        }
        results_df = parameter_search(env, param_grid)
        print(results_df)