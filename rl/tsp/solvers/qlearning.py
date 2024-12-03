import numpy as np
import random
from rl.env.tsp_env import TSPEnv
from utils.logger import Logger

logger =Logger('QLearningTSP').logger

class QLearningTSP:
    def __init__(self, env: TSPEnv, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.99,
                 min_epsilon=0.1,alpha_decay=0.995, min_alpha=0.1):
        """
        Initialize the Q-learning algorithm for the TSP problem.

        Parameters:
            env (TSPEnv): The environment representing the Traveling Salesman Problem.
            episodes (int): Number of episodes to train.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial probability of choosing a random action (exploration rate).
            epsilon_decay (float): Factor to decay epsilon after each episode.
            min_epsilon (float): Minimum value for epsilon to ensure some exploration.
        """
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.gamma = gamma
        self.epsilon = epsilon  # 设置初始 epsilon 为 0.5
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay  # Alpha decay factor

        self.min_epsilon = min_epsilon
        self.num_cities = env.num_cities
        self.distance_matrix = env.distance_matrix
        self.q_table = np.zeros((self.num_cities, self.num_cities))
        # self.q_table = np.full((self.num_cities, self.num_cities), -np.inf)
        # self.q_table = np.full((self.num_cities, self.num_cities), -1e6)

        self.best_solution = None
        self.best_distance = float('inf')
        self.fitness_history = []
        # Set the random seed for reproducibility
        seed= 2
        random.seed(seed)
        np.random.seed(seed)

    def choose_action(self, current_city, visited):
        """
        Choose the next city to visit based on epsilon-greedy policy.

        Parameters:
            current_city (int): The current city index.
            visited (set): Set of visited cities.

        Returns:
            int: The index of the next city to visit.
        """
        if len(visited) == self.num_cities:
            return list(visited)[0]
        if random.random() < self.epsilon:
            # Explore: choose a random unvisited city
            unvisited = [city for city in range(self.num_cities) if city not in visited]
            a = random.choice(unvisited) if unvisited else None
            if current_city == 0:
                logger.info(f'->pick random {a}')
            return a
        else:
            # Exploit: choose the best action based on Q-table
            q_values = [(city, self.q_table[current_city, city]) for city in range(self.num_cities) if
                        city not in visited]
            a=max(q_values, key=lambda x: x[1])[0] if q_values else None
            if current_city ==0:
                logger.info(f'->pick {a} q_values[0:]={[round(x[1], 2) for x in q_values]} ')
            return a

    def update_q_table(self, current_city, next_city, reward):
        """
        Update the Q-table based on the Q-learning formula.

        Parameters:
            current_city (int): The current city index.
            next_city (int): The next city index.
            reward (float): The reward received for moving to the next city.
        """
        max_future_q = max(self.q_table[next_city, :]) if next_city is not None else 0
        current_q = self.q_table[current_city, next_city]
        if current_city ==0:
            logger.info(f'before update_q_table reward={reward}  future_q[{next_city}:]={self.q_table[next_city, :]}\nq_table[{current_city}:]={self.q_table[current_city, :]}')
        self.q_table[current_city, next_city] += self.alpha * (reward + self.gamma * max_future_q - current_q)
        if current_city ==0:
            logger.info(f'after update_q_table reward={reward} add_={self.alpha * (reward + self.gamma * max_future_q - current_q)}\nq_table[{current_city}:]={self.q_table[current_city, :]}')
        # logger.info()

    def run_episode(self):
        """
        Run a single episode of TSP using Q-learning.

        Returns:
            Tuple[List[int], float]: The sequence of visited cities and the total distance.
        """
        start=current_city = 0 # current_city = random.randint(0, self.num_cities - 1)
        visited = set([current_city]) # auto ordered
        tour = [current_city]
        total_distance = 0
        rewards = []
        # while len(visited) < self.num_cities:
        while len(tour) <= self.num_cities:
            next_city = self.choose_action(current_city, visited)
            if next_city is None:
                break

            # Calculate reward as negative distance (since we aim to minimize distance)
            distance = self.distance_matrix[current_city, next_city]
            reward = 1/distance
            if next_city == start:
                reward += 1 # bonus if the trip is returned
            self.update_q_table(current_city, next_city, reward)

            total_distance += distance
            visited.add(next_city)
            tour.append(next_city)
            current_city = next_city
            rewards.append(reward)

        # # Add distance to return to the starting city
        # if len(tour) == self.num_cities:
        #     total_distance += self.distance_matrix[current_city, tour[0]]
        #     tour.append(tour[0])  # Close the tour by returning to the start city

        return tour, total_distance

    def run(self):
        """
        Train the Q-learning algorithm to solve the TSP and return the best tour found.

        Returns:
            Tuple[List[int], float, List[float]]:
                - Best solution found (list of city indices).
                - Distance of the best solution.
                - History of best distances per episode.
        """
        for episode in range(self.episodes):

            tour, total_distance = self.run_episode()
            if total_distance < self.best_distance:
                self.best_distance = total_distance
                self.best_solution = tour.copy()

            self.fitness_history.append(self.best_distance)

            # Decay epsilon but ensure it doesn't go below min_epsilon
            # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            # self.alpha = max(0.0001, self.alpha * self.alpha_decay)  # Ensure alpha doesn't go below 0.01

            if episode % 8 == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)  # Ensure alpha doesn't go below 0.01
                logger.info(f"------------------------------------------------Episode {episode}: a={self.alpha} Best Distance = {self.best_distance:.2f} vs ture{self.env.calculate_tour_distance(opt_tour=True)} , Epsilon = {self.epsilon:.4f} \n{tour} best-{self.best_solution} \n{self.env.opt_tour} {len(self.env.opt_tour)}<-true best tour")

        return self.best_solution, self.best_distance, self.fitness_history