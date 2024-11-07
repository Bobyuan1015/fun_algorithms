# algorithms/ant_colony.py

import numpy as np
import random

from rl.env.tsp_env import TSPEnv


class AntColony:
    def __init__(self, env: TSPEnv, num_ants=20, generations=500, alpha=1.0, beta=5.0, evaporation_rate=0.5, Q=100):
        self.env = env
        self.num_ants = num_ants
        self.generations = generations
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta    # Influence of heuristic
        self.evaporation_rate = evaporation_rate
        self.Q = Q  # Constant for pheromone update
        self.num_cities = env.num_cities
        self.distance_matrix = env.distance_matrix
        self.pheromone = np.ones((self.num_cities, self.num_cities)) / self.num_cities
        self.best_distance = float('inf')
        self.best_solution = None
        self.fitness_history = []

    def run(self):
        for generation in range(self.generations):
            all_solutions = []
            all_distances = []
            for _ in range(self.num_ants):
                solution = self.construct_solution()
                distance = self.calculate_distance(solution)
                all_solutions.append(solution)
                all_distances.append(distance)
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_solution = solution.copy()
            self.update_pheromones(all_solutions, all_distances)
            self.fitness_history.append(self.best_distance)
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Distance = {self.best_distance:.2f}")
        return self.best_solution, self.best_distance, self.fitness_history

    def construct_solution(self):
        solution = []
        visited = set()
        current_city = random.randint(0, self.num_cities - 1)
        solution.append(current_city)
        visited.add(current_city)
        for _ in range(self.num_cities - 1):
            probabilities = self.calculate_transition_probabilities(current_city, visited)
            next_city = np.random.choice(range(self.num_cities), p=probabilities)
            solution.append(next_city)
            visited.add(next_city)
            current_city = next_city
        return solution

    def calculate_transition_probabilities(self, current_city, visited):
        pheromone = self.pheromone[current_city]
        heuristic = 1 / (self.distance_matrix[current_city] + 1e-10)
        pheromone = np.power(pheromone, self.alpha)
        heuristic = np.power(heuristic, self.beta)
        probabilities = pheromone * heuristic
        probabilities[list(visited)] = 0
        total = np.sum(probabilities)
        if total == 0:
            return np.array([1 / self.num_cities] * self.num_cities)
        return probabilities / total

    def calculate_distance(self, solution):
        distance = 0
        for i in range(self.num_cities):
            distance += self.distance_matrix[solution[i], solution[(i + 1) % self.num_cities]]
        return distance

    def update_pheromones(self, solutions, distances):
        self.pheromone *= (1 - self.evaporation_rate)
        for solution, distance in zip(solutions, distances):
            for i in range(self.num_cities):
                from_city = solution[i]
                to_city = solution[(i + 1) % self.num_cities]
                self.pheromone[from_city, to_city] += self.Q / distance
                self.pheromone[to_city, from_city] += self.Q / distance  # Assuming undirected graph

