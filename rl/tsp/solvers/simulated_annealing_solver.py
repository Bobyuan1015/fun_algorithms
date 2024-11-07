# algorithms/simulated_annealing.py

import numpy as np
import random
from rl.env.tsp_env import TSPEnv

class SimulatedAnnealing:
    def __init__(self, env: TSPEnv, initial_temperature=1000, cooling_rate=0.995, generations=10000):
        self.env = env
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.generations = generations
        self.num_cities = env.num_cities
        self.distance_matrix = env.distance_matrix
        self.current_solution = list(range(self.num_cities))
        random.shuffle(self.current_solution)
        self.current_distance = self.calculate_distance(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_distance = self.current_distance
        self.temperature = self.initial_temperature
        self.fitness_history = []

    def calculate_distance(self, solution):
        distance = 0
        for i in range(self.num_cities):
            distance += self.distance_matrix[solution[i], solution[(i + 1) % self.num_cities]]
        return distance

    def run(self):
        for generation in range(self.generations):
            # Generate neighbor by swapping two cities
            new_solution = self.current_solution.copy()
            i, j = random.sample(range(self.num_cities), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_distance = self.calculate_distance(new_solution)
            delta = new_distance - self.current_distance

            # Decide whether to accept the new solution
            if delta < 0 or random.random() < np.exp(-delta / self.temperature):
                self.current_solution = new_solution
                self.current_distance = new_distance
                if new_distance < self.best_distance:
                    self.best_solution = new_solution.copy()
                    self.best_distance = new_distance

            # Cool down
            self.temperature *= self.cooling_rate
            self.fitness_history.append(self.best_distance)

            if generation % 1000 == 0:
                print(f"Generation {generation}: Best Distance = {self.best_distance:.2f}")

            # Early stopping if temperature is low enough
            if self.temperature < 1e-8:
                break

        return self.best_solution, self.best_distance, self.fitness_history
