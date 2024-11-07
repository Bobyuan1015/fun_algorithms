# algorithms/genetic_algorithm.py

import numpy as np
import random
from rl.env.tsp_env import TSPEnv

class GeneticAlgorithm:
    def __init__(self, env: TSPEnv, population_size=100, generations=500, mutation_rate=0.01, crossover_rate=0.9):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_cities = env.num_cities
        self.distance_matrix = env.distance_matrix
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_distance = float('inf')
        self.fitness_history = []

    def initialize_population(self):
        population = []
        base = list(range(self.num_cities))
        for _ in range(self.population_size):
            individual = base.copy()
            random.shuffle(individual)
            population.append(individual)
        return population

    def fitness(self, individual):
        distance = 0
        for i in range(self.num_cities):
            from_city = individual[i]
            to_city = individual[(i + 1) % self.num_cities]
            distance += self.distance_matrix[from_city, to_city]
        return 1 / distance  # Fitness is inverse of distance

    def selection(self):
        fitnesses = [self.fitness(ind) for ind in self.population]
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        selected = np.random.choice(self.population_size, size=2, replace=False, p=probabilities)
        return [self.population[selected[0]], self.population[selected[1]]]

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy()
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child_p1 = parent1[start:end]
        child_p2 = [city for city in parent2 if city not in child_p1]
        child = child_p2[:start] + child_p1 + child_p2[start:]
        return child

    def mutate(self, individual):
        for swapped in range(self.num_cities):
            if random.random() < self.mutation_rate:
                swap_with = random.randint(0, self.num_cities - 1)
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
        return individual

    def evolve_population(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def run(self):
        for generation in range(self.generations):
            self.evolve_population()
            # Evaluate current population
            for individual in self.population:
                current_distance = sum(
                    self.distance_matrix[individual[i], individual[(i + 1) % self.num_cities]]
                    for i in range(self.num_cities)
                )
                if current_distance < self.best_distance:
                    self.best_distance = current_distance
                    self.best_solution = individual.copy()
            self.fitness_history.append(self.best_distance)
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Distance = {self.best_distance:.2f}")
        return self.best_solution, self.best_distance, self.fitness_history
