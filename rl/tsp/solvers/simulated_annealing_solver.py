# algorithms/simulated_annealing.py

import numpy as np
import random
from rl.env.tsp_env import TSPEnv

class SimulatedAnnealing:
    """
    Simulated Annealing algorithm for solving the Traveling Salesman Problem (TSP).

    The algorithm starts with an initial random solution and iteratively attempts to find a better solution by making small changes.
    It accepts worse solutions with a probability that decreases over time, allowing the algorithm to escape local minima and converge towards a global minimum.

    **Algorithm Process:**
    1. Initialize with a random solution and set the initial temperature.
    2. For a predefined number of generations:
        a. Generate a neighboring solution by swapping two cities.
        b. Calculate the change in distance (delta) between the new solution and the current solution.
        c. Decide whether to accept the new solution based on delta and the current temperature.
        d. If accepted and it's the best solution so far, update the best solution.
        e. Cool down the temperature according to the cooling rate.
        f. Record the best distance found.
        g. Optionally, print progress at intervals and perform early stopping if the temperature is sufficiently low.
    3. Return the best solution found along with its distance and the history of fitness values.

    **Convergence Principle:**
    The temperature gradually decreases, reducing the probability of accepting worse solutions over time.
    This allows the algorithm to explore the solution space widely at high temperatures and fine-tune the solution as the temperature cools,
    ultimately converging to a near-optimal solution.
    """

    def __init__(self, env: TSPEnv, initial_temperature=1000, cooling_rate=0.995, generations=10000):
        """
        Initialize the Simulated Annealing algorithm with the given environment and parameters.

        Args:
            env (TSPEnv): The TSP environment containing city information and distance matrix.
            initial_temperature (float): The starting temperature for the annealing process.
            cooling_rate (float): The rate at which the temperature decreases each generation.
            generations (int): The maximum number of generations to run the algorithm.
        """
        self.env = env
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.generations = generations
        self.num_cities = env.num_cities
        self.distance_matrix = env.distance_matrix

        # Initialize the current solution as a random permutation of cities
        self.current_solution = list(range(self.num_cities))
        random.shuffle(self.current_solution)
        self.current_distance = self.calculate_distance(self.current_solution)

        # Initialize the best solution found so far
        self.best_solution = self.current_solution.copy()
        self.best_distance = self.current_distance

        # Set the initial temperature
        self.temperature = self.initial_temperature

        # History of best distances for analysis
        self.fitness_history = []

    def calculate_distance(self, solution):
        """
        Calculate the total distance of the given TSP solution.

        Args:
            solution (list): A permutation of city indices representing the tour.

        Returns:
            float: The total distance of the tour.
        """
        distance = 0
        for i in range(self.num_cities):
            # Sum the distance between consecutive cities, wrapping around to the start
            distance += self.distance_matrix[solution[i], solution[(i + 1) % self.num_cities]]
        return distance

    def run(self):
        """
        Execute the Simulated Annealing algorithm to find the shortest possible tour.

        Returns:
            tuple: A tuple containing the best solution found, its total distance, and the fitness history.
        """
        for generation in range(self.generations):
            # Generate a neighbor by swapping two randomly selected cities
            new_solution = self.current_solution.copy()
            i, j = random.sample(range(self.num_cities), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

            # Calculate the distance of the new solution
            new_distance = self.calculate_distance(new_solution)
            delta = new_distance - self.current_distance

            # Decide whether to accept the new solution
            if delta < 0 or random.random() < np.exp(-delta / self.temperature):
                self.current_solution = new_solution
                self.current_distance = new_distance

                # Update the best solution if the new one is better
                if new_distance < self.best_distance:
                    self.best_solution = new_solution.copy()
                    self.best_distance = new_distance

            # Cool down the temperature
            self.temperature *= self.cooling_rate
            self.fitness_history.append(self.best_distance)

            # Print progress every 1000 generations
            if generation % 1000 == 0:
                print(f"Generation {generation}: Best Distance = {self.best_distance:.2f}")

            # Early stopping if temperature is sufficiently low
            if self.temperature < 1e-8:
                print("Temperature has cooled down sufficiently. Stopping early.")
                break

        return self.best_solution, self.best_distance, self.fitness_history
