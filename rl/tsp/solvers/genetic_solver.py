# algorithms/genetic_algorithm.py

import numpy as np
import random
from rl.env.tsp_env import TSPEnv

class GeneticAlgorithm:
    def __init__(self, env: TSPEnv, population_size=100, generations=500, mutation_rate=0.01, crossover_rate=0.9):
        """
        Initialize the Genetic Algorithm with the given parameters.

        Parameters:
            env (TSPEnv): The environment representing the Traveling Salesman Problem.
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to evolve.
            mutation_rate (float): Probability of mutation for each gene.
            crossover_rate (float): Probability of performing crossover between parents.

        Convergence Principles:
            The Genetic Algorithm (GA) converges towards an optimal or near-optimal solution through the following mechanisms:

            1. **Selection Pressure:**
               - By selecting fitter individuals more frequently for reproduction, the population gradually shifts towards better solutions.

            2. **Crossover (Recombination):**
               - Combining genetic information from two parents allows the algorithm to explore new regions of the solution space, potentially discovering superior offspring.

            3. **Mutation:**
               - Introducing random changes helps maintain genetic diversity within the population, preventing premature convergence to local optima and allowing the exploration of new solutions.

            4. **Elitism (Implicit in Best Solution Tracking):**
               - By keeping track of the best solution found, the algorithm ensures that the quality of solutions does not degrade over generations.

            5. **Fitness Landscape Navigation:**
               - The fitness function guides the search towards regions of the solution space with higher fitness, enabling the algorithm to navigate towards optimal solutions.

            6. **Balancing Exploration and Exploitation:**
               - Crossover and mutation balance the exploration of new solutions and the exploitation of known good solutions, facilitating steady progress towards convergence.
        """
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
        """
        Initialize the population with random permutations of city indices.

        Returns:
            List[List[int]]: A list of individuals, each representing a possible tour.
        """
        population = []
        base = list(range(self.num_cities))  # Base list of city indices
        for _ in range(self.population_size):
            individual = base.copy()         # Create a copy of the base list
            random.shuffle(individual)        # Shuffle to create a random tour
            population.append(individual)     # Add the individual to the population
        return population

    def fitness(self, individual):
        """
        Calculate the fitness of an individual.

        Fitness is defined as the inverse of the total distance of the tour.
        A shorter distance results in a higher fitness value.

        Parameters:
            individual (List[int]): A permutation of city indices representing a tour.

        Returns:
            float: The fitness value of the individual.
        """
        distance = 0
        for i in range(self.num_cities):
            from_city = individual[i]
            to_city = individual[(i + 1) % self.num_cities]  # Ensure the tour is circular
            distance += self.distance_matrix[from_city, to_city]
        return 1 / distance  # Fitness is inverse of distance

    def selection(self):
        """
        Select two individuals from the population based on their fitness.

        Uses roulette wheel selection where the probability of selection is proportional to fitness.

        Returns:
            List[List[int]]: Two selected parent individuals.
        """
        fitnesses = [self.fitness(ind) for ind in self.population]  # Calculate fitness for all individuals
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]     # Normalize fitness values to probabilities
        selected = np.random.choice(self.population_size, size=2, replace=False, p=probabilities)
        return [self.population[selected[0]], self.population[selected[1]]]

    def crossover(self, parent1, parent2):
        """
        Perform Ordered Crossover (OX) between two parents to produce an offspring.

        Ordered Crossover (OX) is a genetic operator used to recombine two parent solutions
        to produce a child solution. It ensures that the child inherits a valid sequence
        of cities without duplicates, maintaining the permutation nature of the Traveling
        Salesman Problem (TSP).

        Benefits of Ordered Crossover (OX):
            1. **Preserves Relative Order:** Maintains the relative ordering of cities from both parents,
               which can help in retaining good sub-tours.
            2. **Prevents Duplicates:** Ensures that each city appears exactly once in the offspring,
               maintaining the validity of the tour.
            3. **Promotes Diversity:** Combines segments from both parents, promoting genetic diversity
               and exploration of the solution space.
            4. **Efficient Exploration:** Facilitates the combination of beneficial traits from both parents,
               potentially leading to better offspring.

        Parameters:
            parent1 (List[int]): The first parent individual.
            parent2 (List[int]): The second parent individual.

        Returns:
            List[int]: The resulting child individual after crossover.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()  # No crossover; return a copy of parent1

        # Select two random crossover points
        start, end = sorted(random.sample(range(self.num_cities), 2))

        # Initialize child with None to indicate unfilled positions
        child = [None] * self.num_cities

        # Step 1: Copy the subsequence from parent1 to child
        child[start:end] = parent1[start:end]

        # Step 2: Fill the remaining positions with cities from parent2 in order
        p2_index = end
        c_index = end
        while None in child:
            if p2_index >= self.num_cities:
                p2_index = 0  # Wrap around to the beginning if end is reached
            city = parent2[p2_index]
            if city not in child:
                child[c_index] = city
                c_index += 1
            p2_index += 1

        return child

    def mutate(self, individual):
        """
        Apply mutation to an individual by swapping cities with a certain probability.

        Parameters:
            individual (List[int]): The individual to mutate.

        Returns:
            List[int]: The mutated individual.
        """
        for swapped in range(self.num_cities):
            if random.random() < self.mutation_rate:
                swap_with = random.randint(0, self.num_cities - 1)  # Select a random city to swap with
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]  # Perform the swap
        return individual

    def evolve_population(self):
        """
        Evolve the current population to form a new generation.

        Steps:
            1. Select parents based on fitness.
            2. Perform crossover to produce offspring.
            3. Apply mutation to the offspring.
            4. Add the offspring to the new population.
        """
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.selection()          # Select two parents
            child = self.crossover(parent1, parent2)     # Generate a child through crossover
            child = self.mutate(child)                    # Apply mutation to the child
            new_population.append(child)                  # Add the child to the new population
        self.population = new_population                  # Update the population with the new generation

    def run(self):
        """
        Execute the Genetic Algorithm to find the best solution.

        Returns:
            Tuple[List[int], float, List[float]]:
                - Best solution found (list of city indices).
                - Distance of the best solution.
                - History of best distances per generation.
        """
        for generation in range(self.generations):
            self.evolve_population()  # Create a new generation

            # Evaluate current population to find the best individual
            for individual in self.population:
                current_distance = sum(
                    self.distance_matrix[individual[i], individual[(i + 1) % self.num_cities]]
                    for i in range(self.num_cities)
                )
                if current_distance < self.best_distance:
                    self.best_distance = current_distance       # Update the best distance found
                    self.best_solution = individual.copy()      # Update the best solution found

            self.fitness_history.append(self.best_distance)     # Record the best distance for this generation

            if generation % 50 == 0:
                print(f"Generation {generation}: Best Distance = {self.best_distance:.2f}")

        return self.best_solution, self.best_distance, self.fitness_history
