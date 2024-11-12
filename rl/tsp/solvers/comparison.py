import time
import csv
import matplotlib.pyplot as plt

from confs.path_conf import system_ex_comparison_model_dir
from rl.env.tsp_env import TSPEnv
from rl.tsp.solvers.ant_colony_solver import AntColony
from rl.tsp.solvers.genetic_solver import GeneticAlgorithm
from rl.tsp.solvers.simulated_annealing_solver import SimulatedAnnealing


def plot_fitness_history(history_dict):
    """
    Plots the convergence curves for each TSP algorithm.

    Args:
        history_dict (dict): A dictionary containing fitness histories for each algorithm.
    """
    plt.figure(figsize=(10, 6))
    for label, history in history_dict.items():
        plt.plot(history, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('Best Distance')
    plt.title('Convergence of TSP Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_results_to_csv(results, filename='results.csv'):
    """
    Saves the summary results to a CSV file.

    Args:
        results (dict): A dictionary containing the best distance and time taken for each algorithm.
        filename (str): The name of the CSV file to save the results.
    """
    with open(filename, mode='w', newline='') as csv_file:
        fieldnames = ['Algorithm', 'Best Distance', 'Time Taken (seconds)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for algo, res in results.items():
            writer.writerow({
                'Algorithm': algo,
                'Best Distance': f"{res['Distance']:.2f}",
                'Time Taken (seconds)': f"{res['Time']:.2f}"
            })
    print(f"Summary results saved to {filename}")


def save_fitness_history_to_csv(fitness_histories):
    """
    Saves the convergence history of each algorithm to separate CSV files.

    Args:
        fitness_histories (dict): A dictionary containing the convergence histories for each algorithm.
    """
    for algo, history in fitness_histories.items():
        # Create a filename by replacing spaces with underscores and converting to lowercase
        filename = f"{system_ex_comparison_model_dir}/fitness_history_{algo.replace(' ', '_').lower()}.csv"
        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Iteration', 'Best Distance'])
            for iteration, distance in enumerate(history, start=1):
                writer.writerow([iteration, f"{distance:.2f}"])
        print(f"Convergence data for {algo} saved to {filename}")


def main():
    """
    Main function to run the comparison experiment between different TSP algorithms.
    """
    # Initialize TSP Environment
    env = TSPEnv(num_cities=30, use_tsplib=False)  # Adjust the number of cities as needed
    print(f"TSP Environment initialized with {env.num_cities} cities.")

    # Dictionaries to store results and fitness histories
    results = {}
    fitness_histories = {}

    # Genetic Algorithm
    ga = GeneticAlgorithm(env, population_size=200, generations=1000, mutation_rate=0.02, crossover_rate=0.9)
    start_time = time.time()
    ga_solution, ga_distance, ga_history = ga.run()
    ga_time = time.time() - start_time
    results['Genetic Algorithm'] = {'Distance': ga_distance, 'Time': ga_time}
    fitness_histories['Genetic Algorithm'] = ga_history
    print(f"Genetic Algorithm: Best Distance = {ga_distance:.2f}, Time Taken = {ga_time:.2f} seconds")

    # Ant Colony Optimization
    ac = AntColony(env, num_ants=50, generations=1000, alpha=1.0, beta=5.0, evaporation_rate=0.5, Q=100)
    start_time = time.time()
    ac_solution, ac_distance, ac_history = ac.run()
    ac_time = time.time() - start_time
    results['Ant Colony'] = {'Distance': ac_distance, 'Time': ac_time}
    fitness_histories['Ant Colony'] = ac_history
    print(f"Ant Colony: Best Distance = {ac_distance:.2f}, Time Taken = {ac_time:.2f} seconds")

    # Simulated Annealing
    sa = SimulatedAnnealing(env, initial_temperature=1000, cooling_rate=0.995, generations=10000)
    start_time = time.time()
    sa_solution, sa_distance, sa_history = sa.run()
    sa_time = time.time() - start_time
    results['Simulated Annealing'] = {'Distance': sa_distance, 'Time': sa_time}
    fitness_histories['Simulated Annealing'] = sa_history
    print(f"Simulated Annealing: Best Distance = {sa_distance:.2f}, Time Taken = {sa_time:.2f} seconds")

    # Display Comparison Results
    print("\n=== Comparison Results ===")
    for algo, res in results.items():
        print(f"{algo}: Best Distance = {res['Distance']:.2f}, Time Taken = {res['Time']:.2f} seconds")

    # Save Summary Results to CSV
    save_results_to_csv(results, filename=system_ex_comparison_model_dir+'results.csv')

    # Save Convergence Histories to CSV
    save_fitness_history_to_csv(fitness_histories)

    # Plot Convergence Curves
    plot_fitness_history(fitness_histories)

    # Optionally, render the best solution using TSPEnv
    # Here, using the Genetic Algorithm's best solution as an example
    env.path = ga_solution
    env.total_distance = ga_distance
    env.render()


if __name__ == "__main__":
    main()
