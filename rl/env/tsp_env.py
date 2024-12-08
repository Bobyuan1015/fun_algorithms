import json
import os
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from tqdm import tqdm
import copy

from confs.path_conf import tsp_data_dir
from rl.tsp.common.tsplib95_generator import generate_data_by_tsplib

class TSPEnv:
    map_cities = {16: 0, 22: 1, 48: 2, 51: 3, 96: 8, 100: 9, 101: 13, 105: 14, 130: 15}

    def __init__(self, num_cities=10, use_tsplib=False):
        """
        Initializes the TSP environment by randomly generating city coordinates,
        computing the distance matrix, and setting the environment to the initial state.

        Args:
            num_cities (int): The number of cities in the TSP problem.
        """
        self.save_file = tsp_data_dir + f'{num_cities}-env.tour'
        if self.load_saved_data(num_cities):
            print("Loaded saved data successfully.")
            self.distance_matrix = self.calculate_distance_matrix()  # Compute the distance matrix
        else:
            if use_tsplib:
                data = generate_data_by_tsplib()  # 0 for 16, 1 for 22, 2 for 48, 3 for 51, 8 for 96, 9\10\11\12 for 100, 13 for 101, 14 for 105 15 for 130

                if num_cities in self.map_cities:
                    if isinstance(data, list):
                        data = data[self.map_cities[num_cities]]
                        self.num_cities = data['dimension']
                        self.city_coords = data['node_coords'].numpy()
                        self.distance_matrix = self.calculate_distance_matrix()  # Compute the distance matrix

                        min_distance, best_path = self.find_shortest_tour()  # 16 cities
                        print(f'num_cities={num_cities} 计算的{min_distance}, {best_path},')
                        self.opt_tour = data['solution']
                        self.opt_tour_distance = self.calculate_tour_distance(opt_tour=True)
                        results = {"num_city": self.num_cities, "min_distance": min_distance, "best_path": best_path,
                                   "lib--opt_tour_distance": self.opt_tour_distance, "lib--opt_tour": self.opt_tour}

                        with open(tsp_data_dir + f'{self.num_cities}-枚举计算最优结果.json', 'w') as f:
                            json.dump(results, f, indent=4)
                        print(f'num_cities={num_cities} 提供的{self.opt_tour_distance}, {self.opt_tour}')
                else:
                    print(f'cannot pick data from the tsplib, please set use_tsplib=False')

            else:
                self.num_cities = num_cities
                self.city_coords = np.random.rand(num_cities, 2)  # Generate random coordinates
                self.distance_matrix = self.calculate_distance_matrix()  # Compute the distance matrix

                max_processes = 8
                self.opt_tour_distance, self.opt_tour = self.find_shortest_tour(max_processes)
            print(f'Init env num_cities = {self.num_cities} opt_tour={self.opt_tour} opt_tour_distance={self.opt_tour_distance}')
            self.save_initialization_data()
        self.reset()

    def save_initialization_data(self):
        """
        Saves the initialized data to a JSON file for later reuse.
        """
        data_to_save = {
            'num_cities': self.num_cities,
            'city_coords': self.city_coords.tolist(),
            'opt_tour': self.opt_tour,
            'opt_tour_distance': self.opt_tour_distance
        }
        with open(self.save_file, 'w') as file:
            json.dump(data_to_save, file)
        print(f"Saved initialization data to {self.save_file}")

    def load_saved_data(self, num_cities):
        """
        Loads the saved data from the file if it matches the current configuration.
        Returns:
            bool: True if the saved data was loaded successfully, False otherwise.
        """
        if not os.path.exists(self.save_file):
            return False

        with open(self.save_file, 'r') as file:
            saved_data = json.load(file)

        if saved_data['num_cities'] != num_cities:
            print("Saved data exists but does not match the current number of cities.")
            return False
        self.num_cities = num_cities
        self.city_coords = np.array(saved_data['city_coords'])
        self.opt_tour = saved_data['opt_tour']
        self.opt_tour_distance = saved_data['opt_tour_distance']
        return True

    def find_min_path(self, starting_from_second_city):
        first_two_cities = starting_from_second_city[0]
        cities = starting_from_second_city[1]
        shortest_distance = float('inf')

        # Initialize variables
        shortest_path = None
        shortest_distance = float('inf')

        # Generate all possible paths starting from the third city
        remaining_cities = list(range(cities))
        for i in first_two_cities:
            remaining_cities.remove(i)
        # Start recursion with the first two cities fixed
        num_paths = 1
        for i in range(cities - len(first_two_cities)):
            num_paths *= (cities - len(first_two_cities) - i)

        pbar = tqdm(total=num_paths, desc=f"Generating paths from: {first_two_cities}")

        def recursive_add_with_pbar(path, remaining_cities):
            nonlocal shortest_path, shortest_distance  # Access outer scope variables
            if len(remaining_cities) == 0:
                distance = 0
                for i in range(len(path) - 1):
                    distance += self.distance_matrix[path[i]][path[i + 1]]
                distance += self.distance_matrix[path[-1]][path[0]]  # Add distance from last city to first
                path.append(0)
                if distance < shortest_distance:
                    shortest_path = path[:]
                    shortest_distance = distance
                    print(f'前缀{first_two_cities} 更新最优：{shortest_distance}，{shortest_path}')
                pbar.update(1)
                return
            for j in range(len(remaining_cities)):
                new_path = path + [remaining_cities[j]]
                new_remaining_cities = remaining_cities[:j] + remaining_cities[j + 1:]
                recursive_add_with_pbar(new_path, new_remaining_cities)

        recursive_add_with_pbar(first_two_cities, remaining_cities)
        pbar.close()

        print(f'进程任务完成：  前缀{first_two_cities} 更新最优：{shortest_distance}，{shortest_path}')
        return shortest_path, shortest_distance

    def find_shortest_tour(self, max_processes=8):
        """
        Find the overall shortest path using multiprocessing.
        """
        starting_from_second_city = [[[0, i + 1], self.num_cities] for i in range(self.num_cities - 1)]

        with multiprocessing.Pool(processes=max_processes) as pool:
            shortest_paths = pool.map(self.find_min_path, starting_from_second_city)

        overall_min_distance = float('inf')
        overall_min_path = None

        for min_path, min_distance in shortest_paths:
            if min_distance < overall_min_distance:
                overall_min_distance = min_distance
                overall_min_path = min_path

        return overall_min_distance, overall_min_path

    def calculate_tour_distance(self, tour=None, opt_tour=False):
        if not opt_tour:
            current_distance = sum(
                self.distance_matrix[tour[i], tour[(i + 1) % self.num_cities]]
                for i in range(self.num_cities)
            )
        else:
            if len(self.opt_tour) > 0:
                for index, i in enumerate(range(self.num_cities)):
                    print(index, self.opt_tour[i] - 1, self.opt_tour[(i + 1) % self.num_cities] - 1)
                current_distance = sum(
                    self.distance_matrix[self.opt_tour[i] - 1, self.opt_tour[(i + 1) % self.num_cities] - 1]
                    for i in range(self.num_cities)
                )  # opt_tour indexing from 1, instead of 0
            else:
                return -1
        return current_distance

    def calculate_distance_matrix(self):
        """
        Calculates the distance matrix between cities.

        Returns:
            np.ndarray: A 2D array representing the distances between each pair of cities.
        """
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.linalg.norm(self.city_coords[i] - self.city_coords[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def reset(self,fix_start=False):
        """
        Resets the environment by randomly choosing a starting city and setting the state back to the initial state.

        Returns:
            dict: The initial state containing the current path and distances.
        """
        self.visited = [False] * self.num_cities
        self.path = []  # Travel path
        self.total_distance = 0
        self.step_distance = 0
        if fix_start:
            self.current_city = 0
        else:
            self.current_city = np.random.choice(self.num_cities)  # Randomly choose a start city
        self.path.append(self.current_city)  # Add the starting city to the path at first
        self.visited[self.current_city] = True
        return self.get_state()

    def step(self, action):
        """
        Takes a step in the environment, choosing a new city based on the action,
        and returns the next state, reward, and whether the task is complete.

        Args:
            action (int): The index of the city to visit next.

        Returns:
            tuple: A tuple containing the new state, reward, and a boolean indicating if the task is done.
        """
        # Compute the distance from the current city to the chosen city
        current_distance = self.distance_matrix[self.current_city, action]

        # If the city has already been visited, end the game
        if action in self.path:
            reward = -100  # Penalty
            done = True  # End the game
            self.step_distance = float('inf')
            self.total_distance += self.step_distance
        else:
            reward = -current_distance  # Reward is negative of the path length (shorter path is better)
            self.total_distance += current_distance
            self.visited[action] = True
            self.path.append(action)
            self.current_city = action
            self.step_distance = current_distance
            done = all(self.visited)  # Check if all cities have been visited

        if done:
            # On the last step, return to the starting city and calculate the distance back
            current_distance = self.distance_matrix[self.current_city, self.path[0]]
            self.current_city = self.path[0]  # Return to the starting city
            reward += -current_distance
            self.total_distance += current_distance
            self.step_distance += current_distance

        return self.get_state(), reward, done

    def get_state(self):
        """
        Returns the current state of the environment, including the step distance,
        total distance, the current path, and unvisited cities.

        Returns:
            dict: The current state of the environment.
        """
        return {
            "step_distance": self.step_distance,
            "total_distance": self.total_distance,
            "current_path":  copy.deepcopy(self.path),
            "current_city": self.current_city,
            "unvisited_cities": np.where(np.array(self.visited) == False)[0].tolist()
        }

    def render(self, disappear=False):
        """
        Renders the TSP path and related information using a scatter plot.
        This includes plotting the cities, the path taken, and the distances between cities.

        Displays:
            A plot showing the cities, the path, and the distance between cities.
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(self.city_coords[:, 0], self.city_coords[:, 1], c='red', label="Cities", zorder=5)

        for i, coord in enumerate(self.city_coords):
            plt.text(coord[0], coord[1], str(i), fontsize=12, ha='right', color='blue')

        for i in range(len(self.path) - 1):
            start, end = self.path[i], self.path[i + 1]
            start_coord = self.city_coords[start]
            end_coord = self.city_coords[end]
            plt.annotate('', xy=end_coord, xytext=start_coord,
                         arrowprops=dict(facecolor='black', edgecolor='black', width=0.5, alpha=0.7))
            # Draw the distance between cities along the path
            mid_point = (start_coord + end_coord) / 2
            plt.text(mid_point[0], mid_point[1], f'{self.distance_matrix[start, end]:.2f}', fontsize=10, color='green')

        total_dist_text = f"Total Distance: {self.total_distance:.2f}"
        plt.text(0.5, 0.05, total_dist_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=14,
                 color='purple')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('TSP Path')
        plt.grid(True)
        if disappear:
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Create the TSP environment
    NUM_CITIES = 10
    env = TSPEnv(num_cities=NUM_CITIES)

    # Reset the environment and get the initial state
    state = env.reset()

    print("Initial State:")
    print(state)
    done = False
    # Take a few steps by choosing cities to visit
    # Cities to visit (0-based indexing)
    for action in range(NUM_CITIES):
        new_state, reward, done = env.step(action)
        print(f"Step to city {action}: Reward = {reward:.2f}, Done = {done} PATH={env.path}")
        if done:
            break
    env.render()
