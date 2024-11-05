# 1. Create the TSP environment. The state consists of all points,
# where visited locations are set to 1 and unvisited locations are set to 0.
# Each environment state is represented as a vector, for example,
# [0, 0, 1, 0, 0, 0] indicates that there are 6 locations,
# the third location has been visited, while the others have not.

# 2. The action corresponds to the index of the maximum value returned
# by the Q-network, allowing the selection of the location with the
# highest Q-value among the 6 points.

# 3. The reward is calculated as the cumulative distance based on the
# order of visited locations, summing the distances between each pair of points.

# 4. The Q-value for visited locations is set to negative infinity,
# preventing revisits to those points.
import torch
import numpy as np
import random, pickle, os

import matplotlib.pyplot as plt
from itertools import permutations

from confs.path_conf import system_data_dir


class TSPEnvironment:
    def __init__(self, model_name, my_logger, num_locations=10):
        self.num_locations = num_locations
        self.logger = my_logger
        self.model_name = model_name

        self.save_dir = os.path.join(system_data_dir, self.model_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_path = f"{self.save_dir}/{model_name}_tsp_data.pkl"
        self.render_path = f"{self.save_dir}/{self.model_name}_TSP_path.png"


        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                # Check if the number of locations matches
                if data['num_locations'] == self.num_locations:
                    self.locations = data['locations']
                    self.distances = data['distances']
                    self.best_path = data['best_path']
                    self.logger.info("Loaded data from file.")
                else:
                    self.logger.info("Number of locations mismatch. Generating new data.")
                    self.locations = self.generate_random_locations()
                    self.distances = self.calculate_distances()
                    self.save_data()
        else:
            self.logger.info("No existing data file found. Generating new data.")
            self.locations = self.generate_random_locations()
            self.distances = self.calculate_distances()
            self.save_data()
        self.reset()

    def save_data(self):
        data = {
            'num_locations': self.num_locations,
            'locations': self.locations,
            'distances': self.distances,
            'best_path': self.best_path
        }


        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)

    def calculate_path_distance(self, index_array):
        # Clean the index_array #[0, 3, 7, 8, 8, 1, 8, 10, 2, 5, 4, 6, 9, 0]  这个路径要清洗
        cleaned_array = []
        for i, elem in enumerate(index_array):
            if elem not in cleaned_array or (i == len(index_array) - 1 and elem == 0):
                cleaned_array.append(elem)

        # Calculate the total distance
        total_distance = 0
        if len(cleaned_array) != len(self.locations) + 1:
            self.logger.info(f'-----生成的路径不对，cleaned_array={cleaned_array}  原始路径={index_array}')
        for i in range(len(cleaned_array) - 1):
            total_distance += self.distances[cleaned_array[i], cleaned_array[i + 1]]

        return total_distance

    def calculate_total_distance_from_first(self):
        n = self.distances.shape[0]  # 获取坐标的数量
        # 固定第一个点，生成剩余点的排列
        remaining_points = range(1, n)  # 剩余的点从1到n-1
        all_paths = permutations(remaining_points)  # 生成所有剩余点的排列
        path_distances = []  # 存储每条路径的总距离

        for path in all_paths:
            # 添加固定的第一个点
            full_path = (0,) + path + (0,)  # 形成完整路径，起点是0，终点也返回0
            total_distance = 0
            # 计算路径的总距离
            for i in range(len(full_path) - 1):
                total_distance += self.distances[full_path[i], full_path[i + 1]]  # 计算相邻点之间的距离
            path_distances.append((full_path, total_distance))  # 存储路径及其总距离

        # 按照总距离从小到大排序
        sorted_path_distances = sorted(path_distances, key=lambda x: x[1], reverse=False)

        for i in range(len(sorted_path_distances)):
            path, distance = sorted_path_distances[i]
            # 将计算结果拼接
            sorted_path_distances[i] = (path, distance)  # 只保留路径和距离

        return sorted_path_distances

    def generate_random_locations(self):
        return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(self.num_locations)]

    def calculate_distances(self):
        self.distances = np.zeros((self.num_locations, self.num_locations))
        for i in range(self.num_locations):
            for j in range(self.num_locations):
                self.distances[i][j] = np.sqrt((self.locations[i][0] - self.locations[j][0]) ** 2 +
                                               (self.locations[i][1] - self.locations[j][1]) ** 2)
        self.all_paths = self.calculate_total_distance_from_first()
        self.best_path = self.all_paths[0]
        return self.distances

    def reset(self):
        self.current_location = 0
        self.unvisited_locations = set(range(1, self.num_locations))
        self.visited_locations = [self.current_location]
        self.total_distance = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros(self.num_locations)
        state[self.visited_locations] = 1
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def step(self, action):
        big_penalty = -np.inf
        if action not in self.unvisited_locations:
            return self.get_state(), big_penalty, False

        distance_to_next = self.distances[self.current_location][action]
        self.unvisited_locations.remove(action)
        self.visited_locations.append(action)
        self.total_distance += distance_to_next
        self.current_location = action

        done = len(self.unvisited_locations) == 0
        if done:  # 要加上 终点到起点的距离
            distance_to_next += self.distances[action][0]

        reward = -distance_to_next
        return self.get_state(), reward, done

    def render(self, distance, path, true_distance, save=False):
        x, y = zip(*self.locations)
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, color="red", label="Locations")
        plt.plot(x, y, "o")

        # Add arrows to show visit order and display location values

        for i in range(1, len(path)):
            start_idx = path[i - 1]
            end_idx = path[i]
            plt.annotate(
                '', xy=self.locations[end_idx], xytext=self.locations[start_idx],
                arrowprops=dict(arrowstyle="->", color="blue", lw=1)
            )
            # Display the coordinates for each location
            plt.text(self.locations[start_idx][0], self.locations[start_idx][1],
                     f"{path[i - 1]}", fontsize=12, ha="right")

        # Display the last point's coordinates as well
        plt.text(self.locations[path[-1]][0], self.locations[path[-1]][1],
                 f"{path[-1]}", fontsize=12, ha="right")

        # Label the start and end points
        plt.annotate("Start", xy=self.locations[path[0]],
                     xytext=(self.locations[path[0]][0] - 0.2,
                             self.locations[path[0]][1] - 0.2),
                     color="green", weight="bold", fontsize=9)
        plt.annotate("End", xy=self.locations[path[-1]],
                     xytext=(self.locations[path[-1]][0] + 0.2,
                             self.locations[path[-1]][1] + 0.2),
                     color="red", weight="bold", fontsize=9)

        plt.legend()
        plt.title(f"TSP Path, distance={distance} true={true_distance}\n {path}", fontsize=9)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        # Save the figure if requested
        if save:
            dir = os.path.join(system_data_dir, self.model_name)
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(self.render_path)
        else:
            plt.show()