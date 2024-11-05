import gym
import os, json
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from itertools import permutations


class TSPEnv(gym.Env):
    """
    自定义的旅行商问题（TSP）环境
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_cities=10, render_mode=None, coordinates_file='city_coordinates.json'):
        super(TSPEnv, self).__init__()

        self.num_cities = num_cities
        self.render_mode = render_mode
        self.coordinates_file = coordinates_file

        # 动作空间：选择下一个要访问的城市（离散空间）
        self.action_space = spaces.Discrete(self.num_cities)

        # 观测空间：城市的坐标和已经访问的城市
        self._load_or_generate_coordinates()  # 加载或生成城市坐标

        # 状态空间：包括当前城市（0到num_cities-1）和已访问城市（多二进制）
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_cities + 1,), dtype=np.float32)

        self.fig, self.ax = None, None  # 用于渲染

        self.reset()

    def _load_or_generate_coordinates(self):
        # 检查是否存在坐标文件
        if os.path.exists(self.coordinates_file):
            with open(self.coordinates_file, 'r') as f:
                data = json.load(f)
                if len(data['city_coordinates']) == self.num_cities:
                    self.best_distance = data['best_distance']
                    self.best_route = data['best_route']
                    self.city_coordinates = np.array(data['city_coordinates'])
                    return
                else:
                    print(f"坐标文件中的城市数量不匹配，生成新的城市坐标。")

        # 随机生成城市坐标并保存
        city_coordinates = np.random.rand(self.num_cities, 2).tolist()
        self.best_distance, self.best_route = self.find_shortest_tsp_route(city_coordinates)
        self.city_coordinates = np.array(city_coordinates)
        dic_save = {'city_coordinates': city_coordinates,
                    'best_distance': self.best_distance,
                    'best_route': self.best_route}
        with open(self.coordinates_file, 'w') as f:
            json.dump(dic_save, f)

    def reset(self):
        # 重置环境状态
        self.current_step = 1
        self.done = False

        self.current_city = 0  # 0为起点
        self.visited = [0] * self.num_cities
        self.visited[self.current_city] = 1

        self.tour = [self.current_city]
        self.total_distance = 0.0

        return [self._get_state()]

    def _get_state(self):
        state = np.zeros(self.num_cities + 1)
        state[0] = self.current_city  # 当前城市
        state[1:] = self.visited  # 已访问的城市
        return state.astype(np.float32)  # 确保数据类型适合神经网络[1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]

    def find_shortest_tsp_route(self, locations):
        """
        Finds the shortest possible route for the Traveling Salesman Problem (TSP).

        Parameters:
            locations (list of list or tuple): A list of [x, y] coordinates representing each location.

        Returns:
            min_distance (float): The total distance of the shortest route.
            best_route (list): The order of location indices representing the shortest route, starting and ending at index 0.
        """
        # Convert the list of locations to a NumPy array for efficient computations
        locations = np.array(locations)
        num_locations = len(locations)

        # Precompute the distance matrix where distance_matrix[i][j] is the Euclidean distance between location i and j
        distance_matrix = np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2)

        # Initialize variables to store the minimum distance and the best route
        min_distance = float('inf')
        best_route = []

        # Generate all possible permutations of the indices from 1 to num_locations-1
        # Index 0 is fixed as the starting and ending point
        indices = list(range(1, num_locations))
        total_permutations = np.math.factorial(num_locations - 1)
        print(f"Total permutations to evaluate: {total_permutations}")

        # Iterate through each permutation to find the one with the smallest total distance
        for idx, perm in enumerate(permutations(indices), 1):
            # Calculate the total distance for the current permutation

            # Distance from the starting point to the first location in the permutation
            current_distance = distance_matrix[0][perm[0]]

            # Distance between consecutive locations in the permutation
            for i in range(len(perm) - 1):
                current_distance += distance_matrix[perm[i]][perm[i + 1]]

            # Distance from the last location back to the starting point
            current_distance += distance_matrix[perm[-1]][0]

            # Update the minimum distance and best route if the current one is better
            if current_distance < min_distance:
                min_distance = current_distance
                best_route = [0] + list(perm) + [0]

            # Optional: Print progress every 100,000 permutations
            if idx % 100000 == 0:
                print(f"Evaluated {idx} permutations...")

        return min_distance, best_route

    def step(self, action):
        if self.done:
            raise Exception("环境已终止，请先调用 reset()")

        if not self.action_space.contains(action):
            raise ValueError(f"无效的动作: {action}")

        if self.visited[action]:
            # 重复访问城市，给予较大的惩罚
            reward = -100.0
            self.done = True
            info = {"message": "重复访问城市"}
            return self._get_state(), reward, self.done, None, info

        # 计算从当前城市到目标城市的距离
        distance = np.linalg.norm(self.city_coordinates[self.current_city] - self.city_coordinates[action])

        self.total_distance += distance
        # print(f'distance={distance} total_distance={self.total_distance}')
        # 更新状态
        self.current_city = action
        self.visited[action] = 1
        self.tour.append(action)
        self.current_step += 1

        # 检查是否所有城市都已访问
        if all(self.visited):
            # 回到起始城市
            return_distance = np.linalg.norm(
                self.city_coordinates[self.current_city] - self.city_coordinates[self.tour[0]])

            self.total_distance += return_distance
            # print(f'return distance={return_distance} total_distance={self.total_distance}')
            self.current_step += 1
            self.tour.append(self.tour[0])
            # reward = -self.total_distance  # 总距离越短，奖励越高（负向）
            reward = -return_distance-distance
            self.done = True
            info = {"message": "完成巡回", "total_distance": self.total_distance}
        else:
            # 奖励为负的步长，鼓励尽快完成
            # reward = -self.total_distance
            reward = -distance
            self.done = False
            info = {}

        return self._get_state(), reward, self.done, None, info



    def render(self, disappear=True,mode='human'):
        if self.render_mode != 'human':
            return

        if self.fig is None and self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()

        # 绘制城市
        self.ax.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], c='blue')

        for i, (x, y) in enumerate(self.city_coordinates):
            self.ax.text(x, y, str(i), fontsize=12, ha='right')

        # 绘制带有方向的路径
        if len(self.tour) > 1:
            for i in range(len(self.tour) - 1):
                start = self.city_coordinates[self.tour[i]]
                end = self.city_coordinates[self.tour[i + 1]]
                arrow = FancyArrowPatch(start, end, arrowstyle='->', color='red', mutation_scale=10, linewidth=0.5)
                self.ax.add_patch(arrow)
        tour_distance=self.calculate_route_distance(self.city_coordinates,self.tour)
        self.ax.set_title(f"TSP Tour - Step {self.current_step}, Current Rout:{tour_distance},{self.tour}\nBest Rout:{self.best_distance},{self.best_route}",fontsize=8)
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_aspect('equal')
        if not disappear:
            plt.show()
        else:
            plt.pause(2)  # 使图形实时更新

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig, self.ax = None, None

    def calculate_route_distance(self,locations, route):
        """
        Calculates the total Euclidean distance of a given TSP route.

        Parameters:
            locations (list of list or tuple): A list of [x, y] coordinates representing each location.
            route (list of int): A list of indices representing the order to visit the locations.
                                 The route should start and end at index 0.

        Returns:
            total_distance (float): The total distance of the route.

        Raises:
            ValueError: If the route does not start and end at index 0 or does not visit all locations exactly once.
        """
        # Convert the list of locations to a NumPy array for efficient computations
        locations = np.array(locations)
        num_locations = len(locations)

        # Validate the route
        if not route:
            raise ValueError("The route is empty.")

        # if route[0] != 0 or route[-1] != 0:
        #     raise ValueError("The route must start and end at index 0.")

        # if len(route) != num_locations + 1:
        #     raise ValueError(f"The route must contain {num_locations + 1} indices (including the return to the start).")

        # if set(route) != set(range(num_locations)):
        #     raise ValueError("The route must visit all locations exactly once (excluding the return to the start).")

        # Initialize total distance
        total_distance = 0.0

        # Iterate through the route and sum the distances between consecutive points
        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            # Calculate Euclidean distance between current_city and next_city
            distance = np.linalg.norm(locations[current_city] - locations[next_city])
            total_distance += distance

        return total_distance


if __name__ == "__main__":
    import gym
    import time

    # 假设 TSPEnv 已经定义在当前作用域中
    # 如果定义在其他模块中，请使用相应的导入语句
    # from tsp_env import TSPEnv

    env = TSPEnv(num_cities=10, render_mode="human")
    observation = env.reset()

    done = False
    total_reward = 0.0

    try:
        while not done:
            # 随机选择下一个城市作为示例
            action = env.action_space.sample()
            observation, reward, _, done, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.5)  # 添加延时以便观察渲染效果
    finally:
        print(f"总奖励: {total_reward}")
        env.close()
