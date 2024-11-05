import numpy as np
import random
import time
import csv
import matplotlib.pyplot as plt


class TSP:
    def __init__(self, cities):
        self.cities = cities
        self.n = len(cities)
        self.path_history = []
        self.best_distance_history = float('inf')
        self.mean_distance = self.calculate_mean_distance()
        self.visited = []  # 用于记录访问的城市
        # self.R_history = []

    def reset(self):
        """重置访问状态和路径历史，返回初始状态"""
        self.visited = []
        self.path_history = []
        # if len(self.R_history) >10:
        #     print(self.R_history[-10:])
        # self.R_history = []
        return 0  # 初始状态为城市 0

    def distance(self, city1, city2):
        return np.linalg.norm(np.array(city1) - np.array(city2))

    def total_distance(self, path):
        # 检查路径是否包含所有城市
        if len(path) != self.n:
            raise ValueError("Path must contain all cities.")

        distance = 0
        for i in range(len(path)):
            distance += self.distance(self.cities[path[i]], self.cities[path[(i + 1) % len(path)]])
        distance += self.distance(self.cities[path[-1]], self.cities[path[0]])  # 回环
        return distance

    def calculate_mean_distance(self):
        total_distance = 0
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                total_distance += self.distance(self.cities[i], self.cities[j])
                count += 1
        return total_distance / count if count > 0 else float('inf')

    def step(self, action):
        self.visited.append(action)
        done = len(self.visited) == self.n
        current_state = self.visited[-2] if len(self.visited) > 1 else 0  # 当前状态是上一个访问的城市
        reward = -self.distance(self.cities[current_state], self.cities[action])

        if done:
            current_path_distance = self.total_distance(self.visited)
            reward += self.n * self.mean_distance  # 完成路径奖励

            if current_path_distance < self.best_distance_history:
                self.best_distance_history = current_path_distance
                reward += 1 * self.mean_distance  # 当前路径短于历史最佳，增加奖励
            else:
                reward -= 1 * self.mean_distance  # 当前路径长于历史最佳，减少奖励
            # self.R_history.append(reward)

        self.path_history.append(self.visited)

        return reward, done, current_state  # 返回奖励、完成状态和当前状态

class QLearningAgent:
    def __init__(self, tsp, alpha=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.9998, min_epsilon=0.01):
        self.tsp = tsp
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {(state, action): np.random.uniform(0, 0.1) for state in range(tsp.n) for action in range(tsp.n)}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, done):
        if done:
            new_q = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * reward
        else:
            next_actions = [a for a in range(self.tsp.n) if a not in self.tsp.visited]
            best_next_q = max([self.get_q_value(next_state, a) for a in next_actions], default=0)
            new_q = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (
                    reward + self.gamma * best_next_q)

        self.q_table[(state, action)] = new_q

    def act(self, state):
        unvisited = [a for a in range(self.tsp.n) if a not in self.tsp.visited]
        if np.random.rand() < self.epsilon:
            return np.random.choice(unvisited) if unvisited else None
        else:
            q_values = [self.get_q_value(state, a) for a in unvisited]
            return unvisited[np.argmax(q_values)] if q_values else None

    def calculate_q_value_changes(self, previous_q_table):
        # 计算当前Q表与上一轮的变化
        changes = []
        for (state, action), q_value in self.q_table.items():
            previous_q_value = previous_q_table.get((state, action), 0)  # 改为负无穷
            changes.append(abs(q_value - previous_q_value))
        return max(changes, default=0)

    def plot_q_values(self, min_q, max_q, mean_q):
        plt.figure(figsize=(10, 5))
        plt.plot(min_q, label='Min Q Value')
        plt.plot(max_q, label='Max Q Value')
        plt.plot(mean_q, label='Mean Q Value')
        plt.xlabel('Episode')
        plt.ylabel('Q Value')
        plt.title('Q Value Statistics Over Episodes')
        plt.legend()
        plt.grid()
        plt.show()

    def train(self, episodes, patience=30, stability_threshold=0.01):
        best_reward = float('-inf')
        best_distance = float('inf')
        best_path = []

        previous_q_table = self.q_table.copy()
        recent_changes = []
        # 用于存储 Q 值的统计数据
        min_q_values = []
        max_q_values = []
        mean_q_values = []
        for episode in range(episodes):
            current_state = self.tsp.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.act(current_state)
                reward, done, next_state = self.tsp.step(action)
                self.update_q_value(current_state, action, reward, next_state, done)
                current_state = next_state
                total_reward += reward

            current_path_distance = self.tsp.total_distance(self.tsp.visited)

            if total_reward > best_reward:
                best_reward = total_reward
                best_path = self.tsp.visited
            if current_path_distance < best_distance:
                best_distance = current_path_distance


            # 收集 Q 值统计
            q_values = list(self.q_table.values())
            min_q_values.append(min(q_values))
            max_q_values.append(max(q_values))
            mean_q_values.append(np.mean(q_values))
            print(f"Episode {episode}, Q-values: min={min(q_values)}, max={max(q_values)}, mean={np.mean(q_values)}")

            # 计算Q值变化
            q_change = self.calculate_q_value_changes(previous_q_table)
            recent_changes.append(q_change)

            # 保持最近的变化数
            if len(recent_changes) > patience:
                recent_changes.pop(0)

            # 判断是否停止
            if len(recent_changes) == patience and all(change < stability_threshold for change in recent_changes):
                print(f"Stopping training at episode {episode}: best_reward={best_reward}, "
                      f"best_distance={best_distance}, path={best_path}")
                break

            previous_q_table = self.q_table.copy()

            # ε 衰减
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            #
            # print(f"episodetest={episode} ε={self.epsilon} q_change={q_change}  R={total_reward} {current_path_distance} "
            #       f"{self.tsp.visited} best:{best_distance} {best_path}")
        # 绘制 Q 值变化曲线
        self.plot_q_values(min_q_values, max_q_values, mean_q_values)
        return best_path, best_distance,episode
# class QLearningAgent:
#     def __init__(self, tsp, alpha=0, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.01):
#         self.tsp = tsp
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay  # ε 衰减率
#         self.min_epsilon = min_epsilon      # 最小 ε 值
#         self.q_table = {}
#
#     def get_q_value(self, state, action):
#         return self.q_table.get((state, action), 0.0)
#
#     def update_q_value(self, state, action, reward, next_state, done):
#         if done:
#             new_q = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * reward
#         else:
#             next_actions = [a for a in range(self.tsp.n) if a not in self.tsp.visited]
#             best_next_q = max([self.get_q_value(next_state, a) for a in next_actions], default=0)
#             new_q = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (
#                     reward + self.gamma * best_next_q)
#
#         self.q_table[(state, action)] = new_q
#
#     def act(self, state):
#         unvisited = [a for a in range(self.tsp.n) if a not in self.tsp.visited]
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(unvisited) if unvisited else None
#         else:
#             q_values = [self.get_q_value(state, a) for a in unvisited]
#             return unvisited[np.argmax(q_values)] if q_values else None
#
#     def train(self, episodes, patience=30):
#         best_reward = float('-inf')
#         best_distance = float('inf')
#         best_path = []
#
#         reward_not_improving = 0
#         distance_not_improving = 0
#         generation = 0
#
#         for episode in range(episodes):
#             current_state = self.tsp.reset()
#             done = False
#             total_reward = 0
#
#             while not done:
#                 action = self.act(current_state)
#                 reward, done, next_state = self.tsp.step(action)
#                 self.update_q_value(current_state, action, reward, next_state, done)
#                 current_state = next_state
#                 total_reward += reward
#
#             current_path_distance = self.tsp.total_distance(self.tsp.visited)
#
#             if total_reward > best_reward:
#                 best_reward = total_reward
#                 reward_not_improving = 0
#                 best_path = self.tsp.visited
#             else:
#                 reward_not_improving += 1
#                 print(f"Stopping reward_not_improving  {reward_not_improving}")
#
#             if current_path_distance < best_distance:
#                 best_distance = current_path_distance
#                 distance_not_improving = 0
#             else:
#
#                 distance_not_improving += 1
#                 print(f"Stopping distance_not_improving  {distance_not_improving}")
#
#             if reward_not_improving >= patience and distance_not_improving >= patience:
#                 print(f"Stopping training at episode {episode}  ε={self.epsilon} best_reward={best_reward} best_distance={best_distance} {best_path}")
#                 generation = episode
#                 break
#
#             # ε 衰减
#             self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
#
#             print(f"{episode} ε={self.epsilon} R={total_reward} {current_path_distance} {self.tsp.visited} best:{best_distance} {best_path}")
#
#         return best_path, best_distance, generation


def genetic_algorithm(cities, patience=50, population_size=100, improvement_threshold=0.001):
    tsp_instance = TSP(cities)  # 在循环外初始化TSP实例
    population = [random.sample(range(len(cities)), len(cities)) for _ in range(population_size)]
    best_path = None
    best_distance = float('inf')
    generation = 0
    last_best_distance = float('inf')
    no_improvement_counter = 0  # 初始化无改进计数器

    while generation < MAX_EPOCHS:
        # 根据路径距离对种群进行排序
        population.sort(key=lambda path: tsp_instance.total_distance(path))
        current_best_distance = tsp_instance.total_distance(population[0])

        # 更新最佳路径和距离
        if current_best_distance < best_distance:
            best_path = population[0]
            best_distance = current_best_distance
            no_improvement_counter = 0  # 重置计数器
        else:
            no_improvement_counter += 1  # 增加计数器

        # 检查是否满足停止条件
        if no_improvement_counter >= patience:  # 连续50次没有改进
            print(f'genetic_algorithm best_distance={best_distance}  {best_path}, {no_improvement_counter}')
            break

        # 选择前一半的种群进行交叉
        next_population = population[:population_size // 2]
        while len(next_population) < population_size:
            parents = random.sample(next_population[:10], 2)
            crossover_point = random.randint(1, len(cities) - 1)
            # 生成子代路径
            child = parents[0][:crossover_point] + [city for city in parents[1] if city not in parents[0][:crossover_point]]
            next_population.append(child)

        population = next_population
        generation += 1

    # 确保路径回到起点
    best_distance = tsp_instance.total_distance(best_path)  # 重新计算距离
    best_path.append(best_path[0])  # 添加回环

    return best_path, best_distance, generation


# 模拟退火算法
def simulated_annealing(cities, patience=50,initial_temp=1000, cooling_rate=0.995, improvement_threshold=1e-6):
    tsp_instance = TSP(cities)  # 在循环外初始化TSP实例
    current_solution = random.sample(range(len(cities)), len(cities))
    current_distance = tsp_instance.total_distance(current_solution)
    best_solution = current_solution[:]
    best_distance = current_distance
    iterations = 0
    no_improvement_counter = 0

    temperature = initial_temp
    while temperature > 1 and iterations < MAX_EPOCHS:
        new_solution = current_solution[:]
        # 随机选择两个城市进行交换
        i, j = random.sample(range(len(cities)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        new_distance = tsp_instance.total_distance(new_solution)
        # 判断是否接受新解
        if new_distance < current_distance or random.uniform(0, 1) < np.exp((current_distance - new_distance) / temperature):
            current_solution = new_solution
            current_distance = new_distance

            # 更新最佳解
            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

        # 降低温度
        temperature *= cooling_rate
        iterations += 1

        # 提前退出条件
        if no_improvement_counter >= patience:
            print(f'simulated_annealing best_distance={best_distance}  {best_solution}, {no_improvement_counter}')
            break

    # 确保最佳解回到起点
    best_distance = tsp_instance.total_distance(best_solution)  # 重新计算距离
    best_solution.append(best_solution[0])  # 添加回环

    return best_solution, best_distance, iterations


# 蚁群算法
def ant_colony(cities, patience=50, n_ants=10, pheromone_evaporation=0.9, improvement_threshold=1e-6):
    tsp_instance = TSP(cities)  # 在循环外初始化TSP实例
    pheromone = np.ones((len(cities), len(cities))) * 0.1
    best_path = None
    best_distance = float('inf')
    iteration = 0
    no_improvement_counter = 0

    while iteration < MAX_EPOCHS:
        paths = []
        for ant in range(n_ants):
            path = [random.randint(0, len(cities) - 1)]  # 随机选择起点
            while len(path) < len(cities):
                pheromone_sum = np.sum(pheromone[path[-1]])
                if pheromone_sum == 0:
                    next_city = random.choice([i for i in range(len(cities)) if i not in path])
                else:
                    probabilities = pheromone[path[-1]] / pheromone_sum
                    probabilities = np.nan_to_num(probabilities)

                    # 设置已访问城市的概率为0
                    for city in range(len(cities)):
                        if city in path:
                            probabilities[city] = 0
                    probabilities = np.nan_to_num(probabilities)
                    probabilities /= np.sum(probabilities)

                    # 根据概率选择下一个城市
                    next_city = np.random.choice(range(len(cities)), p=probabilities)

                path.append(next_city)
            paths.append(path)

        for path in paths:
            distance = tsp_instance.total_distance(path)
            if distance < best_distance:
                best_path = path
                best_distance = distance
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

        pheromone *= pheromone_evaporation  # 信息素蒸发
        for path in paths:
            distance = tsp_instance.total_distance(path)
            for i in range(len(path)):
                pheromone[path[i]][path[(i + 1) % len(path)]] += 1 / distance

        iteration += 1

        # 提前退出条件
        if no_improvement_counter >= patience:
            print(f'ant_colony best_distance={best_distance}  {best_path}, {no_improvement_counter}')
            break

    # 确保最佳路径回到起点
    best_distance = tsp_instance.total_distance(best_path)  # 重新计算距离
    best_path.append(best_path[0])  # 添加回环

    return best_path, best_distance, iteration


# 比较实验
def run_experiments(cities, num_trials=1, output_file='results.csv'):
    metrics = {
        'q_learning': [],
        'genetic_algorithm': [],
        'simulated_annealing': [],
        'ant_colony': []
    }

    for trial in range(num_trials):
        start_time = time.time()
        tsp = TSP(cities)
        q_learning_agent = QLearningAgent(tsp)
        best_path, best_distance, iterations = q_learning_agent.train(episodes=100000)
        metrics['q_learning'].append((best_distance, time.time() - start_time, iterations))

        # 遗传算法
    #     start_time = time.time()
    #     best_ga_path, best_ga_distance, ga_generations = genetic_algorithm(cities)
    #     metrics['genetic_algorithm'].append((best_ga_distance, time.time() - start_time, ga_generations))
    #
    #     # 模拟退火
    #     start_time = time.time()
    #     best_sa_path, best_sa_distance, sa_iterations = simulated_annealing(cities)
    #     metrics['simulated_annealing'].append((best_sa_distance, time.time() - start_time, sa_iterations))
    #
    #     # 蚁群算法
    #     start_time = time.time()
    #     best_ac_path, best_ac_distance, ac_iterations = ant_colony(cities)
    #     metrics['ant_colony'].append((best_ac_distance, time.time() - start_time, ac_iterations))
    #
    # # 保存结果到CSV
    # with open(output_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Algorithm', 'Mean Distance', 'Min Distance', 'Max Distance', 'Std Distance',
    #                      'Mean Time (s)', 'Min Time (s)', 'Max Time (s)', 'Std Time (s)',
    #                      'Mean Iterations/Generations'])
    #
    #     for algorithm, results in metrics.items():
    #         distances = [result[0] for result in results]
    #         times = [result[1] for result in results]
    #         iterations = [result[2] for result in results]
    #
    #         writer.writerow([
    #             algorithm,
    #             np.mean(distances), np.min(distances), np.max(distances), np.std(distances),
    #             np.mean(times), np.min(times), np.max(times), np.std(times),
    #             np.mean(iterations), np.min(iterations), np.max(iterations)
    #         ])


# 示例用法
if __name__ == "__main__":
    MAX_EPOCHS = 100000
    np.random.seed(0)  # 为可重复性设置随机种子
    cities = np.random.rand(100, 2) * 100  # 生成10个城市的坐标
    run_experiments(cities, num_trials=1)
