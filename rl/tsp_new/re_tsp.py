import numpy as np
import matplotlib.pyplot as plt

class TSPEnv:
  def __init__(self, distances, points):
    self.distances = distances
    self.n_cities = distances.shape[0]
    self.points = points
    self.current_city = 0

  def reset(self):
    self.current_city = 0
    return self.current_city

  def step(self, action):
    reward = -self.distances[self.current_city, action]
    self.current_city = action
    return self.current_city, reward, False, {}

def main():
  # TSP 实例
  distances = np.array([[0, 10, 20, 30],
                       [10, 0, 40, 50],
                       [20, 40, 0, 60],
                       [30, 50, 60, 0]])

  # 原始点
  points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])

  # 强化学习参数
  gamma = 0.9
  epsilon = 0.1

  # 初始化 Q 表
  q_table = np.zeros((4, 4))

  # 定义 env 变量
  env = TSPEnv(distances, points)

  # 训练
  for episode in range(1000):
    # 初始化状态
    state = env.reset()

    # 贪婪策略
    while True:
      action = np.argmax(q_table[state, :])
      next_state, reward, over, _ = env.step(action)

      # 更新 Q 表
      q_table[state, action] += gamma * (reward + np.max(q_table[next_state, :]) - q_table[state, action])

      state = next_state

      if over:
        break

  # 测试
  state = env.reset()
  total_distance = 0
  path = []
  while True:
    action = np.argmax(q_table[state, :])
    next_state, reward, over, _ = env.step(action)

    total_distance += reward
    path.append(state)
    state = next_state

    if over:
      path.append(state)
      break

  # 绘制路径
  plt.plot(points[:, 0], points[:, 1], "ro")
  for i in range(len(path) - 1):
    plt.plot([points[path[i], 0], points[path[i+1], 0]], [points[path[i], 1], points[path[i+1], 1]], "b-")

  plt.show()

if __name__ == "__main__":
  main()
