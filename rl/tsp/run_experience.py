import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rl.agents.cluster_agent import train_dqn, evaluate_dqn, greedy_tsp, nearest_neighbor_tsp, genetic_algorithm_tsp
from rl.envs.experience_env import TSPCLUSTEREnv


def generate_tsp_instances(num_instances, num_cities, seed=None):
    if seed is not None:
        np.random.seed(seed)
    instances = []
    for _ in range(num_instances):
        cities = np.random.rand(num_cities, 2) * 100  # 城市坐标在100x100的平面上
        instances.append(cities)
    return instances

def run_experiments(tsp_instances):
    results = []
    for idx, cities in enumerate(tsp_instances):
        print(f"Processing TSP Instance {idx + 1}/{len(tsp_instances)}")
        env = TSPCLUSTEREnv(cities)

        # 强化学习算法
        trained_policy = train_dqn(env, num_episodes=500)
        rl_path, rl_distance = evaluate_dqn(env, trained_policy)

        # 基线算法
        greedy_path, greedy_distance = greedy_tsp(cities)
        nn_path, nn_distance = nearest_neighbor_tsp(cities)
        ga_path, ga_distance = genetic_algorithm_tsp(cities)

        # 收集结果
        results.append({
            'Instance': idx + 1,
            'RL_Distance': rl_distance,
            'Greedy_Distance': greedy_distance,
            'NearestNeighbor_Distance': nn_distance,
            'GeneticAlgorithm_Distance': ga_distance
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # 示例：生成10个包含20个城市的TSP实例
    num_instances = 10
    num_cities = 20
    tsp_instances = generate_tsp_instances(num_instances, num_cities, seed=42)
    # 运行实验
    experiment_results = run_experiments(tsp_instances)

    # 保存结果
    experiment_results.to_csv('tsp_experiment_results.csv', index=False)
    print("实验完成，结果已保存至 'tsp_experiment_results.csv'")

    #———————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # 读取结果
    # experiment_results = pd.read_csv('tsp_experiment_results.csv')

    # 描述性统计
    print(experiment_results.describe())

    # 箱线图比较不同算法的路径长度
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=experiment_results[['RL_Distance', 'Greedy_Distance',
                                         'NearestNeighbor_Distance', 'GeneticAlgorithm_Distance']])
    plt.title('不同算法的TSP路径长度比较')
    plt.ylabel('路径长度')
    plt.savefig('tsp_algorithm_comparison_boxplot.png')
    plt.show()

    # 箱线图展示RL与基线算法的差异
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=experiment_results[['RL_Distance', 'GeneticAlgorithm_Distance']])
    plt.title('强化学习与遗传算法的路径长度比较')
    plt.ylabel('路径长度')
    plt.savefig('rl_vs_ga_comparison_boxplot.png')
    plt.show()

    # 箱线图展示所有基线算法的比较
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=experiment_results[['Greedy_Distance', 'NearestNeighbor_Distance',
                                         'GeneticAlgorithm_Distance']])
    plt.title('基线算法的TSP路径长度比较')
    plt.ylabel('路径长度')
    plt.savefig('baseline_algorithms_comparison_boxplot.png')
    plt.show()

    # 散点图展示RL与遗传算法的路径长度
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='GeneticAlgorithm_Distance', y='RL_Distance', data=experiment_results)
    plt.plot([experiment_results[['GeneticAlgorithm_Distance', 'RL_Distance']].min().min(),
              experiment_results[['GeneticAlgorithm_Distance', 'RL_Distance']].max().max()],
             [experiment_results[['GeneticAlgorithm_Distance', 'RL_Distance']].min().min(),
              experiment_results[['GeneticAlgorithm_Distance', 'RL_Distance']].max().max()],
             'r--')
    plt.title('强化学习与遗传算法路径长度对比')
    plt.xlabel('遗传算法路径长度')
    plt.ylabel('强化学习路径长度')
    plt.savefig('rl_vs_ga_scatterplot.png')
    plt.show()