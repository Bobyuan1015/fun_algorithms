# 设置环境和训练参数
from rl.env.tsp_env import TSPEnv
from rl.tsp.agents.actor_critic_agent import ActorCriticAgent
from rl.tsp.agents.dqn_agent import DQNAgent
from rl.tsp.agents.ppo_agent import PPOAgent
from rl.tsp.agents.q_learning_agent import QLearningAgent


def evaluate(agent):
    num_episodes = 2
    for i in range(num_episodes):
        state = env.reset()
        end = False
        while not end:
            a = agent.choose_action(state, use_sample=False)
            next_state, base_reward, end = env.step(a)
            print(next_state, base_reward, end)
        env.render(disappear=True)


if __name__ == '__main__':
    num_cities = 10  # 假设有10个城市
    num_actions = num_cities  # 每个城市都是一个动作
    num_episodes = 100000  # 训练1000个回合

    # 创建环境和智能体
    env = TSPEnv(num_cities=num_cities)
    # agent = QLearningAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance")
    # agent = DQNAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance")
    # agent = ActorCriticAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance")
    agent = PPOAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance")

    # 开始训练
    agent.train(env, num_episodes)
    # evaluate(agent)
