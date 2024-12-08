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
    num_cities = 5
    num_actions = num_cities
    num_episodes = 2000

    env = TSPEnv(num_cities=num_cities)
    agent = QLearningAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance", state_space_config='visits')
    # agent = DQNAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance")
    # agent = ActorCriticAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance")
    # agent = PPOAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy="negative_distance")

    agent.train(env, num_episodes)
    # evaluate(agent)
    agent.plot_strategy()
    agent.save_results()
    agent.plot_results()
    agent.plot_q_value_changes()
    agent.plot_policy_evolution()
    agent.plot_action_frequencies()
    agent.plot_q_value_trends()