import time
from tqdm import tqdm
import os, sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(project_dir)
sys.path.append(project_dir)
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
    reward_policys = ['negative_distance','negative_distance_with_return_bonus','negative_distance_return_with_best_bonus','zero_step_final_bonus','positive_final_bonus','adaptive_reward','dynamic_penalty_reduction','segment_bonus',
                      'average_baseline','feedback_adjustment','heuristic_mst',
                      'curiosity_driven','diversity_driven'
                      ]
    state_policies = ['step','visits']
    total_len = len(reward_policys)*len(state_policies)
    count = 0
    with tqdm(total=len(reward_policys)*len(state_policies),desc="exp",unit="comb") as pbar:
        for r in reward_policys:
            for state in state_policies:
                agent = QLearningAgent(num_cities=num_cities, num_actions=num_actions, reward_strategy=r, state_space_config=state)
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
                pbar.update(0.5)
                time.sleep(1)

                print(f'{count} in {total_len} ')
                count += 1
