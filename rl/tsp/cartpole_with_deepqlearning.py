#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""
# Usual libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import gym
from collections import deque
import argparse

# Custom RL library
from rl.agents.a2c_agent import ActorCriticAgent
from rl.agents.ppo_agent_functional import PPOAgent
from rl.agents.sarsa_agent import SarsaAgent
from rl.agents.dqn_agent import DQNAgent
from rl import utils
from rl.envs.tsp import TSPEnv

# Constants
N_EPISODES = 100000
MAX_STEPS = 1000
RENDER = False
RENDER_EVERY = 50
AVG_REWARD_WINDOW = 1  # Window size for average reward


def train_agent(agent, env, is_tsp=False):
    """Train the PPO agent in the given environment."""
    rewards = []
    recent_rewards = deque(maxlen=AVG_REWARD_WINDOW)

    for i_episode in range(1, N_EPISODES + 1):
        state = env.reset()[0]
        total_reward = 0

        for i_step in range(1, MAX_STEPS + 1):
            # Render the environment at specified intervals
            if RENDER and (i_episode % RENDER_EVERY == 0):
                frame = env.render()
                plt.imshow(frame)
                plt.axis('off')
                plt.pause(0.001)

            # Agent selects action
            action, log_prob = agent.act(state)

            # Environment responds to action
            next_state, reward, done, truncated, info = env.step(action)

            # Modify reward: Penalize termination
            if done:
                if is_tsp:
                    env.reset()  # Game over due to a bad action, or you will be alive.
                    adjusted_reward = reward
                else:
                    adjusted_reward = -10
            else:
                adjusted_reward = reward  # or another strategy, e.g., +1 per step

            # Store experience in memory
            agent.remember(state, action, log_prob, adjusted_reward, next_state, done)

            # Update state and accumulate reward
            state = next_state
            total_reward += adjusted_reward  # Reflects the penalty if done

            if done:
                print(f"Episode {i_episode}/{N_EPISODES} finished after {i_step} timesteps - epsilon: {agent.epsilon:.2f}")
                break

        # Store and track rewards
        rewards.append(total_reward)
        recent_rewards.append(total_reward)

        # Calculate and print average reward over the window
        if len(recent_rewards) == AVG_REWARD_WINDOW:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {i_episode}: Average Reward over last {AVG_REWARD_WINDOW} episodes: {avg_reward:.2f}")

            # Update best average reward and save model if improved
            agent.update_best_reward(avg_reward)

        # Train the agent after each episode
        agent.train()

    # Close the environment after training
    env.close()

    # Plot the training rewards
    utils.plot_average_running_rewards(rewards)


def evaluate_agent(agent, env, n_episodes=10, deterministic=True,is_tsp=False):
    """Evaluate the trained PPO agent in the given environment."""
    rewards = []

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step = 0

        while not done and step < MAX_STEPS:
            if not is_tsp:
                env.render()  # Render in real-time

            # Agent selects action (deterministic for evaluation)
            action, _ = agent.act(state,test=True)
            state, reward, done, truncated, info = env.step(action)

            # Accumulate reward
            total_reward += reward
            print('r/R:',reward,total_reward)
            step += 1

            if done:
                print(f"Evaluation Episode {i_episode}/{n_episodes} finished after {step} timesteps with reward {total_reward}")
                rewards.append(total_reward)
                break
        if is_tsp:
            env.render(disappear=False)  # Render in real-time

    # Close the environment after evaluation
    env.close()

    # Calculate and print average evaluation reward
    avg_eval_reward = np.mean(rewards)
    print(f"Average Evaluation Reward over {n_episodes} episodes: {avg_eval_reward:.2f}")


def create_env(render=False, is_tsp=False):
    """Create and return the CartPole-v1 environment."""
    if is_tsp:
        env = TSPEnv(num_cities=10, render_mode="human")
    else:
        if render:
            env = gym.make("CartPole-v1", render_mode="human")
        else:
            env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    return env, state_size, action_size


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent on CartPole-v1.")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help="Mode to run: 'train' to train the agent, 'evaluate' to evaluate the trained agent.")
    parser.add_argument('--episodes', type=int, default=1,
                        help="Number of episodes to evaluate (only used in evaluate mode).")
    args = parser.parse_args()
    is_tsp = True
    # is_tsp = False
    # args.mode = 'evaluate'
    # Determine rendering based on mode
    render = True if args.mode == 'evaluate' else False

    # Create the environment
    env, state_size, action_size = create_env(render, is_tsp)

    # agent = DQNAgent(state_size, action_size) #done
    agent = ActorCriticAgent(state_size, action_size)
    # agent = PPOAgent(state_size, action_size)
    if args.mode == 'train':
        train_agent(agent, env, is_tsp=is_tsp)
    elif args.mode == 'evaluate':
        if not os.path.exists(agent.model_path):
            print(f"No saved model found at {agent.model_path}. Please train the agent first.")
            sys.exit(1)
        evaluate_agent(agent, env, n_episodes=args.episodes,is_tsp=is_tsp)

        # Ensure all plots are closed
    plt.close('all')


if __name__ == "__main__":
    main()
# python your_script.py --mode train
# python your_script.py --mode evaluate --episodes 10
