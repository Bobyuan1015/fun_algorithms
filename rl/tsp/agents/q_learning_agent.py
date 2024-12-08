
import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import itertools
from confs.path_conf import system_experience_dir
from rl.tsp.agents.agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self,
                 num_cities,
                 num_actions,
                 alpha=0.5,
                 gamma=0.5,
                 epsilon=0.5,
                 reward_strategy="negative_distance",
                 state_space_config="step",
                 episodes=1000,
                 save_q_every=100 ):

        self.num_cities = num_cities
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_strategy = reward_strategy
        self.state_space_config = state_space_config
        self.episodes = episodes
        self.save_q_every = save_q_every

        date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = f'{system_experience_dir}{date_string}/'
        os.makedirs(self.base_dir, exist_ok=True)

        self.model_path = self.base_dir + 'model_QLearningAgent.pkl'

        # Additional attributes adapted from QLearningTSP
        self.episode_rewards = []
        self.average_rewards = []
        self.cumulative_rewards = []
        self.q_table_snapshots = []
        self.action_frequencies = []
        self.iteration_strategies = []
        self.action_counts = {a: 0 for a in range(self.num_actions)}
        self.q_value_changes = defaultdict(list) if self.state_space_config == "visits" else {s: [] for s in
                                                                                              range(self.num_cities)}

        # strategy_matrix initialization
        # For "step" state space: matrix shape = (num_cities, num_actions)
        # For "visits" state space: dictionary with {state (tuple): {action: count}}
        if self.state_space_config == "step":
            self.strategy_matrix = np.zeros((self.num_cities, self.num_actions))
        elif self.state_space_config == "visits":
            self.strategy_matrix = {}
        else:
            raise ValueError("Invalid state_space_config. Choose 'step' or 'visits'.")

        # For tracking action frequency
        self.action_frequency = np.zeros((self.num_cities, self.num_actions))

        # History best distance from BaseAgent initialization
        self.history_best_distance = float('inf')

        super().__init__(num_cities, num_actions, alpha, gamma, epsilon, reward_strategy, self.model_path)

    def initialize_model(self):
        # Initialize Q-table according to state_space_config
        if self.state_space_config == "step":
            # Simple state: current city => q_table: num_cities x num_actions
            self.q_table = np.zeros((self.num_cities, self.num_actions))
        elif self.state_space_config == "visits":
            # Detailed state: visited cities as a tuple => q_table: dict
            # no lambda or partial due to save constraints, we will handle missing states dynamically
            self.q_table = {}
        else:
            raise ValueError("Invalid state_space_config. Choose 'step' or 'visits'.")
        print("Initialized new Q-learning model.")

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                if (saved_data["num_cities"] == self.num_cities and
                        saved_data.get("state_space_config", None) == self.state_space_config):
                    self.q_table = saved_data["q_table"]
                    self.history_best_distance = saved_data["history_best_distance"]
                    # Load additional attributes if available
                    self.episode_rewards = saved_data.get("episode_rewards", [])
                    self.average_rewards = saved_data.get("average_rewards", [])
                    self.cumulative_rewards = saved_data.get("cumulative_rewards", [])
                    self.q_table_snapshots = saved_data.get("q_table_snapshots", [])
                    self.action_frequencies = saved_data.get("action_frequencies", [])
                    self.iteration_strategies = saved_data.get("iteration_strategies", [])
                    self.action_counts = saved_data.get("action_counts", {a: 0 for a in range(self.num_actions)})
                    self.q_value_changes = saved_data.get("q_value_changes",
                                                          defaultdict(list) if self.state_space_config == "visits" else {s: [] for s in range(self.num_cities)})
                    self.strategy_matrix = saved_data.get("strategy_matrix",
                                                          np.zeros((self.num_cities,
                                                                    self.num_actions)) if self.state_space_config == "step" else {})
                    self.action_frequency = saved_data.get("action_frequency",
                                                           np.zeros((self.num_cities, self.num_actions)))
                    print("Loaded existing Q-learning model.")
                else:
                    print("Model file found, but configuration mismatch. Initializing new model.")
                    self.initialize_model()
        except FileNotFoundError:
            print(f"Model path {self.model_path} does not exist. Initializing new Q-learning model.")
            self.initialize_model()
        except Exception as e:
            print(f"Error loading model: {e}. Initializing new Q-learning model.")
            self.initialize_model()

    def save_model(self):
        saved_data = {
            "num_cities": self.num_cities,
            "q_table": self.q_table,
            "history_best_distance": self.history_best_distance,
            "state_space_config": self.state_space_config,
            "episode_rewards": self.episode_rewards,
            "average_rewards": self.average_rewards,
            "cumulative_rewards": self.cumulative_rewards,
            "q_table_snapshots": self.q_table_snapshots,
            "action_frequencies": self.action_frequencies,
            "iteration_strategies": self.iteration_strategies,
            "action_counts": self.action_counts,
            "q_value_changes": self.q_value_changes,
            "strategy_matrix": self.strategy_matrix,
            "action_frequency": self.action_frequency
        }
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(saved_data, f)
            print(f"Model saved with distance: {self.history_best_distance}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def get_greedy_action(self, state):
        # Greedy action depends on state representation
        if self.state_space_config == "step":
            current_city = self.state_to_index(state)
            return np.argmax(self.q_table[current_city])
        elif self.state_space_config == "visits":
            visited_tuple = self.state_to_index(state)
            q_values = self.q_table.get(visited_tuple, np.zeros(self.num_actions))
            return np.argmax(q_values)

    def state_to_index(self, state):
        # "step": state['current_city']
        # "visits": tuple of visited path
        if self.state_space_config == "step":
            return state['current_city']
        elif self.state_space_config == "visits":
            return tuple(state['current_path'])

    def update_strategy(self, state, action):
        # Update strategy_matrix
        if self.state_space_config == "step":
            current_city = state['current_city']
            self.strategy_matrix[current_city][action] += 1
        elif self.state_space_config == "visits":
            visited_tuple = tuple(state['current_path'])
            if visited_tuple not in self.strategy_matrix:
                self.strategy_matrix[visited_tuple] = {}
            if action not in self.strategy_matrix[visited_tuple]:
                self.strategy_matrix[visited_tuple][action] = 0
            self.strategy_matrix[visited_tuple][action] += 1

    def get_strategy_matrix(self):
        # Convert to a matrix if visits
        if self.state_space_config == "step":
            return self.strategy_matrix
        elif self.state_space_config == "visits":
            # For visits, create a matrix representation for visualization
            # We'll create a matrix of shape (num_cities, num_actions)
            # For visited states, use last visited city to position the counts
            strategy_matrix = np.zeros((self.num_cities, self.num_actions))
            for state, actions in self.strategy_matrix.items():
                current_city = state[-1]  # last visited city
                for action, count in actions.items():
                    strategy_matrix[current_city][action] = count
            return strategy_matrix

    def update(self, state, action, reward, next_state, done):
        # Q-update according to state_space_config
        if self.state_space_config == "step":
            current_index = self.state_to_index(state)
            next_index = self.state_to_index(next_state)
            current_q = self.q_table[current_index][action]
            best_next_action = np.argmax(self.q_table[next_index])
            td_target = reward + self.gamma * self.q_table[next_index, best_next_action] * (1 - done)
            td_error = td_target - current_q
            updated_q = current_q + self.alpha * td_error
            self.q_table[current_index, action] = updated_q
            self.q_value_changes[current_index].append(updated_q)
        elif self.state_space_config == "visits":
            current_tuple = self.state_to_index(state)
            next_tuple = self.state_to_index(next_state)
            current_q_values = self.q_table.get(current_tuple, np.zeros(self.num_actions))
            next_q_values = self.q_table.get(next_tuple, np.zeros(self.num_actions))
            current_q = current_q_values[action]
            best_next_action = np.argmax(next_q_values)
            td_target = reward + self.gamma * next_q_values[best_next_action] * (1 - done)
            td_error = td_target - current_q
            updated_q = current_q + self.alpha * td_error
            # Store updated q-values back
            new_q_values = current_q_values.copy()
            new_q_values[action] = updated_q
            self.q_table[current_tuple] = new_q_values
            self.q_value_changes[current_tuple].append(updated_q)

    def train(self, env, num_episodes):
        stable_episode = None
        for episode in range(num_episodes):
            state = env.reset()
            # For "step" config: state is simple (current_city)
            # For "visits" config: state includes visited path in state['current_path']
            total_distance = float('inf')  # start as inf
            episode_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                self.action_counts[action] += 1

                next_state, base_reward, done = env.step(action)
                reward_params = {
                    "distance": next_state['step_distance'],
                    "done": done,
                    "total_distance": next_state['total_distance'],
                    "visited": next_state['current_path'],
                    "env": env
                }
                reward = self.adjust_reward(reward_params)
                episode_reward += reward

                self.update(state, action, reward, next_state, done)

                # Update frequency and strategy
                if self.state_space_config == "step":
                    current_index = self.state_to_index(state)
                    self.action_frequency[current_index][action] += 1
                elif self.state_space_config == "visits":
                    # We can also track frequency by last visited city
                    current_city = state['current_city']
                    self.action_frequency[current_city][action] += 1

                self.update_strategy(state, action)

                state = next_state
                total_distance = next_state['total_distance']

            # Episode ended
            self.episode_rewards.append(episode_reward)
            self.cumulative_rewards.append(sum(self.episode_rewards))
            self.average_rewards.append(np.mean(self.episode_rewards))

            if len(self.episode_rewards) > 10:
                reward_diff = np.abs(self.episode_rewards[-1] - np.mean(self.episode_rewards[-10:]))
                if reward_diff < 1e-3 and stable_episode is None:
                    stable_episode = episode

            # Save best model if improved
            if total_distance < self.history_best_distance:
                self.history_best_distance = total_distance
                self.save_model()

            # Save snapshots
            if (episode + 1) % self.save_q_every == 0:
                # Snapshot q_table
                if self.state_space_config == "step":
                    self.q_table_snapshots.append(self.q_table.copy())
                elif self.state_space_config == "visits":
                    # Sample a portion of the q_table
                    # Convert dict items to list and slice
                    items_list = list(self.q_table.items())
                    sampled_states = items_list[:100] if len(items_list) > 100 else items_list
                    sampled_q_table = {state: q.copy() for state, q in sampled_states}
                    self.q_table_snapshots.append(sampled_q_table)
                # track action frequencies
                frequencies = [count / (episode + 1) for count in self.action_counts.values()]
                self.action_frequencies.append(frequencies)
                # track iteration strategies
                strategy_matrix = self.get_strategy_matrix()
                self.iteration_strategies.append(strategy_matrix)
                self.save_model()

        print(f"Training converged at episode: {stable_episode}")

    def save_results(self):
        # Save csv results into self.base_dir
        pd.DataFrame({"Episode": range(1, len(self.episode_rewards) + 1),
                      "Reward": self.episode_rewards}).to_csv(self.base_dir + "episode_rewards.csv", index=False)
        pd.DataFrame({"Episode": range(1, len(self.cumulative_rewards) + 1),
                      "Cumulative Reward": self.cumulative_rewards}).to_csv(self.base_dir + "cumulative_rewards.csv",
                                                                            index=False)
        pd.DataFrame({"Episode": range(1, len(self.average_rewards) + 1),
                      "Average Reward": self.average_rewards}).to_csv(self.base_dir + "average_rewards.csv",
                                                                      index=False)

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards, label="Episode Rewards")
        plt.plot(range(1, len(self.average_rewards) + 1), self.average_rewards, label="Average Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.title("Training Progress")
        plt.grid(True)
        plt.savefig(self.base_dir + "training_progress.png")
        plt.show()

    def plot_q_value_changes(self):
        plt.figure(figsize=(12, 6))
        # Use a color map that isn't black
        # For visits, sample a few states; for step, plot all
        if self.state_space_config == "step":
            for state, q_values in self.q_value_changes.items():
                if len(q_values) > 0:
                    plt.plot(q_values, label=f"State {state}")
        elif self.state_space_config == "visits":
            # Sample a few states
            sampled_states = list(itertools.islice(self.q_value_changes.items(), 10))
            for state, q_values in sampled_states:
                if len(q_values) > 0:
                    plt.plot(q_values, label=f"State {state}")
        plt.xlabel("Updates")
        plt.ylabel("Q-value")
        plt.legend()
        plt.title("Q-value Changes Over Time")
        plt.grid(True)
        plt.savefig(self.base_dir + "q_value_changes.png")
        plt.show()

    def plot_policy_evolution(self):
        if not self.q_table_snapshots:
            print("No Q-table snapshots to plot.")
            return

        # Plot each snapshot individually
        for i, q_table in enumerate(self.q_table_snapshots):
            plt.figure(figsize=(10, 8))
            if self.state_space_config == "step":
                plt.imshow(q_table, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Q-table at Snapshot {i + 1}")
                plt.xlabel("Action")
                plt.ylabel("State")
            elif self.state_space_config == "visits":
                # For visits, q_table is a dict; plot a sample
                sampled_states = list(q_table.items())
                sampled_q = np.array([q for _, q in sampled_states])
                plt.imshow(sampled_q, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Q-table Snapshot {i + 1} (Sampled States)")
                plt.xlabel("Action")
                plt.ylabel("Sampled State Index")
            plt.savefig(self.base_dir + f"q_table_snapshot_{i + 1}.png")
            plt.show()

    def plot_action_frequencies(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.action_frequency, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.title("Action Frequency Heatmap")
        plt.xlabel("Action")
        plt.ylabel("State")
        plt.savefig(self.base_dir + "action_frequency_heatmap.png")
        plt.show()

    def plot_q_value_trends(self):
        if self.state_space_config == "step":
            x, y = np.meshgrid(range(self.num_actions), range(self.num_cities))
            q_values = self.q_table
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, q_values, cmap='viridis')
            ax.set_title("Q-value Trends")
            ax.set_xlabel("Action")
            ax.set_ylabel("State")
            ax.set_zlabel("Q-value")
            plt.savefig(self.base_dir + "q_value_trends.png")
            plt.show()
        elif self.state_space_config == "visits":
            # Aggregate Q-values for visits scenario
            if not self.q_table_snapshots:
                print("No Q-table snapshots available for plotting Q-value trends.")
                return
            # Take last snapshot
            last_snapshot = self.q_table_snapshots[-1]
            # Aggregate Q-values
            aggregate_q = np.zeros(self.num_actions)
            count = 0
            for q_values in last_snapshot.values():
                aggregate_q += q_values
                count += 1
            if count > 0:
                aggregate_q /= count
            plt.figure(figsize=(10, 6))
            plt.bar(range(self.num_actions), aggregate_q, color='skyblue')
            plt.xlabel("Action")
            plt.ylabel("Average Q-value")
            plt.title("Average Q-value per Action (Aggregated)")
            plt.savefig(self.base_dir + "q_value_trends_aggregated.png")
            plt.show()

    def get_policy(self):
        policy = {}
        if self.state_space_config == "step":
            for state in range(self.num_cities):
                best_action = np.argmax(self.q_table[state])
                policy[state] = best_action
        elif self.state_space_config == "visits":
            for state, q_values in self.q_table.items():
                best_action = np.argmax(q_values)
                policy[state] = best_action
        return policy

    def plot_strategy(self):
        # Plot each strategy_matrix in iteration_strategies
        # Each entry in iteration_strategies is a matrix
        for i, strategy in enumerate(self.iteration_strategies):
            plt.figure(figsize=(8, 6))
            plt.imshow(strategy, cmap='viridis', interpolation='nearest')
            plt.title(f'Strategy at Iteration {i * 100}')
            plt.xticks(ticks=np.arange(self.num_actions), labels=np.arange(self.num_actions))
            plt.yticks(ticks=np.arange(self.num_cities), labels=np.arange(self.num_cities))
            plt.xlabel('Destination City (Action)')
            plt.ylabel('Current City (State)')
            plt.colorbar()
            plt.savefig(self.base_dir + f"strategy_iteration_{i * 100}.png")
            plt.show()


