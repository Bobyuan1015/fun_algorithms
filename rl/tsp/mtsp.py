import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

# Create mTSP Environment
class MTSPEnvironment:
    def __init__(self, num_salesmen=2, num_locations=10):
        self.num_salesmen = num_salesmen
        self.num_locations = num_locations
        self.locations = [i for i in range(1, num_locations + 1)]
        self.reset()

    def reset(self):
        self.current_locations = [0] * self.num_salesmen
        self.visited_locations = [[False] * self.num_locations for _ in range(self.num_salesmen)]
        for i in range(self.num_salesmen):
            self.visited_locations[i][self.current_locations[i]] = True
        self.steps_taken = 0
        return self.current_locations

    def step(self, actions):
        if len(actions) != self.num_salesmen:
            raise ValueError("Invalid number of actions!")

        rewards = []
        for i in range(self.num_salesmen):
            action = actions[i]
            if not 0 <= action < self.num_locations:
                raise ValueError(f"Invalid action for salesman {i}!")

            if self.visited_locations[i][action]:
                reward = -10  # Penalize revisiting a location
            else:
                reward = -1

            self.current_locations[i] = action
            self.visited_locations[i][action] = True
            rewards.append(reward)

        self.steps_taken += 1
        done = all(all(v) for v in self.visited_locations) or self.steps_taken >= self.num_locations
        return self.current_locations, rewards, done

# Create DQN Model
class DQN(nn.Module):
    def __init__(self, num_salesmen, num_locations):
        super(DQN, self).__init__()
        self.num_salesmen = num_salesmen
        self.num_locations = num_locations
        self.fc1 = nn.Linear(num_salesmen * num_locations, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_salesmen * num_locations)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Reinforcement Learning Algorithm
def dqn_algorithm(env, model, num_episodes=1000, epsilon=0.1, gamma=0.9, batch_size=32):
    memory = []
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        states = [env.reset()]
        states = torch.tensor(states).flatten().unsqueeze(0)

        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                actions = [random.randint(0, env.num_locations - 1) for _ in range(env.num_salesmen)]
            else:
                with torch.no_grad():
                    q_values = model(states)
                    q_values = q_values.reshape(env.num_salesmen, -1)
                    actions = [torch.argmax(q_values[i]).item() for i in range(env.num_salesmen)]

            next_states, rewards, done = env.step(actions)
            next_states = torch.tensor(next_states).flatten().unsqueeze(0)
            total_reward += sum(rewards)

            memory.append((states, actions, rewards, next_states, done))

            states = next_states

            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.cat(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards).float()
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones).float()

                q_values = model(states)
                q_values = q_values.reshape(env.num_salesmen, -1)
                next_q_values = model(next_states)
                next_q_values = next_q_values.reshape(env.num_salesmen, -1)

                targets = rewards + gamma * torch.max(next_q_values, dim=1).values * (1 - dones)
                targets = targets.unsqueeze(1)

                q_values = q_values.gather(1, actions)
                loss = criterion(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model

# Visualize the Path
def plot_path(locations, paths, title="mTSP Solution"):
    plt.figure(figsize=(8, 6))
    for i, path in enumerate(paths):
        x = [locations[p][0] for p in path]
        y = [locations[p][1] for p in path]
        plt.plot(x, y, marker='o', label=f"Salesman {i + 1}")
    for i, loc in enumerate(locations):
        plt.annotate(f'{i}', (loc[0], loc[1]), textcoords="offset points", xytext=(0, 5), ha='center')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Function
if __name__ == "__main__":
    num_salesmen = 2
    num_locations = 10
    env = MTSPEnvironment(num_salesmen, num_locations)
    model = DQN(num_salesmen, num_locations)

    print("Training DQN model...")
    model = dqn_algorithm(env, model)

    print("Testing the model...")
    states = [env.reset()]
    states = torch.tensor(states).flatten().unsqueeze(0)

    done = False
    paths = [[0] for _ in range(num_salesmen)]
    while not done:
        with torch.no_grad():
            q_values = model(states)
            q_values = q_values.reshape(num_salesmen, -1)
            actions = [torch.argmax(q_values[i]).item() for i in range(num_salesmen)]

        next_states, _, done = env.step(actions)
        next_states = torch.tensor(next_states).flatten().unsqueeze(0)
        for i in range(num_salesmen):
            paths[i].append(next_states[0][i].item())

        states = next_states

    print("Paths found:", paths)
    plot_path(env.locations, paths)