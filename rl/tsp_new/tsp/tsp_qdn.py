import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

# Create tsp Environment
class TSPEnvironment:
    def __init__(self, num_locations=10):
        self.num_locations = num_locations
        self.locations = self.generate_random_locations()
        self.distances = self.calculate_distances()
        self.stops = []
        self.reset()

    def generate_random_locations(self):
        return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(self.num_locations)]

    def calculate_distances(self):
        distances = np.zeros((self.num_locations, self.num_locations))
        for i in range(self.num_locations):
            for j in range(self.num_locations):
                distances[i][j] = np.sqrt((self.locations[i][0] - self.locations[j][0]) ** 2 +
                                          (self.locations[i][1] - self.locations[j][1]) ** 2)
        return distances

    def reset(self):
        self.current_location = 0
        self.unvisited_locations = set(range(1, self.num_locations))
        self.visited_locations = [self.current_location]
        self.total_distance = 0
        self.stops = []
        return self.current_location

    def step(self, action):
        if action not in self.unvisited_locations:
            raise ValueError("Invalid action!")
        distance_to_next_location = self.distances[self.current_location][action]
        self.unvisited_locations.remove(action)
        self.current_location = action
        self.visited_locations.append(self.current_location)
        self.total_distance += distance_to_next_location
        self.stops.append(action)

        done = len(self.unvisited_locations) == 0
        reward = -distance_to_next_location
        return self.current_location, reward, done

    def render(self, return_img=False):
        # import matplotlib

        # matplotlib.use('Agg')
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig = plt.figure(figsize=(7, 7))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Show stops
        x = np.array([i[0] for i in self.locations])
        y = np.array([i[1] for i in self.locations])
        ax.scatter(x, y, c="red", s=50)

        # Show START
        if len(self.stops) > 0:
            xy = self.locations[self.stops[0]]
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        # Show itinerary
        if len(self.stops) > 1:
            ax.plot(x[self.stops], y[self.stops], c="blue", linewidth=1, linestyle="--")

            # Annotate END
            xy = self.locations[self.stops[-1]]
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        plt.xticks([])
        plt.yticks([])

        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw_idle()
            canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()

# Create DQN Model
class DQN(nn.Module):
    def __init__(self, num_locations):
        super(DQN, self).__init__()
        self.num_locations = num_locations
        self.fc1 = nn.Linear(num_locations, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_locations)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Reinforcement Learning Algorithm
def dqn_algorithm(env, model, num_episodes=10, epsilon=0.1, gamma=0.9, batch_size=32):
    memory = []
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    statistics_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.eye(env.num_locations)[state].unsqueeze(0)

        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(list(env.unvisited_locations))
            else:
                with torch.no_grad():
                    q_values = model(state)
                    q_values.numpy()[0][env.visited_locations] = -np.inf
                    action = np.argmax(q_values)
                    # action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(int(action))
            next_state = torch.eye(env.num_locations)[next_state].unsqueeze(0)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))

            state = next_state

            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.cat(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards).float()
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones).float()

                q_values = model(states)
                next_q_values = model(next_states)
                max_next_q_values = torch.max(next_q_values, dim=1).values
                targets = rewards + gamma * max_next_q_values * (1 - dones)
                targets = targets.unsqueeze(1)

                q_values = q_values.gather(1, actions)
                loss = criterion(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        statistics_rewards.append(total_reward)
    plt.figure(figsize=(15, 3))
    plt.title("Rewards over training")
    plt.plot(statistics_rewards)
    plt.show()



# Visualize the Path
def plot_path(locations, path, title="tsp Solution"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(path)), path, marker='o')
    for i, loc in enumerate(locations):
        plt.annotate(f'{i}', (i, path[i]), textcoords="offset points", xytext=(0, 5), ha='center')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Main Function
if __name__ == "__main__":
    num_locations = 50
    env = TSPEnvironment(num_locations)
    model = DQN(num_locations)

    print("Training DQN model...")
    dqn_algorithm(env, model)

    print("Testing the model...")
    state = env.reset()
    state = torch.eye(num_locations)[state].unsqueeze(0)

    done = False
    path = [0]
    while not done:
        with torch.no_grad():
            q_values = model(state)
            print(env.visited_locations)
            q_values.numpy()[0][env.visited_locations] = -np.inf
            action = int(np.argmax(q_values))
            path.append(action)
            if action in env.visited_locations:
                print(f'action in {action}')

        next_state, _, done = env.step(action)
        state = torch.eye(num_locations)[next_state].unsqueeze(0)

    # Return to the starting location
    path.append(0)

    print("Path found:", path)
    # plot_path(env.locations, path)
    env.render()