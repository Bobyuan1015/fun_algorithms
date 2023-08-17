import torch,random,time
from IPython import display
from ReinforcementLearning.env.gym_env import GymEnv

def select_action(state):
    """
    This function selects an action based on the state.

    Parameters:
    state

    Returns:
    int: index of selected action
    """
    state = torch.FloatTensor(state).reshape(1, 3)
    mu, std = model(state)
    #action = random.normalvariate(mu=mu.item(), sigma=std.item())
    action = torch.distributions.Normal(mu, std).sample().item()

    return action

def get_data():
    '''Obtain data for a game session.'''
    states = []
    rewards = []
    actions = []
    next_states = []
    overs = []

    # Initialize the game.
    state = env.reset()
    # Play until the game is over.
    over = False
    while not over:

        action = select_action(state)
        next_state, reward, over, _ = env.step(action)

        # Record a data sample
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        overs.append(over)

        state = next_state

    # [b, 4]
    states = torch.FloatTensor(states).reshape(-1, 4)
    # [b, 1]
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    # [b, 1]
    actions = torch.LongTensor(actions).reshape(-1, 1)
    # [b, 4]
    next_states = torch.FloatTensor(next_states).reshape(-1, 4)
    # [b, 1]
    overs = torch.LongTensor(overs).reshape(-1, 1)
    return states, rewards, actions, next_states, overs


def get_advantages(deltas):
    '''advantage function'''
    advantages = []

    #Iterate backwards through the deltas.
    s = 0.0
    for delta in deltas[::-1]:
        s = 0.9 * 0.9 * s + delta
        advantages.append(s)

    advantages.reverse()
    return advantages


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_statu = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
        )

        self.fc_mu = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )

        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
        )

    def forward(self, state):
        state = self.fc_statu(state)

        mu = self.fc_mu(state) * 2.0
        std = self.fc_std(state)
        #Since this involves continuous actions, the mean and standard deviation of the entire distribution.
        return mu, std


def test(play):
    """
    This function test performance of the new policy.

    Parameters:
    play: display game

    Returns:
    reward_sum: Sum of rewards.
    """
    state = env.reset()

    reward_sum = 0
    over = False
    while not over:

        action = select_action(state)
        state, reward, over, _ = env.step(action)
        reward_sum += reward

        # skip frames and display animation.
        if play and random.random() < 0.2:
            display.clear_output(wait=True)
            env.show()
            time.sleep(1)
    return reward_sum

def train():
    """
    This function implements training process
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_td = torch.optim.Adam(model_td.parameters(), lr=5e-3)
    loss_fn = torch.nn.MSELoss()

    # Play N games, training M times per game.
    for epoch in range(3000):
        # Play a game and obtain data

        #states -> [b, 3]
        #rewards -> [b, 1]
        #actions -> [b, 1]
        #next_states -> [b, 3]
        #overs -> [b, 1]
        states, rewards, actions, next_states, overs = get_data()

        # Offset the rewards to facilitate training.
        rewards = (rewards + 8) / 8

        # Calculate values and targets.
        #[b, 3] -> [b, 1]
        values = model_td(states)

        #[b, 3] -> [b, 1]
        targets = model_td(next_states).detach()
        targets = targets * 0.98
        targets *= (1 - overs)
        targets += rewards
        '''Calculate advantages; here, "advantages" resemble the "reward_sum" in policy gradient.
        It's just that here we calculate not the reward, but the difference between the target and the value.'''
        #[b, 1]
        deltas = (targets - values).squeeze(dim=1).tolist()
        advantages = get_advantages(deltas)
        advantages = torch.FloatTensor(advantages).reshape(-1, 1)

        # Retrieve the probability of each action at every step.
        #[b, 3] -> [b, 1],[b, 1]
        mu, std = model(states)
        #Gaussian density probability.    [b, 1]
        old_probs = torch.distributions.Normal(mu, std)
        old_probs = old_probs.log_prob(actions).exp().detach()

        # Train the batch of data repeatedly for 10 epochs.
        for _ in range(10):
            # Recalculate the probabilities for each action at every step.
            #[b, 3] -> [b, 1],[b, 1]
            mu, std = model(states)
            #[b, 1]
            new_probs = torch.distributions.Normal(mu, std)
            new_probs = new_probs.log_prob(actions).exp()

            # Calculate the change in probabilities.
            #[b, 1] - [b, 1] -> [b, 1]
            ratios = new_probs / old_probs

            # Calculate both truncated and untruncated losses, and choose the smaller one.
            #[b, 1] * [b, 1] -> [b, 1]
            surr1 = ratios * advantages
            #[b, 1] * [b, 1] -> [b, 1]
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages

            loss = -torch.min(surr1, surr2)
            loss = loss.mean()

            # Recalculate values and compute temporal difference loss.
            values = model_td(states)
            loss_td = loss_fn(values, targets)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_td.zero_grad()
            loss_td.backward()
            optimizer_td.step()

        if epoch % 200 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(epoch, test_result)


if __name__ == '__main__':
    model = Model()

    model_td = torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    datas = []
    env = GymEnv('Pendulum-v1')
    env.reset()
    train()
    test(True)
