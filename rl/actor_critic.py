
import torch,random,time
from IPython import display
from rl.env.gym_env import GymEnv

def select_action(state):
    """
    This function selects an action based on the state.

    Parameters:
    state

    Returns:
    int: index of selected action
    """
    state = torch.FloatTensor(state).reshape(1, 4)
    #[1, 4] -> [1, 2]
    prob = model(state)

    # Select an action based on probabilities.
    action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_td = torch.optim.Adam(model_td.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    # Play N games, training once per game.
    for i in range(1000):
        # Play a game and obtain data

        #states -> [b, 4]
        #rewards -> [b, 1]
        #actions -> [b, 1]
        #next_states -> [b, 4]
        #overs -> [b, 1]
        states, rewards, actions, next_states, overs = get_data()

        #Caculate values and targets for TD
        #[b, 4] -> [b ,1]
        values = model_td(states)

        #[b, 4] -> [b ,1]
        targets = model_td(next_states) * 0.98
        #[b ,1] * [b ,1] -> [b ,1]
        targets *= (1 - overs)
        #[b ,1] + [b ,1] -> [b ,1]
        targets += rewards

        #td_error, do not gradient update
        #[b ,1] - [b ,1] -> [b ,1]
        delta = (targets - values).detach()#

        # Recalculate the probability for the corresponding action
        #[b, 4] -> [b ,2]
        probs = model(states)
        #[b ,2] -> [b ,1]
        probs = probs.gather(dim=1, index=actions)

        '''the derivative of the policy gradient algorithm
        Replace the "reward_sum" in the formula with temporal difference error (to achieve optimization),
        because the variance of "reward_sum" is relatively high.'''
        #[b ,1] * [b ,1] -> [b ,1] -> scala
        loss = (-probs.log() * delta).mean()
        loss_td = loss_fn(values, targets.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer_td.zero_grad()
        loss_td.backward()
        optimizer_td.step()

        if i % 100 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(i, test_result)

if __name__ == '__main__':

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
        torch.nn.Softmax(dim=1),
    )
    model_td = sequential = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    env = GymEnv()
    env.reset()
    train()
    test(True)