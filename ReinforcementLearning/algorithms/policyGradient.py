'''Policy gradient doesn't compute Q-values; it adjusts different action probabilities based on rewards.'''
import time, torch, random
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

        state = next_state

    return states, rewards, actions

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

    # Play N games, training once per game.
    for epoch in range(1000):
        # Play a game and obtain data
        states, rewards, actions = get_data()
        optimizer.zero_grad()
        reward_sum = 0

        # Starting from the last step
        for i in reversed(range(len(states))):

            # Cumulative feedback, calculated starting from the feedback of the last step.
            reward_sum *= 0.98
            reward_sum += rewards[i]

            # Recalculate the probabilities for corresponding actions.
            state = torch.FloatTensor(states[i]).reshape(1, 4)
            #[1, 4] -> [1, 2]
            prob = model(state)
            #[1, 2] -> scala
            prob = prob[0, actions[i]]

            # According to the derivative formula, the reversal of the sign is because we are calculating the loss,
            # so the optimization direction is opposite.
            loss = -prob.log() * reward_sum

            # Accumulate gradients.
            loss.backward(retain_graph=True)

        optimizer.step()

        if epoch % 100 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(epoch, test_result)



if __name__ == '__main__':

    type_network = ''

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
        torch.nn.Softmax(dim=1), # Probability distribution of actions.
    )
    env = GymEnv()
    env.reset()
    train()
    test(play=True)
