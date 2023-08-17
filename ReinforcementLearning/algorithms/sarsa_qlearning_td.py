'''Here is the implementation code for the Q-learning/Sarsa algorithm.'''
import random
import numpy as np
from ReinforcementLearning.env.cliff_env import CliffEnv

def select_action(row, col):
    """
    This function selects an action based on the state.

    Parameters:
    state(row, col)

    Returns:
    int: index of selected action
    """
    # 1. choose a random action with a small probability.
    if random.random() < 0.1:
        return random.choice(range(4))

    # 2. Otherwise, choose the action with the highest score.
    return Q[row, col].argmax()


def get_update(alogrithm, row, col, action, reward, next_row, next_col, next_action):
    """
    This function updates the Q score; each update depends on the current cell, current action, reward, next cell, and the next cell's action -> SARSA.

    Parameters:
    alogrithm: Q learning or Sarsa
    state(row, col)
    action
    reward
    next_state(next_row，next_col)
    next_action

    Returns:
    value of Q's update
    """
    if alogrithm == 'sarsa':
        # Calculate the target.
        target = 0.9 * Q[next_row, next_col, next_action]
    elif alogrithm == 'Q':
        # The target is the highest score in the next cell
        target = 0.9 * Q[next_row, next_col].max()
    else:
        target = 0
    target += reward

    # Calculate the value by directly looking up the Q-table.
    value = Q[row, col, action]

    '''Calculate the update amount for each step.
    According to the Temporal Difference (TD) algorithm, q(s,a) = gamma*q(next_a,next_s) + reward.
    Here, you're calculating the difference between the two values; the closer it is to 0, the better.
    '''
    loss = target - value

    # 0.1 is like the learning rate in DL
    loss *= 0.1

    return loss


def train():
    """
    This function implements training process
    """
    for epoch in range(1500):
        # Initialize the current position.
        row = random.choice(range(4))
        col = 0

        # Initialize the first action.
        action = select_action(row, col)

        # Calculate the cumulative reward. The larger this value, the better.
        reward_sum = 0

        # Loop until reaching the terminal state or falling into a trap.
        while env.get_state(row, col) not in ['terminal', 'trap']:
            # Execute an action.
            next_row, next_col, reward = env.move(row, col, action)
            reward_sum += reward

            # Determine a new action for the new position.
            next_action = select_action(next_row, next_col)

            # Update score
            update = get_update('Q', row, col, action, reward, next_row, next_col, next_action)

            Q[row, col, action] += update

            # Update the current position.
            row = next_row
            col = next_col
            action = next_action

        if epoch % 150 == 0:
            print(epoch, reward_sum)


if __name__ == '__main__':
    # Initialize the scores for taking each action in every cell, all initialized to 0, because there is no prior knowledge.
    Q = np.zeros([4, 12, 4])
    env = CliffEnv()
    train()
    env.final_result(Q)
