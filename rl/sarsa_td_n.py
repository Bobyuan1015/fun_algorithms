'''Here is the implementation code for the n-step SARSA algorithm.'''
import random
import numpy as np
from rl.env.cliff_env import CliffEnv

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



def get_td_error_list(next_row, next_col, next_action):
    """
    This function retrieves the scores for the 5 time steps.

    Parameters:
    state(next_row, next_col)
    action(next_action)

    Returns:
    int: index of selected action
    """
    # The initialized target is the score of the last state and the last action.
    target = Q[next_row, next_col, next_action]

    '''Calculate the target for each step.
    The target for each step is equal to the next step's target multiplied by 0.9, plus the reward for the current step.
    Backtracking in time from the end, the earlier the time step, the more accumulated information the target will have.
    [4, 3, 2, 1, 0]
    '''
    target_list = []
    for i in reversed(range(5)):
        target = 0.9 * target + reward_list[i] # S A R S A
        target_list.append(target)

    # Reverse the order of time steps.
    target_list = list(reversed(target_list))

    # Calculate the value for each step.
    value_list = []
    for i in range(5):
        row, col = state_list[i]
        action = action_list[i]
        value_list.append(Q[row, col, action])

    td_error_list = []
    for i in range(5):
        '''Calculate the update amount for each step.
        According to the Temporal Difference (TD) algorithm, q(s,a) = gamma*q(next_a,next_s) + reward.
        Here, you're calculating the difference between the two values; the closer it is to 0, the better.
        '''
        td_error = target_list[i] - value_list[i]

        # 0.1 is like the learning rate in DL
        td_error *= 0.1

        td_error_list.append(td_error)

    return td_error_list


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

        # Initialize three lists.
        state_list.clear()
        action_list.clear()
        reward_list.clear()

        # Loop until reaching the terminal state or falling into a trap.
        while env.get_state(row, col) not in ['terminal', 'trap']:

            # Execute an action.
            next_row, next_col, reward = env.move(row, col, action)
            reward_sum += reward

            # Determine a new action for the new position.
            next_action = select_action(next_row, next_col)

            # Record historical data.
            state_list.append([row, col])
            action_list.append(action)
            reward_list.append(reward)

            # Accumulate for 5 steps before starting to update parameters.
            if len(state_list) == 5:

                # Calculate the score.
                td_error_list = get_td_error_list(next_row, next_col, next_action)

                # Update the score only for the first step.
                row, col = state_list[0]
                action = action_list[0]
                update = td_error_list[0]

                Q[row, col, action] += update

                # Remove the first step, so that the list remains with 5 elements in the next loop iteration.
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)

            # Update the current position.
            row = next_row
            col = next_col
            action = next_action

        # After reaching the terminal state, update the remaining steps.
        for i in range(len(state_list)):
            row, col = state_list[i]
            action = action_list[i]
            update = td_error_list[i]
            Q[row, col, action] += update

        if epoch % 100 == 0:
            print(epoch, reward_sum)


if __name__ == '__main__':
    # Initialize the scores for taking each action in every cell, all initialized to 0, as there is no prior knowledge.
    Q = np.zeros([4, 12, 4])
    # Initialize three lists to store historical data of states, actions, and rewards, as they will be used for backtracking later.
    state_list = []
    action_list = []
    reward_list = []

    env = CliffEnv()
    train()
    env.final_result(Q)
