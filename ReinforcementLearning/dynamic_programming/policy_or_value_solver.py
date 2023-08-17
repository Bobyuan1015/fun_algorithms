'''Dynamic programming algorithm requires complete knowledge of the environment, i.e., it's a model-based algorithm.'''
import numpy as np
from ReinforcementLearning.env.cliff_env import CliffEnv

def get_q_value(row, col, action):
    """
    This function calculates the score of taking an action in a given state, the Q(s,a).

    Parameters:
    row,col (int): state
    action (int): action

    Returns:
    int: The value of Q(s,q)
    """

    # Performing an action in the current state yields the next state and reward.
    next_row, next_col, reward = env.move(row, col, action)
    '''Calculating the score of the next state,
    take the score recorded in the values, with 0.9 as the discount factor, as future values are uncertain.'''
    value = env.values[next_row, next_col] * 0.9
    # If the next state is a terminal state or a trap, then the score of the next state is 0.
    if env.get_state(next_row, next_col) in ['trap', 'terminal']:
        value = 0
    # The score of an action is the reward, and plus the score of the next state
    return value + reward

def update_v_value(algorithm='policy_based'):
    """
    This function calculates Policy evaluation, V(s).

    Parameters:
    algorithm: algorithm type

    Returns:
    The values of V(s)
    """
    # Initialize a new set of values, reevaluate the scores of all cells, set them all to 0
    new_values = np.zeros([4, 12])
    # iterate through all cells
    for row in range(4):
        for col in range(12):

            # Calculate the scores for each of the 4 actions in the current cell
            action_value = np.zeros(4)
            # iterate through all actions, applying full exploration
            for action in range(4):
                action_value[action] = get_q_value(row, col, action)

            if algorithm == 'policy_based':
                '''Policy iteration'''

                # multiply the score of each action by its probability (transition probability)
                action_value *= pi[row, col]

                # finally, the score of this cell is the sum of scores of all actions in that cell.
                new_values[row, col] = action_value.sum()
            elif algorithm == 'value_based':
                '''Value iteration
                the only difference from the policy iteration:  action transitions are not required
                '''

                # for each cell, compute its score as the maximum score among all actions in that cell.
                new_values[row, col] = action_value.max()

    return new_values

def update_pi():
    """
    This function updates the policy function., Pi function.
    Recompute the probabilities for each action based on the Q values of each cell.

    Returns:
    The improved Pi function
    """
    # Reinitialize the probabilities of taking actions in each cell
    new_pi = np.zeros([4, 12, 4])
    # iterate through all cells, calculate the scores for each of the 4 actions in the current cell
    for row in range(4):
        for col in range(12):
            action_value = np.zeros(4)
            # iterate through all actions
            for action in range(4):
                action_value[action] = get_q_value(row, col, action)

            # compute how many actions achieve the maximum score in the current state
            count = (action_value == action_value.max()).sum()
            # evenly distribute probabilities among these actions.
            for action in range(4):
                if action_value[action] == action_value.max():
                    new_pi[row, col, action] = 1 / count
                else:
                    new_pi[row, col, action] = 0
    return new_pi

def train():
    """
    This function iterates in a loop between policy evaluation and policy improvement, seeking the optimal solution.

    Returns:
    The improved Pi function
    """
    global pi
    for _ in range(5):
        for _ in range(10):
            env.values = update_v_value('value_based')
        pi = update_pi()

if __name__ == '__main__':
    pi = np.ones([4, 12, 4]) * 0.25
    env = CliffEnv()   # create environment
    train()  # train an optimal model
    env.final_result(pi) # test the optimal model