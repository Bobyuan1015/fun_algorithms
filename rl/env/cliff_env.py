'''Cliff game.(environment)'''
import numpy as np

class CliffEnv(object):
    def __init__(self):
        # Initialize the probabilities of taking actions in each cell.
        self.values = np.zeros([4, 12])

    def show(self, row, col, action):
        """
        This function prints the game states.

        Parameters:
        row:
        col:
        action:
        """

        graph = [
            '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
            '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
            '□', '□', '□', '□', '□', '□', '□', '□', '□', '○', '○', '○', '○', '○',
            '○', '○', '○', '○', '○', '❤'
        ]
        action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
        graph[row * 12 + col] = action
        graph = ''.join(graph)
        for i in range(0, 4 * 12, 12):
            print(graph[i:i + 12])
        print()
        print()

    def get_state(self, row, col):
        """
        This function retrieves the state of a cell

        Parameters:
        row:
        col:

        Returns:
        str: state
        """
        if row != 3:
            return 'ground'
        if row == 3 and col == 0:
            return 'ground'
        if row == 3 and col == 11:
            return 'terminal'
        return 'trap'


    def move(self, row, col, action):
        """
        This function performs an action in a cell.

        Parameters:
        row,col (int): state
        action (int): action

        Returns:
        int: row, col, reward
        """
        # If currently in a trap or a terminal state, no action can be taken, and the feedback is always 0.
        if self.get_state(row, col) in ['trap', 'terminal']:
            return row, col, 0

        if action == 0: #↑
            row -= 1
        elif action == 1: #↓
            row += 1
        elif action == 2: #←
            col -= 1
        elif action == 3: #→
            col += 1
        else:
            pass
        # Moving outside of the map is not allowed.
        row = max(0, row)
        row = min(3, row)
        col = max(0, col)
        col = min(11, col)

        '''If it's a trap, the reward is -100; otherwise, it's -1.
        This forces the machine to end the game as soon as possible, as each step taken results in a deduction of one point.
        Ideally, the game should end by reaching the terminal state to avoid the penalty of -100 points.
        '''
        reward = -1
        if self.get_state(row, col) == 'trap':
            reward = -100
        return row, col, reward

    def final_result(self,pi):
        """
        This function prints the action tendencies for all cell

        Parameters:
        pi: policy of actions
        """
        for row in range(4):
            line = ''
            for col in range(12):
                action = pi[row, col].argmax()
                action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
                line += action
            print(line)


