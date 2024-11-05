import numpy as np
import random
'''
There are several slot machines, and the winning probabilities of each machine are unknown. 
Problem: The number of attempts is limited. The goal is to achieve the maximum return.
Solution:The number of attempts is divided into two parts: exploration and exploitation. 
1)The exploration part involves trying each machine as I am unaware of their respective reward returns. 
This helps in exploring their approximate winning probabilities. 
2)The exploitation part is based on known winning probabilities, where the machine with the highest probability of winning is chosen.
'''




def choose_one(method='greedy'):
    """
    This function is the core algorithmic function, offering three different methods for policy evaluation.

    Parameters:
    method(str): method type---greedy、decayed_random、ucb

    Returns:
    int: index of selected machine
    """

    if method == 'greedy':
        # Always choose to play the machine with the highest historical winning rate.
        if random.random() < 0.01:
            print('exploration')
            # Randomly choose for a better exploration.
            return random.randint(0, 9)

        print(f'exploitation    rewards ={rewards}')
        rewards_mean = [np.mean(i) for i in rewards]
        print(f'rewards_mean ={rewards_mean}')
        # Choose the arm with the highest estimated expected reward.
        return np.argmax(rewards_mean)
    elif method == 'decayed_random':
        # A greedy algorithm with a lower exploration behaviour.
        played_count = sum([len(i) for i in rewards])
        if random.random() < 1 / played_count:
            return random.randint(0, 9)
        # Compute the average reward for each slot machine.
        rewards_mean = [np.mean(i) for i in rewards]
        # Choose the arm with the highest estimated expected reward.
        return np.argmax(rewards_mean)
    elif method == 'ucb':
        # The upper confidence bound: explore more on the machines that have been played less.
        played_count = [len(i) for i in rewards]
        played_count = np.array(played_count)
        '''Calculate the upper confidence bound
        1.The numerator is the total number of plays, square rooted to slow down its growth rate
        2.The denominator is the number of plays for each slot machine, multiplied by 2 to speed up its growth rate
        3.As the number of plays increases, the denominator will quickly exceed the growth rate of the numerator, causing the score to decrease
        4.Specifically for each slot machine, the more it's played, the smaller the score, which means the weighted value of the UCB decreases
        5.Therefore, UCB measures the uncertainty of each slot machine; the greater the uncertainty, the higher the value of exploration.
        '''
        numerator = played_count.sum() ** 0.5
        denominator = played_count * 2
        '''This is a vector where machines that have been played less will have a smaller denominator, leading to a larger UCB.
        Slot machines that have been played less frequently will have high scores. 
        On the other hand, those have been played more frequently will have low scores.'''
        ucb = numerator / denominator

        ucb = ucb ** 0.5
        rewards_mean = [np.mean(i) for i in rewards]
        rewards_mean = np.array(rewards_mean)
        ucb += rewards_mean

        return ucb.argmax()
    elif method == 'Thompson':

        # Calculate the number of times each slot machine outputs 1, then add 1.
        count_1 = [sum(i) + 1 for i in rewards]

        # Calculate the number of times each slot machine outputs 0, then add 1.
        count_0 = [sum(1 - np.array(i)) + 1 for i in rewards]

        # Calculate the reward distribution using the beta distribution, which can be considered as the probability of winning in each slot machine.
        beta = np.random.beta(count_1, count_0)
        # The mean of the Beta distribution is α' / (α' + β'), which in this example is win / win+loss. This means that after considering the observed data,
        # we believe that the best estimate for the probability of win is the mean value.
        return beta.argmax()
    else:
        print(f'no sample method')
        return


def try_and_play():
    """
    This function implements exploration and exploitation to calculate the winning probabilities of each machine.
    """
    i = choose_one(method)

    reward = 0
    if random.random() < probs[i]:
        reward = 1  # win

    # Record the outcomes of the plays.
    rewards[i].append(reward)


def main():
    # Play N times.
    for _ in range(5000):
        try_and_play()

    # Expected optimal outcome.
    target = probs.max() * 5000

    # Actual achieved outcomes.
    result = sum([sum(i) for i in rewards])
    print(f'Winning rate of the machine：{probs}')
    return target, result


if __name__ == '__main__':
    # create 10 armed-bandits. Winning probabilities for each slot machine, uniformly distributed between 0 and 1.
    probs = np.random.uniform(size=10)

    # Record the payout of each slot machine.
    rewards = [[1] for _ in range(10)]
    method = 'ucb'
    target, result = main()
    print(f'Expected target={target}             Actual result={result}')