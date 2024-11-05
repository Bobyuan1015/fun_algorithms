'''SAC (Soft Actor-Critic) is a reinforcement learning algorithm designed to address problems in continuous action spaces.
It combines the Actor-Critic architecture with maximum entropy policy optimization, aiming to achieve stable and efficient
 policy learning.
1.Maximum Entropy Policy: SAC introduces the concept of maximum entropy policy, which maintains a balance between
performance and exploration by adding randomness to the policy. This enhances the algorithm's exploration capabilities.

2.Q-Networks: Similar to Double Q-Learning, SAC employs two Q-networks to mitigate the problem of overestimation,
resulting in more stable action value estimation.

3.Automatically Adjusted Temperature Parameter: SAC introduces a temperature parameter to balance policy exploration and maximizing
the expected cumulative reward. By optimizing this temperature parameter, SAC adapts the level of policy randomness and exploration.

4.Handling Discrete Action Spaces: SAC can also be adapted for problems with both continuous and discrete action spaces
by discretizing the continuous action space.

Maximize the Q-function while maximizing the entropy of the Q-function.     max->Q(a,s) + alpha * H[Q(s,*)]
-->The advantage of entropy is that it allows the machine to explore different actions without becoming overly deterministic.
By adjusting the alpha coefficient, you can control the level of exploration to be more or less.  bigger alpha for more exploration
'''
import torch, random, math
from rl.env.gym_env import GymEnv
from ReinforcementLearning.utils.tools import test, update_data, soft_update, get_sample


class ModelAction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
        )
        self.fc_mu = torch.nn.Linear(128, 1)
        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
        )

    def forward(self, state):
        # [b, 3] -> [b, 128]
        state = self.fc_state(state)

        # [b, 128] -> [b, 1]
        mu = self.fc_mu(state)

        # [b, 128] -> [b, 1]
        std = self.fc_std(state)

        # Define b Gaussian distributions based on mu and std.
        dist = torch.distributions.Normal(mu, std)

        '''Sample b samples.
        "rsample" indicates re-sampling. Essentially, it involves first sampling from a standard normal distribution, 
        then multiplying by the standard deviation and adding the mean.'''
        sample = dist.rsample()

        # Clip the samples to the range between -1 and 1, and calculate the actions.
        action = torch.tanh(sample)
        log_prob = dist.log_prob(sample)

        # Entropy of the actions.
        entropy = log_prob - (1 - action.tanh() ** 2 + 1e-7).log()
        entropy = -entropy

        return action * 2, entropy


class ModelValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, state, action):
        # [b, 3+1] -> [b, 4]
        state = torch.cat([state, action], dim=1)
        # [b, 4] -> [b, 1]
        return self.sequential(state)


def select_action(state):
    """
    This function selects an action based on the state.

    Parameters:
    state

    Returns:
    int: index of selected action
    """
    if continuous_action == 1:
        state = torch.FloatTensor(state).reshape(1, 3)
        action, _ = model_action(state)
        return action.item()
    else:
        state = torch.FloatTensor(state).reshape(1, 4)
        prob = model_action(state)
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        return action


def get_target(reward, next_state, over):
    # First, use the model_action to calculate the actions and entropy. [b, 4] -> [b, 1],[b, 1]
    if continuous_action == 1:
        action, entropy = model_action(next_state)
        # Evaluate the value of the next_state. [b, 4],[b, 1] -> [b, 1]
        target1 = model_value_next1(next_state, action)
        target2 = model_value_next2(next_state, action)
    else:
        prob = model_action(next_state)
        # Calculate the entropy of actions. [b, 2]
        entropy = prob * torch.log(prob + 1e-8)
        # Sum the entropy of all actions.[b, 2] -> [b, 1]
        entropy = -entropy.sum(dim=1, keepdim=True)
        target1 = model_value_next1(next_state)
        target2 = model_value_next2(next_state)
    # Select the one with the lower value, considering stability  [b, 1]
    target = torch.min(target1, target2)
    if continuous_action != 1:
        # estimation of the target. [b, 2] * [b, 2] -> [b, 2]
        target = (prob * target)
        # [b, 2] -> [b, 1]
        target = target.sum(dim=1, keepdim=True)

    # the operation involves adding the entropy of the action to the target. [b, 1] - [b, 1] -> [b, 1]
    target += alpha.exp() * entropy

    # [b, 1]
    target *= 0.99
    target *= (1 - over)
    target += reward

    return target


def get_loss_action(state):
    '''Calculate the loss of the action network.'''
    # Calculate the action and entropy.  [b, 3] -> [b, 1],[b, 1]
    if continuous_action == 1:
        action, entropy = model_action(state)
        # Use the value network to evaluate the value of the action.[b, 3],[b, 1] -> [b, 1]
        value1 = model_value1(state, action)
        value2 = model_value2(state, action)
    else:
        # [b, 4] -> [b, 2]
        prob = model_action(state)
        # [b, 2]
        entropy = prob * (prob + 1e-8).log()
        value1 = model_value1(state)
        value2 = model_value2(state)

    #Choose the one with the lower value for the sake of stability. [b, 1]
    value = torch.min(value1, value2)
    if continuous_action == 1:
        #  After restoring alpha, multiply it by entropy. The desired value is larger for better exploration,
        #  but since we are calculating the loss here, the sign is reversed.
        # [1] - [b, 1] -> [b, 1]
        loss_action = -alpha.exp() * entropy
        # Subtract from the value, so larger values are better for value optimization, resulting in smaller loss
        loss_action -= value
        return loss_action.mean(), entropy
    else:
        # Weight the value according to the probability of the action
        # [b, 2] * [b, 2] -> [b, 2]
        value *= prob
        # [b, 2] -> [b, 1]
        value = value.sum(dim=1, keepdim=True)

        # the operation involves adding the entropy of the action to the target. A larger value is preferable
        # [b, 1] + [b, 1] -> [b, 1]
        loss_action = value + alpha.exp() * entropy

        # Because it's calculating the loss, the sign of this value is reversed.
        return -loss_action.mean(), entropy


def train():
    """
    This function implements training process
    """
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=3e-4)
    optimizer_value1 = torch.optim.Adam(model_value1.parameters(), lr=3e-3)
    optimizer_value2 = torch.optim.Adam(model_value2.parameters(), lr=3e-3)

    optimizer_alpha = torch.optim.Adam([alpha], lr=3e-4)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(1):

        update_data(env, datas, continuous_action, select_action)

        # Learn N times after each data update.
        for i in range(200):

            state, action, reward, next_state, over = get_sample(continuous_action, datas)

            if continuous_action == 1:
                # Offset the rewards for ease of training.
                reward = (reward + 8) / 8

            # Calculate the target, which already takes into account the entropy of actions.  [b, 1]
            target = get_target(reward, next_state, over)
            target = target.detach()

            if continuous_action == 1:
                value1 = model_value1(state, action)
                value2 = model_value2(state, action)
            else:
                value1 = model_value1(state).gather(dim=1, index=action)
                value2 = model_value2(state).gather(dim=1, index=action)

            loss_value1 = loss_fn(value1, target)
            loss_value2 = loss_fn(value2, target)

            optimizer_value1.zero_grad()
            loss_value1.backward()
            optimizer_value1.step()

            optimizer_value2.zero_grad()
            loss_value2.backward()
            optimizer_value2.step()

            loss_action, entropy = get_loss_action(state)
            optimizer_action.zero_grad()
            loss_action.backward()
            optimizer_action.step()

            # Entropy multiplied by alpha is the loss of alpha   [b, 1] -> [1]
            loss_alpha = (entropy + 1).detach() * alpha.exp()
            loss_alpha = loss_alpha.mean()

            optimizer_alpha.zero_grad()
            loss_alpha.backward()
            optimizer_alpha.step()

            soft_update(model_value1, model_value_next1)
            soft_update(model_value2, model_value_next2)

        if epoch % 10 == 0:
            test_result = sum([test(env, False, continuous_action, select_action) for _ in range(10)]) / 10
            print(epoch, len(datas), alpha.exp().item(), test_result)

if __name__ == '__main__':
    continuous_action = 0
    alpha = torch.tensor(math.log(0.01))
    alpha.requires_grad = True

    if continuous_action == 1:
        name = 'Pendulum-v1'
        model_action = ModelAction()
        model_value1 = ModelValue()
        model_value2 = ModelValue()
        model_value_next1 = ModelValue()
        model_value_next2 = ModelValue()
    else:
        name = 'CartPole-v1'
        model_action = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=1),
        )
        model_value1 = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )
        model_value2 = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )
        model_value_next1 = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )
        model_value_next2 = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )

    env = GymEnv(name, continuous_action)
    env.reset()
    model_value_next1.load_state_dict(model_value1.state_dict())
    model_value_next2.load_state_dict(model_value2.state_dict())
    datas = []
    train()
    test(env, True, continuous_action, select_action)
