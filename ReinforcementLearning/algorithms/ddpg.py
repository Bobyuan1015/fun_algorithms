import torch, random, time
from IPython import display
from ReinforcementLearning.env.gym_env import GymEnv
from ReinforcementLearning.utils.tools import update_data, soft_update


def select_action(state):
    """
    This function selects an action based on the state.

    Parameters:
    state

    Returns:
    int: index of selected action
    """
    action = model_action(state).item()
    # Add noise to actions to increase exploration. This prevents the agent from becoming overly deterministic.
    action += random.normalvariate(mu=0, sigma=0.01)
    return action


def get_sample():
    # Sample from the experience replay buffer.
    samples = random.sample(datas, 64)

    # [b, 3]
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, 3)
    # [b, 1]
    action = torch.FloatTensor([i[1] for i in samples]).reshape(-1, 1)
    # [b, 1]
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    # [b, 3]
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, 3)
    # [b, 1]
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1, 1)

    return state, action, reward, next_state, over


def get_value(state, action):
    '''Directly evaluate the value that combines both the state and action'''
    # [b, 3+1] -> [b, 4]
    input = torch.cat([state, action], dim=1)
    # [b, 4] -> [b, 1]
    return model_value(input)


def get_target(next_state, reward, over):
    # Evaluating the next_state requires first calculating the corresponding action for it. Here, "model_action_next" for it.
    # [b, 3] -> [b, 1]
    action = model_action_next(next_state)

    # Similar to the calculation of value, incorporate the action into the next_state for a comprehensive calculation.
    # [b, 3+1] -> [b, 4]
    input = torch.cat([next_state, action], dim=1)

    # [b, 4] -> [b, 1]
    target = model_value_next(input) * 0.98

    # [b, 1] * [b, 1] -> [b, 1]
    target *= (1 - over)

    # [b, 1] + [b, 1] -> [b, 1]
    target += reward

    return target


def get_loss_action(state):
    # First, calculate the action.[b, 3] -> [b, 1]
    action = model_action(state)

    # Similar to the calculation of value, combine the state and action to calculate it comprehensively. [b, 3+1] -> [b, 4]
    input = torch.cat([state, action], dim=1)

    '''Use the value network to evaluate the value of actions, where higher values are better. 
    Since we are calculating the loss here and lower loss is better, the sign is negated.  [b, 4] -> [b, 1] -> [1]'''
    loss = -model_value(input).mean()
    return loss


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),  # In the range from -1 to 1
        )

    def forward(self, state):
        # In the range from -2 to 2.
        return self.sequential(state) * 2.0


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
    model_action.train()
    model_value.train()
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=5e-4)
    optimizer_value = torch.optim.Adam(model_value.parameters(), lr=5e-3)
    loss_fn = torch.nn.MSELoss()


    for epoch in range(200):

        update_data(env,select_action,datas)

        # After each data update, perform N learning iterations.
        for i in range(200):
            # Calculate value and target.
            state, action, reward, next_state, over = get_sample()

            # Sample a batch of data.
            value = get_value(state, action)
            target = get_target(next_state, reward, over)
            loss_value = loss_fn(value, target)

            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()

            # Use the value network to evaluate the loss of the action network, and update the parameters accordingly.
            loss_action = get_loss_action(state)

            optimizer_action.zero_grad()
            loss_action.backward()
            optimizer_action.step()

            # Update with a small proportion.
            soft_update(model_action, model_action_next)
            soft_update(model_value, model_value_next)

        if epoch % 20 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(epoch, len(datas), test_result)


if __name__ == '__main__':
    model_action = Model()
    model_action_next = Model()  # delay update
    model_action_next.load_state_dict(model_action.state_dict())

    model_value = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    model_value_next = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )  # delay update
    model_value_next.load_state_dict(model_value.state_dict())

    # 样本池
    datas = []
    env = GymEnv('Pendulum-v1')
    env.reset()
    train()
    test(True)
