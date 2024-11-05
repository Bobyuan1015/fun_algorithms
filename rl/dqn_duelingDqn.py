import torch, random, os

from rl.env.gym_env import GymEnv
from ReinforcementLearning.utils.tools import get_sample

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


# Set the timer interval to 5000 milliseconds.
# fig = plt.figure()
# timer = fig.canvas.new_timer(interval = 500)
# timer.add_callback(plt.close)


class VAnet(torch.nn.Module):
    '''The key distinction of Dueling DQN compared to other DQN models lies in its utilization of a distinct model structure.'''
    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
        )

        self.fc_A = torch.nn.Linear(128, 11)  # matrix
        self.fc_V = torch.nn.Linear(128, 1)  # vector

    def forward(self, x):
        '''Exactly the same as DQN, the only difference lies in this part where the training process remains identical.
         A+V=Q， a constraint is added to A: the sum of its columns is equal to 0.
         This makes updating A more challenging, thus compelling V to be updated to optimize Q.
         As V is a vector, its dimension is lower, and updating a single V can result in a good Q calculation.
         Eventually, this enhances efficiency.'''
        # [5, 11] -> [5, 128] -> [5, 11]
        A = self.fc_A(self.fc(x))

        # [5, 11] -> [5, 128] -> [5, 1]
        V = self.fc_V(self.fc(x))

        # [5, 11] -> [5] -> [5, 1]
        A_mean = A.mean(dim=1).reshape(-1, 1)

        # [5, 11] - [5, 1] = [5, 11]
        A -= A_mean  # Mean normalization, so that the sum of each column becomes 0

        # Q-values are calculated from the combination of value (V) and advantage (A) values.
        # [5, 11] + [5, 1] = [5, 11]
        Q = A + V

        return Q

    def load_state_dict(self, state_dic):
        self.fc.load_state_dict(state_dic)

    def get_state_weight(self):
        return self.fc.state_dict()

class NormalNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.fc(x)

    def load_state_dict(self, state_dic):
        self.fc.load_state_dict(state_dic)

    def get_state_weight(self):
        return self.fc.state_dict()

def test_env():
    state = env.reset()
    print("The state of this game is represented by four numbers. I don't know the specific meaning of these four numbers, but they collectively describe the entire state of the game.")
    print('state=', state)
    # state= [ 0.03490619  0.04873464  0.04908862 -0.00375859]

    print('This game has a total of two actions: either 0 or 1.')
    print('env.action_space=', env.action_space)
    #If the actions are continuous values, they can be discretized
    # env.action_space= Discrete(2)

    print('Randomly select an action.')
    action = env.action_space.sample()
    print('action=', action)
    # action= 1

    print('Perform an action, obtain the next state, reward, and whether the game has ended.')
    state, reward, over, _ = env.step(action)

    print('state=', state)
    # state= [ 0.02018229 -0.16441101  0.01547085  0.2661691 ]

    print('reward=', reward)
    # reward= 1.0

    print('over=', over)
    # over= False



def select_action(state):
    """
    This function selects an action based on the state.

    Parameters:
    state

    Returns:
    int: index of selected action
    """
    # 1. choose a random action with a small probability.
    if random.random() < 0.1:
        return random.choice(range(4))

    # 2. Otherwise, the neural network generates an action,instead of looking up Q-values,
    state = torch.FloatTensor(state).reshape(1, 4)
    return model(state).argmax().item()()

def update_data():
    '''Add N pieces of data to the experience replay buffer and remove the oldest M pieces of data.
    The maximum size of buffer is 10,000'''
    old_count = len(datas)

    while len(datas) - old_count < 200:

        state = env.reset()
        # Play until the game is over.
        over = False
        while not over:
            action = select_action(state)
            next_state, reward, over, _ = env.step(action)
            # Collecting data samples.
            datas.append((state, action, reward, next_state, over))
            state = next_state
    update_count = len(datas) - old_count
    drop_count = max(len(datas) - 10000, 0)

    while len(datas) > 10000:
        datas.pop(0)
    return update_count, drop_count



def get_qvalue(state, action):
    """
    This function calculates Q-values using a neural network.

    Parameters:
    state
    action

    Returns:
    q(a,s)
    """
    # Compute action logits from the given state. [b, 4] -> [b, 2]
    value = model(state)

    # [b, 2] -> [b, 1]
    value = value.gather(dim=1, index=action)
    return value

def get_target(reward, next_state, over):
    """
     This function calculates Q-values using a neural network.
     For a given state, its expected score can be estimated using accumulated experience from the past models.
     In DQN, the "next_model" is used for evaluation.
     There is clearly no exact solution. The delayed update of the "next_model" is used for evaluation.

     Parameters:
     reward
     next_state
     over

     Returns:
     q(a,s)
     """
    improved_target = False
    with torch.no_grad():
        # [b, 4] -> [b, 2]
        target = next_model(next_state)

    if improved_target:
        # Because targeting can lead to overestimation, the following strategy is introduced.
        """difference between Double DQN and DQN  start"""
        with torch.no_grad():
            # [b, 3] -> [b, 11]
            model_target = model(next_state)

        # Choose the index with the highest score.   [b, 11] -> [b, 1]
        model_target = model_target.max(dim=1)[1]
        model_target = model_target.reshape(-1, 1)
        # [b, 11] -> [b]
        # Use this index to retrieve values from the target. This can mitigate the issue of overestimation in DQN.
        target = target.gather(dim=1, index=model_target)
        """difference between Double DQN and DQN  end"""
    else:
        # DQN
        # [b, 2] -> [b, 1]
        target = target.max(dim=1)[0]
        target = target.reshape(-1, 1)

    #The score of the next state is multiplied by a coefficient, acting as a weight.
    target *= 0.98
    '''If the next_state corresponds to the end of the game, its score is 0.
    This is because if the game has already ended in the next step, 
    it's evident that there's no need to continue playing, and considering next_state becomes unnecessary.'''
    target *= (1 - over)
    # [b, 1] + [b, 1] -> [b, 1]
    target += reward
    return target

def train():
    """
    This function implements training process
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()  # Mean squared error loss.

    # Train N times.   200 ofr DQN, 500 for duelingDQN
    for epoch in range(500):
        # update a batch of data
        update_count, drop_count = update_data()

        # After each data update, perform N learning iterations.
        for i in range(200):
            # Sample a batch of data.
            state, action, reward, next_state, over = get_sample(datas)

            value = get_qvalue(state, action)
            target = get_target(reward, next_state, over)

            # update network weights
            loss = loss_fn(value, target)
            #Initialize the model's parameters and gradients to 0.
            optimizer.zero_grad()
            loss.backward()
            #All optimizers have implemented the step() method, which updates all parameters.
            optimizer.step()

            # delay update:  Copy the parameters of the model to the next_model. 50 for duelingDQN
            if (i + 1) % 10 == 0:
                next_model.load_state_dict(model.get_state_weight())

        if epoch % 50 == 0:
            test_result = sum([test(play=False) for _ in range(20)]) / 20
            print(epoch, len(datas), update_count, drop_count, test_result)


if __name__ == '__main__':
    type_network = ''
    if type_network == 'duelingDQN':
        # The model for calculating actions
        model = VAnet()
        # Experience network used to evaluate the score of a state
        next_model = VAnet()
    else:
        # action network
        model = NormalNet()
        '''Experience network used to evaluate the score of a state. Experience model to evaluate the action model; 
        If the evaluation method keeps changing, training becomes difficult to converge. 
        Therefore, here is delayed updating after several steps.'''
        next_model = NormalNet()

    next_model.load_state_dict(model.get_state_weight())

    datas = []
    env = GymEnv()
    env.reset()
    train()
    test(env,select_action,play=True)
