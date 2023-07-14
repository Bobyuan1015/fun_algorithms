import gym, torch, random
from IPython import display

from ReinforcementLearning.env.gym_env import GymEnv


#得到一个动作
def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 4)

    #[1, 4] -> [1, 2]
    prob = model(state)

    #根据概率选择一个动作
    action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]

    return action


#得到一局游戏的数据
def get_data():
    states = []
    rewards = []
    actions = []

    #初始化游戏
    state = env.reset()

    #玩到游戏结束为止
    over = False
    while not over:
        #根据当前状态得到一个动作
        action = get_action(state)

        #执行动作,得到反馈
        next_state, reward, over, _ = env.step(action)

        #记录数据样本
        states.append(state)
        rewards.append(reward)
        actions.append(action)

        #更新游戏状态,开始下一个动作
        state = next_state

    return states, rewards, actions





def test(play):
    #初始化游戏
    state = env.reset()

    #记录反馈值的和,这个值越大越好
    reward_sum = 0

    #玩到游戏结束为止
    over = False
    while not over:
        #根据当前状态得到一个动作
        action = get_action(state)

        #执行动作,得到反馈
        state, reward, over, _ = env.step(action)
        reward_sum += reward

        #打印动画
        if play and random.random() < 0.2:  #跳帧
            display.clear_output(wait=True)
            env.show()

    return reward_sum

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #玩N局游戏,每局游戏训练一次
    for epoch in range(1000):
        #玩一局游戏,得到数据
        states, rewards, actions = get_data()

        optimizer.zero_grad()

        #反馈的和,初始化为0
        reward_sum = 0

        #从最后一步算起
        for i in reversed(range(len(states))):

            #反馈的和,从最后一步的反馈开始计算
            #每往前一步,>>和<<都衰减0.02,然后再加上当前步的反馈
            reward_sum *= 0.98
            reward_sum += rewards[i]

            #重新计算对应动作的概率
            state = torch.FloatTensor(states[i]).reshape(1, 4)
            #[1, 4] -> [1, 2]
            prob = model(state)
            #[1, 2] -> scala
            prob = prob[0, actions[i]]

            #根据求导公式,符号取反是因为这里是求loss,所以优化方向相反
            loss = -prob.log() * reward_sum

            #累积梯度
            loss.backward(retain_graph=True)

        optimizer.step()

        if epoch % 100 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(epoch, test_result)



if __name__ == '__main__':
    # 计算动作的模型,也是真正要用的模型
    type_network = ''
    # 定义模型
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
        torch.nn.Softmax(dim=1),
    )

    env = GymEnv()
    env.reset()
    train()
    test(play=True)
