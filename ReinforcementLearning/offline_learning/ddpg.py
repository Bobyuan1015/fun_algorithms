import torch
import random
import numpy as np



from IPython import display

from ReinforcementLearning.env.gym_env import GymEnv


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
        state, reward, over, _ = env.step([action])
        reward_sum += reward

        #打印动画
        if play and random.random() < 0.2:  #跳帧
            display.clear_output(wait=True)
            env.show()

    return reward_sum


#获取一批数据样本
def get_sample():
    #从样本池中采样
    samples = random.sample(datas, 64)

    #[b, 3]
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, 3)
    #[b, 1]
    action = torch.FloatTensor([i[1] for i in samples]).reshape(-1, 1)
    #[b, 1]
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    #[b, 3]
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, 3)
    #[b, 1]
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1, 1)

    return state, action, reward, next_state, over


def get_value(state, action):
    #直接评估综合了state和action的value
    #[b, 3+1] -> [b, 4]
    input = torch.cat([state, action], dim=1)

    #[b, 4] -> [b, 1]
    return model_value(input)


def get_target(next_state, reward, over):
    #对next_state的评估需要先把它对应的动作计算出来,这里用model_action_next来计算
    #[b, 3] -> [b, 1]
    action = model_action_next(next_state)

    #和value的计算一样,action拼合进next_state里综合计算
    #[b, 3+1] -> [b, 4]
    input = torch.cat([next_state, action], dim=1)

    #[b, 4] -> [b, 1]
    target = model_value_next(input) * 0.98

    #[b, 1] * [b, 1] -> [b, 1]
    target *= (1 - over)

    #[b, 1] + [b, 1] -> [b, 1]
    target += reward

    return target

def get_loss_action(state):
    #首先把动作计算出来
    #[b, 3] -> [b, 1]
    action = model_action(state)

    #像value计算那里一样,拼合state和action综合计算
    #[b, 3+1] -> [b, 4]
    input = torch.cat([state, action], dim=1)

    #使用value网络评估动作的价值,价值是越高越好
    #因为这里是在计算loss,loss是越小越好,所以符号取反
    #[b, 4] -> [b, 1] -> [1]
    loss = -model_value(input).mean()

    return loss


def soft_update(model, model_next):
    for param, param_next in zip(model.parameters(), model_next.parameters()):
        #以一个小的比例更新
        value = param_next.data * 0.995 + param.data * 0.005
        param_next.data.copy_(value)

def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 3)
    action = model_action(state).item()
    #给动作添加噪声,增加探索。这样就不会才机械
    action += random.normalvariate(mu=0, sigma=0.01)
    return action



#向样本池中添加N条数据,删除M条最古老的数据
def update_data():
    #初始化游戏
    state = env.reset()

    #玩到游戏结束为止
    over = False
    while not over:
        #根据当前状态得到一个动作
        action = get_action(state)

        #执行动作,得到反馈
        next_state, reward, over, _ = env.step([action])

        #记录数据样本
        datas.append((state, action, reward, next_state, over))

        #更新游戏状态,开始下一个动作
        state = next_state

    #数据上限,超出时从最古老的开始删除
    while len(datas) > 10000:
        datas.pop(0)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),#-1---1区间
        )

    def forward(self, state):
        return self.sequential(state) * 2.0     #-2---2区间


def train():
    model_action.train()
    model_value.train()
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=5e-4)
    optimizer_value = torch.optim.Adam(model_value.parameters(), lr=5e-3)
    loss_fn = torch.nn.MSELoss()

    #训练N次
    for epoch in range(200):
        #更新N条数据
        update_data()

        #每次更新过数据后,学习N次
        for i in range(200):
            #采样一批数据
            state, action, reward, next_state, over = get_sample()

            #计算value和target
            value = get_value(state, action)
            target = get_target(next_state, reward, over)

            #两者求差,计算loss,更新参数
            loss_value = loss_fn(value, target)

            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()

            #使用value网络评估action网络的loss,更新参数
            loss_action = get_loss_action(state)

            optimizer_action.zero_grad()
            loss_action.backward()
            optimizer_action.step()

            #以一个小的比例更新
            soft_update(model_action, model_action_next)
            soft_update(model_value, model_value_next)

        if epoch % 20 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(epoch, len(datas), test_result)


if __name__ == '__main__':


    model_action = Model()
    model_action_next = Model()  # 延迟更新
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
    )  # 延迟更新
    model_value_next.load_state_dict(model_value.state_dict())

    # 样本池
    datas = []
    env = GymEnv('Pendulum-v1')
    env.reset()
    train()
    test(True)
