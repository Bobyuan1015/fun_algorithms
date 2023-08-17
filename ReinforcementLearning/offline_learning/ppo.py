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

def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 3)
    mu, std = model(state)

    #根据概率选择一个动作
    #action = random.normalvariate(mu=mu.item(), sigma=std.item())
    action = torch.distributions.Normal(mu, std).sample().item()

    return action

def get_data():
    states = []
    rewards = []
    actions = []
    next_states = []
    overs = []

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
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        overs.append(over)

        #更新游戏状态,开始下一个动作
        state = next_state

    #[b, 3]
    states = torch.FloatTensor(states).reshape(-1, 3)
    #[b, 1]
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    #[b, 1]
    actions = torch.FloatTensor(actions).reshape(-1, 1)
    #[b, 3]
    next_states = torch.FloatTensor(next_states).reshape(-1, 3)
    #[b, 1]
    overs = torch.LongTensor(overs).reshape(-1, 1)

    return states, rewards, actions, next_states, overs

#优势函数
def get_advantages(deltas):
    advantages = []

    #反向遍历deltas
    s = 0.0
    for delta in deltas[::-1]:
        s = 0.9 * 0.9 * s + delta
        advantages.append(s)

    #逆序
    advantages.reverse()
    return advantages


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_statu = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
        )

        self.fc_mu = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )

        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
        )

    def forward(self, state):
        state = self.fc_statu(state)

        mu = self.fc_mu(state) * 2.0
        std = self.fc_std(state)

        return mu, std #因为这里是连续动作，整台分布的均值 和 标准差

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_td = torch.optim.Adam(model_td.parameters(), lr=5e-3)
    loss_fn = torch.nn.MSELoss()

    #玩N局游戏,每局游戏训练M次
    for epoch in range(3000):
        #玩一局游戏,得到数据
        #states -> [b, 3]
        #rewards -> [b, 1]
        #actions -> [b, 1]
        #next_states -> [b, 3]
        #overs -> [b, 1]
        states, rewards, actions, next_states, overs = get_data()

        #偏移reward,便于训练
        rewards = (rewards + 8) / 8

        #计算values和targets
        #[b, 3] -> [b, 1]
        values = model_td(states)

        #[b, 3] -> [b, 1]
        targets = model_td(next_states).detach()
        targets = targets * 0.98
        targets *= (1 - overs)
        targets += rewards

        #计算优势,这里的advantages有点像是策略梯度里的reward_sum
        #只是这里计算的不是reward,而是target和value的差
        #[b, 1]
        deltas = (targets - values).squeeze(dim=1).tolist()
        advantages = get_advantages(deltas)
        advantages = torch.FloatTensor(advantages).reshape(-1, 1)

        #取出每一步动作的概率
        #[b, 3] -> [b, 1],[b, 1]
        mu, std = model(states)
        #[b, 1]
        old_probs = torch.distributions.Normal(mu, std)#高斯密度概率
        old_probs = old_probs.log_prob(actions).exp().detach()

        #每批数据反复训练10次
        for _ in range(10):
            #重新计算每一步动作的概率
            #[b, 3] -> [b, 1],[b, 1]
            mu, std = model(states)
            #[b, 1]
            new_probs = torch.distributions.Normal(mu, std)
            new_probs = new_probs.log_prob(actions).exp()

            #求出概率的变化
            #[b, 1] - [b, 1] -> [b, 1]
            ratios = new_probs / old_probs

            #计算截断的和不截断的两份loss,取其中小的
            #[b, 1] * [b, 1] -> [b, 1]
            surr1 = ratios * advantages
            #[b, 1] * [b, 1] -> [b, 1]
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages

            loss = -torch.min(surr1, surr2)
            loss = loss.mean()

            #重新计算value,并计算时序差分loss
            values = model_td(states)
            loss_td = loss_fn(values, targets)

            #更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_td.zero_grad()
            loss_td.backward()
            optimizer_td.step()

        if epoch % 200 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(epoch, test_result)


if __name__ == '__main__':
    model = Model()

    model_td = torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )


    # 样本池
    datas = []
    env = GymEnv('Pendulum-v1')
    env.reset()
    train()
    test(True)
