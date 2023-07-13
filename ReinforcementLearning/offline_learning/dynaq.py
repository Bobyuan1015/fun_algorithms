import random
import numpy as np
from ReinforcementLearning.env.cliff_env import CliffEnv



#根据状态选择一个动作
def get_action(row, col):
    #有小概率选择随机动作
    global pi
    if random.random() < 0.1:
        return random.choice(range(4))

    #否则选择分数最高的动作
    return pi[row, col].argmax()


#更新分数，每次更新取决于当前的格子，当前的动作，下个格子，和下个格子的动作 ->s a r s a
def get_update(alogrithm, row, col, action, reward, next_row, next_col):
    if alogrithm == 'Q'| 'dynaq':
        # target为下一个格子的最高分数，这里的计算和下一步的动作无关
        target = 0.9 * pi[next_row, next_col].max()
    target += reward

    #计算value,直接查Q表
    value = pi[row, col, action]

    #根据时序差分算法,当前state,action的分数 = 下一个state,action的分数*gamma + reward
    #此处是求两者的差,越接近0越好
    loss = target - value

    #这个0.1相当于lr
    loss *= 0.1

    #更新当前状态和动作的分数
    return loss

def q_planning():
    #Q planning循环,相当于是在反刍历史数据,随机取N个历史数据再进行离线学习
    for _ in range(20):
        #随机选择曾经遇到过的状态动作对
        row, col, action = random.choice(list(history.keys()))

        #再获取下一个状态和反馈
        next_row, next_col, reward = history[(row, col, action)]

        #计算分数
        update = get_update('dynaq',row, col, action, reward, next_row, next_col)

        #更新分数
        pi[row, col, action] += update

#训练
def train():
    for epoch in range(300):
        #初始化当前位置
        row = random.choice(range(4))
        col = 0

        #初始化第一个动作
        action = get_action(row, col)

        #计算反馈的和，这个数字应该越来越小
        reward_sum = 0

        #循环直到到达终点或者掉进陷阱
        while env.get_state(row, col) not in ['terminal', 'trap']:

            #执行动作
            next_row, next_col, reward = env.move(row, col, action)
            reward_sum += reward
            #求新位置的动作
            next_action = get_action(next_row, next_col)
            #计算分数
            update = get_update('Q',row, col, action, reward, next_row, next_col)
            #更新分数
            pi[row, col, action] += update


            #--------------------------------------
            #将数据添加到模型中
            history[(row, col, action)] = next_row, next_col, reward
            #反刍历史数据,进行离线学习
            q_planning()
            # --------------------------------------



            #更新当前位置
            row = next_row
            col = next_col
            action = next_action

        if epoch % 20 == 0:
            print(epoch, reward_sum)


if __name__ == '__main__':
    # 初始化在每一个格子里采取每个动作的分数,初始化都是0,因为没有任何的知识
    pi = np.zeros([4, 12, 4])

    # 保存历史数据,键是(row,col,action),值是(next_row,next_col,reward)
    history = dict()
    env = CliffEnv()
    train()
    env.final_result(pi)
