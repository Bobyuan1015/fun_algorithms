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
def get_update(alogrithm, row, col, action, reward, next_row, next_col, next_action):
    global pi
    if alogrithm == 'sarsa':
        #计算target
        target = 0.9 * pi[next_row, next_col, next_action]
    elif alogrithm == 'Q':
        # target为下一个格子的最高分数，这里的计算和下一步的动作无关
        target = 0.9 * pi[next_row, next_col].max()
    else:
        target = 0
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


#训练
def train():
    global pi
    for epoch in range(1500):
        #初始化当前位置
        row = random.choice(range(4))
        col = 0

        #初始化第一个动作
        action = get_action(row, col)

        #计算反馈的和，这个数字越大越好
        reward_sum = 0

        #循环直到到达终点或者掉进陷阱
        while env.get_state(row, col) not in ['terminal', 'trap']:

            #执行动作
            next_row, next_col, reward = env.move(row, col, action)
            reward_sum += reward

            #求新位置的动作
            next_action = get_action(next_row, next_col)

            #更新分数
            update = get_update('Q',row, col, action, reward, next_row, next_col,next_action)

            pi[row, col, action] += update

             #更新当前位置
            row = next_row
            col = next_col
            action = next_action

        if epoch % 150 == 0:
            print(epoch, reward_sum)


if __name__ == '__main__':
    # 初始化在每一个格子里采取每个动作的分数,初始化都是0,因为没有任何的知识
    pi = np.zeros([4, 12, 4])
    env = CliffEnv()
    train()
    env.final_result(pi)
