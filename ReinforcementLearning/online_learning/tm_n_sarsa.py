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


##获取5个时间步分别的分数
def get_update_list(next_row, next_col, next_action):
    #初始化的target是最后一个state和最后一个action的分数
    target = pi[next_row, next_col, next_action]

    #计算每一步的target
    #每一步的tagret等于下一步的tagret*0.9，再加上本步的reward
    #时间从后往前回溯，越以前的tagret会累加的信息越多
    #[4, 3, 2, 1, 0]
    target_list = []
    for i in reversed(range(5)):
        target = 0.9 * target + reward_list[i]
        target_list.append(target)

    #把时间顺序正过来
    target_list = list(reversed(target_list))

    #计算每一步的value
    value_list = []
    for i in range(5):
        row, col = state_list[i]
        action = action_list[i]
        value_list.append(pi[row, col, action])


    #计算每一步的更新量
    update_list = []
    for i in range(5):
        #根据时序差分算法,当前state,action的分数 = 下一个state,action的分数*gamma + reward
        #此处是求两者的差,越接近0越好
        update = target_list[i] - value_list[i]

        #这个0.1相当于lr
        update *= 0.1

        update_list.append(update)

    return update_list

#训练
def train():
    for epoch in range(1500):
        #初始化当前位置
        row = random.choice(range(4))
        col = 0

        #初始化第一个动作
        action = get_action(row, col)

        #计算反馈的和，这个数字应该越来越小
        reward_sum = 0

        #初始化3个列表
        state_list.clear()
        action_list.clear()
        reward_list.clear()

        #循环直到到达终点或者掉进陷阱
        while env.get_state(row, col) not in ['terminal', 'trap']:

            #执行动作
            next_row, next_col, reward = env.move(row, col, action)
            reward_sum += reward

            #求新位置的动作
            next_action = get_action(next_row, next_col)

            #记录历史数据
            state_list.append([row, col])
            action_list.append(action)
            reward_list.append(reward)

            #积累到5步以后再开始更新参数
            if len(state_list) == 5:

                #计算分数
                update_list = get_update_list(next_row, next_col, next_action)

                #只更新第一步的分数
                row, col = state_list[0]
                action = action_list[0]
                update = update_list[0]

                pi[row, col, action] += update

                #移除第一步，这样在下一次循环时保持列表是5个元素
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)

            #更新当前位置
            row = next_row
            col = next_col
            action = next_action

        #走到终点以后，更新剩下步数的update
        for i in range(len(state_list)):
            row, col = state_list[i]
            action = action_list[i]
            update = update_list[i]
            pi[row, col, action] += update

        if epoch % 100 == 0:
            print(epoch, reward_sum)


if __name__ == '__main__':
    # 初始化在每一个格子里采取每个动作的分数,初始化都是0,因为没有任何的知识
    pi = np.zeros([4, 12, 4])
    # 初始化3个list,用来存储状态,动作,反馈的历史数据,因为后面要回溯这些数据
    state_list = []
    action_list = []
    reward_list = []

    env = CliffEnv()
    train()
    env.final_result(pi)
