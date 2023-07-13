import numpy as np


from ReinforcementLearning.env.cliff_env import CliffEnv


#计算在一个状态下执行动作的分数，Q函数
def get_qsa(row, col, action):
    #在当前状态下执行动作,得到下一个状态和reward
    next_row, next_col, reward = env.move(row, col, action)
    #计算下一个状态的分数,取values当中记录的分数即可,0.9是折扣因子，因为未来的值是不确定的
    value = env.values[next_row, next_col] * 0.9
    #如果下个状态是终点或者陷阱,则下一个状态的分数是0
    if env.get_state(next_row, next_col) in ['trap', 'terminal']:
        value = 0
    #动作的分数本身就是reward,加上下一个状态的分数
    return value + reward

#策略评估，V函数
def update_values(algorithm='policy_based'):
    #初始化一个新的values,重新评估所有格子的分数，全0
    new_values = np.zeros([4, 12])
    #遍历所有格子
    for row in range(4):
        for col in range(12):
            #计算当前格子4个动作分别的分数
            action_value = np.zeros(4)
            #遍历所有动作，完全采用exploration
            for action in range(4):
                action_value[action] = get_qsa(row, col, action)
            if algorithm == 'policy_based':
                #每个动作的分数和它的概率相乘（转移；概率）
                action_value *= env.pi[row, col]
                #最后这个格子的分数,等于该格子下所有动作的分数求和
                new_values[row, col] = action_value.sum()
            elif algorithm == 'value_based':
                # 价值迭代
                """和策略迭代算法唯一的不同点"""  # 不需要做做动作转移
                # 求每一个格子的分数，等于该格子下所有动作的最大分数
                new_values[row, col] = action_value.max()

    return new_values


def update_pi():
    #重新初始化每个格子下采用动作的概率,重新评估
    new_pi = np.zeros([4, 12, 4])
    #遍历所有格子
    for row in range(4):
        for col in range(12):
            #计算当前格子4个动作分别的分数
            action_value = np.zeros(4)
            #遍历所有动作
            for action in range(4):
                action_value[action] = get_qsa(row, col, action)
            #计算当前状态下，达到最大分数的动作有几个
            count = (action_value == action_value.max()).sum()
            #让这些动作均分概率
            for action in range(4):
                if action_value[action] == action_value.max():
                    new_pi[row, col, action] = 1 / count
                else:
                    new_pi[row, col, action] = 0
    return new_pi

# 循环迭代策略评估和策略提升,寻找最优解
def train():
    global pi
    for _ in range(10):
        for _ in range(100):
            env.values = update_values('value_based')
        pi = update_pi()



if __name__ == '__main__':
    pi = np.ones([4, 12, 4]) * 0.25
    env = CliffEnv()
    train()
    env.final_result(pi)
