import numpy as np
from rl.env.data_cache import Pool
from rl.env.gym_env import GymEnv
import random


#玩一局游戏并记录数据
def play(show=False):
    data = []
    reward_sum = 0

    state = env.reset()
    over = False
    while not over:
        action = Q[state].argmax()
        if random.random() < 0.1:
            action = env.action_space.sample()

        next_state, reward, over = env.step(action)

        data.append((state, action, reward, next_state, over))
        reward_sum += reward

        state = next_state

        if show:
            env.show(is_display=True)

    return data, reward_sum



#训练
def train(sarsa=True,qlearning=False):
    #共更新N轮数据
    for epoch in range(1000):
        pool.update()
        #每次更新数据后,训练N次
        for i in range(200):
            #随机抽一条数据
            state, action, reward, next_state, over = pool.sample()
            #Q矩阵当前估计的state下action的价值
            value = Q[state, action]
            if qlearning:
                #实际玩了之后得到的reward+下一个状态的价值*0.9
                target = reward + Q[next_state].max() * 0.9
                #value和target应该是相等的,说明Q矩阵的评估准确
                #如果有误差,则应该以target为准更新Q表,修正它的偏差
                #这就是TD误差,指评估值之间的偏差,以实际成分高的评估为准进行修正
                update = (target - value) * 0.1
            elif sarsa:
                # 求下一个动作,和Q学习唯一的区别点
                next_action = Q[next_state].argmax()
                # 实际玩了之后得到的reward+下一个状态的价值*0.9
                target = reward + Q[next_state, next_action] * 0.9
                # value和target应该是相等的,说明Q矩阵的评估准确
                # 如果有误差,则应该以target为准更新Q表,修正它的偏差
                # 这就是TD误差,指评估值之间的偏差,以实际成分高的评估为准进行修正
                update = (target - value) * 0.02

            #更新Q表
            Q[state, action] += update

        if epoch % 100 == 0:
            print(epoch, len(pool), play()[-1])

if __name__ == "__main__":
    env = GymEnv(continuous=False)
    # 初始化Q表,定义了每个状态下每个动作的价值
    Q = np.zeros((16, 4))
    pool = Pool(play)
    train(sarsa=True,qlearning=False)

    play(True)[-1] #test