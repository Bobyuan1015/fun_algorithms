import numpy as np
from rl.env.data_cache import Pool
from rl.env.gym_env import GymEnv
import random


#玩一局游戏并记录数据
def play(show=False):
    state = []
    action = []
    reward = []
    next_state = []
    over = []

    s = env.reset()
    o = False
    while not o:
        a = Q[s].argmax()
        if random.random() < 0.1:
            a = env.action_space.sample()

        ns, r, o = env.step(a)

        state.append(s)
        action.append(a)
        reward.append(r)
        next_state.append(ns)
        over.append(o)

        s = ns

        if show:

            env.show()

    return state, action, reward, next_state, over, sum(reward)



#训练
def train():
    #训练N局
    for epoch in range(50000):

        #玩一局游戏,得到数据
        state, action, reward, next_state, over, _ = play()
        for i in range(len(state)):
            #计算value
            value = Q[state[i], action[i]]

            #计算target
            #累加未来N步的reward,越远的折扣越大
            #这里是在使用蒙特卡洛方法估计target
            reward_s = 0
            for j in range(i, min(len(state), i + 5)):
                reward_s += reward[j] * 0.9**(j - i)

            #计算最后一步的value,这是target的一部分,按距离给折扣
            target = Q[next_state[j]].max() * 0.9**(j - i + 1)

            #如果最后一步已经结束,则不需要考虑状态价值
            #最后累加reward就是target
            target = target + reward_s

            #更新Q表
            Q[state[i], action[i]] += (target - value) * 0.05

        if epoch % 5000 == 0:
            test_result = sum([play()[-1] for _ in range(20)]) / 20
            print(epoch, test_result)




if __name__ == "__main__":
    env = GymEnv(continuous=False)
    # 初始化Q表,定义了每个状态下每个动作的价值
    Q = np.zeros((16, 4))
    pool = Pool(play)
    train()

    play(True)[-1] #test