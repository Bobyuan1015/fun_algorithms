import numpy as np
import random

# 每个老虎机的中奖概率,0-1之间的均匀分布
probs = np.random.uniform(size=10)

# 记录每个老虎机的返回值
rewards = [[1] for _ in range(10)]
method = 'Thompson'


def choose_one(method='greedy'):
    # 有小概率随机选择一根拉杆
    if method == 'greedy':#总是选择当前机器中胜率（历史）最高的机器玩
        if random.random() < 0.01:
            print('exploration')
            return random.randint(0, 9)#随机,  实验测试，不用探索准确率还高一些。
        print('exploitation')
        # 计算每个老虎机的奖励平均
        print(f'rewards ={rewards}')

        rewards_mean = [np.mean(i) for i in rewards]
        print(f'rewards_mean ={rewards_mean}')
        # 选择期望奖励估值最大的拉杆
        return np.argmax(rewards_mean)
    elif method == 'decayed_random':##随机选择的概率递减的贪婪算法
        # 求出现在已经玩了多少次了
        played_count = sum([len(i) for i in rewards])

        # 随机选择的概率逐渐下降
        if random.random() < 1 / played_count:
            return random.randint(0, 9)

        # 计算每个老虎机的奖励平均
        rewards_mean = [np.mean(i) for i in rewards]

        # 选择期望奖励估值最大的拉杆
        return np.argmax(rewards_mean)
    elif method == 'ucb': #上置信界
        # 求出每个老虎机各玩了多少次
        played_count = [len(i) for i in rewards]
        played_count = np.array(played_count)

        # 求出上置信界
        # 分子是总共玩了多少次,取根号后让他的增长速度变慢
        # 分母是每台老虎机玩的次数,乘以2让他的增长速度变快
        # 随着玩的次数增加,分母会很快超过分子的增长速度,导致分数越来越小
        # 具体到每一台老虎机,则是玩的次数越多,分数就越小,也就是ucb的加权越小
        # 所以ucb衡量了每一台老虎机的不确定性,不确定性越大,探索的价值越大
        fenzi = played_count.sum() ** 0.5
        fenmu = played_count * 2
        ucb = fenzi / fenmu  # 这是一个向量，玩得少的机器分母会小，ucb就会更大

        # ucb本身取根号
        # 大于1的数会被缩小,小于1的数会被放大,这样保持ucb恒定在一定的数值范围内
        ucb = ucb ** 0.5

        # 计算每个老虎机的奖励平均
        rewards_mean = [np.mean(i) for i in rewards]
        rewards_mean = np.array(rewards_mean)

        # ucb和期望求和
        ucb += rewards_mean

        return ucb.argmax()
    elif method == 'Thompson':#汤普森采样算法
        # 求出每个老虎机出1的次数+1
        count_1 = [sum(i) + 1 for i in rewards]

        # 求出每个老虎机出0的次数+1
        count_0 = [sum(1 - np.array(i)) + 1 for i in rewards]

        # 按照beta分布计算奖励分布,这可以认为是每一台老虎机中奖的概率
        beta = np.random.beta(count_1, count_0)

        return beta.argmax()
    else:
        print(f'no sample method')
        return


def try_and_play():
    i = choose_one(method)

    # 玩老虎机,得到结果
    reward = 0
    if random.random() < probs[i]:
        reward = 1  # 赢了

    # 记录玩的结果
    rewards[i].append(reward)


def get_result():
    # 玩N次
    for _ in range(5000):
        try_and_play()

    # 期望的最好结果
    target = probs.max() * 5000

    # 实际玩出的结果
    result = sum([sum(i) for i in rewards])
    print(f'机器的胜率：{probs}')
    return target, result



target, result = get_result()
print(f'target={target}')
print(f'result={result}')
