import numpy as np
import gym

def extract_policy(v, gamma=1.0):
    """ 从状态函数中提取策略 """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        # q_sa: 在状态 s 下的 所有动作价值
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        # print("q_sa: ", q_sa)
        policy[s] = np.argmax(q_sa)
        # print('np.argmax(q_sa): ', policy[s])
    return policy


def value_iteration(env, gamma=1.0):
    """ Value-iteration algorithm """
    # env.env.nS: 16
    # env.env.nA: 4
    # env.env.ncol / env.env.nrow: 4
    v = np.zeros(env.env.nS)
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            # env.env.P[s][a]]: 状态 s 下动作 a 的概率
            q_sa = [sum([p * (r + prev_v[s_]) for p, s_, r, _ in
                         env.env.P[s][a]]) for a in range(env.env.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return v


if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    # env 中是 gym 提供的本次问题的环境信息
    env = gym.make(env_name)
    gamma = 1.0

    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(optimal_v, gamma)
    print('policy:', policy)
    # Value-iteration converged at iteration# 1373.
# policy: [0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]
