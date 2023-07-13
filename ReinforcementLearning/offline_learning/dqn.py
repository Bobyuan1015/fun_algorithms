import gym, torch, random
from matplotlib import pyplot as plt
from IPython import display

#定义环境
class GymEnv(gym.Wrapper):

    def __init__(self):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state, _ = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, info

#测试游戏环境
def test_env():
    state = env.reset()
    print('这个游戏的状态用4个数字表示,我也不知道这4个数字分别是什么意思,反正这4个数字就能描述游戏全部的状态')
    print('state=', state)
    #state= [ 0.03490619  0.04873464  0.04908862 -0.00375859]

    print('这个游戏一共有2个动作,不是0就是1')
    print('env.action_space=', env.action_space)
    #env.action_space= Discrete(2)

    print('随机一个动作')
    action = env.action_space.sample()
    print('action=', action)
    #action= 1

    print('执行一个动作,得到下一个状态,奖励,是否结束')
    state, reward, over, _ = env.step(action)

    print('state=', state)
    #state= [ 0.02018229 -0.16441101  0.01547085  0.2661691 ]

    print('reward=', reward)
    #reward= 1.0

    print('over=', over)
    #over= False


#打印游戏
def show():
    plt.imshow(env.render())
    plt.show()

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
        state, reward, over, _ = env.step(action)
        reward_sum += reward
        #打印动画
        if play and random.random() < 0.2:  #跳帧
            display.clear_output(wait=True)
            show()
    return reward_sum



#得到一个动作
def get_action(state):
    if random.random() < 0.01:
        return random.choice([0, 1])
    #走神经网络,得到一个动作,以前是查Q
    state = torch.FloatTensor(state).reshape(1, 4)
    return model(state).argmax().item()


#向样本池中添加N条数据,删除M条最古老的数据
def update_data():
    old_count = len(datas)
    #玩到新增了N个数据为止
    while len(datas) - old_count < 200:
        #初始化游戏
        state = env.reset()
        #玩到游戏结束为止
        over = False
        while not over:
            #根据当前状态得到一个动作
            action = get_action(state)
            #执行动作,得到反馈
            next_state, reward, over, _ = env.step(action)
            #记录数据样本
            datas.append((state, action, reward, next_state, over))
            #更新游戏状态,开始下一个动作
            state = next_state
    update_count = len(datas) - old_count
    drop_count = max(len(datas) - 10000, 0)
    #数据上限,超出时从最古老的开始删除
    while len(datas) > 10000:
        datas.pop(0)
    return update_count, drop_count


#获取一批数据样本
def get_sample():
    #从样本池中采样
    samples = random.sample(datas, 64)
    #[b, 4]
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, 4)
    #[b, 1]
    action = torch.LongTensor([i[1] for i in samples]).reshape(-1, 1)
    #[b, 1]
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    #[b, 4]
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, 4)
    #[b, 1]
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1, 1)
    return state, action, reward, next_state, over


def get_value(state, action):
    #使用状态计算出动作的logits
    #[b, 4] -> [b, 2]
    value = model(state)
    #根据实际使用的action取出每一个值
    #这个值就是模型评估的在该状态下,执行动作的分数
    #在执行动作前,显然并不知道会得到的反馈和next_state
    #所以这里不能也不需要考虑next_state和reward
    #[b, 2] -> [b, 1]
    value = value.gather(dim=1, index=action)
    return value


def get_target(reward, next_state, over):
    #上面已经把模型认为的状态下执行动作的分数给评估出来了
    #下面使用next_state和reward计算真实的分数
    #针对一个状态,它到底应该多少分,可以使用以往模型积累的经验评估
    #这也是没办法的办法,因为显然没有精确解,这里使用延迟更新的next_model评估

    #使用next_state计算下一个状态的分数
    #[b, 4] -> [b, 2]
    with torch.no_grad():
        target = next_model(next_state)
    #取所有动作中分数最大的
    #[b, 2] -> [b, 1]
    target = target.max(dim=1)[0]
    target = target.reshape(-1, 1)
    #下一个状态的分数乘以一个系数,相当于权重
    target *= 0.98
    #如果next_state已经游戏结束,则next_state的分数是0
    #因为如果下一步已经游戏结束,显然不需要再继续玩下去,也就不需要考虑next_state了.
    #[b, 1] * [b, 1] -> [b, 1]
    target *= (1 - over)
    #加上reward就是最终的分数
    #[b, 1] + [b, 1] -> [b, 1]
    target += reward

    return target


def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()#均方差损失

    #训练N次
    for epoch in range(500):
        #更新N条数据
        update_count, drop_count = update_data()

        #每次更新过数据后,学习N次
        for i in range(200):
            #采样一批数据
            state, action, reward, next_state, over = get_sample()

            #计算一批样本的value和target
            value = get_value(state, action)
            target = get_target(reward, next_state, over)

            #更新参数
            loss = loss_fn(value, target)
            optimizer.zero_grad()#将模型的参数,梯度初始化为0
            loss.backward() #反向传播,计算梯度
            optimizer.step() #所有的optimizer都实现了step()方法，这个方法会更新所有的参数

            #把model的参数复制给next_model
            if (i + 1) % 10 == 0:#延迟更新
                next_model.load_state_dict(model.state_dict())

        if epoch % 50 == 0:
            test_result = sum([test(play=False) for _ in range(20)]) / 20
            print(epoch, len(datas), update_count, drop_count, test_result)


if __name__ == '__main__':
    # 计算动作的模型,也是真正要用的模型
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
    )

    # 经验网络,用于评估一个状态的分数。  经验模型，来评估动作模型；
    # 如果评价的方式总在变，那么训练就很难收殓，所以这里是延迟后几步再更新
    next_model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
    )
    # 把model的参数复制给next_model
    next_model.load_state_dict(model.state_dict())



    # 样本池
    datas = []
    env = GymEnv()
    env.reset()
    train()
    test(play=True)