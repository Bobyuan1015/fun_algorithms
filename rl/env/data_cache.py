


import random




#数据池
def random_pic(pool_list):
    return random.choice(pool_list)

class Pool:

    def __init__(self, play, sample=None):
        self.pool = []
        if not sample:
            self.sample_func = random_pic
        self.sample_func = sample
        self.play_func = play

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    #更新动作池
    def update(self):
        #每次更新不少于N条新数据
        old_len = len(self.pool)
        while len(self.pool) - old_len < 200:
            self.pool.extend(self.play_func()[0])

        #只保留最新的N条数据
        self.pool = self.pool[-1_0000:]

    #获取一批数据样本
    def sample(self):
        return self.sample_func(self.pool)


