import numpy as np
from IPython import display
import time

#悬崖游戏


class CliffEnv(object):
    def __init__(self):
        self.values = np.zeros([4, 12])
        # 初始化每个格子下采用动作的概率


    #打印游戏，方便测试
    def show(self, row, col, action):
        graph = [
            '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
            '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
            '□', '□', '□', '□', '□', '□', '□', '□', '□', '○', '○', '○', '○', '○',
            '○', '○', '○', '○', '○', '❤'
        ]
        action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
        graph[row * 12 + col] = action
        graph = ''.join(graph)
        for i in range(0, 4 * 12, 12):
            print(graph[i:i + 12])
        print()
        print()


    #获取一个格子的状态
    def get_state(self, row, col):
        """
        这是一个自定义函数

        params:
            params1: 第一个参数
            params2: 第二个参数

        return:
            {'data': {}, 'status': 200}
        """
        if row != 3:
            return 'ground'
        if row == 3 and col == 0:
            return 'ground'
        if row == 3 and col == 11:
            return 'terminal'
        return 'trap'

    #在一个格子里做一个动作
    def move(self, row, col, action):
        #如果当前已经在陷阱或者终点，则不能执行任何动作，反馈都是0
        if self.get_state(row, col) in ['trap', 'terminal']:
            return row, col, 0

        if action == 0: #↑
            row -= 1
        elif action == 1: #↓
            row += 1
        elif action == 2: #←
            col -= 1
        elif action == 3: #→
            col += 1
        else:
            pass
        #不允许走到地图外面去
        row = max(0, row)
        row = min(3, row)
        col = max(0, col)
        col = min(11, col)

        #是陷阱的话，奖励是-100，否则都是-1
        #这样强迫了机器尽快结束游戏,因为每走一步都要扣一分
        #结束最好是以走到终点的形式,避免被扣100分
        reward = -1
        if self.get_state(row, col) == 'trap':
            reward = -100
        return row, col, reward

    def final_result(self,pi):
        # 打印所有格子的动作倾向
        for row in range(4):
            line = ''
            for col in range(12):
                action = pi[row, col].argmax()
                action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
                line += action
            print(line)


