import gym
from matplotlib import pyplot as plt
import os


for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


class GymEnv(gym.Wrapper):

    def __init__(self, continuous=False):
        self.continuous = continuous
        if continuous:
            env = gym.make('CartPole-v1', render_mode='rgb_array')
        else:
            env = gym.make('FrozenLake-v1',
                           render_mode='rgb_array',
                           is_slippery=False)
        # env = gym.make(name, render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state, _ = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        """
        This function takes an action in the game.

        Parameters:
        action

        Returns:
         state, reward, end, info
        """
        # if self.continuous == 1:
        #     a = [action]
        # else:
        #     a = action
        a = action
        state, reward, terminated, truncated, info = self.env.step(a)
        end = terminated or truncated
        self.step_n += 1
        if self.continuous:
            if self.step_n >= 200:
                end = True
            # 没坚持到最后,扣分
            if end and self.step_n < 200:
                reward = -1000
        return state, reward, end

    def show(self, is_display=False):
        """
        This function display the game.
        """

        plt.imshow(self.env.render())
        if is_display:
            plt.axis('off')  # 隐藏坐标轴
            plt.draw()
            plt.pause(0.001)  # 暂停以更新图像

