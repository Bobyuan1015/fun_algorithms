import gym
from matplotlib import pyplot as plt
import os

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


class GymEnv(gym.Wrapper):

    def __init__(self, name='CartPole-v1', continuous=0):
        env = gym.make(name, render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.continuous = continuous
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
         state, reward, done, info
        """
        if self.continuous == 1:
            state, reward, terminated, truncated, info = self.env.step([action])
        else:
            state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, info

    def show(self):
        """
        This function display the game.
        """
        plt.imshow(self.env.render())
        plt.show()
