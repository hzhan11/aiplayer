from collections import deque

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

N_DISCRETE_ACTIONS = 3
HEIGHT = 3
WIDTH = 3
N_CHANNELS = 1


# This is the simple pixel game
# It has 5 x 5 with one block at each line
# You can control your character to avoid meet the block

class TempEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TempEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.fig, self.ax = plt.subplots()
        self.reset()

    def step(self, action):
        self._context = np.zeros(self.observation_space.shape)
        self._boxq.append(np.random.randint(0, WIDTH))
        if action == 0:
            if self._boxx > 0:
                self._boxx -= 1
        elif action == 1:
            pass
        elif action == 2:
            if self._boxx < WIDTH - 1:
                self._boxx += 1
        else:
            raise Exception("error action")

        for i in range(len(self._boxq)):
            self._context[len(self._boxq) - 1 - i, self._boxq[i]] = 128
        self._context[HEIGHT - 1, self._boxx] = 255
        self.count += 1

        if len(self._boxq) == HEIGHT and self._boxq[0] == self._boxx:
            return self._context, -1, True, None
        else:
            return self._context, 1, False, None

    def reset(self):
        self.count = 0
        self._boxq = deque(maxlen=HEIGHT)
        self._context = np.zeros(self.observation_space.shape)
        self._boxx = int(WIDTH / 2)
        self._context[HEIGHT - 1, self._boxx] = 255
        return self._context

    def render(self, mode='human', close=False):
        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.set_title("Steps:{}".format(self.count), fontsize=20)
        self.ax.imshow(self._context)
        plt.pause(0.1)


if __name__ == "__main__":
    env = TempEnv()
    observation = env.reset()

    count = 0
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(observation, reward, done, info)
        env.render()
        count += 1
        if done:
            observation = env.reset()
            print(count)
            count = 0
    env.close()
