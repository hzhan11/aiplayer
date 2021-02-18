from collections import deque

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

W = 32
H = 20
C = 3

LOW_P = 0.1
HIGH_P = 0.6
CLASS = 3


class JumpEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def rand_color(self):
        c_list = [0, 125, 255]
        return np.random.choice(c_list, 3)

    def __init__(self):
        super(JumpEnv, self).__init__()
        self.action_space = spaces.Discrete(CLASS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, C), dtype=np.uint8)

        self._fig, self._ax = plt.subplots()
        self.reset()

    def generate_next_state(self):
        yy = np.random.uniform(low=LOW_P, high=HIGH_P)
        #xx = np.array(np.random.rand(H, W, C) * 255, dtype=int)
        xx = np.array(np.ones((H, W, C)) * 100, dtype=int)
        # 1. draw block 1
        if np.random.rand() > 0.5:
            x1 = int(W / 2 + W / 2 * yy)
            x2 = int(W / 2 - W / 2 * yy)
        else:
            x1 = int(W / 2 - W / 2 * yy)
            x2 = int(W / 2 + W / 2 * yy)
        y1 = int(H / 2 - H / 2 * yy)
        y2 = int(H / 2 + H / 2 * yy)

        lenx1 = np.random.randint(3, 7)
        leny1 = np.random.randint(1, 3)
        xx[y1 - leny1:y1 + leny1, x1 - lenx1:x1 + lenx1, :] = self.rand_color()

        # 2. generate several blocks to the bottom
        x3 = int(W / 2)
        y3 = H - 4
        z1 = np.polyfit([x2, x3], [y2, y3], 1)
        sx, ex = 0, 0
        if x2 < x3:
            sx = x2
            ex = x3
        else:
            sx = x3
            ex = x2
        xxtmp = np.random.uniform(low=sx, high=ex, size=(3,))
        for x in xxtmp:
            xb = int(x)
            yb = int(z1[0] * xb + z1[1])
            lenx3 = np.random.randint(3, 7)
            leny3 = np.random.randint(1, 3)
            xx[yb - leny3:yb + leny3, xb - lenx3:xb + lenx3, :] = self.rand_color()

        # 3. draw box2 and character
        lenx2 = np.random.randint(3, 7)
        leny2 = np.random.randint(1, 3)
        xx[y2 - leny2:y2 + leny2, x2 - lenx2:x2 + lenx2, :] = self.rand_color()
        xx[y2 - 6:y2, x2 - 2:x2 + 3, :] = [0, 0, 0]

        yy = int((yy - LOW_P) / ((HIGH_P - LOW_P) / CLASS))

        return xx, yy

    def step(self, action):
        if action != self._next_action:
            self._context = np.zeros((H, W, C), dtype=int)
            return self._context, -1, True, None
        else:
            self._score += 1
            self._context, self._next_action = self.generate_next_state()
            return self._context, 1, False, None

    def reset(self):
        self._count = 0
        self._score = 0
        self._context, self._next_action = self.generate_next_state()
        return self._context

    def render(self, mode='human', close=False):
        self._ax.cla()
        self._ax.set_axis_off()
        self._ax.set_title("Steps:{}, Score:{}".format(self._count, self._score), fontsize=20)
        self._ax.imshow(self._context)
        plt.pause(1)


if __name__ == "__main__":
    env = JumpEnv()
    observation = env.reset()
    env.render()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            observation = env.reset()
            env.render()
    env.close()
