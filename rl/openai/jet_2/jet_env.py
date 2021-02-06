from collections import deque

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

N_DISCRETE_ACTIONS = 3
HEIGHT = 2
WIDTH = 5
N_CHANNELS = 1


class JetEnv2(gym.Env):

    metadata = {'render.modes': ['human']}

    Enemy_Color = 85
    Bonus_Color = 170

    def __init__(self):
        super(JetEnv2, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self._fig, self._ax = plt.subplots()
        self.reset()

    def step(self, action):
        self._context = np.zeros(self.observation_space.shape)
        self._enemy_q.append(np.random.randint(0, WIDTH))
        self._bonus_q.append(np.random.randint(0, WIDTH))
        if action == 0:
            if self._boxx > 0:
                self._boxx -= 1
        elif action == 1:
            pass
        elif action == 2:
            if self._boxx < WIDTH-1:
                self._boxx += 1
        else:
            raise Exception("error action")

        for i in range(len(self._enemy_q)):
            self._context[len(self._enemy_q) - 1 - i, self._enemy_q[i]] = JetEnv2.Enemy_Color
        for i in range(len(self._bonus_q)):
            self._context[len(self._bonus_q) - 1 - i, self._bonus_q[i]] = JetEnv2.Bonus_Color

        self._context[HEIGHT - 1, self._boxx] = 255
        self._count += 1

        if len(self._bonus_q) == HEIGHT and self._bonus_q[0] == self._boxx:
            self._score += 1
            return self._context, 1, False, None
        elif len(self._enemy_q) == HEIGHT and self._enemy_q[0] == self._boxx:
            return self._context, -1, True, None
        else:
            return self._context, 1, False, None

    def reset(self):
        self._count = 0
        self._score = 0
        self._enemy_q = deque(maxlen=HEIGHT)
        self._bonus_q = deque(maxlen=HEIGHT)
        self._context = np.zeros(self.observation_space.shape)
        self._boxx = int(WIDTH / 2)
        self._context[HEIGHT-1, self._boxx] = 255
        return self._context

    def render(self, mode='human', close=False):
        self._ax.cla()
        self._ax.set_axis_off()
        self._ax.set_title("Steps:{}, Score:{}".format(self._count, self._score), fontsize=20)
        self._ax.imshow(self._context)
        plt.pause(0.1)


if __name__ == "__main__":
    env = JetEnv2()
    observation = env.reset()

    count = 0
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(observation, reward, done, info)
        env.render()
        count += 1
        if done:
            observation = env.reset()
            print(count)
            count = 0
    env.close()
