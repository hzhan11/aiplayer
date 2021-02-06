from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL

from rl.maze_simple1.maze import Maze


class Agent:

    def __init__(self, maze):
        self.maze = maze
        self.MAX_EPISODES = 1000
        self.EPSILON = 0.95
        self.GAMMA = 0.9
        self.ALPHA = 0.1
        self.count = 0
        self.fig, self.ax = plt.subplots()

    def build_q_table(self):
        self.table = np.zeros((Maze.Y, Maze.X, len(Maze.ACTIONS)))
        #print(self.table)

    def choose_next_action(self, state, train=True):
        state_actions = self.table[state[0], state[1], :]
        if train and ((np.random.uniform() > self.EPSILON) or (state_actions.all() == 0)):
            action_name = np.random.choice(Maze.ACTIONS)
        else:
            action_name = Maze.ACTIONS[state_actions.argmax()]
        return action_name

    def update(self):
        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.set_title(str(self.count), fontsize=20)
        total = (self.table.shape[0])
        for yy in range(0, self.table.shape[0]+1):
            x_axis = np.linspace(0, total)
            y_axis = np.zeros(50) + yy
            self.ax.plot(x_axis, y_axis, color="blue")
            self.ax.plot(y_axis, x_axis, color="blue")

        for yy in range(self.table.shape[0]-1, -1, -1):
            for xx in range(0, self.table.shape[1]):
                #left
                text = "{:.2f}".format(self.table[self.table.shape[0] - 1 - yy, xx][0])
                if text != "0.00":
                    self.ax.text(0.05 + xx, 0.5 + yy, str(text), fontsize=8)
                # up
                text = "{:.2f}".format(self.table[self.table.shape[0] - 1 - yy, xx][1])
                if text != "0.00":
                    self.ax.text(0.4 + xx, 0.80 + yy, str(text), fontsize=8)
                # right
                text = "{:.2f}".format(self.table[self.table.shape[0] - 1 - yy, xx][2])
                if text != "0.00":
                    self.ax.text(0.60 + xx, 0.5 + yy, str(text), fontsize=8)
                # down
                text = "{:.2f}".format(self.table[self.table.shape[0] - 1 - yy, xx][3])
                if text != "0.00":
                    self.ax.text(0.4 + xx, 0.1 + yy, str(text), fontsize=8)

        self.count+=1
        #plt.pause(Maze.REFRESH_WAIT)

    def train(self):
        self.build_q_table()
        for episode in range(self.MAX_EPISODES):
            s0 = self.maze.reset()
            self.update()
            while True:
                action = self.choose_next_action(s0)
                end, reward, s1 = self.maze.next(action)
                self.update()
                #print(s0, action, end, reward, s1)
                q_predict = self.table[s0[0], s0[1], Maze.ACTIONS.index(action)]
                if end:
                    q_target = reward
                else:
                    q_target = reward + self.GAMMA * self.table[s1[0], s1[1], :].max()
                self.table[s0[0], s0[1], Maze.ACTIONS.index(action)] += self.ALPHA * (q_target - q_predict)
                s0 = s1
                if end:
                    #print(self.table)
                    break
        print("training completed")

    def play(self):
        s0 = self.maze.reset()
        while True:
            action = self.choose_next_action(s0, False)
            end, reward, s1 = self.maze.next(action)
            s0 = s1
            if end:
                break


if __name__ == '__main__':
    m = Maze()
    a = Agent(m)
    a.train()
    #a.play()
