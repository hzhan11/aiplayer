import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.maze_simple1.maze import Maze


class Agent:

    def __init__(self, maze):
        self.maze = maze
        self.MAX_EPISODES = 500
        self.EPSILON = 0.9
        self.GAMMA = 0.9
        self.ALPHA = 0.1

        # 后向观测算法, eligibility trace.
        self.TRACE_DECAY = 0.9

        self.count = 0
        self.fig, self.ax = plt.subplots()
        self.fig1, self.ax1 = plt.subplots()

    def build_q_table(self):
        self.table = np.zeros((Maze.Y, Maze.X, len(Maze.ACTIONS)))
        self.eligibility_trace = np.zeros((Maze.Y, Maze.X, len(Maze.ACTIONS)))

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
        for yy in range(0, self.table.shape[0] + 1):
            x_axis = np.linspace(0, total)
            y_axis = np.zeros(50) + yy
            self.ax.plot(x_axis, y_axis, color="blue")
            self.ax.plot(y_axis, x_axis, color="blue")
        for yy in range(self.table.shape[0] - 1, -1, -1):
            for xx in range(0, self.table.shape[1]):
                # left
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

        self.count += 1

        self.ax1.cla()
        self.ax1.set_axis_off()
        for yy in range(0, self.eligibility_trace.shape[0] + 1):
            x_axis = np.linspace(0, total)
            y_axis = np.zeros(50) + yy
            self.ax1.plot(x_axis, y_axis, color="gray")
            self.ax1.plot(y_axis, x_axis, color="gray")
        for yy in range(self.eligibility_trace.shape[0] - 1, -1, -1):
            for xx in range(0, self.eligibility_trace.shape[1]):
                # left
                text = "{:.2f}".format(self.eligibility_trace[self.eligibility_trace.shape[0] - 1 - yy, xx][0])
                if text != "0.00":
                    self.ax1.text(0.05 + xx, 0.5 + yy, str(text), fontsize=8)
                # up
                text = "{:.2f}".format(self.eligibility_trace[self.eligibility_trace.shape[0] - 1 - yy, xx][1])
                if text != "0.00":
                    self.ax1.text(0.4 + xx, 0.80 + yy, str(text), fontsize=8)
                # right
                text = "{:.2f}".format(self.eligibility_trace[self.eligibility_trace.shape[0] - 1 - yy, xx][2])
                if text != "0.00":
                    self.ax1.text(0.60 + xx, 0.5 + yy, str(text), fontsize=8)
                # down
                text = "{:.2f}".format(self.eligibility_trace[self.eligibility_trace.shape[0] - 1 - yy, xx][3])
                if text != "0.00":
                    self.ax1.text(0.4 + xx, 0.1 + yy, str(text), fontsize=8)

        # plt.pause(Maze.REFRESH_WAIT)

    def train(self):
        self.build_q_table()
        for episode in range(self.MAX_EPISODES):
            s0 = self.maze.reset()
            action0 = self.choose_next_action(s0)
            self.update()

            while True:
                end, reward, s1 = self.maze.next(action0)
                action1 = self.choose_next_action(s1)
                # print(s0, action0, end, reward, s1, action1)
                self.update()
                q_predict = self.table[s0[0], s0[1], Maze.ACTIONS.index(action0)]
                if end:
                    q_target = reward
                else:
                    q_target = reward + self.GAMMA * self.table[s1[0], s1[1], Maze.ACTIONS.index(action1)]
                error = q_target - q_predict
                #Method 1:
                self.eligibility_trace[s0[0], s0[1], :] *= 0
                self.eligibility_trace[s0[0], s0[1], Maze.ACTIONS.index(action0)] = 1

                #self.table[s0[0], s0[1], Maze.ACTIONS.index(action0)] += self.ALPHA * (q_target - q_predict)
                self.table += self.ALPHA * error * self.eligibility_trace
                self.eligibility_trace *= self.GAMMA * self.TRACE_DECAY
                s0 = s1
                action0 = action1
                if end:
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
    a.play()
