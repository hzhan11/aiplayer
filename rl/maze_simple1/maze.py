import matplotlib.pyplot as plt
import numpy as np


class Maze:
    X = 5
    Y = 5
    YOU = 80
    TARGET = 160
    BLOCK = 240
    ACTIONS = ['left', 'up', 'right', 'down']

    REFRESH_WAIT = 0.01
    END_WAIT = 0.05

    def __init__(self):
        self.x_target = Maze.X - 1
        self.y_target = Maze.Y - 1
        self.life = 20
        self.generate_block()
        self.fig, self.ax = plt.subplots()

    def generate_block(self):
        self.blocks = []
        for yy in range(1, Maze.Y - 1):
            for xx in range(1, Maze.X - 1):
                if np.random.uniform() > 0.7:
                    self.blocks.append([yy, xx])

    def move_left(self):
        if self.x > 0:
            self.x -= 1

    def move_up(self):
        if self.y > 0:
            self.y -= 1

    def move_right(self):
        if self.x < Maze.X - 1:
            self.x += 1

    def move_down(self):
        if self.y < Maze.Y - 1:
            self.y += 1

    def reset(self):
        self.x = 0
        self.y = 0
        self.step_count = 0
        self.life = 20
        self.update()
        return self.y, self.x

    def next(self, action):
        bxy = [self.y, self.x]
        eval("self.move_" + action + "()")
        self.step_count += 1
        end = False
        reward = 0
        if self.x == self.x_target and self.y == self.y_target:
            end = True
            reward = 10
        elif [self.y, self.x] in self.blocks:
            self.y = bxy[0]
            self.x = bxy[1]
            end = False
            reward = -2
            self.life -= 2
            if self.life == 0:
                end = True
                reward = -10
        else:
            end = False
            reward = 0
        self.update()
        return end, reward, (self.y, self.x)

    def update(self):
        array = np.zeros((Maze.Y, Maze.X), dtype=int)
        self.ax.cla()
        self.ax.set_axis_off()
        if self.step_count != 0:
            for b in self.blocks:
                array[b[0], b[1]] = Maze.BLOCK
            array[self.y, self.x] = Maze.YOU
            array[self.y_target, self.x_target] = Maze.TARGET
            self.ax.set_title("Current:{}".format(self.life), fontsize=20)
            self.ax.imshow(array)
            plt.pause(Maze.REFRESH_WAIT)
        else:
            self.ax.set_title("Win/End", fontsize=20)
            self.ax.imshow(array)
            plt.pause(Maze.END_WAIT)
