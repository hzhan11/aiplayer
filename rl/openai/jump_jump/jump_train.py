import sys

import numpy as np
import random
from keras.optimizers import Adam

from collections import deque

from tensorflow import keras
from tensorflow.keras import layers

import logging

from tensorflow.python.keras.layers import Rescaling

from rl.openai.jump_jump.jump_env import JumpEnv, H, W, C, CLASS


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.train_loss = -1

    def create_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(H, W, C)),
                Rescaling(scale=1.0 / 255),
                layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.1),
                layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.1),
                # layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
                # layers.MaxPooling2D(pool_size=(2, 2)),
                # layers.Dropout(0.1),
                layers.Flatten(),
                # layers.Dense(128, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(32),
                layers.Dropout(0.3),
                layers.Dense(CLASS, activation='softmax'),
            ]
        )
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def act(self, state, is_play=False):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon and not is_play:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        losses = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            loss = self.model.fit(state, target, epochs=1, verbose=0).history["loss"][0]
            losses.append(loss)
        self.train_loss = np.mean(losses)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save_weights(fn)

    def load_model(self, fn):
        self.model.load_weights(fn)


def train(display=False):
    env = JumpEnv()

    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    # dqn_agent.load_model("model.weights")
    for trial in range(trials):
        cur_state = env.reset().reshape((1, H, W, C))
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            if display:
                env.render()
            # reward = reward if not done or (done and step == 199) else -10.0
            new_state = new_state.reshape((1, H, W, C))
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            # print("step {} complete".format(step))
            if done:
                break
        logging.info("step {} complete with e {}. mem {}, training loss avg is {}".format(step, dqn_agent.epsilon, len(dqn_agent.memory), dqn_agent.train_loss))
        if trial % 100 == 0:
            dqn_agent.save_model("model/model.weights")


def play():
    env = JumpEnv()
    dqn_agent = DQN(env=env)
    dqn_agent.load_model("model/model.weights")
    while True:
        cur_state = env.reset().reshape((1, H, W, C))
        for step in range(2000):
            env.render()
            action = dqn_agent.act(cur_state, True)
            new_state, reward, done, _ = env.step(action)
            cur_state = new_state.reshape((1, H, W, C))
            if done:
                print(step)
                break


if __name__ == "__main__":
    if True:
        file_handler = logging.FileHandler(filename='training.log')
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(level=logging.INFO, handlers=handlers)
        # logging.basicConfig(filename='training.log', level=logging.INFO)
        logging.info('Started')
        train(False)
    else:
        play()
