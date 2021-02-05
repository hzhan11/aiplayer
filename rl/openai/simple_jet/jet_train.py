import numpy as np
import random
from keras.optimizers import Adam

from collections import deque

from tensorflow import keras
from tensorflow.keras import layers

from openai.simple_jet.jet_env import JetEnv, WIDTH, HEIGHT

import logging


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

    def create_model(self):
        state_shape = self.env.observation_space.shape
        model = keras.Sequential(
            [
                keras.Input(shape=state_shape),
                #layers.Conv2D(8, kernel_size=(3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(32, activation="relu"),
                layers.Dense(8, activation="relu"),
                #layers.Dropout(0.3),
                layers.Dense(self.env.action_space.n, activation="relu"),
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
        logging.info("training loss avg is {}".format(np.mean(losses)))

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
    env = JetEnv()

    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    #dqn_agent.load_model("model.weights")
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape((1, HEIGHT, WIDTH, 1))
        cur_state = cur_state / 255
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            if display:
                env.render()
            #reward = reward if not done or (done and step == 199) else -10.0
            new_state = new_state.reshape((1, HEIGHT, WIDTH, 1))
            new_state = new_state / 255
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            # print("step {} complete".format(step))
            if done:
                break
        logging.info("step {} complete with e {}. mem {}".format(step, dqn_agent.epsilon, len(dqn_agent.memory)))
        dqn_agent.save_model("model.weights")


def play():
    env = JetEnv()
    dqn_agent = DQN(env=env)
    dqn_agent.load_model("model.weights")
    while True:
        cur_state = env.reset().reshape((1, HEIGHT, WIDTH, 1)) / 255
        for step in range(2000):
            env.render()
            action = dqn_agent.act(cur_state, True)
            new_state, reward, done, _ = env.step(action)
            cur_state = new_state.reshape((1, HEIGHT, WIDTH, 1)) / 255
            if done:
                print(step)
                break


if __name__ == "__main__":
    if False:
        file_handler = logging.FileHandler(filename='training.log')
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(level=logging.INFO, handlers=handlers)
        #logging.basicConfig(filename='training.log', level=logging.INFO)
        logging.info('Started')
        train(True)
    else:
        play()
