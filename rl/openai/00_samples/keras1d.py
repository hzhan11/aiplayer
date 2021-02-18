import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.python.keras.optimizers import Adam, SGD

LEN_OF_X = 200
CATA = 3


def build_data(COUNT=6000):
    X = []
    Y = []
    for c in range(COUNT):
        xx = np.random.rand(LEN_OF_X)
        #xx = np.zeros(LEN_OF_X)
        yy = np.zeros(CATA)
        rr = np.random.randint(0, CATA)
        #dist = int((rr + 1) * LEN_OF_X / 5)
        #x1 = np.random.randint(0, LEN_OF_X / 3)
        #x2 = x1 + dist
        #xx[x1:x2] = 255
        xx = xx / (rr+1)
        yy[rr] = 1
        X.append(xx)
        Y.append(yy)
    return np.array(X), np.array(Y)


model = keras.Sequential(
    [
        keras.Input(shape=LEN_OF_X),
        # layers.Conv1D(16, 3),
        # layers.Conv1D(16, kernel_size=3, input_shape=LEN_OF_X, activation='relu'),
        # Rescaling(scale=1.0 / 10),
        # layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
        # layers.Flatten(),
        #layers.Dense(512),
        #layers.Dropout(0.1),
        #layers.Dense(128),
        layers.Dense(32),
        layers.Dropout(0.1),
        # layers.Dense(32, activation='softmax'),
        layers.Dense(CATA, activation='softmax'),
    ]
)

# model.compile(loss='mse', metrics=['accuracy',])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy', ])
# for cc in range(500):
#    X, Y = build_data(16)
#    r = model.fit(X, Y, epochs=1, verbose=0)
#    print(cc, r.history["loss"][0])
X, Y = build_data()
r = model.fit(X, Y, epochs=10, verbose=1)
XT, YT = build_data(16)
model.evaluate(X, Y)
