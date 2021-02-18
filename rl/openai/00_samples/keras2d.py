import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.python.keras.optimizers import Adam, SGD

W = 8
H = 8
C = 3

CATA = 3


def build_data(c=16):
    X = []
    Y = []
    for c in range(c):
        xx = np.random.rand(H, W, C)
        yy = np.zeros(CATA)
        rr = np.random.randint(0, CATA)
        xx = xx / (rr+1)
        yy[rr] = 1
        X.append(xx)
        Y.append(yy)
    return np.array(X), np.array(Y)


model = keras.Sequential(
    [
        keras.Input(shape=(H, W, C)),
        layers.Flatten(),
        layers.Dense(32),
        layers.Dense(16),
        layers.Dense(CATA, activation='softmax'),
    ]
)

# model.compile(loss='mse', metrics=['accuracy',])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy', ])
#for cc in range(500):
X, Y = build_data(1000)
r = model.fit(X, Y, epochs=30, verbose=1)
#print(cc, r.history["loss"][0])

XT, YT = build_data(16)
model.evaluate(X, Y)
