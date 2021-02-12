# This sample code is for jump a jump using.
# It is regression example to simulate the game
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.python.keras.optimizers import Adam, SGD

W = 128
H = 80
C = 3


def build_data(COUNT=10000):
    X = []
    # Y = np.random.rand(COUNT)
    Y = np.random.uniform(low=0.2, high=0.8, size=(COUNT,))
    for c in range(COUNT):
        xx = np.array(np.random.rand(H, W, C) * 255, dtype=int)
        yy = Y[c]

        if np.random.rand() > 0.5:
            x1 = int(W / 2 + W / 2 * yy)
            x2 = int(W / 2 - W / 2 * yy)
        else:
            x1 = int(W / 2 - W / 2 * yy)
            x2 = int(W / 2 + W / 2 * yy)
        y1 = int(H / 2 - H / 2 * yy)
        y2 = int(H / 2 + H / 2 * yy)

        xx[y1 - 5:y1 + 5, x1 - 10:x1 + 10, :] = [255, 255, 255]
        xx[y2 - 5:y2 + 5, x2 - 10:x2 + 10, :] = [0, 0, 0]

        X.append(xx)
    return np.array(X), np.array(Y)


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
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(32),
        layers.Dropout(0.3),
        layers.Dense(1),
    ]
)

# model.compile(loss='mse', metrics=['accuracy',])
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()
for cc in range(5000):
    X, Y = build_data(16)
    r = model.fit(X, Y, epochs=1, verbose=0)
    print("{},loss is {:.4f}".format(cc, r.history["loss"][0]))
#X, Y = build_data()
#r = model.fit(X, Y, epochs=50, verbose=1)
XT, YT = build_data(16)
YY = model.predict(XT)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(XT[i])
    plt.title("{:.3f},{:.3f}".format(YT[i], YY[i][0]))
    plt.axis("off")
plt.show()
