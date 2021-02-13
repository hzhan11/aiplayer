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


def rand_color():
    c_list = [0, 125, 255]
    return np.random.choice(c_list, 3)


def build_data(COUNT=10000):
    X_List = []
    # Y = np.random.rand(COUNT)
    Y_List = np.random.uniform(low=0.1, high=0.6, size=(COUNT,))
    for c in range(COUNT):

        xx = np.array(np.random.rand(H, W, C) * 255, dtype=int)
        yy = Y_List[c]

        # 1. draw block 1
        if np.random.rand() > 0.5:
            x1 = int(W / 2 + W / 2 * yy)
            x2 = int(W / 2 - W / 2 * yy)
        else:
            x1 = int(W / 2 - W / 2 * yy)
            x2 = int(W / 2 + W / 2 * yy)
        y1 = int(H / 2 - H / 2 * yy)
        y2 = int(H / 2 + H / 2 * yy)

        lenx1 = np.random.randint(7, 13)
        leny1 = np.random.randint(3, 7)
        xx[y1 - leny1:y1 + leny1, x1 - lenx1:x1 + lenx1, :] = rand_color()

        # 2. generate several blocks to the bottom
        x3 = int(W / 2)
        y3 = H - 8
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
            lenx3 = np.random.randint(7, 13)
            leny3 = np.random.randint(3, 7)
            xx[yb - leny3:yb + leny3, xb - lenx3:xb + lenx3, :] = rand_color()

        # 3. draw box2 and character
        lenx2 = np.random.randint(7, 13)
        leny2 = np.random.randint(3, 7)
        xx[y2 - leny2:y2 + leny2, x2 - lenx2:x2 + lenx2, :] = rand_color()
        xx[y2 - 12:y2, x2 - 3:x2 + 3, :] = [0,0,0]

        #plt.figure()
        #plt.imshow(xx)
        #plt.show()

        X_List.append(xx)
    return np.array(X_List), np.array(Y_List)


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

    if cc % 500 == 0:
        XT, YT = build_data(16)
        YY = model.predict(XT)
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(XT[i])
            plt.title("{:.3f},{:.3f}".format(YT[i], YY[i][0]))
            plt.axis("off")
        plt.show()
