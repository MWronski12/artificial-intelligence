import time

import numpy as np
import matplotlib.pyplot as plt

from network import Network
from layer import FCLayer
from activation import Sigmoid, ReLU

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


digits = load_digits()

X = (digits.data / 255.0).reshape(-1, 64, 1)
y = np.eye(10)[digits.target].reshape(-1, 10, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
)

SIGMOID = "Sigmoid"
RELU = "ReLU"


def layers_factory(sizes, activation):
    layers = []
    for i in range(len(sizes)):
        if i == len(sizes) - 1:
            activation_layer = Sigmoid()
        elif activation == SIGMOID:
            activation_layer = Sigmoid()
        elif activation == RELU:
            activation_layer = ReLU()
        else:
            raise ValueError("Invalid activation function")
        layers.append(FCLayer(sizes[i][0], sizes[i][1]))
        layers.append(activation_layer)

    return layers


architectures = [
    [(64, 32), (32, 10)],
    [(64, 32), (32, 16), (16, 10)],
    [(64, 32), (32, 24), (24, 16), (16, 10)],
]

batch_sizes = [1, 4, 16, 64]


for activation in [SIGMOID, RELU]:
    for architecture in architectures:
        for batch_size in batch_sizes:
            network = Network(
                layers=layers_factory(architecture, activation),
                batch_size=batch_size,
            )

            start = time.time()
            network.fit(X_train, y_train)
            train_time = time.time() - start

            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(network.predict(X_test), axis=1)

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="micro")

            print("layers:", architecture)
            print("activation:", activation)
            print("batch_size:", network.batch_size)
            print("train_time:", train_time)
            print(f"accuracy: {accuracy:.2%}")
            print(f"f1: {f1:.2%}")
            print()

            title = (
                f"batch={batch_size}, "
                f"f1={f1:.2%}, "
                f"activation={activation}, "
                f"train_time={int(train_time * 10e3)}ms, "
            )
            plt.plot(network.errors)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.savefig(f"{title}, n_layers={len(architecture)}.png")
            plt.close()
