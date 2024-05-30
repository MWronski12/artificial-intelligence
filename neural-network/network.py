import numpy as np

from loss import mse, mse_prime

from sklearn.utils import shuffle


class Network:
    def __init__(
        self,
        layers,
        loss=mse,
        loss_prime=mse_prime,
        epochs=1000,
        lr=0.01,
        batch_size=1,
    ):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        self.epochs = epochs
        self.lr = lr
        self.batch_size = float(batch_size)
        self.errors = []

    def fit(self, X_train, y_train):
        """Column vectors are expected for X_train and y_train."""

        for e in range(self.epochs):
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
            X_batches, y_batches = self._make_batches(X_train, y_train)

            error = 0
            for x, y in zip(X_batches, y_batches):
                output = self._forward(x)
                error += mse(y, output)
                output_gradient = self.loss_prime(y, output)
                self._backward(output_gradient)

            error /= len(x)
            self.errors.append(error)

    def predict(self, X):
        return np.array([self._forward(x) for x in X])

    def _forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, self.lr)
        return output_gradient

    def _make_batches(self, X_train, y_train):
        batch_size = int(self.batch_size)
        X_batches = [
            np.hstack(X_train[i : i + batch_size])
            for i in range(0, len(X_train), batch_size)
        ]
        y_batches = [
            np.hstack(y_train[i : i + batch_size])
            for i in range(0, len(y_train), batch_size)
        ]
        return X_batches, y_batches
