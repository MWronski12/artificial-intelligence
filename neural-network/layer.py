import numpy as np


class Layer:
    """Base class for layers."""

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError()

    def backward(self, output_gradient, lr):
        raise NotImplementedError


class FCLayer(Layer):
    """Fully connected layer. Column vectors are expected for input and output, supports batch processing."""

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, lr, momentum=0.9):
        batch_size = self.input.shape[1]
        weights_gradient = np.dot(output_gradient, self.input.T) / batch_size
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True) / batch_size

        # Update velocities
        self.velocity_weights = momentum * self.velocity_weights + weights_gradient
        self.velocity_bias = momentum * self.velocity_bias + bias_gradient

        # Update parameters
        self.weights -= lr * self.velocity_weights
        self.bias -= lr * self.velocity_bias

        return np.dot(self.weights.T, output_gradient)
