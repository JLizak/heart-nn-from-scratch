"""Multilayer perceptron with backpropagation, implemented in NumPy."""

import numpy as np

from heartnn.models.functional import identity, identity_deriv, relu, relu_deriv, sigmoid


class Layer:
    def __init__(self, weights, bias, activation, activation_deriv):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.cache = {}

    def forward(self, x):
        z = x @ self.weights + self.bias
        self.cache["z"] = z
        self.cache["x"] = x
        return self.activation(z)

    def backward(self, grad):
        dz = grad * self.activation_deriv(self.cache["z"])
        self.dW = self.cache["x"].T @ dz
        self.db = np.sum(dz, axis=0, keepdims=True)
        dx = dz @ self.weights.T

        return dx

    def update(self, lr):
        self.weights -= lr * self.dW
        self.bias -= lr * self.db

class MLP:
    def __init__(self, n_layers, input_size, layer_sizes, init_std=0.01, init="normal"):
        self.n_layers = n_layers
        self.init_std = init_std
        self.init = init
        self.layers: list[Layer] = []

        for n in range(0, n_layers):
            # He: std depends on the input size of each layer, keeps the variance of Wx+b for ReLU
            if init == "he":
                std = np.sqrt(2 / input_size)
            else:
                std = self.init_std
            weights = np.random.randn(input_size, layer_sizes[n]) * std
            bias = np.zeros((1, layer_sizes[n]))
            if n == n_layers - 1:
                layer = Layer(weights, bias, identity, identity_deriv)
            else:
                layer = Layer(weights, bias, relu, relu_deriv)
            self.layers.append(layer)

            input_size = layer_sizes[n]

    def print_layers(self):
        for layer in self.layers:
            print(f"Weights size:  {np.shape(layer.weights)}")

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer.forward(z)

        self.p = sigmoid(z)
        return self.p

    def backward(self, y):
        # Gradient of mean BCE w.r.t. the logit of the last layer (sigmoid and BCE combined)
        grad = (self.p - y) / y.shape[0]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def predict(self, x):
        return (self.forward(x) >= 0.5).astype(int)
