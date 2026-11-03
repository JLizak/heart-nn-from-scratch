"""Optimizers for MLP: each step() updates the layers from their stored gradients."""

import numpy as np


class SGD:
    """Plain gradient step: w = w - lr * dL/dw."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            layer.weights -= self.lr * layer.dW
            layer.bias -= self.lr * layer.db

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # m and v for every parameter separately, keyed by (layer index, "W" / "b")
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            self.update_param(layer.weights, layer.dW, (i, "W"))
            self.update_param(layer.bias, layer.db, (i, "b"))

    def update_param(self, param, grad, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2

        # Bias correction, m and v start at 0
        m_hat = self.m[key] / (1 - self.beta1**self.t)
        v_hat = self.v[key] / (1 - self.beta2**self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
