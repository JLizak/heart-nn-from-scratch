import numpy as np


def sigmoid(x):
    # Clipping avoids overflow in exp for very large logits (e.g. on unnormalized data)
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def binary_cross_entropy(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    loss = -1 * y * np.log(p) - (1 - y) * np.log(1 - p)
    return np.mean(loss)


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def identity(z):
    return z


def identity_deriv(z):
    return np.ones_like(z)
