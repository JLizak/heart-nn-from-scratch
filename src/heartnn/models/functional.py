import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y, p):
    p = np.clip(p, 1e-15, 1-1e-15)
    loss = -1 * y * np.log(p) - (1-y) * np.log(1-p)
    return np.mean(loss)
