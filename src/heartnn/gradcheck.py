"""Compare backprop gradients with central finite differences on a small network."""
import numpy as np

from heartnn.models.functional import binary_cross_entropy
from heartnn.models.mlp import MLP


def loss_of(mlp, x, y):
    return binary_cross_entropy(y, mlp.forward(x))


def numerical_grad(mlp, x, y, param, eps=1e-5):
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        old = param[idx]
        param[idx] = old + eps
        loss_plus = loss_of(mlp, x, y)
        param[idx] = old - eps
        loss_minus = loss_of(mlp, x, y)
        param[idx] = old
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))


def check_gradients(seed=0):
    """Max relative error between backprop and finite differences, per (layer, parameter)."""
    np.random.seed(seed)
    layer_sizes = (6, 5, 4, 1)
    mlp = MLP(len(layer_sizes), 4, layer_sizes, init_std=1.0)
    # Non-zero biases: with zero bias a dead input row gives z == 0 exactly, i.e. the ReLU kink,
    # where finite differences are meaningless
    for layer in mlp.layers:
        layer.bias += np.random.randn(*layer.bias.shape)
    x = np.random.randn(8, 4)
    y = np.random.randint(0, 2, size=(8, 1))

    mlp.forward(x)
    mlp.backward(y)

    errors = {}
    for i, layer in enumerate(mlp.layers):
        for name, param, analytic in (("W", layer.weights, layer.dW), ("b", layer.bias, layer.db)):
            errors[i, name] = relative_error(analytic, numerical_grad(mlp, x, y, param))
    return errors


if __name__ == "__main__":
    errors = check_gradients()
    for (i, name), err in errors.items():
        print(f"layer {i} d{name}: max relative error {err:.2e}")
    print("OK" if max(errors.values()) < 1e-6 else "MISMATCH")
