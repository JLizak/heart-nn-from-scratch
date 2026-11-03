import numpy as np
import pytest
from sklearn.datasets import make_classification

from heartnn.gradcheck import check_gradients
from heartnn.models.functional import binary_cross_entropy
from heartnn.models.mlp import MLP
from heartnn.models.optim import SGD, Adam
from heartnn.preprocessing import load_column_splits


def make_data():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    return X, y.reshape(-1, 1).astype(float)


def make_mlp(n_features, **kwargs):
    np.random.seed(0)
    return MLP(2, n_features, (8, 1), **kwargs)


def test_backprop_matches_finite_differences():
    assert max(check_gradients().values()) < 1e-6


def test_output_shapes():
    X, y = make_data()
    mlp = make_mlp(X.shape[1])
    assert mlp.forward(X).shape == y.shape
    assert mlp.predict(X).shape == y.shape


def test_he_init_std_depends_on_layer_input():
    np.random.seed(0)
    mlp = MLP(2, 1000, (500, 1), init="he")
    assert np.isclose(mlp.layers[0].weights.std(), np.sqrt(2 / 1000), rtol=0.05)
    assert np.isclose(mlp.layers[1].weights.std(), np.sqrt(2 / 500), rtol=0.3)


@pytest.mark.parametrize("optimizer", [SGD(0.1), Adam(0.01)], ids=["sgd", "adam"])
def test_optimizer_reduces_loss(optimizer):
    X, y = make_data()
    mlp = make_mlp(X.shape[1], init_std=0.1)
    start = binary_cross_entropy(y, mlp.forward(X))
    for _ in range(200):
        mlp.forward(X)
        mlp.backward(y)
        optimizer.step(mlp.layers)
    assert binary_cross_entropy(y, mlp.forward(X)) < start


def test_update_equals_sgd_step():
    X, y = make_data()
    a, b = make_mlp(X.shape[1], init_std=0.1), make_mlp(X.shape[1], init_std=0.1)
    for mlp in (a, b):
        mlp.forward(X)
        mlp.backward(y)
    a.update(0.1)
    SGD(0.1).step(b.layers)
    assert np.array_equal(a.layers[0].weights, b.layers[0].weights)


def test_column_splits_shape_and_normalization():
    X_train, X_val, X_test, y_train, y_val, y_test = load_column_splits()
    assert [len(s) for s in (X_train, X_val, X_test)] == [550, 184, 184]
    assert y_train.shape == (550, 1)
    assert X_train.shape[1] == 27
    raw = load_column_splits(normalize=False)[0]
    assert raw.max() > 100  # e.g. cholesterol, left on its original scale
