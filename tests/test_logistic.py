import numpy as np
from sklearn.datasets import make_classification

from heartnn.models.logistic import LogisticReg


def make_data():
    return make_classification(n_samples=300, n_features=5, random_state=0)


def test_zero_init_starts_at_ln2():
    X, y = make_data()
    model = LogisticReg(init="zeros", n_epochs=1, lr=1e-12).fit(X, y)
    assert np.isclose(model.history_[0], np.log(2))


def test_loss_decreases():
    X, y = make_data()
    model = LogisticReg(lr=0.1, n_epochs=200, random_state=0).fit(X, y)
    assert len(model.history_) == 200
    assert model.history_[-1] < model.history_[0]


def test_predict_returns_classes():
    X, y = make_data()
    model = LogisticReg(n_epochs=50, random_state=0).fit(X, y)
    assert set(np.unique(model.predict(X))) <= {0, 1}
    assert model.predict_proba(X).shape == (len(X), 2)


def test_tol_stops_early():
    X, y = make_data()
    model = LogisticReg(lr=0.1, n_epochs=5000, tol=1e-4, random_state=0).fit(X, y)
    assert model.n_iter_ < 5000
    assert model.n_iter_ == len(model.history_)


def test_no_tol_runs_all_epochs():
    X, y = make_data()
    model = LogisticReg(n_epochs=100, random_state=0).fit(X, y)
    assert model.n_iter_ == 100
