"""Binary logistic regression trained with gradient descent, implemented in NumPy."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from heartnn.models.functional import binary_cross_entropy, sigmoid


class LogisticReg(ClassifierMixin, BaseEstimator):
    def __init__(
        self, lr=0.1, n_epochs=300, init="normal", init_scale=0.01, random_state=None, tol=None
    ):
        self.lr = lr
        self.n_epochs = n_epochs
        self.init = init
        self.init_scale = init_scale
        self.random_state = random_state
        self.tol = tol

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.random_state)
        if self.init == "zeros":
            self.coef_ = np.zeros(n_features)
        elif self.init == "normal":
            self.coef_ = rng.normal(0.0, self.init_scale, size=n_features)
        elif self.init == "uniform":
            self.coef_ = rng.uniform(-self.init_scale, self.init_scale, size=n_features)

        self.intercept_ = 0.0
    
    def _forward(self, X):
        return sigmoid(X @ self.coef_ + self.intercept_)
   
    def _step(self, X, y):
        p = self._forward(X)
        error = y - p
        
        grad_w = X.T @ error / len(y)
        grad_b = np.mean(error)

        self.coef_ = self.coef_ + self.lr * grad_w
        self.intercept_ = self.intercept_ + self.lr * grad_b

    def fit(self, X, y, callback=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        self.classes_ = np.array([0, 1])
        self._init_params(X.shape[1])
        self.history_ = []
        self.n_iter_ = 0
        
        for epoch in range(1, self.n_epochs + 1):
            self._step(X, y)
            self.history_.append(binary_cross_entropy(y, self._forward(X)))
            if callback is not None:
                callback(epoch, self)
            improvement = self.history_[-2] - self.history_[-1] if epoch >= 2 else None
            if self.tol is not None and epoch >= 2 and self.tol > improvement:
                break
            
        self.n_iter_ = len(self.history_) 
        return self
    
    def predict_proba(self, X):
        p = self._forward(np.asarray(X, dtype=float))
        
        return np.column_stack([1 - p, p])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
