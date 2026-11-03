"""Classification metrics and cross-validation (later also learning curves)."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_score):
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def evaluate(model, X, y):
    """Metrics of a fitted model on (X, y). Works for any sklearn classifier."""
    return classification_metrics(y, model.predict(X), model.predict_proba(X)[:, 1])


class MetricsTracker:
    """Callback that records metrics on given data every `every` epochs."""

    def __init__(self, X, y, every=1):
        self.X, self.y, self.every = X, y, every
        self.epochs, self.history = [], []

    def __call__(self, epoch, model):
        if epoch % self.every == 0:
            self.epochs.append(epoch)
            self.history.append(evaluate(model, self.X, self.y))
