"""Train LogisticReg on a 60/20/20 split and log the run to MLflow."""

import mlflow
from sklearn.model_selection import train_test_split

from heartnn.data import load_clean_data
from heartnn.evaluation import evaluate
from heartnn.models.logistic import LogisticReg
from heartnn.preprocessing import build_preprocessor
from heartnn.tracking import MlflowTracker

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "logreg-from-scratch"


def load_splits(random_state=42):
    X, y = load_clean_data()

    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=random_state
    )

    pre = build_preprocessor()
    X_train = pre.fit_transform(X_train)
    X_val = pre.transform(X_val)
    X_test = pre.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def run(params, splits, run_name=None, tags=None, every=10):
    """One MLflow run, logged on val; test is evaluated once, after training."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(params)

        tracker = MlflowTracker(X_val, y_val, every=every)
        model = LogisticReg(**params).fit(X_train, y_train, callback=tracker)
        tracker.log(model)
        mlflow.log_metric("n_iter", model.n_iter_)

        mlflow.log_metrics({f"test_{k}": v for k, v in evaluate(model, X_test, y_test).items()})

    return model


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    params = {"lr": 0.1, "n_epochs": 1000, "init": "normal", "init_scale": 0.01, "random_state": 0}
    run(params, load_splits(), run_name="default")


if __name__ == "__main__":
    main()
