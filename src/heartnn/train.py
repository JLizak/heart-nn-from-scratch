"""Train LogisticReg on a 60/20/20 split and log the run to MLflow."""

import mlflow

from heartnn.evaluation import evaluate
from heartnn.models.logistic import LogisticReg
from heartnn.preprocessing import load_splits
from heartnn.tracking import TRACKING_URI, MlflowTracker

EXPERIMENT = "logreg-from-scratch"


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
