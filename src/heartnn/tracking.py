"""MLflow logging. The only module besides train.py that imports mlflow."""

import time

import mlflow
import numpy as np
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from heartnn.evaluation import MetricsTracker

TRACKING_URI = "sqlite:///mlflow.db"
MLP_EXPERIMENT = "mlp-from-scratch"
BATCH_SIZE = 1000  # MLflow's limit of metrics per log_batch call


def setup_experiment(name):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(name)


def _send_metrics(metrics):
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    for i in range(0, len(metrics), BATCH_SIZE):
        client.log_batch(run_id, metrics=metrics[i : i + BATCH_SIZE])


class MlflowTracker(MetricsTracker):
    """Collects val metrics during fit; log() sends them with the train loss after fit.

    One log_metric call per epoch takes a few ms on SQLite, i.e. minutes per study,
    so everything goes in batches at the end.
    """

    def log(self, model):
        ts = int(time.time() * 1000)
        metrics = [
            Metric("train_loss", float(loss), ts, epoch)
            for epoch, loss in enumerate(model.history_, start=1)
        ]
        metrics += [
            Metric(f"val_{name}", float(value), ts, epoch)
            for epoch, values in zip(self.epochs, self.history, strict=True)
            for name, value in values.items()
        ]
        _send_metrics(metrics)


def _mean_and_std(seed_runs):
    """Mean and std over seeds of the last value of every curve and of the final metrics."""
    values = {}
    for seed_run in seed_runs:
        last = {name: curve[-1] for name, curve in seed_run["curves"].items()}
        for name, value in {**last, **seed_run["final"]}.items():
            values.setdefault(name, []).append(value)

    summary = {}
    for name, seed_values in values.items():
        finite = np.array([v for v in seed_values if v is not None and np.isfinite(v)])
        if finite.size:
            summary[f"{name}_mean"] = float(finite.mean())
            summary[f"{name}_std"] = float(finite.std())
    return summary


def log_variant(study, label, params, seed_runs, figures=()):
    """One parent run per configuration, with a nested run per seed.

    seed_runs: list of {"seed", "steps", "curves": {name: values}, "final": {name: value}}.
    Everything is logged after training, so tracking cannot affect the results.
    """
    tags = {"study": study}
    with mlflow.start_run(run_name=f"{study}: {label}", tags=tags):
        mlflow.log_params(params)
        mlflow.log_metrics(_mean_and_std(seed_runs))
        for figure in figures:
            mlflow.log_artifact(str(figure))

        for seed_run in seed_runs:
            run_name = f"seed={seed_run['seed']}"
            with mlflow.start_run(run_name=run_name, nested=True, tags=tags):
                mlflow.log_params({**params, "seed": seed_run["seed"]})
                ts = int(time.time() * 1000)
                _send_metrics([
                    Metric(name, float(value), ts, int(step))
                    for name, curve in seed_run["curves"].items()
                    for step, value in zip(seed_run["steps"], curve, strict=True)
                ])
                final = {k: float(v) for k, v in seed_run["final"].items() if v is not None}
                mlflow.log_metrics(final)
