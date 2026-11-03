"""MLflow logging. The only module besides train.py that imports mlflow."""

import time

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from heartnn.evaluation import MetricsTracker

BATCH_SIZE = 1000  # MLflow's limit of metrics per log_batch call


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

        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        for i in range(0, len(metrics), BATCH_SIZE):
            client.log_batch(run_id, metrics=metrics[i : i + BATCH_SIZE])
