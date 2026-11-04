import math

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from heartnn.tracking import log_variant


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield mlflow.set_experiment("test")
    mlflow.set_tracking_uri(previous)


def seed_run(seed, last_loss, test_acc=0.5):
    return {
        "seed": seed,
        "steps": np.array([1, 2]),
        "curves": {"train_loss": np.array([1.0, last_loss])},
        "final": {"test_acc": test_acc, "tol_stop_epoch": None},
    }


def find_runs(experiment):
    runs = MlflowClient().search_runs([experiment.experiment_id])
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    return parent, [r for r in runs if r is not parent]


def test_parent_summarizes_nested_seed_runs(experiment, tmp_path):
    figure = tmp_path / "study.png"
    figure.write_bytes(b"png")
    log_variant("depth", "2 hidden", {"lr": 0.01}, [seed_run(0, 0.2), seed_run(1, 0.4)], [figure])

    parent, children = find_runs(experiment)
    assert parent.data.tags["study"] == "depth"
    assert parent.data.metrics["train_loss_mean"] == pytest.approx(0.3)
    assert parent.data.metrics["train_loss_std"] == pytest.approx(0.1)
    assert [a.path for a in MlflowClient().list_artifacts(parent.info.run_id)] == ["study.png"]

    assert len(children) == 2
    assert {c.data.params["seed"] for c in children} == {"0", "1"}
    assert all(c.data.tags["mlflow.parentRunId"] == parent.info.run_id for c in children)
    assert all(c.data.tags["study"] == "depth" for c in children)
    assert all("tol_stop_epoch" not in c.data.metrics for c in children)


def test_full_curve_is_logged_with_steps(experiment):
    log_variant("depth", "2 hidden", {}, [seed_run(0, 0.2)])

    _, (child,) = find_runs(experiment)
    history = MlflowClient().get_metric_history(child.info.run_id, "train_loss")
    assert [(m.step, m.value) for m in history] == [(1, 1.0), (2, 0.2)]


def test_diverged_seeds_are_left_out_of_the_summary(experiment):
    log_variant("init", "broken", {}, [seed_run(0, math.nan), seed_run(1, 0.4)])

    parent, _ = find_runs(experiment)
    assert parent.data.metrics["train_loss_mean"] == pytest.approx(0.4)
