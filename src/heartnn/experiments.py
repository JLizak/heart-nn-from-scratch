"""Convergence studies from configs/experiments.yaml, one MLflow run per variant.

python -m heartnn.experiments            # all studies
python -m heartnn.experiments lr init    # selected ones
"""

import argparse

import matplotlib.pyplot as plt
import mlflow
import yaml
from sklearn.model_selection import ParameterGrid

from heartnn.plotting import plot_loss_comparison
from heartnn.settings import BASE_DIR, FIGURES_DIR
from heartnn.train import EXPERIMENT, TRACKING_URI, load_splits, run

DEFAULT_CONFIG = BASE_DIR / "configs" / "experiments.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def variants(config, study):
    """Full parameter dicts for one study: base, then study overrides, then the grid point."""
    spec = config["studies"][study]
    base = {**config["base"], **spec.get("params", {})}
    for point in ParameterGrid(spec["grid"]):
        yield point, {**base, **point}


def run_study(config, study, splits):
    curves = {}
    for point, params in variants(config, study):
        label = ", ".join(f"{k}={v}" for k, v in point.items())
        model = run(params, splits, run_name=f"{study}: {label}", tags={"study": study})
        curves[label] = model.history_

    fig = plot_loss_comparison(curves, title=config["studies"][study]["description"])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{study}.png", dpi=150)
    plt.close(fig)


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("studies", nargs="*", help=f"any of {list(config['studies'])}")
    studies = parser.parse_args().studies or list(config["studies"])
    unknown = set(studies) - set(config["studies"])
    if unknown:
        parser.error(f"unknown studies: {sorted(unknown)}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    splits = load_splits()
    for study in studies:
        run_study(config, study, splits)


if __name__ == "__main__":
    main()
