"""List 3, part 2: batch modes, weight initialization and plain gradient vs Adam.

python -m heartnn.mlp_optimization

Every configuration is trained with several seeds; curves show the mean and the min-max band.
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from heartnn.mlp_architecture import FIG_DIR, GRID, SERIES_COLORS, TEXT, TEXT_MUTED
from heartnn.models.functional import binary_cross_entropy
from heartnn.models.mlp import MLP
from heartnn.models.optim import SGD, Adam
from heartnn.preprocessing import load_column_splits
from heartnn.tracking import MLP_EXPERIMENT, log_variant, setup_experiment

SEEDS = (0, 1, 2)
TOL = 1e-6


def train(hidden_sizes, init, init_std, optimizer, batch_size, n_epochs, splits, seed):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    np.random.seed(seed)
    layer_sizes = (*hidden_sizes, 1)
    mlp = MLP(len(layer_sizes), X_train.shape[1], layer_sizes, init_std=init_std, init=init)

    history = {"epoch": [], "updates": [], "time": [], "train_loss": [], "val_loss": [],
               "val_acc": [], "grad_norm": []}
    history["init_grad_norms"] = layer_grad_norms(mlp, X_train, y_train)

    n_updates = 0
    train_time = 0.0
    for epoch in range(1, n_epochs + 1):
        start = time.perf_counter()
        order = np.random.permutation(len(X_train))
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            mlp.forward(X_train[batch])
            mlp.backward(y_train[batch])
            optimizer.step(mlp.layers)
            n_updates += 1
        train_time += time.perf_counter() - start

        # Metrics on whole splits; the gradient norm of the first layer shows plateaus
        p_val = mlp.forward(X_val)
        history["val_loss"].append(binary_cross_entropy(y_val, p_val))
        history["val_acc"].append(np.mean((p_val >= 0.5) == y_val))
        history["train_loss"].append(binary_cross_entropy(y_train, mlp.forward(X_train)))
        mlp.backward(y_train)
        history["grad_norm"].append(np.linalg.norm(mlp.layers[0].dW))
        history["epoch"].append(epoch)
        history["updates"].append(n_updates)
        history["time"].append(train_time)

    history = {k: np.array(v) for k, v in history.items()}
    history["test"] = test_metrics(y_test, mlp.forward(X_test))
    history["tol_stop"] = tol_stop_epoch(history["train_loss"])
    return history


def layer_grad_norms(mlp, X, y):
    """Norm of dL/dW in every layer before training: vanishing or exploding gradients."""
    mlp.forward(X)
    mlp.backward(y)
    return [np.linalg.norm(layer.dW) for layer in mlp.layers]


def test_metrics(y, p):
    if not np.all(np.isfinite(p)):
        return {k: np.nan for k in ("accuracy", "f1", "precision", "recall", "roc_auc")}
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
    }


def tol_stop_epoch(losses):
    """Epoch at which a stop rule 'loss decreased by less than TOL' would end training."""
    for epoch in range(1, len(losses)):
        if losses[epoch - 1] - losses[epoch] < TOL:
            return epoch + 1
    return None


def run_study(name, title, configs, panels, splits, extra_plot=None):
    """configs: list of (label, kwargs for train). Returns {label: [history per seed]}."""
    print(f"== {name}")
    runs = {}
    for label, params in configs:
        runs[label] = [train(splits=splits, seed=seed, **make_params(params)) for seed in SEEDS]
        train_loss = np.nanmean([h["train_loss"][-1] for h in runs[label]])
        test_acc = np.nanmean([h["test"]["accuracy"] for h in runs[label]])
        print(f"   {label}: train loss {train_loss:.3f}, test acc {test_acc:.3f}")
    plot_study(name, title, runs, panels)
    figures = [FIG_DIR / f"{name}.png"]
    if extra_plot is not None:
        figures.append(extra_plot(runs))
    log_study(name, configs, runs, figures)
    return runs


def log_study(name, configs, runs, figures):
    curve_names = ("train_loss", "val_loss", "val_acc", "grad_norm", "updates", "time")
    for label, params in configs:
        optimizer = params["optimizer"]()
        logged = {k: v for k, v in params.items() if k != "optimizer"}
        logged.update(optimizer=type(optimizer).__name__, lr=optimizer.lr)
        seed_runs = [
            {
                "seed": seed,
                "steps": history["epoch"],
                "curves": {key: history[key] for key in curve_names},
                "final": {
                    **{f"test_{k}": v for k, v in history["test"].items()},
                    **{f"init_grad_norm_layer_{i}": norm
                       for i, norm in enumerate(history["init_grad_norms"], start=1)},
                    "tol_stop_epoch": history["tol_stop"],
                },
            }
            for seed, history in zip(SEEDS, runs[label], strict=True)
        ]
        log_variant(name, label, logged, seed_runs, figures=figures)


def make_params(params):
    # A fresh optimizer for every run, Adam keeps state between steps
    params = dict(params)
    params["optimizer"] = params["optimizer"]()
    return params


def plot_study(name, title, runs, panels):
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.4), constrained_layout=True)
    for ax, (key, x_key, panel_title, log_x, log_y) in zip(axes, panels, strict=True):
        for color, (label, histories) in zip(SERIES_COLORS, runs.items(), strict=False):
            values = np.stack([h[key] for h in histories])
            x = np.mean([h[x_key] for h in histories], axis=0)
            ax.fill_between(x, np.nanmin(values, axis=0), np.nanmax(values, axis=0),
                            color=color, alpha=0.15, lw=0)
            ax.plot(x, np.nanmean(values, axis=0), color=color, lw=2, label=label)
        style_axis(ax, panel_title, {"epoch": "epoka", "updates": "liczba aktualizacji wag",
                                     "time": "czas uczenia [s]"}[x_key])
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

    axes[0].legend(frameon=False, labelcolor=TEXT)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_init_grad_norms(runs):
    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    n_configs = len(runs)
    width = 0.8 / n_configs
    for i, (color, (label, histories)) in enumerate(zip(SERIES_COLORS, runs.items(), strict=False)):
        norms = np.mean([h["init_grad_norms"] for h in histories], axis=0)
        x = np.arange(len(norms)) + (i - (n_configs - 1) / 2) * width
        ax.bar(x, norms, width=width * 0.9, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(norms)), [f"warstwa {k + 1}" for k in range(len(norms))])
    style_axis(ax, "Norma ‖dL/dW‖ w każdej warstwie przed uczeniem", None)
    ax.legend(frameon=False, labelcolor=TEXT)
    path = FIG_DIR / "p2_init_grad_norms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def style_axis(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def summarize(runs):
    """Mean and std over seeds of the numbers used in the report."""
    summary = {}
    for label, histories in runs.items():
        row = {
            "train_loss": [h["train_loss"][-1] for h in histories],
            "val_loss": [h["val_loss"][-1] for h in histories],
            "best_val_loss": [
                np.nanmin(h["val_loss"]) if np.any(np.isfinite(h["val_loss"])) else np.nan
                for h in histories
            ],
            "updates": [h["updates"][-1] for h in histories],
            "time": [h["time"][-1] for h in histories],
            "tol_stop": [h["tol_stop"] if h["tol_stop"] is not None else np.nan for h in histories],
            "init_grad_norm_first": [h["init_grad_norms"][0] for h in histories],
            "init_grad_norm_last": [h["init_grad_norms"][-1] for h in histories],
        }
        for metric in histories[0]["test"]:
            row["test_" + metric] = [h["test"][metric] for h in histories]
        summary[label] = {k: {"mean": float(np.nanmean(v)) if np.any(np.isfinite(v)) else None,
                              "std": float(np.nanstd(v)) if np.any(np.isfinite(v)) else None}
                          for k, v in row.items()}
    return summary


def main():
    np.seterr(all="ignore")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_experiment(MLP_EXPERIMENT)
    splits = load_column_splits()
    d = splits[0].shape[1]

    loss_panels = [
        ("train_loss", "epoch", "Strata BCE — trening", False, False),
        ("val_loss", "epoch", "Strata BCE — walidacja", False, False),
        ("val_acc", "epoch", "Accuracy — walidacja", False, False),
        ("grad_norm", "epoch", "‖dL/dW‖ pierwszej warstwy", False, True),
    ]
    results = {}

    # 1. Batch modes: 2 hidden layers, init std 0.01, plain gradient with lr 0.01 (as in part 1)
    batch_configs = [
        ("pełny zbiór (550)", dict(batch_size=550)),
        ("paczki (32)", dict(batch_size=32)),
        ("pojedyncze przykłady (1)", dict(batch_size=1)),
    ]
    runs = run_study(
        "p2_batch",
        "Tryb uczenia (2 warstwy ukryte, init N(0, 0.01²), spadek gradientu, lr 0.01)",
        [(label, dict(hidden_sizes=(d, d), init="normal", init_std=0.01,
                      optimizer=lambda: SGD(0.01), n_epochs=1000, **p))
         for label, p in batch_configs],
        [("train_loss", "epoch", "Strata treningowa od epoki", False, False),
         ("train_loss", "updates", "Strata treningowa od liczby aktualizacji", True, False),
         ("train_loss", "time", "Strata treningowa od czasu", True, False),
         ("val_loss", "epoch", "Strata walidacyjna od epoki", False, False)],
        splits)
    results["batch"] = summarize(runs)

    # 2. Initialization: 4 hidden layers, mini-batch 32, plain gradient with lr 0.01
    init_configs = [
        ("N(0, 0.01²)", dict(init="normal", init_std=0.01)),
        ("N(0, 1)", dict(init="normal", init_std=1.0)),
        ("He, N(0, 2/d_in)", dict(init="he", init_std=None)),
        ("zepsuta, N(0, 10²)", dict(init="normal", init_std=10.0)),
    ]
    runs = run_study(
        "p2_init", "Inicjalizacja wag (4 warstwy ukryte, paczki 32, spadek gradientu, lr 0.01)",
        [(label, dict(hidden_sizes=(d,) * 4, optimizer=lambda: SGD(0.01), batch_size=32,
                      n_epochs=200, **p)) for label, p in init_configs],
        [("train_loss", "epoch", "Strata BCE — trening", False, True)] + loss_panels[1:],
        splits, extra_plot=plot_init_grad_norms)
    results["init"] = summarize(runs)

    # 3. Plain gradient vs Adam: 6 hidden layers, mini-batch 32, small and He initialization
    optim_configs = [
        ("gradient, N(0, 0.01²)", dict(init="normal", init_std=0.01, optimizer=lambda: SGD(0.01))),
        ("Adam, N(0, 0.01²)", dict(init="normal", init_std=0.01, optimizer=lambda: Adam(0.001))),
        ("gradient, He", dict(init="he", init_std=None, optimizer=lambda: SGD(0.01))),
        ("Adam, He", dict(init="he", init_std=None, optimizer=lambda: Adam(0.001))),
    ]
    runs = run_study(
        "p2_adam", "Spadek gradientu (lr 0.01) vs Adam (lr 0.001), 6 warstw ukrytych, paczki 32",
        [(label, dict(hidden_sizes=(d,) * 6, batch_size=32, n_epochs=200, **p))
         for label, p in optim_configs],
        loss_panels, splits)
    results["adam"] = summarize(runs)

    (FIG_DIR / "results2.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
