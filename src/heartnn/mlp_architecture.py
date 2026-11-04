"""List 3, part 1: hidden layer width, normalization and depth of the MLP on heart disease.

python -m heartnn.mlp_architecture

Full-batch gradient descent, lr 0.01, weights from N(0, 0.01^2), as required by the list.
Every configuration is trained with several seeds; curves show the mean and the min-max band.
"""
import matplotlib.pyplot as plt
import numpy as np

from heartnn.models.functional import binary_cross_entropy
from heartnn.models.mlp import MLP
from heartnn.preprocessing import load_column_splits
from heartnn.settings import FIGURES_DIR
from heartnn.tracking import MLP_EXPERIMENT, log_variant, setup_experiment

LR = 0.01
INIT_STD = 0.01
N_EPOCHS = 20000
EVAL_EVERY = 100
SEEDS = (0, 1, 2)

FIG_DIR = FIGURES_DIR
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def accuracy(y, p):
    return np.mean((p >= 0.5) == y)


def train(hidden_sizes, splits, seed):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    np.random.seed(seed)
    layer_sizes = (*hidden_sizes, 1)
    mlp = MLP(len(layer_sizes), X_train.shape[1], layer_sizes, init_std=INIT_STD)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, N_EPOCHS + 1):
        mlp.forward(X_train)
        mlp.backward(y_train)
        mlp.update(LR)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            p_train = mlp.forward(X_train)
            p_val = mlp.forward(X_val)
            history["epoch"].append(epoch)
            history["train_loss"].append(binary_cross_entropy(y_train, p_train))
            history["val_loss"].append(binary_cross_entropy(y_val, p_val))
            history["train_acc"].append(accuracy(y_train, p_train))
            history["val_acc"].append(accuracy(y_val, p_val))

    history = {k: np.array(v) for k, v in history.items()}
    history["test_acc"] = accuracy(y_test, mlp.forward(X_test))
    return history


def final_values(histories, key):
    return np.array([h[key][-1] if np.ndim(h[key]) else h[key] for h in histories])


def run_study(name, title, configs):
    """configs: list of (label, hidden_sizes, normalize). Returns rows for the results table."""
    print(f"== {name}")
    runs = {}
    split_cache = {}
    for label, hidden_sizes, normalize in configs:
        if normalize not in split_cache:
            split_cache[normalize] = load_column_splits(normalize)
        runs[label] = [train(hidden_sizes, split_cache[normalize], seed) for seed in SEEDS]
        print(f"   {label}: val acc {np.mean([h['val_acc'][-1] for h in runs[label]]):.3f}")

    plot_study(name, title, runs)
    log_study(name, configs, runs)

    rows = []
    for label, histories in runs.items():
        keys = ("train_loss", "val_loss", "train_acc", "val_acc", "test_acc")
        rows.append((name, label, *(final_values(histories, key) for key in keys)))
    return rows


def log_study(name, configs, runs):
    curve_names = ("train_loss", "val_loss", "train_acc", "val_acc")
    for label, hidden_sizes, normalize in configs:
        params = {
            "hidden_sizes": hidden_sizes, "normalize": normalize, "lr": LR, "init": "normal",
            "init_std": INIT_STD, "n_epochs": N_EPOCHS, "eval_every": EVAL_EVERY,
        }
        seed_runs = [
            {
                "seed": seed,
                "steps": history["epoch"],
                "curves": {key: history[key] for key in curve_names},
                "final": {"test_acc": history["test_acc"]},
            }
            for seed, history in zip(SEEDS, runs[label], strict=True)
        ]
        log_variant(name, label, params, seed_runs, figures=[FIG_DIR / f"{name}.png"])


def plot_study(name, title, runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    panels = [("train_loss", "Strata BCE — trening"), ("val_loss", "Strata BCE — walidacja"),
              ("val_acc", "Accuracy — walidacja")]

    for ax, (key, panel_title) in zip(axes, panels, strict=True):
        for color, (label, histories) in zip(SERIES_COLORS, runs.items(), strict=False):
            epochs = histories[0]["epoch"]
            values = np.stack([h[key] for h in histories])
            ax.fill_between(epochs, values.min(axis=0), values.max(axis=0),
                            color=color, alpha=0.15, lw=0)
            ax.plot(epochs, values.mean(axis=0), color=color, lw=2, label=label)

        ax.set_title(panel_title, loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("epoka", color=TEXT_MUTED)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    axes[0].legend(frameon=False, labelcolor=TEXT)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def write_table(rows):
    def fmt(v):
        return f"{v.mean():.3f} ± {v.std():.3f}"

    lines = [
        f"Mean ± std over seeds {SEEDS}, after {N_EPOCHS} epochs (lr {LR}, init std {INIT_STD}).",
        "",
        "| study | configuration | train loss | val loss | train acc | val acc | test acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, label, *metrics in rows:
        lines.append(f"| {name} | {label} | " + " | ".join(fmt(m) for m in metrics) + " |")
    (FIG_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_experiment(MLP_EXPERIMENT)
    n_features = load_column_splits()[0].shape[1]

    rows = []
    width_title = "Szerokość warstwy ukrytej (1 warstwa ukryta, dane znormalizowane)"
    rows += run_study("hidden_width", width_title, [
        ("3 neurony", (3,), True),
        (f"{n_features} neuronów (= liczba cech)", (n_features,), True),
        (f"{10 * n_features} neuronów", (10 * n_features,), True),
    ])
    norm_title = f"Normalizacja danych (1 warstwa ukryta, {n_features} neuronów)"
    rows += run_study("normalization", norm_title, [
        ("znormalizowane", (n_features,), True),
        ("nieznormalizowane", (n_features,), False),
    ])
    depth_title = f"Liczba warstw ukrytych (po {n_features} neuronów, dane znormalizowane)"
    rows += run_study("depth", depth_title, [
        (f"{k} ukryt{'a' if k == 1 else 'e'}", (n_features,) * k, True) for k in (1, 2, 3, 4)
    ])
    write_table(rows)


if __name__ == "__main__":
    main()
