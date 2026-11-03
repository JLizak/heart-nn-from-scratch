"""Learning-curve figures. Functions return the figure; the caller shows or saves it."""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(loss, metric_epochs=None, metric_history=None, title="Learning curve"):
    """One run: loss per epoch, plus metrics per epoch if given."""
    n_axes = 2 if metric_history else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 4), squeeze=False)

    ax = axes[0, 0]
    ax.plot(np.arange(1, len(loss) + 1), loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)

    if metric_history:
        ax = axes[0, 1]
        for name in metric_history[0]:
            ax.plot(metric_epochs, [m[name] for m in metric_history], label=name)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_loss_comparison(curves, title, log_x=True):
    """Several runs on one plot, e.g. {"lr=0.01": history, "lr=0.1": history}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, loss in curves.items():
        ax.plot(np.arange(1, len(loss) + 1), loss, label=label)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig
