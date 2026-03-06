"""
4-panel results visualization.

Panel 1: AUROC + AUPRC bar chart (all baselines)
Panel 2: Training curves (loss + cascade weight)
Panel 3: Contagion breakdown per test scenario
Panel 4: Cascade prediction scatter (true vs predicted)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_results(
    results: List[Dict],
    cascade_met: Dict,
    train_history: Dict,
    scenarios_test: List[Dict],
    cascade_preds: np.ndarray,
    save_path: str = "cascadenet_results.png",
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CascadeNet — Results Overview", fontsize=16, fontweight="bold")

    _plot_bar_chart(axes[0, 0], results)
    _plot_training_curves(axes[0, 1], train_history)
    _plot_contagion(axes[1, 0], scenarios_test)
    _plot_cascade_scatter(axes[1, 1], scenarios_test, cascade_preds, cascade_met)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [Viz] Saved to {save_path}")
    plt.close()


def _plot_bar_chart(ax, results):
    names = [r["name"].replace(": ", "\n") for r in results]
    aurocs = [r["auroc"] for r in results]
    auprcs = [r.get("auprc", 0) for r in results]

    x = np.arange(len(names))
    w = 0.35

    bars1 = ax.bar(x - w/2, aurocs, w, label="AUROC", color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x + w/2, auprcs, w, label="AUPRC", color="#7c3aed", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, ha="center")
    ax.set_ylabel("Score")
    ax.set_title("Head A: Default Prediction")
    ax.legend(fontsize=8)

    lo = min(min(aurocs), min(auprcs)) - 0.05
    ax.set_ylim(max(0, lo), 1.02)

    for bar, val in zip(bars1, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)


def _plot_training_curves(ax, history):
    if not history["train_loss"]:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", alpha=0.7, label="Train")
    ax.plot(epochs, history["val_loss"], "r-", alpha=0.7, label="Val")

    ax2 = ax.twinx()
    ax2.plot(epochs, history["cascade_weight"], "g--", alpha=0.5, label="CW")
    ax2.set_ylabel("Cascade Weight", color="green", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="green")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)


def _plot_contagion(ax, scenarios):
    n_sc = len(scenarios)
    idx = range(n_sc)
    init = [s["n_initial_defaults"] for s in scenarios]
    cont = [s["n_contagion_defaults"] for s in scenarios]

    ax.bar(idx, init, color="#94a3b8", label="Fundamental")
    ax.bar(idx, cont, bottom=init, color="#ef4444", label="Contagion")

    ax.set_xlabel("Test Scenario")
    ax.set_ylabel("# Defaults")
    ax.set_title("Contagion Breakdown")
    ax.legend(fontsize=9)


def _plot_cascade_scatter(ax, scenarios, preds, metrics):
    all_true = np.concatenate([s["cascade_sizes"] for s in scenarios])

    # Plot nodes with nonzero cascade
    nonzero = all_true > 0
    n_pts = min(3000, nonzero.sum())
    if n_pts > 0:
        idx = np.random.choice(np.where(nonzero)[0], n_pts, replace=False)
        ax.scatter(
            np.log1p(all_true[idx]),
            np.log1p(preds[idx]),
            alpha=0.2, s=10, color="#2563eb",
        )
        lim = max(np.log1p(all_true[idx]).max(), np.log1p(preds[idx]).max())
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="Perfect")

    rho = metrics["spearman_rho"]
    ax.set_xlabel("True cascade (log1p)")
    ax.set_ylabel("Predicted cascade (log1p)")
    ax.set_title(f"Head B: Cascade (ρ={rho:.3f})")
    ax.legend(fontsize=9)