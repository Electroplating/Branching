"""绘制训练曲线与对比柱状图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def plot_training_curves():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for exp_file in sorted(RESULTS_DIR.glob("*.json")):
        if exp_file.name in ("summary.json", "sim_compare.json"):
            continue
        with open(exp_file) as f:
            data = json.load(f)
        history = data.get("history", [])
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        loss = [h["train_loss"] for h in history]
        top1 = [h["val_top1"] for h in history]
        axes[0].plot(epochs, loss, label=data["name"])
        axes[1].plot(epochs, top1, label=data["name"])

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Top-1 Accuracy")
    axes[1].set_title("Validation Top-1 Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_curves.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'training_curves.png'}")


def plot_final_comparison():
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        return
    with open(summary_path) as f:
        summary = json.load(f)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    names = list(summary.keys())
    metrics = ["gnn_top1", "vsids_top1", "oracle_top1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    x = range(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [summary[n][metric] for n in names]
        ax.bar([xi + i * width for xi in x], vals, width, label=metric, color=color)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Top-1 Accuracy Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_comparison.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'final_comparison.png'}")


def main():
    plot_training_curves()
    plot_final_comparison()


if __name__ == "__main__":
    main()
