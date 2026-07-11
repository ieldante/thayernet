#!/usr/bin/env python3
"""Plot grouped v0.2 training and same-manifest development comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run-dir", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    return parser.parse_args()


def save_new(fig: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {path}")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    audit_run = args.audit_run_dir.resolve()
    training_run = args.training_run_dir.resolve()
    output_dir = audit_run / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(
        training_run / "tables" / "training_history.csv",
        float_precision="round_trip",
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(history["epoch"], history["train_loss"], marker="o", ms=3, label="train weighted loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", ms=3, label="validation weighted loss")
    best = history.loc[history["val_loss"].idxmin()]
    ax.scatter([best["epoch"]], [best["val_loss"]], marker="*", s=150, color="#b2182b", zorder=5,
               label=f"best epoch {int(best['epoch'])}")
    ax.set(xlabel="Epoch", ylabel="Weighted residual loss", title="Grouped v0.2 Moderate retrain")
    ax.set_xticks(np.arange(1, int(history["epoch"].max()) + 1, 2))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_new(fig, output_dir / "grouped_retrain_training_history.png")

    comparison = pd.read_csv(
        audit_run / "tables" / "grouped_retrain_comparison_summary.csv",
        float_precision="round_trip",
    )
    comparison = comparison[
        comparison["method"].isin(
            ["v02_moderate_old_split", "v02_moderate_grouped_retrain"]
        )
    ].copy()
    suite_order = ["normal", "hard_stress", "compact_bright", "high_core_obstruction"]
    method_order = ["v02_moderate_old_split", "v02_moderate_grouped_retrain"]
    labels = ["Normal", "Hard stress", "Compact bright", "High core"]
    colors = ["#5b8db8", "#d87c4a"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    x = np.arange(len(suite_order))
    width = 0.36
    for offset, method, color, label in zip(
        (-width / 2, width / 2),
        method_order,
        colors,
        ("Historical checkpoint (diagnostic)", "Grouped retrain"),
    ):
        rows = comparison[comparison["method"] == method].set_index("suite")
        mse = [rows.loc[s, "affected_mse_macro"] for s in suite_order]
        ratio = [rows.loc[s, "improvement_ratio_vs_identity"] for s in suite_order]
        axes[0].bar(x + offset, mse, width, color=color, label=label)
        axes[1].bar(x + offset, ratio, width, color=color, label=label)

    axes[0].set_title("Affected-region MSE")
    axes[0].set_ylabel("Macro MSE (lower is better)")
    axes[1].set_title("Identity MSE / model MSE")
    axes[1].set_ylabel("Affected-MSE reduction factor (higher is better)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Same grouped development manifests; neither result is final-paper performance", fontsize=11)
    fig.tight_layout()
    save_new(fig, output_dir / "grouped_existing_vs_retrain.png")


if __name__ == "__main__":
    main()
