"""Create paper-ready figures from a completed stress-test run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NORMAL_AFFECTED_IDENTITY_MSE = 0.062555
NORMAL_AFFECTED_MODEL_MSE = 0.004428


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from stress-test CSV outputs."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Completed run directory, for example outputs/runs/stress_test_YYYYMMDD_HHMMSS.",
    )
    return parser.parse_args()


def unique_path(path: Path) -> Path:
    """Return a non-overwriting path by adding a numeric suffix if needed."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 1000):
        candidate = path.with_name(f"{stem}_{idx:02d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused filename for {path}.")


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = unique_path(output_dir / filename)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def grouped_results_path(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "results" / "stress_test_grouped_results.csv",
        run_dir / "results" / "stress_test_measured_severity_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ordered_methods(df: pd.DataFrame) -> pd.DataFrame:
    order = ["identity", "threshold", "model"]
    present = [method for method in order if method in set(df["method"])]
    return df.set_index("method").loc[present].reset_index()


def metric_column(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Missing metric column: {preferred} or {fallback}")


def bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
) -> Path:
    frame = ordered_methods(df)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    colors = ["#8a8f98", "#b36b5e", "#2f6f8f"][: len(frame)]
    ax.bar(frame["method"], frame[metric], color=colors, width=0.62)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, filename)


def improvement_chart(aggregate: pd.DataFrame, output_dir: Path) -> Path:
    frame = aggregate.set_index("method")
    stress_identity = float(frame.loc["identity", "affected_mse"])
    stress_model = float(frame.loc["model", "affected_mse"])
    stress_ratio = stress_identity / stress_model
    normal_ratio = NORMAL_AFFECTED_IDENTITY_MSE / NORMAL_AFFECTED_MODEL_MSE

    labels = ["normal held-out", "hard stress"]
    values = [normal_ratio, stress_ratio]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(labels, values, color=["#587a4d", "#2f6f8f"], width=0.58)
    ax.set_title("Affected-Region MSE Improvement")
    ax.set_ylabel("Identity MSE / Model MSE")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=9)
    return save_figure(fig, output_dir, "normal_vs_stress_affected_improvement.png")


def grouped_bar(
    grouped: pd.DataFrame,
    grouping: str,
    order: Iterable[str],
    title: str,
    output_dir: Path,
    filename: str,
) -> Path | None:
    frame = grouped[grouped["grouping"] == grouping].copy()
    if frame.empty:
        return None
    order = [item for item in order if item in set(frame["group"])]
    if order:
        frame = frame.set_index("group").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(frame["group"], frame["model_affected_mse"], color="#2f6f8f", width=0.62)
    ax.set_title(title)
    ax.set_ylabel("Model affected-region MSE")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, filename)


def scatter_plot(
    per_sample: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
    log_y: bool = False,
) -> Path:
    frame = per_sample[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(frame[x_col], frame[y_col], s=14, alpha=0.38, color="#2f6f8f", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, filename)


def ratio_histogram(per_sample: pd.DataFrame, output_dir: Path) -> Path:
    values = (
        per_sample["model_improvement_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    upper = float(np.nanpercentile(values, 99))
    clipped = values.clip(upper=upper)
    ax.hist(clipped, bins=35, color="#2f6f8f", alpha=0.86)
    ax.axvline(1.0, color="#b36b5e", linewidth=1.6, label="identity parity")
    ax.set_title("Model Improvement Ratio Distribution")
    ax.set_xlabel("Identity affected MSE / model affected MSE")
    ax.set_ylabel("Samples")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, "hist_model_improvement_ratio.png")


def failure_gallery(run_dir: Path, output_dir: Path) -> Path | None:
    figure_dir = run_dir / "figures"
    largest_failure = figure_dir / "stress_test_largest_model_failure_example.png"
    if not largest_failure.exists():
        largest_failure = figure_dir / "stress_test_hard_failure_example.png"
    files = [
        ("Success", figure_dir / "stress_test_success_example.png"),
        ("Partial failure", figure_dir / "stress_test_partial_failure_example.png"),
        ("Largest model failure", largest_failure),
    ]
    existing = [(label, path) for label, path in files if path.exists()]
    if not existing:
        return None

    fig, axes = plt.subplots(len(existing), 1, figsize=(10, 3.2 * len(existing)))
    if len(existing) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, existing):
        ax.imshow(mpimg.imread(path))
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    return save_figure(fig, output_dir, "failure_gallery.png")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_dir = run_dir / "results"
    output_dir = run_dir / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = read_required_csv(results_dir / "stress_test_results.csv")
    per_sample = read_required_csv(results_dir / "stress_test_per_sample_results.csv")
    grouped_path = grouped_results_path(run_dir)
    grouped = pd.read_csv(grouped_path) if grouped_path is not None else pd.DataFrame()

    affected_col = metric_column(aggregate, "affected_mse", "masked_mse")
    whole_col = metric_column(aggregate, "mse", "mse")
    aggregate = aggregate.copy()
    aggregate["affected_mse"] = aggregate[affected_col]

    written: list[Path] = []
    written.append(
        bar_chart(
            aggregate,
            "affected_mse",
            "Stress-Test Affected-Region MSE",
            "Affected-region MSE",
            output_dir,
            "affected_region_mse_bar.png",
        )
    )
    written.append(
        bar_chart(
            aggregate,
            whole_col,
            "Stress-Test Whole-Image MSE",
            "Whole-image MSE",
            output_dir,
            "whole_image_mse_bar.png",
        )
    )
    written.append(improvement_chart(aggregate, output_dir))

    if not grouped.empty:
        if "model_affected_mse" not in grouped.columns and "model_masked_mse" in grouped.columns:
            grouped["model_affected_mse"] = grouped["model_masked_mse"]
        maybe_path = grouped_bar(
            grouped,
            "blend_severity_bin",
            ["easy", "medium", "hard"],
            "Model Error by Blend Severity",
            output_dir,
            "model_affected_mse_by_blend_severity_bin.png",
        )
        if maybe_path is not None:
            written.append(maybe_path)
        maybe_path = grouped_bar(
            grouped,
            "core_overlap_bin",
            ["low", "medium", "high"],
            "Model Error by Core Obstruction",
            output_dir,
            "model_affected_mse_by_core_overlap_bin.png",
        )
        if maybe_path is not None:
            written.append(maybe_path)

    written.append(
        scatter_plot(
            per_sample,
            "blend_severity_score",
            "model_affected_mse",
            "Blend Severity vs Model Error",
            "Blend severity score",
            "Model affected-region MSE",
            output_dir,
            "scatter_blend_severity_vs_model_affected_mse.png",
            log_y=True,
        )
    )
    written.append(
        scatter_plot(
            per_sample,
            "core_obstruction_fraction",
            "model_improvement_ratio",
            "Core Obstruction vs Model Improvement",
            "Core obstruction fraction",
            "Identity affected MSE / model affected MSE",
            output_dir,
            "scatter_core_obstruction_vs_model_improvement_ratio.png",
        )
    )
    written.append(ratio_histogram(per_sample, output_dir))
    maybe_gallery = failure_gallery(run_dir, output_dir)
    if maybe_gallery is not None:
        written.append(maybe_gallery)

    for path in written:
        print(path.relative_to(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
