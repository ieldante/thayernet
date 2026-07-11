"""Run a hard-case stress evaluation for the direct U-Net checkpoint.

The script is intentionally output-safe: each invocation writes to a new
timestamped run directory under ``outputs/runs/`` and reads the trained
checkpoint without modifying it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import baselines
from src import blend as gd_blend
from src import data as gd_data
from src import models
from src import train as gd_train
from src import utils as gd_utils


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "unet_direct_5000train_800val_800test_20ep.pth"
)
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "docs" / "experiment_log.md"

NORMAL_SECTION = """## Direct U-Net Normal Evaluation

- Dataset split: 5,000 training blends / 800 validation blends / 800 held-out test blends.
- Training: compact direct U-Net, 20 epochs, controlled synthetic deblending.
- Whole-image MSE: identity baseline 0.005224 -> direct U-Net 0.000566.
- Affected-region MSE: identity baseline 0.062555 -> direct U-Net 0.004428.
- Affected-region MSE improvement: about 14.1x lower than identity.
- Interpretation: the direct model clearly improves over identity on the controlled synthetic held-out set, especially in pixels changed by blending. Whole-image metrics remain less diagnostic because most pixels are unchanged.
- Terminology: generation difficulty comes from sampled parameters; blend severity comes from affected-region contaminant damage; core obstruction measures affected target-core overlap; model failure comes from model affected error and improvement versus identity.
"""

RESIDUAL_PLAN_SECTION = """## Planned Experiment 2: Residual Prediction

The current direct model predicts the clean target image directly: blended -> target.

The residual model should instead predict the contaminant/residual layer: blended -> residual. The reconstructed target would then be computed as blended - predicted residual.

Expected benefit: residual prediction may preserve target structure better and reduce over-smoothing in stress/core-obstruction cases, because unchanged target light can pass through the subtraction path instead of being regenerated from scratch.

Evaluation should compare the direct U-Net and residual U-Net on the same normal held-out test set and the same hard-case stress-test set. Residual training should not begin until the direct-model stress test shows whether stress/core-obstruction failures justify it.
"""

STRESS_DEFAULTS: dict[str, Any] = {
    "n_stress_blends": 1000,
    "stress_source_subset": 800,
    "seed": 20260708,
    "max_shift": 18,
    "brightness_range": [0.8, 1.4],
    "blur_range": [0.0, 0.15],
    "noise_range": [0.0, 0.006],
    "rotation_range": [0.0, 0.0],
    "min_size_ratio": 0.75,
    "min_mask_fraction": 0.01,
    "affected_region_threshold": 0.02,
    "max_attempt_multiplier": 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained direct U-Net on hard synthetic blends."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the project YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained direct U-Net checkpoint.",
    )
    parser.add_argument(
        "--n-stress-blends",
        type=int,
        default=STRESS_DEFAULTS["n_stress_blends"],
        help="Number of accepted hard-case blends to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Use 'auto' to prefer CUDA/MPS when available.",
    )
    parser.add_argument(
        "--skip-log-update",
        action="store_true",
        help="Do not update docs/experiment_log.md.",
    )
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def make_run_dir(output_root: Path) -> Path:
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = runs_root / f"stress_test_{stamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = runs_root / f"stress_test_{stamp}_{suffix:02d}"
    for child in ("results", "figures", "diagnostics", "logs"):
        (candidate / child).mkdir(parents=True, exist_ok=False)
    return candidate


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    return config


def save_run_config(run_dir: Path, config: dict[str, Any], stress: dict[str, Any]) -> None:
    serialisable = {
        "project_root": ".",
        "config": config,
        "stress_settings": stress,
    }
    with (run_dir / "logs" / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serialisable, handle, sort_keys=False)


def replace_or_append_section(text: str, heading: str, section: str) -> str:
    lines = text.rstrip().splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx
            break

    section_lines = section.rstrip().splitlines()
    if start is None:
        return text.rstrip() + "\n\n" + section.rstrip() + "\n"

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break

    new_lines = lines[:start] + section_lines + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def update_experiment_log(stress_section: str) -> None:
    if EXPERIMENT_LOG_PATH.exists():
        text = EXPERIMENT_LOG_PATH.read_text(encoding="utf-8")
    else:
        text = "# Experiment Log\n"

    text = replace_or_append_section(
        text,
        "## Direct U-Net Normal Evaluation",
        NORMAL_SECTION,
    )
    text = replace_or_append_section(
        text,
        "## Hard-Case Stress Test",
        stress_section,
    )
    text = replace_or_append_section(
        text,
        "## Planned Experiment 2: Residual Prediction",
        RESIDUAL_PLAN_SECTION,
    )
    EXPERIMENT_LOG_PATH.write_text(text, encoding="utf-8")


def write_missing_checkpoint_diagnostic(
    run_dir: Path,
    checkpoint_path: Path,
    stress: dict[str, Any],
    update_log: bool,
) -> None:
    message = (
        "The hard-case stress test could not run because the trained direct "
        f"U-Net checkpoint was not found at `{project_relative(checkpoint_path)}`. "
        "No retraining was started and no existing outputs were modified."
    )
    diagnostic = {
        "status": "blocked_missing_checkpoint",
        "checkpoint_path": project_relative(checkpoint_path),
        "run_dir": project_relative(run_dir),
        "stress_settings": stress,
        "action_taken": "Stopped before loading data or generating blends.",
    }
    with (run_dir / "diagnostics" / "missing_checkpoint.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(diagnostic, handle, indent=2)
    with (run_dir / "diagnostics" / "missing_checkpoint.md").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(
            "# Missing Checkpoint Diagnostic\n\n"
            f"{message}\n\n"
            "Planned stress-test settings:\n\n"
            + "\n".join(f"- {key}: {value}" for key, value in stress.items())
            + "\n"
        )

    stress_section = f"""## Hard-Case Stress Test

Status: blocked on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} local time.

{message}

Exact planned settings:

| Setting | Value |
| --- | ---: |
{format_settings_rows(stress)}

Generated stress blends: 0. The script stopped before data generation because model evaluation would be incomplete without the direct U-Net checkpoint.

Metrics table: not available.

Affected-region MSE improvement ratio: not available.

Comparison to normal held-out test: not available. The normal direct U-Net result remains the current headline result: affected-region MSE 0.062555 -> 0.004428, about 14.1x lower than identity.

Coherence/suspicion status: blocked diagnostic, not a completed stress-test result. The missing checkpoint prevents checking whether the model beats identity/threshold baselines on hard cases.

Recommended next step: restore or place the trained checkpoint at `{project_relative(checkpoint_path)}` and rerun `python3 scripts/run_stress_test.py`. Do not start residual-prediction training until this evaluation completes.

Diagnostic files were saved under `{project_relative(run_dir / "diagnostics")}`.
"""
    if update_log:
        update_experiment_log(stress_section)

    print(message)
    print(f"Diagnostic run directory: {project_relative(run_dir)}")


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state_dict.")
    return state


def load_model(
    checkpoint_path: Path, model_config: dict[str, Any], device: torch.device
) -> models.UNet:
    model = models.UNet(**model_config)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def metric_values(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    whole = gd_utils.compute_metrics(
        pred, target, metrics=("mse", "mae", "psnr", "ssim")
    )
    return {
        "mse": whole["mse"],
        "mae": whole["mae"],
        "psnr": whole["psnr"],
        "ssim": whole["ssim"],
        "masked_mse": gd_utils.masked_mse(pred, target, mask),
        "masked_mae": gd_utils.masked_mae(pred, target, mask),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def target_core_mask(
    target: np.ndarray,
    aperture_fraction: float = 0.18,
    core_percentile: float = 85.0,
) -> np.ndarray:
    """Estimate a simple bright central target-core mask."""
    gray = target.mean(axis=-1)
    height, width = gray.shape
    center_y, center_x = height / 2.0, width / 2.0
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = aperture_fraction * min(height, width)
    aperture = np.hypot(y_grid - center_y, x_grid - center_x) <= radius
    values = gray[aperture]
    threshold = float(np.percentile(values, core_percentile))
    mask = aperture & (gray >= threshold)
    return mask if np.any(mask) else aperture


def generate_stress_blends(
    images: np.ndarray,
    stress: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(stress["seed"]))
    n_requested = int(stress["n_stress_blends"])
    max_attempts = int(stress["max_attempt_multiplier"]) * n_requested
    max_shift = int(stress["max_shift"])
    brightness_range = tuple(float(x) for x in stress["brightness_range"])
    blur_range = tuple(float(x) for x in stress["blur_range"])
    noise_range = tuple(float(x) for x in stress["noise_range"])
    rotation_range = tuple(float(x) for x in stress["rotation_range"])
    min_size_ratio = float(stress["min_size_ratio"])
    min_mask_fraction = float(stress["min_mask_fraction"])
    threshold = float(stress["affected_region_threshold"])

    accepted: list[dict[str, Any]] = []
    relaxed_candidates: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        target_idx, contaminant_idx = rng.choice(images.shape[0], size=2, replace=False)
        target = images[target_idx]
        contaminant = images[contaminant_idx]
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        brightness = float(rng.uniform(*brightness_range))
        blur_sigma = float(rng.uniform(*blur_range))
        noise_std = float(rng.uniform(*noise_range))
        rotation = (
            float(rng.uniform(*rotation_range))
            if rotation_range != (0.0, 0.0)
            else 0.0
        )

        blended, info = gd_blend.blend_pair(
            target=target,
            contaminant=contaminant,
            shift=(dx, dy),
            rotation=rotation,
            brightness=brightness,
            blur_sigma=blur_sigma,
            noise_std=noise_std,
            rng=rng,
        )
        affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
        mask_fraction = float(affected.mean())
        size_ratio = float(info.get("size_ratio", float("nan")))
        finite_ratio = np.isfinite(size_ratio)
        core = target_core_mask(target)
        core_obstruction = (
            float(np.logical_and(affected, core).sum() / core.sum())
            if np.any(core)
            else 0.0
        )

        sample = {
            "target": target,
            "contaminant": contaminant,
            "blended": blended,
            "info": {
                **info,
                "target_index": int(target_idx),
                "contaminant_index": int(contaminant_idx),
                "attempt": int(attempt),
                "mask_fraction": mask_fraction,
                "core_obstruction_fraction": core_obstruction,
            },
        }
        if mask_fraction >= min_mask_fraction and finite_ratio and size_ratio >= min_size_ratio:
            accepted.append(sample)
        elif mask_fraction >= min_mask_fraction and finite_ratio and size_ratio >= 0.5:
            relaxed_candidates.append(sample)

        if len(accepted) >= n_requested:
            break

    if len(accepted) < n_requested:
        needed = n_requested - len(accepted)
        relaxed_candidates.sort(
            key=lambda row: (
                row["info"]["core_obstruction_fraction"],
                row["info"]["mask_fraction"],
                row["info"]["size_ratio"],
            ),
            reverse=True,
        )
        accepted.extend(relaxed_candidates[:needed])

    if len(accepted) < n_requested:
        raise RuntimeError(
            "Could not generate enough stress blends with non-trivial affected "
            f"regions. Requested {n_requested}, accepted {len(accepted)}."
        )

    return accepted[:n_requested]


def predict_model(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
) -> list[np.ndarray]:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            inputs = np.stack([sample["blended"] for sample in batch_samples], axis=0)
            tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
            outputs = model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            predictions.extend(np.clip(outputs, 0.0, 1.0).astype(np.float32))
    return predictions


def blend_severity_bins(blend_severity_score: pd.Series) -> pd.Series:
    try:
        return pd.qcut(
            blend_severity_score.rank(method="first"),
            q=3,
            labels=["easy", "medium", "hard"],
        )
    except ValueError:
        return pd.Series(
            ["unbinned"] * len(blend_severity_score),
            index=blend_severity_score.index,
        )


def core_overlap_bins(core_obstruction_fraction: pd.Series) -> pd.Series:
    """Bin target-core obstruction into transparent low/medium/high groups."""
    return pd.cut(
        core_obstruction_fraction,
        bins=[-0.001, 1.0 / 3.0, 2.0 / 3.0, 1.001],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )


def evaluate_samples(
    samples: list[dict[str, Any]],
    model_predictions: list[np.ndarray],
    affected_region_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_sample_rows: list[dict[str, Any]] = []

    if len(samples) != len(model_predictions):
        raise ValueError(
            "Sample/prediction count mismatch: "
            f"{len(samples)} samples versus {len(model_predictions)} predictions. "
            "Refusing silent zip truncation."
        )

    for index, (sample, model_pred) in enumerate(zip(samples, model_predictions)):
        target = sample["target"]
        blended = sample["blended"]
        threshold_pred = baselines.threshold_baseline(blended)
        mask = gd_utils.affected_region_mask(
            target,
            blended,
            threshold=affected_region_threshold,
        )
        identity_metrics = metric_values(blended, target, mask)
        threshold_metrics = metric_values(threshold_pred, target, mask)
        model_metrics = metric_values(model_pred, target, mask)
        info = sample["info"]
        shift = info.get("shift", (0, 0))
        identity_affected_mae = identity_metrics["masked_mae"]
        mask_fraction = float(mask.mean())
        core_obstruction = float(info.get("core_obstruction_fraction", 0.0))
        # Blend severity measures contaminant damage, not model difficulty.
        blend_severity_score = (
            mask_fraction * identity_affected_mae * (1.0 + core_obstruction)
        )

        row = {
            "index": index,
            "target_index": info.get("target_index"),
            "contaminant_index": info.get("contaminant_index"),
            "generation_difficulty": info.get(
                "generation_difficulty", info.get("difficulty")
            ),
            "shift_x": shift[0],
            "shift_y": shift[1],
            "shift_distance": abs(shift[0]) + abs(shift[1]),
            "brightness": info.get("brightness"),
            "blur_sigma": info.get("blur_sigma"),
            "noise_std": info.get("noise_std"),
            "rotation": info.get("rotation"),
            "target_radius": info.get("target_radius"),
            "contaminant_radius": info.get("contaminant_radius"),
            "size_ratio": info.get("size_ratio"),
            "mask_fraction": mask_fraction,
            "core_obstruction_fraction": core_obstruction,
            "blend_severity_score": blend_severity_score,
        }
        for prefix, values in (
            ("identity", identity_metrics),
            ("threshold", threshold_metrics),
            ("model", model_metrics),
        ):
            for metric_name, value in values.items():
                row[f"{prefix}_{metric_name}"] = value
        row["identity_affected_mse"] = row["identity_masked_mse"]
        row["identity_affected_mae"] = row["identity_masked_mae"]
        row["threshold_affected_mse"] = row["threshold_masked_mse"]
        row["threshold_affected_mae"] = row["threshold_masked_mae"]
        row["model_affected_mse"] = row["model_masked_mse"]
        row["model_affected_mae"] = row["model_masked_mae"]
        row["model_improvement_ratio"] = safe_ratio(
            row["identity_affected_mse"], row["model_affected_mse"]
        )
        row["model_vs_identity_masked_mse_ratio"] = row["model_improvement_ratio"]
        row["threshold_vs_identity_masked_mse_ratio"] = safe_ratio(
            row["identity_affected_mse"], row["threshold_affected_mse"]
        )
        row["model_delta_mse"] = (
            row["model_affected_mse"] - row["identity_affected_mse"]
        )
        row["model_minus_identity_masked_mse"] = row["model_delta_mse"]
        row["model_failure_score"] = row["model_affected_mse"]
        row["model_minus_threshold_masked_mse"] = (
            row["model_masked_mse"] - row["threshold_masked_mse"]
        )
        per_sample_rows.append(row)

    per_sample = pd.DataFrame(per_sample_rows)
    per_sample["blend_severity_bin"] = blend_severity_bins(
        per_sample["blend_severity_score"]
    )
    per_sample["core_overlap_bin"] = core_overlap_bins(
        per_sample["core_obstruction_fraction"]
    )
    per_sample["shift_bin"] = pd.cut(
        per_sample["shift_distance"],
        bins=[-0.1, 12, 24, 36, np.inf],
        labels=["core_overlap", "high_overlap", "moderate_overlap", "lower_overlap"],
    )
    per_sample["brightness_bin"] = pd.cut(
        per_sample["brightness"],
        bins=[0.0, 0.9, 1.1, 1.4, np.inf],
        labels=["0.8-0.9", "0.9-1.1", "1.1-1.4", ">1.4"],
    )
    per_sample["size_ratio_bin"] = pd.cut(
        per_sample["size_ratio"],
        bins=[0.0, 0.75, 1.0, 1.5, np.inf],
        labels=["below_filter", "similar_smaller", "similar_larger", "large_contaminant"],
    )

    aggregate = aggregate_metrics(per_sample)
    severity = severity_table(per_sample)
    return aggregate, per_sample, severity


def aggregate_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in ("identity", "threshold", "model"):
        row = {"method": method, "n": len(per_sample)}
        for metric in ("mse", "mae", "psnr", "ssim", "masked_mse", "masked_mae"):
            row[metric] = float(per_sample[f"{method}_{metric}"].mean())
        row["mean_mask_fraction"] = float(per_sample["mask_fraction"].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result["affected_mse"] = result["masked_mse"]
    result["affected_mae"] = result["masked_mae"]
    identity_masked_mse = float(
        result.loc[result["method"] == "identity", "masked_mse"].iloc[0]
    )
    result["affected_mse_improvement_vs_identity"] = result["masked_mse"].map(
        lambda value: safe_ratio(identity_masked_mse, float(value))
    )
    return result


def group_summary(per_sample: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = per_sample.dropna(subset=[group_col]).groupby(group_col, observed=True)
    for group, frame in grouped:
        identity_mse = float(frame["identity_masked_mse"].mean())
        model_mse = float(frame["model_masked_mse"].mean())
        threshold_mse = float(frame["threshold_masked_mse"].mean())
        rows.append(
            {
                "grouping": group_col,
                "group": str(group),
                "n": int(len(frame)),
                "identity_masked_mse": identity_mse,
                "identity_affected_mse": identity_mse,
                "threshold_masked_mse": threshold_mse,
                "threshold_affected_mse": threshold_mse,
                "model_masked_mse": model_mse,
                "model_affected_mse": model_mse,
                "identity_masked_mae": float(frame["identity_masked_mae"].mean()),
                "identity_affected_mae": float(frame["identity_affected_mae"].mean()),
                "model_masked_mae": float(frame["model_masked_mae"].mean()),
                "model_affected_mae": float(frame["model_affected_mae"].mean()),
                "model_improvement_ratio": safe_ratio(identity_mse, model_mse),
                "threshold_improvement_ratio": safe_ratio(identity_mse, threshold_mse),
                "mean_mask_fraction": float(frame["mask_fraction"].mean()),
                "mean_core_obstruction_fraction": float(
                    frame["core_obstruction_fraction"].mean()
                ),
                "mean_brightness": float(frame["brightness"].mean()),
                "mean_size_ratio": float(frame["size_ratio"].mean()),
                "mean_shift_distance": float(frame["shift_distance"].mean()),
            }
        )
    return pd.DataFrame(rows)


def severity_table(per_sample: pd.DataFrame) -> pd.DataFrame:
    tables = [
        group_summary(per_sample, "generation_difficulty"),
        group_summary(per_sample, "blend_severity_bin"),
        group_summary(per_sample, "core_overlap_bin"),
        group_summary(per_sample, "shift_bin"),
        group_summary(per_sample, "brightness_bin"),
        group_summary(per_sample, "size_ratio_bin"),
    ]
    return pd.concat(tables, ignore_index=True)


def save_rankings(run_dir: Path, per_sample: pd.DataFrame) -> None:
    results_dir = run_dir / "results"
    per_sample.sort_values(
        "model_improvement_ratio", ascending=False
    ).head(10).to_csv(results_dir / "stress_test_top_10_best_improvements.csv", index=False)
    per_sample.sort_values("model_delta_mse", ascending=False).head(
        10
    ).to_csv(results_dir / "stress_test_top_10_worst_failures.csv", index=False)
    per_sample.sort_values(
        ["blend_severity_score", "core_obstruction_fraction"], ascending=False
    ).head(10).to_csv(results_dir / "stress_test_top_10_high_severity.csv", index=False)


def closest_index(frame: pd.DataFrame, column: str, target: float) -> int:
    distances = (frame[column].fillna(0.0) - target).abs()
    row_label = distances.sort_values().index[0]
    return int(frame.loc[row_label, "index"])


def choose_examples(per_sample: pd.DataFrame) -> dict[str, int]:
    finite = per_sample.replace([np.inf, -np.inf], np.nan)
    success_candidates = finite[
        (finite["model_improvement_ratio"] > 2.0)
        & (finite["model_masked_mse"] < finite["identity_masked_mse"])
    ]
    if success_candidates.empty:
        success_idx = closest_index(finite, "model_improvement_ratio", 99.0)
    else:
        success_idx = int(
            success_candidates.sort_values(
                ["model_improvement_ratio", "blend_severity_score"],
                ascending=False,
            ).iloc[0]["index"]
        )

    partial_candidates = finite[
        (finite["model_improvement_ratio"] > 1.0)
        & (finite["model_improvement_ratio"] <= 3.0)
        & (finite["blend_severity_bin"].astype(str) == "hard")
    ]
    if partial_candidates.empty:
        partial_idx = closest_index(finite, "model_improvement_ratio", 1.5)
    else:
        partial_idx = int(
            partial_candidates.sort_values("blend_severity_score", ascending=False).iloc[0][
                "index"
            ]
        )

    failure_candidates = finite[
        (finite["blend_severity_bin"].astype(str) == "hard")
        | (finite["core_obstruction_fraction"] > finite["core_obstruction_fraction"].median())
    ]
    if failure_candidates.empty:
        failure_idx = closest_index(finite, "model_improvement_ratio", 0.0)
    else:
        failure_idx = int(
            failure_candidates.sort_values(
                ["model_improvement_ratio", "blend_severity_score"],
                ascending=[True, False],
            ).iloc[0]["index"]
        )

    return {
        "success": success_idx,
        "partial_failure": partial_idx,
        "largest_failure": failure_idx,
    }


def save_example_figure(
    run_dir: Path,
    samples: list[dict[str, Any]],
    model_predictions: list[np.ndarray],
    per_sample: pd.DataFrame,
    index: int,
    filename: str,
) -> None:
    sample = samples[index]
    row = per_sample.loc[per_sample["index"] == index].iloc[0]
    threshold_pred = baselines.threshold_baseline(sample["blended"])
    images = [
        sample["target"],
        sample["contaminant"],
        sample["blended"],
        threshold_pred,
        model_predictions[index],
    ]
    titles = [
        "Target",
        "Contaminant",
        "Blend",
        "Threshold",
        "Direct U-Net",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(title)
        ax.axis("off")
    ratio = float(row["model_improvement_ratio"])
    warning = " direct U-Net worse than identity" if ratio < 1.0 else ""
    fig.suptitle(
        "idx={idx} blend_severity={severity} shift={shift:.0f} brightness={brightness:.2f} "
        "size_ratio={size_ratio:.2f} model improvement ratio={ratio:.2f}x{warning}".format(
            idx=index,
            severity=row["blend_severity_bin"],
            shift=row["shift_distance"],
            brightness=row["brightness"],
            size_ratio=row["size_ratio"],
            ratio=ratio,
            warning=warning,
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_results(
    run_dir: Path,
    aggregate: pd.DataFrame,
    per_sample: pd.DataFrame,
    severity: pd.DataFrame,
    samples: list[dict[str, Any]],
    model_predictions: list[np.ndarray],
) -> dict[str, int]:
    results_dir = run_dir / "results"
    aggregate.to_csv(results_dir / "stress_test_results.csv", index=False)
    per_sample.to_csv(results_dir / "stress_test_per_sample_results.csv", index=False)
    severity.to_csv(results_dir / "stress_test_grouped_results.csv", index=False)
    save_rankings(run_dir, per_sample)
    examples = choose_examples(per_sample)
    save_example_figure(
        run_dir,
        samples,
        model_predictions,
        per_sample,
        examples["success"],
        "stress_test_success_example.png",
    )
    save_example_figure(
        run_dir,
        samples,
        model_predictions,
        per_sample,
        examples["partial_failure"],
        "stress_test_partial_failure_example.png",
    )
    save_example_figure(
        run_dir,
        samples,
        model_predictions,
        per_sample,
        examples["largest_failure"],
        "stress_test_largest_model_failure_example.png",
    )
    return examples


def format_settings_rows(stress: dict[str, Any]) -> str:
    rows = []
    for key, value in stress.items():
        display = value
        if isinstance(value, list):
            display = f"{value[0]} to {value[1]}"
        rows.append(f"| {key} | {display} |")
    return "\n".join(rows)


def format_metrics_table(aggregate: pd.DataFrame) -> str:
    rows = [
        "| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Affected MAE | Improvement vs identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in aggregate.iterrows():
        rows.append(
            "| {method} | {mse:.6f} | {mae:.6f} | {psnr:.3f} | {ssim:.6f} | "
            "{masked_mse:.6f} | {masked_mae:.6f} | {ratio:.2f}x |".format(
                method=row["method"],
                mse=row["mse"],
                mae=row["mae"],
                psnr=row["psnr"],
                ssim=row["ssim"],
                masked_mse=row["masked_mse"],
                masked_mae=row["masked_mae"],
                ratio=row["affected_mse_improvement_vs_identity"],
            )
        )
    return "\n".join(rows)


def evaluate_coherence(
    aggregate: pd.DataFrame,
    per_sample: pd.DataFrame,
) -> tuple[bool, list[str]]:
    identity = aggregate.loc[aggregate["method"] == "identity"].iloc[0]
    threshold = aggregate.loc[aggregate["method"] == "threshold"].iloc[0]
    model = aggregate.loc[aggregate["method"] == "model"].iloc[0]
    warnings: list[str] = []

    if model["masked_mse"] > identity["masked_mse"] * 1.25:
        warnings.append("identity baseline beats the model by a large margin")
    if model["masked_mse"] < 1e-6 or model["mse"] < 1e-7:
        warnings.append("model metrics are unrealistically close to perfect")
    if model["masked_mse"] < 0.004428 / 3.0:
        warnings.append("stress-test affected MSE is much better than normal test")
    if threshold["masked_mse"] < model["masked_mse"] * 0.75:
        warnings.append("threshold baseline is much better than the learned model")
    if per_sample["mask_fraction"].mean() < 0.005:
        warnings.append("affected-region mask fraction is near zero")
    if per_sample["blend_severity_bin"].nunique(dropna=True) < 2:
        warnings.append("blend severity bins collapsed to fewer than two bins")
    if per_sample["core_overlap_bin"].nunique(dropna=True) < 2:
        warnings.append("core overlap bins collapsed to fewer than two bins")

    return len(warnings) == 0, warnings


def write_diagnostics(run_dir: Path, warnings: Iterable[str], per_sample: pd.DataFrame) -> None:
    warnings = list(warnings)
    if not warnings:
        return
    with (run_dir / "diagnostics" / "stress_test_warnings.md").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("# Stress Test Warnings\n\n")
        for warning in warnings:
            handle.write(f"- {warning}\n")
    per_sample.sort_values("model_delta_mse", ascending=False).head(
        25
    ).to_csv(run_dir / "diagnostics" / "largest_model_regressions.csv", index=False)


def stress_log_section(
    run_dir: Path,
    stress: dict[str, Any],
    aggregate: pd.DataFrame,
    severity: pd.DataFrame,
    coherent: bool,
    warnings: list[str],
    examples: dict[str, int],
) -> str:
    identity = aggregate.loc[aggregate["method"] == "identity"].iloc[0]
    model = aggregate.loc[aggregate["method"] == "model"].iloc[0]
    ratio = safe_ratio(float(identity["masked_mse"]), float(model["masked_mse"]))
    status = "coherent" if coherent else "suspicious; deeper analysis stopped"
    warning_text = "None." if not warnings else "; ".join(warnings) + "."
    blend_severity = severity[severity["grouping"] == "blend_severity_bin"].copy()
    severity_rows = [
        "| Blend severity bin | n | Identity affected MSE | Model affected MSE | Improvement | Mean core obstruction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in blend_severity.iterrows():
        severity_rows.append(
            "| {group} | {n} | {identity:.6f} | {model:.6f} | {ratio:.2f}x | {core:.3f} |".format(
                group=row["group"],
                n=int(row["n"]),
                identity=row["identity_masked_mse"],
                model=row["model_masked_mse"],
                ratio=row["model_improvement_ratio"],
                core=row["mean_core_obstruction_fraction"],
            )
        )

    core_overlap = severity[severity["grouping"] == "core_overlap_bin"].copy()
    core_rows = [
        "| Core overlap bin | n | Identity affected MSE | Model affected MSE | Improvement | Mean core obstruction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in core_overlap.iterrows():
        core_rows.append(
            "| {group} | {n} | {identity:.6f} | {model:.6f} | {ratio:.2f}x | {core:.3f} |".format(
                group=row["group"],
                n=int(row["n"]),
                identity=row["identity_masked_mse"],
                model=row["model_masked_mse"],
                ratio=row["model_improvement_ratio"],
                core=row["mean_core_obstruction_fraction"],
            )
        )

    return f"""## Hard-Case Stress Test

Status: {status}. Run directory: `{project_relative(run_dir)}`.

Exact stress-test settings:

| Setting | Value |
| --- | ---: |
{format_settings_rows(stress)}

Generated stress blends: {int(stress["n_stress_blends"])} from the test partition source subset only.

Metrics table:

{format_metrics_table(aggregate)}

Affected-region MSE improvement ratio: {ratio:.2f}x lower than identity on the stress set.

Comparison to normal held-out test: the normal direct U-Net affected-region improvement was about 14.1x. This stress-test ratio is {'lower than the normal result, which is the expected direction for a harder distribution' if ratio < 14.1 else 'higher than the normal result, which should be inspected against the diagnostics and figures'}.

Blend severity breakdown:

{chr(10).join(severity_rows)}

Core overlap breakdown:

{chr(10).join(core_rows)}

Figure selections:

- Clean success index: {examples["success"]} -> `figures/stress_test_success_example.png`
- Partial failure index: {examples["partial_failure"]} -> `figures/stress_test_partial_failure_example.png`
- Largest model-failure index: {examples["largest_failure"]} -> `figures/stress_test_largest_model_failure_example.png`

Coherence/suspicion status: {warning_text}

Interpretation: the stress set concentrates small shifts, bright contaminants, similar-or-larger contaminant sizes, and target-core obstruction where available. Blend severity bins describe contaminant damage, while model difficulty should be read from model affected error, model delta-MSE, and model improvement ratio.

Recommended next step: if the figures match the blend severity and model-failure breakdowns, use this run to motivate a residual-prediction U-Net comparison on the same normal and hard stress-test sets. If any warnings above are present, inspect diagnostics before continuing.
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    run_dir = make_run_dir(output_root)
    stress = dict(STRESS_DEFAULTS)
    stress["n_stress_blends"] = int(args.n_stress_blends)
    save_run_config(run_dir, config, stress)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        write_missing_checkpoint_diagnostic(
            run_dir,
            checkpoint_path,
            stress,
            update_log=not args.skip_log_update,
        )
        return 2

    checkpoint_stat_before = checkpoint_path.stat()
    resolved_device = gd_train.resolve_accelerator(args.device)
    data_path = PROJECT_ROOT / config["dataset_path"]
    images_raw, labels, _metadata = gd_data.load_galaxy10(data_path)
    images = gd_data.normalise_images(images_raw)
    (_train, _val, test) = gd_data.split_dataset(
        images,
        labels,
        train_frac=float(config["splits"]["train_frac"]),
        val_frac=float(config["splits"]["val_frac"]),
        test_frac=float(config["splits"]["test_frac"]),
        shuffle=True,
        seed=int(config["seed"]),
    )
    test_images, _test_labels = test
    source_subset = int(stress["stress_source_subset"])
    if source_subset > 0:
        test_images = test_images[:source_subset]

    samples = generate_stress_blends(test_images, stress)
    model = load_model(checkpoint_path, config["model"], resolved_device)
    model_predictions = predict_model(
        model,
        samples,
        resolved_device,
        batch_size=int(config["training"].get("batch_size", 8)),
    )
    aggregate, per_sample, severity = evaluate_samples(
        samples,
        model_predictions,
        affected_region_threshold=float(stress["affected_region_threshold"]),
    )
    coherent, warnings = evaluate_coherence(aggregate, per_sample)
    examples = write_results(
        run_dir, aggregate, per_sample, severity, samples, model_predictions
    )
    write_diagnostics(run_dir, warnings, per_sample)

    checkpoint_stat_after = checkpoint_path.stat()
    checkpoint_unchanged = (
        checkpoint_stat_before.st_size == checkpoint_stat_after.st_size
        and checkpoint_stat_before.st_mtime_ns == checkpoint_stat_after.st_mtime_ns
    )
    with (run_dir / "logs" / "checkpoint_integrity.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "checkpoint_path": project_relative(checkpoint_path),
                "size_bytes_before": checkpoint_stat_before.st_size,
                "size_bytes_after": checkpoint_stat_after.st_size,
                "mtime_ns_before": checkpoint_stat_before.st_mtime_ns,
                "mtime_ns_after": checkpoint_stat_after.st_mtime_ns,
                "unchanged": checkpoint_unchanged,
            },
            handle,
            indent=2,
        )
    if not checkpoint_unchanged:
        raise RuntimeError("Checkpoint metadata changed during evaluation.")

    if not args.skip_log_update:
        update_experiment_log(
            stress_log_section(
                run_dir, stress, aggregate, severity, coherent, warnings, examples
            )
        )

    print(f"Stress-test run directory: {project_relative(run_dir)}")
    print(format_metrics_table(aggregate))
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0 if coherent else 3


if __name__ == "__main__":
    raise SystemExit(main())
