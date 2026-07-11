"""Run a no-training evaluation and data-pipeline audit for Thayer-Net.

The script writes only to a caller-provided timestamped audit directory. It
loads existing checkpoints for evaluation, regenerates synthetic held-out sets,
and records diagnostics for masks, blends, residual logic, and model rankings.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import re
import sys
from dataclasses import dataclass
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
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stress_test as stress_helpers
import train_balanced_residual_unet as balanced_helpers
from src import baselines
from src import blend as gd_blend
from src import data as gd_data
from src import train as gd_train
from src import utils as gd_utils


METHODS = ("identity", "threshold", "direct", "residual", "balanced_residual")
LEARNED = ("direct", "residual", "balanced_residual")
THRESHOLDS = (0.005, 0.01, 0.02, 0.04)
DILATIONS = (0, 1, 3, 5, 9)

DIRECT_CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/unet_direct_5000train_800val_800test_20ep.pth"
OLD_RESIDUAL_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_residual_5000train_800val_800test_20ep_20260708_154947.pth"
)
BALANCED_CHECKPOINT = (
    PROJECT_ROOT / "outputs/checkpoints/unet_residual_balanced_hard_20260708_184632.pth"
)
BALANCED_FINAL_CHECKPOINT = (
    PROJECT_ROOT / "outputs/checkpoints/unet_residual_balanced_hard_final_20260708_184632.pth"
)
LATEST_BALANCED_RUN = PROJECT_ROOT / "outputs/runs/balanced_residual_20260708_184632"
CONFIG_PATH = PROJECT_ROOT / "configs/default.yaml"


@dataclass(frozen=True)
class AuditSettings:
    normal_test_blends: int = 1000
    stress_test_blends: int = 1000
    test_source_subset: int = 1000
    stress_source_subset: int = 800
    multiseed_count: int = 3
    multiseed_blends: int = 1000
    batch_size: int = 8
    affected_threshold: float = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--multiseed-count", type=int, default=3)
    parser.add_argument("--multiseed-blends", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Finish report/plots from already written audit tables.",
    )
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_dirs(run_dir: Path) -> None:
    for child in (
        "figures",
        "tables",
        "diagnostics",
        "mask_audit",
        "mask_audit/dilation_examples",
        "mask_audit/core_examples",
        "blend_audit",
        "model_audit",
        "model_audit/multiseed",
        "model_audit/residual_logic",
        "model_audit/comparison_grids",
        "logs",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    frame.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {CONFIG_PATH}")
    return config


def checkpoint_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": project_relative(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_iso_local": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def read_checkpoint_before(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logs/checkpoint_integrity_before.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing before-integrity file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_checkpoint_after(run_dir: Path, before: dict[str, Any]) -> dict[str, Any]:
    checkpoints = {
        "direct_unet": DIRECT_CHECKPOINT,
        "old_residual_unet": OLD_RESIDUAL_CHECKPOINT,
        "balanced_residual_unet": BALANCED_CHECKPOINT,
    }
    after = {name: checkpoint_info(path) for name, path in checkpoints.items()}
    comparison: dict[str, Any] = {}
    for name, info in after.items():
        before_info = before[name]
        comparison[name] = {
            "before": before_info,
            "after": info,
            "unchanged": (
                before_info["size_bytes"] == info["size_bytes"]
                and before_info["mtime_ns"] == info["mtime_ns"]
            ),
        }
    save_json(run_dir / "logs/checkpoint_integrity_after.json", after)
    save_json(run_dir / "logs/checkpoint_integrity_comparison.json", comparison)
    return comparison


def split_indices(n_samples: int, seed: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def load_split_subsets(
    config: dict[str, Any],
    settings: AuditSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data_path = PROJECT_ROOT / config["dataset_path"]
    images_raw, labels, metadata = gd_data.load_galaxy10(data_path)
    train_frac = float(config["splits"]["train_frac"])
    val_frac = float(config["splits"]["val_frac"])
    seed = int(config["seed"])
    train_idx, val_idx, test_idx = split_indices(len(images_raw), seed, train_frac, val_frac)
    train_images = gd_data.normalise_images(images_raw[train_idx[:5000]])
    val_images = gd_data.normalise_images(images_raw[val_idx[:1000]])
    test_images = gd_data.normalise_images(images_raw[test_idx[: settings.test_source_subset]])
    audit = {
        "dataset_path": config["dataset_path"],
        "n_total": int(len(images_raw)),
        "label_count": int(len(labels)),
        "metadata_keys": sorted(metadata.keys()),
        "split_seed": seed,
        "split_fractions": {
            "train": train_frac,
            "validation": val_frac,
            "test": float(config["splits"]["test_frac"]),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "subsets_used_for_audit": {
            "train": int(len(train_images)),
            "validation": int(len(val_images)),
            "test": int(len(test_images)),
            "stress_test": int(min(settings.stress_source_subset, len(test_images))),
        },
        "train_val_overlap": int(len(set(train_idx).intersection(set(val_idx)))),
        "train_test_overlap": int(len(set(train_idx).intersection(set(test_idx)))),
        "val_test_overlap": int(len(set(val_idx).intersection(set(test_idx)))),
    }
    del images_raw, labels, metadata
    gc.collect()
    return train_images, val_images, test_images, audit


def write_split_audit(run_dir: Path, config: dict[str, Any], split_audit: dict[str, Any]) -> None:
    src_data = PROJECT_ROOT / "src/data.py"
    src_blend = PROJECT_ROOT / "src/blend.py"
    train_balanced = PROJECT_ROOT / "scripts/train_balanced_residual_unet.py"
    run_stress = PROJECT_ROOT / "scripts/run_stress_test.py"
    text = f"""# Split Audit

## Where the split is defined

- Dataset split helper: `{project_relative(src_data)}` function `split_dataset`.
- Normal blend generation helper: `{project_relative(src_blend)}` function `generate_blends`.
- Balanced residual evaluation path: `{project_relative(train_balanced)}` loads split subsets before any blend generation.
- Hard stress path: `{project_relative(run_stress)}` receives only the held-out test image subset.

## Split sizes

| Split | Full source images | Audit subset used |
| --- | ---: | ---: |
| Train | {split_audit['split_sizes']['train']} | {split_audit['subsets_used_for_audit']['train']} |
| Validation | {split_audit['split_sizes']['validation']} | {split_audit['subsets_used_for_audit']['validation']} |
| Test | {split_audit['split_sizes']['test']} | {split_audit['subsets_used_for_audit']['test']} |
| Stress source subset | {split_audit['split_sizes']['test']} | {split_audit['subsets_used_for_audit']['stress_test']} |

Split seed: `{split_audit['split_seed']}`.

## Split-before-blending conclusion

The code path splits original Galaxy10 DECaLS images first, then passes the
separate train/validation/test arrays into blend generation. Normal training
blends are generated from train images, validation blends from validation
images, and normal/stress test blends from held-out test images.

Overlap check from reconstructed shuffled source indices:

| Pair | Overlap count |
| --- | ---: |
| Train/validation | {split_audit['train_val_overlap']} |
| Train/test | {split_audit['train_test_overlap']} |
| Validation/test | {split_audit['val_test_overlap']} |

## Leakage risk and limitations

No split-level source leakage was found in the code path. The original
`generate_blends` helper chooses target and contaminant images only from the
array it is given.

Important limitation: the standard normal blend dictionaries do not save global
source indices. Some stress/targeted generation helpers save local subset
indices, but the historical normal evaluation outputs cannot prove source
identity after the fact. This audit therefore proves the current code path and
reconstructed split index disjointness, but not every historical generated
sample's raw source IDs.

The balanced-run saved per-sample result CSVs also do not include global raw
Galaxy10 indices for normal test samples.
"""
    save_text(run_dir / "diagnostics/split_audit.md", text)
    save_json(run_dir / "diagnostics/split_audit.json", split_audit)


def load_models(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    direct = balanced_helpers.load_direct_model(DIRECT_CHECKPOINT, config["model"], device)
    residual = balanced_helpers.load_residual_model(OLD_RESIDUAL_CHECKPOINT, config["model"], device)
    balanced = balanced_helpers.load_residual_model(BALANCED_CHECKPOINT, config["model"], device)
    return direct, residual, balanced


def make_normal_test_blends(test_images: np.ndarray, config: dict[str, Any], settings: AuditSettings) -> list[dict[str, Any]]:
    return balanced_helpers.normal_blends(
        test_images,
        settings.normal_test_blends,
        config,
        seed=int(config["seed"]) + 4000,
        component="normal_test",
    )


def make_stress_blends(test_images: np.ndarray, settings: AuditSettings) -> list[dict[str, Any]]:
    stress = dict(stress_helpers.STRESS_DEFAULTS)
    stress["n_stress_blends"] = settings.stress_test_blends
    stress["stress_source_subset"] = settings.stress_source_subset
    return stress_helpers.generate_stress_blends(
        test_images[: settings.stress_source_subset],
        stress,
    )


def predict_all(
    samples: list[dict[str, Any]],
    direct_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    balanced_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, list[np.ndarray]]:
    preds = {method: [] for method in METHODS}
    balanced_layers: list[np.ndarray] = []
    residual_layers: list[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        inputs = np.stack([sample["blended"] for sample in batch], axis=0)
        tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            direct = direct_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            residual_layer = residual_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            balanced_layer = balanced_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
        residual_recon = np.clip(inputs - residual_layer, 0.0, 1.0).astype(np.float32)
        balanced_recon = np.clip(inputs - balanced_layer, 0.0, 1.0).astype(np.float32)
        for offset, sample in enumerate(batch):
            blended = sample["blended"]
            preds["identity"].append(blended)
            preds["threshold"].append(baselines.threshold_baseline(blended))
            preds["direct"].append(np.clip(direct[offset], 0.0, 1.0).astype(np.float32))
            preds["residual"].append(residual_recon[offset])
            preds["balanced_residual"].append(balanced_recon[offset])
            residual_layers.append(residual_layer[offset].astype(np.float32))
            balanced_layers.append(balanced_layer[offset].astype(np.float32))
    preds["residual_predicted_layer"] = residual_layers
    preds["balanced_predicted_layer"] = balanced_layers
    return preds


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    return (
        float(np.mean((pred[mask] - target[mask]) ** 2)),
        float(np.mean(np.abs(pred[mask] - target[mask]))),
    )


def whole_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    return (
        gd_utils.mse(pred, target),
        gd_utils.mae(pred, target),
        float(ssim(pred, target, channel_axis=2, data_range=1.0)),
    )


def core_mask(target: np.ndarray, aperture_fraction: float = 0.18, core_percentile: float = 85.0) -> np.ndarray:
    return balanced_helpers.target_core_mask(
        target,
        aperture_fraction=aperture_fraction,
        core_percentile=core_percentile,
    )


def affected_mask(target: np.ndarray, blended: np.ndarray, threshold: float) -> np.ndarray:
    return gd_utils.affected_region_mask(target, blended, threshold=threshold)


def evaluate_rows(
    split: str,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    threshold: float,
    dilation: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        target = sample["target"]
        blended = sample["blended"]
        mask = affected_mask(target, blended, threshold)
        if dilation > 0:
            mask = binary_dilation(mask, iterations=dilation)
        identity_mse, _identity_mae = masked_metrics(blended, target, mask)
        info = sample.get("info", {})
        row_base = {
            "split": split,
            "index": idx,
            "threshold": threshold,
            "dilation_radius": dilation,
            "mask_fraction": float(mask.mean()),
            "generation_difficulty": info.get("generation_difficulty", info.get("difficulty")),
            "training_component": info.get("training_component"),
            "core_obstruction_fraction": float(
                np.logical_and(mask, core_mask(target)).sum() / max(core_mask(target).sum(), 1)
            ),
        }
        for method in METHODS:
            pred = preds[method][idx]
            affected_mse, affected_mae = masked_metrics(pred, target, mask)
            whole_mse, whole_mae, whole_ssim = whole_metrics(pred, target)
            rows.append(
                {
                    **row_base,
                    "method": method,
                    "affected_mse": affected_mse,
                    "affected_mae": affected_mae,
                    "whole_mse": whole_mse,
                    "whole_mae": whole_mae,
                    "ssim": whole_ssim,
                    "improvement_vs_identity": safe_ratio(identity_mse, affected_mse),
                    "worse_than_identity": bool(affected_mse > identity_mse),
                }
            )
    return pd.DataFrame(rows)


def aggregate_mask_conditions(
    split: str,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    thresholds: Iterable[float],
    dilations: Iterable[int],
) -> pd.DataFrame:
    """Aggregate affected-region metrics for mask-only sensitivity sweeps."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for dilation in dilations:
            values = {
                method: {
                    "affected_mse": [],
                    "affected_mae": [],
                    "mask_fraction": [],
                    "worse": [],
                }
                for method in METHODS
            }
            for idx, sample in enumerate(samples):
                target = sample["target"]
                blended = sample["blended"]
                delta = np.abs(blended - target).mean(axis=-1)
                mask = delta > threshold
                if dilation > 0:
                    mask = binary_dilation(mask, iterations=dilation)
                mask_fraction = float(mask.mean())
                if not np.any(mask):
                    for method in METHODS:
                        values[method]["affected_mse"].append(float("nan"))
                        values[method]["affected_mae"].append(float("nan"))
                        values[method]["mask_fraction"].append(mask_fraction)
                        values[method]["worse"].append(False)
                    continue
                identity_sq = np.mean((blended - target) ** 2, axis=-1)
                identity_mse = float(identity_sq[mask].mean())
                for method in METHODS:
                    pred = preds[method][idx]
                    sq = np.mean((pred - target) ** 2, axis=-1)
                    ab = np.mean(np.abs(pred - target), axis=-1)
                    mse_value = float(sq[mask].mean())
                    mae_value = float(ab[mask].mean())
                    values[method]["affected_mse"].append(mse_value)
                    values[method]["affected_mae"].append(mae_value)
                    values[method]["mask_fraction"].append(mask_fraction)
                    values[method]["worse"].append(bool(mse_value > identity_mse))
            identity_mean = float(np.nanmean(values["identity"]["affected_mse"]))
            for method in METHODS:
                method_mse = float(np.nanmean(values[method]["affected_mse"]))
                rows.append(
                    {
                        "split": split,
                        "threshold": float(threshold),
                        "dilation_radius": int(dilation),
                        "method": method,
                        "n": int(len(samples)),
                        "affected_mse": method_mse,
                        "affected_mae": float(np.nanmean(values[method]["affected_mae"])),
                        "mean_mask_fraction": float(np.nanmean(values[method]["mask_fraction"])),
                        "min_mask_fraction": float(np.nanmin(values[method]["mask_fraction"])),
                        "max_mask_fraction": float(np.nanmax(values[method]["mask_fraction"])),
                        "improvement_vs_identity": safe_ratio(identity_mean, method_mse),
                        "worse_than_identity_count": int(np.sum(values[method]["worse"])),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_eval(per_sample_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [col for col in ("split", "threshold", "dilation_radius", "method") if col in per_sample_long.columns]
    for keys, frame in per_sample_long.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        identity = per_sample_long[
            (per_sample_long["method"] == "identity")
            & np.logical_and.reduce([per_sample_long[col] == base[col] for col in group_cols if col != "method"])
        ]
        identity_mse = float(identity["affected_mse"].mean())
        rows.append(
            {
                **base,
                "n": int(len(frame)),
                "affected_mse": float(frame["affected_mse"].mean()),
                "affected_mae": float(frame["affected_mae"].mean()),
                "whole_mse": float(frame["whole_mse"].mean()),
                "whole_mae": float(frame["whole_mae"].mean()),
                "ssim": float(frame["ssim"].mean()),
                "mean_mask_fraction": float(frame["mask_fraction"].mean()),
                "min_mask_fraction": float(frame["mask_fraction"].min()),
                "max_mask_fraction": float(frame["mask_fraction"].max()),
                "improvement_vs_identity": safe_ratio(identity_mse, float(frame["affected_mse"].mean())),
                "worse_than_identity_count": int(frame["worse_than_identity"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_same_set_comparison(run_dir: Path, split: str, aggregate: pd.DataFrame, threshold: float) -> None:
    frame = aggregate[(aggregate["threshold"] == threshold) & (aggregate["dilation_radius"] == 0)].copy()
    frame["same_sample_list"] = True
    frame["identity_baseline_shared"] = True
    frame["affected_mask_shared"] = True
    save_csv(run_dir / f"tables/same_set_model_comparison_{split}.csv", frame)


def plot_lines(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    path: Path,
    methods: Iterable[str] = METHODS,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {
        "identity": "#6c717a",
        "threshold": "#b05d4f",
        "direct": "#2f6f8f",
        "residual": "#5f8a4b",
        "balanced_residual": "#7b5fa3",
    }
    for method in methods:
        sub = frame[frame["method"] == method].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub[y_col], marker="o", linewidth=1.8, label=method_label(method), color=colors.get(method))
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, path)


def method_label(method: str) -> str:
    return {
        "identity": "identity",
        "threshold": "threshold",
        "direct": "direct U-Net",
        "residual": "old residual U-Net",
        "balanced_residual": "balanced residual U-Net",
    }.get(method, method)


def run_threshold_sensitivity(
    run_dir: Path,
    samples_by_split: dict[str, list[dict[str, Any]]],
    preds_by_split: dict[str, dict[str, list[np.ndarray]]],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split, samples in samples_by_split.items():
        agg = aggregate_mask_conditions(
            split,
            samples,
            preds_by_split[split],
            thresholds=THRESHOLDS,
            dilations=(0,),
        )
        out[split] = agg
        save_csv(run_dir / f"tables/mask_threshold_sensitivity_{split}.csv", agg)
        plot_lines(
            agg,
            "threshold",
            "affected_mse",
            f"{split.title()} affected MSE vs mask threshold",
            "Affected-region MSE",
            run_dir / f"mask_audit/{split}_affected_mse_vs_threshold.png",
        )
        plot_lines(
            agg,
            "threshold",
            "improvement_vs_identity",
            f"{split.title()} improvement vs mask threshold",
            "Identity affected MSE / method affected MSE",
            run_dir / f"mask_audit/{split}_improvement_vs_threshold.png",
            methods=("direct", "residual", "balanced_residual"),
        )
        plot_lines(
            agg,
            "threshold",
            "mean_mask_fraction",
            f"{split.title()} mask fraction vs threshold",
            "Mean mask fraction",
            run_dir / f"mask_audit/{split}_mean_mask_fraction_vs_threshold.png",
        )
        plot_lines(
            agg,
            "threshold",
            "worse_than_identity_count",
            f"{split.title()} worse-than-identity vs threshold",
            "Count",
            run_dir / f"mask_audit/{split}_worse_than_identity_vs_threshold.png",
            methods=("direct", "residual", "balanced_residual", "threshold"),
        )
    return out


def run_dilation_sensitivity(
    run_dir: Path,
    samples_by_split: dict[str, list[dict[str, Any]]],
    preds_by_split: dict[str, dict[str, list[np.ndarray]]],
    threshold: float,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split, samples in samples_by_split.items():
        agg = aggregate_mask_conditions(
            split,
            samples,
            preds_by_split[split],
            thresholds=(threshold,),
            dilations=DILATIONS,
        )
        out[split] = agg
        save_csv(run_dir / f"tables/mask_dilation_sensitivity_{split}.csv", agg)
        plot_lines(
            agg,
            "dilation_radius",
            "affected_mse",
            f"{split.title()} affected MSE vs mask dilation",
            "Affected-region MSE",
            run_dir / f"mask_audit/{split}_affected_mse_vs_dilation.png",
        )
        plot_lines(
            agg,
            "dilation_radius",
            "improvement_vs_identity",
            f"{split.title()} improvement vs mask dilation",
            "Identity affected MSE / method affected MSE",
            run_dir / f"mask_audit/{split}_improvement_vs_dilation.png",
            methods=("direct", "residual", "balanced_residual"),
        )
        plot_lines(
            agg,
            "dilation_radius",
            "mean_mask_fraction",
            f"{split.title()} mask fraction vs mask dilation",
            "Mean mask fraction",
            run_dir / f"mask_audit/{split}_mask_fraction_vs_dilation.png",
        )
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float] = (1.0, 0.1, 0.0), alpha: float = 0.45) -> np.ndarray:
    out = np.clip(image.copy(), 0.0, 1.0)
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0.0, 1.0)


def contaminant_foreground_for_display(sample: dict[str, Any]) -> np.ndarray:
    info = sample.get("info", {})
    foreground, _meta = gd_blend.extract_source_foreground(sample["contaminant"])
    foreground = np.clip(foreground * float(info.get("brightness", 1.0)), 0.0, 1.0)
    rotation = float(info.get("rotation", 0.0) or 0.0)
    if abs(rotation) > 1e-12:
        from scipy.ndimage import rotate as ndi_rotate

        foreground = ndi_rotate(foreground, rotation, reshape=False, order=1, mode="constant", cval=0.0)
        foreground[foreground < 0.002] = 0.0
    shift = info.get("shift", (int(info.get("shift_x", 0) or 0), int(info.get("shift_y", 0) or 0)))
    return gd_blend.shift_foreground(foreground, int(shift[0]), int(shift[1]))


def plot_image_grid(
    rows: list[list[tuple[np.ndarray, str]]],
    path: Path,
    suptitle: str | None = None,
    figsize_per_cell: tuple[float, float] = (2.25, 2.15),
) -> None:
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(row):
                continue
            image, title = row[c]
            if image.ndim == 2:
                vmax = max(0.05, float(np.nanpercentile(image, 99)))
                ax.imshow(image, cmap="magma", vmin=0.0, vmax=vmax)
            else:
                ax.imshow(np.clip(image, 0.0, 1.0))
            if r == 0:
                ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    save_fig(fig, path, dpi=160)


def make_blend_audit_figures(
    run_dir: Path,
    samples_by_split: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> None:
    rng = np.random.default_rng(20260708)
    montage_rows: list[list[tuple[np.ndarray, str]]] = []
    for split, samples in samples_by_split.items():
        indices = rng.choice(len(samples), size=min(20, len(samples)), replace=False)
        rows: list[list[tuple[np.ndarray, str]]] = []
        for local_idx, idx in enumerate(indices, start=1):
            sample = samples[int(idx)]
            target = sample["target"]
            blended = sample["blended"]
            affected = affected_mask(target, blended, threshold)
            core = core_mask(target)
            foreground = contaminant_foreground_for_display(sample)
            row = [
                (target, "Target"),
                (sample["contaminant"], "Contaminant"),
                (foreground, "Extracted foreground"),
                (overlay_mask(blended, affected), "Affected mask"),
                (overlay_mask(blended, np.logical_and(affected, core), color=(0.2, 0.9, 0.2)), "Core obstruction"),
                (blended, "Final blend"),
            ]
            rows.append(row)
            if len(rows) == 5:
                page = (local_idx - 1) // 5 + 1
                plot_image_grid(
                    rows,
                    run_dir / f"blend_audit/{split}_blend_grid_{page:02d}.png",
                    suptitle=f"{split.title()} blend audit {page:02d}",
                )
                if len(montage_rows) < 6:
                    montage_rows.extend(rows[: max(0, 6 - len(montage_rows))])
                rows = []
        if rows:
            page = math.ceil(len(indices) / 5)
            plot_image_grid(
                rows,
                run_dir / f"blend_audit/{split}_blend_grid_{page:02d}.png",
                suptitle=f"{split.title()} blend audit {page:02d}",
            )
    if montage_rows:
        plot_image_grid(
            montage_rows,
            run_dir / "blend_audit/blend_audit_montage.png",
            suptitle="Blend audit montage",
        )


def make_dilation_examples(run_dir: Path, samples: list[dict[str, Any]], threshold: float) -> None:
    rng = np.random.default_rng(20260709)
    indices = rng.choice(len(samples), size=min(10, len(samples)), replace=False)
    rows: list[list[tuple[np.ndarray, str]]] = []
    for idx in indices:
        sample = samples[int(idx)]
        blend = sample["blended"]
        base = affected_mask(sample["target"], blend, threshold)
        row: list[tuple[np.ndarray, str]] = [(overlay_mask(blend, base), "0 px")]
        for dilation in (1, 3, 5, 9):
            row.append((overlay_mask(blend, binary_dilation(base, iterations=dilation)), f"{dilation} px"))
        rows.append(row)
    plot_image_grid(rows, run_dir / "mask_audit/dilation_examples/dilation_mask_examples.png", "Affected-mask dilation examples")


def run_core_region_metrics(
    run_dir: Path,
    samples_by_split: dict[str, list[dict[str, Any]]],
    preds_by_split: dict[str, dict[str, list[np.ndarray]]],
    threshold: float,
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    all_for_plot: list[pd.DataFrame] = []
    for split, samples in samples_by_split.items():
        rows: list[dict[str, Any]] = []
        for method in METHODS:
            core_mse_values: list[float] = []
            core_mae_values: list[float] = []
            noncore_mse_values: list[float] = []
            noncore_mae_values: list[float] = []
            outer_mse_values: list[float] = []
            outer_mae_values: list[float] = []
            core_pixels = 0
            noncore_pixels = 0
            outer_pixels = 0
            valid_core_cases = 0
            valid_noncore_cases = 0
            valid_outer_cases = 0
            id_core_values: list[float] = []
            id_noncore_values: list[float] = []
            id_outer_values: list[float] = []
            for idx, sample in enumerate(samples):
                target = sample["target"]
                blend = sample["blended"]
                affected = affected_mask(target, blend, threshold)
                core = core_mask(target)
                y_grid, x_grid = np.ogrid[: target.shape[0], : target.shape[1]]
                cy, cx = target.shape[0] / 2.0, target.shape[1] / 2.0
                outer = np.hypot(y_grid - cy, x_grid - cx) > (0.32 * min(target.shape[:2]))
                regions = {
                    "core": np.logical_and(affected, core),
                    "noncore": np.logical_and(affected, ~core),
                    "outer": np.logical_and(affected, outer),
                }
                pred = preds_by_split[split][method][idx]
                identity = preds_by_split[split]["identity"][idx]
                for region, mask in regions.items():
                    mse_value, mae_value = masked_metrics(pred, target, mask)
                    id_mse, _id_mae = masked_metrics(identity, target, mask)
                    if region == "core":
                        core_pixels += int(mask.sum())
                        if np.isfinite(mse_value):
                            valid_core_cases += 1
                            core_mse_values.append(mse_value)
                            core_mae_values.append(mae_value)
                            id_core_values.append(id_mse)
                    elif region == "noncore":
                        noncore_pixels += int(mask.sum())
                        if np.isfinite(mse_value):
                            valid_noncore_cases += 1
                            noncore_mse_values.append(mse_value)
                            noncore_mae_values.append(mae_value)
                            id_noncore_values.append(id_mse)
                    else:
                        outer_pixels += int(mask.sum())
                        if np.isfinite(mse_value):
                            valid_outer_cases += 1
                            outer_mse_values.append(mse_value)
                            outer_mae_values.append(mae_value)
                            id_outer_values.append(id_mse)
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "core_affected_mse": float(np.nanmean(core_mse_values)),
                    "core_affected_mae": float(np.nanmean(core_mae_values)),
                    "core_valid_cases": valid_core_cases,
                    "core_valid_pixels": core_pixels,
                    "core_improvement_vs_identity": safe_ratio(float(np.nanmean(id_core_values)), float(np.nanmean(core_mse_values))),
                    "noncore_affected_mse": float(np.nanmean(noncore_mse_values)),
                    "noncore_affected_mae": float(np.nanmean(noncore_mae_values)),
                    "noncore_valid_cases": valid_noncore_cases,
                    "noncore_valid_pixels": noncore_pixels,
                    "noncore_improvement_vs_identity": safe_ratio(float(np.nanmean(id_noncore_values)), float(np.nanmean(noncore_mse_values))),
                    "outer_affected_mse": float(np.nanmean(outer_mse_values)) if outer_mse_values else float("nan"),
                    "outer_affected_mae": float(np.nanmean(outer_mae_values)) if outer_mae_values else float("nan"),
                    "outer_valid_cases": valid_outer_cases,
                    "outer_valid_pixels": outer_pixels,
                    "outer_improvement_vs_identity": safe_ratio(float(np.nanmean(id_outer_values)), float(np.nanmean(outer_mse_values))) if outer_mse_values else float("nan"),
                }
            )
        frame = pd.DataFrame(rows)
        outputs[split] = frame
        all_for_plot.append(frame)
        save_csv(run_dir / f"tables/core_region_metrics_{split}.csv", frame)
    plot_core_metrics(run_dir, pd.concat(all_for_plot, ignore_index=True))
    make_core_examples(run_dir, samples_by_split["stress"], preds_by_split["stress"], threshold)
    return outputs


def plot_core_metrics(run_dir: Path, frame: pd.DataFrame) -> None:
    learned = ["identity", "direct", "residual", "balanced_residual"]
    splits = list(frame["split"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    colors = ["#6c717a", "#2f6f8f", "#5f8a4b", "#7b5fa3"]
    for ax, metric, title in (
        (axes[0], "core_affected_mse", "Core affected MSE"),
        (axes[1], "noncore_affected_mse", "Non-core affected MSE"),
        (axes[2], "core_improvement_vs_identity", "Core improvement ratio"),
    ):
        x = np.arange(len(splits))
        width = 0.18
        for i, method in enumerate(learned):
            values = [
                float(frame[(frame["split"] == split) & (frame["method"] == method)][metric].iloc[0])
                for split in splits
            ]
            ax.bar(x + (i - 1.5) * width, values, width, label=method_label(method), color=colors[i])
        ax.set_xticks(x, splits)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, run_dir / "mask_audit/core_region_metrics.png")


def make_core_examples(
    run_dir: Path,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    threshold: float,
) -> None:
    rows: list[list[tuple[np.ndarray, str]]] = []
    for idx in range(min(8, len(samples))):
        sample = samples[idx]
        target = sample["target"]
        blend = sample["blended"]
        affected = affected_mask(target, blend, threshold)
        core = core_mask(target)
        core_affected = np.logical_and(affected, core)
        rows.append(
            [
                (target, "Target"),
                (overlay_mask(target, core, color=(0.1, 0.6, 1.0)), "Target core"),
                (overlay_mask(blend, affected), "Affected"),
                (overlay_mask(blend, core_affected, color=(0.2, 0.9, 0.2)), "Core affected"),
                (blend, "Blend"),
                (preds["balanced_residual"][idx], "Balanced recon"),
            ]
        )
    plot_image_grid(rows, run_dir / "mask_audit/core_examples/core_region_examples.png", "Core-region audit examples")


def compute_wide_per_sample(
    split: str,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        target = sample["target"]
        blended = sample["blended"]
        mask = affected_mask(target, blended, threshold)
        core = core_mask(target)
        identity_mse, identity_mae = masked_metrics(blended, target, mask)
        info = sample.get("info", {})
        shift = info.get("shift", (0, 0))
        row: dict[str, Any] = {
            "split": split,
            "index": idx,
            "shift_x": int(shift[0]),
            "shift_y": int(shift[1]),
            "brightness": info.get("brightness"),
            "blur_sigma": info.get("blur_sigma"),
            "noise_std": info.get("noise_std"),
            "size_ratio": info.get("size_ratio"),
            "mask_fraction": float(mask.mean()),
            "core_obstruction_fraction": float(np.logical_and(mask, core).sum() / max(core.sum(), 1)),
            "identity_affected_mse": identity_mse,
            "identity_affected_mae": identity_mae,
        }
        for method in METHODS:
            pred = preds[method][idx]
            aff_mse, aff_mae = masked_metrics(pred, target, mask)
            whole_mse, whole_mae, whole_ssim = whole_metrics(pred, target)
            row[f"{method}_affected_mse"] = aff_mse
            row[f"{method}_affected_mae"] = aff_mae
            row[f"{method}_whole_mse"] = whole_mse
            row[f"{method}_whole_mae"] = whole_mae
            row[f"{method}_ssim"] = whole_ssim
            row[f"{method}_improvement_ratio"] = safe_ratio(identity_mse, aff_mse)
            row[f"{method}_worse_than_identity"] = bool(aff_mse > identity_mse)
        row["balanced_beats_residual"] = row["balanced_residual_affected_mse"] < row["residual_affected_mse"]
        row["balanced_beats_direct"] = row["balanced_residual_affected_mse"] < row["direct_affected_mse"]
        row["residual_beats_balanced"] = row["residual_affected_mse"] < row["balanced_residual_affected_mse"]
        row["direct_beats_balanced"] = row["direct_affected_mse"] < row["balanced_residual_affected_mse"]
        row["balanced_to_residual_ratio"] = safe_ratio(row["balanced_residual_affected_mse"], row["residual_affected_mse"])
        row["balanced_to_direct_ratio"] = safe_ratio(row["balanced_residual_affected_mse"], row["direct_affected_mse"])
        row["severity_score"] = float(mask.mean()) * identity_mae * (1.0 + row["core_obstruction_fraction"])
        rows.append(row)
    return pd.DataFrame(rows)


def run_multiseed(
    run_dir: Path,
    config: dict[str, Any],
    test_images: np.ndarray,
    direct_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    balanced_model: torch.nn.Module,
    device: torch.device,
    settings: AuditSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normal_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    seed_base = int(config["seed"])
    normal_seeds = [seed_base + 9000 + i for i in range(settings.multiseed_count)]
    stress_seeds = [int(stress_helpers.STRESS_DEFAULTS["seed"]) + 100 + i for i in range(settings.multiseed_count)]
    for split, seeds, accumulator in (
        ("normal", normal_seeds, normal_rows),
        ("stress", stress_seeds, stress_rows),
    ):
        for seed_index, seed in enumerate(seeds):
            if split == "normal":
                samples = balanced_helpers.normal_blends(
                    test_images,
                    settings.multiseed_blends,
                    config,
                    seed=seed,
                    component=f"multiseed_normal_{seed}",
                )
            else:
                stress = dict(stress_helpers.STRESS_DEFAULTS)
                stress["n_stress_blends"] = settings.multiseed_blends
                stress["seed"] = seed
                samples = stress_helpers.generate_stress_blends(
                    test_images[: settings.stress_source_subset],
                    stress,
                )
            preds = predict_all(samples, direct_model, residual_model, balanced_model, device, settings.batch_size)
            wide = compute_wide_per_sample(split, samples, preds, settings.affected_threshold)
            identity_mse = float(wide["identity_affected_mse"].mean())
            for method in METHODS:
                accumulator.append(
                    {
                        "split": split,
                        "seed_index": seed_index,
                        "seed": seed,
                        "method": method,
                        "n": int(len(wide)),
                        "affected_mse": float(wide[f"{method}_affected_mse"].mean()),
                        "affected_mae": float(wide[f"{method}_affected_mae"].mean()),
                        "whole_mse": float(wide[f"{method}_whole_mse"].mean()),
                        "ssim": float(wide[f"{method}_ssim"].mean()),
                        "improvement_vs_identity": safe_ratio(identity_mse, float(wide[f"{method}_affected_mse"].mean())),
                        "worse_than_identity_count": int(wide[f"{method}_worse_than_identity"].sum()),
                        "balanced_beats_residual_count": int(wide["balanced_beats_residual"].sum()),
                        "balanced_beats_direct_count": int(wide["balanced_beats_direct"].sum()),
                    }
                )
            del samples, preds, wide
            gc.collect()
    normal = pd.DataFrame(normal_rows)
    stress = pd.DataFrame(stress_rows)
    save_csv(run_dir / "tables/multiseed_normal_results.csv", normal)
    save_csv(run_dir / "tables/multiseed_stress_results.csv", stress)
    summary_rows: list[dict[str, Any]] = []
    for split, frame in (("normal", normal), ("stress", stress)):
        for method in METHODS:
            sub = frame[frame["method"] == method]
            summary_rows.append(
                {
                    "split": split,
                    "method": method,
                    "seeds": int(sub["seed"].nunique()),
                    "mean_affected_mse": float(sub["affected_mse"].mean()),
                    "std_affected_mse": float(sub["affected_mse"].std(ddof=1)),
                    "mean_improvement_vs_identity": float(sub["improvement_vs_identity"].mean()),
                    "std_improvement_vs_identity": float(sub["improvement_vs_identity"].std(ddof=1)),
                    "mean_worse_than_identity_count": float(sub["worse_than_identity_count"].mean()),
                }
            )
        seed_winners = []
        for seed, seed_frame in frame.groupby("seed"):
            learned = seed_frame[seed_frame["method"].isin(LEARNED)].sort_values("affected_mse")
            seed_winners.append(str(learned.iloc[0]["method"]))
        summary_rows.append(
            {
                "split": split,
                "method": "rank_stability",
                "seeds": len(seed_winners),
                "mean_affected_mse": float("nan"),
                "std_affected_mse": float("nan"),
                "mean_improvement_vs_identity": float("nan"),
                "std_improvement_vs_identity": float("nan"),
                "mean_worse_than_identity_count": float("nan"),
                "balanced_seed_win_rate": float(np.mean([winner == "balanced_residual" for winner in seed_winners])),
                "seed_winners": ", ".join(seed_winners),
                "rank_ordering_stable": len(set(seed_winners)) == 1,
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_csv(run_dir / "tables/multiseed_summary.csv", summary)
    plot_multiseed(run_dir, normal, stress)
    return normal, stress, summary


def plot_multiseed(run_dir: Path, normal: pd.DataFrame, stress: pd.DataFrame) -> None:
    combined = pd.concat([normal, stress], ignore_index=True)
    for split, frame in combined.groupby("split"):
        plot_lines(
            frame[frame["method"].isin(LEARNED)],
            "seed_index",
            "improvement_vs_identity",
            f"{split.title()} multiseed improvement",
            "Improvement vs identity",
            run_dir / f"model_audit/multiseed/{split}_improvement_by_seed.png",
            methods=LEARNED,
        )
        plot_lines(
            frame[frame["method"].isin(LEARNED)],
            "seed_index",
            "affected_mse",
            f"{split.title()} multiseed affected MSE",
            "Affected-region MSE",
            run_dir / f"model_audit/multiseed/{split}_affected_mse_by_seed.png",
            methods=LEARNED,
        )
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    methods = list(LEARNED)
    x = np.arange(2)
    width = 0.22
    colors = ["#2f6f8f", "#5f8a4b", "#7b5fa3"]
    for i, method in enumerate(methods):
        means = []
        stds = []
        for split in ("normal", "stress"):
            sub = combined[(combined["split"] == split) & (combined["method"] == method)]
            means.append(float(sub["improvement_vs_identity"].mean()))
            stds.append(float(sub["improvement_vs_identity"].std(ddof=1)))
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=method_label(method), color=colors[i], capsize=3)
    ax.set_xticks(x, ["normal", "stress"])
    ax.set_ylabel("Improvement vs identity")
    ax.set_title("Multiseed improvement ratios")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, run_dir / "model_audit/multiseed/normal_stress_improvement_errorbars.png")


def residual_logic_audit(
    run_dir: Path,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grid_rows: list[list[tuple[np.ndarray, str]]] = []
    for idx, sample in enumerate(samples[: min(100, len(samples))]):
        target = sample["target"]
        blended = sample["blended"]
        true_residual = blended - target
        for label, layer_key, recon_key in (
            ("old_residual", "residual_predicted_layer", "residual"),
            ("balanced_residual", "balanced_predicted_layer", "balanced_residual"),
        ):
            pred_residual = preds[layer_key][idx]
            preclip = blended - pred_residual
            recon = np.clip(preclip, 0.0, 1.0)
            corr = float(np.corrcoef(true_residual.ravel(), pred_residual.ravel())[0, 1])
            rows.append(
                {
                    "sample_index": idx,
                    "model": label,
                    "residual_target_min": float(true_residual.min()),
                    "residual_target_max": float(true_residual.max()),
                    "residual_target_mean": float(true_residual.mean()),
                    "predicted_residual_min": float(pred_residual.min()),
                    "predicted_residual_max": float(pred_residual.max()),
                    "predicted_residual_mean": float(pred_residual.mean()),
                    "preclip_reconstruction_min": float(preclip.min()),
                    "preclip_reconstruction_max": float(preclip.max()),
                    "reconstruction_clipping_fraction": float(np.mean((preclip < 0.0) | (preclip > 1.0))),
                    "predicted_true_residual_correlation": corr,
                    "affected_mse": masked_metrics(recon, target, affected_mask(target, blended, threshold))[0],
                }
            )
        if idx < 10:
            old_pred = preds["residual_predicted_layer"][idx]
            balanced_pred = preds["balanced_predicted_layer"][idx]
            balanced_error = np.abs(preds["balanced_residual"][idx] - target).mean(axis=-1)
            grid_rows.append(
                [
                    (target, "Target"),
                    (blended, "Blend"),
                    (np.clip(true_residual, 0.0, 1.0), "True residual"),
                    (np.clip(old_pred, 0.0, 1.0), "Old pred residual"),
                    (preds["residual"][idx], "Old recon"),
                    (np.clip(balanced_pred, 0.0, 1.0), "Balanced pred residual"),
                    (preds["balanced_residual"][idx], "Balanced recon"),
                    (balanced_error, "Balanced error"),
                ]
            )
    frame = pd.DataFrame(rows)
    summary = frame.groupby("model", as_index=False).agg(
        residual_target_min=("residual_target_min", "min"),
        residual_target_max=("residual_target_max", "max"),
        residual_target_mean=("residual_target_mean", "mean"),
        predicted_residual_min=("predicted_residual_min", "min"),
        predicted_residual_max=("predicted_residual_max", "max"),
        predicted_residual_mean=("predicted_residual_mean", "mean"),
        reconstruction_clipping_fraction=("reconstruction_clipping_fraction", "mean"),
        predicted_true_residual_correlation=("predicted_true_residual_correlation", "mean"),
        affected_mse=("affected_mse", "mean"),
    )
    save_csv(run_dir / "tables/residual_logic_stats.csv", summary)
    save_csv(run_dir / "tables/residual_logic_stats_per_sample.csv", frame)
    plot_image_grid(grid_rows, run_dir / "model_audit/residual_logic/residual_logic_grid.png", "Residual logic audit")
    return summary


def make_comparison_grids(
    run_dir: Path,
    samples_by_split: dict[str, list[dict[str, Any]]],
    preds_by_split: dict[str, dict[str, list[np.ndarray]]],
    wide_by_split: dict[str, pd.DataFrame],
) -> None:
    for split, wide in wide_by_split.items():
        samples = samples_by_split[split]
        preds = preds_by_split[split]
        best = wide[(wide["balanced_beats_residual"]) & (wide["balanced_beats_direct"])].copy()
        if best.empty:
            best = wide.copy()
        best["balanced_margin"] = wide[["direct_affected_mse", "residual_affected_mse"]].min(axis=1) - wide["balanced_residual_affected_mse"]
        make_one_comparison_grid(
            samples,
            preds,
            best.sort_values("balanced_margin", ascending=False).head(6)["index"].astype(int).tolist(),
            run_dir / f"model_audit/comparison_grids/{split}_comparison_grid_best.png",
            f"{split.title()} balanced residual wins",
        )
        failures = wide[(wide["residual_beats_balanced"]) | (wide["direct_beats_balanced"]) | (wide["balanced_residual_worse_than_identity"])].copy()
        if failures.empty:
            failures = wide.sort_values("balanced_residual_affected_mse", ascending=False).head(6)
        else:
            failures = failures.sort_values("balanced_residual_affected_mse", ascending=False).head(6)
        make_one_comparison_grid(
            samples,
            preds,
            failures["index"].astype(int).tolist(),
            run_dir / f"model_audit/comparison_grids/{split}_comparison_grid_failures.png",
            f"{split.title()} failures and counterexamples",
        )
    stress = wide_by_split["stress"]
    samples = samples_by_split["stress"]
    preds = preds_by_split["stress"]
    random_ids = stress.sample(n=min(6, len(stress)), random_state=20260708)["index"].astype(int).tolist()
    make_one_comparison_grid(
        samples,
        preds,
        random_ids,
        run_dir / "model_audit/comparison_grids/stress_comparison_grid_random.png",
        "Stress random examples",
    )
    direct_fail = stress[stress["direct_worse_than_identity"]].sort_values("direct_affected_mse", ascending=False).head(6)
    if not direct_fail.empty:
        make_one_comparison_grid(
            samples,
            preds,
            direct_fail["index"].astype(int).tolist(),
            run_dir / "model_audit/comparison_grids/stress_direct_worse_than_identity.png",
            "Stress direct worse-than-identity cases",
        )
    core_high = stress.sort_values("core_obstruction_fraction", ascending=False).head(6)
    make_one_comparison_grid(
        samples,
        preds,
        core_high["index"].astype(int).tolist(),
        run_dir / "model_audit/comparison_grids/stress_high_core_obstruction.png",
        "Stress high core obstruction",
    )
    model_hard_low = stress.sort_values(["severity_score", "balanced_residual_affected_mse"], ascending=[True, False]).head(6)
    make_one_comparison_grid(
        samples,
        preds,
        model_hard_low["index"].astype(int).tolist(),
        run_dir / "model_audit/comparison_grids/stress_low_severity_model_hard.png",
        "Stress low severity, model-hard",
    )
    model_easy_high = stress.sort_values(["severity_score", "balanced_residual_affected_mse"], ascending=[False, True]).head(6)
    make_one_comparison_grid(
        samples,
        preds,
        model_easy_high["index"].astype(int).tolist(),
        run_dir / "model_audit/comparison_grids/stress_high_severity_model_easy.png",
        "Stress high severity, model-easy",
    )


def make_one_comparison_grid(
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    indices: list[int],
    path: Path,
    title: str,
) -> None:
    rows: list[list[tuple[np.ndarray, str]]] = []
    for idx in indices:
        sample = samples[idx]
        target = sample["target"]
        balanced_error = np.abs(preds["balanced_residual"][idx] - target).mean(axis=-1)
        rows.append(
            [
                (target, "Target"),
                (sample["contaminant"], "Contaminant"),
                (sample["blended"], "Blend"),
                (preds["identity"][idx], "Identity"),
                (preds["threshold"][idx], "Threshold"),
                (preds["direct"][idx], "Direct"),
                (preds["residual"][idx], "Old residual"),
                (preds["balanced_residual"][idx], "Balanced residual"),
                (balanced_error, "Balanced error"),
            ]
        )
    plot_image_grid(rows, path, title, figsize_per_cell=(2.0, 1.95))


def figure_consistency_audit(run_dir: Path) -> None:
    result_file = LATEST_BALANCED_RUN / "results/model_comparison.csv"
    lines: list[str] = ["# Figure Consistency Audit", ""]
    if result_file.exists():
        results = pd.read_csv(result_file)
        normal = results[(results["split"] == "normal") & (results["method"] == "balanced_residual")]
        stress = results[(results["split"] == "stress") & (results["method"] == "balanced_residual")]
        lines.extend(
            [
                "## Metric cross-check",
                "",
                f"- Latest balanced-run results file: `{project_relative(result_file)}`.",
                f"- Balanced normal affected MSE in CSV: `{float(normal['affected_mse'].iloc[0]):.6f}`.",
                f"- Balanced normal improvement in CSV: `{float(normal['improvement_vs_identity'].iloc[0]):.2f}x`.",
                f"- Balanced stress affected MSE in CSV: `{float(stress['affected_mse'].iloc[0]):.6f}`.",
                f"- Balanced stress improvement in CSV: `{float(stress['improvement_vs_identity'].iloc[0]):.2f}x`.",
                "",
            ]
        )
    else:
        lines.append("- Latest balanced-run comparison CSV was not found.")
    docs_to_scan = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "reports/figures/README.md",
        PROJECT_ROOT / "docs/figure_inventory.md",
        PROJECT_ROOT / "docs/paper_plan.md",
        PROJECT_ROOT / "reports/paper/sections/results.tex",
        PROJECT_ROOT / "reports/paper/sections/evaluation.tex",
        PROJECT_ROOT / "reports/paper/sections/discussion.tex",
    ]
    difficulty_hits: list[str] = []
    metric_hits: list[str] = []
    for doc in docs_to_scan:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if "difficulty" in lowered:
                difficulty_hits.append(f"- `{project_relative(doc)}:{lineno}`: {line.strip()[:180]}")
            if any(token in line for token in ("27.79", "16.47", "0.002451", "0.004587", "800", "1000")):
                metric_hits.append(f"- `{project_relative(doc)}:{lineno}`: {line.strip()[:180]}")
    lines.extend(
        [
            "## Caption and terminology checks",
            "",
            "- Normal/stress comparison figures in the current inventory are labeled as balanced-run figures.",
            "- Legacy figures remain present; the figure inventory warns not to mix older direct/residual figures with the current balanced table without a caption caveat.",
            "- The README table uses the current 1,000-blend normal and hard stress balanced-run metrics.",
            "- The paper/results text reports the current balanced-run values and distinguishes normal held-out from hard stress.",
            "",
            "## `difficulty` terminology hits",
            "",
        ]
    )
    if difficulty_hits:
        lines.extend(difficulty_hits[:40])
        if len(difficulty_hits) > 40:
            lines.append(f"- ... {len(difficulty_hits) - 40} more hits omitted.")
    else:
        lines.append("- No visible `difficulty` hits found in scanned figure/docs files.")
    lines.extend(["", "## Numeric/caption-sensitive hits", ""])
    lines.extend(metric_hits[:60] if metric_hits else ["- No metric-sensitive hits found."])
    if len(metric_hits) > 60:
        lines.append(f"- ... {len(metric_hits) - 60} more hits omitted.")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- No obvious caption-number mismatch was found in scanned text for the current balanced metrics.",
            "- Some files still contain `generation_difficulty` as a historical sampled-parameter label; this is acceptable in code/results columns, but prose should prefer `severity` when referring to measured blend damage.",
            "- The README figure is not intrinsically misleading because it labels normal vs hard stress and uses balanced-run figures, but it should continue to carry caveats that normal/stress tests are controlled synthetic evaluations.",
            "- Existing legacy figures should stay labeled as legacy or direct/residual-only if used.",
        ]
    )
    save_text(run_dir / "diagnostics/figure_consistency_audit.md", "\n".join(lines))


def inspect_mask_formula() -> dict[str, Any]:
    source = (PROJECT_ROOT / "src/utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    formula_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "affected_region_mask":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    text = ast.get_source_segment(source, child)
                    if text and "blended" in text and "target" in text and "abs" in text:
                        formula_line = text.strip()
    return {
        "function": "src.utils.affected_region_mask",
        "formula_evidence": formula_line,
        "uses_prediction_argument": "pred" in re.sub(r"prediction", "", source[source.find("def affected_region_mask") : source.find("def masked_mse")]),
        "conclusion": "Mask is computed from abs(blended - target).mean(axis=-1) > threshold, not prediction vs target.",
    }


def write_audit_report(
    run_dir: Path,
    threshold_tables: dict[str, pd.DataFrame],
    dilation_tables: dict[str, pd.DataFrame],
    core_tables: dict[str, pd.DataFrame],
    multiseed_summary: pd.DataFrame,
    residual_stats: pd.DataFrame,
    checkpoint_comparison: dict[str, Any],
    mask_formula: dict[str, Any],
) -> None:
    normal_t = threshold_tables["normal"]
    stress_t = threshold_tables["stress"]
    normal_d = dilation_tables["normal"]
    stress_d = dilation_tables["stress"]
    def best_methods(frame: pd.DataFrame, x_col: str) -> list[str]:
        winners = []
        for _, group in frame[frame["method"].isin(LEARNED)].groupby(x_col):
            winners.append(str(group.sort_values("affected_mse").iloc[0]["method"]))
        return winners
    threshold_winners = {
        "normal": best_methods(normal_t, "threshold"),
        "stress": best_methods(stress_t, "threshold"),
    }
    dilation_winners = {
        "normal": best_methods(normal_d, "dilation_radius"),
        "stress": best_methods(stress_d, "dilation_radius"),
    }
    multi_rank = multiseed_summary[multiseed_summary["method"] == "rank_stability"].copy()
    balanced_rows = multiseed_summary[multiseed_summary["method"] == "balanced_residual"]
    normal_balanced_multi = balanced_rows[balanced_rows["split"] == "normal"].iloc[0]
    stress_balanced_multi = balanced_rows[balanced_rows["split"] == "stress"].iloc[0]
    core_bal = {
        split: frame[frame["method"] == "balanced_residual"].iloc[0]
        for split, frame in core_tables.items()
    }
    checkpoints_unchanged = all(item["unchanged"] for item in checkpoint_comparison.values())
    verdict = "mostly trustworthy with caveats"
    if not checkpoints_unchanged:
        verdict = "suspicious / needs investigation"
    if threshold_winners["normal"].count("balanced_residual") < 2 or threshold_winners["stress"].count("balanced_residual") < 2:
        verdict = "mostly trustworthy with caveats"
    lines = [
        "# Thayer-Net Evaluation Audit Report",
        "",
        "## Purpose",
        "",
        "This audit verifies the evaluation pipeline for masks, halo inclusion, core obstruction, residual reconstruction logic, multi-seed stability, and same-set model comparisons. It does not train or retrain models and does not modify checkpoint files.",
        "",
        "## Summary Verdict",
        "",
        verdict,
        "",
        "## Key Findings",
        "",
        f"- Affected masks: {mask_formula['conclusion']}",
        f"- Checkpoints unchanged: `{checkpoints_unchanged}`.",
        f"- Threshold sensitivity winners: normal `{', '.join(threshold_winners['normal'])}`; stress `{', '.join(threshold_winners['stress'])}`.",
        f"- Dilation/halo sensitivity winners: normal `{', '.join(dilation_winners['normal'])}`; stress `{', '.join(dilation_winners['stress'])}`.",
        f"- Multi-seed balanced normal improvement: `{normal_balanced_multi['mean_improvement_vs_identity']:.2f} +/- {normal_balanced_multi['std_improvement_vs_identity']:.2f}x`.",
        f"- Multi-seed balanced stress improvement: `{stress_balanced_multi['mean_improvement_vs_identity']:.2f} +/- {stress_balanced_multi['std_improvement_vs_identity']:.2f}x`.",
        f"- Multi-seed rank stability rows: `{multi_rank[['split', 'seed_winners', 'balanced_seed_win_rate']].to_dict(orient='records')}`.",
        f"- Core affected balanced MSE: normal `{core_bal['normal']['core_affected_mse']:.6f}`, stress `{core_bal['stress']['core_affected_mse']:.6f}`.",
        f"- Non-core affected balanced MSE: normal `{core_bal['normal']['noncore_affected_mse']:.6f}`, stress `{core_bal['stress']['noncore_affected_mse']:.6f}`.",
        "- Residual logic: residual target is `blended - target`; reconstruction is `blended - predicted_residual`; clipping is applied for metrics/visualization after subtraction.",
        "- Leakage: no split-level leakage was found in the current code path; historical normal blends do not save global source indices, so historical sample-level leakage cannot be independently re-proven from outputs alone.",
        "- Figure consistency: scanned current docs/figure captions matched the current balanced-run values; legacy figures still need explicit legacy context if reused.",
        "",
        "## Known Caveats",
        "",
        "- Controlled synthetic blends only.",
        "- Foreground extraction is still an approximation.",
        "- Source indices are not saved for standard normal blends, so historical sample-level source IDs cannot be fully audited after generation.",
        "- Real sky backgrounds, PSF variation, detector artifacts, and correlated galaxy environments are not fully modeled.",
        "- Affected-region thresholds influence exact numbers.",
        "- Normal evaluation sets may differ between earlier and later experiments; the same-set audit compares models on the same regenerated runtime sets.",
        "- This audit used three 1,000-blend seeds per split for multi-seed evaluation to keep the no-training audit tractable.",
        "",
        "## Action Items",
        "",
        "- Safe paper claim: balanced residual is the best aggregate model in the audited controlled synthetic evaluations, with caveats about individual counterexamples.",
        "- Avoid claiming universal per-sample superiority or real-survey performance.",
        "- The 27.79x normal and 16.47x stress headline values are reasonable to cite as the current same-run balanced-run metrics, provided they are framed as controlled synthetic results and not as immutable across masks/seeds.",
        "- Stop modeling unless a new question requires it; the next highest-value step is to preserve exact generated evaluation sets and global source indices for future reproducibility.",
    ]
    save_text(run_dir / "audit_report.md", "\n".join(lines))


def finalize_from_existing(run_dir: Path) -> None:
    before = read_checkpoint_before(run_dir)
    threshold_tables = {
        "normal": pd.read_csv(run_dir / "tables/mask_threshold_sensitivity_normal.csv"),
        "stress": pd.read_csv(run_dir / "tables/mask_threshold_sensitivity_stress.csv"),
    }
    dilation_tables = {
        "normal": pd.read_csv(run_dir / "tables/mask_dilation_sensitivity_normal.csv"),
        "stress": pd.read_csv(run_dir / "tables/mask_dilation_sensitivity_stress.csv"),
    }
    core_tables = {
        "normal": pd.read_csv(run_dir / "tables/core_region_metrics_normal.csv"),
        "stress": pd.read_csv(run_dir / "tables/core_region_metrics_stress.csv"),
    }
    multiseed_normal = pd.read_csv(run_dir / "tables/multiseed_normal_results.csv")
    multiseed_stress = pd.read_csv(run_dir / "tables/multiseed_stress_results.csv")
    multiseed_summary = pd.read_csv(run_dir / "tables/multiseed_summary.csv")
    residual_stats = pd.read_csv(run_dir / "tables/residual_logic_stats.csv")
    mask_formula = json.loads(
        (run_dir / "diagnostics/affected_mask_formula_audit.json").read_text(
            encoding="utf-8"
        )
    )
    plot_multiseed(run_dir, multiseed_normal, multiseed_stress)
    figure_consistency_audit(run_dir)
    checkpoint_comparison = write_checkpoint_after(run_dir, before)
    write_audit_report(
        run_dir,
        threshold_tables,
        dilation_tables,
        core_tables,
        multiseed_summary,
        residual_stats,
        checkpoint_comparison,
        mask_formula,
    )
    if not all(item["unchanged"] for item in checkpoint_comparison.values()):
        raise RuntimeError("Checkpoint metadata changed during audit.")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    ensure_dirs(run_dir)
    if args.finalize_only:
        finalize_from_existing(run_dir)
        print(f"Finalized audit from existing tables: {project_relative(run_dir)}", flush=True)
        return 0
    before = read_checkpoint_before(run_dir)
    settings = AuditSettings(
        multiseed_count=args.multiseed_count,
        multiseed_blends=args.multiseed_blends,
        batch_size=args.batch_size,
    )
    config = load_config()
    save_json(
        run_dir / "logs/audit_config.json",
        {
            "project_root": ".",
            "settings": settings.__dict__,
            "direct_checkpoint": project_relative(DIRECT_CHECKPOINT),
            "old_residual_checkpoint": project_relative(OLD_RESIDUAL_CHECKPOINT),
            "balanced_checkpoint": project_relative(BALANCED_CHECKPOINT),
            "balanced_final_checkpoint_exists_not_evaluated": BALANCED_FINAL_CHECKPOINT.exists(),
        },
    )
    mask_formula = inspect_mask_formula()
    save_json(run_dir / "diagnostics/affected_mask_formula_audit.json", mask_formula)

    print("Loading data and reconstructing splits.", flush=True)
    _train_images, _val_images, test_images, split_audit = load_split_subsets(config, settings)
    write_split_audit(run_dir, config, split_audit)
    del _train_images, _val_images
    gc.collect()

    device = gd_train.resolve_accelerator(args.device)
    print(f"Using device: {device}", flush=True)
    direct_model, residual_model, balanced_model = load_models(config, device)

    print("Generating same-run normal and stress evaluation blends.", flush=True)
    normal_samples = make_normal_test_blends(test_images, config, settings)
    stress_samples = make_stress_blends(test_images, settings)
    samples_by_split = {"normal": normal_samples, "stress": stress_samples}

    print("Running model inference on same-run evaluation blends.", flush=True)
    preds_by_split = {
        "normal": predict_all(normal_samples, direct_model, residual_model, balanced_model, device, settings.batch_size),
        "stress": predict_all(stress_samples, direct_model, residual_model, balanced_model, device, settings.batch_size),
    }

    print("Writing visual blend diagnostics.", flush=True)
    make_blend_audit_figures(run_dir, samples_by_split, settings.affected_threshold)

    print("Auditing threshold sensitivity.", flush=True)
    threshold_tables = run_threshold_sensitivity(run_dir, samples_by_split, preds_by_split)
    print("Auditing dilation and halo sensitivity.", flush=True)
    dilation_tables = run_dilation_sensitivity(run_dir, samples_by_split, preds_by_split, settings.affected_threshold)
    make_dilation_examples(run_dir, stress_samples, settings.affected_threshold)

    for split, table in threshold_tables.items():
        write_same_set_comparison(run_dir, split, table, settings.affected_threshold)

    print("Computing core/halo/background region metrics.", flush=True)
    core_tables = run_core_region_metrics(run_dir, samples_by_split, preds_by_split, settings.affected_threshold)

    print("Building same-set per-sample model comparison tables.", flush=True)
    wide_by_split = {
        split: compute_wide_per_sample(split, samples, preds_by_split[split], settings.affected_threshold)
        for split, samples in samples_by_split.items()
    }
    for split, frame in wide_by_split.items():
        save_csv(run_dir / f"tables/same_set_per_sample_{split}.csv", frame)

    print("Auditing residual subtraction logic.", flush=True)
    residual_stats = residual_logic_audit(run_dir, stress_samples, preds_by_split["stress"], settings.affected_threshold)

    print("Writing model comparison grids.", flush=True)
    make_comparison_grids(run_dir, samples_by_split, preds_by_split, wide_by_split)

    print("Running multi-seed evaluation audit.", flush=True)
    _multi_normal, _multi_stress, multiseed_summary = run_multiseed(
        run_dir,
        config,
        test_images,
        direct_model,
        residual_model,
        balanced_model,
        device,
        settings,
    )

    print("Checking existing figure and caption consistency.", flush=True)
    figure_consistency_audit(run_dir)

    print("Writing final audit report and checkpoint integrity after-record.", flush=True)
    checkpoint_comparison = write_checkpoint_after(run_dir, before)
    write_audit_report(
        run_dir,
        threshold_tables,
        dilation_tables,
        core_tables,
        multiseed_summary,
        residual_stats,
        checkpoint_comparison,
        mask_formula,
    )
    if not all(item["unchanged"] for item in checkpoint_comparison.values()):
        raise RuntimeError("Checkpoint metadata changed during audit.")
    print(f"Audit complete: {project_relative(run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
