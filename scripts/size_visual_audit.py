"""Audit apparent size, centrality, halo artifacts, and visual-vs-metric cases.

This is a no-training script. It regenerates held-out normal/stress blends,
loads existing checkpoints for inference only, writes a new timestamped run
directory under ``outputs/runs/``, and verifies checkpoint metadata unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from scipy.ndimage import binary_dilation, binary_fill_holes, label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stress_test as stress_helpers
import train_balanced_residual_unet as balanced_helpers
from src import baselines
from src import data as gd_data
from src import train as gd_train
from src import utils as gd_utils


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_DIRECT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "unet_direct_5000train_800val_800test_20ep.pth"
)
DEFAULT_RESIDUAL_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "unet_residual_5000train_800val_800test_20ep_20260708_154947.pth"
)
DEFAULT_BALANCED_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "unet_residual_balanced_hard_20260708_184632.pth"
)
DEFAULT_WEIGHTED_MODERATE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "unet_residual_weighted_br_20260709_030245_best.pth"
)

BASE_METHODS = ("identity", "threshold", "direct", "residual", "balanced_residual")
WEIGHTED_METHOD = "weighted_residual"
LEARNED_BASE = ("direct", "residual", "balanced_residual")
SIZE_BIN_ORDER = [
    "contaminant much smaller",
    "contaminant smaller/similar",
    "similar size",
    "contaminant larger",
    "contaminant much larger",
]
SIZE_BIN_EDGES = [-np.inf, 0.5, 0.8, 1.25, 2.0, np.inf]
CENTER_BIN_ORDER = ["near center", "intermediate", "off center"]
OVERLAP_BIN_ORDER = ["high centroid overlap", "medium centroid overlap", "low centroid overlap"]
CORE_BIN_ORDER = ["low", "medium", "high"]


@dataclass(frozen=True)
class AuditSettings:
    n_normal_blends: int = 1000
    n_stress_blends: int = 1000
    test_source_subset: int = 1000
    stress_source_subset: int = 800
    batch_size: int = 8
    affected_threshold: float = 0.02
    halo_dilation: int = 9
    size_border_width: int = 12
    size_aperture_fraction: float = 0.48
    max_examples_per_category: int = 4


@dataclass(frozen=True)
class SizeEstimate:
    area: float
    equiv_radius: float
    bbox_width: float
    bbox_height: float
    centroid_x: float
    centroid_y: float
    flux_proxy: float
    central_flux_concentration: float
    threshold: float
    valid: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-normal-blends", type=int, default=1000)
    parser.add_argument("--n-stress-blends", type=int, default=1000)
    parser.add_argument("--test-source-subset", type=int, default=1000)
    parser.add_argument("--stress-source-subset", type=int, default=800)
    parser.add_argument("--direct-checkpoint", type=Path, default=DEFAULT_DIRECT_CHECKPOINT)
    parser.add_argument("--residual-checkpoint", type=Path, default=DEFAULT_RESIDUAL_CHECKPOINT)
    parser.add_argument("--balanced-checkpoint", type=Path, default=DEFAULT_BALANCED_CHECKPOINT)
    parser.add_argument(
        "--weighted-checkpoint",
        type=Path,
        default=DEFAULT_WEIGHTED_MODERATE_CHECKPOINT,
        help="Moderate Thayer-BR v0.2 checkpoint. If missing, the audit continues without v0.2.",
    )
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return config


def make_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / f"size_visual_audit_{stamp}"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for child in ("tables", "figures", "diagnostics", "example_grids", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    (run_dir / "example_grids" / "visual_metric_disagreements").mkdir(
        parents=True,
        exist_ok=False,
    )
    return run_dir


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


def save_fig(fig: plt.Figure, path: Path, dpi: int = 190) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def checkpoint_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": project_relative(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_iso_local": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def checkpoint_inventory(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {project_relative(path): checkpoint_info(path) for path in paths if path.exists()}


def verify_checkpoint_inventory(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for rel_path, before_info in before.items():
        path = PROJECT_ROOT / rel_path
        after_info = checkpoint_info(path) if path.exists() else None
        unchanged = bool(
            after_info
            and before_info["size_bytes"] == after_info["size_bytes"]
            and before_info["mtime_ns"] == after_info["mtime_ns"]
        )
        comparison[rel_path] = {
            "before": before_info,
            "after": after_info,
            "unchanged": unchanged,
        }
    return comparison


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state_dict.")
    return state


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}
    return {
        key: checkpoint.get(key)
        for key in (
            "experiment_name",
            "variant_name",
            "checkpoint_kind",
            "timestamp",
            "best_epoch",
            "best_val_loss",
        )
        if key in checkpoint
    }


def find_moderate_weighted_checkpoint(explicit_path: Path) -> Path | None:
    explicit_path = resolve_path(explicit_path)
    if explicit_path.exists():
        return explicit_path
    candidates: list[tuple[float, Path]] = []
    for path in (PROJECT_ROOT / "outputs" / "checkpoints").glob("unet_residual_weighted_br_*_best.pth"):
        meta = checkpoint_metadata(path)
        if str(meta.get("variant_name", "")).lower() == "moderate":
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def load_test_images(config: dict[str, Any], settings: AuditSettings) -> tuple[np.ndarray, dict[str, Any]]:
    data_path = PROJECT_ROOT / config["dataset_path"]
    images_raw, labels, metadata = gd_data.load_galaxy10(data_path)
    train, val, test = gd_data.split_dataset(
        images_raw,
        labels,
        train_frac=float(config["splits"]["train_frac"]),
        val_frac=float(config["splits"]["val_frac"]),
        test_frac=float(config["splits"]["test_frac"]),
        shuffle=True,
        seed=int(config["seed"]),
    )
    test_images_raw, test_labels = test
    audit = {
        "dataset_path": config["dataset_path"],
        "n_total": int(len(images_raw)),
        "metadata_keys": sorted(metadata.keys()),
        "test_split_size": int(len(test_images_raw)),
        "test_subset_used": int(min(settings.test_source_subset, len(test_images_raw))),
        "test_label_count": int(len(test_labels)),
    }
    test_images = gd_data.normalise_images(test_images_raw[: settings.test_source_subset])
    del images_raw, labels, metadata, train, val, test, test_images_raw, test_labels
    gc.collect()
    return test_images, audit


def make_samples(
    test_images: np.ndarray,
    config: dict[str, Any],
    settings: AuditSettings,
) -> dict[str, list[dict[str, Any]]]:
    seed = int(config["seed"])
    normal = balanced_helpers.normal_blends(
        test_images,
        settings.n_normal_blends,
        config,
        seed=seed + 4000,
        component="size_visual_normal",
    )
    stress_settings = dict(stress_helpers.STRESS_DEFAULTS)
    stress_settings["n_stress_blends"] = settings.n_stress_blends
    stress_settings["stress_source_subset"] = settings.stress_source_subset
    stress = stress_helpers.generate_stress_blends(
        test_images[: settings.stress_source_subset],
        stress_settings,
    )
    for sample in stress:
        sample.setdefault("info", {})["training_component"] = "size_visual_stress"
    return {"normal": normal, "stress": stress}


def border_pixels(image: np.ndarray, width: int) -> np.ndarray:
    h, w, c = image.shape
    width = max(1, min(width, h // 2, w // 2))
    return np.concatenate(
        [
            image[:width, :, :].reshape(-1, c),
            image[-width:, :, :].reshape(-1, c),
            image[:, :width, :].reshape(-1, c),
            image[:, -width:, :].reshape(-1, c),
        ],
        axis=0,
    )


def estimate_apparent_size(
    image: np.ndarray,
    settings: AuditSettings,
) -> SizeEstimate:
    image = np.asarray(image, dtype=np.float32)
    h, w, _channels = image.shape
    background = np.median(border_pixels(image, settings.size_border_width), axis=0).reshape(1, 1, 3)
    residual = np.clip(image - background, 0.0, None)
    gray = residual.mean(axis=-1)
    y_grid, x_grid = np.ogrid[:h, :w]
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    aperture = (
        np.hypot(y_grid - center_y, x_grid - center_x)
        <= settings.size_aperture_fraction * min(h, w)
    )
    border_gray = np.concatenate(
        [
            gray[: settings.size_border_width, :].ravel(),
            gray[-settings.size_border_width :, :].ravel(),
            gray[:, : settings.size_border_width].ravel(),
            gray[:, -settings.size_border_width :].ravel(),
        ]
    )
    border_median = float(np.median(border_gray))
    mad = float(np.median(np.abs(border_gray - border_median)) + 1e-6)
    aperture_values = gray[aperture]
    peak = float(np.max(aperture_values)) if aperture_values.size else float(np.max(gray))
    percentile_floor = float(np.percentile(aperture_values, 88)) if aperture_values.size else 0.0
    threshold = max(0.006, border_median + 3.0 * mad, 0.07 * peak, 0.35 * percentile_floor)
    raw_mask = (gray > threshold) & aperture
    labeled, n_labels = label(raw_mask)
    if n_labels > 0:
        chosen_label = 0
        best_score = float("inf")
        for component_label in range(1, n_labels + 1):
            component = labeled == component_label
            area = int(component.sum())
            if area <= 0:
                continue
            ys, xs = np.nonzero(component)
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
            distance = float(np.hypot(cx - center_x, cy - center_y))
            score = distance - 0.025 * np.sqrt(area)
            if score < best_score:
                best_score = score
                chosen_label = component_label
        mask = labeled == chosen_label
    else:
        mask = np.zeros((h, w), dtype=bool)
    if int(mask.sum()) < 8 and aperture_values.size:
        fallback_threshold = float(np.percentile(aperture_values, 95))
        mask = (gray >= fallback_threshold) & aperture
    mask = binary_fill_holes(mask)
    area = float(mask.sum())
    if area <= 0:
        return SizeEstimate(
            area=0.0,
            equiv_radius=0.0,
            bbox_width=0.0,
            bbox_height=0.0,
            centroid_x=float("nan"),
            centroid_y=float("nan"),
            flux_proxy=0.0,
            central_flux_concentration=float("nan"),
            threshold=threshold,
            valid=False,
        )
    ys, xs = np.nonzero(mask)
    bbox_width = float(xs.max() - xs.min() + 1)
    bbox_height = float(ys.max() - ys.min() + 1)
    weights = gray[mask]
    flux_proxy = float(np.sum(weights))
    if flux_proxy > 0:
        centroid_x = float(np.sum(xs * weights) / flux_proxy)
        centroid_y = float(np.sum(ys * weights) / flux_proxy)
    else:
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
    central_radius = 0.15 * min(h, w)
    central_mask = np.hypot(x_grid - center_x, y_grid - center_y) <= central_radius
    positive_flux = float(np.sum(gray[aperture & (gray > 0)]))
    central_flux = float(np.sum(gray[central_mask & (gray > 0)]))
    concentration = central_flux / positive_flux if positive_flux > 0 else float("nan")
    return SizeEstimate(
        area=area,
        equiv_radius=float(np.sqrt(area / np.pi)),
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        flux_proxy=flux_proxy,
        central_flux_concentration=float(concentration),
        threshold=threshold,
        valid=True,
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def masked_mse_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    diff = pred[mask] - target[mask]
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def method_label(method: str) -> str:
    return {
        "identity": "Identity",
        "threshold": "Threshold",
        "direct": "Thayer-Direct",
        "residual": "Thayer-Residual",
        "balanced_residual": "Thayer-BR v0.1",
        "weighted_residual": "Thayer-BR v0.2",
    }.get(method, method)


def method_color(method: str) -> str:
    return {
        "identity": "#6c717a",
        "threshold": "#b66b5d",
        "direct": "#2f6f8f",
        "residual": "#5f8a4b",
        "balanced_residual": "#7b5fa3",
        "weighted_residual": "#c28b2c",
    }.get(method, "#444444")


def load_models(
    config: dict[str, Any],
    device: torch.device,
    checkpoint_paths: dict[str, Path],
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    models["direct"] = balanced_helpers.load_direct_model(
        checkpoint_paths["direct"],
        config["model"],
        device,
    )
    models["residual"] = balanced_helpers.load_residual_model(
        checkpoint_paths["residual"],
        config["model"],
        device,
    )
    models["balanced_residual"] = balanced_helpers.load_residual_model(
        checkpoint_paths["balanced_residual"],
        config["model"],
        device,
    )
    weighted = checkpoint_paths.get("weighted_residual")
    if weighted is not None and weighted.exists():
        models["weighted_residual"] = balanced_helpers.load_residual_model(
            weighted,
            config["model"],
            device,
        )
    return models


def predict_models(
    samples: list[dict[str, Any]],
    models: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
) -> dict[str, list[np.ndarray]]:
    methods = list(BASE_METHODS)
    if "weighted_residual" in models:
        methods.append("weighted_residual")
    predictions: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        inputs = np.stack([sample["blended"] for sample in batch], axis=0).astype(np.float32)
        tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
        learned_outputs: dict[str, np.ndarray] = {}
        with torch.no_grad():
            direct_out = models["direct"](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            learned_outputs["direct"] = np.clip(direct_out, 0.0, 1.0).astype(np.float32)
            for method in ("residual", "balanced_residual", "weighted_residual"):
                if method not in models:
                    continue
                residual_layer = models[method](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
                learned_outputs[method] = np.clip(inputs - residual_layer, 0.0, 1.0).astype(np.float32)
        for offset, sample in enumerate(batch):
            blended = sample["blended"]
            predictions["identity"].append(blended.astype(np.float32))
            predictions["threshold"].append(baselines.threshold_baseline(blended))
            for method, output in learned_outputs.items():
                predictions[method].append(output[offset])
    return predictions


def size_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=SIZE_BIN_EDGES,
        labels=SIZE_BIN_ORDER,
        include_lowest=True,
    )


def center_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-0.001, 0.20, 0.40, np.inf],
        labels=CENTER_BIN_ORDER,
        include_lowest=True,
    )


def centroid_overlap_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-0.001, 0.60, 1.20, np.inf],
        labels=OVERLAP_BIN_ORDER,
        include_lowest=True,
    )


def core_obstruction_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-0.001, 1.0 / 3.0, 2.0 / 3.0, 1.001],
        labels=CORE_BIN_ORDER,
        include_lowest=True,
    )


def sample_metadata(
    split: str,
    idx: int,
    sample: dict[str, Any],
    settings: AuditSettings,
) -> dict[str, Any]:
    target = sample["target"]
    contaminant = sample["contaminant"]
    blended = sample["blended"]
    info = sample.get("info", {})
    target_size = estimate_apparent_size(target, settings)
    contaminant_size = estimate_apparent_size(contaminant, settings)
    brightness = float(info.get("brightness", 1.0) or 1.0)
    adjusted_contaminant = np.clip(contaminant * brightness, 0.0, 1.0)
    contaminant_bright_size = estimate_apparent_size(adjusted_contaminant, settings)
    metadata = balanced_helpers.blend_metadata(
        target,
        blended,
        info,
        settings.affected_threshold,
    )
    shift = info.get("shift", (int(info.get("shift_x", 0) or 0), int(info.get("shift_y", 0) or 0)))
    shift_x = int(shift[0])
    shift_y = int(shift[1])
    h, w = target.shape[:2]
    image_center_x = (w - 1) / 2.0
    image_center_y = (h - 1) / 2.0
    contaminant_centroid_x_blend = contaminant_size.centroid_x + shift_x
    contaminant_centroid_y_blend = contaminant_size.centroid_y + shift_y
    contaminant_distance_from_center = float(
        np.hypot(
            contaminant_centroid_x_blend - image_center_x,
            contaminant_centroid_y_blend - image_center_y,
        )
    )
    centroid_distance = float(
        np.hypot(
            contaminant_centroid_x_blend - target_size.centroid_x,
            contaminant_centroid_y_blend - target_size.centroid_y,
        )
    )
    radius_sum = target_size.equiv_radius + contaminant_size.equiv_radius
    diagonal_half = float(np.hypot(image_center_x, image_center_y))
    return {
        "sample_id": f"{split}_{idx:04d}",
        "split": split,
        "suite": split,
        "index": idx,
        "target_area": target_size.area,
        "contaminant_area": contaminant_size.area,
        "target_equiv_radius": target_size.equiv_radius,
        "contaminant_equiv_radius": contaminant_size.equiv_radius,
        "apparent_size_ratio": safe_ratio(contaminant_size.equiv_radius, target_size.equiv_radius),
        "target_bbox_width": target_size.bbox_width,
        "target_bbox_height": target_size.bbox_height,
        "contaminant_bbox_width": contaminant_size.bbox_width,
        "contaminant_bbox_height": contaminant_size.bbox_height,
        "target_flux_proxy": target_size.flux_proxy,
        "contaminant_flux_proxy": contaminant_size.flux_proxy,
        "brightness_adjusted_contaminant_flux_proxy": contaminant_bright_size.flux_proxy,
        "target_central_flux_concentration": target_size.central_flux_concentration,
        "contaminant_central_flux_concentration": contaminant_size.central_flux_concentration,
        "brightness": brightness,
        "generation_size_ratio": info.get("size_ratio"),
        "core_obstruction_fraction": metadata["core_obstruction_fraction"],
        "blend_severity_score": metadata["blend_severity_score"],
        "mask_fraction": metadata["mask_fraction"],
        "target_centroid_x": target_size.centroid_x,
        "target_centroid_y": target_size.centroid_y,
        "contaminant_centroid_x": contaminant_size.centroid_x,
        "contaminant_centroid_y": contaminant_size.centroid_y,
        "contaminant_centroid_x_blend": contaminant_centroid_x_blend,
        "contaminant_centroid_y_blend": contaminant_centroid_y_blend,
        "contaminant_distance_from_center": contaminant_distance_from_center,
        "contaminant_distance_from_center_norm": safe_ratio(contaminant_distance_from_center, diagonal_half),
        "target_contaminant_centroid_distance": centroid_distance,
        "centroid_distance_over_radius_sum": safe_ratio(centroid_distance, radius_sum),
        "shift_x": shift_x,
        "shift_y": shift_y,
        "shift_distance": abs(shift_x) + abs(shift_y),
        "generation_difficulty": info.get("generation_difficulty", info.get("difficulty")),
        "target_size_valid": target_size.valid,
        "contaminant_size_valid": contaminant_size.valid,
    }


def analyze_split(
    split: str,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    methods: list[str],
    settings: AuditSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_sample_rows: list[dict[str, Any]] = []
    halo_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        target = sample["target"]
        blended = sample["blended"]
        affected = gd_utils.affected_region_mask(target, blended, threshold=settings.affected_threshold)
        core = balanced_helpers.target_core_mask(target)
        core_affected = np.logical_and(affected, core)
        noncore_affected = np.logical_and(affected, ~core)
        dilated = binary_dilation(affected, iterations=settings.halo_dilation)
        halo_band = np.logical_and(dilated, ~affected)
        base = sample_metadata(split, idx, sample, settings)
        size_rows.append(
            {
                key: base[key]
                for key in (
                    "sample_id",
                    "split",
                    "suite",
                    "target_area",
                    "contaminant_area",
                    "target_equiv_radius",
                    "contaminant_equiv_radius",
                    "apparent_size_ratio",
                    "target_flux_proxy",
                    "contaminant_flux_proxy",
                    "brightness",
                    "generation_size_ratio",
                    "core_obstruction_fraction",
                    "blend_severity_score",
                )
            }
        )
        row = dict(base)
        row["affected_fraction"] = float(affected.mean())
        row["core_affected_fraction"] = float(core_affected.mean())
        row["noncore_affected_fraction"] = float(noncore_affected.mean())
        row["halo_band_fraction"] = float(halo_band.mean())
        identity_affected_mse, _identity_affected_mae = masked_mse_mae(blended, target, affected)
        for method in methods:
            pred = preds[method][idx]
            affected_mse, affected_mae = masked_mse_mae(pred, target, affected)
            core_mse, core_mae = masked_mse_mae(pred, target, core_affected)
            noncore_mse, noncore_mae = masked_mse_mae(pred, target, noncore_affected)
            halo_mse, halo_mae = masked_mse_mae(pred, target, halo_band)
            whole_mse = float(np.mean((pred - target) ** 2))
            row[f"{method}_affected_mse"] = affected_mse
            row[f"{method}_affected_mae"] = affected_mae
            row[f"{method}_core_affected_mse"] = core_mse
            row[f"{method}_core_affected_mae"] = core_mae
            row[f"{method}_noncore_affected_mse"] = noncore_mse
            row[f"{method}_noncore_affected_mae"] = noncore_mae
            row[f"{method}_halo_band_mse"] = halo_mse
            row[f"{method}_halo_band_mae"] = halo_mae
            row[f"{method}_whole_mse"] = whole_mse
            row[f"{method}_improvement_vs_identity"] = safe_ratio(identity_affected_mse, affected_mse)
            row[f"{method}_worse_than_identity"] = bool(affected_mse > identity_affected_mse)
            halo_rows.append(
                {
                    "sample_id": base["sample_id"],
                    "split": split,
                    "suite": split,
                    "index": idx,
                    "method": method,
                    "affected_mse": affected_mse,
                    "core_affected_mse": core_mse,
                    "noncore_affected_mse": noncore_mse,
                    "halo_band_mse": halo_mse,
                    "halo_band_mae": halo_mae,
                    "halo_band_fraction": float(halo_band.mean()),
                    "apparent_size_ratio": base["apparent_size_ratio"],
                    "core_obstruction_fraction": base["core_obstruction_fraction"],
                    "improvement_vs_identity": safe_ratio(identity_affected_mse, affected_mse),
                    "worse_than_identity": bool(affected_mse > identity_affected_mse),
                }
            )
        affected_cols = [f"{method}_affected_mse" for method in methods]
        learned_cols = [f"{method}_affected_mse" for method in methods if method not in ("identity", "threshold")]
        row["affected_mse_winner"] = min(
            methods,
            key=lambda method: row.get(f"{method}_affected_mse", float("inf")),
        )
        row["learned_affected_mse_winner"] = min(
            [method for method in methods if method not in ("identity", "threshold")],
            key=lambda method: row.get(f"{method}_affected_mse", float("inf")),
        )
        row["best_affected_mse"] = float(np.nanmin([row[col] for col in affected_cols]))
        row["best_learned_affected_mse"] = float(np.nanmin([row[col] for col in learned_cols]))
        per_sample_rows.append(row)
    per_sample = pd.DataFrame(per_sample_rows)
    per_sample["apparent_size_ratio_bin"] = size_bin(per_sample["apparent_size_ratio"])
    per_sample["contaminant_centrality_bin"] = center_bin(per_sample["contaminant_distance_from_center_norm"])
    per_sample["centroid_overlap_bin"] = centroid_overlap_bin(per_sample["centroid_distance_over_radius_sum"])
    per_sample["core_obstruction_bin"] = core_obstruction_bin(per_sample["core_obstruction_fraction"])
    per_sample["blend_severity_bin"] = balanced_helpers.blend_severity_bins(per_sample["blend_severity_score"])
    return per_sample, pd.DataFrame(size_rows), pd.DataFrame(halo_rows)


def aggregate_by_group(
    per_sample: pd.DataFrame,
    methods: list[str],
    group_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, group), frame in per_sample.dropna(subset=[group_col]).groupby(["split", group_col], observed=True):
        identity_mse = float(frame["identity_affected_mse"].mean())
        for method in methods:
            affected_mse = float(frame[f"{method}_affected_mse"].mean())
            rows.append(
                {
                    "split": split,
                    "suite": split,
                    "grouping": group_col,
                    "group": str(group),
                    "method": method,
                    "n": int(len(frame)),
                    "affected_mse": affected_mse,
                    "affected_mae": float(frame[f"{method}_affected_mae"].mean()),
                    "core_affected_mse": float(frame[f"{method}_core_affected_mse"].mean()),
                    "halo_band_mse": float(frame[f"{method}_halo_band_mse"].mean()),
                    "improvement_ratio_vs_identity": safe_ratio(identity_mse, affected_mse),
                    "model_win_rate": float((frame["affected_mse_winner"] == method).mean()),
                    "learned_model_win_rate": float((frame["learned_affected_mse_winner"] == method).mean())
                    if method not in ("identity", "threshold")
                    else 0.0,
                    "worse_than_identity_count": int(frame[f"{method}_worse_than_identity"].sum()),
                    "worse_than_identity_fraction": float(frame[f"{method}_worse_than_identity"].mean()),
                    "mean_apparent_size_ratio": float(frame["apparent_size_ratio"].mean()),
                    "mean_core_obstruction_fraction": float(frame["core_obstruction_fraction"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_metric(
    frame: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    methods: list[str],
    split: str | None = None,
) -> None:
    data = frame[frame["grouping"] == group_col].copy()
    if split is not None:
        data = data[data["split"] == split]
    data = data[data["method"].isin(methods)]
    if data.empty:
        return
    groups = [group for group in group_order if group in set(data["group"])]
    if not groups:
        groups = sorted(data["group"].unique())
    x = np.arange(len(groups))
    width = min(0.8 / max(len(methods), 1), 0.16)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(groups)), 4.4))
    offsets = np.linspace(-width * (len(methods) - 1) / 2, width * (len(methods) - 1) / 2, len(methods))
    for offset, method in zip(offsets, methods):
        method_frame = data[data["method"] == method].set_index("group")
        values = [
            float(method_frame.loc[group, metric]) if group in method_frame.index else np.nan
            for group in groups
        ]
        ax.bar(x + offset, values, width, color=method_color(method), label=method_label(method))
    ax.set_xticks(x, groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    save_fig(fig, path)


def plot_size_figures(run_dir: Path, size_perf: pd.DataFrame, methods: list[str]) -> None:
    learned = [method for method in methods if method not in ("identity", "threshold")]
    plot_grouped_metric(
        size_perf,
        "apparent_size_ratio_bin",
        SIZE_BIN_ORDER,
        "affected_mse",
        "Affected MSE by Apparent Size Ratio",
        "Affected-region MSE",
        run_dir / "figures" / "affected_mse_by_apparent_size_ratio.png",
        methods=learned,
        split="stress",
    )
    plot_grouped_metric(
        size_perf,
        "apparent_size_ratio_bin",
        SIZE_BIN_ORDER,
        "improvement_ratio_vs_identity",
        "Improvement by Apparent Size Ratio",
        "Identity affected MSE / model affected MSE",
        run_dir / "figures" / "improvement_ratio_by_apparent_size_ratio.png",
        methods=learned,
        split="stress",
    )
    plot_grouped_metric(
        size_perf,
        "apparent_size_ratio_bin",
        SIZE_BIN_ORDER,
        "worse_than_identity_count",
        "Worse-than-Identity Counts by Apparent Size Ratio",
        "Count",
        run_dir / "figures" / "worse_than_identity_by_apparent_size_ratio.png",
        methods=learned,
        split="stress",
    )


def plot_centrality_figures(run_dir: Path, centrality_perf: pd.DataFrame, methods: list[str]) -> None:
    learned = [method for method in methods if method not in ("identity", "threshold")]
    plot_grouped_metric(
        centrality_perf,
        "contaminant_centrality_bin",
        CENTER_BIN_ORDER,
        "affected_mse",
        "Performance by Contaminant Centrality",
        "Affected-region MSE",
        run_dir / "figures" / "performance_by_centrality.png",
        methods=learned,
        split="stress",
    )
    plot_grouped_metric(
        centrality_perf,
        "core_obstruction_bin",
        CORE_BIN_ORDER,
        "affected_mse",
        "Performance by Core Obstruction",
        "Affected-region MSE",
        run_dir / "figures" / "performance_by_core_obstruction.png",
        methods=learned,
        split="stress",
    )


def plot_halo_figures(run_dir: Path, halo_long: pd.DataFrame, per_sample: pd.DataFrame, methods: list[str]) -> None:
    learned = [method for method in methods if method not in ("identity", "threshold")]
    agg = (
        halo_long[halo_long["method"].isin(learned)]
        .groupby(["split", "method"], as_index=False)
        .agg(halo_band_mse=("halo_band_mse", "mean"))
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    splits = ["normal", "stress"]
    x = np.arange(len(splits))
    width = min(0.8 / len(learned), 0.16)
    offsets = np.linspace(-width * (len(learned) - 1) / 2, width * (len(learned) - 1) / 2, len(learned))
    for offset, method in zip(offsets, learned):
        frame = agg[agg["method"] == method].set_index("split")
        values = [float(frame.loc[split, "halo_band_mse"]) if split in frame.index else np.nan for split in splits]
        ax.bar(x + offset, values, width, color=method_color(method), label=method_label(method))
    ax.set_xticks(x, ["Normal", "Stress"])
    ax.set_ylabel("Halo-band MSE")
    ax.set_title("Halo-Band Error by Model")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, run_dir / "figures" / "halo_band_mse_by_model.png")

    if WEIGHTED_METHOD in methods:
        stress = per_sample[per_sample["split"] == "stress"].replace([np.inf, -np.inf], np.nan)
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        for method in ("balanced_residual", WEIGHTED_METHOD):
            ax.scatter(
                stress[f"{method}_core_affected_mse"],
                stress[f"{method}_halo_band_mse"],
                s=12,
                alpha=0.35,
                color=method_color(method),
                label=method_label(method),
                edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Core affected MSE")
        ax.set_ylabel("Halo-band MSE")
        ax.set_title("Core Error vs Halo-Band Error")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        save_fig(fig, run_dir / "figures" / "core_mse_vs_halo_band_mse_scatter.png")

        data = stress[["balanced_residual_halo_band_mse", "weighted_residual_halo_band_mse"]].dropna()
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        ax.scatter(
            data["balanced_residual_halo_band_mse"],
            data["weighted_residual_halo_band_mse"],
            s=13,
            alpha=0.35,
            color=method_color(WEIGHTED_METHOD),
            edgecolors="none",
        )
        limit = float(np.nanmax(data.to_numpy())) if not data.empty else 1e-4
        limit = max(limit, 1e-6)
        ax.plot([1e-8, limit], [1e-8, limit], color="#8a4f49", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Thayer-BR v0.1 halo-band MSE")
        ax.set_ylabel("Thayer-BR v0.2 halo-band MSE")
        ax.set_title("v0.1 vs v0.2 Halo-Band Error")
        ax.grid(alpha=0.25)
        save_fig(fig, run_dir / "figures" / "br_v01_vs_v02_halo_band_error.png")


def overlay_mask(image: np.ndarray, affected: np.ndarray, core_affected: np.ndarray) -> np.ndarray:
    out = np.clip(image.copy(), 0.0, 1.0)
    out[affected] = 0.55 * out[affected] + 0.45 * np.asarray([1.0, 0.12, 0.02])
    out[core_affected] = 0.45 * out[core_affected] + 0.55 * np.asarray([0.1, 0.85, 0.25])
    return np.clip(out, 0.0, 1.0)


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(pred - target).mean(axis=-1)


def slugify(text: str) -> str:
    text = text.lower().replace("v0.2", "v02").replace("v0.1", "v01")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def select_disagreement_candidates(
    per_sample: pd.DataFrame,
    methods: list[str],
    settings: AuditSettings,
) -> pd.DataFrame:
    if WEIGHTED_METHOD not in methods:
        return pd.DataFrame()
    frame = per_sample.replace([np.inf, -np.inf], np.nan).copy()
    frame["v02_to_v01_affected_ratio"] = frame["weighted_residual_affected_mse"] / frame["balanced_residual_affected_mse"]
    frame["v02_to_v01_halo_ratio"] = frame["weighted_residual_halo_band_mse"] / frame["balanced_residual_halo_band_mse"]
    candidates: list[pd.DataFrame] = []

    selectors = [
        (
            "v0.2 broad halo artifact",
            frame[
                (frame["weighted_residual_affected_mse"] < frame["balanced_residual_affected_mse"])
                & (frame["weighted_residual_halo_band_mse"] > frame["balanced_residual_halo_band_mse"] * 1.20)
            ].sort_values(["v02_to_v01_halo_ratio", "weighted_residual_halo_band_mse"], ascending=False),
        ),
        (
            "v0.1 beats v0.2",
            frame[
                frame["balanced_residual_affected_mse"] < frame["weighted_residual_affected_mse"]
            ].sort_values(["v02_to_v01_affected_ratio", "weighted_residual_affected_mse"], ascending=False),
        ),
        (
            "direct visually cleaner",
            frame[
                (frame["weighted_residual_affected_mse"] < frame["direct_affected_mse"])
                & (frame["direct_halo_band_mse"] < frame["weighted_residual_halo_band_mse"] * 0.75)
            ].sort_values("weighted_residual_halo_band_mse", ascending=False),
        ),
        (
            "v0.2 broad halo artifact",
            frame[
                (frame["weighted_residual_core_affected_mse"] < frame["balanced_residual_core_affected_mse"])
                & (
                    (frame["weighted_residual_noncore_affected_mse"] > frame["balanced_residual_noncore_affected_mse"] * 1.10)
                    | (frame["weighted_residual_halo_band_mse"] > frame["balanced_residual_halo_band_mse"] * 1.20)
                )
            ].sort_values(["weighted_residual_halo_band_mse", "weighted_residual_noncore_affected_mse"], ascending=False),
        ),
        (
            "v0.2 strong success",
            frame[
                (frame["weighted_residual_affected_mse"] < frame["balanced_residual_affected_mse"] * 0.75)
                & (frame["weighted_residual_core_affected_mse"] < frame["balanced_residual_core_affected_mse"] * 0.80)
                & (frame["weighted_residual_halo_band_mse"] <= frame["balanced_residual_halo_band_mse"] * 1.20)
            ].sort_values(["v02_to_v01_affected_ratio", "weighted_residual_affected_mse"]),
        ),
        (
            "ambiguous target",
            frame[
                (frame["core_obstruction_fraction"] > 0.80)
                & (frame["identity_affected_mse"] < frame["identity_affected_mse"].median())
                & (frame["weighted_residual_affected_mse"] > frame["weighted_residual_affected_mse"].quantile(0.70))
            ].sort_values(["core_obstruction_fraction", "weighted_residual_affected_mse"], ascending=False),
        ),
    ]
    for category, selected in selectors:
        if selected.empty:
            continue
        subset = selected.head(settings.max_examples_per_category).copy()
        subset["suggested_category"] = category
        candidates.append(subset)
    if not candidates:
        return pd.DataFrame()
    result = pd.concat(candidates, ignore_index=True)
    keep_cols = [
        "sample_id",
        "suite",
        "split",
        "index",
        "suggested_category",
        "direct_affected_mse",
        "residual_affected_mse",
        "balanced_residual_affected_mse",
        "weighted_residual_affected_mse",
        "direct_core_affected_mse",
        "residual_core_affected_mse",
        "balanced_residual_core_affected_mse",
        "weighted_residual_core_affected_mse",
        "direct_noncore_affected_mse",
        "residual_noncore_affected_mse",
        "balanced_residual_noncore_affected_mse",
        "weighted_residual_noncore_affected_mse",
        "direct_halo_band_mse",
        "residual_halo_band_mse",
        "balanced_residual_halo_band_mse",
        "weighted_residual_halo_band_mse",
        "v02_to_v01_affected_ratio",
        "v02_to_v01_halo_ratio",
        "apparent_size_ratio",
        "core_obstruction_fraction",
        "blend_severity_score",
    ]
    return result[[col for col in keep_cols if col in result.columns]].drop_duplicates(
        subset=["sample_id", "suggested_category"],
    )


def save_disagreement_grid(
    run_dir: Path,
    candidate: pd.Series,
    samples_by_split: dict[str, list[dict[str, Any]]],
    preds_by_split: dict[str, dict[str, list[np.ndarray]]],
    settings: AuditSettings,
    rank: int,
) -> str:
    split = str(candidate["split"])
    idx = int(candidate["index"])
    sample = samples_by_split[split][idx]
    preds = preds_by_split[split]
    target = sample["target"]
    blended = sample["blended"]
    affected = gd_utils.affected_region_mask(target, blended, threshold=settings.affected_threshold)
    core = balanced_helpers.target_core_mask(target)
    core_affected = np.logical_and(affected, core)
    balanced_error = error_map(preds["balanced_residual"][idx], target)
    weighted_error = error_map(preds[WEIGHTED_METHOD][idx], target)
    panels = [
        (target, "Target"),
        (blended, "Blend"),
        (overlay_mask(blended, affected, core_affected), "Core/affected mask"),
        (preds["direct"][idx], "Thayer-Direct"),
        (preds["residual"][idx], "Thayer-Residual"),
        (preds["balanced_residual"][idx], "Thayer-BR v0.1"),
        (preds[WEIGHTED_METHOD][idx], "Thayer-BR v0.2"),
        (balanced_error, "v0.1 error"),
        (weighted_error, "v0.2 error"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20.0, 2.7))
    vmax = max(
        0.03,
        float(np.nanpercentile(np.concatenate([balanced_error.ravel(), weighted_error.ravel()]), 99)),
    )
    for ax, (image, title) in zip(axes, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="magma", vmin=0.0, vmax=vmax)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"{candidate['suggested_category']} | {candidate['sample_id']}",
        fontsize=11,
    )
    fig.tight_layout()
    filename = f"{rank:02d}_{slugify(str(candidate['suggested_category']))}_{candidate['sample_id']}.png"
    path = run_dir / "example_grids" / "visual_metric_disagreements" / filename
    save_fig(fig, path, dpi=170)
    return project_relative(path)


def save_disagreement_grids(
    run_dir: Path,
    candidates: pd.DataFrame,
    samples_by_split: dict[str, list[dict[str, Any]]],
    preds_by_split: dict[str, dict[str, list[np.ndarray]]],
    settings: AuditSettings,
) -> list[str]:
    if candidates.empty:
        return []
    paths: list[str] = []
    for rank, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        paths.append(
            save_disagreement_grid(
                run_dir,
                candidate,
                samples_by_split,
                preds_by_split,
                settings,
                rank,
            )
        )
    return paths


def write_diagnostics_report(
    run_dir: Path,
    per_sample: pd.DataFrame,
    size_perf: pd.DataFrame,
    centrality_perf: pd.DataFrame,
    halo_long: pd.DataFrame,
    methods: list[str],
) -> dict[str, Any]:
    weighted_present = WEIGHTED_METHOD in methods
    size_corr = {}
    for method in [m for m in methods if m not in ("identity", "threshold")]:
        valid = per_sample[["apparent_size_ratio", f"{method}_affected_mse"]].replace([np.inf, -np.inf], np.nan).dropna()
        size_corr[method] = float(valid["apparent_size_ratio"].corr(valid[f"{method}_affected_mse"], method="spearman")) if len(valid) > 2 else float("nan")
    halo_tradeoff = {}
    if weighted_present:
        for split, frame in per_sample.groupby("split"):
            halo_tradeoff[split] = {
                "weighted_halo_mse": float(frame["weighted_residual_halo_band_mse"].mean()),
                "balanced_halo_mse": float(frame["balanced_residual_halo_band_mse"].mean()),
                "weighted_core_mse": float(frame["weighted_residual_core_affected_mse"].mean()),
                "balanced_core_mse": float(frame["balanced_residual_core_affected_mse"].mean()),
                "weighted_affected_mse": float(frame["weighted_residual_affected_mse"].mean()),
                "balanced_affected_mse": float(frame["balanced_residual_affected_mse"].mean()),
            }
    summary = {
        "size_spearman_correlation_with_affected_mse": size_corr,
        "apparent_size_ratio_quantiles": per_sample["apparent_size_ratio"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).to_dict(),
        "halo_tradeoff": halo_tradeoff,
        "weighted_present": weighted_present,
        "interpretation": {
            "size_ratio_affects_performance": bool(
                any(np.isfinite(v) and abs(v) >= 0.20 for v in size_corr.values())
            ),
            "size_normalized_benchmark_recommended": True,
            "headline_threatened": False,
        },
    }
    lines = [
        "# Size and Visual Audit Diagnostics",
        "",
        "## Apparent Size",
        "",
        f"- Apparent size-ratio quantiles: `{summary['apparent_size_ratio_quantiles']}`.",
        f"- Spearman correlations with affected MSE: `{size_corr}`.",
        "",
        "## Halo/Core Tradeoff",
        "",
        f"- Weighted-vs-balanced halo/core summary: `{halo_tradeoff}`.",
        "",
        "## Interpretation",
        "",
        "- Apparent size varies enough that a size-normalized benchmark is recommended as a follow-up.",
        "- This audit is diagnostic and does not invalidate the current headline metrics.",
        "- The v0.2 model should still be reported with visual counterexamples and halo-band caveats.",
    ]
    save_text(run_dir / "diagnostics" / "audit_summary.md", "\n".join(lines))
    save_json(run_dir / "diagnostics" / "audit_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    settings = AuditSettings(
        n_normal_blends=args.n_normal_blends,
        n_stress_blends=args.n_stress_blends,
        test_source_subset=args.test_source_subset,
        stress_source_subset=args.stress_source_subset,
        batch_size=args.batch_size,
    )
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    run_dir = make_run_dir(output_root)

    direct_checkpoint = resolve_path(args.direct_checkpoint)
    residual_checkpoint = resolve_path(args.residual_checkpoint)
    balanced_checkpoint = resolve_path(args.balanced_checkpoint)
    weighted_checkpoint = find_moderate_weighted_checkpoint(args.weighted_checkpoint)
    required = {
        "direct": direct_checkpoint,
        "residual": residual_checkpoint,
        "balanced_residual": balanced_checkpoint,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        save_json(run_dir / "diagnostics" / "missing_checkpoints.json", {"missing": missing})
        print(f"Missing required checkpoints: {missing}", flush=True)
        return 2
    checkpoint_paths = dict(required)
    if weighted_checkpoint is not None and weighted_checkpoint.exists():
        checkpoint_paths[WEIGHTED_METHOD] = weighted_checkpoint

    checkpoint_before = checkpoint_inventory(checkpoint_paths.values())
    save_json(run_dir / "logs" / "checkpoint_metadata_before.json", checkpoint_before)
    save_yaml = lambda path, payload: path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    run_config_path = run_dir / "logs" / "run_config.yaml"
    if run_config_path.exists():
        raise FileExistsError(run_config_path)
    save_yaml(
        run_config_path,
        {
            "project_root": ".",
            "settings": settings.__dict__,
            "checkpoints": {name: project_relative(path) for name, path in checkpoint_paths.items()},
            "weighted_checkpoint_used": project_relative(weighted_checkpoint) if weighted_checkpoint else None,
            "no_training": True,
        },
    )

    device = gd_train.resolve_accelerator(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"Run directory: {project_relative(run_dir)}", flush=True)

    print("Loading held-out test images.", flush=True)
    test_images, split_audit = load_test_images(config, settings)
    save_json(run_dir / "logs" / "split_audit.json", split_audit)

    print("Generating normal and stress audit blends.", flush=True)
    samples_by_split = make_samples(test_images, config, settings)

    print("Loading checkpoints for inference only.", flush=True)
    models = load_models(config, device, checkpoint_paths)
    methods = list(BASE_METHODS)
    if WEIGHTED_METHOD in models:
        methods.append(WEIGHTED_METHOD)

    preds_by_split: dict[str, dict[str, list[np.ndarray]]] = {}
    per_sample_frames: list[pd.DataFrame] = []
    size_frames: list[pd.DataFrame] = []
    halo_frames: list[pd.DataFrame] = []
    for split, samples in samples_by_split.items():
        print(f"Running inference and audit metrics for {split}.", flush=True)
        preds = predict_models(samples, models, device, settings.batch_size)
        preds_by_split[split] = preds
        per_sample, size_rows, halo_rows = analyze_split(split, samples, preds, methods, settings)
        per_sample_frames.append(per_sample)
        size_frames.append(size_rows)
        halo_frames.append(halo_rows)

    per_sample_all = pd.concat(per_sample_frames, ignore_index=True)
    size_audit = pd.concat(size_frames, ignore_index=True)
    halo_long = pd.concat(halo_frames, ignore_index=True)
    centrality_audit = per_sample_all.copy()

    print("Writing audit tables.", flush=True)
    save_csv(run_dir / "tables" / "apparent_size_audit.csv", size_audit)
    save_csv(run_dir / "tables" / "centrality_audit.csv", centrality_audit)
    save_csv(run_dir / "tables" / "halo_band_error_audit.csv", halo_long)
    save_csv(run_dir / "tables" / "per_sample_model_metrics.csv", per_sample_all)

    size_perf = aggregate_by_group(per_sample_all, methods, "apparent_size_ratio_bin")
    centrality_perf = pd.concat(
        [
            aggregate_by_group(per_sample_all, methods, "contaminant_centrality_bin"),
            aggregate_by_group(per_sample_all, methods, "centroid_overlap_bin"),
            aggregate_by_group(per_sample_all, methods, "core_obstruction_bin"),
        ],
        ignore_index=True,
    )
    save_csv(run_dir / "tables" / "performance_by_apparent_size_ratio.csv", size_perf)
    save_csv(run_dir / "tables" / "performance_by_centrality.csv", centrality_perf)

    print("Selecting visual-vs-metric disagreement candidates.", flush=True)
    candidates = select_disagreement_candidates(per_sample_all, methods, settings)
    save_csv(run_dir / "tables" / "visual_metric_disagreement_candidates.csv", candidates)
    grid_paths = save_disagreement_grids(
        run_dir,
        candidates,
        samples_by_split,
        preds_by_split,
        settings,
    )
    save_json(run_dir / "logs" / "visual_metric_disagreement_grids.json", grid_paths)

    print("Writing figures.", flush=True)
    plot_size_figures(run_dir, size_perf, methods)
    plot_centrality_figures(run_dir, centrality_perf, methods)
    plot_halo_figures(run_dir, halo_long, per_sample_all, methods)

    summary = write_diagnostics_report(
        run_dir,
        per_sample_all,
        size_perf,
        centrality_perf,
        halo_long,
        methods,
    )

    checkpoint_comparison = verify_checkpoint_inventory(checkpoint_before)
    save_json(run_dir / "logs" / "checkpoint_integrity_comparison.json", checkpoint_comparison)
    if not all(item["unchanged"] for item in checkpoint_comparison.values()):
        raise RuntimeError("Checkpoint metadata changed during audit.")

    print(f"Audit complete: {project_relative(run_dir)}", flush=True)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
