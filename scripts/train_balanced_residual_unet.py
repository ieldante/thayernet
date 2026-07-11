"""Train Experiment 3: balanced hard-case residual U-Net.

The script is output-safe:

- previous direct and residual checkpoints are read but never modified;
- new checkpoints and run directories are timestamped;
- best-validation and final checkpoints use different filenames;
- suspicious results are written to diagnostics instead of being hidden.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stress_test as stress_helpers
from src import baselines
from src import blend as gd_blend
from src import data as gd_data
from src import models
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
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "docs" / "experiment_log.md"
PAPER_PLAN_PATH = PROJECT_ROOT / "docs" / "paper_plan.md"

METHODS = ("identity", "threshold", "direct", "residual", "balanced_residual")
LEARNED_METHODS = ("direct", "residual", "balanced_residual")


@dataclass(frozen=True)
class BalancedSettings:
    """Default balanced residual scale and training composition."""

    n_train_blends: int = 8000
    n_val_blends: int = 1000
    n_normal_test_blends: int = 1000
    n_stress_blends: int = 1000
    train_source_subset: int = 5000
    val_source_subset: int = 1000
    test_source_subset: int = 1000
    num_epochs: int = 20
    normal_fraction: float = 0.50
    high_overlap_fraction: float = 0.30
    brightness_size_fraction: float = 0.20
    affected_region_threshold: float = 0.02


class ResidualBlendDataset(Dataset):
    """PyTorch dataset whose target is residual = blended - target."""

    def __init__(self, blends: list[dict[str, Any]]) -> None:
        if not blends:
            raise ValueError("ResidualBlendDataset requires at least one blend.")
        self.blends = blends

    def __len__(self) -> int:
        return len(self.blends)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.blends[idx]
        blended = np.asarray(sample["blended"], dtype=np.float32)
        target = np.asarray(sample["target"], dtype=np.float32)
        residual = blended - target
        return (
            torch.from_numpy(blended.transpose(2, 0, 1)).float(),
            torch.from_numpy(residual.transpose(2, 0, 1)).float(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train balanced hard-case residual U-Net for Thayer-Net."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--direct-checkpoint", type=Path, default=DEFAULT_DIRECT_CHECKPOINT)
    parser.add_argument(
        "--residual-checkpoint",
        type=Path,
        default=DEFAULT_RESIDUAL_CHECKPOINT,
        help="Previous residual U-Net checkpoint to compare against.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-train-blends", type=int, default=8000)
    parser.add_argument("--n-val-blends", type=int, default=1000)
    parser.add_argument("--n-normal-test-blends", type=int, default=1000)
    parser.add_argument("--n-stress-blends", type=int, default=1000)
    parser.add_argument("--train-source-subset", type=int, default=5000)
    parser.add_argument("--val-source-subset", type=int, default=1000)
    parser.add_argument("--test-source-subset", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-doc-update", action="store_true")
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return config


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def is_memory_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state_dict.")
    return state


def make_residual_unet(model_config: dict[str, Any]) -> models.UNet:
    model = models.UNet(**model_config)
    model.out_activation = nn.Identity()
    return model


def load_direct_model(
    checkpoint_path: Path,
    model_config: dict[str, Any],
    device: torch.device,
) -> models.UNet:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = models.UNet(**model_config)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def load_residual_model(
    checkpoint_path: Path,
    model_config: dict[str, Any],
    device: torch.device,
) -> models.UNet:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = make_residual_unet(model_config)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def checkpoint_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": project_relative(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def verify_checkpoint_unchanged(
    before: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    after = checkpoint_info(path)
    unchanged = (
        before["size_bytes"] == after["size_bytes"]
        and before["mtime_ns"] == after["mtime_ns"]
    )
    return {"before": before, "after": after, "unchanged": unchanged}


def make_run_paths(output_root: Path, stamp: str) -> tuple[Path, Path, Path]:
    run_dir = output_root / "runs" / f"balanced_residual_{stamp}"
    best_checkpoint = (
        output_root / "checkpoints" / f"unet_residual_balanced_hard_{stamp}.pth"
    )
    final_checkpoint = (
        output_root / "checkpoints" / f"unet_residual_balanced_hard_final_{stamp}.pth"
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if best_checkpoint.exists():
        raise FileExistsError(f"Checkpoint path already exists: {best_checkpoint}")
    if final_checkpoint.exists():
        raise FileExistsError(f"Checkpoint path already exists: {final_checkpoint}")
    for child in ("results", "figures", "paper_figures", "diagnostics", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, best_checkpoint, final_checkpoint


def load_split_subsets(
    config: dict[str, Any],
    settings: BalancedSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = PROJECT_ROOT / config["dataset_path"]
    images_raw, labels, _metadata = gd_data.load_galaxy10(data_path)
    train, val, test = gd_data.split_dataset(
        images_raw,
        labels,
        train_frac=float(config["splits"]["train_frac"]),
        val_frac=float(config["splits"]["val_frac"]),
        test_frac=float(config["splits"]["test_frac"]),
        shuffle=True,
        seed=int(config["seed"]),
    )
    train_images_raw, _train_labels = train
    val_images_raw, _val_labels = val
    test_images_raw, _test_labels = test

    train_images = gd_data.normalise_images(
        train_images_raw[: settings.train_source_subset]
    )
    val_images = gd_data.normalise_images(val_images_raw[: settings.val_source_subset])
    test_images = gd_data.normalise_images(test_images_raw[: settings.test_source_subset])
    del images_raw, labels, train_images_raw, val_images_raw, test_images_raw
    gc.collect()
    return train_images, val_images, test_images


def blend_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    blend_config = dict(config["blending"])
    return {
        "max_shift": int(blend_config["max_shift"]),
        "brightness_range": tuple(float(x) for x in blend_config["brightness_range"]),
        "blur_range": tuple(float(x) for x in blend_config["blur_range"]),
        "noise_range": tuple(float(x) for x in blend_config["noise_range"]),
        "rotation_range": tuple(float(x) for x in blend_config["rotation_range"]),
    }


def target_core_mask(
    target: np.ndarray,
    aperture_fraction: float = 0.18,
    core_percentile: float = 85.0,
) -> np.ndarray:
    return gd_utils.evaluation_core_mask_p85_v1(
        target,
        aperture_fraction=aperture_fraction,
        core_percentile=core_percentile,
    )


def blend_metadata(
    target: np.ndarray,
    blended: np.ndarray,
    info: dict[str, Any],
    threshold: float,
) -> dict[str, float]:
    affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
    mask_fraction = float(affected.mean())
    core = target_core_mask(target)
    core_obstruction = (
        float(np.logical_and(affected, core).sum() / core.sum())
        if np.any(core)
        else 0.0
    )
    size_ratio = float(info.get("size_ratio", float("nan")))
    identity_affected_mae = gd_utils.masked_mae(blended, target, affected)
    blend_severity_score = (
        mask_fraction * identity_affected_mae * (1.0 + core_obstruction)
    )
    return {
        "mask_fraction": mask_fraction,
        "core_obstruction_fraction": core_obstruction,
        "size_ratio": size_ratio,
        "blend_severity_score": blend_severity_score,
    }


def normal_blends(
    images: np.ndarray,
    n_blends: int,
    config: dict[str, Any],
    seed: int,
    component: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    blends = gd_blend.generate_blends(
        images,
        n_blends=n_blends,
        rng=rng,
        **blend_params_from_config(config),
    )
    for sample in blends:
        sample["info"]["training_component"] = component
    return blends


def generate_targeted_blends(
    images: np.ndarray,
    n_blends: int,
    seed: int,
    component: str,
    threshold: float,
    max_shift: int,
    brightness_range: tuple[float, float],
    blur_range: tuple[float, float],
    noise_range: tuple[float, float],
    min_size_ratio: float,
    min_mask_fraction: float,
    min_core_obstruction: float | None,
    max_attempt_multiplier: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if n_blends <= 0:
        return [], {"component": component, "requested": 0, "accepted": 0}

    rng = np.random.default_rng(seed)
    accepted: list[dict[str, Any]] = []
    relaxed: list[dict[str, Any]] = []
    relaxed_candidates_used = 0
    attempts = 0
    max_attempts = max_attempt_multiplier * n_blends

    while attempts < max_attempts and len(accepted) < n_blends:
        attempts += 1
        target_idx, contaminant_idx = rng.choice(images.shape[0], size=2, replace=False)
        target = images[target_idx]
        contaminant = images[contaminant_idx]
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        brightness = float(rng.uniform(*brightness_range))
        blur_sigma = float(rng.uniform(*blur_range))
        noise_std = float(rng.uniform(*noise_range))
        blended, info = gd_blend.blend_pair(
            target=target,
            contaminant=contaminant,
            shift=(dx, dy),
            rotation=0.0,
            brightness=brightness,
            blur_sigma=blur_sigma,
            noise_std=noise_std,
            rng=rng,
        )
        metadata = blend_metadata(target, blended, info, threshold)
        info = {
            **info,
            **metadata,
            "target_index": int(target_idx),
            "contaminant_index": int(contaminant_idx),
            "attempt": int(attempts),
            "training_component": component,
        }
        sample = {
            "target": target,
            "contaminant": contaminant,
            "blended": blended,
            "info": info,
        }
        size_ratio = metadata["size_ratio"]
        finite_size = np.isfinite(size_ratio)
        mask_ok = metadata["mask_fraction"] >= min_mask_fraction
        size_ok = finite_size and size_ratio >= min_size_ratio
        core_ok = (
            True
            if min_core_obstruction is None
            else metadata["core_obstruction_fraction"] >= min_core_obstruction
        )
        if mask_ok and size_ok and core_ok:
            accepted.append(sample)
        elif mask_ok and finite_size and size_ratio >= 0.5:
            relaxed.append(sample)

    if len(accepted) < n_blends:
        needed = n_blends - len(accepted)
        relaxed.sort(
            key=lambda row: (
                row["info"]["core_obstruction_fraction"],
                row["info"]["mask_fraction"],
                row["info"]["size_ratio"],
                row["info"]["brightness"],
            ),
            reverse=True,
        )
        relaxed_selected = relaxed[:needed]
        relaxed_candidates_used = len(relaxed_selected)
        accepted.extend(relaxed_selected)

    if len(accepted) < n_blends:
        raise RuntimeError(
            f"Could not generate enough {component} blends. "
            f"Requested {n_blends}, accepted {len(accepted)} after {attempts} attempts."
        )

    selected = accepted[:n_blends]
    diagnostics = {
        "component": component,
        "requested": n_blends,
        "accepted": len(selected),
        "attempts": attempts,
        "mean_mask_fraction": float(
            np.mean([sample["info"]["mask_fraction"] for sample in selected])
        ),
        "mean_core_obstruction_fraction": float(
            np.mean(
                [sample["info"]["core_obstruction_fraction"] for sample in selected]
            )
        ),
        "mean_size_ratio": float(
            np.mean([sample["info"]["size_ratio"] for sample in selected])
        ),
        "mean_brightness": float(
            np.mean([sample["info"]["brightness"] for sample in selected])
        ),
        "relaxed_candidates_used": relaxed_candidates_used,
    }
    return selected, diagnostics


def component_counts(total: int, settings: BalancedSettings) -> dict[str, int]:
    normal = int(round(total * settings.normal_fraction))
    high_overlap = int(round(total * settings.high_overlap_fraction))
    brightness_size = total - normal - high_overlap
    return {
        "normal": normal,
        "high_overlap_core": high_overlap,
        "brightness_size": brightness_size,
    }


def generate_balanced_blends(
    images: np.ndarray,
    total: int,
    config: dict[str, Any],
    seed: int,
    settings: BalancedSettings,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = component_counts(total, settings)
    threshold = settings.affected_region_threshold
    components: list[dict[str, Any]] = []
    blends: list[dict[str, Any]] = []

    normal = normal_blends(
        images,
        counts["normal"],
        config,
        seed=seed + 11,
        component=f"{split_name}_normal",
    )
    blends.extend(normal)
    components.append(
        {
            "component": f"{split_name}_normal",
            "requested": len(normal),
            "accepted": len(normal),
        }
    )

    high_overlap, high_diag = generate_targeted_blends(
        images=images,
        n_blends=counts["high_overlap_core"],
        seed=seed + 23,
        component=f"{split_name}_high_overlap_core",
        threshold=threshold,
        max_shift=18,
        brightness_range=(0.8, 1.4),
        blur_range=(0.0, 0.15),
        noise_range=(0.0, 0.006),
        min_size_ratio=0.75,
        min_mask_fraction=0.01,
        min_core_obstruction=0.66,
    )
    blends.extend(high_overlap)
    components.append(high_diag)

    brightness_size, brightness_diag = generate_targeted_blends(
        images=images,
        n_blends=counts["brightness_size"],
        seed=seed + 37,
        component=f"{split_name}_brightness_size",
        threshold=threshold,
        max_shift=36,
        brightness_range=(1.05, 1.5),
        blur_range=(0.0, 0.15),
        noise_range=(0.0, 0.006),
        min_size_ratio=0.75,
        min_mask_fraction=0.01,
        min_core_obstruction=None,
    )
    blends.extend(brightness_size)
    components.append(brightness_diag)

    rng = np.random.default_rng(seed + 101)
    rng.shuffle(blends)
    return blends, {"split": split_name, "total": len(blends), "components": components}


def metric_values(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    whole = gd_utils.compute_metrics(pred, target, metrics=("mse", "mae", "psnr", "ssim"))
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


def core_overlap_bins(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-0.001, 1.0 / 3.0, 2.0 / 3.0, 1.001],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )


def blend_severity_bins(values: pd.Series) -> pd.Series:
    try:
        return pd.qcut(
            values.rank(method="first"),
            q=3,
            labels=["low", "medium", "high"],
        )
    except ValueError:
        return pd.Series(["unbinned"] * len(values), index=values.index)


def empty_residual_stats() -> dict[str, float]:
    return {
        "residual_pred_min": float("inf"),
        "residual_pred_max": float("-inf"),
        "residual_pred_abs_max": 0.0,
        "residual_pred_sum": 0.0,
        "residual_pred_sq_sum": 0.0,
        "preclip_low_count": 0.0,
        "preclip_high_count": 0.0,
        "pixel_count": 0.0,
    }


def update_residual_stats(
    stats: dict[str, float],
    residual_pred: np.ndarray,
    reconstruction_preclip: np.ndarray,
) -> None:
    stats["residual_pred_min"] = min(stats["residual_pred_min"], float(residual_pred.min()))
    stats["residual_pred_max"] = max(stats["residual_pred_max"], float(residual_pred.max()))
    stats["residual_pred_abs_max"] = max(
        stats["residual_pred_abs_max"], float(np.max(np.abs(residual_pred)))
    )
    stats["residual_pred_sum"] += float(np.sum(residual_pred))
    stats["residual_pred_sq_sum"] += float(np.sum(residual_pred**2))
    stats["preclip_low_count"] += float(np.count_nonzero(reconstruction_preclip < 0.0))
    stats["preclip_high_count"] += float(np.count_nonzero(reconstruction_preclip > 1.0))
    stats["pixel_count"] += float(residual_pred.size)


def finalise_residual_stats(stats: dict[str, float]) -> dict[str, float]:
    pixel_count = max(stats["pixel_count"], 1.0)
    mean = stats["residual_pred_sum"] / pixel_count
    variance = max(stats["residual_pred_sq_sum"] / pixel_count - mean**2, 0.0)
    low_fraction = stats["preclip_low_count"] / pixel_count
    high_fraction = stats["preclip_high_count"] / pixel_count
    return {
        "residual_pred_min": stats["residual_pred_min"],
        "residual_pred_max": stats["residual_pred_max"],
        "residual_pred_abs_max": stats["residual_pred_abs_max"],
        "residual_pred_mean": mean,
        "residual_pred_std": float(np.sqrt(variance)),
        "reconstruction_preclip_low_fraction": low_fraction,
        "reconstruction_preclip_high_fraction": high_fraction,
        "reconstruction_preclip_total_clip_fraction": low_fraction + high_fraction,
    }


def train_one_attempt(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame]:
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model.to(device)
    rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for blended, residual in train_loader:
            blended = blended.to(device)
            residual = residual.to(device)
            optimiser.zero_grad()
            output = model(blended)
            loss = criterion(output, residual)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimiser.step()
            train_loss += float(loss.item()) * blended.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blended, residual in val_loader:
                blended = blended.to(device)
                residual = residual.to(device)
                output = model(blended)
                loss = criterion(output, residual)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite validation loss at epoch {epoch}.")
                val_loss += float(loss.item()) * blended.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        if train_loss > 1.0 or val_loss > 1.0:
            raise RuntimeError(
                f"Training loss exploded at epoch {epoch}: "
                f"train={train_loss:.6f}, val={val_loss:.6f}."
            )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
            }
        )
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"train residual loss={train_loss:.6f}, "
            f"val residual loss={val_loss:.6f}, "
            f"best={best_val:.6f} @ {best_epoch}",
            flush=True,
        )

    return model, best_state, pd.DataFrame(rows)


def train_with_memory_retry(
    train_ds: Dataset,
    val_ds: Dataset,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    settings: BalancedSettings,
    requested_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame, int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    batch_size = requested_batch_size
    while batch_size >= 1:
        seed_everything(seed)
        model = make_residual_unet(model_config)
        try:
            trained_model, best_state, history = train_one_attempt(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                num_epochs=settings.num_epochs,
                batch_size=batch_size,
                learning_rate=float(training_config["learning_rate"]),
                weight_decay=float(training_config.get("weight_decay", 0.0)),
                device=device,
                seed=seed,
            )
            attempts.append({"batch_size": batch_size, "status": "completed"})
            return trained_model, best_state, history, batch_size, attempts
        except RuntimeError as exc:
            if not is_memory_error(exc) or batch_size == 1:
                attempts.append(
                    {"batch_size": batch_size, "status": "failed", "error": str(exc)}
                )
                raise
            next_batch = max(1, batch_size // 2)
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "memory_failed",
                    "retry_batch_size": next_batch,
                    "error": str(exc),
                }
            )
            print(
                f"Memory failure at batch size {batch_size}; retrying at {next_batch}.",
                flush=True,
            )
            del model
            clear_torch_cache()
            gc.collect()
            batch_size = next_batch
    raise RuntimeError("Could not train with any batch size.")


def save_checkpoint(
    path: Path,
    model_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    settings: BalancedSettings,
    history: pd.DataFrame,
    batch_size: int,
    stamp: str,
    checkpoint_kind: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    payload = {
        "model_state_dict": model_state,
        "config": config,
        "experiment_settings": settings.__dict__,
        "checkpoint_kind": checkpoint_kind,
        "residual_target": "blended_minus_target",
        "reconstruction": "blended_minus_predicted_residual",
        "output_activation": "identity",
        "timestamp": stamp,
        "batch_size": batch_size,
        "final_train_loss": float(history["train_loss"].iloc[-1]),
        "final_val_loss": float(history["val_loss"].iloc[-1]),
        "best_epoch": int(history["best_epoch"].iloc[-1]),
        "best_val_loss": float(history["best_val_loss"].iloc[-1]),
    }
    torch.save(payload, path)


def predict_batch(
    batch_samples: list[dict[str, Any]],
    direct_model: nn.Module,
    residual_model: nn.Module,
    balanced_model: nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.stack([sample["blended"] for sample in batch_samples], axis=0)
    tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
    with torch.no_grad():
        direct = direct_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
        residual_layer = (
            residual_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
        )
        balanced_layer = (
            balanced_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
        )
    residual_recon = np.clip(inputs - residual_layer, 0.0, 1.0).astype(np.float32)
    balanced_preclip = inputs - balanced_layer
    balanced_recon = np.clip(balanced_preclip, 0.0, 1.0).astype(np.float32)
    return (
        np.clip(direct, 0.0, 1.0).astype(np.float32),
        residual_recon,
        balanced_layer,
        balanced_recon,
    )


def evaluate_samples(
    split_name: str,
    samples: list[dict[str, Any]],
    direct_model: nn.Module,
    residual_model: nn.Module,
    balanced_model: nn.Module,
    device: torch.device,
    threshold: float,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    balanced_stats = empty_residual_stats()

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        direct_preds, residual_preds, balanced_layers, balanced_preds = predict_batch(
            batch_samples,
            direct_model,
            residual_model,
            balanced_model,
            device,
        )
        inputs = np.stack([sample["blended"] for sample in batch_samples], axis=0)
        update_residual_stats(balanced_stats, balanced_layers, inputs - balanced_layers)

        for offset, sample in enumerate(batch_samples):
            index = start + offset
            target = sample["target"]
            blended = sample["blended"]
            threshold_pred = baselines.threshold_baseline(blended)
            affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
            metadata = blend_metadata(target, blended, sample.get("info", {}), threshold)
            info = sample.get("info", {})
            shift = info.get("shift", (0, 0))
            row: dict[str, Any] = {
                "split": split_name,
                "index": index,
                "generation_difficulty": info.get(
                    "generation_difficulty", info.get("difficulty")
                ),
                "training_component": info.get("training_component"),
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
                "mask_fraction": metadata["mask_fraction"],
                "core_obstruction_fraction": metadata["core_obstruction_fraction"],
                "blend_severity_score": metadata["blend_severity_score"],
            }
            predictions = {
                "identity": blended,
                "threshold": threshold_pred,
                "direct": direct_preds[offset],
                "residual": residual_preds[offset],
                "balanced_residual": balanced_preds[offset],
            }
            metric_map = {
                method: metric_values(pred, target, affected)
                for method, pred in predictions.items()
            }
            identity_mse = metric_map["identity"]["masked_mse"]
            for method, values in metric_map.items():
                for metric_name, value in values.items():
                    row[f"{method}_{metric_name}"] = value
                row[f"{method}_affected_mse"] = values["masked_mse"]
                row[f"{method}_affected_mae"] = values["masked_mae"]
                row[f"{method}_improvement_ratio"] = safe_ratio(
                    identity_mse, values["masked_mse"]
                )
                row[f"{method}_worse_than_identity"] = (
                    values["masked_mse"] > identity_mse
                )

            row["balanced_beats_residual"] = (
                row["balanced_residual_affected_mse"] < row["residual_affected_mse"]
            )
            row["balanced_beats_direct"] = (
                row["balanced_residual_affected_mse"] < row["direct_affected_mse"]
            )
            row["residual_beats_balanced"] = (
                row["residual_affected_mse"] < row["balanced_residual_affected_mse"]
            )
            row["direct_beats_balanced"] = (
                row["direct_affected_mse"] < row["balanced_residual_affected_mse"]
            )
            row["balanced_residual_to_residual_mse_ratio"] = safe_ratio(
                row["balanced_residual_affected_mse"], row["residual_affected_mse"]
            )
            row["balanced_residual_to_direct_mse_ratio"] = safe_ratio(
                row["balanced_residual_affected_mse"], row["direct_affected_mse"]
            )
            rows.append(row)

    per_sample = pd.DataFrame(rows)
    per_sample["blend_severity_bin"] = blend_severity_bins(
        per_sample["blend_severity_score"]
    )
    per_sample["core_overlap_bin"] = core_overlap_bins(
        per_sample["core_obstruction_fraction"]
    )
    aggregate = aggregate_metrics(per_sample)
    grouped = grouped_metrics(per_sample)
    return aggregate, per_sample, grouped, finalise_residual_stats(balanced_stats)


def aggregate_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity_mse = float(per_sample["identity_masked_mse"].mean())
    for method in METHODS:
        row: dict[str, Any] = {
            "method": method,
            "n": int(len(per_sample)),
            "mean_mask_fraction": float(per_sample["mask_fraction"].mean()),
        }
        for metric in ("mse", "mae", "psnr", "ssim", "masked_mse", "masked_mae"):
            row[metric] = float(per_sample[f"{method}_{metric}"].mean())
        row["affected_mse"] = row["masked_mse"]
        row["affected_mae"] = row["masked_mae"]
        row["affected_mse_improvement_vs_identity"] = safe_ratio(
            identity_mse, row["masked_mse"]
        )
        worse = per_sample[f"{method}_worse_than_identity"]
        row["worse_than_identity_n"] = int(worse.sum())
        row["worse_than_identity_fraction"] = float(worse.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_col in ("generation_difficulty", "blend_severity_bin", "core_overlap_bin"):
        frame = per_sample.dropna(subset=[group_col])
        grouped = frame.groupby(group_col, observed=True)
        for group, group_frame in grouped:
            identity_mse = float(group_frame["identity_masked_mse"].mean())
            for method in METHODS:
                rows.append(
                    {
                        "grouping": group_col,
                        "group": str(group),
                        "method": method,
                        "n": int(len(group_frame)),
                        "affected_mse": float(group_frame[f"{method}_masked_mse"].mean()),
                        "affected_mae": float(group_frame[f"{method}_masked_mae"].mean()),
                        "whole_mse": float(group_frame[f"{method}_mse"].mean()),
                        "ssim": float(group_frame[f"{method}_ssim"].mean()),
                        "improvement_vs_identity": safe_ratio(
                            identity_mse,
                            float(group_frame[f"{method}_masked_mse"].mean()),
                        ),
                        "worse_than_identity_n": int(
                            group_frame[f"{method}_worse_than_identity"].sum()
                        ),
                        "worse_than_identity_fraction": float(
                            group_frame[f"{method}_worse_than_identity"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def comparison_summary(split_name: str, per_sample: pd.DataFrame) -> pd.DataFrame:
    identity_mse = float(per_sample["identity_affected_mse"].mean())
    rows = []
    for method in LEARNED_METHODS:
        affected_mse = float(per_sample[f"{method}_affected_mse"].mean())
        rows.append(
            {
                "split": split_name,
                "method": method,
                "n": int(len(per_sample)),
                "affected_mse": affected_mse,
                "affected_mae": float(per_sample[f"{method}_affected_mae"].mean()),
                "whole_mse": float(per_sample[f"{method}_mse"].mean()),
                "ssim": float(per_sample[f"{method}_ssim"].mean()),
                "improvement_vs_identity": safe_ratio(identity_mse, affected_mse),
                "worse_than_identity_n": int(
                    per_sample[f"{method}_worse_than_identity"].sum()
                ),
                "worse_than_identity_fraction": float(
                    per_sample[f"{method}_worse_than_identity"].mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    balanced_mse = float(
        summary.loc[summary["method"] == "balanced_residual", "affected_mse"].iloc[0]
    )
    old_residual_mse = float(
        summary.loc[summary["method"] == "residual", "affected_mse"].iloc[0]
    )
    direct_mse = float(summary.loc[summary["method"] == "direct", "affected_mse"].iloc[0])
    summary["balanced_residual_beats_old_residual_fraction"] = float(
        per_sample["balanced_beats_residual"].mean()
    )
    summary["balanced_residual_beats_direct_fraction"] = float(
        per_sample["balanced_beats_direct"].mean()
    )
    summary["balanced_to_old_residual_aggregate_mse_ratio"] = safe_ratio(
        balanced_mse, old_residual_mse
    )
    summary["balanced_to_direct_aggregate_mse_ratio"] = safe_ratio(
        balanced_mse, direct_mse
    )
    return summary


def format_metrics_table(aggregate: pd.DataFrame) -> str:
    rows = [
        "| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "identity": "identity",
        "threshold": "threshold",
        "direct": "direct U-Net",
        "residual": "residual U-Net",
        "balanced_residual": "balanced residual U-Net",
    }
    for _, row in aggregate.iterrows():
        rows.append(
            "| {method} | {mse:.6f} | {mae:.6f} | {psnr:.3f} | {ssim:.6f} | "
            "{masked_mse:.6f} | {masked_mae:.6f} | {ratio:.2f}x | {worse}/{n} |".format(
                method=labels.get(row["method"], row["method"]),
                mse=row["mse"],
                mae=row["mae"],
                psnr=row["psnr"],
                ssim=row["ssim"],
                masked_mse=row["masked_mse"],
                masked_mae=row["masked_mae"],
                ratio=row["affected_mse_improvement_vs_identity"],
                worse=int(row["worse_than_identity_n"]),
                n=int(row["n"]),
            )
        )
    return "\n".join(rows)


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    if path.exists():
        raise FileExistsError(f"Figure already exists: {path}")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def method_label(method: str) -> str:
    return {
        "identity": "identity",
        "threshold": "threshold",
        "direct": "direct",
        "residual": "residual",
        "balanced_residual": "balanced residual",
    }.get(method, method)


def affected_mse_bar(
    normal_aggregate: pd.DataFrame,
    stress_aggregate: pd.DataFrame,
    output_dir: Path,
) -> Path:
    colors = ["#747b84", "#b66b5d", "#2f6f8f", "#5f8a4b", "#7b5fa3"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, title, aggregate in (
        (axes[0], "Normal held-out", normal_aggregate),
        (axes[1], "Hard stress", stress_aggregate),
    ):
        frame = aggregate.set_index("method").loc[list(METHODS)]
        labels = [method_label(method) for method in frame.index]
        ax.bar(labels, frame["affected_mse"], color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel("Affected-region MSE")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Affected-Region MSE")
    fig.tight_layout()
    return save_figure(fig, output_dir, "affected_region_mse_bar.png")


def improvement_chart(
    normal_aggregate: pd.DataFrame,
    stress_aggregate: pd.DataFrame,
    output_dir: Path,
) -> Path:
    splits = [("normal held-out", normal_aggregate), ("hard stress", stress_aggregate)]
    learned = ["direct", "residual", "balanced_residual"]
    x = np.arange(len(splits))
    width = 0.24
    colors = ["#2f6f8f", "#5f8a4b", "#7b5fa3"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, method in enumerate(learned):
        values = [
            float(
                aggregate.set_index("method").loc[
                    method, "affected_mse_improvement_vs_identity"
                ]
            )
            for _split, aggregate in splits
        ]
        positions = x + (idx - 1) * width
        ax.bar(positions, values, width, label=method_label(method), color=colors[idx])
        for xpos, value in zip(positions, values):
            ax.text(xpos, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [name for name, _aggregate in splits])
    ax.set_ylabel("Identity affected MSE / model affected MSE")
    ax.set_title("Normal vs Stress Improvement")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, "normal_vs_stress_improvement_ratio.png")


def scatter_compare(
    per_sample: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
) -> Path:
    data = per_sample[[x_col, y_col, "split"]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    colors = {"normal": "#5f8a4b", "stress": "#2f6f8f"}
    for split, frame in data.groupby("split"):
        ax.scatter(
            frame[x_col],
            frame[y_col],
            s=13,
            alpha=0.34,
            color=colors.get(split, "#747b84"),
            label=split,
            edgecolors="none",
        )
    limit = float(np.nanmax([data[x_col].max(), data[y_col].max(), 1e-4]))
    ax.plot([1e-6, limit], [1e-6, limit], color="#8a4f49", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, filename)


def ratio_histogram(per_sample: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for split, color in (("normal", "#5f8a4b"), ("stress", "#2f6f8f")):
        values = (
            per_sample.loc[
                per_sample["split"] == split,
                "balanced_residual_to_residual_mse_ratio",
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        values = values[(values > 0) & np.isfinite(values)]
        if values.empty:
            continue
        clipped = values.clip(
            lower=float(np.nanpercentile(values, 1)),
            upper=float(np.nanpercentile(values, 99)),
        )
        ax.hist(clipped, bins=35, alpha=0.48, color=color, label=split)
    ax.axvline(1.0, color="#8a4f49", linewidth=1.5, label="parity")
    ax.set_title("Balanced Residual / Old Residual Error Ratio")
    ax.set_xlabel("Balanced residual affected MSE / old residual affected MSE")
    ax.set_ylabel("Samples")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, "hist_balanced_to_old_residual_ratio.png")


def worse_than_identity_chart(
    normal_aggregate: pd.DataFrame,
    stress_aggregate: pd.DataFrame,
    output_dir: Path,
) -> Path:
    learned = ["direct", "residual", "balanced_residual"]
    splits = [("normal", normal_aggregate), ("stress", stress_aggregate)]
    x = np.arange(len(splits))
    width = 0.24
    colors = ["#2f6f8f", "#5f8a4b", "#7b5fa3"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for idx, method in enumerate(learned):
        values = [
            int(aggregate.set_index("method").loc[method, "worse_than_identity_n"])
            for _split, aggregate in splits
        ]
        ax.bar(x + (idx - 1) * width, values, width, label=method_label(method), color=colors[idx])
    ax.set_xticks(x, [name for name, _aggregate in splits])
    ax.set_ylabel("Samples worse than identity")
    ax.set_title("Worse-than-Identity Counts")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, "worse_than_identity_counts.png")


def grouped_performance_chart(
    grouped: pd.DataFrame,
    split_name: str,
    grouping: str,
    output_dir: Path,
    filename: str,
) -> Path | None:
    frame = grouped[
        (grouped["split"] == split_name)
        & (grouped["grouping"] == grouping)
        & (grouped["method"].isin(["direct", "residual", "balanced_residual"]))
    ].copy()
    if frame.empty:
        return None
    order = [item for item in ["low", "medium", "high", "easy", "hard"] if item in set(frame["group"])]
    if not order:
        order = sorted(frame["group"].unique())
    methods = ["direct", "residual", "balanced_residual"]
    x = np.arange(len(order))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    colors = ["#2f6f8f", "#5f8a4b", "#7b5fa3"]
    for idx, method in enumerate(methods):
        method_frame = frame[frame["method"] == method].set_index("group")
        values = [
            float(method_frame.loc[group, "affected_mse"])
            if group in method_frame.index
            else np.nan
            for group in order
        ]
        ax.bar(x + (idx - 1) * width, values, width, label=method_label(method), color=colors[idx])
    ax.set_xticks(x, order)
    ax.set_ylabel("Affected-region MSE")
    ax.set_title(f"{split_name.title()} Performance by {grouping.replace('_', ' ')}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, filename)


def predict_single(
    sample: dict[str, Any],
    direct_model: nn.Module,
    residual_model: nn.Module,
    balanced_model: nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct, residual, _balanced_layer, balanced = predict_batch(
        [sample], direct_model, residual_model, balanced_model, device
    )
    return direct[0], residual[0], balanced[0]


def save_example_figure(
    sample: dict[str, Any],
    row: pd.Series,
    direct_model: nn.Module,
    residual_model: nn.Module,
    balanced_model: nn.Module,
    device: torch.device,
    output_dir: Path,
    filename: str,
    title: str,
) -> Path:
    direct, residual, balanced = predict_single(
        sample, direct_model, residual_model, balanced_model, device
    )
    target = sample["target"]
    blended = sample["blended"]
    balanced_error = np.abs(balanced - target).mean(axis=-1)
    residual_error = np.abs(residual - target).mean(axis=-1)
    images = [
        (target, "Target"),
        (blended, "Blend"),
        (direct, "Direct"),
        (residual, "Residual"),
        (balanced, "Balanced residual"),
        (residual_error, "Old residual error"),
        (balanced_error, "Balanced error"),
    ]
    fig, axes = plt.subplots(1, 7, figsize=(19.5, 3.4))
    for ax, (image, panel_title) in zip(axes, images):
        if image.ndim == 2:
            vmax = max(0.2, float(image.max()))
            ax.imshow(image, cmap="magma", vmin=0.0, vmax=vmax)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    metadata = {
        "split": str(row["split"]),
        "index": int(row["index"]),
        "direct_affected_mse": float(row["direct_affected_mse"]),
        "residual_affected_mse": float(row["residual_affected_mse"]),
        "balanced_residual_affected_mse": float(row["balanced_residual_affected_mse"]),
        "blend_severity_bin": str(row["blend_severity_bin"]),
        "core_overlap_bin": str(row["core_overlap_bin"]),
        "generation_difficulty": str(row["generation_difficulty"]),
    }
    save_json(output_dir / f"{Path(filename).stem}_metadata.json", metadata)
    return save_figure(fig, output_dir, filename)


def choose_examples(per_sample: pd.DataFrame) -> dict[str, pd.Series | None]:
    finite = per_sample.replace([np.inf, -np.inf], np.nan)
    examples: dict[str, pd.Series | None] = {
        "balanced_improves": None,
        "balanced_failure": None,
        "old_or_direct_better": None,
    }
    improves = finite[
        (finite["balanced_beats_residual"])
        & (finite["balanced_residual_affected_mse"] < finite["identity_affected_mse"])
    ].copy()
    if not improves.empty:
        examples["balanced_improves"] = improves.sort_values(
            ["balanced_residual_to_residual_mse_ratio", "identity_affected_mse"],
            ascending=[True, False],
        ).iloc[0]

    failures = finite[
        (finite["balanced_residual_worse_than_identity"])
        | (finite["balanced_residual_improvement_ratio"] <= 1.25)
    ].copy()
    if failures.empty:
        failures = finite.copy()
    if not failures.empty:
        examples["balanced_failure"] = failures.sort_values(
            ["balanced_residual_affected_mse", "identity_affected_mse"],
            ascending=False,
        ).iloc[0]

    old_or_direct = finite[
        (finite["residual_beats_balanced"]) | (finite["direct_beats_balanced"])
    ].copy()
    if not old_or_direct.empty:
        old_or_direct["best_prior_mse"] = old_or_direct[
            ["direct_affected_mse", "residual_affected_mse"]
        ].min(axis=1)
        old_or_direct["balanced_to_best_prior_ratio"] = (
            old_or_direct["balanced_residual_affected_mse"]
            / old_or_direct["best_prior_mse"]
        )
        examples["old_or_direct_better"] = old_or_direct.sort_values(
            ["balanced_to_best_prior_ratio", "balanced_residual_affected_mse"],
            ascending=False,
        ).iloc[0]
    return examples


def sample_for_row(
    row: pd.Series,
    normal_samples: list[dict[str, Any]],
    stress_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if row["split"] == "normal":
        return normal_samples[int(row["index"])]
    if row["split"] == "stress":
        return stress_samples[int(row["index"])]
    raise KeyError(f"Unknown split: {row['split']}")


def save_all_figures(
    run_dir: Path,
    normal_aggregate: pd.DataFrame,
    normal_per_sample: pd.DataFrame,
    normal_grouped: pd.DataFrame,
    stress_aggregate: pd.DataFrame,
    stress_per_sample: pd.DataFrame,
    stress_grouped: pd.DataFrame,
    normal_samples: list[dict[str, Any]],
    stress_samples: list[dict[str, Any]],
    direct_model: nn.Module,
    residual_model: nn.Module,
    balanced_model: nn.Module,
    device: torch.device,
) -> dict[str, str | None]:
    output_dir = run_dir / "paper_figures"
    combined = pd.concat([normal_per_sample, stress_per_sample], ignore_index=True)
    grouped = pd.concat(
        [
            normal_grouped.assign(split="normal"),
            stress_grouped.assign(split="stress"),
        ],
        ignore_index=True,
    )
    written: dict[str, str | None] = {}
    written["affected_mse_bar"] = project_relative(
        affected_mse_bar(normal_aggregate, stress_aggregate, output_dir)
    )
    written["improvement_ratio"] = project_relative(
        improvement_chart(normal_aggregate, stress_aggregate, output_dir)
    )
    written["old_residual_vs_balanced_scatter"] = project_relative(
        scatter_compare(
            combined,
            "residual_affected_mse",
            "balanced_residual_affected_mse",
            "Old Residual vs Balanced Residual",
            "Old residual affected MSE",
            "Balanced residual affected MSE",
            output_dir,
            "old_residual_vs_balanced_scatter.png",
        )
    )
    written["direct_vs_balanced_scatter"] = project_relative(
        scatter_compare(
            combined,
            "direct_affected_mse",
            "balanced_residual_affected_mse",
            "Direct vs Balanced Residual",
            "Direct affected MSE",
            "Balanced residual affected MSE",
            output_dir,
            "direct_vs_balanced_scatter.png",
        )
    )
    written["balanced_old_ratio_histogram"] = project_relative(
        ratio_histogram(combined, output_dir)
    )
    written["worse_than_identity_counts"] = project_relative(
        worse_than_identity_chart(normal_aggregate, stress_aggregate, output_dir)
    )
    maybe = grouped_performance_chart(
        grouped,
        "stress",
        "core_overlap_bin",
        output_dir,
        "stress_performance_by_core_overlap_bin.png",
    )
    written["core_overlap_performance"] = project_relative(maybe) if maybe else None
    maybe = grouped_performance_chart(
        grouped,
        "stress",
        "blend_severity_bin",
        output_dir,
        "stress_performance_by_blend_severity_bin.png",
    )
    written["blend_severity_performance"] = project_relative(maybe) if maybe else None

    examples = choose_examples(combined)
    for key, filename, title in (
        (
            "balanced_improves",
            "balanced_residual_improves_over_old_residual.png",
            "Balanced residual improves over old residual",
        ),
        (
            "balanced_failure",
            "balanced_residual_failure_example.png",
            "Balanced residual failure case",
        ),
        (
            "old_or_direct_better",
            "old_or_direct_beats_balanced_residual.png",
            "Old residual or direct still beats balanced residual",
        ),
    ):
        row = examples[key]
        if row is None:
            written[key] = None
            continue
        sample = sample_for_row(row, normal_samples, stress_samples)
        written[key] = project_relative(
            save_example_figure(
                sample,
                row,
                direct_model,
                residual_model,
                balanced_model,
                device,
                output_dir,
                filename,
                title,
            )
        )
    return written


def stop_condition_warnings(
    split_name: str,
    aggregate: pd.DataFrame,
    per_sample: pd.DataFrame,
    stats: dict[str, float],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    stops: list[str] = []
    frame = aggregate.set_index("method")
    identity = frame.loc["identity"]
    balanced = frame.loc["balanced_residual"]
    residual = frame.loc["residual"]
    if balanced["masked_mse"] > identity["masked_mse"]:
        stops.append(
            f"{split_name}: balanced residual is worse than identity on aggregate affected-region MSE"
        )
    if stats["residual_pred_abs_max"] > 5.0:
        stops.append(f"{split_name}: balanced residual predictions exceeded absolute value 5")
    if stats["reconstruction_preclip_total_clip_fraction"] > 0.20:
        stops.append(
            f"{split_name}: more than 20% of balanced reconstruction pixels needed clipping"
        )
    if stats["reconstruction_preclip_total_clip_fraction"] > 0.05:
        warnings.append(
            f"{split_name}: more than 5% of balanced reconstruction pixels needed clipping"
        )
    if balanced["masked_mse"] > residual["masked_mse"] * 1.5:
        warnings.append(
            f"{split_name}: balanced residual affected MSE is more than 50% worse than old residual"
        )
    if per_sample["mask_fraction"].mean() < 0.005:
        stops.append(f"{split_name}: affected mask fraction is suspiciously low")
    return warnings, stops


def write_diagnostics(
    run_dir: Path,
    warnings: Iterable[str],
    stops: Iterable[str],
    normal_per_sample: pd.DataFrame | None = None,
    stress_per_sample: pd.DataFrame | None = None,
) -> None:
    warnings = list(warnings)
    stops = list(stops)
    if not warnings and not stops:
        return
    with (run_dir / "diagnostics" / "balanced_residual_diagnostics.md").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("# Balanced Residual Diagnostics\n\n")
        if stops:
            handle.write("## Stop Conditions\n\n")
            for item in stops:
                handle.write(f"- {item}\n")
            handle.write("\n")
        if warnings:
            handle.write("## Warnings\n\n")
            for item in warnings:
                handle.write(f"- {item}\n")
    if normal_per_sample is not None:
        normal_per_sample.sort_values(
            "balanced_residual_to_residual_mse_ratio", ascending=False
        ).head(25).to_csv(
            run_dir / "diagnostics" / "normal_largest_balanced_regressions.csv",
            index=False,
        )
    if stress_per_sample is not None:
        stress_per_sample.sort_values(
            "balanced_residual_to_residual_mse_ratio", ascending=False
        ).head(25).to_csv(
            run_dir / "diagnostics" / "stress_largest_balanced_regressions.csv",
            index=False,
        )


def append_experiment_log(
    run_dir: Path,
    best_checkpoint: Path,
    final_checkpoint: Path,
    settings: BalancedSettings,
    composition: dict[str, Any],
    history: pd.DataFrame,
    normal_aggregate: pd.DataFrame,
    normal_summary: pd.DataFrame,
    stress_aggregate: pd.DataFrame,
    stress_summary: pd.DataFrame,
    warnings: list[str],
    stops: list[str],
    figure_paths: dict[str, str | None],
) -> None:
    status = "coherent" if not stops else "suspicious; inspect diagnostics"
    warning_text = "None." if not warnings and not stops else "; ".join(stops + warnings) + "."
    best_epoch = int(history["best_epoch"].iloc[-1])
    best_val = float(history["best_val_loss"].iloc[-1])
    final_train = float(history["train_loss"].iloc[-1])
    final_val = float(history["val_loss"].iloc[-1])
    normal_frame = normal_summary.set_index("method")
    stress_frame = stress_summary.set_index("method")
    normal_balanced = normal_frame.loc["balanced_residual"]
    normal_residual = normal_frame.loc["residual"]
    stress_balanced = stress_frame.loc["balanced_residual"]
    stress_residual = stress_frame.loc["residual"]
    stress_direct = stress_frame.loc["direct"]
    figure_lines = [
        f"- `{path}`" for path in figure_paths.values() if path is not None
    ] or ["- Not generated before stop condition."]

    section = f"""

## Experiment 3: Balanced Hard-Case Residual U-Net

Status: {status}. Run directory: `{project_relative(run_dir)}`.

### Training Composition

- Task: residual prediction, `blended -> blended - target`.
- Reconstruction: `blended - predicted_residual`.
- Train/validation blends: {settings.n_train_blends:,} / {settings.n_val_blends:,}.
- Normal held-out/stress test blends: {settings.n_normal_test_blends:,} / {settings.n_stress_blends:,}.
- Epochs: {settings.num_epochs}.
- Composition target: 50% normal/random, 30% high-overlap/core-obstruction, 20% brightness/size stress.
- Actual training composition: {json.dumps(composition["train"]["components"])}.
- Saved best model checkpoint: `{project_relative(best_checkpoint)}`.
- Saved final model checkpoint: `{project_relative(final_checkpoint)}`.
- Best validation loss: {best_val:.6f} at epoch {best_epoch}.
- Final train/validation loss: {final_train:.6f} / {final_val:.6f}.

### Normal Held-Out Metrics

{format_metrics_table(normal_aggregate)}

Balanced residual vs old residual on normal aggregate:

- Old residual affected MSE: {normal_residual["affected_mse"]:.6f}.
- Balanced residual affected MSE: {normal_balanced["affected_mse"]:.6f}.
- Balanced beats old residual cases: {normal_frame.loc["balanced_residual", "balanced_residual_beats_old_residual_fraction"]:.1%}.

### Hard Stress-Test Metrics

{format_metrics_table(stress_aggregate)}

Balanced residual vs previous models on hard stress:

- Direct affected MSE: {stress_direct["affected_mse"]:.6f}.
- Old residual affected MSE: {stress_residual["affected_mse"]:.6f}.
- Balanced residual affected MSE: {stress_balanced["affected_mse"]:.6f}.
- Balanced beats old residual cases: {stress_frame.loc["balanced_residual", "balanced_residual_beats_old_residual_fraction"]:.1%}.
- Balanced beats direct cases: {stress_frame.loc["balanced_residual", "balanced_residual_beats_direct_fraction"]:.1%}.
- Balanced worse-than-identity cases: {int(stress_balanced["worse_than_identity_n"])}/{int(stress_balanced["n"])}.

### Interpretation

Stress robustness {'improved' if stress_balanced["affected_mse"] < stress_residual["affected_mse"] else 'did not improve'} relative to the older residual saved model. Normal performance {'degraded' if normal_balanced["affected_mse"] > normal_residual["affected_mse"] else 'did not degrade'} by aggregate affected-region MSE. Direct and old residual can still win on individual cases, so this experiment should be read as a distribution-shaping test rather than a universal replacement.

Coherence/suspicion status: {warning_text}

Paper figures:

{chr(10).join(figure_lines)}
"""
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(section)


def update_paper_plan() -> None:
    text = PAPER_PLAN_PATH.read_text(encoding="utf-8")
    marker = "## Experiment 3 Addition"
    if marker in text or "### Balanced Residual Improvement" in text:
        return
    addition = """

## Experiment 3 Addition

The paper arc now includes a balanced hard-case residual U-Net:

- Direct U-Net baseline tests whether a compact model can recover targets on
  normal controlled blends.
- Residual U-Net tests whether predicting the contaminant layer improves
  reconstruction, especially on stress cases.
- Balanced hard-case residual U-Net tests whether changing the training
  distribution toward normal + high-overlap/core-obstruction + brightness/size
  stress blends improves stress robustness without sacrificing too much normal
  performance.

The balanced residual run should be reported as a distribution-shaping experiment. If it
improves stress metrics but hurts normal metrics, the paper should state that
tradeoff directly and motivate balanced or weighted objectives rather than
claiming a universal model improvement.
"""
    PAPER_PLAN_PATH.write_text(text.rstrip() + addition, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    settings = BalancedSettings(
        n_train_blends=args.n_train_blends,
        n_val_blends=args.n_val_blends,
        n_normal_test_blends=args.n_normal_test_blends,
        n_stress_blends=args.n_stress_blends,
        train_source_subset=args.train_source_subset,
        val_source_subset=args.val_source_subset,
        test_source_subset=args.test_source_subset,
        num_epochs=args.num_epochs,
    )
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, best_checkpoint, final_checkpoint = make_run_paths(output_root, stamp)
    save_yaml(
        run_dir / "logs" / "run_config.yaml",
        {
            "project_root": ".",
            "timestamp": stamp,
            "settings": settings.__dict__,
            "config": config,
            "direct_checkpoint": project_relative(args.direct_checkpoint),
            "residual_checkpoint": project_relative(args.residual_checkpoint),
            "stress_settings": {
                **stress_helpers.STRESS_DEFAULTS,
                "n_stress_blends": settings.n_stress_blends,
            },
        },
    )

    direct_checkpoint = args.direct_checkpoint
    residual_checkpoint = args.residual_checkpoint
    if not direct_checkpoint.is_absolute():
        direct_checkpoint = PROJECT_ROOT / direct_checkpoint
    if not residual_checkpoint.is_absolute():
        residual_checkpoint = PROJECT_ROOT / residual_checkpoint
    direct_checkpoint = direct_checkpoint.resolve()
    residual_checkpoint = residual_checkpoint.resolve()

    stops: list[str] = []
    warnings: list[str] = []
    for path, label in (
        (direct_checkpoint, "direct checkpoint"),
        (residual_checkpoint, "residual checkpoint"),
    ):
        if not path.exists():
            stops.append(f"Missing {label}: {project_relative(path)}")
    if stops:
        write_diagnostics(run_dir, warnings, stops)
        print("\n".join(stops))
        return 2

    direct_before = checkpoint_info(direct_checkpoint)
    residual_before = checkpoint_info(residual_checkpoint)
    save_json(
        run_dir / "logs" / "protected_checkpoint_metadata_before.json",
        {"direct": direct_before, "residual": residual_before},
    )

    seed = int(config["seed"])
    seed_everything(seed)
    device_status = {
        "requested_device": str(args.device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_fallback_allowed": False,
    }
    try:
        device = gd_train.resolve_accelerator(args.device)
    except RuntimeError as exc:
        save_json(
            run_dir / "logs" / "device_selection.json",
            {**device_status, "status": "stopped", "error": str(exc)},
        )
        stops.append(str(exc))
        write_diagnostics(run_dir, warnings, stops)
        print(f"Stopping before full training/evaluation: {exc}", flush=True)
        return 2
    save_json(
        run_dir / "logs" / "device_selection.json",
        {**device_status, "status": "selected", "selected_device": str(device)},
    )
    print(f"Using device: {device}", flush=True)

    try:
        direct_model = load_direct_model(direct_checkpoint, config["model"], device)
        residual_model = load_residual_model(residual_checkpoint, config["model"], device)
    except Exception as exc:
        stops.append(f"Could not load protected checkpoints: {exc}")
        write_diagnostics(run_dir, warnings, stops)
        print(stops[-1])
        return 2

    print("Loading dataset and reconstructing splits.", flush=True)
    train_images, val_images, test_images = load_split_subsets(config, settings)

    print("Generating balanced training blends.", flush=True)
    train_blends, train_composition = generate_balanced_blends(
        train_images,
        total=settings.n_train_blends,
        config=config,
        seed=seed + 1000,
        settings=settings,
        split_name="train",
    )
    print("Generating balanced validation blends.", flush=True)
    val_blends, val_composition = generate_balanced_blends(
        val_images,
        total=settings.n_val_blends,
        config=config,
        seed=seed + 2000,
        settings=settings,
        split_name="val",
    )
    composition = {"train": train_composition, "validation": val_composition}
    save_json(run_dir / "logs" / "training_composition.json", composition)

    component_rows = []
    for split, payload in composition.items():
        for component in payload["components"]:
            component_rows.append({"split": split, **component})
    pd.DataFrame(component_rows).to_csv(
        run_dir / "results" / "training_composition.csv", index=False
    )
    for component in component_rows:
        mean_mask = component.get("mean_mask_fraction")
        if mean_mask is not None and mean_mask < 0.005:
            stops.append(
                f"{component['split']} {component['component']}: suspiciously low mean mask fraction {mean_mask:.6f}"
            )
    if stops:
        write_diagnostics(run_dir, warnings, stops)
        print("Stopped before training due to suspicious blend generation.")
        return 3

    train_ds = ResidualBlendDataset(train_blends)
    val_ds = ResidualBlendDataset(val_blends)
    requested_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["training"].get("batch_size", 8))
    )

    print("Training balanced residual U-Net.", flush=True)
    try:
        final_model, best_state, history, used_batch_size, attempts = train_with_memory_retry(
            train_ds,
            val_ds,
            model_config=config["model"],
            training_config=config["training"],
            settings=settings,
            requested_batch_size=requested_batch_size,
            device=device,
            seed=seed + 3000,
        )
    except Exception as exc:
        stops.append(f"Training failed: {exc}")
        write_diagnostics(run_dir, warnings, stops)
        print(stops[-1])
        return 3

    history.to_csv(run_dir / "results" / "training_history.csv", index=False)
    save_json(run_dir / "logs" / "training_attempts.json", attempts)
    final_state = {
        key: value.detach().cpu().clone()
        for key, value in final_model.state_dict().items()
    }
    save_checkpoint(
        best_checkpoint,
        best_state,
        config,
        settings,
        history,
        used_batch_size,
        stamp,
        "best_validation",
    )
    save_checkpoint(
        final_checkpoint,
        final_state,
        config,
        settings,
        history,
        used_batch_size,
        stamp,
        "final_epoch",
    )
    balanced_model = make_residual_unet(config["model"])
    balanced_model.load_state_dict(best_state)
    balanced_model.to(device)
    balanced_model.eval()
    print(f"Saved best checkpoint: {project_relative(best_checkpoint)}", flush=True)
    print(f"Saved final checkpoint: {project_relative(final_checkpoint)}", flush=True)

    del train_blends, val_blends, train_ds, val_ds, train_images, val_images, final_model
    clear_torch_cache()
    gc.collect()

    print("Generating normal held-out test blends.", flush=True)
    normal_test_blends = normal_blends(
        test_images,
        settings.n_normal_test_blends,
        config,
        seed=seed + 4000,
        component="normal_test",
    )
    print("Evaluating normal held-out test set.", flush=True)
    normal_aggregate, normal_per_sample, normal_grouped, normal_stats = evaluate_samples(
        "normal",
        normal_test_blends,
        direct_model,
        residual_model,
        balanced_model,
        device,
        settings.affected_region_threshold,
        used_batch_size,
    )
    normal_summary = comparison_summary("normal", normal_per_sample)
    normal_aggregate.to_csv(run_dir / "results" / "normal_results.csv", index=False)
    normal_per_sample.to_csv(
        run_dir / "results" / "normal_per_sample_results.csv", index=False
    )
    normal_grouped.to_csv(
        run_dir / "results" / "normal_grouped_results.csv", index=False
    )
    normal_summary.to_csv(
        run_dir / "results" / "normal_model_comparison.csv", index=False
    )
    save_json(
        run_dir / "diagnostics" / "normal_balanced_residual_output_stats.json",
        normal_stats,
    )
    split_warnings, split_stops = stop_condition_warnings(
        "normal", normal_aggregate, normal_per_sample, normal_stats
    )
    warnings.extend(split_warnings)
    stops.extend(split_stops)

    print("Generating hard stress-test blends.", flush=True)
    stress_settings = dict(stress_helpers.STRESS_DEFAULTS)
    stress_settings["n_stress_blends"] = settings.n_stress_blends
    stress_images = test_images[: int(stress_settings["stress_source_subset"])]
    stress_blends = stress_helpers.generate_stress_blends(stress_images, stress_settings)
    print("Evaluating hard stress-test set.", flush=True)
    stress_aggregate, stress_per_sample, stress_grouped, stress_stats = evaluate_samples(
        "stress",
        stress_blends,
        direct_model,
        residual_model,
        balanced_model,
        device,
        float(stress_settings["affected_region_threshold"]),
        used_batch_size,
    )
    stress_summary = comparison_summary("stress", stress_per_sample)
    stress_aggregate.to_csv(run_dir / "results" / "stress_results.csv", index=False)
    stress_per_sample.to_csv(
        run_dir / "results" / "stress_per_sample_results.csv", index=False
    )
    stress_grouped.to_csv(
        run_dir / "results" / "stress_grouped_results.csv", index=False
    )
    stress_summary.to_csv(
        run_dir / "results" / "stress_model_comparison.csv", index=False
    )
    pd.concat([normal_summary, stress_summary], ignore_index=True).to_csv(
        run_dir / "results" / "model_comparison.csv", index=False
    )
    save_json(
        run_dir / "diagnostics" / "stress_balanced_residual_output_stats.json",
        stress_stats,
    )
    split_warnings, split_stops = stop_condition_warnings(
        "stress", stress_aggregate, stress_per_sample, stress_stats
    )
    warnings.extend(split_warnings)
    stops.extend(split_stops)

    normal_balanced = normal_aggregate.set_index("method").loc["balanced_residual"]
    normal_old = normal_aggregate.set_index("method").loc["residual"]
    stress_balanced = stress_aggregate.set_index("method").loc["balanced_residual"]
    stress_old = stress_aggregate.set_index("method").loc["residual"]
    if (
        normal_balanced["masked_mse"] > normal_old["masked_mse"] * 1.5
        and stress_balanced["masked_mse"] > stress_old["masked_mse"] * 1.5
    ):
        stops.append(
            "balanced residual is much worse than old residual on both normal and stress affected-region MSE"
        )

    write_diagnostics(run_dir, warnings, stops, normal_per_sample, stress_per_sample)

    figure_paths: dict[str, str | None] = {}
    if not stops:
        print("Generating paper figures.", flush=True)
        figure_paths = save_all_figures(
            run_dir,
            normal_aggregate,
            normal_per_sample,
            normal_grouped,
            stress_aggregate,
            stress_per_sample,
            stress_grouped,
            normal_test_blends,
            stress_blends,
            direct_model,
            residual_model,
            balanced_model,
            device,
        )

    if not args.skip_doc_update:
        append_experiment_log(
            run_dir,
            best_checkpoint,
            final_checkpoint,
            settings,
            composition,
            history,
            normal_aggregate,
            normal_summary,
            stress_aggregate,
            stress_summary,
            warnings,
            stops,
            figure_paths,
        )
        update_paper_plan()

    integrity = {
        "direct": verify_checkpoint_unchanged(direct_before, direct_checkpoint),
        "residual": verify_checkpoint_unchanged(residual_before, residual_checkpoint),
    }
    save_json(run_dir / "logs" / "protected_checkpoint_integrity.json", integrity)
    if not integrity["direct"]["unchanged"] or not integrity["residual"]["unchanged"]:
        raise RuntimeError("Protected checkpoint metadata changed during the run.")

    print(f"Balanced residual run directory: {project_relative(run_dir)}")
    print(f"Best checkpoint: {project_relative(best_checkpoint)}")
    print(f"Final checkpoint: {project_relative(final_checkpoint)}")
    print("Normal metrics:")
    print(format_metrics_table(normal_aggregate))
    print("Stress metrics:")
    print(format_metrics_table(stress_aggregate))
    if warnings or stops:
        print("Diagnostics:")
        for item in stops + warnings:
            print(f"- {item}")
    return 0 if not stops else 3


if __name__ == "__main__":
    raise SystemExit(main())
