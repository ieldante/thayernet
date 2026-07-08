"""Train and evaluate a residual-prediction U-Net.

This script is intentionally output-safe:

- the existing direct U-Net checkpoint is read but never modified;
- residual checkpoints include a timestamp in the filename;
- every evaluation run writes into a new timestamped directory;
- suspicious results are written as diagnostics instead of being papered over.
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
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "docs" / "experiment_log.md"

NORMAL_BASELINE_DIRECT_AFFECTED_MSE = 0.004428
NORMAL_BASELINE_IDENTITY_AFFECTED_MSE = 0.062555
STRESS_BASELINE_DIRECT_AFFECTED_MSE = 0.009390
STRESS_BASELINE_IDENTITY_AFFECTED_MSE = 0.075541


@dataclass(frozen=True)
class ExperimentSettings:
    """First-pass residual experiment settings."""

    n_train_blends: int = 5000
    n_val_blends: int = 800
    n_test_blends: int = 800
    train_subset: int = 5000
    val_subset: int = 800
    test_subset: int = 800
    num_epochs: int = 20
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
        description="Train and evaluate residual-prediction U-Net deblending."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--direct-checkpoint",
        type=Path,
        default=DEFAULT_DIRECT_CHECKPOINT,
        help="Existing direct U-Net checkpoint to read for comparison.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-train-blends", type=int, default=5000)
    parser.add_argument("--n-val-blends", type=int, default=800)
    parser.add_argument("--n-test-blends", type=int, default=800)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--val-subset", type=int, default=800)
    parser.add_argument("--test-subset", type=int, default=800)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-stress-blends", type=int, default=1000)
    parser.add_argument(
        "--skip-log-update",
        action="store_true",
        help="Do not append docs/experiment_log.md.",
    )
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


def make_run_paths(
    output_root: Path,
    settings: ExperimentSettings,
) -> tuple[str, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / f"residual_unet_{stamp}"
    checkpoint_path = (
        output_root
        / "checkpoints"
        / (
            "unet_residual_"
            f"{settings.n_train_blends}train_"
            f"{settings.n_val_blends}val_"
            f"{settings.n_test_blends}test_"
            f"{settings.num_epochs}ep_{stamp}.pth"
        )
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint path already exists: {checkpoint_path}")
    for child in ("results", "figures", "paper_figures", "diagnostics", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return stamp, run_dir, checkpoint_path


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_residual_unet(model_config: dict[str, Any]) -> models.UNet:
    model = models.UNet(**model_config)
    model.out_activation = nn.Identity()
    return model


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state_dict.")
    return state


def load_direct_model(
    checkpoint_path: Path,
    model_config: dict[str, Any],
    device: torch.device,
) -> models.UNet:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = models.UNet(**model_config)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def is_memory_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def clear_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def load_split_subsets(
    config: dict[str, Any],
    settings: ExperimentSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the dataset and return the same source subsets as the notebook."""
    data_path = PROJECT_ROOT / config["dataset_path"]
    images_raw, labels, _metadata = gd_data.load_galaxy10(data_path)
    (train, val, test) = gd_data.split_dataset(
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

    train_images = gd_data.normalise_images(train_images_raw[: settings.train_subset])
    val_images = gd_data.normalise_images(val_images_raw[: settings.val_subset])
    test_images = gd_data.normalise_images(test_images_raw[: settings.test_subset])

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


def generate_normal_blends(
    train_images: np.ndarray,
    val_images: np.ndarray,
    test_images: np.ndarray,
    config: dict[str, Any],
    settings: ExperimentSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(config["seed"]))
    params = blend_params_from_config(config)
    train_blends = gd_blend.generate_blends(
        train_images,
        n_blends=settings.n_train_blends,
        rng=rng,
        **params,
    )
    val_blends = gd_blend.generate_blends(
        val_images,
        n_blends=settings.n_val_blends,
        rng=rng,
        **params,
    )
    test_blends = gd_blend.generate_blends(
        test_images,
        n_blends=settings.n_test_blends,
        rng=rng,
        **params,
    )
    return train_blends, val_blends, test_blends


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
) -> tuple[nn.Module, pd.DataFrame]:
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

    rows: list[dict[str, float | int]] = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for blended, residual in train_loader:
            blended = blended.to(device)
            residual = residual.to(device)
            optimiser.zero_grad()
            output = model(blended)
            loss = criterion(output, residual)
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
                val_loss += float(loss.item()) * blended.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "batch_size": batch_size,
            }
        )
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"train residual loss={train_loss:.6f}, val residual loss={val_loss:.6f}",
            flush=True,
        )

    return model, pd.DataFrame(rows)


def train_with_memory_retry(
    train_ds: Dataset,
    val_ds: Dataset,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    settings: ExperimentSettings,
    requested_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, pd.DataFrame, int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    batch_size = requested_batch_size
    while batch_size >= 1:
        seed_everything(seed)
        model = make_residual_unet(model_config)
        try:
            trained_model, history = train_one_attempt(
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
            return trained_model, history, batch_size, attempts
        except RuntimeError as exc:
            if not is_memory_error(exc) or batch_size == 1:
                attempts.append(
                    {
                        "batch_size": batch_size,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                raise
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "memory_failed",
                    "error": str(exc),
                }
            )
            print(
                f"Memory failure at batch size {batch_size}; retrying at "
                f"{max(1, batch_size // 2)}.",
                flush=True,
            )
            del model
            clear_torch_cache()
            gc.collect()
            batch_size = max(1, batch_size // 2)
    raise RuntimeError("Could not train residual U-Net with any batch size.")


def metric_values(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    whole = gd_utils.compute_metrics(
        pred,
        target,
        metrics=("mse", "mae", "psnr", "ssim"),
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


def empty_residual_stats() -> dict[str, float]:
    return {
        "residual_pred_min": float("inf"),
        "residual_pred_max": float("-inf"),
        "residual_pred_abs_max": 0.0,
        "residual_pred_mean_sum": 0.0,
        "residual_pred_sq_sum": 0.0,
        "reconstruction_preclip_low_count": 0.0,
        "reconstruction_preclip_high_count": 0.0,
        "pixel_count": 0.0,
    }


def update_residual_stats(
    stats: dict[str, float],
    residual_pred: np.ndarray,
    reconstruction_preclip: np.ndarray,
) -> None:
    stats["residual_pred_min"] = min(
        stats["residual_pred_min"],
        float(np.min(residual_pred)),
    )
    stats["residual_pred_max"] = max(
        stats["residual_pred_max"],
        float(np.max(residual_pred)),
    )
    stats["residual_pred_abs_max"] = max(
        stats["residual_pred_abs_max"],
        float(np.max(np.abs(residual_pred))),
    )
    stats["residual_pred_mean_sum"] += float(np.sum(residual_pred))
    stats["residual_pred_sq_sum"] += float(np.sum(residual_pred**2))
    stats["reconstruction_preclip_low_count"] += float(
        np.count_nonzero(reconstruction_preclip < 0.0)
    )
    stats["reconstruction_preclip_high_count"] += float(
        np.count_nonzero(reconstruction_preclip > 1.0)
    )
    stats["pixel_count"] += float(residual_pred.size)


def finalise_residual_stats(stats: dict[str, float]) -> dict[str, float]:
    pixel_count = max(stats["pixel_count"], 1.0)
    mean = stats["residual_pred_mean_sum"] / pixel_count
    variance = max(stats["residual_pred_sq_sum"] / pixel_count - mean**2, 0.0)
    low_fraction = stats["reconstruction_preclip_low_count"] / pixel_count
    high_fraction = stats["reconstruction_preclip_high_count"] / pixel_count
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


def predict_batch(
    direct_model: nn.Module,
    residual_model: nn.Module,
    batch_samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.stack([sample["blended"] for sample in batch_samples], axis=0)
    tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
    with torch.no_grad():
        direct = direct_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
        residual = (
            residual_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
        )
    residual_recon_preclip = inputs - residual
    residual_recon = np.clip(residual_recon_preclip, 0.0, 1.0).astype(np.float32)
    direct = np.clip(direct, 0.0, 1.0).astype(np.float32)
    return direct, residual, residual_recon


def evaluate_samples(
    split_name: str,
    samples: list[dict[str, Any]],
    direct_model: nn.Module,
    residual_model: nn.Module,
    device: torch.device,
    affected_region_threshold: float,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    stats = empty_residual_stats()
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        direct_preds, residual_preds, residual_recons = predict_batch(
            direct_model,
            residual_model,
            batch_samples,
            device,
        )
        batch_blends = np.stack([sample["blended"] for sample in batch_samples], axis=0)
        residual_recon_preclip = batch_blends - residual_preds
        update_residual_stats(stats, residual_preds, residual_recon_preclip)

        for offset, sample in enumerate(batch_samples):
            index = start + offset
            target = sample["target"]
            blended = sample["blended"]
            threshold_pred = baselines.threshold_baseline(blended)
            direct_pred = direct_preds[offset]
            residual_pred = residual_recons[offset]
            mask = gd_utils.affected_region_mask(
                target,
                blended,
                threshold=affected_region_threshold,
            )
            row: dict[str, Any] = {
                "split": split_name,
                "index": index,
                "mask_fraction": float(mask.mean()),
            }
            info = sample.get("info", {})
            shift = info.get("shift", (0, 0))
            row.update(
                {
                    "generation_difficulty": info.get(
                        "generation_difficulty",
                        info.get("difficulty"),
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
                    "core_obstruction_fraction": info.get(
                        "core_obstruction_fraction",
                        np.nan,
                    ),
                }
            )
            metric_map = {
                "identity": metric_values(blended, target, mask),
                "threshold": metric_values(threshold_pred, target, mask),
                "direct": metric_values(direct_pred, target, mask),
                "residual": metric_values(residual_pred, target, mask),
            }
            for method, values in metric_map.items():
                for metric_name, value in values.items():
                    row[f"{method}_{metric_name}"] = value
                row[f"{method}_affected_mse"] = values["masked_mse"]
                row[f"{method}_affected_mae"] = values["masked_mae"]
                row[f"{method}_improvement_ratio"] = safe_ratio(
                    metric_map["identity"]["masked_mse"],
                    values["masked_mse"],
                )
            row["residual_beats_direct"] = (
                row["residual_affected_mse"] < row["direct_affected_mse"]
            )
            row["direct_beats_residual"] = (
                row["direct_affected_mse"] < row["residual_affected_mse"]
            )
            row["direct_worse_than_identity"] = (
                row["direct_affected_mse"] > row["identity_affected_mse"]
            )
            row["residual_worse_than_identity"] = (
                row["residual_affected_mse"] > row["identity_affected_mse"]
            )
            row["residual_direct_affected_mse_ratio"] = safe_ratio(
                row["residual_affected_mse"],
                row["direct_affected_mse"],
            )
            row["direct_residual_affected_mse_ratio"] = safe_ratio(
                row["direct_affected_mse"],
                row["residual_affected_mse"],
            )
            rows.append(row)

    per_sample = pd.DataFrame(rows)
    aggregate = aggregate_metrics(per_sample)
    return aggregate, per_sample, finalise_residual_stats(stats)


def aggregate_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity_mse = float(per_sample["identity_masked_mse"].mean())
    for method in ("identity", "threshold", "direct", "residual"):
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
            identity_mse,
            row["masked_mse"],
        )
        if method in {"direct", "residual"}:
            worse = per_sample[f"{method}_worse_than_identity"]
            row["worse_than_identity_n"] = int(worse.sum())
            row["worse_than_identity_fraction"] = float(worse.mean())
        else:
            row["worse_than_identity_n"] = 0
            row["worse_than_identity_fraction"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_summary(split_name: str, per_sample: pd.DataFrame) -> pd.DataFrame:
    identity_mse = float(per_sample["identity_affected_mse"].mean())
    direct_mse = float(per_sample["direct_affected_mse"].mean())
    residual_mse = float(per_sample["residual_affected_mse"].mean())
    rows = [
        {
            "split": split_name,
            "n": int(len(per_sample)),
            "identity_affected_mse": identity_mse,
            "direct_affected_mse": direct_mse,
            "residual_affected_mse": residual_mse,
            "identity_affected_mae": float(per_sample["identity_affected_mae"].mean()),
            "direct_affected_mae": float(per_sample["direct_affected_mae"].mean()),
            "residual_affected_mae": float(
                per_sample["residual_affected_mae"].mean()
            ),
            "identity_whole_mse": float(per_sample["identity_mse"].mean()),
            "direct_whole_mse": float(per_sample["direct_mse"].mean()),
            "residual_whole_mse": float(per_sample["residual_mse"].mean()),
            "identity_ssim": float(per_sample["identity_ssim"].mean()),
            "direct_ssim": float(per_sample["direct_ssim"].mean()),
            "residual_ssim": float(per_sample["residual_ssim"].mean()),
            "direct_improvement_vs_identity": safe_ratio(identity_mse, direct_mse),
            "residual_improvement_vs_identity": safe_ratio(
                identity_mse,
                residual_mse,
            ),
            "direct_worse_than_identity_n": int(
                per_sample["direct_worse_than_identity"].sum()
            ),
            "direct_worse_than_identity_fraction": float(
                per_sample["direct_worse_than_identity"].mean()
            ),
            "residual_worse_than_identity_n": int(
                per_sample["residual_worse_than_identity"].sum()
            ),
            "residual_worse_than_identity_fraction": float(
                per_sample["residual_worse_than_identity"].mean()
            ),
            "residual_beats_direct_n": int(per_sample["residual_beats_direct"].sum()),
            "residual_beats_direct_fraction": float(
                per_sample["residual_beats_direct"].mean()
            ),
            "direct_beats_residual_n": int(per_sample["direct_beats_residual"].sum()),
            "direct_beats_residual_fraction": float(
                per_sample["direct_beats_residual"].mean()
            ),
            "aggregate_residual_to_direct_affected_mse_ratio": safe_ratio(
                residual_mse,
                direct_mse,
            ),
        }
    ]
    return pd.DataFrame(rows)


def format_metrics_table(aggregate: pd.DataFrame) -> str:
    rows = [
        "| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in aggregate.iterrows():
        rows.append(
            "| {method} | {mse:.6f} | {mae:.6f} | {psnr:.3f} | {ssim:.6f} | "
            "{masked_mse:.6f} | {masked_mae:.6f} | {ratio:.2f}x | {worse}/{n} |".format(
                method=row["method"],
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


def format_comparison_table(summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    return "\n".join(
        [
            "| Metric | Direct U-Net | Residual U-Net |",
            "| --- | ---: | ---: |",
            f"| Affected-region MSE | {row['direct_affected_mse']:.6f} | {row['residual_affected_mse']:.6f} |",
            f"| Affected-region MAE | {row['direct_affected_mae']:.6f} | {row['residual_affected_mae']:.6f} |",
            f"| Whole-image MSE | {row['direct_whole_mse']:.6f} | {row['residual_whole_mse']:.6f} |",
            f"| SSIM | {row['direct_ssim']:.6f} | {row['residual_ssim']:.6f} |",
            f"| Improvement vs identity | {row['direct_improvement_vs_identity']:.2f}x | {row['residual_improvement_vs_identity']:.2f}x |",
            f"| Worse than identity | {int(row['direct_worse_than_identity_n'])}/{int(row['n'])} | {int(row['residual_worse_than_identity_n'])}/{int(row['n'])} |",
            f"| Residual beats direct | - | {int(row['residual_beats_direct_n'])}/{int(row['n'])} ({row['residual_beats_direct_fraction']:.1%}) |",
        ]
    )


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    if path.exists():
        raise FileExistsError(f"Figure path already exists: {path}")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def affected_mse_bar(
    normal_aggregate: pd.DataFrame,
    stress_aggregate: pd.DataFrame,
    output_dir: Path,
) -> Path:
    methods = ["identity", "threshold", "direct", "residual"]
    colors = ["#747b84", "#b66b5d", "#2f6f8f", "#5f8a4b"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, title, aggregate in (
        (axes[0], "Normal held-out", normal_aggregate),
        (axes[1], "Hard stress", stress_aggregate),
    ):
        frame = aggregate.set_index("method").loc[methods]
        ax.bar(methods, frame["affected_mse"], color=colors, width=0.62)
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
    rows: list[dict[str, Any]] = []
    for split_name, aggregate in (
        ("normal held-out", normal_aggregate),
        ("hard stress", stress_aggregate),
    ):
        frame = aggregate.set_index("method")
        for method in ("direct", "residual"):
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "ratio": float(
                        frame.loc[method, "affected_mse_improvement_vs_identity"]
                    ),
                }
            )
    data = pd.DataFrame(rows)
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    direct = data[data["method"] == "direct"]["ratio"].to_numpy()
    residual = data[data["method"] == "residual"]["ratio"].to_numpy()
    ax.bar(x - width / 2, direct, width, label="direct", color="#2f6f8f")
    ax.bar(x + width / 2, residual, width, label="residual", color="#5f8a4b")
    ax.set_xticks(x, ["normal held-out", "hard stress"])
    ax.set_ylabel("Identity affected MSE / model affected MSE")
    ax.set_title("Normal vs Stress Improvement")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for positions, values in ((x - width / 2, direct), (x + width / 2, residual)):
        for xpos, value in zip(positions, values):
            ax.text(xpos, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)
    return save_figure(fig, output_dir, "normal_vs_stress_improvement_ratio.png")


def direct_vs_residual_scatter(
    normal_per_sample: pd.DataFrame,
    stress_per_sample: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=False, sharey=False)
    for ax, title, frame in (
        (axes[0], "Normal held-out", normal_per_sample),
        (axes[1], "Hard stress", stress_per_sample),
    ):
        data = frame[
            ["direct_affected_mse", "residual_affected_mse"]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(
            data["direct_affected_mse"],
            data["residual_affected_mse"],
            s=13,
            alpha=0.36,
            color="#5f8a4b",
            edgecolors="none",
        )
        limit = float(
            np.nanmax(
                [
                    data["direct_affected_mse"].max(),
                    data["residual_affected_mse"].max(),
                ]
            )
        )
        limit = max(limit, 1e-4)
        ax.plot([1e-6, limit], [1e-6, limit], color="#8a4f49", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Direct affected MSE")
        ax.set_ylabel("Residual affected MSE")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Direct vs Residual Per-Sample Error")
    fig.tight_layout()
    return save_figure(fig, output_dir, "direct_vs_residual_affected_mse_scatter.png")


def residual_direct_ratio_histogram(
    normal_per_sample: pd.DataFrame,
    stress_per_sample: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, frame, color in (
        ("normal", normal_per_sample, "#5f8a4b"),
        ("stress", stress_per_sample, "#2f6f8f"),
    ):
        values = (
            frame["residual_direct_affected_mse_ratio"]
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
        ax.hist(
            clipped,
            bins=35,
            alpha=0.48,
            label=label,
            color=color,
        )
    ax.axvline(1.0, color="#8a4f49", linewidth=1.5, label="parity")
    ax.set_xlabel("Residual affected MSE / direct affected MSE")
    ax.set_ylabel("Samples")
    ax.set_title("Residual-to-Direct Error Ratio")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return save_figure(fig, output_dir, "hist_residual_direct_error_ratio.png")


def predict_single(
    sample: dict[str, Any],
    direct_model: nn.Module,
    residual_model: nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct_batch, residual_batch, residual_recon_batch = predict_batch(
        direct_model,
        residual_model,
        [sample],
        device,
    )
    return direct_batch[0], residual_batch[0], residual_recon_batch[0]


def save_example_figure(
    sample: dict[str, Any],
    row: pd.Series,
    direct_model: nn.Module,
    residual_model: nn.Module,
    device: torch.device,
    output_dir: Path,
    filename: str,
    title: str,
) -> Path:
    direct_pred, _residual_layer, residual_recon = predict_single(
        sample,
        direct_model,
        residual_model,
        device,
    )
    target = sample["target"]
    blended = sample["blended"]
    direct_error = np.abs(direct_pred - target).mean(axis=-1)
    residual_error = np.abs(residual_recon - target).mean(axis=-1)
    images = [
        (target, "Target"),
        (blended, "Blend"),
        (direct_pred, "Direct"),
        (residual_recon, "Residual"),
        (direct_error, "Direct error"),
        (residual_error, "Residual error"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(17.5, 3.5))
    for ax, (image, panel_title) in zip(axes, images):
        if image.ndim == 2:
            ax.imshow(image, cmap="magma", vmin=0.0, vmax=max(0.2, float(image.max())))
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(
        "{title}; split={split} idx={idx} direct={direct:.5f} residual={resid:.5f} ratio={ratio:.2f}".format(
            title=title,
            split=row["split"],
            idx=int(row["index"]),
            direct=float(row["direct_affected_mse"]),
            resid=float(row["residual_affected_mse"]),
            ratio=float(row["residual_direct_affected_mse_ratio"]),
        ),
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, filename)


def example_lookup(
    split_name: str,
    index: int,
    normal_samples: list[dict[str, Any]],
    stress_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if split_name == "normal":
        return normal_samples[index]
    if split_name == "stress":
        return stress_samples[index]
    raise KeyError(f"Unknown split: {split_name}")


def choose_example_rows(
    normal_per_sample: pd.DataFrame,
    stress_per_sample: pd.DataFrame,
) -> dict[str, pd.Series | None]:
    combined = pd.concat([normal_per_sample, stress_per_sample], ignore_index=True)
    finite = combined.replace([np.inf, -np.inf], np.nan)
    examples: dict[str, pd.Series | None] = {
        "residual_success": None,
        "residual_failure": None,
        "direct_beats_residual": None,
    }
    success = finite[
        (finite["residual_beats_direct"])
        & (finite["residual_affected_mse"] < finite["identity_affected_mse"])
    ].copy()
    if not success.empty:
        examples["residual_success"] = success.sort_values(
            ["direct_residual_affected_mse_ratio", "identity_affected_mse"],
            ascending=False,
        ).iloc[0]

    failure = finite[
        (finite["residual_worse_than_identity"])
        | (finite["residual_improvement_ratio"] <= 1.25)
    ].copy()
    if failure.empty:
        failure = finite.copy()
    if not failure.empty:
        examples["residual_failure"] = failure.sort_values(
            ["residual_affected_mse", "identity_affected_mse"],
            ascending=False,
        ).iloc[0]

    direct_better = finite[finite["direct_beats_residual"]].copy()
    if not direct_better.empty:
        examples["direct_beats_residual"] = direct_better.sort_values(
            ["residual_direct_affected_mse_ratio", "residual_affected_mse"],
            ascending=False,
        ).iloc[0]
    return examples


def save_all_figures(
    run_dir: Path,
    normal_aggregate: pd.DataFrame,
    normal_per_sample: pd.DataFrame,
    stress_aggregate: pd.DataFrame,
    stress_per_sample: pd.DataFrame,
    normal_samples: list[dict[str, Any]],
    stress_samples: list[dict[str, Any]],
    direct_model: nn.Module,
    residual_model: nn.Module,
    device: torch.device,
) -> dict[str, str | None]:
    output_dir = run_dir / "paper_figures"
    written: dict[str, str | None] = {}
    written["affected_mse_bar"] = project_relative(
        affected_mse_bar(normal_aggregate, stress_aggregate, output_dir)
    )
    written["improvement_chart"] = project_relative(
        improvement_chart(normal_aggregate, stress_aggregate, output_dir)
    )
    written["scatter"] = project_relative(
        direct_vs_residual_scatter(normal_per_sample, stress_per_sample, output_dir)
    )
    written["ratio_histogram"] = project_relative(
        residual_direct_ratio_histogram(normal_per_sample, stress_per_sample, output_dir)
    )

    examples = choose_example_rows(normal_per_sample, stress_per_sample)
    figure_specs = [
        (
            "residual_success",
            "residual_success_over_direct.png",
            "Residual improves over direct",
        ),
        (
            "residual_failure",
            "residual_failure_example.png",
            "Residual failure case",
        ),
        (
            "direct_beats_residual",
            "direct_beats_residual_example.png",
            "Direct beats residual",
        ),
    ]
    for key, filename, title in figure_specs:
        row = examples[key]
        if row is None:
            written[key] = None
            continue
        sample = example_lookup(
            split_name=str(row["split"]),
            index=int(row["index"]),
            normal_samples=normal_samples,
            stress_samples=stress_samples,
        )
        written[key] = project_relative(
            save_example_figure(
                sample=sample,
                row=row,
                direct_model=direct_model,
                residual_model=residual_model,
                device=device,
                output_dir=output_dir,
                filename=filename,
                title=title,
            )
        )
    return written


def stop_condition_warnings(
    split_name: str,
    aggregate: pd.DataFrame,
    per_sample: pd.DataFrame,
    residual_stats: dict[str, float],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    stops: list[str] = []
    frame = aggregate.set_index("method")
    identity = frame.loc["identity"]
    residual = frame.loc["residual"]
    threshold = frame.loc["threshold"]

    if (
        residual["masked_mse"] > identity["masked_mse"]
        and residual["masked_mae"] > identity["masked_mae"]
    ):
        stops.append(
            f"{split_name}: residual is worse than identity on both affected MSE and affected MAE"
        )
    if residual["masked_mse"] < 1e-6 or residual["mse"] < 1e-7:
        stops.append(f"{split_name}: residual metrics are unrealistically perfect")
    if residual_stats["residual_pred_abs_max"] > 5.0:
        stops.append(f"{split_name}: residual predictions exceeded absolute value 5")
    if residual_stats["reconstruction_preclip_total_clip_fraction"] > 0.20:
        stops.append(
            f"{split_name}: more than 20% of reconstruction pixels needed clipping"
        )
    if threshold["masked_mse"] < residual["masked_mse"] * 0.75:
        warnings.append(
            f"{split_name}: threshold baseline is substantially better than residual"
        )
    if float(per_sample["residual_worse_than_identity"].mean()) > 0.75:
        stops.append(
            f"{split_name}: residual is worse than identity for more than 75% of samples"
        )
    if residual_stats["reconstruction_preclip_total_clip_fraction"] > 0.05:
        warnings.append(
            f"{split_name}: more than 5% of reconstruction pixels needed clipping"
        )
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
    with (run_dir / "diagnostics" / "residual_warnings.md").open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("# Residual U-Net Diagnostics\n\n")
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
            "residual_direct_affected_mse_ratio",
            ascending=False,
        ).head(25).to_csv(
            run_dir / "diagnostics" / "normal_largest_residual_regressions.csv",
            index=False,
        )
    if stress_per_sample is not None:
        stress_per_sample.sort_values(
            "residual_direct_affected_mse_ratio",
            ascending=False,
        ).head(25).to_csv(
            run_dir / "diagnostics" / "stress_largest_residual_regressions.csv",
            index=False,
        )


def append_experiment_log(
    run_dir: Path,
    checkpoint_path: Path,
    settings: ExperimentSettings,
    batch_size: int,
    normal_aggregate: pd.DataFrame,
    normal_summary: pd.DataFrame,
    stress_aggregate: pd.DataFrame | None,
    stress_summary: pd.DataFrame | None,
    warnings: list[str],
    stops: list[str],
    figure_paths: dict[str, str | None],
) -> None:
    status = "coherent" if not stops else "suspicious; stopped early"
    warning_text = "None." if not warnings and not stops else "; ".join(stops + warnings) + "."
    stress_text = (
        "Stress-test evaluation did not run because a stop condition was hit before that stage."
        if stress_aggregate is None or stress_summary is None
        else "\n".join(
            [
                "Stress-test metrics:",
                "",
                format_metrics_table(stress_aggregate),
                "",
                "Stress direct-vs-residual comparison:",
                "",
                format_comparison_table(stress_summary),
            ]
        )
    )
    residual_normal = normal_aggregate.set_index("method").loc["residual"]
    direct_normal = normal_aggregate.set_index("method").loc["direct"]
    residual_beats_normal = (
        "yes"
        if residual_normal["masked_mse"] < direct_normal["masked_mse"]
        else "no"
    )
    if stress_aggregate is not None:
        residual_stress = stress_aggregate.set_index("method").loc["residual"]
        direct_stress = stress_aggregate.set_index("method").loc["direct"]
        residual_beats_stress = (
            "yes"
            if residual_stress["masked_mse"] < direct_stress["masked_mse"]
            else "no"
        )
    else:
        residual_beats_stress = "not evaluated"

    figure_lines = [
        f"- `{path}`" for path in figure_paths.values() if path is not None
    ]
    if not figure_lines:
        figure_lines = ["- Not generated before stop condition."]

    section = f"""

## Residual U-Net Experiment

Status: {status}. Run directory: `{project_relative(run_dir)}`.

### Training Settings

- Task: predict residual contaminant layer as `blended - target`; reconstruction is `blended - predicted_residual`.
- Dataset: Galaxy10 DECaLS from `data/Galaxy10_DECals.h5`.
- Split seed: 42.
- Train/validation/held-out blends: {settings.n_train_blends:,} / {settings.n_val_blends:,} / {settings.n_test_blends:,}.
- Epochs: {settings.num_epochs}.
- Batch size used: {batch_size}.
- Model: compact U-Net with a linear residual output head.
- Checkpoint: `{project_relative(checkpoint_path)}`.

### Normal Held-Out Metrics

{format_metrics_table(normal_aggregate)}

Normal direct-vs-residual comparison:

{format_comparison_table(normal_summary)}

### Hard Stress-Test Metrics

{stress_text}

### Comparison Notes

- Normal affected-region MSE baseline reference: identity {NORMAL_BASELINE_IDENTITY_AFFECTED_MSE:.6f}, direct {NORMAL_BASELINE_DIRECT_AFFECTED_MSE:.6f}.
- Stress affected-region MSE baseline reference: identity {STRESS_BASELINE_IDENTITY_AFFECTED_MSE:.6f}, direct {STRESS_BASELINE_DIRECT_AFFECTED_MSE:.6f}.
- Residual beat direct on normal aggregate affected-region MSE: {residual_beats_normal}.
- Residual beat direct on hard stress aggregate affected-region MSE: {residual_beats_stress}.
- Coherence/suspicion status: {warning_text}

Paper figures:

{chr(10).join(figure_lines)}

Recommended next step: inspect the direct-vs-residual scatter, residual/direct error-ratio histogram, and example panels. If residual improves core-overlap cases without increasing target-detail loss, repeat with a slightly larger residual model or a loss that weights affected regions.
"""
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(section)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    settings: ExperimentSettings,
    history: pd.DataFrame,
    batch_size: int,
    stamp: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}")
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "experiment_settings": settings.__dict__,
        "residual_target": "blended_minus_target",
        "reconstruction": "blended_minus_predicted_residual",
        "output_activation": "identity",
        "timestamp": stamp,
        "batch_size": batch_size,
        "final_train_loss": (
            float(history["train_loss"].iloc[-1]) if not history.empty else None
        ),
        "final_val_loss": (
            float(history["val_loss"].iloc[-1]) if not history.empty else None
        ),
    }
    torch.save(payload, path)


def record_checkpoint_integrity(
    run_dir: Path,
    direct_checkpoint_path: Path,
    before: Any,
    after: Any,
) -> None:
    unchanged = (
        before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    )
    save_json(
        run_dir / "logs" / "direct_checkpoint_integrity.json",
        {
            "checkpoint_path": project_relative(direct_checkpoint_path),
            "size_bytes_before": before.st_size,
            "size_bytes_after": after.st_size,
            "mtime_ns_before": before.st_mtime_ns,
            "mtime_ns_after": after.st_mtime_ns,
            "unchanged": unchanged,
        },
    )
    if not unchanged:
        raise RuntimeError("Direct checkpoint metadata changed during residual run.")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    settings = ExperimentSettings(
        n_train_blends=args.n_train_blends,
        n_val_blends=args.n_val_blends,
        n_test_blends=args.n_test_blends,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
        num_epochs=args.num_epochs,
    )
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    stamp, run_dir, checkpoint_path = make_run_paths(output_root, settings)
    save_yaml(
        run_dir / "logs" / "run_config.yaml",
        {
            "project_root": ".",
            "timestamp": stamp,
            "settings": settings.__dict__,
            "config": config,
            "direct_checkpoint": project_relative(args.direct_checkpoint),
            "stress_settings": {
                **stress_helpers.STRESS_DEFAULTS,
                "n_stress_blends": args.n_stress_blends,
            },
        },
    )

    direct_checkpoint_path = args.direct_checkpoint
    if not direct_checkpoint_path.is_absolute():
        direct_checkpoint_path = PROJECT_ROOT / direct_checkpoint_path
    direct_checkpoint_path = direct_checkpoint_path.resolve()
    if not direct_checkpoint_path.exists():
        message = f"Direct checkpoint not found: {project_relative(direct_checkpoint_path)}"
        write_diagnostics(run_dir, warnings=[], stops=[message])
        print(message)
        return 2

    seed = int(config["seed"])
    seed_everything(seed)
    device = gd_train.resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    direct_stat_before = direct_checkpoint_path.stat()
    try:
        direct_model = load_direct_model(
            direct_checkpoint_path,
            config["model"],
            device,
        )
    except Exception as exc:
        message = f"Could not load direct checkpoint: {exc}"
        write_diagnostics(run_dir, warnings=[], stops=[message])
        print(message)
        return 2

    print("Loading dataset and reconstructing splits.", flush=True)
    train_images, val_images, test_images = load_split_subsets(config, settings)
    print("Generating normal train/validation/test blends.", flush=True)
    train_blends, val_blends, normal_test_blends = generate_normal_blends(
        train_images,
        val_images,
        test_images,
        config,
        settings,
    )
    train_ds = ResidualBlendDataset(train_blends)
    val_ds = ResidualBlendDataset(val_blends)
    requested_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["training"].get("batch_size", 8))
    )

    print("Training residual U-Net.", flush=True)
    residual_model, history, used_batch_size, train_attempts = train_with_memory_retry(
        train_ds=train_ds,
        val_ds=val_ds,
        model_config=config["model"],
        training_config=config["training"],
        settings=settings,
        requested_batch_size=requested_batch_size,
        device=device,
        seed=seed,
    )
    history.to_csv(run_dir / "results" / "training_history.csv", index=False)
    save_json(run_dir / "logs" / "training_attempts.json", train_attempts)
    save_checkpoint(
        checkpoint_path,
        residual_model,
        config,
        settings,
        history,
        used_batch_size,
        stamp,
    )
    print(f"Saved residual checkpoint: {project_relative(checkpoint_path)}", flush=True)

    del train_blends, val_blends, train_ds, val_ds, train_images, val_images
    clear_torch_cache()
    gc.collect()

    all_warnings: list[str] = []
    all_stops: list[str] = []
    print("Evaluating normal held-out test set.", flush=True)
    normal_aggregate, normal_per_sample, normal_stats = evaluate_samples(
        split_name="normal",
        samples=normal_test_blends,
        direct_model=direct_model,
        residual_model=residual_model,
        device=device,
        affected_region_threshold=settings.affected_region_threshold,
        batch_size=used_batch_size,
    )
    normal_summary = comparison_summary("normal", normal_per_sample)
    normal_aggregate.to_csv(run_dir / "results" / "normal_results.csv", index=False)
    normal_per_sample.to_csv(
        run_dir / "results" / "normal_per_sample_results.csv",
        index=False,
    )
    normal_summary.to_csv(
        run_dir / "results" / "normal_direct_vs_residual_comparison.csv",
        index=False,
    )
    save_json(run_dir / "diagnostics" / "normal_residual_output_stats.json", normal_stats)
    warnings, stops = stop_condition_warnings(
        "normal",
        normal_aggregate,
        normal_per_sample,
        normal_stats,
    )
    all_warnings.extend(warnings)
    all_stops.extend(stops)
    if all_stops:
        write_diagnostics(run_dir, all_warnings, all_stops, normal_per_sample)
        if not args.skip_log_update:
            append_experiment_log(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                settings=settings,
                batch_size=used_batch_size,
                normal_aggregate=normal_aggregate,
                normal_summary=normal_summary,
                stress_aggregate=None,
                stress_summary=None,
                warnings=all_warnings,
                stops=all_stops,
                figure_paths={},
            )
        record_checkpoint_integrity(
            run_dir,
            direct_checkpoint_path,
            direct_stat_before,
            direct_checkpoint_path.stat(),
        )
        print("Stopped after normal evaluation due to suspicious residual results.")
        return 3

    print("Generating and evaluating hard stress-test blends.", flush=True)
    stress_settings = dict(stress_helpers.STRESS_DEFAULTS)
    stress_settings["n_stress_blends"] = int(args.n_stress_blends)
    stress_source_subset = int(stress_settings["stress_source_subset"])
    stress_images = test_images[:stress_source_subset]
    stress_blends = stress_helpers.generate_stress_blends(stress_images, stress_settings)
    stress_aggregate, stress_per_sample, stress_stats = evaluate_samples(
        split_name="stress",
        samples=stress_blends,
        direct_model=direct_model,
        residual_model=residual_model,
        device=device,
        affected_region_threshold=float(stress_settings["affected_region_threshold"]),
        batch_size=used_batch_size,
    )
    stress_summary = comparison_summary("stress", stress_per_sample)
    stress_aggregate.to_csv(run_dir / "results" / "stress_results.csv", index=False)
    stress_per_sample.to_csv(
        run_dir / "results" / "stress_per_sample_results.csv",
        index=False,
    )
    stress_summary.to_csv(
        run_dir / "results" / "stress_direct_vs_residual_comparison.csv",
        index=False,
    )
    combined_summary = pd.concat([normal_summary, stress_summary], ignore_index=True)
    combined_summary.to_csv(
        run_dir / "results" / "direct_vs_residual_comparison.csv",
        index=False,
    )
    save_json(run_dir / "diagnostics" / "stress_residual_output_stats.json", stress_stats)
    warnings, stops = stop_condition_warnings(
        "stress",
        stress_aggregate,
        stress_per_sample,
        stress_stats,
    )
    all_warnings.extend(warnings)
    all_stops.extend(stops)
    write_diagnostics(
        run_dir,
        all_warnings,
        all_stops,
        normal_per_sample,
        stress_per_sample,
    )

    figure_paths: dict[str, str | None] = {}
    if not all_stops:
        print("Generating paper figures.", flush=True)
        figure_paths = save_all_figures(
            run_dir=run_dir,
            normal_aggregate=normal_aggregate,
            normal_per_sample=normal_per_sample,
            stress_aggregate=stress_aggregate,
            stress_per_sample=stress_per_sample,
            normal_samples=normal_test_blends,
            stress_samples=stress_blends,
            direct_model=direct_model,
            residual_model=residual_model,
            device=device,
        )

    if not args.skip_log_update:
        append_experiment_log(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            settings=settings,
            batch_size=used_batch_size,
            normal_aggregate=normal_aggregate,
            normal_summary=normal_summary,
            stress_aggregate=stress_aggregate,
            stress_summary=stress_summary,
            warnings=all_warnings,
            stops=all_stops,
            figure_paths=figure_paths,
        )

    record_checkpoint_integrity(
        run_dir,
        direct_checkpoint_path,
        direct_stat_before,
        direct_checkpoint_path.stat(),
    )

    print(f"Residual run directory: {project_relative(run_dir)}")
    print(f"Residual checkpoint: {project_relative(checkpoint_path)}")
    print("Normal metrics:")
    print(format_metrics_table(normal_aggregate))
    print("Stress metrics:")
    print(format_metrics_table(stress_aggregate))
    if all_warnings or all_stops:
        print("Diagnostics:")
        for item in all_stops + all_warnings:
            print(f"- {item}")
    return 0 if not all_stops else 3


if __name__ == "__main__":
    raise SystemExit(main())
