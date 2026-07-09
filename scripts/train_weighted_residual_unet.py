"""Train Experiment 4: Thayer-BR v0.2 weighted residual U-Net.

This script keeps the Experiment 3 residual formulation:

- input: blended RGB image
- target residual: blended - target
- prediction: predicted residual
- reconstruction: blended - predicted residual

All run artifacts are timestamped under ``outputs/``. Existing checkpoints are
read for comparison only and are verified unchanged before the script exits.
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
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "docs" / "experiment_log.md"
RESULTS_INTERPRETATION_PATH = PROJECT_ROOT / "docs" / "results_interpretation.md"
PAPER_PLAN_PATH = PROJECT_ROOT / "docs" / "paper_plan.md"

METHODS = (
    "identity",
    "threshold",
    "direct",
    "residual",
    "balanced_residual",
    "weighted_residual",
)
LEARNED_METHODS = (
    "direct",
    "residual",
    "balanced_residual",
    "weighted_residual",
)


@dataclass(frozen=True)
class WeightedSettings:
    """Balanced residual scale, evaluation scale, and training composition."""

    n_train_blends: int = 12000
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
    multiseed_count: int = 3
    multiseed_blends: int = 1000


@dataclass(frozen=True)
class WeightedLossConfig:
    """Configurable affected/core-weighted residual objective."""

    background_weight: float = 1.0
    affected_extra_weight: float = 3.0
    core_extra_weight: float = 2.0
    affected_threshold: float = 0.02
    core_aperture_fraction: float = 0.18
    core_brightness_fraction: float = 0.55
    eps: float = 1e-8


class WeightedResidualBlendDataset(Dataset):
    """PyTorch dataset whose target is residual = blended - target."""

    def __init__(self, blends: list[dict[str, Any]]) -> None:
        if not blends:
            raise ValueError("WeightedResidualBlendDataset requires at least one blend.")
        self.blends = blends

    def __len__(self) -> int:
        return len(self.blends)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.blends[idx]
        blended = np.asarray(sample["blended"], dtype=np.float32)
        target = np.asarray(sample["target"], dtype=np.float32)
        residual = blended - target
        return (
            torch.from_numpy(blended.transpose(2, 0, 1)).float(),
            torch.from_numpy(residual.transpose(2, 0, 1)).float(),
            torch.from_numpy(target.transpose(2, 0, 1)).float(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Thayer-BR v0.2 with affected/core-weighted residual loss."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--direct-checkpoint", type=Path, default=DEFAULT_DIRECT_CHECKPOINT)
    parser.add_argument("--residual-checkpoint", type=Path, default=DEFAULT_RESIDUAL_CHECKPOINT)
    parser.add_argument("--balanced-checkpoint", type=Path, default=DEFAULT_BALANCED_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variant-name", default="moderate")
    parser.add_argument("--n-train-blends", type=int, default=12000)
    parser.add_argument("--n-val-blends", type=int, default=1000)
    parser.add_argument("--n-normal-test-blends", type=int, default=1000)
    parser.add_argument("--n-stress-blends", type=int, default=1000)
    parser.add_argument("--train-source-subset", type=int, default=5000)
    parser.add_argument("--val-source-subset", type=int, default=1000)
    parser.add_argument("--test-source-subset", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--affected-extra-weight", type=float, default=3.0)
    parser.add_argument("--core-extra-weight", type=float, default=2.0)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--affected-threshold", type=float, default=0.02)
    parser.add_argument("--core-aperture-fraction", type=float, default=0.18)
    parser.add_argument("--core-brightness-fraction", type=float, default=0.55)
    parser.add_argument("--multiseed-count", type=int, default=3)
    parser.add_argument("--multiseed-blends", type=int, default=1000)
    parser.add_argument("--skip-doc-update", action="store_true")
    parser.add_argument("--skip-multiseed", action="store_true")
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


def safe_write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def save_yaml(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    frame.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 220) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


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


def make_run_paths(output_root: Path, stamp: str) -> tuple[Path, Path, Path]:
    run_dir = output_root / "runs" / f"weighted_residual_{stamp}"
    best_checkpoint = (
        output_root / "checkpoints" / f"unet_residual_weighted_br_{stamp}_best.pth"
    )
    final_checkpoint = (
        output_root / "checkpoints" / f"unet_residual_weighted_br_{stamp}_final.pth"
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if best_checkpoint.exists():
        raise FileExistsError(f"Checkpoint path already exists: {best_checkpoint}")
    if final_checkpoint.exists():
        raise FileExistsError(f"Checkpoint path already exists: {final_checkpoint}")
    for child in ("tables", "figures", "paper_figures", "diagnostics", "logs", "example_grids"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, best_checkpoint, final_checkpoint


def checkpoint_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": project_relative(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_iso_local": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def checkpoint_inventory(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    if not checkpoint_dir.exists():
        return {}
    return {
        project_relative(path): checkpoint_info(path)
        for path in sorted(checkpoint_dir.glob("*.pth"))
    }


def verify_checkpoint_inventory(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for rel_path, before_info in before.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            comparison[rel_path] = {
                "before": before_info,
                "after": None,
                "unchanged": False,
            }
            continue
        after_info = checkpoint_info(path)
        comparison[rel_path] = {
            "before": before_info,
            "after": after_info,
            "unchanged": (
                before_info["size_bytes"] == after_info["size_bytes"]
                and before_info["mtime_ns"] == after_info["mtime_ns"]
            ),
        }
    return comparison


def resolve_checkpoint(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def target_core_mask_torch(target: torch.Tensor, loss_config: WeightedLossConfig) -> torch.Tensor:
    """Return a central bright-core mask with shape (B, 1, H, W)."""
    gray = target.mean(dim=1, keepdim=True)
    batch, _channels, height, width = gray.shape
    device = target.device
    y_grid = torch.arange(height, device=device, dtype=target.dtype).view(1, 1, height, 1)
    x_grid = torch.arange(width, device=device, dtype=target.dtype).view(1, 1, 1, width)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = loss_config.core_aperture_fraction * min(height, width)
    aperture = (
        torch.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2) <= radius
    )
    aperture = aperture.expand(batch, 1, height, width)
    dark_floor = torch.full_like(gray, -1.0)
    aperture_gray = torch.where(aperture, gray, dark_floor)
    aperture_max = aperture_gray.amax(dim=(2, 3), keepdim=True)
    threshold = aperture_max * loss_config.core_brightness_fraction
    core = aperture & (gray >= threshold)
    empty = core.flatten(1).sum(dim=1) == 0
    if bool(empty.any().detach().cpu()):
        core[empty] = aperture[empty]
    return core


def weighted_residual_loss(
    predicted_residual: torch.Tensor,
    true_residual: torch.Tensor,
    target: torch.Tensor,
    loss_config: WeightedLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    affected = (
        torch.mean(torch.abs(true_residual), dim=1, keepdim=True)
        > loss_config.affected_threshold
    )
    core = target_core_mask_torch(target, loss_config)
    core_affected = affected & core
    weight = torch.full_like(true_residual[:, :1], loss_config.background_weight)
    weight = weight + affected.float() * loss_config.affected_extra_weight
    weight = weight + core_affected.float() * loss_config.core_extra_weight
    squared_error = (predicted_residual - true_residual) ** 2
    normalizer = weight.sum() * squared_error.shape[1] + loss_config.eps
    loss = (weight * squared_error).sum() / normalizer
    stats = {
        "affected_fraction": float(affected.float().mean().detach().cpu()),
        "core_affected_fraction": float(core_affected.float().mean().detach().cpu()),
        "mean_weight": float(weight.mean().detach().cpu()),
    }
    return loss, stats


def make_residual_unet(model_config: dict[str, Any]) -> nn.Module:
    return balanced_helpers.make_residual_unet(model_config)


def masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    return (
        float(np.mean((pred[mask] - target[mask]) ** 2)),
        float(np.mean(np.abs(pred[mask] - target[mask]))),
    )


def whole_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return gd_utils.compute_metrics(pred, target, metrics=("mse", "mae", "psnr", "ssim"))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def evaluate_validation_diagnostics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_config: WeightedLossConfig,
) -> dict[str, float]:
    model.eval()
    n_samples = 0
    weighted_loss_sum = 0.0
    residual_mse_sum = 0.0
    affected_mse_values: list[float] = []
    residual_min = float("inf")
    residual_max = float("-inf")
    residual_sum = 0.0
    residual_sq_sum = 0.0
    pixel_count = 0.0
    clip_count = 0.0
    black_white_count = 0.0
    recon_sum = 0.0
    weight_stats = {"affected_fraction": [], "core_affected_fraction": [], "mean_weight": []}

    with torch.no_grad():
        for blended, true_residual, target in val_loader:
            blended = blended.to(device)
            true_residual = true_residual.to(device)
            target = target.to(device)
            predicted_residual = model(blended)
            loss, stats = weighted_residual_loss(
                predicted_residual,
                true_residual,
                target,
                loss_config,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")
            batch_size = blended.size(0)
            n_samples += batch_size
            weighted_loss_sum += float(loss.item()) * batch_size
            residual_mse_sum += float(torch.mean((predicted_residual - true_residual) ** 2).item()) * batch_size
            for key, value in stats.items():
                weight_stats[key].append(value)

            reconstruction_preclip = blended - predicted_residual
            reconstruction = torch.clamp(reconstruction_preclip, 0.0, 1.0)
            clip_count += float(
                torch.count_nonzero((reconstruction_preclip < 0.0) | (reconstruction_preclip > 1.0)).item()
            )
            black_white_count += float(
                torch.count_nonzero((reconstruction < 0.01) | (reconstruction > 0.99)).item()
            )
            recon_sum += float(reconstruction.sum().item())
            pixel_count += float(reconstruction.numel())

            residual_min = min(residual_min, float(predicted_residual.min().item()))
            residual_max = max(residual_max, float(predicted_residual.max().item()))
            residual_sum += float(predicted_residual.sum().item())
            residual_sq_sum += float(torch.sum(predicted_residual**2).item())

            affected = (
                torch.mean(torch.abs(true_residual), dim=1)
                > loss_config.affected_threshold
            )
            sq = torch.mean((reconstruction - target) ** 2, dim=1)
            for idx in range(batch_size):
                mask = affected[idx]
                if bool(mask.any().detach().cpu()):
                    affected_mse_values.append(float(sq[idx][mask].mean().item()))

    residual_mean = residual_sum / max(pixel_count, 1.0)
    residual_var = max(residual_sq_sum / max(pixel_count, 1.0) - residual_mean**2, 0.0)
    return {
        "val_loss": weighted_loss_sum / max(n_samples, 1),
        "val_unweighted_residual_mse": residual_mse_sum / max(n_samples, 1),
        "val_affected_region_mse": float(np.nanmean(affected_mse_values)) if affected_mse_values else float("nan"),
        "val_reconstruction_clip_fraction": clip_count / max(pixel_count, 1.0),
        "val_reconstruction_black_white_fraction": black_white_count / max(pixel_count, 1.0),
        "val_reconstruction_mean": recon_sum / max(pixel_count, 1.0),
        "val_predicted_residual_min": residual_min,
        "val_predicted_residual_max": residual_max,
        "val_predicted_residual_mean": residual_mean,
        "val_predicted_residual_std": float(np.sqrt(residual_var)),
        "val_weighted_affected_fraction": float(np.mean(weight_stats["affected_fraction"])),
        "val_weighted_core_affected_fraction": float(np.mean(weight_stats["core_affected_fraction"])),
        "val_mean_loss_weight": float(np.mean(weight_stats["mean_weight"])),
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
    loss_config: WeightedLossConfig,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame]:
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
    recent_val_losses: list[float] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight_stats = {"affected_fraction": [], "core_affected_fraction": [], "mean_weight": []}
        for blended, true_residual, target in train_loader:
            blended = blended.to(device)
            true_residual = true_residual.to(device)
            target = target.to(device)
            optimiser.zero_grad()
            predicted_residual = model(blended)
            loss, stats = weighted_residual_loss(
                predicted_residual,
                true_residual,
                target,
                loss_config,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimiser.step()
            train_loss += float(loss.item()) * blended.size(0)
            for key, value in stats.items():
                train_weight_stats[key].append(value)

        train_loss /= len(train_ds)
        val_diag = evaluate_validation_diagnostics(model, val_loader, device, loss_config)
        val_loss = val_diag["val_loss"]
        if train_loss > 2.0 or val_loss > 2.0:
            raise RuntimeError(
                f"Training loss exploded at epoch {epoch}: "
                f"train={train_loss:.6f}, val={val_loss:.6f}."
            )
        recent_val_losses.append(val_loss)
        if len(recent_val_losses) >= 4:
            best_recent = min(recent_val_losses[:-1])
            if val_loss > best_recent * 10.0:
                raise RuntimeError(
                    f"Validation loss exploded at epoch {epoch}: "
                    f"val={val_loss:.6f}, previous_best={best_recent:.6f}."
                )
        if val_diag["val_reconstruction_clip_fraction"] > 0.20:
            raise RuntimeError(
                f"Extreme clipping at epoch {epoch}: "
                f"{val_diag['val_reconstruction_clip_fraction']:.3%}."
            )
        if val_diag["val_reconstruction_black_white_fraction"] > 0.98:
            raise RuntimeError(
                f"Validation reconstructions are mostly black/white at epoch {epoch}: "
                f"{val_diag['val_reconstruction_black_white_fraction']:.3%}."
            )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            "train_weighted_affected_fraction": float(np.mean(train_weight_stats["affected_fraction"])),
            "train_weighted_core_affected_fraction": float(np.mean(train_weight_stats["core_affected_fraction"])),
            "train_mean_loss_weight": float(np.mean(train_weight_stats["mean_weight"])),
            **val_diag,
        }
        rows.append(row)
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"train weighted residual loss={train_loss:.6f}, "
            f"val weighted residual loss={val_loss:.6f}, "
            f"val affected MSE={val_diag['val_affected_region_mse']:.6f}, "
            f"clip={val_diag['val_reconstruction_clip_fraction']:.3%}, "
            f"best={best_val:.6f} @ {best_epoch}",
            flush=True,
        )

    return model, best_state, pd.DataFrame(rows)


def train_with_memory_retry(
    train_ds: Dataset,
    val_ds: Dataset,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    settings: WeightedSettings,
    requested_batch_size: int,
    device: torch.device,
    seed: int,
    loss_config: WeightedLossConfig,
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
                loss_config=loss_config,
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
    settings: WeightedSettings,
    loss_config: WeightedLossConfig,
    history: pd.DataFrame,
    batch_size: int,
    stamp: str,
    checkpoint_kind: str,
    variant_name: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    payload = {
        "model_state_dict": model_state,
        "config": config,
        "experiment_name": "Thayer-BR v0.2 weighted residual loss",
        "variant_name": variant_name,
        "experiment_settings": settings.__dict__,
        "loss_config": loss_config.__dict__,
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


def predict_all(
    samples: list[dict[str, Any]],
    direct_model: nn.Module,
    residual_model: nn.Module,
    balanced_model: nn.Module,
    weighted_model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, float]]:
    preds: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    weighted_stats = balanced_helpers.empty_residual_stats()
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        inputs = np.stack([sample["blended"] for sample in batch_samples], axis=0)
        tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            direct = direct_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            residual_layer = residual_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            balanced_layer = balanced_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            weighted_layer = weighted_model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
        residual_recon = np.clip(inputs - residual_layer, 0.0, 1.0).astype(np.float32)
        balanced_recon = np.clip(inputs - balanced_layer, 0.0, 1.0).astype(np.float32)
        weighted_preclip = inputs - weighted_layer
        weighted_recon = np.clip(weighted_preclip, 0.0, 1.0).astype(np.float32)
        balanced_helpers.update_residual_stats(weighted_stats, weighted_layer, weighted_preclip)
        for offset, sample in enumerate(batch_samples):
            blended = sample["blended"]
            preds["identity"].append(blended)
            preds["threshold"].append(baselines.threshold_baseline(blended))
            preds["direct"].append(np.clip(direct[offset], 0.0, 1.0).astype(np.float32))
            preds["residual"].append(residual_recon[offset])
            preds["balanced_residual"].append(balanced_recon[offset])
            preds["weighted_residual"].append(weighted_recon[offset])
    return preds, balanced_helpers.finalise_residual_stats(weighted_stats)


def compute_per_sample(
    split: str,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        target = sample["target"]
        blended = sample["blended"]
        affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
        core = balanced_helpers.target_core_mask(target)
        core_affected = np.logical_and(affected, core)
        noncore_affected = np.logical_and(affected, ~core)
        info = sample.get("info", {})
        metadata = balanced_helpers.blend_metadata(target, blended, info, threshold)
        shift = info.get("shift", (0, 0))
        identity_affected_mse, identity_affected_mae = masked_metrics(blended, target, affected)
        row: dict[str, Any] = {
            "split": split,
            "index": idx,
            "generation_difficulty": info.get("generation_difficulty", info.get("difficulty")),
            "training_component": info.get("training_component"),
            "shift_x": int(shift[0]),
            "shift_y": int(shift[1]),
            "shift_distance": abs(int(shift[0])) + abs(int(shift[1])),
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
            "core_affected_fraction": float(core_affected.mean()),
            "noncore_affected_fraction": float(noncore_affected.mean()),
        }
        for method in METHODS:
            pred = preds[method][idx]
            whole = whole_metrics(pred, target)
            affected_mse, affected_mae = masked_metrics(pred, target, affected)
            core_mse, core_mae = masked_metrics(pred, target, core_affected)
            noncore_mse, noncore_mae = masked_metrics(pred, target, noncore_affected)
            for metric_name, value in whole.items():
                row[f"{method}_{metric_name}"] = value
            row[f"{method}_affected_mse"] = affected_mse
            row[f"{method}_affected_mae"] = affected_mae
            row[f"{method}_core_affected_mse"] = core_mse
            row[f"{method}_core_affected_mae"] = core_mae
            row[f"{method}_noncore_affected_mse"] = noncore_mse
            row[f"{method}_noncore_affected_mae"] = noncore_mae
            row[f"{method}_improvement_ratio"] = safe_ratio(identity_affected_mse, affected_mse)
            row[f"{method}_core_improvement_ratio"] = safe_ratio(
                row["identity_core_affected_mse"] if "identity_core_affected_mse" in row else core_mse,
                core_mse,
            )
            row[f"{method}_noncore_improvement_ratio"] = safe_ratio(
                row["identity_noncore_affected_mse"] if "identity_noncore_affected_mse" in row else noncore_mse,
                noncore_mse,
            )
            row[f"{method}_worse_than_identity"] = bool(affected_mse > identity_affected_mse)
        row["weighted_beats_balanced_residual"] = (
            row["weighted_residual_affected_mse"] < row["balanced_residual_affected_mse"]
        )
        row["weighted_beats_direct"] = (
            row["weighted_residual_affected_mse"] < row["direct_affected_mse"]
        )
        row["weighted_beats_residual"] = (
            row["weighted_residual_affected_mse"] < row["residual_affected_mse"]
        )
        row["balanced_beats_weighted_residual"] = (
            row["balanced_residual_affected_mse"] < row["weighted_residual_affected_mse"]
        )
        row["weighted_to_balanced_residual_mse_ratio"] = safe_ratio(
            row["weighted_residual_affected_mse"],
            row["balanced_residual_affected_mse"],
        )
        row["weighted_to_direct_mse_ratio"] = safe_ratio(
            row["weighted_residual_affected_mse"],
            row["direct_affected_mse"],
        )
        row["weighted_to_residual_mse_ratio"] = safe_ratio(
            row["weighted_residual_affected_mse"],
            row["residual_affected_mse"],
        )
        rows.append(row)

    per_sample = pd.DataFrame(rows)
    per_sample["blend_severity_bin"] = balanced_helpers.blend_severity_bins(
        per_sample["blend_severity_score"]
    )
    per_sample["core_overlap_bin"] = balanced_helpers.core_overlap_bins(
        per_sample["core_obstruction_fraction"]
    )
    return per_sample


def aggregate_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity_affected = float(per_sample["identity_affected_mse"].mean())
    identity_core = float(per_sample["identity_core_affected_mse"].mean())
    identity_noncore = float(per_sample["identity_noncore_affected_mse"].mean())
    for method in METHODS:
        affected_mse = float(per_sample[f"{method}_affected_mse"].mean())
        core_mse = float(per_sample[f"{method}_core_affected_mse"].mean())
        noncore_mse = float(per_sample[f"{method}_noncore_affected_mse"].mean())
        row = {
            "split": str(per_sample["split"].iloc[0]),
            "method": method,
            "n": int(len(per_sample)),
            "whole_mse": float(per_sample[f"{method}_mse"].mean()),
            "whole_mae": float(per_sample[f"{method}_mae"].mean()),
            "psnr": float(per_sample[f"{method}_psnr"].mean()),
            "ssim": float(per_sample[f"{method}_ssim"].mean()),
            "affected_mse": affected_mse,
            "affected_mae": float(per_sample[f"{method}_affected_mae"].mean()),
            "core_affected_mse": core_mse,
            "core_affected_mae": float(per_sample[f"{method}_core_affected_mae"].mean()),
            "noncore_affected_mse": noncore_mse,
            "noncore_affected_mae": float(per_sample[f"{method}_noncore_affected_mae"].mean()),
            "improvement_vs_identity": safe_ratio(identity_affected, affected_mse),
            "core_improvement_vs_identity": safe_ratio(identity_core, core_mse),
            "noncore_improvement_vs_identity": safe_ratio(identity_noncore, noncore_mse),
            "worse_than_identity_count": int(per_sample[f"{method}_worse_than_identity"].sum()),
            "worse_than_identity_fraction": float(per_sample[f"{method}_worse_than_identity"].mean()),
            "mean_mask_fraction": float(per_sample["mask_fraction"].mean()),
            "mean_core_obstruction_fraction": float(per_sample["core_obstruction_fraction"].mean()),
        }
        if method == "weighted_residual":
            row["weighted_beats_balanced_residual_fraction"] = float(
                per_sample["weighted_beats_balanced_residual"].mean()
            )
            row["weighted_beats_direct_fraction"] = float(per_sample["weighted_beats_direct"].mean())
            row["weighted_beats_residual_fraction"] = float(per_sample["weighted_beats_residual"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ("generation_difficulty", "blend_severity_bin", "core_overlap_bin")
    for group_col in group_cols:
        frame = per_sample.dropna(subset=[group_col])
        if frame.empty:
            continue
        for group, group_frame in frame.groupby(group_col, observed=True):
            identity_affected = float(group_frame["identity_affected_mse"].mean())
            for method in METHODS:
                affected_mse = float(group_frame[f"{method}_affected_mse"].mean())
                rows.append(
                    {
                        "split": str(group_frame["split"].iloc[0]),
                        "grouping": group_col,
                        "group": str(group),
                        "method": method,
                        "n": int(len(group_frame)),
                        "affected_mse": affected_mse,
                        "affected_mae": float(group_frame[f"{method}_affected_mae"].mean()),
                        "core_affected_mse": float(group_frame[f"{method}_core_affected_mse"].mean()),
                        "noncore_affected_mse": float(group_frame[f"{method}_noncore_affected_mse"].mean()),
                        "whole_mse": float(group_frame[f"{method}_mse"].mean()),
                        "ssim": float(group_frame[f"{method}_ssim"].mean()),
                        "improvement_vs_identity": safe_ratio(identity_affected, affected_mse),
                        "worse_than_identity_count": int(
                            group_frame[f"{method}_worse_than_identity"].sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def model_win_rates(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in per_sample.groupby("split"):
        rows.append(
            {
                "split": split,
                "n": int(len(frame)),
                "weighted_vs_balanced_win_rate": float(frame["weighted_beats_balanced_residual"].mean()),
                "weighted_vs_direct_win_rate": float(frame["weighted_beats_direct"].mean()),
                "weighted_vs_old_residual_win_rate": float(frame["weighted_beats_residual"].mean()),
                "weighted_to_balanced_aggregate_mse_ratio": safe_ratio(
                    float(frame["weighted_residual_affected_mse"].mean()),
                    float(frame["balanced_residual_affected_mse"].mean()),
                ),
                "weighted_to_direct_aggregate_mse_ratio": safe_ratio(
                    float(frame["weighted_residual_affected_mse"].mean()),
                    float(frame["direct_affected_mse"].mean()),
                ),
                "weighted_to_old_residual_aggregate_mse_ratio": safe_ratio(
                    float(frame["weighted_residual_affected_mse"].mean()),
                    float(frame["residual_affected_mse"].mean()),
                ),
                "weighted_worse_than_identity_count": int(frame["weighted_residual_worse_than_identity"].sum()),
                "balanced_worse_than_identity_count": int(frame["balanced_residual_worse_than_identity"].sum()),
            }
        )
    return pd.DataFrame(rows)


def method_label(method: str) -> str:
    return {
        "identity": "identity",
        "threshold": "threshold",
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


def format_metrics_table(aggregate: pd.DataFrame) -> str:
    rows = [
        "| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Core affected MSE | Non-core affected MSE | Improvement vs identity | Worse than identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in aggregate.iterrows():
        rows.append(
            "| {method} | {whole_mse:.6f} | {whole_mae:.6f} | {psnr:.3f} | {ssim:.6f} | "
            "{affected_mse:.6f} | {core_mse:.6f} | {noncore_mse:.6f} | {ratio:.2f}x | {worse}/{n} |".format(
                method=method_label(str(row["method"])),
                whole_mse=row["whole_mse"],
                whole_mae=row["whole_mae"],
                psnr=row["psnr"],
                ssim=row["ssim"],
                affected_mse=row["affected_mse"],
                core_mse=row["core_affected_mse"],
                noncore_mse=row["noncore_affected_mse"],
                ratio=row["improvement_vs_identity"],
                worse=int(row["worse_than_identity_count"]),
                n=int(row["n"]),
            )
        )
    return "\n".join(rows)


def plot_metric_bars(
    aggregate: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    include_all: bool = True,
) -> Path:
    methods = list(METHODS if include_all else LEARNED_METHODS)
    splits = ["normal", "stress"]
    x = np.arange(len(splits))
    width = 0.12 if include_all else 0.18
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    offsets = np.linspace(
        -width * (len(methods) - 1) / 2.0,
        width * (len(methods) - 1) / 2.0,
        len(methods),
    )
    for offset, method in zip(offsets, methods):
        values = []
        for split in splits:
            sub = aggregate[(aggregate["split"] == split) & (aggregate["method"] == method)]
            values.append(float(sub[metric].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + offset, values, width, color=method_color(method), label=method_label(method))
    ax.set_xticks(x, ["Normal", "Stress"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    return save_fig(fig, path)


def scatter_weighted_vs_balanced(per_sample: pd.DataFrame, path: Path) -> Path:
    frame = per_sample[
        ["split", "balanced_residual_affected_mse", "weighted_residual_affected_mse"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(5.7, 4.7))
    for split, color in (("normal", "#5f8a4b"), ("stress", "#2f6f8f")):
        sub = frame[frame["split"] == split]
        ax.scatter(
            sub["balanced_residual_affected_mse"],
            sub["weighted_residual_affected_mse"],
            s=14,
            alpha=0.34,
            color=color,
            label=split,
            edgecolors="none",
        )
    finite_max = float(np.nanmax(frame[["balanced_residual_affected_mse", "weighted_residual_affected_mse"]].to_numpy()))
    limit = max(finite_max, 1e-4)
    ax.plot([1e-6, limit], [1e-6, limit], color="#8a4f49", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Thayer-BR v0.1 affected MSE")
    ax.set_ylabel("Thayer-BR v0.2 affected MSE")
    ax.set_title("Weighted vs BR v0.1 Per Sample")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return save_fig(fig, path)


def ratio_histogram(per_sample: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    for split, color in (("normal", "#5f8a4b"), ("stress", "#2f6f8f")):
        values = (
            per_sample.loc[per_sample["split"] == split, "weighted_to_balanced_residual_mse_ratio"]
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
        ax.hist(clipped, bins=35, alpha=0.50, color=color, label=split)
    ax.axvline(1.0, color="#8a4f49", linewidth=1.5, label="parity")
    ax.set_xlabel("Thayer-BR v0.2 / Thayer-BR v0.1 affected MSE")
    ax.set_ylabel("Samples")
    ax.set_title("Weighted-to-BR v0.1 Error Ratio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_fig(fig, path)


def grouped_performance_chart(
    grouped: pd.DataFrame,
    grouping: str,
    path: Path,
    title: str,
) -> Path | None:
    frame = grouped[
        (grouped["split"] == "stress")
        & (grouped["grouping"] == grouping)
        & (grouped["method"].isin(LEARNED_METHODS))
    ].copy()
    if frame.empty:
        return None
    preferred = ["low", "medium", "high", "easy", "hard"]
    order = [item for item in preferred if item in set(frame["group"])]
    if not order:
        order = sorted(frame["group"].unique())
    x = np.arange(len(order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for idx, method in enumerate(LEARNED_METHODS):
        method_frame = frame[frame["method"] == method].set_index("group")
        values = [
            float(method_frame.loc[group, "affected_mse"])
            if group in method_frame.index
            else np.nan
            for group in order
        ]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            color=method_color(method),
            label=method_label(method),
        )
    ax.set_xticks(x, order)
    ax.set_ylabel("Affected-region MSE")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    return save_fig(fig, path)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.12, 0.0),
    alpha: float = 0.42,
) -> np.ndarray:
    out = np.clip(image.copy(), 0.0, 1.0)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def save_example_grid(
    sample: dict[str, Any],
    row: pd.Series,
    preds: dict[str, list[np.ndarray]],
    threshold: float,
    path: Path,
    title: str,
) -> Path:
    idx = int(row["index"])
    target = sample["target"]
    blended = sample["blended"]
    affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
    core = balanced_helpers.target_core_mask(target)
    core_affected = np.logical_and(affected, core)
    weighted = preds["weighted_residual"][idx]
    balanced = preds["balanced_residual"][idx]
    weighted_error = np.abs(weighted - target).mean(axis=-1)
    balanced_error = np.abs(balanced - target).mean(axis=-1)
    panels = [
        (target, "Target"),
        (overlay_mask(blended, affected), "Affected blend"),
        (overlay_mask(blended, core_affected, color=(0.1, 0.75, 0.25)), "Core affected"),
        (preds["direct"][idx], "Thayer-Direct"),
        (preds["residual"][idx], "Thayer-Residual"),
        (balanced, "Thayer-BR v0.1"),
        (weighted, "Thayer-BR v0.2"),
        (balanced_error, "BR v0.1 error"),
        (weighted_error, "BR v0.2 error"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 2.45))
    for ax, (image, panel_title) in zip(axes, panels):
        if image.ndim == 2:
            vmax = max(0.05, float(np.nanpercentile(image, 99)))
            ax.imshow(image, cmap="magma", vmin=0.0, vmax=vmax)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    metadata = {
        "split": str(row["split"]),
        "index": int(row["index"]),
        "weighted_affected_mse": float(row["weighted_residual_affected_mse"]),
        "balanced_affected_mse": float(row["balanced_residual_affected_mse"]),
        "weighted_to_balanced_ratio": float(row["weighted_to_balanced_residual_mse_ratio"]),
        "weighted_core_affected_mse": float(row["weighted_residual_core_affected_mse"]),
        "balanced_core_affected_mse": float(row["balanced_residual_core_affected_mse"]),
        "blend_severity_bin": str(row["blend_severity_bin"]),
        "core_overlap_bin": str(row["core_overlap_bin"]),
    }
    save_json(path.with_suffix(".json"), metadata)
    return save_fig(fig, path, dpi=180)


def choose_examples(per_sample: pd.DataFrame) -> dict[str, pd.Series | None]:
    finite = per_sample.replace([np.inf, -np.inf], np.nan)
    examples: dict[str, pd.Series | None] = {
        "weighted_improves": None,
        "balanced_beats_weighted": None,
        "remaining_failure": None,
    }
    improves = finite[
        (finite["weighted_beats_balanced_residual"])
        & (finite["weighted_residual_affected_mse"] < finite["identity_affected_mse"])
    ].copy()
    if not improves.empty:
        examples["weighted_improves"] = improves.sort_values(
            ["weighted_to_balanced_residual_mse_ratio", "identity_affected_mse"],
            ascending=[True, False],
        ).iloc[0]

    balanced_wins = finite[finite["balanced_beats_weighted_residual"]].copy()
    if not balanced_wins.empty:
        examples["balanced_beats_weighted"] = balanced_wins.sort_values(
            ["weighted_to_balanced_residual_mse_ratio", "weighted_residual_affected_mse"],
            ascending=False,
        ).iloc[0]

    failures = finite[
        (finite["weighted_residual_worse_than_identity"])
        | (finite["weighted_residual_improvement_ratio"] <= 1.25)
    ].copy()
    if failures.empty:
        failures = finite[finite["split"] == "stress"].copy()
    if not failures.empty:
        examples["remaining_failure"] = failures.sort_values(
            ["weighted_residual_affected_mse", "core_obstruction_fraction"],
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


def preds_for_row(
    row: pd.Series,
    normal_preds: dict[str, list[np.ndarray]],
    stress_preds: dict[str, list[np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    if row["split"] == "normal":
        return normal_preds
    if row["split"] == "stress":
        return stress_preds
    raise KeyError(f"Unknown split: {row['split']}")


def save_all_figures(
    run_dir: Path,
    aggregate: pd.DataFrame,
    grouped: pd.DataFrame,
    per_sample: pd.DataFrame,
    normal_samples: list[dict[str, Any]],
    stress_samples: list[dict[str, Any]],
    normal_preds: dict[str, list[np.ndarray]],
    stress_preds: dict[str, list[np.ndarray]],
    threshold: float,
) -> dict[str, str | None]:
    output_dir = run_dir / "paper_figures"
    written: dict[str, str | None] = {}
    written["affected_region_mse_bar"] = project_relative(
        plot_metric_bars(
            aggregate,
            "affected_mse",
            "Affected-Region MSE",
            "Affected-region MSE",
            output_dir / "affected_region_mse_bar.png",
            include_all=True,
        )
    )
    written["normal_vs_stress_improvement_ratio"] = project_relative(
        plot_metric_bars(
            aggregate,
            "improvement_vs_identity",
            "Normal vs Stress Improvement",
            "Identity affected MSE / model affected MSE",
            output_dir / "normal_vs_stress_improvement_ratio.png",
            include_all=False,
        )
    )
    written["core_affected_mse"] = project_relative(
        plot_metric_bars(
            aggregate,
            "core_affected_mse",
            "Core Affected MSE",
            "Core affected MSE",
            output_dir / "core_affected_mse_comparison.png",
            include_all=True,
        )
    )
    written["noncore_affected_mse"] = project_relative(
        plot_metric_bars(
            aggregate,
            "noncore_affected_mse",
            "Non-Core Affected MSE",
            "Non-core affected MSE",
            output_dir / "noncore_affected_mse_comparison.png",
            include_all=True,
        )
    )
    written["weighted_vs_br_v01_scatter"] = project_relative(
        scatter_weighted_vs_balanced(
            per_sample,
            output_dir / "weighted_vs_br_v01_per_sample_scatter.png",
        )
    )
    written["weighted_br_ratio_histogram"] = project_relative(
        ratio_histogram(
            per_sample,
            output_dir / "hist_weighted_to_br_v01_affected_mse_ratio.png",
        )
    )
    written["worse_than_identity_counts"] = project_relative(
        plot_metric_bars(
            aggregate,
            "worse_than_identity_count",
            "Worse-than-Identity Counts",
            "Samples worse than identity",
            output_dir / "worse_than_identity_count_chart.png",
            include_all=False,
        )
    )
    maybe = grouped_performance_chart(
        grouped,
        "core_overlap_bin",
        output_dir / "stress_performance_by_core_overlap_bin.png",
        "Stress Performance by Core-Overlap Bin",
    )
    written["core_overlap_bin"] = project_relative(maybe) if maybe else None
    maybe = grouped_performance_chart(
        grouped,
        "blend_severity_bin",
        output_dir / "stress_performance_by_blend_severity_bin.png",
        "Stress Performance by Blend-Severity Bin",
    )
    written["blend_severity_bin"] = project_relative(maybe) if maybe else None

    examples = choose_examples(per_sample)
    for key, filename, title in (
        (
            "weighted_improves",
            "qualitative_weighted_improves_over_br_v01.png",
            "Thayer-BR v0.2 Improves Over v0.1",
        ),
        (
            "balanced_beats_weighted",
            "qualitative_br_v01_beats_weighted.png",
            "Thayer-BR v0.1 Beats v0.2",
        ),
        (
            "remaining_failure",
            "qualitative_weighted_remaining_failure.png",
            "Remaining Failure Case",
        ),
    ):
        row = examples[key]
        if row is None:
            written[key] = None
            continue
        sample = sample_for_row(row, normal_samples, stress_samples)
        preds = preds_for_row(row, normal_preds, stress_preds)
        paper_path = output_dir / filename
        written[key] = project_relative(
            save_example_grid(sample, row, preds, threshold, paper_path, title)
        )
        grid_path = run_dir / "example_grids" / filename
        save_example_grid(sample, row, preds, threshold, grid_path, title)
    return written


def run_multiseed(
    run_dir: Path,
    config: dict[str, Any],
    test_images: np.ndarray,
    direct_model: nn.Module,
    residual_model: nn.Module,
    balanced_model: nn.Module,
    weighted_model: nn.Module,
    device: torch.device,
    settings: WeightedSettings,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    seed_base = int(config["seed"])
    normal_seeds = [seed_base + 9000 + i for i in range(settings.multiseed_count)]
    stress_seeds = [
        int(stress_helpers.STRESS_DEFAULTS["seed"]) + 100 + i
        for i in range(settings.multiseed_count)
    ]
    for split, seeds in (("normal", normal_seeds), ("stress", stress_seeds)):
        for seed_index, seed in enumerate(seeds):
            print(f"Multi-seed evaluation: {split} seed {seed}.", flush=True)
            if split == "normal":
                samples = balanced_helpers.normal_blends(
                    test_images,
                    settings.multiseed_blends,
                    config,
                    seed=seed,
                    component=f"multiseed_normal_{seed}",
                )
                threshold = settings.affected_region_threshold
            else:
                stress = dict(stress_helpers.STRESS_DEFAULTS)
                stress["n_stress_blends"] = settings.multiseed_blends
                stress["seed"] = seed
                samples = stress_helpers.generate_stress_blends(
                    test_images[: int(stress["stress_source_subset"])],
                    stress,
                )
                threshold = float(stress["affected_region_threshold"])
            preds, _stats = predict_all(
                samples,
                direct_model,
                residual_model,
                balanced_model,
                weighted_model,
                device,
                batch_size,
            )
            per_sample = compute_per_sample(split, samples, preds, threshold)
            aggregate = aggregate_metrics(per_sample)
            for _, row in aggregate.iterrows():
                rows.append(
                    {
                        "split": split,
                        "seed_index": seed_index,
                        "seed": seed,
                        "method": row["method"],
                        "n": int(row["n"]),
                        "affected_mse": float(row["affected_mse"]),
                        "affected_mae": float(row["affected_mae"]),
                        "whole_mse": float(row["whole_mse"]),
                        "ssim": float(row["ssim"]),
                        "core_affected_mse": float(row["core_affected_mse"]),
                        "noncore_affected_mse": float(row["noncore_affected_mse"]),
                        "improvement_vs_identity": float(row["improvement_vs_identity"]),
                        "worse_than_identity_count": int(row["worse_than_identity_count"]),
                        "weighted_beats_balanced_count": int(per_sample["weighted_beats_balanced_residual"].sum()),
                    }
                )
            del samples, preds, per_sample, aggregate
            gc.collect()
    result = pd.DataFrame(rows)
    save_csv(run_dir / "tables" / "multiseed_results.csv", result)
    summary_rows: list[dict[str, Any]] = []
    for split, split_frame in result.groupby("split"):
        seed_winners: list[str] = []
        for seed, seed_frame in split_frame.groupby("seed"):
            learned = seed_frame[seed_frame["method"].isin(LEARNED_METHODS)].sort_values("affected_mse")
            seed_winners.append(str(learned.iloc[0]["method"]))
        for method in METHODS:
            sub = split_frame[split_frame["method"] == method]
            summary_rows.append(
                {
                    "split": split,
                    "method": method,
                    "seeds": int(sub["seed"].nunique()),
                    "mean_affected_mse": float(sub["affected_mse"].mean()),
                    "std_affected_mse": float(sub["affected_mse"].std(ddof=1)),
                    "mean_core_affected_mse": float(sub["core_affected_mse"].mean()),
                    "mean_noncore_affected_mse": float(sub["noncore_affected_mse"].mean()),
                    "mean_improvement_vs_identity": float(sub["improvement_vs_identity"].mean()),
                    "std_improvement_vs_identity": float(sub["improvement_vs_identity"].std(ddof=1)),
                    "mean_worse_than_identity_count": float(sub["worse_than_identity_count"].mean()),
                    "seed_winners": ", ".join(seed_winners),
                    "weighted_seed_win_rate": float(np.mean([winner == "weighted_residual" for winner in seed_winners])),
                }
            )
    summary = pd.DataFrame(summary_rows)
    save_csv(run_dir / "tables" / "multiseed_summary.csv", summary)
    return result, summary


def stop_condition_warnings(
    split_name: str,
    aggregate: pd.DataFrame,
    weighted_stats: dict[str, float],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    stops: list[str] = []
    frame = aggregate.set_index("method")
    identity = frame.loc["identity"]
    weighted = frame.loc["weighted_residual"]
    balanced = frame.loc["balanced_residual"]
    if weighted["affected_mse"] > identity["affected_mse"]:
        stops.append(
            f"{split_name}: weighted residual is worse than identity on aggregate affected-region MSE"
        )
    if weighted_stats["residual_pred_abs_max"] > 5.0:
        stops.append(f"{split_name}: weighted residual predictions exceeded absolute value 5")
    if weighted_stats["reconstruction_preclip_total_clip_fraction"] > 0.20:
        stops.append(
            f"{split_name}: more than 20% of weighted reconstruction pixels needed clipping"
        )
    if weighted_stats["reconstruction_preclip_total_clip_fraction"] > 0.05:
        warnings.append(
            f"{split_name}: more than 5% of weighted reconstruction pixels needed clipping"
        )
    if weighted["affected_mse"] > balanced["affected_mse"] * 1.5:
        warnings.append(
            f"{split_name}: weighted affected MSE is more than 50% worse than Thayer-BR v0.1"
        )
    if weighted["worse_than_identity_count"] > balanced["worse_than_identity_count"]:
        warnings.append(
            f"{split_name}: weighted has more worse-than-identity cases than Thayer-BR v0.1"
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
    if warnings or stops:
        lines = ["# Weighted Residual Diagnostics", ""]
        if stops:
            lines.extend(["## Stop Conditions", ""])
            lines.extend(f"- {item}" for item in stops)
            lines.append("")
        if warnings:
            lines.extend(["## Warnings", ""])
            lines.extend(f"- {item}" for item in warnings)
            lines.append("")
        safe_write_text(run_dir / "diagnostics" / "weighted_residual_diagnostics.md", "\n".join(lines))
    if normal_per_sample is not None:
        save_csv(
            run_dir / "diagnostics" / "normal_largest_weighted_regressions.csv",
            normal_per_sample.sort_values("weighted_to_balanced_residual_mse_ratio", ascending=False).head(25),
        )
    if stress_per_sample is not None:
        save_csv(
            run_dir / "diagnostics" / "stress_largest_weighted_regressions.csv",
            stress_per_sample.sort_values("weighted_to_balanced_residual_mse_ratio", ascending=False).head(25),
        )


def decide_outcome(
    aggregate: pd.DataFrame,
    win_rates: pd.DataFrame,
) -> dict[str, Any]:
    frame = aggregate.set_index(["split", "method"])
    normal_weighted = frame.loc[("normal", "weighted_residual")]
    normal_balanced = frame.loc[("normal", "balanced_residual")]
    stress_weighted = frame.loc[("stress", "weighted_residual")]
    stress_balanced = frame.loc[("stress", "balanced_residual")]
    stress_matches = stress_weighted["affected_mse"] <= stress_balanced["affected_mse"] * 1.02
    core_improves = stress_weighted["core_affected_mse"] < stress_balanced["core_affected_mse"]
    normal_ok = normal_weighted["affected_mse"] <= normal_balanced["affected_mse"] * 1.15
    worse_ok = (
        stress_weighted["worse_than_identity_count"]
        <= stress_balanced["worse_than_identity_count"]
    )
    normal_improves = normal_weighted["affected_mse"] < normal_balanced["affected_mse"]
    stress_improves = stress_weighted["affected_mse"] < stress_balanced["affected_mse"]
    if normal_improves and stress_improves and core_improves and worse_ok:
        verdict = "strong_new_best"
        recommendation = "Thayer-BR v0.2 should become the current best model."
    elif stress_matches and core_improves and normal_ok and worse_ok:
        verdict = "useful_tradeoff"
        recommendation = "Thayer-BR v0.2 is useful for stress/core cases, but compare tradeoffs before replacing v0.1."
    elif core_improves and not normal_ok:
        verdict = "core_tradeoff"
        recommendation = "Weighted loss is a targeted ablation/tradeoff, not a replacement for Thayer-BR v0.1."
    else:
        verdict = "reject"
        recommendation = "Weighted loss should not replace Thayer-BR v0.1."
    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "normal_weighted_affected_mse": float(normal_weighted["affected_mse"]),
        "normal_balanced_affected_mse": float(normal_balanced["affected_mse"]),
        "stress_weighted_affected_mse": float(stress_weighted["affected_mse"]),
        "stress_balanced_affected_mse": float(stress_balanced["affected_mse"]),
        "stress_weighted_core_mse": float(stress_weighted["core_affected_mse"]),
        "stress_balanced_core_mse": float(stress_balanced["core_affected_mse"]),
        "stress_weighted_worse_than_identity": int(stress_weighted["worse_than_identity_count"]),
        "stress_balanced_worse_than_identity": int(stress_balanced["worse_than_identity_count"]),
        "normal_weighted_worse_than_identity": int(normal_weighted["worse_than_identity_count"]),
        "normal_balanced_worse_than_identity": int(normal_balanced["worse_than_identity_count"]),
        "win_rates": win_rates.to_dict(orient="records"),
    }


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
    return "\n".join(lines[:start] + section_lines + lines[end:]).rstrip() + "\n"


def append_experiment_log(
    run_dir: Path,
    best_checkpoint: Path,
    final_checkpoint: Path,
    settings: WeightedSettings,
    loss_config: WeightedLossConfig,
    variant_name: str,
    composition: dict[str, Any],
    history: pd.DataFrame,
    aggregate: pd.DataFrame,
    win_rates: pd.DataFrame,
    multiseed_summary: pd.DataFrame | None,
    outcome: dict[str, Any],
    warnings: list[str],
    stops: list[str],
    figure_paths: dict[str, str | None],
) -> None:
    normal_aggregate = aggregate[aggregate["split"] == "normal"]
    stress_aggregate = aggregate[aggregate["split"] == "stress"]
    status = "completed" if not stops else "suspicious; inspect diagnostics"
    warning_text = "None." if not warnings and not stops else "; ".join(stops + warnings) + "."
    best_epoch = int(history["best_epoch"].iloc[-1])
    best_val = float(history["best_val_loss"].iloc[-1])
    final_train = float(history["train_loss"].iloc[-1])
    final_val = float(history["val_loss"].iloc[-1])
    figure_lines = [f"- `{path}`" for path in figure_paths.values() if path is not None]
    if not figure_lines:
        figure_lines = ["- Not generated before stop condition."]
    multiseed_text = "Multi-seed evaluation was skipped."
    if multiseed_summary is not None and not multiseed_summary.empty:
        normal_weighted = multiseed_summary[
            (multiseed_summary["split"] == "normal")
            & (multiseed_summary["method"] == "weighted_residual")
        ].iloc[0]
        stress_weighted = multiseed_summary[
            (multiseed_summary["split"] == "stress")
            & (multiseed_summary["method"] == "weighted_residual")
        ].iloc[0]
        multiseed_text = (
            f"Weighted multi-seed improvement: normal "
            f"{normal_weighted['mean_improvement_vs_identity']:.2f} +/- "
            f"{normal_weighted['std_improvement_vs_identity']:.2f}x; stress "
            f"{stress_weighted['mean_improvement_vs_identity']:.2f} +/- "
            f"{stress_weighted['std_improvement_vs_identity']:.2f}x."
        )
    section = f"""

## Experiment 4: Thayer-BR v0.2 Weighted Residual Loss

Status: {status}. Run directory: `{project_relative(run_dir)}`.

### Training Settings

- Task: residual prediction, `blended -> blended - target`.
- Reconstruction: `blended - predicted_residual`.
- Variant: `{variant_name}`.
- Train/validation blends: {settings.n_train_blends:,} / {settings.n_val_blends:,}.
- Normal held-out/stress test blends: {settings.n_normal_test_blends:,} / {settings.n_stress_blends:,}.
- Epochs: {settings.num_epochs}.
- Batch size: {int(history['batch_size'].iloc[-1])}.
- Composition target: 50% normal/random, 30% high-overlap/core-obstruction, 20% brightness/size stress.
- Actual training composition: {json.dumps(composition["train"]["components"])}.
- Loss formula: `sum(weight * (predicted_residual - true_residual)^2) / sum(weight)`, normalized over channels.
- Loss weights: background `{loss_config.background_weight}`, affected extra `{loss_config.affected_extra_weight}`, core affected extra `{loss_config.core_extra_weight}`.
- Affected threshold: `{loss_config.affected_threshold}`.
- Core mask: central aperture `{loss_config.core_aperture_fraction}` with bright-core fraction `{loss_config.core_brightness_fraction}`.
- Saved best model checkpoint: `{project_relative(best_checkpoint)}`.
- Saved final model checkpoint: `{project_relative(final_checkpoint)}`.
- Best validation loss: {best_val:.6f} at epoch {best_epoch}.
- Final train/validation loss: {final_train:.6f} / {final_val:.6f}.

### Normal Held-Out Metrics

{format_metrics_table(normal_aggregate)}

### Hard Stress-Test Metrics

{format_metrics_table(stress_aggregate)}

### Core and Model-Win Summary

- Stress Thayer-BR v0.1 affected MSE: {outcome['stress_balanced_affected_mse']:.6f}.
- Stress Thayer-BR v0.2 affected MSE: {outcome['stress_weighted_affected_mse']:.6f}.
- Stress Thayer-BR v0.1 core affected MSE: {outcome['stress_balanced_core_mse']:.6f}.
- Stress Thayer-BR v0.2 core affected MSE: {outcome['stress_weighted_core_mse']:.6f}.
- Stress worse-than-identity cases: v0.1 `{outcome['stress_balanced_worse_than_identity']}`, v0.2 `{outcome['stress_weighted_worse_than_identity']}`.
- Model win rates: `{win_rates.to_dict(orient='records')}`.
- {multiseed_text}

### Interpretation

Verdict: `{outcome['verdict']}`. {outcome['recommendation']}

Coherence/suspicion status: {warning_text}

Paper figures:

{chr(10).join(figure_lines)}
"""
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(section)


def update_results_interpretation(outcome: dict[str, Any], run_dir: Path) -> None:
    text = RESULTS_INTERPRETATION_PATH.read_text(encoding="utf-8")
    section = f"""## Weighted Residual Loss Experiment

Experiment 4 tested Thayer-BR v0.2, which keeps residual prediction but gives
more loss weight to affected pixels and extra weight to affected target-core
pixels. The run directory is `{project_relative(run_dir)}`.

Outcome: {outcome['recommendation']}

The key comparison is whether the weighted objective improves contaminated and
core-overlap reconstruction without sacrificing the balanced residual model's
normal-set robustness. Stress affected MSE changed from
`{outcome['stress_balanced_affected_mse']:.6f}` for Thayer-BR v0.1 to
`{outcome['stress_weighted_affected_mse']:.6f}` for Thayer-BR v0.2. Stress
core affected MSE changed from `{outcome['stress_balanced_core_mse']:.6f}` to
`{outcome['stress_weighted_core_mse']:.6f}`.

This result should be framed as a weighted-loss ablation unless it improves the
stress/core metrics without increasing worse-than-identity cases or causing a
large normal-performance drop.
"""
    RESULTS_INTERPRETATION_PATH.write_text(
        replace_or_append_section(text, "## Weighted Residual Loss Experiment", section),
        encoding="utf-8",
    )


def update_paper_plan(outcome: dict[str, Any], run_dir: Path) -> None:
    text = PAPER_PLAN_PATH.read_text(encoding="utf-8")
    role = (
        "final model improvement"
        if outcome["verdict"] in {"strong_new_best", "useful_tradeoff"}
        else "negative/ablation result"
    )
    section = f"""## Experiment 4 Addition

Experiment 4 adds Thayer-BR v0.2, a weighted residual-loss run that emphasizes
affected pixels and affected target-core pixels while preserving the residual
prediction formulation. Run directory: `{project_relative(run_dir)}`.

Paper role: {role}.

- If used as an improvement, report where the weighted loss improves stress or
  core-obstructed reconstruction and state any normal-set tradeoff.
- If used as an ablation, report that simply upweighting affected/core pixels
  did not outperform the balanced hard-case residual objective enough to replace
  Thayer-BR v0.1.
- Do not present the weighted model as a new architecture; it is an objective
  change on the same residual U-Net.
"""
    PAPER_PLAN_PATH.write_text(
        replace_or_append_section(text, "## Experiment 4 Addition", section),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    settings = WeightedSettings(
        n_train_blends=args.n_train_blends,
        n_val_blends=args.n_val_blends,
        n_normal_test_blends=args.n_normal_test_blends,
        n_stress_blends=args.n_stress_blends,
        train_source_subset=args.train_source_subset,
        val_source_subset=args.val_source_subset,
        test_source_subset=args.test_source_subset,
        num_epochs=args.num_epochs,
        affected_region_threshold=args.affected_threshold,
        multiseed_count=0 if args.skip_multiseed else args.multiseed_count,
        multiseed_blends=args.multiseed_blends,
    )
    loss_config = WeightedLossConfig(
        background_weight=args.background_weight,
        affected_extra_weight=args.affected_extra_weight,
        core_extra_weight=args.core_extra_weight,
        affected_threshold=args.affected_threshold,
        core_aperture_fraction=args.core_aperture_fraction,
        core_brightness_fraction=args.core_brightness_fraction,
    )
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, best_checkpoint, final_checkpoint = make_run_paths(output_root, stamp)

    checkpoint_dir = output_root / "checkpoints"
    checkpoint_before = checkpoint_inventory(checkpoint_dir)
    save_json(run_dir / "logs" / "protected_checkpoint_metadata_before.json", checkpoint_before)

    save_yaml(
        run_dir / "logs" / "run_config.yaml",
        {
            "project_root": ".",
            "timestamp": stamp,
            "variant_name": args.variant_name,
            "settings": settings.__dict__,
            "loss_config": loss_config.__dict__,
            "config": config,
            "direct_checkpoint": project_relative(args.direct_checkpoint),
            "residual_checkpoint": project_relative(args.residual_checkpoint),
            "balanced_checkpoint": project_relative(args.balanced_checkpoint),
            "best_checkpoint": project_relative(best_checkpoint),
            "final_checkpoint": project_relative(final_checkpoint),
            "stress_settings": {
                **stress_helpers.STRESS_DEFAULTS,
                "n_stress_blends": settings.n_stress_blends,
            },
        },
    )

    stops: list[str] = []
    warnings: list[str] = []
    direct_checkpoint = resolve_checkpoint(args.direct_checkpoint)
    residual_checkpoint = resolve_checkpoint(args.residual_checkpoint)
    balanced_checkpoint = resolve_checkpoint(args.balanced_checkpoint)
    for path, label in (
        (direct_checkpoint, "direct checkpoint"),
        (residual_checkpoint, "old residual checkpoint"),
        (balanced_checkpoint, "Thayer-BR v0.1 checkpoint"),
    ):
        if not path.exists():
            stops.append(f"Missing {label}: {project_relative(path)}")
    if stops:
        write_diagnostics(run_dir, warnings, stops)
        print("\n".join(stops), flush=True)
        return 2

    seed = int(config["seed"])
    seed_everything(seed)
    device = gd_train.resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    try:
        direct_model = balanced_helpers.load_direct_model(direct_checkpoint, config["model"], device)
        residual_model = balanced_helpers.load_residual_model(residual_checkpoint, config["model"], device)
        balanced_model = balanced_helpers.load_residual_model(balanced_checkpoint, config["model"], device)
    except Exception as exc:
        stops.append(f"Could not load comparison checkpoints: {exc}")
        write_diagnostics(run_dir, warnings, stops)
        print(stops[-1], flush=True)
        return 2

    print("Loading dataset and reconstructing splits.", flush=True)
    balanced_settings = balanced_helpers.BalancedSettings(
        n_train_blends=settings.n_train_blends,
        n_val_blends=settings.n_val_blends,
        n_normal_test_blends=settings.n_normal_test_blends,
        n_stress_blends=settings.n_stress_blends,
        train_source_subset=settings.train_source_subset,
        val_source_subset=settings.val_source_subset,
        test_source_subset=settings.test_source_subset,
        num_epochs=settings.num_epochs,
        normal_fraction=settings.normal_fraction,
        high_overlap_fraction=settings.high_overlap_fraction,
        brightness_size_fraction=settings.brightness_size_fraction,
        affected_region_threshold=settings.affected_region_threshold,
    )
    train_images, val_images, test_images = balanced_helpers.load_split_subsets(
        config,
        balanced_settings,
    )

    print("Generating balanced hard-case training blends.", flush=True)
    train_blends, train_composition = balanced_helpers.generate_balanced_blends(
        train_images,
        total=settings.n_train_blends,
        config=config,
        seed=seed + 1000,
        settings=balanced_settings,
        split_name="train",
    )
    print("Generating balanced hard-case validation blends.", flush=True)
    val_blends, val_composition = balanced_helpers.generate_balanced_blends(
        val_images,
        total=settings.n_val_blends,
        config=config,
        seed=seed + 2000,
        settings=balanced_settings,
        split_name="val",
    )
    composition = {"train": train_composition, "validation": val_composition}
    save_json(run_dir / "logs" / "training_composition.json", composition)
    component_rows = []
    for split, payload in composition.items():
        for component in payload["components"]:
            component_rows.append({"split": split, **component})
    save_csv(run_dir / "tables" / "training_composition.csv", pd.DataFrame(component_rows))
    for component in component_rows:
        mean_mask = component.get("mean_mask_fraction")
        if mean_mask is not None and mean_mask < 0.005:
            stops.append(
                f"{component['split']} {component['component']}: suspiciously low mean mask fraction {mean_mask:.6f}"
            )
    if stops:
        write_diagnostics(run_dir, warnings, stops)
        print("Stopped before training due to suspicious blend generation.", flush=True)
        return 3

    train_ds = WeightedResidualBlendDataset(train_blends)
    val_ds = WeightedResidualBlendDataset(val_blends)
    requested_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["training"].get("batch_size", 8))
    )

    print(
        "Training weighted residual U-Net "
        f"(affected extra={loss_config.affected_extra_weight}, "
        f"core extra={loss_config.core_extra_weight}).",
        flush=True,
    )
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
            loss_config=loss_config,
        )
    except Exception as exc:
        stops.append(f"Training failed: {exc}")
        write_diagnostics(run_dir, warnings, stops)
        print(stops[-1], flush=True)
        return 3

    save_csv(run_dir / "tables" / "training_history.csv", history)
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
        loss_config,
        history,
        used_batch_size,
        stamp,
        "best_validation",
        args.variant_name,
    )
    save_checkpoint(
        final_checkpoint,
        final_state,
        config,
        settings,
        loss_config,
        history,
        used_batch_size,
        stamp,
        "final_epoch",
        args.variant_name,
    )
    weighted_model = make_residual_unet(config["model"])
    weighted_model.load_state_dict(best_state)
    weighted_model.to(device)
    weighted_model.eval()
    print(f"Saved best checkpoint: {project_relative(best_checkpoint)}", flush=True)
    print(f"Saved final checkpoint: {project_relative(final_checkpoint)}", flush=True)

    del train_blends, val_blends, train_ds, val_ds, train_images, val_images, final_model
    clear_torch_cache()
    gc.collect()

    print("Generating normal held-out test blends.", flush=True)
    normal_samples = balanced_helpers.normal_blends(
        test_images,
        settings.n_normal_test_blends,
        config,
        seed=seed + 4000,
        component="normal_test",
    )
    print("Evaluating normal held-out test set.", flush=True)
    normal_preds, normal_weighted_stats = predict_all(
        normal_samples,
        direct_model,
        residual_model,
        balanced_model,
        weighted_model,
        device,
        used_batch_size,
    )
    normal_per_sample = compute_per_sample(
        "normal",
        normal_samples,
        normal_preds,
        settings.affected_region_threshold,
    )
    normal_aggregate = aggregate_metrics(normal_per_sample)
    normal_grouped = grouped_metrics(normal_per_sample)
    save_csv(run_dir / "tables" / "normal_per_sample_results.csv", normal_per_sample)
    save_csv(run_dir / "tables" / "normal_results.csv", normal_aggregate)
    save_csv(run_dir / "tables" / "normal_grouped_results.csv", normal_grouped)
    save_json(
        run_dir / "diagnostics" / "normal_weighted_residual_output_stats.json",
        normal_weighted_stats,
    )
    split_warnings, split_stops = stop_condition_warnings(
        "normal",
        normal_aggregate,
        normal_weighted_stats,
    )
    warnings.extend(split_warnings)
    stops.extend(split_stops)

    print("Generating hard stress-test blends.", flush=True)
    stress_settings = dict(stress_helpers.STRESS_DEFAULTS)
    stress_settings["n_stress_blends"] = settings.n_stress_blends
    stress_samples = stress_helpers.generate_stress_blends(
        test_images[: int(stress_settings["stress_source_subset"])],
        stress_settings,
    )
    print("Evaluating hard stress-test set.", flush=True)
    stress_preds, stress_weighted_stats = predict_all(
        stress_samples,
        direct_model,
        residual_model,
        balanced_model,
        weighted_model,
        device,
        used_batch_size,
    )
    stress_per_sample = compute_per_sample(
        "stress",
        stress_samples,
        stress_preds,
        float(stress_settings["affected_region_threshold"]),
    )
    stress_aggregate = aggregate_metrics(stress_per_sample)
    stress_grouped = grouped_metrics(stress_per_sample)
    save_csv(run_dir / "tables" / "stress_per_sample_results.csv", stress_per_sample)
    save_csv(run_dir / "tables" / "stress_results.csv", stress_aggregate)
    save_csv(run_dir / "tables" / "stress_grouped_results.csv", stress_grouped)
    save_json(
        run_dir / "diagnostics" / "stress_weighted_residual_output_stats.json",
        stress_weighted_stats,
    )
    split_warnings, split_stops = stop_condition_warnings(
        "stress",
        stress_aggregate,
        stress_weighted_stats,
    )
    warnings.extend(split_warnings)
    stops.extend(split_stops)

    aggregate = pd.concat([normal_aggregate, stress_aggregate], ignore_index=True)
    per_sample = pd.concat([normal_per_sample, stress_per_sample], ignore_index=True)
    grouped = pd.concat([normal_grouped, stress_grouped], ignore_index=True)
    win_rates = model_win_rates(per_sample)
    save_csv(run_dir / "tables" / "model_comparison.csv", aggregate)
    save_csv(run_dir / "tables" / "weighted_model_win_rates.csv", win_rates)

    write_diagnostics(run_dir, warnings, stops, normal_per_sample, stress_per_sample)
    if stops:
        print("Stopped before figures/docs due to suspicious weighted results.", flush=True)
        checkpoint_comparison = verify_checkpoint_inventory(checkpoint_before)
        save_json(run_dir / "logs" / "protected_checkpoint_integrity.json", checkpoint_comparison)
        if not all(item["unchanged"] for item in checkpoint_comparison.values()):
            raise RuntimeError("Existing checkpoint metadata changed during the run.")
        return 3

    multiseed_summary: pd.DataFrame | None = None
    if settings.multiseed_count > 0:
        print("Running 3-seed mini multi-seed evaluation.", flush=True)
        _multiseed_results, multiseed_summary = run_multiseed(
            run_dir,
            config,
            test_images,
            direct_model,
            residual_model,
            balanced_model,
            weighted_model,
            device,
            settings,
            used_batch_size,
        )

    outcome = decide_outcome(aggregate, win_rates)
    save_json(run_dir / "logs" / "weighted_residual_outcome.json", outcome)

    print("Generating paper figures.", flush=True)
    figure_paths = save_all_figures(
        run_dir,
        aggregate,
        grouped,
        per_sample,
        normal_samples,
        stress_samples,
        normal_preds,
        stress_preds,
        settings.affected_region_threshold,
    )
    save_json(run_dir / "logs" / "paper_figures.json", figure_paths)

    if not args.skip_doc_update:
        append_experiment_log(
            run_dir,
            best_checkpoint,
            final_checkpoint,
            settings,
            loss_config,
            args.variant_name,
            composition,
            history,
            aggregate,
            win_rates,
            multiseed_summary,
            outcome,
            warnings,
            stops,
            figure_paths,
        )
        update_results_interpretation(outcome, run_dir)
        update_paper_plan(outcome, run_dir)

    checkpoint_comparison = verify_checkpoint_inventory(checkpoint_before)
    save_json(run_dir / "logs" / "protected_checkpoint_integrity.json", checkpoint_comparison)
    if not all(item["unchanged"] for item in checkpoint_comparison.values()):
        raise RuntimeError("Existing checkpoint metadata changed during the run.")

    print(f"Weighted residual run directory: {project_relative(run_dir)}", flush=True)
    print(f"Best checkpoint: {project_relative(best_checkpoint)}", flush=True)
    print(f"Final checkpoint: {project_relative(final_checkpoint)}", flush=True)
    print("Normal metrics:", flush=True)
    print(format_metrics_table(normal_aggregate), flush=True)
    print("Stress metrics:", flush=True)
    print(format_metrics_table(stress_aggregate), flush=True)
    print(f"Outcome: {outcome['verdict']} - {outcome['recommendation']}", flush=True)
    if warnings:
        print("Warnings:", flush=True)
        for warning in warnings:
            print(f"- {warning}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
