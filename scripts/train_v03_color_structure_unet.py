"""Train Thayer-BR v0.3 Color/Structure Candidate.

This experiment keeps the v0.2 residual U-Net architecture and adds low-weight
color, edge, reconstruction, affected/core, and halo-band terms. It writes only
timestamped artifacts under ``outputs/`` and refuses to overwrite checkpoints.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage import color
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
import train_weighted_residual_unet as weighted_helpers
from src import baselines
from src import blend as gd_blend
from src import data as gd_data
from src import train as gd_train
from src import utils as gd_utils


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_DIRECT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/checkpoints/unet_direct_5000train_800val_800test_20ep.pth"
)
DEFAULT_RESIDUAL_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_residual_5000train_800val_800test_20ep_20260708_154947.pth"
)
DEFAULT_BR_V01_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_residual_balanced_hard_20260708_184632.pth"
)
DEFAULT_BR_V02_MODERATE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_residual_weighted_br_20260709_030245_best.pth"
)
DEFAULT_BR_V02_STRONG_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_residual_weighted_br_20260709_043745_best.pth"
)

METHODS = [
    "identity",
    "threshold",
    "direct",
    "residual",
    "br_v01",
    "br_v02_moderate",
    "br_v02_strong",
    "br_v03_color",
]
V03_METHOD_LABEL = "BR v0.3 Color"


@dataclass(frozen=True)
class V03Settings:
    n_train_blends: int = 8000
    n_val_blends: int = 1000
    n_normal_test_blends: int = 1000
    n_suite_blends: int = 1000
    train_source_subset: int = 5000
    val_source_subset: int = 1000
    test_source_subset: int = 1000
    num_epochs: int = 20
    affected_region_threshold: float = 0.02
    max_epochs_if_improving: int = 25
    seed_offset: int = 503


@dataclass(frozen=True)
class V03LossConfig:
    residual_mse_weight: float = 1.0
    reconstruction_l1_weight: float = 0.50
    affected_core_loss_weight: float = 1.0
    gradient_loss_weight: float = 0.10
    color_loss_weight: float = 0.05
    halo_band_loss_weight: float = 0.05
    background_weight: float = 1.0
    affected_weight: float = 3.0
    core_weight: float = 2.0
    affected_threshold: float = 0.02
    core_aperture_fraction: float = 0.18
    core_brightness_fraction: float = 0.55
    halo_dilation_iters: int = 5
    eps: float = 1e-8


class V03ResidualBlendDataset(Dataset):
    """Dataset returning blended image, true residual, and clean target."""

    def __init__(self, blends: list[dict[str, Any]]) -> None:
        if not blends:
            raise ValueError("V03ResidualBlendDataset requires at least one blend.")
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
        description="Train Thayer-BR v0.3 Color/Structure Candidate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-name-prefix", default="br_v03_delta_color")
    parser.add_argument("--checkpoint-name-prefix", default="unet_br_v03_delta_color")
    parser.add_argument(
        "--experiment-title",
        default="Thayer-BR v0.3 Color/Structure Candidate",
    )
    parser.add_argument("--v03-method-label", default="BR v0.3 Color")
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="Prefix for combined tables/logs (derived from the run prefix when omitted).",
    )
    parser.add_argument(
        "--diagnostic-report-name",
        default=None,
        help="Diagnostic Markdown filename (derived from the run prefix when omitted).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--direct-checkpoint", type=Path, default=DEFAULT_DIRECT_CHECKPOINT)
    parser.add_argument(
        "--residual-checkpoint", type=Path, default=DEFAULT_RESIDUAL_CHECKPOINT
    )
    parser.add_argument("--br-v01-checkpoint", type=Path, default=DEFAULT_BR_V01_CHECKPOINT)
    parser.add_argument(
        "--br-v02-moderate-checkpoint",
        type=Path,
        default=DEFAULT_BR_V02_MODERATE_CHECKPOINT,
    )
    parser.add_argument(
        "--br-v02-strong-checkpoint",
        type=Path,
        default=DEFAULT_BR_V02_STRONG_CHECKPOINT,
    )
    parser.add_argument("--n-train-blends", type=int, default=8000)
    parser.add_argument("--n-val-blends", type=int, default=1000)
    parser.add_argument("--n-normal-test-blends", type=int, default=1000)
    parser.add_argument("--n-suite-blends", type=int, default=1000)
    parser.add_argument("--train-source-subset", type=int, default=5000)
    parser.add_argument("--val-source-subset", type=int, default=1000)
    parser.add_argument("--test-source-subset", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--color-loss-weight", type=float, default=0.05)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.10)
    parser.add_argument("--reconstruction-l1-weight", type=float, default=0.50)
    parser.add_argument("--affected-core-loss-weight", type=float, default=1.0)
    parser.add_argument("--halo-band-loss-weight", type=float, default=0.05)
    parser.add_argument("--affected-weight", type=float, default=3.0)
    parser.add_argument("--core-weight", type=float, default=2.0)
    parser.add_argument("--affected-threshold", type=float, default=0.02)
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}.")
    return config


def safe_write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def safe_yaml(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
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


def make_run_paths(
    output_root: Path,
    stamp: str,
    run_dir_arg: Path | None,
    run_name_prefix: str,
    checkpoint_name_prefix: str,
) -> tuple[Path, Path, Path]:
    output_root = output_root.resolve()
    runs_root = (output_root / "runs").resolve()
    if not stamp.replace("_", "").isdigit():
        raise ValueError("stamp must contain only digits and underscores")
    for label, value in (
        ("run_name_prefix", run_name_prefix),
        ("checkpoint_name_prefix", checkpoint_name_prefix),
    ):
        if not value or any(not (char.isalnum() or char in "_-") for char in value):
            raise ValueError(f"Unsafe {label}: {value!r}")
    run_dir = (
        resolve_path(run_dir_arg)
        if run_dir_arg is not None
        else runs_root / f"{run_name_prefix}_{stamp}"
    )
    if runs_root not in run_dir.parents:
        raise ValueError(f"Run directory must be a child of {runs_root}: {run_dir}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in (
        "tables",
        "figures",
        "paper_figures",
        "example_grids",
        "diagnostics",
        "logs",
    ):
        (run_dir / child).mkdir(exist_ok=False)
    best_checkpoint = (
        output_root / "checkpoints" / f"{checkpoint_name_prefix}_{stamp}_best.pth"
    )
    final_checkpoint = (
        output_root / "checkpoints" / f"{checkpoint_name_prefix}_{stamp}_final.pth"
    )
    if best_checkpoint.exists():
        raise FileExistsError(f"Checkpoint exists: {best_checkpoint}")
    if final_checkpoint.exists():
        raise FileExistsError(f"Checkpoint exists: {final_checkpoint}")
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, best_checkpoint, final_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_snapshot(checkpoint_dir: Path) -> dict[str, Any]:
    rows = []
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.pth")):
            stat = path.stat()
            rows.append(
                {
                    "path": project_relative(path),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "mtime_iso_local": datetime.fromtimestamp(stat.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                    "sha256": sha256_file(path),
                }
            )
    return {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": project_relative(checkpoint_dir),
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }


def save_after_checkpoint_integrity(run_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    before_path = run_dir / "logs" / "checkpoint_integrity_before.json"
    after = checkpoint_snapshot(checkpoint_dir)
    safe_json(run_dir / "logs" / "checkpoint_integrity_after.json", after)
    if not before_path.exists():
        comparison = {
            "status": "missing_before_snapshot",
            "old_checkpoints_unchanged": False,
        }
        safe_json(run_dir / "logs" / "checkpoint_integrity_comparison.json", comparison)
        return comparison

    before = json.loads(before_path.read_text(encoding="utf-8"))
    before_rows = {
        row["path"]: row for row in before.get("checkpoints", [])
    }
    after_rows = {row["path"]: row for row in after.get("checkpoints", [])}
    comparisons = []
    for rel_path, before_row in before_rows.items():
        after_row = after_rows.get(rel_path)
        unchanged = (
            after_row is not None
            and before_row["size_bytes"] == after_row["size_bytes"]
            and before_row["mtime_ns"] == after_row["mtime_ns"]
            and before_row.get("sha256") == after_row.get("sha256")
        )
        comparisons.append(
            {
                "path": rel_path,
                "before": before_row,
                "after": after_row,
                "unchanged": unchanged,
            }
        )
    comparison = {
        "status": "compared",
        "old_checkpoints_unchanged": all(row["unchanged"] for row in comparisons),
        "old_checkpoint_count": len(comparisons),
        "comparisons": comparisons,
    }
    safe_json(run_dir / "logs" / "checkpoint_integrity_comparison.json", comparison)
    return comparison


def write_color_metric_report(run_dir: Path) -> None:
    text = """# Color Metric Implementation

Implemented evaluation metrics:

- RGB affected-region MSE and MAE are reported for affected, core-affected, non-core affected, and halo-band regions.
- Lab-space affected-region L1 and L2 errors are computed with `skimage.color.rgb2lab`.
- Delta E 2000 affected-region mean and median are computed with `skimage.color.deltaE_ciede2000`.
- Delta E 76 mean and median are also reported as the Lab Euclidean-distance fallback/check.
- Lab chroma error and RGB saturation-proxy error are reported in affected and core-affected regions.
- Gradient/edge error is reported with finite-difference grayscale gradient magnitude.

CIEDE2000 status: available and used through scikit-image.

Color transform assumption: `skimage.color.rgb2lab` treats clipped input arrays as standard RGB/sRGB-like values in `[0, 1]`.

Caveat: Galaxy10 DECaLS RGB cutouts are survey composite images. They are not guaranteed true human-color photographs, so Delta E 2000 is visual-quality evidence rather than the primary astronomical/scientific metric. Affected-region MSE remains the primary metric.

Training loss note: exact CIEDE2000 is not used as a training loss. The v0.3 training objective uses differentiable RGB chroma and color-direction proxies at low weight, while CIEDE2000 is evaluation-only.
"""
    safe_write_text(run_dir / "diagnostics" / "color_metric_implementation.md", text)


def target_core_mask_np(
    target: np.ndarray,
    aperture_fraction: float = 0.18,
    core_percentile: float = 85.0,
) -> np.ndarray:
    return balanced_helpers.target_core_mask(target, aperture_fraction, core_percentile)


def halo_band_mask(affected: np.ndarray, dilation_iters: int = 5) -> np.ndarray:
    if not np.any(affected):
        return np.zeros_like(affected, dtype=bool)
    dilated = binary_dilation(affected, iterations=dilation_iters)
    return np.logical_and(dilated, ~affected)


def target_core_mask_torch(target: torch.Tensor, loss_config: V03LossConfig) -> torch.Tensor:
    return weighted_helpers.target_core_mask_torch(target, loss_config)


def spatial_masks_torch(
    true_residual: torch.Tensor,
    target: torch.Tensor,
    loss_config: V03LossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    affected = (
        torch.mean(torch.abs(true_residual), dim=1, keepdim=True)
        > loss_config.affected_threshold
    )
    core = target_core_mask_torch(target, loss_config)
    core_affected = affected & core
    weight = torch.full_like(true_residual[:, :1], loss_config.background_weight)
    weight = weight + affected.float() * loss_config.affected_weight
    weight = weight + core_affected.float() * loss_config.core_weight
    return affected, core, core_affected, weight


def weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if value.dim() == 4 and value.shape[1] != weight.shape[1]:
        normalizer = weight.sum() * value.shape[1] + eps
    else:
        normalizer = weight.sum() + eps
    return (value * weight).sum() / normalizer


def gradient_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    pred_dx = reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]
    targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]
    targ_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    weight_dx = 0.5 * (weight[:, :, :, 1:] + weight[:, :, :, :-1])
    weight_dy = 0.5 * (weight[:, :, 1:, :] + weight[:, :, :-1, :])
    return 0.5 * (
        weighted_mean(torch.abs(pred_dx - targ_dx), weight_dx, eps)
        + weighted_mean(torch.abs(pred_dy - targ_dy), weight_dy, eps)
    )


def color_proxy_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    recon = torch.clamp(reconstruction, 0.0, 1.0)
    targ = torch.clamp(target, 0.0, 1.0)
    recon_gray = recon.mean(dim=1, keepdim=True)
    targ_gray = targ.mean(dim=1, keepdim=True)
    recon_chroma = torch.sqrt(torch.sum((recon - recon_gray) ** 2, dim=1, keepdim=True) + eps)
    targ_chroma = torch.sqrt(torch.sum((targ - targ_gray) ** 2, dim=1, keepdim=True) + eps)
    chroma = weighted_mean(torch.abs(recon_chroma - targ_chroma), weight, eps)

    recon_dir = recon / (torch.linalg.vector_norm(recon, dim=1, keepdim=True) + eps)
    targ_dir = targ / (torch.linalg.vector_norm(targ, dim=1, keepdim=True) + eps)
    direction = weighted_mean(torch.abs(recon_dir - targ_dir), weight, eps)
    return chroma + 0.25 * direction


def halo_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    affected: torch.Tensor,
    loss_config: V03LossConfig,
) -> torch.Tensor:
    radius = int(loss_config.halo_dilation_iters)
    if radius <= 0:
        return reconstruction.new_tensor(0.0)
    dilated = F.max_pool2d(
        affected.float(),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ) > 0
    ring = (dilated & ~affected).float()
    if float(ring.sum().detach().cpu()) <= 0.0:
        return reconstruction.new_tensor(0.0)
    return weighted_mean((reconstruction - target) ** 2, ring, loss_config.eps)


def v03_loss(
    blended: torch.Tensor,
    predicted_residual: torch.Tensor,
    true_residual: torch.Tensor,
    target: torch.Tensor,
    loss_config: V03LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    affected, _core, core_affected, weight = spatial_masks_torch(
        true_residual, target, loss_config
    )
    reconstruction = blended - predicted_residual
    residual_mse = torch.mean((predicted_residual - true_residual) ** 2)
    recon_l1 = torch.mean(torch.abs(reconstruction - target))
    affected_core = weighted_mean((reconstruction - target) ** 2, weight, loss_config.eps)
    grad = gradient_loss(reconstruction, target, weight, loss_config.eps)
    col = color_proxy_loss(reconstruction, target, weight, loss_config.eps)
    halo = halo_loss(reconstruction, target, affected, loss_config)
    total = (
        loss_config.residual_mse_weight * residual_mse
        + loss_config.reconstruction_l1_weight * recon_l1
        + loss_config.affected_core_loss_weight * affected_core
        + loss_config.gradient_loss_weight * grad
        + loss_config.color_loss_weight * col
        + loss_config.halo_band_loss_weight * halo
    )
    preclip = reconstruction.detach()
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_residual_mse": float(residual_mse.detach().cpu()),
        "loss_reconstruction_l1": float(recon_l1.detach().cpu()),
        "loss_affected_core": float(affected_core.detach().cpu()),
        "loss_gradient": float(grad.detach().cpu()),
        "loss_color_proxy": float(col.detach().cpu()),
        "loss_halo_band": float(halo.detach().cpu()),
        "affected_fraction": float(affected.float().mean().detach().cpu()),
        "core_affected_fraction": float(core_affected.float().mean().detach().cpu()),
        "mean_weight": float(weight.mean().detach().cpu()),
        "preclip_low_fraction": float((preclip < 0.0).float().mean().cpu()),
        "preclip_high_fraction": float((preclip > 1.0).float().mean().cpu()),
    }
    return total, stats


def average_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_config: V03LossConfig,
) -> dict[str, float]:
    model.eval()
    n_samples = 0
    weighted_stats: list[dict[str, float]] = []
    val_loss_sum = 0.0
    affected_mse_values: list[float] = []
    clip_count = 0.0
    pixel_count = 0.0

    with torch.no_grad():
        for blended, true_residual, target in val_loader:
            blended = blended.to(device)
            true_residual = true_residual.to(device)
            target = target.to(device)
            predicted = model(blended)
            loss, stats = v03_loss(blended, predicted, true_residual, target, loss_config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")
            batch_size = blended.size(0)
            n_samples += batch_size
            val_loss_sum += float(loss.item()) * batch_size
            weighted_stats.append(stats)
            reconstruction_preclip = blended - predicted
            reconstruction = torch.clamp(reconstruction_preclip, 0.0, 1.0)
            clip_count += float(
                torch.count_nonzero(
                    (reconstruction_preclip < 0.0) | (reconstruction_preclip > 1.0)
                ).item()
            )
            pixel_count += float(reconstruction_preclip.numel())
            affected = torch.mean(torch.abs(true_residual), dim=1) > loss_config.affected_threshold
            sq = torch.mean((reconstruction - target) ** 2, dim=1)
            for idx in range(batch_size):
                if bool(affected[idx].any().detach().cpu()):
                    affected_mse_values.append(float(sq[idx][affected[idx]].mean().item()))

    diag = average_stats(weighted_stats)
    diag.update(
        {
            "val_loss": val_loss_sum / max(n_samples, 1),
            "val_affected_mse": float(np.nanmean(affected_mse_values))
            if affected_mse_values
            else float("nan"),
            "val_reconstruction_clip_fraction": clip_count / max(pixel_count, 1.0),
        }
    )
    return diag


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
    loss_config: V03LossConfig,
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
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    recent_val: list[float] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_stats: list[dict[str, float]] = []
        for blended, true_residual, target in train_loader:
            blended = blended.to(device)
            true_residual = true_residual.to(device)
            target = target.to(device)
            optimiser.zero_grad()
            predicted = model(blended)
            loss, stats = v03_loss(blended, predicted, true_residual, target, loss_config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimiser.step()
            train_loss_sum += float(loss.item()) * blended.size(0)
            train_stats.append(stats)

        train_loss = train_loss_sum / len(train_ds)
        val_diag = evaluate_validation(model, val_loader, device, loss_config)
        val_loss = val_diag["val_loss"]
        if train_loss > 2.0 or val_loss > 2.0:
            raise RuntimeError(
                f"Training loss exploded at epoch {epoch}: "
                f"train={train_loss:.6f}, val={val_loss:.6f}."
            )
        recent_val.append(val_loss)
        if len(recent_val) >= 4 and val_loss > min(recent_val[:-1]) * 10.0:
            raise RuntimeError(
                f"Validation loss exploded at epoch {epoch}: "
                f"val={val_loss:.6f}, previous_best={min(recent_val[:-1]):.6f}."
            )
        if val_diag["val_reconstruction_clip_fraction"] > 0.20:
            raise RuntimeError(
                f"Extreme clipping at epoch {epoch}: "
                f"{val_diag['val_reconstruction_clip_fraction']:.3%}."
            )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        train_diag = average_stats(train_stats)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            **{f"train_{key}": value for key, value in train_diag.items()},
            **val_diag,
        }
        rows.append(row)
        print(
            f"Epoch {epoch}/{num_epochs}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, val affected MSE={val_diag['val_affected_mse']:.6f}, "
            f"color={train_diag.get('loss_color_proxy', float('nan')):.6f}, "
            f"grad={train_diag.get('loss_gradient', float('nan')):.6f}, "
            f"best={best_val:.6f} @ {best_epoch}",
            flush=True,
        )

    return model, best_state, pd.DataFrame(rows)


def train_with_memory_retry(
    train_ds: Dataset,
    val_ds: Dataset,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    settings: V03Settings,
    requested_batch_size: int,
    device: torch.device,
    seed: int,
    loss_config: V03LossConfig,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame, int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    batch_size = requested_batch_size
    while batch_size >= 1:
        seed_everything(seed)
        model = weighted_helpers.make_residual_unet(model_config)
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
    settings: V03Settings,
    loss_config: V03LossConfig,
    history: pd.DataFrame,
    batch_size: int,
    stamp: str,
    kind: str,
    experiment_title: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    payload = {
        "model_state_dict": model_state,
        "experiment_name": experiment_title,
        "experiment_settings": asdict(settings),
        "loss_config": asdict(loss_config),
        "config": config,
        "checkpoint_kind": kind,
        "timestamp": stamp,
        "architecture": "unchanged compact residual U-Net from v0.2",
        "residual_target": "blended_minus_target",
        "reconstruction": "blended_minus_predicted_residual",
        "output_activation": "identity",
        "color_loss": "differentiable RGB chroma plus color-direction proxy",
        "delta_e2000": "evaluation_only_skimage.color.deltaE_ciede2000",
        "batch_size": batch_size,
        "final_train_loss": float(history["train_loss"].iloc[-1]),
        "final_val_loss": float(history["val_loss"].iloc[-1]),
        "best_epoch": int(history["best_epoch"].iloc[-1]),
        "best_val_loss": float(history["best_val_loss"].iloc[-1]),
    }
    torch.save(payload, path)


def strip_contaminants(blends: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for sample in blends:
        sample.pop("contaminant", None)
    return blends


def blend_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return balanced_helpers.blend_params_from_config(config)


def normal_blends(
    images: np.ndarray,
    n_blends: int,
    config: dict[str, Any],
    seed: int,
    component: str,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    params = blend_params_from_config(config)
    if overrides:
        params.update(overrides)
    blends = gd_blend.generate_blends(images, n_blends=n_blends, rng=rng, **params)
    for sample in blends:
        sample["info"]["training_component"] = component
    return strip_contaminants(blends)


def blend_metadata(
    target: np.ndarray,
    blended: np.ndarray,
    info: dict[str, Any],
    threshold: float,
) -> dict[str, float]:
    return balanced_helpers.blend_metadata(target, blended, info, threshold)


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
    min_size_ratio: float = 0.0,
    max_size_ratio: float | None = None,
    min_mask_fraction: float = 0.01,
    min_core_obstruction: float | None = None,
    min_shift_distance: int | None = None,
    color_tint: bool = False,
    max_attempt_multiplier: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if n_blends <= 0:
        return [], {"component": component, "requested": 0, "accepted": 0}
    rng = np.random.default_rng(seed)
    accepted: list[dict[str, Any]] = []
    relaxed: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max_attempt_multiplier * n_blends
    while attempts < max_attempts and len(accepted) < n_blends:
        attempts += 1
        target_idx, contaminant_idx = rng.choice(images.shape[0], size=2, replace=False)
        target = images[target_idx]
        contaminant = images[contaminant_idx]
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        if min_shift_distance is not None and abs(dx) + abs(dy) < min_shift_distance:
            continue
        brightness = float(rng.uniform(*brightness_range))
        blur_sigma = float(rng.uniform(*blur_range))
        noise_std = float(rng.uniform(*noise_range))
        if color_tint:
            blended, info = tinted_blend_pair(
                target,
                contaminant,
                shift=(dx, dy),
                brightness=brightness,
                blur_sigma=blur_sigma,
                noise_std=noise_std,
                rng=rng,
            )
        else:
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
            "attempt": attempts,
            "training_component": component,
        }
        sample = {"target": target, "blended": blended, "info": info}
        size_ratio = metadata["size_ratio"]
        finite_size = np.isfinite(size_ratio)
        size_ok = finite_size and size_ratio >= min_size_ratio
        if max_size_ratio is not None:
            size_ok = size_ok and size_ratio <= max_size_ratio
        mask_ok = metadata["mask_fraction"] >= min_mask_fraction
        core_ok = (
            True
            if min_core_obstruction is None
            else metadata["core_obstruction_fraction"] >= min_core_obstruction
        )
        constraints_met = bool(mask_ok and size_ok and core_ok)
        sample["info"]["generation_constraints_met"] = constraints_met
        sample["info"]["generation_relaxed"] = False
        if constraints_met:
            accepted.append(sample)
        elif mask_ok and finite_size:
            relaxed.append(sample)

    if len(accepted) < n_blends:
        needed = n_blends - len(accepted)
        if "compact" in component:
            relaxed.sort(
                key=lambda row: (
                    -abs(row["info"].get("size_ratio", 1.0) - 0.55),
                    row["info"].get("brightness", 0.0),
                    row["info"].get("mask_fraction", 0.0),
                ),
                reverse=True,
            )
        else:
            relaxed.sort(
                key=lambda row: (
                    row["info"].get("core_obstruction_fraction", 0.0),
                    row["info"].get("mask_fraction", 0.0),
                    row["info"].get("brightness", 0.0),
                ),
                reverse=True,
            )
        relaxed_selected = relaxed[:needed]
        for sample in relaxed_selected:
            sample["info"]["generation_relaxed"] = True
        accepted.extend(relaxed_selected)
    if len(accepted) < n_blends:
        raise RuntimeError(
            f"Could not generate enough {component} blends. "
            f"Requested {n_blends}, accepted {len(accepted)} after {attempts} attempts."
        )
    selected = accepted[:n_blends]
    diag = {
        "component": component,
        "requested": n_blends,
        "accepted": len(selected),
        "attempts": attempts,
        "relaxed_candidates_used": int(
            sum(bool(s["info"].get("generation_relaxed", False)) for s in selected)
        ),
        "mean_mask_fraction": float(np.mean([s["info"]["mask_fraction"] for s in selected])),
        "mean_core_obstruction_fraction": float(
            np.mean([s["info"]["core_obstruction_fraction"] for s in selected])
        ),
        "mean_size_ratio": float(np.mean([s["info"]["size_ratio"] for s in selected])),
        "mean_brightness": float(np.mean([s["info"]["brightness"] for s in selected])),
        "artifact_flags_used": False,
    }
    return selected, diag


def tinted_blend_pair(
    target: np.ndarray,
    contaminant: np.ndarray,
    shift: tuple[int, int],
    brightness: float,
    blur_sigma: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(target, dtype=np.float32)
    contaminant = np.asarray(contaminant, dtype=np.float32)
    target_blurred = (
        gaussian_filter(target, sigma=(blur_sigma, blur_sigma, 0))
        if blur_sigma > 0
        else target.copy()
    )
    target_blurred = np.clip(target_blurred, 0.0, 1.0)
    _, target_size = gd_blend.extract_source_foreground(target)
    contaminant_foreground, contaminant_size = gd_blend.extract_source_foreground(contaminant)
    tints = np.asarray(
        [
            [1.35, 0.82, 0.82],
            [0.82, 1.25, 0.90],
            [0.85, 0.95, 1.35],
            [1.25, 1.05, 0.75],
        ],
        dtype=np.float32,
    )
    tint = tints[int(rng.integers(0, len(tints)))].reshape(1, 1, 3)
    contaminant_foreground = np.clip(contaminant_foreground * brightness * tint, 0.0, 1.0)
    shifted = gd_blend.shift_foreground(
        contaminant_foreground,
        dx=int(shift[0]),
        dy=int(shift[1]),
    )
    blended = np.clip(target_blurred + shifted, 0.0, 1.0)
    if noise_std > 0.0:
        blended = np.clip(blended + rng.normal(scale=noise_std, size=blended.shape), 0.0, 1.0)
    target_radius = target_size["radius"]
    contaminant_radius = contaminant_size["radius"]
    size_ratio = contaminant_radius / target_radius if target_radius > 0 else float("nan")
    return blended.astype(np.float32), {
        "shift": (int(shift[0]), int(shift[1])),
        "rotation": 0.0,
        "brightness": float(brightness),
        "blur_sigma": float(blur_sigma),
        "noise_std": float(noise_std),
        "generation_difficulty": "hard",
        "difficulty": "hard",
        "target_area": float(target_size["area"]),
        "target_radius": float(target_radius),
        "contaminant_area": float(contaminant_size["area"]),
        "contaminant_radius": float(contaminant_radius),
        "size_ratio": float(size_ratio),
        "color_tint": tint.reshape(3).astype(float).tolist(),
    }


def component_counts(total: int) -> dict[str, int]:
    normal_clean = int(round(total * 0.40))
    high_overlap = int(round(total * 0.25))
    compact = int(round(total * 0.20))
    brightness_size = int(round(total * 0.10))
    low_overlap = total - normal_clean - high_overlap - compact - brightness_size
    return {
        "normal_clean": normal_clean,
        "high_overlap_core": high_overlap,
        "compact_bright": compact,
        "brightness_size": brightness_size,
        "low_overlap_easy": low_overlap,
    }


def generate_v03_training_blends(
    images: np.ndarray,
    total: int,
    config: dict[str, Any],
    seed: int,
    threshold: float,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = component_counts(total)
    blends: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    normal = normal_blends(
        images,
        counts["normal_clean"],
        config,
        seed + 11,
        f"{split_name}_normal_clean",
    )
    blends.extend(normal)
    diagnostics.append(
        {
            "component": f"{split_name}_normal_clean",
            "requested": len(normal),
            "accepted": len(normal),
            "artifact_flags_used": False,
        }
    )
    specs = [
        (
            "high_overlap_core",
            counts["high_overlap_core"],
            {
                "max_shift": 16,
                "brightness_range": (0.85, 1.40),
                "blur_range": (0.0, 0.15),
                "noise_range": (0.0, 0.006),
                "min_size_ratio": 0.70,
                "min_mask_fraction": 0.01,
                "min_core_obstruction": 0.70,
            },
        ),
        (
            "compact_bright",
            counts["compact_bright"],
            {
                "max_shift": 24,
                "brightness_range": (1.30, 1.90),
                "blur_range": (0.0, 0.10),
                "noise_range": (0.0, 0.006),
                "min_size_ratio": 0.15,
                "max_size_ratio": 0.90,
                "min_mask_fraction": 0.004,
                "min_core_obstruction": None,
            },
        ),
        (
            "brightness_size",
            counts["brightness_size"],
            {
                "max_shift": 36,
                "brightness_range": (1.05, 1.55),
                "blur_range": (0.0, 0.15),
                "noise_range": (0.0, 0.006),
                "min_size_ratio": 0.75,
                "min_mask_fraction": 0.01,
                "min_core_obstruction": None,
            },
        ),
    ]
    for offset, (name, count, kwargs) in enumerate(specs, start=1):
        generated, diag = generate_targeted_blends(
            images=images,
            n_blends=count,
            seed=seed + 101 * offset,
            component=f"{split_name}_{name}",
            threshold=threshold,
            **kwargs,
        )
        blends.extend(generated)
        diagnostics.append(diag)
    easy = normal_blends(
        images,
        counts["low_overlap_easy"],
        config,
        seed + 777,
        f"{split_name}_low_overlap_easy",
        overrides={
            "max_shift": 56,
            "brightness_range": (0.45, 0.95),
            "blur_range": (0.0, 0.08),
            "noise_range": (0.0, 0.004),
        },
    )
    blends.extend(easy)
    diagnostics.append(
        {
            "component": f"{split_name}_low_overlap_easy",
            "requested": len(easy),
            "accepted": len(easy),
            "artifact_flags_used": False,
        }
    )
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(blends)
    return blends, {
        "split": split_name,
        "total": len(blends),
        "target_distribution": {
            "normal_clean": 0.40,
            "high_overlap_core": 0.25,
            "compact_bright": 0.20,
            "brightness_size": 0.10,
            "low_overlap_easy": 0.05,
            "artifact_outlier_controlled": 0.0,
            "artifact_note": "No source-quality artifact flags were available; the unsafe artifact bucket was excluded and redistributed to normal clean blends.",
        },
        "counts": counts,
        "components": diagnostics,
    }


def load_split_subsets(config: dict[str, Any], settings: V03Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    train_images = gd_data.normalise_images(train[0][: settings.train_source_subset])
    val_images = gd_data.normalise_images(val[0][: settings.val_source_subset])
    test_images = gd_data.normalise_images(test[0][: settings.test_source_subset])
    del images_raw, labels, train, val, test
    gc.collect()
    return train_images, val_images, test_images


def metric_pair(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    diff = pred[mask] - target[mask]
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def rgb_saturation_proxy(rgb_pixels: np.ndarray) -> np.ndarray:
    maxc = np.max(rgb_pixels, axis=-1)
    minc = np.min(rgb_pixels, axis=-1)
    return (maxc - minc) / (maxc + 1e-8)


def masked_color_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if not np.any(mask):
        return {
            f"{prefix}_lab_l1": float("nan"),
            f"{prefix}_lab_l2": float("nan"),
            f"{prefix}_delta_e76_mean": float("nan"),
            f"{prefix}_delta_e76_median": float("nan"),
            f"{prefix}_delta_e2000_mean": float("nan"),
            f"{prefix}_delta_e2000_median": float("nan"),
            f"{prefix}_lab_chroma_mae": float("nan"),
            f"{prefix}_rgb_saturation_mae": float("nan"),
        }
    pred_pixels = np.clip(pred[mask], 0.0, 1.0).reshape(-1, 1, 3)
    target_pixels = np.clip(target[mask], 0.0, 1.0).reshape(-1, 1, 3)
    pred_lab = color.rgb2lab(pred_pixels).reshape(-1, 3)
    target_lab = color.rgb2lab(target_pixels).reshape(-1, 3)
    lab_diff = pred_lab - target_lab
    delta_e76 = np.sqrt(np.sum(lab_diff**2, axis=1))
    delta_e2000 = color.deltaE_ciede2000(pred_lab, target_lab)
    pred_chroma = np.sqrt(pred_lab[:, 1] ** 2 + pred_lab[:, 2] ** 2)
    target_chroma = np.sqrt(target_lab[:, 1] ** 2 + target_lab[:, 2] ** 2)
    pred_sat = rgb_saturation_proxy(pred_pixels.reshape(-1, 3))
    target_sat = rgb_saturation_proxy(target_pixels.reshape(-1, 3))
    return {
        f"{prefix}_lab_l1": float(np.mean(np.abs(lab_diff))),
        f"{prefix}_lab_l2": float(np.mean(delta_e76)),
        f"{prefix}_delta_e76_mean": float(np.mean(delta_e76)),
        f"{prefix}_delta_e76_median": float(np.median(delta_e76)),
        f"{prefix}_delta_e2000_mean": float(np.mean(delta_e2000)),
        f"{prefix}_delta_e2000_median": float(np.median(delta_e2000)),
        f"{prefix}_lab_chroma_mae": float(np.mean(np.abs(pred_chroma - target_chroma))),
        f"{prefix}_rgb_saturation_mae": float(np.mean(np.abs(pred_sat - target_sat))),
    }


def gradient_error(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    pred_gray = np.mean(pred, axis=-1)
    target_gray = np.mean(target, axis=-1)
    pred_gy, pred_gx = np.gradient(pred_gray)
    target_gy, target_gx = np.gradient(target_gray)
    pred_mag = np.sqrt(pred_gx**2 + pred_gy**2)
    target_mag = np.sqrt(target_gx**2 + target_gy**2)
    return float(np.mean(np.abs(pred_mag[mask] - target_mag[mask])))


def image_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    affected: np.ndarray,
    core_affected: np.ndarray,
    noncore_affected: np.ndarray,
    halo: np.ndarray,
) -> dict[str, float]:
    whole = gd_utils.compute_metrics(pred, target, metrics=("mse", "mae", "psnr", "ssim"))
    affected_mse, affected_mae = metric_pair(pred, target, affected)
    core_mse, core_mae = metric_pair(pred, target, core_affected)
    noncore_mse, noncore_mae = metric_pair(pred, target, noncore_affected)
    halo_mse, halo_mae = metric_pair(pred, target, halo)
    values = {
        "whole_mse": whole["mse"],
        "whole_mae": whole["mae"],
        "psnr": whole["psnr"],
        "ssim": whole["ssim"],
        "affected_mse": affected_mse,
        "affected_mae": affected_mae,
        "core_affected_mse": core_mse,
        "core_affected_mae": core_mae,
        "noncore_affected_mse": noncore_mse,
        "noncore_affected_mae": noncore_mae,
        "halo_band_mse": halo_mse,
        "halo_band_mae": halo_mae,
        "affected_gradient_error": gradient_error(pred, target, affected),
        "core_affected_gradient_error": gradient_error(pred, target, core_affected),
        "halo_band_gradient_error": gradient_error(pred, target, halo),
    }
    values.update(masked_color_metrics(pred, target, affected, "affected"))
    values.update(masked_color_metrics(pred, target, core_affected, "core_affected"))
    return values


def method_label(method: str) -> str:
    return {
        "identity": "identity",
        "threshold": "threshold",
        "direct": "Thayer-Direct",
        "residual": "Thayer-Residual",
        "br_v01": "Thayer-BR v0.1",
        "br_v02_moderate": "BR v0.2 Moderate",
        "br_v02_strong": "BR v0.2 Strong",
        "br_v03_color": V03_METHOD_LABEL,
    }.get(method, method)


def method_color(method: str) -> str:
    return {
        "identity": "#6c717a",
        "threshold": "#b66b5d",
        "direct": "#2f6f8f",
        "residual": "#5f8a4b",
        "br_v01": "#7b5fa3",
        "br_v02_moderate": "#c28b2c",
        "br_v02_strong": "#9c6b1e",
        "br_v03_color": "#2c8f7b",
    }.get(method, "#444444")


def load_models(
    config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    v03_state: dict[str, torch.Tensor],
) -> dict[str, nn.Module | None]:
    direct_path = resolve_path(args.direct_checkpoint)
    residual_path = resolve_path(args.residual_checkpoint)
    br_v01_path = resolve_path(args.br_v01_checkpoint)
    br_v02_path = resolve_path(args.br_v02_moderate_checkpoint)
    br_v02_strong_path = resolve_path(args.br_v02_strong_checkpoint)
    required = {
        "direct": direct_path,
        "residual": residual_path,
        "br_v01": br_v01_path,
        "br_v02_moderate": br_v02_path,
    }
    missing = [f"{key}: {project_relative(path)}" for key, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison checkpoints: " + "; ".join(missing))
    models: dict[str, nn.Module | None] = {
        "direct": balanced_helpers.load_direct_model(direct_path, config["model"], device),
        "residual": balanced_helpers.load_residual_model(residual_path, config["model"], device),
        "br_v01": balanced_helpers.load_residual_model(br_v01_path, config["model"], device),
        "br_v02_moderate": balanced_helpers.load_residual_model(br_v02_path, config["model"], device),
        "br_v02_strong": None,
    }
    if br_v02_strong_path.exists():
        models["br_v02_strong"] = balanced_helpers.load_residual_model(
            br_v02_strong_path, config["model"], device
        )
    v03_model = weighted_helpers.make_residual_unet(config["model"])
    v03_model.load_state_dict(v03_state)
    v03_model.to(device)
    v03_model.eval()
    models["br_v03_color"] = v03_model
    return models


def predict_methods(
    samples: list[dict[str, Any]],
    models: dict[str, nn.Module | None],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, float]]:
    active_methods = [method for method in METHODS if method != "br_v02_strong" or models.get(method) is not None]
    preds: dict[str, list[np.ndarray]] = {method: [] for method in active_methods}
    stats = balanced_helpers.empty_residual_stats()
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        inputs = np.stack([sample["blended"] for sample in batch_samples], axis=0)
        tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            direct = models["direct"](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            residual_layer = models["residual"](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            br_v01_layer = models["br_v01"](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            br_v02_layer = models["br_v02_moderate"](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            v03_layer = models["br_v03_color"](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
            br_v02_strong_layer = None
            if models.get("br_v02_strong") is not None:
                br_v02_strong_layer = (
                    models["br_v02_strong"](tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
                )
        residual_recon = np.clip(inputs - residual_layer, 0.0, 1.0).astype(np.float32)
        br_v01_recon = np.clip(inputs - br_v01_layer, 0.0, 1.0).astype(np.float32)
        br_v02_recon = np.clip(inputs - br_v02_layer, 0.0, 1.0).astype(np.float32)
        v03_preclip = inputs - v03_layer
        v03_recon = np.clip(v03_preclip, 0.0, 1.0).astype(np.float32)
        br_v02_strong_recon = (
            np.clip(inputs - br_v02_strong_layer, 0.0, 1.0).astype(np.float32)
            if br_v02_strong_layer is not None
            else None
        )
        balanced_helpers.update_residual_stats(stats, v03_layer, v03_preclip)
        for offset, sample in enumerate(batch_samples):
            blended = sample["blended"]
            preds["identity"].append(blended)
            preds["threshold"].append(baselines.threshold_baseline(blended))
            preds["direct"].append(np.clip(direct[offset], 0.0, 1.0).astype(np.float32))
            preds["residual"].append(residual_recon[offset])
            preds["br_v01"].append(br_v01_recon[offset])
            preds["br_v02_moderate"].append(br_v02_recon[offset])
            if br_v02_strong_recon is not None:
                preds["br_v02_strong"].append(br_v02_strong_recon[offset])
            preds["br_v03_color"].append(v03_recon[offset])
    return preds, balanced_helpers.finalise_residual_stats(stats)


def compute_per_sample(
    suite: str,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    threshold: float,
    loss_config: V03LossConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_methods = list(preds.keys())
    for idx, sample in enumerate(samples):
        target = sample["target"]
        blended = sample["blended"]
        affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
        core = target_core_mask_np(target)
        core_affected = np.logical_and(affected, core)
        noncore_affected = np.logical_and(affected, ~core)
        halo = halo_band_mask(affected, loss_config.halo_dilation_iters)
        info = sample.get("info", {})
        metadata = blend_metadata(target, blended, info, threshold)
        shift = info.get("shift", (0, 0))
        row: dict[str, Any] = {
            "suite": suite,
            "index": idx,
            "generation_difficulty": info.get("generation_difficulty", info.get("difficulty")),
            "generation_constraints_met": info.get("generation_constraints_met"),
            "generation_relaxed": bool(info.get("generation_relaxed", False)),
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
            "halo_band_fraction": float(halo.mean()),
        }
        identity_affected_mse = float("nan")
        for method in active_methods:
            values = image_metrics(
                preds[method][idx],
                target,
                affected,
                core_affected,
                noncore_affected,
                halo,
            )
            if method == "identity":
                identity_affected_mse = values["affected_mse"]
            for metric_name, value in values.items():
                row[f"{method}_{metric_name}"] = value
            row[f"{method}_improvement_ratio"] = safe_ratio(
                identity_affected_mse, values["affected_mse"]
            )
            row[f"{method}_worse_than_identity"] = bool(
                values["affected_mse"] > identity_affected_mse
            )
        if "br_v02_moderate" in active_methods:
            row["v03_beats_v02_moderate"] = (
                row["br_v03_color_affected_mse"]
                < row["br_v02_moderate_affected_mse"]
            )
            row["v03_to_v02_moderate_affected_mse_ratio"] = safe_ratio(
                row["br_v03_color_affected_mse"],
                row["br_v02_moderate_affected_mse"],
            )
            row["v03_to_v02_moderate_delta_e2000_ratio"] = safe_ratio(
                row["br_v03_color_affected_delta_e2000_mean"],
                row["br_v02_moderate_affected_delta_e2000_mean"],
            )
        row["v03_beats_direct"] = row["br_v03_color_affected_mse"] < row["direct_affected_mse"]
        row["v03_beats_br_v01"] = row["br_v03_color_affected_mse"] < row["br_v01_affected_mse"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["blend_severity_bin"] = balanced_helpers.blend_severity_bins(
        frame["blend_severity_score"]
    )
    frame["core_overlap_bin"] = balanced_helpers.core_overlap_bins(
        frame["core_obstruction_fraction"]
    )
    return frame


def aggregate_metrics(per_sample: pd.DataFrame, active_methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_suffixes = [
        "whole_mse",
        "whole_mae",
        "psnr",
        "ssim",
        "affected_mse",
        "affected_mae",
        "core_affected_mse",
        "core_affected_mae",
        "noncore_affected_mse",
        "noncore_affected_mae",
        "halo_band_mse",
        "halo_band_mae",
        "affected_gradient_error",
        "core_affected_gradient_error",
        "halo_band_gradient_error",
        "affected_lab_l1",
        "affected_lab_l2",
        "affected_delta_e76_mean",
        "affected_delta_e76_median",
        "affected_delta_e2000_mean",
        "affected_delta_e2000_median",
        "affected_lab_chroma_mae",
        "affected_rgb_saturation_mae",
        "core_affected_delta_e2000_mean",
        "core_affected_lab_chroma_mae",
        "core_affected_rgb_saturation_mae",
    ]
    for suite, suite_frame in per_sample.groupby("suite", sort=False):
        identity_affected = float(suite_frame["identity_affected_mse"].mean())
        identity_core = float(suite_frame["identity_core_affected_mse"].mean())
        for method in active_methods:
            row: dict[str, Any] = {
                "suite": suite,
                "method": method,
                "method_label": method_label(method),
                "n": int(len(suite_frame)),  # backward-compatible total count
                "n_total": int(len(suite_frame)),
                "n_valid_affected": int(
                    suite_frame[f"{method}_affected_mse"].notna().sum()
                ),
                "n_valid_core": int(
                    suite_frame[f"{method}_core_affected_mse"].notna().sum()
                ),
                "n_valid_noncore": int(
                    suite_frame[f"{method}_noncore_affected_mse"].notna().sum()
                ),
                "n_valid_halo": int(
                    suite_frame[f"{method}_halo_band_mse"].notna().sum()
                ),
                "mean_mask_fraction": float(suite_frame["mask_fraction"].mean()),
                "mean_core_obstruction_fraction": float(
                    suite_frame["core_obstruction_fraction"].mean()
                ),
                "worse_than_identity_count": int(
                    suite_frame[f"{method}_worse_than_identity"].sum()
                ),
                "worse_than_identity_fraction": float(
                    suite_frame[f"{method}_worse_than_identity"].mean()
                ),
            }
            for suffix in metric_suffixes:
                col = f"{method}_{suffix}"
                if col in suite_frame.columns:
                    row[suffix] = float(suite_frame[col].mean())
            row["improvement_vs_identity"] = safe_ratio(
                identity_affected, row.get("affected_mse", float("nan"))
            )
            row["core_improvement_vs_identity"] = safe_ratio(
                identity_core, row.get("core_affected_mse", float("nan"))
            )
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_summary(per_sample: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for suite, frame in per_sample.groupby("suite", sort=False):
        agg = aggregate[aggregate["suite"] == suite].set_index("method")
        v03 = agg.loc["br_v03_color"]
        v02 = agg.loc["br_v02_moderate"]
        direct = agg.loc["direct"]
        br_v01 = agg.loc["br_v01"]
        rows.append(
            {
                "suite": suite,
                "n": int(len(frame)),
                "v03_affected_mse": float(v03["affected_mse"]),
                "v02_moderate_affected_mse": float(v02["affected_mse"]),
                "v03_to_v02_affected_mse_ratio": safe_ratio(
                    float(v03["affected_mse"]), float(v02["affected_mse"])
                ),
                "v03_delta_e2000": float(v03["affected_delta_e2000_mean"]),
                "v02_moderate_delta_e2000": float(v02["affected_delta_e2000_mean"]),
                "v03_to_v02_delta_e2000_ratio": safe_ratio(
                    float(v03["affected_delta_e2000_mean"]),
                    float(v02["affected_delta_e2000_mean"]),
                ),
                "v03_gradient_error": float(v03["affected_gradient_error"]),
                "v02_moderate_gradient_error": float(v02["affected_gradient_error"]),
                "v03_chroma_error": float(v03["affected_lab_chroma_mae"]),
                "v02_moderate_chroma_error": float(v02["affected_lab_chroma_mae"]),
                "v03_halo_band_mse": float(v03["halo_band_mse"]),
                "v02_moderate_halo_band_mse": float(v02["halo_band_mse"]),
                "v03_worse_than_identity_count": int(v03["worse_than_identity_count"]),
                "v02_moderate_worse_than_identity_count": int(
                    v02["worse_than_identity_count"]
                ),
                "v03_vs_v02_win_rate": float(frame["v03_beats_v02_moderate"].mean()),
                "v03_vs_direct_win_rate": float(frame["v03_beats_direct"].mean()),
                "v03_vs_br_v01_win_rate": float(frame["v03_beats_br_v01"].mean()),
                "direct_affected_mse": float(direct["affected_mse"]),
                "br_v01_affected_mse": float(br_v01["affected_mse"]),
            }
        )
    return pd.DataFrame(rows)


def color_metric_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "suite",
        "method",
        "method_label",
        "affected_lab_l1",
        "affected_lab_l2",
        "affected_delta_e76_mean",
        "affected_delta_e2000_mean",
        "affected_delta_e2000_median",
        "affected_lab_chroma_mae",
        "affected_rgb_saturation_mae",
        "affected_gradient_error",
    ]
    return aggregate[[col for col in cols if col in aggregate.columns]].copy()


def update_example_bank(
    bank: dict[str, dict[str, Any]],
    suite: str,
    samples: list[dict[str, Any]],
    preds: dict[str, list[np.ndarray]],
    per_sample: pd.DataFrame,
    threshold: float,
) -> None:
    def store(key: str, row: pd.Series, score: float) -> None:
        current = bank.get(key)
        if current is not None and score <= current["score"]:
            return
        idx = int(row["index"])
        sample = samples[idx]
        bank[key] = {
            "score": float(score),
            "suite": suite,
            "threshold": threshold,
            "row": row.to_dict(),
            "sample": {
                "target": sample["target"].copy(),
                "blended": sample["blended"].copy(),
            },
            "preds": {
                "direct": preds["direct"][idx].copy(),
                "br_v02_moderate": preds["br_v02_moderate"][idx].copy(),
                "br_v03_color": preds["br_v03_color"][idx].copy(),
            },
        }

    frame = per_sample.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "br_v03_color_affected_mse",
            "br_v02_moderate_affected_mse",
            "direct_affected_mse",
        ]
    )
    if frame.empty:
        return
    improved = frame[frame["v03_to_v02_moderate_affected_mse_ratio"] < 0.85]
    if not improved.empty:
        row = improved.sort_values("v03_to_v02_moderate_affected_mse_ratio").iloc[0]
        store("v03_color_success_over_v02", row, 1.0 / max(row["v03_to_v02_moderate_affected_mse_ratio"], 1e-8))
    tradeoff = frame[
        (frame["v03_to_v02_moderate_affected_mse_ratio"] > 1.10)
        | (frame["v03_to_v02_moderate_delta_e2000_ratio"] > 1.10)
    ]
    if not tradeoff.empty:
        row = tradeoff.sort_values("v03_to_v02_moderate_affected_mse_ratio", ascending=False).iloc[0]
        store("v03_color_failure_or_tradeoff", row, row["v03_to_v02_moderate_affected_mse_ratio"])
        store("v02_beats_v03_counterexample", row, row["v03_to_v02_moderate_affected_mse_ratio"])
    direct_wins = frame[frame["direct_affected_mse"] < frame["br_v03_color_affected_mse"]]
    if not direct_wins.empty:
        row = direct_wins.assign(
            direct_ratio=direct_wins["br_v03_color_affected_mse"] / direct_wins["direct_affected_mse"]
        ).sort_values("direct_ratio", ascending=False).iloc[0]
        store("direct_still_beats_v03_counterexample", row, row["direct_ratio"])
    if suite == "compact_bright":
        compact = frame[frame["v03_to_v02_moderate_affected_mse_ratio"] < 0.85]
        if not compact.empty:
            row = compact.sort_values("v03_to_v02_moderate_affected_mse_ratio").iloc[0]
            store("compact_contaminant_fixed_by_v03", row, 1.0 / max(row["v03_to_v02_moderate_affected_mse_ratio"], 1e-8))
    color_improved = frame[
        (frame["v03_to_v02_moderate_delta_e2000_ratio"] < 0.85)
        & (frame["br_v03_color_affected_lab_chroma_mae"] <= frame["br_v02_moderate_affected_lab_chroma_mae"])
    ]
    if not color_improved.empty:
        row = color_improved.sort_values("v03_to_v02_moderate_delta_e2000_ratio").iloc[0]
        store("muted_color_improved_by_v03", row, 1.0 / max(row["v03_to_v02_moderate_delta_e2000_ratio"], 1e-8))
    if suite == "halo_band":
        halo = frame.assign(
            halo_ratio=frame["br_v03_color_halo_band_mse"] / frame["br_v02_moderate_halo_band_mse"]
        ).replace([np.inf, -np.inf], np.nan).dropna(subset=["halo_ratio"])
        if not halo.empty:
            row = halo.sort_values("halo_ratio").iloc[0]
            store("halo_artifact_comparison", row, 1.0 / max(row["halo_ratio"], 1e-8))


def delta_e_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_lab = color.rgb2lab(np.clip(pred, 0.0, 1.0))
    target_lab = color.rgb2lab(np.clip(target, 0.0, 1.0))
    return color.deltaE_ciede2000(pred_lab, target_lab)


def save_example_grid_from_bank(item: dict[str, Any], path: Path, title: str) -> Path:
    target = item["sample"]["target"]
    blended = item["sample"]["blended"]
    direct = item["preds"]["direct"]
    v02 = item["preds"]["br_v02_moderate"]
    v03 = item["preds"]["br_v03_color"]
    err_v02 = np.abs(v02 - target).mean(axis=-1)
    err_v03 = np.abs(v03 - target).mean(axis=-1)
    de_v03 = delta_e_map(v03, target)
    panels = [
        (target, "Target"),
        (blended, "Blend"),
        (direct, "Thayer-Direct"),
        (v02, "BR v0.2"),
        (v03, "BR v0.3"),
        (err_v02, "v0.2 error"),
        (err_v03, "v0.3 error"),
        (de_v03, "Delta E"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.05 * len(panels), 2.35))
    for ax, (image, panel_title) in zip(axes, panels):
        if image.ndim == 2:
            vmax = max(0.05, float(np.nanpercentile(image, 99)))
            ax.imshow(image, cmap="magma", vmin=0.0, vmax=vmax)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    safe_json(path.with_suffix(".json"), item["row"])
    return save_fig(fig, path, dpi=180)


def write_example_grids(run_dir: Path, bank: dict[str, dict[str, Any]]) -> dict[str, str | None]:
    titles = {
        "v03_color_success_over_v02": "v0.3 Success Over v0.2",
        "v03_color_failure_or_tradeoff": "v0.3 Failure or Tradeoff",
        "v02_beats_v03_counterexample": "v0.2 Beats v0.3 Counterexample",
        "direct_still_beats_v03_counterexample": "Direct Still Beats v0.3",
        "compact_contaminant_fixed_by_v03": "Compact Contaminant Improved",
        "muted_color_improved_by_v03": "Muted Color Improved",
        "halo_artifact_comparison": "Halo-Band Comparison",
    }
    written: dict[str, str | None] = {}
    for key, title in titles.items():
        item = bank.get(key)
        if item is None:
            written[f"{key}.png"] = None
            continue
        path = run_dir / "example_grids" / f"{key}.png"
        save_example_grid_from_bank(item, path, title)
        written[f"{key}.png"] = project_relative(path)
    return written


def plot_metric_bars(
    aggregate: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
    ylabel: str,
    methods: list[str] | None = None,
) -> Path:
    methods = methods or ["identity", "direct", "br_v01", "br_v02_moderate", "br_v03_color"]
    suites = list(aggregate["suite"].drop_duplicates())
    x = np.arange(len(suites))
    width = min(0.14, 0.72 / len(methods))
    offsets = np.linspace(-width * (len(methods) - 1) / 2, width * (len(methods) - 1) / 2, len(methods))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * len(suites)), 4.5))
    for method, offset in zip(methods, offsets):
        values = []
        for suite in suites:
            sub = aggregate[(aggregate["suite"] == suite) & (aggregate["method"] == method)]
            values.append(float(sub[metric].iloc[0]) if not sub.empty and metric in sub else np.nan)
        ax.bar(x + offset, values, width, color=method_color(method), label=method_label(method))
    ax.set_xticks(x, [suite.replace("_", " ") for suite in suites], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    return save_fig(fig, path)


def scatter_v03_vs_v02(per_sample: pd.DataFrame, metric_suffix: str, path: Path, title: str) -> Path:
    x_col = f"br_v02_moderate_{metric_suffix}"
    y_col = f"br_v03_color_{metric_suffix}"
    frame = per_sample[["suite", x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for suite, group in frame.groupby("suite"):
        ax.scatter(group[x_col], group[y_col], s=12, alpha=0.28, label=suite.replace("_", " "), edgecolors="none")
    max_val = float(np.nanmax(frame[[x_col, y_col]].to_numpy()))
    min_val = max(float(np.nanmin(frame[[x_col, y_col]].to_numpy())), 1e-8)
    ax.plot([min_val, max_val], [min_val, max_val], color="#8a4f49", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"v0.2 {metric_suffix.replace('_', ' ')}")
    ax.set_ylabel(f"v0.3 {metric_suffix.replace('_', ' ')}")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    return save_fig(fig, path)


def ratio_histogram(per_sample: pd.DataFrame, ratio_col: str, path: Path, title: str, xlabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for suite, group in per_sample.groupby("suite"):
        values = group[ratio_col].replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values > 0]
        if values.empty:
            continue
        clipped = values.clip(
            lower=float(np.nanpercentile(values, 1)),
            upper=float(np.nanpercentile(values, 99)),
        )
        ax.hist(clipped, bins=35, alpha=0.35, label=suite.replace("_", " "))
    ax.axvline(1.0, color="#8a4f49", linewidth=1.5, label="parity")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    return save_fig(fig, path)


def write_paper_figures(run_dir: Path, aggregate: pd.DataFrame, per_sample: pd.DataFrame) -> dict[str, str]:
    out = run_dir / "paper_figures"
    written = {
        "affected_mse_comparison": project_relative(
            plot_metric_bars(
                aggregate,
                "affected_mse",
                out / "affected_mse_comparison.png",
                "Affected-Region MSE",
                "Affected-region MSE",
            )
        ),
        "core_affected_mse": project_relative(
            plot_metric_bars(
                aggregate,
                "core_affected_mse",
                out / "core_affected_mse_chart.png",
                "Core Affected MSE",
                "Core affected MSE",
            )
        ),
        "halo_band_mse": project_relative(
            plot_metric_bars(
                aggregate,
                "halo_band_mse",
                out / "halo_band_mse_chart.png",
                "Halo-Band MSE",
                "Halo-band MSE",
            )
        ),
        "delta_e_color_error": project_relative(
            plot_metric_bars(
                aggregate,
                "affected_delta_e2000_mean",
                out / "delta_e2000_color_error_chart.png",
                "Affected Delta E 2000",
                "Mean Delta E 2000",
            )
        ),
        "gradient_edge_error": project_relative(
            plot_metric_bars(
                aggregate,
                "affected_gradient_error",
                out / "gradient_edge_error_chart.png",
                "Affected Gradient Error",
                "Gradient magnitude MAE",
            )
        ),
        "v03_vs_v02_affected_scatter": project_relative(
            scatter_v03_vs_v02(
                per_sample,
                "affected_mse",
                out / "v03_vs_v02_affected_mse_scatter.png",
                "v0.3 vs v0.2 Affected MSE",
            )
        ),
        "v03_vs_v02_color_scatter": project_relative(
            scatter_v03_vs_v02(
                per_sample,
                "affected_delta_e2000_mean",
                out / "v03_vs_v02_delta_e2000_scatter.png",
                "v0.3 vs v0.2 Delta E 2000",
            )
        ),
        "affected_ratio_histogram": project_relative(
            ratio_histogram(
                per_sample,
                "v03_to_v02_moderate_affected_mse_ratio",
                out / "hist_v03_to_v02_affected_mse_ratio.png",
                "v0.3 / v0.2 Affected MSE Ratio",
                "v0.3 affected MSE / v0.2 affected MSE",
            )
        ),
        "color_ratio_histogram": project_relative(
            ratio_histogram(
                per_sample,
                "v03_to_v02_moderate_delta_e2000_ratio",
                out / "hist_v03_to_v02_color_error_ratio.png",
                "v0.3 / v0.2 Color Error Ratio",
                "v0.3 Delta E / v0.2 Delta E",
            )
        ),
    }
    compact = aggregate[aggregate["suite"] == "compact_bright"]
    if not compact.empty:
        written["compact_contaminant_stress"] = project_relative(
            plot_metric_bars(
                aggregate[aggregate["suite"] == "compact_bright"],
                "affected_mse",
                out / "compact_contaminant_stress_chart.png",
                "Compact Contaminant Stress",
                "Affected-region MSE",
            )
        )
    written["weighted_color_variant_comparison"] = project_relative(
        plot_metric_bars(
            aggregate,
            "affected_mse",
            out / "weighted_color_variant_comparison.png",
            "Weighted and Color Variant Comparison",
            "Affected-region MSE",
            methods=["br_v01", "br_v02_moderate", "br_v02_strong", "br_v03_color"],
        )
    )
    return written


def generate_suite(
    suite: str,
    test_images: np.ndarray,
    config: dict[str, Any],
    settings: V03Settings,
    seed: int,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    threshold = settings.affected_region_threshold
    if suite == "normal":
        samples = normal_blends(
            test_images,
            settings.n_normal_test_blends,
            config,
            seed,
            "eval_normal",
        )
        return samples, threshold, {"suite": suite, "requested": len(samples), "accepted": len(samples)}
    if suite == "hard_stress":
        stress = dict(stress_helpers.STRESS_DEFAULTS)
        stress["n_stress_blends"] = settings.n_suite_blends
        stress["seed"] = seed
        samples = stress_helpers.generate_stress_blends(
            test_images[: int(stress["stress_source_subset"])],
            stress,
        )
        samples = strip_contaminants(samples)
        return samples, float(stress["affected_region_threshold"]), {"suite": suite, **stress}
    suite_specs: dict[str, dict[str, Any]] = {
        "compact_bright": {
            "max_shift": 24,
            "brightness_range": (1.35, 1.95),
            "blur_range": (0.0, 0.08),
            "noise_range": (0.0, 0.006),
            "min_size_ratio": 0.15,
            "max_size_ratio": 0.90,
            "min_mask_fraction": 0.004,
            "min_core_obstruction": None,
            "max_attempt_multiplier": 120,
        },
        "high_core_obstruction": {
            "max_shift": 12,
            "brightness_range": (0.9, 1.50),
            "blur_range": (0.0, 0.12),
            "noise_range": (0.0, 0.006),
            "min_size_ratio": 0.70,
            "max_size_ratio": None,
            "min_mask_fraction": 0.01,
            "min_core_obstruction": 0.82,
            "max_attempt_multiplier": 140,
        },
        "halo_band": {
            "max_shift": 28,
            "brightness_range": (0.85, 1.45),
            "blur_range": (0.05, 0.25),
            "noise_range": (0.0, 0.006),
            "min_size_ratio": 0.70,
            "max_size_ratio": None,
            "min_mask_fraction": 0.015,
            "min_core_obstruction": None,
            "max_attempt_multiplier": 100,
        },
        "color_saturation": {
            "max_shift": 24,
            "brightness_range": (1.10, 1.65),
            "blur_range": (0.0, 0.10),
            "noise_range": (0.0, 0.006),
            "min_size_ratio": 0.40,
            "max_size_ratio": None,
            "min_mask_fraction": 0.008,
            "min_core_obstruction": None,
            "color_tint": True,
            "max_attempt_multiplier": 120,
        },
    }
    if suite not in suite_specs:
        raise KeyError(f"Unknown suite: {suite}")
    samples, diag = generate_targeted_blends(
        test_images,
        settings.n_suite_blends,
        seed,
        f"eval_{suite}",
        threshold,
        **suite_specs[suite],
    )
    return samples, threshold, {"suite": suite, **diag}


def decide_outcome(summary: pd.DataFrame) -> dict[str, Any]:
    keyed = summary.set_index("suite")
    required = [suite for suite in ("normal", "hard_stress") if suite in keyed.index]
    primary_ok = []
    color_ok = []
    gradient_ok = []
    chroma_ok = []
    halo_ok = []
    worse_ok = []
    for suite in required:
        row = keyed.loc[suite]
        primary_ok.append(row["v03_to_v02_affected_mse_ratio"] <= 1.02)
        color_ok.append(row["v03_to_v02_delta_e2000_ratio"] < 1.0)
        gradient_ok.append(row["v03_gradient_error"] <= row["v02_moderate_gradient_error"])
        chroma_ok.append(row["v03_chroma_error"] <= row["v02_moderate_chroma_error"])
        halo_ok.append(row["v03_halo_band_mse"] <= row["v02_moderate_halo_band_mse"] * 1.10)
        worse_ok.append(
            row["v03_worse_than_identity_count"]
            <= row["v02_moderate_worse_than_identity_count"]
        )
    compact_improves = False
    if "compact_bright" in keyed.index:
        compact_improves = bool(keyed.loc["compact_bright", "v03_to_v02_affected_mse_ratio"] < 1.0)
    primary_pass = bool(primary_ok and all(primary_ok) and all(worse_ok))
    visual_votes = sum(color_ok) + sum(gradient_ok) + sum(chroma_ok) + [compact_improves].count(True)
    halo_pass = bool(halo_ok and all(halo_ok))
    if primary_pass and visual_votes >= max(2, len(required)) and halo_pass:
        verdict = "candidate_improves"
        recommendation = (
            "v0.3 improves the evaluated color/structure evidence without hurting primary affected-region metrics; still treat as a candidate pending repeat seeds."
        )
    elif visual_votes >= max(2, len(required)) and not primary_pass:
        verdict = "visual_tradeoff"
        recommendation = (
            "v0.3 improves some color/structure metrics but weakens primary MSE enough to document it as a tradeoff, not a new best."
        )
    elif compact_improves:
        verdict = "targeted_ablation"
        recommendation = (
            "v0.3 helps compact-contaminant cases but does not clearly replace v0.2 Moderate in aggregate."
        )
    else:
        verdict = "reject_or_ablation"
        recommendation = (
            "v0.3 does not beat v0.2 Moderate under the primary and visual criteria; document as an ablation."
        )
    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "primary_pass": primary_pass,
        "visual_vote_count": int(visual_votes),
        "halo_pass": halo_pass,
        "compact_improves": compact_improves,
    }


def format_metric_rows(aggregate: pd.DataFrame, suite: str) -> str:
    frame = aggregate[aggregate["suite"] == suite]
    rows = [
        "| Method | Affected MSE | Core MSE | Halo MSE | Delta E 2000 | Chroma error | Gradient error | Improvement | Worse than identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        rows.append(
            "| {method} | {affected:.6f} | {core:.6f} | {halo:.6f} | {de:.3f} | {chroma:.3f} | {grad:.6f} | {ratio:.2f}x | {worse}/{n} |".format(
                method=row["method_label"],
                affected=row["affected_mse"],
                core=row["core_affected_mse"],
                halo=row["halo_band_mse"],
                de=row["affected_delta_e2000_mean"],
                chroma=row["affected_lab_chroma_mae"],
                grad=row["affected_gradient_error"],
                ratio=row["improvement_vs_identity"],
                worse=int(row["worse_than_identity_count"]),
                n=int(row["n"]),
            )
        )
    return "\n".join(rows)


def write_v03_report(
    run_dir: Path,
    best_checkpoint: Path,
    final_checkpoint: Path,
    experiment_title: str,
    settings: V03Settings,
    loss_config: V03LossConfig,
    history: pd.DataFrame,
    aggregate: pd.DataFrame,
    summary: pd.DataFrame,
    outcome: dict[str, Any],
    figure_paths: dict[str, str],
    grid_paths: dict[str, str | None],
    artifact_prefix: str,
    diagnostic_report_name: str,
) -> None:
    best_epoch = int(history["best_epoch"].iloc[-1])
    text = f"""# {experiment_title} Report

## Motivation

This private v0.3 run targets muted/desaturated outputs, blurred wisps and edges, compact bright contaminant misses, and occasional broad low-level residual artifacts seen in individual v0.2 examples.

## Setup

- Architecture: unchanged compact residual U-Net from Thayer-BR v0.2.
- Task: `true_residual = blended - target`; `reconstruction = blended - predicted_residual`.
- Best checkpoint: `{project_relative(best_checkpoint)}`.
- Final checkpoint: `{project_relative(final_checkpoint)}`.
- Train/validation blends: {settings.n_train_blends:,} / {settings.n_val_blends:,}.
- Epochs: {settings.num_epochs}.
- Best validation loss: {float(history['best_val_loss'].iloc[-1]):.6f} at epoch {best_epoch}.
- Final train/validation loss: {float(history['train_loss'].iloc[-1]):.6f} / {float(history['val_loss'].iloc[-1]):.6f}.

## Loss Formula

`loss = residual_mse + alpha*recon_l1 + beta*affected_core_loss + gamma*gradient_loss + delta*color_loss + eta*halo_band_loss`

- `alpha = {loss_config.reconstruction_l1_weight}`
- `beta = {loss_config.affected_core_loss_weight}`
- `gamma = {loss_config.gradient_loss_weight}`
- `delta = {loss_config.color_loss_weight}`
- `eta = {loss_config.halo_band_loss_weight}`
- affected/core weights: `{loss_config.affected_weight}` / `{loss_config.core_weight}`
- affected threshold: `{loss_config.affected_threshold}`

Affected/core and halo losses are normalized by summed spatial weights so larger masks do not dominate solely because they contain more pixels.

## Color Metrics

CIEDE2000 was implemented for evaluation with `skimage.color.deltaE_ciede2000`. Training does not use exact CIEDE2000; it uses a differentiable RGB chroma plus color-direction proxy at low weight. Lab/Delta E assumes standard RGB/sRGB-like arrays in `[0, 1]`. Galaxy10 DECaLS RGB images are survey composites, not guaranteed true human-color photographs, so color metrics are visual-quality evidence and affected-region MSE remains primary.

## Training Distribution

The realized distribution was 40% normal clean, 25% high-overlap/core-obstruction, 20% compact bright contaminants, 10% brightness/size stress, 5% low-overlap/easy stabilizer, and 0% artifact/outlier examples because no source-quality artifact flags were available. The intended 5% artifact bucket was excluded and redistributed to normal clean blends, as documented in `logs/training_composition.json`.

## Normal Metrics

{format_metric_rows(aggregate, 'normal')}

## Stress Metrics

{format_metric_rows(aggregate, 'hard_stress')}

## Compact-Contaminant Metrics

{format_metric_rows(aggregate, 'compact_bright')}

## Outcome

Verdict: `{outcome['verdict']}`.

{outcome['recommendation']}

Comparison summary is saved in `tables/{artifact_prefix}_comparison_summary.csv`. If v0.3 improves color or compact failures but weakens primary MSE, it should be documented as a tradeoff or ablation rather than promoted to current best.

## Figures

Paper figures:

{chr(10).join(f'- `{path}`' for path in figure_paths.values())}

Example grids:

{chr(10).join(f'- `{path}`' for path in grid_paths.values() if path)}

## Recommended Next Step

Repeat the candidate with multiple seeds only if the primary affected/core metrics remain comparable to v0.2 Moderate and the color/edge improvements are visible in the qualitative grids. If stronger color weighting is tested later, treat it as a separate ablation.
"""
    safe_write_text(run_dir / "diagnostics" / diagnostic_report_name, text)


def main() -> int:
    args = parse_args()
    global V03_METHOD_LABEL
    V03_METHOD_LABEL = args.v03_method_label
    config = load_config(args.config)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    is_delta_candidate = "delta_candidate" in args.run_name_prefix.lower()
    artifact_prefix = args.artifact_prefix or (
        "v03_delta" if is_delta_candidate else "v03_color"
    )
    diagnostic_report_name = args.diagnostic_report_name or (
        "v03_delta_candidate_report.md"
        if is_delta_candidate
        else "v03_color_structure_report.md"
    )
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    run_dir, best_checkpoint, final_checkpoint = make_run_paths(
        output_root,
        stamp,
        args.run_dir,
        args.run_name_prefix,
        args.checkpoint_name_prefix,
    )
    before_snapshot_path = run_dir / "logs" / "checkpoint_integrity_before.json"
    safe_json(before_snapshot_path, checkpoint_snapshot(output_root / "checkpoints"))

    def final_integrity_on_exit() -> None:
        comparison_path = run_dir / "logs/checkpoint_integrity_comparison.json"
        if comparison_path.exists():
            return
        try:
            save_after_checkpoint_integrity(run_dir, output_root / "checkpoints")
        except Exception as exc:  # best-effort diagnostic during exception unwinding
            failure_path = run_dir / "diagnostics/checkpoint_integrity_finalize_failure.md"
            if not failure_path.exists():
                failure_path.write_text(
                    "# Checkpoint integrity finalization failure\n\n"
                    f"The exit-time integrity snapshot failed: `{type(exc).__name__}: {exc}`\n",
                    encoding="utf-8",
                )

    atexit.register(final_integrity_on_exit)
    write_color_metric_report(run_dir)
    settings = V03Settings(
        n_train_blends=args.n_train_blends,
        n_val_blends=args.n_val_blends,
        n_normal_test_blends=args.n_normal_test_blends,
        n_suite_blends=args.n_suite_blends,
        train_source_subset=args.train_source_subset,
        val_source_subset=args.val_source_subset,
        test_source_subset=args.test_source_subset,
        num_epochs=args.num_epochs,
        affected_region_threshold=args.affected_threshold,
    )
    loss_config = V03LossConfig(
        reconstruction_l1_weight=args.reconstruction_l1_weight,
        affected_core_loss_weight=args.affected_core_loss_weight,
        gradient_loss_weight=args.gradient_loss_weight,
        color_loss_weight=args.color_loss_weight,
        halo_band_loss_weight=args.halo_band_loss_weight,
        affected_weight=args.affected_weight,
        core_weight=args.core_weight,
        affected_threshold=args.affected_threshold,
    )
    safe_yaml(run_dir / "logs" / "loss_config.yaml", asdict(loss_config))
    safe_yaml(
        run_dir / "logs" / "run_config.yaml",
        {
            "project_root": ".",
            "timestamp": stamp,
            "settings": asdict(settings),
            "loss_config": asdict(loss_config),
            "config": config,
            "architecture": "unchanged compact residual U-Net from Thayer-BR v0.2",
            "experiment_title": args.experiment_title,
            "artifact_prefix": artifact_prefix,
            "diagnostic_report_name": diagnostic_report_name,
            "checkpoints": {
                "direct": project_relative(resolve_path(args.direct_checkpoint)),
                "residual": project_relative(resolve_path(args.residual_checkpoint)),
                "br_v01": project_relative(resolve_path(args.br_v01_checkpoint)),
                "br_v02_moderate": project_relative(resolve_path(args.br_v02_moderate_checkpoint)),
                "br_v02_strong": project_relative(resolve_path(args.br_v02_strong_checkpoint)),
                "br_v03_best": project_relative(best_checkpoint),
                "br_v03_final": project_relative(final_checkpoint),
            },
        },
    )

    seed = int(config["seed"]) + settings.seed_offset
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
        safe_json(
            run_dir / "logs" / "device_selection.json",
            {**device_status, "status": "stopped", "error": str(exc)},
        )
        print(f"Stopping before full training/evaluation: {exc}", flush=True)
        return 2
    safe_json(
        run_dir / "logs" / "device_selection.json",
        {**device_status, "status": "selected", "selected_device": str(device)},
    )
    print(f"Using device: {device}", flush=True)
    print(f"Run directory: {project_relative(run_dir)}", flush=True)

    print("Loading dataset splits.", flush=True)
    train_images, val_images, test_images = load_split_subsets(config, settings)

    print("Generating v0.3 balanced color/structure training blends.", flush=True)
    train_blends, train_comp = generate_v03_training_blends(
        train_images,
        settings.n_train_blends,
        config,
        seed + 1000,
        settings.affected_region_threshold,
        "train",
    )
    print("Generating validation blends.", flush=True)
    val_blends, val_comp = generate_v03_training_blends(
        val_images,
        settings.n_val_blends,
        config,
        seed + 2000,
        settings.affected_region_threshold,
        "val",
    )
    composition = {"train": train_comp, "validation": val_comp}
    safe_json(run_dir / "logs" / "training_composition.json", composition)
    component_rows = []
    for split, payload in composition.items():
        for component in payload["components"]:
            component_rows.append({"split": split, **component})
    safe_csv(run_dir / "tables" / "training_composition.csv", pd.DataFrame(component_rows))
    del train_images, val_images
    gc.collect()

    train_ds = V03ResidualBlendDataset(train_blends)
    val_ds = V03ResidualBlendDataset(val_blends)
    requested_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["training"].get("batch_size", 8))
    )
    print("Training v0.3 color/structure residual U-Net.", flush=True)
    trained_model, best_state, history, used_batch_size, attempts = train_with_memory_retry(
        train_ds,
        val_ds,
        config["model"],
        config["training"],
        settings,
        requested_batch_size,
        device,
        seed + 3000,
        loss_config,
    )
    safe_csv(run_dir / "tables" / "training_history.csv", history)
    safe_json(run_dir / "logs" / "training_attempts.json", attempts)
    final_state = {
        key: value.detach().cpu().clone()
        for key, value in trained_model.state_dict().items()
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
        args.experiment_title,
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
        args.experiment_title,
    )
    print(f"Saved best checkpoint: {project_relative(best_checkpoint)}", flush=True)
    print(f"Saved final checkpoint: {project_relative(final_checkpoint)}", flush=True)
    del trained_model, final_state, train_blends, val_blends, train_ds, val_ds
    clear_torch_cache()
    gc.collect()

    models = load_models(config, device, args, best_state)
    active_methods = [method for method in METHODS if method != "br_v02_strong" or models.get(method) is not None]
    suite_names = [
        "normal",
        "hard_stress",
        "compact_bright",
        "high_core_obstruction",
        "halo_band",
        "color_saturation",
    ]
    all_per_sample: list[pd.DataFrame] = []
    suite_diags: list[dict[str, Any]] = []
    example_bank: dict[str, dict[str, Any]] = {}
    for idx, suite in enumerate(suite_names):
        print(f"Generating evaluation suite: {suite}.", flush=True)
        samples, threshold, suite_diag = generate_suite(
            suite,
            test_images,
            config,
            settings,
            seed + 4000 + idx * 137,
        )
        suite_diags.append(suite_diag)
        print(f"Evaluating suite: {suite} ({len(samples)} samples).", flush=True)
        preds, v03_stats = predict_methods(samples, models, device, used_batch_size)
        safe_json(run_dir / "diagnostics" / f"{suite}_v03_output_stats.json", v03_stats)
        per_sample = compute_per_sample(suite, samples, preds, threshold, loss_config)
        safe_csv(run_dir / "tables" / f"{suite}_per_sample_metrics.csv", per_sample)
        update_example_bank(example_bank, suite, samples, preds, per_sample, threshold)
        all_per_sample.append(per_sample)
        del samples, preds, per_sample
        clear_torch_cache()
        gc.collect()

    safe_json(run_dir / "logs" / "evaluation_suite_generation.json", suite_diags)
    per_sample_all = pd.concat(all_per_sample, ignore_index=True)
    aggregate = aggregate_metrics(per_sample_all, active_methods)
    summary = comparison_summary(per_sample_all, aggregate)
    color_summary = color_metric_summary(aggregate)
    outcome = decide_outcome(summary)
    safe_csv(run_dir / "tables" / f"{artifact_prefix}_per_sample_metrics.csv", per_sample_all)
    safe_csv(run_dir / "tables" / f"{artifact_prefix}_suite_metrics.csv", aggregate)
    safe_csv(run_dir / "tables" / f"{artifact_prefix}_comparison_summary.csv", summary)
    safe_csv(run_dir / "tables" / f"{artifact_prefix}_color_metric_summary.csv", color_summary)
    safe_json(run_dir / "logs" / f"{artifact_prefix}_outcome.json", outcome)

    print("Generating figures and qualitative grids.", flush=True)
    figure_paths = write_paper_figures(run_dir, aggregate, per_sample_all)
    grid_paths = write_example_grids(run_dir, example_bank)
    safe_json(run_dir / "logs" / "paper_figures.json", figure_paths)
    safe_json(run_dir / "logs" / "example_grids.json", grid_paths)
    write_v03_report(
        run_dir,
        best_checkpoint,
        final_checkpoint,
        args.experiment_title,
        settings,
        loss_config,
        history,
        aggregate,
        summary,
        outcome,
        figure_paths,
        grid_paths,
        artifact_prefix,
        diagnostic_report_name,
    )

    integrity = save_after_checkpoint_integrity(run_dir, output_root / "checkpoints")
    if not integrity.get("old_checkpoints_unchanged", False):
        raise RuntimeError("Old checkpoint integrity check failed.")

    print("Normal metrics:", flush=True)
    print(format_metric_rows(aggregate, "normal"), flush=True)
    print("Hard stress metrics:", flush=True)
    print(format_metric_rows(aggregate, "hard_stress"), flush=True)
    print("Compact-contaminant metrics:", flush=True)
    print(format_metric_rows(aggregate, "compact_bright"), flush=True)
    print(f"Outcome: {outcome['verdict']} - {outcome['recommendation']}", flush=True)
    print("Old checkpoints unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
