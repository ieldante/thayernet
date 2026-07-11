"""Train and evaluate the private Thayer-ResUNet v0.4 candidate.

The baseline invocation uses the requested architecture-only ablation:

    .venv/bin/python scripts/train_resunet_v04.py \
      --n-train-blends 8000 --n-val-blends 1000 \
      --n-normal-test-blends 1000 --n-suite-blends 1000 \
      --num-epochs 20 --batch-size 8

The default run/checkpoint prefixes are exactly ``resunet_v04_candidate`` and
``unet_resunet_v04_candidate``. All generated artifacts are timestamped under
``outputs/`` and every writer refuses to overwrite an existing path. Optional
Core+ and Halo-safe runs can use separate CLI prefixes and loss weights without
changing the baseline's architecture or accidentally enabling color losses.
"""

from __future__ import annotations

import argparse
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_balanced_residual_unet as balanced_helpers
import train_v03_color_structure_unet as v03_helpers
import train_weighted_residual_unet as weighted_helpers
from src import baselines
from src import models as gd_models
from src import train as gd_train
from src import utils as gd_utils


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/default.yaml"
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

METHOD_ORDER = (
    "identity",
    "threshold",
    "direct",
    "residual",
    "br_v01",
    "br_v02_moderate",
    "br_v03_delta",
    "resunet_v04_baseline",
    "resunet_v04",
)
SUITES = (
    "normal",
    "hard_stress",
    "compact_bright",
    "high_core_obstruction",
    "halo_band",
    "color_saturation",
)


@dataclass(frozen=True)
class ResUNetSettings:
    """Training scale, evaluation scale, and balanced blend composition."""

    n_train_blends: int = 8000
    n_val_blends: int = 1000
    n_normal_test_blends: int = 1000
    n_suite_blends: int = 1000
    train_source_subset: int = 5000
    val_source_subset: int = 1000
    test_source_subset: int = 1000
    num_epochs: int = 20
    normal_fraction: float = 0.50
    high_overlap_fraction: float = 0.30
    brightness_size_fraction: float = 0.20
    affected_region_threshold: float = 0.02
    seed_offset: int = 704


@dataclass(frozen=True)
class ResUNetLossConfig:
    """v0.2 Moderate weighted residual MSE plus an optional halo term.

    ``affected_extra_weight`` and ``core_extra_weight`` are additive extras,
    exactly matching the proven v0.2 implementation. With defaults, ordinary
    pixels have weight 1, affected pixels 4, and affected core pixels 6.
    """

    background_weight: float = 1.0
    affected_extra_weight: float = 3.0
    core_extra_weight: float = 2.0
    affected_threshold: float = 0.02
    core_aperture_fraction: float = 0.18
    core_brightness_fraction: float = 0.55
    halo_weight: float = 0.0
    halo_dilation_iters: int = 5
    eps: float = 1e-8


class TrainingStopError(RuntimeError):
    """Suspicious completed-epoch stop carrying history for diagnostics."""

    def __init__(self, message: str, history: pd.DataFrame) -> None:
        super().__init__(message)
        self.history = history
        self.attempts: list[dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the private Thayer-ResUNet v0.4 architecture candidate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-name-prefix", default="resunet_v04_candidate")
    parser.add_argument(
        "--checkpoint-name-prefix", default="unet_resunet_v04_candidate"
    )
    parser.add_argument(
        "--experiment-title", default="Thayer-ResUNet v0.4 Candidate"
    )
    parser.add_argument("--variant-name", default="baseline_moderate")
    parser.add_argument(
        "--report-name", default="resunet_v04_candidate_report.md"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--direct-checkpoint", type=Path, default=DEFAULT_DIRECT_CHECKPOINT)
    parser.add_argument(
        "--residual-checkpoint", type=Path, default=DEFAULT_RESIDUAL_CHECKPOINT
    )
    parser.add_argument(
        "--br-v01-checkpoint", type=Path, default=DEFAULT_BR_V01_CHECKPOINT
    )
    parser.add_argument(
        "--br-v02-moderate-checkpoint",
        type=Path,
        default=DEFAULT_BR_V02_MODERATE_CHECKPOINT,
    )
    parser.add_argument(
        "--delta-checkpoint",
        type=Path,
        default=None,
        help="Completed v0.3 Delta best checkpoint; newest matching checkpoint is auto-detected when omitted.",
    )
    parser.add_argument(
        "--baseline-resunet-checkpoint",
        type=Path,
        default=None,
        help="Baseline ResUNet best checkpoint for controlled Core+/Halo-safe comparisons.",
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
    parser.add_argument("--affected-extra-weight", type=float, default=3.0)
    parser.add_argument("--core-extra-weight", type=float, default=2.0)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--affected-threshold", type=float, default=0.02)
    parser.add_argument("--halo-weight", type=float, default=0.0)
    parser.add_argument("--halo-dilation-iters", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=704)
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return config


def safe_write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


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


def save_figure(fig: plt.Figure, path: Path, dpi: int = 220) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def make_run_paths(
    output_root: Path,
    stamp: str,
    run_dir_arg: Path | None,
    run_name_prefix: str,
    checkpoint_name_prefix: str,
) -> tuple[Path, Path, Path]:
    for label, value in (
        ("run-name-prefix", run_name_prefix),
        ("checkpoint-name-prefix", checkpoint_name_prefix),
    ):
        if not value or Path(value).name != value:
            raise ValueError(f"{label} must be a non-empty filename component: {value!r}")
    run_dir = (
        resolve_path(run_dir_arg)
        if run_dir_arg is not None
        else output_root / "runs" / f"{run_name_prefix}_{stamp}"
    )
    best_checkpoint = (
        output_root / "checkpoints" / f"{checkpoint_name_prefix}_{stamp}_best.pth"
    )
    final_checkpoint = (
        output_root / "checkpoints" / f"{checkpoint_name_prefix}_{stamp}_final.pth"
    )
    output_root_resolved = output_root.resolve()
    runs_root_resolved = (output_root / "runs").resolve()
    checkpoints_root_resolved = (output_root / "checkpoints").resolve()
    try:
        run_dir.resolve().relative_to(runs_root_resolved)
        best_checkpoint.resolve().relative_to(checkpoints_root_resolved)
        final_checkpoint.resolve().relative_to(checkpoints_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"All run artifacts must stay below {project_relative(output_root_resolved)}."
        ) from exc
    collisions = [
        str(path)
        for path in (run_dir, best_checkpoint, final_checkpoint)
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "Refusing to reuse existing run/checkpoint paths: " + "; ".join(collisions)
        )
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


def write_checkpoint_integrity_after(
    run_dir: Path, checkpoint_dir: Path
) -> dict[str, Any]:
    before_path = run_dir / "logs/checkpoint_integrity_before.json"
    after_path = run_dir / "logs/checkpoint_integrity_after.json"
    comparison_path = run_dir / "logs/checkpoint_integrity_comparison.json"
    if after_path.exists() or comparison_path.exists():
        raise FileExistsError("Checkpoint integrity after-state was already written.")
    after = checkpoint_snapshot(checkpoint_dir)
    safe_json(after_path, after)
    if not before_path.exists():
        comparison = {
            "status": "missing_before_snapshot",
            "old_checkpoints_unchanged": False,
            "comparisons": [],
        }
        safe_json(comparison_path, comparison)
        return comparison
    before = json.loads(before_path.read_text(encoding="utf-8"))
    before_rows = {row["path"]: row for row in before.get("checkpoints", [])}
    after_rows = {row["path"]: row for row in after.get("checkpoints", [])}
    rows = []
    for rel_path, old in before_rows.items():
        new = after_rows.get(rel_path)
        unchanged = bool(
            new is not None
            and old["size_bytes"] == new["size_bytes"]
            and old["mtime_ns"] == new["mtime_ns"]
            and old["sha256"] == new["sha256"]
        )
        rows.append(
            {"path": rel_path, "before": old, "after": new, "unchanged": unchanged}
        )
    comparison = {
        "status": "compared",
        "old_checkpoint_count": len(rows),
        "old_checkpoints_unchanged": all(row["unchanged"] for row in rows),
        "comparisons": rows,
    }
    safe_json(comparison_path, comparison)
    return comparison


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


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_resunet(model_config: dict[str, Any]) -> gd_models.ResUNet:
    return gd_models.ResUNet(**model_config)


def halo_band_loss_torch(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    true_residual: torch.Tensor,
    loss_config: ResUNetLossConfig,
) -> torch.Tensor:
    """Return channel-normalized reconstruction MSE in a dilated affected ring."""

    radius = int(loss_config.halo_dilation_iters)
    if loss_config.halo_weight <= 0.0 or radius <= 0:
        return reconstruction.new_tensor(0.0)
    affected = (
        torch.mean(torch.abs(true_residual), dim=1, keepdim=True)
        > loss_config.affected_threshold
    )
    # Match scipy.ndimage.binary_dilation's default 2-D cross connectivity,
    # which is also used by the evaluation halo-band mask. Repeating the cross
    # dilation ``radius`` times produces the same Manhattan-distance diamond.
    cross = affected.new_tensor(
        [[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=reconstruction.dtype,
    )
    dilated_float = affected.float()
    for _ in range(radius):
        dilated_float = F.conv2d(dilated_float, cross, padding=1)
        dilated_float = (dilated_float > 0).to(reconstruction.dtype)
    dilated = dilated_float > 0
    ring = (dilated & ~affected).float()
    normalizer = ring.sum() * reconstruction.shape[1] + loss_config.eps
    if float(normalizer.detach().cpu()) <= loss_config.eps:
        return reconstruction.new_tensor(0.0)
    return (ring * (reconstruction - target) ** 2).sum() / normalizer


def resunet_loss(
    blended: torch.Tensor,
    predicted_residual: torch.Tensor,
    true_residual: torch.Tensor,
    target: torch.Tensor,
    loss_config: ResUNetLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the exact v0.2 base objective and optional normalized halo term."""

    base, base_stats = weighted_helpers.weighted_residual_loss(
        predicted_residual,
        true_residual,
        target,
        loss_config,
    )
    reconstruction = blended - predicted_residual
    halo = halo_band_loss_torch(
        reconstruction,
        target,
        true_residual,
        loss_config,
    )
    total = base + loss_config.halo_weight * halo
    stats = {
        **base_stats,
        "base_weighted_residual_loss": float(base.detach().cpu()),
        "halo_band_loss": float(halo.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, stats


def _average_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_config: ResUNetLossConfig,
) -> dict[str, float]:
    model.eval()
    n_samples = 0
    loss_sum = 0.0
    affected_mse_values: list[float] = []
    batch_stats: list[dict[str, float]] = []
    clip_count = 0.0
    extreme_count = 0.0
    pixel_count = 0.0
    with torch.no_grad():
        for blended, true_residual, target in val_loader:
            blended = blended.to(device)
            true_residual = true_residual.to(device)
            target = target.to(device)
            predicted = model(blended)
            loss, stats = resunet_loss(
                blended, predicted, true_residual, target, loss_config
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")
            batch_size = blended.shape[0]
            n_samples += batch_size
            loss_sum += float(loss.item()) * batch_size
            batch_stats.append(stats)
            preclip = blended - predicted
            reconstruction = torch.clamp(preclip, 0.0, 1.0)
            clip_count += float(
                torch.count_nonzero((preclip < 0.0) | (preclip > 1.0)).item()
            )
            extreme_count += float(
                torch.count_nonzero((reconstruction < 0.01) | (reconstruction > 0.99)).item()
            )
            pixel_count += float(preclip.numel())
            affected = (
                torch.mean(torch.abs(true_residual), dim=1)
                > loss_config.affected_threshold
            )
            squared = torch.mean((reconstruction - target) ** 2, dim=1)
            for idx in range(batch_size):
                if bool(affected[idx].any().detach().cpu()):
                    affected_mse_values.append(
                        float(squared[idx][affected[idx]].mean().item())
                    )
    result = {
        f"val_{key}": value for key, value in _average_stats(batch_stats).items()
    }
    result.update(
        {
            "val_loss": loss_sum / max(n_samples, 1),
            "val_affected_mse": float(np.nanmean(affected_mse_values))
            if affected_mse_values
            else float("nan"),
            "val_reconstruction_clip_fraction": clip_count / max(pixel_count, 1.0),
            "val_reconstruction_extreme_fraction": extreme_count
            / max(pixel_count, 1.0),
        }
    )
    return result


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
    loss_config: ResUNetLossConfig,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame]:
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model.to(device)
    rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
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
            loss, stats = resunet_loss(
                blended, predicted, true_residual, target, loss_config
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimiser.step()
            train_loss_sum += float(loss.item()) * blended.shape[0]
            train_stats.append(stats)
        train_loss = train_loss_sum / len(train_ds)
        val_diag = evaluate_validation(model, val_loader, device, loss_config)
        val_loss = val_diag["val_loss"]
        recent_val.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        train_diag = _average_stats(train_stats)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
                **{f"train_{key}": value for key, value in train_diag.items()},
                **val_diag,
            }
        )
        print(
            f"Epoch {epoch}/{num_epochs}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, "
            f"val affected MSE={val_diag['val_affected_mse']:.6f}, "
            f"clip={val_diag['val_reconstruction_clip_fraction']:.3%}, "
            f"best={best_val:.6f} @ {best_epoch}",
            flush=True,
        )
        stop_message = None
        if train_loss > 2.0 or val_loss > 2.0:
            stop_message = (
                f"Training loss exploded at epoch {epoch}: "
                f"train={train_loss:.6f}, val={val_loss:.6f}."
            )
        elif len(recent_val) >= 4 and val_loss > min(recent_val[:-1]) * 10.0:
            stop_message = (
                f"Validation loss exploded at epoch {epoch}: "
                f"val={val_loss:.6f}, previous_best={min(recent_val[:-1]):.6f}."
            )
        elif val_diag["val_reconstruction_clip_fraction"] > 0.20:
            stop_message = (
                f"Extreme clipping at epoch {epoch}: "
                f"{val_diag['val_reconstruction_clip_fraction']:.3%}."
            )
        elif val_diag["val_reconstruction_extreme_fraction"] > 0.98:
            stop_message = (
                f"Validation reconstructions are mostly black/white at epoch {epoch}: "
                f"{val_diag['val_reconstruction_extreme_fraction']:.3%}."
            )
        if stop_message is not None:
            raise TrainingStopError(stop_message, pd.DataFrame(rows))
    if not best_state:
        raise RuntimeError("Training completed without a finite best checkpoint state.")
    return model, best_state, pd.DataFrame(rows)


def train_with_memory_retry(
    train_ds: Dataset,
    val_ds: Dataset,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    settings: ResUNetSettings,
    requested_batch_size: int,
    device: torch.device,
    seed: int,
    loss_config: ResUNetLossConfig,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame, int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    batch_size = requested_batch_size
    while batch_size >= 1:
        seed_everything(seed)
        model = make_resunet(model_config)
        try:
            trained, best_state, history = train_one_attempt(
                model,
                train_ds,
                val_ds,
                settings.num_epochs,
                batch_size,
                float(training_config["learning_rate"]),
                float(training_config.get("weight_decay", 0.0)),
                device,
                seed,
                loss_config,
            )
            attempts.append({"batch_size": batch_size, "status": "completed"})
            return trained, best_state, history, batch_size, attempts
        except RuntimeError as exc:
            if not is_memory_error(exc) or batch_size == 1:
                attempts.append(
                    {"batch_size": batch_size, "status": "failed", "error": str(exc)}
                )
                if isinstance(exc, TrainingStopError):
                    exc.attempts = list(attempts)
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
    settings: ResUNetSettings,
    loss_config: ResUNetLossConfig,
    history: pd.DataFrame,
    batch_size: int,
    stamp: str,
    kind: str,
    experiment_title: str,
    variant_name: str,
    parameter_count: int,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    payload = {
        "model_state_dict": model_state,
        "experiment_name": experiment_title,
        "variant_name": variant_name,
        "architecture": "small residual-block U-Net",
        "architecture_class": "src.models.ResUNet",
        "model_parameter_count": parameter_count,
        "baseline_unet_parameter_count": model_parameter_count(
            gd_models.UNet(**config["model"])
        ),
        "experiment_settings": asdict(settings),
        "loss_config": asdict(loss_config),
        "config": config,
        "checkpoint_kind": kind,
        "timestamp": stamp,
        "residual_target": "blended_minus_target",
        "reconstruction": "blended_minus_predicted_residual",
        "output_activation": "identity",
        "batch_size": batch_size,
        "final_train_loss": float(history["train_loss"].iloc[-1]),
        "final_val_loss": float(history["val_loss"].iloc[-1]),
        "best_epoch": int(history["best_epoch"].iloc[-1]),
        "best_val_loss": float(history["best_val_loss"].iloc[-1]),
    }
    torch.save(payload, path)


def discover_delta_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = list(checkpoint_dir.glob("unet_br_v03_delta_candidate_*_best.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_models(
    config: dict[str, Any],
    device: torch.device,
    direct_path: Path,
    residual_path: Path,
    br_v01_path: Path,
    br_v02_path: Path,
    delta_path: Path | None,
    baseline_resunet_path: Path | None,
    candidate_state: dict[str, torch.Tensor],
) -> dict[str, nn.Module]:
    required = {
        "direct": direct_path,
        "residual": residual_path,
        "br_v01": br_v01_path,
        "br_v02_moderate": br_v02_path,
    }
    missing = [
        f"{name}: {project_relative(path)}"
        for name, path in required.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing comparison checkpoints: " + "; ".join(missing))
    loaded: dict[str, nn.Module] = {
        "direct": balanced_helpers.load_direct_model(
            direct_path, config["model"], device
        ),
        "residual": balanced_helpers.load_residual_model(
            residual_path, config["model"], device
        ),
        "br_v01": balanced_helpers.load_residual_model(
            br_v01_path, config["model"], device
        ),
        "br_v02_moderate": balanced_helpers.load_residual_model(
            br_v02_path, config["model"], device
        ),
    }
    if delta_path is not None:
        if not delta_path.exists():
            raise FileNotFoundError(
                f"Requested Delta checkpoint does not exist: {delta_path}"
            )
        loaded["br_v03_delta"] = balanced_helpers.load_residual_model(
            delta_path, config["model"], device
        )
    if baseline_resunet_path is not None:
        if not baseline_resunet_path.exists():
            raise FileNotFoundError(
                "Requested baseline ResUNet checkpoint does not exist: "
                f"{baseline_resunet_path}"
            )
        try:
            checkpoint = torch.load(
                baseline_resunet_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(baseline_resunet_path, map_location="cpu")
        baseline_resunet = make_resunet(config["model"])
        baseline_resunet.load_state_dict(
            balanced_helpers.checkpoint_state_dict(checkpoint)
        )
        baseline_resunet.to(device)
        baseline_resunet.eval()
        loaded["resunet_v04_baseline"] = baseline_resunet
    candidate = make_resunet(config["model"])
    candidate.load_state_dict(candidate_state)
    candidate.to(device)
    candidate.eval()
    loaded["resunet_v04"] = candidate
    return loaded


def active_methods(loaded_models: dict[str, nn.Module]) -> list[str]:
    return [
        method
        for method in METHOD_ORDER
        if method in {"identity", "threshold"} or method in loaded_models
    ]


def predict_methods(
    samples: list[dict[str, Any]],
    loaded_models: dict[str, nn.Module],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, float]]:
    methods = active_methods(loaded_models)
    predictions: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    candidate_stats = balanced_helpers.empty_residual_stats()
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        inputs = np.stack([sample["blended"] for sample in batch_samples], axis=0)
        tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
        outputs: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for method, model in loaded_models.items():
                outputs[method] = (
                    model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
                )
        candidate_layer = outputs["resunet_v04"]
        candidate_preclip = inputs - candidate_layer
        balanced_helpers.update_residual_stats(
            candidate_stats, candidate_layer, candidate_preclip
        )
        for offset, sample in enumerate(batch_samples):
            blended = sample["blended"]
            predictions["identity"].append(blended)
            predictions["threshold"].append(
                baselines.threshold_baseline(blended)
            )
            for method in loaded_models:
                if method == "direct":
                    reconstruction = np.clip(outputs[method][offset], 0.0, 1.0)
                else:
                    reconstruction = np.clip(
                        inputs[offset] - outputs[method][offset], 0.0, 1.0
                    )
                predictions[method].append(reconstruction.astype(np.float32))
    return predictions, balanced_helpers.finalise_residual_stats(candidate_stats)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def compute_per_sample(
    suite: str,
    samples: list[dict[str, Any]],
    predictions: dict[str, list[np.ndarray]],
    threshold: float,
) -> pd.DataFrame:
    methods = list(predictions)
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        target = sample["target"]
        blended = sample["blended"]
        affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
        core = v03_helpers.target_core_mask_np(target)
        core_affected = affected & core
        noncore_affected = affected & ~core
        halo = v03_helpers.halo_band_mask(affected)
        info = sample.get("info", {})
        metadata = balanced_helpers.blend_metadata(target, blended, info, threshold)
        shift = info.get("shift", (0, 0))
        row: dict[str, Any] = {
            "suite": suite,
            "index": idx,
            "generation_difficulty": info.get(
                "generation_difficulty", info.get("difficulty")
            ),
            "generation_constraints_met": info.get("generation_constraints_met"),
            "generation_relaxed": bool(info.get("generation_relaxed", False)),
            "training_component": info.get("training_component"),
            "target_index": info.get("target_index"),
            "contaminant_index": info.get("contaminant_index"),
            "shift_x": int(shift[0]),
            "shift_y": int(shift[1]),
            "shift_distance": abs(int(shift[0])) + abs(int(shift[1])),
            "brightness": info.get("brightness"),
            "blur_sigma": info.get("blur_sigma"),
            "noise_std": info.get("noise_std"),
            "target_radius": info.get("target_radius"),
            "contaminant_radius": info.get("contaminant_radius"),
            "size_ratio": info.get("size_ratio"),
            "mask_fraction": metadata["mask_fraction"],
            "core_obstruction_fraction": metadata["core_obstruction_fraction"],
            "blend_severity_score": metadata["blend_severity_score"],
            "core_affected_fraction": float(core_affected.mean()),
            "halo_band_fraction": float(halo.mean()),
        }
        identity_affected = float("nan")
        for method in methods:
            metrics = v03_helpers.image_metrics(
                predictions[method][idx],
                target,
                affected,
                core_affected,
                noncore_affected,
                halo,
            )
            if method == "identity":
                identity_affected = metrics["affected_mse"]
            for metric, value in metrics.items():
                row[f"{method}_{metric}"] = value
            row[f"{method}_improvement_ratio"] = safe_ratio(
                identity_affected, metrics["affected_mse"]
            )
            row[f"{method}_worse_than_identity"] = bool(
                metrics["affected_mse"] > identity_affected
            )
        row["resunet_beats_v02_moderate"] = (
            row["resunet_v04_affected_mse"]
            < row["br_v02_moderate_affected_mse"]
        )
        row["resunet_to_v02_affected_mse_ratio"] = safe_ratio(
            row["resunet_v04_affected_mse"],
            row["br_v02_moderate_affected_mse"],
        )
        row["resunet_beats_direct"] = (
            row["resunet_v04_affected_mse"] < row["direct_affected_mse"]
        )
        row["resunet_to_direct_affected_mse_ratio"] = safe_ratio(
            row["resunet_v04_affected_mse"], row["direct_affected_mse"]
        )
        if "br_v03_delta" in methods:
            row["resunet_beats_v03_delta"] = (
                row["resunet_v04_affected_mse"]
                < row["br_v03_delta_affected_mse"]
            )
            row["resunet_to_v03_delta_affected_mse_ratio"] = safe_ratio(
                row["resunet_v04_affected_mse"],
                row["br_v03_delta_affected_mse"],
            )
        if "resunet_v04_baseline" in methods:
            row["resunet_beats_baseline_resunet"] = (
                row["resunet_v04_affected_mse"]
                < row["resunet_v04_baseline_affected_mse"]
            )
            row["resunet_to_baseline_resunet_affected_mse_ratio"] = safe_ratio(
                row["resunet_v04_affected_mse"],
                row["resunet_v04_baseline_affected_mse"],
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["blend_severity_bin"] = balanced_helpers.blend_severity_bins(
        frame["blend_severity_score"]
    )
    frame["core_overlap_bin"] = balanced_helpers.core_overlap_bins(
        frame["core_obstruction_fraction"]
    )
    return frame


METRIC_SUFFIXES = (
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
    "affected_delta_e2000_mean",
    "affected_lab_chroma_mae",
    "affected_rgb_saturation_mae",
)


def aggregate_metrics(
    per_sample: pd.DataFrame, methods: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for suite, frame in per_sample.groupby("suite", sort=False):
        identity_affected = float(frame["identity_affected_mse"].mean())
        for method in methods:
            row: dict[str, Any] = {
                "suite": suite,
                "method": method,
                "method_label": method_label(method),
                "n": int(len(frame)),  # backward-compatible total count
                "n_total": int(len(frame)),
                "n_valid_affected": int(
                    frame[f"{method}_affected_mse"].notna().sum()
                ),
                "n_valid_core": int(
                    frame[f"{method}_core_affected_mse"].notna().sum()
                ),
                "n_valid_noncore": int(
                    frame[f"{method}_noncore_affected_mse"].notna().sum()
                ),
                "n_valid_halo": int(
                    frame[f"{method}_halo_band_mse"].notna().sum()
                ),
                "mean_mask_fraction": float(frame["mask_fraction"].mean()),
                "mean_core_obstruction_fraction": float(
                    frame["core_obstruction_fraction"].mean()
                ),
                "worse_than_identity_count": int(
                    frame[f"{method}_worse_than_identity"].sum()
                ),
                "worse_than_identity_fraction": float(
                    frame[f"{method}_worse_than_identity"].mean()
                ),
            }
            for suffix in METRIC_SUFFIXES:
                column = f"{method}_{suffix}"
                if column in frame:
                    row[suffix] = float(frame[column].mean())
            row["improvement_vs_identity"] = safe_ratio(
                identity_affected, row["affected_mse"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_summary(
    per_sample: pd.DataFrame, aggregate: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for suite, frame in per_sample.groupby("suite", sort=False):
        metrics = aggregate[aggregate["suite"] == suite].set_index("method")
        candidate = metrics.loc["resunet_v04"]
        v02 = metrics.loc["br_v02_moderate"]
        direct = metrics.loc["direct"]
        row = {
            "suite": suite,
            "n": int(len(frame)),
            "resunet_affected_mse": float(candidate["affected_mse"]),
            "v02_moderate_affected_mse": float(v02["affected_mse"]),
            "resunet_to_v02_affected_mse_ratio": safe_ratio(
                float(candidate["affected_mse"]), float(v02["affected_mse"])
            ),
            "resunet_core_affected_mse": float(candidate["core_affected_mse"]),
            "v02_moderate_core_affected_mse": float(v02["core_affected_mse"]),
            "resunet_to_v02_core_mse_ratio": safe_ratio(
                float(candidate["core_affected_mse"]),
                float(v02["core_affected_mse"]),
            ),
            "resunet_halo_band_mse": float(candidate["halo_band_mse"]),
            "v02_moderate_halo_band_mse": float(v02["halo_band_mse"]),
            "resunet_to_v02_halo_mse_ratio": safe_ratio(
                float(candidate["halo_band_mse"]), float(v02["halo_band_mse"])
            ),
            "resunet_worse_than_identity_count": int(
                candidate["worse_than_identity_count"]
            ),
            "v02_moderate_worse_than_identity_count": int(
                v02["worse_than_identity_count"]
            ),
            "resunet_vs_v02_win_rate": float(
                frame["resunet_beats_v02_moderate"].mean()
            ),
            "resunet_vs_direct_win_rate": float(
                frame["resunet_beats_direct"].mean()
            ),
            "direct_affected_mse": float(direct["affected_mse"]),
        }
        if "br_v03_delta" in metrics.index:
            delta = metrics.loc["br_v03_delta"]
            row["delta_affected_mse"] = float(delta["affected_mse"])
            row["resunet_vs_delta_win_rate"] = float(
                frame["resunet_beats_v03_delta"].mean()
            )
        if "resunet_v04_baseline" in metrics.index:
            baseline_resunet = metrics.loc["resunet_v04_baseline"]
            row["baseline_resunet_affected_mse"] = float(
                baseline_resunet["affected_mse"]
            )
            row["baseline_resunet_core_affected_mse"] = float(
                baseline_resunet["core_affected_mse"]
            )
            row["baseline_resunet_halo_band_mse"] = float(
                baseline_resunet["halo_band_mse"]
            )
            row["baseline_resunet_worse_than_identity_count"] = int(
                baseline_resunet["worse_than_identity_count"]
            )
            row["resunet_to_baseline_resunet_affected_mse_ratio"] = safe_ratio(
                float(candidate["affected_mse"]),
                float(baseline_resunet["affected_mse"]),
            )
            row["resunet_to_baseline_resunet_core_mse_ratio"] = safe_ratio(
                float(candidate["core_affected_mse"]),
                float(baseline_resunet["core_affected_mse"]),
            )
            row["resunet_to_baseline_resunet_halo_mse_ratio"] = safe_ratio(
                float(candidate["halo_band_mse"]),
                float(baseline_resunet["halo_band_mse"]),
            )
            row["resunet_vs_baseline_resunet_win_rate"] = float(
                frame["resunet_beats_baseline_resunet"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def decide_outcome(summary: pd.DataFrame) -> dict[str, Any]:
    keyed = summary.set_index("suite")
    required = {"normal", "hard_stress", "compact_bright", "high_core_obstruction", "halo_band"}
    missing = sorted(required - set(keyed.index))
    if missing:
        return {
            "verdict": "incomplete",
            "recommendation": "Required suites are missing; do not compare current-best status.",
            "missing_suites": missing,
            "clear_quantitative_success": False,
        }
    normal_ratio = float(keyed.loc["normal", "resunet_to_v02_affected_mse_ratio"])
    stress_ratio = float(
        keyed.loc["hard_stress", "resunet_to_v02_affected_mse_ratio"]
    )
    compact_ratio = float(
        keyed.loc["compact_bright", "resunet_to_v02_affected_mse_ratio"]
    )
    core_ratio = float(
        keyed.loc["high_core_obstruction", "resunet_to_v02_core_mse_ratio"]
    )
    halo_ratio = float(keyed.loc["halo_band", "resunet_to_v02_halo_mse_ratio"])
    protected_suites = (
        "normal",
        "hard_stress",
        "compact_bright",
        "high_core_obstruction",
        "halo_band",
    )
    worse_ok = all(
        int(keyed.loc[suite, "resunet_worse_than_identity_count"])
        <= int(keyed.loc[suite, "v02_moderate_worse_than_identity_count"])
        for suite in protected_suites
    )
    clear = bool(
        normal_ratio <= 1.0
        and stress_ratio <= 1.0
        and compact_ratio < 1.0
        and core_ratio < 1.0
        and halo_ratio <= 1.10
        and worse_ok
    )
    promising = bool(
        normal_ratio <= 1.02
        and stress_ratio <= 1.0
        and (compact_ratio < 1.0 or core_ratio < 1.0)
        and halo_ratio <= 1.10
        and worse_ok
    )
    baseline_available = "baseline_resunet_affected_mse" in keyed.columns
    tuning_details: dict[str, Any] = {}
    tuning_pass = True
    if baseline_available:
        normal_vs_baseline = float(
            keyed.loc[
                "normal", "resunet_to_baseline_resunet_affected_mse_ratio"
            ]
        )
        stress_vs_baseline = float(
            keyed.loc[
                "hard_stress", "resunet_to_baseline_resunet_affected_mse_ratio"
            ]
        )
        stress_core_vs_baseline = float(
            keyed.loc[
                "hard_stress", "resunet_to_baseline_resunet_core_mse_ratio"
            ]
        )
        halo_vs_baseline = float(
            keyed.loc["halo_band", "resunet_to_baseline_resunet_halo_mse_ratio"]
        )
        worse_vs_baseline_ok = all(
            int(keyed.loc[suite, "resunet_worse_than_identity_count"])
            <= int(
                keyed.loc[
                    suite, "baseline_resunet_worse_than_identity_count"
                ]
            )
            for suite in protected_suites
        )
        tuning_pass = bool(
            (stress_vs_baseline < 1.0 or stress_core_vs_baseline < 1.0)
            and normal_vs_baseline <= 1.02
            and halo_vs_baseline <= 1.10
            and worse_vs_baseline_ok
        )
        tuning_details = {
            "baseline_resunet_comparison_available": True,
            "normal_vs_baseline_resunet_affected_mse_ratio": normal_vs_baseline,
            "stress_vs_baseline_resunet_affected_mse_ratio": stress_vs_baseline,
            "stress_core_vs_baseline_resunet_core_mse_ratio": stress_core_vs_baseline,
            "halo_vs_baseline_resunet_halo_mse_ratio": halo_vs_baseline,
            "worse_than_identity_vs_baseline_gate": worse_vs_baseline_ok,
            "controlled_tuning_gate": tuning_pass,
        }
    if baseline_available and not tuning_pass:
        verdict = "loss_tuning_ablation"
        recommendation = (
            "This tuned ResUNet does not improve stress affected/core error over the "
            "baseline architecture without unacceptable normal, halo, or failure-count "
            "tradeoffs. Keep the baseline ResUNet or v0.2 Moderate, whichever is better."
        )
        clear = False
        promising = False
    elif clear:
        verdict = "new_best_candidate_pending_qualitative_review"
        recommendation = (
            "ResUNet clears the same-run quantitative gate. Inspect the saved grids before "
            "calling it current best; repeat-seed evidence would strengthen promotion."
        )
    elif promising:
        verdict = "promising_architecture_tradeoff"
        recommendation = (
            "ResUNet is close on normal data and improves stress plus at least one targeted "
            "slice, but it does not yet clearly replace v0.2 Moderate."
        )
    else:
        verdict = "architecture_ablation"
        recommendation = (
            "ResUNet does not clear the controlled same-run criteria. Keep v0.2 Moderate "
            "as current best and document this run as an architecture ablation."
        )
    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "clear_quantitative_success": clear,
        "promising": promising,
        "normal_affected_mse_ratio": normal_ratio,
        "stress_affected_mse_ratio": stress_ratio,
        "compact_affected_mse_ratio": compact_ratio,
        "high_core_core_mse_ratio": core_ratio,
        "halo_band_mse_ratio": halo_ratio,
        "worse_than_identity_gate": worse_ok,
        "qualitative_review_required_for_promotion": True,
        **tuning_details,
    }


def method_label(method: str) -> str:
    return {
        "identity": "identity",
        "threshold": "threshold",
        "direct": "Thayer-Direct",
        "residual": "Thayer-Residual",
        "br_v01": "Thayer-BR v0.1",
        "br_v02_moderate": "BR v0.2 Moderate",
        "br_v03_delta": "BR v0.3 Delta",
        "resunet_v04_baseline": "Baseline ResUNet v0.4",
        "resunet_v04": "ResUNet v0.4",
    }.get(method, method)


def method_color(method: str) -> str:
    return {
        "identity": "#6c717a",
        "threshold": "#b66b5d",
        "direct": "#2f6f8f",
        "residual": "#5f8a4b",
        "br_v01": "#7b5fa3",
        "br_v02_moderate": "#c28b2c",
        "br_v03_delta": "#2c8f7b",
        "resunet_v04_baseline": "#8356a3",
        "resunet_v04": "#be4b6f",
    }.get(method, "#444444")


def plot_metric_bars(
    aggregate: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
    suites: tuple[str, ...] | None = None,
    methods: list[str] | None = None,
) -> Path:
    suites = suites or tuple(aggregate["suite"].drop_duplicates())
    methods = methods or list(aggregate["method"].drop_duplicates())
    x = np.arange(len(suites))
    width = 0.82 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.3 * len(suites)), 4.7))
    for idx, method in enumerate(methods):
        values = []
        for suite in suites:
            selected = aggregate[
                (aggregate["suite"] == suite) & (aggregate["method"] == method)
            ]
            values.append(
                float(selected.iloc[0][metric]) if not selected.empty else float("nan")
            )
        ax.bar(
            x + (idx - (len(methods) - 1) / 2) * width,
            values,
            width,
            label=method_label(method),
            color=method_color(method),
        )
    ax.set_xticks(x, [suite.replace("_", " ") for suite in suites], rotation=18, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return save_figure(fig, path)


def scatter_resunet_v02(per_sample: pd.DataFrame, path: Path) -> Path:
    x_column = "br_v02_moderate_affected_mse"
    y_column = "resunet_v04_affected_mse"
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for suite, frame in per_sample.groupby("suite", sort=False):
        ax.scatter(
            frame[x_column],
            frame[y_column],
            s=10,
            alpha=0.25,
            label=suite.replace("_", " "),
            edgecolors="none",
        )
    values = np.concatenate(
        [per_sample[x_column].to_numpy(), per_sample[y_column].to_numpy()]
    )
    values = values[np.isfinite(values)]
    upper = float(np.quantile(values, 0.995)) if values.size else 1.0
    ax.plot([0.0, upper], [0.0, upper], "k--", linewidth=1)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel("BR v0.2 Moderate affected MSE")
    ax.set_ylabel("ResUNet v0.4 affected MSE")
    ax.set_title("ResUNet v0.4 vs BR v0.2 Moderate")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, path)


def ratio_histogram(per_sample: pd.DataFrame, path: Path) -> Path:
    column = "resunet_to_v02_affected_mse_ratio"
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for suite, frame in per_sample.groupby("suite", sort=False):
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        values = np.clip(values, 0.0, 3.0)
        ax.hist(values, bins=35, alpha=0.30, label=suite.replace("_", " "))
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("ResUNet affected MSE / BR v0.2 affected MSE (clipped at 3)")
    ax.set_ylabel("Samples")
    ax.set_title("Per-Sample ResUNet / BR v0.2 Error Ratio")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, path)


def write_paper_figures(
    run_dir: Path, aggregate: pd.DataFrame, per_sample: pd.DataFrame
) -> dict[str, str]:
    output = run_dir / "paper_figures"
    methods = list(aggregate["method"].drop_duplicates())
    comparison_methods = [
        method
        for method in (
            "identity",
            "direct",
            "residual",
            "br_v01",
            "br_v02_moderate",
            "br_v03_delta",
            "resunet_v04_baseline",
            "resunet_v04",
        )
        if method in methods
    ]
    improvement_methods = [
        method
        for method in (
            "direct",
            "residual",
            "br_v01",
            "br_v02_moderate",
            "br_v03_delta",
            "resunet_v04_baseline",
            "resunet_v04",
        )
        if method in methods
    ]
    written = {
        "affected_mse": project_relative(
            plot_metric_bars(
                aggregate,
                "affected_mse",
                output / "affected_region_mse_comparison.png",
                "Affected-Region MSE Comparison",
                methods=comparison_methods,
            )
        ),
        "normal_stress_improvement": project_relative(
            plot_metric_bars(
                aggregate,
                "improvement_vs_identity",
                output / "normal_vs_stress_improvement_ratio.png",
                "Normal vs Hard-Stress Improvement",
                suites=("normal", "hard_stress"),
                methods=improvement_methods,
            )
        ),
        "compact_contaminant": project_relative(
            plot_metric_bars(
                aggregate,
                "affected_mse",
                output / "compact_contaminant_affected_mse.png",
                "Compact Bright Contaminant Stress",
                suites=("compact_bright",),
                methods=comparison_methods,
            )
        ),
        "core_affected_mse": project_relative(
            plot_metric_bars(
                aggregate,
                "core_affected_mse",
                output / "core_affected_mse_comparison.png",
                "Core Affected MSE",
                methods=comparison_methods,
            )
        ),
        "halo_band_mse": project_relative(
            plot_metric_bars(
                aggregate,
                "halo_band_mse",
                output / "halo_band_mse_comparison.png",
                "Halo-Band MSE",
                methods=comparison_methods,
            )
        ),
        "resunet_v02_scatter": project_relative(
            scatter_resunet_v02(
                per_sample, output / "resunet_vs_v02_affected_mse_scatter.png"
            )
        ),
        "error_ratio_histogram": project_relative(
            ratio_histogram(
                per_sample, output / "hist_resunet_to_v02_affected_mse_ratio.png"
            )
        ),
    }
    return written


def update_example_bank(
    bank: dict[str, dict[str, Any]],
    suite: str,
    samples: list[dict[str, Any]],
    predictions: dict[str, list[np.ndarray]],
    per_sample: pd.DataFrame,
) -> None:
    def store(key: str, row: pd.Series, score: float) -> None:
        current = bank.get(key)
        if current is not None and score <= current["score"]:
            return
        idx = int(row["index"])
        bank[key] = {
            "score": float(score),
            "suite": suite,
            "row": row.to_dict(),
            "target": np.array(samples[idx]["target"], copy=True),
            "blended": np.array(samples[idx]["blended"], copy=True),
            "predictions": {
                method: np.array(values[idx], copy=True)
                for method, values in predictions.items()
                if method
                in {
                    "direct",
                    "br_v02_moderate",
                    "br_v03_delta",
                    "resunet_v04_baseline",
                    "resunet_v04",
                }
            },
        }

    improved = per_sample[per_sample["resunet_to_v02_affected_mse_ratio"] < 1.0]
    if not improved.empty:
        row = improved.sort_values("resunet_to_v02_affected_mse_ratio").iloc[0]
        store(
            "resunet_improves",
            row,
            1.0 / max(float(row["resunet_to_v02_affected_mse_ratio"]), 1e-8),
        )
    failures = per_sample[per_sample["resunet_to_v02_affected_mse_ratio"] > 1.0]
    if not failures.empty:
        row = failures.sort_values(
            "resunet_to_v02_affected_mse_ratio", ascending=False
        ).iloc[0]
        store(
            "resunet_failure",
            row,
            float(row["resunet_to_v02_affected_mse_ratio"]),
        )
    direct_wins = per_sample[~per_sample["resunet_beats_direct"]]
    if not direct_wins.empty:
        row = direct_wins.sort_values(
            "resunet_to_direct_affected_mse_ratio", ascending=False
        ).iloc[0]
        store(
            "direct_still_wins",
            row,
            float(row["resunet_to_direct_affected_mse_ratio"]),
        )


def save_example_grid(item: dict[str, Any], path: Path, title: str) -> Path:
    target = item["target"]
    panels: list[tuple[str, np.ndarray]] = [
        ("Blended", item["blended"]),
        ("Target", target),
    ]
    for method in (
        "direct",
        "br_v02_moderate",
        "br_v03_delta",
        "resunet_v04_baseline",
        "resunet_v04",
    ):
        if method in item["predictions"]:
            panels.append((method_label(method), item["predictions"][method]))
    ncols = len(panels)
    fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 6.1))
    for col, (label, image) in enumerate(panels):
        axes[0, col].imshow(np.clip(image, 0.0, 1.0))
        axes[0, col].set_title(label)
        axes[0, col].axis("off")
        error = np.mean(np.abs(image - target), axis=-1)
        axes[1, col].imshow(error, cmap="magma", vmin=0.0, vmax=0.20)
        axes[1, col].set_title("Mean absolute error")
        axes[1, col].axis("off")
    row = item["row"]
    fig.suptitle(
        f"{title} | {item['suite'].replace('_', ' ')} | "
        f"ResUNet/v0.2={row['resunet_to_v02_affected_mse_ratio']:.3f}"
    )
    fig.tight_layout()
    return save_figure(fig, path)


def write_example_grids(
    run_dir: Path, bank: dict[str, dict[str, Any]]
) -> dict[str, str | None]:
    specs = {
        "resunet_improves": (
            "resunet_improvement_over_v02.png",
            "ResUNet Improves Over BR v0.2",
        ),
        "resunet_failure": (
            "resunet_failure_tradeoff.png",
            "ResUNet Failure / Tradeoff",
        ),
        "direct_still_wins": (
            "direct_still_wins_over_resunet.png",
            "Thayer-Direct Still Wins",
        ),
    }
    written: dict[str, str | None] = {}
    for key, (filename, title) in specs.items():
        item = bank.get(key)
        if item is None:
            written[key] = None
            continue
        grid_path = run_dir / "example_grids" / filename
        paper_path = run_dir / "paper_figures" / filename
        save_example_grid(item, grid_path, title)
        save_example_grid(item, paper_path, title)
        written[key] = project_relative(grid_path)
    return written


def format_metrics_table(aggregate: pd.DataFrame, suite: str) -> str:
    frame = aggregate[aggregate["suite"] == suite]
    lines = [
        "| Method | Affected MSE | Core affected MSE | Halo MSE | SSIM | Improvement | Worse than identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['method_label']} | {row['affected_mse']:.6f} | "
            f"{row['core_affected_mse']:.6f} | {row['halo_band_mse']:.6f} | "
            f"{row['ssim']:.6f} | {row['improvement_vs_identity']:.2f}x | "
            f"{int(row['worse_than_identity_count'])}/{int(row['n'])} |"
        )
    return "\n".join(lines)


def write_report(
    run_dir: Path,
    report_name: str,
    experiment_title: str,
    variant_name: str,
    best_checkpoint: Path,
    final_checkpoint: Path,
    settings: ResUNetSettings,
    loss_config: ResUNetLossConfig,
    composition: dict[str, Any],
    history: pd.DataFrame,
    parameter_count: int,
    baseline_parameter_count: int,
    delta_path: Path | None,
    baseline_resunet_path: Path | None,
    aggregate: pd.DataFrame,
    outcome: dict[str, Any],
    figures: dict[str, str],
    grids: dict[str, str | None],
) -> None:
    increase = 100.0 * (parameter_count / baseline_parameter_count - 1.0)
    figure_lines = [f"- `{path}`" for path in figures.values()]
    grid_lines = [
        f"- {key}: `{path}`" if path else f"- {key}: no qualifying case found"
        for key, path in grids.items()
    ]
    delta_text = (
        f"Loaded and evaluated `{project_relative(delta_path)}` on the same suites."
        if delta_path is not None
        else "No completed Delta checkpoint was found, so Delta was omitted from same-run comparisons."
    )
    baseline_resunet_text = (
        "Loaded and evaluated baseline ResUNet checkpoint "
        f"`{project_relative(baseline_resunet_path)}` on the same suites."
        if baseline_resunet_path is not None
        else "This is the baseline architecture run; no earlier ResUNet checkpoint was required."
    )
    text = f"""# {experiment_title} Report

## Purpose and Safety

This private architecture ablation tests whether residual blocks inside the
compact U-Net improve compact-contaminant and ambiguous core-overlap
reconstruction. It keeps the residual prediction task and does not use GANs,
attention, or the v0.3 color-heavy objective. All artifacts use a new
timestamped run directory and checkpoints.

## Architecture

- Variant: `{variant_name}`.
- Model: small residual-block U-Net with encoder/decoder skip connections.
- Input/output: RGB blended image to raw RGB predicted residual.
- Reconstruction: `blended - predicted_residual`, clipped only for evaluation.
- Base-channel width: unchanged from the v0.2 project model config.
- ResUNet trainable parameters: `{parameter_count:,}`.
- Standard v0.2 U-Net parameters: `{baseline_parameter_count:,}`.
- Parameter increase: `{increase:.2f}%`.
- Best checkpoint: `{project_relative(best_checkpoint)}`.
- Final checkpoint: `{project_relative(final_checkpoint)}`.

## Training

- Train/validation blends: {settings.n_train_blends:,} / {settings.n_val_blends:,}.
- Epochs: {settings.num_epochs}; completed in full unless this report is absent and a failure diagnostic exists.
- Used batch size: {int(history['batch_size'].iloc[-1])}.
- Fixed seed offset: {settings.seed_offset}.
- Target distribution: 50% normal, 30% high-overlap/core-obstruction, 20% brightness/size stress.
- Actual training components: `{json.dumps(composition['train']['components'])}`.
- Loss: v0.2 normalized affected/core-weighted residual MSE.
- Additive weights: background `{loss_config.background_weight}`, affected extra `{loss_config.affected_extra_weight}`, affected-core extra `{loss_config.core_extra_weight}`.
- Effective default spatial weights: background `{loss_config.background_weight}`, affected `{loss_config.background_weight + loss_config.affected_extra_weight}`, core-affected `{loss_config.background_weight + loss_config.affected_extra_weight + loss_config.core_extra_weight}`.
- Halo-band weight: `{loss_config.halo_weight}`; when nonzero, it is a channel-normalized MSE over a {loss_config.halo_dilation_iters}-pixel affected-region ring.
- No color, Lab, Delta E, gradient, reconstruction-L1, GAN, or attention loss was used.
- Best validation loss: {float(history['best_val_loss'].iloc[-1]):.6f} at epoch {int(history['best_epoch'].iloc[-1])}.
- Final train/validation loss: {float(history['train_loss'].iloc[-1]):.6f} / {float(history['val_loss'].iloc[-1]):.6f}.
- Final validation affected MSE: {float(history['val_affected_mse'].iloc[-1]):.6f}.

## Comparison Checkpoints

{delta_text}

{baseline_resunet_text}

All listed models were evaluated on the same generated samples within each
suite. Clean-source filtering was not run because the source dataset has no
validated artifact-quality flags; this absence is documented rather than
silently approximated.

## Normal Held-Out

{format_metrics_table(aggregate, 'normal')}

## Hard Stress

{format_metrics_table(aggregate, 'hard_stress')}

## Compact Bright Contaminants

{format_metrics_table(aggregate, 'compact_bright')}

## High Core Obstruction

{format_metrics_table(aggregate, 'high_core_obstruction')}

## Halo-Band Stress

{format_metrics_table(aggregate, 'halo_band')}

## Outcome

Verdict: `{outcome['verdict']}`.

{outcome['recommendation']}

Quantitative gate details: `{json.dumps(outcome, allow_nan=True)}`.

This automated report never promotes the model solely from metrics. A
current-best claim requires reviewing the qualitative grids and confirming no
implausible color, edge, core, or broad halo artifacts. If the gate fails, the
run remains an architecture ablation and BR v0.2 Moderate remains current best.

## Figures

{chr(10).join(figure_lines)}

## Qualitative Grids

{chr(10).join(grid_lines)}
"""
    safe_write_text(run_dir / "diagnostics" / report_name, text)


def write_failure_report(run_dir: Path, exc: BaseException) -> None:
    path = run_dir / "diagnostics/resunet_v04_failure_report.md"
    if path.exists():
        return
    safe_write_text(
        path,
        "# ResUNet v0.4 Failure Diagnostic\n\n"
        f"The sub-experiment stopped without deleting or reusing its run directory.\n\n"
        f"- Exception type: `{type(exc).__name__}`\n"
        f"- Message: `{str(exc)}`\n\n"
        "Inspect training history, generation logs, and checkpoint integrity files before any retry.",
    )


def run_experiment(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    best_checkpoint: Path,
    final_checkpoint: Path,
    output_root: Path,
    stamp: str,
) -> int:
    if Path(args.report_name).name != args.report_name or not args.report_name:
        raise ValueError("report-name must be a non-empty filename, not a path.")
    settings = ResUNetSettings(
        n_train_blends=args.n_train_blends,
        n_val_blends=args.n_val_blends,
        n_normal_test_blends=args.n_normal_test_blends,
        n_suite_blends=args.n_suite_blends,
        train_source_subset=args.train_source_subset,
        val_source_subset=args.val_source_subset,
        test_source_subset=args.test_source_subset,
        num_epochs=args.num_epochs,
        affected_region_threshold=args.affected_threshold,
        seed_offset=args.seed_offset,
    )
    loss_config = ResUNetLossConfig(
        background_weight=args.background_weight,
        affected_extra_weight=args.affected_extra_weight,
        core_extra_weight=args.core_extra_weight,
        affected_threshold=args.affected_threshold,
        halo_weight=args.halo_weight,
        halo_dilation_iters=args.halo_dilation_iters,
    )
    count_values = {
        "n_train_blends": settings.n_train_blends,
        "n_val_blends": settings.n_val_blends,
        "n_normal_test_blends": settings.n_normal_test_blends,
        "n_suite_blends": settings.n_suite_blends,
        "train_source_subset": settings.train_source_subset,
        "val_source_subset": settings.val_source_subset,
        "test_source_subset": settings.test_source_subset,
    }
    nonpositive = [name for name, value in count_values.items() if value <= 0]
    if nonpositive:
        raise ValueError(
            "Blend counts and source subset sizes must be positive: "
            + ", ".join(nonpositive)
        )
    if settings.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if min(
        loss_config.background_weight,
        loss_config.affected_extra_weight,
        loss_config.core_extra_weight,
        loss_config.halo_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")
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
            run_dir / "logs/device_selection.json",
            {**device_status, "status": "stopped", "error": str(exc)},
        )
        print(f"Stopping before full training/evaluation: {exc}", flush=True)
        return 2
    safe_json(
        run_dir / "logs/device_selection.json",
        {**device_status, "status": "selected", "selected_device": str(device)},
    )
    print(f"Using device: {device}", flush=True)
    print(f"Run directory: {project_relative(run_dir)}", flush=True)

    direct_path = resolve_path(args.direct_checkpoint)
    residual_path = resolve_path(args.residual_checkpoint)
    br_v01_path = resolve_path(args.br_v01_checkpoint)
    br_v02_path = resolve_path(args.br_v02_moderate_checkpoint)
    delta_path = (
        resolve_path(args.delta_checkpoint)
        if args.delta_checkpoint is not None
        else discover_delta_checkpoint(output_root / "checkpoints")
    )
    baseline_resunet_path = (
        resolve_path(args.baseline_resunet_checkpoint)
        if args.baseline_resunet_checkpoint is not None
        else None
    )
    prototype = make_resunet(config["model"])
    parameter_count = model_parameter_count(prototype)
    baseline_parameter_count = model_parameter_count(
        gd_models.UNet(**config["model"])
    )
    if parameter_count > baseline_parameter_count * 1.10:
        raise RuntimeError(
            "ResUNet parameter count exceeds the controlled 10% architecture budget: "
            f"{parameter_count} vs {baseline_parameter_count}."
        )
    del prototype
    safe_yaml(run_dir / "logs/loss_config.yaml", asdict(loss_config))
    safe_yaml(
        run_dir / "logs/run_config.yaml",
        {
            "project_root": ".",
            "timestamp": stamp,
            "experiment_title": args.experiment_title,
            "variant_name": args.variant_name,
            "settings": asdict(settings),
            "loss_config": asdict(loss_config),
            "config": config,
            "architecture": {
                "class": "src.models.ResUNet",
                "parameter_count": parameter_count,
                "baseline_unet_parameter_count": baseline_parameter_count,
                "parameter_increase_fraction": parameter_count
                / baseline_parameter_count
                - 1.0,
            },
            "checkpoints": {
                "direct": project_relative(direct_path),
                "residual": project_relative(residual_path),
                "br_v01": project_relative(br_v01_path),
                "br_v02_moderate": project_relative(br_v02_path),
                "br_v03_delta": project_relative(delta_path)
                if delta_path is not None
                else None,
                "baseline_resunet": project_relative(baseline_resunet_path)
                if baseline_resunet_path is not None
                else None,
                "candidate_best": project_relative(best_checkpoint),
                "candidate_final": project_relative(final_checkpoint),
            },
        },
    )
    safe_write_text(
        run_dir / "diagnostics/clean_source_filtered_normal_unavailable.md",
        "# Clean-Source Filtered Normal Suite\n\n"
        "This suite was not generated because Galaxy10 DECaLS provides no validated "
        "source-artifact quality flags in the local HDF5 metadata. No heuristic filter "
        "was silently substituted. Normal, hard-stress, compact-bright, high-core, "
        "halo-band, and color/saturation suites are evaluated instead.",
    )

    print("Loading fixed dataset splits.", flush=True)
    train_images, val_images, test_images = balanced_helpers.load_split_subsets(
        config, settings
    )
    print("Generating 50/30/20 balanced training blends.", flush=True)
    train_blends, train_composition = balanced_helpers.generate_balanced_blends(
        train_images,
        total=settings.n_train_blends,
        config=config,
        seed=seed + 1000,
        settings=settings,
        split_name="train",
    )
    print("Generating balanced validation blends.", flush=True)
    val_blends, val_composition = balanced_helpers.generate_balanced_blends(
        val_images,
        total=settings.n_val_blends,
        config=config,
        seed=seed + 2000,
        settings=settings,
        split_name="val",
    )
    v03_helpers.strip_contaminants(train_blends)
    v03_helpers.strip_contaminants(val_blends)
    composition = {"train": train_composition, "validation": val_composition}
    safe_json(run_dir / "logs/training_composition.json", composition)
    component_rows = [
        {"split": split, **component}
        for split, payload in composition.items()
        for component in payload["components"]
    ]
    safe_csv(
        run_dir / "tables/training_composition.csv", pd.DataFrame(component_rows)
    )
    for row in component_rows:
        if row.get("mean_mask_fraction", 1.0) < 0.005:
            raise RuntimeError(
                f"Suspiciously low mean mask fraction in {row['component']}: "
                f"{row['mean_mask_fraction']:.6f}."
            )
    del train_images, val_images
    gc.collect()

    train_ds = weighted_helpers.WeightedResidualBlendDataset(train_blends)
    val_ds = weighted_helpers.WeightedResidualBlendDataset(val_blends)
    requested_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["training"].get("batch_size", 8))
    )
    if requested_batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    print("Training residual-block U-Net with v0.2 Moderate loss.", flush=True)
    try:
        trained_model, best_state, history, used_batch_size, attempts = (
            train_with_memory_retry(
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
        )
    except TrainingStopError as exc:
        safe_csv(
            run_dir / "tables/training_history_partial.csv", exc.history
        )
        safe_json(run_dir / "logs/training_attempts.json", exc.attempts)
        raise
    safe_csv(run_dir / "tables/training_history.csv", history)
    safe_json(run_dir / "logs/training_attempts.json", attempts)
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
        args.variant_name,
        parameter_count,
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
        args.variant_name,
        parameter_count,
    )
    print(f"Saved best checkpoint: {project_relative(best_checkpoint)}", flush=True)
    print(f"Saved final checkpoint: {project_relative(final_checkpoint)}", flush=True)
    del trained_model, final_state, train_ds, val_ds, train_blends, val_blends
    clear_torch_cache()
    gc.collect()

    loaded_models = load_models(
        config,
        device,
        direct_path,
        residual_path,
        br_v01_path,
        br_v02_path,
        delta_path,
        baseline_resunet_path,
        best_state,
    )
    methods = active_methods(loaded_models)
    eval_settings = v03_helpers.V03Settings(
        n_train_blends=settings.n_train_blends,
        n_val_blends=settings.n_val_blends,
        n_normal_test_blends=settings.n_normal_test_blends,
        n_suite_blends=settings.n_suite_blends,
        train_source_subset=settings.train_source_subset,
        val_source_subset=settings.val_source_subset,
        test_source_subset=settings.test_source_subset,
        num_epochs=settings.num_epochs,
        affected_region_threshold=settings.affected_region_threshold,
    )
    all_per_sample: list[pd.DataFrame] = []
    suite_diagnostics: list[dict[str, Any]] = []
    evaluation_warnings: list[str] = []
    example_bank: dict[str, dict[str, Any]] = {}
    for suite_index, suite in enumerate(SUITES):
        print(f"Generating evaluation suite: {suite}.", flush=True)
        samples, threshold, suite_diag = v03_helpers.generate_suite(
            suite,
            test_images,
            config,
            eval_settings,
            seed + 4000 + suite_index * 137,
        )
        suite_diagnostics.append(suite_diag)
        predictions, candidate_stats = predict_methods(
            samples, loaded_models, device, used_batch_size
        )
        safe_json(
            run_dir / "diagnostics" / f"{suite}_resunet_output_stats.json",
            candidate_stats,
        )
        per_sample = compute_per_sample(suite, samples, predictions, threshold)
        safe_csv(
            run_dir / "tables" / f"{suite}_per_sample_metrics.csv", per_sample
        )
        update_example_bank(
            example_bank, suite, samples, predictions, per_sample
        )
        all_per_sample.append(per_sample)
        if candidate_stats["residual_pred_abs_max"] > 5.0:
            raise RuntimeError(
                f"{suite}: ResUNet predicted residual magnitude exceeded 5."
            )
        if candidate_stats["reconstruction_preclip_total_clip_fraction"] > 0.20:
            raise RuntimeError(
                f"{suite}: more than 20% of ResUNet reconstruction pixels needed clipping."
            )
        if float(per_sample["resunet_v04_affected_mse"].mean()) > float(
            per_sample["identity_affected_mse"].mean()
        ):
            evaluation_warnings.append(
                f"{suite}: ResUNet aggregate affected MSE is worse than identity."
            )
        del samples, predictions, per_sample
        clear_torch_cache()
        gc.collect()
    safe_json(run_dir / "logs/evaluation_suite_generation.json", suite_diagnostics)
    safe_json(run_dir / "logs/evaluation_warnings.json", evaluation_warnings)
    if evaluation_warnings:
        safe_write_text(
            run_dir / "diagnostics/evaluation_warnings.md",
            "# ResUNet Evaluation Warnings\n\n"
            + "\n".join(f"- {warning}" for warning in evaluation_warnings)
            + "\n\nThese are negative experimental results, not reasons to hide or delete the run.",
        )
    per_sample_all = pd.concat(all_per_sample, ignore_index=True)
    aggregate = aggregate_metrics(per_sample_all, methods)
    summary = comparison_summary(per_sample_all, aggregate)
    outcome = decide_outcome(summary)
    safe_csv(
        run_dir / "tables/resunet_v04_per_sample_metrics.csv", per_sample_all
    )
    safe_csv(run_dir / "tables/resunet_v04_suite_metrics.csv", aggregate)
    safe_csv(
        run_dir / "tables/resunet_v04_comparison_summary.csv", summary
    )
    safe_json(run_dir / "logs/resunet_v04_outcome.json", outcome)
    figures = write_paper_figures(run_dir, aggregate, per_sample_all)
    grids = write_example_grids(run_dir, example_bank)
    safe_json(run_dir / "logs/paper_figures.json", figures)
    safe_json(run_dir / "logs/example_grids.json", grids)
    write_report(
        run_dir,
        args.report_name,
        args.experiment_title,
        args.variant_name,
        best_checkpoint,
        final_checkpoint,
        settings,
        loss_config,
        composition,
        history,
        parameter_count,
        baseline_parameter_count,
        delta_path,
        baseline_resunet_path,
        aggregate,
        outcome,
        figures,
        grids,
    )
    print(f"Outcome: {outcome['verdict']} - {outcome['recommendation']}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(stamp).name != stamp or not stamp:
        raise ValueError("stamp must be a non-empty filename component.")
    output_root = PROJECT_ROOT / config.get("output_dir", "outputs")
    try:
        output_root.resolve().relative_to((PROJECT_ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("Configured output_dir must stay under the ignored outputs/ tree.") from exc
    run_dir, best_checkpoint, final_checkpoint = make_run_paths(
        output_root,
        stamp,
        args.run_dir,
        args.run_name_prefix,
        args.checkpoint_name_prefix,
    )
    safe_json(
        run_dir / "logs/checkpoint_integrity_before.json",
        checkpoint_snapshot(output_root / "checkpoints"),
    )
    error: BaseException | None = None
    result = 1
    try:
        result = run_experiment(
            args,
            config,
            run_dir,
            best_checkpoint,
            final_checkpoint,
            output_root,
            stamp,
        )
    except BaseException as exc:
        error = exc
        write_failure_report(run_dir, exc)
    integrity = write_checkpoint_integrity_after(
        run_dir, output_root / "checkpoints"
    )
    if not integrity["old_checkpoints_unchanged"]:
        raise RuntimeError("Old checkpoint integrity verification failed.")
    if error is not None:
        raise error
    print("Old checkpoints unchanged.", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
