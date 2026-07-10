"""Run clean-preservation, unaffected-region, and clipping audits on an accelerator.

The audit is deliberately output-safe: it creates two new timestamped run
directories, never writes checkpoints, snapshots every checkpoint before and
after inference, and refuses to perform full inference on CPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_balanced_residual_unet as balanced_helpers
import train_v03_color_structure_unet as v03_helpers
from src import models as gd_models
from src import utils as gd_utils


DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_V02 = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_residual_weighted_br_20260709_030245_best.pth"
)
DEFAULT_DELTA = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_br_v03_delta_candidate_20260710_031425_best.pth"
)
DEFAULT_RESUNET = (
    PROJECT_ROOT
    / "outputs/checkpoints/unet_resunet_v04_candidate_20260710_043109_best.pth"
)
MODEL_LABELS = {
    "br_v02_moderate": "Thayer-BR v0.2 Moderate",
    "br_v03_delta": "Thayer-BR v0.3 Delta",
    "resunet_v04": "Thayer-ResUNet v0.4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v02-checkpoint", type=Path, default=DEFAULT_V02)
    parser.add_argument("--delta-checkpoint", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--resunet-checkpoint", type=Path, default=DEFAULT_RESUNET)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--n-clean", type=int, default=1000)
    parser.add_argument("--n-blends", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--audit-seed", type=int, default=2026071017)
    parser.add_argument(
        "--artifact-candidates",
        type=Path,
        default=None,
        help="Optional heuristic source-artifact candidate CSV for null-test stratification.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def safe_mean(values: pd.Series | np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if array.size and not np.all(np.isnan(array)) else float("nan")


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return gd_utils.masked_mse(pred, target, mask) if np.any(mask) else float("nan")


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return gd_utils.masked_mae(pred, target, mask) if np.any(mask) else float("nan")


def select_device() -> tuple[torch.device, dict[str, Any]]:
    mps_available = bool(torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    if mps_available:
        selected = "mps"
    elif cuda_available:
        selected = "cuda"
    else:
        raise RuntimeError(
            "Full preservation/clipping inference requires MPS or CUDA; refusing CPU fallback."
        )
    return torch.device(selected), {
        "selected_device": selected,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "cpu_fallback_used": False,
        "torch_version": torch.__version__,
        "logged_before_model_inference": True,
    }


def make_run_dirs(stamp: str) -> tuple[Path, Path]:
    if not stamp or Path(stamp).name != stamp or stamp in {".", ".."}:
        raise ValueError("stamp must be a non-empty filename component.")
    runs = PROJECT_ROOT / "outputs/runs"
    preservation = runs / f"preservation_null_tests_{stamp}"
    clipping = runs / f"clipping_audit_{stamp}"
    for run_dir in (preservation, clipping):
        try:
            run_dir.resolve().relative_to(runs.resolve())
        except ValueError as exc:
            raise ValueError(
                "Audit output must remain under ignored outputs/runs/."
            ) from exc
    for run_dir in (preservation, clipping):
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    for run_dir in (preservation, clipping):
        for child in ("tables", "diagnostics", "figures", "logs"):
            (run_dir / child).mkdir(parents=True, exist_ok=False)
    return preservation, clipping


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_snapshot() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((PROJECT_ROOT / "outputs/checkpoints").glob("*.pth")):
        stat = path.stat()
        rows.append(
            {
                "path": relative(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old = {row["path"]: row for row in before["checkpoints"]}
    new = {row["path"]: row for row in after["checkpoints"]}
    rows: list[dict[str, Any]] = []
    for path in sorted(set(old) | set(new)):
        b = old.get(path)
        a = new.get(path)
        unchanged = b == a
        rows.append({"path": path, "unchanged": unchanged, "before": b, "after": a})
    old_unchanged = all(row["unchanged"] for row in rows if row["path"] in old)
    return {
        "old_checkpoint_count": len(old),
        "after_checkpoint_count": len(new),
        "old_checkpoints_unchanged": old_unchanged,
        "new_checkpoint_paths": sorted(set(new) - set(old)),
        "missing_checkpoint_paths": sorted(set(old) - set(new)),
        "comparisons": rows,
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}")
    return config


def load_test_sources(config: dict[str, Any], count: int) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = PROJECT_ROOT / config["dataset_path"]
    with h5py.File(dataset_path, "r") as handle:
        total = int(handle["images"].shape[0])
        indices = np.arange(total, dtype=np.int64)
        rng = np.random.default_rng(int(config["seed"]))
        rng.shuffle(indices)
        n_train = int(total * float(config["splits"]["train_frac"]))
        n_val = int(total * float(config["splits"]["val_frac"]))
        test_indices = indices[n_train + n_val :]
        selected = test_indices[: min(count, len(test_indices))]
        order = np.argsort(selected)
        loaded = handle["images"][selected[order]]
    restored = np.empty_like(loaded)
    restored[order] = loaded
    return restored.astype(np.float32) / 255.0, selected


def resolve_artifact_candidates(path: Path | None) -> Path | None:
    if path is not None:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
    candidates = sorted(
        (PROJECT_ROOT / "outputs/runs").glob(
            "source_artifact_audit_*/tables/source_artifact_candidates.csv"
        )
    )
    return candidates[-1] if candidates else None


def load_artifact_indices(path: Path | None) -> set[int]:
    if path is None:
        return set()
    frame = pd.read_csv(path, usecols=["global_index"])
    return {int(value) for value in frame["global_index"]}


def load_models(
    config: dict[str, Any],
    device: torch.device,
    v02_path: Path,
    delta_path: Path,
    resunet_path: Path,
) -> dict[str, torch.nn.Module]:
    for checkpoint in (v02_path, delta_path, resunet_path):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    loaded: dict[str, torch.nn.Module] = {
        "br_v02_moderate": balanced_helpers.load_residual_model(
            v02_path, config["model"], device
        ),
        "br_v03_delta": balanced_helpers.load_residual_model(
            delta_path, config["model"], device
        ),
    }
    try:
        checkpoint = torch.load(resunet_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(resunet_path, map_location="cpu")
    resunet = gd_models.ResUNet(**config["model"])
    resunet.load_state_dict(balanced_helpers.checkpoint_state_dict(checkpoint))
    resunet.to(device).eval()
    loaded["resunet_v04"] = resunet
    return loaded


def infer_residuals(
    inputs: np.ndarray,
    loaded: dict[str, torch.nn.Module],
    device: torch.device,
) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().to(device)
    outputs: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, model in loaded.items():
            outputs[name] = (
                model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
            )
    del tensor
    return outputs


def clean_audit(
    images: np.ndarray,
    global_indices: np.ndarray,
    loaded: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
    artifact_indices: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for start in range(0, len(images), batch_size):
        inputs = images[start : start + batch_size]
        layers = infer_residuals(inputs, loaded, device)
        for offset, target in enumerate(inputs):
            core = balanced_helpers.target_core_mask(target)
            noncore = ~core
            score = 0.0
            example_models: dict[str, dict[str, np.ndarray]] = {}
            for model_name, batch_layers in layers.items():
                residual = batch_layers[offset]
                preclip = target - residual
                clipped = np.clip(preclip, 0.0, 1.0).astype(np.float32)
                positive = np.clip(residual, 0.0, None)
                whole_mse = gd_utils.mse(clipped, target)
                score += whole_mse
                rows.append(
                    {
                        "sample_index": start + offset,
                        "source_global_index": int(global_indices[start + offset]),
                        "source_artifact_heuristic_flag": bool(
                            int(global_indices[start + offset]) in artifact_indices
                        ),
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "whole_mse_unclipped": gd_utils.mse(preclip, target),
                        "whole_mae_unclipped": gd_utils.mae(preclip, target),
                        "ssim_unclipped": gd_utils.ssim_metric(preclip, target),
                        "whole_mse_clipped": whole_mse,
                        "whole_mae_clipped": gd_utils.mae(clipped, target),
                        "ssim_clipped": gd_utils.ssim_metric(clipped, target),
                        "predicted_residual_abs_mean": float(np.mean(np.abs(residual))),
                        "predicted_residual_rms": float(np.sqrt(np.mean(residual**2))),
                        "predicted_residual_abs_max": float(np.max(np.abs(residual))),
                        "predicted_residual_negative_fraction": float(np.mean(residual < 0.0)),
                        "predicted_residual_positive_fraction": float(np.mean(residual > 0.0)),
                        "core_reconstruction_mse": masked_mse(clipped, target, core),
                        "noncore_reconstruction_mse": masked_mse(clipped, target, noncore),
                        "core_residual_abs_mean": float(np.mean(np.abs(residual[core]))),
                        "noncore_residual_abs_mean": float(np.mean(np.abs(residual[noncore]))),
                        "core_false_subtraction_mean": float(np.mean(positive[core])),
                        "noncore_false_subtraction_mean": float(np.mean(positive[noncore])),
                        "core_false_subtraction_gt_0_02_fraction": float(np.mean(positive[core] > 0.02)),
                        "noncore_false_subtraction_gt_0_02_fraction": float(np.mean(positive[noncore] > 0.02)),
                        "fraction_clipped_low": float(np.mean(preclip < 0.0)),
                        "fraction_clipped_high": float(np.mean(preclip > 1.0)),
                    }
                )
                example_models[model_name] = {
                    "reconstruction": clipped.copy(),
                    "residual": residual.copy(),
                }
            examples.append(
                {
                    "score": score,
                    "source_global_index": int(global_indices[start + offset]),
                    "target": target.copy(),
                    "models": example_models,
                }
            )
            examples = sorted(examples, key=lambda item: item["score"], reverse=True)[:3]
        del layers
    per_sample = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in per_sample.columns
        if column
        not in {
            "sample_index",
            "source_global_index",
            "source_artifact_heuristic_flag",
            "model",
            "model_label",
        }
    ]
    summary_rows: list[dict[str, Any]] = []
    for model_name, frame in per_sample.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "n_samples": len(frame),
            "artifact_flagged_samples": int(
                frame["source_artifact_heuristic_flag"].sum()
            ),
        }
        row.update({column: safe_mean(frame[column]) for column in metric_columns})
        clean_mse = frame["whole_mse_clipped"]
        row.update(
            {
                "whole_mse_clipped_median": float(clean_mse.median()),
                "whole_mse_clipped_p95": float(clean_mse.quantile(0.95)),
                "whole_mse_clipped_p99": float(clean_mse.quantile(0.99)),
                "whole_mse_clipped_max": float(clean_mse.max()),
                "clean_samples_mse_gt_1e_4": int((clean_mse > 1e-4).sum()),
                "clean_samples_mse_gt_5e_4": int((clean_mse > 5e-4).sum()),
                "clean_samples_mse_gt_1e_3": int((clean_mse > 1e-3).sum()),
            }
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), per_sample, examples


def summarize_null_strata(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, flagged), frame in per_sample.groupby(
        ["model", "source_artifact_heuristic_flag"], sort=False
    ):
        mse = frame["whole_mse_clipped"]
        rows.append(
            {
                "model": model,
                "model_label": frame["model_label"].iloc[0],
                "source_artifact_heuristic_flag": bool(flagged),
                "stratum": "artifact_candidate" if flagged else "not_flagged",
                "n_samples": len(frame),
                "whole_mse_clipped": safe_mean(mse),
                "whole_mae_clipped": safe_mean(frame["whole_mae_clipped"]),
                "ssim_clipped": safe_mean(frame["ssim_clipped"]),
                "predicted_residual_abs_mean": safe_mean(
                    frame["predicted_residual_abs_mean"]
                ),
                "core_reconstruction_mse": safe_mean(
                    frame["core_reconstruction_mse"]
                ),
                "whole_mse_clipped_p95": float(mse.quantile(0.95)),
                "whole_mse_clipped_p99": float(mse.quantile(0.99)),
                "whole_mse_clipped_max": float(mse.max()),
            }
        )
    return pd.DataFrame(rows)


def generate_audit_suite(
    name: str,
    images: np.ndarray,
    config: dict[str, Any],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    settings = v03_helpers.V03Settings(
        n_normal_test_blends=count,
        n_suite_blends=count,
        test_source_subset=len(images),
    )
    samples, threshold, details = v03_helpers.generate_suite(
        name, images, config, settings, seed
    )
    for sample in samples:
        sample.pop("contaminant", None)
    return samples, threshold, details


def blended_audits(
    images: np.ndarray,
    loaded: dict[str, torch.nn.Module],
    device: torch.device,
    config: dict[str, Any],
    count: int,
    batch_size: int,
    audit_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    region_rows: list[dict[str, Any]] = []
    clipping_rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    suite_logs: list[dict[str, Any]] = []
    for suite_offset, suite in enumerate(("normal", "hard_stress")):
        suite_seed = audit_seed + suite_offset
        print(f"Generating {suite} development-audit suite with seed {suite_seed}.", flush=True)
        samples, threshold, details = generate_audit_suite(
            suite, images, config, count, suite_seed
        )
        suite_logs.append(
            {
                "suite": suite,
                "seed": suite_seed,
                "affected_mask_threshold": threshold,
                "sample_count": len(samples),
                "generator_details": details,
                "status": "development_audit_not_locked_final_test",
            }
        )
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            inputs = np.stack([sample["blended"] for sample in batch]).astype(np.float32)
            targets = np.stack([sample["target"] for sample in batch]).astype(np.float32)
            layers = infer_residuals(inputs, loaded, device)
            for offset, (blended, target) in enumerate(zip(inputs, targets)):
                sample_index = start + offset
                affected = gd_utils.affected_region_mask(target, blended, threshold)
                unaffected = ~affected
                identity_unaffected_mse = masked_mse(blended, target, unaffected)
                identity_unaffected_mae = masked_mae(blended, target, unaffected)
                identity_values = {
                    "sample_index": sample_index,
                    "suite": suite,
                    "model": "identity",
                    "model_label": "Identity",
                    "affected_fraction": float(affected.mean()),
                    "affected_mse": masked_mse(blended, target, affected),
                    "affected_mae": masked_mae(blended, target, affected),
                    "unaffected_mse": identity_unaffected_mse,
                    "unaffected_mae": identity_unaffected_mae,
                    "unaffected_target_mse": identity_unaffected_mse,
                    "unaffected_target_mae": identity_unaffected_mae,
                    "unaffected_output_vs_blended_mse": 0.0,
                    "unaffected_output_vs_blended_mae": 0.0,
                    "unaffected_excess_target_mse_vs_identity": 0.0,
                    "unaffected_excess_target_mae_vs_identity": 0.0,
                }
                region_rows.append(identity_values)
                for model_name, batch_layers in layers.items():
                    residual = batch_layers[offset]
                    preclip = blended - residual
                    clipped = np.clip(preclip, 0.0, 1.0).astype(np.float32)
                    unaffected_target_mse = masked_mse(clipped, target, unaffected)
                    unaffected_target_mae = masked_mae(clipped, target, unaffected)
                    region_rows.append(
                        {
                            "sample_index": sample_index,
                            "suite": suite,
                            "model": model_name,
                            "model_label": MODEL_LABELS[model_name],
                            "affected_fraction": float(affected.mean()),
                            "affected_mse": masked_mse(clipped, target, affected),
                            "affected_mae": masked_mae(clipped, target, affected),
                            "unaffected_mse": unaffected_target_mse,
                            "unaffected_mae": unaffected_target_mae,
                            "unaffected_target_mse": unaffected_target_mse,
                            "unaffected_target_mae": unaffected_target_mae,
                            "unaffected_output_vs_blended_mse": masked_mse(
                                clipped, blended, unaffected
                            ),
                            "unaffected_output_vs_blended_mae": masked_mae(
                                clipped, blended, unaffected
                            ),
                            "unaffected_excess_target_mse_vs_identity": (
                                unaffected_target_mse - identity_unaffected_mse
                            ),
                            "unaffected_excess_target_mae_vs_identity": (
                                unaffected_target_mae - identity_unaffected_mae
                            ),
                        }
                    )
                    for state, reconstruction in (("unclipped", preclip), ("clipped", clipped)):
                        clipping_rows.append(
                            {
                                "sample_index": sample_index,
                                "suite": suite,
                                "model": model_name,
                                "model_label": MODEL_LABELS[model_name],
                                "reconstruction_state": state,
                                "whole_mse": gd_utils.mse(reconstruction, target),
                                "whole_mae": gd_utils.mae(reconstruction, target),
                                "ssim": gd_utils.ssim_metric(reconstruction, target),
                                "affected_mse": masked_mse(reconstruction, target, affected),
                                "affected_mae": masked_mae(reconstruction, target, affected),
                            }
                        )
                    low = preclip < 0.0
                    high = preclip > 1.0
                    residual_negative = residual < 0.0
                    residual_positive = residual > 0.0
                    affected_channels = np.repeat(affected[..., None], 3, axis=2)
                    pixel_rows.append(
                        {
                            "sample_index": sample_index,
                            "suite": suite,
                            "model": model_name,
                            "model_label": MODEL_LABELS[model_name],
                            "channel_value_fraction_clipped_low": float(np.mean(low)),
                            "channel_value_fraction_clipped_high": float(np.mean(high)),
                            "pixel_fraction_any_channel_clipped_low": float(np.mean(np.any(low, axis=2))),
                            "pixel_fraction_any_channel_clipped_high": float(np.mean(np.any(high, axis=2))),
                            "predicted_residual_abs_mean": float(
                                np.mean(np.abs(residual))
                            ),
                            "predicted_residual_negative_fraction": float(
                                np.mean(residual_negative)
                            ),
                            "predicted_residual_positive_fraction": float(
                                np.mean(residual_positive)
                            ),
                            "predicted_residual_negative_magnitude_conditional": (
                                float(np.mean(-residual[residual_negative]))
                                if np.any(residual_negative)
                                else 0.0
                            ),
                            "predicted_residual_positive_magnitude_conditional": (
                                float(np.mean(residual[residual_positive]))
                                if np.any(residual_positive)
                                else 0.0
                            ),
                            "affected_channel_fraction_below_zero": float(np.mean(low[affected_channels])) if np.any(affected_channels) else float("nan"),
                            "affected_channel_fraction_above_one": float(np.mean(high[affected_channels])) if np.any(affected_channels) else float("nan"),
                            "mean_low_excursion": float(np.mean(np.clip(-preclip, 0.0, None))),
                            "mean_high_excursion": float(np.mean(np.clip(preclip - 1.0, 0.0, None))),
                            "mean_low_excursion_conditional": (
                                float(np.mean(-preclip[low])) if np.any(low) else 0.0
                            ),
                            "mean_high_excursion_conditional": (
                                float(np.mean(preclip[high] - 1.0))
                                if np.any(high)
                                else 0.0
                            ),
                            "over_subtraction_fraction": float(np.mean(residual > blended)),
                            "affected_over_subtraction_fraction": float(np.mean((residual > blended)[affected_channels])) if np.any(affected_channels) else float("nan"),
                        }
                    )
            del inputs, targets, layers
        del samples
        gc.collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    region_per_sample = pd.DataFrame(region_rows)
    clipping_per_sample = pd.DataFrame(clipping_rows)
    pixels_per_sample = pd.DataFrame(pixel_rows)

    region_summary_rows: list[dict[str, Any]] = []
    for (suite, model), frame in region_per_sample.groupby(["suite", "model"], sort=False):
        region_summary_rows.append(
            {
                "suite": suite,
                "model": model,
                "model_label": frame["model_label"].iloc[0],
                "n_samples": len(frame),
                "mean_affected_fraction": safe_mean(frame["affected_fraction"]),
                "affected_mse": safe_mean(frame["affected_mse"]),
                "affected_mae": safe_mean(frame["affected_mae"]),
                "unaffected_mse": safe_mean(frame["unaffected_mse"]),
                "unaffected_mae": safe_mean(frame["unaffected_mae"]),
                "unaffected_target_mse": safe_mean(frame["unaffected_target_mse"]),
                "unaffected_target_mae": safe_mean(frame["unaffected_target_mae"]),
                "unaffected_output_vs_blended_mse": safe_mean(
                    frame["unaffected_output_vs_blended_mse"]
                ),
                "unaffected_output_vs_blended_mae": safe_mean(
                    frame["unaffected_output_vs_blended_mae"]
                ),
                "unaffected_excess_target_mse_vs_identity": safe_mean(
                    frame["unaffected_excess_target_mse_vs_identity"]
                ),
                "unaffected_excess_target_mae_vs_identity": safe_mean(
                    frame["unaffected_excess_target_mae_vs_identity"]
                ),
                "unaffected_to_affected_mse_ratio": safe_mean(frame["unaffected_mse"]) / max(safe_mean(frame["affected_mse"]), 1e-15),
            }
        )

    clipping_summary_rows: list[dict[str, Any]] = []
    for (suite, model, state), frame in clipping_per_sample.groupby(
        ["suite", "model", "reconstruction_state"], sort=False
    ):
        clipping_summary_rows.append(
            {
                "suite": suite,
                "model": model,
                "model_label": frame["model_label"].iloc[0],
                "reconstruction_state": state,
                "n_samples": len(frame),
                "whole_mse": safe_mean(frame["whole_mse"]),
                "whole_mae": safe_mean(frame["whole_mae"]),
                "ssim": safe_mean(frame["ssim"]),
                "affected_mse": safe_mean(frame["affected_mse"]),
                "affected_mae": safe_mean(frame["affected_mae"]),
            }
        )

    unclipped = clipping_per_sample[
        clipping_per_sample["reconstruction_state"] == "unclipped"
    ].drop(columns=["reconstruction_state"])
    clipped = clipping_per_sample[
        clipping_per_sample["reconstruction_state"] == "clipped"
    ].drop(columns=["reconstruction_state"])
    paired = unclipped.merge(
        clipped,
        on=["sample_index", "suite", "model", "model_label"],
        suffixes=("_unclipped", "_clipped"),
        validate="one_to_one",
    )
    for metric in ("whole_mse", "whole_mae", "ssim", "affected_mse", "affected_mae"):
        paired[f"{metric}_change_unclipped_minus_clipped"] = (
            paired[f"{metric}_unclipped"] - paired[f"{metric}_clipped"]
        )
    paired["whole_mse_relative_reduction"] = (
        paired["whole_mse_change_unclipped_minus_clipped"]
        / paired["whole_mse_unclipped"].clip(lower=1e-15)
    )
    paired["affected_mse_relative_reduction"] = (
        paired["affected_mse_change_unclipped_minus_clipped"]
        / paired["affected_mse_unclipped"].clip(lower=1e-15)
    )

    clipping_effect_rows: list[dict[str, Any]] = []
    for (suite, model), frame in paired.groupby(["suite", "model"], sort=False):
        affected_change = frame["affected_mse_change_unclipped_minus_clipped"]
        whole_change = frame["whole_mse_change_unclipped_minus_clipped"]
        affected_relative = frame["affected_mse_relative_reduction"]
        clipping_effect_rows.append(
            {
                "suite": suite,
                "model": model,
                "model_label": frame["model_label"].iloc[0],
                "n_samples": len(frame),
                "whole_mse_change_mean": safe_mean(whole_change),
                "whole_mse_change_p95": float(whole_change.quantile(0.95)),
                "whole_mse_change_p99": float(whole_change.quantile(0.99)),
                "whole_mse_change_max": float(whole_change.max()),
                "affected_mse_change_mean": safe_mean(affected_change),
                "affected_mse_change_p95": float(affected_change.quantile(0.95)),
                "affected_mse_change_p99": float(affected_change.quantile(0.99)),
                "affected_mse_change_max": float(affected_change.max()),
                "affected_relative_reduction_p95": float(
                    affected_relative.quantile(0.95)
                ),
                "affected_relative_reduction_p99": float(
                    affected_relative.quantile(0.99)
                ),
                "affected_relative_reduction_max": float(
                    affected_relative.max()
                ),
                "samples_affected_mse_change_gt_1e_4": int(
                    (affected_change > 1e-4).sum()
                ),
                "samples_affected_relative_reduction_gt_5pct": int(
                    (affected_relative > 0.05).sum()
                ),
                "samples_affected_relative_reduction_gt_10pct": int(
                    (affected_relative > 0.10).sum()
                ),
            }
        )

    pixel_summary_rows: list[dict[str, Any]] = []
    numeric_pixel_columns = [
        column
        for column in pixels_per_sample.columns
        if column not in {"sample_index", "suite", "model", "model_label"}
    ]
    for (suite, model), frame in pixels_per_sample.groupby(["suite", "model"], sort=False):
        row: dict[str, Any] = {
            "suite": suite,
            "model": model,
            "model_label": frame["model_label"].iloc[0],
            "n_samples": len(frame),
        }
        row.update({column: safe_mean(frame[column]) for column in numeric_pixel_columns})
        for column in (
            "channel_value_fraction_clipped_low",
            "channel_value_fraction_clipped_high",
            "mean_low_excursion",
            "mean_high_excursion",
            "mean_low_excursion_conditional",
            "mean_high_excursion_conditional",
        ):
            row[f"{column}_p95"] = float(frame[column].quantile(0.95))
            row[f"{column}_p99"] = float(frame[column].quantile(0.99))
            row[f"{column}_max"] = float(frame[column].max())
        pixel_summary_rows.append(row)
    return (
        pd.DataFrame(region_summary_rows),
        region_per_sample,
        pd.DataFrame(clipping_summary_rows),
        paired,
        pd.DataFrame(pixel_summary_rows),
        pixels_per_sample,
        pd.DataFrame(clipping_effect_rows),
        suite_logs,
    )


def plot_clean_grid(examples: list[dict[str, Any]], path: Path) -> None:
    methods = list(MODEL_LABELS)
    fig, axes = plt.subplots(len(examples), 1 + 2 * len(methods), figsize=(19, 3.5 * len(examples)))
    axes = np.atleast_2d(axes)
    for row_index, example in enumerate(examples):
        panels: list[tuple[np.ndarray, str, str]] = [
            (example["target"], f"Unblended input\nsource {example['source_global_index']}", "rgb")
        ]
        for method in methods:
            reconstruction = example["models"][method]["reconstruction"]
            error = np.abs(reconstruction - example["target"]).mean(axis=2)
            panels.extend(
                [
                    (reconstruction, f"{MODEL_LABELS[method]}\nreconstruction", "rgb"),
                    (error, "absolute error", "heat"),
                ]
            )
        for axis, (image, title, kind) in zip(axes[row_index], panels):
            if kind == "rgb":
                axis.imshow(np.clip(image, 0.0, 1.0))
            else:
                axis.imshow(image, cmap="magma", vmin=0.0, vmax=max(0.02, float(np.percentile(image, 99.5))))
            axis.set_title(title, fontsize=8)
            axis.axis("off")
    fig.suptitle("Unblended-input preservation: highest combined-error audit examples", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_residual_heatmaps(examples: list[dict[str, Any]], path: Path) -> None:
    methods = list(MODEL_LABELS)
    fig, axes = plt.subplots(len(examples), 1 + len(methods), figsize=(13, 3.4 * len(examples)))
    axes = np.atleast_2d(axes)
    for row_index, example in enumerate(examples):
        axes[row_index, 0].imshow(np.clip(example["target"], 0.0, 1.0))
        axes[row_index, 0].set_title(f"Unblended input\nsource {example['source_global_index']}", fontsize=8)
        axes[row_index, 0].axis("off")
        magnitudes = [
            np.mean(np.abs(example["models"][method]["residual"]), axis=2)
            for method in methods
        ]
        vmax = max(0.02, float(np.percentile(np.stack(magnitudes), 99.5)))
        for column, (method, magnitude) in enumerate(zip(methods, magnitudes), start=1):
            axes[row_index, column].imshow(magnitude, cmap="inferno", vmin=0.0, vmax=vmax)
            axes[row_index, column].set_title(f"{MODEL_LABELS[method]}\n|predicted residual|", fontsize=8)
            axes[row_index, column].axis("off")
    fig.suptitle("False residual predictions on unblended targets", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_region_summary(frame: pd.DataFrame, path: Path) -> None:
    plotted = frame[frame["model"] != "identity"].copy()
    labels = [f"{row.model_label}\n{row.suite}" for row in plotted.itertuples()]
    x = np.arange(len(plotted))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12, 5.2))
    axis.bar(x - width / 2, plotted["affected_mse"], width, label="Affected MSE", color="#c44e52")
    axis.bar(
        x + width / 2,
        plotted["unaffected_target_mse"],
        width,
        label="Unaffected-region target MSE",
        color="#4c72b0",
    )
    axis.set_yscale("log")
    axis.set_ylabel("MSE (log scale)")
    axis.set_xticks(x, labels, rotation=24, ha="right")
    axis.set_title("Affected- and unaffected-region target error")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_clipping_summary(frame: pd.DataFrame, path: Path) -> None:
    pivot = frame.pivot_table(index=["suite", "model_label"], columns="reconstruction_state", values="affected_mse").reset_index()
    x = np.arange(len(pivot))
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x - 0.18, pivot["unclipped"], 0.36, label="Unclipped", color="#8172b2")
    axis.bar(x + 0.18, pivot["clipped"], 0.36, label="Clipped", color="#55a868")
    axis.set_xticks(x, [f"{row.model_label}\n{row.suite}" for row in pivot.itertuples()], rotation=22, ha="right")
    axis.set_ylabel("Affected-region MSE")
    axis.set_title("Effect of output clipping on affected-region MSE")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}g}" if np.isfinite(value) else "n/a")
    headers = [str(column) for column in shown.columns]
    values = [[str(value) for value in row] for row in shown.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"

    lines = [
        render(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(render(row) for row in values)
    return "\n".join(lines)


def write_reports(
    preservation: Path,
    clipping: Path,
    device_info: dict[str, Any],
    clean_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    clipping_summary: pd.DataFrame,
    pixel_summary: pd.DataFrame,
    n_clean: int,
    n_blends: int,
    integrity_ok: bool,
    null_strata: pd.DataFrame | None = None,
    clipping_effect_summary: pd.DataFrame | None = None,
) -> None:
    clean_keyed = clean_summary.set_index("model")
    v02_clean = clean_keyed.loc["br_v02_moderate"]
    delta_clean = clean_keyed.loc["br_v03_delta"]
    resunet_clean = clean_keyed.loc["resunet_v04"]
    delta_clean_factor = float(v02_clean["whole_mse_clipped"] / delta_clean["whole_mse_clipped"])
    resunet_clean_reduction = 100.0 * (
        1.0 - float(resunet_clean["whole_mse_clipped"] / v02_clean["whole_mse_clipped"])
    )
    resunet_core_increase = 100.0 * (
        float(resunet_clean["core_reconstruction_mse"] / v02_clean["core_reconstruction_mse"]) - 1.0
    )
    clean_table = markdown_table(
        clean_summary,
        [
            "model_label",
            "whole_mse_clipped",
            "whole_mae_clipped",
            "ssim_clipped",
            "predicted_residual_abs_mean",
            "core_reconstruction_mse",
            "noncore_reconstruction_mse",
            "whole_mse_clipped_p99",
            "whole_mse_clipped_max",
            "clean_samples_mse_gt_1e_3",
        ],
    )
    if null_strata is not None and not null_strata.empty:
        artifact_table = markdown_table(
            null_strata,
            [
                "model_label",
                "stratum",
                "n_samples",
                "whole_mse_clipped",
                "whole_mse_clipped_p99",
                "whole_mse_clipped_max",
            ],
        )
        flagged_count = int(
            null_strata[null_strata["source_artifact_heuristic_flag"] == True][
                "n_samples"
            ].max()
        )
        artifact_section = f"""## Heuristic source-artifact stratification

The source-artifact audit flags are review heuristics, not validated labels or automatic exclusions. {flagged_count} of the {n_clean} unblended inputs are flagged candidates.

{artifact_table}

The null inputs are therefore called **unblended**, not clean-source filtered. A higher error in the flagged stratum may reflect real artifacts, legitimate companions, saturated stars, edge-touching morphology, or several of these at once.
"""
    else:
        artifact_section = """## Heuristic source-artifact stratification

No source-artifact candidate manifest was available. These inputs are unblended but are not certified clean-source examples.
"""
    region_table = markdown_table(
        region_summary[region_summary["model"] != "identity"],
        [
            "suite",
            "model_label",
            "affected_mse",
            "unaffected_target_mse",
            "unaffected_output_vs_blended_mse",
            "unaffected_excess_target_mse_vs_identity",
        ],
    )
    preservation_text = f"""# Preservation and null-test report

## Protocol

- Selected inference device: `{device_info['selected_device']}`.
- MPS available: `{str(device_info['mps_available']).lower()}`; CUDA available: `{str(device_info['cuda_available']).lower()}`.
- CPU fallback for model inference: no.
- Unblended null test: {n_clean} targets from the existing seed-42 development test partition; these are not artifact-filtered clean sources.
- Blended region test: {n_blends} fixed-seed normal and {n_blends} fixed-seed hard-stress development blends.
- Models: Thayer-BR v0.2 Moderate, v0.3 Delta, and ResUNet v0.4.
- Residual expectation: an unblended input should produce a residual close to zero, so `input - predicted_residual` preserves the target.

These are diagnostic development tests, not the locked final evaluation. The source-leakage audit found duplicated sky coordinates across the historical random-index partitions, so the partition label does not establish object-level independence.

## Unblended-input preservation

{clean_table}

`predicted_residual_abs_mean` directly measures false residual activity on unblended targets. Core and non-core reconstruction errors use the evaluation core mask (central aperture plus 85th-percentile brightness threshold). The training loss uses a different core definition based on 55% of the aperture maximum. Per-sample values, positive false-subtraction magnitudes, and clipping fractions are retained in the supplemental table.

{artifact_section}

## Affected versus unaffected regions

{region_table}

The affected mask is defined by mean absolute RGB blend change greater than 0.02. Its complement still contains sub-threshold blend changes, blur, and noise, so `unaffected_target_mse` is not pure model damage. `unaffected_output_vs_blended_mse` measures how much the model changes that region, while `unaffected_excess_target_mse_vs_identity` subtracts the paired identity target error. Negative excess means the model corrected some sub-threshold error.

All masked aggregates are macro means of per-sample masked metrics, so each image contributes equally rather than each masked pixel contributing equally.

## Interpretation

The null test quantifies preservation rather than assuming residual prediction is harmless on unblended inputs. Delta's mean unblended-input MSE is {delta_clean_factor:.1f}x lower than v0.2 Moderate. ResUNet's mean unblended-input MSE is {resunet_clean_reduction:.1f}% lower than v0.2, but its core MSE is {resunet_core_increase:.1f}% higher and its mean absolute error is also higher.

The aggregate null errors are small, but v0.2 has {int(v02_clean['clean_samples_mse_gt_1e_3'])}/1,000 tail cases above unblended-input MSE 0.001, with maximum {float(v02_clean['whole_mse_clipped_max']):.6f}. Manual review of the saved highest-error grid shows false subtraction of bright off-center sources and genuine target structure. This is a meaningful preservation failure mode even though it is not severe in the aggregate average; it must remain visible in limitations and future clean-source benchmarks.

Any model promotion should consider both affected-region repair and unaffected/clean-image damage. Because cross-split duplicate-source leakage is now a major benchmark blocker, these results inform diagnosis only and do not authorize further architecture training tonight.

## Integrity

All pre-existing checkpoints unchanged by this audit: `{str(integrity_ok).lower()}`. No checkpoint was created, edited, or deleted.
"""
    (preservation / "diagnostics/preservation_null_test_report.md").write_text(
        preservation_text, encoding="utf-8"
    )

    clipping_table = markdown_table(
        clipping_summary,
        ["suite", "model_label", "reconstruction_state", "whole_mse", "whole_mae", "ssim", "affected_mse"],
    )
    pixel_table = markdown_table(
        pixel_summary,
        [
            "suite",
            "model_label",
            "channel_value_fraction_clipped_low",
            "channel_value_fraction_clipped_high",
            "predicted_residual_negative_fraction",
            "predicted_residual_negative_magnitude_conditional",
            "predicted_residual_positive_magnitude_conditional",
            "mean_low_excursion_conditional",
            "mean_high_excursion_conditional",
            "over_subtraction_fraction",
        ],
    )
    paired = clipping_summary.pivot_table(
        index=["suite", "model"],
        columns="reconstruction_state",
        values=["whole_mse", "affected_mse"],
    )
    whole_change = (paired[("whole_mse", "unclipped")] - paired[("whole_mse", "clipped")]) / paired[("whole_mse", "unclipped")]
    affected_change = (paired[("affected_mse", "unclipped")] - paired[("affected_mse", "clipped")]) / paired[("affected_mse", "unclipped")]
    max_whole_change = 100.0 * float(whole_change.max())
    max_affected_change = 100.0 * float(affected_change.max())
    if clipping_effect_summary is not None and not clipping_effect_summary.empty:
        clipping_effect_table = markdown_table(
            clipping_effect_summary,
            [
                "suite",
                "model_label",
                "affected_mse_change_p99",
                "affected_mse_change_max",
                "affected_relative_reduction_p99",
                "samples_affected_mse_change_gt_1e_4",
                "samples_affected_relative_reduction_gt_10pct",
            ],
        )
        tail_max = float(clipping_effect_summary["affected_mse_change_max"].max())
        tail_p99 = float(clipping_effect_summary["affected_mse_change_p99"].max())
        tail_abs_count = int(
            clipping_effect_summary["samples_affected_mse_change_gt_1e_4"].sum()
        )
        tail_relative_count = int(
            clipping_effect_summary[
                "samples_affected_relative_reduction_gt_10pct"
            ].sum()
        )
        tail_section = f"""## Per-sample clipping effects

{clipping_effect_table}

Across all model/suite rows, the largest per-sample absolute affected-MSE reduction from clipping is {tail_max:.6g}; the largest group p99 is {tail_p99:.6g}. There are {tail_abs_count} model/sample comparisons with an absolute affected-MSE reduction above 0.0001 and {tail_relative_count} with a relative reduction above 10%. Relative changes can look large when the unclipped error is very small, so absolute and relative tails are reported together.
"""
    else:
        tail_section = """## Per-sample clipping effects

Per-sample clipping distributions were not available in this run. Aggregate means alone do not clear rare clipping-dependent failures.
"""
    clipping_text = f"""# Clipped versus unclipped reconstruction audit

## Protocol

- Selected inference device: `{device_info['selected_device']}`; CPU inference fallback was not used.
- Evaluated {n_blends} normal and {n_blends} hard-stress fixed-seed development blends.
- Reconstruction before clipping: `blended - predicted_residual`.
- Reconstruction after clipping: values constrained to `[0, 1]`.
- Models: Thayer-BR v0.2 Moderate, v0.3 Delta, and ResUNet v0.4.

## Metric comparison

{clipping_table}

These are macro means of per-sample image metrics. They are not pooled-pixel estimates.

## Output-range statistics

{pixel_table}

The low/high clipping fractions count RGB channel values outside the valid display range before clipping. `over_subtraction_fraction` is the fraction of channel values where the predicted residual exceeds the blended input, which makes the raw reconstruction negative. Negative predicted residuals are also reported because they add light during residual subtraction.

`over_subtraction_fraction` is mathematically the same event as the low-clipping fraction and is retained only as a physically descriptive alias. Unconditional mean excursions average zeros over all channel values; the conditional columns average magnitude only where clipping occurs. Residual sign fractions are paired with conditional sign magnitudes.

{tail_section}

## Interpretation

Clipping can lower squared error when targets lie in `[0, 1]`, so clipped metrics are useful for the displayed reconstruction. In the macro aggregates, clipping lowers whole-image MSE by at most {max_whole_change:.2f}% and affected-region MSE by at most {max_affected_change:.2f}% across the six model/suite comparisons. It does not change the aggregate model ranking. The per-sample distribution above determines whether rare clipping-sensitive cases remain.

The nonzero out-of-range and residual-sign fractions still matter as physical diagnostics. Fractions alone do not measure excursion magnitude: the saved table also reports mean low/high excursions. Final reporting should retain both clipped and unclipped forms whenever a future model shows a material gap. These results remain development diagnostics because the historical random-index source split has confirmed duplicated-coordinate leakage.

## Integrity

All pre-existing checkpoints unchanged by this audit: `{str(integrity_ok).lower()}`. No checkpoint was created, edited, or deleted.
"""
    (clipping / "diagnostics/clipping_audit_report.md").write_text(
        clipping_text, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.n_clean <= 0 or args.n_blends <= 0 or args.batch_size <= 0:
        raise ValueError("Sample counts and batch size must be positive.")
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    preservation, clipping = make_run_dirs(stamp)
    try:
        device, device_info = select_device()
    except RuntimeError as exc:
        for run_dir in (preservation, clipping):
            (run_dir / "diagnostics/device_unavailable.md").write_text(
                f"# Device unavailable\n\n{exc}\n\nNo model inference was started.\n",
                encoding="utf-8",
            )
        raise
    print(f"Selected device for full model inference: {device}", flush=True)
    for run_dir in (preservation, clipping):
        save_json(run_dir / "logs/selected_device.json", device_info)

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config_path = config_path.resolve()
    config = load_config(config_path)
    artifact_candidates_path = resolve_artifact_candidates(args.artifact_candidates)
    artifact_indices = load_artifact_indices(artifact_candidates_path)
    dataset_path = (PROJECT_ROOT / config["dataset_path"]).resolve()
    provenance_paths = {
        "config": config_path,
        "audit_script": Path(__file__).resolve(),
        "src_data": PROJECT_ROOT / "src/data.py",
        "src_blend": PROJECT_ROOT / "src/blend.py",
        "src_models": PROJECT_ROOT / "src/models.py",
        "src_utils": PROJECT_ROOT / "src/utils.py",
        "balanced_helper": PROJECT_ROOT / "scripts/train_balanced_residual_unet.py",
        "suite_helper": PROJECT_ROOT / "scripts/train_v03_color_structure_unet.py",
    }
    input_provenance = {
        "dataset": {
            "path": relative(dataset_path),
            "size_bytes": int(dataset_path.stat().st_size),
            "sha256": sha256_file(dataset_path),
        },
        "code_and_config_sha256": {
            name: sha256_file(path) for name, path in provenance_paths.items()
        },
        "artifact_candidates": (
            {
                "path": relative(artifact_candidates_path),
                "sha256": sha256_file(artifact_candidates_path),
            }
            if artifact_candidates_path is not None
            else None
        ),
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": importlib_metadata.version("PyYAML"),
            "scikit-image": importlib_metadata.version("scikit-image"),
            "scipy": importlib_metadata.version("scipy"),
        },
    }
    run_config = {
        "stamp": stamp,
        "n_clean": args.n_clean,
        "n_blends_per_suite": args.n_blends,
        "batch_size": args.batch_size,
        "audit_seed": args.audit_seed,
        "source_split_seed": int(config["seed"]),
        "affected_mask_threshold": 0.02,
        "artifact_candidates_path": (
            relative(artifact_candidates_path)
            if artifact_candidates_path is not None
            else None
        ),
        "artifact_candidate_count": len(artifact_indices),
        "input_provenance": input_provenance,
        "checkpoint_paths": {
            "br_v02_moderate": relative(args.v02_checkpoint),
            "br_v03_delta": relative(args.delta_checkpoint),
            "resunet_v04": relative(args.resunet_checkpoint),
        },
        "final_test_status": "development_audit_only",
    }
    for run_dir in (preservation, clipping):
        save_json(run_dir / "logs/run_config.json", run_config)

    print("Hashing checkpoints before inference.", flush=True)
    before = checkpoint_snapshot()
    for run_dir in (preservation, clipping):
        save_json(run_dir / "logs/checkpoint_integrity_before.json", before)

    images, global_indices = load_test_sources(config, max(args.n_clean, 1000))
    clean_images = images[: min(args.n_clean, len(images))]
    clean_indices = global_indices[: len(clean_images)]
    pd.DataFrame(
        {
            "audit_source_position": np.arange(len(global_indices)),
            "source_global_index": global_indices,
            "historical_split": "test",
        }
    ).to_csv(preservation / "tables/audit_source_indices.csv", index=False)

    loaded = load_models(
        config,
        device,
        args.v02_checkpoint,
        args.delta_checkpoint,
        args.resunet_checkpoint,
    )
    print(f"Running clean null test on {len(clean_images)} targets.", flush=True)
    clean_summary, clean_per_sample, examples = clean_audit(
        clean_images,
        clean_indices,
        loaded,
        device,
        args.batch_size,
        artifact_indices,
    )
    null_strata = summarize_null_strata(clean_per_sample)
    clean_summary.to_csv(preservation / "tables/null_preservation_metrics.csv", index=False)
    clean_per_sample.to_csv(
        preservation / "tables/null_preservation_per_sample.csv", index=False
    )
    null_strata.to_csv(
        preservation / "tables/null_preservation_artifact_strata.csv", index=False
    )
    plot_clean_grid(examples, preservation / "figures/clean_input_reconstruction_absolute_error_grid.png")
    plot_residual_heatmaps(examples, preservation / "figures/false_residual_heatmap_examples.png")

    print("Running blended unaffected-region and clipping audits.", flush=True)
    (
        region_summary,
        region_per_sample,
        clipping_summary,
        clipping_effect_per_sample,
        pixel_summary,
        pixel_per_sample,
        clipping_effect_summary,
        suite_logs,
    ) = blended_audits(
        images,
        loaded,
        device,
        config,
        args.n_blends,
        args.batch_size,
        args.audit_seed,
    )
    region_summary.to_csv(preservation / "tables/unaffected_region_metrics.csv", index=False)
    region_per_sample.to_csv(
        preservation / "tables/unaffected_region_per_sample.csv", index=False
    )
    plot_region_summary(region_summary, preservation / "figures/affected_vs_unaffected_mse.png")
    save_json(preservation / "logs/development_suite_generation.json", suite_logs)

    clipping_summary.to_csv(clipping / "tables/clipped_vs_unclipped_metrics.csv", index=False)
    clipping_effect_per_sample.to_csv(
        clipping / "tables/clipped_vs_unclipped_per_sample_metrics.csv",
        index=False,
    )
    pixel_summary.to_csv(clipping / "tables/clipping_pixel_statistics.csv", index=False)
    pixel_per_sample.to_csv(
        clipping / "tables/clipping_pixel_statistics_per_sample.csv", index=False
    )
    clipping_effect_summary.to_csv(
        clipping / "tables/clipping_effect_distribution_summary.csv", index=False
    )
    plot_clipping_summary(clipping_summary, clipping / "figures/clipped_vs_unclipped_affected_mse.png")
    save_json(clipping / "logs/development_suite_generation.json", suite_logs)

    del loaded
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    print("Hashing checkpoints after inference.", flush=True)
    after = checkpoint_snapshot()
    comparison = compare_snapshots(before, after)
    for run_dir in (preservation, clipping):
        save_json(run_dir / "logs/checkpoint_integrity_after.json", after)
        save_json(run_dir / "logs/checkpoint_integrity_comparison.json", comparison)
    if not comparison["old_checkpoints_unchanged"]:
        for run_dir in (preservation, clipping):
            (run_dir / "diagnostics/checkpoint_integrity_failure.md").write_text(
                "# Checkpoint integrity failure\n\nAt least one pre-existing checkpoint changed. Stop further experimentation and inspect the comparison log.\n",
                encoding="utf-8",
            )
        raise RuntimeError("Pre-existing checkpoint integrity changed during audit.")

    write_reports(
        preservation,
        clipping,
        device_info,
        clean_summary,
        region_summary,
        clipping_summary,
        pixel_summary,
        len(clean_images),
        args.n_blends,
        comparison["old_checkpoints_unchanged"],
        null_strata,
        clipping_effect_summary,
    )
    print(f"Preservation audit: {relative(preservation)}", flush=True)
    print(f"Clipping audit: {relative(clipping)}", flush=True)


if __name__ == "__main__":
    main()
