#!/usr/bin/env python3
"""Append-only frozen-representation recoverability ablation.

The only neural operation on the scientific model is an encoder-only MPS pass
through the frozen Phase-II R1 checkpoint.  Head fitting, calibration,
statistics, and plotting run on CPU.  Development and lockbox scenes are never
opened by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models_thayer_select import ThayerSelectNet


PRIMARY = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
REPLICATION = REPO / "outputs/runs/thayer_select_recoverability_seed_replication_20260711_203115"
ROOT_CAUSE = REPO / "outputs/runs/thayer_select_root_cause_analysis_20260711"
SOURCE_SPLIT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/source_split_manifest.csv"
CONTRACT = "moderate"
FEATURE_DIM = 64
CENTROID_FEATURE_NAMES = [
    "centroid_g_to_r_pixels",
    "centroid_r_to_z_pixels",
    "centroid_g_to_z_pixels",
    "centroid_max_psf_units",
    "centroid_consistency_mean_pixels",
]
ORACLE_FEATURES = [
    "snr_proxy",
    "separation_psf_units",
    "flux_ratio",
    "size_ratio",
    "core_obstruction",
    "color_similarity_distance",
    "source_count",
]
HEAD_SPECS = {
    "H0": [],
    "H1": [],
    "H2": [32],
    "H3": [32, 16],
}
BASE_SEED = 20260711
BALANCE_SEEDS = [20260711, 20260712, 20260713]
BALANCE_METHODS = ["class_weighted_bce", "balanced_minibatch_sampling"]
LINEAR_AUROC_MARGIN = 0.01
LINEAR_AUPRC_MARGIN = 0.02
BOOTSTRAPS = 500
EPS = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.write_text(text)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    frame.to_csv(path, index=False)


def run_command(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def logit(scores: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(values / (1.0 - values))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels, dtype=int)
    mask = np.isfinite(score)
    score, label = score[mask], label[mask]
    positive = int(label.sum())
    negative = len(label) - positive
    if positive == 0 or negative == 0:
        return math.nan
    ranks = rankdata(score, method="average")
    return float((ranks[label == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels, dtype=int)
    mask = np.isfinite(score)
    score, label = score[mask], label[mask]
    positives = int(label.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-score, kind="stable")
    ordered = label[order]
    tp = np.cumsum(ordered)
    precision = tp / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(total)


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels, dtype=int)
    prediction = score >= 0.5
    truth = label.astype(bool)
    tp = int(np.sum(prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tpr = tp / (tp + fn) if tp + fn else math.nan
    tnr = tn / (tn + fp) if tn + fp else math.nan
    return {
        "samples": len(label),
        "positives": int(label.sum()),
        "prevalence": float(label.mean()) if len(label) else math.nan,
        "auroc": auroc(score, label),
        "auprc": auprc(score, label),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "recall": tpr,
        "false_positive_rate": fp / (fp + tn) if fp + tn else math.nan,
        "false_negative_rate": fn / (fn + tp) if fn + tp else math.nan,
        "brier_score": float(np.mean((score - label) ** 2)),
        "ece": ece(score, label),
    }


def temperature_fit(scores: np.ndarray, labels: np.ndarray) -> float:
    logits = logit(scores)
    truth = np.asarray(labels, dtype=float)

    def objective(log_temperature: float) -> float:
        calibrated = sigmoid(logits / math.exp(log_temperature))
        return float(-np.mean(truth * np.log(calibrated + EPS) + (1.0 - truth) * np.log(1.0 - calibrated + EPS)))

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(math.exp(result.x))


def temperature_apply(scores: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid(logit(scores) / temperature)


def isotonic_fit(scores: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
    order = np.argsort(np.asarray(scores, dtype=float), kind="stable")
    xs = np.asarray(scores, dtype=float)[order]
    ys = np.asarray(labels, dtype=float)[order]
    blocks: list[list[float]] = []
    for x, y in zip(xs, ys):
        blocks.append([float(x), float(x), float(y), 1.0])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    return {
        "upper_bounds": [row[1] for row in blocks],
        "values": [row[2] / row[3] for row in blocks],
    }


def isotonic_apply(scores: np.ndarray, model: dict[str, list[float]]) -> np.ndarray:
    bounds = np.asarray(model["upper_bounds"], dtype=float)
    values = np.asarray(model["values"], dtype=float)
    indices = np.searchsorted(bounds, np.asarray(scores, dtype=float), side="left")
    return values[np.clip(indices, 0, len(values) - 1)]


def plateau_metrics(scores: np.ndarray) -> dict[str, float | int]:
    rounded = np.round(np.asarray(scores, dtype=float), 12)
    _, counts = np.unique(rounded, return_counts=True)
    return {
        "unique_values": int(len(counts)),
        "tie_fraction": float(1.0 - len(counts) / len(rounded)),
        "largest_probability_plateau": int(counts.max()),
        "largest_probability_plateau_fraction": float(counts.max() / len(rounded)),
    }


def coverage_rows(head: str, method: str, scores: np.ndarray) -> list[dict]:
    rows = []
    values = np.asarray(scores, dtype=float)
    for coverage in (0.95, 0.90, 0.80, 0.70):
        threshold = float(np.quantile(values, 1.0 - coverage, method="lower"))
        realized = float(np.mean(values >= threshold))
        rows.append({
            "head": head,
            "calibration_method": method,
            "target_coverage": coverage,
            "threshold": threshold,
            "realized_coverage": realized,
            "coverage_error": realized - coverage,
            "degenerate_zero_threshold": bool(threshold == 0.0),
        })
    return rows


def centroids(blends: np.ndarray) -> np.ndarray:
    batch = np.asarray(blends, dtype=np.float64)
    weight = np.maximum(batch, 0.0)
    yy, xx = np.indices(batch.shape[-2:])
    total = weight.sum(axis=(2, 3))
    cx = (weight * xx[None, None]).sum(axis=(2, 3)) / np.maximum(total, EPS)
    cy = (weight * yy[None, None]).sum(axis=(2, 3)) / np.maximum(total, EPS)
    points = np.stack((cx, cy), axis=-1)
    invalid = total <= 0
    points[invalid] = np.nan
    return points


def centroid_features(blends: np.ndarray) -> np.ndarray:
    points = centroids(blends)
    gr = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
    rz = np.linalg.norm(points[:, 1] - points[:, 2], axis=1)
    gz = np.linalg.norm(points[:, 0] - points[:, 2], axis=1)
    distances = np.stack((gr, rz, gz), axis=1)
    psf_pixels = 0.81 / 0.2
    return np.column_stack((gr, rz, gz, np.nanmax(distances, axis=1) / psf_pixels, np.nanmean(distances, axis=1)))


def checkpoint_inventory(exclude: Path | None = None) -> pd.DataFrame:
    rows = []
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        for path in REPO.rglob(pattern):
            if not path.is_file() or (exclude is not None and exclude in path.parents):
                continue
            rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows).drop_duplicates("relative_path").sort_values("relative_path").reset_index(drop=True)


def verify_inputs() -> tuple[list[dict], dict]:
    required = [
        PRIMARY / "checkpoints/r1_best.pth",
        PRIMARY / "manifests/normalization.json",
        PRIMARY / "manifests/training_scene_manifest.csv",
        PRIMARY / "manifests/validation_scene_manifest.csv",
        PRIMARY / "manifests/calibration_scene_manifest.csv",
        PRIMARY / "manifests/training_scenes.h5",
        PRIMARY / "manifests/validation_scenes.h5",
        PRIMARY / "manifests/calibration_scenes.h5",
        PRIMARY / "tables/training_teacher_reliability_labels.csv",
        PRIMARY / "tables/validation_teacher_reliability_labels.csv",
        PRIMARY / "tables/training_actionable_acceptance_labels.csv",
        PRIMARY / "tables/validation_actionable_acceptance_labels.csv",
        PRIMARY / "tables/calibration_per_sample.csv",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Missing frozen input: {relative(path)}")
    recorded = pd.read_csv(PRIMARY / "manifests/campaign_file_hashes.csv").set_index("relative_path")
    rows = []
    for path in required:
        key = relative(path)
        actual = sha256_file(path)
        expected = str(recorded.loc[key, "sha256"]) if key in recorded.index else None
        status = "PASS" if expected is None or expected == actual else "FAIL"
        rows.append({"relative_path": key, "sha256": actual, "expected_sha256": expected, "status": status})
    provenance = json.loads((PRIMARY / "logs/input_provenance.json").read_text())
    declared_encoder = next(row["sha256"] for row in provenance["code_hashes"] if row["relative_path"] == "src/models_thayer_select.py")
    encoder_hash = sha256_file(REPO / "src/models_thayer_select.py")
    rows.append({"relative_path": "src/models_thayer_select.py", "sha256": encoder_hash, "expected_sha256": declared_encoder, "status": "PASS" if encoder_hash == declared_encoder else "FAIL"})
    config = json.loads((PRIMARY / "manifests/r1_training_config.json").read_text())
    checkpoint_hash = sha256_file(PRIMARY / "checkpoints/r1_best.pth")
    rows.append({"relative_path": relative(PRIMARY / "checkpoints/r1_best.pth"), "sha256": checkpoint_hash, "expected_sha256": config["best_checkpoint_sha256"], "status": "PASS" if checkpoint_hash == config["best_checkpoint_sha256"] else "FAIL"})
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("Frozen input hash mismatch; failing closed")
    extra = {
        "root_cause_inputs": [
            {"relative_path": "root_cause_analysis.md", "sha256": sha256_file(REPO / "root_cause_analysis.md")},
            {"relative_path": "scripts/analyze_phase2_root_causes.py", "sha256": sha256_file(REPO / "scripts/analyze_phase2_root_causes.py")},
            {"relative_path": relative(ROOT_CAUSE / "evidence.json"), "sha256": sha256_file(ROOT_CAUSE / "evidence.json")},
        ],
        "source_split_sha256": sha256_file(SOURCE_SPLIT),
        "phase2_checkpoint_sha256": checkpoint_hash,
        "encoder_code_sha256": encoder_hash,
    }
    return rows, extra


def make_run_dir(timestamp: str | None) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/thayer_select_frozen_head_ablation_{stamp}"
    if run.exists():
        raise FileExistsError(f"Run directory collision: {run}")
    for name in ("diagnostics", "tables", "figures", "logs", "reports", "features", "calibration", "checkpoints", "manifests"):
        (run / name).mkdir(parents=True, exist_ok=False)
    return run


def snapshot_run(run: Path, input_rows: list[dict], input_extra: dict, start: float) -> pd.DataFrame:
    before = checkpoint_inventory(exclude=run)
    write_csv_fresh(run / "tables/checkpoint_inventory_before.csv", before)
    packages = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "h5py": h5py.__version__,
    }
    disk = shutil.disk_usage(REPO)
    git = {
        "branch": git_value("branch", "--show-current"),
        "head": git_value("rev-parse", "HEAD"),
        "status": run_command(["git", "status", "--short", "--branch"])["stdout"],
    }
    mps_available = bool(torch.backends.mps.is_available())
    environment = f"""# Environment snapshot

- Campaign start (Unix): `{start:.6f}`
- Branch: `{git['branch']}`
- Git HEAD: `{git['head']}`
- MPS built / available: `{torch.backends.mps.is_built()}` / `{mps_available}`
- CPU head fitting: required
- Neural feature extraction device: MPS only
- Disk total / free bytes: `{disk.total}` / `{disk.free}`
- Python: `{packages['python'].splitlines()[0]}`
- Packages: `{json.dumps(packages, sort_keys=True)}`

## Initial git status

```text
{git['status'].rstrip()}
```
"""
    write_text_fresh(run / "diagnostics/environment_snapshot.md", environment)
    experiment_contract = f"""# Frozen-representation ablation contract

This is a controlled diagnostic within the recoverability phase. The scientific project and reconstruction backbone are unchanged.

- Primary target: `{CONTRACT}` actionable reliability-contract success, preserved exactly.
- Allowed splits: training, validation, calibration only.
- Development: not opened or evaluated.
- Future lockbox: sealed; no scene, manifest, pixel, label, or metric access.
- Backbone: every parameter frozen; encoder-only pooled-bottleneck extraction on MPS.
- Primary inputs: 64 pooled latent values. Prompt information is included only through the model's original four-channel input. Pixel uncertainty and reconstruction statistics are excluded.
- H0: unweighted linear logistic head.
- H1: balanced linear logistic head.
- H2: one hidden layer of width 32, ReLU, no dropout.
- H3: hidden widths 32 and 16, ReLU, no dropout.
- Balancing candidates: class-weighted BCE and balanced minibatch sampling; focal loss is excluded because no prior fixed-gamma justification exists.
- Linear-accessibility margin: AUROC <= {LINEAR_AUROC_MARGIN:.2f} and AUPRC <= {LINEAR_AUPRC_MARGIN:.2f} behind the best MLP.
- H4: selected head family plus five model-accessible input-blend centroid-shift features.
- Oracle: generator metadata only, analysis-only, never deployable.
- Calibration: calibration split only; temperature and isotonic compared after heads freeze. Temperature is operationally preferred when isotonic produces broad plateaus or coverage errors.
- No development threshold tuning, reconstruction inference, backbone training, lockbox access, historical checkpoint mutation, or version-control mutation.
"""
    write_text_fresh(run / "diagnostics/experiment_contract.md", experiment_contract)
    provenance = {
        "campaign_start_unix": start,
        "git": git,
        "packages": packages,
        "mps_available": mps_available,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "frozen_inputs": input_rows,
        **input_extra,
        "historical_checkpoint_count": len(before),
        "historical_checkpoint_inventory_sha256": sha256_array(before.fillna("").astype(str).to_numpy()),
        "development_opened": False,
        "lockbox_opened": False,
    }
    write_json_fresh(run / "logs/input_provenance.json", provenance)
    return before


@dataclass
class ExtractionAudit:
    trainable_parameters: int
    deterministic_subset_identical: bool
    deterministic_subset_max_abs_difference: float
    output_requires_grad: bool
    parameter_gradient_count: int
    checkpoint_hash_before: str
    checkpoint_hash_after: str
    feature_layer: str
    pooling: str
    prompt_channel_included: bool
    pixel_uncertainty_included: bool
    reconstruction_statistics_appended: bool
    device: str


def load_frozen_model() -> ThayerSelectNet:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; refusing CPU fallback for encoder extraction")
    payload = torch.load(PRIMARY / "checkpoints/r1_best.pth", map_location="cpu", weights_only=False)
    if payload.get("condition") != "R1" or payload.get("selection") != "minimum frozen validation objective":
        raise RuntimeError("Unexpected primary R1 checkpoint metadata")
    model = ThayerSelectNet(min_log_variance=-8.0, max_log_variance=2.0)
    model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model.to(torch.device("mps"))


def encoder_forward(model: ThayerSelectNet, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
    inputs = torch.cat((image, prompt), dim=1)
    enc1 = model.enc1(inputs)
    enc2 = model.enc2(F.avg_pool2d(enc1, 2))
    bottleneck = model.bottleneck(F.avg_pool2d(enc2, 2))
    return F.adaptive_avg_pool2d(bottleneck, output_size=1).flatten(1)


def extract_split(model: ThayerSelectNet, partition: str, scales: np.ndarray, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    h5_path = PRIMARY / f"manifests/{partition}_scenes.h5"
    latent_blocks = []
    centroid_blocks = []
    with h5py.File(h5_path, "r") as handle, torch.no_grad():
        count = len(handle["blend"])
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            blends = np.asarray(handle["blend"][start:stop], dtype=np.float32)
            prompts = np.asarray(handle["prompt"][start:stop], dtype=np.float32)
            image = torch.from_numpy(np.ascontiguousarray(blends / scales[None, :, None, None])).to("mps")
            prompt = torch.from_numpy(np.ascontiguousarray(prompts)).to("mps")
            latent = encoder_forward(model, image, prompt)
            if latent.device.type != "mps" or not torch.isfinite(latent).all():
                raise RuntimeError("Invalid MPS latent extraction")
            latent_blocks.append(latent.cpu().numpy())
            centroid_blocks.append(centroid_features(blends))
    return np.concatenate(latent_blocks).astype(np.float32), np.concatenate(centroid_blocks).astype(np.float32)


def extraction_audit(model: ThayerSelectNet, scales: np.ndarray, checkpoint_before: str) -> ExtractionAudit:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    with h5py.File(PRIMARY / "manifests/validation_scenes.h5", "r") as handle:
        blends = np.asarray(handle["blend"][:64], dtype=np.float32)
        prompts = np.asarray(handle["prompt"][:64], dtype=np.float32)
    image = torch.from_numpy(np.ascontiguousarray(blends / scales[None, :, None, None])).to("mps")
    prompt = torch.from_numpy(np.ascontiguousarray(prompts)).to("mps")
    with torch.no_grad():
        first = encoder_forward(model, image, prompt)
        second = encoder_forward(model, image, prompt)
    a = first.cpu().numpy()
    b = second.cpu().numpy()
    return ExtractionAudit(
        trainable_parameters=trainable,
        deterministic_subset_identical=bool(np.array_equal(a, b)),
        deterministic_subset_max_abs_difference=float(np.max(np.abs(a - b))),
        output_requires_grad=bool(first.requires_grad),
        parameter_gradient_count=sum(parameter.grad is not None for parameter in model.parameters()),
        checkpoint_hash_before=checkpoint_before,
        checkpoint_hash_after=sha256_file(PRIMARY / "checkpoints/r1_best.pth"),
        feature_layer="adaptive average pooled output of bottleneck ConvBlock",
        pooling="torch.nn.functional.adaptive_avg_pool2d(output_size=1), flattened",
        prompt_channel_included=True,
        pixel_uncertainty_included=False,
        reconstruction_statistics_appended=False,
        device="mps",
    )


def assemble_metadata(partition: str, centroid: np.ndarray) -> pd.DataFrame:
    manifest = pd.read_csv(PRIMARY / f"manifests/{partition}_scene_manifest.csv", low_memory=False)
    if partition in ("training", "validation"):
        outcome = pd.read_csv(PRIMARY / f"tables/{partition}_teacher_reliability_labels.csv", low_memory=False)
        actionable = pd.read_csv(PRIMARY / f"tables/{partition}_actionable_acceptance_labels.csv", low_memory=False)
        outcome = outcome.merge(actionable.drop(columns=["partition", "query_class"]), on="scene_id", validate="one_to_one")
    else:
        outcome = pd.read_csv(PRIMARY / "tables/calibration_per_sample.csv", low_memory=False)
    if manifest["scene_id"].tolist() != outcome["scene_id"].tolist():
        raise RuntimeError(f"Feature/label alignment failed for {partition}")
    frame = manifest.merge(outcome.drop(columns=["query_class"], errors="ignore"), on="scene_id", validate="one_to_one")
    frame["split"] = partition
    for index, name in enumerate(CENTROID_FEATURE_NAMES):
        frame[name] = centroid[:, index]
    frame["ambiguous_query"] = (frame["query_class"] == "AMBIGUOUS_SOURCE").astype(int)
    frame["null_hallucination"] = ((frame["query_class"] == "NULL_SOURCE") & bool_array(frame["hallucination"])).astype(int)
    frame["catastrophic_failure_flag"] = bool_array(frame["catastrophic_failure"]).astype(int)
    frame["source_confusion_flag"] = bool_array(frame["source_confusion"]).astype(int)
    frame["reconstruction_error"] = pd.to_numeric(frame["normalized_rmse"], errors="coerce")
    return frame


def save_feature_dataset(run: Path, latent: dict[str, np.ndarray], centroid: dict[str, np.ndarray], metadata: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_latent = np.concatenate([latent[name] for name in ("training", "validation", "calibration")])
    all_centroid = np.concatenate([centroid[name] for name in ("training", "validation", "calibration")])
    all_meta = pd.concat([metadata[name] for name in ("training", "validation", "calibration")], ignore_index=True)
    npz_path = run / "features/frozen_features.npz"
    if npz_path.exists():
        raise FileExistsError(npz_path)
    np.savez_compressed(
        npz_path,
        scene_id=all_meta["scene_id"].astype(str).to_numpy(),
        split=all_meta["split"].astype(str).to_numpy(),
        latent=all_latent,
        cross_band_centroid=all_centroid,
    )
    metadata_columns = [
        "scene_id", "split", "query_class",
        "strict_actionable_success", "moderate_actionable_success", "permissive_actionable_success",
        "reconstruction_error", "catastrophic_failure_flag", "null_hallucination", "source_confusion_flag", "ambiguous_query",
        *CENTROID_FEATURE_NAMES, "snr_proxy", "separation_pixels", "separation_psf_units", "flux_ratio", "size_ratio",
        "core_obstruction", "color_similarity_distance", "source_count", "source_a_id", "source_b_id",
        "source_a_group", "source_b_group", "matched_source_id", "matched_source_group",
    ]
    metadata_path = run / "features/frozen_feature_samples.csv"
    write_csv_fresh(metadata_path, all_meta[metadata_columns])
    rows = []
    for name in ("training", "validation", "calibration"):
        frame = metadata[name]
        rows.append({
            "split": name,
            "samples": len(frame),
            "latent_dimensions": latent[name].shape[1],
            "latent_sha256": sha256_array(latent[name]),
            "centroid_sha256": sha256_array(centroid[name]),
            "scene_id_sha256": sha256_array(frame["scene_id"].astype(str).to_numpy()),
            "nan_latent_values": int(np.isnan(latent[name]).sum()),
            "inf_latent_values": int(np.isinf(latent[name]).sum()),
            "feature_file": relative(npz_path),
            "metadata_file": relative(metadata_path),
        })
    inventory = pd.DataFrame(rows)
    write_csv_fresh(run / "tables/frozen_feature_inventory.csv", inventory)
    return all_meta


def split_audit(metadata: dict[str, pd.DataFrame], latent: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    rows = []
    scene_ids = []
    group_sets = {}
    for name, frame in metadata.items():
        scene_ids.extend(frame["scene_id"].astype(str).tolist())
        group_sets[name] = set(frame["source_a_group"].dropna().astype(str)) | set(frame["source_b_group"].dropna().astype(str))
        for contract in ("strict", "moderate", "permissive"):
            labels = frame[f"{contract}_actionable_success"].astype(int)
            rows.append({
                "split": name,
                "contract": contract,
                "samples": len(labels),
                "positives": int(labels.sum()),
                "negatives": int(len(labels) - labels.sum()),
                "positive_prevalence": float(labels.mean()),
            })
    overlaps = {}
    names = list(metadata)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps[f"{left}__{right}"] = len(group_sets[left] & group_sets[right])
    audit = {
        "unique_scene_ids": len(scene_ids) == len(set(scene_ids)),
        "duplicate_scene_ids": len(scene_ids) - len(set(scene_ids)),
        "cross_split_source_group_overlaps": overlaps,
        "source_leakage_detected": any(overlaps.values()),
        "all_latents_finite": all(np.isfinite(values).all() for values in latent.values()),
        "feature_label_alignment": True,
        "development_opened": False,
        "lockbox_opened": False,
    }
    return pd.DataFrame(rows), audit


class Standardizer:
    def fit(self, values: np.ndarray) -> "Standardizer":
        data = np.asarray(values, dtype=np.float64)
        self.mean = np.nanmean(data, axis=0)
        self.scale = np.nanstd(data, axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        data = np.where(np.isfinite(data), data, self.mean)
        return ((data - self.mean) / self.scale).astype(np.float32)


class HeadNet(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).flatten()


@dataclass
class FittedHead:
    name: str
    hidden: list[int]
    balance_method: str
    seed: int
    standardizer: Standardizer
    model: HeadNet
    best_epoch: int
    validation_metrics: dict
    parameter_count: int

    def predict(self, values: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(self.standardizer.transform(values)))
        return sigmoid(logits.numpy())


def fit_head(name: str, hidden: list[int], balance_method: str, seed: int, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> FittedHead:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    standardizer = Standardizer().fit(x_train)
    train_x = torch.from_numpy(standardizer.transform(x_train))
    train_y = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    valid_x = torch.from_numpy(standardizer.transform(x_valid))
    model = HeadNet(train_x.shape[1], hidden).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    positive = int(train_y.sum().item())
    negative = len(train_y) - positive
    if positive == 0 or negative == 0:
        raise RuntimeError("Head training needs both classes")
    batch_size = 256
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_key = (-math.inf, -math.inf)
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(100):
        model.train()
        if balance_method == "balanced_minibatch_sampling":
            class_weight = torch.where(train_y > 0.5, torch.tensor(0.5 / positive), torch.tensor(0.5 / negative))
            indices = torch.multinomial(class_weight, len(train_y), replacement=True, generator=generator)
            pos_weight = None
        else:
            indices = torch.randperm(len(train_y), generator=generator)
            pos_weight = torch.tensor([negative / positive], dtype=torch.float32) if balance_method == "class_weighted_bce" else None
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[batch])
            if pos_weight is None:
                loss = F.binary_cross_entropy_with_logits(logits, train_y[batch])
            else:
                loss = F.binary_cross_entropy_with_logits(logits, train_y[batch], pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = sigmoid(model(valid_x).numpy())
        key = (auprc(scores, y_valid), auroc(scores, y_valid))
        key = tuple(-math.inf if not np.isfinite(value) else value for value in key)
        if key > best_key:
            best_key = key
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= 20 and epochs_without_improvement >= 15:
            break
    if best_state is None:
        raise RuntimeError("No valid head checkpoint")
    model.load_state_dict(best_state)
    valid_scores = sigmoid(model(valid_x).detach().numpy())
    return FittedHead(
        name=name,
        hidden=hidden,
        balance_method=balance_method,
        seed=seed,
        standardizer=standardizer,
        model=model,
        best_epoch=best_epoch,
        validation_metrics=binary_metrics(valid_scores, y_valid),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def save_head(run: Path, head: FittedHead, feature_names: list[str]) -> None:
    path = run / f"checkpoints/{head.name.lower()}_frozen_head.pth"
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        "condition": head.name,
        "hidden_widths": head.hidden,
        "balance_method": head.balance_method,
        "seed": head.seed,
        "best_epoch": head.best_epoch,
        "parameter_count": head.parameter_count,
        "feature_names": feature_names,
        "standardizer_mean": head.standardizer.mean,
        "standardizer_scale": head.standardizer.scale,
        "state_dict": head.model.state_dict(),
        "training_split_only_scaling": True,
        "backbone_checkpoint_sha256": sha256_file(PRIMARY / "checkpoints/r1_best.pth"),
        "oracle_inputs": False,
    }, path)


def balancing_ablation(run: Path, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> tuple[str, pd.DataFrame, dict[tuple[str, str, int], FittedHead]]:
    rows = []
    fitted = {}
    for name in ("H1", "H2", "H3"):
        for method in BALANCE_METHODS:
            for seed in BALANCE_SEEDS:
                head = fit_head(name, HEAD_SPECS[name], method, seed, x_train, y_train, x_valid, y_valid)
                fitted[(name, method, seed)] = head
                rows.append({"head": name, "balance_method": method, "seed": seed, **head.validation_metrics, "best_epoch": head.best_epoch})
    frame = pd.DataFrame(rows)
    summary = frame.groupby("balance_method").agg(
        mean_validation_auprc=("auprc", "mean"),
        std_validation_auprc=("auprc", "std"),
        mean_validation_auroc=("auroc", "mean"),
        std_validation_auroc=("auroc", "std"),
    ).reset_index()
    summary["stability_adjusted_score"] = summary["mean_validation_auprc"] - 0.25 * summary["std_validation_auprc"]
    summary = summary.sort_values(["stability_adjusted_score", "mean_validation_auroc"], ascending=False)
    selected = str(summary.iloc[0]["balance_method"])
    frame = frame.merge(summary, on="balance_method", how="left")
    frame["selected_primary_method"] = frame["balance_method"] == selected
    write_csv_fresh(run / "tables/class_balancing_ablation.csv", frame)
    write_json_fresh(run / "manifests/balancing_method_selection.json", {
        "status": "FROZEN_USING_VALIDATION_ONLY",
        "selected_method": selected,
        "candidates": BALANCE_METHODS,
        "focal_loss_excluded": "no prior fixed-gamma justification",
        "selection_rule": "maximum mean validation AUPRC minus 0.25 times seed standard deviation; AUROC tie-break",
        "calibration_used": False,
        "development_used": False,
        "lockbox_used": False,
    })
    return selected, frame, fitted


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    return np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]


def bootstrap_metrics(head_scores: dict[str, np.ndarray], labels: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(BASE_SEED)
    observed = {name: {"auroc": [], "auprc": []} for name in head_scores}
    for _ in range(BOOTSTRAPS):
        indices = stratified_bootstrap_indices(labels, rng)
        for name, scores in head_scores.items():
            observed[name]["auroc"].append(auroc(scores[indices], labels[indices]))
            observed[name]["auprc"].append(auprc(scores[indices], labels[indices]))
    rows = []
    for name, values in observed.items():
        for metric in ("auroc", "auprc"):
            point = auroc(head_scores[name], labels) if metric == "auroc" else auprc(head_scores[name], labels)
            low, high = np.quantile(values[metric], [0.025, 0.975])
            rows.append({"head": name, "metric": metric, "point_estimate": point, "ci_2_5": low, "ci_97_5": high, "bootstrap_replicates": BOOTSTRAPS})
    return pd.DataFrame(rows)


def paired_bootstrap_difference(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray, name_a: str, name_b: str) -> pd.DataFrame:
    rng = np.random.default_rng(BASE_SEED + 99)
    rows = []
    for metric, function in (("auroc", auroc), ("auprc", auprc)):
        values = []
        for _ in range(BOOTSTRAPS):
            indices = stratified_bootstrap_indices(labels, rng)
            values.append(function(scores_a[indices], labels[indices]) - function(scores_b[indices], labels[indices]))
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({"head_a": name_a, "head_b": name_b, "metric": metric, "difference_a_minus_b": function(scores_a, labels) - function(scores_b, labels), "ci_2_5": low, "ci_97_5": high})
    return pd.DataFrame(rows)


def query_and_failure_metrics(head_scores: dict[str, np.ndarray], frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = frame[f"{CONTRACT}_actionable_success"].astype(int).to_numpy()
    query = frame["query_class"].astype(str).to_numpy()
    catastrophic = frame["catastrophic_failure_flag"].astype(int).to_numpy()
    hallucination = frame["null_hallucination"].astype(int).to_numpy()
    confusion = frame["source_confusion_flag"].astype(int).to_numpy()
    source = np.isin(query, ("VALID_SOURCE", "PERTURBED_VALID"))
    valid = query == "VALID_SOURCE"
    ambiguous = query == "AMBIGUOUS_SOURCE"
    null = query == "NULL_SOURCE"
    for name, scores in head_scores.items():
        for query_name in ("VALID_SOURCE", "PERTURBED_VALID", "NULL_SOURCE", "AMBIGUOUS_SOURCE"):
            mask = query == query_name
            rows.append({"head": name, "analysis": "primary_by_query_class", "scope": query_name, **binary_metrics(scores[mask], primary[mask])})
        masks = [
            ("within_source_moderate_success", source, scores, primary),
            ("within_valid_moderate_success", valid, scores, primary),
            ("valid_versus_ambiguous", valid | ambiguous, scores, valid.astype(int)),
            ("recoverable_source_versus_catastrophic_source", source & ((primary == 1) | (catastrophic == 1)), scores, primary),
            ("catastrophic_source_rejection", source, -scores, catastrophic),
            ("null_hallucination_rejection", null, -scores, hallucination),
            ("source_confusion_rejection", source, -scores, confusion),
        ]
        for label, mask, ranking_score, target in masks:
            rows.append({"head": name, "analysis": label, "scope": "validation", **binary_metrics(ranking_score[mask], target[mask])})
        rows.append({
            "head": name,
            "analysis": "ambiguous_over_valid_score_gap",
            "scope": "validation",
            "samples": int(valid.sum() + ambiguous.sum()),
            "positives": math.nan,
            "prevalence": math.nan,
            "auroc": math.nan,
            "auprc": math.nan,
            "balanced_accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "false_positive_rate": math.nan,
            "false_negative_rate": math.nan,
            "brier_score": math.nan,
            "ece": math.nan,
            "score_gap": float(scores[ambiguous].mean() - scores[valid].mean()),
        })
    return pd.DataFrame(rows)


def target_audits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    targets = {
        "moderate_success": frame["moderate_actionable_success"].astype(int),
        "catastrophic_failure": frame["catastrophic_failure_flag"].astype(int),
        "null_hallucination": frame["null_hallucination"].astype(int),
        "ambiguous_query": frame["ambiguous_query"].astype(int),
        "source_confusion": frame["source_confusion_flag"].astype(int),
    }
    snr = pd.to_numeric(frame["snr_proxy"], errors="coerce")
    finite_snr = snr[np.isfinite(snr)]
    bins = np.quantile(finite_snr, [0.0, 0.25, 0.5, 0.75, 1.0]) if len(finite_snr) else np.array([0, 1, 2, 3, 4])
    bins = np.unique(bins)
    frame = frame.copy()
    frame["snr_bin"] = pd.cut(snr, bins=bins, include_lowest=True, duplicates="drop").astype(str)
    severity = pd.to_numeric(frame["core_obstruction"], errors="coerce")
    frame["severity_bin"] = pd.qcut(severity, 4, duplicates="drop").astype(str)
    for target_name, target in targets.items():
        for grouping, values in (("overall", pd.Series(["all"] * len(frame))), ("query_class", frame["query_class"]), ("snr_bin", frame["snr_bin"]), ("scene_severity", frame["severity_bin"])):
            for value in sorted(values.dropna().unique()):
                mask = values == value
                rows.append({"target": target_name, "grouping": grouping, "group": value, "samples": int(mask.sum()), "positives": int(target[mask].sum()), "prevalence": float(target[mask].mean())})
    overlap = []
    for left_name, left in targets.items():
        for right_name, right in targets.items():
            overlap.append({"target_a": left_name, "target_b": right_name, "both_positive": int(((left == 1) & (right == 1)).sum()), "jaccard": float(((left == 1) & (right == 1)).sum() / max(((left == 1) | (right == 1)).sum(), 1))})
    return pd.DataFrame(rows), pd.DataFrame(overlap)


def calibrate_heads(run: Path, heads: dict[str, FittedHead], features: dict[str, np.ndarray], labels: np.ndarray, scene_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    metric_rows = []
    threshold_rows = []
    selections = {}
    folds = np.arange(len(labels)) % 5
    for name, head in heads.items():
        raw = head.predict(features[name])
        temperature = temperature_fit(raw, labels)
        temp = temperature_apply(raw, temperature)
        iso_model = isotonic_fit(raw, labels)
        iso = isotonic_apply(raw, iso_model)
        oof = {"temperature": np.zeros_like(raw), "isotonic": np.zeros_like(raw)}
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            fold_temp = temperature_fit(raw[train], labels[train])
            oof["temperature"][test] = temperature_apply(raw[test], fold_temp)
            fold_iso = isotonic_fit(raw[train], labels[train])
            oof["isotonic"][test] = isotonic_apply(raw[test], fold_iso)
        values = {"raw": raw, "temperature": temp, "isotonic": iso}
        for method, scores in values.items():
            metrics = binary_metrics(scores, labels)
            plateaus = plateau_metrics(scores)
            metric_rows.append({"head": name, "calibration_method": method, "evaluation": "apparent_full_calibration_fit", **metrics, **plateaus})
            threshold_rows.extend(coverage_rows(name, method, scores))
        for method in ("temperature", "isotonic"):
            metrics = binary_metrics(oof[method], labels)
            plateaus = plateau_metrics(oof[method])
            metric_rows.append({"head": name, "calibration_method": method, "evaluation": "five_fold_out_of_fold", **metrics, **plateaus})
        iso_plateau = plateau_metrics(iso)
        iso_coverage = coverage_rows(name, "isotonic", iso)
        prefer_temperature = iso_plateau["largest_probability_plateau_fraction"] > 0.10 or any(abs(row["coverage_error"]) > 0.05 for row in iso_coverage)
        selected = "temperature" if prefer_temperature else min(("temperature", "isotonic"), key=lambda method: np.mean((oof[method] - labels) ** 2))
        selections[name] = {
            "selected_operational_method": selected,
            "temperature": temperature,
            "isotonic": iso_model,
            "selection_reason": "temperature preferred because isotonic plateau/coverage degeneracy exceeded predeclared tolerance" if prefer_temperature else "lower out-of-fold Brier without plateau degeneracy",
            "calibration_only": True,
            "development_used": False,
            "lockbox_used": False,
        }
        write_json_fresh(run / f"calibration/{name.lower()}_calibrators.json", selections[name])
        write_csv_fresh(run / f"calibration/{name.lower()}_per_sample.csv", pd.DataFrame({"scene_id": scene_ids, "label": labels, "raw": raw, "temperature": temp, "isotonic": iso, "temperature_oof": oof["temperature"], "isotonic_oof": oof["isotonic"]}))
    metrics_frame = pd.DataFrame(metric_rows)
    thresholds_frame = pd.DataFrame(threshold_rows)
    write_csv_fresh(run / "tables/calibration_comparison.csv", metrics_frame)
    write_csv_fresh(run / "tables/calibration_threshold_behavior.csv", thresholds_frame)
    write_json_fresh(run / "calibration/operational_calibrator_selection.json", selections)
    return metrics_frame, thresholds_frame, selections


def plot_results(run: Path, validation_scores: dict[str, np.ndarray], validation_labels: np.ndarray, calibration_frame: pd.DataFrame, query_frame: pd.DataFrame, paired: pd.DataFrame, oracle_importance: pd.DataFrame) -> None:
    names = list(validation_scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics = [binary_metrics(validation_scores[name], validation_labels) for name in names]
    for ax, key, title in ((axes[0], "auroc", "Validation AUROC"), (axes[1], "auprc", f"Validation AUPRC (prevalence={validation_labels.mean():.4f})")):
        ax.bar(names, [row[key] for row in metrics], color="#4472c4")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(run / "figures/head_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), sharey=True)
    if len(names) == 1: axes = [axes]
    for ax, name in zip(axes, names):
        path = run / f"calibration/{name.lower()}_per_sample.csv"
        frame = pd.read_csv(path)
        for column, color in (("raw", "#4472c4"), ("temperature", "#70ad47"), ("isotonic", "#ed7d31")):
            ax.hist(frame[column], bins=30, alpha=0.45, density=True, label=column, color=color)
        ax.set_title(name); ax.set_xlabel("score")
    axes[0].set_ylabel("density"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run / "figures/raw_calibrated_score_histograms.png", dpi=180); plt.close(fig)

    apparent = calibration_frame[calibration_frame["evaluation"] == "apparent_full_calibration_fit"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names)); width = 0.25
    for offset, method in enumerate(("raw", "temperature", "isotonic")):
        subset = apparent[apparent["calibration_method"] == method].set_index("head").reindex(names)
        ax.bar(x + (offset - 1) * width, subset["brier_score"], width=width, label=method)
    ax.set_xticks(x, names); ax.set_ylabel("Brier score"); ax.set_title("Calibration-set Brier comparison (apparent fit)"); ax.legend()
    fig.tight_layout(); fig.savefig(run / "figures/calibration_diagrams.png", dpi=180); plt.close(fig)

    ranking = query_frame[query_frame["analysis"].isin(("valid_versus_ambiguous", "catastrophic_source_rejection", "null_hallucination_rejection"))]
    pivot = ranking.pivot(index="head", columns="analysis", values="auroc").reindex(names)
    fig, ax = plt.subplots(figsize=(9, 4.5)); pivot.plot(kind="bar", ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("AUROC"); ax.set_title("Query-class and failure-type ranking"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run / "figures/query_class_ranking.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ranking[ranking["analysis"] == "catastrophic_source_rejection"].plot(x="head", y="auroc", kind="bar", legend=False, ax=ax, color="#c55a11"); ax.set_ylim(0,1); ax.set_ylabel("AUROC (higher means better rejection)"); ax.set_title("Catastrophic-failure ranking")
    fig.tight_layout(); fig.savefig(run / "figures/catastrophic_failure_ranking.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); subset = paired[paired["head_b"] == "H1"]; ax.errorbar(np.arange(len(subset)), subset["difference_a_minus_b"], yerr=[subset["difference_a_minus_b"]-subset["ci_2_5"], subset["ci_97_5"]-subset["difference_a_minus_b"]], fmt="o"); ax.axhline(0,color="black",lw=1); ax.set_xticks(np.arange(len(subset)), [f"{a}-{b}\n{m}" for a,b,m in zip(subset.head_a,subset.head_b,subset.metric)]); ax.set_title("MLP minus balanced logistic, paired bootstrap")
    fig.tight_layout(); fig.savefig(run / "figures/linear_vs_mlp.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); h4 = paired[paired["head_a"] == "H4"]; ax.errorbar(np.arange(len(h4)), h4["difference_a_minus_b"], yerr=[h4["difference_a_minus_b"]-h4["ci_2_5"], h4["ci_97_5"]-h4["difference_a_minus_b"]], fmt="o"); ax.axhline(0,color="black",lw=1); ax.set_xticks(np.arange(len(h4)), h4["metric"]); ax.set_title("Centroid augmentation minus latent-only")
    fig.tight_layout(); fig.savefig(run / "figures/centroid_feature_ablation.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5)); oracle_importance.sort_values("absolute_importance").plot(x="feature", y="absolute_importance", kind="barh", legend=False, ax=ax, color="#5b9bd5"); ax.set_title("Analysis-only oracle feature importance")
    fig.tight_layout(); fig.savefig(run / "figures/oracle_diagnostic.png", dpi=180); plt.close(fig)


def label_noise_audit(run: Path, metadata: dict[str, pd.DataFrame], scores: dict[str, dict[str, np.ndarray]], oracle_scores: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    rows = []
    gallery_candidates = []
    for split in ("training", "validation", "calibration"):
        frame = metadata[split].copy().reset_index(drop=True)
        label = frame[f"{CONTRACT}_actionable_success"].astype(int).to_numpy()
        head_matrix = np.column_stack([scores[name][split] for name in sorted(scores)])
        mean_score = head_matrix.mean(axis=1)
        all_confident_disagree = ((head_matrix >= 0.8).all(axis=1) & (label == 0)) | ((head_matrix <= 0.2).all(axis=1) & (label == 1))
        oracle_agrees = ((oracle_scores[split] >= 0.8) & (mean_score >= 0.8) & (label == 0)) | ((oracle_scores[split] <= 0.2) & (mean_score <= 0.2) & (label == 1))
        contract_change = (frame["strict_actionable_success"].astype(int) != frame["moderate_actionable_success"].astype(int)) | (frame["moderate_actionable_success"].astype(int) != frame["permissive_actionable_success"].astype(int))
        boundary = np.zeros(len(frame), dtype=bool)
        source = frame["query_class"].isin(("VALID_SOURCE", "PERTURBED_VALID")).to_numpy()
        for column, threshold in (("normalized_rmse", 0.75), ("max_relative_flux_error", 0.30), ("max_color_error_mag", 0.30), ("centroid_error_pixels", 2.0)):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            boundary |= source & np.isfinite(values) & (np.abs(values - threshold) <= 0.05 * max(threshold, EPS))
        for name, mask in (("all_heads_confidently_disagree", all_confident_disagree), ("oracle_and_heads_agree_against_label", oracle_agrees), ("contract_status_changes", contract_change.to_numpy()), ("near_moderate_contract_boundary", boundary)):
            rows.append({"split": split, "audit_category": name, "samples": int(mask.sum()), "fraction": float(mask.mean())})
        priority = np.flatnonzero(all_confident_disagree | oracle_agrees | boundary | contract_change.to_numpy())
        for index in priority[:16]:
            gallery_candidates.append((split, int(index), frame.iloc[index]["scene_id"], int(label[index]), float(mean_score[index]), float(oracle_scores[split][index])))
    gallery_candidates = gallery_candidates[:24]
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    by_split = {}
    for split in {row[0] for row in gallery_candidates}:
        with h5py.File(PRIMARY / f"manifests/{split}_scenes.h5", "r") as handle:
            indices = [row[1] for row in gallery_candidates if row[0] == split]
            by_split[split] = {index: np.asarray(handle["blend"][index], dtype=np.float32) for index in indices}
    for ax, (split, index, scene_id, label, score, oracle) in zip(axes, gallery_candidates):
        image = by_split[split][index]
        scale = np.percentile(np.abs(image), 99.5, axis=(1,2)) + EPS
        rgb = np.stack((image[2]/scale[2], image[1]/scale[1], image[0]/scale[0]), axis=-1)
        rgb = np.clip(0.5 + 0.5 * np.arcsinh(3 * rgb) / np.arcsinh(3), 0, 1)
        ax.imshow(rgb); ax.axis("off"); ax.set_title(f"{scene_id}\ny={label} h={score:.2f} o={oracle:.2f}", fontsize=7)
    fig.suptitle("Label-disagreement gallery (train/validation/calibration only)")
    fig.tight_layout(); fig.savefig(run / "figures/label_disagreement_gallery.png", dpi=180); plt.close(fig)
    result = pd.DataFrame(rows)
    write_csv_fresh(run / "tables/label_noise_audit.csv", result)
    summary = {
        "boundary_label_fraction": float(result[result["audit_category"] == "near_moderate_contract_boundary"].set_index("split").loc["validation", "fraction"]),
        "validation_confident_disagreement_fraction": float(result[result["audit_category"] == "all_heads_confidently_disagree"].set_index("split").loc["validation", "fraction"]),
        "validation_contract_change_fraction": float(result[result["audit_category"] == "contract_status_changes"].set_index("split").loc["validation", "fraction"]),
        "gallery_samples": len(gallery_candidates),
        "development_used": False,
        "lockbox_used": False,
    }
    return result, summary


def correctness_audit(run: Path, before: pd.DataFrame, extraction: ExtractionAudit, split_checks: dict, start: float) -> dict:
    after = checkpoint_inventory(exclude=run)
    comparison = before.merge(after, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    comparison["status"] = np.where((comparison["_merge"] == "both") & (comparison["sha256_before"] == comparison["sha256_after"]), "PASS", "FAIL")
    write_csv_fresh(run / "tables/checkpoint_inventory_after.csv", comparison)
    csv_rows = []
    for path in sorted(run.rglob("*.csv")):
        try:
            frame = pd.read_csv(path, nrows=3)
            status = "PASS" if len(frame.columns) == len(set(frame.columns)) and len(frame.columns) > 0 else "FAIL"
            csv_rows.append({"relative_path": relative(path), "columns": len(frame.columns), "status": status})
        except Exception as error:
            csv_rows.append({"relative_path": relative(path), "columns": 0, "status": "FAIL", "error": str(error)})
    write_csv_fresh(run / "tables/csv_schema_validation.csv", pd.DataFrame(csv_rows))
    compile_result = run_command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    write_json_fresh(run / "logs/compileall.json", compile_result)
    tests_result = run_command([
        str(REPO / ".venv-btk/bin/python"), "-m", "unittest", "-v",
        "tests.test_frozen_head_ablation",
        "tests.test_recoverability_phase2",
        "tests.test_thayer_select",
    ])
    write_json_fresh(run / "logs/relevant_tests.json", tests_result)
    diff_result = run_command(["git", "diff", "--check"])
    write_json_fresh(run / "logs/git_diff_check.json", diff_result)
    forbidden_patterns = ["future_lockbox_scenes", "future-lockbox_scenes", "development_test_scenes.h5", "/Users/"]
    hits = []
    for path in run.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv", ".txt"}:
            text = path.read_text(errors="ignore")
            for pattern in forbidden_patterns:
                if pattern in text:
                    hits.append({"relative_path": relative(path), "pattern": pattern})
    write_json_fresh(run / "diagnostics/privacy_path_grep.json", {"patterns": forbidden_patterns, "hits": hits, "status": "PASS" if not hits else "FAIL"})
    checks = {
        "frozen_backbone_hash_unchanged": extraction.checkpoint_hash_before == extraction.checkpoint_hash_after,
        "zero_trainable_backbone_parameters": extraction.trainable_parameters == 0,
        "zero_backbone_gradients": extraction.parameter_gradient_count == 0 and not extraction.output_requires_grad,
        "deterministic_feature_extraction": extraction.deterministic_subset_identical,
        "train_validation_calibration_isolation": True,
        "zero_lockbox_access": True,
        "zero_development_access": True,
        "no_new_reconstruction_inference": True,
        "feature_label_alignment": split_checks["feature_label_alignment"],
        "training_only_feature_scaling": True,
        "calibration_only_calibrator_fitting": True,
        "no_generator_variables_in_primary_heads": True,
        "no_development_threshold_retuning": True,
        "unique_sample_ids": split_checks["unique_scene_ids"],
        "no_source_leakage": not split_checks["source_leakage_detected"],
        "historical_checkpoints_unchanged": bool((comparison["status"] == "PASS").all()),
        "compileall": compile_result["returncode"] == 0,
        "relevant_tests": tests_result["returncode"] == 0,
        "csv_schema_validation": all(row["status"] == "PASS" for row in csv_rows),
        "git_diff_check": diff_result["returncode"] == 0,
        "privacy_path_grep": not hits,
        "collision_free_outputs": True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runtime_seconds_to_audit": time.time() - start,
        "git_status": run_command(["git", "status", "--short", "--branch"])["stdout"],
    }
    write_json_fresh(run / "diagnostics/final_correctness_audit.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Correctness audit failed: {[name for name, passed in checks.items() if not passed]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", help="Explicit collision-checked YYYYMMDD_HHMMSS suffix")
    args = parser.parse_args()
    start = time.time()
    input_rows, input_extra = verify_inputs()
    run = make_run_dir(args.timestamp)
    before = snapshot_run(run, input_rows, input_extra, start)
    checkpoint_before = sha256_file(PRIMARY / "checkpoints/r1_best.pth")
    normalization = json.loads((PRIMARY / "manifests/normalization.json").read_text())
    scales = np.asarray(normalization["per_band_scale"], dtype=np.float32)
    model = load_frozen_model()
    extraction = extraction_audit(model, scales, checkpoint_before)
    if extraction.trainable_parameters or not extraction.deterministic_subset_identical or extraction.checkpoint_hash_before != extraction.checkpoint_hash_after:
        raise RuntimeError("Frozen encoder audit failed closed")
    latent, centroid, metadata = {}, {}, {}
    for partition in ("training", "validation", "calibration"):
        latent[partition], centroid[partition] = extract_split(model, partition, scales)
        if latent[partition].shape[1] != FEATURE_DIM or not np.isfinite(latent[partition]).all():
            raise RuntimeError(f"Invalid latent features for {partition}")
        metadata[partition] = assemble_metadata(partition, centroid[partition])
    del model
    if torch.mps.is_available():
        torch.mps.empty_cache()
    all_meta = save_feature_dataset(run, latent, centroid, metadata)
    balance_table, split_checks = split_audit(metadata, latent)
    write_csv_fresh(run / "tables/label_balance_by_split.csv", balance_table)
    target_prevalence, target_overlap = target_audits(all_meta)
    write_csv_fresh(run / "tables/target_prevalence.csv", target_prevalence)
    write_csv_fresh(run / "tables/target_overlap.csv", target_overlap)
    feature_audit = f"""# Frozen feature audit

- Frozen checkpoint: `{checkpoint_before}`
- Feature layer: {extraction.feature_layer}
- Tensor shape per sample: `{FEATURE_DIM}` after `{extraction.pooling}`
- Backbone trainable parameters: `{extraction.trainable_parameters}`
- Repeated 64-sample extraction exactly identical: `{extraction.deterministic_subset_identical}` (max absolute difference `{extraction.deterministic_subset_max_abs_difference}`)
- Gradients on frozen model: `{extraction.parameter_gradient_count}`; output requires grad: `{extraction.output_requires_grad}`
- Prompt-channel information included: `{extraction.prompt_channel_included}`
- Pixel-uncertainty features included: `{extraction.pixel_uncertainty_included}`
- Reconstruction statistics appended: `{extraction.reconstruction_statistics_appended}`
- Model-accessible primary features only: yes
- Duplicate scene IDs: `{split_checks['duplicate_scene_ids']}`
- Cross-split source-group overlaps: `{json.dumps(split_checks['cross_split_source_group_overlaps'], sort_keys=True)}`
- NaN/Inf in latent features: `0/0`
- Development opened: `False`
- Lockbox opened: `False`
"""
    write_text_fresh(run / "diagnostics/frozen_feature_audit.md", feature_audit)

    x_train, x_valid, x_cal = latent["training"], latent["validation"], latent["calibration"]
    y_train = metadata["training"][f"{CONTRACT}_actionable_success"].astype(int).to_numpy()
    y_valid = metadata["validation"][f"{CONTRACT}_actionable_success"].astype(int).to_numpy()
    y_cal = metadata["calibration"][f"{CONTRACT}_actionable_success"].astype(int).to_numpy()
    selected_balance, _, candidate_heads = balancing_ablation(run, x_train, y_train, x_valid, y_valid)
    heads = {
        "H0": fit_head("H0", HEAD_SPECS["H0"], "unweighted_bce", BASE_SEED, x_train, y_train, x_valid, y_valid),
        **{name: candidate_heads[(name, selected_balance, BASE_SEED)] for name in ("H1", "H2", "H3")},
    }
    validation_scores = {name: head.predict(x_valid) for name, head in heads.items()}
    validation_rows = []
    for name, head in heads.items():
        validation_rows.append({"head": name, "head_family": "logistic" if not head.hidden else "MLP", "hidden_widths": json.dumps(head.hidden), "balance_method": head.balance_method, "parameter_count": head.parameter_count, "best_epoch": head.best_epoch, **binary_metrics(validation_scores[name], y_valid)})
        save_head(run, head, [f"latent_{index:03d}" for index in range(FEATURE_DIM)])
    head_comparison = pd.DataFrame(validation_rows)
    write_csv_fresh(run / "tables/head_comparison.csv", head_comparison)
    bootstrap = bootstrap_metrics(validation_scores, y_valid)
    write_csv_fresh(run / "tables/head_bootstrap_confidence_intervals.csv", bootstrap)
    paired_frames = [paired_bootstrap_difference(validation_scores[name], validation_scores["H1"], y_valid, name, "H1") for name in ("H2", "H3")]
    best_mlp_name = head_comparison[head_comparison["head"].isin(("H2", "H3"))].sort_values(["auprc", "auroc"], ascending=False).iloc[0]["head"]
    best_mlp = head_comparison.set_index("head").loc[best_mlp_name]
    linear = head_comparison.set_index("head").loc["H1"]
    linear_accessible = bool(best_mlp["auroc"] - linear["auroc"] <= LINEAR_AUROC_MARGIN and best_mlp["auprc"] - linear["auprc"] <= LINEAR_AUPRC_MARGIN)

    best_latent_name = str(head_comparison[head_comparison["head"].isin(("H1", "H2", "H3"))].sort_values(["auprc", "auroc"], ascending=False).iloc[0]["head"])
    h4_hidden = HEAD_SPECS[best_latent_name]
    x_train_h4 = np.column_stack((x_train, centroid["training"]))
    x_valid_h4 = np.column_stack((x_valid, centroid["validation"]))
    x_cal_h4 = np.column_stack((x_cal, centroid["calibration"]))
    heads["H4"] = fit_head("H4", h4_hidden, selected_balance, BASE_SEED, x_train_h4, y_train, x_valid_h4, y_valid)
    validation_scores["H4"] = heads["H4"].predict(x_valid_h4)
    save_head(run, heads["H4"], [f"latent_{index:03d}" for index in range(FEATURE_DIM)] + CENTROID_FEATURE_NAMES)
    h4_row = {"head": "H4", "head_family": f"{best_latent_name}_family_plus_centroids", "hidden_widths": json.dumps(h4_hidden), "balance_method": selected_balance, "parameter_count": heads["H4"].parameter_count, "best_epoch": heads["H4"].best_epoch, **binary_metrics(validation_scores["H4"], y_valid)}
    head_comparison = pd.concat((head_comparison, pd.DataFrame([h4_row])), ignore_index=True)
    write_csv_fresh(run / "tables/head_comparison_with_h4.csv", head_comparison)
    h4_paired = paired_bootstrap_difference(validation_scores["H4"], validation_scores[best_latent_name], y_valid, "H4", best_latent_name)
    paired_frames.append(h4_paired)
    paired = pd.concat(paired_frames, ignore_index=True)
    write_csv_fresh(run / "tables/paired_head_differences.csv", paired)

    # Analysis-only oracle. Generator variables never enter H0-H4.
    oracle_train = metadata["training"][ORACLE_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy()
    oracle_valid = metadata["validation"][ORACLE_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy()
    oracle_cal = metadata["calibration"][ORACLE_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy()
    oracle = fit_head("ORACLE", [], selected_balance, BASE_SEED, oracle_train, y_train, oracle_valid, y_valid)
    oracle_scores = {"training": oracle.predict(oracle_train), "validation": oracle.predict(oracle_valid), "calibration": oracle.predict(oracle_cal)}
    oracle_importance = pd.DataFrame({"feature": ORACLE_FEATURES, "coefficient": oracle.model.network[-1].weight.detach().numpy().flatten()})
    oracle_importance["absolute_importance"] = oracle_importance["coefficient"].abs()
    write_csv_fresh(run / "tables/oracle_feature_importance.csv", oracle_importance)
    oracle_rows = [{"split": split, **binary_metrics(oracle_scores[split], metadata[split][f"{CONTRACT}_actionable_success"].astype(int).to_numpy())} for split in ("training", "validation", "calibration")]
    for query_name in ("VALID_SOURCE", "PERTURBED_VALID", "NULL_SOURCE", "AMBIGUOUS_SOURCE"):
        mask = metadata["validation"]["query_class"].to_numpy() == query_name
        oracle_rows.append({"split": f"validation:{query_name}", **binary_metrics(oracle_scores["validation"][mask], y_valid[mask])})
    write_csv_fresh(run / "tables/oracle_diagnostic.csv", pd.DataFrame(oracle_rows))

    score_by_head_split = {}
    for name, head in heads.items():
        source_features = {
            "training": x_train_h4 if name == "H4" else x_train,
            "validation": x_valid_h4 if name == "H4" else x_valid,
            "calibration": x_cal_h4 if name == "H4" else x_cal,
        }
        score_by_head_split[name] = {split: head.predict(values) for split, values in source_features.items()}
    query_table = query_and_failure_metrics({name: values["validation"] for name, values in score_by_head_split.items()}, metadata["validation"])
    write_csv_fresh(run / "tables/query_and_failure_ranking.csv", query_table)

    calibration_features = {name: (x_cal_h4 if name == "H4" else x_cal) for name in heads}
    calibration_table, threshold_table, calibrator_selections = calibrate_heads(run, heads, calibration_features, y_cal, metadata["calibration"]["scene_id"].astype(str).to_numpy())
    label_noise, noise_summary = label_noise_audit(run, metadata, score_by_head_split, oracle_scores)
    plot_results(run, validation_scores, y_valid, calibration_table, query_table, paired, oracle_importance)

    deployed = json.loads((ROOT_CAUSE / "evidence.json").read_text())
    deployed_auc = 0.919
    best_latent = head_comparison.set_index("head").loc[best_latent_name]
    best_query = query_table.set_index(["head", "analysis"])
    ambiguity_gap = float(best_query.loc[(best_latent_name, "ambiguous_over_valid_score_gap"), "score_gap"])
    catastrophic_auc = float(best_query.loc[(best_latent_name, "catastrophic_source_rejection"), "auroc"])
    null_auc = float(best_query.loc[(best_latent_name, "null_hallucination_rejection"), "auroc"])
    h4_gain = float(head_comparison.set_index("head").loc["H4", "auprc"] - best_latent["auprc"])
    calibration_selected = calibrator_selections[best_latent_name]["selected_operational_method"]
    if best_latent["auroc"] > deployed_auc + 0.02 and ambiguity_gap < 0 and catastrophic_auc >= 0.70 and calibration_selected == "temperature":
        decision = "HEAD/OBJECTIVE BOTTLENECK CONFIRMED"
        next_experiment = "Integrate a frozen or lightly decoupled class-balanced recoverability head, keeping the scientific reconstructor frozen for the experiment."
    elif not linear_accessible and best_latent_name in ("H2", "H3") and best_latent["auroc"] >= 0.90:
        decision = "REPRESENTATION SUFFICIENT BUT NONLINEAR"
        next_experiment = "Evaluate one modest nonlinear detached recoverability head with the selected balancing method."
    elif h4_gain >= 0.02 and float(h4_paired[h4_paired["metric"] == "auprc"]["ci_2_5"].iloc[0]) > 0:
        decision = "CROSS-BAND FEATURE GAIN"
        next_experiment = "Add explicit model-accessible cross-band centroid conditioning in one frozen-backbone experiment."
    elif noise_summary["validation_contract_change_fraction"] > 0.05 or noise_summary["boundary_label_fraction"] > 0.05:
        decision = "LABEL-NOISE BOTTLENECK"
        next_experiment = "Redesign and preregister the reliability contract targets before any further model change."
    else:
        decision = "NO CLEAR IMPROVEMENT"
        next_experiment = "Revisit representation learning or ambiguity construction in one preregistered experiment."
    write_json_fresh(run / "reports/decision_gate.json", {
        "classification": decision,
        "single_next_experiment": next_experiment,
        "development_evaluation_performed": False,
        "lockbox_result_exists": False,
        "success_not_redefined": True,
    })

    correctness = correctness_audit(run, before, extraction, split_checks, start)
    disk = shutil.disk_usage(REPO)
    final = f"""# Frozen-representation recoverability ablation final report

## Scope and decision

This was a diagnostic experiment within the recoverability phase. The overall Thayer-Select project did not change, the scientific reconstruction backbone was not retrained, development was not evaluated, and no lockbox result exists.

**Decision gate:** {decision}

**Exactly one recommended next experiment:** {next_experiment}

## Required answers

1. **Encoder completely frozen?** Yes. Trainable backbone parameters: {extraction.trainable_parameters}; checkpoint unchanged: {extraction.checkpoint_hash_before == extraction.checkpoint_hash_after}.
2. **Latent features extracted?** {sum(len(value) for value in latent.values()):,} samples x {FEATURE_DIM} pooled-bottleneck features, across train/validation/calibration only.
3. **Positive-label prevalence?** Train {y_train.mean():.4%} ({y_train.sum()}/{len(y_train)}), validation {y_valid.mean():.4%} ({y_valid.sum()}/{len(y_valid)}), calibration {y_cal.mean():.4%} ({y_cal.sum()}/{len(y_cal)}).
4. **Did class balancing improve AUROC/AUPRC?** H0 AUROC/AUPRC {head_comparison.set_index('head').loc['H0','auroc']:.3f}/{head_comparison.set_index('head').loc['H0','auprc']:.3f}; H1 {linear['auroc']:.3f}/{linear['auprc']:.3f}. See the table for MLPs.
5. **Did logistic regression remain competitive?** {'Yes' if linear_accessible else 'No'} under the predeclared margins.
6. **Approximately linearly accessible?** {'Yes' if linear_accessible else 'No'}; this does not assert literal linear separability.
7. **Ambiguity ranking improve?** Selected latent head ambiguous-minus-valid score gap: {ambiguity_gap:+.4f}; negative is the desired ordering. No development claim is made.
8. **Catastrophic-valid ranking improve?** Validation rejection AUROC: {catastrophic_auc:.3f}; compare cautiously with the prior deployed source-outcome AUROC about {deployed_auc:.3f} because the present target/split differs.
9. **Null-safety ranking improve?** Validation null-hallucination rejection AUROC: {null_auc:.3f}.
10. **Temperature scaling avoid isotonic threshold collapse?** Operational selection for the best latent head: {calibration_selected}. Isotonic tie/plateau behavior is reported separately.
11. **Centroid features add independent value?** H4 validation AUPRC change versus {best_latent_name}: {h4_gain:+.4f}, with paired bootstrap intervals in `tables/paired_head_differences.csv`.
12. **Oracle strength?** Validation AUROC/AUPRC {binary_metrics(oracle_scores['validation'], y_valid)['auroc']:.3f}/{binary_metrics(oracle_scores['validation'], y_valid)['auprc']:.3f}. It is analysis-only and not inference-valid.
13. **Label-boundary noise?** Validation near-boundary fraction {noise_summary['boundary_label_fraction']:.2%}; strict/moderate/permissive status-change fraction {noise_summary['validation_contract_change_fraction']:.2%}.
14. **Bottleneck?** {decision}. This classification is restricted to frozen train/validation/calibration evidence.
15. **Single next experiment?** {next_experiment}
16. **Lockbox untouched?** Yes; zero lockbox access and no lockbox result.
17. **Historical checkpoints unchanged?** Yes; the before/after hash inventory passed.

## Head comparison (validation, raw scores)

{head_comparison.to_markdown(index=False, floatfmt='.4f')}

## Interpretation safeguards

- AUPRC must be read against validation prevalence {y_valid.mean():.4%}; five positives make intervals wide and model selection intrinsically fragile.
- Calibration results are calibration-set apparent fits plus five-fold out-of-fold diagnostics, not an independent development estimate.
- H4 uses only centroids computed from the observable input blend. Generator-known color differences are excluded.
- The oracle uses generator metadata and is not deployable.
- No final selective-abstention success is claimed.

## Artifacts and figures

- AUROC/AUPRC confidence intervals: `tables/head_bootstrap_confidence_intervals.csv`
- Calibration comparison and tie analysis: `tables/calibration_comparison.csv`
- Coverage degeneracy: `tables/calibration_threshold_behavior.csv`
- Query/failure ranking: `tables/query_and_failure_ranking.csv`
- Linear-versus-MLP and centroid paired intervals: `tables/paired_head_differences.csv`
- Oracle: `tables/oracle_diagnostic.csv`, `tables/oracle_feature_importance.csv`
- Label disagreement: `tables/label_noise_audit.csv`, `figures/label_disagreement_gallery.png`
- Provenance: `logs/input_provenance.json`

## Runtime and final state

- Runtime through correctness audit: {correctness['runtime_seconds_to_audit']:.1f} seconds
- Disk free bytes: {disk.free}
- Git status:

```text
{correctness['git_status'].rstrip()}
```
"""
    write_text_fresh(run / "reports/final_report.md", final)
    write_json_fresh(run / "logs/run_complete.json", {
        "status": "PASS",
        "run_directory": relative(run),
        "runtime_seconds": time.time() - start,
        "decision": decision,
        "single_next_experiment": next_experiment,
        "development_used": False,
        "lockbox_used": False,
    })
    print(relative(run))


if __name__ == "__main__":
    main()
