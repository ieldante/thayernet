#!/usr/bin/env python3
"""Train Thayer-BR v0.2 Moderate on immutable grouped blend manifests.

This entry point is intentionally fail-closed. It verifies the grouped source
partition, cryptographic manifest provenance, and exact replay hashes before
constructing a model. Full training requires MPS or CUDA; CPU is supported
only by ``--loss-self-test-only`` for a tiny analytic loss check.

The manifest rows are replayed once into a disk-backed cache. This avoids both
holding all float32 samples in host memory and regenerating 9,000 blends for
every epoch. Historical checkpoints are inventoried before and after the run
and are never opened for writing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import blend as gd_blend
from src import models
from src import train as gd_train
from src import utils as gd_utils


DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "data/Galaxy10_DECals.h5"
OUTPUT_RUN_ROOT = (PROJECT_ROOT / "outputs/runs").resolve()
CHECKPOINT_ROOT = (PROJECT_ROOT / "outputs/checkpoints").resolve()
STAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
EXPECTED_FILES = {"train": "train_blends.csv", "validation": "val_blends.csv"}
ACCEPTED_SUITE_LABELS = {
    "train": {"train", "train_balanced"},
    "validation": {"validation", "validation_balanced"},
}
REPLAY_CACHE_LIMIT_BYTES = 25 * 1024**3
REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "suite",
    "source_split",
    "target_source_index",
    "target_group_id",
    "target_label",
    "contaminant_source_index",
    "contaminant_group_id",
    "contaminant_label",
    "sample_seed",
    "attempt_index",
    "shift_x",
    "shift_y",
    "brightness_scale",
    "blur_sigma",
    "noise_std",
    "rotation_degrees",
    "affected_threshold",
    "size_ratio",
    "mask_fraction",
    "core_obstruction_fraction",
    "blend_severity_score",
    "target_pxscale",
    "contaminant_pxscale",
    "blend_sha256",
    "affected_mask_sha256",
    "core_mask_sha256",
    "halo_mask_sha256",
    "generator_version",
    "generator_sha256",
    "dataset_sha256",
    "source_split_manifest_sha256",
    "training_component",
    "schema_version",
    "array_hash_method",
    "manifest_builder_sha256",
    "development_not_final",
}


@dataclass(frozen=True)
class LossConfig:
    background_weight: float = 1.0
    affected_extra_weight: float = 3.0
    core_extra_weight: float = 2.0
    affected_threshold: float = 0.02
    core_aperture_fraction: float = 0.18
    core_brightness_fraction: float = 0.55
    eps: float = 1e-8


@dataclass(frozen=True)
class TrainingConfig:
    n_train_blends: int = 8000
    n_val_blends: int = 1000
    num_epochs: int = 20
    requested_batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    training_seed: int = 3042
    normal_fraction: float = 0.50
    high_overlap_fraction: float = 0.30
    brightness_size_fraction: float = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--source-split-manifest", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run/checkpoint timestamp in YYYYMMDD_HHMMSS format.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda"), default="auto")
    parser.add_argument("--n-train-blends", type=int, default=8000)
    parser.add_argument("--n-val-blends", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--training-seed", type=int, default=3042)
    parser.add_argument(
        "--preload-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sequentially preload the uint8 HDF5 image array before one-time replay.",
    )
    parser.add_argument(
        "--loss-self-test-only",
        action="store_true",
        help="Run tiny analytic CPU loss checks and exit without creating a run.",
    )
    return parser.parse_args()


def resolved_input(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    return candidate.resolve()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def validate_sha(value: Any, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return normalized


def safe_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def safe_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content.rstrip() + "\n")


def command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.rstrip()


def manifest_float(value: Any, label: str, *, finite: bool = False) -> float:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan"}):
        parsed = float("nan")
    else:
        parsed = float(value)
    if finite and not math.isfinite(parsed):
        raise ValueError(f"Manifest field {label} must be finite, got {value!r}")
    return parsed


def unique_manifest_value(frames: Iterable[pd.DataFrame], column: str) -> str:
    values = {
        str(value).strip()
        for frame in frames
        for value in frame[column].tolist()
        if str(value).strip()
    }
    if len(values) != 1:
        raise ValueError(f"Manifest column {column!r} must have one common value: {values}")
    return next(iter(values))


def validate_configuration(args: argparse.Namespace) -> TrainingConfig:
    if not STAMP_PATTERN.fullmatch(args.stamp):
        raise ValueError("--stamp must use YYYYMMDD_HHMMSS")
    if args.manifest_dir is None or args.source_split_manifest is None:
        raise ValueError("--manifest-dir and --source-split-manifest are required")
    for name in ("n_train_blends", "n_val_blends", "num_epochs", "batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    return TrainingConfig(
        n_train_blends=int(args.n_train_blends),
        n_val_blends=int(args.n_val_blends),
        num_epochs=int(args.num_epochs),
        requested_batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        training_seed=int(args.training_seed),
    )


def make_run_paths(stamp: str) -> tuple[Path, Path, Path]:
    run_dir = OUTPUT_RUN_ROOT / f"br_v02_moderate_grouped_retrain_{stamp}"
    best = CHECKPOINT_ROOT / f"unet_br_v02_moderate_grouped_retrain_{stamp}_best.pth"
    final = CHECKPOINT_ROOT / f"unet_br_v02_moderate_grouped_retrain_{stamp}_final.pth"
    collisions = [path for path in (run_dir, best, final) if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, collisions)))
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("diagnostics", "tables", "logs", "replay_cache"):
        (run_dir / child).mkdir(exist_ok=False)
    return run_dir, best, final


def checkpoint_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if CHECKPOINT_ROOT.is_dir():
        candidates = (
            path
            for path in CHECKPOINT_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
        )
        for path in sorted(candidates):
            stat = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": file_sha256(path),
                }
            )
    return pd.DataFrame(rows, columns=("path", "size_bytes", "mtime_ns", "sha256"))


def checkpoint_integrity(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    before_map = before.set_index("path").to_dict("index") if not before.empty else {}
    after_map = after.set_index("path").to_dict("index") if not after.empty else {}
    comparisons: list[dict[str, Any]] = []
    for path, old in before_map.items():
        new = after_map.get(path)
        unchanged = new == old
        comparisons.append({"path": path, "unchanged": unchanged, "before": old, "after": new})
    return {
        "protected_checkpoint_count": len(before_map),
        "all_unchanged": all(row["unchanged"] for row in comparisons),
        "comparisons": comparisons,
        "new_checkpoint_paths": sorted(set(after_map) - set(before_map)),
    }


def recursive_summary_hashes(node: Any, filename: str) -> set[str]:
    """Find filename-specific SHA values in common summary layouts."""

    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if Path(str(key)).name == filename and isinstance(value, str):
                try:
                    found.add(validate_sha(value, f"summary hash for {filename}"))
                except ValueError:
                    pass
            if isinstance(value, dict):
                path_value = next(
                    (value.get(name) for name in ("path", "file", "filename", "name") if value.get(name)),
                    None,
                )
                sha_value = next(
                    (value.get(name) for name in ("sha256", "file_sha256") if value.get(name)),
                    None,
                )
                if path_value is not None and Path(str(path_value)).name == filename and sha_value:
                    found.add(validate_sha(sha_value, f"summary hash for {filename}"))
            found.update(recursive_summary_hashes(value, filename))
    elif isinstance(node, list):
        for value in node:
            found.update(recursive_summary_hashes(value, filename))
    return found


def load_blend_manifests(
    manifest_dir: Path,
    settings: TrainingConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest_dir = resolved_input(manifest_dir)
    if not manifest_dir.is_dir():
        raise FileNotFoundError(manifest_dir)
    summary_path = manifest_dir / "manifest_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Immutable manifest anchor missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, Any] = {
        "manifest_directory": str(manifest_dir),
        "manifest_summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "files": {},
    }
    all_sample_ids: list[str] = []
    expected_counts = {
        "train": settings.n_train_blends,
        "validation": settings.n_val_blends,
    }
    for split, filename in EXPECTED_FILES.items():
        path = manifest_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = file_sha256(path)
        expected_hashes = recursive_summary_hashes(summary, filename)
        if not expected_hashes:
            raise ValueError(f"manifest_summary.json does not anchor {filename} with SHA-256")
        if expected_hashes != {actual_sha}:
            raise ValueError(
                f"{filename} hash mismatch: summary={sorted(expected_hashes)}, actual={actual_sha}"
            )
        # Exact manifest replay requires binary64 CSV parameters to round-trip;
        # the default fast parser may shorten a final decimal digit.
        frame = pd.read_csv(
            path,
            keep_default_na=False,
            float_precision="round_trip",
        )
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{filename} lacks columns: {', '.join(missing)}")
        if len(frame) != expected_counts[split]:
            raise ValueError(
                f"{filename} has {len(frame)} rows; expected exactly {expected_counts[split]}"
            )
        split_values = set(frame["source_split"].astype(str).str.lower())
        if split_values != {split}:
            raise ValueError(f"{filename} source_split values are {split_values}, expected {split}")
        suite_values = set(frame["suite"].astype(str))
        if not suite_values or not suite_values.issubset(ACCEPTED_SUITE_LABELS[split]):
            raise ValueError(
                f"{filename} suite values are {suite_values}, expected one of "
                f"{sorted(ACCEPTED_SUITE_LABELS[split])}"
            )
        expected_components = {
            "normal": int(expected_counts[split] * 0.50),
            "high_overlap_core": int(expected_counts[split] * 0.30),
        }
        expected_components["brightness_size"] = expected_counts[split] - sum(
            expected_components.values()
        )
        observed_components = frame["training_component"].astype(str).value_counts().to_dict()
        if observed_components != expected_components:
            raise ValueError(
                f"{filename} balanced component counts are {observed_components}; "
                f"expected exactly {expected_components}"
            )
        ids = frame["sample_id"].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample_id within {filename}")
        all_sample_ids.extend(ids)
        frames[split] = frame.reset_index(drop=True)
        provenance["files"][split] = {
            "path": str(path),
            "sha256": actual_sha,
            "rows": len(frame),
            "summary_hash_verified": True,
        }
    if len(all_sample_ids) != len(set(all_sample_ids)):
        raise ValueError("sample_id values overlap between train and validation manifests")
    return frames, provenance


def load_source_manifest(path: Path) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    path = resolved_input(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    required = {
        "source_index",
        "split",
        "group_id",
        "label",
        "exact_sha256",
        "coordinate_group_key",
        "group_size",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source split manifest lacks columns: {', '.join(missing)}")
    frame["source_index"] = pd.to_numeric(frame["source_index"], errors="raise").astype(int)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["group_size"] = pd.to_numeric(frame["group_size"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["exact_sha256"] = frame["exact_sha256"].astype(str).str.lower()
    if frame["source_index"].duplicated().any():
        raise ValueError("Source split manifest contains duplicate source indices")
    if frame["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Source split manifest contains an empty group_id")
    if (frame["group_size"] <= 0).any():
        raise ValueError("Source split manifest contains a non-positive group_size")
    invalid_exact_hashes = [
        value
        for value in frame["exact_sha256"].astype(str)
        if len(value) != 64
        or any(
            character not in "0123456789abcdef" for character in value.lower()
        )
    ]
    if invalid_exact_hashes:
        raise ValueError(
            f"Source split manifest contains {len(invalid_exact_hashes)} invalid exact SHA-256 values"
        )
    if set(frame["source_index"]) != set(range(len(frame))):
        raise ValueError("Source split manifest must contain every contiguous HDF5 source index")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected source split names: {set(frame['split'])}")
    if frame.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("A source group appears in more than one split")
    observed_group_sizes = frame.groupby("group_id")["source_index"].size()
    recorded_group_sizes = frame.groupby("group_id")["group_size"].nunique()
    if recorded_group_sizes.max() != 1:
        raise ValueError("Inconsistent group_size within a source group")
    recorded = frame.groupby("group_id")["group_size"].first()
    if not observed_group_sizes.equals(recorded):
        raise ValueError("Recorded source group sizes do not match manifest membership")
    exact = frame.loc[frame["exact_sha256"].astype(str).ne("")]
    if not exact.empty and exact.groupby("exact_sha256")["split"].nunique().max() != 1:
        raise ValueError("Exact image hashes leak across source splits")
    coordinate = frame.loc[frame["coordinate_group_key"].astype(str).ne("")]
    if not coordinate.empty and coordinate.groupby("coordinate_group_key")["split"].nunique().max() != 1:
        raise ValueError("Exact coordinate groups leak across source splits")
    mapping = {
        int(row.source_index): {
            "split": str(row.split),
            "group_id": str(row.group_id),
            "label": int(row.label),
        }
        for row in frame.itertuples(index=False)
    }
    return frame, mapping


def verify_common_provenance(
    frames: dict[str, pd.DataFrame],
    source_manifest_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    values = list(frames.values())
    actual_source_sha = file_sha256(source_manifest_path)
    stored_source_sha = validate_sha(
        unique_manifest_value(values, "source_split_manifest_sha256"),
        "source_split_manifest_sha256",
    )
    if actual_source_sha != stored_source_sha:
        raise ValueError(
            f"Source manifest hash mismatch: rows={stored_source_sha}, actual={actual_source_sha}"
        )
    actual_dataset_sha = file_sha256(dataset_path)
    stored_dataset_sha = validate_sha(
        unique_manifest_value(values, "dataset_sha256"), "dataset_sha256"
    )
    if actual_dataset_sha != stored_dataset_sha:
        raise ValueError(
            f"Dataset hash mismatch: rows={stored_dataset_sha}, actual={actual_dataset_sha}"
        )
    generator_path = PROJECT_ROOT / "src/blend.py"
    actual_generator_sha = file_sha256(generator_path)
    stored_generator_sha = validate_sha(
        unique_manifest_value(values, "generator_sha256"), "generator_sha256"
    )
    if actual_generator_sha != stored_generator_sha:
        raise ValueError(
            f"Generator code hash mismatch: rows={stored_generator_sha}, actual={actual_generator_sha}"
        )
    builder_path = PROJECT_ROOT / "scripts/build_grouped_blend_manifests.py"
    if not builder_path.is_file():
        raise FileNotFoundError(builder_path)
    actual_builder_sha = file_sha256(builder_path)
    stored_builder_sha = validate_sha(
        unique_manifest_value(values, "manifest_builder_sha256"),
        "manifest_builder_sha256",
    )
    if actual_builder_sha != stored_builder_sha:
        raise ValueError(
            f"Manifest-builder code hash mismatch: rows={stored_builder_sha}, "
            f"actual={actual_builder_sha}"
        )
    schema_versions = sorted(
        {str(value) for frame in values for value in frame["schema_version"].tolist()}
    )
    if schema_versions != ["thayer_grouped_blend_manifest_schema_v1"]:
        raise ValueError(f"Unsupported grouped blend manifest schema: {schema_versions}")
    hash_methods = sorted(
        {str(value) for frame in values for value in frame["array_hash_method"].tolist()}
    )
    if hash_methods != ["sha256(dtype_ascii+shape_ascii+contiguous_bytes)_v1"]:
        raise ValueError(f"Unsupported grouped array hash contract: {hash_methods}")
    versions = sorted(
        {str(value) for frame in values for value in frame["generator_version"].tolist()}
    )
    return {
        "source_split_manifest": {
            "path": str(source_manifest_path),
            "sha256": actual_source_sha,
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": actual_dataset_sha,
            "size_bytes": dataset_path.stat().st_size,
        },
        "generator": {
            "path": str(generator_path),
            "sha256": actual_generator_sha,
            "versions": versions,
        },
        "manifest_builder": {
            "path": str(builder_path),
            "sha256": actual_builder_sha,
            "schema_versions": schema_versions,
            "array_hash_methods": hash_methods,
        },
        "benchmark_role": {
            "development_not_final_values": sorted(
                {
                    str(value)
                    for frame in values
                    for value in frame["development_not_final"].tolist()
                }
            ),
            "paper_final_claim_permitted": False,
        },
    }


def verify_role(row: pd.Series, split: str, mapping: dict[int, dict[str, Any]]) -> None:
    for role in ("target", "contaminant"):
        source_index = int(row[f"{role}_source_index"])
        source = mapping.get(source_index)
        if source is None:
            raise ValueError(f"sample_id={row['sample_id']}: unknown {role} source {source_index}")
        if source["split"] != split:
            raise ValueError(
                f"sample_id={row['sample_id']}: {role} source is {source['split']}, expected {split}"
            )
        if str(row[f"{role}_group_id"]) != source["group_id"]:
            raise ValueError(f"sample_id={row['sample_id']}: {role} group mismatch")
        if int(row[f"{role}_label"]) != source["label"]:
            raise ValueError(f"sample_id={row['sample_id']}: {role} label mismatch")
    if str(row["target_group_id"]) == str(row["contaminant_group_id"]):
        raise ValueError(f"sample_id={row['sample_id']}: target/contaminant share a source group")


def verify_manifest_role_separation(
    frames: dict[str, pd.DataFrame], mapping: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    source_sets: dict[str, set[int]] = {}
    group_sets: dict[str, set[str]] = {}
    target_sets: dict[str, set[int]] = {}
    contaminant_sets: dict[str, set[int]] = {}
    for split, frame in frames.items():
        for _, row in frame.iterrows():
            verify_role(row, split, mapping)
        target_sets[split] = set(frame["target_source_index"].astype(int))
        contaminant_sets[split] = set(frame["contaminant_source_index"].astype(int))
        source_sets[split] = target_sets[split] | contaminant_sets[split]
        group_sets[split] = set(frame["target_group_id"].astype(str)) | set(
            frame["contaminant_group_id"].astype(str)
        )
    source_overlap = source_sets["train"] & source_sets["validation"]
    group_overlap = group_sets["train"] & group_sets["validation"]
    cross_role_overlap = (
        target_sets["train"] & contaminant_sets["validation"]
    ) | (contaminant_sets["train"] & target_sets["validation"])
    if source_overlap or group_overlap or cross_role_overlap:
        raise ValueError(
            "Grouped train/validation role separation failed: "
            f"sources={len(source_overlap)}, groups={len(group_overlap)}, "
            f"cross_roles={len(cross_role_overlap)}"
        )
    return {
        "status": "pass",
        "train_unique_sources": len(source_sets["train"]),
        "validation_unique_sources": len(source_sets["validation"]),
        "train_unique_groups": len(group_sets["train"]),
        "validation_unique_groups": len(group_sets["validation"]),
        "cross_split_source_overlap": 0,
        "cross_split_group_overlap": 0,
        "cross_split_target_contaminant_role_overlap": 0,
    }


def replay_row(
    row: pd.Series,
    split: str,
    images: Any,
    labels: np.ndarray,
    pxscale: np.ndarray,
    source_mapping: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    verify_role(row, split, source_mapping)
    target_index = int(row["target_source_index"])
    contaminant_index = int(row["contaminant_source_index"])
    if not (0 <= target_index < len(labels) and 0 <= contaminant_index < len(labels)):
        raise IndexError(f"sample_id={row['sample_id']}: source index outside HDF5 range")
    if int(labels[target_index]) != int(row["target_label"]):
        raise ValueError(f"sample_id={row['sample_id']}: target HDF5 label mismatch")
    if int(labels[contaminant_index]) != int(row["contaminant_label"]):
        raise ValueError(f"sample_id={row['sample_id']}: contaminant HDF5 label mismatch")
    target_pxscale = manifest_float(row["target_pxscale"], "target_pxscale", finite=True)
    contaminant_pxscale = manifest_float(
        row["contaminant_pxscale"], "contaminant_pxscale", finite=True
    )
    if not np.isclose(float(pxscale[target_index]), target_pxscale, rtol=0, atol=1e-12):
        raise ValueError(f"sample_id={row['sample_id']}: target pxscale mismatch")
    if not np.isclose(
        float(pxscale[contaminant_index]), contaminant_pxscale, rtol=0, atol=1e-12
    ):
        raise ValueError(f"sample_id={row['sample_id']}: contaminant pxscale mismatch")

    target_u8 = np.asarray(images[target_index], dtype=np.uint8)
    contaminant_u8 = np.asarray(images[contaminant_index], dtype=np.uint8)
    target = target_u8.astype(np.float32) / 255.0
    contaminant = contaminant_u8.astype(np.float32) / 255.0
    rng = np.random.default_rng(int(row["sample_seed"]))
    blended, info = gd_blend.blend_pair(
        target=target,
        contaminant=contaminant,
        shift=(int(row["shift_x"]), int(row["shift_y"])),
        rotation=manifest_float(row["rotation_degrees"], "rotation_degrees", finite=True),
        brightness=manifest_float(row["brightness_scale"], "brightness_scale", finite=True),
        blur_sigma=manifest_float(row["blur_sigma"], "blur_sigma", finite=True),
        noise_std=manifest_float(row["noise_std"], "noise_std", finite=True),
        rng=rng,
    )
    threshold = manifest_float(row["affected_threshold"], "affected_threshold", finite=True)
    affected = gd_utils.affected_region_mask(target, blended, threshold=threshold)
    core = gd_utils.evaluation_core_mask_p85_v1(target)
    core_affected = affected & core
    halo = gd_utils.halo_band_mask_manhattan_v1(affected, dilation_iters=5)
    actual_hashes = {
        "blend_sha256": array_sha256(blended),
        # Grouped-manifest v1 stores masks as uint8 before hashing.
        "affected_mask_sha256": array_sha256(affected.astype(np.uint8)),
        "core_mask_sha256": array_sha256(core.astype(np.uint8)),
        "halo_mask_sha256": array_sha256(halo.astype(np.uint8)),
    }
    for column, actual in actual_hashes.items():
        expected = validate_sha(row[column], column)
        if actual != expected:
            raise ValueError(
                f"sample_id={row['sample_id']}: {column} replay mismatch "
                f"manifest={expected}, actual={actual}"
            )
    checks = {
        "size_ratio": float(info["size_ratio"]),
        "mask_fraction": float(affected.mean()),
        "core_obstruction_fraction": (
            float(core_affected.sum() / core.sum()) if np.any(core) else 0.0
        ),
    }
    for column, actual in checks.items():
        expected = manifest_float(row[column], column)
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True):
            raise ValueError(
                f"sample_id={row['sample_id']}: {column} replay mismatch "
                f"manifest={expected}, actual={actual}"
            )
    severity = float(affected.mean()) * gd_utils.masked_mae(blended, target, affected) * (
        1.0 + checks["core_obstruction_fraction"]
    )
    expected_severity = manifest_float(row["blend_severity_score"], "blend_severity_score")
    if not np.isclose(severity, expected_severity, rtol=1e-6, atol=1e-10, equal_nan=True):
        raise ValueError(f"sample_id={row['sample_id']}: blend severity replay mismatch")
    return target_u8, blended


def estimate_cache_bytes(frames: dict[str, pd.DataFrame], image_shape: tuple[int, ...]) -> int:
    sample_count = sum(len(frame) for frame in frames.values())
    values = int(np.prod(image_shape))
    return sample_count * values * (np.dtype(np.float32).itemsize + np.dtype(np.uint8).itemsize)


def build_replay_cache(
    run_dir: Path,
    frames: dict[str, pd.DataFrame],
    dataset_path: Path,
    source_mapping: dict[int, dict[str, Any]],
    preload_images: bool,
) -> dict[str, Any]:
    cache_dir = run_dir / "replay_cache"
    result: dict[str, Any] = {"preload_images": preload_images, "splits": {}}
    with h5py.File(dataset_path, "r") as handle:
        required = {"images", "ans", "pxscale"}
        missing = required - set(handle.keys())
        if missing:
            raise KeyError(f"Dataset lacks keys: {sorted(missing)}")
        images_ds = handle["images"]
        if images_ds.ndim != 4 or images_ds.shape[-1] != 3 or images_ds.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 NHWC Galaxy10 images; found {images_ds.shape} {images_ds.dtype}"
            )
        labels = np.asarray(handle["ans"][:]).squeeze().astype(np.int64)
        pxscale = np.asarray(handle["pxscale"][:]).squeeze().astype(np.float64)
        if len(labels) != images_ds.shape[0] or len(pxscale) != images_ds.shape[0]:
            raise ValueError("HDF5 label/pxscale arrays do not align with images")
        if len(source_mapping) != images_ds.shape[0]:
            raise ValueError(
                f"Source split has {len(source_mapping)} rows but HDF5 has "
                f"{images_ds.shape[0]} images"
            )
        estimated_bytes = estimate_cache_bytes(frames, tuple(images_ds.shape[1:]))
        if estimated_bytes > REPLAY_CACHE_LIMIT_BYTES:
            raise RuntimeError(
                f"Replay cache estimate {estimated_bytes} exceeds 25 GiB safety limit"
            )
        free_bytes = shutil.disk_usage(cache_dir).free
        if free_bytes < estimated_bytes + 5 * 1024**3:
            raise RuntimeError(
                f"Insufficient free disk for replay cache: need {estimated_bytes} + 5 GiB, "
                f"have {free_bytes}"
            )
        print(
            f"Replay cache estimate: {estimated_bytes / 1024**3:.2f} GiB; "
            f"free disk: {free_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
        if preload_images:
            print(
                f"Sequentially preloading {images_ds.shape[0]} uint8 source images "
                f"({images_ds.nbytes / 1024**3:.2f} GiB).",
                flush=True,
            )
            images: Any = images_ds[:]
        else:
            print("Using direct HDF5 source reads for one-time replay cache generation.", flush=True)
            images = images_ds

        for split, frame in frames.items():
            n = len(frame)
            shape = (n, *images_ds.shape[1:])
            blended_path = cache_dir / f"{split}_blended_float32.npy"
            target_path = cache_dir / f"{split}_target_uint8.npy"
            blended_cache = np.lib.format.open_memmap(
                blended_path, mode="w+", dtype=np.float32, shape=shape
            )
            target_cache = np.lib.format.open_memmap(
                target_path, mode="w+", dtype=np.uint8, shape=shape
            )
            print(f"Replaying and hashing {split}: {n} immutable rows", flush=True)
            for index, (_, row) in enumerate(frame.iterrows()):
                target_u8, blended = replay_row(
                    row, split, images, labels, pxscale, source_mapping
                )
                target_cache[index] = target_u8
                blended_cache[index] = blended
                if (index + 1) % 250 == 0 or index + 1 == n:
                    print(f"  {split} replay verified {index + 1}/{n}", flush=True)
            blended_cache.flush()
            target_cache.flush()
            del blended_cache, target_cache
            cache_files = {
                "blended": {
                    "path": str(blended_path),
                    "size_bytes": blended_path.stat().st_size,
                    "sha256": file_sha256(blended_path),
                },
                "target": {
                    "path": str(target_path),
                    "size_bytes": target_path.stat().st_size,
                    "sha256": file_sha256(target_path),
                },
            }
            os.chmod(blended_path, 0o444)
            os.chmod(target_path, 0o444)
            result["splits"][split] = {
                "rows": n,
                "shape": list(shape),
                "exact_replay_rows": n,
                "files": cache_files,
            }
        if preload_images:
            del images
    result["estimated_cache_bytes"] = estimated_bytes
    result["actual_cache_bytes"] = sum(
        details["size_bytes"]
        for split in result["splits"].values()
        for details in split["files"].values()
    )
    if result["actual_cache_bytes"] > REPLAY_CACHE_LIMIT_BYTES:
        raise RuntimeError("Actual replay cache exceeded the 25 GiB safety limit")
    return result


class ReplayCacheDataset(Dataset):
    """Read-only disk-backed residual training dataset."""

    def __init__(self, blended_path: Path, target_path: Path) -> None:
        self.blended = np.load(blended_path, mmap_mode="r")
        self.target = np.load(target_path, mmap_mode="r")
        if self.blended.shape != self.target.shape or self.blended.dtype != np.float32:
            raise ValueError("Replay cache arrays are incompatible")
        if self.target.dtype != np.uint8:
            raise ValueError("Target replay cache must use uint8 source images")

    def __len__(self) -> int:
        return int(self.blended.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blended = np.array(self.blended[index], dtype=np.float32, copy=True)
        target = np.array(self.target[index], dtype=np.float32, copy=True) / 255.0
        residual = blended - target
        return (
            torch.from_numpy(blended.transpose(2, 0, 1).copy()),
            torch.from_numpy(residual.transpose(2, 0, 1).copy()),
            torch.from_numpy(target.transpose(2, 0, 1).copy()),
        )


def target_core_mask_torch(target: torch.Tensor, config: LossConfig) -> torch.Tensor:
    """Historical v0.2 training-loss core mask (not the evaluation p85 mask)."""

    gray = target.mean(dim=1, keepdim=True)
    batch, _channels, height, width = gray.shape
    y = torch.arange(height, device=target.device, dtype=target.dtype).view(1, 1, height, 1)
    x = torch.arange(width, device=target.device, dtype=target.dtype).view(1, 1, 1, width)
    radius = config.core_aperture_fraction * min(height, width)
    aperture = torch.sqrt((y - (height - 1) / 2.0) ** 2 + (x - (width - 1) / 2.0) ** 2) <= radius
    aperture = aperture.expand(batch, 1, height, width)
    aperture_gray = torch.where(aperture, gray, torch.full_like(gray, -1.0))
    threshold = aperture_gray.amax(dim=(2, 3), keepdim=True) * config.core_brightness_fraction
    core = aperture & (gray >= threshold)
    empty = core.flatten(1).sum(dim=1) == 0
    return torch.where(empty.view(-1, 1, 1, 1), aperture, core)


def weighted_residual_loss(
    predicted_residual: torch.Tensor,
    true_residual: torch.Tensor,
    target: torch.Tensor,
    config: LossConfig,
    *,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if predicted_residual.shape != true_residual.shape or target.shape != true_residual.shape:
        raise ValueError("Predicted residual, true residual, and target must have equal BCHW shape")
    affected = true_residual.abs().mean(dim=1, keepdim=True) > config.affected_threshold
    core_affected = affected & target_core_mask_torch(target, config)
    weight = torch.full_like(true_residual[:, :1], config.background_weight)
    weight = weight + affected.float() * config.affected_extra_weight
    weight = weight + core_affected.float() * config.core_extra_weight
    squared_error = (predicted_residual - true_residual).square()
    denominator = weight.sum() * squared_error.shape[1] + config.eps
    loss = (weight * squared_error).sum() / denominator
    if not collect_stats:
        return loss, {}
    return loss, {
        "affected_fraction": float(affected.float().mean().detach().cpu()),
        "core_affected_fraction": float(core_affected.float().mean().detach().cpu()),
        "mean_weight": float(weight.mean().detach().cpu()),
    }


def run_loss_self_test() -> dict[str, Any]:
    config = LossConfig()
    true = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    true[:, :, 1, 1] = 1.0
    prediction = torch.zeros_like(true)
    target = torch.zeros_like(true)
    target[:, :, 1, 1] = 1.0
    loss, stats = weighted_residual_loss(prediction, true, target, config)
    expected = 18.0 / 63.0
    zero, _ = weighted_residual_loss(true, true, target, config)
    passed = bool(
        np.isclose(float(loss), expected, rtol=0, atol=1e-7)
        and float(zero) == 0.0
        and np.isclose(stats["mean_weight"], 21.0 / 16.0, rtol=0, atol=1e-7)
    )
    result = {
        "passed": passed,
        "observed_weighted_loss": float(loss),
        "expected_weighted_loss": expected,
        "zero_error_loss": float(zero),
        "stats": stats,
        "device": "cpu_tiny_analytic_self_test_only",
    }
    if not passed:
        raise RuntimeError(f"Weighted loss self-test failed: {result}")
    return result


def make_model(model_config: dict[str, Any]) -> nn.Module:
    expected = {"in_channels": 3, "out_channels": 3, "base_channels": 32, "norm": True}
    normalized = {
        "in_channels": int(model_config.get("in_channels", 3)),
        "out_channels": int(model_config.get("out_channels", 3)),
        "base_channels": int(model_config.get("base_channels", 32)),
        "norm": bool(model_config.get("norm", True)),
    }
    if normalized != expected:
        raise ValueError(
            f"Grouped retrain must preserve historical v0.2 architecture: {expected}; got {normalized}"
        )
    model = models.UNet(**normalized)
    model.out_activation = nn.Identity()
    return model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def is_memory_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def append_progress(path: Path, payload: dict[str, Any]) -> None:
    mode = "x" if not path.exists() else "a"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: LossConfig,
) -> dict[str, float]:
    model.eval()
    sample_count = 0
    loss_sum = 0.0
    affected_mse: list[float] = []
    residual_mse_sum = 0.0
    clip_count = 0
    value_count = 0
    with torch.no_grad():
        for blended, true_residual, target in loader:
            blended = blended.to(device)
            true_residual = true_residual.to(device)
            target = target.to(device)
            predicted = model(blended)
            loss, _ = weighted_residual_loss(
                predicted,
                true_residual,
                target,
                loss_config,
                collect_stats=False,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")
            batch_size = blended.shape[0]
            sample_count += batch_size
            loss_sum += float(loss.item()) * batch_size
            residual_mse_sum += float((predicted - true_residual).square().mean().item()) * batch_size
            reconstruction_unclipped = blended - predicted
            reconstruction = reconstruction_unclipped.clamp(0.0, 1.0)
            clip_count += int(
                ((reconstruction_unclipped < 0.0) | (reconstruction_unclipped > 1.0)).sum().item()
            )
            value_count += reconstruction_unclipped.numel()
            affected = true_residual.abs().mean(dim=1) > loss_config.affected_threshold
            spatial_mse = (reconstruction - target).square().mean(dim=1)
            affected_count = affected.flatten(1).sum(dim=1)
            affected_sum = (spatial_mse * affected).flatten(1).sum(dim=1)
            valid = affected_count > 0
            values = (affected_sum[valid] / affected_count[valid]).detach().cpu().tolist()
            affected_mse.extend(float(value) for value in values)
    if sample_count == 0:
        raise RuntimeError("Validation loader was empty")
    return {
        "val_loss": loss_sum / sample_count,
        "val_affected_mse": float(np.mean(affected_mse)) if affected_mse else float("nan"),
        "val_unweighted_residual_mse": residual_mse_sum / sample_count,
        "val_reconstruction_clip_fraction": clip_count / max(value_count, 1),
        "val_affected_valid_samples": len(affected_mse),
    }


def train_attempt(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    settings: TrainingConfig,
    loss_config: LossConfig,
    batch_size: int,
    device: torch.device,
    progress_path: Path,
    attempt_number: int,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame]:
    seed_everything(settings.training_seed)
    generator = torch.Generator()
    generator.manual_seed(settings.training_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    if {parameter.device.type for parameter in model.parameters()} != {device.type}:
        raise RuntimeError("Model parameters were not placed entirely on the selected accelerator")
    optimiser = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, settings.num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for blended, true_residual, target in train_loader:
            blended = blended.to(device)
            true_residual = true_residual.to(device)
            target = target.to(device)
            optimiser.zero_grad(set_to_none=True)
            predicted = model(blended)
            loss, _ = weighted_residual_loss(
                predicted,
                true_residual,
                target,
                loss_config,
                collect_stats=False,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimiser.step()
            train_loss_sum += float(loss.item()) * blended.shape[0]
            train_count += blended.shape[0]
        diagnostics = validation_epoch(model, val_loader, device, loss_config)
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = diagnostics["val_loss"]
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(f"Non-finite epoch loss at epoch {epoch}")
        if train_loss > 2.0 or val_loss > 2.0:
            raise RuntimeError(
                f"Loss explosion at epoch {epoch}: train={train_loss}, validation={val_loss}"
            )
        if diagnostics["val_reconstruction_clip_fraction"] > 0.20:
            raise RuntimeError(
                f"Suspicious validation clipping at epoch {epoch}: "
                f"{diagnostics['val_reconstruction_clip_fraction']:.3%}"
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        row = {
            "attempt": attempt_number,
            "epoch": epoch,
            "batch_size": batch_size,
            "train_loss": train_loss,
            **diagnostics,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
        }
        history.append(row)
        append_progress(progress_path, row)
        print(
            f"Epoch {epoch}/{settings.num_epochs}: train={train_loss:.7f}, "
            f"val={val_loss:.7f}, val affected MSE={diagnostics['val_affected_mse']:.7f}, "
            f"best={best_loss:.7f} @ {best_epoch}",
            flush=True,
        )
    if not best_state:
        raise RuntimeError("Training completed without a best validation state")
    return model, best_state, pd.DataFrame(history)


def train_with_oom_retry(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_config: dict[str, Any],
    settings: TrainingConfig,
    loss_config: LossConfig,
    device: torch.device,
    progress_path: Path,
) -> tuple[nn.Module, dict[str, torch.Tensor], pd.DataFrame, int, list[dict[str, Any]]]:
    batch_size = settings.requested_batch_size
    attempts: list[dict[str, Any]] = []
    attempt_number = 0
    while batch_size >= 1:
        attempt_number += 1
        seed_everything(settings.training_seed)
        model = make_model(model_config)
        try:
            trained, best_state, history = train_attempt(
                model,
                train_dataset,
                val_dataset,
                settings,
                loss_config,
                batch_size,
                device,
                progress_path,
                attempt_number,
            )
            attempts.append({"attempt": attempt_number, "batch_size": batch_size, "status": "complete"})
            return trained, best_state, history, batch_size, attempts
        except RuntimeError as error:
            if not is_memory_error(error) or batch_size == 1:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "batch_size": batch_size,
                        "status": "failed",
                        "error": str(error),
                    }
                )
                raise
            next_batch = max(1, batch_size // 2)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "batch_size": batch_size,
                    "status": "mps_or_cuda_oom_retry",
                    "retry_batch_size": next_batch,
                    "error": str(error),
                }
            )
            print(
                f"Accelerator OOM at batch size {batch_size}; resetting model and retrying "
                f"from the fixed seed with batch size {next_batch}.",
                flush=True,
            )
            del model
            clear_accelerator_cache()
            gc.collect()
            batch_size = next_batch
    raise RuntimeError("No viable accelerator batch size")


def save_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    kind: str,
    epoch: int,
    history: pd.DataFrame,
    model_config: dict[str, Any],
    settings: TrainingConfig,
    loss_config: LossConfig,
    used_batch_size: int,
    device: torch.device,
    provenance: dict[str, Any],
    cache: dict[str, Any],
    stamp: str,
    training_history_sha: str,
    parameter_count: int,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    payload = {
        "model_state_dict": state,
        "experiment_name": "Thayer-BR v0.2 Moderate Grouped Retrain",
        "variant_name": "moderate_grouped",
        "checkpoint_kind": kind,
        "checkpoint_epoch": epoch,
        "residual_target": "blended_minus_target",
        "reconstruction": "blended_minus_predicted_residual",
        "output_activation": "identity",
        "architecture": "compact_unet_historical_v02",
        "model_config": model_config,
        "config": {"model": model_config},
        "parameter_count": parameter_count,
        "training_config": asdict(settings),
        "loss_config": asdict(loss_config),
        "used_batch_size": used_batch_size,
        "selected_device_type": device.type,
        "training_seed": settings.training_seed,
        "best_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"]),
        "best_val_loss": float(history["val_loss"].min()),
        "final_train_loss": float(history.iloc[-1]["train_loss"]),
        "final_val_loss": float(history.iloc[-1]["val_loss"]),
        "final_val_affected_mse": float(history.iloc[-1]["val_affected_mse"]),
        "training_history_sha256": training_history_sha,
        "provenance": provenance,
        "replay_cache_sha256": {
            split: {
                kind_name: details["sha256"]
                for kind_name, details in split_details["files"].items()
            }
            for split, split_details in cache["splits"].items()
        },
        "timestamp": stamp,
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = resolved_input(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Config {path} lacks a model mapping")
    model_config = dict(config["model"])
    # Construct once only after device logging in main; validation here is scalar-only.
    expected = {"in_channels": 3, "out_channels": 3, "base_channels": 32, "norm": True}
    normalized = {
        "in_channels": int(model_config.get("in_channels", 3)),
        "out_channels": int(model_config.get("out_channels", 3)),
        "base_channels": int(model_config.get("base_channels", 32)),
        "norm": bool(model_config.get("norm", True)),
    }
    if normalized != expected:
        raise ValueError(f"Historical v0.2 architecture mismatch: expected {expected}, got {normalized}")
    return normalized, {"path": str(path), "sha256": file_sha256(path), "full_config": config}


def finalize_checkpoint_integrity(run_dir: Path, before: pd.DataFrame) -> dict[str, Any]:
    after = checkpoint_inventory()
    after_path = run_dir / "tables/checkpoint_inventory_after.csv"
    if not after_path.exists():
        safe_csv(after_path, after)
    integrity = checkpoint_integrity(before, after)
    integrity_path = run_dir / "logs/checkpoint_integrity.json"
    if not integrity_path.exists():
        safe_json(integrity_path, integrity)
    return integrity


def write_failure(run_dir: Path, error: BaseException, device: str | None) -> None:
    path = run_dir / "diagnostics/training_failure.md"
    if path.exists():
        return
    safe_text(
        path,
        f"""# Grouped v0.2 retrain failure diagnostic

- Status: stopped; no success claim is permitted.
- Selected device: `{device or 'not selected'}`
- Exception: `{type(error).__name__}: {error}`
- Run directory: `{run_dir}`

The interrupted run and any partial replay cache/checkpoint are intentionally
preserved. Re-run only with a new timestamp after addressing the error.

## Traceback

```text
{traceback.format_exc()}
```
""",
    )


def main() -> int:
    args = parse_args()
    if args.loss_self_test_only:
        print(json.dumps(run_loss_self_test(), indent=2))
        return 0

    settings = validate_configuration(args)
    loss_config = LossConfig()
    dataset_path = resolved_input(args.dataset)
    source_manifest_path = resolved_input(args.source_split_manifest)
    manifest_dir = resolved_input(args.manifest_dir)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    run_dir, best_path, final_path = make_run_paths(args.stamp)
    before = checkpoint_inventory()
    safe_csv(run_dir / "tables/checkpoint_inventory_before.csv", before)
    safe_text(run_dir / "logs/command.txt", shlex.join(sys.argv))
    selected_device: torch.device | None = None
    try:
        frames, manifest_provenance = load_blend_manifests(manifest_dir, settings)
        source_frame, source_mapping = load_source_manifest(source_manifest_path)
        role_integrity = verify_manifest_role_separation(frames, source_mapping)
        provenance = verify_common_provenance(
            frames, source_manifest_path, dataset_path
        )
        provenance["blend_manifests"] = manifest_provenance
        provenance["source_split_rows"] = len(source_frame)
        provenance["manifest_role_integrity"] = role_integrity
        model_config, config_provenance = load_model_config(args.config)
        provenance["configuration"] = config_provenance
        provenance["code"] = {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "src/models.py",
                PROJECT_ROOT / "src/train.py",
                PROJECT_ROOT / "src/utils.py",
                PROJECT_ROOT / "src/blend.py",
            )
        }
        provenance["git"] = {
            "head": command_output(["git", "rev-parse", "HEAD"]),
            "branch": command_output(["git", "branch", "--show-current"]),
            "status_porcelain": command_output(["git", "status", "--short"]),
        }

        # Device selection and durable logging happen before any tensor/model is created.
        device_payload: dict[str, Any] = {
            "requested_device": args.device,
            "torch_version": torch.__version__,
            "mps_built": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            ),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "python": sys.version,
            "platform": platform.platform(),
            "logged_before_tensor_creation": True,
        }
        try:
            selected_device = gd_train.resolve_accelerator(args.device)
        except RuntimeError:
            device_payload.update(
                {
                    "selected_device": None,
                    "selected_device_type": None,
                    "status": "stopped_no_approved_accelerator",
                }
            )
            safe_json(run_dir / "logs/device.json", device_payload)
            print(
                "No approved MPS/CUDA training device is available; stopping before tensors.",
                flush=True,
            )
            raise
        device_payload.update(
            {
                "selected_device": str(selected_device),
                "selected_device_type": selected_device.type,
                "status": "approved_accelerator_selected",
            }
        )
        safe_json(run_dir / "logs/device.json", device_payload)
        print(f"Selected full-training device: {selected_device}", flush=True)

        self_test = run_loss_self_test()
        safe_json(run_dir / "logs/weighted_loss_self_test.json", self_test)
        cache = build_replay_cache(
            run_dir,
            frames,
            dataset_path,
            source_mapping,
            bool(args.preload_images),
        )
        safe_json(run_dir / "logs/replay_cache_inventory.json", cache)
        gc.collect()
        clear_accelerator_cache()
        safe_json(
            run_dir / "logs/run_config.json",
            {
                "experiment": "Thayer-BR v0.2 Moderate Grouped Retrain",
                "stamp": args.stamp,
                "training": asdict(settings),
                "loss": asdict(loss_config),
                "model": model_config,
                "best_checkpoint": str(best_path),
                "final_checkpoint": str(final_path),
                "preload_images": bool(args.preload_images),
                "selected_device": str(selected_device),
            },
        )
        safe_json(run_dir / "logs/provenance.json", provenance)

        train_dataset = ReplayCacheDataset(
            Path(cache["splits"]["train"]["files"]["blended"]["path"]),
            Path(cache["splits"]["train"]["files"]["target"]["path"]),
        )
        val_dataset = ReplayCacheDataset(
            Path(cache["splits"]["validation"]["files"]["blended"]["path"]),
            Path(cache["splits"]["validation"]["files"]["target"]["path"]),
        )
        progress_path = run_dir / "tables/training_progress.jsonl"
        trained, best_state, history, used_batch, attempts = train_with_oom_retry(
            train_dataset,
            val_dataset,
            model_config,
            settings,
            loss_config,
            selected_device,
            progress_path,
        )
        safe_csv(run_dir / "tables/training_history.csv", history)
        safe_json(run_dir / "logs/training_attempts.json", attempts)
        history_sha = file_sha256(run_dir / "tables/training_history.csv")
        final_state = {
            name: value.detach().cpu().clone() for name, value in trained.state_dict().items()
        }
        parameter_count = int(sum(parameter.numel() for parameter in trained.parameters()))
        best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
        save_checkpoint(
            best_path,
            best_state,
            "best_validation",
            best_epoch,
            history,
            model_config,
            settings,
            loss_config,
            used_batch,
            selected_device,
            provenance,
            cache,
            args.stamp,
            history_sha,
            parameter_count,
        )
        save_checkpoint(
            final_path,
            final_state,
            "final_epoch",
            settings.num_epochs,
            history,
            model_config,
            settings,
            loss_config,
            used_batch,
            selected_device,
            provenance,
            cache,
            args.stamp,
            history_sha,
            parameter_count,
        )
        checkpoint_outputs = {
            "best": {
                "path": str(best_path),
                "sha256": file_sha256(best_path),
                "size_bytes": best_path.stat().st_size,
                "epoch": best_epoch,
            },
            "final": {
                "path": str(final_path),
                "sha256": file_sha256(final_path),
                "size_bytes": final_path.stat().st_size,
                "epoch": settings.num_epochs,
            },
        }
        safe_json(run_dir / "logs/checkpoint_outputs.json", checkpoint_outputs)
        integrity = finalize_checkpoint_integrity(run_dir, before)
        if not integrity["all_unchanged"]:
            raise RuntimeError("A pre-existing checkpoint changed during grouped retraining")
        safe_json(
            run_dir / "logs/run_status.json",
            {
                "status": "training_complete_evaluation_pending",
                "best_epoch": best_epoch,
                "best_val_loss": float(history["val_loss"].min()),
                "best_val_affected_mse": float(
                    history.loc[history["val_loss"].idxmin(), "val_affected_mse"]
                ),
                "used_batch_size": used_batch,
                "selected_device": str(selected_device),
                "checkpoint_outputs": checkpoint_outputs,
                "note": "Grouped test evaluation must be run separately before model claims.",
            },
        )
        print(f"Grouped v0.2 training complete: {run_dir}", flush=True)
        print(f"Best checkpoint: {best_path}", flush=True)
        print(f"Final checkpoint: {final_path}", flush=True)
        return 0
    except BaseException as error:
        write_failure(run_dir, error, str(selected_device) if selected_device else None)
        integrity = finalize_checkpoint_integrity(run_dir, before)
        if not integrity["all_unchanged"]:
            print("CRITICAL: pre-existing checkpoint integrity changed", file=sys.stderr)
        status_path = run_dir / "logs/run_failure.json"
        if not status_path.exists():
            safe_json(
                status_path,
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "selected_device": str(selected_device) if selected_device else None,
                    "old_checkpoint_integrity": integrity["all_unchanged"],
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
