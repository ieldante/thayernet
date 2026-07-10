#!/usr/bin/env python3
"""Evaluate a strictly identified Thayer-BR v0.2 checkpoint on grouped tests.

Historical mode diagnoses a model trained on the original row-index split.
Grouped-retrain mode evaluates a newly grouped-trained model on the same
grouped development tests. Both modes refuse CPU inference, replay and verify
every row, and report unclipped plus clipped metrics. Neither mode is a locked
final-paper evaluation. The grouped mode uses separate checkpoint semantics
and output names; it never weakens the historical-checkpoint checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import baselines
from src import blend as gd_blend
from src import models
from src import train as gd_train
from src import utils as gd_utils


SUITE_FILES = {
    "normal": "normal_test_blends.csv",
    "hard_stress": "hard_stress_test_blends.csv",
    "compact_bright": "compact_bright_test_blends.csv",
    "high_core_obstruction": "high_core_obstruction_test_blends.csv",
}

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
}

MODE_CONFIG: dict[str, dict[str, Any]] = {
    "historical-old": {
        "method": "v02_moderate_old_split",
        "evaluation_status": "diagnostic_old_split_checkpoint_not_final",
        "log_prefix": "grouped_existing_v02",
        "outputs": {
            "suite": Path("tables/grouped_existing_v02_suite_metrics.csv"),
            "sample": Path("tables/grouped_existing_v02_per_sample_metrics.csv"),
            "report": Path("reports/existing_v02_grouped_eval.md"),
            "device": Path("logs/grouped_existing_v02_device.json"),
            "provenance": Path("logs/grouped_existing_v02_provenance.json"),
        },
    },
    "grouped-retrain": {
        "method": "v02_moderate_grouped_retrain",
        "evaluation_status": "grouped_retrain_development_test_not_final",
        "log_prefix": "grouped_retrain_v02",
        "outputs": {
            "suite": Path("tables/grouped_retrain_suite_metrics.csv"),
            "sample": Path("tables/grouped_retrain_per_sample_metrics.csv"),
            "comparison": Path("tables/grouped_retrain_comparison_summary.csv"),
            "report": Path("diagnostics/grouped_retrain_v02_report.md"),
            "device": Path("logs/grouped_retrain_v02_device.json"),
            "provenance": Path("logs/grouped_retrain_v02_provenance.json"),
        },
    },
}

REGIONS = ("affected", "core_affected", "noncore_affected", "halo")
STATES = ("unclipped", "clipped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the existing v0.2 Moderate checkpoint on immutable, "
            "duplicate-safe grouped test manifests. This is diagnostic, not final."
        )
    )
    parser.add_argument(
        "--audit-run-dir",
        type=Path,
        required=True,
        help="Master research_correctness_audit_* run directory.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing the four grouped test blend CSV manifests.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/Galaxy10_DECals.h5",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Explicit best checkpoint matching --checkpoint-mode.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=tuple(MODE_CONFIG),
        default="historical-old",
        help=(
            "Strict checkpoint semantic contract. Historical mode remains the default; "
            "grouped-retrain requires the grouped experiment metadata."
        ),
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="Optional required SHA-256 for the supplied checkpoint.",
    )
    parser.add_argument(
        "--source-split-manifest",
        type=Path,
        help=(
            "Grouped source_split_manifest.csv. If omitted, resolve it by the "
            "SHA-256 stored in the blend manifests."
        ),
    )
    parser.add_argument(
        "--existing-v02-suite-metrics",
        type=Path,
        help=(
            "Required in grouped-retrain mode: the prior "
            "grouped_existing_v02_suite_metrics.csv for aligned comparison."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--preload-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sequentially preload the immutable uint8 HDF5 image array into host "
            "RAM before replay. This changes I/O only, not sample values."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Full evaluation accelerator: auto, mps, or cuda. CPU is rejected.",
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Omit the threshold sanity baseline (identity is always included).",
    )
    return parser.parse_args()


def resolved_input(path: Path) -> Path:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    return path.resolve()


def validate_sha(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256.")
    return normalized


def manifest_float(value: Any, label: str, *, require_finite: bool = False) -> float:
    """Parse CSV numeric values while preserving explicit missing/NaN metadata."""

    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan"}):
        parsed = float("nan")
    else:
        parsed = float(value)
    if require_finite and not math.isfinite(parsed):
        raise ValueError(f"Manifest field {label} must be finite, got {value!r}.")
    return parsed


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes using grouped-manifest v1 semantics."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


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


def unique_manifest_value(frames: Iterable[pd.DataFrame], column: str) -> str:
    values: set[str] = set()
    for frame in frames:
        values.update(str(value).strip() for value in frame[column].dropna().tolist())
    if len(values) != 1:
        raise ValueError(
            f"Manifest column {column!r} must contain one common value; got {sorted(values)}."
        )
    return next(iter(values))


def preflight_outputs(
    audit_run_dir: Path,
    output_names: dict[str, Path],
) -> dict[str, Path]:
    output_root = (PROJECT_ROOT / "outputs/runs").resolve()
    audit_run_dir = resolved_input(audit_run_dir)
    if not audit_run_dir.is_relative_to(output_root):
        raise ValueError(f"Audit output must remain under {output_root}: {audit_run_dir}")
    if not audit_run_dir.is_dir():
        raise FileNotFoundError(f"Audit run directory does not exist: {audit_run_dir}")
    required_directories = sorted({str(path.parent) for path in output_names.values()})
    for child in required_directories:
        directory = audit_run_dir / child
        if not directory.is_dir():
            raise FileNotFoundError(f"Required audit subdirectory is missing: {directory}")
    outputs = {key: audit_run_dir / relative for key, relative in output_names.items()}
    collisions = [str(path) for path in outputs.values() if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(collisions))
    return outputs


def load_manifests(manifest_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest_dir = resolved_input(manifest_dir)
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"Grouped blend manifest directory not found: {manifest_dir}")
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {}
    all_ids: list[str] = []
    for expected_suite, filename in SUITE_FILES.items():
        path = manifest_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required grouped test manifest missing: {path}")
        # Exact float32 blend replay depends on recovering the binary64 CSV
        # parameters without the default fast-parser rounding.  Pandas'
        # round-trip parser preserves the emitted Python float representation.
        frame = pd.read_csv(
            path,
            keep_default_na=False,
            float_precision="round_trip",
        )
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {', '.join(missing)}")
        if frame.empty:
            raise ValueError(f"Grouped test manifest is empty: {path}")
        ids = frame["sample_id"].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample_id values within {path.name}.")
        split_values = set(frame["source_split"].astype(str).str.lower())
        if split_values != {"test"}:
            raise ValueError(f"{path.name} must use source_split=test, got {split_values}.")
        suite_values = set(frame["suite"].astype(str))
        accepted_suite_values = {
            expected_suite,
            f"{expected_suite}_test",
            expected_suite.replace("_obstruction", ""),
            f"{expected_suite.replace('_obstruction', '')}_test",
            Path(filename).stem.replace("_blends", ""),
            Path(filename).stem.replace("_test_blends", ""),
        }
        if not suite_values.issubset(accepted_suite_values):
            raise ValueError(
                f"{path.name} has unexpected suite labels {suite_values}; "
                f"expected {accepted_suite_values}."
            )
        all_ids.extend(ids)
        frames[expected_suite] = frame.reset_index(drop=True)
        metadata[expected_suite] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": int(len(frame)),
        }
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("sample_id values must be globally unique across grouped suites.")
    summary_path = manifest_dir / "manifest_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Grouped manifest summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_hashes = summary.get("provenance", {}).get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError(
            "manifest_summary.json lacks provenance.output_sha256 file anchors."
        )
    for suite, suite_metadata in metadata.items():
        filename = Path(suite_metadata["path"]).name
        expected = validate_sha(str(output_hashes.get(filename, "")), filename)
        if suite_metadata["sha256"] != expected:
            raise ValueError(
                f"Manifest file hash mismatch for {filename}: "
                f"summary={expected}, actual={suite_metadata['sha256']}."
            )
        suite_metadata["summary_anchored_sha256"] = expected
    metadata_summary = {
        "path": str(summary_path),
        "sha256": file_sha256(summary_path),
        "status": summary.get("status"),
        "manifest_role": summary.get("manifest_role"),
        "paper_ready_final_test": summary.get("paper_ready_final_test"),
    }
    if metadata_summary["status"] != "complete":
        raise ValueError(f"Grouped manifest summary is not complete: {metadata_summary}")
    if metadata_summary["paper_ready_final_test"] is not False:
        raise ValueError(
            "This evaluator expects grouped development manifests explicitly marked not final."
        )
    for suite_metadata in metadata.values():
        suite_metadata["manifest_summary"] = metadata_summary
    return frames, metadata


def resolve_source_split_manifest(
    explicit_path: Path | None,
    expected_sha: str,
    manifest_dir: Path,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(resolved_input(explicit_path))
    summary_path = manifest_dir / "manifest_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in (
            "source_split_manifest_path",
            "source_split_manifest",
            "grouped_source_split_manifest",
        ):
            value = summary.get(key)
            if isinstance(value, str) and value:
                raw = Path(value)
                candidates.append((raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve())
    candidates.append(manifest_dir / "source_split_manifest.csv")
    candidates.extend(
        sorted(
            (PROJECT_ROOT / "data/manifests").glob(
                "grouped_source_split_*/source_split_manifest.csv"
            )
        )
    )
    seen: set[Path] = set()
    available: list[tuple[Path, str]] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        actual = file_sha256(candidate)
        available.append((candidate, actual))
        if actual == expected_sha:
            return candidate
    detail = ", ".join(f"{path} ({sha})" for path, sha in available) or "none found"
    raise FileNotFoundError(
        "Could not resolve the grouped source split manifest with SHA-256 "
        f"{expected_sha}. Candidates: {detail}"
    )


def source_manifest_mapping(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    aliases = {
        "source_index": ("source_index", "global_source_index", "index"),
        "group_id": ("group_id", "source_group_id"),
        "split": ("split", "source_split"),
        "label": ("label", "class_label"),
    }
    chosen: dict[str, str] = {}
    for canonical, options in aliases.items():
        selected = next((name for name in options if name in frame.columns), None)
        if selected is None:
            raise ValueError(
                f"Source split manifest {path} lacks a column for {canonical}: {options}."
            )
        chosen[canonical] = selected
    indices = pd.to_numeric(frame[chosen["source_index"]], errors="raise").astype(int)
    if indices.duplicated().any():
        raise ValueError("Source split manifest contains duplicate source indices.")
    mapping: dict[int, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        source_index = int(row[chosen["source_index"]])
        mapping[source_index] = {
            "group_id": str(row[chosen["group_id"]]),
            "split": str(row[chosen["split"]]).lower(),
            "label": int(row[chosen["label"]]),
        }
    return mapping, chosen


def verify_manifest_provenance(
    frames: dict[str, pd.DataFrame],
    manifest_dir: Path,
    dataset: Path,
    source_split_arg: Path | None,
) -> tuple[Path, dict[int, dict[str, Any]], dict[str, Any]]:
    frame_values = list(frames.values())
    stored_dataset_sha = validate_sha(
        unique_manifest_value(frame_values, "dataset_sha256"), "dataset_sha256"
    )
    actual_dataset_sha = file_sha256(dataset)
    if stored_dataset_sha != actual_dataset_sha:
        raise ValueError(
            f"Dataset SHA mismatch: manifest={stored_dataset_sha}, actual={actual_dataset_sha}."
        )
    stored_generator_sha = validate_sha(
        unique_manifest_value(frame_values, "generator_sha256"), "generator_sha256"
    )
    generator_path = PROJECT_ROOT / "src/blend.py"
    actual_generator_sha = file_sha256(generator_path)
    if stored_generator_sha != actual_generator_sha:
        raise ValueError(
            "Blend generator code changed after manifest creation: "
            f"manifest={stored_generator_sha}, actual={actual_generator_sha}."
        )
    stored_source_sha = validate_sha(
        unique_manifest_value(frame_values, "source_split_manifest_sha256"),
        "source_split_manifest_sha256",
    )
    source_path = resolve_source_split_manifest(
        source_split_arg, stored_source_sha, manifest_dir
    )
    source_mapping, source_columns = source_manifest_mapping(source_path)
    versions = sorted(
        {
            str(value)
            for frame in frame_values
            for value in frame["generator_version"].tolist()
        }
    )
    return source_path, source_mapping, {
        "dataset": {
            "path": str(dataset),
            "size_bytes": dataset.stat().st_size,
            "sha256": actual_dataset_sha,
        },
        "generator": {
            "path": str(generator_path),
            "sha256": actual_generator_sha,
            "versions": versions,
        },
        "source_split_manifest": {
            "path": str(source_path),
            "sha256": stored_source_sha,
            "rows": len(source_mapping),
            "resolved_columns": source_columns,
        },
    }


def verify_source_membership(
    row: pd.Series,
    source_mapping: dict[int, dict[str, Any]],
) -> None:
    for role in ("target", "contaminant"):
        source_index = int(row[f"{role}_source_index"])
        if source_index not in source_mapping:
            raise ValueError(
                f"sample_id={row['sample_id']}: {role} source {source_index} "
                "is absent from source split manifest."
            )
        source = source_mapping[source_index]
        if source["split"] != "test":
            raise ValueError(
                f"sample_id={row['sample_id']}: {role} source {source_index} "
                f"belongs to {source['split']}, not test."
            )
        if str(row[f"{role}_group_id"]) != source["group_id"]:
            raise ValueError(
                f"sample_id={row['sample_id']}: {role} group ID does not match "
                "the source split manifest."
            )
        if int(row[f"{role}_label"]) != source["label"]:
            raise ValueError(
                f"sample_id={row['sample_id']}: {role} label does not match "
                "the source split manifest."
            )
    if str(row["target_group_id"]) == str(row["contaminant_group_id"]):
        raise ValueError(
            f"sample_id={row['sample_id']}: target and contaminant share a source group."
        )


def replay_row(
    row: pd.Series,
    images: h5py.Dataset,
    labels: h5py.Dataset,
    pxscale: h5py.Dataset,
    source_mapping: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    verify_source_membership(row, source_mapping)
    target_index = int(row["target_source_index"])
    contaminant_index = int(row["contaminant_source_index"])
    n_sources = int(images.shape[0])
    if not (0 <= target_index < n_sources and 0 <= contaminant_index < n_sources):
        raise IndexError(f"sample_id={row['sample_id']}: source index out of range.")
    if int(labels[target_index]) != int(row["target_label"]):
        raise ValueError(f"sample_id={row['sample_id']}: target HDF5 label mismatch.")
    if int(labels[contaminant_index]) != int(row["contaminant_label"]):
        raise ValueError(f"sample_id={row['sample_id']}: contaminant HDF5 label mismatch.")
    target_pxscale = manifest_float(
        row["target_pxscale"], "target_pxscale", require_finite=True
    )
    contaminant_pxscale = manifest_float(
        row["contaminant_pxscale"], "contaminant_pxscale", require_finite=True
    )
    if not np.isclose(float(pxscale[target_index]), target_pxscale, rtol=0, atol=1e-12):
        raise ValueError(f"sample_id={row['sample_id']}: target pxscale mismatch.")
    if not np.isclose(
        float(pxscale[contaminant_index]),
        contaminant_pxscale,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(f"sample_id={row['sample_id']}: contaminant pxscale mismatch.")

    target = np.asarray(images[target_index], dtype=np.float32) / 255.0
    contaminant = np.asarray(images[contaminant_index], dtype=np.float32) / 255.0
    noise_seed_column = "noise_seed" if "noise_seed" in row.index else "sample_seed"
    rng = np.random.default_rng(int(row[noise_seed_column]))
    blended, info = gd_blend.blend_pair(
        target=target,
        contaminant=contaminant,
        shift=(int(row["shift_x"]), int(row["shift_y"])),
        rotation=manifest_float(
            row["rotation_degrees"], "rotation_degrees", require_finite=True
        ),
        brightness=manifest_float(
            row["brightness_scale"], "brightness_scale", require_finite=True
        ),
        blur_sigma=manifest_float(
            row["blur_sigma"], "blur_sigma", require_finite=True
        ),
        noise_std=manifest_float(row["noise_std"], "noise_std", require_finite=True),
        rng=rng,
    )
    affected = gd_utils.affected_region_mask(
        target,
        blended,
        threshold=manifest_float(
            row["affected_threshold"], "affected_threshold", require_finite=True
        ),
    )
    core = gd_utils.evaluation_core_mask_p85_v1(target)
    core_affected = affected & core
    noncore_affected = affected & ~core
    halo = gd_utils.halo_band_mask_manhattan_v1(affected, dilation_iters=5)
    arrays = {
        "affected": affected,
        "core": core,
        "core_affected": core_affected,
        "noncore_affected": noncore_affected,
        "halo": halo,
    }
    expected_hashes = {
        "blend_sha256": array_sha256(blended),
        # Manifest v1 canonicalizes boolean masks to uint8 before hashing so
        # the byte contract is explicit across CSV producers/consumers.
        "affected_mask_sha256": array_sha256(affected.astype(np.uint8)),
        "core_mask_sha256": array_sha256(core.astype(np.uint8)),
        "halo_mask_sha256": array_sha256(halo.astype(np.uint8)),
    }
    for column, actual in expected_hashes.items():
        expected = validate_sha(str(row[column]), column)
        if actual != expected:
            raise ValueError(
                f"sample_id={row['sample_id']}: replay hash mismatch for {column}: "
                f"manifest={expected}, replay={actual}."
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
                f"sample_id={row['sample_id']}: replay metadata mismatch for {column}: "
                f"manifest={expected}, replay={actual}."
            )
    identity_affected_mae = gd_utils.masked_mae(blended, target, affected)
    severity = float(affected.mean()) * identity_affected_mae * (
        1.0 + checks["core_obstruction_fraction"]
    )
    if not np.isclose(
        severity,
        manifest_float(row["blend_severity_score"], "blend_severity_score"),
        rtol=1e-6,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError(
            f"sample_id={row['sample_id']}: blend_severity_score replay mismatch."
        )
    return target, blended, arrays


def load_v02_checkpoint(
    path: Path,
    expected_sha: str | None,
    device: torch.device,
    checkpoint_mode: str,
) -> tuple[nn.Module, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    actual_sha = file_sha256(path)
    if expected_sha is not None:
        expected = validate_sha(expected_sha, "expected checkpoint SHA-256")
        if actual_sha != expected:
            raise ValueError(
                f"Checkpoint SHA mismatch: expected={expected}, actual={actual_sha}."
            )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise TypeError("Expected a structured v0.2 checkpoint with model_state_dict.")
    common_metadata = {
        "checkpoint_kind": "best_validation",
        "residual_target": "blended_minus_target",
        "reconstruction": "blended_minus_predicted_residual",
        "output_activation": "identity",
    }
    mode_metadata = {
        "historical-old": {
            "experiment_name": "Thayer-BR v0.2 weighted residual loss",
            "variant_name": "moderate",
        },
        "grouped-retrain": {
            "experiment_name": "Thayer-BR v0.2 Moderate Grouped Retrain",
            "variant_name": "moderate_grouped",
        },
    }
    if checkpoint_mode not in mode_metadata:
        raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")
    required_metadata = {**mode_metadata[checkpoint_mode], **common_metadata}
    mismatches = {
        key: {"expected": expected, "actual": checkpoint.get(key)}
        for key, expected in required_metadata.items()
        if checkpoint.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint semantic metadata mismatch: {mismatches}")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint lacks config.model architecture metadata.")
    model_config = dict(config["model"])
    if checkpoint_mode == "grouped-retrain":
        top_level_model_config = checkpoint.get("model_config")
        if not isinstance(top_level_model_config, dict):
            raise ValueError(
                "Grouped-retrain checkpoint lacks required top-level model_config."
            )
        if dict(top_level_model_config) != model_config:
            raise ValueError(
                "Grouped-retrain checkpoint model_config disagrees with config.model."
            )
    if model_config.get("in_channels") != 3 or model_config.get("out_channels") != 3:
        raise ValueError(f"Unexpected v0.2 channel configuration: {model_config}")
    model = models.UNet(**model_config)
    model.out_activation = nn.Identity()
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise TypeError("model_state_dict must be a non-empty mapping.")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"Checkpoint state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    model.to(device)
    model.eval()
    if model.training:
        raise RuntimeError("Model remained in training mode after model.eval().")
    parameter_device_types = {parameter.device.type for parameter in model.parameters()}
    if parameter_device_types != {device.type}:
        raise RuntimeError(
            f"Mixed or wrong model devices: expected {device.type}, got {parameter_device_types}."
        )
    metadata = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": actual_sha,
        **{key: checkpoint[key] for key in required_metadata},
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "model_config": model_config,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "safe_torch_load_weights_only": True,
        "checkpoint_mode": checkpoint_mode,
    }
    return model, metadata


def masked_values(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    return (
        gd_utils.masked_mse(prediction, target, mask),
        gd_utils.masked_mae(prediction, target, mask),
    )


def state_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray],
    state: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"whole_mse_{state}": gd_utils.mse(prediction, target),
        f"whole_mae_{state}": gd_utils.mae(prediction, target),
        f"psnr_{state}": gd_utils.psnr(prediction, target, data_range=1.0),
        f"ssim_{state}": gd_utils.ssim_metric(prediction, target),
    }
    for region in REGIONS:
        region_mse, region_mae = masked_values(prediction, target, masks[region])
        row[f"{region}_mse_{state}"] = region_mse
        row[f"{region}_mae_{state}"] = region_mae
    return row


def evaluate(
    frames: dict[str, pd.DataFrame],
    dataset_path: Path,
    source_mapping: dict[int, dict[str, Any]],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    include_threshold: bool,
    checkpoint_sha: str,
    learned_method: str,
    preload_images: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    rows: list[dict[str, Any]] = []
    replay_count = 0
    with h5py.File(dataset_path, "r") as handle:
        required_keys = {"images", "ans", "pxscale"}
        missing = required_keys - set(handle.keys())
        if missing:
            raise KeyError(f"Dataset lacks required keys: {sorted(missing)}")
        images_dataset = handle["images"]
        labels = np.asarray(handle["ans"][:]).squeeze()
        pxscale = np.asarray(handle["pxscale"][:]).squeeze()
        if (
            images_dataset.ndim != 4
            or images_dataset.shape[-1] != 3
            or images_dataset.dtype != np.uint8
        ):
            raise ValueError(
                "Expected uint8 Galaxy10 RGB images (N,H,W,3), got "
                f"{images_dataset.shape} {images_dataset.dtype}."
            )
        if preload_images:
            print(
                "Sequentially preloading immutable uint8 source images "
                f"({images_dataset.nbytes / 1024**3:.2f} GiB).",
                flush=True,
            )
            images: h5py.Dataset | np.ndarray = np.asarray(
                images_dataset[:], dtype=np.uint8
            )
        else:
            images = images_dataset
        for canonical_suite, frame in frames.items():
            print(f"Replaying and evaluating {canonical_suite}: {len(frame)} samples", flush=True)
            for start in range(0, len(frame), batch_size):
                batch_frame = frame.iloc[start : start + batch_size]
                replayed: list[tuple[pd.Series, np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
                for _, manifest_row in batch_frame.iterrows():
                    target, blended, masks = replay_row(
                        manifest_row, images, labels, pxscale, source_mapping
                    )
                    replayed.append((manifest_row, target, blended, masks))
                    replay_count += 1
                inputs = np.stack([item[2] for item in replayed]).astype(np.float32)
                tensor = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).to(device)
                if tensor.device.type != device.type:
                    raise RuntimeError(
                        f"Input tensor placed on {tensor.device}, expected {device}."
                    )
                model.eval()
                with torch.no_grad():
                    predicted_residual = (
                        model(tensor).detach().cpu().numpy().transpose(0, 2, 3, 1)
                    )
                if predicted_residual.shape != inputs.shape:
                    raise RuntimeError(
                        f"v0.2 output shape {predicted_residual.shape} != input {inputs.shape}."
                    )
                if not np.isfinite(predicted_residual).all():
                    raise RuntimeError("v0.2 produced non-finite residual predictions.")

                for offset, (manifest_row, target, blended, masks) in enumerate(replayed):
                    methods: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
                        "identity": (baselines.identity_baseline(blended), None),
                        learned_method: (
                            blended - predicted_residual[offset],
                            predicted_residual[offset],
                        ),
                    }
                    if include_threshold:
                        methods["threshold"] = (baselines.threshold_baseline(blended), None)
                    for method, (unclipped, residual) in methods.items():
                        unclipped = np.asarray(unclipped, dtype=np.float32)
                        clipped = np.clip(unclipped, 0.0, 1.0).astype(np.float32)
                        low = unclipped < 0.0
                        high = unclipped > 1.0
                        result: dict[str, Any] = {
                            "sample_id": str(manifest_row["sample_id"]),
                            "suite": canonical_suite,
                            "manifest_suite": str(manifest_row["suite"]),
                            "manifest_row_index": int(start + offset),
                            "method": method,
                            "source_split": "test",
                            "target_source_index": int(manifest_row["target_source_index"]),
                            "target_group_id": str(manifest_row["target_group_id"]),
                            "contaminant_source_index": int(
                                manifest_row["contaminant_source_index"]
                            ),
                            "contaminant_group_id": str(
                                manifest_row["contaminant_group_id"]
                            ),
                            "sample_seed": int(manifest_row["sample_seed"]),
                            "attempt_index": int(manifest_row["attempt_index"]),
                            "affected_threshold": manifest_float(
                                manifest_row["affected_threshold"],
                                "affected_threshold",
                                require_finite=True,
                            ),
                            "affected_pixel_count": int(masks["affected"].sum()),
                            "core_affected_pixel_count": int(
                                masks["core_affected"].sum()
                            ),
                            "noncore_affected_pixel_count": int(
                                masks["noncore_affected"].sum()
                            ),
                            "halo_pixel_count": int(masks["halo"].sum()),
                            "whole_pixel_count": int(target.shape[0] * target.shape[1]),
                            "channel_value_fraction_clipped_low": float(low.mean()),
                            "channel_value_fraction_clipped_high": float(high.mean()),
                            "pixel_fraction_any_channel_clipped_low": float(
                                np.any(low, axis=2).mean()
                            ),
                            "pixel_fraction_any_channel_clipped_high": float(
                                np.any(high, axis=2).mean()
                            ),
                            "checkpoint_sha256": (
                                checkpoint_sha if method == learned_method else ""
                            ),
                        }
                        if residual is None:
                            result.update(
                                {
                                    "predicted_residual_abs_mean": float("nan"),
                                    "predicted_residual_negative_fraction": float("nan"),
                                    "predicted_residual_positive_fraction": float("nan"),
                                    "over_subtraction_fraction": float("nan"),
                                }
                            )
                        else:
                            result.update(
                                {
                                    "predicted_residual_abs_mean": float(
                                        np.mean(np.abs(residual))
                                    ),
                                    "predicted_residual_negative_fraction": float(
                                        np.mean(residual < 0.0)
                                    ),
                                    "predicted_residual_positive_fraction": float(
                                        np.mean(residual > 0.0)
                                    ),
                                    "over_subtraction_fraction": float(
                                        np.mean(residual > blended)
                                    ),
                                }
                            )
                        result.update(state_metrics(unclipped, target, masks, "unclipped"))
                        result.update(state_metrics(clipped, target, masks, "clipped"))
                        rows.append(result)
                del tensor, predicted_residual, inputs, replayed
                completed = min(start + batch_size, len(frame))
                if completed % 100 == 0 or completed == len(frame):
                    print(
                        f"  {canonical_suite}: replayed/evaluated {completed}/{len(frame)}",
                        flush=True,
                    )
    per_sample = pd.DataFrame(rows)
    if per_sample.empty:
        raise RuntimeError("Evaluation produced no per-sample rows.")
    per_sample = add_paired_identity_outcomes(per_sample)
    return per_sample, {
        "replay_rows_verified": replay_count,
        "replay_hashes_checked_per_row": [
            "blend_sha256",
            "affected_mask_sha256",
            "core_mask_sha256",
            "halo_mask_sha256",
        ],
        "replay_status": "all_exact",
        "preload_images": bool(preload_images),
    }


def add_paired_identity_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    result_parts: list[pd.DataFrame] = []
    for suite, suite_frame in frame.groupby("suite", sort=False):
        identity = suite_frame[suite_frame["method"] == "identity"].copy()
        identity_ids = identity["sample_id"].tolist()
        if len(identity_ids) != len(set(identity_ids)):
            raise ValueError(f"Identity sample IDs are not unique for {suite}.")
        identity = identity.set_index("sample_id", drop=False)
        for method, method_frame in suite_frame.groupby("method", sort=False):
            method_frame = method_frame.copy()
            method_ids = method_frame["sample_id"].tolist()
            if method_ids != identity_ids:
                raise ValueError(
                    f"Immutable sample alignment failed for suite={suite}, method={method}."
                )
            for state in STATES:
                candidate = method_frame[f"affected_mse_{state}"].to_numpy(float)
                reference = identity.loc[method_ids, f"affected_mse_{state}"].to_numpy(float)
                valid = np.isfinite(candidate) & np.isfinite(reference)
                ratio = np.full(len(candidate), np.nan, dtype=float)
                positive = valid & (candidate > 0)
                ratio[positive] = reference[positive] / candidate[positive]
                ratio[valid & (candidate == 0) & (reference > 0)] = np.inf
                ratio[valid & (candidate == 0) & (reference == 0)] = 1.0
                method_frame[f"improvement_ratio_vs_identity_{state}"] = ratio
                method_frame[f"valid_identity_pair_{state}"] = valid
                method_frame[f"worse_than_identity_{state}"] = valid & (
                    candidate > reference
                )
                method_frame[f"beats_identity_{state}"] = valid & (
                    candidate < reference
                )
                method_frame[f"ties_identity_{state}"] = valid & (
                    candidate == reference
                )
            result_parts.append(method_frame)
    result = pd.concat(result_parts, ignore_index=True)
    method_order = {
        "identity": 0,
        "threshold": 1,
        "v02_moderate_old_split": 2,
        "v02_moderate_grouped_retrain": 3,
    }
    result["_method_order"] = result["method"].map(method_order)
    if result["_method_order"].isna().any():
        unknown = sorted(result.loc[result["_method_order"].isna(), "method"].unique())
        raise ValueError(f"Unknown method labels prevent deterministic alignment: {unknown}")
    result = result.sort_values(
        ["suite", "manifest_row_index", "_method_order"], kind="stable"
    ).drop(columns="_method_order")
    expected = result.groupby(["suite", "method"])["sample_id"].nunique()
    totals = result.groupby(["suite", "method"]).size()
    if not expected.equals(totals):
        raise ValueError("Duplicate sample IDs detected after paired-outcome construction.")
    return result.reset_index(drop=True)


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def weighted_micro(frame: pd.DataFrame, metric: str, region: str, state: str) -> float:
    value_column = f"{region}_{metric}_{state}"
    count_column = "whole_pixel_count" if region == "whole" else f"{region}_pixel_count"
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(float)
    counts = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & (counts > 0)
    if not valid.any():
        return float("nan")
    return float(np.sum(values[valid] * counts[valid]) / np.sum(counts[valid]))


def aggregate(per_sample: pd.DataFrame, evaluation_status: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (suite, method), frame in per_sample.groupby(["suite", "method"], sort=False):
        ids = frame["sample_id"].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Non-unique aggregate IDs for {suite}/{method}.")
        for state in STATES:
            n_valid_pairs = int(frame[f"valid_identity_pair_{state}"].sum())
            worse_count = int(frame[f"worse_than_identity_{state}"].sum())
            win_count = int(frame[f"beats_identity_{state}"].sum())
            tie_count = int(frame[f"ties_identity_{state}"].sum())
            row: dict[str, Any] = {
                "suite": suite,
                "method": method,
                "reconstruction_state": state,
                "n_total": int(len(frame)),
                "n_valid_affected": int(frame[f"affected_mse_{state}"].notna().sum()),
                "n_valid_core_affected": int(
                    frame[f"core_affected_mse_{state}"].notna().sum()
                ),
                "n_valid_noncore_affected": int(
                    frame[f"noncore_affected_mse_{state}"].notna().sum()
                ),
                "n_valid_halo": int(frame[f"halo_mse_{state}"].notna().sum()),
                "whole_mse": finite_mean(frame[f"whole_mse_{state}"]),
                "whole_mae": finite_mean(frame[f"whole_mae_{state}"]),
                "psnr": finite_mean(frame[f"psnr_{state}"]),
                "ssim": finite_mean(frame[f"ssim_{state}"]),
                "per_sample_improvement_ratio_vs_identity_mean": finite_mean(
                    frame[f"improvement_ratio_vs_identity_{state}"]
                ),
                "per_sample_improvement_ratio_vs_identity_median": finite_median(
                    frame[f"improvement_ratio_vs_identity_{state}"]
                ),
                "n_valid_pairs_vs_identity": n_valid_pairs,
                "worse_than_identity_count": worse_count,
                "worse_than_identity_rate": (
                    worse_count / n_valid_pairs if n_valid_pairs else float("nan")
                ),
                "win_count_vs_identity": win_count,
                "win_rate_vs_identity": (
                    win_count / n_valid_pairs if n_valid_pairs else float("nan")
                ),
                "tie_count_vs_identity": tie_count,
                "tie_rate_vs_identity": (
                    tie_count / n_valid_pairs if n_valid_pairs else float("nan")
                ),
                "channel_value_fraction_clipped_low": finite_mean(
                    frame["channel_value_fraction_clipped_low"]
                ),
                "channel_value_fraction_clipped_high": finite_mean(
                    frame["channel_value_fraction_clipped_high"]
                ),
                "pixel_fraction_any_channel_clipped_low": finite_mean(
                    frame["pixel_fraction_any_channel_clipped_low"]
                ),
                "pixel_fraction_any_channel_clipped_high": finite_mean(
                    frame["pixel_fraction_any_channel_clipped_high"]
                ),
                "predicted_residual_abs_mean": finite_mean(
                    frame["predicted_residual_abs_mean"]
                ),
                "predicted_residual_negative_fraction": finite_mean(
                    frame["predicted_residual_negative_fraction"]
                ),
                "over_subtraction_fraction": finite_mean(
                    frame["over_subtraction_fraction"]
                ),
                "aggregation_primary": "macro_per_sample",
                "mask_definition": "blend_minus_target_independent_of_prediction",
                "evaluation_status": evaluation_status,
            }
            for region in ("whole", *REGIONS):
                for metric in ("mse", "mae"):
                    column = f"{region}_{metric}_{state}"
                    macro = finite_mean(frame[column])
                    micro = weighted_micro(frame, metric, region, state)
                    row[f"{region}_{metric}_macro"] = macro
                    row[f"{region}_{metric}_micro"] = micro
                    if region != "whole":
                        row[f"{region}_{metric}"] = macro
            rows.append(row)
    summary = pd.DataFrame(rows)
    # The aggregate improvement used for development comparisons is a ratio of
    # paired suite-level macro affected MSEs, not the mean per-sample ratio.
    summary["affected_mse_ratio_identity_over_method"] = np.nan
    for (suite, state), group in summary.groupby(
        ["suite", "reconstruction_state"], sort=False
    ):
        identity_rows = group[group["method"] == "identity"]
        if len(identity_rows) != 1:
            raise ValueError(f"Expected one identity aggregate for {suite}/{state}.")
        identity_mse = float(identity_rows.iloc[0]["affected_mse_macro"])
        indices = group.index
        denominators = summary.loc[indices, "affected_mse_macro"].to_numpy(float)
        ratios = np.divide(
            identity_mse,
            denominators,
            out=np.full_like(denominators, np.nan),
            where=np.isfinite(denominators) & (denominators > 0),
        )
        summary.loc[indices, "affected_mse_ratio_identity_over_method"] = ratios
    summary["improvement_ratio_vs_identity"] = summary[
        "affected_mse_ratio_identity_over_method"
    ]
    return summary


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "NA"
            if np.isposinf(value):
                return "inf"
            if np.isneginf(value):
                return "-inf"
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def build_grouped_retrain_comparison(
    current_suite_metrics: pd.DataFrame,
    historical_suite_metrics_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine aligned grouped-test summaries without treating old dev values as final."""

    path = resolved_input(historical_suite_metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"Existing grouped v0.2 metrics table not found: {path}")
    before_sha = file_sha256(path)
    historical = pd.read_csv(path)
    required = {
        "suite",
        "method",
        "reconstruction_state",
        "n_total",
        "whole_mse",
        "affected_mse_macro",
        "core_affected_mse_macro",
        "noncore_affected_mse_macro",
        "halo_mse_macro",
        "improvement_ratio_vs_identity",
        "worse_than_identity_count",
        "ssim",
        "psnr",
    }
    missing = sorted(required - set(historical.columns))
    if missing:
        raise ValueError(
            f"Historical grouped metrics table lacks columns: {', '.join(missing)}"
        )
    historical_clipped = historical[
        historical["reconstruction_state"] == "clipped"
    ].copy()
    old_model = historical_clipped[
        historical_clipped["method"] == "v02_moderate_old_split"
    ].copy()
    if old_model.empty:
        raise ValueError(
            "Historical grouped table lacks clipped v02_moderate_old_split rows."
        )
    if not (old_model.groupby("suite").size() == 1).all():
        raise ValueError("Historical grouped table has duplicate old-model suite rows.")
    current_clipped = current_suite_metrics[
        current_suite_metrics["reconstruction_state"] == "clipped"
    ].copy()
    expected_suites = set(current_clipped["suite"])
    if set(old_model["suite"]) != expected_suites:
        raise ValueError(
            "Existing-old and grouped-retrain suite sets are not identical: "
            f"old={sorted(set(old_model['suite']))}, current={sorted(expected_suites)}"
        )

    # Identity metrics must be identical because both evaluations replay the
    # same immutable manifests. This catches misalignment before comparison.
    old_identity = historical_clipped[
        historical_clipped["method"] == "identity"
    ].set_index("suite")
    current_identity = current_clipped[
        current_clipped["method"] == "identity"
    ].set_index("suite")
    if old_identity.index.duplicated().any() or current_identity.index.duplicated().any():
        raise ValueError("Identity comparison rows must be unique by suite.")
    if set(old_identity.index) != expected_suites or set(current_identity.index) != expected_suites:
        raise ValueError("Identity rows are incomplete in grouped comparison inputs.")
    identity_columns = (
        "n_total",
        "whole_mse",
        "affected_mse_macro",
        "core_affected_mse_macro",
        "noncore_affected_mse_macro",
        "halo_mse_macro",
        "ssim",
        "psnr",
    )
    for suite in sorted(expected_suites):
        for column in identity_columns:
            old_value = float(old_identity.loc[suite, column])
            new_value = float(current_identity.loc[suite, column])
            if not np.isclose(old_value, new_value, rtol=1e-12, atol=1e-15, equal_nan=True):
                raise ValueError(
                    "Grouped comparison identity mismatch for "
                    f"suite={suite}, metric={column}: old={old_value}, current={new_value}."
                )

    old_model["comparison_role"] = "historical_old_split_checkpoint_on_grouped_test"
    current_clipped["comparison_role"] = current_clipped["method"].map(
        {
            "identity": "sanity_identity",
            "threshold": "sanity_threshold",
            "v02_moderate_grouped_retrain": "grouped_retrained_candidate",
        }
    )
    comparison = pd.concat([current_clipped, old_model], ignore_index=True, sort=False)
    method_order = {
        "identity": 0,
        "threshold": 1,
        "v02_moderate_old_split": 2,
        "v02_moderate_grouped_retrain": 3,
    }
    comparison["_method_order"] = comparison["method"].map(method_order)
    if comparison["_method_order"].isna().any():
        raise ValueError("Grouped comparison contains an unexpected method label.")
    comparison = comparison.sort_values(
        ["suite", "_method_order"], kind="stable"
    ).drop(columns="_method_order")
    after_sha = file_sha256(path)
    if before_sha != after_sha:
        raise RuntimeError("Historical grouped metrics table changed while being read.")
    return comparison.reset_index(drop=True), {
        "path": str(path),
        "sha256": before_sha,
        "rows": int(len(historical)),
        "identity_alignment": "exact_within_rtol_1e-12_atol_1e-15",
        "status": "aligned_grouped_development_comparison",
    }


def make_report(
    suite_metrics: pd.DataFrame,
    device: torch.device,
    checkpoint: dict[str, Any],
    manifest_metadata: dict[str, Any],
    provenance: dict[str, Any],
    checkpoint_mode: str,
    comparison_summary: pd.DataFrame | None,
) -> str:
    clipped = suite_metrics[suite_metrics["reconstruction_state"] == "clipped"].copy()
    columns = [
        "suite",
        "method",
        "n_total",
        "affected_mse_macro",
        "core_affected_mse_macro",
        "noncore_affected_mse_macro",
        "halo_mse_macro",
        "ssim",
        "affected_mse_ratio_identity_over_method",
        "worse_than_identity_count",
    ]
    manifest_rows = sum(int(value["rows"]) for value in manifest_metadata.values())
    if checkpoint_mode == "historical-old":
        title = "Existing v0.2 Moderate on grouped tests"
        status_text = """This is a **diagnostic grouped-source evaluation, not a final-paper result**.
The evaluated Thayer-BR v0.2 Moderate checkpoint was trained on the original
row-index split, before duplicate/source-level grouping was enforced. The
grouped test manifests are useful for measuring its generalization, but only a
new model trained on the grouped training/validation manifests can support a
duplicate-safe training claim."""
        table_path = "tables/grouped_existing_v02_suite_metrics.csv"
        claim_text = (
            "No claim in this report replaces the historical v0.2 development result "
            "or constitutes a locked final-test result."
        )
        comparison_section = ""
    else:
        title = "Grouped-retrained v0.2 Moderate on grouped tests"
        status_text = """This is a **duplicate-safe grouped development evaluation, not a locked
final-paper result**. The checkpoint was trained on grouped training and
validation manifests and is evaluated on disjoint grouped test sources. The
test suite remains a development benchmark until a separately locked final
manifest is evaluated without further model selection."""
        table_path = "tables/grouped_retrain_suite_metrics.csv"
        claim_text = (
            "The grouped number should be reported separately from the historical "
            "row-index development result; this report does not establish final-paper "
            "performance or training-seed robustness."
        )
        if comparison_summary is None:
            raise ValueError("Grouped-retrain report requires a comparison summary.")
        comparison_columns = [
            "suite",
            "method",
            "comparison_role",
            "affected_mse_macro",
            "core_affected_mse_macro",
            "halo_mse_macro",
            "improvement_ratio_vs_identity",
            "worse_than_identity_count",
        ]
        comparison_section = f"""
## Existing-old versus grouped-retrained comparison

{markdown_table(comparison_summary[comparison_columns])}

The old development headline is not inserted into this same-manifest table:
it was measured on different generated development suites and is therefore a
context value, not a sample-aligned comparator.
"""
    return f"""# {title}

## Status

{status_text}

## Execution and identity

- Selected tensor-inference device: `{device}` (CPU fallback was refused).
- Checkpoint: `{checkpoint['path']}`.
- Checkpoint SHA-256: `{checkpoint['sha256']}`.
- Checkpoint kind: `{checkpoint['checkpoint_kind']}`; variant: `{checkpoint['variant_name']}`.
- Residual target: `{checkpoint['residual_target']}`.
- Reconstruction: `{checkpoint['reconstruction']}`.
- Grouped manifest rows replayed exactly: `{manifest_rows}`.
- Replay status: `{provenance['replay_validation']['replay_status']}`.
- Source split manifest: `{provenance['input_provenance']['source_split_manifest']['path']}`.

## Clipped reconstruction metrics

{markdown_table(clipped[columns])}

The primary aggregate is the macro mean of per-sample metrics. Micro
(mask-pixel-weighted) values and explicit valid counts are retained in
`{table_path}`. Every comparator is evaluated
on the same replayed blend, target, affected mask, core mask, and halo mask.
Win/loss fields require identical, ordered, unique `sample_id` sequences.

## Clipping and mask interpretation

Both pre-clipping and post-clipping reconstructions are reported. Pixel and
channel clipping fractions, predicted-residual sign fractions, and an
over-subtraction indicator are retained so clipping cannot conceal unphysical
outputs. The affected mask is computed only from `blended - target`, never from
the prediction. Empty regional masks are represented by NaN and are excluded
with their coverage reported explicitly.

Identity and threshold are sanity checks, not competitive astronomical
deblenders. These experiments use synthetic RGB display-cutout compositing,
not calibrated FITS-band physical injection or survey-grade source separation.

## Provenance checks

- Manifest files: `{len(manifest_metadata)}` suites, `{manifest_rows}` rows.
- Dataset SHA-256 matched every manifest row.
- Generator SHA-256 matched the current audited `src/blend.py`.
- Group IDs, split labels, class labels, and pixel scales matched the grouped
  source manifest and HDF5 source data.
- Blend, affected-mask, core-mask, and halo-mask hashes matched exact replay.
- Tensor inference used only the blended RGB image; target and masks were used
  only after inference for metric calculation.

{comparison_section}
{claim_text}
"""


def verify_inputs_unchanged(
    checkpoint: dict[str, Any],
    manifest_metadata: dict[str, Any],
    input_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Re-hash immutable evaluation inputs after inference and fail on mutation."""

    checkpoint_path = Path(checkpoint["path"])
    checkpoint_after = {
        "size_bytes": checkpoint_path.stat().st_size,
        "mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "sha256": file_sha256(checkpoint_path),
    }
    checkpoint_unchanged = (
        checkpoint_after["size_bytes"] == checkpoint["size_bytes"]
        and checkpoint_after["mtime_ns"] == checkpoint["mtime_ns"]
        and checkpoint_after["sha256"] == checkpoint["sha256"]
    )
    manifest_checks: dict[str, Any] = {}
    for suite, metadata in manifest_metadata.items():
        path = Path(metadata["path"])
        after_sha = file_sha256(path)
        manifest_checks[suite] = {
            "path": str(path),
            "before_sha256": metadata["sha256"],
            "after_sha256": after_sha,
            "unchanged": after_sha == metadata["sha256"],
        }
    summary_records = {
        str(metadata["manifest_summary"]["path"]): metadata["manifest_summary"]
        for metadata in manifest_metadata.values()
    }
    if len(summary_records) != 1:
        raise RuntimeError(
            f"Expected one common manifest summary, found {sorted(summary_records)}"
        )
    summary_path_text, summary_before = next(iter(summary_records.items()))
    summary_path = Path(summary_path_text)
    summary_after_sha = file_sha256(summary_path)
    source_metadata = input_provenance["source_split_manifest"]
    source_path = Path(source_metadata["path"])
    source_after_sha = file_sha256(source_path)
    generator_metadata = input_provenance["generator"]
    generator_path = Path(generator_metadata["path"])
    generator_after_sha = file_sha256(generator_path)
    result = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "before": {
                "size_bytes": checkpoint["size_bytes"],
                "mtime_ns": checkpoint["mtime_ns"],
                "sha256": checkpoint["sha256"],
            },
            "after": checkpoint_after,
            "unchanged": checkpoint_unchanged,
        },
        "manifests": manifest_checks,
        "manifest_summary": {
            "path": str(summary_path),
            "before_sha256": summary_before["sha256"],
            "after_sha256": summary_after_sha,
            "unchanged": summary_after_sha == summary_before["sha256"],
        },
        "source_split_manifest": {
            "path": str(source_path),
            "before_sha256": source_metadata["sha256"],
            "after_sha256": source_after_sha,
            "unchanged": source_after_sha == source_metadata["sha256"],
        },
        "generator": {
            "path": str(generator_path),
            "before_sha256": generator_metadata["sha256"],
            "after_sha256": generator_after_sha,
            "unchanged": generator_after_sha == generator_metadata["sha256"],
        },
    }
    all_unchanged = (
        checkpoint_unchanged
        and all(check["unchanged"] for check in manifest_checks.values())
        and result["manifest_summary"]["unchanged"]
        and result["source_split_manifest"]["unchanged"]
        and result["generator"]["unchanged"]
    )
    result["all_unchanged"] = bool(all_unchanged)
    if not all_unchanged:
        raise RuntimeError(f"An immutable evaluation input changed during inference: {result}")
    return result


def run_evaluation(
    args: argparse.Namespace,
    outputs: dict[str, Path],
    attempt_stamp: str,
) -> int:
    mode = MODE_CONFIG[args.checkpoint_mode]
    evaluation_status = str(mode["evaluation_status"])
    learned_method = str(mode["method"])
    log_prefix = str(mode["log_prefix"])
    if args.checkpoint_mode == "grouped-retrain" and args.existing_v02_suite_metrics is None:
        raise ValueError(
            "--existing-v02-suite-metrics is required in grouped-retrain mode."
        )
    if args.checkpoint_mode == "historical-old" and args.existing_v02_suite_metrics is not None:
        raise ValueError(
            "--existing-v02-suite-metrics is only valid in grouped-retrain mode."
        )
    manifest_dir = resolved_input(args.manifest_dir)
    dataset = resolved_input(args.dataset)
    checkpoint_path = resolved_input(args.checkpoint)
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    # Resolve and persist the accelerator before any model tensor is created.
    device = gd_train.resolve_accelerator(args.device)
    selected_at = datetime.now(timezone.utc).isoformat()
    device_record = {
        "selected_at_utc": selected_at,
        "requested_device": str(args.device),
        "selected_device": str(device),
        "selected_device_type": device.type,
        "cpu_full_inference_allowed": False,
        "mps_built": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "attempt_stamp": attempt_stamp,
    }
    print(f"Selected full-evaluation device: {device}", flush=True)
    device_attempt_path = (
        resolved_input(args.audit_run_dir)
        / "logs"
        / f"{log_prefix}_device_attempt_{attempt_stamp}.json"
    )
    safe_json(device_attempt_path, device_record)

    frames, manifest_metadata = load_manifests(manifest_dir)
    source_sha = validate_sha(
        unique_manifest_value(
            list(frames.values()), "source_split_manifest_sha256"
        ),
        "source_split_manifest_sha256",
    )
    source_path = resolve_source_split_manifest(
        args.source_split_manifest, source_sha, manifest_dir
    )
    source_path, source_mapping, input_provenance = verify_manifest_provenance(
        frames, manifest_dir, dataset, source_path
    )
    model, checkpoint_metadata = load_v02_checkpoint(
        checkpoint_path,
        args.expected_checkpoint_sha256,
        device,
        args.checkpoint_mode,
    )

    per_sample, replay_validation = evaluate(
        frames=frames,
        dataset_path=dataset,
        source_mapping=source_mapping,
        model=model,
        device=device,
        batch_size=args.batch_size,
        include_threshold=not args.skip_threshold,
        checkpoint_sha=checkpoint_metadata["sha256"],
        learned_method=learned_method,
        preload_images=bool(args.preload_images),
    )
    suite_metrics = aggregate(per_sample, evaluation_status)
    comparison_summary: pd.DataFrame | None = None
    comparison_provenance: dict[str, Any] | None = None
    if args.checkpoint_mode == "grouped-retrain":
        comparison_summary, comparison_provenance = build_grouped_retrain_comparison(
            suite_metrics,
            args.existing_v02_suite_metrics,
        )
    input_integrity_after = verify_inputs_unchanged(
        checkpoint_metadata, manifest_metadata, input_provenance
    )

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": evaluation_status,
        "checkpoint_mode": args.checkpoint_mode,
        "learned_method": learned_method,
        "command": " ".join(sys.argv),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "git": {
            "branch": command_output(["git", "branch", "--show-current"]),
            "head": command_output(["git", "rev-parse", "HEAD"]),
            "status_porcelain": command_output(["git", "status", "--porcelain"]),
        },
        "device": device_record,
        "pre_inference_device_attempt_log": str(device_attempt_path),
        "checkpoint": checkpoint_metadata,
        "manifests": manifest_metadata,
        "input_provenance": input_provenance,
        "input_integrity_after_inference": input_integrity_after,
        "replay_validation": replay_validation,
        "comparison_input": comparison_provenance,
        "metric_protocol": {
            "primary_aggregation": "macro_per_sample",
            "secondary_aggregation": "mask_pixel_weighted_micro",
            "affected_mask": "mean(abs(blended-target),axis=RGB)>affected_threshold",
            "core_mask": "evaluation_core_mask_p85_v1",
            "halo_mask": "halo_band_mask_manhattan_v1(iterations=5)",
            "states": list(STATES),
            "data_range": 1.0,
            "sample_alignment": "ordered_unique_sample_id_required",
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    report = make_report(
        suite_metrics,
        device,
        checkpoint_metadata,
        manifest_metadata,
        provenance,
        args.checkpoint_mode,
        comparison_summary,
    )

    # Results are written only after every input, replay, inference, and
    # alignment check has completed successfully.
    safe_csv(outputs["sample"], per_sample)
    safe_csv(outputs["suite"], suite_metrics)
    if comparison_summary is not None:
        safe_csv(outputs["comparison"], comparison_summary)
    safe_text(outputs["report"], report)
    safe_json(outputs["provenance"], provenance)
    safe_json(outputs["device"], device_record)
    print(
        f"Wrote {args.checkpoint_mode} grouped evaluation under "
        f"{resolved_input(args.audit_run_dir)}"
    )
    return 0


def main() -> int:
    args = parse_args()
    mode = MODE_CONFIG[args.checkpoint_mode]
    output_names = dict(mode["outputs"])
    log_prefix = str(mode["log_prefix"])
    attempt_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    audit_run_dir = resolved_input(args.audit_run_dir)
    try:
        outputs = preflight_outputs(audit_run_dir, output_names)
        return run_evaluation(args, outputs, attempt_stamp)
    except Exception as exc:
        output_root = (PROJECT_ROOT / "outputs/runs").resolve()
        logs_dir = audit_run_dir / "logs"
        if audit_run_dir.is_relative_to(output_root) and logs_dir.is_dir():
            failure_path = (
                logs_dir
                / f"{log_prefix}_failure_attempt_{attempt_stamp}.json"
            )
            payload = {
                "attempt_stamp": attempt_stamp,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "status": "failed_no_result_claim",
                "canonical_result_outputs_written": {
                    key: bool((audit_run_dir / relative).exists())
                    for key, relative in output_names.items()
                },
            }
            try:
                safe_json(failure_path, payload)
            except Exception as log_exc:  # pragma: no cover - last-resort stderr only
                print(
                    f"Could not write failure diagnostic {failure_path}: {log_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
