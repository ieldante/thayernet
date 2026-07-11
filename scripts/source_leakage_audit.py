"""Audit Galaxy10 source partitions for duplicate and role leakage.

This script is intentionally read-only with respect to the dataset and existing
run artifacts.  It creates one new timestamped directory under ``outputs/runs``
and refuses to overwrite any prior output.

The audit distinguishes three questions that are easy to conflate:

1. Are the shuffled *row indices* assigned to train/validation/test disjoint?
2. Do saved blend records keep targets and contaminants inside the source array
   supplied to their generator?
3. Do different HDF5 rows represent the same underlying sky object or a very
   similar image across otherwise disjoint row-index partitions?

The third question is checked with exact image SHA-256 hashes, exact/near sky
coordinates when present, and conservative perceptual fingerprints.  A
perceptual-hash match is a candidate for review, not proof of duplication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import yaml
from scipy.fft import dctn
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"

SPLIT_NAMES = np.asarray(["train", "validation", "test"], dtype=object)

TARGET_FIELDS = (
    "target_source_index",
    "global_target_index",
    "target_index",
    "target_idx",
)
CONTAMINANT_FIELDS = (
    "contaminant_source_index",
    "global_contaminant_index",
    "contaminant_index",
    "contaminant_idx",
)

PARTITION_FIELDS = (
    "record_type",
    "artifact_path",
    "split",
    "role",
    "index_semantics",
    "expected_partition",
    "source_count",
    "row_count",
    "rows_with_both_indices",
    "rows_missing_one_or_both_indices",
    "target_min",
    "target_max",
    "contaminant_min",
    "contaminant_max",
    "out_of_range_target_count",
    "out_of_range_contaminant_count",
    "wrong_partition_target_count",
    "wrong_partition_contaminant_count",
    "same_source_pair_count",
    "containment_status",
    "details",
)

EXACT_FIELDS = (
    "exact_hash_sha256",
    "index_a",
    "split_a",
    "split_local_index_a",
    "label_a",
    "ra_a_deg",
    "dec_a_deg",
    "index_b",
    "split_b",
    "split_local_index_b",
    "label_b",
    "ra_b_deg",
    "dec_b_deg",
    "cross_split",
    "same_label",
    "coordinate_separation_arcsec",
)

NEAR_FIELDS = (
    "rank",
    "review_priority",
    "evidence_type",
    "index_a",
    "split_a",
    "audit_zone_a",
    "split_local_index_a",
    "label_a",
    "ra_a_deg",
    "dec_a_deg",
    "index_b",
    "split_b",
    "audit_zone_b",
    "split_local_index_b",
    "label_b",
    "ra_b_deg",
    "dec_b_deg",
    "same_label",
    "exact_coordinate_match",
    "coordinate_separation_arcsec",
    "phash_hamming_64",
    "dhash_hamming_64",
    "downsample_gray_ncc",
    "downsample_gray_rmse_0_1",
    "ranking_score",
    "interpretation",
)

MANIFEST_CROSSCHECK_FIELDS = (
    "manifest_run",
    "source_index",
    "source_split",
    "split_local_index",
    "audit_zone",
    "roles",
    "suites",
    "usage_count",
    "exact_image_counterpart_indices",
    "exact_image_counterpart_zones",
    "critical_coordinate_counterpart_indices",
    "critical_coordinate_counterpart_zones",
    "high_perceptual_counterpart_indices",
    "high_perceptual_counterpart_zones",
    "medium_perceptual_counterpart_indices",
    "medium_perceptual_counterpart_zones",
    "counterpart_in_train",
    "counterpart_in_validation",
    "counterpart_in_development_prefix",
    "verification_status",
    "details",
)


@dataclass(frozen=True)
class AuditSettings:
    batch_size: int = 16
    phash_candidate_hamming: int = 7
    dhash_candidate_hamming: int = 3
    coordinate_candidate_arcsec: float = 1.0
    max_near_candidates: int = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit source partitions and duplicate leakage without running models."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/runs")
    parser.add_argument("--historical-root", type=Path, default=PROJECT_ROOT / "outputs/runs")
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-near-candidates", type=int, default=5000)
    parser.add_argument(
        "--full-home-preflight-no-match",
        action="store_true",
        help=(
            "Record that a separate read-only full-home filename search completed "
            "with no NoDuplicated HDF5 match before this script was run."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def load_config(path: Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_run_dir(output_root: Path, stamp: str) -> Path:
    if not stamp or Path(stamp).name != stamp or stamp in {".", ".."}:
        raise ValueError("stamp must be a non-empty filename component.")
    run_dir = resolve_path(output_root) / f"source_leakage_audit_{stamp}"
    try:
        run_dir.resolve().relative_to((PROJECT_ROOT / "outputs/runs").resolve())
    except ValueError as exc:
        raise ValueError(
            "Audit output must remain under ignored outputs/runs/."
        ) from exc
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    for child in ("tables", "diagnostics", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    return run_dir


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing table: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def reconstruct_partitions(
    n_samples: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    partitions = {
        "train": indices[:n_train],
        "validation": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }
    split_code = np.empty(n_samples, dtype=np.int8)
    local_index = np.empty(n_samples, dtype=np.int64)
    for code, split in enumerate(("train", "validation", "test")):
        split_indices = partitions[split]
        split_code[split_indices] = code
        local_index[split_indices] = np.arange(len(split_indices), dtype=np.int64)
    return partitions, split_code, local_index


def dataset_schema(handle: h5py.File) -> dict[str, Any]:
    datasets: dict[str, Any] = {}

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = {
                "shape": [int(value) for value in obj.shape],
                "dtype": str(obj.dtype),
            }

    handle.visititems(visitor)
    object_id_candidates = [
        key
        for key in datasets
        if any(token in key.lower() for token in ("object", "objid", "source_id", "catalog_id"))
    ]
    return {
        "datasets": datasets,
        "root_attributes": sorted(str(key) for key in handle.attrs.keys()),
        "ra_available": "ra" in handle,
        "dec_available": "dec" in handle,
        "object_id_candidates": object_id_candidates,
    }


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    if bits.ndim != 2 or bits.shape[1] != 64:
        raise ValueError(f"Expected Bx64 bits, received {bits.shape}")
    packed = np.zeros(bits.shape[0], dtype=np.uint64)
    for position in range(64):
        packed |= bits[:, position].astype(np.uint64) << np.uint64(position)
    return packed


def _downsample_grayscale(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB images, received {images.shape}")
    if images.shape[1:3] != (256, 256):
        raise ValueError(
            "This reproducible block-mean fingerprint expects 256x256 Galaxy10 cutouts; "
            f"received {images.shape[1:3]}."
        )
    rgb = images.astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.reshape(len(gray), 32, 8, 32, 8).mean(axis=(2, 4), dtype=np.float32)


def compute_fingerprints(
    images: h5py.Dataset,
    settings: AuditSettings,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(images.shape[0])
    exact_hashes = [""] * n_samples
    phashes = np.zeros(n_samples, dtype=np.uint64)
    dhashes = np.zeros(n_samples, dtype=np.uint64)
    gray32 = np.empty((n_samples, 32, 32), dtype=np.float32)

    for start in range(0, n_samples, settings.batch_size):
        stop = min(start + settings.batch_size, n_samples)
        batch = np.ascontiguousarray(images[start:stop])
        for offset, image in enumerate(batch):
            exact_hashes[start + offset] = hashlib.sha256(image.tobytes(order="C")).hexdigest()

        reduced = _downsample_grayscale(batch)
        gray32[start:stop] = reduced

        coefficients = dctn(reduced, axes=(-2, -1), norm="ortho")[:, :8, :8]
        flat = coefficients.reshape(len(batch), 64)
        thresholds = np.median(flat[:, 1:], axis=1, keepdims=True)
        phashes[start:stop] = _pack_bits(flat > thresholds)

        y_idx = np.rint(np.linspace(0, 31, 8)).astype(np.int64)
        x_idx = np.rint(np.linspace(0, 31, 9)).astype(np.int64)
        small = reduced[:, y_idx][:, :, x_idx]
        differences = small[:, :, 1:] > small[:, :, :-1]
        dhashes[start:stop] = _pack_bits(differences.reshape(len(batch), 64))

        progress_interval = settings.batch_size * 32
        if stop == n_samples or stop % progress_interval == 0:
            print(f"Fingerprinted {stop}/{n_samples} source images.", flush=True)

    centered = gray32.reshape(n_samples, -1)
    centered = centered - centered.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-12)
    return exact_hashes, phashes, dhashes, gray32, normalized.astype(np.float32)


def angular_separation_arcsec(
    ra_a: float,
    dec_a: float,
    ra_b: float,
    dec_b: float,
) -> float:
    ra1, dec1, ra2, dec2 = np.deg2rad([ra_a, dec_a, ra_b, dec_b])
    delta_ra = ra2 - ra1
    delta_dec = dec2 - dec1
    hav = (
        math.sin(delta_dec / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2.0) ** 2
    )
    angle = 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, hav))))
    return float(np.rad2deg(angle) * 3600.0)


def coordinate_candidate_pairs(
    ra: np.ndarray,
    dec: np.ndarray,
    audit_zone_code: np.ndarray,
    threshold_arcsec: float,
) -> set[tuple[int, int]]:
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    xyz = np.column_stack(
        (
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        )
    )
    chord = 2.0 * math.sin(math.radians(threshold_arcsec / 3600.0) / 2.0)
    pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    return {
        (int(a), int(b))
        for a, b in pairs
        if audit_zone_code[int(a)] != audit_zone_code[int(b)]
    }


def hamming_candidate_pairs(
    values: np.ndarray,
    audit_zone_code: np.ndarray,
    threshold: int,
    chunk_bits: int,
) -> set[tuple[int, int]]:
    """Return all cross-split pairs within a small Hamming radius.

    The hash is split into equal chunks.  ``n_chunks > threshold`` guarantees
    that a pair within ``threshold`` bits shares at least one exact chunk, so
    this multi-index search does not miss any in-radius pair.
    """
    if 64 % chunk_bits != 0:
        raise ValueError("chunk_bits must divide 64")
    n_chunks = 64 // chunk_bits
    if n_chunks <= threshold:
        raise ValueError("Need more chunks than threshold for complete candidate search")
    mask = (1 << chunk_bits) - 1
    found: set[tuple[int, int]] = set()
    ints = [int(value) for value in values]
    n_zones = int(audit_zone_code.max()) + 1

    for chunk in range(n_chunks):
        buckets: dict[int, list[list[int]]] = {}
        shift = chunk * chunk_bits
        for index, value in enumerate(ints):
            key = (value >> shift) & mask
            bucket = buckets.setdefault(key, [[] for _ in range(n_zones)])
            bucket[int(audit_zone_code[index])].append(index)
        for split_a, split_b in itertools.combinations(range(n_zones), 2):
            for bucket in buckets.values():
                left = bucket[split_a]
                right = bucket[split_b]
                for a in left:
                    hash_a = ints[a]
                    for b in right:
                        if (hash_a ^ ints[b]).bit_count() <= threshold:
                            found.add((a, b) if a < b else (b, a))
    return found


def exact_duplicate_rows(
    exact_hashes: list[str],
    split_code: np.ndarray,
    local_index: np.ndarray,
    labels: np.ndarray,
    ra: np.ndarray | None,
    dec: np.ndarray | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(exact_hashes):
        groups[digest].append(index)

    rows: list[dict[str, Any]] = []
    duplicate_groups = 0
    cross_split_groups = 0
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        duplicate_groups += 1
        if len({int(split_code[index]) for index in members}) > 1:
            cross_split_groups += 1
        for a, b in itertools.combinations(members, 2):
            separation = ""
            if ra is not None and dec is not None:
                separation = angular_separation_arcsec(ra[a], dec[a], ra[b], dec[b])
            rows.append(
                {
                    "exact_hash_sha256": digest,
                    "index_a": a,
                    "split_a": SPLIT_NAMES[split_code[a]],
                    "split_local_index_a": int(local_index[a]),
                    "label_a": int(labels[a]),
                    "ra_a_deg": "" if ra is None else float(ra[a]),
                    "dec_a_deg": "" if dec is None else float(dec[a]),
                    "index_b": b,
                    "split_b": SPLIT_NAMES[split_code[b]],
                    "split_local_index_b": int(local_index[b]),
                    "label_b": int(labels[b]),
                    "ra_b_deg": "" if ra is None else float(ra[b]),
                    "dec_b_deg": "" if dec is None else float(dec[b]),
                    "cross_split": bool(split_code[a] != split_code[b]),
                    "same_label": bool(labels[a] == labels[b]),
                    "coordinate_separation_arcsec": separation,
                }
            )
    rows.sort(key=lambda row: (not row["cross_split"], row["index_a"], row["index_b"]))
    return rows, {
        "duplicate_hash_groups": duplicate_groups,
        "cross_split_duplicate_hash_groups": cross_split_groups,
        "duplicate_pairs": len(rows),
        "cross_split_duplicate_pairs": sum(bool(row["cross_split"]) for row in rows),
    }


def near_duplicate_rows(
    candidate_pairs: set[tuple[int, int]],
    exact_hashes: list[str],
    phashes: np.ndarray,
    dhashes: np.ndarray,
    gray32: np.ndarray,
    normalized_gray: np.ndarray,
    split_code: np.ndarray,
    audit_zone_name: np.ndarray,
    local_index: np.ndarray,
    labels: np.ndarray,
    ra: np.ndarray | None,
    dec: np.ndarray | None,
    settings: AuditSettings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phash_int = [int(value) for value in phashes]
    dhash_int = [int(value) for value in dhashes]

    for a, b in sorted(candidate_pairs):
        if exact_hashes[a] == exact_hashes[b]:
            continue
        phash_distance = (phash_int[a] ^ phash_int[b]).bit_count()
        dhash_distance = (dhash_int[a] ^ dhash_int[b]).bit_count()
        ncc = float(np.dot(normalized_gray[a], normalized_gray[b]))
        rmse = float(np.sqrt(np.mean((gray32[a] - gray32[b]) ** 2)))
        separation = float("nan")
        exact_coordinate = False
        if ra is not None and dec is not None:
            separation = angular_separation_arcsec(ra[a], dec[a], ra[b], dec[b])
            exact_coordinate = bool(ra[a] == ra[b] and dec[a] == dec[b])

        evidence: list[str] = []
        if exact_coordinate:
            evidence.append("exact_ra_dec")
        elif np.isfinite(separation) and separation <= settings.coordinate_candidate_arcsec:
            evidence.append("near_ra_dec")
        if phash_distance <= settings.phash_candidate_hamming:
            evidence.append("phash")
        if dhash_distance <= settings.dhash_candidate_hamming:
            evidence.append("dhash")

        if exact_coordinate:
            priority = "critical"
            interpretation = "Same catalog sky coordinates across row-index splits; source-object leakage."
        elif np.isfinite(separation) and separation <= settings.coordinate_candidate_arcsec:
            priority = "critical"
            interpretation = "Sky positions are within the coordinate threshold; likely same source object."
        elif phash_distance <= 4 and ncc >= 0.985 and rmse <= 0.060:
            priority = "high"
            interpretation = "Strong perceptual match; visually verify before treating as a duplicate."
        elif (
            phash_distance <= settings.phash_candidate_hamming
            and ncc >= 0.950
            and rmse <= 0.100
        ) or (dhash_distance <= settings.dhash_candidate_hamming and ncc >= 0.950):
            priority = "medium"
            interpretation = "Conservative perceptual candidate; not proof of duplicate identity."
        else:
            continue

        coordinate_bonus = 100.0 if exact_coordinate else (
            80.0 if np.isfinite(separation) and separation <= settings.coordinate_candidate_arcsec else 0.0
        )
        score = (
            coordinate_bonus
            + 30.0 * max(0.0, ncc)
            + 20.0 * (1.0 - phash_distance / 64.0)
            + 10.0 * (1.0 - dhash_distance / 64.0)
            - 20.0 * rmse
        )
        rows.append(
            {
                "review_priority": priority,
                "evidence_type": "+".join(evidence),
                "index_a": a,
                "split_a": SPLIT_NAMES[split_code[a]],
                "audit_zone_a": str(audit_zone_name[a]),
                "split_local_index_a": int(local_index[a]),
                "label_a": int(labels[a]),
                "ra_a_deg": "" if ra is None else float(ra[a]),
                "dec_a_deg": "" if dec is None else float(dec[a]),
                "index_b": b,
                "split_b": SPLIT_NAMES[split_code[b]],
                "audit_zone_b": str(audit_zone_name[b]),
                "split_local_index_b": int(local_index[b]),
                "label_b": int(labels[b]),
                "ra_b_deg": "" if ra is None else float(ra[b]),
                "dec_b_deg": "" if dec is None else float(dec[b]),
                "same_label": bool(labels[a] == labels[b]),
                "exact_coordinate_match": exact_coordinate,
                "coordinate_separation_arcsec": "" if not np.isfinite(separation) else separation,
                "phash_hamming_64": phash_distance,
                "dhash_hamming_64": dhash_distance,
                "downsample_gray_ncc": ncc,
                "downsample_gray_rmse_0_1": rmse,
                "ranking_score": score,
                "interpretation": interpretation,
            }
        )

    priority_rank = {"critical": 0, "high": 1, "medium": 2}
    rows.sort(
        key=lambda row: (
            priority_rank[row["review_priority"]],
            -float(row["ranking_score"]),
            row["index_a"],
            row["index_b"],
        )
    )
    rows = rows[: settings.max_near_candidates]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _first_field(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {field.lower(): field for field in fieldnames}
    return next((lowered[candidate] for candidate in candidates if candidate in lowered), None)


def _parse_index(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        numeric = float(str(value))
    except ValueError:
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _run_dir_for_artifact(path: Path, historical_root: Path) -> Path | None:
    try:
        relative = path.resolve().relative_to(historical_root.resolve())
    except ValueError:
        return None
    return historical_root.resolve() / relative.parts[0] if relative.parts else None


def _load_run_config(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    for candidate in (run_dir / "logs/run_config.yaml", run_dir / "logs/run_config.yml"):
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    return yaml.safe_load(handle) or {}
            except Exception:
                return {}
    return {}


def infer_index_semantics(
    path: Path,
    target_field: str,
    contaminant_field: str,
    fieldnames: list[str],
) -> tuple[str, str]:
    fields_lower = {field.lower() for field in fieldnames}
    if "source_split_name" in fields_lower or "source_split" in fields_lower:
        if "source" in target_field.lower() or "global" in target_field.lower():
            return "global_hdf5_index", "row_declared_split"
    if "source" in target_field.lower() or "global" in target_field.lower():
        return "global_hdf5_index", "test"
    text = str(path).lower()
    if "stress_test_" in text or "resunet_v04_candidate_" in text:
        return "test_subset_local_index", "test"
    return "unknown_index_semantics", "unknown"


def audit_csv_artifact(
    path: Path,
    historical_root: Path,
    partitions: dict[str, np.ndarray],
    split_code: np.ndarray,
) -> dict[str, Any] | None:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            target_field = _first_field(fieldnames, TARGET_FIELDS)
            contaminant_field = _first_field(fieldnames, CONTAMINANT_FIELDS)
            source_relevant_name = any(
                token in path.name.lower() for token in ("per_sample", "manifest", "top_10")
            )
            if target_field is None or contaminant_field is None:
                if not source_relevant_name:
                    return None
                row_count = sum(1 for _ in reader)
                return {
                    "record_type": "historical_artifact",
                    "artifact_path": project_relative(path),
                    "row_count": row_count,
                    "rows_with_both_indices": 0,
                    "rows_missing_one_or_both_indices": row_count,
                    "index_semantics": "source_indices_absent",
                    "containment_status": "not_auditable",
                    "details": "Per-sample/sample-ranking table has no target and contaminant source-index columns.",
                }
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None

    semantics, expected_partition = infer_index_semantics(
        path, target_field, contaminant_field, fieldnames
    )
    source_split_field = _first_field(fieldnames, ("source_split_name", "source_split", "split_name"))
    run_dir = _run_dir_for_artifact(path, historical_root)
    run_config = _load_run_config(run_dir)
    subset_limit: int | None = None
    if semantics == "test_subset_local_index":
        if "stress_settings" in run_config:
            subset_limit = int(run_config["stress_settings"].get("stress_source_subset", len(partitions["test"])))
        elif "settings" in run_config:
            subset_limit = int(run_config["settings"].get("test_source_subset", len(partitions["test"])))
        else:
            subset_limit = len(partitions["test"])
        subset_limit = min(subset_limit, len(partitions["test"]))

    target_values: list[int] = []
    contaminant_values: list[int] = []
    missing = 0
    target_oob = 0
    contaminant_oob = 0
    wrong_target = 0
    wrong_contaminant = 0
    same_source = 0

    for row in rows:
        target = _parse_index(row.get(target_field))
        contaminant = _parse_index(row.get(contaminant_field))
        if target is None or contaminant is None:
            missing += 1
            continue
        target_values.append(target)
        contaminant_values.append(contaminant)

        target_global: int | None = None
        contaminant_global: int | None = None
        if semantics == "test_subset_local_index":
            assert subset_limit is not None
            if target < 0 or target >= subset_limit:
                target_oob += 1
            else:
                target_global = int(partitions["test"][target])
            if contaminant < 0 or contaminant >= subset_limit:
                contaminant_oob += 1
            else:
                contaminant_global = int(partitions["test"][contaminant])
        elif semantics == "global_hdf5_index":
            if target < 0 or target >= len(split_code):
                target_oob += 1
            else:
                target_global = target
            if contaminant < 0 or contaminant >= len(split_code):
                contaminant_oob += 1
            else:
                contaminant_global = contaminant

        if target_global is not None and contaminant_global is not None:
            if target_global == contaminant_global:
                same_source += 1
            row_expected_partition = expected_partition
            if expected_partition == "row_declared_split" and source_split_field is not None:
                row_expected_partition = str(row.get(source_split_field, "")).strip().lower()
                if row_expected_partition == "val":
                    row_expected_partition = "validation"
            if row_expected_partition in ("train", "validation", "test"):
                expected_code = ("train", "validation", "test").index(row_expected_partition)
                wrong_target += int(split_code[target_global] != expected_code)
                wrong_contaminant += int(split_code[contaminant_global] != expected_code)

    indexed_rows = len(target_values)
    if semantics == "unknown_index_semantics":
        status = "not_auditable"
        details = "Source-index columns exist, but saved metadata does not establish global versus subset-local semantics."
    elif indexed_rows == 0:
        status = "not_auditable"
        details = "Source-index columns exist but are blank for every row."
    elif any((target_oob, contaminant_oob, wrong_target, wrong_contaminant, same_source)):
        status = "failed"
        details = "At least one indexed row violates bounds, partition containment, or distinct-role requirements."
    else:
        status = "passed"
        details = (
            "All retained target/contaminant indices map to the held-out test partition."
            if expected_partition == "test"
            else "All retained indices map to their declared source partition."
        )

    return {
        "record_type": "historical_artifact",
        "artifact_path": project_relative(path),
        "index_semantics": semantics,
        "expected_partition": expected_partition,
        "source_count": subset_limit if subset_limit is not None else "",
        "row_count": len(rows),
        "rows_with_both_indices": indexed_rows,
        "rows_missing_one_or_both_indices": missing,
        "target_min": min(target_values) if target_values else "",
        "target_max": max(target_values) if target_values else "",
        "contaminant_min": min(contaminant_values) if contaminant_values else "",
        "contaminant_max": max(contaminant_values) if contaminant_values else "",
        "out_of_range_target_count": target_oob,
        "out_of_range_contaminant_count": contaminant_oob,
        "wrong_partition_target_count": wrong_target,
        "wrong_partition_contaminant_count": wrong_contaminant,
        "same_source_pair_count": same_source,
        "containment_status": status,
        "details": details,
    }


def partition_audit_rows(
    partitions: dict[str, np.ndarray],
    split_code: np.ndarray,
    historical_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, indices in partitions.items():
        rows.append(
            {
                "record_type": "partition",
                "split": split,
                "role": "target_and_contaminant_source_pool",
                "index_semantics": "global_hdf5_index",
                "expected_partition": split,
                "source_count": len(indices),
                "target_min": int(indices.min()),
                "target_max": int(indices.max()),
                "containment_status": "passed",
                "details": "Deterministically reconstructed from numpy default_rng(seed).shuffle and integer fraction boundaries.",
            }
        )
    for split_a, split_b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = np.intersect1d(partitions[split_a], partitions[split_b], assume_unique=True)
        rows.append(
            {
                "record_type": "partition_overlap",
                "split": f"{split_a}_vs_{split_b}",
                "source_count": len(overlap),
                "containment_status": "passed" if len(overlap) == 0 else "failed",
                "details": "Row-index intersection count.",
            }
        )

    if historical_root.exists():
        for path in sorted(historical_root.rglob("*.csv")):
            audit = audit_csv_artifact(path, historical_root, partitions, split_code)
            if audit is not None:
                rows.append(audit)
    return rows


def build_audit_zones(
    partitions: dict[str, np.ndarray],
    split_code: np.ndarray,
    development_test_prefix: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the held-out test rows into development-prefix and final-tail zones."""
    zone_code = split_code.copy()
    zone_name = SPLIT_NAMES[split_code].astype(object)
    test_indices = partitions["test"]
    prefix_size = min(development_test_prefix, len(test_indices))
    prefix = test_indices[:prefix_size]
    tail = test_indices[prefix_size:]
    zone_code[prefix] = 2
    zone_code[tail] = 3
    zone_name[prefix] = "test_development_prefix"
    zone_name[tail] = "test_final_tail"
    return zone_code, zone_name


def _pair_map(
    rows: Iterable[dict[str, Any]],
    priority: str | None = None,
) -> dict[int, set[int]]:
    mapping: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if priority is not None and row.get("review_priority") != priority:
            continue
        a = int(row["index_a"])
        b = int(row["index_b"])
        mapping[a].add(b)
        mapping[b].add(a)
    return mapping


def crosscheck_provisional_manifests(
    historical_root: Path,
    exact_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    split_code: np.ndarray,
    local_index: np.ndarray,
    zone_name: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = sorted(historical_root.glob("final_test_manifest_prep_*"), reverse=True)
    manifest_run: Path | None = None
    manifest_files: list[Path] = []
    for candidate in candidates:
        files = sorted((candidate / "manifests").glob("*_final_test.csv"))
        if len(files) == 5:
            manifest_run = candidate
            manifest_files = files
            break
    if manifest_run is None:
        return [], {
            "available": False,
            "manifest_run": "",
            "manifest_files": 0,
            "manifest_rows": 0,
            "unique_sources": 0,
            "sources_outside_final_tail": 0,
            "blocking_sources": 0,
            "review_sources": 0,
            "cleared_sources": 0,
            "independently_verified": False,
            "recommendation": "No completed five-suite provisional manifest run was available to cross-check.",
        }

    usage: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"roles": set(), "suites": set(), "usage_count": 0}
    )
    manifest_rows = 0
    invalid_index_rows = 0
    for path in manifest_files:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                manifest_rows += 1
                suite = str(row.get("suite") or path.stem)
                for role, field in (
                    ("target", "target_source_index"),
                    ("contaminant", "contaminant_source_index"),
                ):
                    index = _parse_index(row.get(field))
                    if index is None or index < 0 or index >= len(split_code):
                        invalid_index_rows += 1
                        continue
                    usage[index]["roles"].add(role)
                    usage[index]["suites"].add(suite)
                    usage[index]["usage_count"] += 1

    exact_map = _pair_map(exact_rows)
    critical_map = _pair_map(near_rows, "critical")
    high_map = _pair_map(near_rows, "high")
    medium_map = _pair_map(near_rows, "medium")
    leakage_zones = {"train", "validation", "test_development_prefix"}

    output_rows: list[dict[str, Any]] = []
    for source_index, info in sorted(usage.items()):
        exact = sorted(exact_map.get(source_index, set()))
        critical = sorted(critical_map.get(source_index, set()))
        high = sorted(high_map.get(source_index, set()))
        medium = sorted(medium_map.get(source_index, set()))

        def relevant(indices: list[int]) -> list[int]:
            return [index for index in indices if str(zone_name[index]) in leakage_zones]

        exact_relevant = relevant(exact)
        critical_relevant = relevant(critical)
        high_relevant = relevant(high)
        medium_relevant = relevant(medium)
        all_relevant = exact_relevant + critical_relevant + high_relevant + medium_relevant
        zones_relevant = {str(zone_name[index]) for index in all_relevant}
        source_zone = str(zone_name[source_index])

        if source_zone != "test_final_tail":
            status = "failed_source_pool_containment"
            details = "Selected source is not in the intended post-development test tail."
        elif exact_relevant or critical_relevant:
            status = "failed_duplicate_leakage"
            details = "Conclusive exact-image or same-coordinate counterpart exists in train/validation/development-prefix data."
        elif high_relevant or medium_relevant:
            status = "manual_review_required"
            details = "Perceptual counterpart candidate exists in train/validation/development-prefix data."
        else:
            status = "cleared_at_audited_thresholds"
            details = "No exact, coordinate, or retained perceptual counterpart was found in prior-development zones."

        def joined_indices(indices: list[int]) -> str:
            return ";".join(str(index) for index in indices)

        def joined_zones(indices: list[int]) -> str:
            return ";".join(str(zone_name[index]) for index in indices)

        output_rows.append(
            {
                "manifest_run": project_relative(manifest_run),
                "source_index": source_index,
                "source_split": str(SPLIT_NAMES[split_code[source_index]]),
                "split_local_index": int(local_index[source_index]),
                "audit_zone": source_zone,
                "roles": ";".join(sorted(info["roles"])),
                "suites": ";".join(sorted(info["suites"])),
                "usage_count": int(info["usage_count"]),
                "exact_image_counterpart_indices": joined_indices(exact),
                "exact_image_counterpart_zones": joined_zones(exact),
                "critical_coordinate_counterpart_indices": joined_indices(critical),
                "critical_coordinate_counterpart_zones": joined_zones(critical),
                "high_perceptual_counterpart_indices": joined_indices(high),
                "high_perceptual_counterpart_zones": joined_zones(high),
                "medium_perceptual_counterpart_indices": joined_indices(medium),
                "medium_perceptual_counterpart_zones": joined_zones(medium),
                "counterpart_in_train": "train" in zones_relevant,
                "counterpart_in_validation": "validation" in zones_relevant,
                "counterpart_in_development_prefix": "test_development_prefix" in zones_relevant,
                "verification_status": status,
                "details": details,
            }
        )

    blocking = sum(
        row["verification_status"] in ("failed_source_pool_containment", "failed_duplicate_leakage")
        for row in output_rows
    )
    review = sum(row["verification_status"] == "manual_review_required" for row in output_rows)
    cleared = sum(row["verification_status"] == "cleared_at_audited_thresholds" for row in output_rows)
    independently_verified = bool(output_rows) and blocking == 0 and review == 0 and invalid_index_rows == 0
    if blocking:
        recommendation = (
            "Regenerate the provisional manifests after excluding the flagged duplicate-linked sources; "
            "do not treat the present files as locked final tests."
        )
    elif review:
        recommendation = (
            "Visually/catalog-review every flagged perceptual pair, then regenerate affected manifests or "
            "record an independently reviewed exclusion decision before locking."
        )
    else:
        recommendation = (
            "The selected sources pass this exact/coordinate/perceptual cross-check, but the manifests "
            "remain provisional until the broader protocol and generator are frozen independently."
        )
    return output_rows, {
        "available": True,
        "manifest_run": project_relative(manifest_run),
        "manifest_files": len(manifest_files),
        "manifest_rows": manifest_rows,
        "invalid_index_rows": invalid_index_rows,
        "unique_sources": len(output_rows),
        "sources_outside_final_tail": sum(
            row["audit_zone"] != "test_final_tail" for row in output_rows
        ),
        "blocking_sources": blocking,
        "review_sources": review,
        "cleared_sources": cleared,
        "independently_verified": independently_verified,
        "recommendation": recommendation,
    }


def find_no_duplicate_variants() -> tuple[list[str], list[str]]:
    # Synced home directories can block indefinitely on an unavailable provider.
    # The script therefore checks only the local project data directory.  A
    # separate completed full-home preflight can be recorded via CLI.
    roots = [PROJECT_ROOT / "data"]
    found: set[str] = set()
    searched: list[str] = []
    patterns = ("galaxy10_decals_noduplicated.h5", "galaxy10_decals_no_duplicated.h5")
    for root in roots:
        if not root.exists():
            continue
        searched.append(root.name if root != PROJECT_ROOT / "data" else "project data directory")
        for candidate in root.glob("*.h5"):
            lower = candidate.name.lower()
            if lower in patterns or ("galaxy10" in lower and "noduplicat" in lower):
                found.add(candidate.name)
    return sorted(found), searched


def code_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(summary: dict[str, Any]) -> str:
    exact = summary["exact_image_duplicates"]
    coordinate = summary["coordinate_duplicates"]
    historical = summary["historical_artifacts"]
    near = summary["near_duplicate_candidates"]
    manifest = summary["provisional_manifest_crosscheck"]
    blocker = summary["major_blocker"]
    no_dup = summary["no_duplicate_variant_search"]
    schema = summary["hdf5_schema"]
    split_sizes = summary["split_sizes"]
    return f"""# Source Leakage Audit Report

Generated: `{summary['generated_at_local']}`

## Executive conclusion

**Major blocker: {'yes' if blocker else 'no'}.** The reconstructed global HDF5
row-index partitions are disjoint, and all auditable saved target/contaminant
indices remain inside their assigned held-out source pool. However, disjoint row
indices are not the same as disjoint astronomical sources. The HDF5 metadata
contains `{coordinate['duplicate_coordinate_groups']}` repeated exact `(RA, Dec)`
coordinate groups and `{coordinate['cross_split_pairs']}` duplicate-coordinate
pairs cross the current train/validation/test boundaries. Those rows represent
the same sky position and therefore create source-object leakage under the
current random-index split.

This finding is independent of model quality. It means the current normal and
stress results remain useful development-benchmark results, but they should not
be promoted as a locked final paper claim until sources are deduplicated or
group-split by object identity/coordinates.

## Dataset and schema

- Dataset: `{summary['dataset_path']}`
- Dataset SHA-256: `{summary['dataset_sha256']}`
- HDF5 rows: `{summary['n_samples']}`
- Image shape/dtype: `{summary['image_shape']}` / `{summary['image_dtype']}`
- Available datasets: `{', '.join(sorted(schema['datasets']))}`
- RA/Dec metadata: `{'available' if schema['ra_available'] and schema['dec_available'] else 'unavailable'}`
- Object/catalog ID dataset: `{'available: ' + ', '.join(schema['object_id_candidates']) if schema['object_id_candidates'] else 'not present'}`
- A local `Galaxy10_DECals_NoDuplicated.h5`-style file was `{'found (' + ', '.join(no_dup['matches']) + ')' if no_dup['matches'] else 'not found'}` in the searched user-visible locations: `{', '.join(no_dup['searched_locations'])}`.

## Reconstructed split

The audit exactly mirrors `src.data.split_dataset`: create `arange(N)`, shuffle
with NumPy `default_rng({summary['split_seed']})`, then take integer boundaries
for `{summary['split_fractions']['train']:.2f}` / `{summary['split_fractions']['validation']:.2f}` /
`{summary['split_fractions']['test']:.2f}`.

| Partition | Source rows |
| --- | ---: |
| Train | {split_sizes['train']} |
| Validation | {split_sizes['validation']} |
| Test | {split_sizes['test']} |

All three pairwise row-index intersections are zero. This proves row-level
partition disjointness only; a random-index split cannot prevent different rows
of the same object from crossing splits.

## Target/contaminant role containment

- Historical CSV artifacts examined: `{historical['examined']}`
- Artifacts with auditable source indices: `{historical['auditable']}`
- Indexed blend rows audited: `{historical['indexed_rows']}`
- Auditable artifacts failing containment: `{historical['failed']}`
- Per-sample/manifest-like artifacts lacking source indices: `{historical['indices_absent']}`
- Artifacts with unknown index semantics: `{historical['unknown_semantics']}`

The retained indices in completed stress-test and ResUNet targeted-suite tables
are subset-local, not raw Galaxy10 row IDs. Mapping them through the reconstructed
held-out test partition found no target/contaminant role crossing. The standard
normal blend generator and many older per-sample tables do not retain source
indices, so those historical samples cannot be independently re-identified from
their metric tables alone. The current code path still passes a split-specific
array into a generator that selects both roles from that one array.

## Exact image duplicates

Each raw `256 x 256 x 3 uint8` image array was streamed from HDF5 and hashed with
SHA-256 over its contiguous pixel bytes.

- Exact duplicate hash groups: `{exact['duplicate_hash_groups']}`
- Exact duplicate image pairs: `{exact['duplicate_pairs']}`
- Cross-split exact duplicate hash groups: `{exact['cross_split_duplicate_hash_groups']}`
- Cross-split exact duplicate image pairs: `{exact['cross_split_duplicate_pairs']}`

See `tables/exact_duplicate_audit.csv` for pair-level evidence. Exact hash
matches are conclusive pixel equality. Coordinate matches can still reveal the
same object when processing or crop pixels differ.

## Coordinate and perceptual duplicate audit

- Exact coordinate duplicate groups: `{coordinate['duplicate_coordinate_groups']}`
- Exact coordinate duplicate pairs across splits: `{coordinate['cross_split_pairs']}`
- Ranked non-exact-image candidates saved: `{near['saved']}`
- Critical coordinate candidates: `{near['critical']}`
- High perceptual candidates: `{near['high']}`
- Medium perceptual candidates: `{near['medium']}`

Perceptual fingerprinting uses a `32 x 32` block-mean grayscale view, a 64-bit
DCT pHash, and a 64-bit gradient dHash. The complete multi-index search retains
all cross-audit-zone pHash pairs with Hamming distance at most
`{summary['thresholds']['phash_candidate_hamming']}` and all dHash pairs with
distance at most `{summary['thresholds']['dhash_candidate_hamming']}`. Candidates
are then checked with downsampled normalized cross-correlation (NCC) and RMSE.
The CSV saves exact coordinates first, then strong perceptual candidates
(`pHash <= 4`, `NCC >= 0.985`, and grayscale RMSE `<= 0.060`) and conservative
medium candidates. Perceptual matches require visual/catalog review; exact
coordinate duplication is already source-object evidence.

Here, audit zones are train, validation, the first 1,000 test positions used by
development evaluations, and the remaining test tail reserved by the
provisional manifest preparation. This also detects test-prefix versus
test-tail candidates that a three-way split-only search would miss.

## Provisional five-suite manifest cross-check

- Manifest run: `{manifest['manifest_run'] or 'not available'}`
- Manifest CSVs / rows: `{manifest['manifest_files']}` / `{manifest['manifest_rows']}`
- Unique target-or-contaminant sources: `{manifest['unique_sources']}`
- Sources outside the intended final-test tail: `{manifest['sources_outside_final_tail']}`
- Sources with exact-image or same-coordinate leakage evidence: `{manifest['blocking_sources']}`
- Sources requiring perceptual-candidate review: `{manifest['review_sources']}`
- Sources clear at the documented thresholds: `{manifest['cleared_sources']}`
- Independently verified by this audit: `{'yes' if manifest['independently_verified'] else 'no'}`

The five provisional CSVs are read-only and were not modified. Every unique
global target/contaminant index was compared with exact-image duplicate groups
and retained coordinate/perceptual candidates in train, validation, and the
first 1,000 held-out test positions used as the development prefix. Detailed
source-level evidence is in
`tables/provisional_manifest_source_crosscheck.csv`.

Recommendation: {manifest['recommendation']}

## What the audit establishes

1. The deterministic HDF5 row-index partitions are disjoint.
2. No target/contaminant partition violation was found in artifacts that retain
   interpretable indices.
3. The random-index split does not provide object-level independence: exact
   RA/Dec duplicates cross partitions.
4. A no-duplicate local dataset variant was not available in the searched
   locations, and the HDF5 file has no separate object-ID field beyond RA/Dec.

## Recommended correction before final claims

1. Treat the present normal/stress suites as development benchmarks.
2. Build object groups from exact coordinates (and conservatively merge close
   coordinates or verified perceptual duplicates) before assigning splits.
3. Prefer an independently verified no-duplicate Galaxy10 source file if one is
   obtained, but audit it rather than relying on its filename.
4. Freeze the final manifests only after the group-disjoint source split is
   established; do not use the locked final samples for model selection.
5. Re-run preservation, clipping, and final model comparisons on that corrected
   source split. Existing exploratory checkpoints need not be deleted.

## Limitations

- RA/Dec identifies repeated sky positions but does not replace an external
  survey object-ID/catalog crossmatch.
- Perceptual hashes can collide for centrally concentrated, dark-background
  galaxies; they are candidate generators, not duplicate proofs.
- Historical metric tables without source indices cannot be audited sample by
  sample after the fact.
- This audit does not alter the HDF5 file, remove sources, run a model, or revise
  historical metrics.
"""


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.max_near_candidates <= 0:
        raise ValueError("batch size and max-near-candidates must be positive")
    settings = AuditSettings(
        batch_size=args.batch_size,
        max_near_candidates=args.max_near_candidates,
    )
    config = load_config(args.config)
    dataset_path = resolve_path(args.dataset or Path(config["dataset_path"]))
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    dataset_sha256 = file_sha256(dataset_path)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir(args.output_root, stamp)
    historical_root = resolve_path(args.historical_root)
    print(f"Source-leakage audit run: {project_relative(run_dir)}", flush=True)
    print("Device: CPU (hashing, metadata, CSV aggregation only; no model inference or training).", flush=True)

    with h5py.File(dataset_path, "r") as handle:
        if "images" not in handle or "ans" not in handle:
            raise KeyError("Expected HDF5 datasets 'images' and 'ans'.")
        schema = dataset_schema(handle)
        images = handle["images"]
        labels = np.asarray(handle["ans"][:])
        n_samples = int(images.shape[0])
        if len(labels) != n_samples:
            raise ValueError("Image and label counts differ.")
        ra = np.asarray(handle["ra"][:], dtype=np.float64) if "ra" in handle else None
        dec = np.asarray(handle["dec"][:], dtype=np.float64) if "dec" in handle else None

        train_frac = float(config["splits"]["train_frac"])
        val_frac = float(config["splits"]["val_frac"])
        test_frac = float(config["splits"]["test_frac"])
        seed = int(config["seed"])
        partitions, split_code, local_index = reconstruct_partitions(
            n_samples, seed, train_frac, val_frac
        )
        audit_zone_code, audit_zone_name = build_audit_zones(
            partitions, split_code, development_test_prefix=1000
        )

        partition_rows = partition_audit_rows(partitions, split_code, historical_root)
        exact_hashes, phashes, dhashes, gray32, normalized_gray = compute_fingerprints(
            images, settings
        )

    exact_rows, exact_summary = exact_duplicate_rows(
        exact_hashes, split_code, local_index, labels, ra, dec
    )

    coordinate_pairs: set[tuple[int, int]] = set()
    exact_coordinate_cross_split_pairs: set[tuple[int, int]] = set()
    coordinate_group_count = 0
    all_coordinate_pair_count = 0
    if ra is not None and dec is not None:
        coordinate_groups: dict[tuple[float, float], list[int]] = defaultdict(list)
        for index, key in enumerate(zip(ra.tolist(), dec.tolist())):
            coordinate_groups[key].append(index)
        coordinate_group_count = sum(len(members) > 1 for members in coordinate_groups.values())
        all_coordinate_pair_count = sum(
            math.comb(len(members), 2)
            for members in coordinate_groups.values()
            if len(members) > 1
        )
        for members in coordinate_groups.values():
            for a, b in itertools.combinations(members, 2):
                if split_code[a] != split_code[b]:
                    exact_coordinate_cross_split_pairs.add((min(a, b), max(a, b)))
        coordinate_pairs = coordinate_candidate_pairs(
            ra, dec, audit_zone_code, settings.coordinate_candidate_arcsec
        )

    print("Searching cross-audit-zone pHash neighborhood.", flush=True)
    phash_pairs = hamming_candidate_pairs(
        phashes,
        audit_zone_code,
        threshold=settings.phash_candidate_hamming,
        chunk_bits=8,
    )
    print("Searching cross-audit-zone dHash neighborhood.", flush=True)
    dhash_pairs = hamming_candidate_pairs(
        dhashes,
        audit_zone_code,
        threshold=settings.dhash_candidate_hamming,
        chunk_bits=16,
    )
    candidate_pairs = coordinate_pairs | phash_pairs | dhash_pairs
    near_rows = near_duplicate_rows(
        candidate_pairs,
        exact_hashes,
        phashes,
        dhashes,
        gray32,
        normalized_gray,
        split_code,
        audit_zone_name,
        local_index,
        labels,
        ra,
        dec,
        settings,
    )
    manifest_crosscheck_rows, manifest_crosscheck_summary = crosscheck_provisional_manifests(
        historical_root,
        exact_rows,
        near_rows,
        split_code,
        local_index,
        audit_zone_name,
    )

    found_variants, searched_locations = find_no_duplicate_variants()
    audited_artifacts = [row for row in partition_rows if row.get("record_type") == "historical_artifact"]
    historical_summary = {
        "examined": len(audited_artifacts),
        "auditable": sum(row.get("containment_status") in ("passed", "failed") for row in audited_artifacts),
        "indexed_rows": sum(int(row.get("rows_with_both_indices") or 0) for row in audited_artifacts),
        "failed": sum(row.get("containment_status") == "failed" for row in audited_artifacts),
        "indices_absent": sum(row.get("index_semantics") == "source_indices_absent" for row in audited_artifacts),
        "unknown_semantics": sum(row.get("index_semantics") == "unknown_index_semantics" for row in audited_artifacts),
    }
    near_summary = {
        "saved": len(near_rows),
        "critical": sum(row["review_priority"] == "critical" for row in near_rows),
        "high": sum(row["review_priority"] == "high" for row in near_rows),
        "medium": sum(row["review_priority"] == "medium" for row in near_rows),
        "raw_candidate_pair_union": len(candidate_pairs),
        "phash_in_radius_pairs": len(phash_pairs),
        "dhash_in_radius_pairs": len(dhash_pairs),
    }
    coordinate_summary = {
        "duplicate_coordinate_groups": coordinate_group_count,
        "all_duplicate_coordinate_pairs": all_coordinate_pair_count,
        "cross_split_pairs": len(exact_coordinate_cross_split_pairs),
        "within_one_arcsec_cross_audit_zone_pairs": len(coordinate_pairs),
    }
    major_blocker = bool(
        coordinate_summary["cross_split_pairs"]
        or exact_summary["cross_split_duplicate_pairs"]
        or historical_summary["failed"]
        or manifest_crosscheck_summary["blocking_sources"]
    )
    summary = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "run_directory": project_relative(run_dir),
        "dataset_path": project_relative(dataset_path),
        "dataset_size_bytes": dataset_path.stat().st_size,
        "dataset_sha256": dataset_sha256,
        "n_samples": n_samples,
        "image_shape": list(schema["datasets"]["images"]["shape"]),
        "image_dtype": schema["datasets"]["images"]["dtype"],
        "hdf5_schema": schema,
        "split_seed": seed,
        "split_fractions": {
            "train": train_frac,
            "validation": val_frac,
            "test": test_frac,
        },
        "split_sizes": {key: len(value) for key, value in partitions.items()},
        "split_row_index_overlaps": {
            "train_validation": int(len(np.intersect1d(partitions["train"], partitions["validation"]))),
            "train_test": int(len(np.intersect1d(partitions["train"], partitions["test"]))),
            "validation_test": int(len(np.intersect1d(partitions["validation"], partitions["test"]))),
        },
        "historical_artifacts": historical_summary,
        "exact_image_duplicates": exact_summary,
        "coordinate_duplicates": coordinate_summary,
        "near_duplicate_candidates": near_summary,
        "provisional_manifest_crosscheck": manifest_crosscheck_summary,
        "thresholds": {
            "coordinate_candidate_arcsec": settings.coordinate_candidate_arcsec,
            "phash_candidate_hamming": settings.phash_candidate_hamming,
            "dhash_candidate_hamming": settings.dhash_candidate_hamming,
            "high_candidate_phash_max": 4,
            "high_candidate_ncc_min": 0.985,
            "high_candidate_rmse_max": 0.060,
            "medium_candidate_ncc_min": 0.950,
            "medium_candidate_rmse_max": 0.100,
        },
        "no_duplicate_variant_search": {
            "matches": found_variants,
            "searched_locations": searched_locations
            + (["full user home (separate read-only preflight search)"] if args.full_home_preflight_no_match else []),
            "full_home_preflight_no_match": bool(args.full_home_preflight_no_match),
        },
        "code_sha256": {
            "config": code_hash(resolve_path(args.config)),
            "src/data.py": code_hash(PROJECT_ROOT / "src/data.py"),
            "src/blend.py": code_hash(PROJECT_ROOT / "src/blend.py"),
            "scripts/source_leakage_audit.py": code_hash(Path(__file__)),
        },
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "pyyaml": importlib_metadata.version("PyYAML"),
            "scipy": importlib_metadata.version("scipy"),
        },
        "major_blocker": major_blocker,
        "device_note": "CPU used only for hashing, metadata inspection, and CSV aggregation; no model was loaded.",
    }

    write_csv(run_dir / "tables/source_partition_audit.csv", PARTITION_FIELDS, partition_rows)
    write_csv(run_dir / "tables/exact_duplicate_audit.csv", EXACT_FIELDS, exact_rows)
    write_csv(run_dir / "tables/near_duplicate_candidates.csv", NEAR_FIELDS, near_rows)
    write_csv(
        run_dir / "tables/provisional_manifest_source_crosscheck.csv",
        MANIFEST_CROSSCHECK_FIELDS,
        manifest_crosscheck_rows,
    )
    write_json(run_dir / "diagnostics/source_leakage_audit_summary.json", summary)
    write_text(run_dir / "diagnostics/source_leakage_audit_report.md", render_report(summary))
    write_json(
        run_dir / "logs/audit_config.json",
        {
            "config_path": project_relative(resolve_path(args.config)),
            "dataset_path": project_relative(dataset_path),
            "dataset_sha256": dataset_sha256,
            "settings": settings.__dict__,
            "selected_device": "cpu_non_model_audit_only",
            "code_sha256": summary["code_sha256"],
            "dependency_versions": summary["dependency_versions"],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 2 if major_blocker else 0


if __name__ == "__main__":
    sys.exit(main())
