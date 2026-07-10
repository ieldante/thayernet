"""Build a deterministic duplicate-safe Galaxy10 source split manifest.

The script groups pixel-identical images and rows with identical finite
``(ra, dec)`` coordinates before assigning any source to train, validation, or
test.  It never edits the HDF5 dataset and refuses to overwrite an existing
manifest directory or file.

Near-duplicate candidates are deliberately *not* grouped automatically.  A
reviewed CSV can be supplied explicitly with ``--verified-near-duplicate-csv``
and ``--group-verified-near-duplicates``.  This keeps uncertain perceptual
matches out of the correctness-critical split unless a human has verified
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shlex
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/Galaxy10_DECals.h5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/manifests"
SPLIT_NAMES = ("train", "validation", "test")
GROUPING_VERSION = "grouped_source_split_v1_exact_pixels_exact_coordinates"
STAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


class UnionFind:
    """Small deterministic disjoint-set implementation for source groups."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        while parent != int(self.parent[parent]):
            parent = int(self.parent[parent])
        while value != parent:
            next_value = int(self.parent[value])
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = int(self.rank[root_left])
        rank_right = int(self.rank[root_right])
        if rank_left < rank_right:
            root_left, root_right = root_right, root_left
            rank_left, rank_right = rank_right, rank_left
        # A stable root makes group construction reproducible even on rank ties.
        if rank_left == rank_right and root_right < root_left:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if rank_left == rank_right:
            self.rank[root_left] += 1


@dataclass(frozen=True)
class SourceGroup:
    group_id: str
    members: tuple[int, ...]
    label_counts: np.ndarray
    contains_exact_duplicate: bool
    contains_coordinate_duplicate: bool
    contains_verified_near_duplicate: bool

    @property
    def size(self) -> int:
        return len(self.members)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--stamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp suffix in YYYYMMDD_HHMMSS format.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--hash-batch-size", type=int, default=32)
    parser.add_argument(
        "--verified-near-duplicate-csv",
        type=Path,
        help="Optional human-reviewed pair table; ignored unless grouping is explicitly enabled.",
    )
    parser.add_argument(
        "--group-verified-near-duplicates",
        action="store_true",
        help="Union only pairs explicitly verified as the same source in the supplied CSV.",
    )
    parser.add_argument(
        "--audit-run-dir",
        type=Path,
        help="Optional existing research_correctness_audit_* run receiving collision-safe summary copies.",
    )
    return parser.parse_args()


def resolve_existing_file(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_output_root(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved != allowed:
        raise ValueError(f"Output root must be exactly {allowed}; received {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_audit_run(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    allowed = (PROJECT_ROOT / "outputs/runs").resolve()
    if allowed not in resolved.parents:
        raise ValueError(f"Audit run must be under {allowed}")
    if not resolved.name.startswith("research_correctness_audit_"):
        raise ValueError("Audit run must be named research_correctness_audit_*")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    for child in ("tables", "manifests", "logs"):
        if not (resolved / child).is_dir():
            raise FileNotFoundError(resolved / child)
    return resolved


def validate_configuration(args: argparse.Namespace) -> tuple[float, float, float]:
    if not STAMP_PATTERN.fullmatch(args.stamp):
        raise ValueError("--stamp must use YYYYMMDD_HHMMSS")
    fractions = (float(args.train_frac), float(args.val_frac), float(args.test_frac))
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("All split fractions must be strictly between 0 and 1")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("train/validation/test fractions must sum to 1")
    if args.hash_batch_size <= 0:
        raise ValueError("--hash-batch-size must be positive")
    if args.group_verified_near_duplicates and args.verified_near_duplicate_csv is None:
        raise ValueError(
            "--group-verified-near-duplicates requires --verified-near-duplicate-csv"
        )
    return fractions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False)


def safe_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def metadata_vector(handle: h5py.File, key: str, n_sources: int) -> np.ndarray:
    if key not in handle:
        return np.full(n_sources, np.nan, dtype=np.float64)
    values = np.asarray(handle[key][:]).squeeze()
    if values.ndim != 1 or len(values) != n_sources:
        raise ValueError(f"Metadata {key!r} has incompatible shape {values.shape}")
    return values.astype(np.float64, copy=False)


def coordinate_key(ra: float, dec: float) -> str:
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return ""
    # Float hex is an exact, locale-independent representation of the stored
    # value. Canonicalize signed zero because IEEE equality treats +/-0 alike.
    canonical_ra = 0.0 if float(ra) == 0.0 else float(ra)
    canonical_dec = 0.0 if float(dec) == 0.0 else float(dec)
    return f"{canonical_ra.hex()}|{canonical_dec.hex()}"


def exact_image_hashes(
    images: h5py.Dataset,
    batch_size: int,
) -> tuple[list[str], dict[str, list[int]]]:
    n_sources = int(images.shape[0])
    hashes = [""] * n_sources
    groups: dict[str, list[int]] = defaultdict(list)
    for start in range(0, n_sources, batch_size):
        stop = min(start + batch_size, n_sources)
        batch = np.ascontiguousarray(images[start:stop])
        for offset, image in enumerate(batch):
            source_index = start + offset
            digest = hashlib.sha256(image.tobytes(order="C")).hexdigest()
            hashes[source_index] = digest
            groups[digest].append(source_index)
    return hashes, dict(groups)


def union_members(union_find: UnionFind, members: Iterable[int]) -> None:
    ordered = list(members)
    if len(ordered) < 2:
        return
    first = int(ordered[0])
    for member in ordered[1:]:
        union_find.union(first, int(member))


def truthy_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "verified", "same_source", "duplicate"}
    )


def load_verified_near_pairs(
    path: Path | None,
    n_sources: int,
    enabled: bool,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    if path is None:
        return [], {
            "candidate_csv": None,
            "grouping_enabled": False,
            "verified_pairs_grouped": 0,
            "policy": "near duplicates were not automatically grouped",
        }
    resolved = resolve_existing_file(path)
    if not enabled:
        return [], {
            "candidate_csv": str(resolved),
            "candidate_csv_sha256": sha256_file(resolved),
            "grouping_enabled": False,
            "verified_pairs_grouped": 0,
            "policy": "candidate CSV recorded but not used without explicit grouping flag",
        }

    frame = pd.read_csv(resolved)
    column_pairs = (
        ("source_index_a", "source_index_b"),
        ("index_a", "index_b"),
        ("source_a", "source_b"),
    )
    index_columns = next(
        ((left, right) for left, right in column_pairs if {left, right}.issubset(frame.columns)),
        None,
    )
    if index_columns is None:
        raise ValueError(
            "Verified near-duplicate CSV needs source_index_a/source_index_b "
            "(or index_a/index_b) columns"
        )
    verification_columns = (
        "verified_same_source",
        "human_verified_same_source",
        "review_decision",
        "verified_duplicate",
    )
    verification_column = next(
        (name for name in verification_columns if name in frame.columns), None
    )
    if verification_column is None:
        raise ValueError(
            "Explicit near-duplicate grouping requires a human verification column: "
            + ", ".join(verification_columns)
        )
    selected = frame.loc[truthy_values(frame[verification_column])]
    pairs: set[tuple[int, int]] = set()
    left_column, right_column = index_columns
    for row in selected.itertuples(index=False):
        left = int(getattr(row, left_column))
        right = int(getattr(row, right_column))
        if left == right:
            continue
        if not (0 <= left < n_sources and 0 <= right < n_sources):
            raise IndexError(f"Near-duplicate pair ({left}, {right}) is outside the dataset")
        pairs.add((min(left, right), max(left, right)))
    return sorted(pairs), {
        "candidate_csv": str(resolved),
        "candidate_csv_sha256": sha256_file(resolved),
        "grouping_enabled": True,
        "verification_column": verification_column,
        "verified_pairs_grouped": len(pairs),
        "policy": "only explicitly human-verified same-source pairs were grouped",
    }


def stable_group_id(members: Iterable[int]) -> str:
    material = ",".join(str(value) for value in sorted(members)).encode("ascii")
    return "grp_" + hashlib.sha256(material).hexdigest()[:20]


def build_source_groups(
    union_find: UnionFind,
    labels: np.ndarray,
    exact_groups: dict[str, list[int]],
    coordinate_groups: dict[str, list[int]],
    near_pairs: list[tuple[int, int]],
) -> tuple[list[SourceGroup], np.ndarray]:
    n_sources = len(labels)
    member_map: dict[int, list[int]] = defaultdict(list)
    for source_index in range(n_sources):
        member_map[union_find.find(source_index)].append(source_index)

    exact_duplicate_sources = {
        member for members in exact_groups.values() if len(members) > 1 for member in members
    }
    coordinate_duplicate_sources = {
        member
        for members in coordinate_groups.values()
        if len(members) > 1
        for member in members
    }
    verified_near_sources = {member for pair in near_pairs for member in pair}
    label_values = np.unique(labels)
    label_to_position = {int(label): position for position, label in enumerate(label_values)}
    source_group_index = np.empty(n_sources, dtype=np.int64)
    groups: list[SourceGroup] = []
    for group_position, members in enumerate(
        sorted(member_map.values(), key=lambda values: (min(values), len(values)))
    ):
        ordered_members = tuple(sorted(int(value) for value in members))
        counts = np.zeros(len(label_values), dtype=np.int64)
        for member in ordered_members:
            counts[label_to_position[int(labels[member])]] += 1
            source_group_index[member] = group_position
        member_set = set(ordered_members)
        groups.append(
            SourceGroup(
                group_id=stable_group_id(ordered_members),
                members=ordered_members,
                label_counts=counts,
                contains_exact_duplicate=bool(member_set & exact_duplicate_sources),
                contains_coordinate_duplicate=bool(member_set & coordinate_duplicate_sources),
                contains_verified_near_duplicate=bool(member_set & verified_near_sources),
            )
        )
    if len({group.group_id for group in groups}) != len(groups):
        raise RuntimeError("Stable group ID collision; increase the digest prefix length")
    return groups, source_group_index


def assignment_objective(
    assigned_totals: np.ndarray,
    assigned_labels: np.ndarray,
    target_totals: np.ndarray,
    target_labels: np.ndarray,
) -> float:
    total_scale = np.maximum(target_totals, 1.0)
    label_scale = np.maximum(target_labels, 1.0)
    total_error = np.sum(np.square(assigned_totals - target_totals) / total_scale)
    label_error = np.sum(np.square(assigned_labels - target_labels) / label_scale)
    return float(total_error + 0.75 * label_error)


def assign_groups(
    groups: list[SourceGroup],
    labels: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    label_values = np.unique(labels)
    n_labels = len(label_values)
    total_label_counts = np.asarray(
        [np.count_nonzero(labels == label) for label in label_values], dtype=np.float64
    )
    fraction_array = np.asarray(fractions, dtype=np.float64)
    target_totals = fraction_array * len(labels)
    target_labels = fraction_array[:, None] * total_label_counts[None, :]
    assigned_totals = np.zeros(3, dtype=np.float64)
    assigned_labels = np.zeros((3, n_labels), dtype=np.float64)

    rng = np.random.default_rng(seed)
    jitter = rng.random(len(groups))
    order = sorted(
        range(len(groups)),
        key=lambda index: (
            -groups[index].size,
            -int(groups[index].label_counts.max()),
            float(jitter[index]),
            groups[index].group_id,
        ),
    )
    split_priority = list(rng.permutation(3).astype(int))
    priority_rank = {split: rank for rank, split in enumerate(split_priority)}
    assignments: dict[str, str] = {}
    for group_index in order:
        group = groups[group_index]
        candidates: list[tuple[float, int, int]] = []
        for split_index in range(3):
            candidate_totals = assigned_totals.copy()
            candidate_labels = assigned_labels.copy()
            candidate_totals[split_index] += group.size
            candidate_labels[split_index] += group.label_counts
            score = assignment_objective(
                candidate_totals, candidate_labels, target_totals, target_labels
            )
            candidates.append((score, priority_rank[split_index], split_index))
        _, _, selected_split = min(candidates)
        assigned_totals[selected_split] += group.size
        assigned_labels[selected_split] += group.label_counts
        assignments[group.group_id] = SPLIT_NAMES[selected_split]

    if set(assignments) != {group.group_id for group in groups}:
        raise RuntimeError("Not every source group received exactly one split")
    details = {
        "label_values": [int(value) for value in label_values],
        "target_source_counts": {
            SPLIT_NAMES[index]: float(target_totals[index]) for index in range(3)
        },
        "actual_source_counts": {
            SPLIT_NAMES[index]: int(assigned_totals[index]) for index in range(3)
        },
        "target_label_counts": {
            SPLIT_NAMES[split_index]: {
                str(int(label)): float(target_labels[split_index, label_index])
                for label_index, label in enumerate(label_values)
            }
            for split_index in range(3)
        },
        "actual_label_counts": {
            SPLIT_NAMES[split_index]: {
                str(int(label)): int(assigned_labels[split_index, label_index])
                for label_index, label in enumerate(label_values)
            }
            for split_index in range(3)
        },
        "objective": assignment_objective(
            assigned_totals, assigned_labels, target_totals, target_labels
        ),
    }
    return assignments, details


def clean_scalar(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def group_reason(group: SourceGroup) -> str:
    reasons = []
    if group.contains_exact_duplicate:
        reasons.append("exact_pixels")
    if group.contains_coordinate_duplicate:
        reasons.append("exact_coordinates")
    if group.contains_verified_near_duplicate:
        reasons.append("verified_near_duplicate")
    return "+".join(reasons) if reasons else "singleton"


def build_manifest(
    labels: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    pxscale: np.ndarray,
    exact_hashes: list[str],
    coordinate_keys: list[str],
    groups: list[SourceGroup],
    source_group_index: np.ndarray,
    assignments: dict[str, str],
    seed: int,
) -> pd.DataFrame:
    local_counters = Counter()
    rows: list[dict[str, Any]] = []
    for source_index in range(len(labels)):
        group = groups[int(source_group_index[source_index])]
        split = assignments[group.group_id]
        local_index = local_counters[split]
        local_counters[split] += 1
        rows.append(
            {
                # Canonical columns used by build_grouped_blend_manifests.py.
                "source_index": source_index,
                "split": split,
                "group_id": group.group_id,
                "label": int(labels[source_index]),
                "ra": clean_scalar(ra[source_index]),
                "dec": clean_scalar(dec[source_index]),
                "redshift": clean_scalar(redshift[source_index]),
                "pxscale": clean_scalar(pxscale[source_index]),
                "exact_sha256": exact_hashes[source_index],
                "coordinate_group_key": coordinate_keys[source_index],
                "group_size": group.size,
                # Additional audit and replay provenance.
                "split_local_index": local_index,
                "group_reason": group_reason(group),
                "group_contains_exact_duplicate": group.contains_exact_duplicate,
                "group_contains_coordinate_duplicate": group.contains_coordinate_duplicate,
                "group_contains_verified_near_duplicate": (
                    group.contains_verified_near_duplicate
                ),
                "split_seed": seed,
                "grouping_version": GROUPING_VERSION,
            }
        )
    return pd.DataFrame(rows)


def duplicate_check_frame(
    groups_by_evidence: dict[str, list[int]],
    manifest: pd.DataFrame,
    evidence_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_by_source = manifest.set_index("source_index")["split"].to_dict()
    group_by_source = manifest.set_index("source_index")["group_id"].to_dict()
    for evidence, members in sorted(groups_by_evidence.items()):
        if len(members) < 2:
            continue
        ordered_members = sorted(int(value) for value in members)
        splits = sorted({str(split_by_source[value]) for value in ordered_members})
        group_ids = sorted({str(group_by_source[value]) for value in ordered_members})
        rows.append(
            {
                evidence_column: evidence,
                "source_count": len(ordered_members),
                "source_indices": ";".join(str(value) for value in ordered_members),
                "group_ids": ";".join(group_ids),
                "assigned_splits": ";".join(splits),
                "group_id_count": len(group_ids),
                "split_count": len(splits),
                "cross_split_leakage": len(splits) > 1,
                "status": "pass" if len(splits) == 1 and len(group_ids) == 1 else "fail",
            }
        )
    columns = [
        evidence_column,
        "source_count",
        "source_indices",
        "group_ids",
        "assigned_splits",
        "group_id_count",
        "split_count",
        "cross_split_leakage",
        "status",
    ]
    return pd.DataFrame(rows, columns=columns)


def group_integrity_frame(
    groups: list[SourceGroup], assignments: dict[str, str], manifest: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed_by_group = {
        str(group_id): {
            "splits": sorted(selected["split"].unique()),
            "source_indices": sorted(selected["source_index"].astype(int)),
            "source_count": len(selected),
        }
        for group_id, selected in manifest.groupby("group_id", sort=False)
    }
    for group in groups:
        observed = observed_by_group.get(
            group.group_id,
            {"splits": [], "source_indices": [], "source_count": 0},
        )
        splits = observed["splits"]
        source_indices = observed["source_indices"]
        passed = (
            observed["source_count"] == group.size
            and len(splits) == 1
            and splits[0] == assignments[group.group_id]
            and source_indices == list(group.members)
        )
        rows.append(
            {
                "group_id": group.group_id,
                "group_size": group.size,
                "assigned_split": assignments[group.group_id],
                "observed_split_count": len(splits),
                "observed_source_count": observed["source_count"],
                "source_indices": ";".join(str(value) for value in group.members),
                "group_reason": group_reason(group),
                "status": "pass" if passed else "fail",
            }
        )
    return pd.DataFrame(rows)


def verification_summary(
    manifest: pd.DataFrame,
    groups: list[SourceGroup],
    exact_check: pd.DataFrame,
    coordinate_check: pd.DataFrame,
    expected_n_sources: int,
    fractions: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], bool]:
    split_sets_per_group = manifest.groupby("group_id")["split"].nunique()
    source_counts = manifest["source_index"].value_counts()
    observed_indices = set(manifest["source_index"].astype(int))
    expected_indices = set(range(expected_n_sources))
    maximum_group_size = max(group.size for group in groups)
    split_counts = manifest["split"].value_counts()
    maximum_split_count_deviation = max(
        abs(int(split_counts.get(name, 0)) - fraction * expected_n_sources)
        for name, fraction in zip(SPLIT_NAMES, fractions, strict=True)
    )
    checks = [
        {
            "check": "all_dataset_sources_present_once",
            "observed": int(manifest["source_index"].nunique()),
            "expected": expected_n_sources,
            "status": (
                "pass"
                if source_counts.eq(1).all() and observed_indices == expected_indices
                else "fail"
            ),
        },
        {
            "check": "no_source_index_in_multiple_splits",
            "observed": int(manifest.groupby("source_index")["split"].nunique().max()),
            "expected": 1,
            "status": (
                "pass"
                if manifest.groupby("source_index")["split"].nunique().max() == 1
                else "fail"
            ),
        },
        {
            "check": "no_group_in_multiple_splits",
            "observed": int(split_sets_per_group.max()),
            "expected": 1,
            "status": "pass" if split_sets_per_group.max() == 1 else "fail",
        },
        {
            "check": "all_groups_assigned",
            "observed": int(manifest["group_id"].nunique()),
            "expected": len(groups),
            "status": "pass" if manifest["group_id"].nunique() == len(groups) else "fail",
        },
        {
            "check": "zero_cross_split_exact_image_groups",
            "observed": int(exact_check["cross_split_leakage"].sum()),
            "expected": 0,
            "status": "pass" if not exact_check["cross_split_leakage"].any() else "fail",
        },
        {
            "check": "zero_cross_split_exact_coordinate_groups",
            "observed": int(coordinate_check["cross_split_leakage"].sum()),
            "expected": 0,
            "status": (
                "pass" if not coordinate_check["cross_split_leakage"].any() else "fail"
            ),
        },
        {
            "check": "valid_split_names_only",
            "observed": ";".join(sorted(manifest["split"].unique())),
            "expected": ";".join(sorted(SPLIT_NAMES)),
            "status": (
                "pass" if set(manifest["split"].unique()) == set(SPLIT_NAMES) else "fail"
            ),
        },
        {
            "check": "split_source_counts_within_group_tolerance",
            "observed": float(maximum_split_count_deviation),
            "expected": f"<= {maximum_group_size + 1}",
            "status": (
                "pass" if maximum_split_count_deviation <= maximum_group_size + 1 else "fail"
            ),
        },
    ]
    return checks, all(row["status"] == "pass" for row in checks)


def copy_summary_files(
    audit_run: Path,
    stamp: str,
    output_dir: Path,
    filenames: Iterable[str],
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for filename in filenames:
        source = output_dir / filename
        destination_parent = audit_run / ("manifests" if filename.endswith(".json") else "tables")
        destination = destination_parent / f"grouped_source_split_{stamp}_{filename}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite audit summary copy {destination}")
        shutil.copy2(source, destination)
        copied[filename] = str(destination)
    return copied


def main() -> None:
    args = parse_args()
    fractions = validate_configuration(args)
    dataset = resolve_existing_file(args.dataset)
    output_root = resolve_output_root(args.output_root)
    audit_run = resolve_audit_run(args.audit_run_dir)
    output_dir = output_root / f"grouped_source_split_{args.stamp}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest directory {output_dir}")
    output_dir.mkdir(parents=False)

    try:
        dataset_sha256 = sha256_file(dataset)
        script_sha256 = sha256_file(Path(__file__).resolve())
        with h5py.File(dataset, "r") as handle:
            missing = [key for key in ("images", "ans") if key not in handle]
            if missing:
                raise KeyError(f"Dataset is missing required keys: {', '.join(missing)}")
            images = handle["images"]
            labels = np.asarray(handle["ans"][:]).squeeze()
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"Expected NHWC RGB images; found {images.shape}")
            n_sources = int(images.shape[0])
            image_shape = list(images.shape[1:])
            image_dtype = str(images.dtype)
            if labels.ndim != 1 or len(labels) != n_sources:
                raise ValueError(f"Labels have incompatible shape {labels.shape}")
            labels = labels.astype(np.int64, copy=False)
            ra = metadata_vector(handle, "ra", n_sources)
            dec = metadata_vector(handle, "dec", n_sources)
            redshift = metadata_vector(handle, "redshift", n_sources)
            pxscale = metadata_vector(handle, "pxscale", n_sources)
            exact_hashes, exact_groups = exact_image_hashes(images, args.hash_batch_size)

        coordinate_keys = [coordinate_key(ra[index], dec[index]) for index in range(n_sources)]
        coordinate_groups_dd: dict[str, list[int]] = defaultdict(list)
        for source_index, key in enumerate(coordinate_keys):
            if key:
                coordinate_groups_dd[key].append(source_index)
        coordinate_groups = dict(coordinate_groups_dd)

        near_pairs, near_provenance = load_verified_near_pairs(
            args.verified_near_duplicate_csv,
            n_sources,
            args.group_verified_near_duplicates,
        )
        union_find = UnionFind(n_sources)
        for members in exact_groups.values():
            union_members(union_find, members)
        for members in coordinate_groups.values():
            union_members(union_find, members)
        for left, right in near_pairs:
            union_find.union(left, right)

        groups, source_group_index = build_source_groups(
            union_find, labels, exact_groups, coordinate_groups, near_pairs
        )
        assignments, assignment_details = assign_groups(groups, labels, fractions, args.seed)
        manifest = build_manifest(
            labels,
            ra,
            dec,
            redshift,
            pxscale,
            exact_hashes,
            coordinate_keys,
            groups,
            source_group_index,
            assignments,
            args.seed,
        )
        exact_check = duplicate_check_frame(exact_groups, manifest, "exact_sha256")
        coordinate_check = duplicate_check_frame(
            coordinate_groups, manifest, "coordinate_group_key"
        )
        group_check = group_integrity_frame(groups, assignments, manifest)
        verification, all_checks_pass = verification_summary(
            manifest,
            groups,
            exact_check,
            coordinate_check,
            n_sources,
            fractions,
        )
        if not all_checks_pass or not group_check["status"].eq("pass").all():
            raise RuntimeError(
                "Grouped split integrity verification failed before output finalization: "
                + json.dumps(verification)
            )

        safe_csv(output_dir / "source_split_manifest.csv", manifest)
        safe_csv(output_dir / "group_integrity_check.csv", group_check)
        safe_csv(output_dir / "cross_split_duplicate_check.csv", exact_check)
        safe_csv(output_dir / "cross_split_coordinate_check.csv", coordinate_check)
        output_hashes = {
            name: sha256_file(output_dir / name)
            for name in (
                "source_split_manifest.csv",
                "group_integrity_check.csv",
                "cross_split_duplicate_check.csv",
                "cross_split_coordinate_check.csv",
            )
        }

        split_counts = manifest["split"].value_counts().to_dict()
        group_counts = (
            manifest[["group_id", "split"]]
            .drop_duplicates()["split"]
            .value_counts()
            .to_dict()
        )
        summary = {
            "status": "complete",
            "created_at": datetime.now().astimezone().isoformat(),
            "output_directory": str(output_dir),
            "dataset": {
                "path": str(dataset),
                "sha256": dataset_sha256,
                "size_bytes": dataset.stat().st_size,
                "n_sources": n_sources,
                "image_shape": image_shape,
                "image_dtype": image_dtype,
                "label_values": [int(value) for value in np.unique(labels)],
                "finite_ra_dec_count": int(np.count_nonzero(np.isfinite(ra) & np.isfinite(dec))),
            },
            "grouping": {
                "version": GROUPING_VERSION,
                "total_groups": len(groups),
                "multi_source_groups": int(sum(group.size > 1 for group in groups)),
                "maximum_group_size": int(max(group.size for group in groups)),
                "exact_duplicate_evidence_groups": int(
                    sum(len(members) > 1 for members in exact_groups.values())
                ),
                "coordinate_duplicate_evidence_groups": int(
                    sum(len(members) > 1 for members in coordinate_groups.values())
                ),
                "near_duplicate_policy": near_provenance,
            },
            "split_assignment": {
                "seed": args.seed,
                "fractions": dict(zip(SPLIT_NAMES, fractions, strict=True)),
                "source_counts": {name: int(split_counts.get(name, 0)) for name in SPLIT_NAMES},
                "group_counts": {name: int(group_counts.get(name, 0)) for name in SPLIT_NAMES},
                "source_fractions": {
                    name: float(split_counts.get(name, 0) / n_sources) for name in SPLIT_NAMES
                },
                "label_balance": assignment_details,
            },
            "integrity_checks": verification,
            "integrity_passed": all_checks_pass,
            "provenance": {
                "command": shlex.join(sys.argv),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "h5py": h5py.__version__,
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": script_sha256,
                "output_sha256": output_hashes,
            },
        }
        safe_json(output_dir / "split_summary.json", summary)

        copied: dict[str, str] = {}
        if audit_run is not None:
            copied = copy_summary_files(
                audit_run,
                args.stamp,
                output_dir,
                (
                    "split_summary.json",
                    "cross_split_duplicate_check.csv",
                    "cross_split_coordinate_check.csv",
                ),
            )
            safe_json(
                audit_run / "manifests" / f"grouped_source_split_{args.stamp}_location.json",
                {
                    "manifest_directory": str(output_dir),
                    "source_split_manifest_sha256": output_hashes["source_split_manifest.csv"],
                    "summary_copies": copied,
                },
            )

        print(json.dumps({
            "status": "complete",
            "output_directory": str(output_dir),
            "source_counts": summary["split_assignment"]["source_counts"],
            "group_counts": summary["split_assignment"]["group_counts"],
            "integrity_passed": all_checks_pass,
            "audit_copies": copied,
        }, indent=2))
    except Exception as error:
        failure_path = output_dir / "generation_failure.json"
        if not failure_path.exists():
            safe_json(
                failure_path,
                {
                    "status": "failed",
                    "created_at": datetime.now().astimezone().isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "command": shlex.join(sys.argv),
                    "note": "Interrupted/failed directory intentionally preserved; rerun with a new stamp.",
                },
            )
        raise


if __name__ == "__main__":
    main()
