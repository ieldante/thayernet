#!/usr/bin/env python3
"""Audit the pinned Galaxy Zoo DESI catalog and prepare DR10 source samples.

The script is deliberately phase-gated and append-only:

* ``--phase audit`` writes the four Part B audit artifacts.
* ``--phase engineering`` writes a deterministic, morphology-stratified
  engineering manifest and the pilot-selection protocol.
* ``--phase pilot`` is a later, separate action.  It refuses to run without a
  frozen raw cutout size, an existing field-of-view report, an existing file
  documenting the fixed engineering rules, and the engineering manifest.

Every output is opened in exclusive-create mode.  Existing outputs are never
replaced.  PyArrow column projection and record batches keep unrelated Parquet
columns out of memory, and conditional morphology nulls are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "data/catalogs/galaxy_zoo_desi/"
    "gz_desi_deep_learning_catalog_friendly.parquet"
)
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "outputs/runs/dr10_foundation_20260711_024415"
)
DEFAULT_SEED = 20260711
SOUTH_DEC_LIMIT_DEG = 32.375

IDENTIFIER_COLUMNS = [
    "dr8_id",
    "brickid",
    "objid",
    "__index_level_0__",
    "hdf5_loc",
]
COORDINATE_COLUMNS = ["ra", "dec"]

TASK_COLUMNS: dict[str, list[str]] = {
    "smooth-or-featured": [
        "smooth-or-featured_smooth_fraction",
        "smooth-or-featured_featured-or-disk_fraction",
        "smooth-or-featured_artifact_fraction",
    ],
    "disk-edge-on": [
        "disk-edge-on_yes_fraction",
        "disk-edge-on_no_fraction",
    ],
    "has-spiral-arms": [
        "has-spiral-arms_yes_fraction",
        "has-spiral-arms_no_fraction",
    ],
    "bar": [
        "bar_strong_fraction",
        "bar_weak_fraction",
        "bar_no_fraction",
    ],
    "bulge-size": [
        "bulge-size_dominant_fraction",
        "bulge-size_large_fraction",
        "bulge-size_moderate_fraction",
        "bulge-size_small_fraction",
        "bulge-size_none_fraction",
    ],
    "how-rounded": [
        "how-rounded_round_fraction",
        "how-rounded_in-between_fraction",
        "how-rounded_cigar-shaped_fraction",
    ],
    "edge-on-bulge": [
        "edge-on-bulge_boxy_fraction",
        "edge-on-bulge_none_fraction",
        "edge-on-bulge_rounded_fraction",
    ],
    "spiral-winding": [
        "spiral-winding_tight_fraction",
        "spiral-winding_medium_fraction",
        "spiral-winding_loose_fraction",
    ],
    "spiral-arm-count": [
        "spiral-arm-count_1_fraction",
        "spiral-arm-count_2_fraction",
        "spiral-arm-count_3_fraction",
        "spiral-arm-count_4_fraction",
        "spiral-arm-count_more-than-4_fraction",
        "spiral-arm-count_cant-tell_fraction",
    ],
    "merging": [
        "merging_none_fraction",
        "merging_minor-disturbance_fraction",
        "merging_major-disturbance_fraction",
        "merging_merger_fraction",
    ],
}
MORPHOLOGY_COLUMNS = [
    column for columns in TASK_COLUMNS.values() for column in columns
]

STRATUM_ORDER = [
    "artifact",
    "merger",
    "major_disturbance",
    "minor_disturbance",
    "featured_edge_on",
    "featured_spiral",
    "featured_strong_bar",
    "featured_other",
    "smooth_round",
    "smooth_in_between",
    "smooth_cigar",
    "smooth_other",
]
ENGINEERING_WEIGHTS = {
    "artifact": 8,
    "merger": 10,
    "major_disturbance": 8,
    "minor_disturbance": 6,
    "featured_edge_on": 10,
    "featured_spiral": 10,
    "featured_strong_bar": 8,
    "featured_other": 8,
    "smooth_round": 8,
    "smooth_in_between": 8,
    "smooth_cigar": 6,
    "smooth_other": 10,
}
PILOT_WEIGHTS = {
    "artifact": 2,
    "merger": 5,
    "major_disturbance": 4,
    "minor_disturbance": 2,
    "featured_edge_on": 8,
    "featured_spiral": 12,
    "featured_strong_bar": 8,
    "featured_other": 9,
    "smooth_round": 15,
    "smooth_in_between": 15,
    "smooth_cigar": 7,
    "smooth_other": 13,
}

SELECTION_COLUMNS = list(
    dict.fromkeys(
        [
            "dr8_id",
            "__index_level_0__",
            "ra",
            "dec",
            *TASK_COLUMNS["smooth-or-featured"],
            *TASK_COLUMNS["merging"],
            *TASK_COLUMNS["disk-edge-on"],
            *TASK_COLUMNS["has-spiral-arms"],
            *TASK_COLUMNS["bar"],
            *TASK_COLUMNS["how-rounded"],
        ]
    )
)
MANIFEST_CATALOG_COLUMNS = [
    "dr8_id",
    "__index_level_0__",
    "ra",
    "dec",
    "brickid",
    "objid",
    "hdf5_loc",
    *MORPHOLOGY_COLUMNS,
]
MANIFEST_FIELDS = [
    "source_id",
    "catalog_row_index",
    "ra",
    "dec",
    "brickid",
    "objid",
    "hdf5_loc",
    "provisional_group_id",
    "selection_phase",
    "morphology_stratum",
    "selection_reason",
    "selection_seed",
    "selection_rank_within_stratum",
    "selection_hash",
    "geometric_prefilter",
    "coverage_status",
    "brightness_selection_status",
    "size_selection_status",
    "raw_cutout_size",
    "fov_study_report",
    "fov_study_report_sha256",
    "engineering_rules_file",
    "engineering_rules_sha256",
    *MORPHOLOGY_COLUMNS,
]


@dataclass(frozen=True)
class DuplicateMetrics:
    total_rows: int
    distinct_keys: int
    duplicate_groups: int
    duplicate_rows_including_first: int
    excess_duplicate_rows: int
    max_group_size: int


@dataclass
class TaskAudit:
    name: str
    columns: list[str]
    available: int = 0
    missing: int = 0
    partial_rows: int = 0
    nonfinite_nonnull_values: int = 0
    out_of_range_values: int = 0
    ties: int = 0
    argmax_counts: list[int] | None = None
    sum_min: float = math.inf
    sum_max: float = -math.inf
    max_abs_sum_minus_one: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("audit", "engineering", "pilot")
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--engineering-count", type=int, default=100)
    parser.add_argument("--pilot-count", type=int, default=2_500)
    parser.add_argument(
        "--engineering-manifest",
        type=Path,
        help="Existing engineering CSV; required by pilot (defaults inside run).",
    )
    parser.add_argument(
        "--raw-cutout-size",
        type=int,
        choices=(128, 192, 256),
        help="Frozen raw cutout size; required only by --phase pilot.",
    )
    parser.add_argument(
        "--fov-study-report",
        type=Path,
        help="Existing report that freezes the FOV choice; required by pilot.",
    )
    parser.add_argument(
        "--engineering-rules-file",
        type=Path,
        help="Existing file documenting fixed engineering rules; required by pilot.",
    )
    return parser.parse_args()


def resolve_existing_file(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_run_dir(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    allowed = (PROJECT_ROOT / "outputs/runs").resolve()
    if allowed not in resolved.parents:
        raise ValueError(f"Run directory must be under {allowed}: {resolved}")
    if not resolved.name.startswith("dr10_foundation_"):
        raise ValueError("Run directory must be named dr10_foundation_*")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    for child in ("diagnostics", "tables", "manifests"):
        if not (resolved / child).is_dir():
            raise FileNotFoundError(resolved / child)
    return resolved


def ensure_absent(paths: Iterable[Path]) -> None:
    collisions = [path for path in paths if path.exists()]
    if collisions:
        joined = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{joined}")


def exclusive_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def exclusive_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_column(batch: pa.RecordBatch, name: str) -> pa.Array:
    return batch.column(batch.schema.get_field_index(name))


def numpy_column(batch: pa.RecordBatch, name: str) -> np.ndarray:
    return batch_column(batch, name).to_numpy(zero_copy_only=False)


def duplicate_metrics_from_sorted(sorted_values: np.ndarray) -> DuplicateMetrics:
    total = int(len(sorted_values))
    if total == 0:
        return DuplicateMetrics(0, 0, 0, 0, 0, 0)
    equal = sorted_values[1:] == sorted_values[:-1]
    starts = np.r_[0, np.flatnonzero(~equal) + 1]
    lengths = np.diff(np.r_[starts, total])
    duplicated = lengths[lengths > 1]
    excess = int((duplicated - 1).sum()) if len(duplicated) else 0
    return DuplicateMetrics(
        total_rows=total,
        distinct_keys=total - excess,
        duplicate_groups=int(len(duplicated)),
        duplicate_rows_including_first=int(duplicated.sum()),
        excess_duplicate_rows=excess,
        max_group_size=int(duplicated.max()) if len(duplicated) else 1,
    )


def coordinate_duplicate_metrics(
    ra: np.ndarray, dec: np.ndarray
) -> DuplicateMetrics:
    order = np.lexsort((dec, ra))
    sorted_ra = ra[order]
    sorted_dec = dec[order]
    total = int(len(order))
    if total == 0:
        return DuplicateMetrics(0, 0, 0, 0, 0, 0)
    equal = (sorted_ra[1:] == sorted_ra[:-1]) & (
        sorted_dec[1:] == sorted_dec[:-1]
    )
    starts = np.r_[0, np.flatnonzero(~equal) + 1]
    lengths = np.diff(np.r_[starts, total])
    duplicated = lengths[lengths > 1]
    excess = int((duplicated - 1).sum()) if len(duplicated) else 0
    return DuplicateMetrics(
        total_rows=total,
        distinct_keys=total - excess,
        duplicate_groups=int(len(duplicated)),
        duplicate_rows_including_first=int(duplicated.sum()),
        excess_duplicate_rows=excess,
        max_group_size=int(duplicated.max()) if len(duplicated) else 1,
    )


def semantic_role(column: str) -> tuple[str, str, str]:
    if column in {"ra", "dec"}:
        return "coordinate", "", "ICRS sky coordinate in degrees"
    if column == "dr8_id":
        return "stable_source_id", "", "Unique within the pinned catalog"
    if column in {"brickid", "objid"}:
        return "source_id_component", "", "Component of dr8_id"
    if column == "__index_level_0__":
        return "catalog_row_index", "", "Pinned-file row index; not cross-version ID"
    if column == "hdf5_loc":
        return "upstream_provenance", "", "Constant upstream HDF5 locator"
    for task, columns in TASK_COLUMNS.items():
        if column in columns:
            return (
                "morphology_vote_fraction",
                task,
                "Conditional nulls are preserved; morphology is analysis metadata only",
            )
    return "unclassified", "", ""


def audit_identifiers_and_coordinates(
    parquet: pq.ParquetFile, batch_size: int
) -> dict[str, Any]:
    columns = [
        "dr8_id",
        "brickid",
        "objid",
        "__index_level_0__",
        "hdf5_loc",
        "ra",
        "dec",
    ]
    ra_parts: list[np.ndarray] = []
    dec_parts: list[np.ndarray] = []
    composite_parts: list[np.ndarray] = []
    row_offset = 0
    result: dict[str, Any] = {
        "ra_null": 0,
        "dec_null": 0,
        "ra_nonfinite": 0,
        "dec_nonfinite": 0,
        "ra_out_of_range": 0,
        "dec_out_of_range": 0,
        "ra_min": math.inf,
        "ra_max": -math.inf,
        "dec_min": math.inf,
        "dec_max": -math.inf,
        "south_dec_count": 0,
        "dr8_id_null": 0,
        "dr8_id_empty": 0,
        "dr8_id_composite_mismatch": 0,
        "index_sequence_mismatch": 0,
        "hdf5_values": set(),
        "brickid_min": math.inf,
        "brickid_max": -math.inf,
        "objid_min": math.inf,
        "objid_max": -math.inf,
    }

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        n = batch.num_rows
        ra_array = batch_column(batch, "ra")
        dec_array = batch_column(batch, "dec")
        result["ra_null"] += ra_array.null_count
        result["dec_null"] += dec_array.null_count
        ra = np.asarray(numpy_column(batch, "ra"), dtype=np.float64)
        dec = np.asarray(numpy_column(batch, "dec"), dtype=np.float64)
        ra_finite = np.isfinite(ra)
        dec_finite = np.isfinite(dec)
        result["ra_nonfinite"] += int((~ra_finite).sum())
        result["dec_nonfinite"] += int((~dec_finite).sum())
        result["ra_out_of_range"] += int(
            (ra_finite & ((ra < 0.0) | (ra >= 360.0))).sum()
        )
        result["dec_out_of_range"] += int(
            (dec_finite & ((dec < -90.0) | (dec > 90.0))).sum()
        )
        if ra_finite.any():
            result["ra_min"] = min(result["ra_min"], float(ra[ra_finite].min()))
            result["ra_max"] = max(result["ra_max"], float(ra[ra_finite].max()))
        if dec_finite.any():
            result["dec_min"] = min(
                result["dec_min"], float(dec[dec_finite].min())
            )
            result["dec_max"] = max(
                result["dec_max"], float(dec[dec_finite].max())
            )
        both_finite = ra_finite & dec_finite
        result["south_dec_count"] += int(
            (both_finite & (dec < SOUTH_DEC_LIMIT_DEG)).sum()
        )
        ra_parts.append(ra[both_finite].copy())
        dec_parts.append(dec[both_finite].copy())

        brick = np.asarray(numpy_column(batch, "brickid"), dtype=np.int64)
        obj = np.asarray(numpy_column(batch, "objid"), dtype=np.int64)
        if np.any(brick < 0) or np.any(obj < 0):
            raise ValueError("Negative brickid/objid cannot be packed losslessly")
        result["brickid_min"] = min(result["brickid_min"], int(brick.min()))
        result["brickid_max"] = max(result["brickid_max"], int(brick.max()))
        result["objid_min"] = min(result["objid_min"], int(obj.min()))
        result["objid_max"] = max(result["objid_max"], int(obj.max()))
        packed = (brick.astype(np.uint64) << np.uint64(32)) | obj.astype(
            np.uint64
        )
        composite_parts.append(packed)

        dr8 = batch_column(batch, "dr8_id")
        result["dr8_id_null"] += dr8.null_count
        empty = pc.fill_null(pc.equal(dr8, ""), False)
        result["dr8_id_empty"] += int(pc.sum(pc.cast(empty, pa.int64())).as_py())
        expected = pc.binary_join_element_wise(
            pc.cast(batch_column(batch, "brickid"), pa.string()),
            pc.cast(batch_column(batch, "objid"), pa.string()),
            "_",
        )
        matches = pc.fill_null(pc.equal(dr8, expected), False)
        result["dr8_id_composite_mismatch"] += n - int(
            pc.sum(pc.cast(matches, pa.int64())).as_py()
        )

        index_values = np.asarray(
            numpy_column(batch, "__index_level_0__"), dtype=np.int64
        )
        expected_index = np.arange(row_offset, row_offset + n, dtype=np.int64)
        result["index_sequence_mismatch"] += int(
            (index_values != expected_index).sum()
        )
        row_offset += n
        result["hdf5_values"].update(
            value
            for value in pc.unique(batch_column(batch, "hdf5_loc")).to_pylist()
            if value is not None
        )

    ra_all = np.concatenate(ra_parts)
    dec_all = np.concatenate(dec_parts)
    composite = np.concatenate(composite_parts)
    result["coordinate_duplicates"] = coordinate_duplicate_metrics(ra_all, dec_all)
    composite.sort()
    result["composite_duplicates"] = duplicate_metrics_from_sorted(composite)
    if result["dr8_id_composite_mismatch"]:
        raise ValueError(
            "dr8_id does not exactly match brickid_objid; cannot infer exact ID "
            "duplicate counts from the numeric composite"
        )
    result["dr8_id_duplicates"] = result["composite_duplicates"]
    result["rows_scanned"] = row_offset
    result["finite_coordinate_rows"] = int(len(ra_all))
    return result


def audit_task(
    parquet: pq.ParquetFile,
    name: str,
    columns: list[str],
    batch_size: int,
) -> TaskAudit:
    result = TaskAudit(name=name, columns=columns, argmax_counts=[0] * len(columns))
    rows = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows += batch.num_rows
        arrays = [batch_column(batch, column) for column in columns]
        present = np.column_stack(
            [~array.is_null().to_numpy(zero_copy_only=False) for array in arrays]
        )
        values = np.column_stack(
            [array.to_numpy(zero_copy_only=False) for array in arrays]
        ).astype(np.float64, copy=False)
        finite = np.isfinite(values)
        valid = finite.all(axis=1)
        any_finite = finite.any(axis=1)
        result.available += int(valid.sum())
        result.partial_rows += int((any_finite & ~valid).sum())
        result.nonfinite_nonnull_values += int((present & ~finite).sum())
        finite_values = values[finite]
        result.out_of_range_values += int(
            ((finite_values < 0.0) | (finite_values > 1.0)).sum()
        )
        if valid.any():
            valid_values = values[valid]
            maxima = valid_values.max(axis=1)
            result.ties += int(
                (np.sum(valid_values == maxima[:, None], axis=1) > 1).sum()
            )
            argmax = np.argmax(valid_values, axis=1)
            for category in range(len(columns)):
                assert result.argmax_counts is not None
                result.argmax_counts[category] += int((argmax == category).sum())
            sums = valid_values.sum(axis=1, dtype=np.float64)
            result.sum_min = min(result.sum_min, float(sums.min()))
            result.sum_max = max(result.sum_max, float(sums.max()))
            result.max_abs_sum_minus_one = max(
                result.max_abs_sum_minus_one,
                float(np.max(np.abs(sums - 1.0))),
            )
    result.missing = rows - result.available
    return result


def audit_gating(parquet: pq.ParquetFile, batch_size: int) -> dict[str, int]:
    columns = [
        "smooth-or-featured_smooth_fraction",
        "smooth-or-featured_featured-or-disk_fraction",
        "disk-edge-on_yes_fraction",
        "disk-edge-on_no_fraction",
        "has-spiral-arms_yes_fraction",
        "bar_strong_fraction",
        "bulge-size_dominant_fraction",
        "how-rounded_round_fraction",
        "edge-on-bulge_boxy_fraction",
        "spiral-winding_tight_fraction",
        "spiral-arm-count_1_fraction",
    ]
    mismatches = Counter()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        values = {name: numpy_column(batch, name) for name in columns}
        smooth = values[columns[0]]
        featured = values[columns[1]]
        edge_yes = values[columns[2]]
        edge_no = values[columns[3]]
        spiral_yes = values[columns[4]]
        has_spiral_available = np.isfinite(spiral_yes)
        bar_available = np.isfinite(values[columns[5]])
        bulge_available = np.isfinite(values[columns[6]])
        rounded_available = np.isfinite(values[columns[7]])
        edge_bulge_available = np.isfinite(values[columns[8]])
        winding_available = np.isfinite(values[columns[9]])
        count_available = np.isfinite(values[columns[10]])
        disk_available = np.isfinite(edge_yes) & np.isfinite(edge_no)

        mismatches["rounded_iff_smooth_gt_half"] += int(
            np.logical_xor(rounded_available, smooth > 0.5).sum()
        )
        mismatches["disk_iff_featured_gt_half"] += int(
            np.logical_xor(disk_available, featured > 0.5).sum()
        )
        face_score = featured * edge_no
        face_expected = face_score > 0.5
        mismatches["spiral_question_iff_featured_not_edge_gt_half"] += int(
            np.logical_xor(has_spiral_available, face_expected).sum()
        )
        mismatches["bar_iff_featured_not_edge_gt_half"] += int(
            np.logical_xor(bar_available, face_expected).sum()
        )
        mismatches["bulge_iff_featured_not_edge_gt_half"] += int(
            np.logical_xor(bulge_available, face_expected).sum()
        )
        mismatches["edge_bulge_iff_featured_edge_gt_half"] += int(
            np.logical_xor(edge_bulge_available, featured * edge_yes > 0.5).sum()
        )
        spiral_expected = featured * edge_no * spiral_yes > 0.5
        mismatches["winding_iff_cumulative_spiral_gt_half"] += int(
            np.logical_xor(winding_available, spiral_expected).sum()
        )
        mismatches["arm_count_iff_cumulative_spiral_gt_half"] += int(
            np.logical_xor(count_available, spiral_expected).sum()
        )
    return dict(mismatches)


def build_column_inventory(
    parquet: pq.ParquetFile,
) -> list[dict[str, Any]]:
    metadata = parquet.metadata
    if metadata.num_row_groups != 1:
        raise ValueError("Pinned catalog audit expects exactly one Parquet row group")
    group = metadata.row_group(0)
    rows: list[dict[str, Any]] = []
    for ordinal, field in enumerate(parquet.schema_arrow):
        column_meta = group.column(ordinal)
        statistics = column_meta.statistics
        null_count = statistics.null_count if statistics is not None else ""
        minimum = statistics.min if statistics is not None else ""
        maximum = statistics.max if statistics is not None else ""
        role, task, notes = semantic_role(field.name)
        rows.append(
            {
                "ordinal": ordinal,
                "column_name": field.name,
                "arrow_type": str(field.type),
                "parquet_physical_type": column_meta.physical_type,
                "semantic_role": role,
                "morphology_question": task,
                "null_count": null_count,
                "non_null_count": (
                    metadata.num_rows - int(null_count) if null_count != "" else ""
                ),
                "null_fraction": (
                    int(null_count) / metadata.num_rows if null_count != "" else ""
                ),
                "min": minimum,
                "max": maximum,
                "notes": notes,
            }
        )
    return rows


def category_label(column: str) -> str:
    suffix = column.rsplit("_", 1)[0]
    for prefix in (
        "smooth-or-featured_",
        "disk-edge-on_",
        "has-spiral-arms_",
        "bar_",
        "bulge-size_",
        "how-rounded_",
        "edge-on-bulge_",
        "spiral-winding_",
        "spiral-arm-count_",
        "merging_",
    ):
        if suffix.startswith(prefix):
            return suffix[len(prefix) :]
    return suffix


def audit_report_markdown(
    catalog: Path,
    parquet: pq.ParquetFile,
    identity: dict[str, Any],
    tasks: dict[str, TaskAudit],
    gating: dict[str, int],
) -> str:
    lines = [
        "# Galaxy Zoo DESI Friendly Catalog Schema and Scientific Audit",
        "",
        "## Scope and provenance",
        "",
        f"- Catalog: `{catalog}`",
        f"- File size: {catalog.stat().st_size:,} bytes",
        f"- Exact rows: {parquet.metadata.num_rows:,}",
        f"- Exact columns: {parquet.metadata.num_columns}",
        f"- Parquet row groups: {parquet.metadata.num_row_groups}",
        f"- Writer: `{parquet.metadata.created_by}`",
        "- Embedded metadata contains Arrow/Pandas schema metadata but no Galaxy Zoo release-version field. Version provenance therefore depends on the pinned path and checksums recorded separately.",
        "- Audit implementation projected only required columns and processed record batches; it did not load the full table at once.",
        "",
        "## Exact schema",
        "",
        "```text",
        str(parquet.schema_arrow).rstrip(),
        "```",
        "",
        "## Available scientific fields",
        "",
        "- Coordinates: `ra`, `dec` (double precision, degrees).",
        "- Stable pinned-catalog identifier: `dr8_id`; it exactly equals `brickid_objid` for every row.",
        "- Pinned-file row identifier: `__index_level_0__`, exactly 0 through 8,689,369. It must not be assumed stable across catalog versions.",
        "- Morphology: 34 deep-learning vote-fraction columns across ten Galaxy Zoo decision-tree questions.",
        "- No magnitude or flux columns are present.",
        "- No angular-size, radius, axis-ratio, position-angle, or other shape-measurement columns are present.",
        "- No explicit survey-region, camera, photometric-system, or data-release field is present. `brickid` is a spatial tiling ID, not a survey-region label.",
        "- `hdf5_loc` is constant and is upstream provenance, not a region field.",
        "",
        "## Coordinate and identifier quality",
        "",
        "| Metric | Exact result |",
        "|---|---:|",
        f"| Missing RA | {identity['ra_null']:,} |",
        f"| Missing Dec | {identity['dec_null']:,} |",
        f"| Nonfinite RA | {identity['ra_nonfinite']:,} |",
        f"| Nonfinite Dec | {identity['dec_nonfinite']:,} |",
        f"| RA outside [0,360) | {identity['ra_out_of_range']:,} |",
        f"| Dec outside [-90,90] | {identity['dec_out_of_range']:,} |",
        f"| RA range | {identity['ra_min']!r} to {identity['ra_max']!r} deg |",
        f"| Dec range | {identity['dec_min']!r} to {identity['dec_max']!r} deg |",
        f"| Empty/null dr8_id | {identity['dr8_id_empty'] + identity['dr8_id_null']:,} |",
        f"| dr8_id vs brickid_objid mismatches | {identity['dr8_id_composite_mismatch']:,} |",
        f"| Row-index sequence mismatches | {identity['index_sequence_mismatch']:,} |",
        f"| Exact dr8_id duplicate groups | {identity['dr8_id_duplicates'].duplicate_groups:,} |",
        f"| Exact (brickid,objid) duplicate groups | {identity['composite_duplicates'].duplicate_groups:,} |",
        f"| Exact (RA,Dec) duplicate groups | {identity['coordinate_duplicates'].duplicate_groups:,} |",
        "",
        "The absence of exact catalog/coordinate duplicates does not rule out near-coordinate or pixel-identical duplicates after DR10 download. Those later duplicate tests remain required before splitting.",
        "",
        "## Morphology completeness and derived distributions",
        "",
        "The catalog contains continuous vote-fraction estimates, not hard truth labels. Argmax counts below are audit summaries only; morphology remains analysis metadata and must never be a model input.",
        "",
        "| Question | Available | Missing | Argmax tie rows | Derived argmax distribution |",
        "|---|---:|---:|---:|---|",
    ]
    for name, audit in tasks.items():
        assert audit.argmax_counts is not None
        pieces = []
        for column, count in zip(audit.columns, audit.argmax_counts, strict=True):
            fraction = count / audit.available if audit.available else 0.0
            pieces.append(
                f"{category_label(column)}: {count:,} ({100.0 * fraction:.4f}%)"
            )
        lines.append(
            f"| {name} | {audit.available:,} | {audit.missing:,} | "
            f"{audit.ties:,} | {'; '.join(pieces)} |"
        )
    lines.extend(
        [
            "",
            "All finite morphology fractions are in [0,1]. Each question sums to one within float32 rounding; the largest observed absolute row-sum deviation is shown in `catalog_quality_summary.csv`.",
            "",
            "## Conditional-null semantics",
            "",
            "Conditional morphology nulls are scientifically meaningful decision-tree gating, not missing measurements to replace with zero. Exact mismatch counts for the observed gates were:",
            "",
            "| Gate | Mismatching rows |",
            "|---|---:|",
        ]
    )
    for name, mismatch in gating.items():
        lines.append(f"| `{name}` | {mismatch:,} |")
    lines.extend(
        [
            "",
            "The observed exact gates are: rounded iff smooth > 0.5; disk orientation iff featured/disk > 0.5; face-on questions iff featured/disk × not-edge-on > 0.5; edge-on bulge iff featured/disk × edge-on > 0.5; spiral-detail questions iff featured/disk × not-edge-on × spiral > 0.5.",
            "",
            "## Conservative southern prefilter",
            "",
            f"A finite-coordinate `dec < {SOUTH_DEC_LIMIT_DEG}` prefilter retains exactly **{identity['south_dec_count']:,}** rows ({100.0 * identity['south_dec_count'] / parquet.metadata.num_rows:.4f}%).",
            "",
            "The official DR10 description notes a more nuanced historical north/south geometry involving the Galactic Plane. This campaign intentionally uses the stricter declination-only subset for candidate generation. Declination is not treated as proof of coverage: every selected coordinate must later pass an actual `layer=ls-dr10-south`, `bands=grz` FITS validity and three-band coverage check.",
            "",
            "Official documentation: https://www.legacysurvey.org/dr10/description/",
            "",
            "## Selection implications",
            "",
            "- Random sampling would be dominated by smooth and merging-none predictions, so the engineering sample uses deterministic, exclusive morphology strata.",
            "- The friendly catalog cannot impose a brightness or size cut. Those quantities must be measured from validated scientific cutouts or obtained later through a separately documented official-product join.",
            "- No external or advanced morphology catalog is needed for the initial coordinate/morphology selection and none is downloaded by this script.",
            "- Artifact-like rows are retained as an explicit engineering stratum because they are useful for developing transparent quality flags; they are not automatically accepted into a clean source library.",
            "- Exact and near-duplicate/pixel-hash grouping remains a later pre-split gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def quality_summary_rows(
    catalog: Path,
    parquet: pq.ParquetFile,
    identity: dict[str, Any],
    tasks: dict[str, TaskAudit],
    gating: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        metric: str,
        value: Any,
        unit: str,
        status: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "unit": unit,
                "status": status,
                "notes": notes,
            }
        )

    add("file", "size", catalog.stat().st_size, "bytes", "observed", str(catalog))
    add("file", "rows", parquet.metadata.num_rows, "rows", "pass", "exact")
    add("file", "columns", parquet.metadata.num_columns, "columns", "pass", "exact")
    add(
        "file",
        "row_groups",
        parquet.metadata.num_row_groups,
        "row_groups",
        "observed",
        "column projection still avoids unrelated columns",
    )
    for coordinate in ("ra", "dec"):
        add(
            "coordinates",
            f"{coordinate}_null_count",
            identity[f"{coordinate}_null"],
            "rows",
            "pass" if identity[f"{coordinate}_null"] == 0 else "fail",
            "",
        )
        add(
            "coordinates",
            f"{coordinate}_nonfinite_count",
            identity[f"{coordinate}_nonfinite"],
            "rows",
            "pass" if identity[f"{coordinate}_nonfinite"] == 0 else "fail",
            "",
        )
        add(
            "coordinates",
            f"{coordinate}_min",
            identity[f"{coordinate}_min"],
            "degrees",
            "observed",
            "finite rows",
        )
        add(
            "coordinates",
            f"{coordinate}_max",
            identity[f"{coordinate}_max"],
            "degrees",
            "observed",
            "finite rows",
        )
    add(
        "selection",
        "conservative_dec_south_count",
        identity["south_dec_count"],
        "rows",
        "candidate_only",
        f"finite RA/Dec and dec < {SOUTH_DEC_LIMIT_DEG}; actual grz coverage pending",
    )
    add(
        "measurements",
        "magnitude_or_flux_columns",
        0,
        "columns",
        "absent",
        "brightness cannot be filtered from friendly Parquet",
    )
    add(
        "measurements",
        "size_or_shape_columns",
        0,
        "columns",
        "absent",
        "size cannot be filtered from friendly Parquet",
    )
    add(
        "measurements",
        "explicit_survey_region_columns",
        0,
        "columns",
        "absent",
        "actual ls-dr10-south grz validation is mandatory",
    )
    for name, audit in tasks.items():
        add(
            "morphology",
            f"{name}_available_rows",
            audit.available,
            "rows",
            "observed",
            "gated nulls preserved",
        )
        add(
            "morphology",
            f"{name}_missing_rows",
            audit.missing,
            "rows",
            "expected_gating" if audit.missing else "pass",
            "",
        )
        add(
            "morphology",
            f"{name}_max_abs_sum_minus_one",
            audit.max_abs_sum_minus_one,
            "fraction",
            "pass" if audit.max_abs_sum_minus_one < 1e-6 else "review",
            "float32 rounding tolerance",
        )
        add(
            "morphology",
            f"{name}_out_of_range_values",
            audit.out_of_range_values,
            "values",
            "pass" if audit.out_of_range_values == 0 else "fail",
            "expected range [0,1]",
        )
    for gate, count in gating.items():
        add(
            "morphology_gating",
            gate,
            count,
            "rows",
            "pass" if count == 0 else "review",
            "exact decision-tree availability relation",
        )
    return rows


def duplicate_summary_rows(identity: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        (
            "dr8_id",
            identity["dr8_id_duplicates"],
            "dr8_id exactly matched brickid_objid for every row; numeric composite sorted exactly",
        ),
        (
            "brickid_objid",
            identity["composite_duplicates"],
            "uint64 packed numeric composite sorted exactly",
        ),
        (
            "exact_ra_dec",
            identity["coordinate_duplicates"],
            "float64 numeric equality after lexicographic sort; no rounding",
        ),
    ]
    rows = []
    for name, metrics, method in specs:
        rows.append(
            {
                "duplicate_key": name,
                "total_rows": metrics.total_rows,
                "distinct_keys": metrics.distinct_keys,
                "duplicate_groups": metrics.duplicate_groups,
                "duplicate_rows_including_first": metrics.duplicate_rows_including_first,
                "excess_duplicate_rows": metrics.excess_duplicate_rows,
                "max_group_size": metrics.max_group_size,
                "status": "pass" if metrics.duplicate_groups == 0 else "group_required",
                "method": method,
            }
        )
    return rows


def run_audit(catalog: Path, run_dir: Path, batch_size: int) -> None:
    outputs = [
        run_dir / "diagnostics/catalog_schema_report.md",
        run_dir / "tables/catalog_column_inventory.csv",
        run_dir / "tables/catalog_quality_summary.csv",
        run_dir / "tables/catalog_duplicate_summary.csv",
    ]
    ensure_absent(outputs)
    parquet = pq.ParquetFile(catalog)
    expected = set(IDENTIFIER_COLUMNS + COORDINATE_COLUMNS + MORPHOLOGY_COLUMNS)
    missing = sorted(expected - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    identity = audit_identifiers_and_coordinates(parquet, batch_size)
    if identity["rows_scanned"] != parquet.metadata.num_rows:
        raise RuntimeError("Identifier/coordinate scan row count mismatch")
    tasks = {
        name: audit_task(parquet, name, columns, batch_size)
        for name, columns in TASK_COLUMNS.items()
    }
    gating = audit_gating(parquet, batch_size)
    inventory = build_column_inventory(parquet)
    quality = quality_summary_rows(catalog, parquet, identity, tasks, gating)
    duplicates = duplicate_summary_rows(identity)
    report = audit_report_markdown(catalog, parquet, identity, tasks, gating)

    exclusive_csv(
        outputs[1],
        [
            "ordinal",
            "column_name",
            "arrow_type",
            "parquet_physical_type",
            "semantic_role",
            "morphology_question",
            "null_count",
            "non_null_count",
            "null_fraction",
            "min",
            "max",
            "notes",
        ],
        inventory,
    )
    exclusive_csv(
        outputs[2],
        ["section", "metric", "value", "unit", "status", "notes"],
        quality,
    )
    exclusive_csv(
        outputs[3],
        [
            "duplicate_key",
            "total_rows",
            "distinct_keys",
            "duplicate_groups",
            "duplicate_rows_including_first",
            "excess_duplicate_rows",
            "max_group_size",
            "status",
            "method",
        ],
        duplicates,
    )
    exclusive_text(outputs[0], report)
    for path in outputs:
        print(path)


def scaled_quotas(total: int, weights: dict[str, int]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("Requested sample count must be positive")
    weight_sum = sum(weights.values())
    raw = {name: total * weight / weight_sum for name, weight in weights.items()}
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    priority = sorted(
        weights,
        key=lambda name: (-(raw[name] - quotas[name]), STRATUM_ORDER.index(name)),
    )
    for name in priority[:remainder]:
        quotas[name] += 1
    return quotas


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64, copy=False) ^ np.uint64(seed & int(mask))
        z = (z + np.uint64(0x9E3779B97F4A7C15)) & mask
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
        return z ^ (z >> np.uint64(31))


def exclusive_stratum_masks(batch: pa.RecordBatch) -> list[tuple[str, np.ndarray]]:
    top_columns = TASK_COLUMNS["smooth-or-featured"]
    merging_columns = TASK_COLUMNS["merging"]
    top = np.column_stack([numpy_column(batch, column) for column in top_columns])
    merging = np.column_stack(
        [numpy_column(batch, column) for column in merging_columns]
    )
    top_argmax = np.argmax(top, axis=1)
    merging_argmax = np.argmax(merging, axis=1)
    dec = np.asarray(numpy_column(batch, "dec"), dtype=np.float64)
    ra = np.asarray(numpy_column(batch, "ra"), dtype=np.float64)
    eligible = np.isfinite(ra) & np.isfinite(dec) & (dec < SOUTH_DEC_LIMIT_DEG)
    assigned = np.zeros(batch.num_rows, dtype=bool)
    masks: list[tuple[str, np.ndarray]] = []

    def assign(name: str, condition: np.ndarray) -> None:
        mask = eligible & ~assigned & condition
        masks.append((name, mask))
        assigned[mask] = True

    assign("artifact", top_argmax == 2)
    assign("merger", merging_argmax == 3)
    assign("major_disturbance", merging_argmax == 2)
    assign("minor_disturbance", merging_argmax == 1)

    featured = top_argmax == 1
    edge_yes = numpy_column(batch, "disk-edge-on_yes_fraction")
    edge_no = numpy_column(batch, "disk-edge-on_no_fraction")
    is_edge = np.isfinite(edge_yes) & np.isfinite(edge_no) & (edge_yes > edge_no)
    assign("featured_edge_on", featured & is_edge)

    spiral_yes = numpy_column(batch, "has-spiral-arms_yes_fraction")
    spiral_no = numpy_column(batch, "has-spiral-arms_no_fraction")
    is_spiral = (
        np.isfinite(spiral_yes)
        & np.isfinite(spiral_no)
        & (spiral_yes > spiral_no)
    )
    assign("featured_spiral", featured & is_spiral)

    bar = np.column_stack(
        [numpy_column(batch, column) for column in TASK_COLUMNS["bar"]]
    )
    is_strong_bar = np.isfinite(bar).all(axis=1) & (np.argmax(bar, axis=1) == 0)
    assign("featured_strong_bar", featured & is_strong_bar)
    assign("featured_other", featured)

    smooth = top_argmax == 0
    rounded = np.column_stack(
        [numpy_column(batch, column) for column in TASK_COLUMNS["how-rounded"]]
    )
    rounded_valid = np.isfinite(rounded).all(axis=1)
    rounded_argmax = np.argmax(np.nan_to_num(rounded, nan=-math.inf), axis=1)
    assign("smooth_round", smooth & rounded_valid & (rounded_argmax == 0))
    assign("smooth_in_between", smooth & rounded_valid & (rounded_argmax == 1))
    assign("smooth_cigar", smooth & rounded_valid & (rounded_argmax == 2))
    assign("smooth_other", smooth)

    if np.any(eligible & ~assigned):
        raise RuntimeError("Eligible rows remained outside exclusive strata")
    return masks


def retain_smallest_hashes(
    heap: list[tuple[int, int]],
    row_indices: np.ndarray,
    hashes: np.ndarray,
    quota: int,
) -> None:
    for row_index, hash_value in zip(row_indices, hashes, strict=True):
        row = int(row_index)
        value = int(hash_value)
        candidate = (-value, -row)
        if len(heap) < quota:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)


def select_ranked_rows(
    parquet: pq.ParquetFile,
    batch_size: int,
    seed: int,
    quotas: dict[str, int],
    excluded_rows: set[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    heaps: dict[str, list[tuple[int, int]]] = {name: [] for name in STRATUM_ORDER}
    available = Counter()
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=SELECTION_COLUMNS
    ):
        row_indices = np.asarray(
            numpy_column(batch, "__index_level_0__"), dtype=np.int64
        )
        not_excluded = ~np.isin(
            row_indices,
            np.fromiter(excluded_rows, dtype=np.int64),
            assume_unique=False,
        ) if excluded_rows else np.ones(batch.num_rows, dtype=bool)
        for stratum, raw_mask in exclusive_stratum_masks(batch):
            mask = raw_mask & not_excluded
            positions = np.flatnonzero(mask)
            available[stratum] += int(len(positions))
            if not len(positions):
                continue
            rows = row_indices[positions]
            hashes = splitmix64(rows.astype(np.uint64), seed)
            retain_smallest_hashes(heaps[stratum], rows, hashes, quotas[stratum])

    ranked: list[dict[str, Any]] = []
    for stratum in STRATUM_ORDER:
        quota = quotas[stratum]
        if available[stratum] < quota:
            raise RuntimeError(
                f"Stratum {stratum!r} has {available[stratum]} rows, below quota {quota}"
            )
        values = sorted(
            [(-neg_hash, -neg_row) for neg_hash, neg_row in heaps[stratum]],
            key=lambda item: (item[0], item[1]),
        )
        if len(values) != quota:
            raise RuntimeError(f"Selection heap for {stratum!r} did not reach quota")
        for rank, (hash_value, row_index) in enumerate(values, start=1):
            ranked.append(
                {
                    "catalog_row_index": row_index,
                    "morphology_stratum": stratum,
                    "selection_rank_within_stratum": rank,
                    "selection_hash": f"{hash_value:016x}",
                }
            )
    return ranked, dict(available)


def collect_manifest_records(
    parquet: pq.ParquetFile,
    batch_size: int,
    ranked: list[dict[str, Any]],
    phase: str,
    seed: int,
    raw_cutout_size: int | None = None,
    fov_report: Path | None = None,
    rules_file: Path | None = None,
) -> list[dict[str, Any]]:
    selection = {int(row["catalog_row_index"]): row for row in ranked}
    selected_indices = np.array(sorted(selection), dtype=np.int64)
    records: dict[int, dict[str, Any]] = {}
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=MANIFEST_CATALOG_COLUMNS
    ):
        row_indices = np.asarray(
            numpy_column(batch, "__index_level_0__"), dtype=np.int64
        )
        positions = np.flatnonzero(
            np.isin(row_indices, selected_indices, assume_unique=True)
        )
        for position in positions:
            index = int(row_indices[position])
            selected = selection[index]
            catalog_values = {
                name: batch_column(batch, name)[int(position)].as_py()
                for name in MANIFEST_CATALOG_COLUMNS
            }
            source_id = str(catalog_values.pop("dr8_id"))
            catalog_values.pop("__index_level_0__")
            reason = (
                f"{phase}_morphology_stratum={selected['morphology_stratum']};"
                f"finite_ra_dec;dec_lt_{SOUTH_DEC_LIMIT_DEG};"
                "deterministic_splitmix64_rank"
            )
            records[index] = {
                "source_id": source_id,
                "catalog_row_index": index,
                **catalog_values,
                "provisional_group_id": f"dr8_{source_id}",
                "selection_phase": phase,
                "morphology_stratum": selected["morphology_stratum"],
                "selection_reason": reason,
                "selection_seed": seed,
                "selection_rank_within_stratum": selected[
                    "selection_rank_within_stratum"
                ],
                "selection_hash": selected["selection_hash"],
                "geometric_prefilter": f"finite_ra_dec_and_dec_lt_{SOUTH_DEC_LIMIT_DEG}",
                "coverage_status": "pending_actual_ls_dr10_south_grz_fits_validation",
                "brightness_selection_status": "not_available_in_friendly_catalog",
                "size_selection_status": "not_available_in_friendly_catalog",
                "raw_cutout_size": raw_cutout_size,
                "fov_study_report": str(fov_report) if fov_report else "",
                "fov_study_report_sha256": sha256_file(fov_report) if fov_report else "",
                "engineering_rules_file": str(rules_file) if rules_file else "",
                "engineering_rules_sha256": sha256_file(rules_file) if rules_file else "",
            }
    if set(records) != set(selection):
        missing = sorted(set(selection) - set(records))
        raise RuntimeError(f"Could not retrieve selected catalog rows: {missing[:10]}")
    output: list[dict[str, Any]] = []
    for stratum in STRATUM_ORDER:
        stratum_rows = sorted(
            (
                row
                for row in ranked
                if row["morphology_stratum"] == stratum
            ),
            key=lambda row: row["selection_rank_within_stratum"],
        )
        output.extend(records[int(row["catalog_row_index"])] for row in stratum_rows)
    return output


def protocol_markdown(
    catalog: Path,
    run_dir: Path,
    seed: int,
    count: int,
    quotas: dict[str, int],
    available: dict[str, int],
) -> str:
    lines = [
        "# DR10 Engineering and Pilot Source-Selection Protocol",
        "",
        "## Status",
        "",
        f"The deterministic **engineering sample of {count} sources** was created with seed `{seed}`. The pilot candidate CSV was deliberately **not** created in this phase. Pilot selection is gated on a frozen field-of-view choice and documented fixed engineering rules.",
        "",
        "## Input and permitted metadata",
        "",
        f"- Input: pinned friendly Parquet `{catalog}`.",
        "- Candidate geometry: finite RA/Dec and the conservative `dec < 32.375` subset.",
        "- The declination rule is only a prefilter. Actual `ls-dr10-south`, `bands=grz` coverage and valid three-band FITS data remain mandatory.",
        "- The friendly Parquet has no brightness/flux or size/shape measurement. No brightness or size filter was invented.",
        "- No external or advanced morphology catalog was downloaded or used.",
        "- Morphology fractions are retained as analysis and stratification metadata only; they are never model inputs.",
        "- Conditional morphology nulls are preserved as empty CSV fields. They are decision-tree gating, not zeros.",
        "",
        "## Deterministic engineering selection",
        "",
        "Rows are assigned to exactly one stratum in priority order: artifact; merger; major disturbance; minor disturbance; featured edge-on; featured spiral; featured strong-bar; other featured; smooth round; smooth in-between; smooth cigar; other smooth. This makes all strata exclusive even when a galaxy has several notable fractions.",
        "",
        "Within each stratum, a deterministic SplitMix64 key is computed from the pinned catalog row index and selection seed. The lowest keys are selected. This is independent of PyArrow batch boundaries and exactly replayable.",
        "",
        "| Exclusive stratum | Eligible southern rows | Engineering quota |",
        "|---|---:|---:|",
    ]
    for stratum in STRATUM_ORDER:
        lines.append(
            f"| {stratum} | {available.get(stratum, 0):,} | {quotas[stratum]:,} |"
        )
    lines.extend(
        [
            "",
            "Artifact-like and disturbed rows are intentionally represented in the engineering sample to develop transparent quality/rejection rules. Membership here is not acceptance into the future clean source library.",
            "",
            "## Later pilot gate",
            "",
            "The separate pilot phase excludes every engineering catalog row and defaults to 2,500 morphology-stratified candidates. It refuses to run unless all of the following are explicit and existing:",
            "",
            "1. the engineering manifest;",
            "2. the completed field-of-view study report;",
            "3. a selected raw cutout size of 128, 192, or 256 pixels;",
            "4. a file documenting the fixed engineering quality/isolation rules.",
            "",
            "The pilot remains a candidate list until actual southern `g,r,z` FITS validation and the fixed source-quality rules are applied. No future-lockbox role is assigned here.",
            "",
            "Template command after those gates pass:",
            "",
            "```bash",
            ".venv/bin/python scripts/prepare_dr10_catalog_samples.py \\",
            "  --phase pilot \\",
            f"  --catalog {catalog} \\",
            f"  --run-dir {run_dir} \\",
            f"  --seed {seed} \\",
            "  --pilot-count 2500 \\",
            f"  --engineering-manifest {run_dir / 'manifests/dr10_engineering_sources.csv'} \\",
            "  --raw-cutout-size <128-or-192-or-256> \\",
            "  --fov-study-report <completed-field-of-view-report> \\",
            "  --engineering-rules-file <fixed-engineering-rules-file>",
            "```",
            "",
            "The command uses exclusive creation and will refuse to replace `manifests/dr10_pilot_source_candidates.csv` if it already exists.",
        ]
    )
    return "\n".join(lines) + "\n"


def read_engineering_rows(path: Path) -> set[int]:
    rows: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"catalog_row_index", "source_id", "selection_phase"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Engineering manifest lacks required columns: {path}")
        for row in reader:
            if row["selection_phase"] != "engineering":
                raise ValueError("Engineering manifest contains a non-engineering row")
            index = int(row["catalog_row_index"])
            if index in rows:
                raise ValueError("Engineering manifest repeats a catalog row")
            rows.add(index)
    if not rows:
        raise ValueError("Engineering manifest is empty")
    return rows


def run_engineering(
    catalog: Path,
    run_dir: Path,
    batch_size: int,
    seed: int,
    count: int,
) -> None:
    manifest_path = run_dir / "manifests/dr10_engineering_sources.csv"
    protocol_path = run_dir / "diagnostics/pilot_selection_protocol.md"
    ensure_absent([manifest_path, protocol_path])
    parquet = pq.ParquetFile(catalog)
    quotas = scaled_quotas(count, ENGINEERING_WEIGHTS)
    ranked, available = select_ranked_rows(
        parquet, batch_size, seed, quotas, excluded_rows=set()
    )
    records = collect_manifest_records(
        parquet, batch_size, ranked, "engineering", seed
    )
    if len(records) != count:
        raise RuntimeError(f"Expected {count} engineering records; got {len(records)}")
    if len({row["source_id"] for row in records}) != count:
        raise RuntimeError("Engineering source IDs are not unique")
    if any(float(row["dec"]) >= SOUTH_DEC_LIMIT_DEG for row in records):
        raise RuntimeError("Engineering selection escaped conservative Dec cut")
    protocol = protocol_markdown(
        catalog, run_dir, seed, count, quotas, available
    )
    exclusive_csv(manifest_path, MANIFEST_FIELDS, records)
    exclusive_text(protocol_path, protocol)
    print(manifest_path)
    print(protocol_path)


def run_pilot(
    catalog: Path,
    run_dir: Path,
    batch_size: int,
    seed: int,
    count: int,
    engineering_manifest: Path,
    raw_cutout_size: int | None,
    fov_report: Path | None,
    rules_file: Path | None,
) -> None:
    if raw_cutout_size is None:
        raise ValueError("--phase pilot requires --raw-cutout-size")
    if fov_report is None:
        raise ValueError("--phase pilot requires --fov-study-report")
    if rules_file is None:
        raise ValueError("--phase pilot requires --engineering-rules-file")
    fov_report = resolve_existing_file(fov_report)
    rules_file = resolve_existing_file(rules_file)
    engineering_manifest = resolve_existing_file(engineering_manifest)
    output = run_dir / "manifests/dr10_pilot_source_candidates.csv"
    ensure_absent([output])
    excluded = read_engineering_rows(engineering_manifest)
    parquet = pq.ParquetFile(catalog)
    quotas = scaled_quotas(count, PILOT_WEIGHTS)
    ranked, _ = select_ranked_rows(parquet, batch_size, seed, quotas, excluded)
    records = collect_manifest_records(
        parquet,
        batch_size,
        ranked,
        "pilot",
        seed,
        raw_cutout_size,
        fov_report,
        rules_file,
    )
    if len(records) != count:
        raise RuntimeError(f"Expected {count} pilot records; got {len(records)}")
    if set(int(row["catalog_row_index"]) for row in records) & excluded:
        raise RuntimeError("Pilot selection overlaps engineering sample")
    exclusive_csv(output, MANIFEST_FIELDS, records)
    print(output)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    catalog = resolve_existing_file(args.catalog)
    run_dir = resolve_run_dir(args.run_dir)
    if args.phase == "audit":
        run_audit(catalog, run_dir, args.batch_size)
    elif args.phase == "engineering":
        run_engineering(
            catalog,
            run_dir,
            args.batch_size,
            args.seed,
            args.engineering_count,
        )
    else:
        engineering_manifest = args.engineering_manifest or (
            run_dir / "manifests/dr10_engineering_sources.csv"
        )
        run_pilot(
            catalog,
            run_dir,
            args.batch_size,
            args.seed,
            args.pilot_count,
            engineering_manifest,
            args.raw_cutout_size,
            args.fov_study_report,
            args.engineering_rules_file,
        )


if __name__ == "__main__":
    main()
