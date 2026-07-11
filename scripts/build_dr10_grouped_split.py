#!/usr/bin/env python3
"""Build a deterministic, duplicate-safe five-way DR10 source split.

Only sources classified ``accepted_clean_source`` enter the split.  Connected
components are formed from exact stable identifiers, exact parsed coordinates,
exact decoded-pixel hashes, and explicitly high-confidence duplicate evidence.
No angular-radius or visual-similarity merge is performed.  All outputs are
created exclusively under a new ``data/manifests/dr10_grouped_source_split_*``
directory and integrity failures abort closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from astropy.io import fits


SPLIT_VERSION = "dr10_grouped_source_split_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OUTPUT_ROOT = (PROJECT_ROOT / "data/manifests").resolve()
ROLES = (
    "train",
    "validation",
    "calibration",
    "development_test",
    "future_lockbox",
)
DEFAULT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.08,
    "development_test": 0.07,
    "future_lockbox": 0.05,
}
NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a"}
MANDATORY_STABLE_ID_COLUMNS = ("source_id", "dr8_id")
FLUX_LIKE_BRIGHTNESS_COLUMNS = ("central_flux_r", "flux_r")


class UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}
        self.rank = {key: 0 for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


@dataclass(frozen=True)
class EvidenceBucket:
    evidence_type: str
    evidence_key: str
    member_keys: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_pixel_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def current_fits_pixel_hash(path: Path) -> str:
    with fits.open(path, mode="readonly", memmap=False, checksum=True) as hdul:
        hdul.verify("exception")
        image_hdus = [hdu for hdu in hdul if hdu.data is not None]
        if len(image_hdus) == 1 and np.asarray(image_hdus[0].data).ndim == 3:
            cube = np.asarray(image_hdus[0].data)
        elif len(image_hdus) == 3 and all(
            np.asarray(hdu.data).ndim == 2 for hdu in image_hdus
        ):
            cube = np.stack([np.asarray(hdu.data) for hdu in image_hdus], axis=0)
        else:
            shapes = [tuple(np.asarray(hdu.data).shape) for hdu in image_hdus]
            raise ValueError(f"unsupported audited FITS layout during split: {shapes}")
        if cube.ndim != 3 or cube.shape[0] != 3:
            raise ValueError(
                f"audited FITS no longer has canonical band-first shape: {cube.shape}"
            )
        return semantic_pixel_hash(cube)


def write_csv_exclusive(
    path: Path, rows: Sequence[dict[str, Any]], preferred: Sequence[str]
) -> None:
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen and not key.startswith("_"):
                fields.append(key)
                seen.add(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key, ""))
                    for key in fields
                }
            )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def normalized_token(value: Any) -> str:
    token = str(value).strip()
    return "" if token.lower() in NULL_TOKENS else token


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "high"}


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def record_key(row: dict[str, str], position: int) -> str:
    catalog_index = normalized_token(row.get("catalog_row_index", ""))
    if catalog_index:
        return f"catalog_row:{catalog_index}"
    path = normalized_token(row.get("path") or row.get("fits_path") or "")
    if path:
        return f"path:{Path(path).expanduser().resolve()}"
    source_id = normalized_token(row.get("source_id") or row.get("dr8_id") or "")
    if source_id:
        return f"source:{source_id}"
    return f"input_position:{position}"


def _index_rows(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    indices: dict[str, dict[str, list[dict[str, str]]]] = {
        "catalog_row_index": defaultdict(list),
        "source_id": defaultdict(list),
        "dr8_id": defaultdict(list),
        "path": defaultdict(list),
    }
    for row in rows:
        for column, aliases in {
            "catalog_row_index": ("catalog_row_index",),
            "source_id": ("source_id",),
            "dr8_id": ("dr8_id",),
            "path": ("path", "fits_path"),
        }.items():
            raw_values = {
                normalized_token(row.get(alias, ""))
                for alias in aliases
                if normalized_token(row.get(alias, ""))
            }
            values = (
                {str(Path(value).expanduser().resolve()) for value in raw_values}
                if column == "path"
                else raw_values
            )
            if len(values) > 1:
                raise ValueError(
                    f"Conflicting aliases for {column} in audit row: {sorted(values)}"
                )
            if not values:
                continue
            value = next(iter(values))
            indices[column][value].append(row)
    return indices


def unique_audit_match(
    source: dict[str, str],
    indices: dict[str, dict[str, list[dict[str, str]]]],
    table_name: str,
    preferred_path: str = "",
) -> dict[str, str] | None:
    candidates: list[tuple[str, str]] = []
    catalog_index = normalized_token(source.get("catalog_row_index", ""))
    source_id = normalized_token(source.get("source_id", ""))
    dr8_id = normalized_token(source.get("dr8_id", ""))
    source_paths = {
        normalized_token(value)
        for value in (preferred_path, source.get("path", ""), source.get("fits_path", ""))
        if normalized_token(value)
    }
    resolved_paths = {
        str(Path(value).expanduser().resolve()) for value in source_paths
    }
    if len(resolved_paths) > 1:
        raise ValueError(
            f"Conflicting {table_name} source paths: {sorted(source_paths)}"
        )
    if catalog_index:
        candidates.append(("catalog_row_index", catalog_index))
    if source_id:
        candidates.append(("source_id", source_id))
    if dr8_id:
        candidates.append(("dr8_id", dr8_id))
    if resolved_paths:
        candidates.append(("path", next(iter(resolved_paths))))
    matched_rows: list[tuple[str, str, dict[str, str]]] = []
    for column, value in candidates:
        matches = indices[column].get(value, [])
        if len(matches) == 1:
            matched_rows.append((column, value, matches[0]))
            continue
        if len(matches) > 1:
            canonical = {json.dumps(row, sort_keys=True) for row in matches}
            if len(canonical) == 1:
                matched_rows.append((column, value, matches[0]))
                continue
            raise ValueError(
                f"Ambiguous {table_name} join for {column}={value!r}: {len(matches)} rows"
            )
    if not matched_rows:
        return None

    canonical_matches = {
        json.dumps(row, sort_keys=True) for _, _, row in matched_rows
    }
    if len(canonical_matches) != 1:
        evidence = ", ".join(f"{column}={value!r}" for column, value, _ in matched_rows)
        raise ValueError(
            f"Conflicting {table_name} join keys resolve to different rows: {evidence}"
        )
    matched = matched_rows[0][2]

    # A unique match on one key is not sufficient: every identifier populated
    # in both records must agree. This catches stale or accidentally reordered
    # audit tables before any accepted-source decision is trusted.
    row_values = {
        "catalog_row_index": normalized_token(matched.get("catalog_row_index", "")),
        "source_id": normalized_token(matched.get("source_id", "")),
        "dr8_id": normalized_token(matched.get("dr8_id", "")),
        "path": normalized_token(matched.get("path") or matched.get("fits_path") or ""),
    }
    source_values = dict(candidates)
    for column in ("catalog_row_index", "source_id", "dr8_id", "path"):
        left = source_values.get(column, "")
        right = row_values[column]
        if not left or not right:
            continue
        if column == "path":
            left = str(Path(left).expanduser().resolve())
            right = str(Path(right).expanduser().resolve())
        if left != right:
            raise ValueError(
                f"Conflicting {table_name} identifiers for matched row: "
                f"{column} source={left!r}, audit={right!r}"
            )
    return matched


def join_accepted_sources(
    sources: Sequence[dict[str, str]],
    decisions: Sequence[dict[str, str]],
    quality: Sequence[dict[str, str]],
    isolation: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    decision_index = _index_rows(decisions)
    quality_index = _index_rows(quality)
    isolation_index = _index_rows(isolation)
    accepted: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    used_keys: set[str] = set()
    for position, source in enumerate(sources):
        key = record_key(source, position)
        if key in used_keys:
            raise ValueError(f"Non-unique source record key: {key}")
        used_keys.add(key)
        decision = unique_audit_match(source, decision_index, "quality decision")
        if decision is None:
            excluded["missing_quality_decision"] += 1
            continue
        status = normalized_token(decision.get("decision", ""))
        if status != "accepted_clean_source":
            excluded[status or "empty_quality_decision"] += 1
            continue
        decision_path = normalized_token(decision.get("path", ""))
        quality_row = unique_audit_match(
            source, quality_index, "FITS quality", preferred_path=decision_path
        )
        isolation_row = unique_audit_match(
            source, isolation_index, "source isolation", preferred_path=decision_path
        )
        if quality_row is None:
            raise ValueError(f"Accepted source {key} has no FITS quality row")
        if isolation_row is None:
            raise ValueError(f"Accepted source {key} has no isolation row")
        pixel_hash = normalized_token(quality_row.get("pixel_hash", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{64}", pixel_hash):
            raise ValueError(f"Accepted source {key} lacks a valid exact pixel hash")
        row: dict[str, Any] = dict(source)
        row.update(
            _record_key=key,
            fits_path=decision_path or normalized_token(quality_row.get("path", "")),
            source_quality_decision=status,
            source_quality_rule_version=decision.get("rule_version", ""),
            pixel_hash=pixel_hash.lower(),
            audited_file_sha256=str(quality_row.get("file_sha256", "")).strip().lower(),
            audit_exact_duplicate_group_id=quality_row.get("exact_duplicate_group_id", ""),
            central_flux_g=isolation_row.get("central_flux_g", ""),
            central_flux_r=isolation_row.get("central_flux_r", ""),
            central_flux_z=isolation_row.get("central_flux_z", ""),
        )
        accepted.append(row)
    return accepted, excluded


def revalidate_accepted_fits(records: Sequence[dict[str, Any]]) -> int:
    """Rehash every accepted FITS immediately before lockbox allocation."""
    seen_paths: set[str] = set()
    for row in records:
        path_text = normalized_token(row.get("fits_path", ""))
        if not path_text:
            raise ValueError(f"Accepted source lacks fits_path: {row.get('_record_key')}")
        path = Path(path_text).expanduser().resolve()
        if "lockbox" in str(path).lower():
            raise ValueError(f"Refusing lockbox-like path before split allocation: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Accepted FITS is missing before allocation: {path}")
        path_key = str(path)
        if path_key in seen_paths:
            raise ValueError(f"Accepted FITS path is repeated before grouping: {path}")
        seen_paths.add(path_key)

        expected_file_hash = normalized_token(row.get("audited_file_sha256", "")).lower()
        expected_pixel_hash = normalized_token(row.get("pixel_hash", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_file_hash):
            raise ValueError(f"Accepted source lacks audited full-file SHA-256: {path}")
        if file_sha256(path) != expected_file_hash:
            raise ValueError(f"Accepted FITS full-file SHA-256 changed after audit: {path}")
        if current_fits_pixel_hash(path) != expected_pixel_hash:
            raise ValueError(f"Accepted FITS decoded-pixel hash changed after audit: {path}")
        if file_sha256(path) != expected_file_hash:
            raise ValueError(f"Accepted FITS changed during preallocation revalidation: {path}")
        row["preallocation_file_sha256"] = expected_file_hash
        row["preallocation_pixel_hash"] = expected_pixel_hash
        row["preallocation_hash_revalidation_pass"] = 1
    return len(records)


def exact_coordinate_key(row: dict[str, Any]) -> str:
    ra, dec = safe_float(row.get("ra")), safe_float(row.get("dec"))
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return ""
    if not (0.0 <= ra <= 360.0 and -90.0 <= dec <= 90.0):
        raise ValueError(f"Out-of-range coordinate for {row['_record_key']}: {ra}, {dec}")
    return f"{ra.hex()}|{dec.hex()}"


def make_bucket(
    evidence_type: str, evidence_key: str, member_keys: Iterable[str]
) -> EvidenceBucket | None:
    members = tuple(sorted(set(member_keys)))
    if len(members) < 2:
        return None
    return EvidenceBucket(evidence_type, evidence_key, members)


def construct_components(
    records: list[dict[str, Any]],
    stable_id_columns: Sequence[str],
    duplicate_rows: Sequence[dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], list[EvidenceBucket]]:
    keys = [str(row["_record_key"]) for row in records]
    union_find = UnionFind(keys)
    buckets: list[EvidenceBucket] = []

    def add_grouped(evidence_type: str, values: dict[str, list[str]]) -> None:
        for value, members in sorted(values.items()):
            bucket = make_bucket(evidence_type, value, members)
            if bucket is None:
                continue
            buckets.append(bucket)
            first = bucket.member_keys[0]
            for member in bucket.member_keys[1:]:
                union_find.union(first, member)

    effective_stable_columns = list(
        dict.fromkeys((*MANDATORY_STABLE_ID_COLUMNS, *stable_id_columns))
    )
    for column in effective_stable_columns:
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in records:
            value = normalized_token(row.get(column, ""))
            if value:
                grouped[value].append(str(row["_record_key"]))
        add_grouped(f"stable_id:{column}", grouped)

    if all(any(column in row for row in records) for column in ("brickid", "objid")):
        composites: dict[str, list[str]] = defaultdict(list)
        for row in records:
            brick = normalized_token(row.get("brickid", ""))
            obj = normalized_token(row.get("objid", ""))
            if brick and obj:
                composites[f"{brick}|{obj}"].append(str(row["_record_key"]))
        add_grouped("stable_id:brickid+objid", composites)

    coordinates: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, list[str]] = defaultdict(list)
    source_ids: dict[str, list[str]] = defaultdict(list)
    for row in records:
        coordinate = exact_coordinate_key(row)
        if coordinate:
            coordinates[coordinate].append(str(row["_record_key"]))
        hashes[str(row["pixel_hash"])].append(str(row["_record_key"]))
        source_id = normalized_token(row.get("source_id") or row.get("dr8_id") or "")
        if source_id:
            source_ids[source_id].append(str(row["_record_key"]))
    add_grouped("exact_coordinate", coordinates)
    add_grouped("exact_pixel_hash", hashes)

    for index, duplicate in enumerate(duplicate_rows):
        if not truthy(duplicate.get("high_confidence") or duplicate.get("is_high_confidence")):
            continue
        left_id = normalized_token(
            duplicate.get("source_id_a") or duplicate.get("left_source_id") or ""
        )
        right_id = normalized_token(
            duplicate.get("source_id_b") or duplicate.get("right_source_id") or ""
        )
        if not left_id or not right_id:
            raise ValueError(
                "High-confidence duplicate rows require source_id_a/source_id_b"
            )
        left_members, right_members = source_ids.get(left_id, []), source_ids.get(right_id, [])
        if not left_members or not right_members:
            raise ValueError(
                f"High-confidence duplicate references unknown source: {left_id}, {right_id}"
            )
        members = tuple(sorted(set(left_members + right_members)))
        evidence_label = normalized_token(duplicate.get("evidence_type", "")) or "explicit"
        bucket = EvidenceBucket(
            f"high_confidence_duplicate:{evidence_label}",
            f"row_{index:07d}:{left_id}|{right_id}",
            members,
        )
        buckets.append(bucket)
        for member in members[1:]:
            union_find.union(members[0], member)

    component_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        component_members[union_find.find(str(row["_record_key"]))].append(row)
    groups: dict[str, list[dict[str, Any]]] = {}
    for members in component_members.values():
        member_keys = sorted(str(row["_record_key"]) for row in members)
        digest = hashlib.sha256("\n".join(member_keys).encode("utf-8")).hexdigest()[:16]
        group_id = f"dr10_group_{digest}"
        for row in members:
            row["group_id"] = group_id
        groups[group_id] = sorted(members, key=lambda row: str(row["_record_key"]))
    return dict(sorted(groups.items())), buckets


def infer_morphology_columns(
    records: Sequence[dict[str, Any]], explicit: Sequence[str]
) -> list[str]:
    if explicit:
        missing = [column for column in explicit if not any(column in row for row in records)]
        if missing:
            raise ValueError(f"Requested morphology columns are absent: {missing}")
        return list(explicit)
    available = sorted({key for row in records for key in row})
    top_level = [
        key
        for key in available
        if ("smooth-or-featured" in key.lower() or "smooth_or_featured" in key.lower())
        and ("fraction" in key.lower() or "vote" in key.lower())
    ]
    if top_level:
        return top_level
    return [
        key
        for key in available
        if "morph" in key.lower() and ("fraction" in key.lower() or "vote" in key.lower())
    ][:8]


def annotate_balance_strata(
    records: list[dict[str, Any]], morphology_columns: Sequence[str], brightness_bins: int
) -> tuple[list[float], str]:
    coverage = {
        column: sum(np.isfinite(safe_float(row.get(column))) for row in records)
        for column in FLUX_LIKE_BRIGHTNESS_COLUMNS
    }
    brightness_column = max(
        FLUX_LIKE_BRIGHTNESS_COLUMNS,
        key=lambda column: (coverage[column], -FLUX_LIKE_BRIGHTNESS_COLUMNS.index(column)),
    )
    if coverage[brightness_column] == 0:
        brightness_column = "unavailable"
    brightness_values = []
    for row in records:
        value = (
            safe_float(row.get(brightness_column))
            if brightness_column != "unavailable"
            else float("nan")
        )
        row["balance_brightness_value"] = value if np.isfinite(value) else ""
        row["balance_brightness_source"] = brightness_column
        if np.isfinite(value):
            brightness_values.append(value)
        morphology_values = [safe_float(row.get(column)) for column in morphology_columns]
        finite_indices = [index for index, value in enumerate(morphology_values) if np.isfinite(value)]
        if finite_indices:
            dominant = max(finite_indices, key=lambda index: morphology_values[index])
            row["balance_morphology_label"] = morphology_columns[dominant]
        else:
            row["balance_morphology_label"] = "morphology_unavailable"

    if brightness_values and brightness_bins > 1:
        edges = np.unique(
            np.quantile(
                np.asarray(brightness_values, dtype=float),
                np.linspace(0.0, 1.0, brightness_bins + 1)[1:-1],
            )
        )
    else:
        edges = np.array([], dtype=float)
    for row in records:
        value = safe_float(row.get("balance_brightness_value"))
        brightness_label = (
            f"brightness_q{int(np.searchsorted(edges, value, side='right'))}"
            if np.isfinite(value)
            else "brightness_unavailable"
        )
        row["balance_brightness_bin"] = brightness_label
        row["balance_stratum"] = f"{row['balance_morphology_label']}|{brightness_label}"
    return [float(edge) for edge in edges], brightness_column


def deterministic_tie(seed: int, group_id: str, role: str) -> str:
    return hashlib.sha256(f"{seed}|{group_id}|{role}".encode("utf-8")).hexdigest()


def assign_group_roles(
    groups: dict[str, list[dict[str, Any]]],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    total_sources = sum(len(members) for members in groups.values())
    target_counts = {role: fractions[role] * total_sources for role in ROLES}
    total_strata = Counter(
        str(row["balance_stratum"])
        for members in groups.values()
        for row in members
    )
    target_strata = {
        role: {stratum: fractions[role] * count for stratum, count in total_strata.items()}
        for role in ROLES
    }
    current_counts = Counter({role: 0 for role in ROLES})
    current_strata = {role: Counter() for role in ROLES}
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            -len(groups[group_id]),
            deterministic_tie(seed, group_id, "order"),
        ),
    )
    assignment: dict[str, str] = {}
    for group_id in ordered_groups:
        members = groups[group_id]
        group_strata = Counter(str(row["balance_stratum"]) for row in members)
        candidates = []
        for role in ROLES:
            target = max(target_counts[role], 1.0)
            before = ((current_counts[role] - target_counts[role]) / target) ** 2
            after = ((current_counts[role] + len(members) - target_counts[role]) / target) ** 2
            score = after - before
            for stratum, amount in group_strata.items():
                stratum_target = max(target_strata[role][stratum], 1.0)
                before_s = (
                    (current_strata[role][stratum] - target_strata[role][stratum])
                    / stratum_target
                ) ** 2
                after_s = (
                    (
                        current_strata[role][stratum]
                        + amount
                        - target_strata[role][stratum]
                    )
                    / stratum_target
                ) ** 2
                score += 0.35 * (after_s - before_s)
            candidates.append((score, deterministic_tie(seed, group_id, role), role))
        role = min(candidates)[2]
        assignment[group_id] = role
        current_counts[role] += len(members)
        current_strata[role].update(group_strata)

    if len(groups) >= len(ROLES):
        empty_roles = [role for role in ROLES if role not in assignment.values()]
        for empty_role in empty_roles:
            donor_counts = Counter(assignment.values())
            donors = [role for role in ROLES if donor_counts[role] > 1]
            if not donors:
                raise RuntimeError("Unable to populate all five roles without splitting a group")
            donor = max(
                donors,
                key=lambda role: current_counts[role] - target_counts[role],
            )
            movable = [group_id for group_id, role in assignment.items() if role == donor]
            chosen = min(
                movable,
                key=lambda group_id: (
                    len(groups[group_id]), deterministic_tie(seed, group_id, empty_role)
                ),
            )
            assignment[chosen] = empty_role
            current_counts[donor] -= len(groups[chosen])
            current_counts[empty_role] += len(groups[chosen])
    return assignment


def balance_diagnostics(
    manifest_rows: Sequence[dict[str, Any]], fractions: dict[str, float]
) -> dict[str, Any]:
    """Report achieved source and stratum balance without hiding group constraints."""
    total = len(manifest_rows)
    role_counts = Counter(str(row["role"]) for row in manifest_rows)
    source_balance: dict[str, dict[str, float | int]] = {}
    for role in ROLES:
        actual = int(role_counts[role])
        target = float(fractions[role] * total)
        actual_fraction = float(actual / total) if total else float("nan")
        source_balance[role] = {
            "target_fraction": float(fractions[role]),
            "actual_fraction": actual_fraction,
            "absolute_fraction_deviation": abs(actual_fraction - fractions[role]),
            "target_source_count": target,
            "actual_source_count": actual,
            "source_count_deviation": actual - target,
        }

    stratum_totals = Counter(str(row["balance_stratum"]) for row in manifest_rows)
    stratum_balance: dict[str, dict[str, dict[str, float | int]]] = {}
    maximum_stratum_deviation = 0.0
    for stratum, stratum_total in sorted(stratum_totals.items()):
        role_details: dict[str, dict[str, float | int]] = {}
        for role in ROLES:
            actual = sum(
                1
                for row in manifest_rows
                if str(row["balance_stratum"]) == stratum and row["role"] == role
            )
            actual_fraction = actual / stratum_total
            deviation = abs(actual_fraction - fractions[role])
            maximum_stratum_deviation = max(maximum_stratum_deviation, deviation)
            role_details[role] = {
                "target_fraction": float(fractions[role]),
                "actual_fraction": float(actual_fraction),
                "absolute_fraction_deviation": float(deviation),
                "target_source_count": float(fractions[role] * stratum_total),
                "actual_source_count": int(actual),
            }
        stratum_balance[stratum] = role_details

    return {
        "source_role_balance": source_balance,
        "stratum_role_balance": stratum_balance,
        "max_absolute_source_fraction_deviation": max(
            detail["absolute_fraction_deviation"]
            for detail in source_balance.values()
        ),
        "max_absolute_stratum_fraction_deviation": float(maximum_stratum_deviation),
        "balance_is_diagnostic_not_a_leakage_gate": True,
    }


def integrity_tables(
    manifest_rows: Sequence[dict[str, Any]], evidence: Sequence[EvidenceBucket]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key = {str(row["_record_key"]): row for row in manifest_rows}
    if len(by_key) != len(manifest_rows):
        raise ValueError("Duplicate source record keys in split manifest")
    for row in manifest_rows:
        by_group[str(row["group_id"])].append(row)
    group_checks = []
    for group_id, members in sorted(by_group.items()):
        roles = sorted({str(row["role"]) for row in members})
        group_checks.append(
            {
                "group_id": group_id,
                "member_count": len(members),
                "role_count": len(roles),
                "roles": ";".join(roles),
                "member_keys_json": json.dumps(
                    sorted(str(row["_record_key"]) for row in members)
                ),
                "pass": int(len(roles) == 1),
            }
        )
    duplicate_checks = []
    for bucket in evidence:
        missing = [key for key in bucket.member_keys if key not in by_key]
        if missing:
            raise ValueError(f"Evidence references missing split members: {missing}")
        roles = sorted({str(by_key[key]["role"]) for key in bucket.member_keys})
        groups = sorted({str(by_key[key]["group_id"]) for key in bucket.member_keys})
        passed = len(roles) == 1 and len(groups) == 1
        duplicate_checks.append(
            {
                "evidence_type": bucket.evidence_type,
                "evidence_key": bucket.evidence_key,
                "member_count": len(bucket.member_keys),
                "group_count": len(groups),
                "role_count": len(roles),
                "groups": ";".join(groups),
                "roles": ";".join(roles),
                "member_keys_json": json.dumps(bucket.member_keys),
                "pass": int(passed),
            }
        )
    return group_checks, duplicate_checks


def fail_closed_integrity(
    manifest_rows: Sequence[dict[str, Any]],
    group_checks: Sequence[dict[str, Any]],
    duplicate_checks: Sequence[dict[str, Any]],
) -> None:
    invalid_roles = sorted({str(row.get("role")) for row in manifest_rows} - set(ROLES))
    if invalid_roles:
        raise ValueError(f"Unknown split roles: {invalid_roles}")
    if any(not bool(int(row["pass"])) for row in group_checks):
        raise RuntimeError("Cross-role group leakage detected; refusing outputs")
    if any(not bool(int(row["pass"])) for row in duplicate_checks):
        raise RuntimeError("Cross-split duplicate leakage detected; refusing outputs")
    if len(manifest_rows) >= len(ROLES):
        missing_roles = set(ROLES) - {str(row["role"]) for row in manifest_rows}
        if missing_roles:
            raise RuntimeError(f"Five-way protocol has empty roles: {sorted(missing_roles)}")


def parse_fractions(value: str) -> dict[str, float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != len(ROLES) or any(part <= 0 for part in parts):
        raise ValueError("--fractions must contain five positive comma-separated values")
    total = sum(parts)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split fractions must sum to 1; got {total}")
    return dict(zip(ROLES, parts, strict=True))


def safe_stamp(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("stamp must be one safe filename component")
    return value


def lockbox_policy(seed: int, version: str) -> str:
    return f"""# Future-lockbox policy

Created: `{utc_now()}`  
Split version: `{version}`  
Allocation seed: `{seed}`

The `future_lockbox` role is allocated once at the grouped-source level. After
this allocation, its FITS pixels and generated blends must not be opened,
rendered, visually inspected, debugged against, used for normalization,
threshold tuning, model selection, early stopping, uncertainty calibration, or
development reporting. Only a future, explicitly authorized final evaluation
may access it.

Stable identifiers, exact parsed coordinates, exact decoded-pixel hashes, and
explicitly high-confidence duplicate evidence are unioned before role
assignment. Uncertain visual similarities are not automatically merged. Every
group has exactly one role; the target and contaminant for any synthetic blend
must both be drawn from that same role. Calibration is separate from validation
and development testing. Any cross-role group or duplicate evidence causes a
fail-closed abort. Morphology and measured brightness are used only to improve
split balance and for downstream analysis; morphology is never a model input.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--quality-decisions", type=Path, required=True)
    parser.add_argument("--fits-quality", type=Path, required=True)
    parser.add_argument("--isolation-metrics", type=Path, required=True)
    parser.add_argument("--duplicate-evidence", type=Path)
    parser.add_argument("--output-root", type=Path, default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=731_029)
    parser.add_argument(
        "--fractions",
        default=",".join(str(DEFAULT_FRACTIONS[role]) for role in ROLES),
        help="train,validation,calibration,development_test,future_lockbox",
    )
    parser.add_argument(
        "--stable-id-columns", default="source_id,dr8_id,ls_id,iauname"
    )
    parser.add_argument("--morphology-columns", default="")
    parser.add_argument("--brightness-bins", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fractions = parse_fractions(args.fractions)
    if args.brightness_bins < 1 or args.brightness_bins > 20:
        raise SystemExit("--brightness-bins must be in [1, 20]")
    requested_stable_columns = [
        item.strip() for item in args.stable_id_columns.split(",") if item.strip()
    ]
    stable_columns = list(
        dict.fromkeys((*MANDATORY_STABLE_ID_COLUMNS, *requested_stable_columns))
    )
    morphology_explicit = [item.strip() for item in args.morphology_columns.split(",") if item.strip()]

    sources = read_csv(args.source_manifest)
    decisions = read_csv(args.quality_decisions)
    quality = read_csv(args.fits_quality)
    isolation = read_csv(args.isolation_metrics)
    duplicate_evidence = read_csv(args.duplicate_evidence) if args.duplicate_evidence else []
    accepted, excluded = join_accepted_sources(sources, decisions, quality, isolation)
    if len(accepted) < len(ROLES):
        raise SystemExit("Fewer than five accepted sources; refusing a five-way split")
    preallocation_revalidated_count = revalidate_accepted_fits(accepted)

    morphology_columns = infer_morphology_columns(accepted, morphology_explicit)
    brightness_edges, brightness_definition = annotate_balance_strata(
        accepted, morphology_columns, args.brightness_bins
    )
    groups, evidence = construct_components(accepted, stable_columns, duplicate_evidence)
    if len(groups) < len(ROLES):
        raise SystemExit("Fewer than five duplicate-safe groups; refusing a five-way split")
    assignment = assign_group_roles(groups, fractions, args.seed)
    manifest_rows: list[dict[str, Any]] = []
    for group_id, members in groups.items():
        for row in members:
            row["role"] = assignment[group_id]
            row["split_seed"] = args.seed
            row["split_version"] = SPLIT_VERSION
            row["group_member_count"] = len(members)
            manifest_rows.append(row)
    manifest_rows.sort(key=lambda row: (ROLES.index(str(row["role"])), str(row["group_id"]), str(row["_record_key"])))
    group_checks, duplicate_checks = integrity_tables(manifest_rows, evidence)
    fail_closed_integrity(manifest_rows, group_checks, duplicate_checks)

    output_root = args.output_root.expanduser().resolve()
    if output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError(
            f"Split outputs must remain in {EXPECTED_OUTPUT_ROOT}; got {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"dr10_grouped_source_split_{safe_stamp(args.stamp)}"
    output_dir.mkdir(exist_ok=False)
    output_files = {
        "manifest": output_dir / "source_split_manifest.csv",
        "summary": output_dir / "split_summary.json",
        "groups": output_dir / "group_integrity_check.csv",
        "duplicates": output_dir / "cross_split_duplicate_check.csv",
        "policy": output_dir / "lockbox_policy.md",
    }
    role_counts = Counter(str(row["role"]) for row in manifest_rows)
    group_role_counts = Counter(assignment.values())
    achieved_balance = balance_diagnostics(manifest_rows, fractions)
    morphology_counts = {
        role: dict(
            sorted(
                Counter(
                    str(row["balance_morphology_label"])
                    for row in manifest_rows
                    if row["role"] == role
                ).items()
            )
        )
        for role in ROLES
    }
    brightness_counts = {
        role: dict(
            sorted(
                Counter(
                    str(row["balance_brightness_bin"])
                    for row in manifest_rows
                    if row["role"] == role
                ).items()
            )
        )
        for role in ROLES
    }
    summary = {
        "created_utc": utc_now(),
        "split_version": SPLIT_VERSION,
        "seed": args.seed,
        "fractions": fractions,
        "candidate_source_count": len(sources),
        "accepted_source_count": len(manifest_rows),
        "excluded_counts": dict(sorted(excluded.items())),
        "group_count": len(groups),
        "source_role_counts": dict(role_counts),
        "group_role_counts": dict(group_role_counts),
        "stable_id_columns": stable_columns,
        "mandatory_stable_id_columns": list(MANDATORY_STABLE_ID_COLUMNS),
        "morphology_balance_columns": morphology_columns,
        "brightness_balance_definition": brightness_definition,
        "brightness_quantile_edges": brightness_edges,
        "morphology_role_counts": morphology_counts,
        "brightness_role_counts": brightness_counts,
        "balance_diagnostics": achieved_balance,
        "evidence_bucket_count": len(evidence),
        "group_integrity_pass": all(bool(row["pass"]) for row in group_checks),
        "cross_split_duplicate_check_pass": all(
            bool(row["pass"]) for row in duplicate_checks
        ),
        "preallocation_file_and_pixel_hash_revalidation_pass": True,
        "preallocation_revalidated_source_count": preallocation_revalidated_count,
        "lockbox_pixels_accessed": False,
        "lockbox_visualized": False,
    }
    preferred_manifest = [
        "source_id",
        "catalog_row_index",
        "ra",
        "dec",
        "fits_path",
        "pixel_hash",
        "audited_file_sha256",
        "preallocation_file_sha256",
        "preallocation_pixel_hash",
        "preallocation_hash_revalidation_pass",
        "group_id",
        "group_member_count",
        "role",
        "source_quality_decision",
        "balance_morphology_label",
        "balance_brightness_value",
        "balance_brightness_source",
        "balance_brightness_bin",
        "balance_stratum",
        "split_seed",
        "split_version",
    ]
    write_csv_exclusive(output_files["manifest"], manifest_rows, preferred_manifest)
    write_json_exclusive(output_files["summary"], summary)
    write_csv_exclusive(
        output_files["groups"],
        group_checks,
        ["group_id", "member_count", "role_count", "roles", "pass", "member_keys_json"],
    )
    write_csv_exclusive(
        output_files["duplicates"],
        duplicate_checks,
        [
            "evidence_type",
            "evidence_key",
            "member_count",
            "group_count",
            "role_count",
            "groups",
            "roles",
            "pass",
            "member_keys_json",
        ],
    )
    write_text_exclusive(output_files["policy"], lockbox_policy(args.seed, SPLIT_VERSION))
    print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
