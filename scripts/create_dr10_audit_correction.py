#!/usr/bin/env python3
"""Create an append-only correction for the DR10 artifact-candidate table.

The first pilot audit correctly identified artifact candidates but omitted the
``catalog_row_index`` value from those rows.  This utility joins that immutable
table to the immutable FITS-quality table by path, validates identity fields,
and writes a separately named corrected table plus a provenance report.  It
never replaces either input or an existing output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


CORRECTION_VERSION = "dr10_artifact_catalog_row_index_correction_v1"
IDENTITY_FIELDS = ("filename", "source_id", "group_id")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def corrected_rows(
    artifact_rows: Sequence[dict[str, str]],
    quality_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    quality_by_path: dict[str, dict[str, str]] = {}
    for row in quality_rows:
        path = row.get("path", "")
        if not path or path in quality_by_path:
            raise ValueError(f"FITS-quality path is blank or duplicated: {path!r}")
        quality_by_path[path] = row

    corrected: list[dict[str, str]] = []
    seen: set[str] = set()
    for artifact in artifact_rows:
        path = artifact.get("path", "")
        if not path or path in seen:
            raise ValueError(f"Artifact path is blank or duplicated: {path!r}")
        seen.add(path)
        if path not in quality_by_path:
            raise ValueError(f"Artifact path is absent from FITS quality: {path}")
        quality = quality_by_path[path]
        for field in IDENTITY_FIELDS:
            if artifact.get(field, "") != quality.get(field, ""):
                raise ValueError(f"Identity mismatch for {field} at {path}")
        if artifact.get("catalog_row_index", "").strip():
            raise ValueError(
                "Correction refuses a nonblank original catalog_row_index; "
                f"unexpected input at {path}"
            )
        catalog_row_index = quality.get("catalog_row_index", "").strip()
        if not catalog_row_index:
            raise ValueError(f"Quality row lacks catalog_row_index at {path}")
        corrected.append({**artifact, "catalog_row_index": catalog_row_index})
    return corrected


def write_csv_exclusive(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-table", required=True, type=Path)
    parser.add_argument("--fits-quality", required=True, type=Path)
    parser.add_argument("--output-table", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_path = args.artifact_table.expanduser().resolve()
    quality_path = args.fits_quality.expanduser().resolve()
    output_path = args.output_table.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Correction outputs already exist; refusing overwrite")

    artifact_fields, artifacts = read_csv(artifact_path)
    _quality_fields, quality = read_csv(quality_path)
    if "catalog_row_index" not in artifact_fields:
        raise ValueError("Artifact table has no catalog_row_index field")
    corrected = corrected_rows(artifacts, quality)

    input_hashes_before = {
        "artifact_table": sha256_file(artifact_path),
        "fits_quality": sha256_file(quality_path),
    }
    write_csv_exclusive(output_path, artifact_fields, corrected)
    input_hashes_after = {
        "artifact_table": sha256_file(artifact_path),
        "fits_quality": sha256_file(quality_path),
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("An input changed while the correction was being written")

    report = {
        "correction_version": CORRECTION_VERSION,
        "original_artifact_table": str(artifact_path),
        "original_artifact_table_sha256": input_hashes_before["artifact_table"],
        "fits_quality_table": str(quality_path),
        "fits_quality_table_sha256": input_hashes_before["fits_quality"],
        "corrected_artifact_table": str(output_path),
        "corrected_artifact_table_sha256": sha256_file(output_path),
        "corrected_row_count": len(corrected),
        "correction": "catalog_row_index copied by exact path after identity validation",
        "original_table_preserved": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
