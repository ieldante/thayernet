#!/usr/bin/env python3
"""Show why the earlier provisional final pool is not final-test independent.

This is a metadata-only, append-only audit.  It never edits either manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FINAL_MANIFEST_NAMES = (
    "normal_final_test.csv",
    "hard_stress_final_test.csv",
    "compact_bright_final_test.csv",
    "high_core_obstruction_final_test.csv",
    "halo_artifact_stress_final_test.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run-dir", type=Path, required=True)
    parser.add_argument("--provisional-manifest-dir", type=Path, required=True)
    parser.add_argument("--grouped-source-split", type=Path, required=True)
    parser.add_argument("--grouped-blend-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_run = args.audit_run_dir.resolve()
    provisional_dir = args.provisional_manifest_dir.resolve()
    source_split_path = args.grouped_source_split.resolve()
    blend_dir = args.grouped_blend_dir.resolve()

    output_table = audit_run / "tables" / "provisional_final_pool_grouped_overlap.csv"
    summary_table = audit_run / "tables" / "provisional_final_pool_overlap_summary.csv"
    report_path = audit_run / "diagnostics" / "provisional_final_pool_superseded.md"
    provenance_path = audit_run / "logs" / "provisional_final_pool_overlap_provenance.json"
    for output in (output_table, summary_table, report_path, provenance_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite audit output: {output}")

    split = read_csv(source_split_path)
    required_split_columns = {"source_index", "split", "group_id"}
    if not required_split_columns.issubset(split.columns):
        raise ValueError("Grouped source split is missing required columns")
    if split["source_index"].duplicated().any():
        raise ValueError("Grouped source split contains duplicate source indices")
    split_by_source = split.set_index("source_index")

    source_roles: dict[int, set[str]] = {}
    manifest_hashes: dict[str, str] = {}
    for name in FINAL_MANIFEST_NAMES:
        path = provisional_dir / name
        frame = read_csv(path)
        manifest_hashes[name] = sha256(path)
        for role, column in (
            ("target", "target_source_index"),
            ("contaminant", "contaminant_source_index"),
        ):
            for source_index in frame[column].astype(int).unique():
                source_roles.setdefault(int(source_index), set()).add(role)

    pool = sorted(source_roles)
    missing = sorted(set(pool).difference(split_by_source.index.astype(int)))
    if missing:
        raise ValueError(f"Provisional sources absent from grouped split: {missing[:10]}")

    use_counts: dict[tuple[int, str, str], int] = {}
    grouped_blend_hashes: dict[str, str] = {}
    for split_name, filename in (("train", "train_blends.csv"), ("validation", "val_blends.csv")):
        path = blend_dir / filename
        frame = read_csv(path)
        grouped_blend_hashes[filename] = sha256(path)
        for role, column in (
            ("target", "target_source_index"),
            ("contaminant", "contaminant_source_index"),
        ):
            counts = frame[column].astype(int).value_counts()
            for source_index, count in counts.items():
                use_counts[(int(source_index), split_name, role)] = int(count)

    rows: list[dict[str, object]] = []
    for source_index in pool:
        grouped = split_by_source.loc[source_index]
        train_target = use_counts.get((source_index, "train", "target"), 0)
        train_contaminant = use_counts.get((source_index, "train", "contaminant"), 0)
        val_target = use_counts.get((source_index, "validation", "target"), 0)
        val_contaminant = use_counts.get((source_index, "validation", "contaminant"), 0)
        rows.append(
            {
                "source_index": source_index,
                "group_id": grouped["group_id"],
                "grouped_split": grouped["split"],
                "provisional_roles": "+".join(sorted(source_roles[source_index])),
                "grouped_train_target_uses": train_target,
                "grouped_train_contaminant_uses": train_contaminant,
                "grouped_validation_target_uses": val_target,
                "grouped_validation_contaminant_uses": val_contaminant,
                "used_in_grouped_train": (train_target + train_contaminant) > 0,
                "used_in_grouped_validation": (val_target + val_contaminant) > 0,
                "used_in_grouped_train_or_validation": (
                    train_target + train_contaminant + val_target + val_contaminant
                )
                > 0,
                "final_eligible_after_grouped_retrain": False,
            }
        )
    detail = pd.DataFrame(rows)

    split_counts = detail["grouped_split"].value_counts().to_dict()
    n_train_used = int(detail["used_in_grouped_train"].sum())
    n_val_used = int(detail["used_in_grouped_validation"].sum())
    n_train_or_val = int(detail["used_in_grouped_train_or_validation"].sum())
    summary = pd.DataFrame(
        [
            {"measure": "provisional_unique_sources", "count": len(detail), "fraction": 1.0},
            {
                "measure": "mapped_to_grouped_train",
                "count": int(split_counts.get("train", 0)),
                "fraction": float(split_counts.get("train", 0)) / len(detail),
            },
            {
                "measure": "mapped_to_grouped_validation",
                "count": int(split_counts.get("validation", 0)),
                "fraction": float(split_counts.get("validation", 0)) / len(detail),
            },
            {
                "measure": "mapped_to_grouped_test",
                "count": int(split_counts.get("test", 0)),
                "fraction": float(split_counts.get("test", 0)) / len(detail),
            },
            {
                "measure": "actually_used_in_grouped_train_blends",
                "count": n_train_used,
                "fraction": n_train_used / len(detail),
            },
            {
                "measure": "actually_used_in_grouped_validation_blends",
                "count": n_val_used,
                "fraction": n_val_used / len(detail),
            },
            {
                "measure": "actually_used_in_grouped_train_or_validation_blends",
                "count": n_train_or_val,
                "fraction": n_train_or_val / len(detail),
            },
        ]
    )

    expected = {
        "pool": 1000,
        "train_split": 683,
        "validation_split": 173,
        "test_split": 144,
        "train_used": 499,
        "validation_used": 91,
        "train_or_validation_used": 590,
    }
    observed = {
        "pool": len(detail),
        "train_split": int(split_counts.get("train", 0)),
        "validation_split": int(split_counts.get("validation", 0)),
        "test_split": int(split_counts.get("test", 0)),
        "train_used": n_train_used,
        "validation_used": n_val_used,
        "train_or_validation_used": n_train_or_val,
    }
    if observed != expected:
        raise AssertionError(f"Unexpected overlap counts: {observed} != {expected}")

    detail.to_csv(output_table, index=False)
    summary.to_csv(summary_table, index=False)
    report_path.write_text(
        "# Earlier provisional final pool is superseded\n\n"
        "## Verdict\n\n"
        "The metadata-only pool prepared under the historical row-index split is "
        "**not eligible for final evaluation after the grouped resplit and retrain**. "
        "It remains preserved for provenance and must not be deleted or relabeled as a valid final set.\n\n"
        "## Direct overlap evidence\n\n"
        f"- Provisional unique source pool: `{len(detail)}`.\n"
        f"- Mapping under the new grouped split: `{observed['train_split']}` train, "
        f"`{observed['validation_split']}` validation, `{observed['test_split']}` test.\n"
        f"- Actually used in grouped training blends: `{n_train_used}` sources.\n"
        f"- Actually used in grouped validation blends: `{n_val_used}` sources.\n"
        f"- Used in either grouped training or validation: `{n_train_or_val}` sources (`59.0%`).\n\n"
        "This is a protocol-ordering failure, not evidence of a numerical metric bug. "
        "The grouped development tests remain valid for development comparisons because "
        "their source and group roles are disjoint from grouped training and validation. "
        "They are not a locked final-paper benchmark.\n\n"
        "## Required correction\n\n"
        "After the model, generator, metric code, and analysis protocol are frozen, allocate "
        "a fresh untouched source-group partition and generate new final manifests from it. "
        "Do not inspect examples or tune models on that future pool. The optional second "
        "training seed was not launched in this audit because final-test independence is "
        "the higher-priority unresolved infrastructure blocker.\n"
    )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass_overlap_reproduced_provisional_pool_superseded",
        "provisional_manifest_directory": str(provisional_dir),
        "grouped_source_split": {
            "path": str(source_split_path),
            "sha256": sha256(source_split_path),
        },
        "grouped_blend_directory": str(blend_dir),
        "provisional_manifest_sha256": manifest_hashes,
        "grouped_train_validation_manifest_sha256": grouped_blend_hashes,
        "observed": observed,
        "outputs": [str(output_table), str(summary_table), str(report_path)],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(observed, sort_keys=True))


if __name__ == "__main__":
    main()
