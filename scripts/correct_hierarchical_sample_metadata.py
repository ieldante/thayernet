#!/usr/bin/env python3
"""Append-only correction for NULL query-state CSV parsing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATASETS = ("q_training", "q_validation", "r_training", "r_validation", "natural_calibration", "stratified_calibration")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run = args.run_dir.resolve(); inventory = []
    for dataset in DATASETS:
        manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False)
        sample = pd.read_csv(run / f"features/v2_{dataset}_samples.csv", keep_default_na=False, low_memory=False)
        if manifest.scene_id.tolist() != sample.scene_id.tolist():
            raise RuntimeError(f"Scene alignment failed for {dataset}")
        original_blank = int((sample.query_state == "").sum())
        sample["query_state"] = manifest.query_state.astype(str)
        unique = sample.query_state == "UNIQUE_VALID"
        expected_applicable = unique.astype(int)
        if not np.array_equal(sample.applicable_valid_risk.to_numpy(dtype=int), expected_applicable.to_numpy(dtype=int)):
            raise RuntimeError(f"Applicability mismatch in {dataset}")
        if set(sample.query_state) - {"UNIQUE_VALID", "NULL", "AMBIGUOUS"}:
            raise RuntimeError(f"Invalid query state in {dataset}")
        output = run / f"features/v3_{dataset}_samples.csv"
        write_csv_fresh(output, sample)
        inventory.append({
            "dataset": dataset, "rows": len(sample), "blank_query_states_before": original_blank, "blank_query_states_after": int((sample.query_state == "").sum()),
            "unique_valid": int((sample.query_state == "UNIQUE_VALID").sum()), "null": int((sample.query_state == "NULL").sum()),
            "ambiguous": int((sample.query_state == "AMBIGUOUS").sum()), "applicability_mismatches": 0,
            "supersedes": f"features/v2_{dataset}_samples.csv", "relative_path": str(output.relative_to(REPO)), "sha256": sha256_file(output),
        })
    table = pd.DataFrame(inventory)
    write_csv_fresh(run / "tables/sample_metadata_correction_inventory.csv", table)
    write_json_fresh(run / "logs/sample_metadata_correction_complete.json", {
        "status": "PASS", "cause": "pandas default missing-token parsing of literal NULL during feature-sample assembly",
        "authoritative_manifests_changed": False, "feature_arrays_changed": False, "risk_values_changed": False,
        "corrected_sample_namespace": "v3", "total_blank_query_states_corrected": int(table.blank_query_states_before.sum()),
        "applicability_mismatches": int(table.applicability_mismatches.sum()), "head_training_started_before_correction": False,
        "development_accessed": False, "lockbox_accessed": False,
    })


if __name__ == "__main__":
    main()
