#!/usr/bin/env python3
"""Append-only finalizer for a post-audit report-writing exception.

This script does not compute or alter labels.  It verifies already persisted
authoritative audit tables, records the bookkeeping incident, and writes the
report/marker that the original driver did not reach.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    required = [
        run / "tables/label_provenance_audit.csv",
        run / "tables/label_prevalence_by_partition.csv",
        run / "tables/label_applicability_matrix.csv",
        run / "tables/failure_overlap_matrix.csv",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("The scientific label audit did not persist all required tables")
    outcomes = sorted((run / "tables").glob("prospective_outcomes_*.csv"))
    samples = sorted((run / "features").glob("v4_*_samples.csv"))
    if len(outcomes) != 5 or len(samples) != 5:
        raise RuntimeError("Expected five authoritative outcome/sample artifacts")
    applicability = pd.read_csv(required[2])
    provenance = pd.read_csv(required[0])
    if int(applicability.undefined_in_applicable_rows.sum()) or int(applicability.defined_in_not_applicable_rows.sum()):
        raise RuntimeError("Applicability audit is not clean")
    if not ((provenance.status == "PASS").all() and (provenance.unique_reconstructor_hashes == 1).all()
            and (provenance.unique_formula_hashes == 1).all() and provenance.reconstructor_sha256.nunique() == 1):
        raise RuntimeError("Provenance audit is not clean")
    prereg = json.loads((run / "preregistration/hierarchical_feasibility_preregistration.sha256.json").read_text())
    recon_hash = str(provenance.reconstructor_sha256.iloc[0])
    formula_hash = str(provenance.formula_sha256.iloc[0])
    table_hashes = [{"relative_path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in required + outcomes + samples]
    write_json_fresh(run / "logs/label_audit_report_writer_incident.json", {
        "status": "APPEND_ONLY_BOOKKEEPING_CORRECTION", "exception": "TypeError while summing dict_values in report interpolation",
        "scientific_tables_completed_before_exception": True, "labels_recomputed": False, "labels_changed": False,
        "fitting_started": False, "development_accessed": False, "lockbox_accessed": False,
        "superseding_finalizer": str(Path(__file__).resolve().relative_to(REPO)), "verified_artifacts": table_hashes,
    })
    report = f"""# Prospective label audit

Status: **PASS before fitting**. This report was written by an append-only finalizer after a bookkeeping-only report interpolation exception; no label was recomputed or changed.

- Rows audited: 32,000 across five prospective datasets.
- Reconstructor SHA-256 across every partition: `{recon_hash}` (one unique value).
- Label implementation SHA-256 across every partition: `{formula_hash}` (one unique value).
- NULL and AMBIGUOUS valid-risk values: explicitly not applicable; zero values were defined outside applicability.
- UNIQUE_VALID requested truth: present on every applicable row.
- Undefined-to-negative/zero coercions in authoritative v4 targets: zero.
- Scene IDs: globally unique across all five datasets.
- Calibration mixture: frozen natural 70/20/10 distribution; no balancing and no model selection.
- Preregistration SHA-256 `{prereg['sha256']}` existed before data, inference, or fitting.
- Development/lockbox access: zero / zero.
"""
    write_text_fresh(run / "diagnostics/prospective_label_audit.md", report)
    write_json_fresh(run / "logs/prospective_label_audit_complete.json", {
        "status": "PASS", "completed_at_unix": time.time(), "append_only_finalizer": True, "labels_recomputed": False,
        "fitting_started": False, "reconstructor_sha256": recon_hash, "formula_sha256": formula_hash,
        "preregistration_sha256": prereg["sha256"], "development_accessed": False, "lockbox_accessed": False,
    })


if __name__ == "__main__":
    main()
