#!/usr/bin/env python3
"""Supersede the stopped Thayer-FF checkpoint inventory with full output scope."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
AUTHORITATIVE = (
    REPO
    / "outputs/runs/thayer_output_parameterization_20260713_023120"
    / "tables/checkpoint_inventory_after.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_fresh(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing empty checkpoint table")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve() or not run.name.startswith(
        "thayer_fixed_feature_audit_"
    ):
        raise ValueError("unexpected run directory")
    prior = json.loads(
        (run / "diagnostics/final_correctness_audit_closure_superseding_v2.json").read_text()
    )
    with AUTHORITATIVE.open(newline="", encoding="utf-8") as handle:
        authoritative_rows = list(csv.DictReader(handle))
    authoritative = {row["path"]: row for row in authoritative_rows}

    current_rows = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if not path.is_file():
            continue
        current_rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv_fresh(run / "tables/checkpoint_inventory_full_closure.csv", current_rows)
    current = {str(row["path"]): row for row in current_rows}
    comparison = []
    for path in sorted(set(authoritative) | set(current)):
        left = authoritative.get(path)
        right = current.get(path)
        comparison.append(
            {
                "path": path,
                "authoritative_sha256": left["sha256"] if left else "",
                "closure_sha256": right["sha256"] if right else "",
                "status": (
                    "UNCHANGED"
                    if left and right and left["sha256"] == right["sha256"]
                    else "MISMATCH"
                ),
            }
        )
    write_csv_fresh(
        run / "tables/checkpoint_inventory_authoritative_comparison.csv",
        comparison,
    )
    mismatch_count = sum(row["status"] != "UNCHANGED" for row in comparison)
    passed = (
        len(authoritative_rows) == 600
        and len(current_rows) == 600
        and mismatch_count == 0
    )
    correctness = {
        **prior,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "FAIL_CLOSED_WITH_PUBLIC_SURFACE_CLOSURE_PASS"
            if passed and prior["closure_validation_pass"]
            else "FAIL_CLOSED_WITH_CLOSURE_FAILURE"
        ),
        "closure_validation_pass": bool(passed and prior["closure_validation_pass"]),
        "historical_checkpoint_count_before": 600,
        "historical_checkpoint_count_after": 600,
        "historical_checkpoint_mismatch_count": mismatch_count,
        "checkpoint_inventory_scope": "ALL_PTH_FILES_UNDER_OUTPUTS",
        "checkpoint_reference": str(AUTHORITATIVE.relative_to(REPO)),
        "original_582_file_inventory_scope": "OUTPUTS_RUNS_ONLY_SUPERSEDED",
        "supersedes": "diagnostics/final_correctness_audit_closure_superseding_v2.json",
    }
    write_json_fresh(
        run / "diagnostics/final_correctness_audit_closure_superseding_v3.json",
        correctness,
    )
    addendum = f"""# Checkpoint-inventory scope addendum

The initial stopped-run inventory reported 582 files because its search root
was limited to `outputs/runs`. It omitted 18 root-level checkpoint files under
`outputs/checkpoints`. This was an inventory-scope defect, not a checkpoint
mutation.

The superseding audit uses the authoritative Thayer-OP closure inventory as
its pre-campaign reference and rehashes every `.pth` file under `outputs`:

- Authoritative reference count: `{len(authoritative_rows)}`.
- Current closure count: `{len(current_rows)}`.
- Hash mismatches, additions, or removals: `{mismatch_count}`.
- Result: **{'PASS' if passed else 'FAIL'}**.

The scientific campaign remains **NOT RUN BY PROTECTED-DATA GATE**. This
scope correction changes no scientific conclusion and does not authorize the
capacity ladder.
"""
    write_text_fresh(run / "reports/checkpoint_inventory_scope_addendum.md", addendum)
    write_json_fresh(
        run / "logs/closure_superseding_v3.json",
        {
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": correctness["status"],
            "checkpoint_inventory_pass": passed,
            "historical_checkpoint_count": len(current_rows),
            "historical_checkpoint_mismatch_count": mismatch_count,
            "scientific_campaign_status": correctness["scientific_campaign_status"],
            "capacity_ladder_authorized": False,
        },
    )
    print(json.dumps(correctness, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
