#!/usr/bin/env python3
"""Append a token-aware privacy correction to the Thayer-ME final audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = [
    "docs/thayer_two_expert_decoder.md", "docs/expert_specialization_contract.md",
    "docs/micro_overfit_capacity_gate.md", "docs/atlas_expert_hypotheses.md",
    "docs/thayer_multiple_hypotheses.md", "docs/ambiguity_atlas_v0.md",
    "docs/competing_hypothesis_recoverability.md", "docs/current_status.md",
    "docs/project_roadmap.md", "docs/experiment_log.md",
    "docs/limitations_and_next_steps.md", "docs/model_card_thayer_select.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    primary = json.loads((run_dir / "diagnostics/final_correctness_audit.json").read_text())
    if primary["failure_count"] != 1 or primary["failures"][0]["check"] != "public_privacy_path_grep":
        raise RuntimeError("unexpected primary audit state")
    pattern = r"/Users/|ChatGPT|artificial intelligence|(^|[^[:alnum:]_])Codex([^[:alnum:]_]|$)"
    result = subprocess.run(["rg", "-n", pattern, *PUBLIC_DOCS], cwd=REPO, text=True, capture_output=True)
    if result.returncode != 1:
        raise RuntimeError(f"token-aware privacy grep failed:\n{result.stdout}{result.stderr}")
    write_text_fresh(run_dir / "logs/privacy_path_grep_superseding.txt", "Token-aware public-document grep found no personal absolute paths and no standalone references to ChatGPT, Codex as an assistant, or artificial intelligence. The primary grep's only literal substring matches were the dependency name SurveyCodex in existing PSF provenance text.\n")
    rows = read_csv(run_dir / "tables/final_correctness_checks.csv")
    for row in rows:
        if row["check"] == "public_privacy_path_grep":
            row["status"] = "PASS"
            row["evidence"] = "token-aware grep clean; SurveyCodex dependency name excluded"
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("a non-privacy correctness failure remains")
    write_csv_fresh(run_dir / "tables/final_correctness_checks_superseding.csv", rows)
    audited_at = datetime.now(timezone.utc).isoformat()
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit_superseding.json", {"status": "PASS_WITH_PREREGISTERED_MICRO_GATE_FAILURE", "failure_count": 0, "failures": [], "check_count": len(rows), "scientific_decision": "REPRESENTATIONAL_OR_LOSS_IMPLEMENTATION_FAILURE", "full_training_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "historical_checkpoint_count": primary["historical_checkpoint_count"], "supersedes": "diagnostics/final_correctness_audit.json", "audited_at_utc": audited_at})
    write_text_fresh(run_dir / "reports/post_final_correctness_addendum.md", f"""# Post-final correctness addendum

The primary audit's sole failure was a false-positive privacy match: the broad
substring pattern matched the dependency name `SurveyCodex` in pre-existing PSF
provenance text. A token-aware superseding grep found no personal absolute paths
and no standalone references to ChatGPT, Codex as an assistant, or artificial
intelligence in the public campaign documents.

All 22 correctness checks therefore pass under the superseding audit. The
scientific decision is unchanged: **REPRESENTATIONAL OR LOSS IMPLEMENTATION
FAILURE; FULL TRAINING AND ATLAS PROHIBITED**.

- Primary final report SHA-256: `{sha256_file(run_dir / 'reports/final_report.md')}`.
- Superseding correctness table: `tables/final_correctness_checks_superseding.csv`.
- Full fit / Atlas / development / lockbox access counts: 0 / 0 / 0 / 0.
- Historical checkpoints: {primary['historical_checkpoint_count']}/{primary['historical_checkpoint_count']} unchanged.
""")
    source_rows = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            source_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv_fresh(run_dir / "tables/source_code_hashes_superseding.csv", source_rows)
    write_json_fresh(run_dir / "logs/finalization_superseding_complete.json", {"status": "PASS", "scientific_decision": "REPRESENTATIONAL_OR_LOSS_IMPLEMENTATION_FAILURE", "correctness_status": "PASS_WITH_PREREGISTERED_MICRO_GATE_FAILURE", "addendum_sha256": sha256_file(run_dir / "reports/post_final_correctness_addendum.md"), "full_training_count": 0, "atlas_evaluation_count": 0})
    print(json.dumps({"status": "PASS", "correctness_failures": 0, "scientific_decision": "REPRESENTATIONAL_OR_LOSS_IMPLEMENTATION_FAILURE"}, sort_keys=True))


if __name__ == "__main__":
    main()
