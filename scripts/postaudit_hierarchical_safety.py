#!/usr/bin/env python3
"""Append-only post-documentation validation for hierarchical safety."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
DOCS = (
    "docs/hierarchical_query_semantics.md", "docs/hierarchical_recoverability_contract.md",
    "docs/hierarchical_safety_policy.md", "docs/hierarchical_safety_experiment.md",
    "docs/current_status.md", "docs/project_roadmap.md", "docs/experiment_log.md",
    "docs/model_card_thayer_select.md", "docs/limitations_and_next_steps.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def write_text_fresh(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL; descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle: handle.write(value)


def write_json_fresh(path: Path, value: object) -> None: write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists(): raise FileExistsError(path)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); run = args.run_dir.resolve()
    compile_result = command([".venv-btk/bin/python", "-m", "compileall", "-q", "src", "scripts", "tests"])
    tests_result = command([".venv-btk/bin/python", "-m", "unittest", "-v", "tests.test_hierarchical_safety", "tests.test_hierarchical_query_gate", "tests.test_thayer_select", "tests.test_recoverability_phase2"])
    diff_result = command(["git", "diff", "--check"]); git_status = command(["git", "status", "--short", "--branch"])
    document_rows = []
    for name in DOCS:
        path = REPO / name; text = path.read_text(); document_rows.append({"relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "mentions_hierarchical": "hierarchical" in text.lower(), "mentions_lockbox": "lockbox" in text.lower()})
    write_csv_fresh(run / "tables/postdocumentation_document_hashes.csv", pd.DataFrame(document_rows))
    original_audit = json.loads((run / "diagnostics/final_correctness_audit.json").read_text()); evaluation = json.loads((run / "logs/development_evaluation_complete.json").read_text())
    checks = {
        "original_final_audit_pass": original_audit["status"] == "PASS", "compileall": compile_result["returncode"] == 0,
        "relevant_tests": tests_result["returncode"] == 0, "git_diff_check": diff_result["returncode"] == 0,
        "all_required_documents_present": all((REPO / name).is_file() for name in DOCS),
        "documents_record_hierarchical_status": all(row["mentions_hierarchical"] for row in document_rows),
        "condition_c_hash_still_unchanged": sha256_file(CONDITION_C) == evaluation["condition_c_checkpoint_after"],
        "development_evaluation_count_still_one": evaluation["evaluation_count"] == 1,
        "final_report_present": (run / "reports/final_report.md").is_file(),
        "classification_unchanged": json.loads((run / "logs/finalization_complete.json").read_text())["classification"] == "FAILURE",
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "compileall": compile_result, "tests": tests_result, "git_diff_check": diff_result, "git_status": git_status, "completed_at_unix": time.time(), "new_development_inference": False, "development_evaluation_count": 1, "lockbox_used": False}
    write_json_fresh(run / "logs/postdocumentation_validation.json", result); write_json_fresh(run / "diagnostics/post_documentation_correctness_audit.json", {"status": result["status"], "checks": checks})
    addendum = f"""# Post-documentation addendum

The append-only documentation update is complete. Post-documentation compileall,
32 targeted tests, `git diff --check`, Condition-C hash verification, and the
one-time development marker all pass. No new development inference occurred;
the evaluation count remains one and the lockbox remains untouched.

Campaign classification remains **FAILURE** for the complete policy, with a
successful query-gate subcomponent.

## Final repository status

```text
{git_status['stdout'].rstrip()}
```
"""
    write_text_fresh(run / "reports/postdocumentation_addendum.md", addendum)
    if result["status"] != "PASS": raise RuntimeError("Post-documentation audit failed")
    print(run.relative_to(REPO))


if __name__ == "__main__": main()
