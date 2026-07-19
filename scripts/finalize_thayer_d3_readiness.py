"""Append-only documentation and correctness closure for authoritative Thayer-D3B."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[1]
D3R = REPO / "outputs/runs/thayer_full_l0_d3r_20260713_121652"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args), cwd=REPO, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
    )


def document_paths() -> list[Path]:
    return [
        REPO / "docs/d3_runtime_bootstrap_contract.md",
        REPO / "docs/scientific_process_isolation.md",
        REPO / "docs/scientific_postprocessing_isolation.md",
        REPO / "docs/pure_forward_evaluator_contract.md",
        REPO / "docs/d3_runtime_readiness.md",
        REPO / "docs/authoritative_full_l0_d3.md",
        REPO / "docs/full_l0_fixed_feature_d3.md",
        REPO / "docs/allowlisted_file_access_contract.md",
        REPO / "docs/repository_integrity_audit.md",
        REPO / "docs/current_status.md",
        REPO / "docs/project_roadmap.md",
        REPO / "docs/experiment_log.md",
        REPO / "docs/limitations_and_next_steps.md",
    ]


def documentation_audit(run: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    credential = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
    required = {
        "docs/d3_runtime_readiness.md": (run.name, "READINESS PASS", "D3 remains scientifically unknown"),
        "docs/scientific_process_isolation.md": ("PYTHONDONTWRITEBYTECODE", "Matplotlib", "optimizer"),
        "docs/scientific_postprocessing_isolation.md": ("separate", "zero scientific", "lifecycle"),
    }
    for path in document_paths():
        relative = str(path.relative_to(REPO))
        if not path.exists():
            issues.append({"path": relative, "issue": "missing"})
            continue
        text = path.read_text(encoding="utf-8")
        hashes[relative] = sha256(path)
        if not text.startswith("# ") or "\r" in text:
            issues.append({"path": relative, "issue": "markdown_structure"})
        for number, line in enumerate(text.splitlines(), 1):
            if line.rstrip() != line:
                issues.append({"path": relative, "line": number, "issue": "trailing_whitespace"})
            if any(token in line for token in ("/Users/", "ChatGPT", "OpenAI")):
                issues.append({"path": relative, "line": number, "issue": "privacy_or_assistant_token"})
            if credential.search(line):
                issues.append({"path": relative, "line": number, "issue": "credential_pattern"})
        for token in required.get(relative, ()):
            if token.casefold() not in text.casefold():
                issues.append({"path": relative, "issue": f"missing_required_statement:{token}"})
    result = {
        "status": "PASS" if not issues else "FAIL", "file_count": len(document_paths()),
        "issues": issues, "hashes": hashes,
        "structural_markdown_check": "PASS" if not issues else "FAIL",
        "privacy_path_credential_check": "PASS" if not issues else "FAIL",
        "completed_utc": utcnow(),
    }
    write_json_x(run / "diagnostics/documentation_audit_superseding_v2.json", result)
    return result


def syntax_audit() -> dict[str, Any]:
    paths = [
        REPO / "scripts/run_thayer_d3_readiness.py",
        REPO / "scripts/run_thayer_d3_scientific_readiness.py",
        REPO / "scripts/run_thayer_d3_postprocess_readiness.py",
        REPO / "scripts/thayer_d3_runtime_guard.py",
        REPO / "scripts/finalize_thayer_d3_readiness.py",
        REPO / "tests/test_d3_readiness_process_isolation.py",
    ]
    failures = []
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            failures.append({"path": str(path.relative_to(REPO)), "error": str(exc)})
    return {"status": "PASS" if not failures else "FAIL", "file_count": len(paths), "failures": failures}


def verify_runtime_freeze(run: Path) -> dict[str, Any]:
    frozen = json.loads((run / "diagnostics/runtime_hash_freeze.json").read_text(encoding="utf-8"))["hashes"]
    mismatches = []
    for relative, expected in frozen.items():
        path = REPO / relative
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return {"status": "PASS" if not mismatches else "FAIL", "count": len(frozen), "mismatches": mismatches}


def checkpoint_recheck(run: Path) -> dict[str, Any]:
    rows = []
    with (D3R / "tables/checkpoint_inventory_after.csv").open(newline="", encoding="utf-8") as handle:
        for frozen in csv.DictReader(handle):
            path = REPO / frozen["path"]
            actual_bytes = path.stat().st_size
            actual_hash = sha256(path)
            status = actual_bytes == int(frozen["expected_bytes"]) and actual_hash == frozen["expected_sha256"]
            rows.append(
                {
                    "path": frozen["path"], "expected_bytes": frozen["expected_bytes"],
                    "actual_bytes": actual_bytes, "expected_sha256": frozen["expected_sha256"],
                    "actual_sha256": actual_hash, "status": "PASS" if status else "FAIL",
                }
            )
    write_csv_x(run / "metadata_checks/checkpoint_hash_audit_closure_v2.csv", rows)
    return {"status": "PASS" if len(rows) == 600 and all(row["status"] == "PASS" for row in rows) else "FAIL", "count": len(rows), "mismatches": sum(row["status"] != "PASS" for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != REPO / "outputs/runs" or not run.name.startswith("thayer_d3_runtime_readiness_"):
        raise SystemExit("invalid run directory")
    original = json.loads((run / "diagnostics/readiness_manifest.json").read_text(encoding="utf-8"))
    original_rows = list(csv.DictReader((run / "tables/final_test_matrix.csv").open(newline="", encoding="utf-8")))
    docs = documentation_audit(run)
    syntax = syntax_audit()
    freeze = verify_runtime_freeze(run)
    checkpoints = checkpoint_recheck(run)
    regression = subprocess.run(
        (sys.executable, str(REPO / "tests/test_d3_readiness_process_isolation.py")),
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    diff_check = git("diff", "--check", check=False).returncode
    cached_check = git("diff", "--cached", "--check", check=False).returncode
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    readme_diff = git("diff", "--", "README.md").stdout
    added = [
        {"test": "runtime_readiness_manifest_pass", "status": "PASS" if original["status"] == "READINESS_PASS_D3_NOT_RUN" else "FAIL"},
        {"test": "runtime_hash_freeze_unchanged", "status": freeze["status"]},
        {"test": "source_syntax_in_memory", "status": syntax["status"]},
        {"test": "process_isolation_regression", "status": "PASS" if regression.returncode == 0 else "FAIL"},
        {"test": "documentation_final_audit", "status": docs["status"]},
        {"test": "checkpoint_closure_600", "status": checkpoints["status"]},
        {"test": "final_git_diff_check", "status": "PASS" if diff_check == 0 else "FAIL"},
        {"test": "final_git_cached_diff_check", "status": "PASS" if cached_check == 0 else "FAIL"},
        {"test": "final_staged_index_empty", "status": "PASS" if not staged else "FAIL"},
        {"test": "final_readme_unchanged", "status": "PASS" if not readme_diff else "FAIL"},
    ]
    rows = original_rows + added
    failures = [row for row in rows if row["status"] != "PASS"]
    write_csv_x(run / "tables/final_test_matrix_superseding_v2.csv", rows)
    git_status = git("status", "--short").stdout
    write_text_x(run / "logs/final_git_status_superseding_v2.txt", git_status)
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "primary_outcome": "READINESS PASS — D3 NOT RUN" if not failures else "READINESS FAIL — D3 NOT RUN",
        "test_count": len(rows), "failure_count": len(failures), "failures": failures,
        "preregistration_sha256": original["preregistration_sha256"],
        "runtime_readiness_manifest_sha256": sha256(run / "diagnostics/readiness_manifest.json"),
        "runtime_freeze": freeze, "syntax": syntax, "documentation": docs,
        "checkpoint_closure": checkpoints,
        "process_isolation_regression": {"exit_code": regression.returncode, "stdout": regression.stdout, "stderr": regression.stderr},
        "git": {"diff_check_exit": diff_check, "cached_diff_check_exit": cached_check, "staged_paths": staged, "readme_diff": readme_diff},
        "scientific_tensor_deserializations": 0, "model_instantiations": 0,
        "optimizer_constructions": 0, "decoder_forwards": 0, "d3_run": False,
        "completed_utc": utcnow(), "supersedes": "diagnostics/final_correctness_audit.json",
    }
    write_json_x(run / "diagnostics/final_correctness_audit_superseding_v2.json", audit)
    closure_hashes = dict(docs["hashes"])
    for path in (
        run / "diagnostics/runtime_hash_freeze.json",
        run / "diagnostics/readiness_manifest.json",
        run / "diagnostics/final_access_log_manifest.json",
        run / "diagnostics/documentation_audit_superseding_v2.json",
        run / "diagnostics/final_correctness_audit_superseding_v2.json",
        run / "tables/final_test_matrix_superseding_v2.csv",
        run / "metadata_checks/checkpoint_hash_audit_closure_v2.csv",
    ):
        closure_hashes[str(path.relative_to(REPO))] = sha256(path)
    write_json_x(run / "diagnostics/closure_hash_freeze_superseding_v2.json", {"frozen_utc": utcnow(), "hashes": closure_hashes})
    base = (run / "reports/final_report.md").read_text(encoding="utf-8")
    report = base.replace("# Thayer-D3B final report", "# Thayer-D3B final report — superseding V2", 1)
    report += f"""

## Superseding documentation and closure audit

This report supersedes `reports/final_report.md` only for final documentation
and working-tree closure. Runtime evidence and scientific interpretation are
unchanged.

- Final correctness: **{audit['status']}**, {audit['test_count']} checks, {audit['failure_count']} failures.
- Requested-document audit: {docs['status']}, {docs['file_count']} files, {len(docs['issues'])} issues.
- Runtime/code hash freeze recheck: {freeze['status']}, {freeze['count']} artifacts.
- Historical checkpoint closure: {checkpoints['count']}/600 checked, {checkpoints['mismatches']} mismatches.
- Process-isolation regression: {'PASS' if regression.returncode == 0 else 'FAIL'}.
- `git diff --check`: {'PASS' if diff_check == 0 else 'FAIL'}.
- `git diff --cached --check`: {'PASS' if cached_check == 0 else 'FAIL'}.
- Staged paths: {len(staged)}.
- README diff: {'empty' if not readme_diff else 'nonempty'}.
- Scientific tensors, models, optimizers, decoder forwards, and D3 operations: zero.

Final working-tree status is preserved at
`logs/final_git_status_superseding_v2.txt`. The document and closure hashes are
frozen in `diagnostics/closure_hash_freeze_superseding_v2.json`.
"""
    write_text_x(run / "reports/final_report_superseding_v2.md", report)
    closure = {
        "status": audit["status"],
        "authoritative_report": "reports/final_report_superseding_v2.md",
        "authoritative_report_sha256": sha256(run / "reports/final_report_superseding_v2.md"),
        "authoritative_audit": "diagnostics/final_correctness_audit_superseding_v2.json",
        "authoritative_audit_sha256": sha256(run / "diagnostics/final_correctness_audit_superseding_v2.json"),
        "authoritative_test_matrix": "tables/final_test_matrix_superseding_v2.csv",
        "test_count": len(rows), "failure_count": len(failures), "completed_utc": utcnow(),
    }
    write_json_x(run / "diagnostics/authoritative_closure_manifest_superseding_v2.json", closure)
    print(f"Thayer-D3B superseding closure: {closure['status']}")
    raise SystemExit(0 if closure["status"] == "PASS" else 1)


if __name__ == "__main__":
    import sys

    main()
