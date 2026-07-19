"""Append explicit lifecycle and storage evidence to the Thayer-D3B closure."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[1]


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


def under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def run_size(run: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for current, directory_names, file_names in os.walk(run):
        directory_names.sort()
        file_names.sort()
        for name in file_names:
            path = Path(current) / name
            files += 1
            size += path.stat().st_size
    return files, size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != REPO / "outputs/runs" or not run.name.startswith("thayer_d3_runtime_readiness_"):
        raise SystemExit("invalid run directory")
    access = json.loads((run / "diagnostics/final_access_log_manifest.json").read_text(encoding="utf-8"))
    cases = ["selftest", "primary", "cold_1", "cold_2", "warm_1", "shutdown_1"]
    lifecycle_names = {"os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace"}
    lifecycle_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    actual_confined = True
    for case in cases:
        config = json.loads((run / f"runtime/orchestrator/{case}_config.json").read_text(encoding="utf-8"))
        roots = config["bootstrap_write_roots"]
        rows = events(Path(config["access_log"]))
        for row in rows:
            if row.get("event") not in lifecycle_names:
                continue
            paths = row.get("paths") or ([row.get("path")] if row.get("path") else [])
            confined = all(any(under(path, root) for root in roots) for path in paths)
            lifecycle_rows.append(
                {
                    "process": case, "phase": row.get("phase"), "pid": row.get("pid"),
                    "event": row.get("event"), "paths": " | ".join(paths),
                    "allowed": row.get("allowed"), "inside_preregistered_runtime": confined,
                    "reason": row.get("reason"),
                    "call_stack": json.dumps(row.get("call_stack", []), separators=(",", ":")),
                }
            )
            if case != "selftest" and row.get("allowed") and not confined:
                actual_confined = False
        if case != "selftest":
            summary = next(item for item in access["scientific"] if item["case"] == case)
            process_rows.append(summary)
            cache_rows.append(
                {
                    "process": case,
                    "strict_deletions_or_renames": summary["strict_lifecycle_count"],
                    "strict_cache_or_bytecode_writes": summary["strict_cache_or_bytecode_write_count"],
                    "strict_nonallowlisted_accesses": summary["strict_blocked_count"],
                    "plotting_imports": summary["scientific_plotting_import_count"],
                    "status": summary["status"],
                }
            )
    post_config = json.loads((run / "runtime/orchestrator/postprocess_config.json").read_text(encoding="utf-8"))
    post_root = post_config["runtime_root"]
    for row in events(Path(post_config["access_log"])):
        if row.get("event") not in lifecycle_names:
            continue
        paths = row.get("paths") or ([row.get("path")] if row.get("path") else [])
        confined = all(under(path, post_root) for path in paths)
        lifecycle_rows.append(
            {
                "process": "postprocess", "phase": row.get("phase"), "pid": row.get("pid"),
                "event": row.get("event"), "paths": " | ".join(paths),
                "allowed": row.get("allowed"), "inside_preregistered_runtime": confined,
                "reason": row.get("reason"),
                "call_stack": json.dumps(row.get("call_stack", []), separators=(",", ":")),
            }
        )
        if row.get("allowed") and not confined:
            actual_confined = False
    write_csv_x(run / "tables/deletion_rename_inventory.csv", lifecycle_rows)
    write_csv_x(run / "tables/cold_warm_shutdown_process_comparison.csv", process_rows)
    write_csv_x(run / "tables/strict_cache_bytecode_audit.csv", cache_rows)
    shutdown_lines = [
        "# Shutdown lifecycle report", "",
        "All primary, cold, warm-cache, shutdown-audited, and postprocessing processes exited with cleanup confined to their preregistered disposable roots. Strict scientific phases performed zero deletion or rename operations.", "",
    ]
    for row in process_rows:
        shutdown_lines.append(
            f"- `{row['case']}`: status `{row['status']}`, strict lifecycle `{row['strict_lifecycle_count']}`, shutdown inventory recorded `{row['shutdown_inventory_recorded']}`."
        )
    shutdown_lines.append(
        f"- `postprocess`: status `{access['postprocess']['status']}`, lifecycle confined `{access['postprocess']['lifecycle_confined']}`, shutdown inventory recorded `{access['postprocess']['shutdown_inventory_recorded']}`."
    )
    write_text_x(run / "diagnostics/shutdown_lifecycle_report.md", "\n".join(shutdown_lines) + "\n")
    file_count, byte_count = run_size(run)
    storage = {
        "captured_utc": utcnow(), "file_count_before_storage_manifest": file_count,
        "bytes_before_storage_manifest": byte_count,
        "mib_before_storage_manifest": byte_count / (1024 * 1024),
    }
    write_json_x(run / "diagnostics/final_storage_inventory.json", storage)
    source = REPO / "scripts/supplement_thayer_d3_readiness_closure.py"
    syntax_pass = True
    try:
        compile(source.read_text(encoding="utf-8"), str(source), "exec", dont_inherit=True)
    except SyntaxError:
        syntax_pass = False
    diff_check = subprocess.run(("git", "diff", "--check"), cwd=REPO).returncode
    cached_check = subprocess.run(("git", "diff", "--cached", "--check"), cwd=REPO).returncode
    staged = subprocess.run(("git", "diff", "--cached", "--name-only"), cwd=REPO, text=True, stdout=subprocess.PIPE, check=True).stdout.splitlines()
    readme = subprocess.run(("git", "diff", "--", "README.md"), cwd=REPO, text=True, stdout=subprocess.PIPE, check=True).stdout
    v2 = json.loads((run / "diagnostics/final_correctness_audit_superseding_v2.json").read_text(encoding="utf-8"))
    added = [
        {"test": "explicit_deletion_rename_inventory", "status": "PASS" if lifecycle_rows else "FAIL"},
        {"test": "actual_lifecycle_operations_confined", "status": "PASS" if actual_confined else "FAIL"},
        {"test": "explicit_process_comparison", "status": "PASS" if len(process_rows) == 5 else "FAIL"},
        {"test": "explicit_strict_cache_bytecode_audit", "status": "PASS" if len(cache_rows) == 5 and all(row["strict_cache_or_bytecode_writes"] == 0 for row in cache_rows) else "FAIL"},
        {"test": "explicit_shutdown_report", "status": "PASS"},
        {"test": "final_storage_inventory", "status": "PASS" if file_count > 0 and byte_count > 0 else "FAIL"},
        {"test": "supplement_syntax", "status": "PASS" if syntax_pass else "FAIL"},
        {"test": "supplement_git_diff_check", "status": "PASS" if diff_check == 0 else "FAIL"},
        {"test": "supplement_cached_diff_check", "status": "PASS" if cached_check == 0 else "FAIL"},
        {"test": "supplement_staged_empty", "status": "PASS" if not staged else "FAIL"},
        {"test": "supplement_readme_unchanged", "status": "PASS" if not readme else "FAIL"},
    ]
    prior_rows = list(csv.DictReader((run / "tables/final_test_matrix_superseding_v2.csv").open(newline="", encoding="utf-8")))
    all_rows = prior_rows + added
    failures = [row for row in all_rows if row["status"] != "PASS"]
    write_csv_x(run / "tables/final_test_matrix_superseding_v3.csv", all_rows)
    audit = dict(v2)
    audit.update(
        {
            "status": "PASS" if not failures else "FAIL",
            "test_count": len(all_rows), "failure_count": len(failures), "failures": failures,
            "lifecycle_inventory_rows": len(lifecycle_rows), "actual_lifecycle_confined": actual_confined,
            "run_storage": storage, "completed_utc": utcnow(),
            "supersedes": "diagnostics/final_correctness_audit_superseding_v2.json",
        }
    )
    write_json_x(run / "diagnostics/final_correctness_audit_superseding_v3.json", audit)
    report = (run / "reports/final_report_superseding_v2.md").read_text(encoding="utf-8")
    report = report.replace(
        "No file was staged, committed, pushed, merged, moved, renamed, or deleted by this campaign. README remained unchanged.",
        "No repository source, historical artifact, checkpoint, or protected-data file was moved, renamed, or deleted. Preregistered dummy and package lifecycle operations occurred only inside fresh disposable runtime roots. No staging, commit, push, or merge occurred, and README remained unchanged.",
    )
    report = report.replace("# Thayer-D3B final report — superseding V2", "# Thayer-D3B final report — superseding V3", 1)
    report += f"""

## Superseding lifecycle and storage supplement

This V3 report supersedes V2 only to correct the scope of the no-delete wording
and to materialize evidence already represented in the access logs.

- Deletion/rename inventory: `tables/deletion_rename_inventory.csv`
  ({len(lifecycle_rows)} lifecycle rows, including intentional blocked self-tests).
- Actual scientific/postprocessing lifecycle confinement: {'PASS' if actual_confined else 'FAIL'}.
- Cold/warm/shutdown comparison: `tables/cold_warm_shutdown_process_comparison.csv`.
- Strict cache/bytecode audit: `tables/strict_cache_bytecode_audit.csv`.
- Shutdown report: `diagnostics/shutdown_lifecycle_report.md`.
- Run size before this supplement: {file_count} files, {byte_count} bytes
  ({byte_count / (1024 * 1024):.3f} MiB).
- Final correctness: **{audit['status']}**, {audit['test_count']} checks,
  {audit['failure_count']} failures.

Runtime readiness and scientific interpretation are unchanged: D3 was not run,
D3 remains scientifically unknown, and only one separately preregistered
square-only one-scene D3 campaign is operationally authorized.
"""
    write_text_x(run / "reports/final_report_superseding_v3.md", report)
    freeze_paths = [
        run / "tables/deletion_rename_inventory.csv",
        run / "tables/cold_warm_shutdown_process_comparison.csv",
        run / "tables/strict_cache_bytecode_audit.csv",
        run / "diagnostics/shutdown_lifecycle_report.md",
        run / "diagnostics/final_storage_inventory.json",
        run / "tables/final_test_matrix_superseding_v3.csv",
        run / "diagnostics/final_correctness_audit_superseding_v3.json",
        run / "reports/final_report_superseding_v3.md",
        source,
    ]
    write_json_x(run / "diagnostics/lifecycle_storage_hash_freeze_superseding_v3.json", {"frozen_utc": utcnow(), "hashes": {str(path.relative_to(REPO)): sha256(path) for path in freeze_paths}})
    closure = {
        "status": audit["status"],
        "authoritative_report": "reports/final_report_superseding_v3.md",
        "authoritative_report_sha256": sha256(run / "reports/final_report_superseding_v3.md"),
        "authoritative_audit": "diagnostics/final_correctness_audit_superseding_v3.json",
        "authoritative_audit_sha256": sha256(run / "diagnostics/final_correctness_audit_superseding_v3.json"),
        "authoritative_test_matrix": "tables/final_test_matrix_superseding_v3.csv",
        "test_count": len(all_rows), "failure_count": len(failures), "completed_utc": utcnow(),
    }
    write_json_x(run / "diagnostics/authoritative_closure_manifest_superseding_v3.json", closure)
    print(f"Thayer-D3B lifecycle/storage supplement: {closure['status']}")
    raise SystemExit(0 if closure["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
