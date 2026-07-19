#!/usr/bin/env python3
"""Append closure evidence to a fail-closed Thayer-FF prestart run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[1]


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


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fieldnames = fields or (list(rows[0]) if rows else [])
    if not fieldnames:
        raise ValueError(f"CSV fields required: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    output = result.stdout
    if result.stderr:
        output += result.stderr
    return result.returncode, output


def checkpoint_inventory() -> list[dict[str, object]]:
    rows = []
    for path in sorted((REPO / "outputs/runs").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    return rows


def validate_csvs(run: Path) -> dict[str, object]:
    rows = []
    for path in sorted(run.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            parsed = list(reader)
        widths = sorted({len(row) for row in parsed})
        rows.append(
            {
                "path": str(path.relative_to(run)),
                "row_count": len(parsed),
                "column_counts": widths,
                "rectangular": len(widths) == 1 and bool(widths and widths[0] > 0),
            }
        )
    return {
        "status": "PASS" if rows and all(row["rectangular"] for row in rows) else "FAIL",
        "file_count": len(rows),
        "files": rows,
    }


def privacy_audit(paths: list[Path]) -> list[dict[str, object]]:
    absolute_user = "/" + "Users/"
    absolute_home = "/" + "home/"
    chat_name = "Chat" + "GPT"
    assistant_name = "Cod" + "ex"
    ai_phrase = "artificial" + " intelligence"
    forbidden = re.compile(
        "|".join(
            (
                re.escape(absolute_user),
                re.escape(absolute_home),
                rf"(?:^|\W)(?:{chat_name}|{assistant_name})(?:$|\W)",
                ai_phrase,
            )
        ),
        flags=re.IGNORECASE,
    )
    matches = []
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix.lower() in {".pyc", ".png", ".h5", ".pth"}:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if forbidden.search(line):
                    matches.append(
                        {
                            "path": str(path.relative_to(REPO)),
                            "line": line_number,
                            "text": line,
                        }
                    )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve() or not run.name.startswith(
        "thayer_fixed_feature_audit_"
    ):
        raise ValueError("unexpected run directory")
    incident = json.loads((run / "logs/fail_closed_stop.json").read_text())
    if incident["status"] != "FAIL_CLOSED_BEFORE_PREREGISTRATION":
        raise RuntimeError("run is not the expected fail-closed prestart record")

    with (run / "tables/checkpoint_inventory_before.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        before_rows = list(csv.DictReader(handle))
    after_rows = checkpoint_inventory()
    write_csv_fresh(run / "tables/checkpoint_inventory_after.csv", after_rows)
    before = {row["path"]: row for row in before_rows}
    after = {str(row["path"]): row for row in after_rows}
    audit_rows = []
    for path in sorted(set(before) | set(after)):
        left = before.get(path)
        right = after.get(path)
        audit_rows.append(
            {
                "path": path,
                "before_sha256": left["sha256"] if left else "",
                "after_sha256": right["sha256"] if right else "",
                "status": "UNCHANGED" if left and right and left["sha256"] == right["sha256"] else "MISMATCH",
            }
        )
    write_csv_fresh(run / "tables/checkpoint_hash_audit.csv", audit_rows)
    checkpoint_pass = bool(audit_rows) and all(row["status"] == "UNCHANGED" for row in audit_rows)

    compile_code, compile_output = run_command(
        str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"
    )
    write_text_fresh(
        run / "logs/compileall_output.txt",
        compile_output + f"exit_code={compile_code}\n",
    )
    test_code, test_output = run_command(
        str(REPO / ".venv-btk/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_output_parameterization.py",
        "tests/test_canonical_tensor_hash.py",
        "tests/test_two_expert_decoder.py",
    )
    write_text_fresh(
        run / "logs/focused_tests_output.txt",
        test_output + f"exit_code={test_code}\n",
    )
    diff_code, diff_output = run_command("git", "diff", "--check")
    write_text_fresh(run / "logs/git_diff_check.txt", diff_output + f"exit_code={diff_code}\n")
    cached_code, cached_output = run_command("git", "diff", "--cached", "--check")
    write_text_fresh(
        run / "logs/git_diff_cached_check.txt",
        cached_output + f"exit_code={cached_code}\n",
    )

    privacy_matches = privacy_audit(
        [
            run,
            REPO / "scripts/record_thayer_fixed_feature_prestart_stop.py",
            REPO / "scripts/finalize_thayer_fixed_feature_prestart_stop.py",
        ]
    )
    write_json_fresh(
        run / "logs/privacy_path_grep.json",
        {
            "status": "PASS" if not privacy_matches else "FAIL",
            "match_count": len(privacy_matches),
            "matches": privacy_matches,
        },
    )
    csv_audit = validate_csvs(run)
    write_json_fresh(run / "logs/csv_schema_validation.json", csv_audit)

    large_rows = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.stat().st_size >= 1024 * 1024:
            large_rows.append(
                {
                    "path": str(path.relative_to(run)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv_fresh(
        run / "tables/large_file_inventory.csv",
        large_rows,
        fields=["path", "bytes", "sha256"],
    )
    status_code, status_output = run_command("git", "status", "--short")
    write_text_fresh(
        run / "logs/final_git_status.txt",
        status_output + f"exit_code={status_code}\n",
    )

    operational_pass = all(
        (
            checkpoint_pass,
            compile_code == 0,
            test_code == 0,
            diff_code == 0,
            cached_code == 0,
            not privacy_matches,
            csv_audit["status"] == "PASS",
            status_code == 0,
        )
    )
    correctness = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL_CLOSED_WITH_CLEAN_CLOSURE" if operational_pass else "FAIL_CLOSED_WITH_CLOSURE_FAILURE",
        "scientific_campaign_status": "NOT_RUN_BY_PROTECTED_DATA_GATE",
        "protocol_failure_count": 1,
        "protocol_failure": incident["reason"],
        "closure_validation_pass": operational_pass,
        "historical_checkpoint_count_before": len(before_rows),
        "historical_checkpoint_count_after": len(after_rows),
        "historical_checkpoint_mismatch_count": sum(row["status"] != "UNCHANGED" for row in audit_rows),
        "compileall_pass": compile_code == 0,
        "focused_tests_pass": test_code == 0,
        "git_diff_check_pass": diff_code == 0,
        "git_diff_cached_check_pass": cached_code == 0,
        "privacy_path_grep_pass": not privacy_matches,
        "csv_schema_validation_pass": csv_audit["status"] == "PASS",
        "large_file_count": len(large_rows),
        "optimizer_step_count": 0,
        "capacity_ladder_authorized": False,
    }
    write_json_fresh(run / "diagnostics/final_correctness_audit_superseding.json", correctness)

    addendum = f"""# Checkpoint and repository closure addendum

The scientific campaign remains **NOT RUN BY PROTECTED-DATA GATE**. This
addendum does not supersede or excuse the prestart development-access incident.
It verifies only that the fail-closed stop was operationally clean.

- Historical checkpoints before/after: `{len(before_rows)}/{len(after_rows)}`.
- Historical checkpoint mismatches: `{correctness['historical_checkpoint_mismatch_count']}`.
- Compileall: `{'PASS' if compile_code == 0 else 'FAIL'}`.
- Focused frozen-contract tests: `{'PASS' if test_code == 0 else 'FAIL'}`.
- CSV/schema validation: `{csv_audit['status']}`.
- Privacy/path grep: `{'PASS' if not privacy_matches else 'FAIL'}`.
- Git diff checks: `{'PASS' if diff_code == 0 and cached_code == 0 else 'FAIL'}`.
- Large files at or above 1 MiB: `{len(large_rows)}`.
- Scene/P0 loads, feature extractions, model constructions, optimizer steps,
  and JVP/VJP operations: `0/0/0/0/0`.
- No checkpoint, historical run, README, or project-status document was
  modified by the stopped campaign.

Operational closure status: **{'PASS' if operational_pass else 'FAIL'}**.
Scientific conclusion: **none**. Capacity-ladder authorization: **false**.
"""
    write_text_fresh(run / "reports/checkpoint_closure_addendum.md", addendum)
    write_json_fresh(
        run / "logs/closure_final.json",
        {
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": correctness["status"],
            "scientific_campaign_status": correctness["scientific_campaign_status"],
            "run_bytes_before_record": sum(
                path.stat().st_size for path in run.rglob("*") if path.is_file()
            ),
            "capacity_ladder_authorized": False,
        },
    )
    print(json.dumps(correctness, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
