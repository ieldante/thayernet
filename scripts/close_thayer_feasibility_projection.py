#!/usr/bin/env python3
"""Post-final repository and schema closure for Thayer-FP."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def main() -> None:
    source_rows = []
    for root in ("src", "scripts", "tests"):
        for path in sorted((REPO / root).rglob("*.py")):
            source_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size})
    path = RUN / "tables/source_code_hashes_closure.csv"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0])); writer.writeheader(); writer.writerows(source_rows)

    csv_errors = []
    csv_count = 0
    for csv_path in sorted(RUN.rglob("*.csv")):
        csv_count += 1
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        widths = {len(row) for row in rows}
        if not rows or len(widths) != 1:
            csv_errors.append({"path": str(csv_path.relative_to(REPO)), "rows": len(rows), "widths": sorted(widths)})
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True)
    cached = subprocess.run(["git", "diff", "--cached", "--check"], cwd=REPO, text=True, capture_output=True)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True, capture_output=True)
    readme = subprocess.run(["git", "status", "--short", "README.md"], cwd=REPO, text=True, capture_output=True)
    status = subprocess.run(["git", "status", "--short"], cwd=REPO, text=True, capture_output=True)
    run_bytes_before_record = sum(item.stat().st_size for item in RUN.rglob("*") if item.is_file())
    closure = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "csv_count": csv_count,
        "csv_schema_pass": not csv_errors, "csv_errors": csv_errors,
        "git_diff_check_pass": diff.returncode == 0, "git_diff_cached_check_pass": cached.returncode == 0,
        "staged_index_empty": not staged.stdout.strip(), "readme_unchanged": not readme.stdout.strip(),
        "run_bytes_before_closure_record": run_bytes_before_record,
        "strict_correctness_status": "FAIL — output-contract stop rule not enforced at epoch 1",
        "scientific_decision": "FAILURE — PROJECTED TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT MEMORIZE THEM",
    }
    fresh_text(RUN / "logs/repository_closure_checks.json", json.dumps(closure, indent=2, sort_keys=True) + "\n")
    fresh_text(RUN / "reports/repository_closure_addendum.md", f"""# Repository closure addendum

- All `{csv_count}` final campaign CSV files are rectangular: `{'PASS' if not csv_errors else 'FAIL'}`.
- `git diff --check`: `{'PASS' if diff.returncode == 0 else 'FAIL'}`.
- `git diff --cached --check`: `{'PASS' if cached.returncode == 0 else 'FAIL'}`.
- Staged index empty: `{'PASS' if not staged.stdout.strip() else 'FAIL'}`.
- README unchanged: `{'PASS' if not readme.stdout.strip() else 'FAIL'}`.
- Run bytes before this closure record: `{run_bytes_before_record}`.

Scientific decision remains **FAILURE — PROJECTED TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT MEMORIZE THEM**. Strict correctness remains **FAIL** solely because negative model outputs were observed at epoch 1 and the preregistered output-contract stop rule was not enforced then.

Final Git status:

```text
{status.stdout.rstrip()}
```
""")
    print(json.dumps(closure, indent=2))


if __name__ == "__main__":
    main()
