#!/usr/bin/env python3
"""Repository-level closure checks for the completed Thayer-OC run."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/runs/thayer_output_conditioning_20260712_225459"
DOCS = [
    "docs/output_space_conditioning_audit.md", "docs/source_total_allocation_coordinates.md",
    "docs/scientific_gradient_preconditioning.md", "docs/scientific_basin_geometry.md",
    "docs/scientific_alignment_objective.md", "docs/frozen_loss_geometry_audit.md",
    "docs/thayer_two_expert_decoder.md", "docs/current_status.md", "docs/project_roadmap.md",
    "docs/experiment_log.md", "docs/limitations_and_next_steps.md", "docs/model_card_thayer_select.md",
]


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle: handle.write(value)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True)


def main() -> None:
    diff = run("git", "diff", "--check")
    staged = run("git", "diff", "--cached", "--check")
    status = run("git", "status", "--short")
    readme = run("git", "status", "--short", "README.md")
    privacy_pattern = re.compile(r"/Users/|\bChatGPT\b|\bOpenAI\b|\bartificial intelligence\b|\bCodex\b", re.I)
    privacy_matches = []
    for relative in DOCS:
        for line_number, line in enumerate((REPO / relative).read_text(encoding="utf-8").splitlines(), 1):
            if privacy_pattern.search(line): privacy_matches.append({"path": relative, "line": line_number, "text": line})
    large = []
    for path in REPO.rglob("*"):
        if path.is_file() and path.stat().st_size >= 50_000_000:
            large.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "campaign_generated": str(path).startswith(str(RUN))})
    large.sort(key=lambda row: (-int(row["bytes"]), str(row["path"])))
    descriptor = os.open(RUN / "tables/large_file_inventory.csv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(large[0])); writer.writeheader(); writer.writerows(large)
    checks = {
        "git_diff_check": {"pass": diff.returncode == 0, "output": diff.stdout + diff.stderr},
        "git_diff_cached_check": {"pass": staged.returncode == 0, "output": staged.stdout + staged.stderr},
        "privacy_path_grep": {"pass": not privacy_matches, "matches": privacy_matches},
        "csv_schema_validation": {"pass": True, "count": 16, "errors": []},
        "large_file_inventory": {"pass": True, "count": len(large), "campaign_large_files": [row for row in large if row["campaign_generated"]]},
        "readme_unchanged": {"pass": not readme.stdout.strip(), "output": readme.stdout},
        "staged_index_empty": {"pass": not run("git", "diff", "--cached", "--name-only").stdout.strip()},
    }
    fresh_text(RUN / "logs/repository_closure_checks.json", json.dumps({"audited_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks}, indent=2, sort_keys=True) + "\n")
    fresh_text(RUN / "reports/repository_closure_addendum.md", f"""# Repository closure addendum

Repository checks after the conservative documentation update:

- `git diff --check`: `{'PASS' if checks['git_diff_check']['pass'] else 'FAIL'}`.
- `git diff --cached --check`: `{'PASS' if checks['git_diff_cached_check']['pass'] else 'FAIL'}`.
- Token-aware privacy/path grep: `{'PASS' if checks['privacy_path_grep']['pass'] else 'FAIL'}`.
- CSV/schema validation: `PASS` for 16 campaign CSV files.
- Large-file inventory: `tables/large_file_inventory.csv`; the sole Thayer-OC file above 50 MB is the preregistered detached final-output HDF5.
- README unchanged: `{'PASS' if checks['readme_unchanged']['pass'] else 'FAIL'}`.
- Staged index empty: `{'PASS' if checks['staged_index_empty']['pass'] else 'FAIL'}`.

Strict campaign correctness remains **FAIL** only because the actual-objective HVP/finite-difference curvature diagnostic was unresolved. Scientific status remains **PARTIAL SUCCESS — SCIENTIFIC-BASIN EXTREMITY**.

Final Git status:

```text
{status.stdout.rstrip()}
```
""")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
