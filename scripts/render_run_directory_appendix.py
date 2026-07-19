#!/usr/bin/env python3
"""Render the audited run ledger into the canonical map appendix.

This is a metadata-only renderer. It does not open scientific arrays or execute
campaign code.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "research_archive" / "experiment_ledger.csv"
MAP = ROOT / "docs" / "thayer_research_program_map.md"
BEGIN = "<!-- RUN_DIRECTORY_APPENDIX -->"
END = "<!-- END_RUN_DIRECTORY_APPENDIX -->"


def cell(value: str) -> str:
    """Escape a compact value for one Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", " ").strip()


def main() -> None:
    with LEDGER.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_rows = [row for row in rows if row["run_directory"].startswith("outputs/runs/")]
    if len(run_rows) != 124:
        raise RuntimeError(f"expected the frozen 124-run inventory, found {len(run_rows)}")

    text = MAP.read_text(encoding="utf-8")
    if BEGIN not in text:
        raise RuntimeError(f"missing appendix marker in {MAP}")
    prefix, tail = text.split(BEGIN, maxsplit=1)
    if END in tail:
        _, suffix = tail.split(END, maxsplit=1)
    else:
        suffix = ""

    lines = [
        BEGIN,
        "",
        "| Timestamp | Run directory | Type | Authority status | Supersession status | Git evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in run_rows:
        run_name = Path(row["run_directory"]).name
        lines.append(
            "| {timestamp} | `{run}` | `{kind}` | `{authority}` | `{supersession}` | `{commit}` |".format(
                timestamp=cell(row["timestamp"]),
                run=cell(run_name),
                kind=cell(row["campaign_type"]),
                authority=cell(row["authority_status"]),
                supersession=cell(row["supersession_status"]),
                commit=cell(row["commit_artifact"]),
            )
        )
    lines.extend(
        [
            "",
            "`REPORT_PRESENT_REVIEW_AUTHORITY_MATRIX` means a final-report locator exists; it does not make directory recency authoritative. `ENGINEERING_OR_INCOMPLETE_RECORD` means the run supplies engineering history or lacks a compact final scientific authority. Claim-level authority and grouped launch-attempt resolutions remain in the claim matrix and supersession ledger.",
            "",
            END,
        ]
    )
    MAP.write_text(prefix + "\n".join(lines) + suffix, encoding="utf-8")


if __name__ == "__main__":
    main()
