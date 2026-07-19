#!/usr/bin/env python3
"""Build the audited commit manifest from the Git index.

The index, rather than the working tree, is authoritative. Deletions fail
closed. The manifest omits its own byte hash because embedding a file's exact
SHA-256 inside that same file is self-referential; Git verifies that final blob
after commit.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "research_archive" / "commit_manifest.md"
OUTPUT_REL = OUTPUT.relative_to(ROOT).as_posix()


def git(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=True,
        text=not binary,
    )
    return result.stdout


def category(path: str) -> tuple[str, str]:
    suffix = Path(path).suffix.lower()
    if path == "README.md":
        return "documentation", "Canonical repository entry point."
    if path.startswith("src/"):
        return "source", "Scientific, solver, audit, or contract implementation."
    if path.startswith("scripts/"):
        return "script", "Campaign, replay, validation, or archive driver."
    if path.startswith("tests/"):
        return "test", "Unit, contract, replay, or artifact validation."
    if path.startswith("docs/experiment_archive/"):
        if suffix == ".png":
            return "selected_figure", "Compact central figure cleared by protection review."
        if suffix in {".csv", ".json"}:
            return "archive_table_or_manifest", "Compact claim-bearing archive evidence."
        if path.endswith("SOURCE_PROVENANCE.md"):
            return "archive_provenance", "Original/archive path, size, hash, authority, and supersession."
        return "experiment_archive", "Compact report, protocol, code, or archive index."
    if path.startswith("docs/research_archive/"):
        if suffix == ".csv":
            return "research_audit_table", "Machine-readable experiment or large-file inventory."
        return "research_archive", "Canonical authority, validation, data-use, or audit documentation."
    if path.startswith("docs/"):
        return "documentation", "Scientific protocol, contract, history, or interpretation."
    if path.startswith("reports/"):
        return "report", "Repository-level scientific authority required by code and archive provenance."
    if path.startswith("configs/"):
        return "config", "Portable experiment configuration."
    if path == ".gitignore":
        return "ignore_rule", "Protection rule for local scientific artifacts."
    return "other_reviewed", "Reviewed repository artifact in the proposed index."


def escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    status = str(git("diff", "--cached", "--name-status"))
    deleted = [line for line in status.splitlines() if line.startswith("D\t")]
    if deleted:
        raise RuntimeError(f"refusing to manifest staged deletions: {deleted}")

    raw = git(
        "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z",
        binary=True,
    )
    assert isinstance(raw, bytes)
    paths = sorted(
        path.decode("utf-8") for path in raw.split(b"\0") if path
    )
    paths = [path for path in paths if path != OUTPUT_REL]
    if not paths:
        raise RuntimeError("Git index has no proposed files")

    rows: list[tuple[str, str, int, str, str]] = []
    digest = hashlib.sha256()
    for path in paths:
        data = git("show", f":{path}", binary=True)
        assert isinstance(data, bytes)
        sha = hashlib.sha256(data).hexdigest()
        kind, reason = category(path)
        rows.append((path, kind, len(data), sha, reason))
        digest.update(path.encode("utf-8") + b"\0")
        digest.update(str(len(data)).encode("ascii") + b"\0")
        digest.update(sha.encode("ascii") + b"\n")

    counts: dict[str, int] = {}
    for _, kind, _, _, _ in rows:
        counts[kind] = counts.get(kind, 0) + 1
    lines = [
        "# Commit manifest",
        "",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Base HEAD: `{str(git('rev-parse', 'HEAD')).strip()}`",
        f"- Indexed files listed below: {len(rows)}",
        f"- Indexed bytes listed below: {sum(row[2] for row in rows)}",
        f"- Canonical staged-record SHA-256: `{digest.hexdigest()}`",
        "- Deletions: 0",
        "",
        "The table inventories every proposed committed file except this manifest itself. Embedding this file's exact SHA-256 in its own bytes is self-referential; after commit, Git supplies its final blob identity and the final handoff reports the complete commit statistics. No other path is exempt.",
        "",
        "## Category counts",
        "",
        "| Category | Files |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{kind}` | {count} |" for kind, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## File inventory",
            "",
            "| Path | Category | Bytes | SHA-256 | Reason included |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    lines.extend(
        f"| `{escape(path)}` | `{kind}` | {size} | `{sha}` | {escape(reason)} |"
        for path, kind, size, sha, reason in rows
    )
    lines.append("")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
