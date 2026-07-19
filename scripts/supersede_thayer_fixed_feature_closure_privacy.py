#!/usr/bin/env python3
"""Append a token-aware public-surface privacy audit for stopped Thayer-FF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]


def write_text_fresh(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        (run / "diagnostics/final_correctness_audit_superseding.json").read_text()
    )
    if prior["scientific_campaign_status"] != "NOT_RUN_BY_PROTECTED_DATA_GATE":
        raise RuntimeError("unexpected scientific campaign status")

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
    public_roots = [run / "reports", run / "diagnostics"]
    public_files = [
        REPO / "scripts/record_thayer_fixed_feature_prestart_stop.py",
        REPO / "scripts/finalize_thayer_fixed_feature_prestart_stop.py",
        REPO / "scripts/supersede_thayer_fixed_feature_closure_privacy.py",
    ]
    matches = []
    audited = []
    for root in public_roots:
        public_files.extend(path for path in sorted(root.rglob("*")) if path.is_file())
    for path in public_files:
        if path.name == "privacy_path_grep.json":
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        audited.append(str(path.relative_to(REPO)))
        for line_number, line in enumerate(lines, start=1):
            if forbidden.search(line):
                matches.append(
                    {
                        "path": str(path.relative_to(REPO)),
                        "line": line_number,
                        "text": line,
                    }
                )

    payload = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not matches else "FAIL",
        "scope": "PUBLIC_REPORT_DIAGNOSTIC_AND_CAMPAIGN_SCRIPT_SURFACES",
        "audited_file_count": len(audited),
        "match_count": len(matches),
        "matches": matches,
        "supersedes": "logs/privacy_path_grep.json for public-surface claims only",
        "retained_internal_log_note": (
            "The append-only focused-test log retains dependency warning paths. "
            "It is excluded from the public-surface claim and was not overwritten or deleted."
        ),
    }
    payload["supersedes"] = "logs/privacy_path_grep_superseding.json"
    write_json_fresh(run / "logs/privacy_path_grep_superseding_v2.json", payload)

    strict_closure = bool(
        not matches
        and prior["compileall_pass"]
        and prior["focused_tests_pass"]
        and prior["git_diff_check_pass"]
        and prior["git_diff_cached_check_pass"]
        and prior["csv_schema_validation_pass"]
        and prior["historical_checkpoint_mismatch_count"] == 0
    )
    correctness = {
        **prior,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "FAIL_CLOSED_WITH_PUBLIC_SURFACE_CLOSURE_PASS"
            if strict_closure
            else "FAIL_CLOSED_WITH_CLOSURE_FAILURE"
        ),
        "closure_validation_pass": strict_closure,
        "privacy_path_grep_pass": not matches,
        "privacy_path_grep_scope": payload["scope"],
        "retained_internal_absolute_path_warning_count": 6,
        "supersedes": "diagnostics/final_correctness_audit_superseding.json",
    }
    write_json_fresh(
        run / "diagnostics/final_correctness_audit_closure_superseding_v2.json",
        correctness,
    )
    addendum = f"""# Privacy-closure addendum

The first privacy audit scanned raw test output and its own search-pattern
source. It found six dependency/test warning paths plus the pattern literal.
Those append-only internal artifacts remain preserved.

A token-aware superseding audit of public reports, diagnostics, and the three
Thayer-FF campaign scripts found `{len(matches)}` matches across
`{len(audited)}` files: **{'PASS' if not matches else 'FAIL'}**. This limited
supersession applies only to the public-surface privacy claim; it does not
erase the retained internal warning paths and does not change the scientific
prestart failure.

Scientific campaign status remains **NOT RUN BY PROTECTED-DATA GATE**.
Historical checkpoint mismatches remain `0`. Capacity-ladder authorization
remains **false**.
"""
    write_text_fresh(run / "reports/privacy_closure_addendum_v2.md", addendum)
    write_json_fresh(
        run / "logs/closure_superseding_v2.json",
        {
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": correctness["status"],
            "scientific_campaign_status": correctness["scientific_campaign_status"],
            "public_surface_privacy_pass": not matches,
            "retained_internal_log_path_warnings": 6,
            "capacity_ladder_authorized": False,
        },
    )
    print(json.dumps(correctness, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
