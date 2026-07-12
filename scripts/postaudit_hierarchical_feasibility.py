#!/usr/bin/env python3
"""Append-only corrections for final feasibility reporting semantics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


REPO = Path(__file__).resolve().parents[1]


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run = args.run_dir.resolve()
    privacy = json.loads((run / "diagnostics/privacy_path_grep.json").read_text())
    expected = {
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/logs/compileall.json",
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/logs/relevant_tests.json",
    }
    observed = set(privacy["absolute_user_path_hits"])
    if observed != expected or privacy["development_scene_files"] or privacy["lockbox_scene_files"] or privacy["sealed_lockbox_group_overlap"] != 0:
        raise RuntimeError("Privacy/lockbox result is not the expected command-path-only case")
    write_json_fresh(run / "diagnostics/privacy_path_grep_superseding_command_paths.json", {
        "status": "PASS_WITH_EXPECTED_COMMAND_PATHS", "supersedes": "diagnostics/privacy_path_grep.json",
        "absolute_user_path_hits": sorted(observed), "classification": "interpreter command paths recorded by test logs",
        "scientific_data_path_hits": [], "development_scene_files": [], "lockbox_scene_files": [],
        "sealed_lockbox_group_overlap": 0, "lockbox_access": False,
    })
    audit = json.loads((run / "diagnostics/final_correctness_audit.json").read_text())
    audit["zero_lockbox_access"] = True
    audit["privacy_path_grep_only_expected_command_paths"] = True
    audit["supersedes"] = "diagnostics/final_correctness_audit.json"
    audit["correction"] = "separate benign interpreter command paths from lockbox scene/group access"
    audit["status"] = "PASS" if all(value for key, value in audit.items() if key not in {"status", "supersedes", "correction"} and isinstance(value, bool)) else "FAIL"
    write_json_fresh(run / "diagnostics/final_correctness_audit_superseding_lockbox_semantics.json", audit)
    addendum = """# Authoritative final-report addendum

This append-only addendum supersedes two statements in `reports/final_report.md`; all other measurements and the **PARTIAL SUCCESS** classification remain unchanged.

## Catastrophic-valid component decision

The catastrophic-valid head achieved five-seed validation AUROC/AUPRC 0.987/0.997 at prevalence 0.8165 and natural-calibration AUROC/AUPRC 0.987/0.997. It materially exceeds the prior 0.654 AUROC and is scientifically highly rankable.

Nevertheless, the frozen preregistration required AUPRC at least `1.25 × prevalence`, which is 1.0206 here and therefore exceeds the mathematical maximum AUPRC of 1.0. The gate may not be changed after seeing results. The authoritative component decision is consequently **FAIL under a defective preregistered AUPRC gate**, not PASS. This is a prospective gate-design defect, not a model-performance failure. Overall remains **PARTIAL SUCCESS**.

Therefore answers 17–19 are corrected:

- Components: query/image/flux/centroid/confusion PASS; catastrophic-valid FAIL under the frozen impossible AUPRC gate; calibration PARTIAL.
- A future full hierarchical-policy campaign is **not yet justified**.
- The one next experiment is a separately preregistered train/validation/calibration-only conditional-calibration correction. It must preflight every gate for attainability before hashing, use a bounded prevalence-adjusted AP lift such as `(AUPRC - prevalence) / (1 - prevalence)`, keep Condition C and all heads frozen, calibrate image/flux residuals across frozen SNR/overlap groups with partial pooling, and require both attainable ranking gates and 85–95% coverage with bounded 95th-percentile width in every subgroup. It still must not use development or lockbox data.

## Correctness-audit semantics

The original final audit set `zero_lockbox_access=false` only because it reused the aggregate privacy/path status. The only path hits were the absolute `.venv-btk` interpreter commands recorded in compile/test logs. Direct checks found zero development files, zero lockbox files, zero sealed-lockbox group overlap, and every campaign access marker false. The superseding correctness audit is **PASS** and correctly records zero lockbox access.
"""
    write_text_fresh(run / "reports/final_report_addendum.md", addendum)
    write_json_fresh(run / "logs/postaudit_complete.json", {
        "status": audit["status"], "classification": "PARTIAL SUCCESS", "catastrophic_component": "FAIL_UNDER_IMPOSSIBLE_FROZEN_AUPRC_GATE",
        "corrected_zero_lockbox_access": True, "development_accessed": False, "lockbox_accessed": False,
        "thresholds_changed": False, "models_changed": False, "calibration_changed": False, "completed_at_unix": time.time(),
    })
    print(json.dumps({"classification": "PARTIAL SUCCESS", "audit": audit["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
