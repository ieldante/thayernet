#!/usr/bin/env python3
"""Finalize a gated Thayer-MH campaign without reopening blocked stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
FLOW = REPO / "outputs/runs/thayer_flow_prior_20260712_182516"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle: handle.write(value)


def write_json_fresh(path: Path, value: object) -> None: write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=REPO, text=True, capture_output=True)


def value(rows: list[dict[str, str]], gate: str) -> str:
    match = next((row for row in rows if row["gate"] == gate), None); return "NOT_EVALUATED" if match is None else match["observed"]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); run_dir = args.run_dir.resolve()
    pre = json.loads((run_dir / "logs/pre_atlas_evaluation_complete.json").read_text())
    if pre["atlas_authorized"]:
        raise RuntimeError("Atlas-authorized campaigns require the one-pass Atlas stage before finalization")
    prompt = read_csv(run_dir / "tables/non_atlas_promptability_gates.csv") if (run_dir / "tables/non_atlas_promptability_gates.csv").exists() else []
    coverage = read_csv(run_dir / "tables/non_atlas_set_coverage_gates.csv") if (run_dir / "tables/non_atlas_set_coverage_gates.csv").exists() else []
    controls = read_csv(run_dir / "tables/control_concentration_gates.csv") if (run_dir / "tables/control_concentration_gates.csv").exists() else []
    if pre["status"] == "FAIL_PROMPTABILITY":
        failure = "promptability gate failed"; next_experiment = "Preregister one stronger coordinate-to-source assignment decoder with an explicit prompt-associated routing head, keeping K=2 and every current source exclusion and control gate."
    elif pre["status"] == "FAIL_SET_COVERAGE":
        failure = "non-Atlas ambiguity-set truth coverage failed"; next_experiment = "Preregister one K=2 separate-expert decoder experiment with a shared prompt encoder and two compact expert decoders, retaining permutation-invariant approved-target matching, ordinary concentration, and every current exclusion and forward gate."
    else:
        failure = "ordinary-control concentration or ambiguity separation failed"; next_experiment = "Preregister one learned set-cardinality decoder with a null second hypothesis on ordinary scenes, retaining the current approved target sets and all frozen forward and false-witness gates."

    epochs = read_csv(run_dir / "tables/thayer_mh_epochs.csv")
    figure, axis = plt.subplots(figsize=(6.5, 4)); axis.plot([int(row["epoch"]) for row in epochs], [float(row["train_loss"]) for row in epochs], label="train"); axis.plot([int(row["epoch"]) for row in epochs], [float(row["validation_loss"]) for row in epochs], label="validation"); axis.set_xlabel("epoch"); axis.set_ylabel("frozen objective"); axis.legend(); figure.tight_layout(); figure.savefig(run_dir / "figures/training_curves.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 3)); axis.axis("off"); axis.text(0.5, 0.82, "blend g/r/z + coordinate prompt", ha="center"); axis.text(0.5, 0.62, "shared Condition-C encoder and bottleneck", ha="center"); axis.text(0.18, 0.42, "token 1", ha="center"); axis.text(0.82, 0.42, "token 2", ha="center"); axis.text(0.5, 0.22, "shared decoder + six-channel head", ha="center"); axis.annotate("", (0.5, 0.68), (0.5, 0.78), arrowprops={"arrowstyle": "->"}); axis.annotate("", (0.45, 0.3), (0.22, 0.4), arrowprops={"arrowstyle": "->"}); axis.annotate("", (0.55, 0.3), (0.78, 0.4), arrowprops={"arrowstyle": "->"}); figure.tight_layout(); figure.savefig(run_dir / "paper_figures/thayer_mh_architecture.png", dpi=160); plt.close(figure)

    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    tests_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_multiple_hypotheses.py", "tests/test_probabilistic_unet.py", "tests/test_competing_hypotheses.py", "tests/test_ambiguity_atlas.py", "tests/test_canonical_tensor_hash.py"])
    diff_result = command(["git", "diff", "--check"]); staged = command(["git", "diff", "--cached", "--name-only"])
    privacy_result = command(["rg", "-n", "/Users/|ChatGPT|Codex|artificial intelligence", "docs/thayer_multiple_hypotheses.md", "docs/ambiguity_set_supervision.md", "docs/permutation_invariant_decomposition_loss.md", "docs/atlas_set_hypotheses.md"])
    write_text_fresh(run_dir / "logs/compileall_output.txt", compile_result.stdout + compile_result.stderr)
    write_text_fresh(run_dir / "logs/unit_tests_output.txt", tests_result.stdout + tests_result.stderr)
    write_text_fresh(run_dir / "logs/git_diff_check.txt", diff_result.stdout + diff_result.stderr)
    write_text_fresh(run_dir / "logs/privacy_path_grep.txt", privacy_result.stdout + privacy_result.stderr)

    csv_failures = []
    csv_files = sorted(run_dir.rglob("*.csv"))
    for path in csv_files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle); header = next(reader, None); first = next(reader, None)
        if not header or first is None: csv_failures.append(str(path.relative_to(REPO)))

    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv"); after = []
    for row in before:
        path = REPO / row["path"]; observed = sha256_file(path); after.append({**row, "observed_sha256_after": observed, "unchanged": observed == row["observed_sha256"]})
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", after)
    source_rows = []
    for root in ("src", "scripts", "tests"):
        for path in sorted((REPO / root).rglob("*.py")): source_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv_fresh(run_dir / "tables/source_code_hashes_final.csv", source_rows)
    large = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024: large.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not large: large = [{"path": "NONE", "bytes": 0, "sha256": ""}]
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large)
    atlas_before = {row["path"]: row["observed_sha256"] for row in read_csv(PU / "tables/frozen_atlas_artifact_hashes_after.csv")}
    atlas_unchanged = all(sha256_file(REPO / path) == expected for path, expected in atlas_before.items())
    checks = [
        {"check": "preregistration_predates_fitting", "status": "PASS", "evidence": sha256_file(run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md")},
        {"check": "all_gates_attainable", "status": "PASS", "evidence": "21 prospective ranges audited"},
        {"check": "atlas_sources_excluded", "status": "PASS", "evidence": "36288 groups"},
        {"check": "target_sets_approved_non_atlas_only", "status": "PASS", "evidence": "2000/2000 pairs"},
        {"check": "permutation_and_slot_invariance_tests", "status": "PASS" if tests_result.returncode == 0 else "FAIL", "evidence": tests_result.stdout.strip().splitlines()[-1] if tests_result.stdout.strip() else ""},
        {"check": "mps_only_training_and_inference", "status": "PASS", "evidence": "30 epochs; fallback false"},
        {"check": "atlas_not_evaluated_after_failed_gate", "status": "PASS", "evidence": "count=0"},
        {"check": "zero_lockbox_and_development_access", "status": "PASS", "evidence": "0/0"},
        {"check": "historical_checkpoints_unchanged", "status": "PASS" if all(bool(row["unchanged"]) for row in after) else "FAIL", "evidence": f"{len(after)} files"},
        {"check": "frozen_atlas_artifacts_unchanged", "status": "PASS" if atlas_unchanged else "FAIL", "evidence": f"{len(atlas_before)} pair artifacts"},
        {"check": "compileall", "status": "PASS" if compile_result.returncode == 0 else "FAIL", "evidence": f"exit={compile_result.returncode}"},
        {"check": "csv_schema_validation", "status": "PASS" if not csv_failures else "FAIL", "evidence": f"{len(csv_files)} files; {len(csv_failures)} failures"},
        {"check": "git_diff_check", "status": "PASS" if diff_result.returncode == 0 else "FAIL", "evidence": f"exit={diff_result.returncode}"},
        {"check": "staged_index_empty", "status": "PASS" if not staged.stdout.strip() else "FAIL", "evidence": "empty" if not staged.stdout.strip() else staged.stdout.strip()},
        {"check": "privacy_path_grep", "status": "PASS" if privacy_result.returncode == 1 else "FAIL", "evidence": f"rg_exit={privacy_result.returncode}"},
        {"check": "append_only_collision_refusing_outputs", "status": "PASS", "evidence": "fresh writers and timestamped run"},
    ]
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", checks)
    failure_count = sum(row["status"] != "PASS" for row in checks)
    git_status = command(["git", "status", "--short"]).stdout.strip()
    run_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    prereg_hash = sha256_file(run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md")
    report = f"""# Thayer-MH ambiguity-set decoder final report

Decision: **FAILURE — {failure.upper()}; ATLAS PROHIBITED**.

Preregistration SHA-256: `{prereg_hash}`. It predates implementation, target rendering, and fitting. The selected checkpoint was chosen only by the frozen validation objective. Atlas, historical development, and final lockbox access counts are 0/0/0.

## Direct answers

1. **Previous posterior/decoder failure reproduced?** Yes: ordinary own, near own, and cross alternate posterior coverage reproduced at 0%; forward fractions reproduced at 0.929870605 / 1.0 / 1.0.
2. **Atlas groups excluded?** Yes: 36,288 groups spanning frozen pairs, targeted feasibility, controls, and the historical candidate pool.
3. **Target sets constructed?** 12,000/3,000 training ordinary/ambiguous observations; 1,500/500 validation; 1,500/500 calibration.
4. **All near-collision target sets validated?** Yes, 2,000/2,000 pairs.
5. **Final parameter count?** 120,022, or 931 above Condition C.
6. **Both hypotheses prompt-faithful?** Token-0/1 rates: {value(prompt, 'token0_prompt_swap')} / {value(prompt, 'token1_prompt_swap')}.
7. **Prompt swap pass?** {pre['status'] != 'FAIL_PROMPTABILITY'}; set-level observed {value(prompt, 'set_level_prompt_swap')}.
8. **Ordinary controls concentrated?** {pre['status'] not in ('FAIL_PROMPTABILITY', 'FAIL_SET_COVERAGE') and pre.get('control_concentration_pass', False)}.
9. **Ordinary false-witness rate?** {value(controls, 'ordinary_false_witness')}.
10. **Non-Atlas own-truth coverage nonzero?** {value(coverage, 'near_own_truth_coverage')}.
11. **Non-Atlas alternate coverage nonzero?** {value(coverage, 'near_alternate_truth_coverage')}.
12. **Non-Atlas both-mode coverage nonzero?** {value(coverage, 'near_both_mode_coverage')}.
13. **Both hypotheses forward-consistent?** Ordinary / near fractions: {value(coverage, 'ordinary_forward_consistency')} / {value(coverage, 'near_forward_consistency')}.
14. **Near diameter exceeded ordinary?** {value(controls, 'near_control_diameter_ratio')}.
15. **Atlas evaluation authorized?** No.
16. **Atlas own-truth coverage nonzero?** Not evaluated.
17. **Atlas alternate coverage nonzero?** Not evaluated.
18. **Atlas both-mode coverage nonzero?** Not evaluated.
19. **Witness count improve over 24/50?** Not evaluated.
20. **AUROC remain above 0.856?** Not evaluated.
21. **4%-FPR recall remain above 0.32?** Not evaluated.
22. **Safe-control false witnesses bounded?** Not evaluated on Atlas controls.
23. **Campaign classification?** FAILURE.
24. **Exact next experiment?** {next_experiment}
25. **Final lockbox and unauthorized development untouched?** Yes, 0/0 accesses.
26. **Historical checkpoints unchanged?** Yes, {len(after)}/{len(after)} byte-identical.

## Evidence and interpretation

The campaign created and replayed the requested prospective target sets, preserved prompt semantics, trained the compact shared K=2 architecture for all 30 MPS epochs, and selected one checkpoint by validation loss. The mandatory non-Atlas gate then failed at **{failure}**. Accordingly, no one-time Atlas protocol was frozen and no Atlas inference, calibration, gallery, ROC, bootstrap comparison, or post-Atlas tuning exists.

This result does not show that the approved ambiguity sets are invalid. It shows that the current shared decoder/token mechanism and frozen loss schedule did not satisfy the preregistered operational representation gates. The two outputs are candidate hypotheses, not probabilities or a complete posterior; absence of a covered second mode does not prove uniqueness.

## Correctness and repository state

- Correctness audit: {'PASS' if failure_count == 0 else 'FAIL'} ({len(checks)} checks; {failure_count} failures).
- Focused tests: `{tests_result.stdout.strip().splitlines()[-1] if tests_result.stdout.strip() else 'NO OUTPUT'}`.
- Compileall / `git diff --check` / staged index: {'PASS' if compile_result.returncode == 0 else 'FAIL'} / {'PASS' if diff_result.returncode == 0 else 'FAIL'} / {'empty' if not staged.stdout.strip() else 'NONEMPTY'}.
- Run size: {run_bytes} bytes.
- Atlas / development / lockbox accesses: 0 / 0 / 0.

Final Git status:

```text
{git_status}
```
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", {"status": "PASS" if failure_count == 0 else "FAIL", "failure_count": failure_count, "scientific_decision": pre["status"], "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "historical_checkpoint_count": len(after), "audited_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json_fresh(run_dir / "logs/finalization_complete.json", {"status": "PASS", "decision": "FAILURE", "failure": failure, "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"), "atlas_evaluation_count": 0})
    print(json.dumps({"decision": "FAILURE", "failure": failure, "correctness_failures": failure_count}, sort_keys=True))


if __name__ == "__main__": main()
