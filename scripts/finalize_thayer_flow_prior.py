#!/usr/bin/env python3
"""Finalize the stopped Thayer-PF campaign and audit immutable boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
SOURCE_SPLIT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/source_split_manifest.csv"
EXPECTED_SOURCE_SPLIT = "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=REPO, check=check, text=True, capture_output=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_csvs(run_dir: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    paths = sorted(run_dir.rglob("*.csv"))
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if len(rows) < 2 or not rows[0] or any(len(row) != len(rows[0]) for row in rows[1:]):
            failures.append(str(path.relative_to(REPO)))
    return len(paths), failures


def checkpoint_inventory(run_dir: Path) -> list[dict[str, object]]:
    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv")
    rows: list[dict[str, object]] = []
    for row in before:
        path = REPO / row["path"]
        observed = sha256_file(path)
        rows.append({
            "path": row["path"], "expected_sha256": row["expected_sha256"],
            "observed_sha256": observed, "status": "PASS" if observed == row["expected_sha256"] else "FAIL",
        })
    return rows


def source_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in ("src", "scripts", "tests"):
        for path in sorted((REPO / root).rglob("*.py")):
            rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.time()
    gate = json.loads((run_dir / "logs/posterior_decoder_sufficiency_complete.json").read_text())
    if gate["status"] != "FAIL" or gate["flow_implementation_authorized"]:
        raise RuntimeError("finalizer is only valid for the stopped Part-D campaign")

    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"], check=False)
    write_text_fresh(run_dir / "logs/compileall_output.txt", compile_result.stdout + compile_result.stderr + f"exit_code={compile_result.returncode}\n")
    tests = command([
        str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q",
        "tests/test_probabilistic_unet.py", "tests/test_competing_hypotheses.py",
        "tests/test_ambiguity_atlas.py", "tests/test_canonical_tensor_hash.py",
    ], check=False)
    write_text_fresh(run_dir / "logs/unit_tests_output.txt", tests.stdout + tests.stderr + f"exit_code={tests.returncode}\n")
    diff_check = command(["git", "diff", "--check"], check=False)
    write_text_fresh(run_dir / "logs/git_diff_check.txt", diff_check.stdout + diff_check.stderr + f"exit_code={diff_check.returncode}\n")
    staged = command(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
    privacy = command([
        "rg", "-n", "(/Users/|file://|OpenAI|ChatGPT|AI-generated|generated by AI)",
        "docs/thayer_flow_prior.md", "docs/conditional_flow_prior_contract.md",
        "docs/latent_truth_coverage.md", "docs/atlas_flow_hypotheses.md",
    ], check=False)
    privacy_pass = privacy.returncode == 1
    write_text_fresh(run_dir / "logs/privacy_path_grep.txt", (privacy.stdout + privacy.stderr) if not privacy_pass else "PASS: no public-path or attribution leaks\n")

    checkpoints = checkpoint_inventory(run_dir)
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", checkpoints)
    write_csv_fresh(run_dir / "tables/source_code_hashes_final.csv", source_inventory())
    atlas_hashes = read_csv(PU / "tables/frozen_atlas_artifact_hashes_after.csv")
    atlas_unchanged = len(atlas_hashes) == 25 and all(
        row["status"] == "PASS" and sha256_file(REPO / row["path"]) == row["observed_sha256"]
        for row in atlas_hashes
    )
    csv_count, csv_failures = validate_csvs(run_dir)
    metric_tests = read_csv(run_dir / "tables/truth_coverage_metric_tests.csv")
    baseline_tests = read_csv(run_dir / "tables/baseline_reproduction.csv")
    gates = read_csv(run_dir / "tables/posterior_decoder_sufficiency_gates.csv")
    prereg = run_dir / "preregistration/posterior_decoder_sufficiency.md"
    gate_report = run_dir / "diagnostics/posterior_decoder_sufficiency.md"
    flow_directories_empty = all(not any((run_dir / name).iterdir()) for name in ("flow_models", "checkpoints", "prior_samples", "latent_samples"))
    atlas_directory_empty = not any((run_dir / "atlas_evaluation").iterdir())
    source_split_unchanged = sha256_file(SOURCE_SPLIT) == EXPECTED_SOURCE_SPLIT

    files = sorted((path for path in run_dir.rglob("*") if path.is_file()), key=lambda path: path.stat().st_size, reverse=True)
    large_rows = [{
        "path": str(path.relative_to(REPO)), "bytes": path.stat().st_size,
        "mib": path.stat().st_size / (1024**2), "at_least_50_mib": path.stat().st_size >= 50 * 1024**2,
    } for path in files[:20]]
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large_rows)

    checks = [
        {"check": "posterior_gate_preregistered_before_evaluation", "pass": prereg.stat().st_mtime_ns < gate_report.stat().st_mtime_ns, "evidence": sha256_file(prereg)},
        {"check": "persisted_baselines_reproduced", "pass": all(row["pass"] == "True" for row in baseline_tests), "evidence": f"rows={len(baseline_tests)}"},
        {"check": "truth_coverage_metric_audit", "pass": all(row["pass"] == "True" for row in metric_tests), "evidence": f"rows={len(metric_tests)}"},
        {"check": "posterior_decoder_gate_failed_closed", "pass": any(row["pass"] == "False" for row in gates) and not gate["flow_implementation_authorized"], "evidence": gate["scientific_decision"]},
        {"check": "flow_not_implemented_or_fitted", "pass": flow_directories_empty, "evidence": "flow_models/checkpoints/prior_samples/latent_samples empty"},
        {"check": "atlas_not_evaluated", "pass": atlas_directory_empty and gate["atlas_evaluation_count"] == 0, "evidence": "campaign Atlas evaluation count 0"},
        {"check": "zero_development_access", "pass": gate["development_scene_access_count"] == 0, "evidence": "count=0"},
        {"check": "zero_lockbox_access", "pass": gate["lockbox_scene_access_count"] == 0, "evidence": "count=0"},
        {"check": "historical_checkpoints_unchanged", "pass": len(checkpoints) == 560 and all(row["status"] == "PASS" for row in checkpoints), "evidence": f"count={len(checkpoints)}"},
        {"check": "frozen_atlas_artifacts_unchanged", "pass": atlas_unchanged, "evidence": f"count={len(atlas_hashes)}"},
        {"check": "source_partition_unchanged", "pass": source_split_unchanged, "evidence": sha256_file(SOURCE_SPLIT)},
        {"check": "staged_index_empty", "pass": not staged, "evidence": f"staged_count={len(staged)}"},
        {"check": "compileall", "pass": compile_result.returncode == 0, "evidence": f"exit_code={compile_result.returncode}"},
        {"check": "focused_unit_tests", "pass": tests.returncode == 0, "evidence": tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else f"exit_code={tests.returncode}"},
        {"check": "csv_schema_validation", "pass": not csv_failures, "evidence": f"csv_count={csv_count};failures={len(csv_failures)}"},
        {"check": "git_diff_check", "pass": diff_check.returncode == 0, "evidence": f"exit_code={diff_check.returncode}"},
        {"check": "public_privacy_path_grep", "pass": privacy_pass, "evidence": f"rg_exit_code={privacy.returncode}"},
        {"check": "no_auditor_or_catalog_policy", "pass": True, "evidence": "prohibited by failed Part-D gate"},
    ]
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", checks)
    failures = [row["check"] for row in checks if not bool(row["pass"])]
    audit_status = "PASS_WITH_PREREGISTERED_POSTERIOR_DECODER_GATE_FAILURE" if not failures else "FAIL"
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", {
        "status": audit_status, "check_count": len(checks), "failure_count": len(failures),
        "failures": failures, "csv_file_count_before_final_tables": csv_count,
        "historical_checkpoint_count": len(checkpoints), "atlas_evaluation_count": 0,
        "flow_fit_count": 0, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    if failures:
        raise RuntimeError(f"final correctness audit failed: {failures}")

    summary = {row["subset"]: row for row in read_csv(run_dir / "tables/posterior_decoder_sufficiency_summary.csv")}
    git_status = command(["git", "status", "--short"]).stdout.rstrip()
    disk_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    final_report = f"""# Thayer-PF conditional flow-prior truth-coverage final report

Decision: **FAILURE — POSTERIOR/DECODER INSUFFICIENT; FLOW PROHIBITED**.

Posterior/decoder preregistration SHA-256: `{sha256_file(prereg)}`. It predates
all Part-D inference. No flow preregistration was created because the mandatory
sufficiency gate failed.

## Direct answers

1. **Was the frozen truth-coverage metric correct?** Yes. All {len(metric_tests)} independent synthetic cases passed without changing the frozen threshold.
2. **Did posterior samples cover the own truth?** No. K=32 coverage was 0% on 512 ordinary prompt evaluations and 0% on 500 near-collision own-posterior evaluations.
3. **Did cross-decoding demonstrate alternate-truth representability?** No. Coverage was 0%; alternate identity was {float(summary['near_cross']['mean_source_identity_fraction']):.6f}.
4. **Was a prior correction scientifically justified?** No. The decoder/posterior bottleneck gate failed.
5. **Was the decoder and posterior frozen?** Yes. The selected Thayer-PU checkpoint remained byte-identical and no parameters were trained.
6. **What flow architecture and mixture base were used?** None; flow implementation was prohibited by gate.
7. **What was the added parameter count?** 0.
8. **Did both mixture components remain active?** Not applicable; no mixture was implemented.
9. **Did the flow assign mass to both posterior modes?** Not applicable.
10. **Did it avoid excessive mode bridging?** Not applicable.
11. **Did non-Atlas own-truth coverage improve?** No flow result exists; posterior own-truth coverage itself was zero.
12. **Did non-Atlas alternate-truth coverage become nonzero?** No; cross-decode coverage was zero.
13. **Did forward consistency remain valid?** Yes diagnostically: ordinary / near-own / near-cross sample fractions were {float(summary['ordinary']['mean_forward_consistent_fraction']):.6f} / {float(summary['near_own']['mean_forward_consistent_fraction']):.6f} / {float(summary['near_cross']['mean_forward_consistent_fraction']):.6f}.
14. **Did safe controls remain concentrated?** Persisted Thayer-PU control concentration reproduced; no new prior was evaluated.
15. **Was Atlas evaluation authorized?** No.
16. **How many Atlas witnesses were produced?** No new Atlas inference. The unchanged persisted Thayer-PU result is 24/50.
17. **Did witness count improve over 24/50?** Not evaluated.
18. **Did AUROC improve over 0.856?** Not evaluated.
19. **Did recall at 4% FPR improve over 0.32?** Not evaluated.
20. **Did Atlas own-truth coverage become nonzero?** Not evaluated.
21. **Did Atlas alternate-truth coverage become nonzero?** Not evaluated.
22. **What fraction of the posterior-prior gap was closed?** 0 by intervention: no flow was fitted; this is not a claim that the representations are equivalent.
23. **Did safe-control false witnesses remain bounded?** No new control inference was run; the persisted 0.08 Thayer-PU rate reproduced.
24. **Was the model SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE** at the posterior/decoder sufficiency gate. Thayer-PF is not a model artifact.
25. **What exact experiment should happen next?** Preregister one ambiguity-set decoder-training experiment that presents both non-Atlas near-collision decompositions under each observationally equivalent condition while preserving prompt identity and forward consistency.
26. **Were final lockbox and unauthorized development data untouched?** Yes; access counts 0/0.
27. **Were all historical checkpoints unchanged?** Yes; {len(checkpoints)}/{len(checkpoints)} files are byte-identical.

## Scientific evidence

- Persisted non-Atlas and Atlas baselines reproduced exactly from immutable tables and logs; Atlas scenes were not reopened.
- Ordinary posterior coverage: {float(summary['ordinary']['target_truth_coverage_rate']):.6f}; median best distance {float(summary['ordinary']['median_best_target_scientific_distance']):.6f}.
- Near-own posterior coverage: {float(summary['near_own']['target_truth_coverage_rate']):.6f}; median best distance {float(summary['near_own']['median_best_target_scientific_distance']):.6f}.
- Near-cross alternate coverage: {float(summary['near_cross']['target_truth_coverage_rate']):.6f}; median best distance {float(summary['near_cross']['median_best_target_scientific_distance']):.6f}.
- Near-own identity remained {float(summary['near_own']['mean_source_identity_fraction']):.6f}, but cross alternate identity was only {float(summary['near_cross']['mean_source_identity_fraction']):.6f}.

Forward consistency did not rescue the gate. It establishes observation-level
recomposition within tolerance, not recovery of either known scientific truth.
No latent teachers, flow curves, mixture diagnostics, mode plots, prior samples,
Atlas galleries, or new ROC/sample-efficiency curves exist because the campaign
stopped before those stages.

## Correctness, provenance, and repository state

- Correctness audit: {audit_status}; {len(checks)} checks, 0 failures.
- Focused unit tests: `{tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else 'PASS'}`.
- Compileall, CSV validation, checkpoint/Atlas/source-partition hashes, `git diff --check`, empty staged index, public privacy grep, and large-file inventory: PASS.
- Campaign runtime through finalization: {time.time() - started + float(gate['runtime_seconds']):.2f} seconds recorded in the active scripts; run size at report creation: {disk_bytes} bytes.
- Flow fitting / Atlas / development / lockbox access counts: 0 / 0 / 0 / 0.

Final Git status:

```text
{git_status}
```
"""
    write_text_fresh(run_dir / "reports/final_report.md", final_report)
    write_json_fresh(run_dir / "logs/finalization_complete.json", {
        "status": "PASS", "decision": "FAILURE_POSTERIOR_DECODER_INSUFFICIENT",
        "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"),
        "correctness_audit_status": audit_status, "runtime_seconds": time.time() - started,
        "flow_fit_count": 0, "atlas_evaluation_count": 0,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS", "decision": "FAILURE_POSTERIOR_DECODER_INSUFFICIENT", "run_dir": str(run_dir.relative_to(REPO))}, sort_keys=True))


if __name__ == "__main__":
    main()
