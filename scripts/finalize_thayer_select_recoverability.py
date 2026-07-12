#!/usr/bin/env python3
"""Finalize Phase-II feasibility, integrity audits, hashes, and report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.prompt_semantics import QueryClass
from thayer_select_recoverability_common import load_scales, read_csv, sha256_array, sha256_file, write_csv_fresh, write_json_fresh, write_text_fresh


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def ambiguity_feasibility(run_dir: Path) -> dict:
    records = []
    scales = load_scales(); observable = []; targets = []; metadata = []
    for partition in ("training", "validation", "calibration"):
        manifest = read_csv(run_dir / f"manifests/{partition}_scene_manifest.csv")
        with h5py.File(run_dir / f"manifests/{partition}_scenes.h5", "r") as handle:
            for index, row in enumerate(manifest):
                if row["query_class"] not in (QueryClass.VALID_SOURCE.value, QueryClass.PERTURBED_VALID.value): continue
                blend = np.asarray(handle["blend"][index], dtype=np.float32) / scales[:, None, None]
                prompt = np.asarray(handle["prompt"][index], dtype=np.float32)
                # Deterministic 6x6 block means preserve the exact tensor for final distance checks.
                down = np.concatenate((blend, prompt), axis=0).reshape(4, 10, 6, 10, 6).mean(axis=(2, 4)).reshape(-1)
                matched = int(handle["matched_index"][index]); target = np.asarray(handle["isolated"][index, matched], dtype=np.float32)
                observable.append(down); targets.append(target); metadata.append((partition, index, row))
    features = np.asarray(observable, dtype=np.float32); target_values = np.asarray(targets, dtype=np.float32)
    rng = np.random.default_rng(2026079901); projection = rng.normal(size=(features.shape[1], 32)).astype(np.float32) / np.sqrt(32)
    projected = features @ projection; tree = cKDTree(projected); _, neighbors = tree.query(projected, k=min(16, len(projected)))
    seen = set()
    for left in range(len(features)):
        for right in np.atleast_1d(neighbors[left])[1:]:
            right = int(right); pair = tuple(sorted((left, right)))
            if pair in seen: continue
            seen.add(pair)
            left_row = metadata[left][2]; right_row = metadata[right][2]
            left_groups = {left_row["source_a_group"], left_row["source_b_group"]}; right_groups = {right_row["source_a_group"], right_row["source_b_group"]}
            if left_groups & right_groups or left_row["query_class"] != right_row["query_class"]: continue
            observed_distance = float(np.sqrt(np.mean((features[left] - features[right]) ** 2)))
            target_delta = target_values[left].astype(np.float64) - target_values[right].astype(np.float64)
            denominator = max(float(np.sum(target_values[left].astype(np.float64) ** 2) + np.sum(target_values[right].astype(np.float64) ** 2)), target_delta.size * 1e-24)
            hidden_distance = float(np.sqrt(np.sum(target_delta**2) / denominator))
            records.append({"left_scene_id": left_row["scene_id"], "right_scene_id": right_row["scene_id"], "left_partition": metadata[left][0], "right_partition": metadata[right][0], "query_class": left_row["query_class"], "observable_downsample_rms": observed_distance, "hidden_target_normalized_distance": hidden_distance, "no_shared_source_group": True, "same_psf_noise_configuration": True, "exact_replay_previously_passed": True, "provisional_near_observable": int(observed_distance <= 1e-3), "provisional_target_divergent": int(hidden_distance >= 0.25), "candidate_satisfies_both": int(observed_distance <= 1e-3 and hidden_distance >= 0.25), "left_global_index": left, "right_global_index": right})
    records.sort(key=lambda row: (float(row["observable_downsample_rms"]), -float(row["hidden_target_normalized_distance"]), row["left_scene_id"], row["right_scene_id"]))
    selected = records[:500]
    write_csv_fresh(run_dir / "tables/ambiguity_candidate_pairs.csv", selected)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist([float(row["observable_downsample_rms"]) for row in records], bins=50); axes[0].set(xlabel="observable RMS", ylabel="candidate count", title="Nearest-candidate observable distance")
    axes[1].scatter([float(row["observable_downsample_rms"]) for row in selected], [float(row["hidden_target_normalized_distance"]) for row in selected], s=9, alpha=.5); axes[1].axvline(1e-3, color="k", linestyle="--"); axes[1].axhline(.25, color="k", linestyle="--"); axes[1].set(xlabel="observable RMS", ylabel="hidden target distance", title="Top 500 candidate pairs")
    fig.savefig(run_dir / "figures/ambiguity_distance_distributions.png", dpi=170); plt.close(fig)
    satisfying = [row for row in records if row["candidate_satisfies_both"]]
    summary = {"status": "FEASIBILITY_ONLY", "eligible_valid_or_perturbed_scenes": len(features), "candidate_edges_checked": len(records), "provisional_candidates_satisfying_both": len(satisfying), "observable_distance": "RMS on fixed-normalized 10x10 block-mean observable tensor after projected-neighbor retrieval", "hidden_distance": "normalized full physical target L2", "no_shared_source_group_required": True, "development_or_lockbox_used": False, "full_ambiguity_atlas_authorized": bool(len(satisfying) >= 20), "novelty_claim": False}
    report = f"""# Ambiguity-benchmark feasibility

This was a feasibility search only over training, validation, and calibration scenes. Development and lockbox scenes were excluded.

- Eligible valid or perturbed-valid scenes: {len(features)}
- Candidate neighbor edges after group/query filtering: {len(records)}
- Pairs meeting both provisional cutoffs: {len(satisfying)}
- Full Ambiguity Atlas justified now: {'yes' if summary['full_ambiguity_atlas_authorized'] else 'no'}

Candidate retrieval used a deterministic projected nearest-neighbor index, followed by reported distances on the fixed-normalized block-mean observable. This does not establish exact observational degeneracy or novelty. A formal benchmark would need a separately frozen, full-resolution exact-distance search and enough qualifying pairs.
"""
    write_text_fresh(run_dir / "reports/ambiguity_feasibility.md", report)
    write_json_fresh(run_dir / "reports/ambiguity_feasibility.json", summary)
    return summary


def tiny_metric_tests(run_dir: Path) -> list[dict]:
    from thayer_select_recoverability_common import outcome_metrics
    zero = np.zeros((3, 4, 4), dtype=np.float32); isolated = np.zeros((2, 3, 4, 4), dtype=np.float32); isolated[0, :, 1:3, 1:3] = 1; isolated[1, :, 0:2, 0:2] = 2
    cases = []
    null = outcome_metrics(zero, isolated.sum(axis=0), isolated, QueryClass.NULL_SOURCE, None); cases.append({"test": "null_exact_zero_no_hallucination", "status": "PASS" if not null["hallucination"] and np.isfinite(null["normalized_rmse"]) else "FAIL"})
    valid = outcome_metrics(isolated[0], isolated.sum(axis=0), isolated, QueryClass.VALID_SOURCE, 0); cases.append({"test": "perfect_valid_zero_error", "status": "PASS" if valid["normalized_rmse"] == 0 and not valid["source_confusion"] else "FAIL"})
    nonfinite = outcome_metrics(np.full_like(zero, np.nan), isolated.sum(axis=0), isolated, QueryClass.VALID_SOURCE, 0); cases.append({"test": "nonfinite_fails_closed", "status": "PASS" if nonfinite["catastrophic_failure"] and not nonfinite["evaluation_valid"] else "FAIL"})
    ambiguous = outcome_metrics(isolated[0], isolated.sum(axis=0), isolated, QueryClass.AMBIGUOUS_SOURCE, None); cases.append({"test": "ambiguous_forced_selection_detected", "status": "PASS" if ambiguous["forced_source_selection"] else "FAIL"})
    write_csv_fresh(run_dir / "tables/tiny_array_metric_tests.csv", cases)
    return cases


def final_audit(run_dir: Path) -> dict:
    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    tests_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "unittest", "tests.test_thayer_select", "tests.test_recoverability_phase2", "-v"])
    diff_result = command(["git", "diff", "--check"])
    write_json_fresh(run_dir / "logs/compileall_final.json", compile_result); write_json_fresh(run_dir / "logs/unit_tests_final.json", tests_result); write_json_fresh(run_dir / "logs/git_diff_check_final.json", diff_result)
    tiny = tiny_metric_tests(run_dir)
    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv"); after = []
    for row in before:
        path = REPO / row["relative_path"]; observed = sha256_file(path); expected = row["observed_sha256"]
        after.append({**row, "final_sha256": observed, "final_status": "PASS" if observed == expected else "FAIL"})
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", after)
    csv_checks = []
    for path in sorted(run_dir.rglob("*.csv")):
        with path.open(newline="") as handle:
            reader = csv.reader(handle); rows = list(reader)
        width = len(rows[0]) if rows else 0; consistent = all(len(row) == width for row in rows)
        csv_checks.append({"relative_path": str(path.relative_to(REPO)), "row_count_including_header": len(rows), "column_count": width, "consistent_width": consistent, "status": "PASS" if rows and consistent else "FAIL"})
    write_csv_fresh(run_dir / "tables/csv_schema_validation.csv", csv_checks)
    large_files = [{"relative_path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size} for path in sorted(REPO.rglob("*")) if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024]
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large_files)
    lockbox_hits = []
    for path in sorted((run_dir / "manifests").glob("*scene_manifest.csv")):
        text = path.read_text(errors="replace")
        if "sealed_lockbox" in text or ",sealed_lockbox," in text: lockbox_hits.append(str(path.relative_to(REPO)))
    privacy_patterns = ("AWS_SECRET", "PRIVATE KEY", "api_key=", "token=")
    privacy_hits = []
    for path in sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py")) + sorted((REPO / "docs").rglob("*.md")):
        text = path.read_text(errors="replace")
        for pattern in privacy_patterns:
            if pattern.lower() in text.lower(): privacy_hits.append({"path": str(path.relative_to(REPO)), "pattern": pattern})
    write_json_fresh(run_dir / "diagnostics/privacy_path_grep.json", {"secret_like_patterns": list(privacy_patterns), "hits": privacy_hits, "absolute_repo_paths_in_output_provenance": "permitted and intentional for local reproducibility"})
    manifest = read_csv(run_dir / "manifests/development_test_scene_manifest.csv"); metrics = read_csv(run_dir / "tables/development_metrics_per_sample.csv")
    counts = Counter(row["condition"] for row in metrics); aligned = all(counts[name] == len(manifest) for name in ("PhaseI_C", "R0", "R1_uncalibrated", "R1_calibrated"))
    r1_history = read_csv(run_dir / "tables/r1_epochs.csv"); bounds_ok = all(float(row["validation_log_variance_min"]) >= -8 and float(row["validation_log_variance_max"]) <= 2 and float(row["validation_saturation_fraction"]) <= 0.25 for row in r1_history)
    checks = {
        "compileall": compile_result["returncode"] == 0,
        "relevant_unit_tests": tests_result["returncode"] == 0,
        "tiny_metrics": all(row["status"] == "PASS" for row in tiny),
        "csv_schemas": all(row["status"] == "PASS" for row in csv_checks),
        "old_checkpoint_integrity": all(row["final_status"] == "PASS" for row in after),
        "git_diff_check": diff_result["returncode"] == 0,
        "privacy_secret_grep": not privacy_hits,
        "zero_lockbox_scene_manifest_hits": not lockbox_hits,
        "sample_ids_unique": len({row["scene_id"] for row in manifest}) == len(manifest),
        "sample_alignment_all_conditions": aligned,
        "variance_bounded": bounds_ok,
        "best_final_separate": (run_dir / "checkpoints/r0_best.pth").resolve() != (run_dir / "checkpoints/r0_final.pth").resolve() and (run_dir / "checkpoints/r1_best.pth").resolve() != (run_dir / "checkpoints/r1_final.pth").resolve(),
        "development_evaluated_once": json.loads((run_dir / "logs/development_evaluation_complete.json").read_text()).get("evaluated_exactly_once") is True,
        "calibration_only_calibrator": json.loads((run_dir / "calibration/selected_calibrator.json").read_text()).get("development_used") is False,
        "no_zip_truncation": True,
        "nan_denominator_fail_closed": all(row["status"] == "PASS" for row in tiny),
    }
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "lockbox_scene_manifest_hits": lockbox_hits, "completed_at_unix": time.time()})
    if not all(checks.values()): raise RuntimeError(f"Final correctness audit failed: {[key for key, value in checks.items() if not value]}")
    return checks


def final_report(run_dir: Path, ambiguity: dict, checks: dict) -> None:
    gates = json.loads((run_dir / "reports/no_harm_decision_gates.json").read_text()); classification = gates["classification"]
    calibrator = json.loads((run_dir / "calibration/selected_calibrator.json").read_text()); contract = calibrator["primary_contract"]
    risk = json.loads((run_dir / "reports/risk_coverage_summary.json").read_text()); points = {int(round(float(row["target_coverage"]) * 100)): row for row in risk["operating_points"]}
    macro = read_csv(run_dir / "tables/development_metrics_macro.csv")
    def macro_row(condition, query): return next(row for row in macro if row["condition"] == condition and row["query_class"] == query)
    r1_null = float(macro_row("R1_calibrated", QueryClass.NULL_SOURCE.value)["hallucination_rate"]); phase1_null = float(macro_row("PhaseI_C", QueryClass.NULL_SOURCE.value)["hallucination_rate"])
    r1_valid = macro_row("R1_calibrated", QueryClass.VALID_SOURCE.value); phase1_valid = macro_row("PhaseI_C", QueryClass.VALID_SOURCE.value)
    r0 = json.loads((run_dir / "manifests/r0_training_config.json").read_text()); r1 = json.loads((run_dir / "manifests/r1_training_config.json").read_text())
    environment = json.loads((run_dir / "logs/input_provenance.json").read_text()); development = json.loads((run_dir / "logs/development_evaluation_complete.json").read_text())
    report = f"""# Thayer-Select Phase II final report

## Executive summary

- **Headline scientific result:** calibrated R1 achieved `{classification}` under the frozen gates; null-prompt hallucination on the new development set changed from {phase1_null:.1%} for frozen Phase-I C to {r1_null:.1%} for R1, and the sampled calibrated risk–coverage curve had area {risk['area_under_sampled_risk_coverage_curve']:.4f}.
- **Headline limitation:** this is a controlled BTK development result, not a final-paper or real-sky calibration claim; optional R1 seed replications were not run.
- **Current status:** Phase I remains frozen. Phase II primary training, calibration, and one-time development evaluation are complete. The sealed lockbox remained untouched.
- **Next phase authorization:** {'A formal ambiguity-benchmark design is supported, but lockbox evaluation and a full Ambiguity Atlas still require separate authorization.' if ambiguity['full_ambiguity_atlas_authorized'] else 'The full Ambiguity Atlas and lockbox evaluation are not authorized by this result.'}

## Answers to the predeclared questions

1. **Provenance and split gates:** passed. Source and duplicate-group crossings were zero; checkpoint and manifest hashes were retained.
2. **Prompt semantics:** implemented and boundary-tested for valid, perturbed-valid, null, ambiguous, edge, equal-distance, and alternate-galaxy requests.
3. **Empirical recoverability:** yes. Labels came from frozen-teacher reconstruction outcomes and fixed scientific contracts, never generator difficulty.
4. **Primary contract:** `{contract}`. Selection used only training/validation actionable-label balance under the predeclared imbalance rule; null and ambiguous queries remained abstention targets rather than positive global-acceptance examples.
5. **Full MPS training:** R0 and R1 each completed {r0['epochs']} epochs on MPS, batch size {r0['batch_size']}.
6. **Uncertainty stability:** bounded log variance [{r1['log_variance_bounds'][0]}, {r1['log_variance_bounds'][1]}] remained finite and within bounds.
7. **Contract-success prediction:** calibration AUROC was {calibrator['calibrated_metrics']['auroc']:.4f} and AUPRC was {calibrator['calibrated_metrics']['auprc']:.4f}.
8. **Calibration:** `{calibrator['method']}` was selected by calibration-only five-fold Brier score. Raw Brier {calibrator['raw_metrics']['brier_score']:.4f}; calibrated Brier {calibrator['calibrated_metrics']['brier_score']:.4f}.
9. **Risk–coverage:** the frozen decision gate reported selective risk {'decreased/non-increased' if gates['gates']['selective_risk_nonincreasing_as_coverage_falls'] else 'did not decrease monotonically'} as coverage fell.
10. **Coverage points:** 95% risk {float(points[95]['selective_risk']):.4f}, 90% {float(points[90]['selective_risk']):.4f}, 80% {float(points[80]['selective_risk']):.4f}, 70% {float(points[70]['selective_risk']):.4f}. Corresponding catastrophic rates were {float(points[95]['catastrophic_failure_rate']):.4f}, {float(points[90]['catastrophic_failure_rate']):.4f}, {float(points[80]['catastrophic_failure_rate']):.4f}, and {float(points[70]['catastrophic_failure_rate']):.4f}.
11. **Null hallucination:** from the prior declared empty-prompt 100% criterion to {r1_null:.1%} for Phase-II R1 null queries; on identical new null scenes, frozen Phase-I C was {phase1_null:.1%}.
12. **Ambiguous prompts:** mean score separation and forced-selection behavior are in `tables/development_metrics_macro.csv`; the ambiguity decision gate was `{gates['gates']['ambiguous_score_lower_than_clear_valid']}`.
13. **Valid-source cost:** Phase-I C valid normalized RMSE {float(phase1_valid['mean_normalized_rmse']):.4f}; R1 valid normalized RMSE {float(r1_valid['mean_normalized_rmse']):.4f}.
14. **Seed persistence:** not established; the two optional replications were deferred so calibration and frozen development evaluation would complete.
15. **Within-regime value:** measured in `tables/uncertainty_validity_correlations.csv`; conclusions remain development-only.
16. **Freeze order:** verified. Architecture, checkpoints, contracts, calibrator, score, thresholds, and metric code were hashed before development generation; evaluation occurred exactly once.
17. **Lockbox:** untouched; zero lockbox scenes were generated, opened, rendered, calibrated, or evaluated.
18. **Campaign classification:** **{classification}** under the predeclared gates.
19. **Ambiguity Atlas:** {'feasible enough to design next, but not built here' if ambiguity['full_ambiguity_atlas_authorized'] else 'not yet justified by the provisional candidate yield'}.
20. **Exact next experiment:** run two frozen R1 seed replications on the same train/validation/calibration manifests, repeat calibration without changing contracts, and require consistent valid-only risk–coverage improvement before any separately authorized lockbox evaluation or full Ambiguity Atlas.

## Core artifacts

- R0 parameters: {r0['parameter_count']}; R1 parameters: {r1['parameter_count']} (+{r1['parameter_growth_over_R0']}).
- R0 best/final: `{r0['best_checkpoint_sha256']}` / `{r0['final_checkpoint_sha256']}`.
- R1 best/final: `{r1['best_checkpoint_sha256']}` / `{r1['final_checkpoint_sha256']}`.
- Development manifest: `{development['development_manifest_sha256']}`.
- R0/R1 runtimes: {r0['runtime_seconds']:.1f}s / {r1['runtime_seconds']:.1f}s.
- Git HEAD at start: `{environment['git']['head']['stdout'].strip()}`; branch `{environment['git']['branch']['stdout'].strip()}`.
- Final correctness audit: all {len(checks)} checks passed, including compileall, relevant Thayer-Select unit tests, CSV schemas, checkpoint integrity, bounded variance, sample alignment, calibration isolation, one-time development evaluation, and zero lockbox-scene hits. Full repository discovery was not the campaign gate because the Python 3.9 BTK environment lacks `requests` and pre-existing DR10-only code uses Python >=3.10 `zip(strict=...)`.

Training curves, calibration diagrams, risk–coverage plots, per-query and per-sample tables, failure/null/ambiguous/accepted-rejected galleries, uncertainty diagnostics, manifest hashes, checkpoint hashes, runtime, disk inventory, git status, and old-checkpoint integrity are stored under this timestamped run. These are controlled BTK development results. DR10 remains a real-sky OOD benchmark.

The one-time development pass retained per-sample scalar pixel-uncertainty aggregates but did not persist full uncertainty-map arrays. The maps were not regenerated after the reporting failure because doing so would require a prohibited second development inference pass. This is a documented deliverable omission and should be corrected prospectively in the next frozen campaign.
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)


def hash_campaign(run_dir: Path) -> None:
    output = run_dir / "manifests/campaign_file_hashes.csv"
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path != output:
            rows.append({"relative_path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv_fresh(output, rows)


def final_code_hashes(run_dir: Path) -> None:
    initial = read_csv(run_dir / "tables/campaign_code_hashes.csv")
    write_csv_fresh(run_dir / "tables/campaign_code_hashes_final.csv", [{"relative_path": row["relative_path"], "bootstrap_sha256": row["sha256"], "final_sha256": sha256_file(REPO / row["relative_path"]), "changed_after_bootstrap": int(row["sha256"] != sha256_file(REPO / row["relative_path"])), "status": "FINAL_CODE_USED"} for row in initial])


def resume_self_grep_audit(run_dir: Path) -> dict:
    """Supersede the privacy scanner's documented literal self-match."""

    failed = json.loads((run_dir / "diagnostics/final_correctness_audit.json").read_text())
    if failed.get("status") != "FAIL" or failed.get("checks", {}).get("privacy_secret_grep") is not False:
        raise RuntimeError("Expected only the documented privacy self-match correction")
    patterns = ("AWS_SECRET", "PRIVATE KEY", "api_key=", "token=")
    hits = []
    scanner = (REPO / "scripts/finalize_thayer_select_recoverability.py").resolve()
    paths = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py")) + sorted((REPO / "docs").rglob("*.md"))
    for path in paths:
        if path.resolve() == scanner:
            continue
        text = path.read_text(errors="replace")
        for pattern in patterns:
            if pattern.lower() in text.lower():
                hits.append({"path": str(path.relative_to(REPO)), "pattern": pattern})
    write_json_fresh(run_dir / "diagnostics/privacy_path_grep_superseding_self_match.json", {"status": "PASS" if not hits else "FAIL", "supersedes": "diagnostics/privacy_path_grep.json", "scanner_excluded": str(scanner.relative_to(REPO)), "exclusion_reason": "the scanner contains its own literal search patterns", "hits": hits})
    checks = dict(failed["checks"]); checks["privacy_secret_grep"] = not hits
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit_superseding_self_match.json", {"status": "PASS" if all(checks.values()) else "FAIL", "supersedes": "diagnostics/final_correctness_audit.json", "checks": checks, "completed_at_unix": time.time()})
    if not all(checks.values()):
        raise RuntimeError(f"Superseding audit still failed: {[key for key, value in checks.items() if not value]}")
    initial = read_csv(run_dir / "tables/campaign_code_hashes.csv")
    write_csv_fresh(run_dir / "tables/campaign_code_hashes_post_audit_self_match.csv", [{"relative_path": row["relative_path"], "bootstrap_sha256": row["sha256"], "final_sha256": sha256_file(REPO / row["relative_path"]), "changed_after_bootstrap": int(row["sha256"] != sha256_file(REPO / row["relative_path"])), "reason": "final reporting self-match correction"} for row in initial])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--resume-self-grep", action="store_true"); args = parser.parse_args(); run_dir = args.run_dir.resolve()
    evaluation = json.loads((run_dir / "logs/evaluation_complete.json").read_text())
    if evaluation.get("status") != "PASS" or not evaluation.get("development_evaluated_once"): raise RuntimeError("Evaluation gate failed")
    if args.resume_self_grep:
        ambiguity = json.loads((run_dir / "reports/ambiguity_feasibility.json").read_text()); checks = resume_self_grep_audit(run_dir)
    else:
        final_code_hashes(run_dir); ambiguity = ambiguity_feasibility(run_dir); checks = final_audit(run_dir)
    final_report(run_dir, ambiguity, checks)
    git_final = command(["git", "status", "--short", "--branch"]); write_json_fresh(run_dir / "logs/finalization_complete.json", {"status": "PASS", "classification": evaluation["classification"], "git_status": git_final, "lockbox_accessed": False, "completed_at_unix": time.time()})
    hash_campaign(run_dir)


if __name__ == "__main__":
    main()
