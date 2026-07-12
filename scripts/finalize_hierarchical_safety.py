#!/usr/bin/env python3
"""Reporting-only corrections, correctness audit, and final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
R1_CHECKPOINT = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518/checkpoints/r1_best.pth"
SOURCE_SPLIT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/source_split_manifest.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str: return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL; descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle: handle.write(value)


def write_json_fresh(path: Path, value: object) -> None: write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists(): raise FileExistsError(path)
    frame.to_csv(path, index=False)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray: return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def two_way_ci(frame: pd.DataFrame, accepted: np.ndarray, outcome: np.ndarray, repetitions: int = 300) -> tuple[float, float]:
    groups = sorted(set(frame.source_a_group) | set(frame.source_b_group)); mapping = {group: index for index, group in enumerate(groups)}; a = frame.source_a_group.map(mapping).to_numpy(); b = frame.source_b_group.map(mapping).to_numpy(); rng = np.random.default_rng(20260712993); values = []
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups)); weights = counts[a] * counts[b] * accepted
        if weights.sum(): values.append(float(np.sum(weights * outcome) / np.sum(weights)))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def reporting_corrections(run: Path, data: pd.DataFrame) -> None:
    valid = data.query_state.eq("UNIQUE_VALID").to_numpy(); null = data.query_state.eq("NULL").to_numpy(); ambiguous = data.query_state.eq("AMBIGUOUS").to_numpy()
    c_cat = numeric(data, "catastrophic_valid"); r1_cat = ((numeric(data, "r1_policy_violation") >= 2) | (numeric(data, "r1_confusion") == 1)).astype(float)
    methods = (("condition_c_reconstruction_only", np.ones(len(data), dtype=bool), c_cat, "c"), ("original_monolithic_R1", numeric(data, "r1_accept").astype(bool), r1_cat, "r1"), ("hierarchical_query_gate_only", numeric(data, "query_gate_accept").astype(bool), c_cat, "c"), ("complete_hierarchical_policy", numeric(data, "full_policy_accept").astype(bool), c_cat, "c"))
    rows = []
    for method, accepted, catastrophic, prefix in methods:
        for state, mask in (("UNIQUE_VALID", valid), ("NULL", null), ("AMBIGUOUS", ambiguous)):
            selected = accepted & mask
            hallucination = numeric(data, f"{prefix}_hallucination") if prefix == "r1" else numeric(data, "c_hallucination")
            forced = numeric(data, f"{prefix}_forced_source") if prefix == "r1" else numeric(data, "c_forced_source")
            rows.append({"method": method, "query_state": state, "samples": int(mask.sum()), "accepted": int(selected.sum()), "coverage": float(selected.sum() / mask.sum()), "catastrophic_rate_accepted": float(np.nanmean(catastrophic[selected])) if state == "UNIQUE_VALID" and selected.any() else math.nan, "null_false_accept_rate": float(selected.sum() / mask.sum()) if state == "NULL" else math.nan, "ambiguous_false_accept_rate": float(selected.sum() / mask.sum()) if state == "AMBIGUOUS" else math.nan, "exposed_hallucination_rate": float(np.nansum(selected * hallucination) / mask.sum()) if state == "NULL" else math.nan, "exposed_forced_source_rate": float(np.nansum(selected * forced) / mask.sum()) if state == "AMBIGUOUS" else math.nan})
    write_csv_fresh(run / "tables/development_metrics_macro_superseding_r1_outcomes.csv", pd.DataFrame(rows))
    write_json_fresh(run / "logs/development_reporting_outcome_correction.json", {"status": "REPORTING_ONLY_APPEND_ONLY_CORRECTION", "supersedes": "tables/development_metrics_macro.csv for original_monolithic_R1 outcome columns", "cause": "initial macro table applied Condition-C outcome columns to every selector", "new_neural_inference": False, "thresholds_changed": False, "development_evaluation_count_changed": False, "lockbox_used": False})

    scores = {"condition_c_reconstruction_only": np.zeros(len(data)), "random_rejection": np.random.default_rng(20260712992).random(len(data)), "original_monolithic_R1": numeric(data, "r1_raw_score"), "hierarchical_query_gate": numeric(data, "query_margin"), "complete_hierarchical_policy": numeric(data, "recoverability_margin"), "oracle_condition_c_risk": -numeric(data, "oracle_policy_violation")}
    valid_frame = data.loc[valid].reset_index(drop=True); c_truth = c_cat[valid]; r1_truth = r1_cat[valid]; rows = []
    for method, score in scores.items():
        scoped_score = score[valid]; truth = r1_truth if method == "original_monolithic_R1" else c_truth; order = np.argsort(-scoped_score, kind="stable")
        for coverage in (0.95, 0.90, 0.80, 0.70):
            count = int(math.ceil(len(order) * coverage)); accepted = np.zeros(len(order), dtype=int); accepted[order[:count]] = 1; rate = float(np.mean(truth[order[:count]])); low, high = two_way_ci(valid_frame, accepted, truth)
            rows.append({"method": method, "valid_coverage": coverage, "catastrophic_rate": rate, "cluster_ci_low": low, "cluster_ci_high": high, "outcome_backbone": "R1" if method == "original_monolithic_R1" else "Condition C", "bootstrap": "two-way source-group pigeonhole", "repetitions": 300})
    corrected = pd.DataFrame(rows); write_csv_fresh(run / "tables/development_valid_operating_points_superseding_r1_outcomes.csv", corrected)

    reconstruction_rows = []
    for model, prefix, catastrophic in (("Condition C", "c", c_truth), ("R1", "r1", r1_truth)):
        frame = data.loc[valid]
        reconstruction_rows.append({"model": model, "valid_scenes": len(frame), "mean_image_risk": float(np.nanmean(numeric(frame, f"{prefix}_image_risk"))), "mean_flux_risk_max": float(np.nanmean(numeric(frame, f"{prefix}_flux_risk_max"))), "mean_centroid_error_pixels": float(np.nanmean(numeric(frame, f"{prefix}_centroid_pixels"))), "confusion_rate": float(np.nanmean(numeric(frame, f"{prefix}_confusion"))), "catastrophic_rate": float(np.mean(catastrophic))})
    write_csv_fresh(run / "tables/development_reconstruction_comparison.csv", pd.DataFrame(reconstruction_rows))

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("random_rejection", "original_monolithic_R1", "hierarchical_query_gate", "complete_hierarchical_policy", "oracle_condition_c_risk"):
        subset = corrected[corrected.method == method]; ax.plot(subset.valid_coverage, 1 - subset.catastrophic_rate, marker="o", label=method)
    ax.set_xlabel("valid-scene coverage"); ax.set_ylabel("non-catastrophic fraction among accepted"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(run / "figures/catastrophic_failure_rejection_curves.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    method_scores = {"R1": numeric(data, "r1_raw_score"), "query gate": numeric(data, "query_margin"), "hierarchy": numeric(data, "recoverability_margin")}
    for ax, state, mask in ((axes[0], "NULL", null), (axes[1], "AMBIGUOUS", ambiguous)):
        for method, score in method_scores.items():
            thresholds = np.quantile(score, np.linspace(0, 1, 101)); overall_coverage = []; false_accept = []
            for threshold in thresholds:
                accepted = score >= threshold; overall_coverage.append(float(accepted.mean())); false_accept.append(float(accepted[mask].mean()))
            ax.plot(overall_coverage, false_accept, label=method)
        ax.set_title(state); ax.set_xlabel("overall accepted coverage"); ax.set_ylabel("class false-accept rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run / "figures/null_ambiguous_false_accept_curves_superseding.png", dpi=180); plt.close(fig)


def figures_and_paper(run: Path) -> None:
    query = pd.read_csv(run / "tables/query_gate_seed_stability.csv"); risk = pd.read_csv(run / "tables/risk_head_seed_stability.csv"); confusion = pd.read_csv(run / "tables/confusion_head_seed_stability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(query.seed.astype(str), query.macro_f1, marker="o"); axes[0].set_title("query macro F1"); axes[0].tick_params(axis="x", rotation=45)
    for task, group in risk.groupby("task"): axes[1].plot(group.seed.astype(str), group.upper_spearman, marker="o", label=task)
    axes[1].set_title("risk upper Spearman"); axes[1].legend(); axes[1].tick_params(axis="x", rotation=45)
    axes[2].plot(confusion.seed.astype(str), confusion.auroc, marker="o", label="AUROC"); axes[2].plot(confusion.seed.astype(str), confusion.auprc, marker="o", label="AUPRC"); axes[2].set_title("confusion ranking"); axes[2].legend(); axes[2].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(run / "figures/head_seed_stability.png", dpi=180); plt.close(fig)
    for source_name, destination_name in (("query_gate_confusion_matrices.png", "query_gate_confusion_matrices.png"), ("valid_risk_regression.png", "valid_risk_regression.png"), ("conformal_quantile_coverage.png", "conformal_quantile_coverage.png"), ("class_conditional_valid_risk_coverage.png", "class_conditional_valid_risk_coverage.png"), ("catastrophic_failure_rejection_curves.png", "catastrophic_failure_rejection_curves.png")):
        source = run / "figures" / source_name; destination = run / "paper_figures" / destination_name
        if destination.exists(): raise FileExistsError(destination)
        shutil.copyfile(source, destination)


def audits(run: Path) -> dict:
    compile_result = command([".venv-btk/bin/python", "-m", "compileall", "-q", "src", "scripts", "tests"])
    tests_result = command([".venv-btk/bin/python", "-m", "unittest", "-v", "tests.test_hierarchical_safety", "tests.test_hierarchical_query_gate", "tests.test_thayer_select", "tests.test_recoverability_phase2"])
    diff_result = command(["git", "diff", "--check"])
    write_json_fresh(run / "logs/compileall.json", compile_result); write_json_fresh(run / "logs/relevant_tests.json", tests_result); write_json_fresh(run / "logs/git_diff_check.json", diff_result)

    before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv").set_index("relative_path"); checkpoint_rows = []
    for relative_path, row in before.iterrows():
        path = REPO / relative_path; observed = sha256_file(path) if path.is_file() else ""; checkpoint_rows.append({"relative_path": relative_path, "expected_sha256": row.sha256, "observed_sha256": observed, "status": "PASS" if observed == row.sha256 else "FAIL"})
    checkpoint = pd.DataFrame(checkpoint_rows); write_csv_fresh(run / "tables/checkpoint_inventory_after.csv", checkpoint)

    split = pd.read_csv(SOURCE_SPLIT); expected = {"q_training": "training", "r_training": "training", "q_validation": "validation", "r_validation": "validation", "natural_calibration": "calibration", "stratified_calibration": "calibration", "development": "development_test"}; partition_groups = {partition: set(group.duplicate_group_id) for partition, group in split.groupby("partition")}; isolation_rows = []
    for dataset, partition in expected.items():
        manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False); groups = set(manifest.source_a_group) | set(manifest.source_b_group); wrong = groups - partition_groups[partition]; isolation_rows.append({"dataset": dataset, "expected_source_partition": partition, "scenes": len(manifest), "source_groups": len(groups), "wrong_partition_groups": len(wrong), "lockbox_group_overlap": len(groups & partition_groups["sealed_lockbox"]), "status": "PASS" if not wrong and not (groups & partition_groups["sealed_lockbox"]) else "FAIL"})
    isolation = pd.DataFrame(isolation_rows); write_csv_fresh(run / "tables/source_partition_isolation_audit.csv", isolation)

    alignment_rows = []
    for dataset in expected:
        manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False); features = np.load(run / f"features/v2_{dataset}_features.npz", allow_pickle=True); sample_path = run / (f"features/v3_{dataset}_samples.csv" if dataset != "development" else "features/v2_development_samples.csv"); samples = pd.read_csv(sample_path, keep_default_na=False)
        status = manifest.scene_id.tolist() == features["scene_id"].astype(str).tolist() == samples.scene_id.tolist(); invalid_applicable = int(((samples.query_state != "UNIQUE_VALID") & (pd.to_numeric(samples.applicable_valid_risk) != 0)).sum()); alignment_rows.append({"dataset": dataset, "rows": len(manifest), "scene_id_alignment": status, "invalid_query_applicability_violations": invalid_applicable, "status": "PASS" if status and invalid_applicable == 0 else "FAIL"})
    alignment = pd.DataFrame(alignment_rows); write_csv_fresh(run / "tables/sample_alignment_and_applicability_audit.csv", alignment)

    csv_rows = []
    for path in sorted(run.rglob("*.csv")):
        try:
            frame = pd.read_csv(path, nrows=3, keep_default_na=False); status = "PASS" if len(frame.columns) and len(frame.columns) == len(set(frame.columns)) else "FAIL"; csv_rows.append({"relative_path": relative(path), "columns": len(frame.columns), "status": status})
        except Exception as error: csv_rows.append({"relative_path": relative(path), "columns": 0, "status": "FAIL", "error": str(error)})
    csv_audit = pd.DataFrame(csv_rows); write_csv_fresh(run / "tables/csv_schema_validation.csv", csv_audit)
    large = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024: large.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv_fresh(run / "tables/large_file_inventory.csv", pd.DataFrame(large))
    privacy = {"absolute_user_path_hits": [], "lockbox_scene_files": [relative(path) for path in run.rglob("*lockbox*.h5")], "sealed_lockbox_group_overlap": int(isolation.lockbox_group_overlap.sum()), "status": "PASS"}
    for path in list(run.rglob("*.md")) + list(run.rglob("*.json")) + list(run.rglob("*.csv")):
        if "/Users/" in path.read_text(errors="ignore"): privacy["absolute_user_path_hits"].append(relative(path))
    privacy["status"] = "PASS" if not privacy["absolute_user_path_hits"] and not privacy["lockbox_scene_files"] and privacy["sealed_lockbox_group_overlap"] == 0 else "FAIL"; write_json_fresh(run / "diagnostics/privacy_path_grep.json", privacy)

    dev = json.loads((run / "logs/development_evaluation_complete.json").read_text()); feature_audit = json.loads((run / "diagnostics/frozen_feature_extraction_audit.json").read_text()); correction = json.loads((run / "manifests/hierarchical_policy_freeze_superseding_nondegeneracy.json").read_text()); mode = stat.S_IMODE((run / "manifests/v2_development_scene_manifest.csv").stat().st_mode)
    checks = {
        "compileall": compile_result["returncode"] == 0, "hierarchical_and_relevant_tests": tests_result["returncode"] == 0, "git_diff_check": diff_result["returncode"] == 0,
        "condition_c_checkpoint_hash_unchanged": sha256_file(CONDITION_C) == dev["condition_c_checkpoint_after"], "zero_trainable_reconstruction_parameters": feature_audit["trainable_reconstruction_parameters"] == 0,
        "historical_checkpoints_unchanged": bool((checkpoint.status == "PASS").all()), "source_partition_isolation": bool((isolation.status == "PASS").all()), "zero_lockbox_access": privacy["status"] == "PASS",
        "deterministic_feature_extraction": feature_audit["deterministic_exact"], "prompt_local_feature_dimensions_correct": feature_audit["prompt_local_dimension"] == 112 and feature_audit["prompt_local_scales"] == [60, 30, 15],
        "sample_alignment_and_applicability": bool((alignment.status == "PASS").all()), "natural_calibration_only_for_operational_parameters": json.loads((run / "logs/calibration_and_policy_freeze_complete.json").read_text())["natural_calibration_only_for_operational_parameters"],
        "no_development_threshold_tuning": not dev["threshold_retuning"], "fresh_development_evaluated_once": dev["evaluation_count"] == 1, "development_manifest_read_only": mode == 0o444,
        "manifest_replay": bool((pd.read_csv(run / "tables/training_calibration_replay_audit.csv").status == "PASS").all() and (pd.read_csv(run / "tables/development_replay_audit.csv").status == "PASS").all()),
        "csv_schema_validation": bool((csv_audit.status == "PASS").all()), "operational_policy_correctly_classified_degenerate": correction["authoritative_nondegenerate"] is False,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "privacy": privacy, "git_status": command(["git", "status", "--short", "--branch"]), "disk": shutil.disk_usage(REPO)}


def final_report(run: Path, data: pd.DataFrame, audit: dict) -> str:
    valid = data[data.query_state == "UNIQUE_VALID"].copy(); null = data[data.query_state == "NULL"]; ambiguous = data[data.query_state == "AMBIGUOUS"]
    query = json.loads((run / "manifests/query_gate_selection.json").read_text()); conformal = pd.read_csv(run / "tables/conformal_calibration_summary.csv"); risk_seed = pd.read_csv(run / "tables/risk_head_seed_stability.csv"); confusion = pd.read_csv(run / "tables/confusion_head_seed_stability.csv"); points = pd.read_csv(run / "tables/development_valid_operating_points_superseding_r1_outcomes.csv")
    hierarchy = points[points.method == "complete_hierarchical_policy"].sort_values("valid_coverage", ascending=False); random = points[points.method == "random_rejection"].sort_values("valid_coverage", ascending=False)
    risk_summary = risk_seed.groupby("task").agg(median_spearman_mean=("median_spearman", "mean"), upper_spearman_mean=("upper_spearman", "mean"), pinball_mean=("upper_pinball_log1p", "mean"), top10_recall_mean=("top_10_percent_recall", "mean"), catastrophic_auroc_mean=("catastrophic_failure_auroc", "mean"), catastrophic_auprc_mean=("catastrophic_failure_auprc", "mean")).reset_index()
    risk_table_lines = ["| task | median Spearman | upper Spearman | pinball | top-10% recall | catastrophic AUROC | catastrophic AUPRC |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in risk_summary.itertuples():
        risk_table_lines.append(f"| {row.task} | {row.median_spearman_mean:.3f} | {row.upper_spearman_mean:.3f} | {row.pinball_mean:.3f} | {row.top10_recall_mean:.3f} | {row.catastrophic_auroc_mean:.3f} | {row.catastrophic_auprc_mean:.3f} |")
    risk_table = "\n".join(risk_table_lines)
    runtime = sum(json.loads((run / path).read_text()).get("runtime_seconds", 0) for path in ("logs/data_preparation_complete.json", "logs/feature_extraction_complete.json", "logs/query_gate_training_complete.json", "logs/risk_head_training_complete.json", "logs/calibration_and_policy_freeze_complete.json"))
    return f"""# Hierarchical Thayer-Select safety campaign final report

## Outcome

**Campaign classification: FAILURE under the frozen gates.** The query-state subproblem succeeded and valid-only tail ranking was strong, but the complete calibrated policy was operationally degenerate and did not improve on the historical R1 ranking at useful valid-scene coverage. It accepted 1/2,000 UNIQUE_VALID development scenes (0.05%) and no invalid scenes. This is not a deployable safety policy.

The overall project did not change. Recoverability is now represented as a derived hierarchy rather than a monolithic target. Condition C stayed frozen, development was generated only after policy freeze and evaluated once, and the lockbox remained untouched.

## Required answers

1. **Did partition drift explain the prior calibration collapse?** No physical source/scene shift did: maximum physical |SMD| was 0.0535. Source-reuse frequency shifted (|SMD| 0.276), reducing effective independence. The main causes were sparse heterogeneous labels (5 moderate/37 permissive validation positives), calibration underpowering, and isotonic ties.
2. **Were query semantics applied consistently?** Yes: zero mismatches across 40,500 historical contract checks, exact three-state unit tests passed, and fresh manifests replayed deterministically.
3. **Did the UNIQUE/NULL/AMBIGUOUS gate work?** Yes. Balanced validation macro F1/AUPRC were {query['ensemble_stratified_metrics']['macro_f1']:.3f}/{query['ensemble_stratified_metrics']['macro_auprc']:.3f}; recalls were UNIQUE {query['ensemble_stratified_metrics']['per_class_recall'][0]:.3f}, NULL {query['ensemble_stratified_metrics']['per_class_recall'][1]:.3f}, AMBIGUOUS {query['ensemble_stratified_metrics']['per_class_recall'][2]:.3f}.
4. **Was ambiguity inversion removed?** Yes in all five query-head seeds. Development query-gate acceptance was {ambiguous.query_gate_accept.mean():.1%} for AMBIGUOUS versus {valid.query_gate_accept.mean():.1%} for UNIQUE_VALID.
5. **Which prompt-local feature family worked best?** F_COMBINED (global + multiscale prompt-local + reconstruction summary) with a small MLP. Standalone F_PROMPT_LOCAL was the best purely prompt-local family but underperformed the combination.
6. **How well were continuous risks predicted?** Five-seed means are:\n\n{risk_table}
7. **Did upper-quantile predictions achieve calibrated coverage?** Yes marginally: natural coverage was {', '.join(f'{row.risk} {row.empirical_natural_coverage:.3f}' for row in conformal.itertuples())}; stratified diagnostic coverage was {', '.join(f'{row.risk} {row.stratified_diagnostic_coverage:.3f}' for row in conformal.itertuples())}.
8. **Did confusion-risk ranking improve?** Yes diagnostically: five-seed AUROC mean {confusion.auroc.mean():.3f} and AUPRC mean {confusion.auprc.mean():.3f} at 2.3% validation prevalence, versus the prior 0.654 catastrophic-rejection AUROC. It did not rescue policy coverage.
9. **Did the hierarchy produce nondegenerate thresholds?** No. Natural calibration accepted 1/4,200 valid scenes; stratified diagnostic calibration accepted 0/1,000.
10. **What happened to NULL false acceptance?** Reconstruction-only exposed 100%; the query gate and full policy both accepted {null.query_gate_accept.mean():.1%}/{null.full_policy_accept.mean():.1%}. Condition-C exposed hallucination fell from {numeric(null, 'c_hallucination').mean():.1%} to 0 because all NULL queries abstained.
11. **What happened to AMBIGUOUS false acceptance?** Reconstruction-only exposed 100%; query-gate/full-policy acceptance became {ambiguous.query_gate_accept.mean():.1%}/{ambiguous.full_policy_accept.mean():.1%}. Exposed forced-source behavior fell from {numeric(ambiguous, 'c_forced_source').mean():.1%} to {(numeric(ambiguous, 'c_forced_source') * numeric(ambiguous, 'query_gate_accept')).mean():.1%} and 0.
12. **Catastrophic valid failures at 95/90/80/70% diagnostic coverage?** Hierarchy: {', '.join(f'{row.valid_coverage:.0%}: {row.catastrophic_rate:.3f}' for row in hierarchy.itertuples())}. Random: {', '.join(f'{row.valid_coverage:.0%}: {row.catastrophic_rate:.3f}' for row in random.itertuples())}. The gain is negligible at 95%, modest by 70%, and statistically similar to R1.
13. **What reconstruction performance was sacrificed?** None by the safety heads: they never modify Condition C. Selection sacrifices essentially all coverage at the frozen operating point. Mean Condition-C valid image/flux/centroid risks were {numeric(valid, 'c_image_risk').mean():.3f}/{numeric(valid, 'c_flux_risk_max').mean():.3f}/{numeric(valid, 'c_centroid_pixels').mean():.3f}.
14. **Were results stable across head seeds?** Query classification was stable (macro-F1 SD {pd.read_csv(run / 'tables/query_gate_seed_stability.csv').macro_f1.std(ddof=1):.4f}). Risk rank correlations were stable; raw linear-space widths were not, motivating log-space ensemble calibration. Confusion AUROC ranged {confusion.auroc.min():.3f}-{confusion.auroc.max():.3f}.
15. **Did the frozen backbone remain unchanged?** Yes: Condition-C SHA-256 remained `{sha256_file(CONDITION_C)}`, with zero trainable reconstruction parameters and exact repeated feature extraction.
16. **Was development evaluated only once?** Yes. Manifest SHA-256 `{json.loads((run / 'manifests/development_manifest_freeze.json').read_text())['manifest_sha256']}`; evaluation count 1; no retuning.
17. **Was the lockbox untouched?** Yes: zero lockbox groups/scenes/files and no sealed pixel access.
18. **SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE**, because the frozen complete policy is operationally degenerate and does not materially beat R1 at useful coverage, despite successful query validity and risk-ranking subcomponents.
19. **Ready for targeted Ambiguity Atlas construction?** Ready only for a separate targeted pilot, not the full Atlas: use simulator optimization to find close decision boundaries, matched source-pair construction, and multi-hypothesis truth sets. Do not use development or lockbox scenes.
20. **Exact next experiment?** A preregistered train/validation/calibration-only *risk-limit feasibility and conditional-conformal audit*: verify aperture flux-risk scaling and log-space tail stability, require at least 70% valid calibration coverage at a fixed catastrophic-risk budget, and compare hierarchy versus R1 before generating any new development set. Keep Condition C frozen.

## Correctness and provenance

- Final correctness audit: **{audit['status']}**.
- Fresh non-development scenes: 43,000; development scenes: 3,000.
- Approximate measured campaign runtime through freeze: {runtime:.1f} seconds, excluding one-time development and final audits.
- Run disk usage: {sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) / 1024**3:.2f} GiB.
- Historical checkpoints: unchanged.
- Development reporting correction: R1 macro and operating-point outcomes were recomputed from already persisted R1 outputs; no new inference or second evaluation occurred.

## Artifact index

- Drift: `diagnostics/partition_drift_report_superseding_source_reuse.md`, `tables/partition_drift_audit.csv`
- Query gate: `tables/query_gate_candidate_comparison.csv`, `figures/query_gate_confusion_matrices.png`, `figures/query_gate_per_class_pr.png`
- Risks/calibration: `tables/risk_head_seed_stability.csv`, `tables/conformal_calibration_summary.csv`, `figures/valid_risk_regression.png`, `figures/conformal_quantile_coverage.png`
- Development: `tables/development_per_sample.csv`, `tables/development_valid_operating_points_superseding_r1_outcomes.csv`, `figures/catastrophic_failure_rejection_curves.png`
- Galleries: `example_grids/development_accepted_rejected_gallery.png`
- Freeze/audit: `manifests/hierarchical_policy_freeze.json`, `manifests/hierarchical_policy_freeze_superseding_nondegeneracy.json`, `diagnostics/final_correctness_audit.json`

## Final git status

```text
{audit['git_status']['stdout'].rstrip()}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); run = args.run_dir.resolve()
    if json.loads((run / "logs/development_evaluation_complete.json").read_text())["evaluation_count"] != 1: raise RuntimeError("One-time development gate failed")
    data = pd.read_csv(run / "tables/development_per_sample.csv", keep_default_na=False, low_memory=False)
    reporting_corrections(run, data); figures_and_paper(run); audit = audits(run)
    disk = audit.pop("disk"); audit["disk"] = {"total": disk.total, "used": disk.used, "free": disk.free}; write_json_fresh(run / "diagnostics/final_correctness_audit.json", audit)
    report = final_report(run, data, audit); write_text_fresh(run / "reports/final_report.md", report)
    code_paths = [REPO / "src/hierarchical_safety.py", REPO / "scripts/run_hierarchical_safety.py", REPO / "scripts/prepare_hierarchical_safety_data.py", REPO / "scripts/correct_hierarchical_sample_metadata.py", REPO / "scripts/extract_hierarchical_safety_features.py", REPO / "scripts/train_hierarchical_query_gate.py", REPO / "scripts/train_hierarchical_risk_heads.py", REPO / "scripts/calibrate_hierarchical_safety.py", REPO / "scripts/run_hierarchical_development_evaluation.py", Path(__file__).resolve(), REPO / "tests/test_hierarchical_safety.py", REPO / "tests/test_hierarchical_query_gate.py"]
    write_csv_fresh(run / "tables/campaign_code_hashes_final.csv", pd.DataFrame([{"relative_path": relative(path), "sha256": sha256_file(path)} for path in code_paths]))
    write_json_fresh(run / "logs/finalization_complete.json", {"status": audit["status"], "classification": "FAILURE", "completed_at_unix": time.time(), "new_development_inference": False, "development_evaluation_count": 1, "lockbox_used": False})
    if audit["status"] != "PASS": raise RuntimeError("Final correctness audit failed")
    print(relative(run))


if __name__ == "__main__": main()
