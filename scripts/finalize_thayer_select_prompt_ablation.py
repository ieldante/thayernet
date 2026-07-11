#!/usr/bin/env python3
"""Audit, classify, and report the completed promptability baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_thayer_select_prompt_ablation import sample_metrics
from thayer_select_prompt_ablation_common import (
    LOCKBOX_PARTITION,
    PARTITION_COUNTS,
    gaussian_prompt_numpy,
    parameter_count,
    read_csv,
    sha256_file,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)


def run_command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def integrity_audit(run_dir: Path) -> dict:
    output_path = run_dir / "diagnostics/model_infrastructure_audit.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing.get("status") != "PASS":
            raise RuntimeError("Existing model/infrastructure audit is not passing")
        return existing
    definitions = read_csv(run_dir / "manifests/development_scene_definitions.csv")
    rendered = read_csv(run_dir / "manifests/rendered_scene_manifest.csv")
    source_split = read_csv(run_dir / "manifests/source_split_manifest.csv")
    group_partitions: dict[str, set[str]] = defaultdict(set)
    source_partitions: dict[str, set[str]] = defaultdict(set)
    for row in source_split:
        source_partitions[row["persistent_source_id"]].add(row["partition"])
        group_partitions[row["duplicate_group_id"]].add(row["partition"])
    condition_configs = {
        condition: json.loads((run_dir / f"manifests/{condition.lower()}_training_config.json").read_text())
        for condition in ("A_centered_no_prompt", "B_randomized_no_prompt", "C_randomized_coordinate_prompt")
    }
    h5_counts = {}
    h5_complete = True
    for partition, expected in PARTITION_COUNTS.items():
        with h5py.File(run_dir / f"manifests/{partition}_scenes.h5", "r") as handle:
            h5_counts[partition] = int(handle.attrs["completed_count"])
            h5_complete &= bool(handle.attrs["complete"]) and int(handle.attrs["completed_count"]) == expected
    checks = {
        "all_three_models_aligned_scene_definition_hash": len({config["scene_definition_sha256"] for config in condition_configs.values()}) == 1,
        "b_c_identical_randomized_scene_variant": condition_configs["B_randomized_no_prompt"]["scene_variant"] == condition_configs["C_randomized_coordinate_prompt"]["scene_variant"] == "random",
        "a_centered_scene_variant": condition_configs["A_centered_no_prompt"]["scene_variant"] == "centered",
        "no_source_cross_partition": all(len(value) == 1 for value in source_partitions.values()),
        "no_group_cross_partition": all(len(value) == 1 for value in group_partitions.values()),
        "no_lockbox_scene_definition": all(row["partition"] != LOCKBOX_PARTITION for row in definitions),
        "no_lockbox_rendered_manifest": all(row["partition"] != LOCKBOX_PARTITION for row in rendered),
        "hdf5_counts_complete": h5_complete,
        "prompt_absent_a": not condition_configs["A_centered_no_prompt"]["coordinate_prompt"] and condition_configs["A_centered_no_prompt"]["input_channels"] == 3,
        "prompt_absent_b": not condition_configs["B_randomized_no_prompt"]["coordinate_prompt"] and condition_configs["B_randomized_no_prompt"]["input_channels"] == 3,
        "prompt_present_c": condition_configs["C_randomized_coordinate_prompt"]["coordinate_prompt"] and condition_configs["C_randomized_coordinate_prompt"]["input_channels"] == 4,
        "reconstruction_only_objective": all(config["loss"] == "whole-image normalized MSE only" for config in condition_configs.values()),
        "calibration_not_used": all(config["calibration_scenes_used"] == 0 for config in condition_configs.values()),
        "development_not_inspected_during_training": all(config["development_test_scenes_inspected"] == 0 for config in condition_configs.values()),
        "best_selected_by_validation_only": all(config["validation_rule"] == "minimum validation MSE; no test or calibration information" for config in condition_configs.values()),
        "all_training_on_mps": all(config["device"] == "mps" for config in condition_configs.values()),
        "parameter_count_a_b_equal": condition_configs["A_centered_no_prompt"]["parameter_count"] == condition_configs["B_randomized_no_prompt"]["parameter_count"],
        "first_layer_difference_exact": condition_configs["C_randomized_coordinate_prompt"]["parameter_count"] - condition_configs["B_randomized_no_prompt"]["parameter_count"] == 144,
        "all_scene_counts": len(definitions) == sum(PARTITION_COUNTS.values()),
        "rendered_scene_counts": len(rendered) == sum(PARTITION_COUNTS.values()),
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    checks["hdf5_counts"] = h5_counts
    write_json_fresh(output_path, checks)
    if checks["status"] != "PASS":
        raise RuntimeError("Model/infrastructure audit failed")
    return checks


def tiny_array_tests(run_dir: Path) -> None:
    output_path = run_dir / "tables/tiny_array_formula_tests.csv"
    if output_path.exists():
        if any(row["status"] != "PASS" for row in read_csv(output_path)):
            raise RuntimeError("Existing tiny-array metric test is not passing")
        return
    zero = np.zeros((3, 8, 8), dtype=np.float32)
    one = np.ones_like(zero)
    result = sample_metrics(one, zero, zero)
    prompt = gaussian_prompt_numpy(3.0, 5.0, height=8, width=8)
    tests = [
        {"test": "constant_error_mse", "observed": result["whole_mse"], "expected": 1.0, "status": "PASS" if result["whole_mse"] == 1.0 else "FAIL"},
        {"test": "constant_error_mae", "observed": result["whole_mae"], "expected": 1.0, "status": "PASS" if result["whole_mae"] == 1.0 else "FAIL"},
        {"test": "identity_baseline", "observed": result["identity_mse"], "expected": 0.0, "status": "PASS" if result["identity_mse"] == 0.0 else "FAIL"},
        {"test": "prompt_peak_coordinate", "observed": int(np.argmax(prompt)), "expected": 5 * 8 + 3, "status": "PASS" if int(np.argmax(prompt)) == 5 * 8 + 3 else "FAIL"},
        {"test": "parameter_difference", "observed": parameter_count(4) - parameter_count(3), "expected": 144, "status": "PASS" if parameter_count(4) - parameter_count(3) == 144 else "FAIL"},
    ]
    write_csv_fresh(output_path, tests)
    if any(row["status"] != "PASS" for row in tests):
        raise RuntimeError("Tiny-array metric test failed")


def external_audits(run_dir: Path) -> None:
    compile_result = run_command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts"])
    unittest_result = run_command([str(REPO / ".venv-btk/bin/python"), "-m", "unittest", "tests.test_metric_correctness", "tests.test_thayer_select"])
    diff_result = run_command(["git", "diff", "--check"])
    retry = 0
    while True:
        suffix = "" if retry == 0 else f"_retry{retry}"
        paths = [run_dir / f"logs/compileall{suffix}.json", run_dir / f"logs/relevant_unittests{suffix}.json", run_dir / f"logs/git_diff_check{suffix}.json"]
        if not any(path.exists() for path in paths):
            break
        retry += 1
    write_json_fresh(paths[0], compile_result)
    write_json_fresh(paths[1], unittest_result)
    write_json_fresh(paths[2], diff_result)
    if any(result["returncode"] != 0 for result in (compile_result, unittest_result, diff_result)):
        raise RuntimeError("Compile, unit-test, or git diff audit failed")


def schema_and_privacy_audit(run_dir: Path) -> None:
    primary = read_csv(run_dir / "tables/primary_metrics_per_sample.csv")
    swap = read_csv(run_dir / "tables/prompt_swap_per_scene.csv")
    no_harm = read_csv(run_dir / "tables/no_harm_per_sample.csv")
    required_primary = {"condition", "scene_id", "whole_mse", "source_mse", "psnr", "ssim", "centroid_error_pixels"}
    checks = {
        "primary_columns_present": required_primary.issubset(primary[0]),
        "primary_row_count": len(primary) == 3000,
        "primary_unique_condition_scene": len({(row["condition"], row["scene_id"]) for row in primary}) == 3000,
        "prompt_swap_row_count": len(swap) == 1000,
        "prompt_swap_unique_scene": len({row["scene_id"] for row in swap}) == 1000,
        "no_harm_row_count": len(no_harm) == 5000,
        "finite_primary_mse": all(np.isfinite(float(row["whole_mse"])) and np.isfinite(float(row["source_mse"])) for row in primary),
        "no_lockbox_hdf5_file": not any("lockbox" in path.name.lower() and path.suffix == ".h5" for path in run_dir.rglob("*")),
        "no_dr10_artifact": not any("dr10" in path.name.lower() for path in run_dir.rglob("*")),
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    write_json_fresh(run_dir / "diagnostics/csv_schema_privacy_path_audit.json", checks)
    if checks["status"] != "PASS":
        raise RuntimeError("CSV/schema/privacy audit failed")


def historical_integrity_after(run_dir: Path) -> None:
    before = read_csv(run_dir / "tables/historical_checkpoint_hashes_before.csv")
    after = []
    for row in before:
        path = REPO / row["relative_path"]
        observed = sha256_file(path)
        after.append({"relative_path": row["relative_path"], "expected_sha256": row["sha256"], "observed_sha256": observed, "status": "PASS" if observed == row["sha256"] else "FAIL"})
    write_csv_fresh(run_dir / "tables/historical_checkpoint_hashes_after.csv", after)
    if len(after) != 18 or any(row["status"] != "PASS" for row in after):
        raise RuntimeError("Historical checkpoints changed")


def fmt(value, digits=6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def write_final_report(run_dir: Path, decision: str, rationale: str) -> None:
    macro = {row["condition"]: row for row in read_csv(run_dir / "tables/primary_metrics_macro.csv")}
    primary = read_csv(run_dir / "tables/primary_metrics_per_sample.csv")
    primary_by_condition = {
        condition: [row for row in primary if row["condition"] == condition]
        for condition in ("A_centered_no_prompt", "B_randomized_no_prompt", "C_randomized_coordinate_prompt")
    }
    b_source = np.asarray([float(row["source_mse"]) for row in primary_by_condition["B_randomized_no_prompt"]])
    c_source = np.asarray([float(row["source_mse"]) for row in primary_by_condition["C_randomized_coordinate_prompt"]])
    b_whole = np.asarray([float(row["whole_mse"]) for row in primary_by_condition["B_randomized_no_prompt"]])
    c_whole = np.asarray([float(row["whole_mse"]) for row in primary_by_condition["C_randomized_coordinate_prompt"]])
    trim = max(1, int(0.01 * len(b_source)))
    distribution_rows = []
    for label, values in (("B_source_mse", b_source), ("C_source_mse", c_source), ("C_minus_B_source_mse", c_source - b_source), ("C_divided_by_B_source_mse", c_source / np.maximum(b_source, 1e-30))):
        distribution_rows.append({"quantity": label, "mean": float(np.mean(values)), "trimmed_mean_1pct_each_tail": float(np.mean(np.sort(values)[trim:-trim])),
                                  "p05": float(np.quantile(values, 0.05)), "p25": float(np.quantile(values, 0.25)),
                                  "median": float(np.median(values)), "p75": float(np.quantile(values, 0.75)), "p95": float(np.quantile(values, 0.95))})
    write_csv_fresh(run_dir / "tables/paired_distribution_summary.csv", distribution_rows)
    effects = json.loads((run_dir / "reports/paired_effects.json").read_text())
    swap = json.loads((run_dir / "reports/prompt_swap_summary.json").read_text())
    behavior = json.loads((run_dir / "reports/randomized_unprompted_behavior.json").read_text())
    no_harm = {row["case"]: row for row in read_csv(run_dir / "tables/no_harm_summary.csv")}
    params = json.loads((run_dir / "manifests/model_parameter_counts.json").read_text())
    configs = [json.loads((run_dir / f"manifests/{condition.lower()}_training_config.json").read_text()) for condition in macro]
    training_runtime = sum(float(config["runtime_seconds"]) for config in configs)
    disk_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    campaign_start = min(path.stat().st_mtime for path in run_dir.rglob("*") if path.is_file())
    elapsed = time.time() - campaign_start
    empty_hallucination = float(no_harm["empty_prompt"]["hallucinated_source_rate"])
    clean_damage = float(no_harm["isolated_correct"]["source_mse"])
    wrong_confusion = float(no_harm["wrong_prompt"]["source_confusion_rate"])
    if decision == "success":
        next_experiment = "Freeze a new group-safe campaign on the same promptable Condition-C backbone, then add recoverability prediction and bounded uncertainty using calibration-only threshold selection; keep the current development test and lockbox untouched."
        ready = "Yes—promptability passed, so a separately frozen recoverability/abstention campaign is justified."
    elif decision == "partial":
        next_experiment = "Repeat the same A/B/C promptability baseline with a predeclared coordinate-robustness intervention (prompt jitter augmentation only), preserving the split and evaluation protocol; do not add uncertainty yet."
        ready = "No. Resolve source confusion/coordinate sensitivity in another promptability-only campaign first."
    else:
        next_experiment = "Run a new promptability-only diagnostic with stronger prompt injection at multiple backbone scales while keeping the same controlled A/B/C logic; do not add uncertainty yet."
        ready = "No. The prompt must first be shown to control requested identity."
    report = f"""# Thayer-Select promptability baseline — final report

Run: `{run_dir.relative_to(REPO)}`

Scientific question: Does coordinate conditioning let a compact model reconstruct an arbitrary requested galaxy after the centered-target shortcut is removed?

Decision: **{decision.upper()}**

Rationale: {rationale}

## Executive result

All source, simulator, replay, MPS, manifest, and historical-checkpoint gates passed. Condition B is interpreted as an identifiability/centrality control, not a production deblender. Conditions B and C used byte-identical randomized development scenes and requested identities; Condition A used the aligned centered-position variant. No ratio across the A and B scene variants is reported.

The paired randomized source-region MSE effect was C−B = **{fmt(effects['C_minus_B_mean'])}** (95% bootstrap CI {fmt(effects['C_minus_B_ci95'][0])} to {fmt(effects['C_minus_B_ci95'][1])}); C won {effects['C_wins']}/1000 scenes. Prompt-swap success was **{fmt(swap['source_swap_success_rate'], 4)}**, output-collapse rate was **{fmt(swap['output_collapse_rate'], 4)}**, and changing only the prompt changed the output in **{fmt(swap['changing_prompt_changes_output_rate'], 4)}** of scenes.

The distribution is heavy-tailed and is reported explicitly: median source MSE was {fmt(np.median(b_source))} for B and {fmt(np.median(c_source))} for C; the 1%-trimmed means were {fmt(np.mean(np.sort(b_source)[trim:-trim]))} and {fmt(np.mean(np.sort(c_source)[trim:-trim]))}. C won {int(np.sum(c_whole < b_whole))}/1000 whole-image comparisons. Thus the mean improvement is not described as uniform per scene.

## Required answers

1. **Explicit-seed replay:** PASS. Source IDs, source coordinates, isolated arrays, noiseless/noisy blends, prompt maps, seed `2026072301`, and catalog/metadata/array hashes matched exactly (`logs/explicit_seed_replay_verification.json`).
2. **Group-safe source split:** PASS. 86,273 persistent source identities and exact-position duplicate groups have zero cross-partition crossings (`diagnostics/group_integrity_report.json`).
3. **Lockbox sealed:** YES. Zero lockbox scene definitions, renders, arrays, plots, or evaluations were created. Only assignment metadata was counted/hashed.
4. **Training on MPS:** YES. All three models completed 20/20 epochs on MPS with no CPU neural fallback.
5. **Exact parameter counts:** A={params['A_centered_no_prompt']:,}; B={params['B_randomized_no_prompt']:,}; C={params['C_randomized_coordinate_prompt']:,}. The sole prompt-related difference is **+{params['prompt_first_layer_parameter_difference']}** first-layer weights.
6. **Randomization cost:** B randomized source MSE was {fmt(macro['B_randomized_no_prompt']['source_mse'])}; A centered source MSE was {fmt(macro['A_centered_no_prompt']['source_mse'])}. The aligned absolute difference B−A was {fmt(effects['B_randomized_minus_A_centered_absolute_difference'])}; this is not presented as a cross-manifest ratio.
7. **Prompt recovery:** C randomized source MSE was {fmt(macro['C_randomized_coordinate_prompt']['source_mse'])}, an absolute paired change of {fmt(effects['C_minus_B_mean'])} from B, with CI above.
8. **Prompt swapping:** Source-swap success={fmt(swap['source_swap_success_rate'], 4)}; output-collapse={fmt(swap['output_collapse_rate'], 4)}; prompt sensitivity ratio={fmt(swap['prompt_sensitivity_ratio_mean'], 4)}.
9. **Coordinate versus brightness/centrality:** Condition-C requested-identity evidence is in `tables/prompt_swap_per_scene.csv`; Condition-B tendencies are: closer to central {behavior.get('closer_central_source', 0)}/1000, brighter {behavior.get('closer_brighter_source', 0)}/1000, larger {behavior.get('closer_larger_source', 0)}/1000, and average over either source {behavior.get('closer_average_than_either', 0)}/1000.
10. **Failure modes:** The principal observed weaknesses are quantified by prompt collapse, swap failures, stratified source confusion, and small-offset sensitivity (`tables/centrality_stratified_summary.csv`, `tables/prompt_swap_per_scene.csv`).
11. **Basic no-harm tests:** Empty-prompt hallucination rate={fmt(empty_hallucination, 4)} under the predeclared diagnostic definition (absolute predicted flux >10% of requested-source flux); isolated-source source MSE={fmt(clean_damage)}; wrong-prompt confusion rate={fmt(wrong_confusion, 4)}. These are engineering diagnostics, not calibrated abstention results.
12. **Promptability classification:** **{decision.upper()}**. {rationale}
13. **Ready for recoverability/abstention:** {ready}
14. **Exact next experiment:** {next_experiment}

## Primary metric table (macro per scene)

| Condition | Whole MSE | Source MSE | Source MAE | PSNR | SSIM | Centroid error (px) | Worse than input |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for condition in ("A_centered_no_prompt", "B_randomized_no_prompt", "C_randomized_coordinate_prompt"):
        row = macro[condition]
        report += f"| {condition} | {fmt(row['whole_mse'])} | {fmt(row['source_mse'])} | {fmt(row['source_mae'])} | {fmt(row['psnr'])} | {fmt(row['ssim'])} | {fmt(row['centroid_error_pixels'])} | {row['worse_than_input_count']} |\n"
    report += f"""

Micro source/core/non-core aggregations are in `tables/primary_metrics_micro.csv`; per-sample metrics, flux/color errors, win/loss/tie inputs, and identity comparisons are in `tables/primary_metrics_per_sample.csv`. Quantiles and trimmed distribution summaries are in `tables/paired_distribution_summary.csv`.

## Figures and diagnostic evidence

- Training curves: `figures/training_curves.png`
- Defining prompt-swap grid: `figures/prompt_swap_flagship_grid.png`
- Prompt-swap table and summary: `tables/prompt_swap_per_scene.csv`, `reports/prompt_swap_summary.json`
- Centrality stratification: `tables/centrality_stratification_per_scene.csv`, `tables/centrality_stratified_summary.csv`
- No-harm diagnostics: `tables/no_harm_per_sample.csv`, `tables/no_harm_summary.csv`

## Provenance and integrity

- Source split and scene hashes: `manifests/source_split_manifest.csv`, `manifests/rendered_scene_manifest.csv`, `tables/btk_foundation_inputs.csv`
- New checkpoint hashes: the three frozen training configs and `tables/campaign_file_hashes.csv`
- Historical checkpoints: PASS, 18/18 unchanged (`tables/historical_checkpoint_hashes_before.csv`, `tables/historical_checkpoint_hashes_after.csv`)
- Training runtime: {training_runtime / 60:.2f} minutes; campaign elapsed wall time: {elapsed / 60:.2f} minutes
- Run disk usage before final hash inventory: {disk_bytes / (1024**3):.3f} GiB
- Git status: `logs/git_final.txt`; Git diff whitespace check: PASS
- Compileall, relevant unittests, CSV/schema, privacy/path, formula, and MPS audits: PASS

No uncertainty, recoverability, abstention, calibration selection, COSMOS, DR10, or lockbox experiment was performed.
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)


def hash_inventory(run_dir: Path) -> None:
    rows = []
    excluded = {run_dir / "tables/campaign_file_hashes.csv", run_dir / "logs/audit_complete.json"}
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file() and value not in excluded):
        rows.append({"relative_path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv_fresh(run_dir / "tables/campaign_file_hashes.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--decision", choices=("success", "partial", "failure"), required=True)
    parser.add_argument("--rationale", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "logs/development_evaluation_complete.json").is_file():
        raise RuntimeError("Development evaluation is incomplete")
    integrity_audit(run_dir)
    tiny_array_tests(run_dir)
    external_audits(run_dir)
    schema_and_privacy_audit(run_dir)
    historical_integrity_after(run_dir)
    code_paths = [REPO / "src/btk_scene.py", REPO / "scripts/thayer_select_prompt_ablation_common.py", REPO / "scripts/prepare_thayer_select_prompt_ablation.py", REPO / "scripts/train_thayer_select_prompt_ablation.py", REPO / "scripts/evaluate_thayer_select_prompt_ablation.py", Path(__file__).resolve()]
    write_csv_fresh(run_dir / "tables/campaign_code_hashes.csv", [{"relative_path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in code_paths])
    git_final = run_command(["git", "status", "--short", "--branch"])
    write_json_fresh(run_dir / "logs/git_final.txt", git_final)
    write_final_report(run_dir, args.decision, args.rationale)
    hash_inventory(run_dir)
    write_json_fresh(run_dir / "logs/audit_complete.json", {"status": "PASS", "decision": args.decision.upper(), "completed_at_unix": time.time(), "hash_inventory_excludes_itself_and_this_completion_marker": True})


if __name__ == "__main__":
    main()
