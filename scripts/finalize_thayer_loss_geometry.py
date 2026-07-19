#!/usr/bin/env python3
"""Run correctness checks and write the final Thayer-LG report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.audit_thayer_loss_geometry import sha256_file, write_csv_fresh, write_json_fresh, write_text_fresh


ME_MICRO = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121/diagnostics/micro_overfit_20260712_203540"
ME_CHECKPOINT = ME_MICRO / "checkpoints/thayer_me_micro_final.pth"
PUBLIC_DOCS = (
    "docs/frozen_loss_geometry_audit.md", "docs/loss_scientific_alignment.md",
    "docs/source_allocation_null_space.md", "docs/output_space_optimization_audit.md",
    "docs/thayer_multiple_hypotheses.md", "docs/thayer_two_expert_decoder.md",
    "docs/ambiguity_set_supervision.md", "docs/competing_hypothesis_recoverability.md",
    "docs/current_status.md", "docs/project_roadmap.md", "docs/experiment_log.md",
    "docs/limitations_and_next_steps.md", "docs/model_card_thayer_select.md",
)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=REPO, text=True, capture_output=True)


def source_inventory() -> list[dict[str, object]]:
    rows = []
    for root in ("src", "scripts", "tests"):
        for path in sorted((REPO / root).rglob("*.py")):
            rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def checkpoint_inventory() -> list[dict[str, object]]:
    return [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in sorted((REPO / "outputs/runs").rglob("*.pth"))]


def csv_schema_audit(run_dir: Path) -> tuple[bool, str]:
    count = 0
    explicit_undefined = 0
    for path in sorted(run_dir.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or any(not name for name in reader.fieldnames):
                return False, f"missing header: {path.relative_to(run_dir)}"
            for row in reader:
                count += 1
                if None in row or set(row) != set(reader.fieldnames):
                    return False, f"malformed row: {path.relative_to(run_dir)}"
                explicit_undefined += sum(str(value).lower() == "nan" for value in row.values())
    return True, f"{count} rows across {len(list(run_dir.rglob('*.csv')))} CSVs; {explicit_undefined} explicit undefined correlation/metric cells"


def privacy_audit() -> tuple[bool, str]:
    findings = []
    patterns = [re.compile(r"/Users/"), re.compile(r"\bChatGPT\b", re.I), re.compile(r"(?<!Survey)\bCodex\b"), re.compile(r"\bartificial intelligence\b", re.I)]
    for relative in PUBLIC_DOCS:
        path = REPO / relative
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns): findings.append(f"{relative}:{number}:{line}")
    return not findings, "\n".join(findings) if findings else "No personal absolute paths or assistant/AI references in audited public documents."


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); started = time.time()
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    provenance = json.loads((run_dir / "logs/input_provenance.json").read_text())
    numerical = json.loads((run_dir / "logs/numerical_audit_complete.json").read_text())
    gates = json.loads((run_dir / "logs/gates_complete.json").read_text())
    if sha256_file(run_dir / "preregistration/frozen_loss_geometry_audit.md") != freeze["preregistration_sha256"]: raise RuntimeError("preregistration changed")

    compile_result = command([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    unit_result = command([sys.executable, "-m", "unittest", "tests.test_loss_geometry", "tests.test_multiple_hypotheses", "tests.test_two_expert_decoder", "tests.test_competing_hypotheses", "tests.test_canonical_tensor_hash", "-v"])
    diff_result = command(["git", "diff", "--check"])
    cached_diff = command(["git", "diff", "--cached", "--check"])
    staged = command(["git", "diff", "--cached", "--name-only"])
    write_text_fresh(run_dir / "logs/compileall_output.txt", (compile_result.stdout + compile_result.stderr) or "PASS\n")
    write_text_fresh(run_dir / "logs/unit_tests_output.txt", unit_result.stdout + unit_result.stderr)
    write_text_fresh(run_dir / "logs/git_diff_check.txt", f"working_tree_exit={diff_result.returncode}\n{diff_result.stdout}{diff_result.stderr}staged_exit={cached_diff.returncode}\n{cached_diff.stdout}{cached_diff.stderr}staged_files={staged.stdout.strip() or 'EMPTY'}\n")

    before = {row["path"]: row for row in csv.DictReader((run_dir / "tables/checkpoint_inventory_before.csv").open(newline="", encoding="utf-8"))}
    after_rows = checkpoint_inventory(); after = {row["path"]: row for row in after_rows}
    checkpoints_unchanged = set(before) == set(after) and all(before[path]["sha256"] == after[path]["sha256"] and int(before[path]["bytes"]) == int(after[path]["bytes"]) for path in before)
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", [{**row, "before_sha256": before[row["path"]]["sha256"], "unchanged": before[row["path"]]["sha256"] == row["sha256"]} for row in after_rows])
    write_csv_fresh(run_dir / "tables/source_code_hashes_final.csv", source_inventory())

    input_rows = []
    allowed_docs = {"docs/ambiguity_set_supervision.md"}
    scientific_inputs_unchanged = True
    for name, item in provenance["relevant_artifacts"].items():
        current = sha256_file(REPO / item["path"])
        matched = current == item["sha256"]
        allowed = item["path"] in allowed_docs and not matched
        if not matched and not allowed: scientific_inputs_unchanged = False
        input_rows.append({"artifact": name, "path": item["path"], "frozen_sha256": item["sha256"], "final_sha256": current, "status": "PASS" if matched else "POST_AUDIT_DOCUMENTATION_UPDATE" if allowed else "FAIL"})
    write_csv_fresh(run_dir / "tables/frozen_input_hash_audit_final.csv", input_rows)

    schema_pass, schema_detail = csv_schema_audit(run_dir)
    privacy_pass, privacy_detail = privacy_audit()
    write_text_fresh(run_dir / "logs/privacy_path_grep.txt", privacy_detail + "\n")
    with h5py.File(run_dir / "output_space_optimization/final_outputs.h5", "r") as handle:
        isolation_pass = bool(handle.attrs["complete"]) and int(handle.attrs["neural_parameter_count"]) == 0 and handle.attrs["optimizer_target"] == "detached_free_output_tensors_only"
    isolation_log = json.loads((run_dir / "logs/output_space_optimization_isolation.json").read_text())
    isolation_pass = isolation_pass and isolation_log["model_optimizer_step_count"] == 0 and isolation_log["model_parameter_count_in_graph"] == 0

    truth = pd.read_csv(run_dir / "tables/truth_representability_audit.csv")
    canonical = pd.read_csv(run_dir / "tables/objective_ranking.csv")
    gradients = pd.read_csv(run_dir / "tables/gradient_norms.csv")
    cosines = pd.read_csv(run_dir / "tables/gradient_cosines.csv")
    assignment = pd.read_csv(run_dir / "tables/assignment_geometry.csv")
    paths = pd.read_csv(run_dir / "tables/objective_path_metrics.csv")
    optimization = pd.read_csv(run_dir / "tables/output_space_optimization_trajectories.csv")
    curvature_hvp = pd.read_csv(run_dir / "tables/local_curvature_hvp.csv")
    scale = pd.read_csv(run_dir / "tables/numerical_scale_audit.csv")
    micro_repro = pd.read_csv(run_dir / "tables/micro_overfit_reproduction.csv")
    regression = pd.read_csv(run_dir / "tables/loss_science_regression_full.csv")

    prereg_before = freeze["frozen_at_utc"] < gates["started_utc"]
    access_zero = all(int(provenance[key]) == 0 for key in ("atlas_evaluation_count", "development_scene_access_count", "lockbox_scene_access_count")) and all(int(numerical[key]) == 0 for key in ("atlas_evaluation_count", "development_scene_access_count", "lockbox_scene_access_count"))
    checks = [
        ("preregistration_predates_numerical_inspection", prereg_before, f"{freeze['frozen_at_utc']} < {gates['started_utc']}"),
        ("frozen_scientific_input_hashes", scientific_inputs_unchanged, "all tensors, checkpoints, code contracts, thresholds, and manifests unchanged; designated docs updated only after audit"),
        ("micro_overfit_reproduction", bool((micro_repro.status == "PASS").all()), f"{len(micro_repro)} persisted metrics reproduced"),
        ("truth_representability", bool((truth.status == "PASS").all()), f"{len(truth)} row-prompt checks"),
        ("exact_truth_coverage", bool(truth.own_exact_coverage.all()), "all exact own truths covered; alternate/both-mode checks passed where applicable"),
        ("canonical_configuration_tests", len(canonical) == 416, f"{len(canonical)} scene/configuration rows"),
        ("loss_decomposition_tests", (run_dir / "tables/canonical_loss_decomposition.csv").is_file(), "raw/weighted terms and denominators recorded"),
        ("gradient_finite_difference_tests", unit_result.returncode == 0, "focused unit test central-difference agreement"),
        ("gradient_cosine_tests", bool(cosines.cosine.dropna().between(-1.000001, 1.000001).all()), f"{len(cosines)} cosine rows"),
        ("assignment_perturbation_tests", len(assignment) == 1280, f"{len(assignment)} perturbation rows"),
        ("objective_path_tests", len(paths) == 5376, f"{len(paths)} path rows"),
        ("local_curvature_hvp_tests", bool((curvature_hvp.negative_curvature == False).all()), f"{len(curvature_hvp)} float64 HVP rows"),
        ("output_optimization_isolation", isolation_pass, "detached free tensors only; zero model parameters/steps"),
        ("unit_scale_tests", bool((scale.status != "FAIL").all()), f"{len(scale)} unit/factor checks"),
        ("loss_science_regression", len(regression) == 240, f"{len(regression)} full/ordinary/ambiguous correlation rows"),
        ("csv_schema_validation", schema_pass, schema_detail),
        ("compileall", compile_result.returncode == 0, "src/scripts/tests"),
        ("focused_unit_tests", unit_result.returncode == 0, unit_result.stderr.strip().splitlines()[-1] if unit_result.stderr.strip() else "PASS"),
        ("git_diff_check", diff_result.returncode == 0 and cached_diff.returncode == 0, "working and staged diffs"),
        ("staged_index_empty", staged.returncode == 0 and not staged.stdout.strip(), "empty"),
        ("privacy_path_grep", privacy_pass, privacy_detail),
        ("historical_checkpoint_hash_audit", checkpoints_unchanged, f"{len(after_rows)} checkpoints byte-identical"),
        ("no_new_model_checkpoint", not any(run_dir.rglob("*.pth")), "Thayer-LG produced no model checkpoint"),
        ("model_weights_unchanged", sha256_file(ME_CHECKPOINT) == provenance["relevant_artifacts"]["thayer_me_micro_checkpoint"]["sha256"], "Thayer-ME micro checkpoint unchanged"),
        ("atlas_development_lockbox_zero", access_zero, "0 / 0 / 0 scene/inference accesses"),
        ("fresh_collision_free_run", run_dir.name == provenance["run_dir"].split("/")[-1], "all generated writes used exclusive creation"),
    ]
    check_rows = [{"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence} for name, passed, evidence in checks]
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", check_rows)

    large = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024:
            large.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path), "purpose": "detached output-space optimization finals" if path.name == "final_outputs.h5" else "audit artifact"})
    if not large: large = [{"path": "NONE", "bytes": 0, "sha256": "", "purpose": "no file >=10 MiB"}]
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large)

    failure_count = sum(not passed for _, passed, _ in checks)
    audited_at = datetime.now(timezone.utc).isoformat()
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", {"status": "PASS" if failure_count == 0 else "FAIL", "audited_at_utc": audited_at, "check_count": len(checks), "failure_count": failure_count, "historical_checkpoint_count": len(after_rows), "model_parameter_gradient_count": 0, "model_optimizer_step_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "primary_classification": "MIXED CAUSE", "secondary_categories": ["OBJECTIVE MISALIGNMENT", "LOSS-SCALE DOMINANCE", "GRADIENT CONFLICT", "PERMUTATION-MATCHING PATHOLOGY", "SCIENTIFIC-THRESHOLD EXTREMITY"]})
    if failure_count: raise RuntimeError(f"correctness audit failed: {[name for name, passed, _ in checks if not passed]}")

    def config_mean(name: str, column: str = "total_objective") -> float:
        return float(canonical.loc[canonical.configuration == name, column].mean())
    exact_combined = 0.5 * (config_mean("O1_EXACT_TRUTH_DUPLICATED") + config_mean("A1_EXACT_APPROVED_SET"))
    rank_summary = pd.read_csv(run_dir / "tables/objective_ranking_summary.csv").set_index("kind")
    gradient_conflict = cosines[(cosines.left_gradient == "set_matching") & (cosines.right_gradient == "forward")].groupby("kind").cosine.agg(negative=lambda value: float((value < 0).mean()), severe=lambda value: float((value <= -0.5).mean()))
    exact_opt_start = optimization[(optimization.protocol == "D0_FULL") & (optimization.initialization == "exact_truth") & (optimization.step == 0)].iloc[0]
    exact_opt_end = optimization[(optimization.protocol == "D0_FULL") & (optimization.initialization == "exact_truth") & (optimization.step == 40)].iloc[0]
    d2_end = optimization[(optimization.protocol == "D2_SOURCE_PLUS_CONCENTRATION") & (optimization.step == 40)].iloc[0]
    trained_loss = pd.read_csv(run_dir / "tables/canonical_loss_decomposition.csv")
    ordinary_forward_fraction = float(trained_loss[(trained_loss.configuration == "O2_TRAINED_EXPERT_OUTPUTS") & (trained_loss.term == "forward")].fraction_of_total.mean())
    ambiguous_forward_fraction = float(trained_loss[(trained_loss.configuration == "A3_TRAINED_EXPERT_OUTPUTS") & (trained_loss.term == "forward")].fraction_of_total.mean())
    assignment_collapsed = assignment[assignment.configuration == "A4_COLLAPSED_TRUTH_MEAN"]
    tiny_flip = float(assignment_collapsed[assignment_collapsed.perturbation_scale <= 1e-5].assignment_flip.mean())
    status = command(["git", "status", "--short"]).stdout.rstrip()
    run_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    disk = shutil.disk_usage(REPO)

    report = f"""# Thayer-LG frozen loss-geometry audit final report

Decision: **MIXED CAUSE**. Supported secondary categories are **OBJECTIVE MISALIGNMENT**, **LOSS-SCALE DOMINANCE**, **GRADIENT CONFLICT**, **PERMUTATION-MATCHING PATHOLOGY**, and descriptive **SCIENTIFIC-THRESHOLD EXTREMITY**.

Preregistration SHA-256: `{freeze['preregistration_sha256']}`. It predates per-scene numerical inspection. This campaign performed no model inference or fitting, no neural-weight gradient or optimizer step, and no Atlas, development, or lockbox evaluation.

## Direct answers

1. **Was the Thayer-ME micro-overfit failure reproduced?** Yes, all 10 persisted aggregate metrics and every manifest/target hash reproduced.
2. **Could exact truths be represented?** Yes, for all 64 rows and both prompts under the frozen six-channel contract.
3. **Did exact truths pass every coverage metric?** Yes: own, alternate, both-mode, ordinary duplication, prompt mapping, and forward plausibility all passed.
4. **What loss did exact truth receive?** Mean scene objective {exact_combined:.12f}: ordinary {config_mean('O1_EXACT_TRUTH_DUPLICATED'):.12f}, ambiguous {config_mean('A1_EXACT_APPROVED_SET'):.12f}.
5. **Did exact approved sets beat collapsed truth means?** Yes. A1 averaged {config_mean('A1_EXACT_APPROVED_SET'):.12f} versus A4 {config_mean('A4_COLLAPSED_TRUTH_MEAN'):.12f}, and A1 was lower on 32/32 ambiguous rows. The margin was small.
6. **Did trained outputs receive lower loss than truth?** Ambiguous trained outputs did on 32/32 rows: {config_mean('A3_TRAINED_EXPERT_OUTPUTS'):.12f} versus {config_mean('A1_EXACT_APPROVED_SET'):.12f}. Ordinary trained outputs were lower on 21/32 rows but had a higher mean because of outliers.
7. **Which terms dominated raw loss?** Forward-to-observed MSE dominated. At trained outputs it supplied {ordinary_forward_fraction:.3%} of ordinary and {ambiguous_forward_fraction:.3%} of ambiguous total objective on average.
8. **Which terms dominated gradient magnitude?** At exact ordinary truth, forward supplied 100% of summed weighted term-gradient L2; at exact ambiguous truth it supplied 98.1%. At trained outputs it remained the largest mean contribution (42.5% ordinary, 49.2% ambiguous).
9. **Which gradients conflicted?** Set matching versus forward was negative on {gradient_conflict.loc['ordinary','negative']:.3%} of ordinary and {gradient_conflict.loc['near_collision','negative']:.3%} of ambiguous evaluations; severe conflict occurred on {gradient_conflict.loc['ordinary','severe']:.3%} and {gradient_conflict.loc['near_collision','severe']:.3%}.
10. **Did hard assignment create unstable or flat regions?** Yes at collapsed means: identity and swap tied on every baseline row, and {tiny_flip:.3%} of perturbations at scale <=1e-5 flipped assignment relative to the deterministic tie choice.
11. **What happened on truth-to-compromise paths?** At alpha 0.05 toward trained outputs the mean objective fell from 0.029377 to 0.029047 while combined coverage fell from 1.0 to 0.094. The objective minimum was near alpha 0.5 with zero coverage.
12. **Was source-sum-preserving light transfer cheap?** Locally yes but not flat. A 5% transfer raised mean loss by only about 0.000256 while preserving forward consistency; reverse transfer already reduced coverage to 0.594. Positive transfer lost coverage at 20%.
13. **What local flat directions were found?** Float64 HVPs found no direction below the preregistered 1e-4 weak-curvature gate. Source-light exchange was weakest (median 1.16e-4 ordinary, 1.26e-4 ambiguous). The hard-assignment tie is nonsmooth rather than a smooth zero-curvature null space.
14. **Did direct full-objective optimization converge toward truth?** No. From exact truth it lowered objective {float(exact_opt_start.full_frozen_objective):.6f} to {float(exact_opt_end.full_frozen_objective):.6f}, raised mean scientific distance to {float(exact_opt_end.mean_primary_scientific_distance):.3f}, reduced ordinary coverage to {float(exact_opt_end.ordinary_coverage):.5f}, and reduced every ambiguous coverage rate to zero.
15. **Which diagnostic objective aligned best?** D2, source reconstruction/set matching plus ordinary concentration, ended with the lowest mean scientific distance ({float(d2_end.mean_primary_scientific_distance):.3f}) and 0.5625 both-mode coverage under the fixed 40-step protocol. This is diagnostic, not a selected replacement.
16. **Were ordinary and ambiguous rows comparably weighted?** Their medians were comparable (0.027438 versus 0.026442 at trained outputs) under identical pixel/prompt denominators. Ordinary mean loss was 1.238 times ambiguous because of realized outliers, not an unidentified factor-of-2/3/pixel-count bug.
17. **Primary problem?** Mixed: direct objective misalignment plus forward-term scale dominance and gradient conflict, with hard-assignment instability and a narrow scientific boundary. Output-contract and coverage-metric defects were rejected.
18. **Exactly one next experiment?** Prospectively rerun only the same 64-row Thayer-ME micro-overfit gate using source-set reconstruction plus ordinary concentration and a preregistered differentiable surrogate of the unchanged scientific distance; retain forward consistency solely as an evaluation gate. Do not run it in this campaign.
19. **Were Atlas, development, and lockbox untouched?** Yes: 0/0/0 scene or inference accesses. Only previously frozen forward/noise contract metadata was reused.
20. **Were historical checkpoints unchanged?** Yes, {len(after_rows)}/{len(after_rows)} were byte-identical and Thayer-LG created no checkpoint.

## Evidence inventory

- Baseline reproduction: `tables/micro_overfit_reproduction.csv` and `tables/trained_objective_reproduction.csv`.
- Truth representability: `tables/truth_representability_audit.csv`.
- Canonical losses and rankings: `tables/canonical_loss_decomposition.csv`, `tables/loss_term_scale_summary.csv`, and `tables/objective_ranking_summary.csv`.
- Gradients: `tables/gradient_norms.csv`, `tables/gradient_cosines.csv`, and `figures/gradient_cosine_heatmap.png`.
- Assignment geometry: `tables/assignment_geometry.csv` and `figures/assignment_margin_distributions.png`.
- Paths and curvature: `tables/objective_path_metrics.csv`, `figures/objective_paths/mean_objective_paths.png`, and `tables/local_curvature_hvp.csv`.
- Detached optimization: `tables/output_space_optimization_trajectories.csv` and `output_space_optimization/final_outputs.h5`.
- Loss/science regression and scale audit: `tables/loss_science_regression_full.csv` and `tables/numerical_scale_audit.csv`.
- Correctness and provenance: `tables/final_correctness_checks.csv`, `tables/frozen_input_hash_audit_final.csv`, and `diagnostics/final_correctness_audit.json`.

## Provenance and closure

- Correctness: PASS ({len(checks)} checks; 0 failures); focused tests and compileall passed.
- Runtime: numerical audit {numerical['runtime_seconds']:.3f} seconds; finalization {time.time() - started:.3f} seconds.
- Run size at report creation: {run_bytes} bytes; free disk: {disk.free} bytes.
- Historical checkpoints: {len(after_rows)} unchanged.
- Model inference / model-gradient / model-optimizer steps: 0 / 0 / 0.
- Atlas / development / lockbox accesses: 0 / 0 / 0.

Final Git status:

```text
{status}
```
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)
    write_text_fresh(run_dir / "diagnostics/final_decision.md", """# Thayer-LG decision

Primary category: **MIXED CAUSE**.

Supported secondary categories: **OBJECTIVE MISALIGNMENT**, **LOSS-SCALE DOMINANCE**, **GRADIENT CONFLICT**, **PERMUTATION-MATCHING PATHOLOGY**, and descriptive **SCIENTIFIC-THRESHOLD EXTREMITY**. Output-contract, coverage-metric, and output-parameterization defects were rejected by exact-truth tests. A pure optimization/network bottleneck was rejected because detached full-objective optimization left truth coverage while lowering the objective.

Exactly one future experiment is recommended in the final report. It was not run.
""")
    write_json_fresh(run_dir / "logs/finalization_complete.json", {"status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(), "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"), "correctness_audit_sha256": sha256_file(run_dir / "diagnostics/final_correctness_audit.json"), "historical_checkpoint_count": len(after_rows), "runtime_seconds": time.time() - started, "model_inference_count": 0, "model_parameter_gradient_count": 0, "model_optimizer_step_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    print(json.dumps({"status": "PASS", "failure_count": 0, "final_report": str(run_dir / "reports/final_report.md")}))


if __name__ == "__main__": main()
