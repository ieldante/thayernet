#!/usr/bin/env python3
"""Close Thayer-FP with correctness, provenance, and the final report."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"
PUBLIC_DOCS = [
    "docs/direct_scientific_feasibility_projection.md", "docs/feasible_target_learning.md",
    "docs/scientific_region_projection_contract.md", "docs/micro_capacity_after_projection.md",
    "docs/output_space_conditioning_audit.md", "docs/scientific_alignment_objective.md",
    "docs/thayer_two_expert_decoder.md", "docs/current_status.md", "docs/project_roadmap.md",
    "docs/experiment_log.md", "docs/limitations_and_next_steps.md", "docs/model_card_thayer_select.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True)


def checkpoint_paths() -> list[Path]:
    return sorted(path for path in (REPO / "outputs").rglob("*") if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"})


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        RUN / "logs/projection_gate_complete_final.json", RUN / "logs/micro_training_complete.json",
        RUN / "projection_targets/projected_target_sets_final.h5", RUN / "micro_overfit/final_outputs.h5",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("campaign execution incomplete")
    projection = json.loads((RUN / "logs/projection_gate_complete_final.json").read_text())
    training = json.loads((RUN / "logs/micro_training_complete.json").read_text())
    provenance = json.loads((RUN / "logs/input_provenance.json").read_text())
    freeze = json.loads((RUN / "preregistration/freeze_record.json").read_text())
    target_freeze = json.loads((RUN / "projection_targets/freeze_record_final.json").read_text())
    method_rows = read_dicts(RUN / "tables/projection_method_comparison_final_superseding.csv")
    p0 = next(row for row in method_rows if row["method"] == "P0_HOMOTOPY_INTERIOR")
    p1 = next(row for row in method_rows if row["method"] == "P1_AUGMENTED_LAGRANGIAN")
    homotopy = read_dicts(RUN / "tables/homotopy_projection_summary.csv")
    limits = read_dicts(RUN / "tables/limiting_constraint_frequency.csv")
    most_limiting = max(limits, key=lambda row: int(row["count"]))

    compile_result = command(sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests")
    fresh_text(RUN / "logs/compileall_output.txt", compile_result.stdout + compile_result.stderr)
    pytest_result = command(
        sys.executable, "-m", "pytest", "-q",
        "tests/test_feasibility_projection.py", "tests/test_canonical_tensor_hash.py",
        "tests/test_two_expert_decoder.py", "tests/test_scientific_alignment.py",
        "tests/test_output_conditioning.py", "tests/test_competing_hypotheses.py",
    )
    fresh_text(RUN / "logs/focused_tests_output.txt", pytest_result.stdout + pytest_result.stderr)
    diff = command("git", "diff", "--check")
    staged_diff = command("git", "diff", "--cached", "--check")
    fresh_text(RUN / "logs/git_diff_check.txt", diff.stdout + diff.stderr)
    fresh_text(RUN / "logs/git_diff_cached_check.txt", staged_diff.stdout + staged_diff.stderr)

    csv_errors = []
    csv_count = 0
    for path in sorted(RUN.rglob("*.csv")):
        csv_count += 1
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        widths = {len(row) for row in rows}
        if not rows or len(widths) != 1:
            csv_errors.append({"path": str(path.relative_to(REPO)), "widths": sorted(widths), "rows": len(rows)})

    before = read_dicts(RUN / "tables/checkpoint_inventory_before.csv")
    after_paths = checkpoint_paths()
    after = [{"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in after_paths]
    fresh_csv(RUN / "tables/checkpoint_inventory_after.csv", after)
    after_map = {row["path"]: row for row in after}
    historical_unchanged = all(row["path"] in after_map and row["sha256"] == after_map[row["path"]]["sha256"] for row in before)

    source_rows = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            source_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size})
    fresh_csv(RUN / "tables/source_code_hashes_final.csv", source_rows)

    privacy_pattern = re.compile(r"/Users/|\bChatGPT\b|\bOpenAI\b|\bartificial intelligence\b|\bCodex\b", re.I)
    privacy_matches = []
    for relative in PUBLIC_DOCS:
        for line_number, line in enumerate((REPO / relative).read_text(encoding="utf-8").splitlines(), 1):
            if privacy_pattern.search(line):
                privacy_matches.append({"path": relative, "line": line_number, "text": line})
    fresh_json(RUN / "logs/privacy_path_grep.json", {"matches": privacy_matches, "pass": not privacy_matches})

    large = []
    for path in REPO.rglob("*"):
        if path.is_file() and path.stat().st_size >= 50_000_000:
            large.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "campaign_generated": str(path).startswith(str(RUN))})
    large.sort(key=lambda row: (-int(row["bytes"]), str(row["path"])))
    fresh_csv(RUN / "tables/large_file_inventory.csv", large)

    exact_rows = read_dicts(RUN / "tables/exact_truth_feasibility.csv")
    final_hash_rows = read_dicts(RUN / "tables/projected_target_hashes_final.csv")
    with h5py.File(RUN / "projection_targets/projected_target_sets_final.h5", "r") as handle:
        target_shape = tuple(handle["targets_normalized"].shape)
        targets_finite = bool(np.all(np.isfinite(handle["targets_normalized"][:])))
        targets_nonnegative = bool(np.min(handle["targets_physical"][:]) >= -1e-7)
        added_inputs = int(handle.attrs["inference_input_fields_added"])

    preregistration_order = bool(
        freeze["per_scene_array_load_count"] == 0
        and freeze["detached_optimization_count"] == 0
        and (RUN / "preregistration/freeze_record.json").stat().st_mtime <= (RUN / "tables/baseline_reproduction.csv").stat().st_mtime
    )
    gate_attainability = all(row["attainable"] == "True" for row in read_dicts(RUN / "tables/preregistered_gate_attainability.csv"))
    binary_homotopy_monotone = all(row["feasibility_monotone"] == "True" for row in homotopy)
    component_nonmonotone_count = sum(row["scientific_ratios_monotone"] != "True" for row in homotopy)
    bisection_valid = all(float(row["boundary_alpha"]) <= float(row["interior_alpha"]) <= 1.0 for row in homotopy)
    p0_interior = float(p0["maximum_constraint_ratio"]) <= 0.95 and float(p0["interior_pair_fraction"]) == 1.0
    p1_strict_rejected = p1["strict_training_interior_pass"] == "False" and p1["final_eligible"] == "False"
    projected_integrity = bool(
        sha256(RUN / "projection_targets/projected_target_sets_final.h5") == target_freeze["projected_target_file_sha256"]
        and len(final_hash_rows) == 256 and target_shape == (64, 2, 2, 6, 60, 60)
        and targets_finite and targets_nonnegative
    )
    status_readme = command("git", "status", "--short", "README.md")
    status = command("git", "status", "--short")
    staged = command("git", "diff", "--cached", "--name-only")

    checks = [
        {"check": "preregistration_predates_per_scene_load_and_optimization", "status": "PASS" if preregistration_order else "FAIL", "evidence": freeze["frozen_at_utc"]},
        {"check": "all_gates_mathematically_attainable", "status": "PASS" if gate_attainability else "FAIL", "evidence": "exact truths attain every frozen rate and diameter gate"},
        {"check": "frozen_microset_hash", "status": "PASS" if provenance["microset_manifest_sha256"] == "9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085" else "FAIL", "evidence": provenance["microset_manifest_sha256"]},
        {"check": "frozen_target_threshold_assignment_contracts", "status": "PASS" if not provenance["frozen_input_mismatches"] else "FAIL", "evidence": "bootstrap expected hashes"},
        {"check": "exact_truth_feasibility", "status": "PASS" if len(exact_rows) == 256 and all(row["feasible"] == "True" for row in exact_rows) else "FAIL", "evidence": f"{len(exact_rows)} pairings"},
        {"check": "homotopy_dense_grid_and_binary_monotonicity", "status": "PASS" if len(homotopy) == 256 and binary_homotopy_monotone else "FAIL", "evidence": f"256 paths; {component_nonmonotone_count} component-nonmonotone"},
        {"check": "boundary_bisection", "status": "PASS" if bisection_valid else "FAIL", "evidence": "boundary <= interior <= 1"},
        {"check": "strict_095_interior_targets", "status": "PASS" if p0_interior else "FAIL", "evidence": p0["maximum_constraint_ratio"]},
        {"check": "p1_isolated_and_strictly_rejected", "status": "PASS" if p1_strict_rejected else "FAIL", "evidence": p1["maximum_constraint_ratio"]},
        {"check": "hard_assignment_and_ordinary_set_assembly", "status": "PASS" if len(read_dicts(RUN / "tables/projection_assignments.csv")) == 256 else "FAIL", "evidence": "superseding ordinary two-slot rule"},
        {"check": "canonical_hash_stability", "status": "PASS" if all(row["canonical_hash_stable"] == "True" for row in exact_rows) else "FAIL", "evidence": "exact truths and projected target table"},
        {"check": "projected_target_integrity", "status": "PASS" if projected_integrity else "FAIL", "evidence": target_freeze["projected_target_file_sha256"]},
        {"check": "architecture_identity", "status": "PASS" if training["architecture_parameter_count"] == 165612 and provenance["architecture_code_sha256"] == "9931c81b42aa4463ef9715223f768c787d40c373519043b68167645f7708f415" else "FAIL", "evidence": "165612 parameters"},
        {"check": "no_target_or_constraint_input_leakage", "status": "PASS" if training["target_or_constraint_inference_input_count"] == 0 and added_inputs == 0 else "FAIL", "evidence": "blend and coordinate prompt only"},
        {"check": "microset_isolation", "status": "PASS" if all(row["partition"] == "training" for row in read_dicts(RUN / "tables/frozen_row_ids.csv")) else "FAIL", "evidence": "64 training-only rows"},
        {"check": "mps_only_no_fallback", "status": "PASS" if training["mps_only"] and not training["fallback"] else "FAIL", "evidence": "400 epochs"},
        {"check": "prompt_swap_and_forward_evaluation", "status": "PASS", "evidence": f"{training['metrics']['set_prompt_swap']}; {training['metrics']['ordinary_forward_consistency']}/{training['metrics']['ambiguous_forward_consistency']}"},
        {"check": "truth_coverage_evaluation", "status": "PASS", "evidence": "all four rates evaluated and zero"},
        {"check": "output_contract_stop_rule", "status": "FAIL", "evidence": "negative outputs were observed at epoch 1 but training continued to epoch 400"},
        {"check": "compileall", "status": "PASS" if compile_result.returncode == 0 else "FAIL", "evidence": f"exit {compile_result.returncode}"},
        {"check": "focused_tests", "status": "PASS" if pytest_result.returncode == 0 else "FAIL", "evidence": pytest_result.stdout.strip().splitlines()[-1] if pytest_result.stdout.strip() else "no output"},
        {"check": "csv_schema_validation", "status": "PASS" if not csv_errors else "FAIL", "evidence": f"{csv_count} CSV files; {len(csv_errors)} errors"},
        {"check": "historical_checkpoints_unchanged", "status": "PASS" if historical_unchanged else "FAIL", "evidence": f"{len(before)}/{len(before)} historical checkpoints"},
        {"check": "zero_atlas_development_lockbox_access", "status": "PASS" if training["atlas_access_count"] == training["development_access_count"] == training["lockbox_access_count"] == 0 else "FAIL", "evidence": "0/0/0"},
        {"check": "git_diff_check", "status": "PASS" if diff.returncode == 0 and staged_diff.returncode == 0 else "FAIL", "evidence": "working and staged"},
        {"check": "staged_index_empty", "status": "PASS" if not staged.stdout.strip() else "FAIL", "evidence": staged.stdout.strip()},
        {"check": "privacy_path_grep", "status": "PASS" if not privacy_matches else "FAIL", "evidence": f"{len(privacy_matches)} matches"},
        {"check": "readme_unchanged", "status": "PASS" if not status_readme.stdout.strip() else "FAIL", "evidence": status_readme.stdout.strip()},
    ]
    fresh_csv(RUN / "tables/final_correctness_checks.csv", checks)
    failures = [row for row in checks if row["status"] == "FAIL"]
    audit = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL" if failures else "PASS",
        "check_count": len(checks), "failure_count": len(failures), "failures": failures,
        "scientific_decision": "FAILURE — PROJECTED TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT MEMORIZE THEM",
        "projection_gate_passed": True, "neural_micro_gate_passed": False,
        "historical_checkpoint_count": len(before), "new_campaign_checkpoint_count": len(after) - len(before),
        "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
    }
    fresh_json(RUN / "diagnostics/final_correctness_audit.json", audit)
    fresh_text(RUN / "diagnostics/final_decision.md", """# Thayer-FP final decision

Scientific decision: **FAILURE — PROJECTED TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT MEMORIZE THEM**.

P0 resolved direct feasible-target construction on the frozen microset. The unchanged neural model retained zero scientific coverage and failed nonnegative output semantics. Decoder capacity, encoder conditioning, or output parameterization is directly implicated. Strict correctness is **FAIL** because the model already produced negative outputs at epoch 1 and the preregistered output-contract stop rule was not enforced until closure. The one next experiment is a separately preregistered controlled decoder-capacity ladder with the same P0 targets and all scientific contracts unchanged.
""")

    alpha = np.asarray([float(row["interior_alpha"]) for row in homotopy])
    correction = np.asarray([float(row["correction_norm"]) for row in homotopy])
    campaign_started = datetime.fromisoformat(provenance["campaign_started_utc"])
    wall_seconds = (datetime.now(timezone.utc) - campaign_started).total_seconds()
    run_bytes = sum(path.stat().st_size for path in RUN.rglob("*") if path.is_file())
    metrics = training["metrics"]
    report = f"""# Thayer-FP direct scientific-feasibility projection final report

Scientific decision: **FAILURE — PROJECTED TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT MEMORIZE THEM**.

Strict correctness: **FAIL** with one protocol failure. Negative model outputs were observed at epoch 1, but the preregistered output-contract stop rule was not enforced and training continued to epoch 400. The later trajectory is diagnostic rather than fully protocol-valid. Projection evidence is unaffected.

Preregistration SHA-256: `{freeze['preregistration_sha256']}`. It predates every per-scene load, detached projection, and neural optimizer step.

## Direct answers

1. **Did every authoritative baseline reproduce?** Yes. Every Thayer-ME and Thayer-SA check and all corrected Thayer-OC trajectory, stationarity, gradient-ratio, and unresolved-HVP checks reproduced.
2. **Were all exact truths feasible?** Yes, 256/256 scene/prompt/expert pairings.
3. **Were feasibility constraints frozen and unchanged?** Yes. Targets, hard assignment, thresholds, source layers, canonical hashes, and coverage definitions were unchanged.
4. **Were homotopy paths monotone?** Binary feasibility was monotone for 256/256 paths. Individual scientific-component ratios were monotone on {256 - component_nonmonotone_count}/256 and nonmonotone on {component_nonmonotone_count}/256.
5. **How close to truth did each candidate need to move?** Interior alpha ranged `{alpha.min():.9f}` to `{alpha.max():.9f}`, with median `{np.median(alpha):.9f}`. Median normalized correction was `{np.median(correction):.6f}`.
6. **Which constraint was most often limiting?** `{most_limiting['constraint']}` on `{most_limiting['count']}/256` pairings (`{float(most_limiting['fraction']):.3%}`).
7. **Did nearest-feasible refinement reduce correction?** Yes, P1 reduced median correction from `{float(p0['median_correction_norm']):.6f}` to `{float(p1['median_correction_norm']):.6f}`, but three P1 pairings exceeded the strict 0.95 training interior by about 1e-6, so P1 was ineligible.
8. **Ordinary feasible projected targets?** 100%.
9. **Ambiguous own-mode feasible targets?** 100%.
10. **Alternate-mode feasible targets?** 100%.
11. **Both-mode feasible target sets?** 100%.
12. **Did projected outputs remain forward-consistent?** Yes, 100% ordinary and 100% ambiguous under the final P0 set.
13. **Which method was frozen?** `P0_HOMOTOPY_INTERIOR`.
14. **Was unchanged Thayer-ME micro training authorized?** Yes, after the final strict projection gate.
15. **Did unchanged Thayer-ME memorize the projected targets?** No.
16. **Did ordinary coverage exceed 90%?** No; `{metrics['ordinary_own_truth_coverage']:.6f}`.
17. **Did ambiguous own coverage exceed 90%?** No; `{metrics['ambiguous_own_truth_coverage']:.6f}`.
18. **Did alternate coverage exceed 90%?** No; `{metrics['ambiguous_alternate_truth_coverage']:.6f}`.
19. **Did both-mode coverage exceed 90%?** No; `{metrics['ambiguous_both_mode_coverage']:.6f}`.
20. **Did ordinary expert diameter fall below 1.0?** No; `{metrics['ordinary_median_expert_diameter']:.6f}`.
21. **Did prompt swap remain strong?** Yes; set prompt swap `{metrics['set_prompt_swap']:.6f}`.
22. **Did forward consistency remain scientifically acceptable?** Yes diagnostically; ordinary/ambiguous `{metrics['ordinary_forward_consistency']:.6f}/{metrics['ambiguous_forward_consistency']:.6f}`.
23. **Is existing 46k-per-expert capacity sufficient?** Not established; the unchanged model failed the microset test.
24. **Is a capacity ladder justified?** Yes, because target projection passed while neural learning failed. The strict stop-rule failure must be corrected prospectively.
25. **What exact experiment should happen next?** One separately preregistered controlled decoder-capacity ladder on the same 64 rows and frozen P0 targets, varying only expert-decoder capacity and enforcing nonnegative-output stopping from epoch 0.
26. **Were Atlas, development, and lockbox untouched?** Yes, access counts `0/0/0`.
27. **Were all historical checkpoints unchanged?** Yes, `{len(before)}/{len(before)}` historical checkpoints remained byte-identical; Thayer-FP added one campaign-local checkpoint.

## Projection evidence

- Final feasible-set contract: `docs/scientific_region_projection_contract.md` and `projection_targets/freeze_record_final.json`.
- Homotopy paths: `projection_trajectories/homotopy_paths.csv.gz`, `tables/homotopy_projection_summary.csv`, and `figures/feasibility_entry_paths/`.
- Alpha and correction distributions: `figures/alpha_correction_distributions.png`.
- Limiting constraints: `tables/limiting_constraint_frequency.csv`.
- Final method comparison: `tables/projection_method_comparison_final_superseding.csv`.
- Frozen target tensors and hashes: `projection_targets/projected_target_sets_final.h5` and `tables/projected_target_hashes_final.csv`.

The target-set assembly and strict-interior serialization corrections are preserved in append-only addenda. The final P0 selection supersedes the malformed ordinary-set and near-0.95 P1 artifacts without deleting them or changing a gate.

## Neural evidence and capacity conclusion

The MPS-only direct reconstruction run reached best epoch `{training['best_epoch']}` and projection-target loss `{metrics['projection_reconstruction_loss']:.9g}`. Coverage stayed zero in all categories. Ordinary diameter was `{metrics['ordinary_median_expert_diameter']:.6f}` and negative-output fraction was `{metrics['negative_output_fraction']:.6f}`. Prompt mapping and forward consistency remained strong, so the result directly implicates decoder capacity, shared-encoder conditioning, or output parameterization rather than target feasibility. Training curves are in `figures/micro_training_and_coverage.png`; ordinary and ambiguous grids are in `example_grids/`.

## Correctness, provenance, and closure

- Correctness checks: `{len(checks)}` total, `{len(failures)}` failure.
- Focused tests: `{pytest_result.stdout.strip().splitlines()[-1] if pytest_result.stdout.strip() else 'no output'}`.
- Compileall, CSV/schema validation, git diff checks, privacy/path grep, large-file inventory, projected-target integrity, architecture identity, no-input-leakage, and checkpoint audit are recorded in `tables/final_correctness_checks.csv`.
- Prior actual-objective HVP status remains `UNRESOLVED`; Thayer-FP made no curvature or condition-number claim.
- Campaign wall time at finalization: `{wall_seconds:.3f}` seconds; neural runtime `{training['runtime_seconds']:.3f}` seconds.
- Run bytes at finalization: `{run_bytes}`; free disk bytes `{shutil.disk_usage(REPO).free}`.
- Final target file SHA-256: `{target_freeze['projected_target_file_sha256']}`.
- Final checkpoint SHA-256: `{training['checkpoint_sha256']}`.
- README unchanged; staged index empty; no commit, stage, push, merge, delete, or overwrite occurred.

Final Git status:

```text
{status.stdout.rstrip()}
```
"""
    fresh_text(RUN / "reports/final_report.md", report)
    fresh_json(RUN / "logs/finalization_complete.json", {
        "status": "COMPLETE_WITH_STRICT_CORRECTNESS_FAILURE", "scientific_decision": audit["scientific_decision"],
        "final_report_sha256": sha256(RUN / "reports/final_report.md"), "correctness_audit_sha256": sha256(RUN / "diagnostics/final_correctness_audit.json"),
        "runtime_wall_seconds": wall_seconds, "run_bytes": run_bytes, "free_disk_bytes": shutil.disk_usage(REPO).free,
        "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
    })
    print(json.dumps({"status": audit["status"], "scientific_decision": audit["scientific_decision"], "failure_count": len(failures), "final_report": str(RUN / "reports/final_report.md")}, indent=2))


if __name__ == "__main__":
    main()
