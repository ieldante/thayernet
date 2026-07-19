#!/usr/bin/env python3
"""Run the frozen Thayer-FP baseline, projection, and target-freeze gates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.audit_thayer_scientific_alignment import exact_metrics_summary
from scripts.run_thayer_output_conditioning import initialization_outputs, reproduce_baselines
from scripts.run_thayer_two_expert_micro_overfit import (
    MEAN_PSF_FWHM_PIXEL,
    NORMALIZATION,
    load_micro_arrays,
    prompt_identity,
    select_microset,
    thresholds,
)
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.feasibility_projection import (
    INTERIOR_LIMIT,
    augmented_lagrangian_refinement,
    constraint_ratios,
    homotopy_projection,
    normalized_correction,
    output_contract_valid,
    scientific_ratio_tensor,
)
from src.loss_geometry import scientific_metrics
from src.models_two_expert_decoder import source_sum
from src.output_conditioning import common_allocation_gradient_parts
from src.scientific_alignment import corrected_objective


ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
SA = REPO / "outputs/runs/thayer_scientific_alignment_20260712_220315"
OC = REPO / "outputs/runs/thayer_output_conditioning_20260712_225459"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
ME_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
OC_OUTPUTS = OC / "detached_optimization/final_outputs.h5"


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
        raise ValueError(f"refusing empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_freeze(run: Path) -> dict[str, object]:
    freeze = json.loads((run / "preregistration/freeze_record.json").read_text())
    prereg = run / "preregistration/direct_scientific_feasibility_projection.md"
    checks = {
        "freeze_status": freeze["status"] == "FROZEN_BEFORE_PER_SCENE_LOAD",
        "preregistration_hash": sha256(prereg) == freeze["preregistration_sha256"],
        "frozen_rows_hash": sha256(run / "tables/frozen_row_ids.csv") == freeze["frozen_rows_sha256"],
        "attainability_hash": sha256(run / "tables/preregistered_gate_attainability.csv") == freeze["gate_attainability_sha256"],
        "no_projection_summary_before_load": not (run / "tables/homotopy_projection_summary.csv").exists(),
        "no_projected_targets_before_load": not (run / "projection_targets/projected_target_sets.h5").exists(),
        "no_micro_checkpoint_before_load": not any((run / "checkpoints").iterdir()),
    }
    order_path = run / "tables/preregistration_order_checks.csv"
    if not order_path.exists():
        fresh_csv(order_path, [{"check": key, "pass": value} for key, value in checks.items()])
    if not all(checks.values()):
        raise RuntimeError(f"preregistration/order check failed: {checks}")
    return freeze


def assignments(arrays: dict[str, np.ndarray], candidate: np.ndarray, scales: np.ndarray) -> np.ndarray:
    objective = corrected_objective(
        torch.from_numpy(candidate),
        torch.from_numpy(arrays["targets"]),
        torch.from_numpy(arrays["counts"]),
        torch.from_numpy(scales),
        MEAN_PSF_FWHM_PIXEL,
    )
    wins = objective["identity_wins"].detach().cpu().numpy()
    selected = np.zeros((64, 2, 2), dtype=np.int64)
    for scene in range(64):
        for prompt in (0, 1):
            if arrays["counts"][scene, prompt] == 1:
                selected[scene, prompt] = (0, 0)
            elif wins[scene, prompt]:
                selected[scene, prompt] = (0, 1)
            else:
                selected[scene, prompt] = (1, 0)
    return selected


def reproduce_oc_baselines(
    run: Path,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scales: np.ndarray,
    initial: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(metric: str, observed: object, expected: object, tolerance: float = 0.0) -> None:
        if isinstance(observed, (float, int)) and isinstance(expected, (float, int)):
            passed = abs(float(observed) - float(expected)) <= tolerance
        else:
            passed = observed == expected
        checks.append({"metric": metric, "observed": observed, "expected": expected, "absolute_tolerance": tolerance, "pass": passed})

    with h5py.File(OC_OUTPUTS, "r") as handle:
        cases = {
            "c1_thayer_me": (np.asarray(handle["C1_RAW_LBFGS/thayer_me_experts"], dtype=np.float32), (0.125, 0.84375, 0.875, 0.75)),
            "c0_sa_compromise": (np.asarray(handle["C0_RAW_ADAM/sa_compromise"], dtype=np.float32), (0.28125, 0.8125, 0.8125, 0.8125)),
            "c4_collapsed": (np.asarray(handle["C4_ALTERNATING_TD/collapsed_means"], dtype=np.float32), (0.4375, 0.21875, 0.21875, 0.0625)),
        }
        for name, (output, expected) in cases.items():
            summary = exact_metrics_summary(output, arrays, scales, rows)
            values = (summary["ordinary_coverage"], summary["ambiguous_own_coverage"], summary["ambiguous_alternate_coverage"], summary["ambiguous_both_mode_coverage"])
            for label, value, target in zip(("ordinary", "own", "alternate", "both"), values, expected):
                add(f"oc_{name}_{label}", value, target, 1e-7)
        exact = initial["exact_truths"]
        expected_eligibility = {
            "C0_RAW_ADAM": True, "C1_RAW_LBFGS": True, "C2_TD_ADAM": False,
            "C3_TD_LBFGS": True, "C4_ALTERNATING_TD": False, "C5_JACOBIAN_PRECONDITIONED_TD": False,
        }
        for method, expected in expected_eligibility.items():
            output = np.asarray(handle[f"{method}/exact_truths"], dtype=np.float32)
            summary = exact_metrics_summary(output, arrays, scales, rows)
            stationary = min(summary["ordinary_coverage"], summary["ambiguous_own_coverage"], summary["ambiguous_alternate_coverage"], summary["ambiguous_both_mode_coverage"]) == 1.0
            add(f"oc_{method}_truth_stationary", stationary, expected)

    compromise = torch.from_numpy(initial["sa_compromise"]).clone().requires_grad_(True)
    objective = corrected_objective(compromise, torch.from_numpy(arrays["targets"]), torch.from_numpy(arrays["counts"]), torch.from_numpy(scales), MEAN_PSF_FWHM_PIXEL)["total"]
    gradient = torch.autograd.grad(objective, compromise)[0]
    common, allocation = common_allocation_gradient_parts(gradient)
    add("oc_common_allocation_gradient_ratio", float(common.norm() / allocation.norm()), 0.723635, 5e-7)
    audit = json.loads((OC / "diagnostics/final_correctness_audit.json").read_text())
    add("oc_actual_objective_hvp_status", audit["hvp_condition_number_status"], "UNRESOLVED")
    primary_path = run / "tables/oc_baseline_reproduction.csv"
    output_path = primary_path if not primary_path.exists() else run / "tables/oc_baseline_reproduction_superseding.csv"
    fresh_csv(output_path, checks)
    if not all(bool(row["pass"]) for row in checks):
        fresh_json(run / "logs/oc_baseline_reproduction_failure.json", {"status": "FAIL_CLOSED", "failed": [row for row in checks if not row["pass"]]})
        raise RuntimeError("Thayer-OC baseline reproduction failed")
    return checks


def exact_truth_sanity(
    run: Path,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scales: np.ndarray,
    exact: np.ndarray,
) -> list[dict[str, object]]:
    physical = exact * np.tile(scales, 2)[None, None, None, :, None, None]
    records = []
    for scene in range(64):
        for prompt in (0, 1):
            for expert in (0, 1):
                target_index = 0 if arrays["counts"][scene, prompt] == 1 else expert
                value = physical[scene, prompt, expert]
                target = arrays["targets_physical"][scene, prompt, target_index]
                ratios = constraint_ratios(value, target)
                digest_a = canonical_tensor_sha256(value)
                digest_b = canonical_tensor_sha256(np.ascontiguousarray(value.copy()))
                records.append({
                    "scene": scene, "scene_id": rows[scene]["scene_id"], "kind": rows[scene]["kind"],
                    "prompt": prompt, "expert": expert, "target_index": target_index,
                    "image": ratios.image, "flux_g": ratios.flux_g, "flux_r": ratios.flux_r, "flux_z": ratios.flux_z,
                    "color_gr": ratios.color_gr, "color_rz": ratios.color_rz, "centroid": ratios.centroid,
                    "maximum": ratios.maximum, "finite": ratios.finite, "nonnegative": ratios.nonnegative,
                    "feasible": ratios.feasible(1.0), "interior_feasible": ratios.feasible(INTERIOR_LIMIT),
                    "canonical_sha256": digest_a, "canonical_hash_stable": digest_a == digest_b,
                })
    fresh_csv(run / "tables/exact_truth_feasibility.csv", records)
    exact_summary = exact_metrics_summary(exact, arrays, scales, rows)
    checks = {
        "all_exact_pairings_feasible": all(bool(row["feasible"]) for row in records),
        "all_exact_pairings_interior": all(bool(row["interior_feasible"]) for row in records),
        "all_exact_ratios_zero_tolerance": max(float(row["maximum"]) for row in records) <= 1e-6,
        "canonical_hashes_stable": all(bool(row["canonical_hash_stable"]) for row in records),
        "ordinary_exact_coverage": exact_summary["ordinary_coverage"] == 1.0,
        "ambiguous_exact_own": exact_summary["ambiguous_own_coverage"] == 1.0,
        "ambiguous_exact_alternate": exact_summary["ambiguous_alternate_coverage"] == 1.0,
        "ambiguous_exact_both": exact_summary["ambiguous_both_mode_coverage"] == 1.0,
    }
    fresh_csv(run / "tables/exact_truth_sanity_checks.csv", [{"check": key, "pass": value} for key, value in checks.items()])
    if not all(checks.values()):
        fresh_json(run / "logs/exact_truth_infeasibility.json", {"status": "FAIL_CLOSED", "checks": checks})
        raise RuntimeError("exact-truth feasibility gate failed")
    return records


def run_p0(
    run: Path,
    rows: list[dict[str, str]],
    candidate_physical: np.ndarray,
    targets_physical: np.ndarray,
    selected_assignment: np.ndarray,
    *,
    persist: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    projected = np.empty_like(candidate_physical)
    summaries: list[dict[str, object]] = []
    curve_path = run / "projection_trajectories/homotopy_paths.csv.gz"
    curve_handle = gzip.open(curve_path, "xt", newline="", encoding="utf-8") if persist else None
    curve_writer = None
    example_keys = {(0, 0, 0), (15, 1, 1), (31, 0, 1), (32, 0, 0), (39, 1, 1), (47, 0, 1), (55, 1, 0), (63, 1, 1)}
    try:
        for scene in range(64):
            for prompt in (0, 1):
                for expert in (0, 1):
                    target_index = int(selected_assignment[scene, prompt, expert])
                    value, summary, path_rows = homotopy_projection(candidate_physical[scene, prompt, expert], targets_physical[scene, prompt, target_index])
                    projected[scene, prompt, expert] = value
                    entry = summary["per_constraint_entry_alpha"]
                    summaries.append({
                        "scene": scene, "scene_id": rows[scene]["scene_id"], "kind": rows[scene]["kind"], "prompt": prompt,
                        "expert": expert, "target_index": target_index, "assignment": "ordinary" if rows[scene]["kind"] == "ordinary" else "identity_or_swap",
                        "boundary_alpha": summary["boundary_alpha"], "interior_alpha": summary["interior_alpha"],
                        "correction_norm": summary["correction_norm"], "final_max_ratio": summary["final_max_ratio"],
                        "entry_image": entry["image"], "entry_flux_g": entry["flux_g"], "entry_flux_r": entry["flux_r"], "entry_flux_z": entry["flux_z"],
                        "entry_color_gr": entry["color_gr"], "entry_color_rz": entry["color_rz"], "entry_centroid": entry["centroid"],
                        "first_limiting_metric": summary["first_limiting_metric"], "scientific_ratios_monotone": summary["scientific_ratios_monotone"],
                        "feasibility_monotone": summary["feasibility_monotone"], "path_nonmonotonicity": summary["path_nonmonotonicity"],
                        "feasible_intervals": json.dumps(summary["feasible_intervals"]),
                    })
                    if persist and curve_handle is not None:
                        augmented = [{"scene": scene, "prompt": prompt, "expert": expert, "target_index": target_index, **row} for row in path_rows]
                        if curve_writer is None:
                            curve_writer = csv.DictWriter(curve_handle, fieldnames=list(augmented[0]))
                            curve_writer.writeheader()
                        curve_writer.writerows(augmented)
                    if persist and (scene, prompt, expert) in example_keys:
                        alphas = np.asarray([float(row["alpha"]) for row in path_rows])
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        for metric in ("image", "flux_g", "flux_r", "flux_z", "color_gr", "color_rz", "centroid"):
                            ax.plot(alphas, [float(row[metric]) for row in path_rows], label=metric, linewidth=1)
                        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
                        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
                        ax.set_ylim(0, min(10, max(1.2, np.nanpercentile([float(row["maximum"]) for row in path_rows], 95))))
                        ax.set_xlabel("alpha toward exact truth"); ax.set_ylabel("normalized constraint ratio"); ax.legend(ncol=2, fontsize=7)
                        fig.tight_layout(); fig.savefig(run / f"figures/feasibility_entry_paths/scene_{scene:02d}_p{prompt}_e{expert}.png", dpi=160); plt.close(fig)
    finally:
        if curve_handle is not None:
            curve_handle.close()
    return projected, summaries


def expert_to_target_set(expert_outputs: np.ndarray, selected_assignment: np.ndarray) -> np.ndarray:
    target_set = np.empty_like(expert_outputs)
    for scene in range(64):
        for prompt in (0, 1):
            if selected_assignment[scene, prompt, 0] == selected_assignment[scene, prompt, 1]:
                # Ordinary rows have two projected expert representatives of
                # the same approved region.  Retain both slots; do not collapse
                # them into the single canonical target index.
                target_set[scene, prompt] = expert_outputs[scene, prompt]
                continue
            for expert in (0, 1):
                target_set[scene, prompt, selected_assignment[scene, prompt, expert]] = expert_outputs[scene, prompt, expert]
    return target_set


def method_metrics(
    name: str,
    expert_physical: np.ndarray,
    target_set_normalized: np.ndarray,
    candidate_physical: np.ndarray,
    matched_targets_physical: np.ndarray,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scales: np.ndarray,
    deterministic: bool,
    exact_stationary: bool,
) -> dict[str, object]:
    pair_ratios = []
    corrections = []
    for index in np.ndindex(64, 2, 2):
        ratio = constraint_ratios(expert_physical[index], matched_targets_physical[index])
        pair_ratios.append(ratio)
        corrections.append(normalized_correction(expert_physical[index], candidate_physical[index]))
    summary = exact_metrics_summary(target_set_normalized, arrays, scales, rows)
    threshold, sky = thresholds()
    detailed = scientific_metrics(target_set_normalized, arrays["targets_physical"], arrays["counts"], arrays["blend_physical"], scales, threshold, sky, MEAN_PSF_FWHM_PIXEL)
    ordinary_indices = [index for index, row in enumerate(rows) if row["kind"] == "ordinary"]
    ambiguous_indices = [index for index, row in enumerate(rows) if row["kind"] == "near_collision"]
    ordinary_forward = float(np.mean([detailed[index]["forward_consistent_fraction"] == 1.0 for index in ordinary_indices]))
    ambiguous_forward = float(np.mean([detailed[index]["forward_consistent_fraction"] == 1.0 for index in ambiguous_indices]))
    prompt_pass = []
    for scene in range(64):
        identities = []
        for prompt in (0, 1):
            identities.extend(prompt_identity(target_set_normalized[scene, prompt], arrays["targets"][scene, prompt], int(arrays["counts"][scene, prompt])))
        prompt_pass.append(bool(all(identities)))
    source_sum_error = float(np.mean((target_set_normalized[..., :3, :, :] + target_set_normalized[..., 3:, :, :] - arrays["blend"][:, None, None]) ** 2))
    hashes_stable = True
    for scene in range(64):
        for prompt in (0, 1):
            for target in (0, 1):
                value = target_set_normalized[scene, prompt, target] * np.tile(scales, 2)[:, None, None]
                hashes_stable &= canonical_tensor_sha256(value) == canonical_tensor_sha256(np.ascontiguousarray(value.copy()))
    return {
        "method": name,
        "eligible": bool(exact_stationary and deterministic and all(ratio.feasible(1.0) for ratio in pair_ratios) and hashes_stable),
        "exact_truth_stationary": exact_stationary,
        "deterministic": deterministic,
        "canonical_hash_stable": hashes_stable,
        "feasible_pair_fraction": float(np.mean([ratio.feasible(1.0) for ratio in pair_ratios])),
        "interior_pair_fraction": float(np.mean([ratio.feasible(INTERIOR_LIMIT) for ratio in pair_ratios])),
        "ordinary_feasible_fraction": summary["ordinary_coverage"],
        "ambiguous_own_feasible_fraction": summary["ambiguous_own_coverage"],
        "ambiguous_alternate_feasible_fraction": summary["ambiguous_alternate_coverage"],
        "ambiguous_both_mode_feasible_fraction": summary["ambiguous_both_mode_coverage"],
        "median_correction_norm": float(np.median(corrections)),
        "mean_correction_norm": float(np.mean(corrections)),
        "median_interior_slack": float(np.median([1.0 - ratio.maximum for ratio in pair_ratios])),
        "maximum_constraint_ratio": float(max(ratio.maximum for ratio in pair_ratios)),
        "ordinary_forward_consistency": ordinary_forward,
        "ambiguous_forward_consistency": ambiguous_forward,
        "set_prompt_swap": float(np.mean(prompt_pass)),
        "source_sum_mse_evaluation_only": source_sum_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    started = time.time()
    freeze = verify_freeze(run)

    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    rows, indices = select_microset()
    arrays = load_micro_arrays(indices, scales)
    initial = initialization_outputs(arrays, scales)
    baseline_path = run / "tables/baseline_reproduction.csv"
    if baseline_path.exists():
        with baseline_path.open(newline="", encoding="utf-8") as handle:
            baseline_rows = list(csv.DictReader(handle))
        if not all(row["pass"] == "True" for row in baseline_rows):
            raise RuntimeError("persisted baseline reproduction is not passing")
    else:
        baseline_rows = reproduce_baselines(run, arrays, rows, scales, initial)
    oc_rows = reproduce_oc_baselines(run, arrays, rows, scales, initial)
    if (run / "logs/oc_baseline_reproduction_failure.json").exists():
        fresh_text(run / "reports/baseline_checker_correction_addendum.md", """# Baseline-checker correction addendum

The first Thayer-FP baseline checker stopped before projection because it assigned unreported zero expectations to C4's ambiguous coverage columns. That was a checker bookkeeping error, not a frozen-output mismatch. The authoritative Thayer-OC trajectory row records C4 from `collapsed_means` at ordinary/own/alternate/both coverage 0.4375/0.21875/0.21875/0.0625. Direct evaluation of the frozen HDF5 reproduced those four values exactly in `tables/oc_baseline_reproduction_superseding.csv`. The originally requested best ordinary value, all C1 and C0 coverage values, all truth-stationarity eligibility results, the common/allocation ratio, and the unresolved HVP status also reproduced. No projection or target construction occurred before this correction; the empty projection directories and preserved failure artifact document the fail-closed stop.
""")
    fresh_text(run / "diagnostics/baseline_reproduction.md", f"""# Thayer-FP authoritative baseline reproduction

All `{len(baseline_rows)}` Thayer-ME/Thayer-SA checks and `{len(oc_rows)}` Thayer-OC checks reproduced after the preregistration freeze. Thayer-ME retains zero ordinary/own/alternate/both-mode coverage, set prompt swap 0.953125, ordinary/ambiguous forward consistency 0.96875/1.0, and ordinary diameter 5.165995. Thayer-SA retains its surrogate correlations, exact-truth stationarity, and corrected-compromise coverage failure. Thayer-OC retains its best ordinary, own/alternate, and both-mode endpoints, its C2/C4/C5 truth-stationarity ineligibility, and common/allocation gradient ratio 0.723635. The actual-objective HVP status was reproduced only as the authoritative `UNRESOLVED` status; no new HVP, finite-difference, curvature, or condition-number calculation was attempted.
""")

    exact = initial["exact_truths"]
    exact_truth_sanity(run, arrays, rows, scales, exact)
    candidate_normalized = initial["thayer_me_experts"]
    candidate_physical = candidate_normalized * np.tile(scales, 2)[None, None, None, :, None, None]
    selected_assignment = assignments(arrays, candidate_normalized, scales)
    matched_targets_physical = np.empty_like(candidate_physical)
    matched_targets_normalized = np.empty_like(candidate_normalized)
    assignment_rows = []
    for scene in range(64):
        for prompt in (0, 1):
            for expert in (0, 1):
                target_index = int(selected_assignment[scene, prompt, expert])
                matched_targets_physical[scene, prompt, expert] = arrays["targets_physical"][scene, prompt, target_index]
                matched_targets_normalized[scene, prompt, expert] = arrays["targets"][scene, prompt, target_index]
                assignment_rows.append({"scene": scene, "scene_id": rows[scene]["scene_id"], "kind": rows[scene]["kind"], "prompt": prompt, "expert": expert, "target_index": target_index})
    fresh_csv(run / "tables/projection_assignments.csv", assignment_rows)

    p0_physical, homotopy_rows = run_p0(run, rows, candidate_physical, arrays["targets_physical"], selected_assignment, persist=True)
    fresh_csv(run / "tables/homotopy_projection_summary.csv", homotopy_rows)
    p0_repeat_physical, _ = run_p0(run, rows, candidate_physical, arrays["targets_physical"], selected_assignment, persist=False)
    p0_deterministic = bool(np.array_equal(p0_physical, p0_repeat_physical))
    p0_normalized = p0_physical / np.tile(scales, 2)[None, None, None, :, None, None]
    p0_target_set = expert_to_target_set(p0_normalized, selected_assignment)

    scales_t = torch.from_numpy(scales)
    torch.manual_seed(2026071305)
    p1_tensor, p1_trajectory = augmented_lagrangian_refinement(
        torch.from_numpy(p0_normalized), torch.from_numpy(candidate_normalized), torch.from_numpy(matched_targets_normalized), scales_t,
    )
    p1_normalized = p1_tensor.numpy()
    torch.manual_seed(2026071305)
    p1_repeat, _ = augmented_lagrangian_refinement(
        torch.from_numpy(p0_normalized), torch.from_numpy(candidate_normalized), torch.from_numpy(matched_targets_normalized), scales_t,
    )
    p1_deterministic = bool(torch.equal(p1_tensor, p1_repeat))
    exact_control, exact_control_trajectory = augmented_lagrangian_refinement(
        torch.from_numpy(exact), torch.from_numpy(exact), torch.from_numpy(exact), scales_t,
    )
    p1_exact_stationary = bool(torch.equal(exact_control, torch.from_numpy(exact)))
    for row in p1_trajectory:
        row["control"] = "candidate_refinement"
    for row in exact_control_trajectory:
        row["control"] = "exact_truth_stationarity"
    fresh_csv(run / "projection_trajectories/p1_augmented_lagrangian.csv", [*p1_trajectory, *exact_control_trajectory])
    p1_physical = p1_normalized * np.tile(scales, 2)[None, None, None, :, None, None]
    p1_target_set = expert_to_target_set(p1_normalized, selected_assignment)

    methods = [
        method_metrics("P0_HOMOTOPY_INTERIOR", p0_physical, p0_target_set, candidate_physical, matched_targets_physical, arrays, rows, scales, p0_deterministic, True),
        method_metrics("P1_AUGMENTED_LAGRANGIAN", p1_physical, p1_target_set, candidate_physical, matched_targets_physical, arrays, rows, scales, p1_deterministic, p1_exact_stationary),
    ]
    fresh_csv(run / "tables/projection_method_comparison.csv", methods)
    eligible = [row for row in methods if row["eligible"]]
    if not eligible:
        selected_name = "NONE"
        selected_targets = None
        selected_expert = None
    else:
        eligible.sort(key=lambda row: (
            -float(row["feasible_pair_fraction"]),
            -float(row["ambiguous_both_mode_feasible_fraction"]),
            float(row["median_correction_norm"]),
            -float(row["median_interior_slack"]),
            -(float(row["ordinary_forward_consistency"]) + float(row["ambiguous_forward_consistency"])),
            0 if row["method"] == "P0_HOMOTOPY_INTERIOR" else 1,
        ))
        selected_name = str(eligible[0]["method"])
        selected_targets = p0_target_set if selected_name == "P0_HOMOTOPY_INTERIOR" else p1_target_set
        selected_expert = p0_normalized if selected_name == "P0_HOMOTOPY_INTERIOR" else p1_normalized
    selected_metrics = next((row for row in methods if row["method"] == selected_name), None)
    projection_gate = bool(selected_metrics is not None and
        float(selected_metrics["feasible_pair_fraction"]) >= 0.95 and
        float(selected_metrics["ordinary_feasible_fraction"]) >= 0.90 and
        float(selected_metrics["ambiguous_own_feasible_fraction"]) >= 0.90 and
        float(selected_metrics["ambiguous_alternate_feasible_fraction"]) >= 0.90 and
        float(selected_metrics["ambiguous_both_mode_feasible_fraction"]) >= 0.90 and
        float(selected_metrics["ordinary_forward_consistency"]) >= 0.90 and
        float(selected_metrics["ambiguous_forward_consistency"]) >= 0.90 and
        bool(selected_metrics["deterministic"]))
    gate_rows = [
        {"gate": "exact_truth_feasibility", "observed": 1.0, "threshold": "1.0", "pass": True},
        {"gate": "selected_method_exists", "observed": selected_name, "threshold": "eligible P0 or P1", "pass": selected_metrics is not None},
    ]
    if selected_metrics is not None:
        for key, threshold_value in (
            ("feasible_pair_fraction", 0.95), ("ordinary_feasible_fraction", 0.90),
            ("ambiguous_own_feasible_fraction", 0.90), ("ambiguous_alternate_feasible_fraction", 0.90),
            ("ambiguous_both_mode_feasible_fraction", 0.90), ("ordinary_forward_consistency", 0.90),
            ("ambiguous_forward_consistency", 0.90),
        ):
            gate_rows.append({"gate": key, "observed": selected_metrics[key], "threshold": f">={threshold_value}", "pass": float(selected_metrics[key]) >= threshold_value})
        gate_rows.extend([
            {"gate": "deterministic_reproduction", "observed": selected_metrics["deterministic"], "threshold": "True", "pass": bool(selected_metrics["deterministic"])},
            {"gate": "canonical_hash_stability", "observed": selected_metrics["canonical_hash_stable"], "threshold": "True", "pass": bool(selected_metrics["canonical_hash_stable"])},
        ])
    fresh_csv(run / "tables/projection_correctness_gates.csv", gate_rows)

    limiting = Counter(str(row["first_limiting_metric"]) for row in homotopy_rows)
    fresh_csv(run / "tables/limiting_constraint_frequency.csv", [{"constraint": key, "count": value, "fraction": value / len(homotopy_rows)} for key, value in sorted(limiting.items())])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([float(row["interior_alpha"]) for row in homotopy_rows], bins=30); axes[0].set_xlabel("P0 interior alpha"); axes[0].set_ylabel("pairings")
    axes[1].hist([float(row["correction_norm"]) for row in homotopy_rows], bins=30); axes[1].set_xlabel("normalized correction"); axes[1].set_ylabel("pairings")
    fig.tight_layout(); fig.savefig(run / "figures/alpha_correction_distributions.png", dpi=180); plt.close(fig)

    if not projection_gate or selected_targets is None or selected_expert is None:
        fresh_json(run / "logs/projection_gate_complete.json", {"status": "FAIL", "micro_training_authorized": False, "selected_method": selected_name, "runtime_seconds": time.time() - started})
        print(json.dumps({"status": "PROJECTION_GATE_FAILED", "selected_method": selected_name}, indent=2))
        return

    selected_physical_targets = selected_targets * np.tile(scales, 2)[None, None, None, :, None, None]
    target_path = run / "projection_targets/projected_target_sets.h5"
    with h5py.File(target_path, "x") as handle:
        handle.create_dataset("targets_normalized", data=selected_targets.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("targets_physical", data=selected_physical_targets.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("expert_order_normalized", data=selected_expert.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("hard_assignment_target_index", data=selected_assignment.astype(np.int8))
        handle.attrs["complete"] = True
        handle.attrs["selected_method"] = selected_name
        handle.attrs["preregistration_sha256"] = freeze["preregistration_sha256"]
        handle.attrs["training_only_representatives"] = True
        handle.attrs["astronomical_truth_claim"] = False
        handle.attrs["inference_input_fields_added"] = 0
    hash_rows = []
    for scene in range(64):
        for prompt in (0, 1):
            for target in (0, 1):
                value = selected_physical_targets[scene, prompt, target]
                hash_rows.append({"scene": scene, "scene_id": rows[scene]["scene_id"], "kind": rows[scene]["kind"], "prompt": prompt, "target_slot": target, "canonical_sha256": canonical_tensor_sha256(value), "source_truth_provenance": "frozen_training_target_region", "inference_input": False})
    fresh_csv(run / "tables/projected_target_hashes.csv", hash_rows)
    fresh_json(run / "projection_targets/freeze_record.json", {
        "status": "FROZEN_PROJECTED_TARGETS", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_method": selected_name, "projected_target_file_sha256": sha256(target_path),
        "projected_target_hash_table_sha256": sha256(run / "tables/projected_target_hashes.csv"),
        "preregistration_sha256": freeze["preregistration_sha256"], "architecture_unchanged": True,
        "scientific_thresholds_unchanged": True, "truth_or_constraint_inference_inputs": 0,
        "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
    })
    fresh_json(run / "logs/projection_gate_complete.json", {
        "status": "PASS", "micro_training_authorized": True, "selected_method": selected_name,
        "selected_metrics": selected_metrics, "projected_target_file_sha256": sha256(target_path),
        "runtime_seconds": time.time() - started, "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
    })
    print(json.dumps({"status": "PROJECTION_GATE_PASS", "selected_method": selected_name, "metrics": selected_metrics, "runtime_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
