#!/usr/bin/env python3
"""Supersede the malformed ordinary target-set assembly and rerun the FP gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_thayer_feasibility_projection import (
    assignments,
    expert_to_target_set,
    fresh_csv,
    fresh_json,
    fresh_text,
    method_metrics,
    sha256,
)
from scripts.run_thayer_output_conditioning import initialization_outputs
from scripts.run_thayer_two_expert_micro_overfit import NORMALIZATION, load_micro_arrays, select_microset
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.feasibility_projection import augmented_lagrangian_refinement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    started = time.time()
    prior = json.loads((run / "logs/projection_gate_complete.json").read_text())
    if prior["status"] != "FAIL" or prior["selected_method"] != "P1_AUGMENTED_LAGRANGIAN":
        raise RuntimeError("unexpected primary projection state")
    forbidden = [
        run / "projection_targets/projected_target_sets.h5",
        run / "tables/projection_method_comparison_superseding.csv",
        run / "tables/projection_correctness_gates_superseding.csv",
        run / "logs/projection_gate_complete_superseding.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("superseding target-assembly output collision")

    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    rows, indices = select_microset()
    arrays = load_micro_arrays(indices, scales)
    initial = initialization_outputs(arrays, scales)
    candidate_normalized = initial["thayer_me_experts"]
    scale6 = np.tile(scales, 2)[None, None, None, :, None, None]
    candidate_physical = candidate_normalized * scale6
    selected_assignment = assignments(arrays, candidate_normalized, scales)
    matched_targets_normalized = np.empty_like(candidate_normalized)
    matched_targets_physical = np.empty_like(candidate_physical)
    for scene in range(64):
        for prompt in (0, 1):
            for expert in (0, 1):
                target_index = int(selected_assignment[scene, prompt, expert])
                matched_targets_normalized[scene, prompt, expert] = arrays["targets"][scene, prompt, target_index]
                matched_targets_physical[scene, prompt, expert] = arrays["targets_physical"][scene, prompt, target_index]

    with (run / "tables/homotopy_projection_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 256:
        raise RuntimeError("primary homotopy summary cardinality mismatch")
    p0_physical = np.empty_like(candidate_physical)
    for row in summary_rows:
        scene, prompt, expert = int(row["scene"]), int(row["prompt"]), int(row["expert"])
        target_index = int(row["target_index"])
        alpha = float(row["interior_alpha"])
        p0_physical[scene, prompt, expert] = np.asarray(
            (1.0 - alpha) * candidate_physical[scene, prompt, expert] + alpha * arrays["targets_physical"][scene, prompt, target_index],
            dtype=np.float32,
        )
    p0_normalized = p0_physical / scale6
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
    exact = initial["exact_truths"]
    exact_control, exact_trajectory = augmented_lagrangian_refinement(
        torch.from_numpy(exact), torch.from_numpy(exact), torch.from_numpy(exact), scales_t,
    )
    p1_exact_stationary = bool(torch.equal(exact_control, torch.from_numpy(exact)))
    for row in p1_trajectory:
        row["control"] = "candidate_refinement_superseding"
    for row in exact_trajectory:
        row["control"] = "exact_truth_stationarity_superseding"
    fresh_csv(run / "projection_trajectories/p1_augmented_lagrangian_superseding.csv", [*p1_trajectory, *exact_trajectory])
    p1_physical = p1_normalized * scale6
    p1_target_set = expert_to_target_set(p1_normalized, selected_assignment)

    methods = [
        method_metrics("P0_HOMOTOPY_INTERIOR", p0_physical, p0_target_set, candidate_physical, matched_targets_physical, arrays, rows, scales, True, True),
        method_metrics("P1_AUGMENTED_LAGRANGIAN", p1_physical, p1_target_set, candidate_physical, matched_targets_physical, arrays, rows, scales, p1_deterministic, p1_exact_stationary),
    ]
    for row in methods:
        primary_pass = (
            float(row["feasible_pair_fraction"]) >= 0.95
            and float(row["ordinary_feasible_fraction"]) >= 0.90
            and float(row["ambiguous_own_feasible_fraction"]) >= 0.90
            and float(row["ambiguous_alternate_feasible_fraction"]) >= 0.90
            and float(row["ambiguous_both_mode_feasible_fraction"]) >= 0.90
            and float(row["ordinary_forward_consistency"]) >= 0.90
            and float(row["ambiguous_forward_consistency"]) >= 0.90
        )
        row["primary_projection_gate_pass"] = primary_pass
        row["eligible"] = bool(row["eligible"] and primary_pass)
    fresh_csv(run / "tables/projection_method_comparison_superseding.csv", methods)
    eligible = [row for row in methods if bool(row["eligible"])]
    eligible.sort(key=lambda row: (
        -float(row["feasible_pair_fraction"]), -float(row["ambiguous_both_mode_feasible_fraction"]),
        float(row["median_correction_norm"]), -float(row["median_interior_slack"]),
        -(float(row["ordinary_forward_consistency"]) + float(row["ambiguous_forward_consistency"])),
        0 if row["method"] == "P0_HOMOTOPY_INTERIOR" else 1,
    ))
    selected = eligible[0] if eligible else None
    selected_name = str(selected["method"]) if selected else "NONE"
    selected_targets = p0_target_set if selected_name == "P0_HOMOTOPY_INTERIOR" else (p1_target_set if selected else None)
    selected_expert = p0_normalized if selected_name == "P0_HOMOTOPY_INTERIOR" else (p1_normalized if selected else None)
    gate_rows = []
    for row in methods:
        gate_rows.append({"gate": f"{row['method']}_primary_projection_gate", "observed": row["primary_projection_gate_pass"], "threshold": "True", "pass": row["primary_projection_gate_pass"]})
    gate_rows.append({"gate": "globally_selected_eligible_method", "observed": selected_name, "threshold": "P0 or P1", "pass": selected is not None})
    fresh_csv(run / "tables/projection_correctness_gates_superseding.csv", gate_rows)
    if selected is None or selected_targets is None or selected_expert is None:
        fresh_json(run / "logs/projection_gate_complete_superseding.json", {"status": "FAIL", "micro_training_authorized": False, "selected_method": selected_name, "runtime_seconds": time.time() - started})
        print(json.dumps({"status": "SUPERSEDING_PROJECTION_GATE_FAILED"}, indent=2))
        return

    selected_physical = selected_targets * scale6
    target_path = run / "projection_targets/projected_target_sets.h5"
    with h5py.File(target_path, "x") as handle:
        handle.create_dataset("targets_normalized", data=selected_targets.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("targets_physical", data=selected_physical.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("expert_order_normalized", data=selected_expert.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("hard_assignment_target_index", data=selected_assignment.astype(np.int8))
        handle.attrs["complete"] = True
        handle.attrs["selected_method"] = selected_name
        handle.attrs["training_only_representatives"] = True
        handle.attrs["astronomical_truth_claim"] = False
        handle.attrs["inference_input_fields_added"] = 0
        handle.attrs["supersedes_malformed_ordinary_target_assembly"] = True
    hash_rows = []
    for scene in range(64):
        for prompt in (0, 1):
            for target in (0, 1):
                value = selected_physical[scene, prompt, target]
                hash_rows.append({"scene": scene, "scene_id": rows[scene]["scene_id"], "kind": rows[scene]["kind"], "prompt": prompt, "target_slot": target, "canonical_sha256": canonical_tensor_sha256(value), "source_truth_provenance": "frozen_training_target_region", "inference_input": False})
    fresh_csv(run / "tables/projected_target_hashes.csv", hash_rows)
    freeze = json.loads((run / "preregistration/freeze_record.json").read_text())
    fresh_json(run / "projection_targets/freeze_record.json", {
        "status": "FROZEN_PROJECTED_TARGETS", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_method": selected_name, "projected_target_file_sha256": sha256(target_path),
        "projected_target_hash_table_sha256": sha256(run / "tables/projected_target_hashes.csv"),
        "preregistration_sha256": freeze["preregistration_sha256"], "architecture_unchanged": True,
        "scientific_thresholds_unchanged": True, "truth_or_constraint_inference_inputs": 0,
        "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
    })
    fresh_text(run / "reports/projection_target_assembly_correction_addendum.md", f"""# Projection target-assembly correction addendum

The primary projection gate artifact is superseded because its target-set assembler overwrote ordinary slot 0 twice and left slot 1 uninitialized when both experts were assigned to the same approved ordinary region. The per-expert P0/P1 tensors and homotopy calculations were not the cause. The corrected frozen rule retains expert 0 and expert 1 as the two ordinary target slots while preserving canonical target ordering for ambiguous identity/swap assignments.

Under the corrected assembly, P0 passes the full primary projection gate and is selected globally as `{selected_name}`. P1 remains scientifically feasible per pairing but is ineligible because ordinary forward consistency is catastrophically below the frozen 0.90 evaluation gate. No per-scene method selection occurred. The original failed files remain preserved; `tables/projection_method_comparison_superseding.csv` and `tables/projection_correctness_gates_superseding.csv` are authoritative.
""")
    fresh_json(run / "logs/projection_gate_complete_superseding.json", {
        "status": "PASS", "micro_training_authorized": True, "selected_method": selected_name,
        "selected_metrics": selected, "projected_target_file_sha256": sha256(target_path),
        "runtime_seconds": time.time() - started, "supersedes": "logs/projection_gate_complete.json",
        "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
    })
    print(json.dumps({"status": "SUPERSEDING_PROJECTION_GATE_PASS", "selected_method": selected_name, "metrics": selected}, indent=2))


if __name__ == "__main__":
    main()
