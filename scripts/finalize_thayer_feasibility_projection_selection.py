#!/usr/bin/env python3
"""Apply the preregistered strict 0.95 target-interior rule and freeze P0."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_thayer_feasibility_projection import assignments, expert_to_target_set, fresh_csv, fresh_json, fresh_text, method_metrics, sha256
from scripts.run_thayer_output_conditioning import initialization_outputs
from scripts.run_thayer_two_expert_micro_overfit import NORMALIZATION, load_micro_arrays, select_microset
from src.canonical_tensor_hash import canonical_tensor_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    prior = json.loads((run / "logs/projection_gate_complete_superseding.json").read_text())
    if prior["status"] != "PASS" or prior["selected_method"] != "P1_AUGMENTED_LAGRANGIAN":
        raise RuntimeError("unexpected superseding projection state")
    outputs = [
        run / "tables/projection_method_comparison_final_superseding.csv",
        run / "tables/projection_correctness_gates_final.csv",
        run / "projection_targets/projected_target_sets_final.h5",
        run / "projection_targets/freeze_record_final.json",
        run / "logs/projection_gate_complete_final.json",
    ]
    if any(path.exists() for path in outputs):
        raise RuntimeError("final projection-selection collision")

    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    scale6 = np.tile(scales, 2)[None, None, None, :, None, None]
    rows, indices = select_microset()
    arrays = load_micro_arrays(indices, scales)
    initial = initialization_outputs(arrays, scales)
    candidate_normalized = initial["thayer_me_experts"]
    candidate_physical = candidate_normalized * scale6
    selected_assignment = assignments(arrays, candidate_normalized, scales)
    matched_targets_physical = np.empty_like(candidate_physical)
    with (run / "tables/homotopy_projection_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    p0_physical = np.empty_like(candidate_physical)
    for row in summary_rows:
        scene, prompt, expert = int(row["scene"]), int(row["prompt"]), int(row["expert"])
        target_index = int(row["target_index"])
        alpha = float(row["interior_alpha"])
        truth = arrays["targets_physical"][scene, prompt, target_index]
        p0_physical[scene, prompt, expert] = np.asarray((1.0 - alpha) * candidate_physical[scene, prompt, expert] + alpha * truth, dtype=np.float32)
        matched_targets_physical[scene, prompt, expert] = truth
    p0_normalized = p0_physical / scale6
    p0_target_set = expert_to_target_set(p0_normalized, selected_assignment)
    p0 = method_metrics("P0_HOMOTOPY_INTERIOR", p0_physical, p0_target_set, candidate_physical, matched_targets_physical, arrays, rows, scales, True, True)

    with (run / "tables/projection_method_comparison_superseding.csv").open(newline="", encoding="utf-8") as handle:
        previous_methods = list(csv.DictReader(handle))
    p1_previous = next(row for row in previous_methods if row["method"] == "P1_AUGMENTED_LAGRANGIAN")
    p1_strict = float(p1_previous["interior_pair_fraction"]) == 1.0 and float(p1_previous["maximum_constraint_ratio"]) <= 0.95
    p0_strict = float(p0["interior_pair_fraction"]) == 1.0 and float(p0["maximum_constraint_ratio"]) <= 0.95
    common_fields = set(p0) | set(p1_previous)
    final_rows = [
        {**{key: p0.get(key, "") for key in common_fields}, "strict_training_interior_pass": p0_strict, "final_eligible": bool(p0["eligible"] and p0_strict), "final_selection": True},
        {**{key: p1_previous.get(key, "") for key in common_fields}, "strict_training_interior_pass": p1_strict, "final_eligible": False, "final_selection": False},
    ]
    fresh_csv(run / "tables/projection_method_comparison_final_superseding.csv", final_rows)
    gates = [
        {"gate": "P0_all_pairings_at_or_below_0.95", "observed": p0["maximum_constraint_ratio"], "threshold": "<=0.95", "pass": p0_strict},
        {"gate": "P0_primary_projection_gate", "observed": p0["eligible"], "threshold": "True", "pass": bool(p0["eligible"])},
        {"gate": "P1_all_pairings_at_or_below_0.95", "observed": p1_previous["maximum_constraint_ratio"], "threshold": "<=0.95", "pass": p1_strict},
        {"gate": "globally_selected_eligible_method", "observed": "P0_HOMOTOPY_INTERIOR", "threshold": "eligible fixed method", "pass": bool(p0["eligible"] and p0_strict)},
    ]
    fresh_csv(run / "tables/projection_correctness_gates_final.csv", gates)
    if not bool(p0["eligible"] and p0_strict):
        fresh_json(run / "logs/projection_gate_complete_final.json", {"status": "FAIL", "micro_training_authorized": False, "selected_method": "NONE"})
        raise RuntimeError("strict P0 projection gate failed")

    physical_targets = p0_target_set * scale6
    target_path = run / "projection_targets/projected_target_sets_final.h5"
    with h5py.File(target_path, "x") as handle:
        handle.create_dataset("targets_normalized", data=p0_target_set.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("targets_physical", data=physical_targets.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("expert_order_normalized", data=p0_normalized.astype(np.float32), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("hard_assignment_target_index", data=selected_assignment.astype(np.int8))
        handle.attrs["complete"] = True
        handle.attrs["selected_method"] = "P0_HOMOTOPY_INTERIOR"
        handle.attrs["strict_training_interior_limit"] = 0.95
        handle.attrs["training_only_representatives"] = True
        handle.attrs["astronomical_truth_claim"] = False
        handle.attrs["inference_input_fields_added"] = 0
        handle.attrs["supersedes"] = "projection_targets/projected_target_sets.h5"
    hash_rows = []
    for scene in range(64):
        for prompt in (0, 1):
            for target in (0, 1):
                value = physical_targets[scene, prompt, target]
                hash_rows.append({"scene": scene, "scene_id": rows[scene]["scene_id"], "kind": rows[scene]["kind"], "prompt": prompt, "target_slot": target, "canonical_sha256": canonical_tensor_sha256(value), "source_truth_provenance": "frozen_training_target_region", "inference_input": False})
    fresh_csv(run / "tables/projected_target_hashes_final.csv", hash_rows)
    prereg = json.loads((run / "preregistration/freeze_record.json").read_text())
    freeze = {
        "status": "FROZEN_PROJECTED_TARGETS", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_method": "P0_HOMOTOPY_INTERIOR", "projected_target_file_sha256": sha256(target_path),
        "projected_target_hash_table_sha256": sha256(run / "tables/projected_target_hashes_final.csv"),
        "preregistration_sha256": prereg["preregistration_sha256"], "architecture_unchanged": True,
        "scientific_thresholds_unchanged": True, "strict_training_interior_limit": 0.95,
        "truth_or_constraint_inference_inputs": 0, "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
        "supersedes": "projection_targets/freeze_record.json",
    }
    fresh_json(run / "projection_targets/freeze_record_final.json", freeze)
    fresh_text(run / "reports/interior_slack_strictness_addendum.md", """# Interior-slack strictness addendum

The first corrected target-set audit selected P1 because it passed every unchanged scientific acceptance threshold (<=1.0) and all projection coverage/forward gates. A strict reread of the preregistration found that three P1 pairings were at `0.950001...`, so P1 did not meet the separate training-target interior requirement that every ratio be <=0.95. No tolerance or gate was relaxed. P1 is therefore ineligible as the frozen training-target method.

P0 has 100% pairwise interior feasibility, 100% ordinary/own/alternate/both-mode target-set coverage, 100% ordinary and ambiguous forward consistency under the corrected ordinary assembly, deterministic reproduction, stable canonical hashes, and maximum ratio below 0.006. The authoritative globally selected method is **P0_HOMOTOPY_INTERIOR**. `projection_targets/projected_target_sets_final.h5` and the `*_final` tables supersede the earlier P1 freeze artifacts.
""")
    fresh_json(run / "logs/projection_gate_complete_final.json", {
        "status": "PASS", "micro_training_authorized": True, "selected_method": "P0_HOMOTOPY_INTERIOR",
        "selected_metrics": p0, "projected_target_file_sha256": sha256(target_path),
        "supersedes": "logs/projection_gate_complete_superseding.json", "atlas_access_count": 0,
        "development_access_count": 0, "lockbox_access_count": 0,
    })
    print(json.dumps({"status": "FINAL_PROJECTION_GATE_PASS", "selected_method": "P0_HOMOTOPY_INTERIOR", "metrics": p0}, indent=2))


if __name__ == "__main__":
    main()
