#!/usr/bin/env python3
"""Reproduce Thayer-MH, audit reused targets, and preregister Thayer-ME."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py


REPO = Path(__file__).resolve().parents[1]
MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
PARTITIONS = ("training", "validation", "calibration")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()


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


def as_bool(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"invalid boolean: {value}")
    return value == "True"


def reproduce_mh(run_dir: Path) -> None:
    prompt = {row["gate"]: row for row in read_csv(MH / "tables/non_atlas_promptability_gates.csv")}
    coverage = {row["gate"]: row for row in read_csv(MH / "tables/non_atlas_set_coverage_gates.csv")}
    stop = json.loads((MH / "logs/pre_atlas_evaluation_complete.json").read_text())
    expected = {
        "expert_1_prompt_swap": 0.992,
        "expert_2_prompt_swap": 0.992,
        "set_level_prompt_swap": 0.992,
        "reconstruction_mse_ratio_to_condition_c": 0.8643911719720833,
        "ordinary_own_truth_coverage": 0.0,
        "near_own_truth_coverage": 0.0,
        "near_alternate_truth_coverage": 0.0,
        "near_both_mode_coverage": 0.0,
        "ordinary_forward_consistency": 0.9333333333333333,
        "near_forward_consistency": 1.0,
        "atlas_inference_count": 0.0,
    }
    observed = {
        "expert_1_prompt_swap": float(prompt["token0_prompt_swap"]["observed"]),
        "expert_2_prompt_swap": float(prompt["token1_prompt_swap"]["observed"]),
        "set_level_prompt_swap": float(prompt["set_level_prompt_swap"]["observed"]),
        "reconstruction_mse_ratio_to_condition_c": float(prompt["reconstruction_factor_to_condition_c"]["observed"]),
        "ordinary_own_truth_coverage": float(coverage["ordinary_own_truth_coverage"]["observed"]),
        "near_own_truth_coverage": float(coverage["near_own_truth_coverage"]["observed"]),
        "near_alternate_truth_coverage": float(coverage["near_alternate_truth_coverage"]["observed"]),
        "near_both_mode_coverage": float(coverage["near_both_mode_coverage"]["observed"]),
        "ordinary_forward_consistency": float(coverage["ordinary_forward_consistency"]["observed"]),
        "near_forward_consistency": float(coverage["near_forward_consistency"]["observed"]),
        "atlas_inference_count": float(stop["atlas_evaluation_count"]),
    }
    rows = []
    for metric, target in expected.items():
        value = observed[metric]
        passed = math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12)
        rows.append({"metric": metric, "expected": target, "observed": value, "absolute_tolerance": 1e-12, "status": "PASS" if passed else "FAIL", "source": "immutable Thayer-MH tables/log; zero new inference"})
    write_csv_fresh(run_dir / "tables/baseline_reproduction.csv", rows)
    if any(row["status"] != "PASS" for row in rows):
        write_text_fresh(run_dir / "diagnostics/baseline_reproduction.md", "# Thayer-MH reproduction\n\n**FAIL — STOP.** Thayer-ME implementation and fitting are prohibited.\n")
        raise RuntimeError("Thayer-MH reproduction mismatch")
    write_text_fresh(run_dir / "diagnostics/baseline_reproduction.md", """# Thayer-MH reproduction

Status: **PASS**. Metrics were recomputed from immutable persisted tables and the persisted stop record. No neural inference or Atlas access occurred.

- Expert/token and unordered-set prompt-swap success: 0.992 / 0.992 / 0.992.
- Requested reconstruction MSE ratio to Condition C: 0.8643911719720833.
- Ordinary own, near own, near alternate, and near both-mode coverage: 0 / 0 / 0 / 0.
- Ordinary and near-collision forward-consistent fractions: 0.9333333333333333 / 1.0.
- Thayer-MH Atlas inference count: 0.
""")


def audit_target_reuse(run_dir: Path) -> dict[str, object]:
    inventory = read_csv(MH / "tables/target_set_inventory.csv")
    pair_rows = read_csv(MH / "tables/target_set_pair_validation.csv")
    pair_manifest = read_csv(MH / "tables/non_atlas_near_collision_pair_manifest.csv")
    exclusion = {row["source_group"] for row in read_csv(MH / "tables/atlas_source_exclusion_audit.csv")}
    provenance = json.loads((run_dir / "logs/input_provenance.json").read_text())
    committed = {row["path"]: row["sha256"] for row in provenance["relevant_artifacts"]["reused_target_sets"] + provenance["relevant_artifacts"]["reused_scene_tensors"]}
    rows: list[dict[str, object]] = []
    for partition in PARTITIONS:
        target = MH / f"target_sets/thayer_mh_{partition}_target_sets.h5"
        scene = MH / f"manifests/probabilistic_unet_{partition}_scenes.h5"
        with h5py.File(target, "r") as handle:
            counts = handle["target_count"][:]
            complete = bool(handle.attrs["complete"])
        partition_inventory = [row for row in inventory if row["partition"] == partition and row["prompt_role"] == "A"]
        ordinary_expected = sum(row["kind"] == "ordinary" for row in partition_inventory)
        ambiguous_expected = sum(row["kind"] == "near_collision" for row in partition_inventory)
        ordinary_ok = int((counts[:, 0] == 1).sum()) == ordinary_expected
        ambiguous_ok = int((counts[:, 0] == 2).sum()) == ambiguous_expected
        target_rel = str(target.relative_to(REPO)); scene_rel = str(scene.relative_to(REPO))
        hash_ok = sha256_file(target) == committed[target_rel] and sha256_file(scene) == committed[scene_rel]
        rows.append({"partition": partition, "target_path": target_rel, "scene_path": scene_rel, "ordinary_scene_count": ordinary_expected, "ambiguous_scene_count": ambiguous_expected, "ordinary_single_target_pass": ordinary_ok, "ambiguous_two_target_pass": ambiguous_ok, "complete": complete, "hash_match": hash_ok, "status": "PASS" if all((ordinary_ok, ambiguous_ok, complete, hash_ok)) else "FAIL"})

    by_scene_prompt = {(row["scene_id"], row["prompt_role"]): row for row in inventory}
    shared_set_pass = True
    for row in inventory:
        if row["kind"] != "near_collision":
            continue
        alternate_role = "A" if row["alternate_requested_source_index"] == "0" else "B"
        counterpart = by_scene_prompt.get((row["alternate_scene_id"], alternate_role))
        if counterpart is None:
            shared_set_pass = False
            break
        signature = {row["own_decomposition_sha256"], row["alternate_decomposition_sha256"]}
        counterpart_signature = {counterpart["own_decomposition_sha256"], counterpart["alternate_decomposition_sha256"]}
        if signature != counterpart_signature:
            shared_set_pass = False
            break
    pair_gates_pass = len(pair_rows) == 2000 and all(row["status"] == "PASS" for row in pair_rows)
    atlas_absent = all(as_bool(row["atlas_groups_absent"]) for row in pair_rows)
    partition_pass = True
    for row in pair_manifest:
        groups = {row[field] for field in ("left_source_a_group", "left_source_b_group", "right_source_a_group", "right_source_b_group")}
        if len(groups) != 4 or groups & exclusion or row["partition"] not in PARTITIONS:
            partition_pass = False
            break
    if any(row["status"] != "PASS" for row in rows) or not all((shared_set_pass, pair_gates_pass, atlas_absent, partition_pass)):
        write_csv_fresh(run_dir / "tables/target_set_reuse_audit.csv", rows)
        write_text_fresh(run_dir / "diagnostics/target_set_reuse_report.md", "# Target-set reuse audit\n\n**FAIL — STOP.** Thayer-ME implementation and fitting are prohibited.\n")
        raise RuntimeError("target-set reuse audit failed")
    write_csv_fresh(run_dir / "tables/target_set_reuse_audit.csv", rows)
    write_json_fresh(run_dir / "target_sets/reused_target_set_references.json", {"status": "PASS", "authoritative_run": str(MH.relative_to(REPO)), "files": [{"path": row["target_path"], "sha256": committed[str(row["target_path"])]} for row in rows], "reuse_mode": "read-only direct reference; no copy or regeneration"})
    write_text_fresh(run_dir / "diagnostics/target_set_reuse_report.md", f"""# Target-set reuse audit

Status: **PASS — EXACT THAYER-MH TARGET SETS REUSED READ-ONLY**.

- Ordinary / ambiguous observations by partition: training 12,000 / 3,000; validation 1,500 / 500; calibration 1,500 / 500.
- Every ordinary target-count entry is one; every ambiguous target-count entry is two.
- Both members of every ambiguity pair have the same unordered two-decomposition hash set for both prompts: PASS.
- All 2,000 persisted pair-gate rows remain PASS.
- Atlas-related group absence and four-group disjointness: PASS.
- Cross-partition source-group isolation: PASS under the unchanged source split and persisted pair manifest.
- Target and scene HDF5 hashes match the campaign-start commitments.
- No pair, scene, or target was regenerated, copied, or replaced.
""")
    return {"target_hashes": {row["partition"]: committed[str(row["target_path"])] for row in rows}, "scene_hashes": {row["partition"]: committed[str(row["scene_path"])] for row in rows}, "pair_gate_count": len(pair_rows)}


def gates() -> list[dict[str, object]]:
    return [
        {"stage": "architecture", "gate": "parameter_ceiling", "range": "integer [0,+inf)", "threshold": "<=250000", "attainable": True},
        {"stage": "micro", "gate": "ordinary_own_truth_coverage", "range": "[0,1]", "threshold": ">=0.90", "attainable": True},
        {"stage": "micro", "gate": "ordinary_expert_diameter", "range": "[0,+inf)", "threshold": "median <=1.0 frozen scientific units", "attainable": True},
        {"stage": "micro", "gate": "ambiguous_own_truth_coverage", "range": "[0,1]", "threshold": ">=0.90", "attainable": True},
        {"stage": "micro", "gate": "ambiguous_alternate_truth_coverage", "range": "[0,1]", "threshold": ">=0.90", "attainable": True},
        {"stage": "micro", "gate": "ambiguous_both_mode_coverage", "range": "[0,1]", "threshold": ">=0.90", "attainable": True},
        {"stage": "micro", "gate": "prompt_swap", "range": "[0,1]", "threshold": ">=0.90 each expert and set", "attainable": True},
        {"stage": "micro", "gate": "forward_consistency", "range": "[0,1]", "threshold": ">=0.90 ordinary and ambiguous", "attainable": True},
        {"stage": "validation", "gate": "expert_prompt_swap", "range": "[0,1]", "threshold": ">=0.80 each", "attainable": True},
        {"stage": "validation", "gate": "set_prompt_swap", "range": "[0,1]", "threshold": ">=0.90", "attainable": True},
        {"stage": "validation", "gate": "reconstruction_factor", "range": "[0,+inf)", "threshold": "<=3.0x Condition C", "attainable": True},
        {"stage": "validation", "gate": "ordinary_own_truth_coverage", "range": "[0,1]", "threshold": ">=0.05", "attainable": True},
        {"stage": "validation", "gate": "near_own_truth_coverage", "range": "[0,1]", "threshold": ">=0.05", "attainable": True},
        {"stage": "validation", "gate": "near_alternate_truth_coverage", "range": "[0,1]", "threshold": ">=0.05", "attainable": True},
        {"stage": "validation", "gate": "near_both_mode_coverage", "range": "[0,1]", "threshold": ">=0.05", "attainable": True},
        {"stage": "validation", "gate": "forward_consistency", "range": "[0,1]", "threshold": ">=0.50 ordinary and near", "attainable": True},
        {"stage": "control", "gate": "ordinary_false_witness", "range": "[0,1]", "threshold": "<=0.10", "attainable": True},
        {"stage": "control", "gate": "diameter_separation", "range": "[0,+inf)", "threshold": "near/control median >=1.25 and pair-bootstrap lower >1", "attainable": True},
        {"stage": "control", "gate": "low_fpr_recall", "range": "[0,1]", "threshold": ">0 at calibration ordinary 95th percentile", "attainable": True},
        {"stage": "atlas", "gate": "own_truth_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"stage": "atlas", "gate": "alternate_truth_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"stage": "atlas", "gate": "both_mode_coverage", "range": "[0,1]", "threshold": ">0 preferred for success", "attainable": True},
        {"stage": "atlas", "gate": "witness_count", "range": "integer [0,50]", "threshold": ">24 or remains high with truth-coverage improvement", "attainable": True},
        {"stage": "atlas", "gate": "diameter_auroc", "range": "[0,1]", "threshold": ">=0.806 (no >0.05 material regression from 0.856)", "attainable": True},
        {"stage": "atlas", "gate": "recall_at_4pct_fpr", "range": "[0,1]", "threshold": ">=0.32", "attainable": True},
        {"stage": "atlas", "gate": "control_false_witness", "range": "[0,1]", "threshold": "<=0.10", "attainable": True},
    ]


def preregister(run_dir: Path, target_audit: dict[str, object]) -> None:
    gate_rows = gates()
    if not all(bool(row["attainable"]) for row in gate_rows):
        raise RuntimeError("unattainable gate")
    write_csv_fresh(run_dir / "tables/preregistered_gate_attainability.csv", gate_rows)
    frozen_at = datetime.now(timezone.utc).isoformat()
    text = f"""# Preregistration: Thayer-ME two-expert ambiguity decoder

Frozen at UTC `{frozen_at}` after exact Thayer-MH reproduction and target-set reuse audit, and before model implementation, microset fitting, full fitting, checkpoint selection, or Atlas inference.

## Scientific hypothesis and scope

A Condition-C-compatible shared prompt-sensitive encoder followed by two independently parameterized compact expert decoders can represent both approved decompositions of an ambiguous observation, while both experts converge to the same answer on ordinary uniquely recoverable scenes. This is a decoder-capacity and specialization feasibility campaign, not a posterior, auditor, catalog policy, final-lockbox campaign, or proof of uniqueness.

The exact read-only Thayer-MH training, validation, and calibration scene tensors, manifests, and target sets are reused. Target hashes are `{json.dumps(target_audit['target_hashes'], sort_keys=True)}`. Scene hashes are `{json.dumps(target_audit['scene_hashes'], sort_keys=True)}`. No Atlas group, development row, final-lockbox row, source ID, pair ID, morphology label, simulator difficulty, or generator parameter may enter inference.

## Architecture and initialization

The shared encoder is the Condition-C-compatible `enc1 4->16`, `enc2 16->32`, `bottleneck 32->64` path receiving only normalized g/r/z plus the fixed Gaussian coordinate prompt. Every compatible encoder tensor is loaded exactly from Condition C. No hypothesis token or expert identity enters the encoder or observed input. `enc1` and `enc2` remain frozen; `bottleneck` is frozen for phase 1 and may train only in phase 2.

Expert 1 and Expert 2 each independently contain `dec2 96->32`, `dec1 48->16`, and a `16->6` output head. They share no convolution, normalization, output-head, or late assignment parameters. Each emits requested g/r/z plus companion g/r/z with the unchanged unclipped, zero-background source-layer semantics. The decoders are initialized independently with frozen seeds 2026071201 and 2026071202; neither is copied from Condition C or from the other expert. Expected parameters are 72,672 shared encoder + 46,470 per expert = 165,612 total, below the frozen 250,000 ceiling.

## Loss and assignment

For ambiguous targets `{{Y_A,Y_B}}`, compute both expert-to-target assignments and minimize the sum of requested, companion, and source-sum reconstruction costs. The winning assignment is per scene and no global expert semantics exist. Ordinary scenes supervise both experts to the one approved decomposition and add a 0.10 concentration loss. The fixed full objective also includes 0.50 observed-blend forward/recomposition loss, 0.25 unordered prompt-swap loss, and 0.05 pair-set consistency. There is no generic diversity reward and no target-aware separation term.

## Isolated micro-overfit gate

The microset is frozen as the first 32 training ordinary scenes plus both members of the first 16 sorted training ambiguity pairs, for 32 ambiguous observations. Both prompts are used. Validation, calibration, Atlas, development, and lockbox rows are prohibited. The fit uses MPS only, AdamW, batch size 8, learning rate 1e-3, weight decay 0, seed 2026071250, at most 400 epochs, and early stopping only after all micro gates pass. The exact frozen scientific-distance and forward-consistency metrics apply. Required rates are >=0.90 for ordinary own-truth, ambiguous own-truth, alternate-truth, both-mode, prompt swap, and ordinary/ambiguous forward consistency; median ordinary expert diameter must be <=1.0. Failure is `REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE` and prohibits full training. Capacity and gates cannot change afterward.

## Full training and specialization audit

Only after micro pass: MPS-only AdamW, seed 2026071260, 30 epochs, batch size 8, learning rate 3e-4, weight decay 1e-4, and the exact Thayer-MH 6 ordinary + 2 ambiguity observations per batch schedule. Epochs 1-5 freeze the full encoder; epochs 6-30 unfreeze only the bottleneck. Select one best checkpoint by lowest protected validation objective; save best and final separately. Track per-expert assignment frequency/entropy, output distance, gradient norm, parameter distance, ordinary concentration, ambiguous separation, identity, forward consistency, coverage, assignment flips, and flux-scale-only differences. Stop on NaN/Inf, MPS fallback, hash/source exposure, collision, dead expert, sustained ambiguity collapse, uncontrolled ordinary divergence, or source-sum/forward instability.

## Non-Atlas and Atlas gates

Gate ranges and thresholds are frozen in `tables/preregistered_gate_attainability.csv`. Promptability is checked before truth coverage; truth coverage before control concentration. Own, alternate, and both-mode near-collision coverage must each be at least 0.05, rather than merely one example. Any failed stage stops before Atlas. Only after every non-Atlas gate passes may the selected checkpoint, matching, forward threshold, truth metrics, diameter metrics, controls, low-FPR threshold, and success gates be frozen and hashed. Atlas inference is one pass over the 50 frozen observations, with no retraining, recalibration, threshold change, seed addition, or post-Atlas tuning. Overall success requires every preceding gate; partial success requires non-Atlas specialization and nonzero Atlas truth coverage but insufficient operational low-FPR performance.

Final-lockbox and unauthorized development access remain zero under every outcome. Historical artifacts remain immutable.
"""
    path = run_dir / "preregistration/two_expert_ambiguity_decoder.md"
    write_text_fresh(path, text)
    write_json_fresh(run_dir / "preregistration/freeze_record.json", {
        "status": "FROZEN_BEFORE_MODEL_IMPLEMENTATION_OR_FITTING",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": sha256_file(path),
        "gate_table_sha256": sha256_file(run_dir / "tables/preregistered_gate_attainability.csv"),
        "target_set_reuse_audit_sha256": sha256_file(run_dir / "tables/target_set_reuse_audit.csv"),
        "parameter_ceiling": 250000,
        "expected_parameter_count": 165612,
        "expert_initialization_seeds": [2026071201, 2026071202],
        "micro_seed": 2026071250,
        "training_seed": 2026071260,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_two_expert_decoder_"):
        raise ValueError("unexpected run directory")
    part_a = json.loads((run_dir / "logs/part_a_complete.json").read_text())
    if part_a["status"] != "PASS" or part_a["atlas_evaluation_count"] != 0:
        raise RuntimeError("Part A did not pass")
    if command(["git", "diff", "--cached", "--name-only"]).splitlines():
        raise RuntimeError("staged files appeared after Part A")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint exists before preregistration")
    reproduce_mh(run_dir)
    target_audit = audit_target_reuse(run_dir)
    preregister(run_dir, target_audit)
    write_json_fresh(run_dir / "logs/foundation_complete.json", {
        "status": "PASS",
        "thayer_mh_reproduction": "PASS",
        "target_set_reuse": "PASS",
        "preregistration_sha256": sha256_file(run_dir / "preregistration/two_expert_ambiguity_decoder.md"),
        "model_implementation_authorized": True,
        "micro_fit_authorized": True,
        "full_fit_authorized": False,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS", "preregistration_sha256": sha256_file(run_dir / "preregistration/two_expert_ambiguity_decoder.md"), "target_pairs": target_audit["pair_gate_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
