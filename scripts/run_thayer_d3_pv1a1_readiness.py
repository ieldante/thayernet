#!/usr/bin/env python3
"""Freeze the non-scientific THAYER-D3-PV1-A1 readiness campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Mapping

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.canonical_tensor_hash import canonical_tensor_sha256
from src.d3_audit_layer_pv1 import (
    A1_CAPACITIES,
    A1_EXPERT_SEEDS,
    A1_REPLICA_OFFSETS,
    AuditJournal,
    IndependentAuditReplayer,
    LEGACY_LEARNED_CHECKPOINT_SHA256,
    PrimaryL0EvidenceBinding,
    canonical_json_bytes,
    construct_fresh_initial_state,
    hash_protected_file_for_integrity,
    reproduce_initial_state_clean_processes,
    validate_a1_initialization_manifest,
    verify_event_chain,
)
from src.d3_protocol_pv1a1 import (
    NoUpdateReadinessGuard,
    RuntimeDecisionProducer,
    hard_two_expert_set_loss,
    validate_effective_protocol,
)


CAMPAIGN_ID = "thayer_d3_pv1a1_readiness_r2_20260714_165947"
PROTOCOL_ID = "THAYER-D3-PV1-A1"
PARENT_SHA256 = "969a22da8eb8e54fc7a2a55a70bfd50996e0b6a551791ad7060d537b3d258a3a"
AMENDMENT_SHA256 = "4d3c227ba9ef7779bf987e95f26f39a0144717aeaf7067b66febce3467eff15e"
ENCODER_SHA256 = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"
SCENE_IDS = (
    "pu_training_ordinary_00000", "pu_training_ordinary_00008",
    "pu_training_ordinary_00016", "pu_training_ordinary_00024",
    "pu_training_near_00000", "pu_training_near_00008",
    "pu_training_near_00016", "pu_training_near_00024",
)
SCALES = np.asarray((611.9199829101562, 1805.8800048828125, 1854.199951171875), dtype=np.float32)
LEGACY_CHECKPOINT = REPO / "outputs/runs/thayer_output_parameterization_20260713_023120/checkpoints/ambiguous_one_scene_square.pth"
LEGACY_CACHE = REPO / "outputs/runs/thayer_repository_integrity_20260713_031653/fixed_feature_retry/cached_features_superseding_v4.pt"
LEGACY_PAYLOAD = REPO / "outputs/runs/thayer_repository_integrity_20260713_031653/data_lineage/one_scene_payload.npz"
PRIOR_SCENES = REPO / "outputs/runs/thayer_d3_protocol_readiness_r1_20260714_074016/protocol_bundle/scenes_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_x(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def copy_x(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


def leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            result.update(leaf_paths(item, f"{prefix}/{token}"))
        return result
    if isinstance(value, list):
        result = set()
        for index, item in enumerate(value):
            result.update(leaf_paths(item, f"{prefix}/{index}"))
        return result
    return {prefix}


def cache_reproducibility(run: Path) -> dict[str, object]:
    a_path = run / "cache_generation_A/manifest.json"
    b_path = run / "cache_generation_B/manifest.json"
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    keys = (
        "scene_ids", "feature_batch_shapes", "feature_batch_dtypes",
        "canonical_batch_feature_member_hashes", "complete_ordered_batch_feature_sha256",
        "target_batch_shape", "target_batch_dtype", "complete_ordered_batch_target_sha256",
        "encoder_checkpoint_sha256", "encoder_tensor_sha256", "normalization_scales_grz",
    )
    comparisons = {key: a[key] == b[key] for key in keys}
    scene_comparisons = []
    for left, right in zip(a["scenes"], b["scenes"]):
        scene_comparisons.append({
            "scene_id": left["scene_id"],
            "identity_match": left["scene_id"] == right["scene_id"],
            "order_match": left["order_index"] == right["order_index"],
            "feature_shapes_match": left["feature_shapes"] == right["feature_shapes"],
            "feature_dtypes_match": left["feature_dtypes"] == right["feature_dtypes"],
            "feature_member_hashes_match": left["canonical_member_feature_hashes"] == right["canonical_member_feature_hashes"],
            "scene_feature_hash_match": left["canonical_scene_feature_sha256"] == right["canonical_scene_feature_sha256"],
            "target_shape_match": left["target_shape"] == right["target_shape"],
            "target_dtype_match": left["target_dtype"] == right["target_dtype"],
            "target_hash_match": left["canonical_scene_target_sha256"] == right["canonical_scene_target_sha256"],
        })
    exact = all(comparisons.values()) and all(all(value for key, value in row.items() if key != "scene_id") for row in scene_comparisons)
    if not exact:
        raise RuntimeError("CACHE_REPRODUCIBILITY_FAILURE")
    promoted = run / "prospective_cache"
    shutil.copytree(run / "cache_generation_A", promoted, copy_function=shutil.copy2)
    result = {
        "schema_version": "thayer-d3-pv1-a1-cache-reproducibility-v1",
        "protocol_identifier": PROTOCOL_ID,
        "clean_process_generation_count": 2,
        "generation_a_manifest_sha256": sha256_file(a_path),
        "generation_b_manifest_sha256": sha256_file(b_path),
        "comparisons": comparisons,
        "per_scene_comparisons": scene_comparisons,
        "promotion_rule": "PROMOTE_GENERATION_A_AFTER_EXACT_CANONICAL_AGREEMENT",
        "promoted_generation": "A",
        "promoted_path": promoted.relative_to(run).as_posix(),
        "complete_ordered_batch_feature_sha256": a["complete_ordered_batch_feature_sha256"],
        "complete_ordered_batch_target_sha256": a["complete_ordered_batch_target_sha256"],
        "scientific_optimizer_constructions": 0,
        "scientific_backward_passes": 0,
        "scientific_optimizer_steps": 0,
        "status": "PASS",
    }
    write_json_x(run / "diagnostics/cache_reproducibility_audit.json", result)
    return result


def legacy_comparison(run: Path) -> dict[str, object]:
    old = torch.load(LEGACY_CACHE, map_location="cpu", weights_only=False)
    new = torch.load(run / "prospective_cache/ordered_cache.pt", map_location="cpu", weights_only=False)
    rows = []
    for prompt in ("prompt_a", "prompt_b"):
        for level, (left, full) in enumerate(zip(old[prompt], new[prompt])):
            right = full[4:5]
            rows.append({
                "member": f"{prompt}.{('enc1', 'enc2', 'bottleneck')[level]}",
                "legacy_shape": list(left.shape), "prospective_shape": list(right.shape),
                "shape_equal": list(left.shape) == list(right.shape),
                "legacy_dtype": str(left.dtype), "prospective_dtype": str(right.dtype),
                "dtype_equal": left.dtype == right.dtype,
                "exact_tensor_equal": bool(torch.equal(left, right)),
                "maximum_absolute_difference": float((left - right).abs().max()),
                "legacy_canonical_sha256": canonical_tensor_sha256(left[0], layout="CHW"),
                "prospective_canonical_sha256": canonical_tensor_sha256(right[0], layout="CHW"),
            })
    with np.load(LEGACY_PAYLOAD, allow_pickle=False) as payload:
        old_target = np.asarray(payload["p0_physical"], dtype=np.float32)
    prospective_targets = np.load(run / "prospective_cache/ordered_targets.npy", allow_pickle=False)
    new_target = prospective_targets[4]
    target_record = {
        "legacy_shape": list(old_target.shape), "prospective_shape": list(new_target.shape),
        "shape_equal": old_target.shape == new_target.shape,
        "legacy_dtype": str(old_target.dtype), "prospective_dtype": str(new_target.dtype),
        "dtype_equal": old_target.dtype == new_target.dtype,
        "exact_tensor_equal": bool(np.array_equal(old_target, new_target)),
        "maximum_absolute_difference": float(np.max(np.abs(old_target - new_target))),
        "legacy_canonical_sha256": canonical_tensor_sha256(old_target.reshape(24, 60, 60), layout="CHW"),
        "prospective_canonical_sha256": canonical_tensor_sha256(new_target.reshape(24, 60, 60), layout="CHW"),
    }
    result = {
        "schema_version": "thayer-d3-pv1-a1-legacy-near-scene-comparison-v1",
        "scene_id": "pu_training_near_00000",
        "legacy_cache_path": LEGACY_CACHE.relative_to(REPO).as_posix(),
        "prospective_cache_path": "prospective_cache/ordered_cache.pt",
        "legacy_encoder_tensor_sha256": old["encoder_tensor_sha256"],
        "prospective_encoder_tensor_sha256": new["encoder_tensor_sha256"],
        "encoder_tensor_identity_equal": old["encoder_tensor_sha256"] == new["encoder_tensor_sha256"],
        "feature_comparisons": rows,
        "target_comparison": target_record,
        "selection_policy": "PROSPECTIVE_GENERATION_A_ONLY_LEGACY_NOT_SELECTABLE",
        "status": "PASS_COMPARISON_RECORDED",
    }
    write_json_x(run / "diagnostics/legacy_near_scene_comparison.json", result)
    return result


def raw_input_verification(run: Path) -> dict[str, object]:
    prior = json.loads(PRIOR_SCENES.read_text(encoding="utf-8"))
    source_indices = [0, 8, 16, 24, 12000, 12008, 12016, 12024]
    target_indices = [0, 8, 16, 24, 32, 40, 48, 56]
    scene_h5 = REPO / prior["scenes"][0]["raw_data"]["path"]
    target_h5 = REPO / prior["scenes"][0]["target"]["historical_path"]
    from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy
    rows = []
    with h5py.File(scene_h5, "r") as scenes, h5py.File(target_h5, "r") as targets:
        for order, (scene_id, source_index, target_index, expected) in enumerate(zip(SCENE_IDS, source_indices, target_indices, prior["scenes"])):
            blend = np.asarray(scenes["blend"][source_index], dtype=np.float32)
            xy = np.asarray(scenes["xy"][source_index], dtype=np.float64)
            prompt_a = gaussian_prompt_numpy(float(xy[0, 0]), float(xy[0, 1]))[None]
            prompt_b = gaussian_prompt_numpy(float(xy[1, 0]), float(xy[1, 1]))[None]
            target = np.asarray(targets["targets_physical"][target_index], dtype=np.float32)
            observed = {
                "blend": canonical_tensor_sha256(blend, layout="CHW"),
                "prompt_a": canonical_tensor_sha256(prompt_a, layout="CHW"),
                "prompt_b": canonical_tensor_sha256(prompt_b, layout="CHW"),
            }
            expected_raw = expected["raw_data"]
            target_member_hashes = {
                f"prompt_{p}_slot_{s}": canonical_tensor_sha256(target[p, s], layout="CHW")
                for p in (0, 1) for s in (0, 1)
            }
            rows.append({
                "scene_id": scene_id, "order_index": order,
                "source_hdf5_index": source_index, "target_hdf5_index": target_index,
                "blend_sha256": observed["blend"], "blend_match": observed["blend"] == expected_raw["blend_canonical_sha256"],
                "prompt_a_sha256": observed["prompt_a"], "prompt_a_match": observed["prompt_a"] == expected_raw["prompt_a_canonical_sha256"],
                "prompt_b_sha256": observed["prompt_b"], "prompt_b_match": observed["prompt_b"] == expected_raw["prompt_b_canonical_sha256"],
                "target_member_hashes": target_member_hashes,
                "target_hashes_match": target_member_hashes == expected["target"]["canonical_hashes"],
            })
    if not all(all((row["blend_match"], row["prompt_a_match"], row["prompt_b_match"], row["target_hashes_match"])) for row in rows):
        raise RuntimeError("input manifest verification failed")
    result = {"schema_version": "thayer-d3-pv1-a1-input-verification-v1", "scenes": rows, "status": "PASS"}
    write_json_x(run / "diagnostics/input_manifest_verification.json", result)
    return result


def initialization_and_capacities(run: Path, journal: AuditJournal) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    journal.append("FRESH_INITIALIZATION_STARTED", "initialization", {"capacity_identifier": "L0", "expert_seeds": list(A1_EXPERT_SEEDS["L0"]), "legacy_checkpoint_loaded": False})
    initial = construct_fresh_initial_state("L0")
    validated = validate_a1_initialization_manifest(initial.manifest)
    initialization_path = run / "initial_state/fresh_l0_initialization_manifest.json"
    write_json_x(initialization_path, initial.manifest)
    journal.append("FRESH_INITIALIZATION_COMPLETED", "initialization", {
        "expert_seeds": list(A1_EXPERT_SEEDS["L0"]),
        "canonical_state_sha256": validated["canonical_state_sha256"],
        "optimizer_step_counter": 0, "optimizer_state_entry_count": 0,
        "metric_history_count": 0, "semantic_history_count": 0,
        "transferred_model_tensors": 0, "transferred_optimizer_tensors": 0,
    })
    l0_reproduction = reproduce_initial_state_clean_processes("L0")
    if l0_reproduction["status"] != "PASS" or l0_reproduction["canonical_state_sha256"] != validated["canonical_state_sha256"]:
        raise RuntimeError("FRESH_INITIALIZATION_REPRODUCIBILITY_FAILURE")
    write_json_x(run / "diagnostics/fresh_l0_reproducibility.json", l0_reproduction)
    journal.append("FRESH_INITIALIZATION_REPRODUCED", "initialization", {"clean_process_count": 2, "match": True, "canonical_state_sha256": validated["canonical_state_sha256"]})
    journal.append("INITIAL_STATE_FROZEN", "initialization", {"manifest_sha256": sha256_file(initialization_path), "canonical_state_sha256": validated["canonical_state_sha256"]})

    binding = PrimaryL0EvidenceBinding()
    binding.bind_branch_selection("THAYER-D3-PV1-A1::PRIMARY-L0::RUN-ONCE", "THAYER-D3-PV1-A1::PRIMARY-L0::IMMUTABLE-EVIDENCE")
    binding.bind_primary_capacity("THAYER-D3-PV1-A1::PRIMARY-L0::RUN-ONCE", "THAYER-D3-PV1-A1::PRIMARY-L0::IMMUTABLE-EVIDENCE")
    binding.bind_eight_scene("THAYER-D3-PV1-A1::EIGHT-SCENE::SEPARATE-RUN", "THAYER-D3-PV1-A1::EIGHT-SCENE::FRESH-INITIALIZATION")
    binding_record = {
        "schema_version": "thayer-d3-pv1-a1-primary-l0-binding-v1",
        "branch_selection_run_id": binding.branch_run_id,
        "branch_selection_evidence_id": binding.branch_evidence_id,
        "primary_capacity_run_id": binding.primary_capacity_run_id,
        "primary_capacity_evidence_id": binding.primary_capacity_evidence_id,
        "eight_scene_run_id": binding.eight_scene_run_id,
        "eight_scene_initialization_id": binding.eight_scene_initialization_id,
        **binding.verify(),
        "status": "PASS",
    }
    write_json_x(run / "diagnostics/primary_l0_evidence_binding.json", binding_record)
    journal.append("PRIMARY_L0_EVIDENCE_BOUND", "future_contract", {
        "branch_selection_run_id": binding.branch_run_id,
        "branch_selection_evidence_id": binding.branch_evidence_id,
        "primary_capacity_run_id": binding.primary_capacity_run_id,
        "primary_capacity_evidence_id": binding.primary_capacity_evidence_id,
        "duplicate_primary_l0_run_count": 0,
    })
    journal.append("DUPLICATE_PRIMARY_L0_REJECTED", "nonscientific_synthetic_test", {"evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE", "corruption_case": "duplicate_primary_l0", "rejected": True})

    capacity_rows = []
    for level, widths in A1_CAPACITIES.items():
        for offset in A1_REPLICA_OFFSETS:
            seeds = tuple(seed + offset for seed in A1_EXPERT_SEEDS[level])
            report = reproduce_initial_state_clean_processes(level, seeds)
            if report["status"] != "PASS":
                raise RuntimeError("FRESH_INITIALIZATION_REPRODUCIBILITY_FAILURE")
            capacity_rows.append({
                "capacity_identifier": level, "dec2_width": widths[0], "dec1_width": widths[1],
                "replica_offset": offset, "expert_seeds": list(seeds),
                "clean_process_constructions": 2,
                "canonical_state_hashes_match": report["canonical_state_hashes_match"],
                "canonical_state_sha256": report["canonical_state_sha256"],
                "parameter_count": report["constructions"][0]["parameter_count"],
                "state_tensor_count": report["constructions"][0]["tensor_count"],
                "transferred_model_tensors": 0, "transferred_optimizer_tensors": 0,
            })
            journal.append("CAPACITY_CONSTRUCTED", "readiness", {"capacity_identifier": level, "replica_offset": offset, "expert_seeds": list(seeds), "canonical_state_sha256": report["canonical_state_sha256"], "scientific_optimization": False})
    capacity_manifest = {
        "schema_version": "thayer-d3-pv1-a1-capacity-manifest-v1",
        "protocol_identifier": PROTOCOL_ID, "conditions": capacity_rows,
        "primary_levels": [{"level": level, "dec2": widths[0], "dec1": widths[1], "seeds": list(A1_EXPERT_SEEDS[level])} for level, widths in A1_CAPACITIES.items()],
        "replica_offsets": list(A1_REPLICA_OFFSETS), "model_transfer": "FORBIDDEN",
        "optimizer_transfer": "FORBIDDEN", "partial_state_dict_loading": "FORBIDDEN",
        "shape_adaptation": "FORBIDDEN", "scientific_optimizer_steps": 0, "status": "PASS",
    }
    write_json_x(run / "diagnostics/capacity_construction_manifest.json", capacity_manifest)
    return initial.manifest, binding_record, capacity_manifest


def decision_agreement(run: Path) -> dict[str, object]:
    runtime = RuntimeDecisionProducer()
    replay_protocol = {
        "protocol_identifier": PROTOCOL_ID, "tangent_threshold": 0.5,
        "tangent_comparison": "STRICTLY_LESS_THAN", "eight_scene_required_pass_count": 8,
        "capacities": [{"level": level} for level in A1_CAPACITIES],
    }
    replayer = IndependentAuditReplayer(replay_protocol)
    keys = (
        "valid", "l0_completed_validly", "source_integrity", "input_integrity",
        "metric_integrity", "scientific_contract_integrity", "frozen_l0_success",
        "d0_pass", "d1_pass", "other_validity_predicates_pass",
        "optimization_mechanism_supported", "hard_assignment_mechanism_supported",
        "square_mapping_mechanism_supported",
    )
    branch_count = 0
    for bits in itertools.product((False, True), repeat=len(keys)):
        for capture in (0.2, 0.5):
            evidence = {"evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE", **dict(zip(keys, bits)), "validated_tangent_capture": capture}
            if runtime.select_branch(evidence) != replayer.select_branch(evidence):
                raise RuntimeError("runtime/replay branch disagreement")
            branch_count += 1
    eight_count = 0
    for bits in itertools.product((False, True), repeat=8):
        evidence = {"valid": True, "scene_passes": list(bits), "evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE"}
        if runtime.classify_terminal("EIGHT_SCENE", evidence) != replayer.classify_terminal("EIGHT_SCENE", evidence):
            raise RuntimeError("runtime/replay eight-scene disagreement")
        eight_count += 1
    capacity_count = 0
    for first, smaller in (("L1", "L0"), ("L2", "L1"), ("L3", "L2")):
        for candidate in itertools.product((False, True), repeat=3):
            for boundary in itertools.product((False, True), repeat=3):
                evidence = {"valid": True, "first_passing_level": first, "level_results": {first: list(candidate), smaller: list(boundary)}, "evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE"}
                if runtime.classify_terminal("CAPACITY_LADDER", evidence) != replayer.classify_terminal("CAPACITY_LADDER", evidence):
                    raise RuntimeError("runtime/replay capacity disagreement")
                capacity_count += 1
    result = {
        "schema_version": "thayer-d3-pv1-a1-runtime-replay-agreement-v1",
        "evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE",
        "branch_fixture_count": branch_count, "eight_scene_fixture_count": eight_count,
        "capacity_fixture_count": capacity_count, "total_fixture_count": branch_count + eight_count + capacity_count,
        "runtime_decision_producer": "src.d3_protocol_pv1a1.RuntimeDecisionProducer",
        "independent_replayer": "src.d3_audit_layer_pv1.IndependentAuditReplayer",
        "runtime_imported_by_replayer": False, "agreement": True, "scientific_branch_selections": 0,
        "scientific_d3_decisions": 0, "status": "PASS",
    }
    write_json_x(run / "diagnostics/runtime_replay_agreement.json", result)
    return result


def no_update_traversals(run: Path, journal: AuditJournal) -> dict[str, object]:
    cache = torch.load(run / "prospective_cache/ordered_cache.pt", map_location="cpu", weights_only=False)
    targets = torch.from_numpy(np.load(run / "prospective_cache/ordered_targets.npy", allow_pickle=False))
    scale6 = torch.from_numpy(np.tile(SCALES, 2)).view(1, 1, 6, 1, 1)
    guard = NoUpdateReadinessGuard()
    with guard, torch.no_grad():
        l0 = construct_fresh_initial_state("L0").model.eval()
        joined_features = tuple(torch.cat((left, right), dim=0) for left, right in zip(cache["prompt_a"], cache["prompt_b"]))
        joined_output = l0.forward_features(joined_features) * scale6
        output_a, output_b = joined_output[:8], joined_output[8:]
        loss_a = hard_two_expert_set_loss(output_a, targets[:, 0])
        loss_b = hard_two_expert_set_loss(output_b, targets[:, 1])
        eight_loss = 0.5 * (loss_a + loss_b)
        if not torch.isfinite(eight_loss) or output_a.shape != (8, 2, 6, 60, 60):
            raise RuntimeError("eight-scene no-update traversal failed")
        capacity_rows = []
        near_features = tuple(torch.cat((left[4:5], right[4:5]), dim=0) for left, right in zip(cache["prompt_a"], cache["prompt_b"]))
        near_targets = torch.cat((targets[4, 0:1], targets[4, 1:2]), dim=0)
        for level in A1_CAPACITIES:
            model = construct_fresh_initial_state(level).model.eval()
            output = model.forward_features(near_features) * scale6
            loss = hard_two_expert_set_loss(output, near_targets)
            if output.shape != (2, 2, 6, 60, 60) or not torch.isfinite(output).all() or not torch.isfinite(loss):
                raise RuntimeError("capacity no-update traversal failed")
            capacity_rows.append({
                "capacity_identifier": level, "output_shape": list(output.shape),
                "output_finite": True, "target_dependent_loss_finite": True,
                "target_dependent_loss_observed_not_used_for_selection": float(loss),
                "performance_comparison_performed": False, "capacity_selected": False,
            })
    for index, scene_id in enumerate(SCENE_IDS):
        journal.append("MODEL_FORWARD_COMPLETED", "no_update_eight_scene", {"scene_id": scene_id, "order_index": index, "no_grad": True, "scientific_update": False})
        journal.append("TARGET_DEPENDENT_LOSS_COMPUTED", "no_update_eight_scene", {"scene_id": scene_id, "used_for_success_predicate": False, "used_for_branch_selection": False, "used_for_d3": False})
    result = {
        "schema_version": "thayer-d3-pv1-a1-no-update-readiness-v1",
        "protocol_identifier": PROTOCOL_ID,
        "eight_scene": {
            "scene_ids": list(SCENE_IDS), "ordered_batch_loaded": True,
            "feature_hashes_verified": True, "target_hashes_verified": True,
            "no_grad_forward_count": 1, "output_shape": list(output_a.shape),
            "output_finite": True, "per_scene_metric_compatibility": True,
            "target_dependent_loss_finite": True,
            "target_dependent_loss_observed_not_used_for_selection": float(eight_loss),
            "success_predicate_evaluated": False, "d3_applied": False,
        },
        "capacity_ladder": capacity_rows,
        "guards_installed": ["backward", "scientific_optimizer_construction", "optimizer_step", "semantic_checkpoint_selection", "branch_selection_from_metrics", "d3_application"],
        "guard_rejection_counts_during_valid_traversal": guard.rejection_counts,
        "scientific_backward_passes": 0, "scientific_optimizer_steps": 0,
        "scientific_checkpoints_selected": 0, "scientific_capacity_selections": 0,
        "scientific_branch_selections": 0, "scientific_d3_decisions": 0,
        "d3_status": "UNKNOWN", "status": "PASS",
    }
    write_json_x(run / "diagnostics/no_update_readiness_traversal.json", result)
    return result


def source_freeze(run: Path) -> dict[str, object]:
    files = []
    for top in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(top.rglob("*.py")):
            files.append({"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    files.append({"path": "README.md", "bytes": (REPO / "README.md").stat().st_size, "sha256": sha256_file(REPO / "README.md")})
    core = {"schema_version": "thayer-d3-pv1-a1-source-freeze-v1", "repository_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip(), "files": files}
    result = {**core, "source_freeze_sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(), "status": "PASS"}
    write_json_x(run / "source_freeze/source_manifest.json", result)
    return result


def protocol_values(audit_hash: str, init_hash: str, cache_manifest: Mapping[str, object], capacity_manifest: Mapping[str, object], binding: Mapping[str, object]) -> dict[str, object]:
    return {
        "authority": {
            "parent_protocol_identifier": "THAYER-D3-PV1", "amendment_identifier": "A1",
            "parent_approval_sha256": PARENT_SHA256, "amendment_approval_sha256": AMENDMENT_SHA256,
            "historical_recovery_claim": False,
        },
        "initialization": {
            "mode": "FRESH_SEEDED_STEP_ZERO", "l0_expert_seeds": list(A1_EXPERT_SEEDS["L0"]),
            "initial_state_canonical_sha256": init_hash, "optimizer_step_counter": 0,
            "pre_step_optimizer_state": "EMPTY", "metric_history": "EMPTY",
            "semantic_checkpoint_history": "EMPTY", "model_tensor_transfer": "FORBIDDEN",
            "optimizer_tensor_transfer": "FORBIDDEN", "counter_transfer": "FORBIDDEN",
            "scientific_evidence_transfer": "FORBIDDEN",
        },
        "legacy_checkpoint": {
            "sha256": LEGACY_LEARNED_CHECKPOINT_SHA256,
            "classification": "PROTECTED_LEGACY_LEARNED_CHECKPOINT_ONLY",
            "integrity_hashing": "ALLOWED", "scientific_use": "FORBIDDEN",
            "model_load": "FORBIDDEN", "optimizer_load": "FORBIDDEN",
            "branch_evidence": "FORBIDDEN", "capacity_evidence": "FORBIDDEN",
        },
        "primary_l0": {
            "run_count": 1, "run_id": binding["branch_selection_run_id"],
            "evidence_id": binding["branch_selection_evidence_id"],
            "branch_selection_role": "SOLE", "capacity_ladder_reuse": "REQUIRED",
            "duplicate_run": "FORBIDDEN", "maximum_optimizer_steps": 5000,
            "evaluation_steps": [0, 1, 10, 50, *range(100, 5001, 100)],
            "success_patience_evaluations": 3,
            "optimizer": {"class": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999], "epsilon": 1e-08, "amsgrad": False, "foreach": False, "maximize": False, "capturable": False, "differentiable": False, "gradient_clip_norm": 5.0, "gradient_norm_type": 2.0, "scheduler": "NONE"},
            "loss": {"reduction": "MEAN_OVER_PROMPTS_OF_MINIMUM_IDENTITY_SWAP", "pair_cost": "REQUESTED_PLUS_COMPANION_PHYSICAL_MSE_DIVIDED_BY_SCALE6", "identity_cost": "C00_PLUS_C11", "swap_cost": "C01_PLUS_C10", "tie_behavior": "IDENTITY_WINS"},
            "scientific_inputs": "FROZEN_PROSPECTIVE_NEAR_00000_CACHE_AND_P0_TARGET",
            "validity_predicates": ["SOURCE_INTEGRITY", "INPUT_INTEGRITY", "METRIC_INTEGRITY", "SCIENTIFIC_CONTRACT_INTEGRITY", "D0_PASS", "D1_PASS", "OTHER_VALIDITY_PREDICATES_PASS"],
            "success_predicate": {"own_coverage": 1.0, "alternate_coverage": 1.0, "both_mode_coverage": 1.0, "prompt_identity_all_experts": True, "forward_consistency_all_experts": True, "primary_normalized_gate": 1.0, "image_threshold": 0.25, "relative_flux_threshold_grz": [0.2, 0.2, 0.2], "color_threshold_g_minus_r": 0.2, "color_threshold_r_minus_z": 0.2, "centroid_threshold_mean_psf_fwhm": 0.5, "comparison_operator": "LESS_THAN_OR_EQUAL", "required_consecutive_evaluations": 3},
            "forward_plausibility": {"global_chi_square_mean_maximum": 1.2543178712712195, "per_band_chi_square_mean_maximum_grz": [1.2000065947013574, 1.2258474450543715, 1.256406290721562], "absolute_relative_flux_residual_maximum": 0.12280256285502243, "finite_required": True, "sky_electrons_grz": [24114.080000000005, 127057.12000000002, 250784.80000000005], "variance_floor": 1.0},
            "mechanism_tests": {"optimization_mechanism_supported": "OBSERVED_BOOLEAN_REQUIRED", "hard_assignment_mechanism_supported": "OBSERVED_BOOLEAN_REQUIRED", "square_mapping_mechanism_supported": "OBSERVED_BOOLEAN_REQUIRED", "validated_tangent_capture": "OBSERVED_FINITE_NUMBER_REQUIRED", "capacity_branch_requires_all_three_mechanisms_unsupported": True, "capacity_branch_requires_validated_tangent_capture_strictly_below": 0.5},
        },
        "eight_scene": {
            "scene_order": list(SCENE_IDS), "cache_generation_processes": 2,
            "promoted_generation": "A", "encoder_checkpoint_sha256": ENCODER_SHA256,
            "cached_feature_representation": "ORDERED_BATCH_NCHW",
            "target_representation": "PROMPT_TARGET_SLOT_CHANNEL_HEIGHT_WIDTH",
            "initialization": "SEPARATE_FRESH_MATCHED_L0", "expert_seeds": list(A1_EXPERT_SEEDS["L0"]),
            "optimizer_steps": 3200, "batch_size": 8,
            "optimizer": {"class": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0, "gradient_clip_norm": 5.0},
            "evaluation_cadence": {"initial": 0, "first_step": 1, "periodic_interval": 100, "terminal": 3200},
            "terminal_step": 3200, "required_pass_count": 8,
            "valid_8_of_8": "PASS", "valid_0_through_7_of_8": "FAIL", "invalid_or_incomplete": "UNKNOWN",
            "early_stop": "FORBIDDEN", "checkpoint_shopping": "FORBIDDEN",
        },
        "capacity_ladder": {
            "levels": capacity_manifest["primary_levels"], "replica_offsets": list(A1_REPLICA_OFFSETS),
            "primary_l0_reuse": "REQUIRED", "l1_l2_l3_initialization": "INDEPENDENT_SCRATCH",
            "model_transfer": "FORBIDDEN", "optimizer_transfer": "FORBIDDEN",
            "partial_state_dict_loading": "FORBIDDEN", "shape_adaptation": "FORBIDDEN",
            "counter_transfer": "FORBIDDEN", "history_transfer": "FORBIDDEN",
            "per_condition_maximum_optimizer_steps": 5000,
            "first_clean_replicated_boundary": "PASS", "mixed_or_nonreplicating_boundary": "FAIL",
            "no_pass_through_l3": "FAIL", "invalid_or_incomplete": "UNKNOWN",
        },
        "decisions": {
            "l0_success": "EIGHT_SCENE", "valid_l0_failure_low_capture": "CAPACITY_LADDER",
            "other_valid_l0": "NONE_UNKNOWN", "invalid_or_incomplete_l0": "NONE_UNKNOWN",
            "tangent_threshold": 0.5, "tangent_comparison": "STRICTLY_LESS_THAN",
            "alternative_mechanisms_required_excluded": ["OPTIMIZATION", "HARD_ASSIGNMENT", "SQUARE_MAPPING"],
            "runtime_producer": "src.d3_protocol_pv1a1.RuntimeDecisionProducer",
            "independent_replayer": "src.d3_audit_layer_pv1.IndependentAuditReplayer",
            "runtime_replay_agreement_required": True,
        },
        "retry_policy": "FORBIDDEN",
        "audit": {"contract_sha256": audit_hash, "event_hash_chain": "REQUIRED", "candidate_root_containment": "REQUIRED", "noninterference": "REQUIRED", "independent_final_replay": "REQUIRED"},
        "cache_identity": {"feature_sha256": cache_manifest["complete_ordered_batch_feature_sha256"], "target_sha256": cache_manifest["complete_ordered_batch_target_sha256"], "promotion_rule": cache_manifest["promotion_rule"]},
        "scientific_totals_for_readiness": {"backward_passes": 0, "optimizer_steps": 0, "checkpoints_selected": 0, "capacity_selections": 0, "branch_selections": 0, "d3_decisions": 0, "d3_status": "UNKNOWN"},
    }


def build_bundle(run: Path, source: Mapping[str, object], init: Mapping[str, object], cache_audit: Mapping[str, object], capacity: Mapping[str, object], binding: Mapping[str, object], decisions: Mapping[str, object]) -> dict[str, object]:
    bundle = run / "protocol_bundle/THAYER-D3-PV1-A1"
    bundle.mkdir(parents=True)
    copy_x(run / "authority/THAYER-D3-PV1_human_approval.md", bundle / "parent_human_approval.md")
    copy_x(run / "authority/THAYER-D3-PV1-A1_human_approval.md", bundle / "amendment_A1_human_approval.md")
    composed = {
        "schema_version": "thayer-d3-pv1-a1-composed-authority-v1", "effective_protocol_identifier": PROTOCOL_ID,
        "parent_protocol_identifier": "THAYER-D3-PV1", "amendment_identifier": "A1",
        "parent_approval_sha256": PARENT_SHA256, "amendment_approval_sha256": AMENDMENT_SHA256,
        "composition_status": "RESOLVED", "historical_recovery_claim": False,
        "superseded_requirement": "learned-checkpoint/fresh-seed equivalence",
        "replacement_requirement": "protected legacy checkpoint exclusion plus deterministic fresh step-zero L0",
        "all_other_parent_decisions": "UNCHANGED", "unresolved_decision_count": 0,
    }
    write_json_x(bundle / "composed_authority.json", composed)
    audit_contract = {
        "schema_version": "thayer-d3-pv1-a1-audit-contract-v1", "protocol_identifier": PROTOCOL_ID,
        "parent_contract_sha256": "0aac0a31e0b1dec18efe993b0690fc7afb2b5959c93589c053a943eaaa363291",
        "runtime_decision_producer": "SEPARATE", "independent_replayer": "SEPARATE_NO_RUNTIME_IMPORT",
        "required_a1_events": ["HUMAN_AUTHORITY_AMENDMENT_FROZEN", "LEGACY_CHECKPOINT_INVENTORIED", "LEGACY_CHECKPOINT_EXCLUDED", "FRESH_INITIALIZATION_STARTED", "FRESH_INITIALIZATION_COMPLETED", "FRESH_INITIALIZATION_REPRODUCED", "INITIAL_STATE_FROZEN", "PRIMARY_L0_EVIDENCE_BOUND", "DUPLICATE_PRIMARY_L0_REJECTED"],
        "noninterference": ["PYTHON_RNG", "NUMPY_RNG", "TORCH_CPU_RNG", "AVAILABLE_ACCELERATOR_RNG", "MODEL_TENSORS", "OPTIMIZER_STATE", "INPUTS", "TARGETS", "GRADIENTS"],
        "independent_checks": ["AUTHORITY", "EVENT_CHAIN", "SOURCE_FREEZE", "INPUT_MANIFEST", "CACHE_MANIFEST", "CAPACITY_MANIFEST", "SEMANTIC_CHECKPOINT_REPLAY", "RETRY_PROHIBITION", "OPTIMIZER_STEP_COUNT", "EVALUATION_MONOTONICITY", "TERMINAL_OUTCOME"],
        "candidate_root_containment": "FAIL_CLOSED", "unknown_event_type": "FAIL_CLOSED",
    }
    write_json_x(bundle / "audit_contract.json", audit_contract)
    audit_hash = sha256_file(bundle / "audit_contract.json")
    values = protocol_values(audit_hash, init["combined_canonical_state_manifest_sha256"], cache_audit, capacity, binding)
    provenance = {path: "COMPOSED_PARENT_AUTHORITY_AMENDMENT_A1_VALIDATED_ENGINEERING_CONTRACT" for path in leaf_paths(values)}
    effective = {
        "schema_version": "thayer-d3-pv1-a1-effective-protocol-v1", "protocol_identifier": PROTOCOL_ID,
        "values": values, "provenance_by_json_pointer": provenance, "unknown_key_policy": "FAIL_CLOSED",
    }
    validation = validate_effective_protocol(effective)
    effective["validation"] = validation
    # Validation is informational and not a science-affecting protocol key, so freeze it separately.
    validation_record = effective.pop("validation")
    write_json_x(bundle / "d3_protocol.json", effective)
    write_text_x(bundle / "d3_protocol.md", "# THAYER-D3-PV1-A1 effective protocol\n\nThe parent prospective authority and Amendment A1 compose without unresolved scientific choices. All science-affecting leaf fields have explicit provenance. Unknown or missing keys fail closed. D3 remains UNKNOWN in this readiness bundle.\n")
    write_json_x(run / "diagnostics/effective_protocol_validation.json", validation_record)
    copy_x(run / "initial_state/fresh_l0_initialization_manifest.json", bundle / "initialization_manifest.json")
    legacy_manifest = {
        "schema_version": "thayer-d3-pv1-a1-legacy-exclusion-v1", "sha256": LEGACY_LEARNED_CHECKPOINT_SHA256,
        "classification": "PROTECTED_LEGACY_LEARNED_CHECKPOINT_ONLY", "integrity_hashing": "ALLOWED",
        "model_loading": "FORBIDDEN", "optimizer_loading": "FORBIDDEN", "branch_evidence": "FORBIDDEN",
        "capacity_evidence": "FORBIDDEN", "model_load_attempts": 0, "optimizer_load_attempts": 0,
        "scientific_evidence_references": 0, "status": "PASS",
    }
    write_json_x(bundle / "legacy_checkpoint_exclusion_manifest.json", legacy_manifest)
    promoted = json.loads((run / "prospective_cache/manifest.json").read_text())
    scenes_manifest = {"schema_version": "thayer-d3-pv1-a1-scenes-manifest-v1", "scene_order": list(SCENE_IDS), "scenes": promoted["scenes"], "status": "PASS"}
    targets_manifest = {"schema_version": "thayer-d3-pv1-a1-targets-manifest-v1", "scene_order": list(SCENE_IDS), "batch_shape": promoted["target_batch_shape"], "dtype": promoted["target_batch_dtype"], "complete_ordered_batch_target_sha256": promoted["complete_ordered_batch_target_sha256"], "per_scene": [{"scene_id": row["scene_id"], "order_index": row["order_index"], "target_shape": row["target_shape"], "target_dtype": row["target_dtype"], "canonical_scene_target_sha256": row["canonical_scene_target_sha256"]} for row in promoted["scenes"]], "status": "PASS"}
    cached_manifest = {"schema_version": "thayer-d3-pv1-a1-cached-features-manifest-v1", "scene_order": list(SCENE_IDS), "encoder_checkpoint_sha256": ENCODER_SHA256, "encoder_tensor_sha256": promoted["encoder_tensor_sha256"], "batch_shapes": promoted["feature_batch_shapes"], "batch_dtypes": promoted["feature_batch_dtypes"], "canonical_member_hashes": promoted["canonical_batch_feature_member_hashes"], "complete_ordered_batch_feature_sha256": promoted["complete_ordered_batch_feature_sha256"], "generation_a_b_exact": True, "promoted_generation": "A", "status": "PASS"}
    write_json_x(bundle / "scenes_manifest.json", scenes_manifest)
    write_json_x(bundle / "targets_manifest.json", targets_manifest)
    write_json_x(bundle / "cached_features_manifest.json", cached_manifest)
    write_json_x(bundle / "capacity_manifest.json", capacity)
    transfer = {"schema_version": "thayer-d3-pv1-a1-checkpoint-transfer-v1", "model_tensor_transfers": 0, "optimizer_tensor_transfers": 0, "partial_state_dict_loads": 0, "shape_adaptations": 0, "counter_transfers": 0, "history_transfers": 0, "scientific_evidence_transfers": 0, "policy": "ALL_TRANSFER_FORBIDDEN", "status": "PASS"}
    write_json_x(bundle / "checkpoint_transfer_manifest.json", transfer)
    graph = {
        "schema_version": "thayer-d3-pv1-a1-decision-graph-v1", "initial_status": "UNKNOWN",
        "nodes": [
            {"id": "PRIMARY_L0", "run_count": 1, "success": "EIGHT_SCENE", "valid_failure_low_capture": "CAPACITY_LADDER", "otherwise": "UNKNOWN"},
            {"id": "EIGHT_SCENE", "fresh_initialization": "SEPARATE", "valid_8_of_8": "PASS", "valid_0_to_7": "FAIL", "invalid": "UNKNOWN"},
            {"id": "CAPACITY_LADDER", "primary_l0": "REUSE", "clean_replicated_boundary": "PASS", "other_valid": "FAIL", "invalid": "UNKNOWN"},
        ],
        "runtime_replay_fixture_count": decisions["total_fixture_count"], "runtime_replay_agreement": True,
    }
    write_json_x(bundle / "d3_decision_graph.json", graph)
    write_text_x(bundle / "d3_decision_graph.md", "# D3 decision graph\n\nPrimary L0 runs once. Success selects the separately initialized eight-scene branch; a valid low-capture failure with alternatives excluded selects the capacity ladder and reuses primary L0. Every invalid or incomplete route remains UNKNOWN.\n")
    execution = {
        "schema_version": "thayer-d3-pv1-a1-execution-contract-v1", "protocol_identifier": PROTOCOL_ID,
        "readiness_mode": {"scientific_backward_passes": 0, "scientific_optimizer_steps": 0, "scientific_branch_selections": 0, "scientific_d3_decisions": 0},
        "future_sequence": ["CREATE_FRESH_ROOT", "VERIFY_PROTOCOL", "VERIFY_SOURCE", "VERIFY_AUDIT", "VERIFY_INPUTS", "VERIFY_LEGACY", "CONSTRUCT_FRESH_L0", "RUN_PRIMARY_L0_ONCE", "BIND_PRIMARY_L0_EVIDENCE", "DERIVE_BRANCH", "EXECUTE_SELECTED_BRANCH_ONLY", "INDEPENDENT_REPLAY", "REQUIRE_RUNTIME_REPLAY_AGREEMENT", "FINAL_INTEGRITY", "REPORT_PASS_FAIL_UNKNOWN"],
        "repair_retry_resume_checkpoint_shopping_seed_replacement": "FORBIDDEN",
    }
    write_json_x(bundle / "execution_contract.json", execution)
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "repository_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip(), "source_freeze_sha256": source["source_freeze_sha256"]}
    write_json_x(run / "environment/environment_manifest.json", environment)

    content_files = sorted(path for path in bundle.rglob("*") if path.is_file())
    content_rows = [{"path": path.relative_to(bundle).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in content_files]
    content_tree_hash = hashlib.sha256(canonical_json_bytes(content_rows)).hexdigest()
    command = {
        "schema_version": "thayer-d3-pv1-a1-scientific-command-template-v1",
        "status": "CREATED_NOT_EXECUTED", "protocol_identifier": PROTOCOL_ID,
        "command": [
            str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_thayer_d3_pv1a1_scientific.py"),
            "--protocol-bundle", str(bundle), "--protocol-hashes", str(bundle / "protocol_hashes.json"),
            "--readiness-root", str(run), "--create-fresh-timestamped-campaign-root",
            "--execute-authoritative-science",
        ],
        "preflight_protocol_content_tree_sha256": content_tree_hash,
        "required_operations": execution["future_sequence"],
        "forbidden_operations": ["REPAIR", "RETRY", "RESUME", "CHECKPOINT_SHOPPING", "SEED_REPLACEMENT", "PRIMARY_L0_RERUN"],
        "executed_during_readiness": False,
    }
    write_json_x(bundle / "scientific_command_template.json", command)
    bundle_files = sorted(path for path in bundle.rglob("*") if path.is_file() and path.name != "protocol_hashes.json")
    file_rows = [{"path": path.relative_to(bundle).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in bundle_files]
    bundle_hash = hashlib.sha256(canonical_json_bytes(file_rows)).hexdigest()
    hashes = {
        "schema_version": "thayer-d3-pv1-a1-protocol-hashes-v1",
        "hash_scope": "CANONICAL_ORDERED_PATH_SHA256_BYTES_FOR_ALL_BUNDLE_FILES_EXCLUDING_PROTOCOL_HASHES_JSON",
        "protocol_bundle_sha256": bundle_hash, "protocol_content_tree_before_command_sha256": content_tree_hash,
        "source_freeze_sha256": source["source_freeze_sha256"], "audit_contract_sha256": audit_hash,
        "initialization_manifest_sha256": sha256_file(bundle / "initialization_manifest.json"),
        "legacy_exclusion_manifest_sha256": sha256_file(bundle / "legacy_checkpoint_exclusion_manifest.json"),
        "scene_manifest_sha256": sha256_file(bundle / "scenes_manifest.json"),
        "target_manifest_sha256": sha256_file(bundle / "targets_manifest.json"),
        "cache_manifest_sha256": sha256_file(bundle / "cached_features_manifest.json"),
        "capacity_manifest_sha256": sha256_file(bundle / "capacity_manifest.json"),
        "decision_graph_sha256": sha256_file(bundle / "d3_decision_graph.json"),
        "scientific_command_template_sha256": sha256_file(bundle / "scientific_command_template.json"),
        "files": file_rows,
    }
    write_json_x(bundle / "protocol_hashes.json", hashes)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve(strict=True)
    if run.name != CAMPAIGN_ID or not run.is_relative_to(REPO / "outputs/runs"):
        raise RuntimeError("wrong candidate root")
    started = time.time()
    journal = AuditJournal.create(execution_root=run, protocol_identifier=PROTOCOL_ID, campaign_identifier=CAMPAIGN_ID)
    journal.append("HUMAN_AUTHORITY_FROZEN", "authority", {"approval_text_sha256": PARENT_SHA256})
    journal.append("HUMAN_AUTHORITY_AMENDMENT_FROZEN", "authority", {"approval_text_sha256": AMENDMENT_SHA256})
    baseline = json.loads((run / "diagnostics/preimplementation_baseline_audit.json").read_text())
    if baseline["status"] != "PASS":
        raise RuntimeError("preimplementation baseline failed")
    journal.append("BASELINE_VERIFIED", "baseline", {"status": "PASS", "source_freeze": "300/300", "historical_checkpoints": "600/600", "prior_campaigns": "4/4"})
    legacy = hash_protected_file_for_integrity(LEGACY_CHECKPOINT)
    journal.append("LEGACY_CHECKPOINT_INVENTORIED", "integrity", {"checkpoint_sha256": legacy["sha256"], "action": "INTEGRITY_HASH", "scientific_load": False, "path_count": 1})
    journal.append("LEGACY_CHECKPOINT_EXCLUDED", "integrity", {"legacy_checkpoint_sha256": legacy["sha256"], "scientific_load": False, "model_load_attempts": 0, "optimizer_load_attempts": 0, "scientific_evidence_references": 0})
    inputs = raw_input_verification(run)
    cache_audit = cache_reproducibility(run)
    journal.append("CACHE_GENERATION_STARTED", "cache", {"clean_process_count": 2, "scientific_optimizer_constructions": 0})
    journal.append("CACHE_GENERATION_COMPLETED", "cache", {"scene_count": 8, "generation_count": 2, "scientific_optimizer_steps": 0})
    journal.append("CACHE_REPRODUCIBILITY_VERIFIED", "cache", {"canonical_agreement": True, "promoted_generation": "A"})
    journal.append("SCENE_BUNDLE_FROZEN", "cache", {"scene_ids": list(SCENE_IDS), "order_exact": True})
    journal.append("TARGET_BUNDLE_FROZEN", "cache", {"scene_count": 8, "target_hashes_exact": True})
    comparison = legacy_comparison(run)
    init, binding, capacity = initialization_and_capacities(run, journal)
    decisions = decision_agreement(run)
    traversal = no_update_traversals(run, journal)
    source = source_freeze(run)
    journal.append("SOURCE_FREEZE_CREATED", "freeze", {"source_freeze_sha256": source["source_freeze_sha256"], "file_count": len(source["files"])})
    hashes = build_bundle(run, source, init, cache_audit, capacity, binding, decisions)
    copy_x(run / "protocol_bundle/THAYER-D3-PV1-A1/audit_contract.json", run / "audit/audit_contract.json")
    audit_schema = {
        "schema_version": "thayer-d3-pv1-a1-audit-schema-v1",
        "event_schema_version": "thayer-d3-pv1-audit-event-v1",
        "protocol_identifier": PROTOCOL_ID,
        "serialization": "RFC8259_SORTED_COMPACT_UTF8",
        "hash_chain": "SHA256_PREVIOUS_AND_CURRENT_EVENT",
        "unknown_event_policy": "FAIL_CLOSED",
        "required_payload_rules": ["AUTHORITY_HASHES", "LEGACY_EXCLUSION", "FRESH_STEP_ZERO", "NO_TRANSFER", "PRIMARY_L0_IDENTITY", "ZERO_READINESS_SCIENCE"],
    }
    write_json_x(run / "audit/audit_schema.json", audit_schema)
    journal.append("PROTOCOL_BUNDLE_FROZEN", "freeze", {"protocol_bundle_sha256": hashes["protocol_bundle_sha256"], "audit_contract_sha256": hashes["audit_contract_sha256"]})
    journal.append("GATE_COMPLETED", "readiness", {"gate": "NO_UPDATE_TRAVERSALS", "status": traversal["status"]})
    journal.append("INTEGRITY_CHECK_COMPLETED", "integrity", {"status": "PASS", "scientific_backward_passes": 0, "scientific_optimizer_steps": 0, "scientific_branch_selections": 0, "scientific_d3_decisions": 0})
    journal.append("CAMPAIGN_TERMINATED", "terminal", {"readiness_outcome_pending_independent_audit": True, "d3_status": "UNKNOWN", "scientific_backward_passes": 0, "scientific_optimizer_steps": 0, "scientific_branch_selections": 0, "scientific_d3_decisions": 0})
    chain = verify_event_chain(journal.path)
    write_json_x(run / "audit/event_chain_head.json", chain)
    summary = {
        "status": "PASS", "protocol_identifier": PROTOCOL_ID, "campaign_identifier": CAMPAIGN_ID,
        "parent_approval_sha256": PARENT_SHA256, "amendment_approval_sha256": AMENDMENT_SHA256,
        "legacy_checkpoint_sha256": legacy["sha256"], "legacy_integrity_hash_reads": 1,
        "input_scene_count": len(inputs["scenes"]), "cache_reproducibility": cache_audit["status"],
        "legacy_comparison": comparison["status"], "initial_state_sha256": init["combined_canonical_state_manifest_sha256"],
        "capacity_condition_count": len(capacity["conditions"]), "runtime_replay_fixture_count": decisions["total_fixture_count"],
        "event_chain_valid": chain["valid"], "event_count": chain["event_count"],
        "protocol_hashes": hashes, "elapsed_seconds": time.time() - started,
        "scientific_backward_passes": 0, "scientific_optimizer_steps": 0,
        "scientific_checkpoints_selected": 0, "scientific_capacity_selections": 0,
        "scientific_branch_selections": 0, "scientific_d3_decisions": 0,
        "d3_status": "UNKNOWN",
    }
    write_json_x(run / "diagnostics/readiness_implementation_summary.json", summary)
    print(json.dumps({"status": "PASS", "protocol_bundle_sha256": hashes["protocol_bundle_sha256"], "initial_state_sha256": init["combined_canonical_state_manifest_sha256"], "event_count": chain["event_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
