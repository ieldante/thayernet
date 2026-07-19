"""Regression-first contract for THAYER-D3-PV1 Amendment A1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import src.d3_audit_layer_pv1 as audit


LEGACY_SHA256 = "8b06e788853a9180df7f83803d25cab17e362aac602c2932efe8dee680fa591e"
SEEDS = (2026071201, 2026071202)


def required(name: str):
    assert hasattr(audit, name), f"A1 contract API missing: {name}"
    return getattr(audit, name)


def test_a1_01_legacy_checkpoint_classified_by_exact_sha256() -> None:
    classify = required("classify_checkpoint_sha256")
    assert classify(LEGACY_SHA256) == "PROTECTED_LEGACY_LEARNED_CHECKPOINT_ONLY"


def test_a1_02_integrity_hash_does_not_deserialize_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hash_file = required("hash_protected_file_for_integrity")
    path = tmp_path / "legacy.pth"
    path.write_bytes(b"protected bytes")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("torch.load must not be called"))
    assert hash_file(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())["read_mode"] == "BYTE_LEVEL_INTEGRITY_ONLY"


def test_a1_03_legacy_model_tensor_load_rejected() -> None:
    reject = required("reject_protected_checkpoint_use")
    with pytest.raises(audit.AuditContractError, match="model"):
        reject(LEGACY_SHA256, "MODEL_STATE_LOAD")


def test_a1_04_legacy_optimizer_state_load_rejected() -> None:
    reject = required("reject_protected_checkpoint_use")
    with pytest.raises(audit.AuditContractError, match="optimizer"):
        reject(LEGACY_SHA256, "OPTIMIZER_STATE_LOAD")


def test_a1_05_legacy_branch_evidence_rejected() -> None:
    reject = required("reject_protected_checkpoint_use")
    with pytest.raises(audit.AuditContractError, match="branch"):
        reject(LEGACY_SHA256, "BRANCH_SELECTION_EVIDENCE")


def test_a1_06_legacy_capacity_evidence_rejected() -> None:
    reject = required("reject_protected_checkpoint_use")
    with pytest.raises(audit.AuditContractError, match="capacity"):
        reject(LEGACY_SHA256, "CAPACITY_EVIDENCE")


def test_a1_07_fresh_l0_uses_exact_approved_seeds() -> None:
    construct = required("construct_fresh_initial_state")
    state = construct("L0", SEEDS)
    assert tuple(state.manifest["expert_seeds"]) == SEEDS


def test_a1_08_two_clean_process_l0_states_match() -> None:
    reproduce = required("reproduce_initial_state_clean_processes")
    report = reproduce("L0", SEEDS, process_count=2)
    assert report["canonical_state_hashes_match"] is True
    assert len(report["constructions"]) == 2


def test_a1_09_fresh_initialization_is_step_zero() -> None:
    state = required("construct_fresh_initial_state")("L0", SEEDS)
    assert state.manifest["optimizer_step_counter"] == 0


def test_a1_10_fresh_initialization_has_no_metric_history() -> None:
    state = required("construct_fresh_initial_state")("L0", SEEDS)
    assert state.manifest["metric_history_count"] == 0


def test_a1_11_fresh_initialization_has_no_semantic_history() -> None:
    state = required("construct_fresh_initial_state")("L0", SEEDS)
    assert state.manifest["semantic_history_count"] == 0


def test_a1_12_new_optimizer_state_is_empty() -> None:
    state = required("construct_fresh_initial_state")("L0", SEEDS)
    optimizer = required("create_fresh_a1_optimizer")(state.model)
    assert len(optimizer.state) == 0


def test_a1_13_no_model_tensor_transfer_recorded() -> None:
    state = required("construct_fresh_initial_state")("L0", SEEDS)
    assert state.manifest["transferred_model_tensors"] == 0


def test_a1_14_no_optimizer_tensor_transfer_recorded() -> None:
    state = required("construct_fresh_initial_state")("L0", SEEDS)
    assert state.manifest["transferred_optimizer_tensors"] == 0


def test_a1_15_initialization_artifact_must_be_candidate_contained(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    freeze = required("freeze_initial_state_manifest")
    state = required("construct_fresh_initial_state")("L0", SEEDS)
    path = freeze(root, "initial_state/l0.json", state.manifest)
    assert path.is_relative_to(root.resolve())


def test_a1_16_branch_and_primary_capacity_l0_share_evidence_identity() -> None:
    bind = required("PrimaryL0EvidenceBinding")
    binding = bind()
    binding.bind_branch_selection("future-l0-run", "evidence-001")
    binding.bind_primary_capacity("future-l0-run", "evidence-001")
    assert binding.verify()["identity_exact"] is True


def test_a1_17_duplicate_primary_l0_run_is_rejected() -> None:
    bind = required("PrimaryL0EvidenceBinding")
    binding = bind()
    binding.bind_branch_selection("future-l0-run", "evidence-001")
    binding.bind_primary_capacity("future-l0-run", "evidence-001")
    with pytest.raises(audit.AuditContractError, match="duplicate"):
        binding.bind_primary_capacity("duplicate-l0-run", "evidence-002")


def test_a1_18_eight_scene_initialization_identity_is_separate() -> None:
    bind = required("PrimaryL0EvidenceBinding")
    binding = bind()
    binding.bind_branch_selection("future-l0-run", "evidence-001")
    binding.bind_eight_scene("future-eight-scene-run", "eight-initialization-001")
    assert binding.verify()["eight_scene_separate"] is True


def test_a1_19_legacy_checkpoint_bytes_remain_exact() -> None:
    path = Path("outputs/runs/thayer_output_parameterization_20260713_023120/checkpoints/ambiguous_one_scene_square.pth")
    record = required("hash_protected_file_for_integrity")(path, expected_sha256=LEGACY_SHA256)
    assert record["sha256"] == LEGACY_SHA256


def test_a1_20_effective_protocol_contains_exclusion_not_equivalence() -> None:
    protocol = required("effective_a1_protocol")()
    encoded = json.dumps(protocol, sort_keys=True).lower()
    assert protocol["initialization"]["mode"] == "FRESH_SEEDED_STEP_ZERO"
    assert protocol["legacy_checkpoint"]["scientific_use"] == "FORBIDDEN"
    assert "learned-checkpoint/fresh-seed equivalence" not in encoded
