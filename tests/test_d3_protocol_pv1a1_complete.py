"""Complete A1 protocol, runtime/replay, corruption, and guard gates."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import pytest
import torch

from src.d3_audit_layer_pv1 import (
    A1_CAPACITIES,
    A1_EXPERT_SEEDS,
    A1_REPLICA_OFFSETS,
    AuditContractError,
    AuditJournal,
    IndependentAuditReplayer,
    LEGACY_LEARNED_CHECKPOINT_SHA256,
    construct_fresh_initial_state,
    validate_a1_event_semantics,
    validate_a1_initialization_manifest,
)
from src.d3_protocol_pv1a1 import (
    NoUpdateReadinessGuard,
    ProtocolContractError,
    RuntimeDecisionProducer,
    hard_two_expert_set_loss,
    validate_effective_protocol,
)


PARENT_SHA = "969a22da8eb8e54fc7a2a55a70bfd50996e0b6a551791ad7060d537b3d258a3a"
AMENDMENT_SHA = "4d3c227ba9ef7779bf987e95f26f39a0144717aeaf7067b66febce3467eff15e"


def protocol_values() -> dict[str, object]:
    return {
        "authority": {"parent_protocol_identifier": "THAYER-D3-PV1", "amendment_identifier": "A1"},
        "initialization": {
            "mode": "FRESH_SEEDED_STEP_ZERO", "l0_expert_seeds": [2026071201, 2026071202],
            "optimizer_step_counter": 0, "pre_step_optimizer_state": "EMPTY",
            "metric_history": "EMPTY", "semantic_checkpoint_history": "EMPTY",
        },
        "legacy_checkpoint": {"sha256": LEGACY_LEARNED_CHECKPOINT_SHA256, "scientific_use": "FORBIDDEN"},
        "primary_l0": {"run_count": 1, "capacity_ladder_reuse": "REQUIRED"},
        "eight_scene": {"scene_order": [f"scene-{i}" for i in range(8)], "optimizer_steps": 3200, "required_pass_count": 8},
        "capacity_ladder": {"levels": ["L0", "L1", "L2", "L3"], "replica_offsets": [0, 10000, 20000]},
        "decisions": {"tangent_threshold": 0.5, "tangent_comparison": "STRICTLY_LESS_THAN"},
        "retry_policy": "FORBIDDEN",
        "audit": {"contract_sha256": "a" * 64},
    }


def leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        result = set()
        for key, item in value.items():
            result |= leaf_paths(item, f"{prefix}/{key}")
        return result
    if isinstance(value, list):
        result = set()
        for index, item in enumerate(value):
            result |= leaf_paths(item, f"{prefix}/{index}")
        return result
    return {prefix}


def protocol() -> dict[str, object]:
    values = protocol_values()
    return {
        "schema_version": "thayer-d3-pv1-a1-effective-protocol-v1",
        "protocol_identifier": "THAYER-D3-PV1-A1",
        "values": values,
        "provenance_by_json_pointer": {path: "TEST_AUTHORITY" for path in leaf_paths(values)},
        "unknown_key_policy": "FAIL_CLOSED",
    }


def replay_protocol() -> dict[str, object]:
    return {
        "protocol_identifier": "THAYER-D3-PV1-A1",
        "tangent_threshold": 0.5,
        "tangent_comparison": "STRICTLY_LESS_THAN",
        "eight_scene_required_pass_count": 8,
        "capacities": [{"level": level} for level in A1_CAPACITIES],
    }


def l0_evidence(bits: tuple[bool, ...], capture: float) -> dict[str, object]:
    keys = (
        "valid", "l0_completed_validly", "source_integrity", "input_integrity",
        "metric_integrity", "scientific_contract_integrity", "frozen_l0_success",
        "d0_pass", "d1_pass", "other_validity_predicates_pass",
        "optimization_mechanism_supported", "hard_assignment_mechanism_supported",
        "square_mapping_mechanism_supported",
    )
    return {"evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE", **dict(zip(keys, bits)), "validated_tangent_capture": capture}


def test_effective_protocol_has_complete_leaf_provenance() -> None:
    result = validate_effective_protocol(protocol())
    assert result["valid"] and result["leaf_field_count"] == result["provenance_field_count"]


@pytest.mark.parametrize("mutation", ["unknown", "missing", "provenance", "null", "placeholder", "legacy_allowed", "wrong_init"])
def test_effective_protocol_corruption_fails_closed(mutation: str) -> None:
    value = protocol()
    if mutation == "unknown":
        value["unexpected"] = True
    elif mutation == "missing":
        del value["values"]["initialization"]["mode"]
    elif mutation == "provenance":
        value["provenance_by_json_pointer"].pop(next(iter(value["provenance_by_json_pointer"])))
    elif mutation == "null":
        value["values"]["retry_policy"] = None
    elif mutation == "placeholder":
        value["values"]["retry_policy"] = "PLACEHOLDER"
    elif mutation == "legacy_allowed":
        value["values"]["legacy_checkpoint"]["scientific_use"] = "ALLOWED"
    else:
        value["values"]["initialization"]["mode"] = "LEARNED_CHECKPOINT"
    with pytest.raises(ProtocolContractError):
        validate_effective_protocol(value)


def test_capacity_widths_seed_schedule_and_replica_offsets() -> None:
    assert A1_CAPACITIES == {"L0": (32, 16), "L1": (80, 40), "L2": (160, 80), "L3": (224, 112)}
    assert A1_EXPERT_SEEDS == {
        "L0": (2026071201, 2026071202), "L1": (2026072201, 2026072202),
        "L2": (2026073201, 2026073202), "L3": (2026074201, 2026074202),
    }
    assert A1_REPLICA_OFFSETS == (0, 10000, 20000)


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_each_capacity_constructs_exact_widths_without_transfer(level: str) -> None:
    state = construct_fresh_initial_state(level)
    dec2, dec1 = A1_CAPACITIES[level]
    assert state.model.expert_1.dec2.block[0].out_channels == dec2
    assert state.model.expert_1.dec1.block[0].out_channels == dec1
    assert state.manifest["transferred_model_tensors"] == 0
    assert state.manifest["transferred_optimizer_tensors"] == 0
    assert validate_a1_initialization_manifest(state.manifest)["valid"] is True


def test_runtime_and_independent_branch_replay_agree_exhaustively() -> None:
    runtime = RuntimeDecisionProducer()
    replay = IndependentAuditReplayer(replay_protocol())
    cases = 0
    for bits in itertools.product((False, True), repeat=13):
        for capture in (0.2, 0.5):
            evidence = l0_evidence(bits, capture)
            assert runtime.select_branch(evidence) == replay.select_branch(evidence)
            cases += 1
    assert cases == 16384


def test_runtime_and_independent_eight_scene_replay_agree_exhaustively() -> None:
    runtime = RuntimeDecisionProducer()
    replay = IndependentAuditReplayer(replay_protocol())
    for bits in itertools.product((False, True), repeat=8):
        evidence = {"valid": True, "scene_passes": list(bits), "evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE"}
        assert runtime.classify_terminal("EIGHT_SCENE", evidence) == replay.classify_terminal("EIGHT_SCENE", evidence)


def test_runtime_and_independent_capacity_replay_agree_exhaustively() -> None:
    runtime = RuntimeDecisionProducer()
    replay = IndependentAuditReplayer(replay_protocol())
    for first in ("L1", "L2", "L3"):
        smaller = {"L1": "L0", "L2": "L1", "L3": "L2"}[first]
        for candidate in itertools.product((False, True), repeat=3):
            for boundary in itertools.product((False, True), repeat=3):
                evidence = {"valid": True, "first_passing_level": first, "level_results": {first: list(candidate), smaller: list(boundary)}}
                assert runtime.classify_terminal("CAPACITY_LADDER", evidence) == replay.classify_terminal("CAPACITY_LADDER", evidence)


def test_no_update_guard_rejects_backward_and_optimizer_construction() -> None:
    tensor = torch.tensor(2.0, requires_grad=True)
    with NoUpdateReadinessGuard():
        with pytest.raises(ProtocolContractError, match="backward"):
            tensor.backward()
        with pytest.raises(ProtocolContractError, match="optimizer"):
            torch.optim.AdamW([tensor], lr=1e-3)


@pytest.mark.parametrize("action", ["optimizer_step", "select_semantic_checkpoint", "select_branch_from_metrics", "apply_d3"])
def test_no_update_guard_rejects_selection_and_update_actions(action: str) -> None:
    guard = NoUpdateReadinessGuard()
    with pytest.raises(ProtocolContractError):
        getattr(guard, action)()


def test_no_grad_target_dependent_loss_is_finite() -> None:
    outputs = torch.zeros(2, 2, 6, 4, 4)
    targets = torch.ones_like(outputs)
    with torch.no_grad():
        loss = hard_two_expert_set_loss(outputs, targets)
    assert loss.shape == () and torch.isfinite(loss) and not loss.requires_grad


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("seed", "seed"), ("step", "step"), ("optimizer", "optimizer"),
        ("metric", "metric"), ("semantic", "semantic"), ("model_transfer", "transfer"),
        ("optimizer_transfer", "transfer"), ("legacy_load", "legacy"), ("manifest", "manifest"),
    ],
)
def test_initial_state_corruption_matrix(mutation: str, message: str) -> None:
    manifest = copy.deepcopy(construct_fresh_initial_state("L0").manifest)
    if mutation == "seed": manifest["expert_seeds"][0] += 1
    elif mutation == "step": manifest["optimizer_step_counter"] = 1
    elif mutation == "optimizer": manifest["optimizer_state_entry_count_before_first_step"] = 1
    elif mutation == "metric": manifest["metric_history_count"] = 1
    elif mutation == "semantic": manifest["semantic_history_count"] = 1
    elif mutation == "model_transfer": manifest["transferred_model_tensors"] = 1
    elif mutation == "optimizer_transfer": manifest["transferred_optimizer_tensors"] = 1
    elif mutation == "legacy_load": manifest["legacy_checkpoint_loaded"] = True
    else: manifest["state_dict_tensors"][0]["canonical_tensor_content_sha256"] = "0" * 64
    with pytest.raises(AuditContractError, match=message):
        validate_a1_initialization_manifest(manifest)


def make_journal(tmp_path: Path, *, omit_amendment: bool = False, amendment_sha: str = AMENDMENT_SHA, mismatch: bool = False, duplicate: int = 0, legacy_action: str = "INTEGRITY_HASH") -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    journal = AuditJournal.create(execution_root=root, protocol_identifier="THAYER-D3-PV1-A1", campaign_identifier="fixture")
    journal.append("HUMAN_AUTHORITY_FROZEN", "authority", {"approval_text_sha256": PARENT_SHA})
    if not omit_amendment:
        journal.append("HUMAN_AUTHORITY_AMENDMENT_FROZEN", "authority", {"approval_text_sha256": amendment_sha})
    journal.append("LEGACY_CHECKPOINT_INVENTORIED", "integrity", {"checkpoint_sha256": LEGACY_LEARNED_CHECKPOINT_SHA256, "action": legacy_action, "scientific_load": legacy_action != "INTEGRITY_HASH"})
    journal.append("LEGACY_CHECKPOINT_EXCLUDED", "integrity", {"legacy_checkpoint_sha256": LEGACY_LEARNED_CHECKPOINT_SHA256, "scientific_load": False})
    journal.append("FRESH_INITIALIZATION_STARTED", "initialization", {"expert_seeds": [2026071201, 2026071202]})
    journal.append("FRESH_INITIALIZATION_COMPLETED", "initialization", {
        "expert_seeds": [2026071201, 2026071202], "optimizer_step_counter": 0,
        "optimizer_state_entry_count": 0, "metric_history_count": 0, "semantic_history_count": 0,
    })
    journal.append("FRESH_INITIALIZATION_REPRODUCED", "initialization", {"match": True})
    journal.append("INITIAL_STATE_FROZEN", "initialization", {"sha256": "a" * 64})
    journal.append("PRIMARY_L0_EVIDENCE_BOUND", "future_contract", {
        "branch_selection_run_id": "run-1", "primary_capacity_run_id": "run-2" if mismatch else "run-1",
        "branch_selection_evidence_id": "evidence-1", "primary_capacity_evidence_id": "evidence-1",
        "duplicate_primary_l0_run_count": duplicate,
    })
    return journal.path


@pytest.mark.parametrize("mutation", ["legacy_model", "legacy_optimizer", "legacy_branch", "amendment", "missing_amendment", "duplicate_l0", "evidence_mismatch"])
def test_a1_event_corruption_matrix(tmp_path: Path, mutation: str) -> None:
    kwargs: dict[str, object] = {}
    if mutation == "legacy_model": kwargs["legacy_action"] = "MODEL_STATE_LOAD"
    elif mutation == "legacy_optimizer": kwargs["legacy_action"] = "OPTIMIZER_STATE_LOAD"
    elif mutation == "legacy_branch": kwargs["legacy_action"] = "BRANCH_SELECTION_EVIDENCE"
    elif mutation == "amendment": kwargs["amendment_sha"] = "0" * 64
    elif mutation == "missing_amendment": kwargs["omit_amendment"] = True
    elif mutation == "duplicate_l0": kwargs["duplicate"] = 1
    else: kwargs["mismatch"] = True
    path = make_journal(tmp_path, **kwargs)
    with pytest.raises(AuditContractError):
        validate_a1_event_semantics(path, parent_approval_sha256=PARENT_SHA, amendment_approval_sha256=AMENDMENT_SHA)


def test_a1_valid_event_replay(tmp_path: Path) -> None:
    path = make_journal(tmp_path)
    assert validate_a1_event_semantics(path, parent_approval_sha256=PARENT_SHA, amendment_approval_sha256=AMENDMENT_SHA)["valid"] is True
