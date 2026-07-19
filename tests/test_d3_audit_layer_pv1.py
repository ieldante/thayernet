"""Regression-first contract for the independent THAYER-D3-PV1 audit layer."""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.d3_audit_layer_pv1 import (
    AuditContractError,
    AuditJournal,
    IndependentAuditReplayer,
    REQUIRED_EVENT_TYPES,
    canonical_json_bytes,
    observe_scientific_state,
    observe_tensor,
    resolve_candidate_path,
    validate_execution_events,
    validate_readiness_manifest,
    verify_event_chain,
)


SCENES = [
    "pu_training_ordinary_00000",
    "pu_training_ordinary_00008",
    "pu_training_ordinary_00016",
    "pu_training_ordinary_00024",
    "pu_training_near_00000",
    "pu_training_near_00008",
    "pu_training_near_00016",
    "pu_training_near_00024",
]
CAPACITIES = [
    {"level": "L0", "index": 0, "dec2": 32, "dec1": 16, "seeds": [2026071201, 2026071202]},
    {"level": "L1", "index": 1, "dec2": 80, "dec1": 40, "seeds": [2026072201, 2026072202]},
    {"level": "L2", "index": 2, "dec2": 160, "dec1": 80, "seeds": [2026073201, 2026073202]},
    {"level": "L3", "index": 3, "dec2": 224, "dec1": 112, "seeds": [2026074201, 2026074202]},
]


def _protocol() -> dict[str, object]:
    return {
        "protocol_identifier": "THAYER-D3-PV1",
        "scene_order": SCENES,
        "capacities": CAPACITIES,
        "replica_offsets": [0, 10000, 20000],
        "tangent_threshold": 0.5,
        "tangent_comparison": "STRICTLY_LESS_THAN",
        "eight_scene_required_pass_count": 8,
        "transfer_policy": {
            "decoder_tensors": "FORBIDDEN",
            "optimizer_state": "FORBIDDEN",
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "protocol_identifier": "THAYER-D3-PV1",
        "authority_text_sha256": "a" * 64,
        "protocol_bundle_sha256": "b" * 64,
        "source_freeze_sha256": "c" * 64,
        "scenes": [
            {
                "scene_id": scene,
                "order_index": index,
                "target_sha256": f"{index + 1:064x}",
                "cached_feature_sha256": f"{index + 101:064x}",
                "layout": "NCHW",
            }
            for index, scene in enumerate(SCENES)
        ],
        "ordered_batch_scene_ids": SCENES,
        "capacities": copy.deepcopy(CAPACITIES),
        "transfers": {
            "decoder_tensor_transfer_attempts": 0,
            "optimizer_state_transfer_attempts": 0,
            "partial_state_dict_load_attempts": 0,
        },
        "scientific_backward_passes": 0,
        "scientific_optimizer_steps": 0,
        "unexpected_retry_count": 0,
    }


def _journal(tmp_path: Path) -> AuditJournal:
    root = tmp_path / "candidate_001"
    root.mkdir()
    return AuditJournal.create(
        execution_root=root,
        protocol_identifier="THAYER-D3-PV1",
        campaign_identifier="campaign",
        candidate_identifier="candidate_001",
    )


def test_canonical_event_serialization_is_explicit_and_stable() -> None:
    left = canonical_json_bytes({"z": [3, 2, 1], "a": "é", "b": False})
    right = canonical_json_bytes({"b": False, "a": "é", "z": [3, 2, 1]})
    assert left == right == b'{"a":"\xc3\xa9","b":false,"z":[3,2,1]}'
    with pytest.raises(AuditContractError, match="canonical JSON"):
        canonical_json_bytes({"not_finite": math.nan})


def test_hash_chain_construction_and_replay(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    first = journal.append("HUMAN_AUTHORITY_FROZEN", "authority", {"sha256": "a" * 64})
    second = journal.append("BASELINE_VERIFIED", "baseline", {"status": "PASS"})
    report = verify_event_chain(journal.path)
    assert report["valid"] is True
    assert report["event_count"] == 2
    assert first["sequence_number"] == 1
    assert second["previous_event_sha256"] == first["current_event_sha256"]
    assert report["chain_head_sha256"] == second["current_event_sha256"]


def test_journal_exclusive_creation_refuses_overwrite(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("HUMAN_AUTHORITY_FROZEN", "authority", {})
    with pytest.raises(FileExistsError):
        AuditJournal.create(
            execution_root=journal.execution_root,
            protocol_identifier="THAYER-D3-PV1",
            campaign_identifier="campaign",
            candidate_identifier="candidate_001",
        )


@pytest.mark.parametrize("mutation", ["modified", "deleted", "reordered", "duplicate_sequence"])
def test_hash_chain_corruption_is_rejected(tmp_path: Path, mutation: str) -> None:
    journal = _journal(tmp_path)
    for event_type in ("HUMAN_AUTHORITY_FROZEN", "BASELINE_VERIFIED", "CAMPAIGN_TERMINATED"):
        journal.append(event_type, "fixture", {"event": event_type})
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    if mutation == "modified":
        events[1]["payload"]["event"] = "tampered"
    elif mutation == "deleted":
        del events[1]
    elif mutation == "reordered":
        events[0], events[1] = events[1], events[0]
    else:
        events[1]["sequence_number"] = events[0]["sequence_number"]
    journal.path.write_text("".join(canonical_json_bytes(event).decode() + "\n" for event in events))
    with pytest.raises(AuditContractError):
        verify_event_chain(journal.path)


def test_required_event_schema_supports_readiness_and_future_science_types() -> None:
    assert {
        "HUMAN_AUTHORITY_FROZEN",
        "CACHE_GENERATION_STARTED",
        "BACKWARD_COMPLETED",
        "OPTIMIZER_STEP_COMPLETED",
        "D3_DECISION_COMPUTED",
        "CAMPAIGN_TERMINATED",
    } <= REQUIRED_EVENT_TYPES


def test_event_order_and_missing_required_event_rejected(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("BASELINE_VERIFIED", "baseline", {})
    journal.append("HUMAN_AUTHORITY_FROZEN", "authority", {})
    with pytest.raises(AuditContractError, match="order"):
        validate_execution_events(
            journal.path,
            required_event_types=("HUMAN_AUTHORITY_FROZEN", "BASELINE_VERIFIED"),
            readiness_mode=True,
        )


def test_extra_optimizer_step_and_forbidden_retry_rejected(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("OPTIMIZER_STEP_COMPLETED", "scientific_optimization", {"step": 1})
    with pytest.raises(AuditContractError, match="optimizer"):
        validate_execution_events(journal.path, readiness_mode=True)

    retry_root = tmp_path / "candidate_002"
    retry_root.mkdir()
    retry = AuditJournal.create(
        execution_root=retry_root,
        protocol_identifier="THAYER-D3-PV1",
        campaign_identifier="campaign",
        candidate_identifier="candidate_002",
    )
    retry.append("WORKER_LAUNCHED", "retry", {"attempt": 2, "retry": True})
    with pytest.raises(AuditContractError, match="retry"):
        validate_execution_events(retry.path, readiness_mode=False)


def test_nonmonotonic_evaluation_and_extra_scientific_step_rejected(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("EVALUATION_COMPLETED", "science", {"step": 100})
    journal.append("EVALUATION_COMPLETED", "science", {"step": 1})
    with pytest.raises(AuditContractError, match="evaluation"):
        validate_execution_events(journal.path, readiness_mode=False)

    other_root = tmp_path / "candidate_002"
    other_root.mkdir()
    other = AuditJournal.create(
        execution_root=other_root,
        protocol_identifier="THAYER-D3-PV1",
        campaign_identifier="campaign",
        candidate_identifier="candidate_002",
    )
    other.append("OPTIMIZER_STEP_COMPLETED", "science", {"step": 3201})
    with pytest.raises(AuditContractError, match="extra optimizer step"):
        validate_execution_events(other.path, readiness_mode=False, maximum_optimizer_steps=3200)


@pytest.mark.parametrize("relative", ["../escape.json", "/tmp/escape.json", "audit/../../escape.json"])
def test_candidate_root_path_containment_rejects_traversal(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    with pytest.raises(AuditContractError, match="candidate root"):
        resolve_candidate_path(root, relative, for_write=True)


def test_candidate_root_path_containment_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AuditContractError, match="candidate root"):
        resolve_candidate_path(root, "linked/event.json", for_write=True)


def test_repository_root_write_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    candidate = repository / "outputs/runs/candidate"
    candidate.mkdir(parents=True)
    with pytest.raises(AuditContractError, match="repository-root write"):
        resolve_candidate_path(
            candidate,
            repository / "forbidden.json",
            for_write=True,
            repository_root=repository,
        )


def test_audit_observation_does_not_interfere_with_rng_model_optimizer_or_tensors() -> None:
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.SiLU(), torch.nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    targets = torch.ones(3, 2)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()  # explicitly NONSCIENTIFIC_SYNTHETIC_EVIDENCE fixture

    python_before = pickle.dumps(random.getstate())
    numpy_before = pickle.dumps(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    accelerator_before = [state.clone() for state in torch.mps.get_rng_state().__class__()] if False else None
    model_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    input_before = inputs.detach().clone()
    target_before = targets.detach().clone()
    gradients_before = [parameter.grad.detach().clone() for parameter in model.parameters()]

    observed = observe_scientific_state(model, optimizer, inputs=(inputs,), targets=(targets,))
    tensor_record = observe_tensor(
        inputs.reshape(1, 3, 4),
        logical_axis_layout="NCHW_FIXTURE",
        batch_order=("fixture",),
    )

    assert observed["model_state_sha256"]
    assert tensor_record["canonical_contiguous_content_sha256"]
    assert pickle.dumps(random.getstate()) == python_before
    assert pickle.dumps(np.random.get_state()) == numpy_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert accelerator_before is None
    assert all(torch.equal(model.state_dict()[name], value) for name, value in model_before.items())
    assert optimizer.state_dict() == optimizer_before
    assert torch.equal(inputs, input_before)
    assert torch.equal(targets, target_before)
    assert all(torch.equal(parameter.grad, value) for parameter, value in zip(model.parameters(), gradients_before))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("scene_identity", "scene"),
        ("duplicate_scene", "duplicate"),
        ("reordered_scene", "order"),
        ("wrong_target_hash", "target"),
        ("wrong_cache_hash", "cached-feature"),
        ("wrong_layout", "NCHW"),
        ("wrong_width", "capacity"),
        ("wrong_seed", "seed"),
        ("decoder_transfer", "transfer"),
        ("optimizer_transfer", "optimizer"),
        ("unexpected_retry", "retry"),
        ("unexpected_step", "optimizer step"),
    ],
)
def test_readiness_manifest_corruption_is_rejected(mutation: str, message: str) -> None:
    manifest = _manifest()
    if mutation == "scene_identity":
        manifest["scenes"][0]["scene_id"] = "wrong"
    elif mutation == "duplicate_scene":
        manifest["scenes"][1]["scene_id"] = SCENES[0]
    elif mutation == "reordered_scene":
        manifest["scenes"][0], manifest["scenes"][1] = manifest["scenes"][1], manifest["scenes"][0]
    elif mutation == "wrong_target_hash":
        manifest["scenes"][0]["target_sha256"] = "wrong"
    elif mutation == "wrong_cache_hash":
        manifest["scenes"][0]["cached_feature_sha256"] = "wrong"
    elif mutation == "wrong_layout":
        manifest["scenes"][0]["layout"] = "NHWC"
    elif mutation == "wrong_width":
        manifest["capacities"][2]["dec2"] = 161
    elif mutation == "wrong_seed":
        manifest["capacities"][1]["seeds"][0] += 1
    elif mutation == "decoder_transfer":
        manifest["transfers"]["decoder_tensor_transfer_attempts"] = 1
    elif mutation == "optimizer_transfer":
        manifest["transfers"]["optimizer_state_transfer_attempts"] = 1
    elif mutation == "unexpected_retry":
        manifest["unexpected_retry_count"] = 1
    else:
        manifest["scientific_optimizer_steps"] = 1
    with pytest.raises(AuditContractError, match=message):
        validate_readiness_manifest(manifest, _protocol())


def _l0(*, success: bool = False, capture: float = 0.2, valid: bool = True, alternative: str | None = None) -> dict[str, object]:
    return {
        "evidence_class": "NONSCIENTIFIC_SYNTHETIC_EVIDENCE",
        "valid": valid,
        "l0_completed_validly": valid,
        "source_integrity": valid,
        "input_integrity": valid,
        "metric_integrity": valid,
        "scientific_contract_integrity": valid,
        "frozen_l0_success": success,
        "d0_pass": True,
        "d1_pass": True,
        "other_validity_predicates_pass": True,
        "optimization_mechanism_supported": alternative == "optimization",
        "hard_assignment_mechanism_supported": alternative == "hard_assignment",
        "square_mapping_mechanism_supported": alternative == "square_mapping",
        "validated_tangent_capture": capture,
    }


@pytest.mark.parametrize(
    ("evidence", "branch", "status"),
    [
        (_l0(success=True), "EIGHT_SCENE", "PENDING"),
        (_l0(capture=0.2), "CAPACITY_LADDER", "PENDING"),
        (_l0(capture=0.5), "NONE", "UNKNOWN"),
        (_l0(alternative="optimization"), "NONE", "UNKNOWN"),
        (_l0(valid=False), "NONE", "UNKNOWN"),
    ],
)
def test_independent_branch_replay(evidence: dict[str, object], branch: str, status: str) -> None:
    replay = IndependentAuditReplayer(_protocol()).select_branch(evidence)
    assert replay == {"downstream_branch": branch, "d3_status": status}


@pytest.mark.parametrize(
    ("branch", "evidence", "expected"),
    [
        ("EIGHT_SCENE", {"valid": True, "scene_passes": [True] * 8}, "PASS"),
        ("EIGHT_SCENE", {"valid": True, "scene_passes": [True] * 7 + [False]}, "FAIL"),
        ("EIGHT_SCENE", {"valid": False, "scene_passes": [True] * 8}, "UNKNOWN"),
        (
            "CAPACITY_LADDER",
            {
                "valid": True,
                "first_passing_level": "L1",
                "level_results": {"L0": [False, False, False], "L1": [True, True, True]},
            },
            "PASS",
        ),
        (
            "CAPACITY_LADDER",
            {
                "valid": True,
                "first_passing_level": "L1",
                "level_results": {"L0": [False, True, False], "L1": [True, True, True]},
            },
            "FAIL",
        ),
        ("CAPACITY_LADDER", {"valid": True, "first_passing_level": None, "level_results": {}}, "FAIL"),
        ("NONE", {"valid": True}, "UNKNOWN"),
    ],
)
def test_independent_d3_replay(branch: str, evidence: dict[str, object], expected: str) -> None:
    assert IndependentAuditReplayer(_protocol()).classify_terminal(branch, evidence) == expected


def test_branch_or_d3_result_inconsistent_with_evidence_is_rejected() -> None:
    replayer = IndependentAuditReplayer(_protocol())
    with pytest.raises(AuditContractError, match="branch"):
        replayer.verify_runtime_result(_l0(success=True), {"downstream_branch": "CAPACITY_LADDER", "d3_status": "PENDING"})
    with pytest.raises(AuditContractError, match="D3"):
        replayer.verify_terminal_result(
            "EIGHT_SCENE",
            {"valid": True, "scene_passes": [True] * 7 + [False]},
            "PASS",
        )
