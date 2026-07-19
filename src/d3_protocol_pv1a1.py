"""Runtime protocol producer and no-update guards for THAYER-D3-PV1-A1.

The independent decision replayer intentionally lives in
``src.d3_audit_layer_pv1`` and is neither imported nor called here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping

import torch


class ProtocolContractError(RuntimeError):
    """Fail-closed protocol-as-data or readiness-guard violation."""


def _boolean(evidence: Mapping[str, object], key: str) -> bool:
    value = evidence.get(key)
    if value is not True and value is not False:
        raise ProtocolContractError(f"missing Boolean runtime evidence: {key}")
    return bool(value)


@dataclass(frozen=True)
class RuntimeDecisionProducer:
    """Pure runtime branch and terminal decision implementation."""

    tangent_threshold: float = 0.5
    required_scene_count: int = 8

    def select_branch(self, evidence: Mapping[str, object]) -> dict[str, str]:
        validity = (
            "valid", "l0_completed_validly", "source_integrity", "input_integrity",
            "metric_integrity", "scientific_contract_integrity",
        )
        if any(not _boolean(evidence, key) for key in validity):
            return {"downstream_branch": "NONE", "d3_status": "UNKNOWN"}
        if _boolean(evidence, "frozen_l0_success"):
            return {"downstream_branch": "EIGHT_SCENE", "d3_status": "PENDING"}
        capacity_valid = (
            _boolean(evidence, "d0_pass")
            and _boolean(evidence, "d1_pass")
            and _boolean(evidence, "other_validity_predicates_pass")
        )
        alternatives_excluded = not any(
            _boolean(evidence, key)
            for key in (
                "optimization_mechanism_supported",
                "hard_assignment_mechanism_supported",
                "square_mapping_mechanism_supported",
            )
        )
        capture = evidence.get("validated_tangent_capture")
        low_capture = (
            isinstance(capture, (int, float))
            and not isinstance(capture, bool)
            and math.isfinite(float(capture))
            and float(capture) < self.tangent_threshold
        )
        if capacity_valid and alternatives_excluded and low_capture:
            return {"downstream_branch": "CAPACITY_LADDER", "d3_status": "PENDING"}
        return {"downstream_branch": "NONE", "d3_status": "UNKNOWN"}

    def classify_terminal(self, branch: str, evidence: Mapping[str, object]) -> str:
        if branch == "NONE":
            return "UNKNOWN"
        if evidence.get("valid") is not True:
            return "UNKNOWN"
        if branch == "EIGHT_SCENE":
            passes = evidence.get("scene_passes")
            if (
                not isinstance(passes, list)
                or len(passes) != self.required_scene_count
                or any(value is not True and value is not False for value in passes)
            ):
                return "UNKNOWN"
            return "PASS" if all(passes) else "FAIL"
        if branch != "CAPACITY_LADDER":
            raise ProtocolContractError("unknown runtime branch")
        if evidence.get("valid") is not True:
            return "UNKNOWN"
        first = evidence.get("first_passing_level")
        results = evidence.get("level_results")
        if first is None:
            return "FAIL"
        levels = ["L0", "L1", "L2", "L3"]
        if first not in levels or first == "L0" or not isinstance(results, Mapping):
            return "UNKNOWN"
        smaller = levels[levels.index(str(first)) - 1]
        candidate = results.get(first)
        boundary = results.get(smaller)
        if (
            not isinstance(candidate, list) or not isinstance(boundary, list)
            or len(candidate) != 3 or len(boundary) != 3
            or any(value is not True and value is not False for value in candidate + boundary)
        ):
            return "UNKNOWN"
        return "PASS" if all(candidate) and not any(boundary) else "FAIL"


REQUIRED_VALUE_PATHS = frozenset(
    {
        "/authority/parent_protocol_identifier",
        "/authority/amendment_identifier",
        "/initialization/mode",
        "/initialization/l0_expert_seeds",
        "/initialization/optimizer_step_counter",
        "/initialization/pre_step_optimizer_state",
        "/initialization/metric_history",
        "/initialization/semantic_checkpoint_history",
        "/legacy_checkpoint/sha256",
        "/legacy_checkpoint/scientific_use",
        "/primary_l0/run_count",
        "/primary_l0/capacity_ladder_reuse",
        "/eight_scene/scene_order",
        "/eight_scene/optimizer_steps",
        "/eight_scene/required_pass_count",
        "/capacity_ladder/levels",
        "/capacity_ladder/replica_offsets",
        "/decisions/tangent_threshold",
        "/decisions/tangent_comparison",
        "/retry_policy",
        "/audit/contract_sha256",
    }
)


def _leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            result.update(_leaf_paths(item, f"{prefix}/{token}"))
        return result
    if isinstance(value, list):
        result = set()
        for index, item in enumerate(value):
            result.update(_leaf_paths(item, f"{prefix}/{index}"))
        return result
    return {prefix}


def _lookup(values: Mapping[str, object], pointer: str) -> object:
    current: object = values
    for token in pointer.strip("/").split("/"):
        if not isinstance(current, Mapping) or token not in current:
            raise ProtocolContractError(f"missing required protocol key: {pointer}")
        current = current[token]
    return current


def validate_effective_protocol(protocol: Mapping[str, object]) -> dict[str, object]:
    """Reject unknown keys, missing science, absent provenance, and fallbacks."""

    allowed = {
        "schema_version", "protocol_identifier", "values",
        "provenance_by_json_pointer", "unknown_key_policy",
    }
    unknown = set(protocol) - allowed
    if unknown:
        raise ProtocolContractError(f"unknown top-level protocol keys: {sorted(unknown)}")
    if protocol.get("schema_version") != "thayer-d3-pv1-a1-effective-protocol-v1":
        raise ProtocolContractError("wrong effective protocol schema")
    if protocol.get("protocol_identifier") != "THAYER-D3-PV1-A1":
        raise ProtocolContractError("wrong effective protocol identifier")
    if protocol.get("unknown_key_policy") != "FAIL_CLOSED":
        raise ProtocolContractError("unknown-key policy must fail closed")
    values = protocol.get("values")
    provenance = protocol.get("provenance_by_json_pointer")
    if not isinstance(values, Mapping) or not isinstance(provenance, Mapping):
        raise ProtocolContractError("protocol values and provenance maps are required")
    for pointer in REQUIRED_VALUE_PATHS:
        _lookup(values, pointer)
    leaf_paths = _leaf_paths(values)
    missing_provenance = sorted(path for path in leaf_paths if path not in provenance)
    extra_provenance = sorted(set(provenance) - leaf_paths)
    if missing_provenance or extra_provenance:
        raise ProtocolContractError(
            f"protocol provenance coverage mismatch: missing={missing_provenance}, extra={extra_provenance}"
        )
    if any(value is None for value in _iter_leaves(values)):
        raise ProtocolContractError("null scientific values are forbidden")
    encoded = json.dumps(values, sort_keys=True).lower()
    if any(token in encoded for token in ('"todo"', '"placeholder"', '"fallback"')):
        raise ProtocolContractError("placeholder or permissive fallback found")
    if _lookup(values, "/initialization/mode") != "FRESH_SEEDED_STEP_ZERO":
        raise ProtocolContractError("wrong A1 initialization mode")
    if _lookup(values, "/legacy_checkpoint/scientific_use") != "FORBIDDEN":
        raise ProtocolContractError("legacy checkpoint is not scientifically excluded")
    return {
        "valid": True,
        "leaf_field_count": len(leaf_paths),
        "provenance_field_count": len(provenance),
        "unknown_key_policy": "FAIL_CLOSED",
    }


def _iter_leaves(value: object):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_leaves(item)
    else:
        yield value


class NoUpdateReadinessGuard:
    """Runtime guard rejecting update, selection, and D3 actions in readiness."""

    def __init__(self) -> None:
        self._original_tensor_backward = None
        self._original_autograd_backward = None
        self._original_adamw_init = None
        self.rejection_counts = {
            "backward": 0,
            "scientific_optimizer_construction": 0,
            "optimizer_step": 0,
            "semantic_checkpoint_selection": 0,
            "branch_selection": 0,
            "d3_application": 0,
        }

    def _reject_backward(self, *args, **kwargs):
        self.rejection_counts["backward"] += 1
        raise ProtocolContractError("backward is forbidden during no-update readiness")

    def _reject_optimizer(self, *args, **kwargs):
        self.rejection_counts["scientific_optimizer_construction"] += 1
        raise ProtocolContractError("scientific optimizer construction is forbidden during readiness")

    def __enter__(self) -> "NoUpdateReadinessGuard":
        self._original_tensor_backward = torch.Tensor.backward
        self._original_autograd_backward = torch.autograd.backward
        self._original_adamw_init = torch.optim.AdamW.__init__
        torch.Tensor.backward = self._reject_backward  # type: ignore[method-assign]
        torch.autograd.backward = self._reject_backward  # type: ignore[assignment]
        torch.optim.AdamW.__init__ = self._reject_optimizer  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        torch.Tensor.backward = self._original_tensor_backward  # type: ignore[method-assign]
        torch.autograd.backward = self._original_autograd_backward  # type: ignore[assignment]
        torch.optim.AdamW.__init__ = self._original_adamw_init  # type: ignore[method-assign]

    def optimizer_step(self) -> None:
        self.rejection_counts["optimizer_step"] += 1
        raise ProtocolContractError("optimizer.step is forbidden during readiness")

    def select_semantic_checkpoint(self) -> None:
        self.rejection_counts["semantic_checkpoint_selection"] += 1
        raise ProtocolContractError("performance-based semantic checkpoint selection is forbidden during readiness")

    def select_branch_from_metrics(self) -> None:
        self.rejection_counts["branch_selection"] += 1
        raise ProtocolContractError("branch selection from readiness metrics is forbidden")

    def apply_d3(self) -> None:
        self.rejection_counts["d3_application"] += 1
        raise ProtocolContractError("D3 application is forbidden during readiness")


def hard_two_expert_set_loss(
    outputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Target-dependent hard identity/swap loss used only for no-grad traversal."""

    if outputs.ndim != 5 or outputs.shape[1:3] != (2, 6):
        raise ProtocolContractError("outputs must have shape (N,2,6,H,W)")
    if targets.shape != outputs.shape:
        raise ProtocolContractError("targets must match outputs")
    cost = lambda left, right: (left - right).square().mean(dim=(-3, -2, -1))
    identity = cost(outputs[:, 0], targets[:, 0]) + cost(outputs[:, 1], targets[:, 1])
    swapped = cost(outputs[:, 0], targets[:, 1]) + cost(outputs[:, 1], targets[:, 0])
    return torch.minimum(identity, swapped).mean()


__all__ = [
    "NoUpdateReadinessGuard",
    "ProtocolContractError",
    "RuntimeDecisionProducer",
    "hard_two_expert_set_loss",
    "validate_effective_protocol",
]
