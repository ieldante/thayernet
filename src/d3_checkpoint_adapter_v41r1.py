"""Frozen-schema production checkpoint adapter for D3 v4.1 R1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import torch

from src.output_parameterization import MappedCompactExpertDecoder


SCHEMA_VERSION = "thayer-d3-checkpoint-v4"
FROZEN_MAP_LOCATION = "cpu"
FROZEN_WEIGHTS_ONLY = True
PRODUCTION_CHECKPOINT_KEYS = (
    "schema_version",
    "step",
    "expert_1_state_dict",
    "expert_2_state_dict",
    "optimizer_state_dict",
    "metrics",
    "physical",
    "penultimate_expert_1",
    "penultimate_expert_2",
    "bridge_sha256",
    "policy_engine_sha256",
)

_ORIGINAL_TORCH_SAVE = torch.save
_ORIGINAL_TORCH_LOAD = torch.load
_TRACE = {"writer_calls": 0, "reader_calls": 0, "routed_writer_calls": 0,
          "routed_reader_calls": 0}
_ROUTING_INSTALLED = False


def reset_adapter_trace() -> None:
    for key in _TRACE:
        _TRACE[key] = 0


def adapter_trace() -> dict[str, int]:
    return dict(_TRACE)


def _synthetic_expert(seed: int) -> MappedCompactExpertDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expert = MappedCompactExpertDecoder(mapping="square")
    with torch.no_grad():
        for parameter in expert.parameters():
            parameter.zero_()
    return expert


def build_synthetic_production_checkpoint_payload() -> dict[str, Any]:
    """Build deterministic non-scientific values with the full frozen schema."""

    expert_1 = _synthetic_expert(4101)
    expert_2 = _synthetic_expert(4102)
    parameters = list(expert_1.parameters()) + list(expert_2.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 1e-4)
    optimizer.step()
    metrics = {
        "evaluation_index": 0,
        "step": 0,
        "objective": 0.0,
        "assignment_prompt_a": "identity",
        "assignment_prompt_b": "identity",
        "selected_terminal_event": "CONTINUE",
        "optimizer_state_sha256": "0" * 64,
        "physical_sha256": "1" * 64,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": 0,
        "expert_1_state_dict": {
            name: value.detach().cpu()
            for name, value in expert_1.state_dict().items()
        },
        "expert_2_state_dict": {
            name: value.detach().cpu()
            for name, value in expert_2.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "physical": torch.zeros((2, 2, 6, 60, 60), dtype=torch.float32),
        "penultimate_expert_1": torch.zeros(
            (2, 16, 60, 60), dtype=torch.float32
        ),
        "penultimate_expert_2": torch.zeros(
            (2, 16, 60, 60), dtype=torch.float32
        ),
        "bridge_sha256": "2" * 64,
        "policy_engine_sha256": "3" * 64,
    }
    validate_production_checkpoint_payload(payload)
    return payload


def validate_production_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != set(PRODUCTION_CHECKPOINT_KEYS):
        raise ValueError("production checkpoint top-level key mismatch")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("production checkpoint schema version mismatch")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("production checkpoint step must be nonnegative int")
    for key in ("expert_1_state_dict", "expert_2_state_dict"):
        state = payload[key]
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"{key} must be a nonempty mapping")
        if not all(isinstance(name, str) and torch.is_tensor(value)
                   for name, value in state.items()):
            raise ValueError(f"{key} contains invalid members")
    optimizer = payload["optimizer_state_dict"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("optimizer state structure mismatch")
    if not optimizer["state"] or not optimizer["param_groups"]:
        raise ValueError("optimizer state must exercise state and param groups")
    if not isinstance(payload["metrics"], Mapping) or "objective" not in payload["metrics"]:
        raise ValueError("metrics structure mismatch")
    if tuple(payload["physical"].shape) != (2, 2, 6, 60, 60):
        raise ValueError("physical tensor shape mismatch")
    for key in ("penultimate_expert_1", "penultimate_expert_2"):
        if tuple(payload[key].shape) != (2, 16, 60, 60):
            raise ValueError(f"{key} shape mismatch")
    for key in ("bridge_sha256", "policy_engine_sha256"):
        value = payload[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{key} must be a SHA-256 token")


def write_production_checkpoint(
    path: Path | str | BinaryIO, payload: Mapping[str, Any]
) -> None:
    """Validate and write through the exact exclusive frozen serialization path."""

    validate_production_checkpoint_payload(payload)
    _TRACE["writer_calls"] += 1
    if hasattr(path, "write"):
        _ORIGINAL_TORCH_SAVE(dict(payload), path)
        return
    target = Path(path)
    with target.open("xb") as handle:
        _ORIGINAL_TORCH_SAVE(dict(payload), handle)


def read_production_checkpoint(
    path: Path | str | BinaryIO,
    *,
    map_location: str = FROZEN_MAP_LOCATION,
    weights_only: bool = FROZEN_WEIGHTS_ONLY,
) -> dict[str, Any]:
    """Read with frozen settings and validate the complete production schema."""

    if map_location != FROZEN_MAP_LOCATION or weights_only is not FROZEN_WEIGHTS_ONLY:
        raise ValueError("production checkpoint reader settings mismatch")
    _TRACE["reader_calls"] += 1
    payload = _ORIGINAL_TORCH_LOAD(
        path, map_location=map_location, weights_only=weights_only
    )
    validate_production_checkpoint_payload(payload)
    return payload


def synthetic_payload_manifest() -> dict[str, Any]:
    payload = build_synthetic_production_checkpoint_payload()
    return {
        "schema_version": SCHEMA_VERSION,
        "top_level_keys": list(PRODUCTION_CHECKPOINT_KEYS),
        "expert_1_state_keys": sorted(payload["expert_1_state_dict"]),
        "expert_2_state_keys": sorted(payload["expert_2_state_dict"]),
        "optimizer_state_member_count": len(payload["optimizer_state_dict"]["state"]),
        "map_location": FROZEN_MAP_LOCATION,
        "weights_only": FROZEN_WEIGHTS_ONLY,
        "scientific_checkpoint_opened": False,
        "scientific_model_tensor_used": False,
        "protected_identifier_count": 0,
    }


def _production_checkpoint_path(value: Any) -> bool:
    if hasattr(value, "name"):
        value = value.name
    try:
        path = Path(value)
    except TypeError:
        return False
    return path.name.startswith("evaluation_step_") or "serialization_prewarm" in path.parts


def routed_torch_save(obj: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(obj, Mapping) and obj.get("schema_version") == SCHEMA_VERSION:
        if args or kwargs:
            raise ValueError("production checkpoint writer received unfrozen options")
        _TRACE["routed_writer_calls"] += 1
        return write_production_checkpoint(destination, obj)
    return _ORIGINAL_TORCH_SAVE(obj, destination, *args, **kwargs)


def routed_torch_load(source: Any, *args: Any, **kwargs: Any) -> Any:
    if _production_checkpoint_path(source):
        if args:
            raise ValueError("production checkpoint reader requires keyword settings")
        _TRACE["routed_reader_calls"] += 1
        return read_production_checkpoint(source, **kwargs)
    return _ORIGINAL_TORCH_LOAD(source, *args, **kwargs)


def install_torch_checkpoint_routing() -> None:
    global _ROUTING_INSTALLED
    if _ROUTING_INSTALLED:
        return
    torch.save = routed_torch_save
    torch.load = routed_torch_load
    _ROUTING_INSTALLED = True


__all__ = [
    "FROZEN_MAP_LOCATION",
    "FROZEN_WEIGHTS_ONLY",
    "PRODUCTION_CHECKPOINT_KEYS",
    "SCHEMA_VERSION",
    "adapter_trace",
    "build_synthetic_production_checkpoint_payload",
    "install_torch_checkpoint_routing",
    "read_production_checkpoint",
    "reset_adapter_trace",
    "synthetic_payload_manifest",
    "validate_production_checkpoint_payload",
    "write_production_checkpoint",
]
