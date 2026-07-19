"""Independent, noninterfering audit support for prospective THAYER-D3-PV1.

This module observes immutable representations, maintains a canonical JSONL hash
chain, validates execution evidence, and independently replays protocol
decisions.  It does not import the runtime protocol decision implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import inspect
import json
import math
import os
import pickle
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.d3_tensor_hash_contract_r1 import CANONICAL_DTYPE, canonical_nchw_tensor_hash
from src.models_probabilistic_unet import ConvBlock
from src.output_parameterization import apply_output_mapping, initial_raw_bias


AUDIT_SCHEMA_VERSION = "thayer-d3-pv1-audit-event-v1"
ZERO_HASH = "0" * 64
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_EVENT_TYPES = frozenset(
    {
        "HUMAN_AUTHORITY_FROZEN",
        "HUMAN_AUTHORITY_AMENDMENT_FROZEN",
        "BASELINE_VERIFIED",
        "LEGACY_CHECKPOINT_INVENTORIED",
        "LEGACY_CHECKPOINT_EXCLUDED",
        "FRESH_INITIALIZATION_STARTED",
        "FRESH_INITIALIZATION_COMPLETED",
        "FRESH_INITIALIZATION_REPRODUCED",
        "INITIAL_STATE_FROZEN",
        "PRIMARY_L0_EVIDENCE_BOUND",
        "DUPLICATE_PRIMARY_L0_REJECTED",
        "SOURCE_FREEZE_CREATED",
        "PROTOCOL_BUNDLE_FROZEN",
        "CACHE_GENERATION_STARTED",
        "CACHE_GENERATION_COMPLETED",
        "CACHE_REPRODUCIBILITY_VERIFIED",
        "SCENE_BUNDLE_FROZEN",
        "TARGET_BUNDLE_FROZEN",
        "INPUT_LOADED",
        "CAPACITY_CONSTRUCTED",
        "WORKER_LAUNCHED",
        "MODEL_FORWARD_COMPLETED",
        "TARGET_DEPENDENT_LOSS_COMPUTED",
        "BACKWARD_COMPLETED",
        "OPTIMIZER_STEP_COMPLETED",
        "EVALUATION_COMPLETED",
        "SEMANTIC_CHECKPOINT_PERSISTED",
        "SEMANTIC_CHECKPOINT_REPLAYED",
        "BRANCH_SELECTED",
        "D3_DECISION_COMPUTED",
        "GATE_COMPLETED",
        "INTEGRITY_CHECK_COMPLETED",
        "CAMPAIGN_TERMINATED",
    }
)
READINESS_FORBIDDEN_EVENTS = frozenset(
    {"BACKWARD_COMPLETED", "OPTIMIZER_STEP_COMPLETED", "D3_DECISION_COMPUTED"}
)


class AuditContractError(RuntimeError):
    """Raised when immutable audit evidence violates the fail-closed contract."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the single explicit canonical JSON representation used by the journal."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AuditContractError(f"value is outside canonical JSON: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_candidate_path(
    execution_root: Path | str,
    candidate_path: Path | str,
    *,
    for_write: bool,
    repository_root: Path | str | None = None,
) -> Path:
    """Resolve a path and reject traversal, symlink, and repository-root escapes."""

    root = Path(execution_root).resolve(strict=True)
    supplied = Path(candidate_path)
    if supplied.is_absolute():
        target = supplied.resolve(strict=False)
    else:
        if ".." in supplied.parts:
            raise AuditContractError("path escapes candidate root through traversal")
        target = (root / supplied).resolve(strict=False)
    if not _is_relative_to(target, root) or target == root:
        if repository_root is not None:
            repository = Path(repository_root).resolve(strict=True)
            if _is_relative_to(target, repository):
                raise AuditContractError("repository-root write is outside candidate root")
        raise AuditContractError("path escapes candidate root")
    if for_write:
        parent = target.parent
        while parent != root and not parent.exists():
            parent = parent.parent
        if parent.exists() and not _is_relative_to(parent.resolve(strict=True), root):
            raise AuditContractError("symlink escapes candidate root")
    elif not target.is_file():
        raise AuditContractError("candidate-root read target does not exist")
    return target


def _canonical_event_hash(event_without_current_hash: Mapping[str, object]) -> str:
    return _sha256_bytes(canonical_json_bytes(event_without_current_hash))


@dataclass
class AuditJournal:
    """One exclusively created append-only candidate-scoped audit journal."""

    execution_root: Path
    path: Path
    protocol_identifier: str
    campaign_identifier: str
    candidate_identifier: str | None
    _sequence_number: int = 0
    _chain_head: str = ZERO_HASH

    @classmethod
    def create(
        cls,
        *,
        execution_root: Path | str,
        protocol_identifier: str,
        campaign_identifier: str,
        candidate_identifier: str | None = None,
        journal_relative_path: str = "audit/events.jsonl",
    ) -> "AuditJournal":
        root = Path(execution_root).resolve(strict=True)
        if not protocol_identifier or not campaign_identifier:
            raise AuditContractError("protocol and campaign identifiers are required")
        path = resolve_candidate_path(root, journal_relative_path, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = path.parent.resolve(strict=True)
        if not _is_relative_to(resolved_parent, root):
            raise AuditContractError("journal parent escapes candidate root")
        with path.open("x", encoding="utf-8"):
            pass
        return cls(
            execution_root=root,
            path=path,
            protocol_identifier=protocol_identifier,
            campaign_identifier=campaign_identifier,
            candidate_identifier=candidate_identifier,
        )

    def append(self, event_type: str, stage: str, payload: Mapping[str, object]) -> dict[str, object]:
        if event_type not in REQUIRED_EVENT_TYPES:
            raise AuditContractError(f"unsupported audit event type: {event_type}")
        if not isinstance(stage, str) or not stage:
            raise AuditContractError("audit stage must be a nonempty string")
        if not isinstance(payload, Mapping):
            raise AuditContractError("audit payload must be a mapping")
        if not self.path.is_file() or self.path.is_symlink():
            raise AuditContractError("audit journal is absent or replaced by a symlink")
        if self._sequence_number:
            verified = verify_event_chain(self.path)
            if verified["chain_head_sha256"] != self._chain_head:
                raise AuditContractError("audit journal changed before append")
        frozen_payload = json.loads(canonical_json_bytes(dict(payload)))
        payload_sha256 = _sha256_bytes(canonical_json_bytes(frozen_payload))
        event_without_hash: dict[str, object] = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "protocol_identifier": self.protocol_identifier,
            "campaign_identifier": self.campaign_identifier,
            "candidate_identifier": self.candidate_identifier,
            "sequence_number": self._sequence_number + 1,
            "event_type": event_type,
            "stage": stage,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "payload": frozen_payload,
            "canonical_payload_sha256": payload_sha256,
            "previous_event_sha256": self._chain_head,
        }
        event = {
            **event_without_hash,
            "current_event_sha256": _canonical_event_hash(event_without_hash),
        }
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        try:
            os.write(descriptor, canonical_json_bytes(event) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sequence_number += 1
        self._chain_head = str(event["current_event_sha256"])
        return event


def verify_event_chain(path: Path | str) -> dict[str, object]:
    """Replay and validate every byte and link of a canonical audit journal."""

    journal = Path(path)
    if not journal.is_file() or journal.is_symlink():
        raise AuditContractError("event journal is absent or is a symlink")
    raw_lines = journal.read_bytes().splitlines(keepends=True)
    previous = ZERO_HASH
    seen_sequences: set[int] = set()
    event_types: list[str] = []
    for expected_sequence, raw_line in enumerate(raw_lines, 1):
        if not raw_line.endswith(b"\n"):
            raise AuditContractError("event journal line is not newline terminated")
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise AuditContractError(f"invalid event JSON: {error}") from error
        if canonical_json_bytes(event) + b"\n" != raw_line:
            raise AuditContractError("event is not canonically serialized")
        sequence = event.get("sequence_number")
        if not isinstance(sequence, int) or sequence in seen_sequences:
            raise AuditContractError("duplicate or invalid sequence number")
        seen_sequences.add(sequence)
        if sequence != expected_sequence:
            raise AuditContractError("event sequence is deleted, reordered, or nonmonotonic")
        if event.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
            raise AuditContractError("unsupported audit event schema")
        if event.get("event_type") not in REQUIRED_EVENT_TYPES:
            raise AuditContractError("unexpected audit event type")
        if event.get("previous_event_sha256") != previous:
            raise AuditContractError("broken previous-event hash chain")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise AuditContractError("event payload must be an object")
        if event.get("canonical_payload_sha256") != _sha256_bytes(canonical_json_bytes(payload)):
            raise AuditContractError("canonical payload hash mismatch")
        current = event.get("current_event_sha256")
        if not isinstance(current, str) or not SHA256_PATTERN.fullmatch(current):
            raise AuditContractError("invalid current-event hash")
        without_hash = {key: value for key, value in event.items() if key != "current_event_sha256"}
        if current != _canonical_event_hash(without_hash):
            raise AuditContractError("current-event hash mismatch")
        previous = current
        event_types.append(str(event["event_type"]))
    return {
        "valid": True,
        "event_count": len(raw_lines),
        "chain_head_sha256": previous,
        "event_types": event_types,
        "journal_file_sha256": _sha256_file(journal),
    }


def _load_events(path: Path | str) -> list[dict[str, object]]:
    verify_event_chain(path)
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def validate_execution_events(
    path: Path | str,
    *,
    required_event_types: Sequence[str] = (),
    readiness_mode: bool,
    maximum_optimizer_steps: int | None = None,
) -> dict[str, object]:
    """Apply ordering, no-retry, cadence, and step-count contracts to a journal."""

    events = _load_events(path)
    types = [str(event["event_type"]) for event in events]
    positions = []
    for required in required_event_types:
        if required not in REQUIRED_EVENT_TYPES or required not in types:
            raise AuditContractError(f"missing required event: {required}")
        positions.append(types.index(required))
    if positions != sorted(positions):
        raise AuditContractError("required events are out of order")
    if readiness_mode:
        unexpected = READINESS_FORBIDDEN_EVENTS.intersection(types)
        if unexpected:
            raise AuditContractError(f"unexpected optimizer/backward/D3 event in readiness: {sorted(unexpected)}")

    optimizer_steps: list[int] = []
    evaluation_steps: list[int] = []
    for event in events:
        stage = str(event["stage"]).lower()
        payload = event["payload"]
        if "retry" in stage or payload.get("retry") is True or (
            isinstance(payload.get("attempt"), int) and payload["attempt"] > 1
        ):
            raise AuditContractError("forbidden retry observed")
        if event["event_type"] == "OPTIMIZER_STEP_COMPLETED":
            step = payload.get("step")
            if not isinstance(step, int) or step <= 0:
                raise AuditContractError("invalid optimizer step")
            optimizer_steps.append(step)
        if event["event_type"] == "EVALUATION_COMPLETED":
            step = payload.get("step")
            if not isinstance(step, int) or step < 0:
                raise AuditContractError("invalid evaluation step")
            evaluation_steps.append(step)
    if optimizer_steps != sorted(set(optimizer_steps)):
        raise AuditContractError("optimizer steps are duplicate or nonmonotonic")
    if maximum_optimizer_steps is not None and (
        len(optimizer_steps) > maximum_optimizer_steps
        or any(step > maximum_optimizer_steps for step in optimizer_steps)
    ):
        raise AuditContractError("extra optimizer step exceeds fixed budget")
    if evaluation_steps != sorted(evaluation_steps):
        raise AuditContractError("non-monotonic evaluation step")
    return {
        "valid": True,
        "event_count": len(events),
        "optimizer_step_count": len(optimizer_steps),
        "evaluation_steps": evaluation_steps,
    }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous()
    array = value.numpy()
    return array.tobytes(order="C")


def observe_tensor(
    tensor: torch.Tensor,
    *,
    logical_axis_layout: str,
    batch_order: Iterable[str] | None,
) -> dict[str, object]:
    """Observe a detached tensor without changing it or any RNG state."""

    if not isinstance(tensor, torch.Tensor):
        raise AuditContractError("tensor observation requires torch.Tensor")
    order = None if batch_order is None else list(batch_order)
    device = str(tensor.device)
    shape = list(tensor.shape)
    dtype = str(tensor.dtype)
    raw_digest = hashlib.sha256()
    metadata = {
        "observation_schema": "thayer-d3-pv1-detached-tensor-observation-v1",
        "shape": shape,
        "dtype": dtype,
        "logical_axis_layout": logical_axis_layout,
        "batch_order": order,
        "memory_order": "C",
    }
    raw_digest.update(canonical_json_bytes(metadata))
    raw_digest.update(b"\0")
    raw_digest.update(_tensor_bytes(tensor))
    result: dict[str, object] = {
        **metadata,
        "device_at_observation": device,
        "canonical_contiguous_content_sha256": raw_digest.hexdigest(),
    }
    if logical_axis_layout == "NCHW":
        if order is None:
            raise AuditContractError("NCHW observation requires batch order")
        record = canonical_nchw_tensor_hash(
            tensor,
            semantic_axis_order="NCHW",
            semantic_member_name="audit.observed_tensor",
            prompt_identity="ordered_batch",
            expert_identity="not_applicable",
            band_order=None,
            canonical_dtype=CANONICAL_DTYPE,
            ordered_sample_ids=order,
        )
        result["validated_tensor_hash_schema"] = record.header["schema_version"]
        result["validated_canonical_tensor_content_sha256"] = record.canonical_semantic_tensor_sha256
        result["canonical_contiguous_content_sha256"] = record.canonical_semantic_tensor_sha256
    return result


def _observe_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256_bytes(_tensor_bytes(value)),
        }
    if isinstance(value, Mapping):
        return {str(key): _observe_tree(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_observe_tree(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def observe_scientific_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    inputs: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
) -> dict[str, object]:
    """Hash detached model/optimizer/tensor/gradient state without interference."""

    model_tree = _observe_tree(model.state_dict())
    optimizer_tree = _observe_tree(optimizer.state_dict())
    gradients = {
        name: _observe_tree(parameter.grad) if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    return {
        "model_state_sha256": _sha256_bytes(canonical_json_bytes(model_tree)),
        "optimizer_state_sha256": _sha256_bytes(canonical_json_bytes(optimizer_tree)),
        "gradient_state_sha256": _sha256_bytes(canonical_json_bytes(gradients)),
        "inputs": [_observe_tree(value) for value in inputs],
        "targets": [_observe_tree(value) for value in targets],
    }


LEGACY_LEARNED_CHECKPOINT_SHA256 = "8b06e788853a9180df7f83803d25cab17e362aac602c2932efe8dee680fa591e"
A1_EXPERT_SEEDS = {
    "L0": (2026071201, 2026071202),
    "L1": (2026072201, 2026072202),
    "L2": (2026073201, 2026073202),
    "L3": (2026074201, 2026074202),
}
A1_CAPACITIES = {
    "L0": (32, 16),
    "L1": (80, 40),
    "L2": (160, 80),
    "L3": (224, 112),
}
A1_REPLICA_OFFSETS = (0, 10000, 20000)
_FORBIDDEN_LEGACY_USES = {
    "MODEL_STATE_LOAD": "model state load from protected legacy checkpoint is forbidden",
    "OPTIMIZER_STATE_LOAD": "optimizer state load from protected legacy checkpoint is forbidden",
    "BRANCH_SELECTION_EVIDENCE": "branch-selection evidence from protected legacy checkpoint is forbidden",
    "CAPACITY_EVIDENCE": "capacity evidence from protected legacy checkpoint is forbidden",
    "CHECKPOINT_TRANSFER": "checkpoint transfer from protected legacy checkpoint is forbidden",
}


def classify_checkpoint_sha256(checkpoint_sha256: str) -> str:
    """Classify the one A1-protected learned checkpoint by exact content hash."""

    _require_sha256(checkpoint_sha256, "checkpoint")
    if checkpoint_sha256 == LEGACY_LEARNED_CHECKPOINT_SHA256:
        return "PROTECTED_LEGACY_LEARNED_CHECKPOINT_ONLY"
    return "NOT_THE_A1_PROTECTED_LEGACY_CHECKPOINT"


def hash_protected_file_for_integrity(
    path: Path | str, *, expected_sha256: str = LEGACY_LEARNED_CHECKPOINT_SHA256
) -> dict[str, object]:
    """Byte-hash a protected file without invoking a tensor/checkpoint loader."""

    _require_sha256(expected_sha256, "expected protected-file")
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise AuditContractError("protected integrity target is absent or is a symlink")
    digest = _sha256_file(target)
    if digest != expected_sha256:
        raise AuditContractError("protected file integrity hash mismatch")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": digest,
        "classification": classify_checkpoint_sha256(digest),
        "read_mode": "BYTE_LEVEL_INTEGRITY_ONLY",
        "torch_load_calls": 0,
        "model_state_loads": 0,
        "optimizer_state_loads": 0,
    }


def reject_protected_checkpoint_use(checkpoint_sha256: str, purpose: str) -> None:
    """Fail closed on every scientific use of the protected legacy hash."""

    _require_sha256(checkpoint_sha256, "checkpoint")
    if checkpoint_sha256 == LEGACY_LEARNED_CHECKPOINT_SHA256 and purpose in _FORBIDDEN_LEGACY_USES:
        raise AuditContractError(_FORBIDDEN_LEGACY_USES[purpose])


class A1CapacityExpertDecoder(nn.Module):
    """Scratch-constructed square-mapped decoder at one approved capacity."""

    def __init__(self, dec2_width: int, dec1_width: int) -> None:
        super().__init__()
        self.dec2_width = dec2_width
        self.dec1_width = dec1_width
        self.dec2 = ConvBlock(96, dec2_width)
        self.dec1 = ConvBlock(dec2_width + 16, dec1_width)
        self.decomposition_head = nn.Conv2d(dec1_width, 6, 1)
        with torch.no_grad():
            self.decomposition_head.weight.zero_()
            self.decomposition_head.bias.fill_(initial_raw_bias("square"))

    def raw_forward(
        self, enc1: torch.Tensor, enc2: torch.Tensor, bottleneck: torch.Tensor
    ) -> torch.Tensor:
        up2 = F.interpolate(bottleneck, size=(30, 30), mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=(60, 60), mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.decomposition_head(dec1)

    def forward(
        self, enc1: torch.Tensor, enc2: torch.Tensor, bottleneck: torch.Tensor
    ) -> torch.Tensor:
        return apply_output_mapping(self.raw_forward(enc1, enc2, bottleneck), "square")


class A1FreshDecoderPair(nn.Module):
    """Two independently seeded experts consuming frozen cached features."""

    architecture_identifier = "THAYER_CACHED_FEATURE_TWO_EXPERT_SQUARE_DECODER_A1_V1"

    def __init__(self, capacity_identifier: str, expert_seeds: Sequence[int]) -> None:
        super().__init__()
        if capacity_identifier not in A1_CAPACITIES:
            raise AuditContractError("unknown A1 capacity identifier")
        if len(expert_seeds) != 2 or any(not isinstance(seed, int) for seed in expert_seeds):
            raise AuditContractError("exactly two integer expert seeds are required")
        dec2, dec1 = A1_CAPACITIES[capacity_identifier]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(expert_seeds[0]))
            self.expert_1 = A1CapacityExpertDecoder(dec2, dec1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(expert_seeds[1]))
            self.expert_2 = A1CapacityExpertDecoder(dec2, dec1)
        self.capacity_identifier = capacity_identifier
        self.expert_seeds = tuple(int(seed) for seed in expert_seeds)

    def forward_features(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return torch.stack((self.expert_1(*features), self.expert_2(*features)), dim=1)


@dataclass(frozen=True)
class FreshInitialState:
    model: A1FreshDecoderPair
    manifest: dict[str, object]


def _rng_manifest() -> dict[str, object]:
    python_raw = pickle.dumps(__import__("random").getstate(), protocol=5)
    numpy_raw = pickle.dumps(np.random.get_state(), protocol=5)
    torch_cpu_raw = torch.random.get_rng_state().cpu().numpy().tobytes()
    accelerator: dict[str, object]
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state = torch.mps.get_rng_state()
        raw = state.cpu().numpy().tobytes()
        accelerator = {"available": True, "backend": "mps", "sha256": _sha256_bytes(raw), "base64": base64.b64encode(raw).decode("ascii")}
    elif torch.cuda.is_available():
        states = [state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()]
        accelerator = {
            "available": True, "backend": "cuda",
            "sha256": _sha256_bytes(b"".join(states)),
            "device_state_base64": [base64.b64encode(raw).decode("ascii") for raw in states],
        }
    else:
        accelerator = {"available": False, "backend": None, "sha256": None}
    return {
        "python": {"sha256": _sha256_bytes(python_raw), "base64": base64.b64encode(python_raw).decode("ascii")},
        "numpy": {"sha256": _sha256_bytes(numpy_raw), "base64": base64.b64encode(numpy_raw).decode("ascii")},
        "torch_cpu": {"sha256": _sha256_bytes(torch_cpu_raw), "base64": base64.b64encode(torch_cpu_raw).decode("ascii")},
        "accelerator": accelerator,
    }


def _state_tensor_record(name: str, value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach().to(device="cpu").contiguous()
    metadata = {
        "schema_version": "thayer-d3-canonical-state-tensor-hash-v1",
        "state_dict_key": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "byte_order": "native-pytorch-cpu",
        "memory_order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(_tensor_bytes(tensor))
    return {**metadata, "canonical_tensor_content_sha256": digest.hexdigest()}


def _canonical_initial_state_manifest(
    model: A1FreshDecoderPair,
    *,
    capacity_identifier: str,
    expert_seeds: tuple[int, int],
    rng_before: Mapping[str, object],
    rng_after: Mapping[str, object],
) -> dict[str, object]:
    tensor_records = [
        _state_tensor_record(name, value)
        for name, value in sorted(model.state_dict().items())
    ]
    constructor_source = "\n".join(
        (inspect.getsource(A1CapacityExpertDecoder), inspect.getsource(A1FreshDecoderPair))
    ).encode("utf-8")
    dec2, dec1 = A1_CAPACITIES[capacity_identifier]
    config = {
        "architecture_identifier": model.architecture_identifier,
        "capacity_identifier": capacity_identifier,
        "dec2_width": dec2,
        "dec1_width": dec1,
        "expert_seeds": list(expert_seeds),
        "output_mapping": "square",
        "initialization_device_policy": "CPU_DETERMINISTIC_CONSTRUCTION",
    }
    state_core = {
        **config,
        "sorted_state_dict_keys": [row["state_dict_key"] for row in tensor_records],
        "state_dict_tensors": tensor_records,
        "constructor_source_sha256": _sha256_bytes(constructor_source),
        "initialization_configuration_sha256": _sha256_bytes(canonical_json_bytes(config)),
    }
    state_hash = _sha256_bytes(canonical_json_bytes(state_core))
    return {
        "schema_version": "thayer-d3-pv1-a1-fresh-initial-state-manifest-v1",
        "protocol_identifier": "THAYER-D3-PV1-A1",
        **state_core,
        "combined_canonical_state_manifest_sha256": state_hash,
        "rng_states_before_construction": dict(rng_before),
        "rng_states_after_construction": dict(rng_after),
        "rng_state_noninterference": canonical_json_bytes(rng_before) == canonical_json_bytes(rng_after),
        "optimizer_step_counter": 0,
        "metric_history_count": 0,
        "semantic_history_count": 0,
        "optimizer_state_entry_count_before_first_step": 0,
        "transferred_model_tensors": 0,
        "transferred_optimizer_tensors": 0,
        "transferred_counters": 0,
        "transferred_scientific_evidence": 0,
        "legacy_checkpoint_sha256": LEGACY_LEARNED_CHECKPOINT_SHA256,
        "legacy_checkpoint_loaded": False,
    }


def construct_fresh_initial_state(
    capacity_identifier: str,
    expert_seeds: Sequence[int] | None = None,
) -> FreshInitialState:
    """Construct one approved scratch state without loading any checkpoint."""

    if capacity_identifier not in A1_CAPACITIES:
        raise AuditContractError("unknown A1 capacity identifier")
    seeds = tuple(A1_EXPERT_SEEDS[capacity_identifier] if expert_seeds is None else expert_seeds)
    if len(seeds) != 2 or any(not isinstance(seed, int) for seed in seeds):
        raise AuditContractError("fresh construction requires exactly two integer seeds")
    rng_before = _rng_manifest()
    model = A1FreshDecoderPair(capacity_identifier, seeds)
    rng_after = _rng_manifest()
    manifest = _canonical_initial_state_manifest(
        model,
        capacity_identifier=capacity_identifier,
        expert_seeds=(int(seeds[0]), int(seeds[1])),
        rng_before=rng_before,
        rng_after=rng_after,
    )
    if not manifest["rng_state_noninterference"]:
        raise AuditContractError("fresh construction interfered with caller RNG state")
    return FreshInitialState(model=model, manifest=manifest)


def create_fresh_a1_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Create the frozen AdamW configuration with a provably empty pre-step state."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    if optimizer.state:
        raise AuditContractError("new optimizer state is not empty")
    return optimizer


def _clean_process_initialization(capacity_identifier: str, expert_seeds: tuple[int, int]) -> dict[str, object]:
    code = (
        "import json; from src.d3_audit_layer_pv1 import construct_fresh_initial_state; "
        f"x=construct_fresh_initial_state({capacity_identifier!r},{expert_seeds!r}); "
        "print(json.dumps({'state_sha256':x.manifest['combined_canonical_state_manifest_sha256'],"
        "'parameter_count':sum(v.numel() for v in x.model.state_dict().values()),"
        "'tensor_count':len(x.model.state_dict())},sort_keys=True))"
    )
    completed = subprocess.run(
        (sys.executable, "-c", code),
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AuditContractError(f"clean-process fresh initialization failed: {completed.stderr.strip()}")
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as error:
        raise AuditContractError("clean-process initialization returned malformed evidence") from error
    return {**result, "process_exit_code": completed.returncode}


def reproduce_initial_state_clean_processes(
    capacity_identifier: str,
    expert_seeds: Sequence[int] | None = None,
    *,
    process_count: int = 2,
) -> dict[str, object]:
    if process_count != 2:
        raise AuditContractError("A1 reproducibility requires exactly two clean processes")
    seeds = tuple(A1_EXPERT_SEEDS[capacity_identifier] if expert_seeds is None else expert_seeds)
    if len(seeds) != 2:
        raise AuditContractError("fresh construction requires two seeds")
    constructions = [
        _clean_process_initialization(capacity_identifier, (int(seeds[0]), int(seeds[1])))
        for _ in range(process_count)
    ]
    hashes = [row["state_sha256"] for row in constructions]
    matched = len(set(hashes)) == 1
    return {
        "protocol_identifier": "THAYER-D3-PV1-A1",
        "capacity_identifier": capacity_identifier,
        "expert_seeds": list(seeds),
        "clean_process_count": process_count,
        "constructions": constructions,
        "canonical_state_hashes_match": matched,
        "canonical_state_sha256": hashes[0] if matched else None,
        "status": "PASS" if matched else "FRESH_INITIALIZATION_REPRODUCIBILITY_FAILURE",
    }


def freeze_initial_state_manifest(
    candidate_root: Path | str,
    relative_path: Path | str,
    manifest: Mapping[str, object],
) -> Path:
    root = Path(candidate_root).resolve(strict=True)
    target = resolve_candidate_path(root, relative_path, for_write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return target.resolve(strict=True)


@dataclass
class PrimaryL0EvidenceBinding:
    """Immutable identity binding for the sole branch/primary-capacity L0 run."""

    branch_run_id: str | None = None
    branch_evidence_id: str | None = None
    primary_capacity_run_id: str | None = None
    primary_capacity_evidence_id: str | None = None
    eight_scene_run_id: str | None = None
    eight_scene_initialization_id: str | None = None

    def bind_branch_selection(self, run_id: str, evidence_id: str) -> None:
        if self.branch_run_id is not None:
            raise AuditContractError("duplicate branch-selection L0 run")
        if not run_id or not evidence_id:
            raise AuditContractError("L0 run and evidence identities are required")
        self.branch_run_id = run_id
        self.branch_evidence_id = evidence_id

    def bind_primary_capacity(self, run_id: str, evidence_id: str) -> None:
        if self.primary_capacity_run_id is not None:
            raise AuditContractError("duplicate primary L0 capacity run rejected")
        if run_id != self.branch_run_id or evidence_id != self.branch_evidence_id:
            raise AuditContractError("branch L0 and primary capacity L0 evidence identity mismatch")
        self.primary_capacity_run_id = run_id
        self.primary_capacity_evidence_id = evidence_id

    def bind_eight_scene(self, run_id: str, initialization_id: str) -> None:
        if run_id == self.branch_run_id or initialization_id == self.branch_evidence_id:
            raise AuditContractError("eight-scene branch requires a separate fresh initialization identity")
        self.eight_scene_run_id = run_id
        self.eight_scene_initialization_id = initialization_id

    def verify(self) -> dict[str, object]:
        identity = (
            self.branch_run_id is not None
            and self.branch_run_id == self.primary_capacity_run_id
            and self.branch_evidence_id == self.primary_capacity_evidence_id
        )
        eight_separate = (
            self.eight_scene_run_id is not None
            and self.eight_scene_initialization_id is not None
            and self.eight_scene_run_id != self.branch_run_id
            and self.eight_scene_initialization_id != self.branch_evidence_id
        )
        return {
            "identity_exact": identity,
            "duplicate_primary_l0_run_count": 0,
            "eight_scene_separate": eight_separate,
        }


def effective_a1_protocol() -> dict[str, object]:
    """Return the strict A1 initialization/exclusion protocol fragment."""

    return {
        "protocol_identifier": "THAYER-D3-PV1-A1",
        "parent_protocol_identifier": "THAYER-D3-PV1",
        "amendment_identifier": "A1",
        "initialization": {
            "mode": "FRESH_SEEDED_STEP_ZERO",
            "l0_expert_seeds": list(A1_EXPERT_SEEDS["L0"]),
            "optimizer_step_counter": 0,
            "optimizer_state_before_first_step": "EMPTY",
            "metric_history": "EMPTY",
            "semantic_checkpoint_history": "EMPTY",
            "transfers": "FORBIDDEN",
        },
        "legacy_checkpoint": {
            "sha256": LEGACY_LEARNED_CHECKPOINT_SHA256,
            "classification": "PROTECTED_LEGACY_LEARNED_CHECKPOINT_ONLY",
            "integrity_hashing": "ALLOWED",
            "scientific_use": "FORBIDDEN",
        },
        "primary_l0": {
            "branch_selection_run_count": 1,
            "capacity_ladder_reuse": "REQUIRED_WHEN_SELECTED",
            "duplicate_primary_l0": "FORBIDDEN",
        },
    }


def validate_a1_initialization_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    """Independently validate the canonical fresh-state and zero-transfer record."""

    if manifest.get("schema_version") != "thayer-d3-pv1-a1-fresh-initial-state-manifest-v1":
        raise AuditContractError("wrong fresh-state manifest schema")
    if manifest.get("protocol_identifier") != "THAYER-D3-PV1-A1":
        raise AuditContractError("wrong fresh-state protocol identifier")
    capacity = manifest.get("capacity_identifier")
    if capacity not in A1_CAPACITIES:
        raise AuditContractError("wrong fresh-state capacity")
    expected_seeds = A1_EXPERT_SEEDS[str(capacity)]
    if manifest.get("expert_seeds") != list(expected_seeds):
        raise AuditContractError("incorrect fresh seed")
    dec2, dec1 = A1_CAPACITIES[str(capacity)]
    if manifest.get("dec2_width") != dec2 or manifest.get("dec1_width") != dec1:
        raise AuditContractError("fresh-state capacity width mismatch")
    if manifest.get("optimizer_step_counter") != 0:
        raise AuditContractError("nonzero initial step counter")
    if manifest.get("optimizer_state_entry_count_before_first_step") != 0:
        raise AuditContractError("nonempty initial optimizer state")
    if manifest.get("metric_history_count") != 0:
        raise AuditContractError("inherited metric history")
    if manifest.get("semantic_history_count") != 0:
        raise AuditContractError("inherited semantic history")
    for key, label in (
        ("transferred_model_tensors", "model tensor transfer"),
        ("transferred_optimizer_tensors", "optimizer tensor transfer"),
        ("transferred_counters", "counter transfer"),
        ("transferred_scientific_evidence", "scientific evidence transfer"),
    ):
        if manifest.get(key) != 0:
            raise AuditContractError(f"forbidden {label}")
    if manifest.get("legacy_checkpoint_sha256") != LEGACY_LEARNED_CHECKPOINT_SHA256:
        raise AuditContractError("wrong protected legacy checkpoint hash")
    if manifest.get("legacy_checkpoint_loaded") is not False:
        raise AuditContractError("legacy checkpoint loaded as model state")
    tensors = manifest.get("state_dict_tensors")
    keys = manifest.get("sorted_state_dict_keys")
    if not isinstance(tensors, list) or not isinstance(keys, list):
        raise AuditContractError("fresh-state tensor inventory missing")
    observed_keys = [row.get("state_dict_key") for row in tensors if isinstance(row, Mapping)]
    if observed_keys != sorted(observed_keys) or observed_keys != keys or len(observed_keys) != len(tensors):
        raise AuditContractError("fresh-state key inventory altered")
    for row in tensors:
        if not isinstance(row, Mapping):
            raise AuditContractError("fresh-state tensor record malformed")
        _require_sha256(row.get("canonical_tensor_content_sha256"), "fresh-state tensor")
    state_core_keys = (
        "architecture_identifier", "capacity_identifier", "dec2_width", "dec1_width",
        "expert_seeds", "output_mapping", "initialization_device_policy",
        "sorted_state_dict_keys", "state_dict_tensors", "constructor_source_sha256",
        "initialization_configuration_sha256",
    )
    state_core = {key: manifest.get(key) for key in state_core_keys}
    expected_state_hash = _sha256_bytes(canonical_json_bytes(state_core))
    if manifest.get("combined_canonical_state_manifest_sha256") != expected_state_hash:
        raise AuditContractError("altered fresh-state manifest")
    if manifest.get("rng_state_noninterference") is not True:
        raise AuditContractError("fresh initialization RNG interference")
    before = manifest.get("rng_states_before_construction")
    after = manifest.get("rng_states_after_construction")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or canonical_json_bytes(before) != canonical_json_bytes(after):
        raise AuditContractError("fresh initialization RNG states differ")
    return {
        "valid": True,
        "capacity_identifier": capacity,
        "tensor_count": len(tensors),
        "canonical_state_sha256": expected_state_hash,
        "optimizer_step_counter": 0,
        "optimizer_state_entry_count": 0,
    }


def validate_a1_event_semantics(
    path: Path | str,
    *,
    parent_approval_sha256: str,
    amendment_approval_sha256: str,
) -> dict[str, object]:
    """Replay A1-specific authority, exclusion, initialization, and identity events."""

    _require_sha256(parent_approval_sha256, "parent approval")
    _require_sha256(amendment_approval_sha256, "amendment approval")
    events = _load_events(path)
    required = (
        "HUMAN_AUTHORITY_FROZEN", "HUMAN_AUTHORITY_AMENDMENT_FROZEN",
        "LEGACY_CHECKPOINT_INVENTORIED", "LEGACY_CHECKPOINT_EXCLUDED",
        "FRESH_INITIALIZATION_STARTED", "FRESH_INITIALIZATION_COMPLETED",
        "FRESH_INITIALIZATION_REPRODUCED", "INITIAL_STATE_FROZEN",
        "PRIMARY_L0_EVIDENCE_BOUND",
    )
    types = [str(event["event_type"]) for event in events]
    positions = []
    for event_type in required:
        if event_type not in types:
            raise AuditContractError(f"missing amendment event: {event_type}")
        positions.append(types.index(event_type))
    if positions != sorted(positions):
        raise AuditContractError("A1 events are out of order")
    parent = next(event for event in events if event["event_type"] == "HUMAN_AUTHORITY_FROZEN")
    amendment = next(event for event in events if event["event_type"] == "HUMAN_AUTHORITY_AMENDMENT_FROZEN")
    if parent["payload"].get("approval_text_sha256") != parent_approval_sha256:
        raise AuditContractError("altered parent authority text")
    if amendment["payload"].get("approval_text_sha256") != amendment_approval_sha256:
        raise AuditContractError("altered amendment authority text")
    for event in events:
        payload = event["payload"]
        digest = payload.get("checkpoint_sha256") or payload.get("legacy_checkpoint_sha256")
        action = str(payload.get("action", ""))
        if digest == LEGACY_LEARNED_CHECKPOINT_SHA256 and action in _FORBIDDEN_LEGACY_USES:
            reject_protected_checkpoint_use(str(digest), action)
        if digest == LEGACY_LEARNED_CHECKPOINT_SHA256 and payload.get("scientific_load") is True:
            raise AuditContractError("legacy checkpoint loaded as scientific state")
    completed = next(event for event in events if event["event_type"] == "FRESH_INITIALIZATION_COMPLETED")
    payload = completed["payload"]
    if payload.get("expert_seeds") != list(A1_EXPERT_SEEDS["L0"]):
        raise AuditContractError("incorrect fresh seed")
    if payload.get("optimizer_step_counter") != 0:
        raise AuditContractError("nonzero initial step counter")
    if payload.get("optimizer_state_entry_count") != 0:
        raise AuditContractError("nonempty initial optimizer state")
    if payload.get("metric_history_count") != 0:
        raise AuditContractError("inherited metric history")
    if payload.get("semantic_history_count") != 0:
        raise AuditContractError("inherited semantic history")
    bound = next(event for event in events if event["event_type"] == "PRIMARY_L0_EVIDENCE_BOUND")["payload"]
    if bound.get("branch_selection_run_id") != bound.get("primary_capacity_run_id"):
        raise AuditContractError("branch L0/capacity L0 evidence mismatch")
    if bound.get("branch_selection_evidence_id") != bound.get("primary_capacity_evidence_id"):
        raise AuditContractError("branch L0/capacity L0 evidence mismatch")
    if bound.get("duplicate_primary_l0_run_count") != 0:
        raise AuditContractError("duplicate primary L0 run")
    return {
        "valid": True,
        "event_count": len(events),
        "legacy_scientific_load_count": 0,
        "primary_l0_identity_exact": True,
    }


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise AuditContractError(f"wrong {label} hash")


def validate_readiness_manifest(
    manifest: Mapping[str, object], protocol: Mapping[str, object]
) -> dict[str, object]:
    """Validate immutable identities, construction, transfer, and zero-update rules."""

    if manifest.get("protocol_identifier") != protocol.get("protocol_identifier"):
        raise AuditContractError("protocol identifier mismatch")
    for field, label in (
        ("authority_text_sha256", "authority"),
        ("protocol_bundle_sha256", "protocol bundle"),
        ("source_freeze_sha256", "source-freeze"),
    ):
        _require_sha256(manifest.get(field), label)
    scenes = manifest.get("scenes")
    expected_scenes = protocol.get("scene_order")
    if not isinstance(scenes, list) or not isinstance(expected_scenes, list) or len(scenes) != len(expected_scenes):
        raise AuditContractError("scene count mismatch")
    observed_ids = [row.get("scene_id") for row in scenes if isinstance(row, Mapping)]
    if len(observed_ids) != len(set(observed_ids)):
        raise AuditContractError("duplicate scene identity")
    if observed_ids != expected_scenes:
        raise AuditContractError("scene identity or order mismatch")
    for index, row in enumerate(scenes):
        if not isinstance(row, Mapping) or row.get("order_index") != index:
            raise AuditContractError("scene order index mismatch")
        _require_sha256(row.get("target_sha256"), "target")
        _require_sha256(row.get("cached_feature_sha256"), "cached-feature")
        if row.get("layout") != "NCHW":
            raise AuditContractError("wrong NCHW batch layout")
    if manifest.get("ordered_batch_scene_ids") != expected_scenes:
        raise AuditContractError("wrong NCHW batch order")

    capacities = manifest.get("capacities")
    expected_capacities = protocol.get("capacities")
    if not isinstance(capacities, list) or not isinstance(expected_capacities, list) or len(capacities) != len(expected_capacities):
        raise AuditContractError("capacity count mismatch")
    for observed, expected in zip(capacities, expected_capacities):
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            raise AuditContractError("capacity record malformed")
        for key in ("level", "index", "dec2", "dec1"):
            if observed.get(key) != expected.get(key):
                raise AuditContractError("wrong capacity width or level")
        if observed.get("seeds") != expected.get("seeds"):
            raise AuditContractError("wrong capacity seed")
    transfers = manifest.get("transfers")
    if not isinstance(transfers, Mapping):
        raise AuditContractError("transfer manifest missing")
    if transfers.get("decoder_tensor_transfer_attempts") != 0 or transfers.get("partial_state_dict_load_attempts") != 0:
        raise AuditContractError("forbidden checkpoint/tensor transfer")
    if transfers.get("optimizer_state_transfer_attempts") != 0:
        raise AuditContractError("forbidden optimizer-state transfer")
    if manifest.get("unexpected_retry_count") != 0:
        raise AuditContractError("unexpected retry")
    if manifest.get("scientific_backward_passes") != 0:
        raise AuditContractError("unexpected scientific backward pass")
    if manifest.get("scientific_optimizer_steps") != 0:
        raise AuditContractError("unexpected scientific optimizer step")
    return {"valid": True, "scene_count": len(scenes), "capacity_count": len(capacities)}


def _required_bool(evidence: Mapping[str, object], key: str) -> bool:
    value = evidence.get(key)
    if value is not True and value is not False:
        raise AuditContractError(f"branch evidence missing Boolean {key}")
    return bool(value)


class IndependentAuditReplayer:
    """Independent THAYER-D3-PV1 decision implementation for persisted evidence."""

    def __init__(self, protocol: Mapping[str, object]) -> None:
        self.protocol = json.loads(canonical_json_bytes(dict(protocol)))
        if self.protocol.get("protocol_identifier") not in {"THAYER-D3-PV1", "THAYER-D3-PV1-A1"}:
            raise AuditContractError("independent replay received wrong protocol")
        if self.protocol.get("tangent_comparison") != "STRICTLY_LESS_THAN":
            raise AuditContractError("independent replay requires strict tangent comparison")

    def select_branch(self, evidence: Mapping[str, object]) -> dict[str, str]:
        required_validity = (
            "valid",
            "l0_completed_validly",
            "source_integrity",
            "input_integrity",
            "metric_integrity",
            "scientific_contract_integrity",
        )
        if any(not _required_bool(evidence, key) for key in required_validity):
            return {"downstream_branch": "NONE", "d3_status": "UNKNOWN"}
        if _required_bool(evidence, "frozen_l0_success"):
            return {"downstream_branch": "EIGHT_SCENE", "d3_status": "PENDING"}
        capacity_validity = _required_bool(evidence, "d0_pass") and _required_bool(evidence, "d1_pass") and _required_bool(
            evidence, "other_validity_predicates_pass"
        )
        alternatives_excluded = not any(
            _required_bool(evidence, key)
            for key in (
                "optimization_mechanism_supported",
                "hard_assignment_mechanism_supported",
                "square_mapping_mechanism_supported",
            )
        )
        capture = evidence.get("validated_tangent_capture")
        low_capture = isinstance(capture, (int, float)) and math.isfinite(float(capture)) and float(capture) < float(
            self.protocol["tangent_threshold"]
        )
        if capacity_validity and alternatives_excluded and low_capture:
            return {"downstream_branch": "CAPACITY_LADDER", "d3_status": "PENDING"}
        return {"downstream_branch": "NONE", "d3_status": "UNKNOWN"}

    def classify_terminal(self, branch: str, evidence: Mapping[str, object]) -> str:
        if branch == "NONE":
            return "UNKNOWN"
        if evidence.get("valid") is not True:
            return "UNKNOWN"
        if branch == "EIGHT_SCENE":
            passes = evidence.get("scene_passes")
            required = int(self.protocol["eight_scene_required_pass_count"])
            if not isinstance(passes, list) or len(passes) != required or any(value is not True and value is not False for value in passes):
                return "UNKNOWN"
            return "PASS" if all(passes) else "FAIL"
        if branch != "CAPACITY_LADDER":
            raise AuditContractError(f"unknown branch for independent replay: {branch}")
        first = evidence.get("first_passing_level")
        results = evidence.get("level_results")
        if first is None:
            return "FAIL"
        levels = [row["level"] for row in self.protocol["capacities"]]
        if first not in levels or first == levels[0] or not isinstance(results, Mapping):
            return "UNKNOWN"
        smaller = levels[levels.index(first) - 1]
        candidate = results.get(first)
        boundary = results.get(smaller)
        if not isinstance(candidate, list) or not isinstance(boundary, list) or len(candidate) != 3 or len(boundary) != 3:
            return "UNKNOWN"
        if any(value is not True and value is not False for value in candidate + boundary):
            return "UNKNOWN"
        return "PASS" if all(candidate) and not any(boundary) else "FAIL"

    def verify_runtime_result(
        self, evidence: Mapping[str, object], runtime_result: Mapping[str, object]
    ) -> dict[str, str]:
        replay = self.select_branch(evidence)
        if dict(runtime_result) != replay:
            raise AuditContractError("runtime branch selection is inconsistent with independent evidence replay")
        return replay

    def verify_terminal_result(
        self, branch: str, evidence: Mapping[str, object], runtime_d3_status: str
    ) -> str:
        replay = self.classify_terminal(branch, evidence)
        if runtime_d3_status != replay:
            raise AuditContractError("runtime D3 result is inconsistent with independent evidence replay")
        return replay


__all__ = [
    "A1_CAPACITIES",
    "A1_EXPERT_SEEDS",
    "A1_REPLICA_OFFSETS",
    "A1CapacityExpertDecoder",
    "A1FreshDecoderPair",
    "AUDIT_SCHEMA_VERSION",
    "AuditContractError",
    "AuditJournal",
    "FreshInitialState",
    "IndependentAuditReplayer",
    "LEGACY_LEARNED_CHECKPOINT_SHA256",
    "PrimaryL0EvidenceBinding",
    "READINESS_FORBIDDEN_EVENTS",
    "REQUIRED_EVENT_TYPES",
    "canonical_json_bytes",
    "classify_checkpoint_sha256",
    "construct_fresh_initial_state",
    "create_fresh_a1_optimizer",
    "effective_a1_protocol",
    "freeze_initial_state_manifest",
    "hash_protected_file_for_integrity",
    "observe_scientific_state",
    "observe_tensor",
    "resolve_candidate_path",
    "reject_protected_checkpoint_use",
    "reproduce_initial_state_clean_processes",
    "validate_execution_events",
    "validate_a1_event_semantics",
    "validate_a1_initialization_manifest",
    "validate_readiness_manifest",
    "verify_event_chain",
]
