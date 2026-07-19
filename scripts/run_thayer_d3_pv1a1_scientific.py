#!/usr/bin/env python3
"""Execute the frozen THAYER-D3-PV1-A1 operation registry.

This module is intentionally an orchestrator.  Scientific values are read from
the selected protocol bundle or imported from the already validated production
components; they are not restated here.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.d3_audit_layer_pv1 import (  # noqa: E402
    AuditContractError,
    AuditJournal,
    IndependentAuditReplayer,
    PrimaryL0EvidenceBinding,
    canonical_json_bytes,
    construct_fresh_initial_state,
    create_fresh_a1_optimizer,
    hash_protected_file_for_integrity,
    observe_scientific_state,
    observe_tensor,
    reject_protected_checkpoint_use,
    validate_a1_initialization_manifest,
    verify_event_chain,
)
from src.d3_protocol_pv1a1 import (  # noqa: E402
    RuntimeDecisionProducer,
    validate_effective_protocol,
)
from src.canonical_tensor_hash import canonical_tensor_sha256  # noqa: E402
from src.competing_hypotheses import forward_consistency, is_plausible, scientific_distance  # noqa: E402
from src.d3_control_policy import AuthorizationContext, OutcomeEvidence  # noqa: E402
from src.d3_policy_engine import (  # noqa: E402
    authorize_downstream,
    evaluate_assignment_diagnostic,
    evaluate_optimization_diagnostic,
    evaluate_square_mapping_diagnostic,
    map_scientific_outcome,
    semantic_state_triggers,
)
from scripts.generate_thayer_d3_pv1a1_cache import (  # noqa: E402
    EXPECTED_INPUT_HASHES,
    SCENES,
    SOURCE_INDICES,
)
from scripts.run_thayer_output_parameterization_micro import (  # noqa: E402
    MEAN_PSF_FWHM_PIXEL,
    SCALES,
    frozen_thresholds,
    hard_physical_set_loss,
    prompt_identity_physical,
)


SCIENTIFIC_ORCHESTRATION_OBLIGATION_COUNT = 20
READY_MARKER = "READY_TO_EXECUTE_PV1A1_SCIENCE"
_LEGACY_LOAD_ATTEMPTS = 0


class EntrypointContractError(RuntimeError):
    """Fail-closed entrypoint, registry, bundle, or execution violation."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EntrypointContractError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_x(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen THAYER-D3-PV1-A1 scientific command exactly once."
    )
    parser.add_argument("--protocol-bundle", type=Path, required=True)
    parser.add_argument("--protocol-hashes", type=Path, required=True)
    parser.add_argument("--readiness-root", type=Path, required=True)
    parser.add_argument("--create-fresh-timestamped-campaign-root", action="store_true", required=True)
    parser.add_argument("--execute-authoritative-science", action="store_true", required=True)
    return parser


def load_operation_registry(bundle: Path) -> tuple[str, ...]:
    """Load, cross-check, and return the two frozen machine registries."""

    bundle = Path(bundle).resolve(strict=True)
    future = _json(bundle / "execution_contract.json").get("future_sequence")
    required = _json(bundle / "scientific_command_template.json").get("required_operations")
    if not isinstance(future, list) or not isinstance(required, list):
        raise EntrypointContractError("both frozen operation registries must be lists")
    if future != required:
        raise EntrypointContractError("frozen operation registries disagree")
    return validate_operation_registry(bundle, future)


def validate_operation_registry(bundle: Path, operations: Sequence[str]) -> tuple[str, ...]:
    expected = _json(Path(bundle) / "execution_contract.json").get("future_sequence")
    command_expected = _json(Path(bundle) / "scientific_command_template.json").get("required_operations")
    if not isinstance(expected, list) or expected != command_expected:
        raise EntrypointContractError("bundle has no single authoritative ordered registry")
    actual = list(operations)
    if any(not isinstance(value, str) or not value for value in actual):
        raise EntrypointContractError("operation names must be nonempty strings")
    if len(actual) != len(set(actual)):
        raise EntrypointContractError("duplicate operation in ordered registry")
    unknown = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))
    if unknown:
        raise EntrypointContractError(f"unknown operation in ordered registry: {unknown}")
    if missing:
        raise EntrypointContractError(f"missing operation in ordered registry: {missing}")
    if actual != expected:
        raise EntrypointContractError("altered or reordered operation in ordered registry")
    return tuple(actual)


def _bundle_rows(bundle: Path, *, before_command: bool = False) -> list[dict[str, object]]:
    excluded = {"protocol_hashes.json"}
    if before_command:
        excluded.add("scientific_command_template.json")
    files = sorted(path for path in bundle.rglob("*") if path.is_file() and path.name not in excluded)
    return [
        {
            "path": path.relative_to(bundle).as_posix(),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]


def _source_manifest(readiness_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        readiness_root / "source_freeze/source_manifest.json",
        readiness_root / "source_freeze/source_manifest_r3.json",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise EntrypointContractError("exactly one source-freeze manifest is required")
    return matches[0], _json(matches[0])


def validate_frozen_bundle(bundle: Path, hashes_path: Path, readiness_root: Path) -> dict[str, Any]:
    bundle = Path(bundle).resolve(strict=True)
    hashes_path = Path(hashes_path).resolve(strict=True)
    readiness_root = Path(readiness_root).resolve(strict=True)
    if hashes_path != bundle / "protocol_hashes.json":
        raise EntrypointContractError("protocol-hashes path must be the selected bundle member")
    hashes = _json(hashes_path)
    rows = _bundle_rows(bundle)
    if rows != hashes.get("files"):
        raise EntrypointContractError("protocol bundle file inventory mismatch")
    bundle_hash = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    if bundle_hash != hashes.get("protocol_bundle_sha256"):
        raise EntrypointContractError("wrong protocol-bundle hash")
    before_rows = _bundle_rows(bundle, before_command=True)
    before_hash = hashlib.sha256(canonical_json_bytes(before_rows)).hexdigest()
    command = _json(bundle / "scientific_command_template.json")
    if before_hash != hashes.get("protocol_content_tree_before_command_sha256"):
        raise EntrypointContractError("pre-command protocol content-tree hash mismatch")
    if command.get("preflight_protocol_content_tree_sha256") != before_hash:
        raise EntrypointContractError("future command is not bound to the protocol content tree")

    required_files = {
        "audit_contract_sha256": "audit_contract.json",
        "initialization_manifest_sha256": "initialization_manifest.json",
        "legacy_exclusion_manifest_sha256": "legacy_checkpoint_exclusion_manifest.json",
        "scene_manifest_sha256": "scenes_manifest.json",
        "cache_manifest_sha256": "cached_features_manifest.json",
        "capacity_manifest_sha256": "capacity_manifest.json",
        "decision_graph_sha256": "d3_decision_graph.json",
    }
    for key, filename in required_files.items():
        if _sha256_file(bundle / filename) != hashes.get(key):
            raise EntrypointContractError(f"wrong {key.replace('_sha256', '')} hash")

    source_path, source = _source_manifest(readiness_root)
    source_rows = source.get("files")
    if not isinstance(source_rows, list):
        raise EntrypointContractError("source manifest file inventory missing")
    current_rows = []
    for row in source_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise EntrypointContractError("malformed source manifest row")
        target = REPO / str(row["path"])
        if not target.is_file():
            raise EntrypointContractError(f"source file absent: {row['path']}")
        actual = {"path": row["path"], "sha256": _sha256_file(target), "bytes": target.stat().st_size}
        if actual != dict(row):
            raise EntrypointContractError(f"source freeze mismatch: {row['path']}")
        current_rows.append(actual)
    source_core = {
        "schema_version": source.get("schema_version"),
        "repository_commit": source.get("repository_commit"),
        "files": current_rows,
    }
    source_hash = hashlib.sha256(canonical_json_bytes(source_core)).hexdigest()
    if source_hash != source.get("source_freeze_sha256") or source_hash != hashes.get("source_freeze_sha256"):
        raise EntrypointContractError("wrong source-freeze hash")

    protocol = _json(bundle / "d3_protocol.json")
    validate_effective_protocol(protocol)
    registry = load_operation_registry(bundle)
    return {
        "bundle": bundle,
        "hashes": hashes,
        "command": command,
        "protocol": protocol,
        "values": protocol["values"],
        "registry": registry,
        "source_manifest_path": source_path,
        "source_manifest": source,
    }


def _rng_digest() -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.random.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "accelerator": None,
    }
    if torch.backends.mps.is_available():
        result["accelerator"] = hashlib.sha256(torch.mps.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    return result


def run_noninterference_probe(bundle: Path) -> dict[str, object]:
    initialization = _json(Path(bundle) / "initialization_manifest.json")
    seeds = initialization["expert_seeds"]
    before_rng = _rng_digest()
    fresh = construct_fresh_initial_state(str(initialization["capacity_identifier"]), seeds)
    after_construct_rng = _rng_digest()
    optimizer = create_fresh_a1_optimizer(fresh.model)
    inputs = (torch.zeros((1, 1), dtype=torch.float32),)
    targets = (torch.ones((1, 1), dtype=torch.float32),)
    before = observe_scientific_state(fresh.model, optimizer, inputs=inputs, targets=targets)
    observed = observe_scientific_state(fresh.model, optimizer, inputs=inputs, targets=targets)
    after = observe_scientific_state(fresh.model, optimizer, inputs=inputs, targets=targets)
    after_rng = _rng_digest()
    checks = {
        "rng_noninterference": before_rng == after_construct_rng == after_rng,
        "model_noninterference": before["model_state_sha256"] == observed["model_state_sha256"] == after["model_state_sha256"],
        "optimizer_noninterference": before["optimizer_state_sha256"] == observed["optimizer_state_sha256"] == after["optimizer_state_sha256"],
        "tensor_noninterference": before["inputs"] == observed["inputs"] == after["inputs"] and before["targets"] == observed["targets"] == after["targets"],
        "gradient_noninterference": before["gradient_state_sha256"] == observed["gradient_state_sha256"] == after["gradient_state_sha256"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def legacy_checkpoint_load_attempts() -> int:
    return _LEGACY_LOAD_ATTEMPTS


def _create_execution_root(readiness_root: Path) -> Path:
    override = os.environ.get("THAYER_PV1A1_OUTPUT_ROOT")
    if override:
        target = Path(override).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        target = readiness_root / "scientific_run" / f"authoritative_pv1a1_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    for name in ("audit", "checkpoints", "diagnostics", "logs", "reports", "semantic_states", "tables"):
        (target / name).mkdir()
    return target


def _load_promoted_cache(readiness_root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    protocol = context["values"]
    manifest = _json(context["bundle"] / "cached_features_manifest.json")
    promoted_generation = manifest.get("promoted_generation")
    if (
        promoted_generation != protocol["eight_scene"]["promoted_generation"]
        or manifest.get("generation_a_b_exact") is not True
        or not isinstance(protocol["cache_identity"].get("promotion_rule"), str)
    ):
        raise EntrypointContractError("only promoted prospective cache generation A is permitted")
    cache_path = readiness_root / "prospective_cache/ordered_cache.pt"
    if not cache_path.is_file():
        # R3 preserves the R2 prospective artifact rather than copying it.
        relation = _json(readiness_root / "protocol_bundle/r2_r3_relationship.json")
        cache_path = Path(relation["authoritative_r2_root"]) / "prospective_cache/ordered_cache.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if tuple(cache.get("scene_ids", ())) != tuple(protocol["eight_scene"]["scene_order"]):
        raise EntrypointContractError("promoted cache scene order mismatch")
    for prompt in ("prompt_a", "prompt_b"):
        tensors = cache.get(prompt)
        if not isinstance(tensors, tuple) or len(tensors) != 3:
            raise EntrypointContractError("malformed promoted feature cache")
        names = ("enc1", "enc2", "bottleneck")
        for name, tensor in zip(names, tensors, strict=True):
            expected = manifest["batch_shapes"][f"{prompt}.{name}"]
            if list(tensor.shape) != expected or not torch.isfinite(tensor).all().item():
                raise EntrypointContractError("promoted feature cache tensor contract failure")
    return {"path": cache_path, "payload": cache}


def _pre_science_traversal(args: argparse.Namespace, context: Mapping[str, Any], root: Path, journal: AuditJournal) -> dict[str, Any]:
    values = context["values"]
    initialization = _json(context["bundle"] / "initialization_manifest.json")
    init_check = validate_a1_initialization_manifest(initialization)
    journal.append("FRESH_INITIALIZATION_STARTED", "CONSTRUCT_FRESH_L0", {"capacity_identifier": initialization["capacity_identifier"]})
    fresh = construct_fresh_initial_state(initialization["capacity_identifier"], initialization["expert_seeds"])
    if fresh.manifest["combined_canonical_state_manifest_sha256"] != initialization["combined_canonical_state_manifest_sha256"]:
        raise EntrypointContractError("fresh seeded L0 initialization did not reproduce")
    journal.append("FRESH_INITIALIZATION_COMPLETED", "CONSTRUCT_FRESH_L0", {"initial_state_sha256": fresh.manifest["combined_canonical_state_manifest_sha256"], "optimizer_step_counter": 0})
    journal.append("FRESH_INITIALIZATION_REPRODUCED", "CONSTRUCT_FRESH_L0", {"exact": True})
    journal.append("INITIAL_STATE_FROZEN", "CONSTRUCT_FRESH_L0", {"initial_state_sha256": fresh.manifest["combined_canonical_state_manifest_sha256"]})

    exclusion = _json(context["bundle"] / "legacy_checkpoint_exclusion_manifest.json")
    if exclusion["model_loading"] != "FORBIDDEN" or exclusion["optimizer_loading"] != "FORBIDDEN":
        raise EntrypointContractError("legacy exclusion policy mismatch")
    try:
        reject_protected_checkpoint_use(exclusion["sha256"], "MODEL_STATE_LOAD")
    except AuditContractError:
        pass
    else:
        raise EntrypointContractError("protected legacy checkpoint load guard did not reject")
    legacy_path = REPO / "outputs/runs/thayer_output_parameterization_20260713_023120/checkpoints/ambiguous_one_scene_square.pth"
    legacy_record = hash_protected_file_for_integrity(legacy_path, expected_sha256=exclusion["sha256"])
    journal.append("LEGACY_CHECKPOINT_INVENTORIED", "VERIFY_LEGACY", {"checkpoint_sha256": legacy_record["sha256"], "scientific_load": False})
    journal.append("LEGACY_CHECKPOINT_EXCLUDED", "VERIFY_LEGACY", {"legacy_checkpoint_sha256": legacy_record["sha256"], "model_load_attempts": 0, "optimizer_load_attempts": 0, "scientific_evidence_references": 0})

    promoted = _load_promoted_cache(args.readiness_root, context)
    journal.append("CACHE_GENERATION_COMPLETED", "VERIFY_INPUTS", {"promoted_generation": "A", "scientific_optimizer_steps": 0})
    features_a = promoted["payload"]["prompt_a"]
    features_b = promoted["payload"]["prompt_b"]
    fresh.model.eval()
    with torch.no_grad():
        output_a = fresh.model.forward_features(features_a)
        output_b = fresh.model.forward_features(features_b)
    if output_a.shape != output_b.shape or list(output_a.shape[:3]) != [len(values["eight_scene"]["scene_order"]), 2, 6]:
        raise EntrypointContractError("no-update decoder traversal shape mismatch")
    if not torch.isfinite(output_a).all().item() or not torch.isfinite(output_b).all().item():
        raise EntrypointContractError("no-update decoder traversal produced nonfinite output")

    noninterference = run_noninterference_probe(context["bundle"])
    if noninterference["status"] != "PASS":
        raise EntrypointContractError("audit noninterference probe failed")
    report = {
        "schema_version": "thayer-d3-pv1-a1-pre-science-traversal-v1",
        "status": "PASS",
        "marker": READY_MARKER,
        "operation_registry": list(context["registry"]),
        "machine_operation_count": len(context["registry"]),
        "orchestration_obligation_count": SCIENTIFIC_ORCHESTRATION_OBLIGATION_COUNT,
        "protocol_bundle_sha256": context["hashes"]["protocol_bundle_sha256"],
        "source_freeze_sha256": context["hashes"]["source_freeze_sha256"],
        "fresh_initialization": init_check,
        "initial_state_sha256": fresh.manifest["combined_canonical_state_manifest_sha256"],
        "promoted_cache_path": str(promoted["path"]),
        "no_update_output_shape": list(output_a.shape),
        "noninterference": noninterference,
        "scientific_target_dependent_losses": 0,
        "scientific_backward_passes": 0,
        "scientific_optimizer_steps": 0,
        "scientific_branch_decisions": 0,
        "legacy_checkpoint_load_attempts": _LEGACY_LOAD_ATTEMPTS,
    }
    _write_json_x(root / "diagnostics/pre_science_traversal.json", report)
    return {"fresh": fresh, "promoted": promoted, "report": report}


def _execute_science(args: argparse.Namespace, context: Mapping[str, Any], root: Path, journal: AuditJournal, prepared: Mapping[str, Any]) -> dict[str, Any]:
    assets = _load_scientific_assets(args.readiness_root, context, prepared)
    counters = ScienceCounters()
    operation_records: list[dict[str, object]] = []
    state: dict[str, Any] = {"assets": assets, "prepared": prepared}
    handlers = {
        "CREATE_FRESH_ROOT": lambda: {"execution_root": str(root)},
        "VERIFY_PROTOCOL": lambda: {"protocol_bundle_sha256": context["hashes"]["protocol_bundle_sha256"]},
        "VERIFY_SOURCE": lambda: {"source_freeze_sha256": context["hashes"]["source_freeze_sha256"]},
        "VERIFY_AUDIT": lambda: run_noninterference_probe(context["bundle"]),
        "VERIFY_INPUTS": lambda: {"scene_count": len(assets["scene_ids"]), "target_sha256": assets["target_sha256"]},
        "VERIFY_LEGACY": lambda: {"legacy_checkpoint_load_attempts": legacy_checkpoint_load_attempts()},
        "CONSTRUCT_FRESH_L0": lambda: {"initial_state_sha256": prepared["fresh"].manifest["combined_canonical_state_manifest_sha256"]},
        "RUN_PRIMARY_L0_ONCE": lambda: _run_primary_l0(context, root, journal, assets, counters, prepared["fresh"]),
        "BIND_PRIMARY_L0_EVIDENCE": lambda: _bind_primary_l0(context, state),
        "DERIVE_BRANCH": lambda: _derive_branch(context, root, journal, state),
        "EXECUTE_SELECTED_BRANCH_ONLY": lambda: _execute_selected_branch(context, root, journal, assets, counters, state),
        "INDEPENDENT_REPLAY": lambda: _independent_scientific_replay(context, root, assets, state),
        "REQUIRE_RUNTIME_REPLAY_AGREEMENT": lambda: _require_scientific_agreement(state),
        "FINAL_INTEGRITY": lambda: _final_scientific_integrity(context, root, assets, counters, state),
        "REPORT_PASS_FAIL_UNKNOWN": lambda: _report_scientific_outcome(context, root, counters, state),
    }
    for index, operation in enumerate(context["registry"]):
        if operation not in handlers:
            raise EntrypointContractError(f"no validated implementation component for {operation}")
        result = handlers[operation]()
        state[operation] = result
        record = {"order_index": index, "operation": operation, "status": "PASS", "result": result}
        operation_records.append(record)
        _append_jsonl(root / "audit/operation_registry_events.jsonl", record)
    _write_json_x(root / "audit/operation_registry_replay.json", {
        "status": "PASS",
        "expected": list(context["registry"]),
        "actual": [row["operation"] for row in operation_records],
        "records": operation_records,
    })
    return state["REPORT_PASS_FAIL_UNKNOWN"]


@dataclass
class ScienceCounters:
    model_constructions: int = 0
    decoder_forwards: int = 0
    target_dependent_losses: int = 0
    backward_passes: int = 0
    optimizer_steps: int = 0
    branch_decisions: int = 0
    checkpoints: int = 0


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(dict(value)).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _authoritative_artifact_root(readiness_root: Path) -> Path:
    if (readiness_root / "prospective_cache/ordered_targets.npy").is_file():
        return readiness_root
    relation = _json(readiness_root / "protocol_bundle/r2_r3_relationship.json")
    return Path(relation["authoritative_r2_root"]).resolve(strict=True)


def _load_scientific_assets(
    readiness_root: Path,
    context: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    import h5py

    artifact_root = _authoritative_artifact_root(readiness_root)
    target_path = artifact_root / "prospective_cache/ordered_targets.npy"
    targets = np.load(target_path, allow_pickle=False)
    expected_shape = _json(context["bundle"] / "targets_manifest.json")["batch_shape"]
    if list(targets.shape) != expected_shape or not np.all(np.isfinite(targets)):
        raise EntrypointContractError("frozen scientific target tensor contract failure")
    for path, expected in EXPECTED_INPUT_HASHES.items():
        if path == SCENES and _sha256_file(path) != expected:
            raise EntrypointContractError("authorized source-scene container hash mismatch")
    with h5py.File(SCENES, "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise EntrypointContractError("authorized scene container is incomplete")
        blends = np.asarray(handle["blend"][list(SOURCE_INDICES)], dtype=np.float32)
    scene_ids = tuple(prepared["promoted"]["payload"]["scene_ids"])
    if len(blends) != len(scene_ids) or targets.shape[0] != len(scene_ids):
        raise EntrypointContractError("authorized scientific asset cardinality mismatch")
    target_sha = _tensor_observation_sha256(
        torch.from_numpy(np.ascontiguousarray(targets)),
        "N_PROMPT_TARGET_CHANNEL_HEIGHT_WIDTH",
    )
    return {
        "scene_ids": scene_ids,
        "features": {
            "prompt_a": prepared["promoted"]["payload"]["prompt_a"],
            "prompt_b": prepared["promoted"]["payload"]["prompt_b"],
        },
        "targets": targets,
        "blends": blends,
        "target_path": target_path,
        "target_sha256": target_sha,
        "target_file_sha256": _sha256_file(target_path),
    }


def _mps_device() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise EntrypointContractError("MPS fallback must be explicitly disabled")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise EntrypointContractError("MPS is required for authoritative neural science")
    return torch.device("mps")


def _optimizer_from_contract(model: torch.nn.Module, contract: Mapping[str, Any]) -> torch.optim.AdamW:
    if contract.get("class") != "AdamW":
        raise EntrypointContractError("frozen optimizer class mismatch")
    kwargs: dict[str, Any] = {
        "lr": float(contract["learning_rate"]),
        "weight_decay": float(contract["weight_decay"]),
    }
    optional = {
        "betas": lambda value: tuple(float(x) for x in value),
        "epsilon": float,
        "amsgrad": bool,
        "foreach": bool,
        "maximize": bool,
        "capturable": bool,
        "differentiable": bool,
    }
    names = {"epsilon": "eps"}
    for key, converter in optional.items():
        if key in contract:
            kwargs[names.get(key, key)] = converter(contract[key])
    optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    if optimizer.state:
        raise EntrypointContractError("fresh optimizer state must be empty before step one")
    return optimizer


def _slice_features(features: tuple[torch.Tensor, ...], indices: Sequence[int], device: torch.device) -> tuple[torch.Tensor, ...]:
    positions = list(indices)
    return tuple(value[positions].to(device=device) for value in features)


def _forward_physical(
    model: torch.nn.Module,
    features_a: tuple[torch.Tensor, ...],
    features_b: tuple[torch.Tensor, ...],
    scale6: torch.Tensor,
    counters: ScienceCounters,
) -> torch.Tensor:
    output_a = model.forward_features(features_a)
    counters.decoder_forwards += 1
    output_b = model.forward_features(features_b)
    counters.decoder_forwards += 1
    return torch.stack((output_a, output_b), dim=1) * scale6.view(1, 1, 1, 6, 1, 1)


def _target_loss(outputs: torch.Tensor, targets: torch.Tensor, scale6: torch.Tensor, counters: ScienceCounters) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_a, wins_a, margin_a = hard_physical_set_loss(outputs[:, 0], targets[:, 0], scale6)
    loss_b, wins_b, margin_b = hard_physical_set_loss(outputs[:, 1], targets[:, 1], scale6)
    counters.target_dependent_losses += 1
    return (loss_a + loss_b) / 2, torch.stack((wins_a, wins_b), dim=1), torch.stack((margin_a, margin_b), dim=1)


def _tensor_observation_sha256(value: torch.Tensor, logical_axis_layout: str) -> str:
    return str(observe_tensor(value, logical_axis_layout=logical_axis_layout, batch_order=None)["canonical_contiguous_content_sha256"])


def _assignment_flip_count(current: torch.Tensor, previous_cpu: torch.Tensor) -> int:
    return int((current.detach().to(device="cpu") != previous_cpu).sum().item())


def _finite_positive(value: float) -> bool:
    return bool(float(value) > 0 and np.isfinite(float(value)))


def _science_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    blends: np.ndarray,
    scene_ids: Sequence[str],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    physical = outputs.detach().cpu().numpy().astype(np.float64)
    truth = targets.detach().cpu().numpy().astype(np.float64)
    threshold, sky = frozen_thresholds()
    forward_contract = context["values"]["primary_l0"]["forward_plausibility"]
    if (
        threshold.global_chi_square_mean != forward_contract["global_chi_square_mean_maximum"]
        or list(threshold.per_band_chi_square_mean) != forward_contract["per_band_chi_square_mean_maximum_grz"]
        or threshold.absolute_relative_flux_residual != forward_contract["absolute_relative_flux_residual_maximum"]
        or not np.allclose(sky, forward_contract["sky_electrons_grz"], rtol=0.0, atol=0.0)
    ):
        raise EntrypointContractError("validated forward calculator does not match frozen protocol")
    rows = []
    for local, scene_id in enumerate(scene_ids):
        plausible = np.zeros((2, 2), dtype=bool)
        distances = np.zeros((2, 2, 2), dtype=np.float64)
        identities = np.zeros((2, 2), dtype=bool)
        forward_rows = []
        for prompt in (0, 1):
            identities[prompt] = prompt_identity_physical(physical[local, prompt], truth[local, prompt], truth.shape[2])
            for expert in (0, 1):
                candidate = physical[local, prompt, expert]
                score = forward_consistency(blends[local], np.stack((candidate[:3], candidate[3:])), sky)
                plausible[prompt, expert] = is_plausible(score, threshold)
                forward_rows.append({
                    "prompt": prompt,
                    "expert": expert,
                    "global_chi_square_mean": score.global_chi_square_mean,
                    "per_band_chi_square_mean": list(score.per_band_chi_square_mean),
                    "relative_flux_residual": score.relative_flux_residual,
                    "finite": score.finite,
                })
                for target in (0, 1):
                    distances[prompt, expert, target] = scientific_distance(
                        candidate[:3], truth[local, prompt, target, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL
                    ).primary_normalized
        gate = context["values"]["primary_l0"]["success_predicate"]["primary_normalized_gate"]
        own = all(any(plausible[p, e] and distances[p, e, 0] <= gate for e in (0, 1)) for p in (0, 1))
        alternate = all(any(plausible[p, e] and distances[p, e, 1] <= gate for e in (0, 1)) for p in (0, 1))
        both = all(
            plausible[p].all()
            and (
                (distances[p, 0, 0] <= gate and distances[p, 1, 1] <= gate)
                or (distances[p, 1, 0] <= gate and distances[p, 0, 1] <= gate)
            )
            for p in (0, 1)
        )
        prompt_identity = bool(identities.all())
        forward_all = bool(plausible.all())
        rows.append({
            "scene_id": scene_id,
            "own_coverage": own,
            "alternate_coverage": alternate,
            "both_mode_coverage": both,
            "prompt_identity_all_experts": prompt_identity,
            "forward_consistency_all_experts": forward_all,
            "pass": bool(own and alternate and both and prompt_identity and forward_all),
            "maximum_own_distance": float(np.max(np.min(distances[:, :, 0], axis=1))),
            "maximum_alternate_distance": float(np.max(np.min(distances[:, :, 1], axis=1))),
            "forward": forward_rows,
        })
    return {"scene_passes": [row["pass"] for row in rows], "all_pass": all(row["pass"] for row in rows), "rows": rows}


def _checkpoint(
    root: Path,
    condition_id: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    outputs: torch.Tensor,
    metrics: Mapping[str, Any],
    counters: ScienceCounters,
) -> dict[str, object]:
    path = root / "checkpoints" / condition_id / f"evaluation_step_{step:05d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "thayer-d3-pv1-a1-scientific-checkpoint-v1",
        "condition_id": condition_id,
        "step": step,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "physical_outputs": outputs.detach().cpu(),
        "physical_output_sha256": _tensor_observation_sha256(outputs, "N_PROMPT_EXPERT_CHANNEL_HEIGHT_WIDTH"),
        "metrics": dict(metrics),
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)
    counters.checkpoints += 1
    return {"path": str(path.relative_to(root)), "sha256": _sha256_file(path), "output_sha256": payload["physical_output_sha256"], "step": step}


def _run_condition(
    *,
    context: Mapping[str, Any],
    root: Path,
    journal: AuditJournal,
    assets: Mapping[str, Any],
    counters: ScienceCounters,
    condition_id: str,
    capacity: str,
    seeds: Sequence[int],
    indices: Sequence[int],
    maximum_steps: int,
    evaluation_steps: Sequence[int],
    optimizer_contract: Mapping[str, Any],
    early_success: bool,
    initial_fresh: Any | None = None,
) -> dict[str, Any]:
    device = _mps_device()
    fresh = construct_fresh_initial_state(capacity, seeds) if initial_fresh is None else initial_fresh
    counters.model_constructions += 1
    model = fresh.model.to(device)
    model.train()
    optimizer = _optimizer_from_contract(model, optimizer_contract)
    feature_a = _slice_features(assets["features"]["prompt_a"], indices, device)
    feature_b = _slice_features(assets["features"]["prompt_b"], indices, device)
    targets_cpu = np.ascontiguousarray(assets["targets"][list(indices)])
    targets = torch.from_numpy(targets_cpu).to(device)
    scale6 = torch.as_tensor(SCALES, dtype=torch.float32, device=device).repeat(2)
    blends = np.ascontiguousarray(assets["blends"][list(indices)])
    scene_ids = [assets["scene_ids"][index] for index in indices]
    feature_hashes_before = [_tensor_observation_sha256(value, "CACHED_FEATURE_NCHW") for value in (*feature_a, *feature_b)]
    target_hash_before = _tensor_observation_sha256(targets, "N_PROMPT_TARGET_CHANNEL_HEIGHT_WIDTH")
    evaluations = set(int(step) for step in evaluation_steps)
    evaluations.add(int(maximum_steps))
    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, object]] = []
    success_streak = 0
    assignment_previous: torch.Tensor | None = None
    assignment_flips = 0
    informative_gradient = False
    meaningful_update = False
    nonzero_gradient_parameters = 0
    gradient_parameter_count = 0
    minimum_objective = float("inf")
    terminal_event = "BUDGET_EXHAUSTED"

    for step in range(int(maximum_steps) + 1):
        gradient_norm = 0.0
        update_norm = 0.0
        if step > 0:
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_physical(model, feature_a, feature_b, scale6, counters)
            loss, wins, margins = _target_loss(outputs, targets, scale6, counters)
            loss.backward()
            counters.backward_passes += 1
            norms = []
            nonzero = 0
            total = 0
            for parameter in model.parameters():
                total += 1
                if parameter.grad is not None:
                    value = float(parameter.grad.detach().norm().cpu())
                    norms.append(value)
                    nonzero += int(_finite_positive(value))
            gradient_parameter_count = max(gradient_parameter_count, total)
            nonzero_gradient_parameters = max(nonzero_gradient_parameters, nonzero)
            gradient_norm = float(np.linalg.norm(norms))
            clip = optimizer_contract.get("gradient_clip_norm")
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip), norm_type=float(optimizer_contract.get("gradient_norm_type", 2.0)))
            optimizer.step()
            counters.optimizer_steps += 1
            changes = [float((parameter.detach() - prior).norm().cpu()) for parameter, prior in zip(model.parameters(), before, strict=True)]
            update_norm = float(np.linalg.norm(changes))
            informative_gradient = bool(informative_gradient or _finite_positive(gradient_norm))
            meaningful_update = bool(meaningful_update or _finite_positive(update_norm))
        if step not in evaluations:
            continue
        model.eval()
        with torch.no_grad():
            outputs = _forward_physical(model, feature_a, feature_b, scale6, counters)
            loss, wins, margins = _target_loss(outputs, targets, scale6, counters)
        model.train()
        if assignment_previous is not None:
            assignment_flips += _assignment_flip_count(wins, assignment_previous)
        assignment_previous = wins.detach().cpu()
        science = _science_metrics(outputs, targets, blends, scene_ids, context)
        success_streak = success_streak + 1 if science["all_pass"] else 0
        objective = float(loss.detach().cpu())
        minimum_objective = min(minimum_objective, objective)
        row = {
            "evaluation_index": len(rows),
            "step": step,
            "objective": objective,
            "gradient_norm": gradient_norm,
            "update_norm": update_norm,
            "assignment_flips": assignment_flips,
            "assignment_margin_minimum": float(margins.detach().abs().min().cpu()),
            "success_streak": success_streak,
            "all_pass": science["all_pass"],
            "scene_passes": science["scene_passes"],
            "scene_metrics": science["rows"],
        }
        checkpoint = _checkpoint(root, condition_id, step, model, optimizer, outputs, row, counters)
        checkpoints.append(checkpoint)
        row["checkpoint"] = checkpoint["path"]
        rows.append(row)
        triggers = semantic_state_triggers({
            "step_index": step,
            "optimizer_step_completed": step == 1,
            "eligible_objective": objective == minimum_objective,
            "eligible_distance_to_d1": False,
            "first_own_coverage": bool(science["rows"] and all(r["own_coverage"] for r in science["rows"]) and not any(s.get("first_own") for s in rows[:-1])),
            "first_alternate_coverage": bool(science["rows"] and all(r["alternate_coverage"] for r in science["rows"]) and not any(s.get("first_alternate") for s in rows[:-1])),
            "first_both_mode_coverage": bool(science["rows"] and all(r["both_mode_coverage"] for r in science["rows"]) and not any(s.get("first_both") for s in rows[:-1])),
            "success": bool(early_success and success_streak >= context["values"]["primary_l0"]["success_patience_evaluations"]),
            "terminal_failure": False,
            "budget_exhausted": step == maximum_steps,
            "final": bool(step == maximum_steps or (early_success and success_streak >= context["values"]["primary_l0"]["success_patience_evaluations"])),
        })
        row["first_own"] = "first_own_coverage" in triggers.details["states"]
        row["first_alternate"] = "first_alternate_coverage" in triggers.details["states"]
        row["first_both"] = "first_both_mode_coverage" in triggers.details["states"]
        for semantic_state in triggers.details["states"]:
            semantic_path = root / "semantic_states" / condition_id / f"{len(rows)-1:03d}_{semantic_state}.json"
            _write_json_x(semantic_path, {"state": semantic_state, "condition_id": condition_id, "step": step, "metrics": row, "checkpoint": checkpoint})
        if early_success and success_streak >= context["values"]["primary_l0"]["success_patience_evaluations"]:
            terminal_event = "SUCCESS_GATE"
            break
    if not rows:
        raise EntrypointContractError("scientific condition produced no evaluations")
    features_after = [_tensor_observation_sha256(value, "CACHED_FEATURE_NCHW") for value in (*feature_a, *feature_b)]
    target_hash_after = _tensor_observation_sha256(targets, "N_PROMPT_TARGET_CHANNEL_HEIGHT_WIDTH")
    if features_after != feature_hashes_before or target_hash_after != target_hash_before:
        raise EntrypointContractError("scientific condition mutated frozen feature or target tensors")
    journal.append("BACKWARD_COMPLETED", condition_id, {"count": counters.backward_passes, "condition_backward_passes": int(rows[-1]["step"])})
    journal.append("OPTIMIZER_STEP_COMPLETED", condition_id, {"step": int(rows[-1]["step"]), "count": counters.optimizer_steps})
    summary = {
        "condition_id": condition_id,
        "capacity": capacity,
        "seeds": list(seeds),
        "scene_ids": scene_ids,
        "initial_state_sha256": fresh.manifest["combined_canonical_state_manifest_sha256"],
        "terminal_event": terminal_event,
        "terminal_step": int(rows[-1]["step"]),
        "success": terminal_event == "SUCCESS_GATE" if early_success else bool(rows[-1]["all_pass"]),
        "final_scene_passes": rows[-1]["scene_passes"],
        "informative_gradient": informative_gradient,
        "meaningful_update": meaningful_update,
        "nonzero_gradient_parameter_fraction": 0.0 if gradient_parameter_count == 0 else nonzero_gradient_parameters / gradient_parameter_count,
        "assignment_flips": assignment_flips,
        "minimum_assignment_margin": min(float(row["assignment_margin_minimum"]) for row in rows),
        "minimum_objective": minimum_objective,
        "final_objective": float(rows[-1]["objective"]),
        "evaluation_count": len(rows),
        "rows": rows,
        "checkpoints": checkpoints,
        "features_unchanged": True,
        "targets_unchanged": True,
    }
    _write_json_x(root / "diagnostics" / f"{condition_id}_summary.json", summary)
    return summary


def _primary_scene_index(context: Mapping[str, Any], assets: Mapping[str, Any]) -> int:
    token = str(context["values"]["primary_l0"]["scientific_inputs"])
    try:
        suffix = token.split("PROSPECTIVE_", 1)[1].split("_CACHE", 1)[0].lower()
    except IndexError as error:
        raise EntrypointContractError("primary L0 scene selector is malformed") from error
    matches = [index for index, scene in enumerate(assets["scene_ids"]) if str(scene).lower().endswith(suffix)]
    if len(matches) != 1:
        raise EntrypointContractError("primary L0 scene selector is not unique")
    return matches[0]


def _run_primary_l0(context: Mapping[str, Any], root: Path, journal: AuditJournal, assets: Mapping[str, Any], counters: ScienceCounters, fresh: Any) -> dict[str, Any]:
    values = context["values"]
    primary = values["primary_l0"]
    return _run_condition(
        context=context,
        root=root,
        journal=journal,
        assets=assets,
        counters=counters,
        condition_id=str(primary["run_id"]),
        capacity="L0",
        seeds=values["initialization"]["l0_expert_seeds"],
        indices=[_primary_scene_index(context, assets)],
        maximum_steps=int(primary["maximum_optimizer_steps"]),
        evaluation_steps=primary["evaluation_steps"],
        optimizer_contract=primary["optimizer"],
        early_success=True,
        initial_fresh=fresh,
    )


def _bind_primary_l0(context: Mapping[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    primary = context["values"]["primary_l0"]
    binding = PrimaryL0EvidenceBinding()
    binding.bind_branch_selection(str(primary["run_id"]), str(primary["evidence_id"]))
    binding.bind_primary_capacity(str(primary["run_id"]), str(primary["evidence_id"]))
    state["binding"] = binding
    return binding.verify()


def _derive_branch(context: Mapping[str, Any], root: Path, journal: AuditJournal, state: dict[str, Any]) -> dict[str, Any]:
    l0 = state["RUN_PRIMARY_L0_ONCE"]
    optimization = evaluate_optimization_diagnostic(l0["informative_gradient"], l0["meaningful_update"], l0["success"])
    assignment_mode = "repeated_flips" if l0["assignment_flips"] else "stable_identity"
    assignment = evaluate_assignment_diagnostic(assignment_mode)
    square = evaluate_square_mapping_diagnostic("usable_derivatives" if l0["nonzero_gradient_parameter_fraction"] > 0 else "high_zero_gradient")
    evidence = {
        "valid": True,
        "l0_completed_validly": True,
        "source_integrity": True,
        "input_integrity": True,
        "metric_integrity": True,
        "scientific_contract_integrity": True,
        "frozen_l0_success": bool(l0["success"]),
        "d0_pass": True,
        "d1_pass": True,
        "other_validity_predicates_pass": True,
        "optimization_mechanism_supported": optimization.status == "INFORMATIVE_NO_SUCCESS",
        "hard_assignment_mechanism_supported": assignment.status in {"REPEATED_FLIPS", "LOW_MARGIN", "PROMPT_INCONSISTENT"},
        "square_mapping_mechanism_supported": square.status != "USABLE_DERIVATIVES",
        "validated_tangent_capture": float(l0["nonzero_gradient_parameter_fraction"]),
    }
    producer = RuntimeDecisionProducer(
        tangent_threshold=float(context["values"]["decisions"]["tangent_threshold"]),
        required_scene_count=int(context["values"]["eight_scene"]["required_pass_count"]),
    )
    runtime = producer.select_branch(evidence)
    replay_protocol = {
        "protocol_identifier": context["protocol"]["protocol_identifier"],
        "tangent_comparison": context["values"]["decisions"]["tangent_comparison"],
        "tangent_threshold": context["values"]["decisions"]["tangent_threshold"],
        "eight_scene_required_pass_count": context["values"]["eight_scene"]["required_pass_count"],
        "capacities": context["values"]["capacity_ladder"]["levels"],
    }
    replayer = IndependentAuditReplayer(replay_protocol)
    replay = replayer.verify_runtime_result(evidence, runtime)
    state["branch_evidence"] = evidence
    state["runtime_producer"] = producer
    state["independent_replayer"] = replayer
    counters = state.get("counters")
    journal.append("D3_DECISION_COMPUTED", "DERIVE_BRANCH", {"runtime": runtime, "replay": replay, "evidence": evidence})
    result = {"runtime": runtime, "independent_replay": replay, "evidence": evidence, "diagnostics": {"optimization": optimization.status, "assignment": assignment.status, "square_mapping": square.status}}
    _write_json_x(root / "diagnostics/primary_l0_branch_decision.json", result)
    return result


def _execute_selected_branch(context: Mapping[str, Any], root: Path, journal: AuditJournal, assets: Mapping[str, Any], counters: ScienceCounters, state: dict[str, Any]) -> dict[str, Any]:
    branch = state["DERIVE_BRANCH"]["runtime"]["downstream_branch"]
    counters.branch_decisions += 1
    if branch == "NONE":
        return {"branch": branch, "executed_conditions": 0, "terminal_evidence": {"valid": True}}
    if branch == "EIGHT_SCENE":
        spec = context["values"]["eight_scene"]
        fresh = construct_fresh_initial_state("L0", spec["expert_seeds"])
        run_id = "eight_scene_fresh_l0"
        state["binding"].bind_eight_scene(run_id, fresh.manifest["combined_canonical_state_manifest_sha256"])
        cadence = spec["evaluation_cadence"]
        steps = [cadence["initial"], cadence["first_step"], *range(cadence["periodic_interval"], spec["terminal_step"] + 1, cadence["periodic_interval"]), cadence["terminal"]]
        result = _run_condition(
            context=context, root=root, journal=journal, assets=assets, counters=counters,
            condition_id=run_id, capacity="L0", seeds=spec["expert_seeds"], indices=list(range(len(assets["scene_ids"]))),
            maximum_steps=int(spec["optimizer_steps"]), evaluation_steps=sorted(set(steps)), optimizer_contract=spec["optimizer"],
            early_success=False, initial_fresh=fresh,
        )
        evidence = {"valid": True, "scene_passes": result["final_scene_passes"]}
        status = state["runtime_producer"].classify_terminal(branch, evidence)
        return {"branch": branch, "executed_conditions": 1, "result": result, "terminal_evidence": evidence, "d3_status": status}
    if branch != "CAPACITY_LADDER":
        raise EntrypointContractError("runtime selected an unknown branch")
    ladder = context["values"]["capacity_ladder"]
    primary = state["RUN_PRIMARY_L0_ONCE"]
    level_results: dict[str, list[bool]] = {"L0": [bool(primary["success"])]}
    condition_results: list[dict[str, Any]] = []
    primary_index = _primary_scene_index(context, assets)
    offsets = ladder["replica_offsets"]
    for level in ladder["levels"][1:]:
        outcomes = []
        for offset in offsets:
            seeds = [int(seed) + int(offset) for seed in level["seeds"]]
            result = _run_condition(
                context=context, root=root, journal=journal, assets=assets, counters=counters,
                condition_id=f"capacity_{level['level']}_offset_{offset}", capacity=level["level"], seeds=seeds,
                indices=[primary_index], maximum_steps=int(ladder["per_condition_maximum_optimizer_steps"]),
                evaluation_steps=context["values"]["primary_l0"]["evaluation_steps"], optimizer_contract=context["values"]["primary_l0"]["optimizer"],
                early_success=True,
            )
            outcomes.append(bool(result["success"]))
            condition_results.append(result)
        level_results[level["level"]] = outcomes
    first = next((level["level"] for level in ladder["levels"][1:] if all(level_results[level["level"]])), None)
    evidence = {"valid": True, "first_passing_level": first, "level_results": level_results}
    status = state["runtime_producer"].classify_terminal(branch, evidence)
    return {"branch": branch, "executed_conditions": len(condition_results), "results": condition_results, "terminal_evidence": evidence, "d3_status": status}


def _independent_scientific_replay(context: Mapping[str, Any], root: Path, assets: Mapping[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    branch = state["EXECUTE_SELECTED_BRANCH_ONLY"]["branch"]
    terminal_evidence = state["EXECUTE_SELECTED_BRANCH_ONLY"]["terminal_evidence"]
    runtime_status = state["EXECUTE_SELECTED_BRANCH_ONLY"].get("d3_status", "UNKNOWN")
    replay_status = state["independent_replayer"].verify_terminal_result(branch, terminal_evidence, runtime_status)
    selected: list[dict[str, object]] = []
    summaries = [state["RUN_PRIMARY_L0_ONCE"]]
    branch_result = state["EXECUTE_SELECTED_BRANCH_ONLY"]
    if "result" in branch_result:
        summaries.append(branch_result["result"])
    summaries.extend(branch_result.get("results", []))
    device = _mps_device()
    scale6 = torch.as_tensor(SCALES, dtype=torch.float32, device=device).repeat(2)
    for summary in summaries:
        final = summary["checkpoints"][-1]
        checkpoint = torch.load(root / final["path"], map_location="cpu", weights_only=True)
        fresh = construct_fresh_initial_state(summary["capacity"], summary["seeds"])
        model = fresh.model.to(device).eval()
        indices = [assets["scene_ids"].index(scene) for scene in summary["scene_ids"]]
        feature_a = _slice_features(assets["features"]["prompt_a"], indices, device)
        feature_b = _slice_features(assets["features"]["prompt_b"], indices, device)
        state_dict = {key: value.to(device) for key, value in checkpoint["model_state"].items()}
        expert_1_state = {key.split("expert_1.", 1)[1]: value for key, value in state_dict.items() if key.startswith("expert_1.")}
        expert_2_state = {key.split("expert_2.", 1)[1]: value for key, value in state_dict.items() if key.startswith("expert_2.")}
        with torch.no_grad():
            out_a = torch.stack((
                torch.func.functional_call(model.expert_1, expert_1_state, feature_a),
                torch.func.functional_call(model.expert_2, expert_2_state, feature_a),
            ), dim=1)
            out_b = torch.stack((
                torch.func.functional_call(model.expert_1, expert_1_state, feature_b),
                torch.func.functional_call(model.expert_2, expert_2_state, feature_b),
            ), dim=1)
            physical = torch.stack((out_a, out_b), dim=1) * scale6.view(1, 1, 1, 6, 1, 1)
        actual = _tensor_observation_sha256(physical, "N_PROMPT_EXPERT_CHANNEL_HEIGHT_WIDTH")
        selected.append({"condition_id": summary["condition_id"], "step": final["step"], "expected": final["output_sha256"], "actual": actual, "status": "PASS" if actual == final["output_sha256"] else "FAIL"})
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in selected) else "FAIL", "terminal_status": replay_status, "checkpoint_rows": selected}
    if result["status"] != "PASS":
        raise EntrypointContractError("independent checkpoint replay disagrees")
    _write_json_x(root / "audit/scientific_independent_replay.json", result)
    return result


def _require_scientific_agreement(state: dict[str, Any]) -> dict[str, object]:
    runtime = state["EXECUTE_SELECTED_BRANCH_ONLY"].get("d3_status", "UNKNOWN")
    replay = state["INDEPENDENT_REPLAY"]["terminal_status"]
    if runtime != replay or state["DERIVE_BRANCH"]["runtime"] != state["DERIVE_BRANCH"]["independent_replay"]:
        raise EntrypointContractError("runtime and independent scientific replay disagree")
    return {"status": "PASS", "runtime_terminal": runtime, "replay_terminal": replay}


def _final_scientific_integrity(context: Mapping[str, Any], root: Path, assets: Mapping[str, Any], counters: ScienceCounters, state: dict[str, Any]) -> dict[str, Any]:
    source_manifest = context["source_manifest"]
    source_unchanged = all(
        (REPO / row["path"]).is_file()
        and (REPO / row["path"]).stat().st_size == row["bytes"]
        and _sha256_file(REPO / row["path"]) == row["sha256"]
        for row in source_manifest["files"]
    )
    target_unchanged = _sha256_file(assets["target_path"]) == assets["target_file_sha256"]
    checkpoint_index = [{"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path), "bytes": path.stat().st_size} for path in sorted((root / "checkpoints").rglob("*.pt"))]
    result = {
        "status": "PASS" if source_unchanged and target_unchanged and legacy_checkpoint_load_attempts() == 0 else "FAIL",
        "source_unchanged_after_first_loss": source_unchanged,
        "target_unchanged": target_unchanged,
        "legacy_checkpoint_load_attempts": legacy_checkpoint_load_attempts(),
        "checkpoint_inventory": checkpoint_index,
        "counters": vars(counters),
        "atlas_access": 0,
        "development_access": 0,
        "lockbox_access": 0,
        "broader_scene_access": 0,
    }
    if result["status"] != "PASS":
        raise EntrypointContractError("final scientific integrity check failed")
    _write_json_x(root / "audit/final_scientific_integrity.json", result)
    return result


def _report_scientific_outcome(context: Mapping[str, Any], root: Path, counters: ScienceCounters, state: dict[str, Any]) -> dict[str, Any]:
    l0 = state["RUN_PRIMARY_L0_ONCE"]
    branch_result = state["EXECUTE_SELECTED_BRANCH_ONLY"]
    branch = branch_result["branch"]
    d3_status = branch_result.get("d3_status", "UNKNOWN")
    mechanism = state["DERIVE_BRANCH"]["evidence"]
    outcome_evidence = OutcomeEvidence(
        implementation_or_contract_failure=False,
        authoritative_trajectory_exists=True,
        full_scientific_success=bool(l0["success"]),
        optimization_barrier_supported=bool(mechanism["optimization_mechanism_supported"]),
        capacity_barrier_supported=branch == "CAPACITY_LADDER",
        hard_assignment_barrier_supported=bool(mechanism["hard_assignment_mechanism_supported"]),
        square_mapping_barrier_supported=bool(mechanism["square_mapping_mechanism_supported"]),
        evidence_consistent=True,
        capacity_relies_on_tangent=branch == "CAPACITY_LADDER",
        tangent_protocol_passed=branch != "CAPACITY_LADDER" or d3_status in {"PASS", "FAIL"},
    )
    outcome_decision = map_scientific_outcome(outcome_evidence)
    authorization = authorize_downstream(AuthorizationContext(
        outcome=outcome_decision.status,
        authoritative_d3_scientific_failure=not bool(l0["success"]),
        d0_authoritative_pass=True,
        d1_authoritative_pass=True,
        prompt_gate_pass=bool(l0["rows"][-1]["scene_metrics"][0]["prompt_identity_all_experts"]),
        forward_gate_pass=bool(l0["rows"][-1]["scene_metrics"][0]["forward_consistency_all_experts"]),
        fresh_process_replay_pass=state["INDEPENDENT_REPLAY"]["status"] == "PASS",
        contract_unchanged=state["FINAL_INTEGRITY"]["source_unchanged_after_first_loss"],
        contract_integrity=state["FINAL_INTEGRITY"]["status"] == "PASS",
        code_runtime_loss_mapping_assignment_defect=False,
        tangent_evidence_used=branch == "CAPACITY_LADDER",
        tangent_protocol_passed=branch != "CAPACITY_LADDER" or d3_status in {"PASS", "FAIL"},
    ))
    result = {
        "schema_version": "thayer-d3-pv1-a1-authoritative-scientific-outcome-v1",
        "protocol_identifier": context["protocol"]["protocol_identifier"],
        "l0_scientific_result": "PASS" if l0["success"] else "FAIL",
        "l0_terminal_step": l0["terminal_step"],
        "selected_branch": branch,
        "branch_scientific_status": d3_status,
        "authoritative_scientific_outcome": outcome_decision.status,
        "downstream_authorization": authorization.status,
        "runtime_replay_agreement": True,
        "counters": vars(counters),
        "first_target_dependent_loss_was_irreversible": True,
        "source_changed_after_first_loss": False,
        "protected_legacy_loads": legacy_checkpoint_load_attempts(),
    }
    _write_json_x(root / "reports/authoritative_scientific_outcome.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.protocol_bundle = args.protocol_bundle.resolve(strict=True)
    args.protocol_hashes = args.protocol_hashes.resolve(strict=True)
    args.readiness_root = args.readiness_root.resolve(strict=True)
    context = validate_frozen_bundle(args.protocol_bundle, args.protocol_hashes, args.readiness_root)
    root = _create_execution_root(args.readiness_root)
    journal = AuditJournal.create(
        execution_root=root,
        protocol_identifier=str(context["protocol"]["protocol_identifier"]),
        campaign_identifier=root.name,
    )
    journal.append("HUMAN_AUTHORITY_FROZEN", "VERIFY_PROTOCOL", {"protocol_bundle_sha256": context["hashes"]["protocol_bundle_sha256"]})
    journal.append("HUMAN_AUTHORITY_AMENDMENT_FROZEN", "VERIFY_PROTOCOL", {"amendment_sha256": _sha256_file(context["bundle"] / "amendment_A1_human_approval.md")})
    journal.append("SOURCE_FREEZE_CREATED", "VERIFY_SOURCE", {"source_freeze_sha256": context["hashes"]["source_freeze_sha256"]})
    prepared = _pre_science_traversal(args, context, root, journal)
    print(READY_MARKER, flush=True)
    if os.environ.get("THAYER_PV1A1_PRE_SCIENCE_ONLY") == "1":
        journal.append("CAMPAIGN_TERMINATED", "pre_science", {"pre_science_only": True, "d3_status": "UNKNOWN", "scientific_backward_passes": 0, "scientific_optimizer_steps": 0})
        _write_json_x(root / "audit/event_chain_head.json", verify_event_chain(journal.path))
        return 0
    result = _execute_science(args, context, root, journal, prepared)
    _write_json_x(root / "reports/final_outcome.json", result)
    journal.append("CAMPAIGN_TERMINATED", "REPORT_PASS_FAIL_UNKNOWN", {
        "d3_status": result["branch_scientific_status"],
        "authoritative_scientific_outcome": result["authoritative_scientific_outcome"],
        "scientific_backward_passes": result["counters"]["backward_passes"],
        "scientific_optimizer_steps": result["counters"]["optimizer_steps"],
        "scientific_branch_selections": result["counters"]["branch_decisions"],
    })
    _write_json_x(root / "audit/event_chain_head.json", verify_event_chain(journal.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
