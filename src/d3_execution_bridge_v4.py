"""Bundle-v3 authority bridge for append-only authoritative D3 execution.

This module contains no scientific thresholds, policy decisions, model
definitions, optimizer definitions, or scientific array values.  It validates
and links the four frozen authorities used by the v4 launch path.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "thayer-d3-execution-bridge-v4"
BRIDGE_VERSION = "4.0.0"
CAPSULE_V1_SHA256 = "8a76ccdfa659a7291f0f9b73e0cb4d4c8adfb317b9902fc8ad5763e6d17b7d21"
CONTINUATION_VALUE = "true_after_v4_preflight"
REQUIRED_SOURCE_PATHS = (
    "scripts/run_thayer_scientific_d3_v4.py",
    "scripts/run_thayer_scientific_d3_process_v4.py",
    "scripts/run_thayer_d3_postprocess_v4.py",
    "src/d3_execution_bridge_v4.py",
)


class IntegrationRequirementFailure(RuntimeError):
    """Fail closed with one canonical integration requirement identifier."""

    def __init__(self, requirement_id: str, message: str):
        super().__init__(f"{requirement_id}: {message}")
        self.requirement_id = requirement_id
        self.message = message


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def read_json(path: Path, requirement_id: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationRequirementFailure(requirement_id, str(exc)) from exc
    if not isinstance(value, dict):
        raise IntegrationRequirementFailure(requirement_id, "JSON root is not an object")
    return value


def require_hash(path: Path, expected: str, requirement_id: str) -> None:
    if not path.is_file():
        raise IntegrationRequirementFailure(requirement_id, f"missing file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise IntegrationRequirementFailure(
            requirement_id, f"SHA-256 mismatch: expected={expected} actual={actual} path={path}"
        )


def _require_record(
    repo: Path, record: Mapping[str, Any], requirement_id: str
) -> Path:
    if not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise IntegrationRequirementFailure(requirement_id, "path/SHA-256 record incomplete")
    path = resolve(repo, str(record["path"]))
    require_hash(path, str(record["sha256"]), requirement_id)
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise IntegrationRequirementFailure(requirement_id, "byte-size mismatch")
    return path


def validate_authority_chain(
    repo: Path, bundle_v3_path: Path, bundle_v3_sha256: str
) -> dict[str, Any]:
    """Validate v3 -> v2 -> capsule-v1/runtime and policy references."""

    repo = repo.resolve()
    bundle_v3_path = bundle_v3_path.resolve()
    require_hash(bundle_v3_path, bundle_v3_sha256, "D3I-AUTH-BUNDLE-V3-SHA")
    bundle_v3 = read_json(bundle_v3_path, "D3I-AUTH-BUNDLE-V3-JSON")
    if bundle_v3.get("schema_version") != "thayer-d3-executable-bundle-v3":
        raise IntegrationRequirementFailure("D3I-AUTH-BUNDLE-V3-SCHEMA", "schema mismatch")

    base_record = bundle_v3.get("base_bundle_v2")
    if not isinstance(base_record, dict):
        raise IntegrationRequirementFailure("D3I-AUTH-BASE-V2-REFERENCE", "base v2 reference missing")
    bundle_v2_path = _require_record(repo, base_record, "D3I-AUTH-BASE-V2-SHA")
    bundle_v2 = read_json(bundle_v2_path, "D3I-AUTH-BASE-V2-JSON")
    if bundle_v2.get("schema_version") != "thayer-d3-executable-bundle-v2":
        raise IntegrationRequirementFailure("D3I-AUTH-BASE-V2-SCHEMA", "schema mismatch")

    for key, requirement_id in (
        ("policy_registry", "D3I-AUTH-POLICY-REGISTRY"),
        ("policy_engine", "D3I-AUTH-POLICY-ENGINE"),
        ("policy_preflight", "D3I-AUTH-POLICY-PREFLIGHT"),
    ):
        record = bundle_v3.get(key)
        if not isinstance(record, dict):
            raise IntegrationRequirementFailure(requirement_id, f"missing v3 {key}")
        _require_record(repo, record, requirement_id)

    capsule_v2_record = bundle_v2.get("capsule_v2")
    if not isinstance(capsule_v2_record, dict):
        raise IntegrationRequirementFailure("D3I-AUTH-CAPSULE-V2", "capsule-v2 record missing")
    capsule_v2_path = _require_record(repo, capsule_v2_record, "D3I-AUTH-CAPSULE-V2")
    capsule_v2 = read_json(capsule_v2_path, "D3I-AUTH-CAPSULE-V2")
    embedded_capsule = capsule_v2.get("scientific_capsule_v1")
    identity = capsule_v2.get("capsule_identity")
    if not isinstance(embedded_capsule, dict) or not isinstance(identity, dict):
        raise IntegrationRequirementFailure("D3I-AUTH-CAPSULE-V1-EMBEDDED", "embedded capsule missing")
    capsule_v1_reference = identity.get("supersedes_without_modifying")
    if not isinstance(capsule_v1_reference, str):
        raise IntegrationRequirementFailure("D3I-AUTH-CAPSULE-V1-REFERENCE", "external capsule path missing")
    capsule_v1_path = resolve(repo, capsule_v1_reference)
    require_hash(capsule_v1_path, CAPSULE_V1_SHA256, "D3I-AUTH-CAPSULE-V1-SHA")
    capsule_v1 = read_json(capsule_v1_path, "D3I-AUTH-CAPSULE-V1-JSON")
    if embedded_capsule != capsule_v1:
        raise IntegrationRequirementFailure(
            "D3I-AUTH-CAPSULE-V1-EMBEDDED-EQUALITY", "embedded and external capsule differ"
        )

    runtime_record = bundle_v2.get("runtime_readiness")
    if not isinstance(runtime_record, dict):
        raise IntegrationRequirementFailure("D3I-AUTH-RUNTIME", "runtime record missing")
    runtime_path = _require_record(repo, runtime_record, "D3I-AUTH-RUNTIME-SHA")
    runtime = read_json(runtime_path, "D3I-AUTH-RUNTIME-JSON")
    if runtime.get("status") != runtime_record.get("expected_status"):
        raise IntegrationRequirementFailure("D3I-AUTH-RUNTIME-STATUS", "readiness status mismatch")
    if runtime.get("checkpoint_before", {}).get("count") != 600:
        raise IntegrationRequirementFailure("D3I-AUTH-RUNTIME-CHECKPOINTS", "checkpoint count mismatch")

    registry_record = bundle_v2.get("requirement_registry")
    if not isinstance(registry_record, dict):
        raise IntegrationRequirementFailure("D3I-AUTH-REQUIREMENT-REGISTRY", "registry missing")
    registry_path = _require_record(repo, registry_record, "D3I-AUTH-REQUIREMENT-REGISTRY")
    registry = read_json(registry_path, "D3I-AUTH-REQUIREMENT-REGISTRY")
    if len(registry.get("requirements", [])) != 180:
        raise IntegrationRequirementFailure("D3I-AUTH-REQUIREMENT-COUNT", "expected 180 requirements")

    return {
        "bundle_v3_path": bundle_v3_path,
        "bundle_v3": bundle_v3,
        "bundle_v2_path": bundle_v2_path,
        "bundle_v2": bundle_v2,
        "capsule_v2_path": capsule_v2_path,
        "capsule_v2": capsule_v2,
        "capsule_v1_path": capsule_v1_path,
        "capsule_v1": capsule_v1,
        "runtime_path": runtime_path,
        "runtime": runtime,
        "registry_path": registry_path,
        "registry": registry,
    }


def source_records(repo: Path, relative_paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = resolve(repo, relative)
        if not path.is_file():
            raise IntegrationRequirementFailure("D3I-SOURCE-PRESENT", f"missing source: {relative}")
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def build_bridge(
    *,
    repo: Path,
    run: Path,
    bundle_v3_path: Path,
    bundle_v3_sha256: str,
    repository_head: str,
    phase: str,
    synthetic_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an in-memory reference-only bridge after validating authorities."""

    if phase not in {"candidate", "frozen"}:
        raise IntegrationRequirementFailure("D3I-BRIDGE-PHASE", f"unsupported phase: {phase}")
    chain = validate_authority_chain(repo, bundle_v3_path, bundle_v3_sha256)
    v3 = chain["bundle_v3"]
    v2 = chain["bundle_v2"]
    sources = source_records(repo, REQUIRED_SOURCE_PATHS)
    bridge = {
        "schema_version": SCHEMA_VERSION,
        "bridge_version": BRIDGE_VERSION,
        "phase": phase,
        "producing_run": str(run.resolve().relative_to(repo.resolve())),
        "repository_head": repository_head,
        "created_utc": utcnow(),
        "authorities": {
            "bundle_v3": {
                "path": str(chain["bundle_v3_path"].relative_to(repo)),
                "sha256": bundle_v3_sha256,
            },
            "base_bundle_v2": {
                "path": str(chain["bundle_v2_path"].relative_to(repo)),
                "sha256": sha256_file(chain["bundle_v2_path"]),
                "resolved_from": "bundle_v3.base_bundle_v2",
            },
            "scientific_capsule_v1": {
                "path": str(chain["capsule_v1_path"].relative_to(repo)),
                "sha256": CAPSULE_V1_SHA256,
                "resolved_from": "base_bundle_v2.capsule_v2 -> capsule_identity.supersedes_without_modifying",
            },
            "runtime_readiness": {
                "path": str(chain["runtime_path"].relative_to(repo)),
                "sha256": sha256_file(chain["runtime_path"]),
                "resolved_from": "base_bundle_v2.runtime_readiness",
            },
        },
        "precedence": {
            "policy_and_continuation": "bundle_v3",
            "architecture_and_artifact_schemas": "base_bundle_v2",
            "scientific_values": "scientific_capsule_v1",
            "runtime": "runtime_readiness",
            "bundle_v2_historical_scope_authoritative": False,
            "automatic_scientific_continuation": CONTINUATION_VALUE,
        },
        "launchers": {
            "orchestrator": sources["scripts/run_thayer_scientific_d3_v4.py"],
            "scientific_worker": sources["scripts/run_thayer_scientific_d3_process_v4.py"],
            "postprocessing_worker": sources["scripts/run_thayer_d3_postprocess_v4.py"],
            "bridge_validator": sources["src/d3_execution_bridge_v4.py"],
            "policy_engine": dict(v3["policy_engine"]),
            "policy_preflight": dict(v3["policy_preflight"]),
            "policy_registry": dict(v3["policy_registry"]),
            "semantic_state_adapter": {
                "path": "src/d3_state_machine.py",
                "sha256": sha256_file(repo / "src/d3_state_machine.py"),
            },
            "pure_evaluator": dict(v2["pure_evaluator"]),
        },
        "cli_propagation": {
            "orchestrator_bundle_v3": "authorities.bundle_v3.path",
            "orchestrator_bundle_v3_sha256": "authorities.bundle_v3.sha256",
            "worker_bridge_path": "orchestrator validated bridge path",
            "worker_bridge_sha256": "orchestrator validated bridge SHA-256",
            "worker_output_root": str(run.resolve()),
            "worker_runtime_root": str((run / "runtime/scientific").resolve()),
            "bundle_v2_user_argument_permitted": False,
            "dropped_bundle_v3_permitted": False,
        },
        "scientific_contract": {
            "l0_constructor": v2["l0_constructor"],
            "artifact_member_contracts_reference": "base_bundle_v2.artifact_member_contracts",
            "optimizer_contract_reference": "base_bundle_v2.optimizer_contract",
            "execution_budget_reference": "base_bundle_v2.execution_budget",
            "policy_contract_reference": "bundle_v3.policy_registry",
            "semantic_state_contract": v3["semantic_state_contract"],
            "outcome_categories": v3["outcome_categories"],
            "authorization_contract": v3["authorization_contract"],
        },
        "synthetic_preflight": dict(synthetic_preflight),
        "flow_invariants": {
            "bundle_v3_sha": bundle_v3_sha256,
            "policy_engine_sha": v3["policy_engine"]["sha256"],
            "base_bundle_v2_sha": sha256_file(chain["bundle_v2_path"]),
        },
    }
    return bridge


def validate_bridge(
    *, repo: Path, bridge_path: Path, bridge_sha256: str, require_frozen: bool
) -> dict[str, Any]:
    """Validate a bridge, all source hashes, and the linked authority chain."""

    require_hash(bridge_path, bridge_sha256, "D3I-BRIDGE-SHA")
    bridge = read_json(bridge_path, "D3I-BRIDGE-JSON")
    if bridge.get("schema_version") != SCHEMA_VERSION or bridge.get("bridge_version") != BRIDGE_VERSION:
        raise IntegrationRequirementFailure("D3I-BRIDGE-SCHEMA", "bridge identity mismatch")
    if require_frozen and bridge.get("phase") != "frozen":
        raise IntegrationRequirementFailure("D3I-BRIDGE-FROZEN", "frozen bridge required")
    precedence = bridge.get("precedence", {})
    if precedence.get("automatic_scientific_continuation") != CONTINUATION_VALUE:
        raise IntegrationRequirementFailure("D3I-BRIDGE-CONTINUATION", "continuation flag absent")
    if precedence.get("bundle_v2_historical_scope_authoritative") is not False:
        raise IntegrationRequirementFailure("D3I-BRIDGE-V2-SCOPE", "historical v2 scope treated as authority")

    authorities = bridge.get("authorities", {})
    v3_record = authorities.get("bundle_v3")
    if not isinstance(v3_record, dict):
        raise IntegrationRequirementFailure("D3I-BRIDGE-V3", "bundle-v3 authority absent")
    chain = validate_authority_chain(
        repo, resolve(repo, str(v3_record.get("path", ""))), str(v3_record.get("sha256", ""))
    )
    expected = {
        "base_bundle_v2": (chain["bundle_v2_path"], sha256_file(chain["bundle_v2_path"])),
        "scientific_capsule_v1": (chain["capsule_v1_path"], CAPSULE_V1_SHA256),
        "runtime_readiness": (chain["runtime_path"], sha256_file(chain["runtime_path"])),
    }
    for name, (path, digest) in expected.items():
        record = authorities.get(name)
        if not isinstance(record, dict):
            raise IntegrationRequirementFailure(f"D3I-BRIDGE-{name.upper()}", "authority absent")
        if resolve(repo, str(record.get("path", ""))) != path or record.get("sha256") != digest:
            raise IntegrationRequirementFailure(f"D3I-BRIDGE-{name.upper()}", "authority substitution")

    launchers = bridge.get("launchers", {})
    for name in (
        "orchestrator",
        "scientific_worker",
        "postprocessing_worker",
        "bridge_validator",
        "policy_engine",
        "policy_preflight",
        "policy_registry",
        "semantic_state_adapter",
        "pure_evaluator",
    ):
        record = launchers.get(name)
        if not isinstance(record, dict):
            raise IntegrationRequirementFailure("D3I-BRIDGE-LAUNCHER", f"missing launcher: {name}")
        _require_record(repo, record, f"D3I-BRIDGE-LAUNCHER-{name.upper()}")
    if bridge.get("cli_propagation", {}).get("bundle_v2_user_argument_permitted") is not False:
        raise IntegrationRequirementFailure("D3I-BRIDGE-CLI-V2", "bundle v2 accepted independently")
    return {"bridge": bridge, "chain": chain, "bridge_sha256": bridge_sha256}


def bridge_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Thayer D3 execution bridge v4",
        "type": "object",
        "required": [
            "schema_version",
            "bridge_version",
            "phase",
            "producing_run",
            "repository_head",
            "authorities",
            "precedence",
            "launchers",
            "cli_propagation",
            "scientific_contract",
            "synthetic_preflight",
            "flow_invariants",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "bridge_version": {"const": BRIDGE_VERSION},
            "phase": {"enum": ["candidate", "frozen"]},
            "authorities": {"type": "object"},
            "precedence": {"type": "object"},
            "launchers": {"type": "object"},
        },
        "additionalProperties": False,
    }
