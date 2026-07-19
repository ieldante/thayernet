#!/usr/bin/env python3
"""Build the append-only executable D3 capsule v2 from one canonical registry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.d3_executable_contract import sha256, write_json_x
from src.d3_requirement_registry import (
    records_by_id,
    required_ids_for_component,
    validate_registry,
)


def required_ids(registry: dict[str, object]) -> frozenset[str]:
    return required_ids_for_component(registry, "builder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capsule-v1", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hash-chain", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    validate_registry(registry)
    ids = required_ids(registry)
    records = records_by_id(registry)
    v1 = json.loads(args.capsule_v1.read_text(encoding="utf-8"))
    source_paths = {
        "registry": args.registry,
        "builder": repo / "scripts/build_d3_executable_capsule_v2.py",
        "validator": repo / "scripts/validate_d3_executable_capsule_v2.py",
        "synthetic_consumer": repo / "scripts/run_thayer_d3_synthetic_preflight.py",
        "scientific_launcher": repo / "scripts/run_thayer_authoritative_d3_v2.py",
        "executable_contract": repo / "src/d3_executable_contract.py",
        "requirement_registry_source": repo / "src/d3_requirement_registry.py",
        "artifact_metadata": repo / "src/d3_artifact_metadata.py",
        "replay": repo / "scripts/replay_thayer_d3_synthetic_checkpoint.py",
    }
    implementation = {
        name: {"path": str(path.relative_to(repo)), "sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in source_paths.items()
    }
    capsule = {
        "capsule_identity": {
            "capsule_id": "thayer-d3-executable-capsule-v2",
            "schema_version": "thayer-d3-executable-capsule-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "producer": "scripts/build_d3_executable_capsule_v2.py",
            "append_only": True,
            "supersedes_without_modifying": str(args.capsule_v1.relative_to(repo)),
            "scientific_d3_executed": False,
        },
        "scientific_capsule_v1": v1,
        "requirement_registry": {
            "path": str(args.registry.relative_to(repo)),
            "sha256": sha256(args.registry),
            "required_count": len(ids),
            "schema_version": registry["schema_version"],
        },
        "requirements": {identifier: records[identifier]["expected_value"] for identifier in sorted(ids)},
        "implementation_contract": implementation,
        "completeness": {
            "required_count": len(ids),
            "present_count": len(ids),
            "unresolved_count": 0,
            "implicit_default_count": 0,
            "placeholder_count": 0,
            "builder_validator_preflight_consumer_set_equality_required": True,
        },
        "scope": {
            "scientific_array_values_loaded_in_builder": 0,
            "scientific_d3_steps": 0,
            "synthetic_preflight_only": True,
            "automatic_scientific_continuation": False,
        },
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "thayer-d3-executable-capsule-v2.schema.json",
        "title": "Thayer executable D3 capsule v2",
        "type": "object",
        "required": [
            "capsule_identity", "scientific_capsule_v1", "requirement_registry",
            "requirements", "implementation_contract", "completeness", "scope",
        ],
        "properties": {
            "capsule_identity": {"type": "object", "required": ["capsule_id", "schema_version", "created_utc", "producer", "append_only", "supersedes_without_modifying", "scientific_d3_executed"]},
            "scientific_capsule_v1": {"type": "object"},
            "requirement_registry": {"type": "object", "required": ["path", "sha256", "required_count", "schema_version"]},
            "requirements": {
                "type": "object",
                "required": sorted(ids),
                "additionalProperties": False,
                "properties": {identifier: {} for identifier in sorted(ids)},
            },
            "implementation_contract": {"type": "object", "required": sorted(implementation)},
            "completeness": {"type": "object", "required": ["required_count", "present_count", "unresolved_count", "implicit_default_count", "placeholder_count", "builder_validator_preflight_consumer_set_equality_required"]},
            "scope": {"type": "object", "required": ["scientific_array_values_loaded_in_builder", "scientific_d3_steps", "synthetic_preflight_only", "automatic_scientific_continuation"]},
        },
        "additionalProperties": False,
    }
    write_json_x(args.schema, schema)
    write_json_x(args.capsule, capsule)
    manifest = {
        "schema_version": "thayer-d3-executable-capsule-manifest-v2",
        "files": {
            "capsule": {"path": str(args.capsule.relative_to(repo)), "bytes": args.capsule.stat().st_size, "sha256": sha256(args.capsule)},
            "schema": {"path": str(args.schema.relative_to(repo)), "bytes": args.schema.stat().st_size, "sha256": sha256(args.schema)},
            "registry": {"path": str(args.registry.relative_to(repo)), "bytes": args.registry.stat().st_size, "sha256": sha256(args.registry)},
        },
        "required_count": len(ids),
        "scientific_array_values_loaded": 0,
    }
    write_json_x(args.manifest, manifest)
    chain = {
        "schema_version": "thayer-d3-executable-capsule-hash-chain-v2",
        "capsule_sha256": sha256(args.capsule),
        "schema_sha256": sha256(args.schema),
        "registry_sha256": sha256(args.registry),
        "manifest_sha256": sha256(args.manifest),
        "implementation_sha256": {name: record["sha256"] for name, record in implementation.items()},
    }
    write_json_x(args.hash_chain, chain)
    print(json.dumps({
        "status": "PASS", "required_count": len(ids),
        "capsule_sha256": sha256(args.capsule), "schema_sha256": sha256(args.schema),
        "manifest_sha256": sha256(args.manifest), "hash_chain_sha256": sha256(args.hash_chain),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
