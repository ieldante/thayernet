#!/usr/bin/env python3
"""Validate the executable D3 capsule v2 against the one canonical registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.d3_executable_contract import sha256
from src.d3_requirement_registry import (
    RequirementFailure,
    required_ids_for_component,
    validate_capsule_requirements,
    validate_registry,
)


def required_ids(registry: dict[str, object]) -> frozenset[str]:
    return required_ids_for_component(registry, "validator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hash-chain", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        capsule = json.loads(args.capsule.read_text(encoding="utf-8"))
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        chain = json.loads(args.hash_chain.read_text(encoding="utf-8"))
        validate_registry(registry)
        ids = required_ids(registry)
        observed = validate_capsule_requirements(capsule, registry)
        if observed != ids:
            raise RequirementFailure("registry.required_set_equality", "validator set differs from capsule set")
        schema_required = set(schema["properties"]["requirements"]["required"])
        if schema_required != ids:
            raise RequirementFailure("registry.required_set_equality", "schema set differs from registry set")
        core_required = set(schema["required"])
        if not core_required.issubset(capsule):
            raise RequirementFailure("capsule.schema", "capsule is missing a schema-required top-level field")
        if set(capsule) != set(schema["properties"]):
            raise RequirementFailure("capsule.schema", "capsule has an unexpected top-level field")
        expected_hashes = {
            "capsule": sha256(args.capsule), "schema": sha256(args.schema), "registry": sha256(args.registry),
        }
        for name, expected in expected_hashes.items():
            if manifest["files"][name]["sha256"] != expected:
                raise RequirementFailure(f"capsule.manifest.{name}", "manifest hash mismatch")
        if chain["capsule_sha256"] != expected_hashes["capsule"]:
            raise RequirementFailure("capsule.hash_chain.capsule", "capsule hash-chain mismatch")
        if chain["schema_sha256"] != expected_hashes["schema"]:
            raise RequirementFailure("capsule.hash_chain.schema", "schema hash-chain mismatch")
        if chain["registry_sha256"] != expected_hashes["registry"]:
            raise RequirementFailure("capsule.hash_chain.registry", "registry hash-chain mismatch")
        if chain["manifest_sha256"] != sha256(args.manifest):
            raise RequirementFailure("capsule.hash_chain.manifest", "manifest hash-chain mismatch")
        result = {
            "status": "PASS", "required_count": len(ids), "required_set_equal": True,
            "capsule_sha256": expected_hashes["capsule"], "schema_sha256": expected_hashes["schema"],
            "registry_sha256": expected_hashes["registry"], "manifest_sha256": sha256(args.manifest),
            "hash_chain_sha256": sha256(args.hash_chain),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    except RequirementFailure as exc:
        print(json.dumps({"status": "FAIL", "canonical_requirement_id": exc.requirement_id, "message": exc.message}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
