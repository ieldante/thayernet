#!/usr/bin/env python3
"""Synthetic-only actual D3 consumer for executable contract v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.d3_executable_contract import run_synthetic_preflight, sha256
from src.d3_requirement_registry import (
    RequirementFailure,
    required_ids_for_component,
    validate_capsule_requirements,
    validate_registry,
)


def required_ids(registry: dict[str, object]) -> frozenset[str]:
    return required_ids_for_component(registry, "synthetic_consumer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--capsule-sha256", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if sha256(args.capsule) != args.capsule_sha256:
            raise RequirementFailure("capsule.identity.sha256", "capsule hash mismatch")
        if sha256(args.registry) != args.registry_sha256:
            raise RequirementFailure("registry.identity.sha256", "registry hash mismatch")
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        capsule = json.loads(args.capsule.read_text(encoding="utf-8"))
        validate_registry(registry)
        required_ids(registry)
        validate_capsule_requirements(capsule, registry)
        if args.validate_only:
            print(json.dumps({"status": "PASS", "model_execution_started": False}, sort_keys=True))
            return 0
        result = run_synthetic_preflight(
            repo=args.repo.resolve(), run=args.run.resolve(),
            capsule_path=args.capsule.resolve(), registry_path=args.registry.resolve(),
            capsule_sha256=args.capsule_sha256, registry_sha256=args.registry_sha256,
            checkpoint_path=args.initial_checkpoint.resolve(), spawn_replay=True,
        )
        if result["status"] != "PASS":
            raise RuntimeError("synthetic consumer did not pass")
        print("ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED")
        print("READY_FOR_AUTHORITATIVE_D3_EXECUTION")
        return 0
    except RequirementFailure as exc:
        print(json.dumps({
            "status": "REJECTED",
            "canonical_requirement_id": exc.requirement_id,
            "message": exc.message,
            "model_execution_started": False,
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
