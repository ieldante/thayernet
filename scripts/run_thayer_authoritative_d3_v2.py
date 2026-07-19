#!/usr/bin/env python3
"""Bundle-only future authoritative D3 launcher with synthetic preflight mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.d3_executable_contract import sha256, write_text_x
from src.d3_requirement_registry import (
    RequirementFailure,
    required_ids_for_component,
    validate_capsule_requirements,
    validate_registry,
)


def required_ids(registry: dict[str, object]) -> frozenset[str]:
    return required_ids_for_component(registry, "scientific_launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-preflight-only", action="store_true")
    return parser.parse_args()


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def main() -> int:
    args = parse_args()
    bundle_path = args.bundle.resolve()
    if sha256(bundle_path) != args.bundle_sha256:
        raise RequirementFailure("bundle.identity.sha256", "bundle hash mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    repo = _resolve(bundle_path.parents[0], bundle["repository_root_relative_to_bundle"]).resolve()
    schema_path = _resolve(repo, bundle["bundle_schema"]["path"])
    if sha256(schema_path) != bundle["bundle_schema"]["sha256"]:
        raise RequirementFailure("bundle.schema.sha256", "bundle schema hash mismatch")
    registry_path = _resolve(repo, bundle["requirement_registry"]["path"])
    capsule_path = _resolve(repo, bundle["capsule_v2"]["path"])
    checkpoint_path = _resolve(repo, bundle["initial_state"]["path"])
    for identifier, path, expected in (
        ("registry.identity.sha256", registry_path, bundle["requirement_registry"]["sha256"]),
        ("capsule.identity.sha256", capsule_path, bundle["capsule_v2"]["sha256"]),
        ("initial_state.checkpoint_sha256", checkpoint_path, bundle["initial_state"]["sha256"]),
    ):
        if sha256(path) != expected:
            raise RequirementFailure(identifier, f"bundle member hash mismatch: {path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    ids = required_ids(registry)
    if validate_capsule_requirements(capsule, registry) != ids:
        raise RequirementFailure("registry.required_set_equality", "scientific launcher set mismatch")
    for identifier, record in bundle["validated_results"].items():
        path = _resolve(repo, record["path"])
        if sha256(path) != record["sha256"]:
            raise RequirementFailure(identifier, "validated-result hash mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not args.synthetic_preflight_only:
        print("READY_FOR_AUTHORITATIVE_D3_EXECUTION")
        raise SystemExit("SCIENTIFIC_D3_REQUIRES_A_SEPARATELY_PREREGISTERED_FUTURE_CAMPAIGN")
    command = [
        str(repo / ".venv-btk/bin/python"), "-B", str(repo / "scripts/run_thayer_d3_synthetic_preflight.py"),
        "--repo", str(repo), "--run", str(output), "--capsule", str(capsule_path),
        "--capsule-sha256", bundle["capsule_v2"]["sha256"],
        "--registry", str(registry_path), "--registry-sha256", bundle["requirement_registry"]["sha256"],
        "--initial-checkpoint", str(checkpoint_path),
    ]
    environment = dict(os.environ)
    environment.update({
        "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    completed = subprocess.run(command, cwd=repo, env=environment, text=True, capture_output=True, check=False)
    write_text_x(output / "launcher_stdout.txt", completed.stdout)
    write_text_x(output / "launcher_stderr.txt", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"synthetic preflight failed: {completed.stderr}")
    markers = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if markers[-2:] != ["ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED", "READY_FOR_AUTHORITATIVE_D3_EXECUTION"]:
        raise RuntimeError(f"synthetic preflight markers invalid: {markers}")
    print("ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED")
    print("READY_FOR_AUTHORITATIVE_D3_EXECUTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
