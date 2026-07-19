"""Focused executable D3 registry, metadata, and consumer-contract tests."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

from src.d3_artifact_metadata import inspect_npz, inspect_torch_zip
from src.d3_executable_contract import (
    architecture_audit,
    load_project_contract_modules,
    metadata_preflight_required_ids,
    model_preflight_required_ids,
)
from src.d3_requirement_registry import (
    RequirementFailure,
    required_ids,
    validate_capsule_requirements,
    validate_registry,
)


REPO = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExecutableD3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        registry_path = os.environ.get("THAYER_D3E_REGISTRY")
        capsule_path = os.environ.get("THAYER_D3E_CAPSULE")
        if not registry_path or not capsule_path:
            raise unittest.SkipTest("THAYER_D3E_REGISTRY and THAYER_D3E_CAPSULE are required")
        cls.registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        cls.capsule = json.loads(Path(capsule_path).read_text(encoding="utf-8"))

    def test_registry_and_capsule_exact_set(self) -> None:
        validate_registry(self.registry)
        self.assertEqual(validate_capsule_requirements(self.capsule, self.registry), required_ids(self.registry))

    def test_every_component_consumes_identical_set(self) -> None:
        builder = load_script("test_d3_builder", REPO / "scripts/build_d3_executable_capsule_v2.py")
        validator = load_script("test_d3_validator", REPO / "scripts/validate_d3_executable_capsule_v2.py")
        consumer = load_script("test_d3_consumer", REPO / "scripts/run_thayer_d3_synthetic_preflight.py")
        launcher = load_script("test_d3_launcher", REPO / "scripts/run_thayer_authoritative_d3_v2.py")
        expected = required_ids(self.registry)
        sets = {
            frozenset(builder.required_ids(self.registry)),
            frozenset(validator.required_ids(self.registry)),
            frozenset(consumer.required_ids(self.registry)),
            frozenset(launcher.required_ids(self.registry)),
            metadata_preflight_required_ids(self.registry),
            model_preflight_required_ids(self.registry),
        }
        self.assertEqual(sets, {expected})

    def test_missing_requirement_fails_with_its_id(self) -> None:
        mutated = json.loads(json.dumps(self.capsule))
        identifier = sorted(required_ids(self.registry))[0]
        del mutated["requirements"][identifier]
        with self.assertRaises(RequirementFailure) as context:
            validate_capsule_requirements(mutated, self.registry)
        self.assertEqual(context.exception.requirement_id, identifier)

    def test_container_headers_read_no_payloads(self) -> None:
        npz = inspect_npz(REPO / "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/optimized_features/d1_penultimate_endpoints.npz")
        self.assertEqual({row["array_payload_bytes_read"] for row in npz["members"]}, {0})
        pt = inspect_torch_zip(REPO / "outputs/runs/thayer_repository_integrity_20260713_031653/fixed_feature_retry/cached_features_superseding_v4.pt")
        self.assertEqual(pt["tensor_storage_payload_bytes_read"], 0)

    def test_exact_architecture_parameter_counts(self) -> None:
        modules = load_project_contract_modules(REPO)
        audit = architecture_audit(
            modules,
            REPO / "outputs/runs/thayer_output_parameterization_20260713_023120/checkpoints/ambiguous_one_scene_square.pth",
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["parameter_counts"], [46470, 46470])


if __name__ == "__main__":
    unittest.main()
