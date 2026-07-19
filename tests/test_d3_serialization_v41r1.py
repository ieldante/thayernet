"""CSV-required production-checkpoint tests for D3 v4.1 R1."""

from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
RUN = Path(
    os.environ.get(
        "D3_I41R1_RUN",
        REPO / "outputs/runs/thayer_d3_i41r1_20260713_221426",
    )
)


def candidate_002() -> bool:
    return os.environ.get("D3_I41R1_TARGET") == "candidate_002"


def adapter_module():
    if candidate_002():
        raise AssertionError("candidate 002 has no production checkpoint adapter")
    return importlib.import_module("src.d3_checkpoint_adapter_v41r1")


def worker_module():
    name = (
        "scripts.run_thayer_scientific_d3_process_v41"
        if candidate_002()
        else "scripts.run_thayer_scientific_d3_process_v41r1"
    )
    return importlib.import_module(name)


def worker_source() -> str:
    path = REPO / (
        "scripts/run_thayer_scientific_d3_process_v41.py"
        if candidate_002()
        else "scripts/run_thayer_scientific_d3_process_v41r1.py"
    )
    return path.read_text(encoding="utf-8")


class D3SerializationV41R1RequiredTests(unittest.TestCase):
    def test_checkpoint_weights_only_and_map_location_match_frozen_contract(self) -> None:
        adapter = adapter_module()
        self.assertEqual(adapter.FROZEN_MAP_LOCATION, "cpu")
        self.assertTrue(adapter.FROZEN_WEIGHTS_ONLY)

    def test_observed_torch_utils_serialization_modules_are_imported_when_present(self) -> None:
        if candidate_002():
            self.fail("candidate 002 lacked this exact mandatory regression test")
        source = worker_source()
        self.assertIn("torch.utils.serialization", source)
        self.assertIn("torch.utils.serialization.config", source)

    def test_prewarm_payload_matches_frozen_checkpoint_schema(self) -> None:
        adapter = adapter_module()
        payload = adapter.build_synthetic_production_checkpoint_payload()
        adapter.validate_production_checkpoint_payload(payload)
        self.assertEqual(set(payload), set(adapter.PRODUCTION_CHECKPOINT_KEYS))
        self.assertEqual(payload["schema_version"], "thayer-d3-checkpoint-v4")

    def test_prewarm_uses_no_scientific_checkpoint_or_model_tensor(self) -> None:
        adapter = adapter_module()
        manifest = adapter.synthetic_payload_manifest()
        self.assertFalse(manifest["scientific_checkpoint_opened"])
        self.assertFalse(manifest["scientific_model_tensor_used"])
        self.assertEqual(manifest["protected_identifier_count"], 0)

    def test_production_checkpoint_reader_is_prewarmed_in_bootstrap_scratch(self) -> None:
        worker = worker_module()
        with tempfile.TemporaryDirectory() as temporary:
            result = worker.run_serialization_contract_probe(
                Path(temporary), strict_verify=True
            )
        self.assertGreaterEqual(result["adapter_trace"]["reader_calls"], 2)
        self.assertTrue(result["bootstrap_reader_pass"])

    def test_production_checkpoint_writer_is_prewarmed_in_bootstrap_scratch(self) -> None:
        worker = worker_module()
        with tempfile.TemporaryDirectory() as temporary:
            result = worker.run_serialization_contract_probe(
                Path(temporary), strict_verify=True
            )
        self.assertEqual(result["adapter_trace"]["writer_calls"], 1)
        self.assertTrue(result["bootstrap_writer_pass"])

    def test_strict_phase_has_complete_serialization_module_set_loaded(self) -> None:
        worker = worker_module()
        with tempfile.TemporaryDirectory() as temporary:
            result = worker.run_serialization_contract_probe(
                Path(temporary), strict_verify=True
            )
        self.assertEqual(
            set(result["required_serialization_modules"]),
            set(result["loaded_before_strict"]),
        )

    def test_strict_phase_production_checkpoint_path_causes_zero_external_pyc_reads(self) -> None:
        worker = worker_module()
        with tempfile.TemporaryDirectory() as temporary:
            result = worker.run_serialization_contract_probe(
                Path(temporary), strict_verify=True
            )
        self.assertEqual(result["strict_external_pyc_reads"], [])

    def test_strict_phase_production_checkpoint_path_causes_zero_new_imports(self) -> None:
        worker = worker_module()
        with tempfile.TemporaryDirectory() as temporary:
            result = worker.run_serialization_contract_probe(
                Path(temporary), strict_verify=True
            )
        self.assertEqual(result["strict_new_imports"], [])

    def test_torch_serialization_is_imported_during_bootstrap(self) -> None:
        source = worker_source()
        tree = ast.parse(source)
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("torch.serialization", names)

    def test_unobserved_serialization_modules_are_not_broadly_imported(self) -> None:
        source = worker_source()
        tree = ast.parse(source)
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.startswith("torch")
        }
        self.assertEqual(
            names,
            {
                "torch",
                "torch.serialization",
                "torch.utils.serialization",
                "torch.utils.serialization.config",
            },
        )


if __name__ == "__main__":
    unittest.main()
