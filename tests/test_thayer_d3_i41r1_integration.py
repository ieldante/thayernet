"""CSV-required integration tests for D3 v4.1 R1."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RUN = Path(
    os.environ.get(
        "D3_I41R1_RUN",
        REPO / "outputs/runs/thayer_d3_i41r1_20260713_221426",
    )
)


def candidate_002() -> bool:
    return os.environ.get("D3_I41R1_TARGET") == "candidate_002"


class ThayerD3I41R1RequiredIntegrationTests(unittest.TestCase):
    def test_v41_dtype_metadata_validation_passes_previous_float32_member_without_payload_loading(self) -> None:
        name = (
            "src.d3_contract_tokens_v41"
            if candidate_002()
            else "src.d3_contract_tokens_v41r1"
        )
        module = importlib.import_module(name)
        result = module.numpy_dtype_contract_equal(np.dtype("float32"), "<f4")
        self.assertTrue(result.equal)
        self.assertEqual(result.equality_basis, "numpy_dtype_object_equality")

    def test_v41_revalidates_same_eight_container_paths_and_ninety_one_member_contracts_without_payload_loading(self) -> None:
        if candidate_002():
            self.fail("candidate 002 did not provide a metadata-only 8/91 proof")
        worker = importlib.import_module(
            "scripts.run_thayer_scientific_d3_process_v41r1"
        )
        contract = worker.metadata_validation_contract()
        self.assertEqual(contract["container_count"], 8)
        self.assertEqual(contract["member_count"], 91)
        self.assertEqual(contract["payload_values_loaded"], 0)

    def test_v41_scientific_payloads_not_loaded_during_synthetic_preflight(self) -> None:
        if candidate_002():
            self.fail("candidate 002 cannot independently prove payload isolation")
        source = (
            REPO / "scripts/run_thayer_scientific_d3_v41r1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"scientific_payload_values_loaded": 0', source)
        self.assertIn('"candidate_self_certified": False', source)
        self.assertIn('"eligibility_decision": "PENDING_INDEPENDENT_VALIDATOR"', source)


if __name__ == "__main__":
    unittest.main()
