"""Structural tests for the R1 independent validator boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
VALIDATOR = REPO / "scripts/validate_thayer_d3_i41r1_candidate.py"


class ThayerD3I41R1IndependentValidatorTests(unittest.TestCase):
    def source(self) -> str:
        return VALIDATOR.read_text(encoding="utf-8")

    def test_validator_reads_frozen_audit_and_csv_directly(self) -> None:
        source = self.source()
        self.assertIn("independent_contract_audit_v2.json", source)
        self.assertIn("v41_required_test_name_audit_v2.csv", source)

    def test_validator_uses_independent_numpy_dtype_oracle(self) -> None:
        source = self.source()
        self.assertIn("np.dtype(actual) == np.dtype(expected)", source)
        self.assertIn("object_equality_ast", source)

    def test_validator_writes_eligibility_result_itself(self) -> None:
        source = self.source()
        self.assertIn("independent_validator/candidate_validation.json", source)
        worker = (
            REPO / "scripts/run_thayer_scientific_d3_process_v41r1.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("independent_validator/candidate_validation.json", worker)

    def test_validator_freezes_all_required_markers(self) -> None:
        source = self.source()
        for marker in (
            "ALL_V41_INDEPENDENT_AUDIT_ROWS_PASS",
            "ALL_REQUIRED_REGRESSION_TESTS_COLLECTED_AND_PASSED",
            "NUMPY_DTYPE_OBJECT_EQUALITY_AUTHORITATIVE",
            "PRODUCTION_CHECKPOINT_ADAPTER_PREWARM_PASS",
            "READY_FOR_V41_SCIENTIFIC_PAYLOAD_ACCESS",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
