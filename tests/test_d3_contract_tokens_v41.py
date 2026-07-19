"""Regression tests for semantic NumPy dtype contract tokens in D3 v4.1."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class D3ContractTokenV41Tests(unittest.TestCase):
    def tokens(self):
        try:
            return importlib.import_module("src.d3_contract_tokens_v41")
        except ModuleNotFoundError as exc:
            self.fail(f"v4.1 contract-token helper is absent: {exc}")

    def test_dtype_float32_display_and_lt_f4_contract_are_equivalent(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal("float32", "<f4")
        self.assertTrue(result.equal)
        self.assertEqual(result.canonical_actual_token, "<f4")

    def test_dtype_numpy_float32_scalar_type_and_lt_f4_are_equivalent(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal(np.float32, "<f4")
        self.assertTrue(result.equal)
        self.assertEqual(result.canonical_actual_token, "<f4")

    def test_dtype_numpy_dtype_float32_and_lt_f4_are_equivalent(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal(np.dtype("float32"), "<f4")
        self.assertTrue(result.equal)

    def test_dtype_native_equal_f4_and_lt_f4_are_equivalent_on_little_endian(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal("=f4", "<f4")
        if sys.byteorder == "little":
            self.assertTrue(result.equal)
        else:
            self.assertFalse(result.equal)

    def test_dtype_big_endian_f4_is_rejected_against_little_endian_contract(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal(">f4", "<f4")
        if sys.byteorder == "little":
            self.assertFalse(result.equal)
            self.assertEqual(result.canonical_actual_token, ">f4")

    def test_dtype_float64_is_rejected_against_float32_contract(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal("float64", "<f4")
        self.assertFalse(result.equal)

    def test_dtype_int32_is_rejected_against_float32_contract(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal("int32", "<f4")
        self.assertFalse(result.equal)

    def test_dtype_comparison_does_not_use_str_dtype_display(self) -> None:
        source = (REPO / "src/d3_contract_tokens_v41.py").read_text(encoding="utf-8")
        self.assertNotIn("str(dtype)", source)
        self.assertNotIn("dtype.name", source)
        self.assertIn("np.dtype", source)
        self.assertIn(".str", source)

    def test_member_contract_preserves_original_expected_token(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal(np.zeros(1, dtype=np.float32), "<f4")
        self.assertEqual(result.original_expected_token, "<f4")

    def test_member_contract_logs_expected_actual_and_canonical_dtype(self) -> None:
        result = self.tokens().numpy_dtype_contract_equal(np.zeros(1, dtype=np.float32), "<f4")
        payload = result.to_dict()
        self.assertEqual(
            set(payload),
            {
                "original_actual_token",
                "original_expected_token",
                "canonical_actual_token",
                "canonical_expected_token",
                "equal",
                "platform_byte_order",
                "failure_reason",
            },
        )
        self.assertEqual(payload["original_actual_token"], "float32")
        self.assertTrue(payload["equal"])


if __name__ == "__main__":
    unittest.main()
