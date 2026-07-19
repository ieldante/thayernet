"""CSV-required dtype tests for the independent D3 v4.1 R1 repair."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def dtype_module():
    name = (
        "src.d3_contract_tokens_v41"
        if os.environ.get("D3_I41R1_TARGET") == "candidate_002"
        else "src.d3_contract_tokens_v41r1"
    )
    return importlib.import_module(name)


class D3ContractTokensV41R1RequiredTests(unittest.TestCase):
    def test_dtype_object_equality_is_authoritative(self) -> None:
        module = dtype_module()
        result = module.numpy_dtype_contract_equal("float32", "<f4")
        self.assertTrue(result.equal)
        self.assertEqual(result.equality_basis, "numpy_dtype_object_equality")
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "numpy_dtype_contract_equal"
        )
        object_comparison = any(
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "actual_dtype"
            and any(isinstance(operator, ast.Eq) for operator in node.ops)
            and any(
                isinstance(comparator, ast.Name)
                and comparator.id == "expected_dtype"
                for comparator in node.comparators
            )
            for node in ast.walk(function)
        )
        self.assertTrue(object_comparison)

    def test_member_contract_logs_original_and_canonical_dtype(self) -> None:
        result = dtype_module().numpy_dtype_contract_equal(
            np.zeros(1, dtype=np.float32), "<f4"
        )
        self.assertEqual(result.original_actual_token, "float32")
        self.assertEqual(result.original_expected_token, "<f4")
        self.assertEqual(result.canonical_actual_token, "<f4")
        self.assertEqual(result.canonical_expected_token, "<f4")
        self.assertEqual(result.actual_kind, "f")
        self.assertEqual(result.expected_kind, "f")
        self.assertEqual(result.actual_itemsize, 4)
        self.assertEqual(result.expected_itemsize, 4)
        self.assertEqual(result.equality_basis, "numpy_dtype_object_equality")

    def test_object_dtype_rejected_by_default(self) -> None:
        module = dtype_module()
        with self.assertRaises(module.UnsupportedNumpyDType):
            module.coerce_numpy_dtype(np.dtype(object))

    def test_structured_dtype_rejected_by_default(self) -> None:
        module = dtype_module()
        with self.assertRaises(module.UnsupportedNumpyDType):
            module.coerce_numpy_dtype(np.dtype([("value", "<f4")]))

    def test_subarray_dtype_rejected_by_default(self) -> None:
        module = dtype_module()
        with self.assertRaises(module.UnsupportedNumpyDType):
            module.coerce_numpy_dtype(np.dtype((np.float32, (2,))))


if __name__ == "__main__":
    unittest.main()
