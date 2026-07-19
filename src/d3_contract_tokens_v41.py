"""Canonical NumPy dtype contract tokens for the D3 v4.1 execution path.

Only dtype token interpretation belongs here. Shapes, member names, semantic
roles, values, hashes, units, and scientific thresholds remain outside this
module and retain their frozen comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from typing import Any, Optional

import numpy as np


class UnsupportedNumpyDType(TypeError):
    """Raised when a dtype is outside the primitive frozen contract domain."""


@dataclass(frozen=True)
class NumpyDTypeContractResult:
    """Typed report for one semantic dtype contract comparison."""

    original_actual_token: str
    original_expected_token: str
    canonical_actual_token: Optional[str]
    canonical_expected_token: Optional[str]
    equal: bool
    platform_byte_order: str
    failure_reason: Optional[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _original_dtype_token(value: Any) -> str:
    """Preserve the caller-facing token for reporting, never comparison."""

    if hasattr(value, "dtype") and not isinstance(value, type):
        return str(value.dtype)
    if isinstance(value, str):
        return value
    if isinstance(value, type) and issubclass(value, np.generic):
        return f"numpy.{value.__name__}"
    return str(value)


def canonical_numpy_dtype_token(value: Any) -> str:
    """Return the primitive NumPy dtype token preserving kind/size/byte order."""

    candidate = value.dtype if hasattr(value, "dtype") and not isinstance(value, type) else value
    try:
        normalized = np.dtype(candidate)
    except (TypeError, ValueError) as exc:
        raise UnsupportedNumpyDType(f"unsupported NumPy dtype input: {value!r}") from exc
    if normalized.fields is not None:
        raise UnsupportedNumpyDType("structured NumPy dtypes are not contract-supported")
    if normalized.hasobject:
        raise UnsupportedNumpyDType("object NumPy dtypes are not contract-supported")
    return normalized.str


def numpy_dtype_contract_equal(actual: Any, expected: Any) -> NumpyDTypeContractResult:
    """Compare dtype tokens semantically while preserving both original tokens."""

    original_actual = _original_dtype_token(actual)
    original_expected = _original_dtype_token(expected)
    try:
        canonical_actual = canonical_numpy_dtype_token(actual)
    except UnsupportedNumpyDType as exc:
        return NumpyDTypeContractResult(
            original_actual_token=original_actual,
            original_expected_token=original_expected,
            canonical_actual_token=None,
            canonical_expected_token=None,
            equal=False,
            platform_byte_order=sys.byteorder,
            failure_reason=f"unsupported_actual_dtype: {exc}",
        )
    try:
        canonical_expected = canonical_numpy_dtype_token(expected)
    except UnsupportedNumpyDType as exc:
        return NumpyDTypeContractResult(
            original_actual_token=original_actual,
            original_expected_token=original_expected,
            canonical_actual_token=canonical_actual,
            canonical_expected_token=None,
            equal=False,
            platform_byte_order=sys.byteorder,
            failure_reason=f"unsupported_expected_dtype: {exc}",
        )
    equal = canonical_actual == canonical_expected
    return NumpyDTypeContractResult(
        original_actual_token=original_actual,
        original_expected_token=original_expected,
        canonical_actual_token=canonical_actual,
        canonical_expected_token=canonical_expected,
        equal=equal,
        platform_byte_order=sys.byteorder,
        failure_reason=None if equal else "canonical_dtype_mismatch",
    )


__all__ = [
    "NumpyDTypeContractResult",
    "UnsupportedNumpyDType",
    "canonical_numpy_dtype_token",
    "numpy_dtype_contract_equal",
]
