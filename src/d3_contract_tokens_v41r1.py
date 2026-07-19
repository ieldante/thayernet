"""Independent-audit-compliant NumPy dtype contracts for D3 v4.1 R1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from typing import Any, Optional

import numpy as np


class UnsupportedNumpyDType(TypeError):
    """Raised when a dtype is outside the frozen primitive contract domain."""


@dataclass(frozen=True)
class NumpyDTypeContractResult:
    """Immutable, complete report for one dtype-object comparison."""

    original_actual_token: str
    original_expected_token: str
    canonical_actual_token: Optional[str]
    canonical_expected_token: Optional[str]
    actual_kind: Optional[str]
    expected_kind: Optional[str]
    actual_itemsize: Optional[int]
    expected_itemsize: Optional[int]
    actual_byteorder: Optional[str]
    expected_byteorder: Optional[str]
    platform_byteorder: str
    equal: bool
    equality_basis: str
    failure_reason: Optional[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _original_dtype_token(value: Any) -> str:
    """Preserve the caller-facing representation for reporting only."""

    if isinstance(value, str):
        return value
    if isinstance(value, type) and issubclass(value, np.generic):
        return f"numpy.{value.__name__}"
    if hasattr(value, "dtype") and not isinstance(value, type):
        return str(value.dtype)
    return str(value)


def coerce_numpy_dtype(value: Any) -> np.dtype[Any]:
    """Coerce one supported primitive dtype and reject compound categories."""

    candidate = (
        value.dtype
        if hasattr(value, "dtype") and not isinstance(value, type)
        else value
    )
    try:
        dtype = value if isinstance(value, np.dtype) else np.dtype(candidate)
    except (TypeError, ValueError) as exc:
        raise UnsupportedNumpyDType(
            f"unsupported NumPy dtype input: {value!r}"
        ) from exc
    if dtype.fields is not None:
        raise UnsupportedNumpyDType(
            "structured NumPy dtypes are not contract-supported"
        )
    if dtype.hasobject:
        raise UnsupportedNumpyDType(
            "object-containing NumPy dtypes are not contract-supported"
        )
    if dtype.subdtype is not None:
        raise UnsupportedNumpyDType(
            "subarray NumPy dtypes are not contract-supported"
        )
    return dtype


def canonical_numpy_dtype_token(value: Any) -> str:
    """Return ``dtype.str`` for logging, never as the equality authority."""

    return coerce_numpy_dtype(value).str


def numpy_dtype_contract_equal(
    actual: Any, expected: Any
) -> NumpyDTypeContractResult:
    """Compare with NumPy dtype objects and preserve reporting representations."""

    original_actual = _original_dtype_token(actual)
    original_expected = _original_dtype_token(expected)
    equality_basis = "numpy_dtype_object_equality"
    try:
        actual_dtype = coerce_numpy_dtype(actual)
    except UnsupportedNumpyDType as exc:
        return NumpyDTypeContractResult(
            original_actual,
            original_expected,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            sys.byteorder,
            False,
            equality_basis,
            f"unsupported_actual_dtype: {exc}",
            "REJECTED",
        )
    try:
        expected_dtype = coerce_numpy_dtype(expected)
    except UnsupportedNumpyDType as exc:
        return NumpyDTypeContractResult(
            original_actual,
            original_expected,
            actual_dtype.str,
            None,
            actual_dtype.kind,
            None,
            actual_dtype.itemsize,
            None,
            actual_dtype.byteorder,
            None,
            sys.byteorder,
            False,
            equality_basis,
            f"unsupported_expected_dtype: {exc}",
            "REJECTED",
        )

    # This NumPy dtype-object comparison is the sole equality authority.
    equal = actual_dtype == expected_dtype
    return NumpyDTypeContractResult(
        original_actual,
        original_expected,
        actual_dtype.str,
        expected_dtype.str,
        actual_dtype.kind,
        expected_dtype.kind,
        actual_dtype.itemsize,
        expected_dtype.itemsize,
        actual_dtype.byteorder,
        expected_dtype.byteorder,
        sys.byteorder,
        bool(equal),
        equality_basis,
        None if equal else "numpy_dtype_object_mismatch",
        "PASS" if equal else "MISMATCH",
    )


__all__ = [
    "NumpyDTypeContractResult",
    "UnsupportedNumpyDType",
    "canonical_numpy_dtype_token",
    "coerce_numpy_dtype",
    "numpy_dtype_contract_equal",
]
