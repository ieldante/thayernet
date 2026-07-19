"""Versioned, batch-invariant hashes for one scientific CHW tensor sample."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - NumPy-only environments remain supported.
    torch = None


SCHEMA_VERSION = "thayer-per-sample-tensor-sha256-v1"
Layout = Literal["CHW", "HWC", "NCHW", "NHWC"]


def _numpy(value: object) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").numpy()
    return np.asarray(value)


def canonical_chw_float32(
    value: object,
    *,
    layout: Layout = "CHW",
    sample_index: int | None = None,
) -> np.ndarray:
    """Return one contiguous little-endian float32 sample in CHW order.

    Batched layouts require an explicit sample index. The selected sample is
    copied before canonical conversion, so batch position, batch size, strides,
    device, and storage layout cannot enter the result.
    """

    array = _numpy(value)
    if layout == "CHW":
        if sample_index is not None or array.ndim != 3:
            raise ValueError("CHW requires one rank-3 sample and no sample_index")
        sample = array
    elif layout == "HWC":
        if sample_index is not None or array.ndim != 3:
            raise ValueError("HWC requires one rank-3 sample and no sample_index")
        sample = np.moveaxis(array, -1, 0)
    elif layout == "NCHW":
        if sample_index is None or array.ndim != 4:
            raise ValueError("NCHW requires a rank-4 batch and sample_index")
        sample = array[sample_index]
    elif layout == "NHWC":
        if sample_index is None or array.ndim != 4:
            raise ValueError("NHWC requires a rank-4 batch and sample_index")
        sample = np.moveaxis(array[sample_index], -1, 0)
    else:  # pragma: no cover - guarded by the public type and runtime check.
        raise ValueError(f"unsupported layout: {layout}")
    if sample.ndim != 3:
        raise ValueError(f"canonical scientific samples must be CHW, got {sample.shape}")
    if not np.issubdtype(sample.dtype, np.number):
        raise TypeError(f"numeric tensor required, got {sample.dtype}")
    canonical = np.asarray(sample, dtype=np.dtype("<f4"), order="C")
    if not np.all(np.isfinite(canonical)):
        raise ValueError("non-finite tensors are not hashable under the scientific contract")
    return np.ascontiguousarray(canonical)


def canonical_tensor_sha256(
    value: object,
    *,
    layout: Layout = "CHW",
    sample_index: int | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Hash one canonical sample, including schema, CHW shape, and `<f4` dtype."""

    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical hash schema: {schema_version}")
    sample = canonical_chw_float32(value, layout=layout, sample_index=sample_index)
    header = {
        "schema_version": schema_version,
        "dimension_order": "CHW",
        "shape": list(sample.shape),
        "dtype": "<f4",
        "byte_order": "little",
        "memory_order": "C",
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(sample.tobytes(order="C"))
    return digest.hexdigest()

