"""Explicit CHW and ordered-batch NCHW canonical tensor hashes for D3 HASH-R1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - NumPy-only tests remain supported.
    torch = None


SCHEMA_VERSION = "thayer-d3-explicit-canonical-tensor-hash-r1"
CANONICAL_DTYPE = "<f4"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CanonicalSampleHashRecord:
    """One sample-ID-bound CHW diagnostic hash."""

    sample_id: str
    header: dict[str, Any]
    canonical_semantic_tensor_sha256: str


@dataclass(frozen=True)
class CanonicalTensorHashRecord:
    """Canonical digest plus separately preserved raw provenance."""

    header: dict[str, Any]
    canonical_semantic_tensor_sha256: str
    raw_file_sha256: str | None
    raw_member_hash: Any
    original_dtype: str
    canonical_dtype: str
    exact_values_preserved: bool
    per_sample_hashes: tuple[CanonicalSampleHashRecord, ...]


def _numpy(value: object) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").numpy()
    return np.asarray(value)


def _nonempty(label: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _band_order(value: Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    result = list(value)
    if not result or any(not isinstance(token, str) or not token for token in result):
        raise ValueError("band_order must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("band_order must not contain duplicates")
    return result


def _provenance(raw_file_sha256: str | None, raw_member_hash: Any) -> None:
    if raw_file_sha256 is not None and not _SHA256.fullmatch(raw_file_sha256):
        raise ValueError("raw_file_sha256 must be a lowercase SHA-256 or None")
    if isinstance(raw_member_hash, str) and not raw_member_hash:
        raise ValueError("raw_member_hash must be nonempty when provided")


def _canonical_float32(value: object, *, rank: int, axis_order: str) -> tuple[np.ndarray, str]:
    array = _numpy(value)
    if array.ndim != rank:
        raise ValueError(f"{axis_order} requires rank {rank}, got rank {array.ndim} shape {array.shape}")
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise TypeError(f"{axis_order} requires float32 dtype semantics, got {array.dtype}")
    canonical = np.ascontiguousarray(array.astype(np.dtype(CANONICAL_DTYPE), copy=False))
    if not np.all(np.isfinite(canonical)):
        raise ValueError("non-finite tensors are outside the canonical hash contract")
    if not np.array_equal(array, canonical):
        raise ValueError("canonicalization changed tensor values")
    return canonical, str(array.dtype)


def _digest(header: dict[str, Any], canonical: np.ndarray) -> str:
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payload = canonical.tobytes(order="C")
    framed = (
        len(header_bytes).to_bytes(8, "big")
        + header_bytes
        + len(payload).to_bytes(8, "big")
        + payload
    )
    return hashlib.sha256(framed).hexdigest()


def canonical_chw_tensor_hash(
    value: object,
    *,
    semantic_axis_order: str,
    semantic_member_name: str,
    prompt_identity: str,
    expert_identity: str,
    band_order: Iterable[str] | None,
    canonical_dtype: str,
    raw_file_sha256: str | None = None,
    raw_member_hash: Any = None,
) -> CanonicalTensorHashRecord:
    """Hash exactly one declared CHW sample; no batch handling or axis transform."""

    if semantic_axis_order != "CHW":
        raise ValueError("canonical_chw_tensor_hash requires semantic_axis_order='CHW'")
    if np.dtype(canonical_dtype).str != CANONICAL_DTYPE:
        raise ValueError("canonical_chw_tensor_hash requires canonical_dtype='<f4'")
    member = _nonempty("semantic_member_name", semantic_member_name)
    prompt = _nonempty("prompt_identity", prompt_identity)
    expert = _nonempty("expert_identity", expert_identity)
    bands = _band_order(band_order)
    _provenance(raw_file_sha256, raw_member_hash)
    canonical, original_dtype = _canonical_float32(value, rank=3, axis_order="CHW")
    if bands is not None and canonical.shape[0] != len(bands):
        raise ValueError(
            "rank 3 CHW channel contract does not match the declared band/channel order"
        )
    header = {
        "schema_version": SCHEMA_VERSION,
        "semantic_member_name": member,
        "prompt_identity": prompt,
        "expert_identity": expert,
        "semantic_axis_order": "CHW",
        "shape": list(canonical.shape),
        "canonical_dtype": CANONICAL_DTYPE,
        "band_order": bands,
        "memory_order": "C",
    }
    return CanonicalTensorHashRecord(
        header=header,
        canonical_semantic_tensor_sha256=_digest(header, canonical),
        raw_file_sha256=raw_file_sha256,
        raw_member_hash=raw_member_hash,
        original_dtype=original_dtype,
        canonical_dtype=CANONICAL_DTYPE,
        exact_values_preserved=True,
        per_sample_hashes=(),
    )


def canonical_nchw_tensor_hash(
    value: object,
    *,
    semantic_axis_order: str,
    semantic_member_name: str,
    prompt_identity: str,
    expert_identity: str,
    band_order: Iterable[str] | None,
    canonical_dtype: str,
    ordered_sample_ids: Iterable[str],
    raw_file_sha256: str | None = None,
    raw_member_hash: Any = None,
) -> CanonicalTensorHashRecord:
    """Hash one complete declared NCHW batch with order and sample IDs bound."""

    if semantic_axis_order != "NCHW":
        raise ValueError("canonical_nchw_tensor_hash requires semantic_axis_order='NCHW'")
    if np.dtype(canonical_dtype).str != CANONICAL_DTYPE:
        raise ValueError("canonical_nchw_tensor_hash requires canonical_dtype='<f4'")
    member = _nonempty("semantic_member_name", semantic_member_name)
    prompt = _nonempty("prompt_identity", prompt_identity)
    expert = _nonempty("expert_identity", expert_identity)
    bands = _band_order(band_order)
    _provenance(raw_file_sha256, raw_member_hash)
    canonical, original_dtype = _canonical_float32(value, rank=4, axis_order="NCHW")
    sample_ids = list(ordered_sample_ids)
    if len(sample_ids) != canonical.shape[0]:
        raise ValueError("ordered_sample_ids length must equal the explicit N dimension")
    for sample_id in sample_ids:
        _nonempty("sample_id", sample_id)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("ordered_sample_ids must be unique")
    if bands is not None and canonical.shape[1] != len(bands):
        raise ValueError(
            "rank 4 NCHW channel contract does not match the declared band/channel order"
        )

    sample_records: list[CanonicalSampleHashRecord] = []
    for index, sample_id in enumerate(sample_ids):
        direct = canonical_chw_tensor_hash(
            canonical[index],
            semantic_axis_order="CHW",
            semantic_member_name=f"{member}.sample",
            prompt_identity=sample_id,
            expert_identity=expert,
            band_order=bands,
            canonical_dtype=CANONICAL_DTYPE,
        )
        sample_records.append(
            CanonicalSampleHashRecord(
                sample_id=sample_id,
                header=direct.header,
                canonical_semantic_tensor_sha256=direct.canonical_semantic_tensor_sha256,
            )
        )

    header = {
        "schema_version": SCHEMA_VERSION,
        "semantic_member_name": member,
        "prompt_identity": prompt,
        "expert_identity": expert,
        "semantic_axis_order": "NCHW",
        "shape": list(canonical.shape),
        "canonical_dtype": CANONICAL_DTYPE,
        "band_order": bands,
        "ordered_sample_ids": sample_ids,
        "memory_order": "C",
    }
    return CanonicalTensorHashRecord(
        header=header,
        canonical_semantic_tensor_sha256=_digest(header, canonical),
        raw_file_sha256=raw_file_sha256,
        raw_member_hash=raw_member_hash,
        original_dtype=original_dtype,
        canonical_dtype=CANONICAL_DTYPE,
        exact_values_preserved=True,
        per_sample_hashes=tuple(sample_records),
    )


def canonical_tensor_hash_by_contract(
    value: object,
    *,
    semantic_axis_order: str,
    semantic_member_name: str,
    prompt_identity: str,
    expert_identity: str,
    band_order: Iterable[str] | None,
    canonical_dtype: str,
    ordered_sample_ids: Iterable[str] | None = None,
    raw_file_sha256: str | None = None,
    raw_member_hash: Any = None,
) -> CanonicalTensorHashRecord:
    """Dispatch only from an explicit semantic axis-order contract."""

    common = {
        "semantic_axis_order": semantic_axis_order,
        "semantic_member_name": semantic_member_name,
        "prompt_identity": prompt_identity,
        "expert_identity": expert_identity,
        "band_order": band_order,
        "canonical_dtype": canonical_dtype,
        "raw_file_sha256": raw_file_sha256,
        "raw_member_hash": raw_member_hash,
    }
    if semantic_axis_order == "CHW":
        if ordered_sample_ids is not None:
            raise ValueError("CHW contract does not accept ordered_sample_ids")
        return canonical_chw_tensor_hash(value, **common)
    if semantic_axis_order == "NCHW":
        if ordered_sample_ids is None:
            raise ValueError("NCHW contract requires ordered_sample_ids")
        return canonical_nchw_tensor_hash(
            value, ordered_sample_ids=ordered_sample_ids, **common
        )
    raise ValueError(f"unsupported semantic axis order: {semantic_axis_order}")


__all__ = [
    "CANONICAL_DTYPE",
    "SCHEMA_VERSION",
    "CanonicalSampleHashRecord",
    "CanonicalTensorHashRecord",
    "canonical_chw_tensor_hash",
    "canonical_nchw_tensor_hash",
    "canonical_tensor_hash_by_contract",
]

