"""Explicit Mode-B correction for candidate 004's runtime feature-hash call site."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Sequence

from src.d3_tensor_hash_contract_r1 import canonical_tensor_hash_by_contract


FEATURE_CONTRACTS = (
    ("cached_features.enc1", (2, 16, 60, 60)),
    ("cached_features.enc2", (2, 32, 30, 30)),
    ("cached_features.bottleneck", (2, 64, 15, 15)),
)


def hash_runtime_feature_batches(
    features: Sequence[object],
    *,
    raw_file_sha256: str,
    ordered_sample_ids: Iterable[str],
    raw_member_hashes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Hash the unchanged three-tensor runtime feature tuple as ordered NCHW."""

    sample_ids = tuple(ordered_sample_ids)
    if sample_ids != ("prompt_a", "prompt_b"):
        raise ValueError("runtime feature order must be explicitly prompt_a then prompt_b")
    if len(features) != len(FEATURE_CONTRACTS):
        raise ValueError("runtime feature tuple must contain enc1, enc2, and bottleneck")
    provenance = (
        tuple(raw_member_hashes)
        if raw_member_hashes is not None
        else (None,) * len(FEATURE_CONTRACTS)
    )
    if len(provenance) != len(FEATURE_CONTRACTS):
        raise ValueError("raw_member_hashes must align with the three feature levels")

    records = []
    for value, (member, expected_shape), raw_member_hash in zip(
        features, FEATURE_CONTRACTS, provenance
    ):
        shape = tuple(int(item) for item in value.shape)
        if shape != expected_shape:
            raise ValueError(
                f"{member} does not match frozen NCHW shape: {shape} != {expected_shape}"
            )
        record = canonical_tensor_hash_by_contract(
            value,
            semantic_axis_order="NCHW",
            semantic_member_name=member,
            prompt_identity="ordered_prompt_batch",
            expert_identity="shared_cached_encoder",
            band_order=None,
            canonical_dtype="<f4",
            ordered_sample_ids=sample_ids,
            raw_file_sha256=raw_file_sha256,
            raw_member_hash=raw_member_hash,
        )
        records.append(asdict(record))
    return {
        "schema_version": "thayer-d3-hash-r1-runtime-callsite-v1",
        "marker": "RANK4_HASH_CONTRACT_RESOLVED",
        "mode": "MODE_B_ORDERED_BATCH_NCHW",
        "ordered_sample_ids": list(sample_ids),
        "records": records,
        "target_dependent_loss_computations": 0,
        "scientific_metric_computations": 0,
        "decoder_forwards": 0,
        "optimizer_steps": 0,
    }


__all__ = ["FEATURE_CONTRACTS", "hash_runtime_feature_batches"]

