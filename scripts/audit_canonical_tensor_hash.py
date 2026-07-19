#!/usr/bin/env python3
"""Write the campaign's fail-closed canonical per-sample hash audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.canonical_tensor_hash import SCHEMA_VERSION, canonical_tensor_sha256  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_probabilistic_unet_"):
        raise ValueError("unexpected run directory")
    if not (run_dir / "logs/part_a_complete.json").is_file():
        raise RuntimeError("Part A did not pass")

    base = (np.arange(3 * 7 * 9, dtype=np.float64).reshape(3, 7, 9) / 11.0)
    reference = canonical_tensor_sha256(base)
    padded = np.zeros((3, 7, 18), dtype=np.float64)
    padded[:, :, ::2] = base
    buffer = run_dir / "diagnostics/canonical_hash_roundtrip_retry1.npy"
    if buffer.exists():
        raise FileExistsError(buffer)
    with buffer.open("xb") as handle:
        np.save(handle, base, allow_pickle=False)
    reloaded = np.load(buffer, allow_pickle=False)
    batch = np.stack((base + 9, base - 4, base))
    cpu_tensor = torch.from_numpy(base.astype(np.float32))
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable for required transfer test")
    mps_tensor = cpu_tensor.to("mps")

    changed_pixel = base.copy()
    changed_pixel[1, 2, 3] += 0.25
    changed_value = base.copy()
    changed_value[0, 0, 0] = np.float64(0.125)
    cases = [
        ("batch_size_1", reference, canonical_tensor_sha256(base[None], layout="NCHW", sample_index=0), True),
        ("batch_size_n_position_2", reference, canonical_tensor_sha256(batch, layout="NCHW", sample_index=2), True),
        ("noncontiguous_storage", reference, canonical_tensor_sha256(padded[:, :, ::2]), True),
        ("channel_last_round_trip", reference, canonical_tensor_sha256(np.moveaxis(base, 0, -1), layout="HWC"), True),
        ("equivalent_clone", reference, canonical_tensor_sha256(base.copy()), True),
        ("serialization_reload", reference, canonical_tensor_sha256(reloaded), True),
        ("mps_cpu_transfer", canonical_tensor_sha256(cpu_tensor), canonical_tensor_sha256(mps_tensor), True),
        ("one_changed_pixel", reference, canonical_tensor_sha256(changed_pixel), False),
        ("channel_permutation", reference, canonical_tensor_sha256(base[[1, 0, 2]]), False),
        ("shape_change", reference, canonical_tensor_sha256(base[:, :, :-1]), False),
        ("dtype_conversion_with_value_change", reference, canonical_tensor_sha256(changed_value), False),
    ]
    rows = []
    for name, left, right, expect_equal in cases:
        observed_equal = left == right
        passed = observed_equal == expect_equal
        rows.append({
            "test": name,
            "expect_equal": expect_equal,
            "observed_equal": observed_equal,
            "left_sha256": left,
            "right_sha256": right,
            "status": "PASS" if passed else "FAIL",
        })
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError(f"canonical hash contract failed: {rows}")
    write_csv_fresh(run_dir / "tables/canonical_hash_tests.csv", rows)
    module = REPO / "src/canonical_tensor_hash.py"
    report = f"""# Canonical per-sample tensor hash contract

Status: **PASS — {len(rows)}/{len(rows)} required tests passed**.

- Schema version: `{SCHEMA_VERSION}`.
- Digest: SHA-256 over a canonical JSON header, one NUL separator, and tensor bytes.
- Header fields: schema version, fixed CHW order, CHW shape, explicit `<f4` dtype,
  little-endian byte order, and contiguous C memory order.
- A batch dimension is removed by explicit sample selection. Batch position,
  batch size, strides, source device, and tensor storage layout are excluded.
- Tensors are detached, moved to CPU, converted to explicit little-endian
  float32, made C-contiguous, and rejected if non-finite.
- This version is campaign-only. Historical candidate hashes are unchanged.
- Implementation SHA-256: `{sha256_file(module)}`.

Invariance passed for batch size 1 versus N, different batch position,
noncontiguous storage, channel-last round trip, cloning, serialization/reload,
and MPS versus CPU transfer. Sensitivity passed for one changed pixel, channel
permutation, shape change, and a pre-conversion dtype/value change.
"""
    write_text_fresh(run_dir / "diagnostics/canonical_hash_contract.md", report)


if __name__ == "__main__":
    main()
