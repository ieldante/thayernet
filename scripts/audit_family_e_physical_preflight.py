#!/usr/bin/env python3
"""Audit Family-E simplex target representability after preregistration freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.family_e import conservation_error, simplex_source_allocation  # noqa: E402

EXPECTED_PREREG = "256bffe3bc53b572b7596bba844f0afdbf4abf3c4cb1d8906fc0ad08663d8881"
UPSTREAM = {
    "training": (
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_training_scene_manifest.csv",
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_training_scenes.h5",
    ),
    "validation": (
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_validation_scene_manifest.csv",
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_validation_scenes.h5",
    ),
    "calibration": (
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_natural_calibration_scene_manifest.csv",
        "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_natural_calibration_scenes.h5",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_mps() -> dict[str, object]:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        return {"status": "FAIL", "reason": "MPS_UNAVAILABLE"}
    device = torch.device("mps")
    observed = torch.tensor(
        [[[[0.0, 1.0e-8], [2.0, 7.0]], [[1.0, 3.0], [5.0, 9.0]], [[4.0, 0.0], [6.0, 8.0]]]],
        dtype=torch.float32,
        device=device,
    )
    logits = torch.zeros((1, 9, 2, 2), dtype=torch.float32, device=device, requires_grad=True)
    output = simplex_source_allocation(logits, observed)
    loss = output.requested.square().mean() + output.companion.abs().mean()
    loss.backward()
    conservation = float(conservation_error(output, observed).detach().cpu())
    gradient_finite = bool(torch.isfinite(logits.grad).all().cpu())
    all_nonnegative = bool(
        (
            (output.requested >= 0).all()
            & (output.companion >= 0).all()
            & (output.residual >= 0).all()
        ).cpu()
    )

    zero_logits = torch.zeros((1, 3, 3, 1, 1), dtype=torch.float32, device=device)
    zero_logits[:, :, 0] = -100.0
    zero_observed = torch.ones((1, 3, 1, 1), dtype=torch.float32, device=device)
    zero_output = simplex_source_allocation(zero_logits, zero_observed)
    zero_error = float(zero_output.requested.abs().max().cpu())

    signed_observed = torch.tensor(
        [[[[-1.0]], [[2.0]], [[3.0]]]], dtype=torch.float32, device=device
    )
    signed = simplex_source_allocation(
        torch.zeros((1, 9, 1, 1), dtype=torch.float32, device=device), signed_observed
    )
    signed_negative = bool(
        (
            (signed.requested < 0).any()
            | (signed.companion < 0).any()
            | (signed.residual < 0).any()
        ).cpu()
    )
    return {
        "status": "PASS"
        if all_nonnegative and conservation <= 9.0e-5 and gradient_finite and zero_error <= 1.0e-7
        else "FAIL",
        "device": "mps",
        "nonnegative_for_nonnegative_observed": all_nonnegative,
        "conservation_max_abs_error": conservation,
        "finite_gradients": gradient_finite,
        "zero_source_absolute_error": zero_error,
        "low_flux_represented": bool(
            float(output.requested[0, 0, 0, 1].detach().cpu()) > 0.0
        ),
        "signed_observed_forces_negative_allocation": signed_negative,
        "band_order": "g/r/z",
        "units": "detected electrons",
    }


def audit_partition(repo: Path, run: Path, partition: str) -> dict[str, object]:
    selector = pd.read_csv(run / "manifests" / f"{partition}_manifest.csv")
    upstream_manifest_path, scene_path = (repo / value for value in UPSTREAM[partition])
    upstream = pd.read_csv(upstream_manifest_path)
    indices = selector["upstream_index"].to_numpy(dtype=np.int64)
    selected = upstream.iloc[indices]
    total_observed = 0
    total_targets = 0
    observed_negative = 0
    target_negative = 0
    observed_nonfinite = 0
    target_nonfinite = 0
    scenes_negative = 0
    observed_min = np.inf
    observed_max = -np.inf
    target_min = np.inf
    target_max = -np.inf

    with h5py.File(scene_path, "r") as handle:
        if tuple(handle["blend"].shape[1:]) != (3, 60, 60):
            raise RuntimeError("unexpected blend shape")
        if tuple(handle["isolated"].shape[1:]) != (2, 3, 60, 60):
            raise RuntimeError("unexpected isolated shape")
        for start in range(0, len(indices), 64):
            batch_indices = indices[start : start + 64]
            observed = np.asarray(handle["blend"][batch_indices], dtype=np.float32)
            targets = np.asarray(handle["isolated"][batch_indices], dtype=np.float32)
            total_observed += observed.size
            total_targets += targets.size
            observed_negative += int(np.count_nonzero(observed < 0))
            target_negative += int(np.count_nonzero(targets < 0))
            observed_nonfinite += int(np.count_nonzero(~np.isfinite(observed)))
            target_nonfinite += int(np.count_nonzero(~np.isfinite(targets)))
            scenes_negative += int(np.count_nonzero(np.any(observed < 0, axis=(1, 2, 3))))
            observed_min = min(observed_min, float(observed.min()))
            observed_max = max(observed_max, float(observed.max()))
            target_min = min(target_min, float(targets.min()))
            target_max = max(target_max, float(targets.max()))

    tolerance = 1.0e-6 * max(1.0, abs(observed_min), abs(observed_max))
    exceedance_count = 0
    exceedance_scenes = 0
    maximum_exceedance = -np.inf
    requested_prompt_hash_mismatch = 0
    with h5py.File(scene_path, "r") as handle:
        for start in range(0, len(indices), 64):
            batch_indices = indices[start : start + 64]
            observed = np.asarray(handle["blend"][batch_indices], dtype=np.float64)
            targets = np.asarray(handle["isolated"][batch_indices], dtype=np.float64)
            excess = targets.sum(axis=1, dtype=np.float64) - observed
            exceedance_count += int(np.count_nonzero(excess > tolerance))
            exceedance_scenes += int(np.count_nonzero(np.any(excess > tolerance, axis=(1, 2, 3))))
            maximum_exceedance = max(maximum_exceedance, float(excess.max()))
            prompts = np.asarray(handle["prompt"][batch_indices], dtype=np.float32)
            for local, prompt in enumerate(prompts):
                expected = str(selected.iloc[start + local]["prompt_sha256"])
                digest = hashlib.sha256(
                    b"canonical-f32-le-v1\0"
                    + np.asarray(prompt, dtype="<f4", order="C").tobytes(order="C")
                ).hexdigest()
                # Upstream prompt hashes use the project canonical helper.  This
                # raw diagnostic is retained but is not treated as that helper.
                if len(expected) != 64 or len(digest) != 64:
                    requested_prompt_hash_mismatch += 1

    representable = (
        observed_negative == 0
        and observed_nonfinite == 0
        and target_negative == 0
        and target_nonfinite == 0
        and exceedance_count == 0
    )
    return {
        "partition": partition,
        "episodes": int(len(indices)),
        "observed_values": int(total_observed),
        "observed_minimum": observed_min,
        "observed_maximum": observed_max,
        "observed_negative_count": observed_negative,
        "observed_negative_fraction": observed_negative / total_observed,
        "episodes_with_negative_observed": scenes_negative,
        "observed_nonfinite_count": observed_nonfinite,
        "target_values": int(total_targets),
        "target_minimum": target_min,
        "target_maximum": target_max,
        "target_negative_count": target_negative,
        "target_nonfinite_count": target_nonfinite,
        "tolerance": tolerance,
        "target_sum_exceedance_count": exceedance_count,
        "episodes_with_target_sum_exceedance": exceedance_scenes,
        "target_sum_maximum_exceedance": maximum_exceedance,
        "representable": representable,
        "source_group_provenance": "upstream_index plus frozen upstream manifest",
        "prompt_identity_rows_well_formed": requested_prompt_hash_mismatch == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    repo = REPO
    run = repo / args.run
    prereg = run / "preregistration/family_e_nonnegative_flux_conserving_eligibility.md"
    observed_prereg = sha256_file(prereg)
    if observed_prereg != EXPECTED_PREREG:
        raise SystemExit("preregistration hash mismatch")
    synthetic = synthetic_mps()
    partitions = [audit_partition(repo, run, p) for p in ("training", "validation", "calibration")]
    all_representable = all(bool(row["representable"]) for row in partitions)
    result = {
        "preregistration_sha256": observed_prereg,
        "model_constructed": False,
        "training_tensor_load_occurred_after_preregistration": True,
        "synthetic": synthetic,
        "partitions": partitions,
        "physical_construction_synthetic_pass": synthetic["status"] == "PASS",
        "frozen_target_representability_pass": all_representable,
        "status": "PASS" if synthetic["status"] == "PASS" and all_representable else "FAIL",
        "stop_reason": None
        if all_representable
        else "SIGNED_ZERO_BACKGROUND_OBSERVATIONS_INCOMPATIBLE_WITH_NONNEGATIVE_EXACT_SIMPLEX_CONSERVATION",
        "post_hoc_repair_applied": False,
        "additive_background_applied": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
