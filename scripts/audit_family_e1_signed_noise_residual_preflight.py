#!/usr/bin/env python3
"""Execute the preregistered signed-noise-residual physical preflight."""
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

from src.family_e_signed_residual import (  # noqa: E402
    apply_witness_numpy,
    conservation_error,
    inverse_target_witness,
    signed_noise_residual_allocation,
)

EXPECTED_PREREG = "be546f7f1aa2ec04f1a76f84bc5305c87521d5b89331c681dc3cdf18a5293d3b"
SCALES = np.asarray(
    [611.9199829101562, 1805.8800048828125, 1854.199951171875],
    dtype=np.float32,
)
FAMILY_E_RUN = Path("outputs/runs/thayer_family_e_v0_20260714_195256")
PARTITIONS = {
    "training": {
        "selector": FAMILY_E_RUN / "manifests/training_manifest.csv",
        "selector_sha256": "4a8768eaa70e1d3f5f7a29fd4035e994c9c6f1494d3553e6ac0f805c8e911bc1",
        "manifest": Path("outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_training_scene_manifest.csv"),
        "scenes": Path("outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_training_scenes.h5"),
        "rows": 10000,
    },
    "validation": {
        "selector": FAMILY_E_RUN / "manifests/validation_manifest.csv",
        "selector_sha256": "bc5c65ffab19baea38e37edcb4d5dabd15bae1c0266b7dfdaa749eba5c6c464d",
        "manifest": Path("outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_validation_scene_manifest.csv"),
        "scenes": Path("outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_validation_scenes.h5"),
        "rows": 2000,
    },
    "calibration": {
        "selector": FAMILY_E_RUN / "manifests/calibration_manifest.csv",
        "selector_sha256": "70326c1835726677e5d98c50323329f919bcd405f0f379420987fcd97e20fa0c",
        "manifest": Path("outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_natural_calibration_scene_manifest.csv"),
        "scenes": Path("outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_natural_calibration_scenes.h5"),
        "rows": 2000,
    },
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
        [[
            [[-3.0, 2.0], [0.0, 4.0]],
            [[5.0, -7.0], [1.0, 9.0]],
            [[-2.0, 6.0], [8.0, -1.0]],
        ]],
        dtype=torch.float32,
        device=device,
    )
    logits = torch.tensor(
        [[
            [[0.0, -1.0], [1.0e-8, 0.5]],
            [[0.1, 0.2], [-0.3, 0.4]],
            [[0.0, 0.3], [0.2, -0.1]],
            [[0.2, 0.0], [-0.4, 0.1]],
            [[-0.2, 0.3], [0.0, 0.2]],
            [[0.1, -0.1], [0.4, 0.0]],
        ]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    scales = torch.tensor(SCALES, dtype=torch.float32, device=device)
    output = signed_noise_residual_allocation(logits, observed, scales)
    error = float(conservation_error(output, observed).detach().cpu())
    loss = output.requested.square().mean() + output.companion.abs().mean()
    loss.backward()

    zero_logits = torch.zeros((1, 6, 2, 2), dtype=torch.float32, device=device)
    zero = signed_noise_residual_allocation(zero_logits, observed, scales)
    negative_logits = -torch.ones((1, 6, 2, 2), dtype=torch.float32, device=device)
    negative = signed_noise_residual_allocation(negative_logits, observed, scales)

    changed_observed = observed + 0.75
    changed_observation_output = signed_noise_residual_allocation(
        logits.detach(), changed_observed, scales
    )
    changed_requested_logits = logits.detach().clone()
    changed_requested_logits[:, :3] += 0.25
    changed_requested_output = signed_noise_residual_allocation(
        changed_requested_logits, observed, scales
    )

    max_magnitude = max(
        1.0,
        float(observed.abs().max().cpu()),
        float(output.requested.abs().max().detach().cpu()),
        float(output.companion.abs().max().detach().cpu()),
        float(output.residual_noise.abs().max().detach().cpu()),
    )
    tolerance = 1.0e-5 * max_magnitude
    source_nonnegative = bool(
        ((output.requested >= 0).all() & (output.companion >= 0).all()).cpu()
    )
    residual_has_both_signs = bool(
        ((output.residual_noise < 0).any() & (output.residual_noise > 0).any()).cpu()
    )
    finite = bool(
        (
            torch.isfinite(output.requested).all()
            & torch.isfinite(output.companion).all()
            & torch.isfinite(output.residual_noise).all()
            & torch.isfinite(logits.grad).all()
        ).cpu()
    )
    zero_exact = bool(
        (
            (zero.requested == 0).all()
            & (zero.companion == 0).all()
            & (negative.requested == 0).all()
            & (negative.companion == 0).all()
        ).cpu()
    )
    observation_isolation = bool(
        torch.equal(changed_observation_output.requested, output.requested.detach())
        and torch.equal(changed_observation_output.companion, output.companion.detach())
        and torch.allclose(
            changed_observation_output.residual_noise - output.residual_noise.detach(),
            torch.full_like(output.residual_noise, 0.75),
            rtol=0.0,
            atol=2.0e-5,
        )
    )
    requested_isolation = bool(
        torch.equal(changed_requested_output.companion, output.companion.detach())
        and not torch.equal(changed_requested_output.requested, output.requested.detach())
    )
    low_flux = float(output.requested[0, 0, 1, 0].detach().cpu())
    passed = (
        source_nonnegative
        and residual_has_both_signs
        and finite
        and zero_exact
        and error <= tolerance
        and observation_isolation
        and requested_isolation
        and low_flux > 0.0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "device": "mps",
        "source_nonnegative": source_nonnegative,
        "signed_residual_has_both_signs": residual_has_both_signs,
        "finite_outputs_and_gradients": finite,
        "zero_and_negative_logits_map_exactly_to_zero": zero_exact,
        "low_positive_flux_output": low_flux,
        "conservation_max_abs_error": error,
        "conservation_tolerance": tolerance,
        "observation_change_isolated_to_residual": observation_isolation,
        "requested_logit_change_leaves_companion_exact": requested_isolation,
        "band_order": "requested g/r/z then companion g/r/z",
        "units": "detected electrons after positive scale inversion",
        "cpu_fallback": False,
    }


def audit_partition(repo: Path, partition: str, spec: dict[str, object]) -> dict[str, object]:
    selector_path = repo / Path(spec["selector"])
    if sha256_file(selector_path) != spec["selector_sha256"]:
        raise RuntimeError(f"{partition} selector hash mismatch")
    selector = pd.read_csv(selector_path)
    if len(selector) != spec["rows"]:
        raise RuntimeError(f"{partition} row-count mismatch")
    upstream = pd.read_csv(repo / Path(spec["manifest"]))
    indices = selector.upstream_index.to_numpy(dtype=np.int64)
    selected = upstream.iloc[indices].reset_index(drop=True)
    if not selected.query_state.eq("UNIQUE_VALID").all():
        raise RuntimeError(f"{partition} contains a non-valid query")
    if not selected.prompt_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError(f"{partition} prompt identity malformed")

    observed_count = 0
    source_count = 0
    residual_count = 0
    observed_nonfinite = 0
    target_nonfinite = 0
    target_negative = 0
    mapped_negative = 0
    residual_nonfinite = 0
    residual_negative = 0
    residual_positive = 0
    residual_zero = 0
    observed_max_abs = 0.0
    target_max_abs = 0.0
    mapped_max_abs = 0.0
    residual_max_abs = 0.0
    residual_min = np.inf
    residual_max = -np.inf
    residual_sum = 0.0
    residual_sumsq = 0.0
    band_sumsq = np.zeros(3, dtype=np.float64)
    band_count = np.zeros(3, dtype=np.int64)
    requested_roundtrip_max = 0.0
    companion_roundtrip_max = 0.0
    conservation32_max = 0.0
    conservation64_max = 0.0

    with h5py.File(repo / Path(spec["scenes"]), "r") as handle:
        if tuple(handle["blend"].shape[1:]) != (3, 60, 60):
            raise RuntimeError("unexpected observed shape")
        if tuple(handle["isolated"].shape[1:]) != (2, 3, 60, 60):
            raise RuntimeError("unexpected isolated shape")
        for start in range(0, len(indices), 64):
            batch_indices = indices[start : start + 64]
            observed = np.asarray(handle["blend"][batch_indices], dtype=np.float32)
            isolated = np.asarray(handle["isolated"][batch_indices], dtype=np.float32)
            matched = selected.matched_source_index.iloc[
                start : start + len(batch_indices)
            ].to_numpy(dtype=np.int64)
            local = np.arange(len(batch_indices))
            requested_target = isolated[local, matched]
            companion_target = isolated[local, 1 - matched]

            requested_logits, companion_logits = inverse_target_witness(
                requested_target, companion_target, SCALES
            )
            requested, companion = apply_witness_numpy(
                requested_logits, companion_logits, SCALES
            )
            residual = observed - requested - companion
            reconstructed = requested + companion + residual

            requested64 = (
                np.maximum(
                    requested_target.astype(np.float64)
                    / SCALES.astype(np.float64)[None, :, None, None],
                    0.0,
                )
                * SCALES.astype(np.float64)[None, :, None, None]
            )
            companion64 = (
                np.maximum(
                    companion_target.astype(np.float64)
                    / SCALES.astype(np.float64)[None, :, None, None],
                    0.0,
                )
                * SCALES.astype(np.float64)[None, :, None, None]
            )
            observed64 = observed.astype(np.float64)
            residual64 = observed64 - requested64 - companion64
            reconstructed64 = requested64 + companion64 + residual64

            observed_count += observed.size
            source_count += requested.size + companion.size
            residual_count += residual.size
            observed_nonfinite += int(np.count_nonzero(~np.isfinite(observed)))
            target_nonfinite += int(
                np.count_nonzero(~np.isfinite(requested_target))
                + np.count_nonzero(~np.isfinite(companion_target))
            )
            target_negative += int(
                np.count_nonzero(requested_target < 0)
                + np.count_nonzero(companion_target < 0)
            )
            mapped_negative += int(
                np.count_nonzero(requested < 0)
                + np.count_nonzero(companion < 0)
            )
            residual_nonfinite += int(np.count_nonzero(~np.isfinite(residual)))
            residual_negative += int(np.count_nonzero(residual < 0))
            residual_positive += int(np.count_nonzero(residual > 0))
            residual_zero += int(np.count_nonzero(residual == 0))
            observed_max_abs = max(observed_max_abs, float(np.max(np.abs(observed))))
            target_max_abs = max(
                target_max_abs,
                float(np.max(np.abs(requested_target))),
                float(np.max(np.abs(companion_target))),
            )
            mapped_max_abs = max(
                mapped_max_abs,
                float(np.max(np.abs(requested))),
                float(np.max(np.abs(companion))),
            )
            residual_max_abs = max(residual_max_abs, float(np.max(np.abs(residual))))
            residual_min = min(residual_min, float(np.min(residual)))
            residual_max = max(residual_max, float(np.max(residual)))
            residual_sum += float(np.sum(residual, dtype=np.float64))
            residual_sumsq += float(
                np.sum(residual.astype(np.float64) ** 2, dtype=np.float64)
            )
            band_sumsq += np.sum(
                residual.astype(np.float64) ** 2, axis=(0, 2, 3), dtype=np.float64
            )
            band_count += np.asarray(
                [len(batch_indices) * 60 * 60] * 3, dtype=np.int64
            )
            requested_roundtrip_max = max(
                requested_roundtrip_max,
                float(np.max(np.abs(requested - requested_target))),
            )
            companion_roundtrip_max = max(
                companion_roundtrip_max,
                float(np.max(np.abs(companion - companion_target))),
            )
            conservation32_max = max(
                conservation32_max,
                float(np.max(np.abs(reconstructed - observed))),
            )
            conservation64_max = max(
                conservation64_max,
                float(np.max(np.abs(reconstructed64 - observed64))),
            )

    source_tolerance = 1.0e-6 * max(1.0, target_max_abs)
    magnitude = max(
        1.0, observed_max_abs, mapped_max_abs, residual_max_abs
    )
    conservation32_tolerance = 1.0e-5 * magnitude
    conservation64_tolerance = 1.0e-10 * magnitude
    passed = (
        observed_nonfinite == 0
        and target_nonfinite == 0
        and target_negative == 0
        and mapped_negative == 0
        and residual_nonfinite == 0
        and residual_negative > 0
        and residual_positive > 0
        and requested_roundtrip_max <= source_tolerance
        and companion_roundtrip_max <= source_tolerance
        and conservation32_max <= conservation32_tolerance
        and conservation64_max <= conservation64_tolerance
    )
    return {
        "partition": partition,
        "status": "PASS" if passed else "FAIL",
        "episodes": int(len(indices)),
        "selector_sha256": spec["selector_sha256"],
        "observed_values": observed_count,
        "source_values": source_count,
        "observed_nonfinite_count": observed_nonfinite,
        "target_nonfinite_count": target_nonfinite,
        "target_negative_count": target_negative,
        "mapped_source_negative_count": mapped_negative,
        "source_roundtrip_tolerance": source_tolerance,
        "requested_roundtrip_max_abs_error": requested_roundtrip_max,
        "companion_roundtrip_max_abs_error": companion_roundtrip_max,
        "residual_values": residual_count,
        "residual_nonfinite_count": residual_nonfinite,
        "residual_minimum": residual_min,
        "residual_maximum": residual_max,
        "residual_mean": residual_sum / residual_count,
        "residual_rms": float(np.sqrt(residual_sumsq / residual_count)),
        "residual_negative_count": residual_negative,
        "residual_positive_count": residual_positive,
        "residual_zero_count": residual_zero,
        "residual_negative_fraction": residual_negative / residual_count,
        "residual_positive_fraction": residual_positive / residual_count,
        "residual_zero_fraction": residual_zero / residual_count,
        "residual_band_rms_g_r_z": [
            float(np.sqrt(band_sumsq[index] / band_count[index]))
            for index in range(3)
        ],
        "float32_conservation_max_abs_error": conservation32_max,
        "float32_conservation_tolerance": conservation32_tolerance,
        "float64_conservation_max_abs_error": conservation64_max,
        "float64_conservation_tolerance": conservation64_tolerance,
        "prompt_identity_well_formed": True,
        "source_provenance_retained": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = REPO / args.run
    prereg = run / "preregistration/signed_noise_residual_physical_contract_preflight.md"
    observed_prereg = sha256_file(prereg)
    if observed_prereg != EXPECTED_PREREG:
        raise SystemExit("preregistration hash mismatch")

    synthetic = synthetic_mps()
    partitions = [
        audit_partition(REPO, partition, spec)
        for partition, spec in PARTITIONS.items()
    ]
    all_partitions = all(row["status"] == "PASS" for row in partitions)
    passed = synthetic["status"] == "PASS" and all_partitions
    result = {
        "campaign": "Thayer-Family-E1-Signed-Noise-Residual-Preflight-v0",
        "preregistration_sha256": observed_prereg,
        "synthetic_mps": synthetic,
        "partitions": partitions,
        "status": "SIGNED_NOISE_RESIDUAL_CONTRACT_PASS"
        if passed
        else "SIGNED_NOISE_RESIDUAL_CONTRACT_FAIL",
        "all_frozen_target_representability_gates_pass": all_partitions,
        "model_constructed": False,
        "optimizer_constructed": False,
        "checkpoint_written": False,
        "reconstruction_written": False,
        "post_hoc_repair_applied": False,
        "truth_used_at_inference": False,
        "truth_used_only_for_offline_inverse_witness": True,
        "development_scene_access_count": 0,
        "atlas_selection_access_count": 0,
        "final_lockbox_access_count": 0,
        "next_campaign_authorized": passed,
        "authorized_next_campaign": "Thayer-Family-E1-v0 — Nonnegative-Source Signed-Residual Model Eligibility"
        if passed
        else None,
        "thayer_audit_v1_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
