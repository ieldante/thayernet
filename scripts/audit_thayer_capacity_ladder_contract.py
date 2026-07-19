#!/usr/bin/env python3
"""Reproduce Thayer-FP and audit the Thayer-CL physical-output prerequisite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_thayer_two_expert_micro_overfit import (  # noqa: E402
    MEAN_PSF_FWHM_PIXEL,
    NORMALIZATION,
    load_micro_arrays,
    prompt_identity,
    select_microset,
    thresholds,
)
from src.competing_hypotheses import forward_consistency, is_plausible, scientific_distance  # noqa: E402


FP = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"
P0 = FP / "projection_targets/projected_target_sets_final.h5"
FP_OUTPUTS = FP / "micro_overfit/final_outputs.h5"
FP_HISTORY = FP / "micro_overfit/micro_gate_history.csv"
FP_COMPLETE = FP / "logs/micro_training_complete.json"
HOMOTOPY = FP / "tables/homotopy_projection_summary.csv"
LIMITING = FP / "tables/limiting_constraint_frequency.csv"
P0_COMPARISON = FP / "tables/projection_method_comparison_final_superseding.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def persisted_metrics(
    outputs_physical: np.ndarray,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scales: np.ndarray,
) -> dict[str, float]:
    """Apply the frozen Thayer-ME evaluator to already-persisted outputs."""

    outputs_normalized = outputs_physical / np.tile(scales, 2)[None, None, None, :, None, None]
    threshold, sky = thresholds()
    ordinary_diameter: list[float] = []
    ordinary_coverage: list[bool] = []
    ordinary_forward: list[bool] = []
    ambiguous_own: list[bool] = []
    ambiguous_alternate: list[bool] = []
    ambiguous_both: list[bool] = []
    ambiguous_forward: list[bool] = []
    expert_prompt: list[list[bool]] = [[], []]
    set_prompt: list[bool] = []
    for index, row in enumerate(rows):
        count = int(arrays["counts"][index, 0])
        plausible = np.zeros((2, 2), dtype=bool)
        own = np.zeros((2, 2), dtype=bool)
        alternate = np.zeros((2, 2), dtype=bool)
        identities = np.zeros((2, 2), dtype=bool)
        diameters = []
        for prompt in (0, 1):
            identities[prompt] = prompt_identity(outputs_normalized[index, prompt], arrays["targets"][index, prompt], count)
            for expert in (0, 1):
                candidate = outputs_physical[index, prompt, expert]
                score = forward_consistency(arrays["blend_physical"][index], np.stack((candidate[:3], candidate[3:])), sky)
                plausible[prompt, expert] = is_plausible(score, threshold)
                own_distance = scientific_distance(
                    candidate[:3], arrays["targets_physical"][index, prompt, 0, :3],
                    mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                ).primary_normalized
                own[prompt, expert] = plausible[prompt, expert] and own_distance <= 1.0
                if count == 2:
                    alternate_distance = scientific_distance(
                        candidate[:3], arrays["targets_physical"][index, prompt, 1, :3],
                        mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                    ).primary_normalized
                    alternate[prompt, expert] = plausible[prompt, expert] and alternate_distance <= 1.0
            diameters.append(scientific_distance(
                outputs_physical[index, prompt, 0, :3], outputs_physical[index, prompt, 1, :3],
                mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
            ).primary_normalized)
        for expert in (0, 1):
            expert_prompt[expert].append(bool(identities[:, expert].all()))
        set_prompt.append(bool(identities.all()))
        both = [bool((own[p, 0] and alternate[p, 1]) or (own[p, 1] and alternate[p, 0])) for p in (0, 1)]
        if row["kind"] == "ordinary":
            ordinary_diameter.append(float(np.mean(diameters)))
            ordinary_coverage.append(bool(own.all()))
            ordinary_forward.append(bool(plausible.all()))
        else:
            ambiguous_own.append(bool(own.any(axis=1).all()))
            ambiguous_alternate.append(bool(alternate.any(axis=1).all()))
            ambiguous_both.append(bool(all(both)))
            ambiguous_forward.append(bool(plausible.all()))
    return {
        "ordinary_own_truth_coverage": float(np.mean(ordinary_coverage)),
        "ordinary_median_expert_diameter": float(np.median(ordinary_diameter)),
        "ambiguous_own_truth_coverage": float(np.mean(ambiguous_own)),
        "ambiguous_alternate_truth_coverage": float(np.mean(ambiguous_alternate)),
        "ambiguous_both_mode_coverage": float(np.mean(ambiguous_both)),
        "expert_1_prompt_swap": float(np.mean(expert_prompt[0])),
        "expert_2_prompt_swap": float(np.mean(expert_prompt[1])),
        "set_prompt_swap": float(np.mean(set_prompt)),
        "ordinary_forward_consistency": float(np.mean(ordinary_forward)),
        "ambiguous_forward_consistency": float(np.mean(ambiguous_forward)),
    }


def add_check(rows: list[dict[str, object]], metric: str, observed: object, expected: object, tolerance: float = 0.0) -> None:
    if isinstance(observed, (float, int)) and isinstance(expected, (float, int)):
        passed = abs(float(observed) - float(expected)) <= tolerance
    else:
        passed = observed == expected
    rows.append({"metric": metric, "observed": observed, "expected": expected, "absolute_tolerance": tolerance, "pass": passed})


def domain_rows(name: str, values: np.ndarray, units: str, provenance: str) -> list[dict[str, object]]:
    labels = ("requested_g", "requested_r", "requested_z", "companion_g", "companion_r", "companion_z")
    rows: list[dict[str, object]] = []
    for expert in (0, 1):
        for channel, label in enumerate(labels):
            value = values[:, :, expert, channel]
            rows.append({
                "domain": name,
                "units": units,
                "provenance": provenance,
                "expert": expert,
                "channel": channel,
                "band_layer": label,
                "minimum": float(np.min(value)),
                "maximum": float(np.max(value)),
                "negative_fraction": float(np.mean(value < 0)),
                "negative_fraction_below_minus_1e_7": float(np.mean(value < -1e-7)),
                "zero_fraction": float(np.mean(value == 0)),
                "nonfinite_fraction": float(np.mean(~np.isfinite(value))),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    freeze = json.loads((run / "preregistration/freeze_record.json").read_text())
    prereg = run / "preregistration/contract_compliant_decoder_capacity_ladder.md"
    if freeze["status"] != "FROZEN_BEFORE_PER_SCENE_LOAD" or sha256(prereg) != freeze["preregistration_sha256"]:
        raise RuntimeError("Thayer-CL preregistration/order gate failed")
    write_json_fresh(run / "logs/per_scene_audit_started.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": freeze["preregistration_sha256"],
        "status": "AUTHORIZED_AFTER_FREEZE",
    })

    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    micro_rows, indices = select_microset()
    arrays = load_micro_arrays(indices, scales)
    with h5py.File(P0, "r") as handle:
        p0_normalized = np.asarray(handle["targets_normalized"], dtype=np.float32)
        p0_physical = np.asarray(handle["targets_physical"], dtype=np.float32)
        p0_attrs = {key: value.item() if hasattr(value, "item") else value for key, value in handle.attrs.items()}
    with h5py.File(FP_OUTPUTS, "r") as handle:
        fp_physical = np.asarray(handle["decompositions_physical"], dtype=np.float32)
        fp_output_attrs = {key: value.item() if hasattr(value, "item") else value for key, value in handle.attrs.items()}
    fp_normalized = fp_physical / np.tile(scales, 2)[None, None, None, :, None, None]

    p0_metrics = persisted_metrics(p0_physical, arrays, micro_rows, scales)
    neural_metrics = persisted_metrics(fp_physical, arrays, micro_rows, scales)
    completion = json.loads(FP_COMPLETE.read_text())
    homotopy = read_csv(HOMOTOPY)
    limiting = {row["constraint"]: int(row["count"]) for row in read_csv(LIMITING)}
    method = next(row for row in read_csv(P0_COMPARISON) if row["method"] == "P0_HOMOTOPY_INTERIOR")
    history = read_csv(FP_HISTORY)
    first_negative = next((row for row in history if float(row["negative_output_fraction"]) > 0), None)

    checks: list[dict[str, object]] = []
    for key in ("ordinary_own_truth_coverage", "ambiguous_own_truth_coverage", "ambiguous_alternate_truth_coverage", "ambiguous_both_mode_coverage"):
        add_check(checks, f"p0_{key}", p0_metrics[key], 1.0, 1e-12)
    add_check(checks, "p0_median_alpha", float(np.median([float(row["interior_alpha"]) for row in homotopy])), 0.999979483, 5e-10)
    add_check(checks, "p0_median_correction", float(np.median([float(row["correction_norm"]) for row in homotopy])), 0.946369, 5e-7)
    add_check(checks, "p0_flux_z_limiting_count", limiting["flux_z"], 173)
    add_check(checks, "p0_pair_feasible_fraction", float(method["feasible_pair_fraction"]), 1.0, 1e-12)
    add_check(checks, "p0_ordinary_forward_consistency", p0_metrics["ordinary_forward_consistency"], 1.0, 1e-12)
    add_check(checks, "p0_ambiguous_forward_consistency", p0_metrics["ambiguous_forward_consistency"], 1.0, 1e-12)
    neural_expected = completion["metrics"]
    for key in (
        "ordinary_own_truth_coverage", "ambiguous_own_truth_coverage", "ambiguous_alternate_truth_coverage",
        "ambiguous_both_mode_coverage", "ordinary_median_expert_diameter", "expert_1_prompt_swap",
        "expert_2_prompt_swap", "set_prompt_swap", "ordinary_forward_consistency", "ambiguous_forward_consistency",
    ):
        add_check(checks, f"fp_neural_{key}", neural_metrics[key], float(neural_expected[key]), 1e-9)
    add_check(checks, "fp_final_negative_fraction", float(np.mean(fp_normalized < -1e-7)), 0.43571650752314817, 1e-12)
    add_check(checks, "fp_first_persisted_negative_epoch", int(first_negative["epoch"]) if first_negative else None, 1)
    add_check(checks, "fp_first_persisted_negative_fraction", float(first_negative["negative_output_fraction"]) if first_negative else None, 0.365785228587963, 1e-12)
    add_check(checks, "fp_first_persisted_negative_minimum_normalized", float(first_negative["output_minimum_normalized"]) if first_negative else None, -2.0505404472351074, 1e-12)
    write_csv_fresh(run / "tables/thayer_fp_reproduction.csv", checks)
    if not all(bool(row["pass"]) for row in checks):
        write_json_fresh(run / "logs/fail_closed_stop.json", {
            "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "PART_C_THAYER_FP_REPRODUCTION",
            "reason": "AUTHORITATIVE_BASELINE_MISMATCH",
            "failed_checks": [row for row in checks if not row["pass"]],
            "model_construction_count": 0,
            "optimizer_step_count": 0,
        })
        raise SystemExit(3)

    reconstructed_physical = p0_normalized * np.tile(scales, 2)[None, None, None, :, None, None]
    reconstructed_normalized = p0_physical / np.tile(scales, 2)[None, None, None, :, None, None]
    roundtrip_rows = []
    labels = ("g", "r", "z", "g", "r", "z")
    for channel, label in enumerate(labels):
        physical_error = np.abs(reconstructed_physical[..., channel, :, :] - p0_physical[..., channel, :, :])
        normalized_error = np.abs(reconstructed_normalized[..., channel, :, :] - p0_normalized[..., channel, :, :])
        roundtrip_rows.append({
            "channel": channel,
            "band": label,
            "scale": float(np.tile(scales, 2)[channel]),
            "physical_max_abs_error": float(np.max(physical_error)),
            "normalized_max_abs_error": float(np.max(normalized_error)),
            "target_minimum_physical": float(np.min(p0_physical[..., channel, :, :])),
            "target_maximum_physical": float(np.max(p0_physical[..., channel, :, :])),
            "finite": bool(np.all(np.isfinite(p0_physical[..., channel, :, :]))),
            "nonnegative": bool(np.min(p0_physical[..., channel, :, :]) >= 0),
            "within_frozen_float32_tolerance": bool(np.max(physical_error) <= 0.0009765625),
        })
    write_csv_fresh(run / "tables/output_contract_roundtrip.csv", roundtrip_rows)

    domain_audit: list[dict[str, object]] = []
    domain_audit.extend(domain_rows("projected_target_normalized", p0_normalized, "normalized", "frozen P0 targets"))
    domain_audit.extend(domain_rows("projected_target_physical", p0_physical, "detected_electrons", "frozen P0 targets"))
    domain_audit.extend(domain_rows("model_raw_normalized", fp_normalized, "normalized", "linear head; no activation"))
    domain_audit.extend(domain_rows("model_post_activation_normalized", fp_normalized, "normalized", "identity mapping; identical to raw"))
    domain_audit.extend(domain_rows("inverse_normalized_physical", fp_physical, "detected_electrons", "positive scale multiplication"))
    domain_audit.extend(domain_rows("metric_evaluation_physical", fp_physical, "detected_electrons", "persisted evaluator input"))
    domain_audit.extend(domain_rows("checkpoint_associated_physical", fp_physical, "detected_electrons", "best epoch 395 persisted output"))
    write_csv_fresh(run / "tables/output_domain_audit.csv", domain_audit)

    candidates = [
        {
            "mapping": "historical_identity_linear_head", "defined_by_frozen_contract": True,
            "nonnegative_by_construction": False, "exact_zero_representable": True, "all_p0_targets_representable": True,
            "finite_gradient_near_zero": True, "post_hoc_clipping": False, "mathematically_admissible_for_new_contract": False,
            "reason": "positive scale multiplication preserves raw negative signs",
        },
        {
            "mapping": "relu_as_neural_head_activation", "defined_by_frozen_contract": False,
            "nonnegative_by_construction": True, "exact_zero_representable": True, "all_p0_targets_representable": True,
            "finite_gradient_near_zero": True, "post_hoc_clipping": False, "mathematically_admissible_for_new_contract": True,
            "reason": "admissible in principle but would be a new unfrozen mapping choice",
        },
        {
            "mapping": "square_as_neural_head_mapping", "defined_by_frozen_contract": False,
            "nonnegative_by_construction": True, "exact_zero_representable": True, "all_p0_targets_representable": True,
            "finite_gradient_near_zero": True, "post_hoc_clipping": False, "mathematically_admissible_for_new_contract": True,
            "reason": "admissible in principle but would be a different new unfrozen mapping choice",
        },
        {
            "mapping": "absolute_value_as_neural_head_mapping", "defined_by_frozen_contract": False,
            "nonnegative_by_construction": True, "exact_zero_representable": True, "all_p0_targets_representable": True,
            "finite_gradient_near_zero": True, "post_hoc_clipping": False, "mathematically_admissible_for_new_contract": True,
            "reason": "subdifferentiable and admissible in principle, but also unfrozen",
        },
        {
            "mapping": "softplus_as_neural_head_activation", "defined_by_frozen_contract": False,
            "nonnegative_by_construction": True, "exact_zero_representable": False, "all_p0_targets_representable": False,
            "finite_gradient_near_zero": True, "post_hoc_clipping": False, "mathematically_admissible_for_new_contract": False,
            "reason": "strictly positive output cannot exactly represent required zeros",
        },
        {
            "mapping": "detached_clamp_min_projection", "defined_by_frozen_contract": False,
            "nonnegative_by_construction": True, "exact_zero_representable": True, "all_p0_targets_representable": True,
            "finite_gradient_near_zero": True, "post_hoc_clipping": True, "mathematically_admissible_for_new_contract": False,
            "reason": "Thayer-OC defines this only as detached audit projection; coverage clipping is prohibited",
        },
    ]
    write_csv_fresh(run / "tables/output_mapping_uniqueness_audit.csv", candidates)
    selected_by_contract = [row for row in candidates if row["defined_by_frozen_contract"] and row["mathematically_admissible_for_new_contract"]]
    admissible_new = [row for row in candidates if row["mathematically_admissible_for_new_contract"]]
    unique = len(selected_by_contract) == 1 and len(admissible_new) == 1

    first_epoch = int(first_negative["epoch"]) if first_negative else None
    report = f"""# Physical output-contract audit

Decision: **FAIL-CLOSED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING**.

## Negative-output provenance

The frozen P0 targets are finite and nonnegative in both normalized and physical space. Thayer-FP's decoder ends in an unconstrained linear `1x1` head. It applies no activation: raw normalized output and post-activation normalized output are identical. Inverse normalization multiplies each channel by a strictly positive scale (`{', '.join(f'{value:.6f}' for value in scales)}`), so it cannot create or remove a negative sign. Negative values therefore entered at the raw linear decoder head and remained negative in the final physical source layers used by scientific evaluation.

The first persisted violation is the end-of-epoch-{first_epoch} evaluation: normalized minimum `{float(first_negative['output_minimum_normalized']):.9f}` and negative fraction `{float(first_negative['negative_output_fraction']):.9f}`. The historical loop did not record batch-level output-contract checks, so the exact first violating batch is **UNRESOLVED**; it cannot be reconstructed from the persisted trajectory without rerunning a protocol that was not recorded. At the stored best-epoch-395 output, the physical negative fraction is `{float(np.mean(fp_physical < -1e-7)):.9f}` and the physical minimum is `{float(np.min(fp_physical)):.9f}` detected electrons.

## Target round trip

The frozen normalized-to-physical mapping is positive per-band scale multiplication. Stored P0 physical tensors match normalized tensors multiplied by those scales with maximum absolute error `{float(np.max(np.abs(reconstructed_physical - p0_physical))):.9g}`, within the frozen float32 inversion tolerance. Every P0 channel is finite and nonnegative. This establishes target-domain validity but does not choose a neural output mapping.

## Uniqueness gate

The historical identity head is the only mapping defined by the frozen neural contracts, and it is not nonnegative by construction. The Thayer-OC `clamp_min` function is explicitly a detached audit projection, not model code, and post-hoc clipping may not manufacture coverage. At least three distinct new mappings—ReLU head activation, squaring, and absolute value—are mathematically capable of representing the nonnegative P0 range including zero while keeping finite (sub)gradients near zero. None is selected by an existing frozen contract.

Thus there are `{len(selected_by_contract)}` contract-selected eligible mappings and `{len(admissible_new)}` distinct unfrozen mathematically admissible mappings. Selecting one here would be an unpreregistered output-parameterization choice. Under Part D and the preregistered uniqueness gate, the campaign stops before decoder construction, synthetic head fitting, one/eight-scene gates, or capacity-ladder training.

## Reproduction and boundary

All `{len(checks)}` Thayer-FP reproduction checks passed. P0 retained 100% ordinary, own, alternate, and both-mode target-set coverage; median alpha/correction and z-band limiting count reproduced. The diagnostic neural trajectory retained zero coverage, diameter `{neural_metrics['ordinary_median_expert_diameter']:.6f}`, prompt swap `{neural_metrics['set_prompt_swap']:.6f}`, forward consistency `{neural_metrics['ordinary_forward_consistency']:.6f}/{neural_metrics['ambiguous_forward_consistency']:.6f}`, and final negative fraction `{float(np.mean(fp_normalized < -1e-7)):.6f}`.

No model was constructed, no optimizer step occurred, and Atlas, development, and lockbox access counts remain zero.
"""
    write_text_fresh(run / "diagnostics/physical_output_contract.md", report)

    stop = {
        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "PART_D_PHYSICAL_OUTPUT_CONTRACT_AUDIT",
        "reason": "NO_UNIQUE_CONTRACT_COMPLIANT_OUTPUT_MAPPING",
        "contract_selected_eligible_mapping_count": len(selected_by_contract),
        "unfrozen_mathematically_admissible_mapping_count": len(admissible_new),
        "historical_identity_mapping_nonnegative": False,
        "first_persisted_negative_epoch": first_epoch,
        "first_negative_batch": "UNRESOLVED_NOT_PERSISTED",
        "physical_negatives_confirmed": True,
        "model_construction_count": 0,
        "synthetic_head_optimizer_step_count": 0,
        "neural_optimizer_step_count": 0,
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
        "capacity_ladder_authorized": False,
        "required_next_campaign": "SEPARATE_PREREGISTERED_OUTPUT_PARAMETERIZATION_CAMPAIGN",
    }
    write_json_fresh(run / "logs/fail_closed_stop.json", stop)
    write_json_fresh(run / "logs/contract_audit_complete.json", {
        "status": "FAIL_CLOSED_NO_UNIQUE_MAPPING",
        "preregistration_sha256": freeze["preregistration_sha256"],
        "fp_reproduction_check_count": len(checks),
        "fp_reproduction_failure_count": 0,
        "p0_attrs": p0_attrs,
        "fp_output_attrs": fp_output_attrs,
        "target_roundtrip_max_abs_physical": float(np.max(np.abs(reconstructed_physical - p0_physical))),
        "target_roundtrip_max_abs_normalized": float(np.max(np.abs(reconstructed_normalized - p0_normalized))),
        "model_construction_count": 0,
        "optimizer_step_count": 0,
    })
    print(json.dumps(stop, sort_keys=True))


if __name__ == "__main__":
    main()
