#!/usr/bin/env python3
"""Run Thayer-OP representability, gradient, stop, synthetic, and MPS preflights."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bootstrap_thayer_output_parameterization import (
    CONDITION_C,
    EXPECTED,
    FORWARD_THRESHOLDS,
    NORMALIZATION,
    P0_HASHES,
    P0_TARGETS,
    sha256,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)
from scripts.run_thayer_two_expert_micro_overfit import require_mps
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.models_two_expert_decoder import parameter_count, warm_start_condition_c_encoder
from src.output_parameterization import (
    INITIAL_PHYSICAL_EPSILON,
    MAPPINGS,
    NUMERICAL_ZERO_TOLERANCE,
    PHYSICAL_NEGATIVE_TOLERANCE,
    ROUNDTRIP_PHYSICAL_ATOL,
    STAGNATION_DERIVATIVE_TOLERANCE,
    MappedThayerMixtureExperts,
    apply_output_mapping,
    decoder_parameter_count,
    encoder_tensor_sha256,
    freeze_encoder,
    initial_raw_bias,
    mapping_derivative,
    raw_inverse_witness,
)


SYNTHETIC_STEPS = 500
SYNTHETIC_LR = 0.03
SCALES = np.asarray([611.9199829101562, 1805.8800048828125, 1854.199951171875], dtype=np.float32)
SCALE6 = np.tile(SCALES, 2).astype(np.float32)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_frozen_run(run: Path) -> dict[str, object]:
    freeze = json.loads((run / "preregistration/freeze_record.json").read_text())
    prereg = run / "preregistration/fixed_l0_output_parameterization.md"
    if freeze["status"] != "FROZEN_BEFORE_PER_SCENE_LOAD":
        raise RuntimeError(f"preregistration did not authorize preflight: {freeze['status']}")
    if sha256(prereg) != freeze["preregistration_sha256"]:
        raise RuntimeError("preregistration hash mismatch")
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch before preflight: {path}")
    if sha256(REPO / "src/output_parameterization.py") != freeze["mapping_code_sha256"]:
        raise RuntimeError("frozen output-mapping implementation changed")
    return freeze


def numpy_mapping(raw: np.ndarray, mapping: str) -> np.ndarray:
    if mapping == "relu":
        return np.maximum(raw, np.float32(0.0)).astype(np.float32, copy=False)
    if mapping == "square":
        return np.square(raw, dtype=np.float32)
    if mapping == "absolute":
        return np.abs(raw).astype(np.float32, copy=False)
    raise ValueError(mapping)


def numpy_inverse(target: np.ndarray, mapping: str) -> np.ndarray:
    if mapping == "square":
        return np.sqrt(target, dtype=np.float32)
    return target.copy()


def target_test_cases(normalized: np.ndarray) -> dict[str, np.ndarray]:
    high = float(np.max(normalized))
    z_values = normalized[..., [2, 5], :, :]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    sparse = np.zeros((6, 4, 4), dtype=np.float32)
    sparse[0, 0, 0] = np.float32(0.1)
    sparse[2, 1, 2] = np.float32(1.0)
    sparse[4, 3, 1] = np.float32(2.0)
    z_extrema = np.zeros((6, 4, 4), dtype=np.float32)
    z_extrema[2, 0, 0] = np.float32(z_min)
    z_extrema[2, 1, 1] = np.float32(z_max)
    z_extrema[5, 2, 2] = np.float32(z_max)
    return {
        "exact_zero": np.zeros((6, 4, 4), dtype=np.float32),
        "numerical_near_zero": np.full((6, 4, 4), NUMERICAL_ZERO_TOLERANCE, dtype=np.float32),
        "sparse_positive": sparse,
        "constant_positive": np.full((6, 4, 4), 0.1, dtype=np.float32),
        "high_target_value": np.full((6, 4, 4), high, dtype=np.float32),
        "z_band_extrema": z_extrema,
    }


def representability_audit(
    run: Path,
    normalized: np.ndarray,
    physical: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    expected_hash_rows = read_csv(P0_HASHES)
    expected_hashes = {
        (int(row["scene"]), int(row["prompt"]), int(row["target_slot"])): row["canonical_sha256"]
        for row in expected_hash_rows
    }
    rows: list[dict[str, object]] = []
    eligible = {}
    cases = target_test_cases(normalized)
    shape = normalized.shape
    for mapping in MAPPINGS:
        raw = numpy_inverse(normalized, mapping)
        mapped_a = numpy_mapping(raw, mapping)
        mapped_b = numpy_mapping(raw.copy(), mapping)
        rebuilt_physical = mapped_a * SCALE6[None, None, None, :, None, None]
        max_physical_error = float(np.max(np.abs(rebuilt_physical - physical)))
        canonical_stable = True
        equal_target_hash_count = 0
        for scene in range(shape[0]):
            for prompt in range(shape[1]):
                for slot in range(shape[2]):
                    sample_a = rebuilt_physical[scene, prompt, slot]
                    sample_b = mapped_b[scene, prompt, slot] * SCALE6[:, None, None]
                    hash_a = canonical_tensor_sha256(sample_a)
                    hash_b = canonical_tensor_sha256(sample_b)
                    canonical_stable = canonical_stable and hash_a == hash_b
                    equal_target_hash_count += int(hash_a == expected_hashes[(scene, prompt, slot)])
        positive = normalized > 0
        derivative = np.ones_like(raw, dtype=np.float32)
        if mapping == "relu":
            derivative = (raw > 0).astype(np.float32)
        elif mapping == "square":
            derivative = 2.0 * raw
        elif mapping == "absolute":
            derivative = np.sign(raw).astype(np.float32)
        positive_saturation_fraction = float(np.mean((derivative == 0)[positive]))
        passed = bool(
            mapped_a.shape == normalized.shape
            and np.all(np.isfinite(raw))
            and np.all(np.isfinite(mapped_a))
            and float(np.min(mapped_a)) >= 0.0
            and max_physical_error <= ROUNDTRIP_PHYSICAL_ATOL
            and canonical_stable
            and positive_saturation_fraction == 0.0
        )
        rows.append(
            {
                "mapping": mapping,
                "test_case": "all_frozen_p0_targets",
                "target_count": int(np.prod(shape[:3])),
                "shape": "x".join(map(str, shape)),
                "finite": bool(np.all(np.isfinite(mapped_a))),
                "nonnegative": bool(float(np.min(mapped_a)) >= 0.0),
                "maximum_physical_roundtrip_error": max_physical_error,
                "physical_roundtrip_atol": ROUNDTRIP_PHYSICAL_ATOL,
                "positive_target_saturation_fraction": positive_saturation_fraction,
                "canonical_hash_stable": canonical_stable,
                "target_hash_equal_fraction": equal_target_hash_count / len(expected_hashes),
                "pass": passed,
            }
        )
        for case_name, target in cases.items():
            witness = numpy_inverse(target, mapping)
            mapped = numpy_mapping(witness, mapping)
            error = float(
                np.max(
                    np.abs(
                        (mapped - target) * SCALE6[:, None, None]
                    )
                )
            )
            case_pass = bool(
                mapped.shape == target.shape
                and np.all(np.isfinite(witness))
                and np.all(np.isfinite(mapped))
                and float(np.min(mapped)) >= 0.0
                and error <= ROUNDTRIP_PHYSICAL_ATOL
                and canonical_tensor_sha256(mapped * SCALE6[:, None, None])
                == canonical_tensor_sha256(numpy_mapping(witness.copy(), mapping) * SCALE6[:, None, None])
            )
            rows.append(
                {
                    "mapping": mapping,
                    "test_case": case_name,
                    "target_count": 1,
                    "shape": "6x4x4",
                    "finite": bool(np.all(np.isfinite(mapped))),
                    "nonnegative": bool(float(np.min(mapped)) >= 0.0),
                    "maximum_physical_roundtrip_error": error,
                    "physical_roundtrip_atol": ROUNDTRIP_PHYSICAL_ATOL,
                    "positive_target_saturation_fraction": 0.0,
                    "canonical_hash_stable": True,
                    "target_hash_equal_fraction": 1.0 if np.array_equal(mapped, target) else 0.0,
                    "pass": case_pass,
                }
            )
        eligible[mapping] = bool(passed and all(bool(row["pass"]) for row in rows if row["mapping"] == mapping))
    write_csv_fresh(run / "tables/mapping_representability.csv", rows)
    write_text_fresh(
        run / "diagnostics/mapping_representability_report.md",
        "# Mapping representability report\n\n"
        + "\n".join(
            f"- {mapping}: **{'PASS' if eligible[mapping] else 'FAIL'}**; every frozen P0 target and six boundary/range cases were audited."
            for mapping in MAPPINGS
        )
        + f"\n\nThe frozen physical round-trip tolerance is `{ROUNDTRIP_PHYSICAL_ATOL}` detected electrons. Canonical stability means two independent applications of the same frozen witness produce the same versioned physical CHW hash; square is not required to be byte-identical to the original target when its deterministic float32 sqrt-square round trip remains within tolerance.\n",
    )
    return rows, eligible


def initial_output_audit(run: Path) -> tuple[list[dict[str, object]], bool]:
    rows = []
    reference = None
    identical = True
    for mapping in MAPPINGS:
        raw_value = np.float32(initial_raw_bias(mapping))
        raw = torch.full((1, 2, 6, 4, 4), float(raw_value), dtype=torch.float32)
        mapped = apply_output_mapping(raw, mapping).detach().numpy()
        physical = mapped * SCALE6[None, None, :, None, None]
        if reference is None:
            reference = mapped.copy()
        else:
            identical = identical and np.array_equal(reference, mapped)
        derivative = mapping_derivative(raw, mapping).detach().numpy()
        rows.append(
            {
                "mapping": mapping,
                "raw_bias": float(raw_value),
                "initial_mapped_normalized_mean": float(mapped.mean()),
                "initial_mapped_normalized_variance": float(mapped.var()),
                "initial_physical_mean": float(physical.mean()),
                "initial_physical_variance": float(physical.var()),
                "requested_g_electrons": float(physical[0, 0, 0, 0, 0]),
                "requested_r_electrons": float(physical[0, 0, 1, 0, 0]),
                "requested_z_electrons": float(physical[0, 0, 2, 0, 0]),
                "companion_g_electrons": float(physical[0, 0, 3, 0, 0]),
                "companion_r_electrons": float(physical[0, 0, 4, 0, 0]),
                "companion_z_electrons": float(physical[0, 0, 5, 0, 0]),
                "local_mapping_derivative": float(derivative.flat[0]),
                "zero_gradient_fraction": float(np.mean(derivative == 0)),
                "matched_initial_tensor": True,
            }
        )
    for row in rows:
        row["matched_initial_tensor"] = identical
    write_csv_fresh(run / "tables/initial_output_match.csv", rows)
    return rows, identical


def gradient_audit(run: Path, normalized: np.ndarray) -> tuple[list[dict[str, object]], dict[str, bool]]:
    positive = normalized[normalized > 0]
    z_positive = normalized[..., [2, 5], :, :]
    points = {
        "initialization": INITIAL_PHYSICAL_EPSILON,
        "numerical_zero": 0.0,
        "low_positive_output": 1e-6,
        "median_positive_target": float(np.median(positive)),
        "high_target": float(np.max(positive)),
        "z_band_minimum": float(np.min(z_positive[z_positive > 0])),
        "z_band_maximum": float(np.max(z_positive)),
    }
    rows = []
    eligible = {}
    for mapping in MAPPINGS:
        material_bad = 0
        material_count = 0
        for point_name, target_value in points.items():
            target = torch.tensor([target_value], dtype=torch.float32)
            raw = raw_inverse_witness(target, mapping).detach().clone().requires_grad_(True)
            mapped = apply_output_mapping(raw, mapping)
            mapped.sum().backward()
            grad = raw.grad.detach()
            analytic = mapping_derivative(raw.detach(), mapping)
            perturb = raw.detach() - NUMERICAL_ZERO_TOLERANCE
            perturbed_output = apply_output_mapping(perturb, mapping)
            derivative_abs = float(torch.abs(analytic).item())
            is_material = target_value > 0 and point_name not in {"initialization"}
            unusable = is_material and (not math.isfinite(derivative_abs) or derivative_abs == 0.0)
            material_count += int(is_material)
            material_bad += int(unusable)
            rows.append(
                {
                    "mapping": mapping,
                    "audit_point": point_name,
                    "target_normalized": target_value,
                    "raw_witness": float(raw.detach().item()),
                    "mapped_normalized": float(mapped.detach().item()),
                    "raw_to_mapped_derivative": float(analytic.item()),
                    "autograd_derivative": float(grad.item()),
                    "zero_gradient_fraction": float(grad.item() == 0.0),
                    "stagnant_derivative_fraction": float(abs(float(grad.item())) <= STAGNATION_DERIVATIVE_TOLERANCE),
                    "nonfinite_gradient_fraction": float(not math.isfinite(float(grad.item()))),
                    "gradient_norm": float(torch.linalg.vector_norm(grad).item()),
                    "negative_raw_perturbation": float(perturb.item()),
                    "perturbed_mapped_output": float(perturbed_output.item()),
                    "material_target_support": is_material,
                    "material_gradient_usable": not unusable,
                }
            )
        eligible[mapping] = material_count > 0 and material_bad == 0
    write_csv_fresh(run / "tables/gradient_numerical_preflight.csv", rows)
    risks = {
        "relu": "zero derivative on the nonpositive raw half-line; positive P0 witnesses remain active",
        "square": "sign symmetry and derivative shrinking toward raw zero; positive P0 witnesses remain nonzero",
        "absolute": "nondifferentiable cusp with PyTorch subgradient zero at raw zero; positive witnesses have derivative one",
    }
    write_text_fresh(
        run / "diagnostics/gradient_numerical_report.md",
        "# Gradient and numerical preflight\n\n"
        + "\n".join(
            f"- {mapping}: **{'PASS' if eligible[mapping] else 'FAIL'}**. {risks[mapping]}."
            for mapping in MAPPINGS
        )
        + "\n\nExact-boundary subgradients are reported rather than used alone as disqualifiers. Eligibility depends on usable finite derivatives throughout sampled strictly positive P0 support.\n",
    )
    return rows, eligible


class ExpectedStop(RuntimeError):
    pass


def injected_guard(
    test_dir: Path,
    *,
    tensor: torch.Tensor | None = None,
    target_hash_ok: bool = True,
    mps_device_ok: bool = True,
    optimizer_steps: int = 0,
) -> None:
    reason = None
    if not target_hash_ok:
        reason = "TARGET_HASH_MISMATCH"
    elif not mps_device_ok:
        reason = "MPS_FALLBACK"
    elif tensor is not None and not bool(torch.all(torch.isfinite(tensor))):
        if bool(torch.any(torch.isnan(tensor))):
            reason = "NAN_PHYSICAL_OUTPUT"
        else:
            reason = "INF_PHYSICAL_OUTPUT"
    elif tensor is not None and float(torch.min(tensor).item()) < PHYSICAL_NEGATIVE_TOLERANCE:
        reason = "NEGATIVE_PHYSICAL_OUTPUT"
    if reason is None:
        return
    test_dir.mkdir(parents=True, exist_ok=False)
    write_json_fresh(
        test_dir / "incident.json",
        {
            "detected_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "run_status": "FAILED_EXPECTED_SENTINEL",
            "optimizer_steps_before": optimizer_steps,
            "optimizer_steps_after": optimizer_steps,
            "checkpoint_promoted": False,
            "synchronous": True,
        },
    )
    raise ExpectedStop(reason)


def stop_rule_self_tests(run: Path) -> tuple[list[dict[str, object]], bool]:
    root = run / "output_contract/stop_self_tests"
    root.mkdir(exist_ok=False)
    cases = {
        "negative_physical_output": {"tensor": torch.tensor([-1e-6], dtype=torch.float32)},
        "nan_physical_output": {"tensor": torch.tensor([float("nan")], dtype=torch.float32)},
        "inf_physical_output": {"tensor": torch.tensor([float("inf")], dtype=torch.float32)},
        "target_hash_mismatch": {"target_hash_ok": False},
        "mps_fallback_simulation": {"mps_device_ok": False},
    }
    rows = []
    for name, kwargs in cases.items():
        caught = False
        reason = ""
        try:
            injected_guard(root / name, optimizer_steps=0, **kwargs)
        except ExpectedStop as exc:
            caught = True
            reason = str(exc)
        incident = root / name / "incident.json"
        payload = json.loads(incident.read_text()) if incident.is_file() else {}
        passed = bool(
            caught
            and incident.is_file()
            and payload.get("run_status") == "FAILED_EXPECTED_SENTINEL"
            and payload.get("optimizer_steps_before") == payload.get("optimizer_steps_after") == 0
            and payload.get("checkpoint_promoted") is False
            and payload.get("synchronous") is True
        )
        rows.append(
            {
                "self_test": name,
                "expected_stop_caught": caught,
                "reason": reason,
                "incident_written_before_termination": incident.is_file(),
                "optimizer_step_advanced": False,
                "checkpoint_promoted": False,
                "local_run_status_failed": payload.get("run_status") == "FAILED_EXPECTED_SENTINEL",
                "pass": passed,
            }
        )
    write_csv_fresh(run / "tables/stop_rule_self_tests.csv", rows)
    write_text_fresh(
        run / "diagnostics/stop_rule_self_test_report.md",
        "# Fail-closed stop-rule self-test\n\n"
        + "\n".join(
            f"- {row['self_test']}: **{'PASS' if row['pass'] else 'FAIL'}** ({row['reason']})."
            for row in rows
        )
        + "\n\nEvery incident was an isolated expected-failure path. The main campaign remains eligible only because each sentinel stopped locally before an optimizer step and promoted no checkpoint.\n",
    )
    return rows, all(bool(row["pass"]) for row in rows)


def z_dominant_crop(normalized: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    z = normalized[..., [2, 5], :, :]
    flat = int(np.argmax(z))
    scene, prompt, slot, z_choice, y, x = np.unravel_index(flat, z.shape)
    channel = (2, 5)[z_choice]
    y0 = min(max(y - 2, 0), 56)
    x0 = min(max(x - 2, 0), 56)
    crop = normalized[scene, prompt, slot, :, y0 : y0 + 4, x0 : x0 + 4]
    return crop.copy(), {
        "scene": int(scene),
        "prompt": int(prompt),
        "target_slot": int(slot),
        "z_channel": int(channel),
        "y_start": int(y0),
        "x_start": int(x0),
    }


def synthetic_targets(normalized: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    sparse = np.zeros((6, 4, 4), dtype=np.float32)
    sparse[0, 0, 0] = 0.05
    sparse[2, 1, 2] = 0.50
    sparse[4, 3, 1] = 1.00
    z_crop, provenance = z_dominant_crop(normalized)
    return {
        "zero_tensor": np.zeros((6, 4, 4), dtype=np.float32),
        "constant_positive_tensor": np.full((6, 4, 4), 0.1, dtype=np.float32),
        "sparse_positive_tensor": sparse,
        "projected_target_crop": normalized[0, 0, 0, :, 28:32, 28:32].copy(),
        "z_band_dominant_target_crop": z_crop,
    }, provenance


def synthetic_fit_audit(
    run: Path,
    normalized: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    targets, z_provenance = synthetic_targets(normalized)
    write_json_fresh(run / "synthetic_preflight/z_crop_provenance.json", z_provenance)
    feature = torch.zeros((1, 16, 4, 4), dtype=torch.float32, device=device)
    for index in range(16):
        feature[0, index, index // 4, index % 4] = 1.0
    scale = torch.from_numpy(SCALE6).to(device).view(1, 6, 1, 1)
    curves = []
    summaries = []
    eligible = {mapping: True for mapping in MAPPINGS}
    for mapping in MAPPINGS:
        for case_name, target_np in targets.items():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026071250)
                head = nn.Conv2d(16, 6, 1).to(device)
            with torch.no_grad():
                head.weight.zero_()
                head.bias.fill_(initial_raw_bias(mapping))
            optimizer = torch.optim.AdamW(head.parameters(), lr=SYNTHETIC_LR, weight_decay=0.0)
            target = torch.from_numpy(target_np[None]).to(device)
            target_physical = target * scale
            initial_loss = None
            final_loss = None
            finite_gradients = True
            negative_count = 0
            minimum_physical = float("inf")
            for step in range(SYNTHETIC_STEPS + 1):
                raw = head(feature)
                mapped = apply_output_mapping(raw, mapping)
                physical = mapped * scale
                weighted_residual = (physical - target_physical) / scale
                loss = weighted_residual.square().mean()
                if initial_loss is None:
                    initial_loss = float(loss.detach().cpu())
                final_loss = float(loss.detach().cpu())
                minimum_physical = min(minimum_physical, float(physical.detach().min().cpu()))
                negative_count += int(float(physical.detach().min().cpu()) < PHYSICAL_NEGATIVE_TOLERANCE)
                if step in {0, 1, SYNTHETIC_STEPS} or step % 10 == 0:
                    derivative = mapping_derivative(raw.detach(), mapping)
                    curves.append(
                        {
                            "mapping": mapping,
                            "case": case_name,
                            "step": step,
                            "loss": final_loss,
                            "physical_minimum": float(physical.detach().min().cpu()),
                            "zero_gradient_fraction": float((derivative == 0).float().mean().cpu()),
                            "stagnation_fraction": float((torch.abs(derivative) <= STAGNATION_DERIVATIVE_TOLERANCE).float().mean().cpu()),
                        }
                    )
                if step == SYNTHETIC_STEPS:
                    break
                if not bool(torch.isfinite(loss).detach().cpu()):
                    finite_gradients = False
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
                finite_gradients = finite_gradients and all(bool(torch.all(torch.isfinite(grad)).detach().cpu()) for grad in grads)
                if not finite_gradients:
                    break
                optimizer.step()
            assert initial_loss is not None and final_loss is not None
            reduction = 0.0 if initial_loss == 0.0 else 1.0 - final_loss / initial_loss
            if case_name == "zero_tensor":
                fit_pass = final_loss <= 1e-12
            else:
                fit_pass = reduction >= 0.95
            passed = bool(fit_pass and finite_gradients and negative_count == 0 and math.isfinite(final_loss))
            eligible[mapping] = eligible[mapping] and passed
            summaries.append(
                {
                    "mapping": mapping,
                    "case": case_name,
                    "device": str(device),
                    "optimizer": "AdamW",
                    "learning_rate": SYNTHETIC_LR,
                    "optimizer_steps": SYNTHETIC_STEPS,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "loss_reduction_fraction": reduction,
                    "finite_gradients": finite_gradients,
                    "physical_negative_events": negative_count,
                    "minimum_physical": minimum_physical,
                    "hidden_posthoc_clipping": False,
                    "pass": passed,
                }
            )
    write_csv_fresh(run / "synthetic_preflight/synthetic_fit_curves.csv", curves)
    write_csv_fresh(run / "tables/synthetic_fit_summary.csv", summaries)
    write_text_fresh(
        run / "synthetic_preflight/synthetic_fit_report.md",
        "# Synthetic L0 output-head fits\n\n"
        + "\n".join(
            f"- {mapping}: **{'PASS' if eligible[mapping] else 'FAIL'}** across zero, constant, sparse, P0-crop, and z-dominant P0-crop targets."
            for mapping in MAPPINGS
        )
        + "\n\nOnly the 16-to-6 L0 output head trained. The encoder was bypassed, the 4x4 spatial basis was deterministic, all optimization ran on MPS, and no post-hoc clipping was used.\n",
    )
    return summaries, eligible


def memory_probes(
    run: Path,
    eligible: dict[str, bool],
    device: torch.device,
) -> tuple[list[dict[str, object]], bool, str | None]:
    rows = []
    reference_encoder_hash = None
    all_pass = True
    for mapping in MAPPINGS:
        if not eligible[mapping]:
            rows.append(
                {
                    "mapping": mapping,
                    "eligible_for_probe": False,
                    "device": "NOT_RUN",
                    "microbatch_size": 8,
                    "effective_batch_size": 8,
                    "encoder_hash_before": "",
                    "encoder_hash_after": "",
                    "encoder_byte_identical": True,
                    "parameters_per_expert": 46470,
                    "total_parameters": 165612,
                    "current_allocated_bytes": 0,
                    "driver_allocated_bytes": 0,
                    "pass": False,
                }
            )
            all_pass = False
            continue
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        model = MappedThayerMixtureExperts(mapping, torch.from_numpy(SCALES))
        warm_start_condition_c_encoder(model, CONDITION_C)
        freeze_encoder(model)
        before = encoder_tensor_sha256(model)
        reference_encoder_hash = before if reference_encoder_hash is None else reference_encoder_hash
        same_reference = before == reference_encoder_hash
        counts = decoder_parameter_count(model)
        model = model.to(device)
        model.encoder.eval()
        blend = torch.zeros((8, 3, 60, 60), dtype=torch.float32, device=device)
        prompt = torch.zeros((8, 1, 60, 60), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(
            list(model.expert_1.parameters()) + list(model.expert_2.parameters()),
            lr=1e-3,
            weight_decay=0.0,
        )
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        output = model.forward_outputs(blend, prompt)
        loss = output.physical.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = bool(torch.isfinite(loss).detach().cpu()) and all(
            bool(torch.all(torch.isfinite(parameter.grad)).detach().cpu())
            for parameter in list(model.expert_1.parameters()) + list(model.expert_2.parameters())
            if parameter.grad is not None
        )
        after = encoder_tensor_sha256(model)
        current = int(torch.mps.current_allocated_memory()) if hasattr(torch.mps, "current_allocated_memory") else 0
        driver = int(torch.mps.driver_allocated_memory()) if hasattr(torch.mps, "driver_allocated_memory") else 0
        passed = bool(
            finite
            and output.physical.device.type == "mps"
            and float(output.physical.min().detach().cpu()) >= PHYSICAL_NEGATIVE_TOLERANCE
            and before == after
            and same_reference
            and counts == (46470, 46470)
            and parameter_count(model) == 165612
            and all(not parameter.requires_grad for parameter in model.encoder.parameters())
            and not any(id(parameter) in optimizer_parameter_ids for parameter in model.encoder.parameters())
        )
        rows.append(
            {
                "mapping": mapping,
                "eligible_for_probe": True,
                "device": output.physical.device.type,
                "microbatch_size": 8,
                "effective_batch_size": 8,
                "encoder_hash_before": before,
                "encoder_hash_after": after,
                "encoder_byte_identical": before == after and same_reference,
                "parameters_per_expert": counts[0],
                "total_parameters": parameter_count(model),
                "current_allocated_bytes": current,
                "driver_allocated_bytes": driver,
                "pass": passed,
            }
        )
        all_pass = all_pass and passed
        del optimizer, loss, output, model, blend, prompt
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    write_csv_fresh(run / "tables/mps_memory_probe.csv", rows)
    write_text_fresh(
        run / "diagnostics/common_compute_contract.md",
        "# Common compute contract\n\n"
        "The frozen microbatch and effective batch sizes are both 8 with accumulation 1. Every real mapping condition receives exactly 3,200 AdamW steps and 25,600 scene presentations, fixed row order, learning rate 1e-3, zero weight decay, no scheduler, and gradient clipping at 5.0. Synthetic MPS probes used the exact L0 encoder/decoder topology and confirmed a common batch of eight before scene fitting.\n\n"
        + "\n".join(
            f"- {row['mapping']}: {row['device']}; current/driver allocated bytes {row['current_allocated_bytes']}/{row['driver_allocated_bytes']}; **{'PASS' if row['pass'] else 'FAIL'}**."
            for row in rows
        )
        + "\n",
    )
    return rows, all_pass, reference_encoder_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    freeze = validate_frozen_run(run)
    started = time.time()
    write_json_fresh(
        run / "logs/p0_representability_load_started.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_sha256": freeze["preregistration_sha256"],
            "per_scene_input_load_count": 0,
            "p0_target_tensor_load_count": 1,
        },
    )
    with h5py.File(P0_TARGETS, "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("P0 target file is incomplete")
        normalized = np.asarray(handle["targets_normalized"], dtype=np.float32)
        physical = np.asarray(handle["targets_physical"], dtype=np.float32)
    if normalized.shape != (64, 2, 2, 6, 60, 60) or physical.shape != normalized.shape:
        raise RuntimeError("P0 target shape mismatch")
    if not np.array_equal(normalized * SCALE6[None, None, None, :, None, None], physical):
        raise RuntimeError("P0 normalized/physical source path mismatch")

    _, representable = representability_audit(run, normalized, physical)
    _, initial_match = initial_output_audit(run)
    _, gradients = gradient_audit(run, normalized)
    _, self_tests = stop_rule_self_tests(run)
    device = require_mps()
    _, synthetic = synthetic_fit_audit(run, normalized, device)
    preprobe_eligible = {
        mapping: bool(representable[mapping] and gradients[mapping] and synthetic[mapping] and self_tests and initial_match)
        for mapping in MAPPINGS
    }
    _, probes_passed, reference_encoder_hash = memory_probes(run, preprobe_eligible, device)
    eligible = {
        mapping: bool(preprobe_eligible[mapping] and next(row for row in read_csv(run / "tables/mps_memory_probe.csv") if row["mapping"] == mapping)["pass"] == "True")
        for mapping in MAPPINGS
    }
    status = "PASS" if all(eligible.values()) and probes_passed else "FAIL"
    complete = {
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "preregistration_sha256": freeze["preregistration_sha256"],
        "p0_target_file_sha256": sha256(P0_TARGETS),
        "p0_hash_table_sha256": sha256(P0_HASHES),
        "representability": representable,
        "gradient_eligibility": gradients,
        "stop_rule_self_tests_passed": self_tests,
        "synthetic_fit_eligibility": synthetic,
        "initial_outputs_matched": initial_match,
        "mps_memory_probes_passed": probes_passed,
        "eligible_mappings": [mapping for mapping in MAPPINGS if eligible[mapping]],
        "ineligible_mappings": [mapping for mapping in MAPPINGS if not eligible[mapping]],
        "reference_encoder_tensor_sha256": reference_encoder_hash,
        "p0_target_tensor_load_count": 1,
        "per_scene_input_load_count": 0,
        "neural_training_device": "mps",
        "cpu_neural_fallback": False,
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
    }
    write_json_fresh(run / "logs/preflight_complete.json", complete)
    print(json.dumps(complete, indent=2, sort_keys=True))
    if status != "PASS":
        raise RuntimeError("Thayer-OP preflight failed closed")


if __name__ == "__main__":
    main()
