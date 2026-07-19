#!/usr/bin/env python3
"""Run matched MPS-only one-scene and eight-scene Thayer-OP gates."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bootstrap_thayer_output_parameterization import (
    ATLAS_NOISE,
    CONDITION_C,
    EIGHT_SCENES,
    EFFECTIVE_BATCH_SIZE,
    EXPECTED,
    FORWARD_THRESHOLDS,
    LEARNING_RATE,
    MICROSET,
    NORMALIZATION,
    OPTIMIZER_STEPS,
    P0_HASHES,
    P0_TARGETS,
    PHYSICAL_NEGATIVE_TOLERANCE,
    sha256,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)
from scripts.run_thayer_two_expert_micro_overfit import load_micro_arrays, require_mps
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.competing_hypotheses import (
    PlausibilityThresholds,
    forward_consistency,
    is_plausible,
    scientific_distance,
)
from src.models_two_expert_decoder import parameter_count, warm_start_condition_c_encoder
from src.output_parameterization import (
    MAPPINGS,
    NUMERICAL_ZERO_TOLERANCE,
    STAGNATION_DERIVATIVE_TOLERANCE,
    MappedThayerMixtureExperts,
    decoder_parameter_count,
    encoder_tensor_sha256,
    freeze_encoder,
    mapping_derivative,
)


SCALES = np.asarray([611.9199829101562, 1805.8800048828125, 1854.199951171875], dtype=np.float32)
SCALE6 = np.tile(SCALES, 2).astype(np.float32)
MEAN_PSF_FWHM_PIXEL = float(np.mean([0.86, 0.81, 0.77]) / 0.2)
EVALUATION_STEPS = {0, 1, OPTIMIZER_STEPS} | set(range(100, OPTIMIZER_STEPS + 1, 100))


class ContractViolation(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frozen_thresholds() -> tuple[PlausibilityThresholds, np.ndarray]:
    raw = json.loads(FORWARD_THRESHOLDS.read_text())
    noise = json.loads(ATLAS_NOISE.read_text())
    return (
        PlausibilityThresholds(
            float(raw["global_chi_square_mean"]),
            tuple(float(value) for value in raw["per_band_chi_square_mean"]),
            float(raw["absolute_relative_flux_residual"]),
            int(raw["calibration_scene_count"]),
            0.99,
            0.995,
            0.99,
        ),
        np.asarray(noise["sky_electrons_grz"], dtype=np.float64),
    )


def model_topology_sha256(model: MappedThayerMixtureExperts) -> str:
    digest = hashlib.sha256()
    for expert_name, expert in (("expert_1", model.expert_1), ("expert_2", model.expert_2)):
        for name, tensor in expert.state_dict().items():
            digest.update(f"{expert_name}.{name}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode("utf-8"))
    return digest.hexdigest()


def physical_direct_cost(
    predicted_physical: torch.Tensor,
    target_physical: torch.Tensor,
    scale6: torch.Tensor,
) -> torch.Tensor:
    view = (1,) * (predicted_physical.ndim - 3) + (6, 1, 1)
    residual = (predicted_physical - target_physical) / scale6.view(view)
    requested = residual[..., :3, :, :].square().mean(dim=(-3, -2, -1))
    companion = residual[..., 3:, :, :].square().mean(dim=(-3, -2, -1))
    return requested + companion


def hard_physical_set_loss(
    outputs_physical: torch.Tensor,
    targets_physical: torch.Tensor,
    scale6: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c00 = physical_direct_cost(outputs_physical[:, 0], targets_physical[:, 0], scale6)
    c10 = physical_direct_cost(outputs_physical[:, 1], targets_physical[:, 0], scale6)
    c01 = physical_direct_cost(outputs_physical[:, 0], targets_physical[:, 1], scale6)
    c11 = physical_direct_cost(outputs_physical[:, 1], targets_physical[:, 1], scale6)
    identity = c00 + c11
    swapped = c01 + c10
    return torch.minimum(identity, swapped).mean(), identity <= swapped, swapped - identity


def weighted_mse_physical(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> float:
    view = (1,) * (left.ndim - 3) + (len(scales), 1, 1)
    return float(np.mean(((left - right) / scales.reshape(view)) ** 2))


def prompt_identity_physical(output: np.ndarray, targets: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros(2, dtype=bool)
    for expert in (0, 1):
        requested = output[expert, :3]
        requested_cost = min(weighted_mse_physical(requested, targets[index, :3], SCALES) for index in range(count))
        companion_cost = min(weighted_mse_physical(requested, targets[index, 3:], SCALES) for index in range(count))
        result[expert] = requested_cost < companion_cost
    return result


def projected_assignment(
    outputs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, bool, float, list[int], float]:
    c00 = weighted_mse_physical(outputs[0], targets[0], SCALE6)
    c10 = weighted_mse_physical(outputs[1], targets[0], SCALE6)
    c01 = weighted_mse_physical(outputs[0], targets[1], SCALE6)
    c11 = weighted_mse_physical(outputs[1], targets[1], SCALE6)
    identity = c00 + c11
    swapped = c01 + c10
    identity_wins = identity <= swapped
    assignment = [0, 1] if identity_wins else [1, 0]
    loss = identity if identity_wins else swapped
    z_errors = []
    for expert, target_index in enumerate(assignment):
        for channel in (2, 5):
            z_errors.append(float(np.mean(((outputs[expert, channel] - targets[target_index, channel]) / SCALE6[channel]) ** 2)))
    return loss, identity_wins, swapped - identity, assignment, float(np.mean(z_errors))


def subset_arrays(arrays: dict[str, np.ndarray], positions: list[int]) -> dict[str, np.ndarray]:
    return {name: value[positions] for name, value in arrays.items()}


def evaluate_condition(
    model: MappedThayerMixtureExperts,
    mapping: str,
    arrays: dict[str, np.ndarray],
    projected_physical: np.ndarray,
    rows: list[dict[str, object]],
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    model.encoder.eval()
    model.expert_1.eval()
    model.expert_2.eval()
    blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"])).to(device)
    prompt_a = torch.from_numpy(np.ascontiguousarray(arrays["prompt_a"])).to(device)
    prompt_b = torch.from_numpy(np.ascontiguousarray(arrays["prompt_b"])).to(device)
    with torch.no_grad():
        joined = model.forward_outputs(torch.cat((blend, blend)), torch.cat((prompt_a, prompt_b)))
    count = len(blend)
    physical = np.stack(
        (
            joined.physical[:count].cpu().numpy(),
            joined.physical[count:].cpu().numpy(),
        ),
        axis=1,
    )
    raw = np.stack(
        (
            joined.raw_normalized[:count].cpu().numpy(),
            joined.raw_normalized[count:].cpu().numpy(),
        ),
        axis=1,
    )
    mapped = np.stack(
        (
            joined.mapped_normalized[:count].cpu().numpy(),
            joined.mapped_normalized[count:].cpu().numpy(),
        ),
        axis=1,
    )
    threshold, sky = frozen_thresholds()
    per_scene: list[dict[str, object]] = []
    ordinary_diameters = []
    projected_losses = []
    z_errors = []
    assignment_margins = []
    identity_wins_all = []
    source_sum_errors = []
    forward_finite = []
    for scene_index, row in enumerate(rows):
        target_count = int(arrays["counts"][scene_index, 0])
        plausible = np.zeros((2, 2), dtype=bool)
        own = np.zeros((2, 2), dtype=bool)
        alternate = np.zeros((2, 2), dtype=bool)
        identities = np.zeros((2, 2), dtype=bool)
        diameters = []
        scene_assignment = []
        scene_z = []
        for prompt_index in (0, 1):
            identities[prompt_index] = prompt_identity_physical(
                physical[scene_index, prompt_index],
                arrays["targets_physical"][scene_index, prompt_index],
                target_count,
            )
            fit_loss, identity_win, margin, assignment, z_error = projected_assignment(
                physical[scene_index, prompt_index],
                projected_physical[scene_index, prompt_index],
            )
            projected_losses.append(fit_loss)
            z_errors.append(z_error)
            scene_z.append(z_error)
            assignment_margins.append(abs(margin))
            identity_wins_all.append(identity_win)
            scene_assignment.append("identity" if identity_win else "swap")
            for expert in (0, 1):
                candidate = physical[scene_index, prompt_index, expert]
                score = forward_consistency(
                    arrays["blend_physical"][scene_index],
                    np.stack((candidate[:3], candidate[3:])),
                    sky,
                )
                forward_finite.append(bool(score.finite))
                plausible[prompt_index, expert] = is_plausible(score, threshold)
                own_distance = scientific_distance(
                    candidate[:3],
                    arrays["targets_physical"][scene_index, prompt_index, 0, :3],
                    mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                ).primary_normalized
                own[prompt_index, expert] = plausible[prompt_index, expert] and own_distance <= 1.0
                if target_count == 2:
                    alternate_distance = scientific_distance(
                        candidate[:3],
                        arrays["targets_physical"][scene_index, prompt_index, 1, :3],
                        mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                    ).primary_normalized
                    alternate[prompt_index, expert] = plausible[prompt_index, expert] and alternate_distance <= 1.0
            diameters.append(
                scientific_distance(
                    physical[scene_index, prompt_index, 0, :3],
                    physical[scene_index, prompt_index, 1, :3],
                    mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                ).primary_normalized
            )
            for expert in (0, 1):
                source_sum_errors.append(
                    weighted_mse_physical(
                        physical[scene_index, prompt_index, expert, :3]
                        + physical[scene_index, prompt_index, expert, 3:],
                        arrays["blend_physical"][scene_index],
                        SCALES,
                    )
                )
        both_modes = [
            bool(
                (own[prompt_index, 0] and alternate[prompt_index, 1])
                or (own[prompt_index, 1] and alternate[prompt_index, 0])
            )
            for prompt_index in (0, 1)
        ]
        ordinary = row["kind"] == "ordinary"
        if ordinary:
            ordinary_diameters.append(float(np.mean(diameters)))
        per_scene.append(
            {
                "micro_index": row["micro_index"],
                "scene_id": row["scene_id"],
                "kind": row["kind"],
                "pair_id": row["pair_id"],
                "ordinary_both_experts_own_truth": bool(own.all()) if target_count == 1 else False,
                "own_truth_coverage": bool(own.any(axis=1).all()),
                "alternate_truth_coverage": bool(alternate.any(axis=1).all()) if target_count == 2 else False,
                "both_mode_coverage": bool(all(both_modes)) if target_count == 2 else False,
                "set_prompt_identity": bool(identities.all()),
                "both_experts_forward_consistent": bool(plausible.all()),
                "forward_evaluation_finite": bool(all(forward_finite[-4:])),
                "expert_diameter": float(np.mean(diameters)),
                "hard_assignment_prompt_a": scene_assignment[0],
                "hard_assignment_prompt_b": scene_assignment[1],
                "z_band_projected_target_mse": float(np.mean(scene_z)),
            }
        )
    ordinary_rows = [row for row in per_scene if row["kind"] == "ordinary"]
    ambiguous_rows = [row for row in per_scene if row["kind"] == "near_collision"]
    raw_tensor = torch.from_numpy(raw)
    derivative = mapping_derivative(raw_tensor, mapping).numpy()
    if mapping == "relu":
        boundary = float(np.mean(raw <= 0.0))
    else:
        boundary = float(np.mean(np.abs(raw) <= NUMERICAL_ZERO_TOLERANCE))
    metrics: dict[str, object] = {
        "scene_count": len(rows),
        "ordinary_scene_count": len(ordinary_rows),
        "ambiguous_scene_count": len(ambiguous_rows),
        "ordinary_coverage": float(np.mean([bool(row["ordinary_both_experts_own_truth"]) for row in ordinary_rows])) if ordinary_rows else 0.0,
        "own_coverage": float(np.mean([bool(row["own_truth_coverage"]) for row in ambiguous_rows])) if ambiguous_rows else 0.0,
        "alternate_coverage": float(np.mean([bool(row["alternate_truth_coverage"]) for row in ambiguous_rows])) if ambiguous_rows else 0.0,
        "both_mode_coverage": float(np.mean([bool(row["both_mode_coverage"]) for row in ambiguous_rows])) if ambiguous_rows else 0.0,
        "ordinary_expert_diameter": float(np.median(ordinary_diameters)) if ordinary_diameters else 0.0,
        "set_prompt_swap": float(np.mean([bool(row["set_prompt_identity"]) for row in per_scene])),
        "ordinary_forward_consistency": float(np.mean([bool(row["both_experts_forward_consistent"]) for row in ordinary_rows])) if ordinary_rows else 0.0,
        "ambiguous_forward_consistency": float(np.mean([bool(row["both_experts_forward_consistent"]) for row in ambiguous_rows])) if ambiguous_rows else 0.0,
        "forward_evaluation_finite_fraction": float(np.mean(forward_finite)),
        "source_sum_consistency_mse": float(np.mean(source_sum_errors)),
        "projected_target_loss": float(np.mean(projected_losses)),
        "z_band_projected_target_mse": float(np.mean(z_errors)),
        "identity_assignment_fraction": float(np.mean(identity_wins_all)),
        "assignment_margin_mean": float(np.mean(assignment_margins)),
        "zero_gradient_fraction": float(np.mean(derivative == 0.0)),
        "stagnation_fraction": float(np.mean(np.abs(derivative) <= STAGNATION_DERIVATIVE_TOLERANCE)),
        "raw_output_mean_absolute": float(np.mean(np.abs(raw))),
        "raw_output_maximum_absolute": float(np.max(np.abs(raw))),
        "mapping_boundary_or_cusp_fraction": boundary,
        "physical_minimum": float(np.min(physical)),
        "physical_negative_fraction": float(np.mean(physical < PHYSICAL_NEGATIVE_TOLERANCE)),
        "finite_output_fraction": float(np.mean(np.isfinite(physical))),
    }
    model.expert_1.train()
    model.expert_2.train()
    return metrics, per_scene, {"physical": physical, "raw_normalized": raw, "mapped_normalized": mapped}


def write_contract_incident(run: Path, gate: str, mapping: str, step: int, reason: str) -> None:
    path = run / "logs" / f"contract_violation_{gate}_{mapping}_step_{step}.json"
    write_json_fresh(
        path,
        {
            "detected_at_utc": datetime.now(timezone.utc).isoformat(),
            "gate": gate,
            "mapping": mapping,
            "optimizer_step": step,
            "reason": reason,
            "status": "FAILED",
            "checkpoint_promoted": False,
            "subsequent_optimizer_step": False,
        },
    )
    raise ContractViolation(reason)


def save_outputs(
    run: Path,
    gate: str,
    mapping: str,
    outputs: dict[str, np.ndarray],
    metrics: dict[str, object],
) -> tuple[Path, list[dict[str, object]]]:
    directory = run / ("eight_scene" if gate == "eight_scene" else "one_scene")
    path = directory / f"{gate}_{mapping}_outputs.h5"
    with h5py.File(path, "x") as handle:
        handle.create_dataset("physical", data=outputs["physical"], compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("raw_normalized", data=outputs["raw_normalized"], compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.create_dataset("mapped_normalized", data=outputs["mapped_normalized"], compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.attrs["mapping"] = mapping
        handle.attrs["gate"] = gate
        handle.attrs["complete"] = True
        handle.attrs["physical_negative_fraction"] = metrics["physical_negative_fraction"]
    hashes = []
    for scene in range(outputs["physical"].shape[0]):
        for prompt in range(2):
            for expert in range(2):
                hashes.append(
                    {
                        "gate": gate,
                        "mapping": mapping,
                        "scene_position": scene,
                        "prompt": prompt,
                        "expert": expert,
                        "canonical_physical_sha256": canonical_tensor_sha256(outputs["physical"][scene, prompt, expert]),
                    }
                )
    return path, hashes


def run_condition(
    run: Path,
    gate: str,
    mapping: str,
    positions: list[int],
    all_arrays: dict[str, np.ndarray],
    all_projected_physical: np.ndarray,
    all_rows: list[dict[str, object]],
    device: torch.device,
    reference_encoder_hash: str,
    p0_stat: tuple[int, int],
) -> dict[str, object]:
    started = time.time()
    torch.manual_seed(2026071250)
    model = MappedThayerMixtureExperts(mapping, torch.from_numpy(SCALES))
    warm_start_condition_c_encoder(model, CONDITION_C)
    freeze_encoder(model)
    if decoder_parameter_count(model) != (46470, 46470) or parameter_count(model) != 165612:
        raise RuntimeError("L0 parameter count changed")
    topology_hash = model_topology_sha256(model)
    model = model.to(device)
    model.encoder.eval()
    encoder_before = encoder_tensor_sha256(model)
    if encoder_before != reference_encoder_hash:
        raise RuntimeError("condition encoder differs from frozen reference")
    decoder_parameters = list(model.expert_1.parameters()) + list(model.expert_2.parameters())
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    if decoder_ids & encoder_ids or any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("encoder isolation failed before optimizer construction")
    optimizer = torch.optim.AdamW(decoder_parameters, lr=LEARNING_RATE, weight_decay=0.0)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != decoder_ids or optimizer_ids & encoder_ids:
        raise RuntimeError("optimizer contains a non-decoder parameter")

    arrays = subset_arrays(all_arrays, positions)
    projected_physical = all_projected_physical[positions]
    rows = [all_rows[position] for position in positions]
    batch_positions = [0] * EFFECTIVE_BATCH_SIZE if len(positions) == 1 else list(range(EFFECTIVE_BATCH_SIZE))
    batch_arrays = subset_arrays(arrays, batch_positions)
    batch_targets = projected_physical[batch_positions]
    blend = torch.from_numpy(np.ascontiguousarray(batch_arrays["blend"])).to(device)
    prompt_a = torch.from_numpy(np.ascontiguousarray(batch_arrays["prompt_a"])).to(device)
    prompt_b = torch.from_numpy(np.ascontiguousarray(batch_arrays["prompt_b"])).to(device)
    target = torch.from_numpy(np.ascontiguousarray(batch_targets)).to(device)
    scale6 = torch.from_numpy(SCALE6).to(device)
    with torch.no_grad():
        features = model.encode(torch.cat((blend, blend)), torch.cat((prompt_a, prompt_b)))
        features = tuple(value.detach() for value in features)
    if model.encoder.training:
        raise RuntimeError("encoder left eval mode while caching features")

    metrics0, per_scene0, outputs0 = evaluate_condition(model, mapping, arrays, projected_physical, rows, device)
    evaluation_rows = [{"gate": gate, "mapping": mapping, "step": 0, **metrics0}]
    training_rows = []
    per_scene_final = per_scene0
    outputs_final = outputs0
    metrics_final = metrics0
    grad_sum = [0.0, 0.0]
    grad_min = [float("inf"), float("inf")]
    dead_run = [0, 0]
    dead_max = [0, 0]
    negative_events = 0
    nonfinite_events = 0
    for step in range(1, OPTIMIZER_STEPS + 1):
        stat = P0_TARGETS.stat()
        if (stat.st_size, stat.st_mtime_ns) != p0_stat:
            write_contract_incident(run, gate, mapping, step, "P0_TARGET_FILE_STAT_CHANGED")
        if model.encoder.training:
            write_contract_incident(run, gate, mapping, step, "ENCODER_LEFT_EVAL_MODE")
        joined = model.decode_features(features)
        if joined.physical.device.type != "mps":
            write_contract_incident(run, gate, mapping, step, "MPS_FALLBACK")
        if not bool(torch.all(torch.isfinite(joined.physical)).detach().cpu()):
            nonfinite_events += 1
            write_contract_incident(run, gate, mapping, step, "NONFINITE_PHYSICAL_OUTPUT")
        minimum = float(joined.physical.detach().min().cpu())
        if minimum < PHYSICAL_NEGATIVE_TOLERANCE:
            negative_events += 1
            write_contract_incident(run, gate, mapping, step, "NEGATIVE_PHYSICAL_OUTPUT")
        out_a = joined.physical[:EFFECTIVE_BATCH_SIZE]
        out_b = joined.physical[EFFECTIVE_BATCH_SIZE:]
        loss_a, wins_a, margin_a = hard_physical_set_loss(out_a, target[:, 0], scale6)
        loss_b, wins_b, margin_b = hard_physical_set_loss(out_b, target[:, 1], scale6)
        loss = 0.5 * (loss_a + loss_b)
        if not bool(torch.isfinite(loss).detach().cpu()):
            write_contract_incident(run, gate, mapping, step, "NONFINITE_TRAINING_LOSS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms = []
        for expert in (model.expert_1, model.expert_2):
            gradients = [parameter.grad.detach().norm() for parameter in expert.parameters() if parameter.grad is not None]
            value = float(torch.linalg.vector_norm(torch.stack(gradients)).cpu()) if gradients else 0.0
            if not math.isfinite(value):
                write_contract_incident(run, gate, mapping, step, "NONFINITE_EXPERT_GRADIENT")
            grad_norms.append(value)
        torch.nn.utils.clip_grad_norm_(decoder_parameters, 5.0)
        optimizer.step()
        for expert in (0, 1):
            grad_sum[expert] += grad_norms[expert]
            grad_min[expert] = min(grad_min[expert], grad_norms[expert])
            dead_run[expert] = dead_run[expert] + 1 if grad_norms[expert] <= 1e-12 else 0
            dead_max[expert] = max(dead_max[expert], dead_run[expert])
        if step == 1 or step % 10 == 0 or step == OPTIMIZER_STEPS:
            derivative = mapping_derivative(joined.raw_normalized.detach(), mapping)
            training_rows.append(
                {
                    "gate": gate,
                    "mapping": mapping,
                    "step": step,
                    "target_loss": float(loss.detach().cpu()),
                    "expert_1_gradient_norm": grad_norms[0],
                    "expert_2_gradient_norm": grad_norms[1],
                    "identity_assignment_fraction": float(torch.cat((wins_a, wins_b)).float().mean().cpu()),
                    "assignment_margin_mean": float(torch.cat((margin_a.abs(), margin_b.abs())).mean().cpu()),
                    "physical_minimum": minimum,
                    "zero_gradient_fraction": float((derivative == 0).float().mean().cpu()),
                    "stagnation_fraction": float((torch.abs(derivative) <= STAGNATION_DERIVATIVE_TOLERANCE).float().mean().cpu()),
                    "elapsed_seconds": time.time() - started,
                    "device": "mps",
                    "fallback": False,
                }
            )
        if step in EVALUATION_STEPS:
            metrics_final, per_scene_final, outputs_final = evaluate_condition(
                model, mapping, arrays, projected_physical, rows, device
            )
            evaluation_rows.append({"gate": gate, "mapping": mapping, "step": step, **metrics_final})
            print(
                json.dumps(
                    {
                        "gate": gate,
                        "mapping": mapping,
                        "step": step,
                        "loss": metrics_final["projected_target_loss"],
                        "ordinary": metrics_final["ordinary_coverage"],
                        "own": metrics_final["own_coverage"],
                        "alternate": metrics_final["alternate_coverage"],
                        "both": metrics_final["both_mode_coverage"],
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if sha256(P0_TARGETS) != EXPECTED[P0_TARGETS] or sha256(P0_HASHES) != EXPECTED[P0_HASHES]:
        write_contract_incident(run, gate, mapping, OPTIMIZER_STEPS, "P0_TARGET_HASH_MISMATCH")
    encoder_after = encoder_tensor_sha256(model)
    encoder_identical = encoder_before == encoder_after == reference_encoder_hash
    if not encoder_identical:
        write_contract_incident(run, gate, mapping, OPTIMIZER_STEPS, "ENCODER_TENSOR_MUTATION")
    active = dead_max[0] < 20 and dead_max[1] < 20 and grad_sum[0] > 0 and grad_sum[1] > 0
    prompt_pass = (
        metrics_final["set_prompt_swap"] >= 0.90
        if gate == "eight_scene"
        else metrics_final["set_prompt_swap"] == 1.0
    )
    common_pass = bool(
        metrics_final["physical_negative_fraction"] == 0.0
        and metrics_final["finite_output_fraction"] == 1.0
        and metrics_final["forward_evaluation_finite_fraction"] == 1.0
        and prompt_pass
        and active
        and encoder_identical
        and negative_events == 0
        and nonfinite_events == 0
    )
    if gate == "ordinary_one_scene":
        gate_pass = bool(
            common_pass
            and metrics_final["ordinary_coverage"] == 1.0
            and metrics_final["ordinary_expert_diameter"] <= 1.0
        )
    elif gate == "ambiguous_one_scene":
        gate_pass = bool(
            common_pass
            and metrics_final["own_coverage"] == 1.0
            and metrics_final["alternate_coverage"] == 1.0
            and metrics_final["both_mode_coverage"] == 1.0
        )
    else:
        gate_pass = common_pass

    checkpoint_path = run / "checkpoints" / f"{gate}_{mapping}.pth"
    with checkpoint_path.open("xb") as handle:
        torch.save(
            {
                "campaign": "Thayer-OP",
                "gate": gate,
                "mapping": mapping,
                "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "optimizer_steps": OPTIMIZER_STEPS,
                "metrics": metrics_final,
                "encoder_tensor_sha256": encoder_after,
                "p0_target_sha256": EXPECTED[P0_TARGETS],
                "parameter_count": parameter_count(model),
            },
            handle,
        )
    output_path, output_hashes = save_outputs(run, gate, mapping, outputs_final, metrics_final)
    result = {
        "gate": gate,
        "mapping": mapping,
        "pass": gate_pass,
        "common_contract_pass": common_pass,
        "optimizer_steps": OPTIMIZER_STEPS,
        "scene_presentations": OPTIMIZER_STEPS * EFFECTIVE_BATCH_SIZE,
        "unique_scene_count": len(positions),
        "runtime_seconds": time.time() - started,
        "encoder_hash_before": encoder_before,
        "encoder_hash_after": encoder_after,
        "encoder_byte_identical": encoder_identical,
        "topology_sha256": topology_hash,
        "parameters_per_expert": 46470,
        "total_parameters": 165612,
        "expert_1_gradient_norm_mean": grad_sum[0] / OPTIMIZER_STEPS,
        "expert_2_gradient_norm_mean": grad_sum[1] / OPTIMIZER_STEPS,
        "expert_1_gradient_norm_minimum": grad_min[0],
        "expert_2_gradient_norm_minimum": grad_min[1],
        "expert_1_max_consecutive_dead_steps": dead_max[0],
        "expert_2_max_consecutive_dead_steps": dead_max[1],
        "both_experts_active": active,
        "physical_negative_events": negative_events,
        "nonfinite_events": nonfinite_events,
        "checkpoint_path": str(checkpoint_path.relative_to(run)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "output_path": str(output_path.relative_to(run)),
        "output_sha256": sha256(output_path),
        **metrics_final,
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "per_scene_rows": per_scene_final,
        "output_hash_rows": output_hashes,
    }
    del optimizer, model, features, blend, prompt_a, prompt_b, target
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return result


def plot_learning_curves(run: Path, results: list[dict[str, object]]) -> None:
    by_gate = sorted({str(result["gate"]) for result in results})
    fig, axes = plt.subplots(len(by_gate), 1, figsize=(9, 3.5 * len(by_gate)), squeeze=False)
    for axis, gate in zip(axes[:, 0], by_gate):
        for result in results:
            if result["gate"] != gate:
                continue
            rows = result["evaluation_rows"]
            axis.plot([row["step"] for row in rows], [row["projected_target_loss"] for row in rows], label=result["mapping"])
        axis.set_yscale("log")
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("projected-target loss")
        axis.set_title(gate.replace("_", " "))
        axis.legend()
    fig.tight_layout()
    fig.savefig(run / "figures/output_mapping_learning_curves.png", dpi=180)
    plt.close(fig)


def selection_decision(results: list[dict[str, object]], one_pass: dict[str, dict[str, bool]]) -> dict[str, object]:
    eight = {result["mapping"]: result for result in results if result["gate"] == "eight_scene"}
    selectable = []
    for mapping, result in eight.items():
        if (
            one_pass[mapping]["ordinary"]
            and one_pass[mapping]["ambiguous"]
            and bool(result["common_contract_pass"])
            and float(result["set_prompt_swap"]) >= 0.90
        ):
            selectable.append(mapping)
    if not selectable or max(float(eight[mapping]["both_mode_coverage"]) for mapping in selectable) == 0.0:
        passed_any_one = any(value["ordinary"] and value["ambiguous"] for value in one_pass.values())
        return {
            "primary_outcome": "NO MAPPING PASSES",
            "selected_mapping": None,
            "selection_stable_under_tie_breaker": True,
            "capacity_ladder_authorized": False,
            "blocker": "eight-scene aggregation/generalization within the microset" if passed_any_one else "one-scene truth-mode memorization",
            "next_experiment": (
                "Run one fixed four-scene aggregation-isolation fit with two ordinary and two ambiguous frozen rows under the same selected one-scene-eligible mappings."
                if passed_any_one
                else "Run one fixed-feature L0 expert-decoder optimization audit on the frozen ambiguous scene, retaining the same hard assignment and mapping while comparing the neural decoder trajectory with direct cached-feature output optimization."
            ),
            "selectable_mappings": selectable,
        }
    def key(mapping: str) -> tuple[float, float, float, float, float]:
        row = eight[mapping]
        minimum = min(
            float(row["ordinary_coverage"]),
            float(row["own_coverage"]),
            float(row["alternate_coverage"]),
            float(row["both_mode_coverage"]),
        )
        return (
            minimum,
            -float(row["projected_target_loss"]),
            -float(row["ordinary_expert_diameter"]),
            -float(row["stagnation_fraction"]),
            -float(row["z_band_projected_target_mse"]),
        )
    best_key = max(key(mapping) for mapping in selectable)
    tied = [mapping for mapping in selectable if key(mapping) == best_key]
    if len(tied) > 1 and "relu" in tied:
        selected = "relu"
    else:
        selected = tied[0]
    primary = {
        "relu": "RELU SELECTED",
        "square": "SQUARE SELECTED",
        "absolute": "ABSOLUTE SELECTED",
    }[selected]
    if len(tied) > 1:
        primary = "MULTIPLE MAPPINGS EQUIVALENT"
    return {
        "primary_outcome": primary,
        "selected_mapping": selected,
        "selection_stable_under_tie_breaker": True,
        "capacity_ladder_authorized": True,
        "blocker": None,
        "next_experiment": "Run one separate preregistered decoder-capacity ladder using only the frozen selected output mapping.",
        "selectable_mappings": selectable,
        "exact_remaining_tie": tied,
        "selection_key": list(best_key),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    preflight = json.loads((run / "logs/preflight_complete.json").read_text())
    freeze = json.loads((run / "preregistration/freeze_record.json").read_text())
    if preflight["status"] != "PASS" or set(preflight["eligible_mappings"]) != set(MAPPINGS):
        raise RuntimeError("preflight did not authorize all three mappings")
    if sha256(run / "preregistration/fixed_l0_output_parameterization.md") != freeze["preregistration_sha256"]:
        raise RuntimeError("preregistration changed before scene fitting")
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch before scene fitting: {path}")
    device = require_mps()
    started = time.time()
    frozen_rows = read_csv(run / "tables/frozen_row_selection.csv")
    micro_rows = read_csv(MICROSET)
    micro_indices = [int(row["micro_index"]) for row in frozen_rows]
    source_indices = np.asarray([int(row["source_h5_index"]) for row in frozen_rows], dtype=np.int64)
    if micro_indices != list(EIGHT_SCENES):
        raise RuntimeError("frozen eight-scene order changed")
    write_json_fresh(
        run / "logs/per_scene_fitting_started.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_sha256": freeze["preregistration_sha256"],
            "frozen_micro_indices": micro_indices,
            "frozen_source_h5_indices": source_indices.tolist(),
            "unique_scene_input_load_count": len(source_indices),
            "remaining_microset_scene_input_load_count": 0,
            "atlas_scene_access_count": 0,
            "development_access_count": 0,
            "lockbox_access_count": 0,
        },
    )
    arrays = load_micro_arrays(source_indices, SCALES)
    with h5py.File(P0_TARGETS, "r") as handle:
        projected_physical = np.asarray(handle["targets_physical"][micro_indices], dtype=np.float32)
    all_rows = [
        {
            "micro_index": int(row["micro_index"]),
            "scene_id": row["scene_id"],
            "kind": row["kind"],
            "pair_id": row["pair_id"],
            "source_h5_index": int(row["source_h5_index"]),
        }
        for row in frozen_rows
    ]
    if any(micro_rows[index]["scene_id"] != all_rows[position]["scene_id"] for position, index in enumerate(micro_indices)):
        raise RuntimeError("scene metadata disagrees with frozen manifest")
    p0_file_stat = P0_TARGETS.stat()
    p0_stat = (p0_file_stat.st_size, p0_file_stat.st_mtime_ns)
    reference_encoder_hash = str(preflight["reference_encoder_tensor_sha256"])
    results: list[dict[str, object]] = []
    all_training_rows = []
    all_evaluation_rows = []
    all_per_scene_rows = []
    all_output_hash_rows = []
    encoder_rows = []
    summary_rows = []

    for gate, positions in (("ordinary_one_scene", [0]), ("ambiguous_one_scene", [4])):
        for mapping in MAPPINGS:
            result = run_condition(
                run,
                gate,
                mapping,
                positions,
                arrays,
                projected_physical,
                all_rows,
                device,
                reference_encoder_hash,
                p0_stat,
            )
            results.append(result)
            all_training_rows.extend(result.pop("training_rows"))
            all_evaluation_rows.extend(result.pop("evaluation_rows"))
            for row in result.pop("per_scene_rows"):
                all_per_scene_rows.append({"gate": gate, "mapping": mapping, **row})
            all_output_hash_rows.extend(result.pop("output_hash_rows"))
            encoder_rows.append(
                {
                    "gate": gate,
                    "mapping": mapping,
                    "encoder_hash_before": result["encoder_hash_before"],
                    "encoder_hash_after": result["encoder_hash_after"],
                    "reference_encoder_hash": reference_encoder_hash,
                    "byte_identical": result["encoder_byte_identical"],
                }
            )
            summary_rows.append({key: value for key, value in result.items() if key not in {"checkpoint_path", "output_path"}})

    one_pass = {
        mapping: {
            "ordinary": bool(next(result for result in results if result["gate"] == "ordinary_one_scene" and result["mapping"] == mapping)["pass"]),
            "ambiguous": bool(next(result for result in results if result["gate"] == "ambiguous_one_scene" and result["mapping"] == mapping)["pass"]),
        }
        for mapping in MAPPINGS
    }
    ambiguous_passers = [mapping for mapping in MAPPINGS if one_pass[mapping]["ambiguous"]]
    eight_candidates = [mapping for mapping in MAPPINGS if one_pass[mapping]["ordinary"] and one_pass[mapping]["ambiguous"]]
    stopped_after_ambiguous = not ambiguous_passers
    if not stopped_after_ambiguous:
        for mapping in eight_candidates:
            result = run_condition(
                run,
                "eight_scene",
                mapping,
                list(range(8)),
                arrays,
                projected_physical,
                all_rows,
                device,
                reference_encoder_hash,
                p0_stat,
            )
            results.append(result)
            all_training_rows.extend(result.pop("training_rows"))
            all_evaluation_rows.extend(result.pop("evaluation_rows"))
            for row in result.pop("per_scene_rows"):
                all_per_scene_rows.append({"gate": "eight_scene", "mapping": mapping, **row})
            all_output_hash_rows.extend(result.pop("output_hash_rows"))
            encoder_rows.append(
                {
                    "gate": "eight_scene",
                    "mapping": mapping,
                    "encoder_hash_before": result["encoder_hash_before"],
                    "encoder_hash_after": result["encoder_hash_after"],
                    "reference_encoder_hash": reference_encoder_hash,
                    "byte_identical": result["encoder_byte_identical"],
                }
            )
            summary_rows.append({key: value for key, value in result.items() if key not in {"checkpoint_path", "output_path"}})

    if stopped_after_ambiguous:
        selection = {
            "primary_outcome": "NO MAPPING PASSES",
            "selected_mapping": None,
            "selection_stable_under_tie_breaker": True,
            "capacity_ladder_authorized": False,
            "blocker": "one-scene ambiguous truth-mode memorization",
            "next_experiment": "Run one fixed-feature L0 expert-decoder optimization audit on the frozen ambiguous scene, retaining the same hard assignment and mapping while comparing the neural decoder trajectory with direct cached-feature output optimization.",
            "selectable_mappings": [],
        }
    else:
        selection = selection_decision(results, one_pass)
    selection["stopped_after_ambiguous_gate"] = stopped_after_ambiguous
    selection["ordinary_one_scene_passers"] = [mapping for mapping in MAPPINGS if one_pass[mapping]["ordinary"]]
    selection["ambiguous_one_scene_passers"] = ambiguous_passers
    selection["eight_scene_candidates"] = eight_candidates
    selection["remaining_56_microset_scene_rows_loaded"] = 0
    selection["atlas_scene_access_count"] = 0
    selection["development_access_count"] = 0
    selection["lockbox_access_count"] = 0

    write_csv_fresh(run / "one_scene/training_curves.csv", [row for row in all_training_rows if row["gate"] != "eight_scene"])
    write_csv_fresh(run / "one_scene/evaluation_history.csv", [row for row in all_evaluation_rows if row["gate"] != "eight_scene"])
    if any(row["gate"] == "eight_scene" for row in all_training_rows):
        write_csv_fresh(run / "eight_scene/training_curves.csv", [row for row in all_training_rows if row["gate"] == "eight_scene"])
        write_csv_fresh(run / "eight_scene/evaluation_history.csv", [row for row in all_evaluation_rows if row["gate"] == "eight_scene"])
    write_csv_fresh(run / "tables/condition_summary.csv", summary_rows)
    write_csv_fresh(run / "tables/condition_encoder_hashes.csv", encoder_rows)
    write_csv_fresh(run / "tables/final_per_scene_metrics.csv", all_per_scene_rows)
    write_csv_fresh(run / "tables/output_canonical_hashes.csv", all_output_hash_rows)
    comparison = []
    for mapping in MAPPINGS:
        ordinary = next(result for result in results if result["gate"] == "ordinary_one_scene" and result["mapping"] == mapping)
        ambiguous = next(result for result in results if result["gate"] == "ambiguous_one_scene" and result["mapping"] == mapping)
        eight = next((result for result in results if result["gate"] == "eight_scene" and result["mapping"] == mapping), None)
        comparison.append(
            {
                "mapping": mapping,
                "representability_pass": True,
                "stop_self_tests_pass": True,
                "synthetic_fit_pass": True,
                "ordinary_one_scene_pass": ordinary["pass"],
                "ambiguous_one_scene_pass": ambiguous["pass"],
                "eight_scene_run": eight is not None,
                "eight_ordinary_coverage": eight["ordinary_coverage"] if eight else "",
                "eight_own_coverage": eight["own_coverage"] if eight else "",
                "eight_alternate_coverage": eight["alternate_coverage"] if eight else "",
                "eight_both_mode_coverage": eight["both_mode_coverage"] if eight else "",
                "eight_minimum_coverage": min(float(eight[key]) for key in ("ordinary_coverage", "own_coverage", "alternate_coverage", "both_mode_coverage")) if eight else "",
                "projected_target_loss": eight["projected_target_loss"] if eight else "",
                "ordinary_expert_diameter": eight["ordinary_expert_diameter"] if eight else "",
                "stagnation_fraction": eight["stagnation_fraction"] if eight else "",
                "z_band_projected_target_mse": eight["z_band_projected_target_mse"] if eight else "",
                "physical_negative_fraction": (eight or ambiguous)["physical_negative_fraction"],
                "set_prompt_swap": (eight or ambiguous)["set_prompt_swap"],
                "selected": selection["selected_mapping"] == mapping,
            }
        )
    write_csv_fresh(run / "tables/mapping_comparison.csv", comparison)
    write_json_fresh(run / "logs/selection.json", selection)
    plot_learning_curves(run, results)
    complete = {
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "condition_count": len(results),
        "optimizer_step_count": len(results) * OPTIMIZER_STEPS,
        "mps_only": True,
        "fallback": False,
        "one_scene_gate_results": one_pass,
        "selection": selection,
        "unique_scene_input_load_count": 8,
        "remaining_56_microset_scene_input_load_count": 0,
        "p0_target_sha256": sha256(P0_TARGETS),
        "historical_input_hashes_match": True,
        "atlas_scene_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
    }
    write_json_fresh(run / "logs/micro_campaign_complete.json", complete)
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
