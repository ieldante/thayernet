#!/usr/bin/env python3
"""Run the preregistered isolated Thayer-ME representational-capacity gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_probabilistic_unet_pre_atlas import prompts
from src.competing_hypotheses import PlausibilityThresholds, forward_consistency, is_plausible, scientific_distance
from src.models_two_expert_decoder import (
    ThayerMixtureExperts,
    expert_parameter_distance,
    parameter_count,
    permutation_invariant_target_loss,
    prompt_swap_set_loss,
    set_training_phase,
    source_sum,
    unordered_set_distance,
    warm_start_condition_c_encoder,
)


MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
PROMPT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
CONDITION_C = PROMPT / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PROMPT / "manifests/normalization.json"
SCENES = MH / "manifests/probabilistic_unet_training_scenes.h5"
TARGETS = MH / "target_sets/thayer_mh_training_target_sets.h5"
DEFINITIONS = MH / "manifests/probabilistic_unet_scene_definitions.csv"
SEED = 2026071250
EPOCHS = 400
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MEAN_PSF_FWHM_PIXEL = float(np.mean([0.86, 0.81, 0.77]) / 0.2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("MPS fallback is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS probe failed")
    return torch.device("mps")


def thresholds() -> tuple[PlausibilityThresholds, np.ndarray]:
    raw = json.loads((PU / "manifests/forward_consistency_thresholds.json").read_text())
    threshold = PlausibilityThresholds(float(raw["global_chi_square_mean"]), tuple(float(value) for value in raw["per_band_chi_square_mean"]), float(raw["absolute_relative_flux_residual"]), int(raw["calibration_scene_count"]), 0.99, 0.995, 0.99)
    sky = np.asarray(json.loads((ATLAS / "manifests/fixed_noise_contract.json").read_text())["sky_electrons_grz"], dtype=np.float64)
    return threshold, sky


def select_microset() -> tuple[list[dict[str, str]], np.ndarray]:
    rows = [row for row in read_csv(DEFINITIONS) if row["partition"] == "training"]
    ordinary = [index for index, row in enumerate(rows) if row["kind"] == "ordinary"][:32]
    by_pair: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["kind"] == "near_collision":
            by_pair.setdefault(row["near_collision_pair_id"], []).append(index)
    selected_pairs = sorted(by_pair)[:16]
    ambiguous = [index for pair_id in selected_pairs for index in sorted(by_pair[pair_id])]
    if len(ordinary) != 32 or len(ambiguous) != 32 or any(len(by_pair[pair_id]) != 2 for pair_id in selected_pairs):
        raise RuntimeError("microset cardinality mismatch")
    indices = np.asarray(sorted(ordinary + ambiguous), dtype=np.int64)
    return [rows[index] for index in indices], indices


def load_micro_arrays(indices: np.ndarray, scales: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(SCENES, "r") as scene, h5py.File(TARGETS, "r") as target:
        if not bool(scene.attrs["complete"]) or not bool(target.attrs["complete"]):
            raise RuntimeError("reused tensors incomplete")
        blend_physical = np.asarray(scene["blend"][indices.tolist()], dtype=np.float32)
        isolated_physical = np.asarray(scene["isolated"][indices.tolist()], dtype=np.float32)
        xy = np.asarray(scene["xy"][indices.tolist()], dtype=np.float64)
        targets_physical = np.asarray(target["targets"][indices.tolist()], dtype=np.float32)
        counts = np.asarray(target["target_count"][indices.tolist()], dtype=np.int64)
    prompt_a, prompt_b = prompts(xy)
    tile_scales = np.tile(scales, 2)
    return {
        "blend_physical": blend_physical,
        "blend": blend_physical / scales[None, :, None, None],
        "isolated_physical": isolated_physical,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "targets_physical": targets_physical,
        "targets": targets_physical / tile_scales[None, None, None, :, None, None],
        "counts": counts,
    }


def batch_loss(model: ThayerMixtureExperts, arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"][indices])).to(device)
    prompt_a = torch.from_numpy(np.ascontiguousarray(arrays["prompt_a"][indices])).to(device)
    prompt_b = torch.from_numpy(np.ascontiguousarray(arrays["prompt_b"][indices])).to(device)
    targets = torch.from_numpy(np.ascontiguousarray(arrays["targets"][indices])).to(device)
    counts = torch.from_numpy(np.ascontiguousarray(arrays["counts"][indices])).to(device)
    joined = model(torch.cat((blend, blend)), torch.cat((prompt_a, prompt_b)))
    output_a, output_b = joined[: len(blend)], joined[len(blend):]
    set_a = permutation_invariant_target_loss(output_a, targets[:, 0], counts[:, 0])
    set_b = permutation_invariant_target_loss(output_b, targets[:, 1], counts[:, 1])
    forward = 0.5 * ((source_sum(output_a) - blend[:, None]).square().mean() + (source_sum(output_b) - blend[:, None]).square().mean())
    prompt_swap = prompt_swap_set_loss(output_a, output_b)
    pair_consistency_terms = []
    for left in range(4, len(indices), 2):
        pair_consistency_terms.append(unordered_set_distance(output_a[left:left + 1], output_a[left + 1:left + 2]).mean())
        pair_consistency_terms.append(unordered_set_distance(output_b[left:left + 1], output_b[left + 1:left + 2]).mean())
    pair_consistency = torch.stack(pair_consistency_terms).mean()
    target_loss = 0.5 * (set_a["loss"] + set_b["loss"])
    total = target_loss + 0.5 * forward + 0.25 * prompt_swap + 0.05 * pair_consistency
    ambiguous = counts[:, 0] == 2
    identity_wins = torch.cat((set_a["identity_wins"][ambiguous], set_b["identity_wins"][ambiguous]))
    return total, {
        "loss": float(total.detach().cpu()),
        "target_set": float(target_loss.detach().cpu()),
        "forward": float(forward.detach().cpu()),
        "prompt_swap": float(prompt_swap.detach().cpu()),
        "pair_consistency": float(pair_consistency.detach().cpu()),
        "expert_1_target_a_assignment_fraction": float(identity_wins.float().mean().detach().cpu()),
    }


def training_batches(rows: list[dict[str, str]], rng: np.random.Generator) -> list[np.ndarray]:
    ordinary = np.asarray([index for index, row in enumerate(rows) if row["kind"] == "ordinary"], dtype=np.int64)
    by_pair: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["kind"] == "near_collision":
            by_pair.setdefault(row["near_collision_pair_id"], []).append(index)
    ordinary = rng.permutation(ordinary)
    pairs = [by_pair[pair_id] for pair_id in sorted(by_pair)]
    pairs = [pairs[index] for index in rng.permutation(len(pairs))]
    return [np.asarray(list(ordinary[4 * batch:4 * batch + 4]) + pairs[2 * batch] + pairs[2 * batch + 1], dtype=np.int64) for batch in range(8)]


def prompt_identity(output: np.ndarray, targets: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros(2, dtype=bool)
    for expert in (0, 1):
        requested = output[expert, :3]
        requested_cost = min(float(np.mean((requested - targets[target_index, :3]) ** 2)) for target_index in range(count))
        companion_cost = min(float(np.mean((requested - targets[target_index, 3:]) ** 2)) for target_index in range(count))
        result[expert] = requested_cost < companion_cost
    return result


def evaluate(model: ThayerMixtureExperts, arrays: dict[str, np.ndarray], rows: list[dict[str, str]], scales: np.ndarray, device: torch.device) -> tuple[dict[str, float], np.ndarray, list[dict[str, object]]]:
    model.eval()
    outputs = np.empty((len(rows), 2, 2, 6, 60, 60), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(rows))
            blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"][start:stop])).to(device)
            prompt_a = torch.from_numpy(np.ascontiguousarray(arrays["prompt_a"][start:stop])).to(device)
            prompt_b = torch.from_numpy(np.ascontiguousarray(arrays["prompt_b"][start:stop])).to(device)
            joined = model(torch.cat((blend, blend)), torch.cat((prompt_a, prompt_b))).cpu().numpy()
            outputs[start:stop] = np.stack((joined[: len(blend)], joined[len(blend):]), axis=1)
    outputs_physical = outputs * np.tile(scales, 2)[None, None, None, :, None, None]
    threshold, sky = thresholds()
    per_scene = []
    expert_prompt = [[], []]
    set_prompt = []
    ordinary_diameter = []
    for index, row in enumerate(rows):
        count = int(arrays["counts"][index, 0])
        plausible = np.zeros((2, 2), dtype=bool)
        own = np.zeros((2, 2), dtype=bool)
        alternate = np.zeros((2, 2), dtype=bool)
        identities = np.zeros((2, 2), dtype=bool)
        diameters = []
        for prompt_index in (0, 1):
            identities[prompt_index] = prompt_identity(outputs[index, prompt_index], arrays["targets"][index, prompt_index], count)
            for expert in (0, 1):
                candidate = outputs_physical[index, prompt_index, expert]
                score = forward_consistency(arrays["blend_physical"][index], np.stack((candidate[:3], candidate[3:])), sky)
                plausible[prompt_index, expert] = is_plausible(score, threshold)
                own_distance = scientific_distance(candidate[:3], arrays["targets_physical"][index, prompt_index, 0, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized
                own[prompt_index, expert] = plausible[prompt_index, expert] and own_distance <= 1.0
                if count == 2:
                    alternate_distance = scientific_distance(candidate[:3], arrays["targets_physical"][index, prompt_index, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized
                    alternate[prompt_index, expert] = plausible[prompt_index, expert] and alternate_distance <= 1.0
            diameters.append(scientific_distance(outputs_physical[index, prompt_index, 0, :3], outputs_physical[index, prompt_index, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized)
        for expert in (0, 1):
            expert_prompt[expert].append(bool(identities[:, expert].all()))
        set_prompt.append(bool(identities.all()))
        both_modes = [bool((own[prompt_index, 0] and alternate[prompt_index, 1]) or (own[prompt_index, 1] and alternate[prompt_index, 0])) for prompt_index in (0, 1)]
        if row["kind"] == "ordinary":
            ordinary_diameter.append(float(np.mean(diameters)))
        per_scene.append({
            "scene_id": row["scene_id"],
            "kind": row["kind"],
            "near_collision_pair_id": row["near_collision_pair_id"],
            "both_experts_forward_consistent": bool(plausible.all()),
            "ordinary_both_experts_own_truth": bool(own.all()) if count == 1 else False,
            "own_truth_coverage": bool(own.any(axis=1).all()),
            "alternate_truth_coverage": bool(alternate.any(axis=1).all()) if count == 2 else False,
            "both_mode_coverage": bool(all(both_modes)) if count == 2 else False,
            "expert_1_prompt_identity": bool(identities[:, 0].all()),
            "expert_2_prompt_identity": bool(identities[:, 1].all()),
            "set_prompt_identity": bool(identities.all()),
            "expert_diameter": float(np.mean(diameters)),
        })
    ordinary = [row for row in per_scene if row["kind"] == "ordinary"]
    ambiguous = [row for row in per_scene if row["kind"] == "near_collision"]
    metrics = {
        "ordinary_own_truth_coverage": float(np.mean([bool(row["ordinary_both_experts_own_truth"]) for row in ordinary])),
        "ordinary_median_expert_diameter": float(np.median(ordinary_diameter)),
        "ambiguous_own_truth_coverage": float(np.mean([bool(row["own_truth_coverage"]) for row in ambiguous])),
        "ambiguous_alternate_truth_coverage": float(np.mean([bool(row["alternate_truth_coverage"]) for row in ambiguous])),
        "ambiguous_both_mode_coverage": float(np.mean([bool(row["both_mode_coverage"]) for row in ambiguous])),
        "expert_1_prompt_swap": float(np.mean(expert_prompt[0])),
        "expert_2_prompt_swap": float(np.mean(expert_prompt[1])),
        "set_prompt_swap": float(np.mean(set_prompt)),
        "ordinary_forward_consistency": float(np.mean([bool(row["both_experts_forward_consistent"]) for row in ordinary])),
        "ambiguous_forward_consistency": float(np.mean([bool(row["both_experts_forward_consistent"]) for row in ambiguous])),
    }
    return metrics, outputs_physical, per_scene


def gates_pass(metrics: dict[str, float]) -> bool:
    return (
        metrics["ordinary_own_truth_coverage"] >= 0.90
        and metrics["ordinary_median_expert_diameter"] <= 1.0
        and metrics["ambiguous_own_truth_coverage"] >= 0.90
        and metrics["ambiguous_alternate_truth_coverage"] >= 0.90
        and metrics["ambiguous_both_mode_coverage"] >= 0.90
        and metrics["expert_1_prompt_swap"] >= 0.90
        and metrics["expert_2_prompt_swap"] >= 0.90
        and metrics["set_prompt_swap"] >= 0.90
        and metrics["ordinary_forward_consistency"] >= 0.90
        and metrics["ambiguous_forward_consistency"] >= 0.90
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    architecture = json.loads((run_dir / "logs/architecture_audit_complete.json").read_text())
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    if architecture["status"] != "FROZEN_ARCHITECTURE_AUDIT_PASS" or sha256_file(run_dir / "preregistration/two_expert_ambiguity_decoder.md") != freeze["preregistration_sha256"]:
        raise RuntimeError("architecture/preregistration gate failed")
    if any((run_dir / "checkpoints").iterdir()) or any((run_dir / "atlas_evaluation").iterdir()):
        raise RuntimeError("master checkpoint or Atlas collision before micro fit")
    if sha256_file(TARGETS) != json.loads((run_dir / "target_sets/reused_target_set_references.json").read_text())["files"][0]["sha256"]:
        raise RuntimeError("training target set changed")
    device = require_mps()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    rows, source_indices = select_microset()
    arrays = load_micro_arrays(source_indices, scales)
    if any(row["partition"] != "training" for row in rows):
        raise RuntimeError("non-training row entered microset")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    micro_dir = run_dir / f"diagnostics/micro_overfit_{stamp}"
    micro_dir.mkdir(parents=False, exist_ok=False)
    for name in ("checkpoints", "logs", "tables", "expert_outputs"):
        (micro_dir / name).mkdir(exist_ok=False)
    manifest_rows = [{"micro_index": index, "source_h5_index": int(source_indices[index]), "scene_id": row["scene_id"], "kind": row["kind"], "pair_id": row["near_collision_pair_id"], "partition": row["partition"], "validation_access": 0, "calibration_access": 0, "atlas_access": 0, "development_access": 0, "lockbox_access": 0} for index, row in enumerate(rows)]
    write_csv_fresh(micro_dir / "tables/microset_manifest.csv", manifest_rows)
    write_json_fresh(micro_dir / "logs/microset_frozen_before_fit.json", {"status": "FROZEN", "ordinary_count": 32, "ambiguous_observation_count": 32, "ambiguous_pair_count": 16, "manifest_sha256": sha256_file(micro_dir / "tables/microset_manifest.csv"), "scene_tensor_sha256": sha256_file(SCENES), "target_tensor_sha256": sha256_file(TARGETS), "seed": SEED, "max_epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "device": "mps", "fallback": False, "validation_access_count": 0, "calibration_access_count": 0, "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0})
    model = ThayerMixtureExperts()
    warm_start_condition_c_encoder(model, CONDITION_C)
    set_training_phase(model, 2)
    model = model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    epoch_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    final_metrics: dict[str, float] | None = None
    final_outputs: np.ndarray | None = None
    final_per_scene: list[dict[str, object]] | None = None
    passed_epoch = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_rows = []
        grad_1 = []
        grad_2 = []
        for batch_indices in training_batches(rows, rng):
            optimizer.zero_grad(set_to_none=True)
            loss, values = batch_loss(model, arrays, batch_indices, device)
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise RuntimeError("NaN/Inf micro loss")
            loss.backward()
            grad_1.append(float(torch.linalg.vector_norm(torch.stack([parameter.grad.detach().norm() for parameter in model.expert_1.parameters() if parameter.grad is not None])).cpu()))
            grad_2.append(float(torch.linalg.vector_norm(torch.stack([parameter.grad.detach().norm() for parameter in model.expert_2.parameters() if parameter.grad is not None])).cpu()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_rows.append(values)
        mean = {key: float(np.mean([row[key] for row in batch_rows])) for key in batch_rows[0]}
        epoch_row: dict[str, object] = {"epoch": epoch, **mean, "expert_1_gradient_norm": float(np.mean(grad_1)), "expert_2_gradient_norm": float(np.mean(grad_2)), "expert_parameter_distance": float(expert_parameter_distance(model)), "elapsed_seconds": time.time() - started, "device": "mps", "fallback": False}
        epoch_rows.append(epoch_row)
        if epoch == 1 or epoch % 20 == 0:
            metrics, outputs, per_scene = evaluate(model, arrays, rows, scales, device)
            evaluation_rows.append({"epoch": epoch, **metrics})
            print(json.dumps({"phase": "micro_overfit", "epoch": epoch, "loss": mean["loss"], **metrics, "elapsed_seconds": time.time() - started}, sort_keys=True), flush=True)
            if gates_pass(metrics):
                passed_epoch = epoch
                final_metrics, final_outputs, final_per_scene = metrics, outputs, per_scene
                break
    if final_metrics is None:
        final_metrics, final_outputs, final_per_scene = evaluate(model, arrays, rows, scales, device)
    status = "PASS" if gates_pass(final_metrics) else "REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE"
    payload = {"model": "Thayer-ME micro-overfit", "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, "epoch": passed_epoch or EPOCHS, "metrics": final_metrics, "seed": SEED, "parameter_count": parameter_count(model), "preregistration_sha256": freeze["preregistration_sha256"], "microset_manifest_sha256": sha256_file(micro_dir / "tables/microset_manifest.csv")}
    with (micro_dir / "checkpoints/thayer_me_micro_final.pth").open("xb") as handle:
        torch.save(payload, handle)
    write_csv_fresh(micro_dir / "tables/micro_epochs.csv", epoch_rows)
    write_csv_fresh(micro_dir / "tables/micro_gate_history.csv", evaluation_rows)
    write_csv_fresh(micro_dir / "tables/micro_per_scene.csv", final_per_scene)
    with h5py.File(micro_dir / "expert_outputs/micro_final_decompositions.h5", "x") as handle:
        handle.create_dataset("decompositions", data=final_outputs, compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.attrs["complete"] = True
    gate_rows = [
        {"gate": "ordinary_own_truth_coverage", "threshold": ">=0.90", "observed": final_metrics["ordinary_own_truth_coverage"], "pass": final_metrics["ordinary_own_truth_coverage"] >= 0.90},
        {"gate": "ordinary_median_expert_diameter", "threshold": "<=1.0", "observed": final_metrics["ordinary_median_expert_diameter"], "pass": final_metrics["ordinary_median_expert_diameter"] <= 1.0},
        {"gate": "ambiguous_own_truth_coverage", "threshold": ">=0.90", "observed": final_metrics["ambiguous_own_truth_coverage"], "pass": final_metrics["ambiguous_own_truth_coverage"] >= 0.90},
        {"gate": "ambiguous_alternate_truth_coverage", "threshold": ">=0.90", "observed": final_metrics["ambiguous_alternate_truth_coverage"], "pass": final_metrics["ambiguous_alternate_truth_coverage"] >= 0.90},
        {"gate": "ambiguous_both_mode_coverage", "threshold": ">=0.90", "observed": final_metrics["ambiguous_both_mode_coverage"], "pass": final_metrics["ambiguous_both_mode_coverage"] >= 0.90},
        {"gate": "expert_1_prompt_swap", "threshold": ">=0.90", "observed": final_metrics["expert_1_prompt_swap"], "pass": final_metrics["expert_1_prompt_swap"] >= 0.90},
        {"gate": "expert_2_prompt_swap", "threshold": ">=0.90", "observed": final_metrics["expert_2_prompt_swap"], "pass": final_metrics["expert_2_prompt_swap"] >= 0.90},
        {"gate": "set_prompt_swap", "threshold": ">=0.90", "observed": final_metrics["set_prompt_swap"], "pass": final_metrics["set_prompt_swap"] >= 0.90},
        {"gate": "ordinary_forward_consistency", "threshold": ">=0.90", "observed": final_metrics["ordinary_forward_consistency"], "pass": final_metrics["ordinary_forward_consistency"] >= 0.90},
        {"gate": "ambiguous_forward_consistency", "threshold": ">=0.90", "observed": final_metrics["ambiguous_forward_consistency"], "pass": final_metrics["ambiguous_forward_consistency"] >= 0.90},
    ]
    write_csv_fresh(micro_dir / "tables/micro_overfit_gates.csv", gate_rows)
    write_text_fresh(micro_dir / "micro_overfit_report.md", f"""# Thayer-ME micro-overfit capacity gate

Status: **{status}**.

- Epoch evaluated: {passed_epoch or EPOCHS} of {EPOCHS} maximum.
- Ordinary both-expert own-truth coverage: {final_metrics['ordinary_own_truth_coverage']:.6f}; median expert diameter {final_metrics['ordinary_median_expert_diameter']:.6f}.
- Ambiguous own / alternate / both-mode coverage: {final_metrics['ambiguous_own_truth_coverage']:.6f} / {final_metrics['ambiguous_alternate_truth_coverage']:.6f} / {final_metrics['ambiguous_both_mode_coverage']:.6f}.
- Expert 1 / expert 2 / set prompt-swap identity: {final_metrics['expert_1_prompt_swap']:.6f} / {final_metrics['expert_2_prompt_swap']:.6f} / {final_metrics['set_prompt_swap']:.6f}.
- Ordinary / ambiguous all-expert forward consistency: {final_metrics['ordinary_forward_consistency']:.6f} / {final_metrics['ambiguous_forward_consistency']:.6f}.
- MPS-only execution; fallback false.
- Validation / calibration / Atlas / development / lockbox access: 0 / 0 / 0 / 0 / 0.
""")
    write_json_fresh(micro_dir / "logs/micro_overfit_complete.json", {"status": status, "passed": status == "PASS", "full_training_authorized": status == "PASS", "epoch": passed_epoch or EPOCHS, "runtime_seconds": time.time() - started, "metrics": final_metrics, "checkpoint_sha256": sha256_file(micro_dir / "checkpoints/thayer_me_micro_final.pth"), "expert_outputs_sha256": sha256_file(micro_dir / "expert_outputs/micro_final_decompositions.h5"), "mps_only": True, "fallback": False, "validation_access_count": 0, "calibration_access_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    write_json_fresh(run_dir / "logs/micro_overfit_complete.json", {"status": status, "passed": status == "PASS", "full_training_authorized": status == "PASS", "micro_run": str(micro_dir.relative_to(run_dir)), "micro_report_sha256": sha256_file(micro_dir / "micro_overfit_report.md"), "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    print(json.dumps({"status": status, "epoch": passed_epoch or EPOCHS, **final_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
