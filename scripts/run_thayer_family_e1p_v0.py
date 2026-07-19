#!/usr/bin/env python3
"""Run the frozen Family-E1P paired-prompt micro-overfit and influence trace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy  # noqa: E402
from src.family_e1 import (  # noqa: E402
    BAND_SCALES,
    EXPECTED_PARAMETERS,
    FamilyE1Output,
    FamilyE1UNet,
    conservation_error,
    source_objective,
    trainable_parameter_count,
)
from src.family_e1p import (  # noqa: E402
    activation_pair_metrics,
    build_paired_prompt_examples,
    paired_prediction_metrics,
)


ORIGINAL_RUN = REPO / "outputs/runs/thayer_family_e1_v0_20260714_214715"
HIERARCHICAL = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
TRAIN_SELECTOR = ORIGINAL_RUN / "manifests/training_manifest.csv"
TRAIN_MANIFEST = HIERARCHICAL / "manifests/v2_r_training_scene_manifest.csv"
TRAIN_H5 = HIERARCHICAL / "manifests/v2_r_training_scenes.h5"
SCALES = np.asarray(BAND_SCALES, dtype=np.float32)
MICRO_SPECS = {
    "difficult_one_scene": {"indices": [6], "steps": 2000, "seed": 2026071512},
    "mixed_eight_scene": {
        "indices": [0, 3, 5, 6, 18, 51, 73, 81],
        "steps": 3000,
        "seed": 2026071513,
    },
}
LAYER_MODULES = (
    ("enc0_first", "encoder", "enc0_first"),
    ("enc0_second", "encoder", "enc0_second"),
    ("down0", "encoder", "down0"),
    ("enc1", "encoder", "enc1"),
    ("down1", "encoder", "down1"),
    ("enc2", "encoder", "enc2"),
    ("down2", "encoder", "down2"),
    ("enc3", "encoder", "enc3"),
    ("dec2_up", "decoder", "dec2.up_convolution"),
    ("dec2_first", "decoder", "dec2.first"),
    ("dec2_second", "decoder", "dec2.second"),
    ("dec1_up", "decoder", "dec1.up_convolution"),
    ("dec1_first", "decoder", "dec1.first"),
    ("dec1_second", "decoder", "dec1.second"),
    ("dec0_up", "decoder", "dec0.up_convolution"),
    ("dec0_first", "decoder", "dec0.first"),
    ("dec0_second", "decoder", "dec0.second"),
    ("source_head_raw", "head", "source_head"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_torch_fresh(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(payload, handle)


def command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)


def validate_run(run: Path) -> dict[str, object]:
    if run.parent.resolve() != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_family_e1p_v0_"):
        raise RuntimeError("unexpected Family-E1P run path")
    freeze = json.loads((run / "logs/preregistration_complete.json").read_text())
    preregistration = REPO / str(freeze["path"])
    if freeze["status"] != "FROZEN_BEFORE_MODEL_CONSTRUCTION_OR_FITTING":
        raise RuntimeError("preregistration was not frozen")
    if sha256_file(preregistration) != freeze["sha256"]:
        raise RuntimeError("preregistration hash mismatch")
    staged = command(["git", "diff", "--cached", "--name-status"])
    if staged.returncode or staged.stdout.strip():
        raise RuntimeError("staged index must remain empty")
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    for item in provenance["authoritative_inputs"]:
        path = REPO / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"authoritative input changed: {path}")
    return provenance


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU fallback is prohibited")
    probe = torch.ones(2, dtype=torch.float32, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


def load_paired_micro(indices: list[int]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    selector = pd.read_csv(TRAIN_SELECTOR)
    upstream_manifest = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    local_indices = np.asarray(indices, dtype=np.int64)
    upstream_indices = selector.iloc[local_indices].upstream_index.to_numpy(dtype=np.int64)
    rows = upstream_manifest.iloc[upstream_indices].reset_index(drop=True)
    with h5py.File(TRAIN_H5, "r") as handle:
        blend = np.asarray(handle["blend"][upstream_indices], dtype=np.float32)
        isolated = np.asarray(handle["isolated"][upstream_indices], dtype=np.float32)
        prompt_a = np.asarray(handle["prompt"][upstream_indices], dtype=np.float32)
        xy = np.asarray(handle["xy"][upstream_indices], dtype=np.float64)
        prompt_xy = np.asarray(handle["prompt_xy"][upstream_indices], dtype=np.float64)
    matched = rows.matched_source_index.to_numpy(dtype=np.int64)
    local = np.arange(len(rows))
    source_a = np.ascontiguousarray(isolated[local, matched])
    source_b = np.ascontiguousarray(isolated[local, 1 - matched])
    prompt_b_values = []
    prompt_b_xy = []
    for index in range(len(rows)):
        offset = prompt_xy[index] - xy[index, matched[index]]
        alternate_xy = xy[index, 1 - matched[index]] + offset
        prompt_b_xy.append(alternate_xy)
        prompt_b_values.append(gaussian_prompt_numpy(float(alternate_xy[0]), float(alternate_xy[1])))
    prompt_b = np.ascontiguousarray(np.asarray(prompt_b_values, dtype=np.float32)[:, None])
    paired = build_paired_prompt_examples(
        blend,
        prompt_a,
        prompt_b,
        source_a,
        source_b,
        band_scales=BAND_SCALES,
    )
    scene_count = len(rows)
    if not np.array_equal(paired["observed"][:scene_count], paired["observed"][scene_count:]):
        raise RuntimeError("paired observations are not byte-identical")
    if not np.array_equal(paired["model_input"][:scene_count, :3], paired["model_input"][scene_count:, :3]):
        raise RuntimeError("non-prompt model inputs differ across paired views")
    if np.array_equal(paired["model_input"][:scene_count, 3:], paired["model_input"][scene_count:, 3:]):
        raise RuntimeError("paired prompt channels are identical")
    if not np.array_equal(paired["requested"][:scene_count], paired["companion"][scene_count:]):
        raise RuntimeError("A requested target did not become B companion target")
    if not np.array_equal(paired["companion"][:scene_count], paired["requested"][scene_count:]):
        raise RuntimeError("A companion target did not become B requested target")

    manifest_rows = []
    for position, (_, row) in enumerate(rows.iterrows()):
        manifest_rows.append(
            {
                "family_e1_index": indices[position],
                "upstream_index": int(upstream_indices[position]),
                "matched_source_index": int(matched[position]),
                "source_a_id": str(row.get("source_a_id", "")),
                "source_b_id": str(row.get("source_b_id", "")),
                "source_a_group": str(row.get("source_a_group", "")),
                "source_b_group": str(row.get("source_b_group", "")),
                "query_state": str(row.get("query_state", "")),
                "observation_sha256": sha256_array(blend[position]),
                "prompt_a_sha256": sha256_array(prompt_a[position]),
                "prompt_b_sha256": sha256_array(prompt_b[position]),
                "prompt_a_x": float(prompt_xy[position, 0]),
                "prompt_a_y": float(prompt_xy[position, 1]),
                "prompt_b_x": float(prompt_b_xy[position][0]),
                "prompt_b_y": float(prompt_b_xy[position][1]),
                "prompt_l1_difference": float(np.mean(np.abs(prompt_a[position] - prompt_b[position]))),
            }
        )
    return paired, manifest_rows


def output_diagnostics(output: FamilyE1Output, observed: torch.Tensor) -> dict[str, float]:
    sources = torch.cat((output.requested.flatten(), output.companion.flatten()))
    residual = output.residual_noise
    magnitude = max(
        1.0,
        float(observed.detach().abs().max().cpu()),
        float(output.requested.detach().abs().max().cpu()),
        float(output.companion.detach().abs().max().cpu()),
        float(residual.detach().abs().max().cpu()),
    )
    return {
        "source_minimum": float(sources.detach().min().cpu()),
        "source_negative_fraction": float((sources < 0).float().mean().detach().cpu()),
        "source_nonfinite_fraction": float((~torch.isfinite(sources)).float().mean().detach().cpu()),
        "residual_negative_fraction": float((residual < 0).float().mean().detach().cpu()),
        "residual_positive_fraction": float((residual > 0).float().mean().detach().cpu()),
        "conservation_error": float(conservation_error(output, observed).detach().cpu()),
        "conservation_tolerance": 1.0e-5 * magnitude,
    }


def snapshot_metrics(
    model: FamilyE1UNet,
    model_input: torch.Tensor,
    observed: torch.Tensor,
    requested: torch.Tensor,
    companion: torch.Tensor,
    scene_count: int,
) -> tuple[dict[str, float], FamilyE1Output]:
    model.eval()
    with torch.no_grad():
        output = model(model_input, observed)
        losses = source_objective(output.requested, output.companion, requested, companion)
        pair = paired_prediction_metrics(
            output.requested,
            requested,
            companion,
            scene_count=scene_count,
            band_scales=BAND_SCALES,
        )
        physical = output_diagnostics(output, observed)
    values = {key: float(value.detach().cpu()) for key, value in losses.items()}
    values.update(pair)
    values.update(physical)
    return values, output


def layerwise_prompt_trace(
    model: FamilyE1UNet,
    model_input: torch.Tensor,
    observed: torch.Tensor,
    *,
    scene_count: int,
    condition: str,
    phase: str,
) -> list[dict[str, object]]:
    model.eval()
    value = model_input.detach().clone().requires_grad_(True)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for layer_name, _, module_path in LAYER_MODULES:
        module = model.get_submodule(module_path)

        def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor, *, name: str = layer_name) -> None:
            captured[name] = output

        handles.append(module.register_forward_hook(capture))
    output = model(value, observed)
    for handle in handles:
        handle.remove()
    scales = model.band_scales.to(dtype=output.requested.dtype)
    ordered: list[tuple[str, str, torch.Tensor]] = [("prompt_input", "input", value[:, 3:4])]
    ordered.extend((name, group, captured[name]) for name, group, _ in LAYER_MODULES)
    ordered.extend(
        (
            ("requested_output", "output", output.requested / scales),
            ("companion_output", "output", output.companion / scales),
        )
    )
    rows = []
    for order, (name, group, activation) in enumerate(ordered):
        metrics = activation_pair_metrics(activation, scene_count=scene_count)
        gradient = torch.autograd.grad(
            activation.square().mean(),
            value,
            retain_graph=order < len(ordered) - 1,
            allow_unused=False,
        )[0][:, 3:4]
        rows.append(
            {
                "condition": condition,
                "phase": phase,
                "layer_order": order,
                "layer": name,
                "group": group,
                **metrics,
                "gradient_wrt_prompt_input_rms": float(torch.sqrt(torch.mean(gradient.square())).detach().cpu()),
                "gradient_wrt_prompt_input_l2": float(torch.linalg.vector_norm(gradient).detach().cpu()),
            }
        )
    del output, captured, value
    torch.mps.empty_cache()
    return rows


def prompt_gradient_metrics(
    model: FamilyE1UNet,
    model_input: torch.Tensor,
    observed: torch.Tensor,
    requested: torch.Tensor,
    companion: torch.Tensor,
    condition: str,
) -> dict[str, object]:
    model.eval()
    value = model_input.detach().clone().requires_grad_(True)
    output = model(value, observed)
    scales = model.band_scales.to(dtype=output.requested.dtype)
    losses = source_objective(output.requested, output.companion, requested, companion)
    scores = (
        ("objective", losses["total"]),
        ("requested_output_energy", (output.requested / scales).square().mean()),
        ("companion_output_energy", (output.companion / scales).square().mean()),
    )
    result: dict[str, object] = {"condition": condition}
    for index, (name, score) in enumerate(scores):
        gradient = torch.autograd.grad(score, value, retain_graph=index < len(scores) - 1)[0][:, 3:4]
        result[f"{name}_prompt_gradient_rms"] = float(torch.sqrt(torch.mean(gradient.square())).detach().cpu())
        result[f"{name}_prompt_gradient_l2"] = float(torch.linalg.vector_norm(gradient).detach().cpu())
    return result


def per_scene_metrics(
    output: FamilyE1Output,
    requested: torch.Tensor,
    companion: torch.Tensor,
    indices: list[int],
) -> list[dict[str, object]]:
    count = len(indices)
    rows = []
    for local, family_index in enumerate(indices):
        pair_indices = torch.tensor([local, count + local], device=output.requested.device)
        values = paired_prediction_metrics(
            output.requested.index_select(0, pair_indices),
            requested.index_select(0, pair_indices),
            companion.index_select(0, pair_indices),
            scene_count=1,
            band_scales=BAND_SCALES,
        )
        rows.append({"family_e1_index": family_index, **values})
    return rows


def cpu_state(model: FamilyE1UNet) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fit_condition(
    run: Path,
    condition: str,
    spec: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    indices = list(spec["indices"])
    steps = int(spec["steps"])
    seed = int(spec["seed"])
    seed_everything(seed)
    paired, manifest_rows = load_paired_micro(indices)
    fresh_csv(run / f"manifests/{condition}_paired_scene_manifest.csv", manifest_rows)
    model_input = torch.from_numpy(paired["model_input"]).to(device)
    observed = torch.from_numpy(paired["observed"]).to(device)
    requested = torch.from_numpy(paired["requested"]).to(device)
    companion = torch.from_numpy(paired["companion"]).to(device)
    scene_count = len(indices)

    model = FamilyE1UNet().to(device)
    if trainable_parameter_count(model) != EXPECTED_PARAMETERS or next(model.parameters()).device.type != "mps":
        raise RuntimeError("frozen architecture count/device mismatch")
    state_keys = tuple(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    if optimizer.param_groups[0]["lr"] != 3.0e-3 or optimizer.param_groups[0]["weight_decay"] != 1.0e-4:
        raise RuntimeError("optimizer contract mismatch")
    head_start = model.source_head.weight.detach().clone()
    decoder_start = model.dec2.up_convolution[0].weight.detach().clone()
    initial, _ = snapshot_metrics(model, model_input, observed, requested, companion, scene_count)
    layer_rows = layerwise_prompt_trace(
        model,
        model_input,
        observed,
        scene_count=scene_count,
        condition=condition,
        phase="initial",
    )
    trace_rows: list[dict[str, object]] = [{"condition": condition, "step": 0, **initial, "gradient_norm": 0.0}]
    checkpoints = {steps // 4, steps // 2, 3 * steps // 4, steps}
    maximum_conservation = initial["conservation_error"]
    maximum_tolerance = initial["conservation_tolerance"]
    hidden_gradient_seen = False
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(model_input, observed)
        losses = source_objective(output.requested, output.companion, requested, companion)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"nonfinite objective: {condition} update {step}")
        losses["total"].backward()
        hidden_gradient = model.dec2.up_convolution[0].weight.grad
        hidden_gradient_seen = hidden_gradient_seen or (
            hidden_gradient is not None and float(hidden_gradient.detach().abs().max().cpu()) > 0.0
        )
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
        optimizer.step()
        closure = float(conservation_error(output, observed).detach().cpu())
        magnitude = max(
            1.0,
            float(observed.detach().abs().max().cpu()),
            float(output.requested.detach().abs().max().cpu()),
            float(output.companion.detach().abs().max().cpu()),
            float(output.residual_noise.detach().abs().max().cpu()),
        )
        maximum_conservation = max(maximum_conservation, closure)
        maximum_tolerance = max(maximum_tolerance, 1.0e-5 * magnitude)
        if step in checkpoints:
            snapshot, _ = snapshot_metrics(model, model_input, observed, requested, companion, scene_count)
            trace_rows.append({"condition": condition, "step": step, **snapshot, "gradient_norm": gradient_norm})
            model.train()

    final, final_output = snapshot_metrics(model, model_input, observed, requested, companion, scene_count)
    layer_rows.extend(
        layerwise_prompt_trace(
            model,
            model_input,
            observed,
            scene_count=scene_count,
            condition=condition,
            phase="final",
        )
    )
    gradient_row = prompt_gradient_metrics(model, model_input, observed, requested, companion, condition)
    scene_rows = per_scene_metrics(final_output, requested, companion, indices)
    total_reduction = 1.0 - final["total"] / max(initial["total"], 1.0e-30)
    requested_reduction = 1.0 - final["requested_l1"] / max(initial["requested_l1"], 1.0e-30)
    companion_reduction = 1.0 - final["companion_l1"] / max(initial["companion_l1"], 1.0e-30)
    head_update = float(torch.linalg.vector_norm(model.source_head.weight.detach() - head_start).cpu())
    decoder_update = float(torch.linalg.vector_norm(model.dec2.up_convolution[0].weight.detach() - decoder_start).cpu())
    physical_pass = (
        final["source_negative_fraction"] == 0.0
        and final["source_nonfinite_fraction"] == 0.0
        and maximum_conservation <= maximum_tolerance
    )
    identity_pass = final["prompt_identity"] >= 0.90
    unchanged_micro_reconstruction_pass = (
        total_reduction >= 0.95 and requested_reduction >= 0.80 and companion_reduction >= 0.80
    )
    result = {
        "condition": condition,
        "scenes": scene_count,
        "prompt_views": 2 * scene_count,
        "steps": steps,
        "seed": seed,
        "device": "mps",
        "cpu_fallback": False,
        "architecture_parameters": trainable_parameter_count(model),
        "state_key_count": len(state_keys),
        "initial_total": initial["total"],
        "final_total": final["total"],
        "objective_reduction": total_reduction,
        "initial_requested_l1": initial["requested_l1"],
        "final_requested_l1": final["requested_l1"],
        "requested_l1_reduction": requested_reduction,
        "initial_companion_l1": initial["companion_l1"],
        "final_companion_l1": final["companion_l1"],
        "companion_l1_reduction": companion_reduction,
        **{key: value for key, value in final.items() if key not in {"total", "requested_l1", "companion_l1", "flux", "centroid", "color"}},
        "maximum_conservation_error": maximum_conservation,
        "maximum_conservation_tolerance": maximum_tolerance,
        "head_update_norm": head_update,
        "nonfinal_decoder_update_norm": decoder_update,
        "hidden_gradient_seen": hidden_gradient_seen,
        "unchanged_micro_reconstruction_pass": unchanged_micro_reconstruction_pass,
        "identity_pass": identity_pass,
        "physical_pass": physical_pass,
        "runtime_seconds": time.time() - started,
    }
    checkpoint = run / f"micro_overfit/{condition}_final_state.pth"
    save_torch_fresh(
        checkpoint,
        {
            "campaign": "Thayer-Family-E1P-v0",
            "condition": condition,
            "micro_only": True,
            "architecture": "FamilyE1UNet",
            "parameters": EXPECTED_PARAMETERS,
            "seed": seed,
            "steps": steps,
            "model_state_dict": cpu_state(model),
            "optimizer": {"name": "AdamW", "lr": 3.0e-3, "weight_decay": 1.0e-4, "gradient_clip": 5.0},
            "final_metrics": result,
        },
    )
    result["micro_checkpoint"] = relative(checkpoint)
    result["micro_checkpoint_sha256"] = sha256_file(checkpoint)
    fresh_csv(run / f"micro_overfit/{condition}_trace.csv", trace_rows)
    return result, layer_rows, scene_rows, gradient_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    provenance = validate_run(run)
    device = require_mps()
    started = time.time()
    all_results = []
    all_layers = []
    all_scenes = []
    all_gradients = []
    for condition, spec in MICRO_SPECS.items():
        result, layers, scenes, gradients = fit_condition(run, condition, spec, device)
        all_results.append(result)
        all_layers.extend(layers)
        all_scenes.extend({"condition": condition, **row} for row in scenes)
        all_gradients.append(gradients)
    fresh_csv(run / "tables/micro_overfit_results.csv", all_results)
    fresh_csv(run / "tables/layerwise_prompt_trace.csv", all_layers)
    fresh_csv(run / "tables/per_scene_pair_metrics.csv", all_scenes)
    fresh_csv(run / "tables/prompt_gradient_metrics.csv", all_gradients)
    first_indistinguishable = []
    for condition in MICRO_SPECS:
        rows = [
            row
            for row in all_layers
            if row["condition"] == condition and row["phase"] == "final" and row["group"] != "input"
        ]
        first = next((str(row["layer"]) for row in rows if bool(row["numerically_indistinguishable"])), "NONE")
        first_indistinguishable.append({"condition": condition, "first_numerically_indistinguishable_layer": first})
    fresh_csv(run / "tables/first_indistinguishable_layer.csv", first_indistinguishable)
    both_identity = all(bool(row["identity_pass"]) for row in all_results)
    all_physical = all(bool(row["physical_pass"]) for row in all_results)
    all_reconstruction = all(bool(row["unchanged_micro_reconstruction_pass"]) for row in all_results)
    fresh_json(
        run / "logs/micro_overfit_complete.json",
        {
            "status": "PASS" if both_identity and all_physical else "FAIL",
            "both_identity_pass": both_identity,
            "all_physical_pass": all_physical,
            "all_unchanged_micro_reconstruction_pass": all_reconstruction,
            "full_training_authorized": both_identity and all_physical,
            "device": "mps",
            "cpu_fallback": False,
            "validation_access_count": 0,
            "calibration_access_count": 0,
            "oof_outputs": 0,
            "safety_labels": 0,
            "auditor_models": 0,
            "runtime_seconds": time.time() - started,
            "bootstrap_historical_checkpoint_count": provenance["historical_checkpoint_count"],
        },
    )
    fresh_json(
        run / "architecture/unchanged_architecture_contract.json",
        {
            "architecture": "FamilyE1UNet",
            "model_source": "src/family_e1.py",
            "model_source_sha256": sha256_file(REPO / "src/family_e1.py"),
            "trainable_parameters": EXPECTED_PARAMETERS,
            "input_channels": 4,
            "raw_output_channels": 6,
            "encoder_widths": [24, 48, 96, 128],
            "source_mapping": "in_forward_relu",
            "signed_residual": "observed-requested-companion",
            "architecture_changes": 0,
            "parameter_changes": 0,
        },
    )
    print(
        json.dumps(
            {
                "run_dir": relative(run),
                "both_identity_pass": both_identity,
                "full_training_authorized": both_identity and all_physical,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
