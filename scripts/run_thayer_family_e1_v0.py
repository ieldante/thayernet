#!/usr/bin/env python3
"""Run the frozen Family-E1 correctness, micro, and training stages."""
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


FAMILY_E = REPO / "outputs/runs/thayer_family_e_v0_20260714_195256"
HIERARCHICAL = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
PARTITIONS = {
    "training": (
        HIERARCHICAL / "manifests/v2_r_training_scene_manifest.csv",
        HIERARCHICAL / "manifests/v2_r_training_scenes.h5",
    ),
    "validation": (
        HIERARCHICAL / "manifests/v2_r_validation_scene_manifest.csv",
        HIERARCHICAL / "manifests/v2_r_validation_scenes.h5",
    ),
    "calibration": (
        HIERARCHICAL / "manifests/v2_natural_calibration_scene_manifest.csv",
        HIERARCHICAL / "manifests/v2_natural_calibration_scenes.h5",
    ),
}
PRIMARY_SEEDS = (2026071501, 2026071502, 2026071503)
MICRO_SPECS = {
    "ordinary_one_scene": ([16], 1500, 2026071511),
    "difficult_one_scene": ([6], 2000, 2026071512),
    "mixed_eight_scene": ([0, 3, 5, 6, 18, 51, 73, 81], 3000, 2026071513),
}
SCALES = np.asarray(BAND_SCALES, dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_run(run: Path, prerequisite: str | None = None) -> None:
    if run.parent.resolve() != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_family_e1_v0_"):
        raise RuntimeError("unexpected Family-E1 run path")
    record = json.loads((run / "logs/preregistration_complete.json").read_text())
    prereg = REPO / record["path"]
    if record["status"] != "FROZEN_BEFORE_MODEL_CONSTRUCTION_OR_FITTING" or sha256_file(prereg) != record["sha256"]:
        raise RuntimeError("preregistration freeze mismatch")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is prohibited")
    staged = os.popen(f"cd {REPO!s} && git diff --cached --name-only").read().strip()
    if staged:
        raise RuntimeError("staged index is not empty")
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    for item in provenance["authoritative_inputs"]:
        path = REPO / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"authoritative input changed: {path}")
    for partition in PARTITIONS:
        current = run / f"manifests/{partition}_manifest.csv"
        authority = FAMILY_E / f"manifests/{partition}_manifest.csv"
        if sha256_file(current) != sha256_file(authority):
            raise RuntimeError(f"selector reference mismatch: {partition}")
    if prerequisite is not None:
        status = json.loads((run / "logs" / prerequisite).read_text())
        if status["status"] != "PASS":
            raise RuntimeError(f"prerequisite did not pass: {prerequisite}")


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, dtype=torch.float32, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.mps.manual_seed(seed)


def selected_manifest(run: Path, partition: str) -> tuple[pd.DataFrame, np.ndarray]:
    selector = pd.read_csv(run / f"manifests/{partition}_manifest.csv")
    upstream = pd.read_csv(PARTITIONS[partition][0], low_memory=False)
    indices = selector.upstream_index.to_numpy(dtype=np.int64)
    selected = upstream.iloc[indices].reset_index(drop=True)
    return selected, indices


def ordered_targets(isolated: np.ndarray, matched: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = np.arange(len(isolated))
    requested = isolated[local, matched]
    companion = isolated[local, 1 - matched]
    return np.ascontiguousarray(requested), np.ascontiguousarray(companion)


def load_rows(run: Path, partition: str, family_indices: list[int] | np.ndarray) -> dict[str, np.ndarray | pd.DataFrame]:
    selected, upstream_indices = selected_manifest(run, partition)
    local_indices = np.asarray(family_indices, dtype=np.int64)
    upstream = upstream_indices[local_indices]
    with h5py.File(PARTITIONS[partition][1], "r") as handle:
        blend = np.asarray(handle["blend"][upstream], dtype=np.float32)
        isolated = np.asarray(handle["isolated"][upstream], dtype=np.float32)
        prompt = np.asarray(handle["prompt"][upstream], dtype=np.float32)
        xy = np.asarray(handle["xy"][upstream], dtype=np.float64)
        prompt_xy = np.asarray(handle["prompt_xy"][upstream], dtype=np.float64)
    rows = selected.iloc[local_indices].reset_index(drop=True)
    matched = rows.matched_source_index.to_numpy(dtype=np.int64)
    requested, companion = ordered_targets(isolated, matched)
    return {
        "rows": rows, "blend": blend, "prompt": prompt, "xy": xy,
        "prompt_xy": prompt_xy, "matched": matched,
        "requested": requested, "companion": companion,
    }


def augmented_prompt_views(data: dict[str, np.ndarray | pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blend = np.asarray(data["blend"], dtype=np.float32)
    prompt = np.asarray(data["prompt"], dtype=np.float32)
    xy = np.asarray(data["xy"], dtype=np.float64)
    prompt_xy = np.asarray(data["prompt_xy"], dtype=np.float64)
    matched = np.asarray(data["matched"], dtype=np.int64)
    requested = np.asarray(data["requested"], dtype=np.float32)
    companion = np.asarray(data["companion"], dtype=np.float32)
    alternate_prompts = []
    for index in range(len(blend)):
        offset = prompt_xy[index] - xy[index, matched[index]]
        alternate_xy = xy[index, 1 - matched[index]] + offset
        alternate_prompts.append(gaussian_prompt_numpy(float(alternate_xy[0]), float(alternate_xy[1]))[None])
    alternate = np.asarray(alternate_prompts, dtype=np.float32)
    normalized = blend / SCALES[None, :, None, None]
    inputs_a = np.concatenate((normalized, prompt), axis=1)
    inputs_b = np.concatenate((normalized, alternate), axis=1)
    return (
        np.ascontiguousarray(np.concatenate((inputs_a, inputs_b), axis=0)),
        np.ascontiguousarray(np.concatenate((blend, blend), axis=0)),
        np.ascontiguousarray(np.concatenate((requested, companion), axis=0)),
        np.ascontiguousarray(np.concatenate((companion, requested), axis=0)),
    )


def source_group_audit(run: Path) -> list[dict[str, object]]:
    selected = {name: selected_manifest(run, name)[0] for name in PARTITIONS}
    result: list[dict[str, object]] = []
    for name, frame in selected.items():
        pairs = frame.apply(lambda row: tuple(sorted((str(row.source_a_id), str(row.source_b_id)))), axis=1)
        result.append({"check": f"{name}_row_count", "observed": len(frame), "required": {"training": 10000, "validation": 2000, "calibration": 2000}[name], "pass": len(frame) == {"training": 10000, "validation": 2000, "calibration": 2000}[name]})
        result.append({"check": f"{name}_duplicate_source_pairs", "observed": int(pairs.duplicated().sum()), "required": 0, "pass": not pairs.duplicated().any()})
        result.append({"check": f"{name}_query_state", "observed": ",".join(sorted(frame.query_state.unique())), "required": "UNIQUE_VALID", "pass": frame.query_state.eq("UNIQUE_VALID").all()})
        result.append({"check": f"{name}_unique_prompt_hashes", "observed": int(frame.prompt_sha256.nunique()), "required": len(frame), "pass": frame.prompt_sha256.nunique() == len(frame)})
    for left, right in (("training", "validation"), ("training", "calibration"), ("validation", "calibration")):
        left_groups = set(selected[left].source_a_group) | set(selected[left].source_b_group)
        right_groups = set(selected[right].source_a_group) | set(selected[right].source_b_group)
        left_pairs = {tuple(sorted(pair)) for pair in zip(selected[left].source_a_id, selected[left].source_b_id)}
        right_pairs = {tuple(sorted(pair)) for pair in zip(selected[right].source_a_id, selected[right].source_b_id)}
        result.append({"check": f"{left}_{right}_source_group_overlap", "observed": len(left_groups & right_groups), "required": 0, "pass": not (left_groups & right_groups)})
        result.append({"check": f"{left}_{right}_source_pair_overlap", "observed": len(left_pairs & right_pairs), "required": 0, "pass": not (left_pairs & right_pairs)})
    training = selected["training"]
    fold_group_sets = []
    for fold in range(5):
        frame = training[pd.read_csv(run / "manifests/training_manifest.csv").oof_fold.to_numpy() == fold]
        fold_group_sets.append(set(frame.source_a_group) | set(frame.source_b_group))
        result.append({"check": f"fold_{fold}_rows", "observed": len(frame), "required": 2000, "pass": len(frame) == 2000})
    overlap = max(len(fold_group_sets[a] & fold_group_sets[b]) for a in range(5) for b in range(a + 1, 5))
    result.append({"check": "maximum_cross_fold_group_overlap", "observed": overlap, "required": 0, "pass": overlap == 0})
    return result


def output_diagnostics(output: FamilyE1Output, observed: torch.Tensor) -> dict[str, float]:
    values = torch.cat((output.requested.flatten(), output.companion.flatten()))
    residual = output.residual_noise
    magnitude = max(
        1.0, float(observed.detach().abs().max().cpu()),
        float(output.requested.detach().abs().max().cpu()),
        float(output.companion.detach().abs().max().cpu()),
        float(residual.detach().abs().max().cpu()),
    )
    return {
        "source_minimum": float(values.detach().min().cpu()),
        "source_maximum": float(values.detach().max().cpu()),
        "source_negative_fraction": float((values < 0).float().mean().detach().cpu()),
        "source_nonfinite_fraction": float((~torch.isfinite(values)).float().mean().detach().cpu()),
        "residual_negative_fraction": float((residual < 0).float().mean().detach().cpu()),
        "residual_positive_fraction": float((residual > 0).float().mean().detach().cpu()),
        "conservation_error": float(conservation_error(output, observed).detach().cpu()),
        "conservation_tolerance": 1.0e-5 * magnitude,
    }


def stage_preflight(run: Path) -> None:
    validate_run(run)
    device = require_mps()
    leakage_rows = source_group_audit(run)
    if not all(bool(row["pass"]) for row in leakage_rows):
        raise RuntimeError("source-group or manifest audit failed")
    fresh_csv(run / "tables/source_group_leakage_tests.csv", leakage_rows)

    seed_everything(2026071501)
    model = FamilyE1UNet().to(device)
    count = trainable_parameter_count(model)
    if count != EXPECTED_PARAMETERS or count > 3_000_000 or next(model.parameters()).device.type != "mps":
        raise RuntimeError("architecture count/device gate failed")
    module_types = [type(module).__name__ for module in model.modules()]
    if any(name in module_types for name in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "MultiheadAttention", "Transformer")):
        raise RuntimeError("prohibited architecture module")
    architecture = {
        "architecture_id": "THAYER_FAMILY_E1_COMPACT_COORDINATE_UNET",
        "input_channels": 4, "output_raw_channels": 6,
        "encoder_widths": [24, 48, 96, 128], "group_norm_groups": 8,
        "activation": "SiLU", "upsampling": "bilinear_align_corners_false",
        "source_mapping": "in_forward_relu", "signed_residual": "observed-requested-companion",
        "trainable_parameters": count, "parameter_ceiling": 3_000_000,
        "model_source": relative(REPO / "src/family_e1.py"),
        "model_source_sha256": sha256_file(REPO / "src/family_e1.py"),
        "architecture_variants_constructed": 1,
        "batch_norm": False, "attention": False, "latent_variables": False,
    }
    fresh_json(run / "architecture/architecture_manifest.json", architecture)

    # Synthetic physical fixture on MPS.
    model.eval()
    synthetic_input = torch.randn(2, 4, 60, 60, device=device)
    synthetic_observed = torch.randn(2, 3, 60, 60, device=device)
    synthetic = model(synthetic_input, synthetic_observed)
    physical = output_diagnostics(synthetic, synthetic_observed)
    zero_raw = torch.zeros(1, 6, 2, 2, device=device, requires_grad=True)
    negative_raw = -torch.ones(1, 6, 2, 2, device=device)
    scales = torch.tensor(BAND_SCALES, device=device).reshape(1, 3, 1, 1)
    zero_sources = torch.relu(zero_raw) * torch.cat((scales, scales), dim=1)
    negative_sources = torch.relu(negative_raw) * torch.cat((scales, scales), dim=1)
    physical.update({
        "device": "mps", "cpu_fallback": False,
        "zero_logits_exact_zero": bool(torch.equal(zero_sources, torch.zeros_like(zero_sources))),
        "negative_logits_exact_zero": bool(torch.equal(negative_sources, torch.zeros_like(negative_sources))),
        "residual_has_both_signs": physical["residual_negative_fraction"] > 0 and physical["residual_positive_fraction"] > 0,
    })
    physical_pass = (
        physical["source_negative_fraction"] == 0.0
        and physical["source_nonfinite_fraction"] == 0.0
        and physical["conservation_error"] <= physical["conservation_tolerance"]
        and physical["zero_logits_exact_zero"] and physical["negative_logits_exact_zero"]
        and physical["residual_has_both_signs"]
    )
    physical["status"] = "PASS" if physical_pass else "FAIL"
    fresh_json(run / "physical_contract/mps_physical_preflight.json", physical)
    if not physical_pass:
        raise RuntimeError("physical MPS preflight failed")

    # Objective alignment: synthetic fixture and the frozen micro crops.
    audit_rows: list[dict[str, object]] = []
    fixture_req = torch.zeros(1, 3, 60, 60, device=device)
    fixture_comp = torch.zeros_like(fixture_req)
    fixture_req[:, 0, 16:25, 13:22] = 1.0
    fixture_req[:, 1, 17:25, 14:23] = 0.7
    fixture_req[:, 2, 18:24, 15:22] = 0.4
    fixture_comp[:, 0, 34:42, 39:47] = 0.3
    fixture_comp[:, 1, 33:43, 38:48] = 0.9
    fixture_comp[:, 2, 35:41, 40:46] = 1.2
    fixtures: list[tuple[str, torch.Tensor, torch.Tensor]] = [("synthetic", fixture_req, fixture_comp)]
    micro_union = sorted({index for spec in MICRO_SPECS.values() for index in spec[0]})
    frozen = load_rows(run, "training", micro_union)
    frozen_req = torch.from_numpy(np.asarray(frozen["requested"], dtype=np.float32)).to(device)
    frozen_comp = torch.from_numpy(np.asarray(frozen["companion"], dtype=np.float32)).to(device)
    for local, index in enumerate(micro_union):
        fixtures.append((f"training_{index}", frozen_req[local:local+1], frozen_comp[local:local+1]))
    stationary = True
    compromise_beats = False
    swapping_penalized = True
    gradients_finite = True
    gradients_nonzero = True
    for name, truth_req, truth_comp in fixtures:
        exact_req = truth_req.detach().clone().requires_grad_(True)
        exact_comp = truth_comp.detach().clone().requires_grad_(True)
        exact = source_objective(exact_req, exact_comp, truth_req, truth_comp)
        exact["total"].backward()
        exact_gradient = max(float(exact_req.grad.abs().max().cpu()), float(exact_comp.grad.abs().max().cpu()))
        stationary = stationary and exact_gradient == 0.0 and float(exact["total"].detach().cpu()) == 0.0
        candidates = {
            "equal_average": ((truth_req + truth_comp) / 2, (truth_req + truth_comp) / 2),
            "swap": (truth_comp, truth_req),
            "requested_scale_0p8": (0.8 * truth_req, truth_comp),
            "companion_scale_0p8": (truth_req, 0.8 * truth_comp),
        }
        for candidate_name, (candidate_req, candidate_comp) in candidates.items():
            req = candidate_req.detach().clone().requires_grad_(True)
            comp = candidate_comp.detach().clone().requires_grad_(True)
            values = source_objective(req, comp, truth_req, truth_comp)
            values["total"].backward()
            grad_max = max(float(req.grad.abs().max().cpu()), float(comp.grad.abs().max().cpu()))
            finite = bool(torch.isfinite(req.grad).all().cpu() and torch.isfinite(comp.grad).all().cpu())
            total = float(values["total"].detach().cpu())
            compromise_beats = compromise_beats or total < float(exact["total"].detach().cpu()) - 1e-12
            if candidate_name == "swap":
                swapping_penalized = swapping_penalized and total > float(exact["total"].detach().cpu())
            gradients_finite = gradients_finite and finite
            gradients_nonzero = gradients_nonzero and grad_max > 0.0
            audit_rows.append({
                "fixture": name, "candidate": candidate_name,
                "exact_total": float(exact["total"].detach().cpu()), "candidate_total": total,
                "candidate_requested_l1": float(values["requested_l1"].detach().cpu()),
                "candidate_companion_l1": float(values["companion_l1"].detach().cpu()),
                "candidate_flux": float(values["flux"].detach().cpu()),
                "candidate_centroid": float(values["centroid"].detach().cpu()),
                "candidate_color": float(values["color"].detach().cpu()),
                "candidate_beats_truth": total < float(exact["total"].detach().cpu()) - 1e-12,
                "gradient_finite": finite, "gradient_max_abs": grad_max,
            })

    ordinary = load_rows(run, "training", [16])
    ordinary_inputs, ordinary_observed, ordinary_req, ordinary_comp = augmented_prompt_views(ordinary)
    model.train(); model.zero_grad(set_to_none=True)
    output = model(torch.from_numpy(ordinary_inputs).to(device), torch.from_numpy(ordinary_observed).to(device))
    init_loss = source_objective(output.requested, output.companion, torch.from_numpy(ordinary_req).to(device), torch.from_numpy(ordinary_comp).to(device))
    init_loss["total"].backward()
    raw_inactive = float((output.raw <= 0).float().mean().detach().cpu())
    raw_inactive_on_positive_target = float(((output.raw <= 0) & (torch.cat((torch.from_numpy(ordinary_req), torch.from_numpy(ordinary_comp)), dim=1).to(device) > 0)).float().sum().detach().cpu() / max(1.0, float((torch.cat((torch.from_numpy(ordinary_req), torch.from_numpy(ordinary_comp)), dim=1) > 0).sum())))
    hidden_gradient = model.dec2.up_convolution[0].weight.grad
    hidden_attached = hidden_gradient is not None and bool(torch.isfinite(hidden_gradient).all().cpu()) and float(hidden_gradient.abs().max().cpu()) > 0
    objective_pass = stationary and not compromise_beats and swapping_penalized and gradients_finite and gradients_nonzero and hidden_attached
    fresh_csv(run / "tables/objective_alignment_audit.csv", audit_rows)
    fresh_json(run / "objective_audit/objective_alignment_summary.json", {
        "status": "PASS" if objective_pass else "FAIL",
        "truth_stationary_minimum": stationary,
        "compromise_beats_truth": compromise_beats,
        "requested_companion_swap_penalized": swapping_penalized,
        "gradients_finite": gradients_finite,
        "gradients_nonzero_away_from_optimum": gradients_nonzero,
        "relu_inactive_fraction_at_initialization": raw_inactive,
        "relu_inactive_fraction_on_positive_target_pixels": raw_inactive_on_positive_target,
        "hidden_decoder_gradient_attached": hidden_attached,
        "zero_target_pixels_representable": True,
        "conservation_exact_within_frozen_tolerance": physical_pass,
    })
    fresh_text(run / "diagnostics/objective_alignment.md", f"""# Family-E1 objective-alignment audit

Status: **{'PASS' if objective_pass else 'FAIL'}**.

- Exact truth stationary global minimum: `{stationary}`.
- Any compromise beat truth: `{compromise_beats}`.
- Requested/companion swap penalized on every fixture: `{swapping_penalized}`.
- Finite, nonzero gradients away from truth: `{gradients_finite and gradients_nonzero}`.
- Initial ReLU inactive fraction / inactive fraction on positive targets: `{raw_inactive:.6f}` / `{raw_inactive_on_positive_target:.6f}`.
- Hidden decoder gradient attached: `{hidden_attached}`.
- Zero target pixels remain exactly representable; signed closure remains within the frozen tolerance.
""")
    if not objective_pass:
        fresh_json(run / "logs/preflight_complete.json", {"status": "FAIL", "reason": "OBJECTIVE_ALIGNMENT"})
        raise RuntimeError("objective alignment failed")
    fresh_json(run / "logs/preflight_complete.json", {
        "status": "PASS", "architecture_parameters": count,
        "architecture_manifest_sha256": sha256_file(run / "architecture/architecture_manifest.json"),
        "model_source_sha256": architecture["model_source_sha256"],
        "objective_alignment": "PASS", "physical_contract": "PASS",
        "source_group_leakage": "PASS", "mps": "PASS", "cpu_fallback": False,
    })


def micro_identity(output: FamilyE1Output, requested: torch.Tensor, companion: torch.Tensor) -> float:
    scales = torch.tensor(BAND_SCALES, device=output.requested.device).reshape(1, 3, 1, 1)
    own = torch.mean(((output.requested - requested) / scales) ** 2, dim=(1, 2, 3))
    inverted = torch.mean(((output.requested - companion) / scales) ** 2, dim=(1, 2, 3))
    return float((own < inverted).float().mean().detach().cpu())


def fit_micro(run: Path, name: str, indices: list[int], steps: int, seed: int, device: torch.device) -> dict[str, object]:
    seed_everything(seed)
    data = load_rows(run, "training", indices)
    model_input_np, observed_np, requested_np, companion_np = augmented_prompt_views(data)
    model_input = torch.from_numpy(model_input_np).to(device)
    observed = torch.from_numpy(observed_np).to(device)
    requested = torch.from_numpy(requested_np).to(device)
    companion = torch.from_numpy(companion_np).to(device)
    model = FamilyE1UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    head_start = model.source_head.weight.detach().clone()
    decoder_start = model.dec2.up_convolution[0].weight.detach().clone()
    model.train()
    with torch.no_grad():
        initial_output = model(model_input, observed)
        initial = source_objective(initial_output.requested, initial_output.companion, requested, companion)
    initial_values = {key: float(value.detach().cpu()) for key, value in initial.items()}
    hidden_gradient_seen = False
    maximum_conservation = 0.0
    maximum_tolerance = 0.0
    checkpoints = {0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1}
    trace_rows: list[dict[str, object]] = []
    started = time.time()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(model_input, observed)
        losses = source_objective(output.requested, output.companion, requested, companion)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"nonfinite micro objective: {name} step {step}")
        losses["total"].backward()
        gradient = model.dec2.up_convolution[0].weight.grad
        hidden_gradient_seen = hidden_gradient_seen or (gradient is not None and float(gradient.detach().abs().max().cpu()) > 0)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
        optimizer.step()
        diagnostics = output_diagnostics(output, observed)
        maximum_conservation = max(maximum_conservation, diagnostics["conservation_error"])
        maximum_tolerance = max(maximum_tolerance, diagnostics["conservation_tolerance"])
        if step in checkpoints:
            trace_rows.append({
                "condition": name, "step": step + 1,
                **{key: float(value.detach().cpu()) for key, value in losses.items()},
                "gradient_norm": gradient_norm,
                "identity_rate": micro_identity(output, requested, companion),
                "negative_fraction": diagnostics["source_negative_fraction"],
                "nonfinite_fraction": diagnostics["source_nonfinite_fraction"],
                "conservation_error": diagnostics["conservation_error"],
            })
    model.eval()
    with torch.no_grad():
        final_output = model(model_input, observed)
        final = source_objective(final_output.requested, final_output.companion, requested, companion)
    final_values = {key: float(value.detach().cpu()) for key, value in final.items()}
    diagnostics = output_diagnostics(final_output, observed)
    total_reduction = 1.0 - final_values["total"] / max(initial_values["total"], 1e-30)
    requested_reduction = 1.0 - final_values["requested_l1"] / max(initial_values["requested_l1"], 1e-30)
    companion_reduction = 1.0 - final_values["companion_l1"] / max(initial_values["companion_l1"], 1e-30)
    identity = micro_identity(final_output, requested, companion)
    head_update = float(torch.linalg.vector_norm(model.source_head.weight.detach() - head_start).cpu())
    decoder_update = float(torch.linalg.vector_norm(model.dec2.up_convolution[0].weight.detach() - decoder_start).cpu())
    passed = (
        total_reduction >= 0.95 and requested_reduction >= 0.80 and companion_reduction >= 0.80
        and identity >= 0.90 and diagnostics["source_negative_fraction"] == 0.0
        and diagnostics["source_nonfinite_fraction"] == 0.0
        and maximum_conservation <= maximum_tolerance and head_update > 0 and decoder_update > 0
        and hidden_gradient_seen and next(model.parameters()).device.type == "mps"
    )
    fresh_csv(run / f"micro_overfit/{name}_trace.csv", trace_rows)
    return {
        "condition": name, "scenes": len(indices), "prompt_views": len(model_input_np),
        "steps": steps, "seed": seed, "device": "mps", "cpu_fallback": False,
        "initial_total": initial_values["total"], "final_total": final_values["total"],
        "objective_reduction": total_reduction,
        "initial_requested_l1": initial_values["requested_l1"], "final_requested_l1": final_values["requested_l1"],
        "requested_l1_reduction": requested_reduction,
        "initial_companion_l1": initial_values["companion_l1"], "final_companion_l1": final_values["companion_l1"],
        "companion_l1_reduction": companion_reduction,
        "prompt_identity_rate": identity, "systematic_source_inversion": identity < 0.5,
        "requested_negative_fraction": float((final_output.requested < 0).float().mean().cpu()),
        "companion_negative_fraction": float((final_output.companion < 0).float().mean().cpu()),
        "nonfinite_fraction": diagnostics["source_nonfinite_fraction"],
        "maximum_conservation_error": maximum_conservation,
        "maximum_conservation_tolerance": maximum_tolerance,
        "head_update_norm": head_update, "nonfinal_decoder_update_norm": decoder_update,
        "hidden_gradient_seen": hidden_gradient_seen,
        "runtime_seconds": time.time() - started, "pass": passed,
    }


def stage_micro(run: Path) -> None:
    validate_run(run, "preflight_complete.json")
    device = require_mps()
    results = [fit_micro(run, name, *spec, device) for name, spec in MICRO_SPECS.items()]
    fresh_csv(run / "tables/micro_overfit_results.csv", results)
    ordinary = next(row for row in results if row["condition"] == "ordinary_one_scene")
    difficult = next(row for row in results if row["condition"] == "difficult_one_scene")
    mixed = next(row for row in results if row["condition"] == "mixed_eight_scene")
    passed = bool(ordinary["pass"] and mixed["pass"])
    fresh_text(run / "diagnostics/micro_overfit_report.md", f"""# Family-E1 micro-overfit report

Authoritative gate: **{'PASS' if passed else 'FAIL'}**. Ordinary / difficult / mixed-eight status: `{ordinary['pass']} / {difficult['pass']} / {mixed['pass']}`.

| Condition | Objective reduction | Requested L1 reduction | Companion L1 reduction | Prompt identity | Closure max / tolerance | Status |
|---|---:|---:|---:|---:|---:|---|
| ordinary | {ordinary['objective_reduction']:.6f} | {ordinary['requested_l1_reduction']:.6f} | {ordinary['companion_l1_reduction']:.6f} | {ordinary['prompt_identity_rate']:.6f} | {ordinary['maximum_conservation_error']:.6g} / {ordinary['maximum_conservation_tolerance']:.6g} | {'PASS' if ordinary['pass'] else 'FAIL'} |
| difficult | {difficult['objective_reduction']:.6f} | {difficult['requested_l1_reduction']:.6f} | {difficult['companion_l1_reduction']:.6f} | {difficult['prompt_identity_rate']:.6f} | {difficult['maximum_conservation_error']:.6g} / {difficult['maximum_conservation_tolerance']:.6g} | {'PASS' if difficult['pass'] else 'FAIL'} |
| mixed eight | {mixed['objective_reduction']:.6f} | {mixed['requested_l1_reduction']:.6f} | {mixed['companion_l1_reduction']:.6f} | {mixed['prompt_identity_rate']:.6f} | {mixed['maximum_conservation_error']:.6g} / {mixed['maximum_conservation_tolerance']:.6g} | {'PASS' if mixed['pass'] else 'FAIL'} |

All neural work used MPS. Requested/companion negative fractions and nonfinite fractions remained exactly zero. Head and non-final decoder updates and attached hidden gradients are recorded in the CSV.
""")
    fresh_json(run / "logs/micro_overfit_complete.json", {
        "status": "PASS" if passed else "FAIL", "ordinary_pass": bool(ordinary["pass"]),
        "difficult_pass": bool(difficult["pass"]), "mixed_eight_pass": bool(mixed["pass"]),
        "full_training_authorized": passed, "cpu_fallback": False,
    })
    if not passed:
        raise RuntimeError("ordinary or eight-scene micro-overfit gate failed")


def load_partition_arrays(run: Path, partition: str) -> dict[str, np.ndarray | pd.DataFrame]:
    selected, indices = selected_manifest(run, partition)
    with h5py.File(PARTITIONS[partition][1], "r") as handle:
        blend = np.asarray(handle["blend"][indices], dtype=np.float32)
        isolated = np.asarray(handle["isolated"][indices], dtype=np.float32)
        prompt = np.asarray(handle["prompt"][indices], dtype=np.float32)
        xy = np.asarray(handle["xy"][indices], dtype=np.float64)
        prompt_xy = np.asarray(handle["prompt_xy"][indices], dtype=np.float64)
    matched = selected.matched_source_index.to_numpy(dtype=np.int64)
    requested, companion = ordered_targets(isolated, matched)
    del isolated
    model_input = np.concatenate((blend / SCALES[None, :, None, None], prompt), axis=1).astype(np.float32)
    return {
        "rows": selected, "model_input": np.ascontiguousarray(model_input),
        "requested": requested, "companion": companion,
        "xy": xy, "prompt_xy": prompt_xy, "matched": matched,
    }


def tensor_batch(array: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array[indices])).to(device)


def validation_metrics(model: FamilyE1UNet, data: dict[str, np.ndarray | pd.DataFrame], device: torch.device, batch_size: int = 16) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("total", "requested_l1", "companion_l1", "flux", "centroid", "color")}
    diagnostics = {"source_minimum": math.inf, "source_maximum": -math.inf, "source_negative_count": 0, "source_nonfinite_count": 0, "source_count": 0, "conservation_error": 0.0, "conservation_tolerance": 0.0}
    n = len(np.asarray(data["model_input"]))
    with torch.no_grad():
        for start in range(0, n, batch_size):
            index = np.arange(start, min(start + batch_size, n))
            model_input = tensor_batch(np.asarray(data["model_input"]), index, device)
            requested = tensor_batch(np.asarray(data["requested"]), index, device)
            companion = tensor_batch(np.asarray(data["companion"]), index, device)
            observed = model_input[:, :3] * model.band_scales
            output = model(model_input, observed)
            losses = source_objective(output.requested, output.companion, requested, companion)
            for key in totals:
                totals[key] += float(losses[key].cpu()) * len(index)
            detail = output_diagnostics(output, observed)
            diagnostics["source_minimum"] = min(diagnostics["source_minimum"], detail["source_minimum"])
            diagnostics["source_maximum"] = max(diagnostics["source_maximum"], detail["source_maximum"])
            combined = torch.cat((output.requested.flatten(), output.companion.flatten()))
            diagnostics["source_negative_count"] += int((combined < 0).sum().cpu())
            diagnostics["source_nonfinite_count"] += int((~torch.isfinite(combined)).sum().cpu())
            diagnostics["source_count"] += combined.numel()
            diagnostics["conservation_error"] = max(diagnostics["conservation_error"], detail["conservation_error"])
            diagnostics["conservation_tolerance"] = max(diagnostics["conservation_tolerance"], detail["conservation_tolerance"])
    metrics = {key: value / n for key, value in totals.items()}
    metrics.update({
        "source_minimum": diagnostics["source_minimum"], "source_maximum": diagnostics["source_maximum"],
        "negative_fraction": diagnostics["source_negative_count"] / diagnostics["source_count"],
        "nonfinite_fraction": diagnostics["source_nonfinite_count"] / diagnostics["source_count"],
        "conservation_error": diagnostics["conservation_error"], "conservation_tolerance": diagnostics["conservation_tolerance"],
    })
    return metrics


def prompt_swap_metric(model: FamilyE1UNet, data: dict[str, np.ndarray | pd.DataFrame], device: torch.device, limit: int = 100) -> float:
    count = min(limit, len(np.asarray(data["model_input"])))
    inputs = np.asarray(data["model_input"][:count]).copy()
    xy = np.asarray(data["xy"][:count]); prompt_xy = np.asarray(data["prompt_xy"][:count]); matched = np.asarray(data["matched"][:count])
    alternate_prompts = []
    for index in range(count):
        offset = prompt_xy[index] - xy[index, matched[index]]
        position = xy[index, 1 - matched[index]] + offset
        alternate_prompts.append(gaussian_prompt_numpy(float(position[0]), float(position[1])))
    alternate_inputs = inputs.copy(); alternate_inputs[:, 3] = np.asarray(alternate_prompts)
    requested = np.asarray(data["requested"][:count]); companion = np.asarray(data["companion"][:count])
    successes = []
    model.eval()
    with torch.no_grad():
        for start in range(0, count, 16):
            end = min(start + 16, count)
            original_input = torch.from_numpy(np.ascontiguousarray(inputs[start:end])).to(device)
            alternate_input = torch.from_numpy(np.ascontiguousarray(alternate_inputs[start:end])).to(device)
            original = model(original_input, original_input[:, :3] * model.band_scales).requested.cpu().numpy()
            alternate = model(alternate_input, alternate_input[:, :3] * model.band_scales).requested.cpu().numpy()
            req = requested[start:end]; comp = companion[start:end]
            original_own = np.mean(((original - req) / SCALES[None, :, None, None]) ** 2, axis=(1, 2, 3))
            original_wrong = np.mean(((original - comp) / SCALES[None, :, None, None]) ** 2, axis=(1, 2, 3))
            alternate_own = np.mean(((alternate - comp) / SCALES[None, :, None, None]) ** 2, axis=(1, 2, 3))
            alternate_wrong = np.mean(((alternate - req) / SCALES[None, :, None, None]) ** 2, axis=(1, 2, 3))
            successes.extend(((original_own < original_wrong) & (alternate_own < alternate_wrong)).tolist())
    return float(np.mean(successes))


def cpu_state(model: FamilyE1UNet) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint_fresh(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)


def train_one(
    run: Path,
    name: str,
    seed: int,
    training: dict[str, np.ndarray | pd.DataFrame],
    validation: dict[str, np.ndarray | pd.DataFrame],
    train_indices: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    model = FamilyE1UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0.0)
    best_tuple = (math.inf, math.inf, math.inf, math.inf)
    best_state = None
    best_epoch = -1
    best_metrics: dict[str, float] | None = None
    no_improvement = 0
    epoch_rows: list[dict[str, object]] = []
    started = time.time()
    input_array = np.asarray(training["model_input"]); req_array = np.asarray(training["requested"]); comp_array = np.asarray(training["companion"])
    for epoch in range(1, 41):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(train_indices)
        totals = {key: 0.0 for key in ("total", "requested_l1", "companion_l1", "flux", "centroid", "color")}
        examples = 0
        gradient_norm_sum = 0.0
        gradient_batches = 0
        epoch_start = {name: value.detach().clone() for name, value in model.named_parameters()}
        train_min = math.inf; train_max = -math.inf; negative_count = 0; nonfinite_count = 0; source_count = 0; closure_max = 0.0; tolerance_max = 0.0
        for start in range(0, len(order), 16):
            index = order[start:start + 16]
            model_input = tensor_batch(input_array, index, device)
            requested = tensor_batch(req_array, index, device)
            companion = tensor_batch(comp_array, index, device)
            observed = model_input[:, :3] * model.band_scales
            optimizer.zero_grad(set_to_none=True)
            output = model(model_input, observed)
            losses = source_objective(output.requested, output.companion, requested, companion)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"nonfinite training loss: {name} epoch {epoch}")
            losses["total"].backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
            if not math.isfinite(gradient_norm):
                raise RuntimeError("nonfinite gradient norm")
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * len(index)
            examples += len(index); gradient_norm_sum += gradient_norm; gradient_batches += 1
            combined = torch.cat((output.requested.flatten(), output.companion.flatten()))
            train_min = min(train_min, float(combined.detach().min().cpu())); train_max = max(train_max, float(combined.detach().max().cpu()))
            negative_count += int((combined < 0).sum().detach().cpu()); nonfinite_count += int((~torch.isfinite(combined)).sum().detach().cpu()); source_count += combined.numel()
            detail = output_diagnostics(output, observed); closure_max = max(closure_max, detail["conservation_error"]); tolerance_max = max(tolerance_max, detail["conservation_tolerance"])
        update_norm_sq = 0.0
        with torch.no_grad():
            for parameter_name, parameter in model.named_parameters():
                update_norm_sq += float(torch.sum((parameter - epoch_start[parameter_name]) ** 2).cpu())
        validation_result = validation_metrics(model, validation, device)
        prompt_swap = prompt_swap_metric(model, validation, device, limit=100)
        current_tuple = (validation_result["requested_l1"], validation_result["companion_l1"], validation_result["flux"], validation_result["centroid"])
        improved = current_tuple < best_tuple
        if improved:
            best_tuple = current_tuple; best_state = cpu_state(model); best_epoch = epoch; best_metrics = dict(validation_result); no_improvement = 0
        else:
            no_improvement += 1
        epoch_rows.append({
            "model": name, "seed": seed, "epoch": epoch,
            **{f"training_{key}": value / examples for key, value in totals.items()},
            **{f"validation_{key}": value for key, value in validation_result.items()},
            "training_source_minimum": train_min, "training_source_maximum": train_max,
            "training_negative_fraction": negative_count / source_count, "training_nonfinite_fraction": nonfinite_count / source_count,
            "training_conservation_error": closure_max, "training_conservation_tolerance": tolerance_max,
            "validation_prompt_swap": prompt_swap,
            "gradient_norm_mean": gradient_norm_sum / gradient_batches, "update_norm": math.sqrt(update_norm_sq),
            "learning_rate": optimizer.param_groups[0]["lr"], "selected_so_far": improved,
        })
        print(json.dumps({"model": name, "epoch": epoch, "train": totals["total"] / examples, "validation": validation_result["total"], "requested": validation_result["requested_l1"], "prompt_swap": prompt_swap, "selected": improved}), flush=True)
        scheduler.step()
        if no_improvement >= 8:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("no selected checkpoint")
    save_checkpoint_fresh(checkpoint_path, {
        "architecture_id": "THAYER_FAMILY_E1_COMPACT_COORDINATE_UNET",
        "seed": seed, "model_name": name, "selected_epoch": best_epoch,
        "selection_rule": "validation_lexicographic_requested_companion_flux_centroid",
        "selection_metrics": best_metrics, "model_state": best_state,
        "model_source_sha256": sha256_file(REPO / "src/family_e1.py"),
        "preregistration_sha256": sha256_file(run / "preregistration/family_e1_nonnegative_source_signed_residual_model.md"),
    })
    epoch_path = run / f"training/{name}_epochs.csv"
    fresh_csv(epoch_path, epoch_rows)
    return {
        "model": name, "seed": seed, "training_rows": len(train_indices),
        "epochs_completed": len(epoch_rows), "selected_epoch": best_epoch,
        "checkpoint_path": relative(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_validation_requested_l1": best_metrics["requested_l1"],
        "selected_validation_companion_l1": best_metrics["companion_l1"],
        "selected_validation_flux": best_metrics["flux"],
        "selected_validation_centroid": best_metrics["centroid"],
        "selected_validation_prompt_swap": epoch_rows[best_epoch - 1]["validation_prompt_swap"],
        "selection_uses_safety_labels": False, "selection_uses_calibration": False,
        "device": "mps", "cpu_fallback": False, "runtime_seconds": time.time() - started,
    }


def stage_train(run: Path, include_folds: bool) -> None:
    validate_run(run, "micro_overfit_complete.json")
    device = require_mps()
    training = load_partition_arrays(run, "training")
    validation = load_partition_arrays(run, "validation")
    primary_rows = []
    if not include_folds:
        for seed in PRIMARY_SEEDS:
            primary_rows.append(train_one(
                run, f"primary_seed_{seed}", seed, training, validation,
                np.arange(len(np.asarray(training["model_input"]))),
                run / f"checkpoints/family_e1_seed_{seed}.pth", device,
            ))
        fresh_csv(run / "tables/primary_training_summary.csv", primary_rows)
        fresh_json(run / "logs/primary_training_complete.json", {
            "status": "PASS", "seeds": list(PRIMARY_SEEDS), "completed": len(primary_rows),
            "primary_seed": PRIMARY_SEEDS[0], "validation_only_selection": True,
            "safety_labels_created": 0, "calibration_access_count": 0, "cpu_fallback": False,
        })
    else:
        validate_run(run, "primary_training_complete.json")
        folds = pd.read_csv(run / "manifests/training_manifest.csv").oof_fold.to_numpy(dtype=np.int64)
        fold_rows = []
        for fold in range(5):
            train_indices = np.flatnonzero(folds != fold)
            fold_rows.append(train_one(
                run, f"oof_fold_{fold}", PRIMARY_SEEDS[0], training, validation,
                train_indices, run / f"checkpoints/family_e1_oof_fold_{fold}.pth", device,
            ) | {"held_fold": fold, "held_rows": int(np.sum(folds == fold))})
        fresh_csv(run / "tables/oof_fold_training_summary.csv", fold_rows)
        fresh_json(run / "logs/oof_fold_training_complete.json", {
            "status": "PASS", "folds": 5, "completed": len(fold_rows),
            "identical_architecture_objective_optimizer_budget": True,
            "fold_specific_tuning": False, "cpu_fallback": False,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "micro", "train-primary", "train-folds"), required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage == "preflight":
        stage_preflight(run)
    elif args.stage == "micro":
        stage_micro(run)
    elif args.stage == "train-primary":
        stage_train(run, include_folds=False)
    else:
        stage_train(run, include_folds=True)


if __name__ == "__main__":
    main()
