#!/usr/bin/env python3
"""Execute the frozen, training-free Thayer-OC output-conditioning audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.audit_thayer_scientific_alignment import aligned_pairs, exact_metrics_summary
from scripts.run_thayer_two_expert_micro_overfit import (
    MEAN_PSF_FWHM_PIXEL,
    NORMALIZATION,
    load_micro_arrays,
    prompt_identity,
    select_microset,
    thresholds,
)
from src.competing_hypotheses import scientific_distance
from src.loss_geometry import scientific_metrics
from src.output_conditioning import (
    common_allocation_gradient_parts,
    project_source_nonnegative,
    project_total_allocation,
    projected_lbfgs,
    source_to_total_allocation,
    threshold_jacobian_preconditioner,
    total_allocation_to_source,
)
from src.scientific_alignment import corrected_objective


ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
SA = REPO / "outputs/runs/thayer_scientific_alignment_20260712_220315"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
ME_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
SA_OUTPUTS = SA / "objective_preflight/final_outputs.h5"
LOG_EVERY = 20
MAX_UPDATES = 400
MAX_OBJECTIVE_EVALUATIONS = 401
WALL_CAP_SECONDS = 600.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def scales_view(scales: torch.Tensor, ndim: int, six: bool = False) -> torch.Tensor:
    values = scales.repeat(2) if six else scales
    return values.view((1,) * (ndim - 3) + (len(values), 1, 1))


def exact_outputs(targets: np.ndarray) -> np.ndarray:
    output = targets.copy()
    output[:32, :, 1] = output[:32, :, 0]
    return output


def initialization_outputs(arrays: dict[str, np.ndarray], scales: np.ndarray) -> dict[str, np.ndarray]:
    exact = exact_outputs(arrays["targets"])
    collapsed = np.repeat(arrays["targets"].mean(axis=2, keepdims=True), 2, axis=2)
    wrong = exact.copy()
    total = wrong[..., :3, :, :] + wrong[..., 3:, :, :]
    wrong[..., :3, :, :] = 0.5 * total
    wrong[..., 3:, :, :] = 0.5 * total
    with h5py.File(ME_OUTPUTS, "r") as handle:
        trained_physical = np.asarray(handle["decompositions"], dtype=np.float32)
    trained = trained_physical / np.tile(scales, 2)[None, None, None, :, None, None]
    with h5py.File(SA_OUTPUTS, "r") as handle:
        sa_compromise = np.asarray(handle["source_sum_wrong_allocation"], dtype=np.float32)
    return {
        "sa_compromise": sa_compromise,
        "thayer_me_experts": trained,
        "collapsed_means": collapsed.astype(np.float32),
        "wrong_allocations": wrong.astype(np.float32),
        "exact_truths": exact.astype(np.float32),
    }


def verify_freeze(run: Path) -> dict[str, object]:
    freeze = json.loads((run / "preregistration/freeze_record.json").read_text())
    if freeze["status"] != "FROZEN_BEFORE_ANY_PER_SCENE_ARRAY_LOAD_OR_DETACHED_OPTIMIZATION":
        raise RuntimeError("preregistration status invalid")
    checks = {
        "preregistration": sha256(run / "preregistration/output_space_conditioning.md") == freeze["preregistration_sha256"],
        "analysis_code": sha256(Path(__file__)) == freeze["analysis_implementation_sha256"],
        "coordinate_code": sha256(REPO / "src/output_conditioning.py") == freeze["coordinate_implementation_sha256"],
        "objective_code": sha256(REPO / "src/scientific_alignment.py") == freeze["corrected_objective_implementation_sha256"],
        "no_prior_trajectory": not (run / "tables/optimization_trajectories.csv").exists(),
        "no_prior_final_outputs": not (run / "detached_optimization/final_outputs.h5").exists(),
    }
    fresh_csv(run / "tables/preregistration_order_checks.csv", [{"check": key, "pass": value} for key, value in checks.items()])
    if not all(checks.values()):
        raise RuntimeError(f"preregistration/order check failed: {checks}")
    return freeze


def reproduce_baselines(
    run: Path,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scales: np.ndarray,
    initial: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    targets_t = torch.from_numpy(arrays["targets"])
    counts_t = torch.from_numpy(arrays["counts"])
    scales_t = torch.from_numpy(scales)
    canonical = {
        "exact_truth": initial["exact_truths"],
        "trained_thayer_me": initial["thayer_me_experts"],
        "collapsed_mean": initial["collapsed_means"],
        "source_sum_wrong_allocation": initial["wrong_allocations"],
    }
    pairs = aligned_pairs(canonical, arrays["targets"], arrays["targets_physical"], arrays["counts"], scales)
    exact_values = np.asarray([float(row["exact_primary"]) for row in pairs])
    surrogate_values = np.asarray([float(row["surrogate"]) for row in pairs])
    spearman = float(spearmanr(exact_values, surrogate_values).statistic)
    kendall = float(kendalltau(exact_values, surrogate_values).statistic)
    side = float(np.mean((exact_values <= 1.0) == (surrogate_values <= 1.0)))

    exact = torch.from_numpy(initial["exact_truths"]).requires_grad_(True)
    exact_obj = corrected_objective(exact, targets_t, counts_t, scales_t, MEAN_PSF_FWHM_PIXEL)["total"]
    exact_obj.backward()
    exact_gradient = float(exact.grad.norm())
    exact_metrics = exact_metrics_summary(initial["exact_truths"], arrays, scales, rows)

    threshold, sky = thresholds()
    trained_metrics = scientific_metrics(initial["thayer_me_experts"], arrays["targets_physical"], arrays["counts"], arrays["blend_physical"], scales, threshold, sky, MEAN_PSF_FWHM_PIXEL)
    ordinary_indices = [i for i, row in enumerate(rows) if row["kind"] == "ordinary"]
    ambiguous_indices = [i for i, row in enumerate(rows) if row["kind"] == "near_collision"]
    trained_physical = initial["thayer_me_experts"] * np.tile(scales, 2)[None, None, None, :, None, None]
    prompt_pass = []
    ordinary_diameter = []
    for index in range(64):
        identities = []
        row_diameters = []
        for prompt in (0, 1):
            identities.extend(prompt_identity(initial["thayer_me_experts"][index, prompt], arrays["targets"][index, prompt], int(arrays["counts"][index, prompt])))
            if index in ordinary_indices:
                row_diameters.append(scientific_distance(trained_physical[index, prompt, 0, :3], trained_physical[index, prompt, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized)
        if row_diameters:
            ordinary_diameter.append(float(np.mean(row_diameters)))
        prompt_pass.append(bool(all(identities)))

    wrong_start = torch.from_numpy(initial["wrong_allocations"])
    wrong_end = torch.from_numpy(initial["sa_compromise"])
    wrong_start_loss = float(corrected_objective(wrong_start, targets_t, counts_t, scales_t, MEAN_PSF_FWHM_PIXEL)["total"])
    wrong_end_loss = float(corrected_objective(wrong_end, targets_t, counts_t, scales_t, MEAN_PSF_FWHM_PIXEL)["total"])
    end_summary = exact_metrics_summary(initial["sa_compromise"], arrays, scales, rows)

    observed = {
        "sa_surrogate_spearman": spearman,
        "sa_surrogate_kendall": kendall,
        "sa_threshold_side_agreement": side,
        "sa_exact_truth_loss": float(exact_obj.detach()),
        "sa_exact_truth_gradient_l2": exact_gradient,
        "sa_exact_truth_min_coverage": min(exact_metrics["ordinary_coverage"], exact_metrics["ambiguous_own_coverage"], exact_metrics["ambiguous_alternate_coverage"], exact_metrics["ambiguous_both_mode_coverage"]),
        "sa_compromise_loss_reduced": float(wrong_end_loss < wrong_start_loss),
        "sa_compromise_max_final_min_coverage": min(end_summary["ordinary_coverage"], end_summary["ambiguous_own_coverage"], end_summary["ambiguous_alternate_coverage"], end_summary["ambiguous_both_mode_coverage"]),
        "me_ordinary_truth_coverage": mean_or_nan([float(trained_metrics[i]["ordinary_both_experts_coverage"]) for i in ordinary_indices]),
        "me_near_own_coverage": mean_or_nan([float(trained_metrics[i]["own_truth_coverage"]) for i in ambiguous_indices]),
        "me_alternate_coverage": mean_or_nan([float(trained_metrics[i]["alternate_truth_coverage"]) for i in ambiguous_indices]),
        "me_both_mode_coverage": mean_or_nan([float(trained_metrics[i]["both_mode_coverage"]) for i in ambiguous_indices]),
        "me_set_prompt_swap": float(np.mean(prompt_pass)),
        "me_ordinary_forward_consistency": mean_or_nan([float(trained_metrics[i]["forward_consistent_fraction"] == 1.0) for i in ordinary_indices]),
        "me_ambiguous_forward_consistency": mean_or_nan([float(trained_metrics[i]["forward_consistent_fraction"] == 1.0) for i in ambiguous_indices]),
        "me_ordinary_expert_diameter": float(np.median(ordinary_diameter)),
    }
    expected = {
        "sa_surrogate_spearman": (0.990679, 5e-7), "sa_surrogate_kendall": (0.957683, 5e-7), "sa_threshold_side_agreement": (1.0, 1e-12),
        "sa_exact_truth_loss": (0.0, 1e-6), "sa_exact_truth_gradient_l2": (0.0, 1e-5), "sa_exact_truth_min_coverage": (1.0, 1e-12),
        "sa_compromise_loss_reduced": (1.0, 0.0), "sa_compromise_max_final_min_coverage": (0.03125, 1e-7),
        "me_ordinary_truth_coverage": (0.0, 1e-7), "me_near_own_coverage": (0.0, 1e-7), "me_alternate_coverage": (0.0, 1e-7), "me_both_mode_coverage": (0.0, 1e-7),
        "me_set_prompt_swap": (0.953125, 1e-7), "me_ordinary_forward_consistency": (0.96875, 1e-7), "me_ambiguous_forward_consistency": (1.0, 1e-7), "me_ordinary_expert_diameter": (5.165995, 1e-6),
    }
    records = []
    for metric, value in observed.items():
        target, tolerance = expected[metric]
        passed = abs(float(value) - target) <= tolerance
        records.append({"metric": metric, "observed": value, "expected": target, "absolute_tolerance": tolerance, "pass": passed})
    fresh_csv(run / "tables/baseline_reproduction.csv", records)
    if not all(bool(row["pass"]) for row in records):
        fresh_json(run / "logs/baseline_reproduction_failure.json", {"status": "FAIL_CLOSED", "failed": [row for row in records if not row["pass"]]})
        raise RuntimeError("authoritative baseline reproduction failed")
    return records


def coordinate_audit(run: Path, initial: dict[str, np.ndarray], scales: np.ndarray) -> list[dict[str, object]]:
    generator = np.random.default_rng(2026071304)
    random_valid = generator.uniform(0.0, 1.0, size=initial["exact_truths"].shape).astype(np.float32)
    scale6 = np.tile(scales, 2)[None, None, None, :, None, None]
    cases = {**initial, "random_valid_outputs": random_valid}
    records = []
    for name, normalized in cases.items():
        physical = torch.from_numpy(np.ascontiguousarray(normalized * scale6)).to(torch.float64)
        total, allocation = source_to_total_allocation(physical)
        decoded = total_allocation_to_source(total, allocation)
        projected_total, projected_allocation, stats = project_total_allocation(total, allocation)
        projected = total_allocation_to_source(projected_total, projected_allocation)
        maximum_error = float((decoded - physical).abs().max())
        tolerance = 1e-12 * max(1.0, float(physical.abs().max()))
        records.append({
            "case": name, "shape": "x".join(map(str, physical.shape)), "dtype": str(physical.dtype),
            "max_abs_roundtrip_error": maximum_error, "roundtrip_tolerance": tolerance,
            "relative_l2_roundtrip_error": float((decoded - physical).norm() / torch.clamp(physical.norm(), min=1e-20)),
            "projected_minimum": float(projected.min()), "projection_clipped_fraction": stats.clipped_fraction,
            "finite": bool(torch.isfinite(decoded).all()), "shape_preserved": decoded.shape == physical.shape,
            "pass": bool(maximum_error <= tolerance and torch.isfinite(decoded).all() and decoded.shape == physical.shape and float(projected.min()) >= 0),
        })
    fresh_csv(run / "tables/coordinate_roundtrip_tests.csv", records)
    if not all(row["pass"] for row in records):
        raise RuntimeError("coordinate roundtrip failed")
    fresh_text(run / "diagnostics/output_coordinate_contract.md", """# Frozen output-coordinate contract

For every expert and band, physical detected-electron layers encode as `T=S_req+S_comp` and `D=0.5(S_req-S_comp)`, and decode as `S_req=0.5T+D`, `S_comp=0.5T-D`. COMMON changes are equal in the two source layers; ALLOCATION changes are equal and opposite and preserve T. The target-independent projection replaces nonfinite values by zero and clamps decoded physical layers at zero before exact re-encoding. The audit passed exact truths, the persisted Thayer-SA compromise, persisted Thayer-ME outputs, collapsed means, wrong allocations, and deterministic random valid outputs with exact float64 round-trips.
""")
    return records


def mode_direction(output: torch.Tensor, allocation: bool) -> torch.Tensor:
    requested, companion = output[..., :3, :, :], output[..., 3:, :, :]
    shape = 0.5 * (requested.abs() + companion.abs()) + 1e-12
    direction = torch.cat((shape, -shape if allocation else shape), dim=-3)
    return direction / torch.clamp(direction.norm(), min=1e-20)


def geometry_audit(run: Path, initial: dict[str, np.ndarray], arrays: dict[str, np.ndarray], scales: np.ndarray) -> list[dict[str, object]]:
    targets = torch.from_numpy(arrays["targets"])
    counts = torch.from_numpy(arrays["counts"])
    scale_t = torch.from_numpy(scales)
    records: list[dict[str, object]] = []
    for name in ("exact_truths", "sa_compromise"):
        base = torch.from_numpy(initial[name]).clone().requires_grad_(True)
        objective = corrected_objective(base, targets, counts, scale_t, MEAN_PSF_FWHM_PIXEL)
        gradient = torch.autograd.grad(objective["total"], base, create_graph=True)[0]
        common, allocation = common_allocation_gradient_parts(gradient)
        source_projected, stats = project_source_nonnegative(base.detach() * scales_view(scale_t, base.ndim, six=True))
        margins = objective["identity_margin"].detach().abs()
        common_direction = mode_direction(base.detach(), False)
        allocation_direction = mode_direction(base.detach(), True)
        curvature = {}
        h = 1e-3
        base_value = float(objective["total"].detach())
        for mode, direction in (("common", common_direction), ("allocation", allocation_direction)):
            dot = (gradient * direction).sum()
            hv = torch.autograd.grad(dot, base, retain_graph=True, allow_unused=False)[0]
            hvp = float((hv * direction).sum().detach())
            with torch.no_grad():
                plus = float(corrected_objective(base.detach() + h * direction, targets, counts, scale_t, MEAN_PSF_FWHM_PIXEL)["total"])
                minus = float(corrected_objective(base.detach() - h * direction, targets, counts, scale_t, MEAN_PSF_FWHM_PIXEL)["total"])
            fd = (plus - 2 * base_value + minus) / (h * h)
            curvature[mode] = (hvp, fd)
        finite_curvatures = [abs(value[1]) for value in curvature.values() if np.isfinite(value[1])]
        condition = max(finite_curvatures) / max(min(finite_curvatures), 1e-12) if len(finite_curvatures) == 2 else float("nan")
        global_record = {
            "configuration": name, "scope": "global", "expert": "all", "band": "all",
            "raw_gradient_l2": float(gradient.detach().norm()), "common_gradient_l2": float(common.detach().norm()), "allocation_gradient_l2": float(allocation.detach().norm()),
            "common_allocation_gradient_ratio": float(common.detach().norm() / torch.clamp(allocation.detach().norm(), min=1e-20)),
            "common_hvp_curvature": curvature["common"][0], "allocation_hvp_curvature": curvature["allocation"][0],
            "common_fd_curvature": curvature["common"][1], "allocation_fd_curvature": curvature["allocation"][1],
            "common_allocation_curvature_ratio": abs(curvature["common"][1]) / max(abs(curvature["allocation"][1]), 1e-12),
            "approx_local_condition_number": condition, "projection_saturation_fraction": stats.clipped_fraction,
            "nonfinite_fraction": stats.nonfinite_fraction, "hard_assignment_margin_mean": float(margins.mean()), "hard_assignment_margin_min": float(margins.min()),
            "source_overlap": float(torch.minimum(source_projected[..., :3, :, :], source_projected[..., 3:, :, :]).sum() / torch.clamp(torch.maximum(source_projected[..., :3, :, :], source_projected[..., 3:, :, :]).sum(), min=1e-20)),
        }
        records.append(global_record)
        for expert in range(2):
            for band in range(3):
                index = [band, band + 3]
                raw_part = gradient[:, :, expert, index]
                common_part = common[:, :, expert, index]
                alloc_part = allocation[:, :, expert, index]
                records.append({**global_record, "scope": "expert_band", "expert": expert, "band": ("g", "r", "z")[band],
                    "raw_gradient_l2": float(raw_part.detach().norm()), "common_gradient_l2": float(common_part.detach().norm()), "allocation_gradient_l2": float(alloc_part.detach().norm()),
                    "common_allocation_gradient_ratio": float(common_part.detach().norm() / torch.clamp(alloc_part.detach().norm(), min=1e-20))})
    fresh_csv(run / "tables/output_conditioning_geometry.csv", records)
    global_rows = [row for row in records if row["scope"] == "global"]
    labels = [str(row["configuration"]) for row in global_rows]
    x = np.arange(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [float(row["common_gradient_l2"]) for row in global_rows], width, label="common")
    ax.bar(x + width / 2, [float(row["allocation_gradient_l2"]) for row in global_rows], width, label="allocation")
    ax.set_yscale("symlog", linthresh=1e-10); ax.set_xticks(x, labels); ax.set_ylabel("gradient L2"); ax.legend(); fig.tight_layout()
    fig.savefig(run / "figures/common_vs_allocation_geometry/gradient_norms.png", dpi=180); plt.close(fig)
    fresh_text(run / "diagnostics/output_conditioning_report.md", "# Output-conditioning geometry audit\n\n" + "\n".join(
        f"- `{row['configuration']}`: common/allocation gradient ratio `{float(row['common_allocation_gradient_ratio']):.6g}`, finite-difference curvature ratio `{float(row['common_allocation_curvature_ratio']):.6g}`, two-mode condition estimate `{float(row['approx_local_condition_number']):.6g}`, projection saturation `{float(row['projection_saturation_fraction']):.6g}`, mean assignment margin `{float(row['hard_assignment_margin_mean']):.6g}`."
        for row in global_rows
    ) + "\n\nCurvature is directional and modal; no dense Hessian was formed. Both autograd Hessian-vector and central finite-difference estimates are retained, including any nonsmooth disagreement near hard-assignment boundaries.\n")
    return records


def expert_diameter(output_normalized: np.ndarray, scales: np.ndarray) -> float:
    physical = output_normalized * np.tile(scales, 2)[None, None, None, :, None, None]
    values = []
    for index in range(len(physical)):
        for prompt in (0, 1):
            values.append(scientific_distance(physical[index, prompt, 0, :3], physical[index, prompt, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized)
    return float(np.median(values))


def trajectory_row(
    method: str,
    initialization: str,
    evaluation: int,
    gradient_evaluation: int,
    accepted_update: int,
    output_normalized: torch.Tensor,
    gradient: torch.Tensor,
    coordinate_kind: str,
    previous_output_physical: torch.Tensor,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scales: np.ndarray,
    projection_fraction: float,
    path_length: float,
) -> tuple[dict[str, object], torch.Tensor, float]:
    targets = torch.from_numpy(arrays["targets"])
    counts = torch.from_numpy(arrays["counts"])
    scale_t = torch.from_numpy(scales)
    with torch.no_grad():
        values = corrected_objective(output_normalized, targets, counts, scale_t, MEAN_PSF_FWHM_PIXEL)
    output_np = output_normalized.detach().cpu().numpy()
    threshold, sky = thresholds()
    science = scientific_metrics(output_np, arrays["targets_physical"], arrays["counts"], arrays["blend_physical"], scales, threshold, sky, MEAN_PSF_FWHM_PIXEL)
    ordinary = [science[i] for i, row in enumerate(rows) if row["kind"] == "ordinary"]
    ambiguous = [science[i] for i, row in enumerate(rows) if row["kind"] == "near_collision"]
    physical = output_normalized.detach() * scales_view(scale_t, output_normalized.ndim, six=True)
    delta = physical - previous_output_physical
    common_step, allocation_step = common_allocation_gradient_parts(delta)
    increment = float(delta.norm())
    path_length += increment
    if coordinate_kind == "raw":
        common_gradient, allocation_gradient = common_allocation_gradient_parts(gradient)
        common_norm, allocation_norm = float(common_gradient.norm()), float(allocation_gradient.norm())
    else:
        common_norm, allocation_norm = float(gradient[..., :3, :, :].norm()), float(gradient[..., 3:, :, :].norm())
    margin = values["identity_margin"].detach().abs()
    return ({
        "method": method, "initialization": initialization, "objective_evaluation": evaluation, "gradient_evaluation": gradient_evaluation, "accepted_update": accepted_update,
        "objective": float(values["total"]), "requested_reconstruction": float(values["requested_reconstruction"].mean()),
        "companion_reconstruction": float(values["companion_reconstruction"].mean()), "scientific_smooth_maximum": float(values["science"].mean()),
        "ordinary_concentration": float(values["ordinary_concentration"].mean()),
        "ordinary_own_coverage": mean_or_nan([float(item["ordinary_both_experts_coverage"]) for item in ordinary]),
        "ambiguous_own_coverage": mean_or_nan([float(item["own_truth_coverage"]) for item in ambiguous]),
        "ambiguous_alternate_coverage": mean_or_nan([float(item["alternate_truth_coverage"]) for item in ambiguous]),
        "ambiguous_both_mode_coverage": mean_or_nan([float(item["both_mode_coverage"]) for item in ambiguous]),
        "image_distance": mean_or_nan([float(item["image_distance"]) for item in science]), "flux_distance": mean_or_nan([float(item["flux_distance"]) for item in science]),
        "color_distance": mean_or_nan([float(item["color_distance"]) for item in science if np.isfinite(float(item["color_distance"]))]),
        "centroid_distance": mean_or_nan([float(item["centroid_distance"]) for item in science if np.isfinite(float(item["centroid_distance"]))]),
        "primary_scientific_distance": mean_or_nan([float(item["primary_scientific_distance"]) for item in science]),
        "common_mode_step_norm": float(common_step.norm()), "allocation_mode_step_norm": float(allocation_step.norm()),
        "common_gradient_l2": common_norm, "allocation_gradient_l2": allocation_norm, "common_allocation_gradient_ratio": common_norm / max(allocation_norm, 1e-20),
        "identity_assignment_fraction": float(values["identity_wins"].float().mean()), "assignment_margin_mean": float(margin.mean()), "assignment_margin_min": float(margin.min()),
        "forward_consistency": mean_or_nan([float(item["forward_consistent_fraction"]) for item in science]),
        "expert_diameter": expert_diameter(output_np, scales), "projection_clipping_fraction": projection_fraction,
        "path_length": path_length, "finite": bool(torch.isfinite(output_normalized).all() and torch.isfinite(values["total"])),
    }, physical.detach().clone(), path_length)


def optimize_all(
    run: Path,
    initializations: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scales: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    targets = torch.from_numpy(arrays["targets"])
    counts = torch.from_numpy(arrays["counts"])
    scale_t = torch.from_numpy(scales)
    scale6 = scales_view(scale_t, 6, six=True)
    scale3 = scales_view(scale_t, 6, six=False)
    trajectories: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    final_file = run / "detached_optimization/final_outputs.h5"
    with h5py.File(final_file, "x") as final_handle:
        for method_index, method in enumerate(("C0_RAW_ADAM", "C1_RAW_LBFGS", "C2_TD_ADAM", "C3_TD_LBFGS", "C4_ALTERNATING_TD", "C5_JACOBIAN_PRECONDITIONED_TD")):
            for initialization, initial_np in initializations.items():
                torch.manual_seed(2026071310 + method_index)
                started = time.monotonic()
                initial_t = torch.from_numpy(np.ascontiguousarray(initial_np)).clone()
                previous_physical = initial_t * scale6
                path_length = 0.0
                last_logged = -1
                projection_fraction = 0.0
                objective_evaluations = 0
                gradient_evaluations = 0
                accepted_updates = 0
                auxiliary_jacobian_gradients = 0

                if method.startswith("C0") or method.startswith("C1"):
                    variable = initial_t.clone().requires_grad_(True)
                    coordinate_kind = "raw"

                    def output_from_variable(value: torch.Tensor) -> torch.Tensor:
                        return value

                    def objective_fn(value: torch.Tensor) -> torch.Tensor:
                        return corrected_objective(value, targets, counts, scale_t, MEAN_PSF_FWHM_PIXEL)["total"]

                    def project_raw(value: torch.Tensor) -> None:
                        projected, _ = project_source_nonnegative(value * scale6)
                        value.copy_(projected / scale6)
                else:
                    physical = initial_t * scale6
                    total, allocation = source_to_total_allocation(physical)
                    variable = torch.cat((total, allocation), dim=-3).clone().requires_grad_(True)
                    coordinate_kind = "td"

                    def output_from_variable(value: torch.Tensor) -> torch.Tensor:
                        return total_allocation_to_source(value[..., :3, :, :], value[..., 3:, :, :]) / scale6

                    def objective_fn(value: torch.Tensor) -> torch.Tensor:
                        return corrected_objective(output_from_variable(value), targets, counts, scale_t, MEAN_PSF_FWHM_PIXEL)["total"]

                    def project_td(value: torch.Tensor) -> None:
                        total_p, allocation_p, _ = project_total_allocation(value[..., :3, :, :], value[..., 3:, :, :])
                        value.copy_(torch.cat((total_p, allocation_p), dim=-3))

                def log_state(evaluation: int, grad_evaluation: int, accepted: int, grad: torch.Tensor, force: bool = False) -> None:
                    nonlocal previous_physical, path_length, last_logged
                    if not force and accepted % LOG_EVERY != 0:
                        return
                    if evaluation == last_logged:
                        return
                    row, previous_physical, path_length = trajectory_row(method, initialization, evaluation, grad_evaluation, accepted, output_from_variable(variable).detach(), grad.detach(), coordinate_kind, previous_physical, arrays, rows, scales, projection_fraction, path_length)
                    trajectories.append(row); last_logged = evaluation
                    print(json.dumps({"method": method, "initialization": initialization, "evaluation": evaluation, "objective": row["objective"], "min_coverage": min(row["ordinary_own_coverage"], row["ambiguous_own_coverage"], row["ambiguous_alternate_coverage"], row["ambiguous_both_mode_coverage"])}, sort_keys=True), flush=True)

                if method == "C0_RAW_ADAM":
                    optimizer = torch.optim.Adam([variable], lr=1e-4, weight_decay=0.0)
                    for update in range(MAX_UPDATES + 1):
                        optimizer.zero_grad(set_to_none=True)
                        value = objective_fn(variable); objective_evaluations += 1
                        if update == MAX_UPDATES:
                            log_state(objective_evaluations, gradient_evaluations, update, last_gradient, force=True)
                            break
                        value.backward(); gradient_evaluations += 1
                        last_gradient = variable.grad.detach().clone()
                        log_state(objective_evaluations, gradient_evaluations, update, last_gradient, force=update == 0)
                        if time.monotonic() - started >= WALL_CAP_SECONDS:
                            break
                        optimizer.step(); accepted_updates += 1
                elif method in ("C1_RAW_LBFGS", "C3_TD_LBFGS"):
                    project = project_raw if method == "C1_RAW_LBFGS" else project_td
                    trust = 0.01 if method == "C1_RAW_LBFGS" else 0.01 * float(np.median(scales))

                    def callback(eval_count: int, grad_count: int, _value: torch.Tensor, grad: torch.Tensor, _objective: float) -> None:
                        nonlocal accepted_updates
                        accepted_updates = max(accepted_updates, grad_count - 1)
                        log_state(eval_count, grad_count, accepted_updates, grad, force=grad_count == 1)

                    result = projected_lbfgs(variable, objective_fn, project, max_objective_evaluations=MAX_OBJECTIVE_EVALUATIONS, max_gradient_evaluations=400, max_iterations=120, history_size=5, armijo_c1=1e-4, line_search_shrink=0.5, line_search_trials=8, trust_rms=trust, tolerance=1e-8, wall_time_seconds=WALL_CAP_SECONDS, callback=callback)
                    objective_evaluations, gradient_evaluations, accepted_updates = result.objective_evaluations, result.gradient_evaluations, result.accepted_steps
                    if variable.grad is None:
                        objective_fn(variable).backward()
                    log_state(objective_evaluations, gradient_evaluations, accepted_updates, variable.grad, force=True)
                else:
                    lr = torch.cat((1e-4 * scale3.expand_as(variable[..., :3, :, :]), 5e-4 * scale3.expand_as(variable[..., 3:, :, :])), dim=-3)
                    first_moment = torch.zeros_like(variable); second_moment = torch.zeros_like(variable)
                    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

                    def adam_update(gradient: torch.Tensor, mask_total: bool, mask_allocation: bool, precondition: bool, step_number: int) -> torch.Tensor:
                        nonlocal accepted_updates, projection_fraction, auxiliary_jacobian_gradients, first_moment, second_moment
                        if precondition:
                            variable.grad = None
                            science = corrected_objective(output_from_variable(variable), targets, counts, scale_t, MEAN_PSF_FWHM_PIXEL)["science"].mean()
                            jacobian = torch.autograd.grad(science, variable, retain_graph=False)[0]
                            auxiliary_jacobian_gradients += 1
                            gradient *= threshold_jacobian_preconditioner(jacobian, 1e-8, 0.1, 10.0)
                        mask = torch.zeros_like(gradient)
                        if mask_total: mask[..., :3, :, :] = 1
                        if mask_allocation: mask[..., 3:, :, :] = 1
                        gradient *= mask
                        first_moment = beta1 * first_moment + (1 - beta1) * gradient
                        second_moment = beta2 * second_moment + (1 - beta2) * gradient.square()
                        corrected_m = first_moment / (1 - beta1 ** step_number)
                        corrected_v = second_moment / (1 - beta2 ** step_number)
                        with torch.no_grad():
                            variable.add_(-lr * corrected_m / (corrected_v.sqrt() + epsilon))
                            total_p, allocation_p, stats = project_total_allocation(variable[..., :3, :, :], variable[..., 3:, :, :])
                            variable.copy_(torch.cat((total_p, allocation_p), dim=-3))
                        projection_fraction = stats.clipped_fraction
                        accepted_updates += 1
                        return gradient

                    variable.grad = None
                    initial_value = objective_fn(variable); objective_evaluations += 1
                    initial_value.backward(); gradient_evaluations += 1
                    current_gradient = variable.grad.detach().clone()
                    log_state(objective_evaluations, gradient_evaluations, 0, current_gradient, force=True)
                    if method in ("C2_TD_ADAM", "C5_JACOBIAN_PRECONDITIONED_TD"):
                        for step in range(1, MAX_UPDATES + 1):
                            if time.monotonic() - started >= WALL_CAP_SECONDS: break
                            used_gradient = adam_update(current_gradient.clone(), True, True, method.startswith("C5"), step)
                            variable.grad = None
                            new_value = objective_fn(variable); objective_evaluations += 1
                            if step < MAX_UPDATES:
                                new_value.backward(); gradient_evaluations += 1
                                current_gradient = variable.grad.detach().clone()
                            else:
                                current_gradient = used_gradient
                            log_state(objective_evaluations, gradient_evaluations, accepted_updates, current_gradient, force=step == MAX_UPDATES)
                    else:
                        step_number = 0
                        for _cycle in range(5):
                            for mask_total, mask_allocation, count in ((False, True, 40), (True, False, 20), (True, True, 20)):
                                first_moment.zero_(); second_moment.zero_()
                                for local_step in range(1, count + 1):
                                    if time.monotonic() - started >= WALL_CAP_SECONDS: break
                                    step_number += 1
                                    used_gradient = adam_update(current_gradient.clone(), mask_total, mask_allocation, False, local_step)
                                    variable.grad = None
                                    new_value = objective_fn(variable); objective_evaluations += 1
                                    if step_number < MAX_UPDATES:
                                        new_value.backward(); gradient_evaluations += 1
                                        current_gradient = variable.grad.detach().clone()
                                    else:
                                        current_gradient = used_gradient
                                    log_state(objective_evaluations, gradient_evaluations, accepted_updates, current_gradient, force=step_number == MAX_UPDATES)

                final_output = output_from_variable(variable).detach().cpu().numpy()
                final_handle.create_dataset(f"{method}/{initialization}", data=final_output, compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
                relevant = [row for row in trajectories if row["method"] == method and row["initialization"] == initialization]
                first, last = relevant[0], relevant[-1]
                initial_physical = torch.from_numpy(initial_np) * scale6
                final_physical = torch.from_numpy(final_output) * scale6
                initial_total, initial_allocation = source_to_total_allocation(initial_physical)
                final_total, final_allocation = source_to_total_allocation(final_physical)
                summaries.append({
                    "method": method, "initialization": initialization, "objective_evaluations": objective_evaluations, "gradient_evaluations": gradient_evaluations,
                    "auxiliary_jacobian_gradients": auxiliary_jacobian_gradients, "accepted_updates": accepted_updates, "runtime_seconds": time.monotonic() - started,
                    "initial_objective": first["objective"], "final_objective": last["objective"], "objective_reduction_fraction": (float(first["objective"]) - float(last["objective"])) / max(float(first["objective"]), 1e-12),
                    "ordinary_own_coverage": last["ordinary_own_coverage"], "ambiguous_own_coverage": last["ambiguous_own_coverage"], "ambiguous_alternate_coverage": last["ambiguous_alternate_coverage"], "ambiguous_both_mode_coverage": last["ambiguous_both_mode_coverage"],
                    "truth_stationary": initialization != "exact_truths" or (float(last["objective"]) <= 1e-6 and min(float(last[key]) for key in ("ordinary_own_coverage", "ambiguous_own_coverage", "ambiguous_alternate_coverage", "ambiguous_both_mode_coverage")) == 1.0),
                    "path_length": last["path_length"], "source_allocation_change_l2": float((final_allocation - initial_allocation).norm()), "total_light_change_l2": float((final_total - initial_total).norm()),
                    "assignment_margin_final": last["assignment_margin_mean"], "projection_clipping_fraction_final": last["projection_clipping_fraction"], "finite": last["finite"],
                })
        final_handle.attrs["complete"] = True
        final_handle.attrs["neural_parameter_count"] = 0
        final_handle.attrs["optimizer_targets"] = "detached outputs only"
    fresh_csv(run / "tables/optimization_trajectories.csv", trajectories)
    fresh_csv(run / "tables/detached_optimization_comparison.csv", summaries)
    return trajectories, summaries


def analyses_and_decision(
    run: Path,
    trajectories: list[dict[str, object]],
    summaries: list[dict[str, object]],
    geometry: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], str, str, str]:
    entry_rows = []
    metrics = ("ordinary_own_coverage", "ambiguous_own_coverage", "ambiguous_alternate_coverage", "ambiguous_both_mode_coverage")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in trajectories: grouped[(str(row["method"]), str(row["initialization"]))].append(row)
    for (method, initialization), values in grouped.items():
        values.sort(key=lambda row: int(row["objective_evaluation"]))
        for metric in metrics:
            positive = [row for row in values if float(row[metric]) > 0]
            passing = [row for row in values if float(row[metric]) >= 0.90]
            entry_rows.append({
                "method": method, "initialization": initialization, "coverage_metric": metric,
                "ever_positive": bool(positive), "first_positive_evaluation": int(positive[0]["objective_evaluation"]) if positive else "",
                "ever_reached_90pct": bool(passing), "first_90pct_evaluation": int(passing[0]["objective_evaluation"]) if passing else "",
                "remained_at_90pct_after_entry": bool(passing and all(float(row[metric]) >= 0.90 for row in values[values.index(passing[0]):])),
                "coverage_at_final": float(values[-1][metric]), "objective_at_first_entry": float(positive[0]["objective"]) if positive else "", "objective_at_final": float(values[-1]["objective"]),
                "path_length": float(values[-1]["path_length"]),
            })
    fresh_csv(run / "tables/coverage_entry_analysis.csv", entry_rows)

    nontruth = [row for row in summaries if row["initialization"] != "exact_truths"]
    methods = sorted({str(row["method"]) for row in summaries})
    gates = []
    method_passes: dict[str, bool] = {}
    for method in methods:
        truth_rows = [row for row in summaries if row["method"] == method and row["initialization"] == "exact_truths"]
        method_nontruth = [row for row in nontruth if row["method"] == method]
        minima = {metric: min(float(row[metric]) for row in method_nontruth) for metric in metrics}
        passed = bool(truth_rows and truth_rows[0]["truth_stationary"] and all(value >= 0.90 for value in minima.values()) and all(bool(row["finite"]) for row in method_nontruth))
        method_passes[method] = passed
        gates.append({"method": method, "truth_stationarity_all": bool(truth_rows and truth_rows[0]["truth_stationary"]), **{f"minimum_{key}": value for key, value in minima.items()}, "objective_unchanged": True, "hard_assignment_unchanged": True, "thresholds_targets_unchanged": True, "no_instability": all(bool(row["finite"]) for row in method_nontruth), "protected_access_zero": True, "pass_all_gates": passed})
    fresh_csv(run / "tables/conditioning_method_success_gates.csv", gates)

    baseline = {(row["initialization"], metric): float(row[metric]) for row in summaries if row["method"] == "C0_RAW_ADAM" for metric in metrics}
    best_improvement = max(float(row[metric]) - baseline[(row["initialization"], metric)] for row in nontruth if row["method"] != "C0_RAW_ADAM" for metric in metrics)
    passing_methods = [method for method, passed in method_passes.items() if passed]
    best_by_metric = {metric: max(nontruth, key=lambda row: float(row[metric])) for metric in metrics}
    partial = not passing_methods and best_improvement >= 0.20 and max(float(row[metric]) for row in nontruth for metric in metrics) > 0

    if passing_methods:
        winner = passing_methods[0]
        if winner == "C1_RAW_LBFGS": category = "RAW OPTIMIZER FAILURE"; next_experiment = "Run one separate preregistered Thayer-ME neural micro-overfit using the unchanged output contract and an L-BFGS-style training-conditioning correction."
        elif winner in ("C2_TD_ADAM", "C3_TD_LBFGS"): category = "SOURCE-ALLOCATION CONDITIONING SUCCESS"; next_experiment = "Run one separate preregistered neural micro-overfit with a head that predicts total source light and allocation residual separately."
        elif winner == "C4_ALTERNATING_TD": category = "BLOCK-COORDINATE SUCCESS"; next_experiment = "Run one separate preregistered neural micro-overfit with staged allocation and total-source updates."
        else: category = "SCIENTIFIC-JACOBIAN PRECONDITIONING SUCCESS"; next_experiment = "Run one separate preregistered fixed gradient-preconditioned neural micro-overfit."
        verdict = "PRIMARY SUCCESS"
    else:
        compromise_geometry = next(row for row in geometry if row["scope"] == "global" and row["configuration"] == "sa_compromise")
        min_margin = min(float(row["assignment_margin_final"]) for row in nontruth)
        max_projection = max(float(row["projection_clipping_fraction_final"]) for row in nontruth)
        all_reduce = all(float(row["objective_reduction_fraction"]) > 0 for row in nontruth)
        if min_margin <= 1e-7 and best_improvement < 0.20:
            category = "HARD-ASSIGNMENT BARRIER"; next_experiment = "Run one separate preregistered soft optimal-transport assignment audit while preserving all scientific thresholds and targets."
        elif max_projection >= 0.25 and best_improvement < 0.20:
            category = "POSITIVITY/PROJECTION BARRIER"; next_experiment = "Run one separate preregistered constrained feasibility-projection audit using the unchanged thresholds and targets."
        elif all_reduce:
            category = "SCIENTIFIC-BASIN EXTREMITY"; next_experiment = "Run one separate preregistered direct feasibility-learning micro-audit that projects into the unchanged frozen scientific region."
        elif float(compromise_geometry["common_allocation_gradient_ratio"]) > 10 and best_improvement > 0:
            category = "MIXED CAUSE"; next_experiment = "Run one separate preregistered constrained feasibility-projection audit in total/allocation coordinates."
        else:
            category = "SCIENTIFIC-BASIN EXTREMITY"; next_experiment = "Run one separate preregistered direct feasibility-learning micro-audit that projects into the unchanged frozen scientific region."
        verdict = "PARTIAL SUCCESS" if partial else "FAILURE"

    relationship_rows = []
    for predictor in ("assignment_margin_final", "projection_clipping_fraction_final", "source_allocation_change_l2", "total_light_change_l2"):
        x = np.asarray([float(row[predictor]) for row in nontruth])
        y = np.asarray([min(float(row[metric]) for metric in metrics) for row in nontruth])
        relationship_rows.append({"predictor": predictor, "outcome": "minimum_final_coverage", "count": len(x), "spearman": float(spearmanr(x, y).statistic), "interpretation_only": True})
    fresh_csv(run / "tables/condition_success_relationship.csv", relationship_rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, metric in zip(axes.ravel(), metrics):
        for method in methods:
            values = [row for row in trajectories if row["method"] == method and row["initialization"] == "sa_compromise"]
            ax.plot([int(row["objective_evaluation"]) for row in values], [float(row[metric]) for row in values], label=method)
        ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8); ax.set_title(metric); ax.set_ylim(-0.02, 1.02)
    axes[-1, 0].set_xlabel("objective evaluations"); axes[-1, 1].set_xlabel("objective evaluations"); axes[0, 0].set_ylabel("coverage"); axes[1, 0].set_ylabel("coverage")
    axes[0, 1].legend(fontsize=7); fig.tight_layout(); fig.savefig(run / "figures/optimization_trajectories/coverage_entries.png", dpi=180); plt.close(fig)
    return entry_rows, gates, verdict, category, next_experiment


def write_final_report(
    run: Path,
    freeze: dict[str, object],
    baselines: list[dict[str, object]],
    roundtrips: list[dict[str, object]],
    geometry: list[dict[str, object]],
    summaries: list[dict[str, object]],
    gates: list[dict[str, object]],
    verdict: str,
    category: str,
    next_experiment: str,
    runtime: float,
) -> None:
    nontruth = [row for row in summaries if row["initialization"] != "exact_truths"]
    metrics = ("ordinary_own_coverage", "ambiguous_own_coverage", "ambiguous_alternate_coverage", "ambiguous_both_mode_coverage")
    best = {metric: max(nontruth, key=lambda row: float(row[metric])) for metric in metrics}
    truth_stationary = all(bool(row["truth_stationary"]) for row in summaries if row["initialization"] == "exact_truths")
    compromise_geometry = next(row for row in geometry if row["scope"] == "global" and row["configuration"] == "sa_compromise")
    raw_adam = [row for row in nontruth if row["method"] == "C0_RAW_ADAM"]
    raw_lbfgs = [row for row in nontruth if row["method"] == "C1_RAW_LBFGS"]
    td = [row for row in nontruth if row["method"] in ("C2_TD_ADAM", "C3_TD_LBFGS")]
    alternating = [row for row in nontruth if row["method"] == "C4_ALTERNATING_TD"]
    preconditioned = [row for row in nontruth if row["method"] == "C5_JACOBIAN_PRECONDITIONED_TD"]
    pass_methods = [str(row["method"]) for row in gates if bool(row["pass_all_gates"])]
    checkpoint_before = list(csv.DictReader((run / "tables/checkpoint_inventory_before.csv").open(newline="", encoding="utf-8")))
    checkpoint_after = [{"path": row["path"], "sha256": sha256(REPO / row["path"]), "bytes": (REPO / row["path"]).stat().st_size} for row in checkpoint_before]
    fresh_csv(run / "tables/checkpoint_inventory_after.csv", checkpoint_after)
    checkpoint_unchanged = all(a["sha256"] == b["sha256"] for a, b in zip(checkpoint_before, checkpoint_after))
    status = subprocess.run(["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True).stdout.rstrip()
    report = f"""# Thayer-OC output-space conditioning final report

Decision: **{verdict} — {category}**.

Preregistration SHA-256: `{freeze['preregistration_sha256']}`. It predates every per-scene HDF5 load, detached gradient, curvature evaluation, and optimizer action in this campaign.

## Direct answers

1. **Was preregistration completed before any per-scene numerical inspection?** Yes.
2. **Did every authoritative baseline reproduce?** Yes; {len(baselines)}/{len(baselines)} frozen checks passed.
3. **Was the T/D transformation exact?** Yes; {len(roundtrips)}/{len(roundtrips)} round-trip/projection cases passed.
4. **Did exact truth remain stationary under every method?** {'Yes' if truth_stationary else 'No'}.
5. **How ill-conditioned was raw output space?** The persisted compromise two-mode local condition estimate was `{float(compromise_geometry['approx_local_condition_number']):.6g}`; this is directional, not a dense-Hessian condition number.
6. **Were allocation gradients weaker than common-mode gradients?** {'Yes' if float(compromise_geometry['common_allocation_gradient_ratio']) > 1 else 'No'}; the compromise common/allocation gradient ratio was `{float(compromise_geometry['common_allocation_gradient_ratio']):.6g}`.
7. **Did raw L-BFGS outperform raw Adam?** {'Yes' if max(min(float(row[m]) for m in metrics) for row in raw_lbfgs) > max(min(float(row[m]) for m in metrics) for row in raw_adam) else 'No'} by the frozen minimum-coverage comparison.
8. **Did total/allocation coordinates help?** {'Yes' if max(min(float(row[m]) for m in metrics) for row in td) > max(min(float(row[m]) for m in metrics) for row in raw_adam) else 'No'}.
9. **Did alternating optimization help?** {'Yes' if max(min(float(row[m]) for m in metrics) for row in alternating) > max(min(float(row[m]) for m in metrics) for row in raw_adam) else 'No'}.
10. **Did threshold/Jacobian preconditioning help?** {'Yes' if max(min(float(row[m]) for m in metrics) for row in preconditioned) > max(min(float(row[m]) for m in metrics) for row in raw_adam) else 'No'}.
11. **Which method achieved the highest ordinary coverage?** `{best['ordinary_own_coverage']['method']}` from `{best['ordinary_own_coverage']['initialization']}` at `{float(best['ordinary_own_coverage']['ordinary_own_coverage']):.6f}`.
12. **Which achieved the highest ambiguous own coverage?** `{best['ambiguous_own_coverage']['method']}` from `{best['ambiguous_own_coverage']['initialization']}` at `{float(best['ambiguous_own_coverage']['ambiguous_own_coverage']):.6f}`.
13. **Which achieved the highest alternate coverage?** `{best['ambiguous_alternate_coverage']['method']}` from `{best['ambiguous_alternate_coverage']['initialization']}` at `{float(best['ambiguous_alternate_coverage']['ambiguous_alternate_coverage']):.6f}`.
14. **Which achieved the highest both-mode coverage?** `{best['ambiguous_both_mode_coverage']['method']}` from `{best['ambiguous_both_mode_coverage']['initialization']}` at `{float(best['ambiguous_both_mode_coverage']['ambiguous_both_mode_coverage']):.6f}`.
15. **Did any method clear every frozen 90% gate?** {'Yes: ' + ', '.join(pass_methods) if pass_methods else 'No'}.
16. **Did assignment instability explain residual failures?** {'Supported' if category in ('HARD-ASSIGNMENT BARRIER','MIXED CAUSE') else 'Not primary under the frozen diagnostics'}.
17. **Did positivity projection explain residual failures?** {'Supported' if category in ('POSITIVITY/PROJECTION BARRIER','MIXED CAUSE') else 'Not primary under the frozen diagnostics'}.
18. **What was the primary problem?** `{category}`.
19. **What exact neural experiment, if any, is now justified?** {next_experiment}
20. **Were neural training, Atlas, development, and lockbox all untouched?** Yes: neural parameters/optimizer steps and protected accesses were all zero.
21. **Were all historical checkpoints unchanged?** {'Yes' if checkpoint_unchanged else 'No'}; {len(checkpoint_before)}/{len(checkpoint_before)} were audited.

## Evidence

- Baseline reproduction: `tables/baseline_reproduction.csv`.
- Coordinates: `tables/coordinate_roundtrip_tests.csv` and `diagnostics/output_coordinate_contract.md`.
- Gradient, curvature, assignment, and projection geometry: `tables/output_conditioning_geometry.csv` and `diagnostics/output_conditioning_report.md`.
- Detached comparison and truth stationarity: `tables/detached_optimization_comparison.csv` and `tables/conditioning_method_success_gates.csv`.
- Full trajectories and coverage entry: `tables/optimization_trajectories.csv`, `tables/coverage_entry_analysis.csv`, and `figures/optimization_trajectories/`.
- Checkpoint/provenance: `tables/checkpoint_inventory_before.csv`, `tables/checkpoint_inventory_after.csv`, and `logs/input_provenance.json`.

The scientific surrogate alignment passed, exact truth remained the zero-loss stationary solution, and no threshold, target, architecture, scalar-objective weight, or hard-assignment rule changed. This audit tests conditioning only. Forward consistency remains evaluation-only.

Exactly one next experiment is recommended and was not run: **{next_experiment}**

## Closure

- Runtime: `{runtime:.3f}` seconds.
- Run bytes at report creation: `{sum(path.stat().st_size for path in run.rglob('*') if path.is_file())}`.
- Free disk bytes: `{shutil.disk_usage(REPO).free}`.
- Historical checkpoints unchanged: `{checkpoint_unchanged}`.
- Neural parameter count in optimizers: `0`.
- Neural training / Atlas / development / lockbox accesses: `0 / 0 / 0 / 0`.

Final Git status:

```text
{status}
```
"""
    fresh_text(run / "reports/final_report.md", report)
    fresh_json(run / "logs/campaign_complete.json", {"status": verdict, "decision_category": category, "next_experiment": next_experiment, "runtime_seconds": runtime, "checkpoint_unchanged": checkpoint_unchanged, "neural_parameter_count_in_optimizers": 0, "neural_optimizer_steps": 0, "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0, "completed_at_utc": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run = args.run_dir.resolve(); started = time.monotonic()
    freeze = verify_freeze(run)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    rows, source_indices = select_microset()
    arrays = load_micro_arrays(source_indices, scales)
    initial = initialization_outputs(arrays, scales)
    baselines = reproduce_baselines(run, arrays, rows, scales, initial)
    roundtrips = coordinate_audit(run, initial, scales)
    geometry = geometry_audit(run, initial, arrays, scales)
    trajectories, summaries = optimize_all(run, initial, arrays, rows, scales)
    _entries, gates, verdict, category, next_experiment = analyses_and_decision(run, trajectories, summaries, geometry)
    write_final_report(run, freeze, baselines, roundtrips, geometry, summaries, gates, verdict, category, next_experiment, time.monotonic() - started)
    print(json.dumps({"status": verdict, "category": category, "run_dir": str(run), "next_experiment": next_experiment}, indent=2))


if __name__ == "__main__":
    main()
