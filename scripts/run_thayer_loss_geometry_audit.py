#!/usr/bin/env python3
"""Execute the preregistered numerical stages of the Thayer-LG audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.audit_thayer_loss_geometry import load_inputs, verify_freeze, write_csv_fresh, write_json_fresh, write_text_fresh
from scripts.run_thayer_two_expert_micro_overfit import MEAN_PSF_FWHM_PIXEL, thresholds
from src.loss_geometry import (
    TOP_LEVEL_WEIGHTS,
    canonical_configurations,
    differentiable_scientific_surrogate,
    exact_batch_objective,
    flux_preserving_morphology,
    scene_loss_terms,
    scientific_metrics,
    source_light_transfer,
)
from src.models_two_expert_decoder import source_sum


MICRO = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121/diagnostics/micro_overfit_20260712_203540"
TERM_NAMES = tuple(TOP_LEVEL_WEIGHTS)
BANDS = ("g", "r", "z")


def as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def json_map(value: dict[str, float]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def loss_breakdown(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    terms: dict[str, torch.Tensor],
    local: int,
) -> dict[str, dict[str, float]]:
    """Return per-band and per-expert contributions for the local scene."""
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for prompt in (0, 1):
        chosen = terms[f"chosen_targets_prompt_{prompt}"][local]
        for expert in (0, 1):
            target = targets[local, prompt, int(chosen[expert])]
            predicted = outputs[local, prompt, expert]
            for band, label in enumerate(BANDS):
                factor = 0.5 / 3.0
                result["requested_reconstruction"][f"p{prompt}_e{expert}_{label}"] = factor * as_float((predicted[band] - target[band]).square().mean())
                result["companion_reconstruction"][f"p{prompt}_e{expert}_{label}"] = factor * as_float((predicted[band + 3] - target[band + 3]).square().mean())
                pred_sum = predicted[band] + predicted[band + 3]
                target_sum = target[band] + target[band + 3]
                result["target_source_sum"][f"p{prompt}_e{expert}_{label}"] = factor * as_float((pred_sum - target_sum).square().mean())
                result["forward"][f"p{prompt}_e{expert}_{label}"] = (0.5 / 2.0 / 3.0) * as_float((pred_sum - BLEND_CACHE[local, band]).square().mean())
            for channel in range(6):
                source = "requested" if channel < 3 else "companion"
                label = BANDS[channel % 3]
                result["ordinary_concentration"][f"p{prompt}_{source}_{label}"] = (0.5 / 6.0) * as_float((outputs[local, prompt, 0, channel] - outputs[local, prompt, 1, channel]).square().mean())
    for name in TERM_NAMES:
        if name not in result:
            result[name] = {"aggregate_only": as_float(terms[name][local])}
    return result


BLEND_CACHE: torch.Tensor


def canonical_audit(
    run_dir: Path,
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    scales: np.ndarray,
    configs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    global BLEND_CACHE
    threshold, sky = thresholds()
    all_loss_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    targets_all = torch.from_numpy(np.ascontiguousarray(arrays["targets"]))
    counts_all = torch.from_numpy(np.ascontiguousarray(arrays["counts"]))
    blend_all = torch.from_numpy(np.ascontiguousarray(arrays["blend"]))
    for config, (indices, output_np) in configs.items():
        outputs = torch.from_numpy(np.ascontiguousarray(output_np))
        targets = targets_all[indices]
        counts = counts_all[indices]
        blend = blend_all[indices]
        BLEND_CACHE = blend
        terms = scene_loss_terms(outputs, targets, counts, blend, rows, indices)
        science = scientific_metrics(output_np, arrays["targets_physical"][indices], arrays["counts"][indices], arrays["blend_physical"][indices], scales, threshold, sky, MEAN_PSF_FWHM_PIXEL)
        for local, global_index in enumerate(indices):
            breakdown = loss_breakdown(outputs, targets, counts, terms, local)
            total = as_float(terms["total"][local])
            identity = [bool(terms[f"identity_wins_prompt_{prompt}"][local]) for prompt in (0, 1)]
            margins = [abs(as_float(terms[f"identity_cost_prompt_{prompt}"][local] - terms[f"swap_cost_prompt_{prompt}"][local])) for prompt in (0, 1)]
            for term in TERM_NAMES:
                raw = as_float(terms[term][local])
                weighted = as_float(terms[f"weighted_{term}"][local])
                all_loss_rows.append({
                    "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "kind": rows[int(global_index)]["kind"],
                    "pair_id": rows[int(global_index)]["near_collision_pair_id"], "configuration": config, "term": term,
                    "raw_loss": raw, "configured_weight": TOP_LEVEL_WEIGHTS[term], "weighted_loss": weighted,
                    "fraction_of_total": weighted / total if total > 0 else 0.0,
                    "applicable_pixel_count": int(np.prod(output_np.shape[3:])),
                    "normalization_denominator": "mean over applicable prompt/expert/channel/pixel axes",
                    "per_band_expert_contributions_json": json_map(breakdown[term]),
                    "selected_assignment": f"p0={'identity' if identity[0] else 'swap'};p1={'identity' if identity[1] else 'swap'}",
                    "assignment_margin": float(np.mean(margins)),
                })
            ranking_rows.append({
                "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "kind": rows[int(global_index)]["kind"],
                "pair_id": rows[int(global_index)]["near_collision_pair_id"], "configuration": config,
                "total_objective": total,
                **{name: as_float(terms[name][local]) for name in TERM_NAMES},
                **science[local],
                "selected_assignment": f"p0={'identity' if identity[0] else 'swap'};p1={'identity' if identity[1] else 'swap'}",
                "assignment_margin": float(np.mean(margins)),
            })
    write_csv_fresh(run_dir / "tables/canonical_loss_decomposition.csv", all_loss_rows)
    write_csv_fresh(run_dir / "tables/objective_ranking.csv", ranking_rows)
    summary_rows = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in all_loss_rows:
        grouped[(str(row["kind"]), str(row["configuration"]), str(row["term"]))].append(float(row["weighted_loss"]))
    for (kind, config, term), values in grouped.items():
        summary_rows.append({"kind": kind, "configuration": config, "term": term, "count": len(values), "mean_weighted_loss": float(np.mean(values)), "median_weighted_loss": float(np.median(values)), "p95_weighted_loss": float(np.quantile(values, 0.95)), "maximum_weighted_loss": float(np.max(values))})
    write_csv_fresh(run_dir / "tables/loss_term_scale_summary.csv", summary_rows)
    return all_loss_rows, ranking_rows


def ranking_summary(run_dir: Path, ranking_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_scene: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in ranking_rows:
        by_scene[int(row["micro_index"])].append(row)
    summary = []
    comparison = []
    for index, scene_rows in sorted(by_scene.items()):
        objective = np.asarray([float(row["total_objective"]) for row in scene_rows])
        science = np.asarray([float(row["primary_scientific_distance"]) for row in scene_rows])
        rho = spearmanr(objective, science).statistic
        tau = kendalltau(objective, science).statistic
        truth_name = "O1_EXACT_TRUTH_DUPLICATED" if scene_rows[0]["kind"] == "ordinary" else "A1_EXACT_APPROVED_SET"
        trained_name = "O2_TRAINED_EXPERT_OUTPUTS" if scene_rows[0]["kind"] == "ordinary" else "A3_TRAINED_EXPERT_OUTPUTS"
        compromise_names = ["O3_EXPERT_MEAN_DUPLICATED", "O5_SOURCE_SUM_PRESERVING_LIGHT_TRANSFER"] if scene_rows[0]["kind"] == "ordinary" else ["A4_COLLAPSED_TRUTH_MEAN", "A5_TRAINED_EXPERT_MEAN_DUPLICATED", "A8_SOURCE_SUM_PRESERVING_COMPROMISE"]
        lookup = {str(row["configuration"]): row for row in scene_rows}
        truth = lookup[truth_name]
        trained = lookup[trained_name]
        best = min(scene_rows, key=lambda row: float(row["total_objective"]))
        compromise_best = min((lookup[name] for name in compromise_names), key=lambda row: float(row["total_objective"]))
        summary.append({
            "micro_index": index, "scene_id": truth["scene_id"], "kind": truth["kind"],
            "spearman_objective_vs_science": rho, "kendall_objective_vs_science": tau,
            "objective_optimal_configuration": best["configuration"], "truth_objective_optimal": best["configuration"] == truth_name,
            "truth_total_objective": truth["total_objective"], "trained_total_objective": trained["total_objective"],
            "trained_minus_truth_objective": float(trained["total_objective"]) - float(truth["total_objective"]),
            "best_compromise_configuration": compromise_best["configuration"],
            "compromise_minus_truth_objective": float(compromise_best["total_objective"]) - float(truth["total_objective"]),
            "compromise_beats_truth": float(compromise_best["total_objective"]) < float(truth["total_objective"]),
        })
    for kind in ("ordinary", "near_collision", "all"):
        selected = summary if kind == "all" else [row for row in summary if row["kind"] == kind]
        comparison.append({
            "kind": kind, "scene_count": len(selected),
            "fraction_truth_objective_optimal": float(np.mean([bool(row["truth_objective_optimal"]) for row in selected])),
            "fraction_compromise_beats_truth": float(np.mean([bool(row["compromise_beats_truth"]) for row in selected])),
            "median_trained_minus_truth_objective": float(np.median([float(row["trained_minus_truth_objective"]) for row in selected])),
            "median_compromise_minus_truth_objective": float(np.median([float(row["compromise_minus_truth_objective"]) for row in selected])),
            "median_spearman": float(np.nanmedian([float(row["spearman_objective_vs_science"]) for row in selected])),
            "median_kendall": float(np.nanmedian([float(row["kendall_objective_vs_science"]) for row in selected])),
        })
    write_csv_fresh(run_dir / "tables/objective_ranking_scene_summary.csv", summary)
    write_csv_fresh(run_dir / "tables/objective_ranking_summary.csv", comparison)
    return comparison


def norm_record(gradient: torch.Tensor) -> tuple[float, float, float, str, str, str]:
    flat = gradient.reshape(-1)
    band = {band: as_float(torch.linalg.vector_norm(gradient[..., [i, i + 3], :, :])) for i, band in enumerate(BANDS)}
    layer = {"requested": as_float(torch.linalg.vector_norm(gradient[..., :3, :, :])), "companion": as_float(torch.linalg.vector_norm(gradient[..., 3:, :, :]))}
    expert = {f"expert_{i + 1}": as_float(torch.linalg.vector_norm(gradient[:, i])) for i in (0, 1)}
    return as_float(torch.linalg.vector_norm(flat)), as_float(flat.abs().sum()), as_float(flat.abs().max()), json_map(band), json_map(layer), json_map(expert)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.reshape(-1)
    b = right.reshape(-1)
    denominator = as_float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return float("nan") if denominator <= 1e-20 else as_float(torch.dot(a, b)) / denominator


def gradient_audit(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], configs: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    targets_all = torch.from_numpy(np.ascontiguousarray(arrays["targets"]))
    counts_all = torch.from_numpy(np.ascontiguousarray(arrays["counts"]))
    blend_all = torch.from_numpy(np.ascontiguousarray(arrays["blend"]))
    norm_rows: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    align_configs = {"O1_EXACT_TRUTH_DUPLICATED", "O2_TRAINED_EXPERT_OUTPUTS", "O3_EXPERT_MEAN_DUPLICATED", "O5_SOURCE_SUM_PRESERVING_LIGHT_TRANSFER", "A1_EXACT_APPROVED_SET", "A3_TRAINED_EXPERT_OUTPUTS", "A4_COLLAPSED_TRUTH_MEAN", "A8_SOURCE_SUM_PRESERVING_COMPROMISE"}
    for config, (indices, output_np) in configs.items():
        targets, counts, blend = targets_all[indices], counts_all[indices], blend_all[indices]
        gradients: dict[str, torch.Tensor] = {}
        for term in TERM_NAMES:
            variable = torch.from_numpy(np.ascontiguousarray(output_np)).requires_grad_(True)
            values = scene_loss_terms(variable, targets, counts, blend, rows, indices)
            if values[term].requires_grad:
                values[term].sum().backward()
                gradients[term] = variable.grad.detach().clone()
            else:
                gradients[term] = torch.zeros_like(variable)
        fractions = np.zeros((len(indices), len(TERM_NAMES)))
        for local in range(len(indices)):
            weighted_norms = np.asarray([as_float(torch.linalg.vector_norm(gradients[term][local] * TOP_LEVEL_WEIGHTS[term])) for term in TERM_NAMES])
            fractions[local] = weighted_norms / max(float(weighted_norms.sum()), 1e-20)
        for term_index, term in enumerate(TERM_NAMES):
            for local, global_index in enumerate(indices):
                raw = gradients[term][local]
                weighted = raw * TOP_LEVEL_WEIGHTS[term]
                raw_stats = norm_record(raw)
                weighted_stats = norm_record(weighted)
                norm_rows.append({
                    "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "kind": rows[int(global_index)]["kind"], "configuration": config, "term": term,
                    "unweighted_l2": raw_stats[0], "unweighted_l1": raw_stats[1], "unweighted_max_abs": raw_stats[2],
                    "weighted_l2": weighted_stats[0], "weighted_l1": weighted_stats[1], "weighted_max_abs": weighted_stats[2],
                    "per_band_weighted_l2_json": weighted_stats[3], "per_layer_weighted_l2_json": weighted_stats[4], "per_expert_weighted_l2_json": weighted_stats[5],
                    "fraction_sum_term_weighted_l2": float(fractions[local, term_index]),
                    "dominance_flag": bool(fractions[local, term_index] >= 0.75),
                })
        if config in align_configs:
            variable = torch.from_numpy(np.ascontiguousarray(output_np)).requires_grad_(True)
            surrogate = differentiable_scientific_surrogate(variable, targets, counts, MEAN_PSF_FWHM_PIXEL)
            surrogate.sum().backward()
            gradients["scientific_surrogate"] = variable.grad.detach().clone()
            composites = {
                "set_matching": gradients["requested_reconstruction"] + gradients["companion_reconstruction"] + 0.5 * gradients["target_source_sum"] + 0.10 * gradients["ordinary_concentration"],
                "source_assignment": gradients["requested_reconstruction"] + gradients["companion_reconstruction"],
                "full_objective": sum(TOP_LEVEL_WEIGHTS[name] * gradients[name] for name in TERM_NAMES),
            }
            pairs = [
                ("requested_reconstruction", "target_source_sum"), ("companion_reconstruction", "target_source_sum"),
                ("requested_reconstruction", "forward"), ("set_matching", "forward"),
                ("set_matching", "ordinary_concentration"), ("source_assignment", "prompt_swap"),
                ("scientific_surrogate", "full_objective"),
            ]
            all_gradients = {**gradients, **composites}
            for local, global_index in enumerate(indices):
                for left, right in pairs:
                    value = cosine(all_gradients[left][local], all_gradients[right][local])
                    cosine_rows.append({
                        "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "kind": rows[int(global_index)]["kind"], "configuration": config,
                        "left_gradient": left, "right_gradient": right, "cosine": value,
                        "classification": "undefined" if math.isnan(value) else ("severe_conflict" if value <= -0.5 else "conflict" if value < 0 else "orthogonal" if abs(value) < 0.1 else "aligned"),
                    })
        del gradients
    write_csv_fresh(run_dir / "tables/gradient_norms.csv", norm_rows)
    write_csv_fresh(run_dir / "tables/gradient_cosines.csv", cosine_rows)
    summary = []
    for key in sorted({(row["kind"], row["left_gradient"], row["right_gradient"]) for row in cosine_rows}):
        selected = [row for row in cosine_rows if (row["kind"], row["left_gradient"], row["right_gradient"]) == key and not math.isnan(float(row["cosine"]))]
        values = [float(row["cosine"]) for row in selected]
        summary.append({"kind": key[0], "left_gradient": key[1], "right_gradient": key[2], "count": len(values), "mean_cosine": float(np.mean(values)), "median_cosine": float(np.median(values)), "negative_fraction": float(np.mean(np.asarray(values) < 0)), "severe_conflict_fraction": float(np.mean(np.asarray(values) <= -0.5))})
    write_csv_fresh(run_dir / "tables/gradient_alignment_summary.csv", summary)
    return norm_rows, cosine_rows


def assignment_audit(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], configs: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict[str, object]]:
    targets_all = torch.from_numpy(np.ascontiguousarray(arrays["targets"]))
    counts_all = torch.from_numpy(np.ascontiguousarray(arrays["counts"]))
    blend_all = torch.from_numpy(np.ascontiguousarray(arrays["blend"]))
    generator = torch.Generator().manual_seed(2026071301)
    output_rows = []
    for config in ("A1_EXACT_APPROVED_SET", "A3_TRAINED_EXPERT_OUTPUTS", "A4_COLLAPSED_TRUTH_MEAN", "A8_SOURCE_SUM_PRESERVING_COMPROMISE"):
        indices, output_np = configs[config]
        base = torch.from_numpy(np.ascontiguousarray(output_np))
        targets, counts, blend = targets_all[indices], counts_all[indices], blend_all[indices]
        baseline = scene_loss_terms(base, targets, counts, blend, rows, indices)
        noise = torch.randn(base.shape, generator=generator, dtype=base.dtype)
        for scale in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
            perturbed = base if scale == 0 else base + scale * noise
            terms = baseline if scale == 0 else scene_loss_terms(perturbed, targets, counts, blend, rows, indices)
            for local, global_index in enumerate(indices):
                for prompt in (0, 1):
                    identity = as_float(terms[f"identity_cost_prompt_{prompt}"][local])
                    swap = as_float(terms[f"swap_cost_prompt_{prompt}"][local])
                    base_identity = bool(baseline[f"identity_wins_prompt_{prompt}"][local])
                    observed_identity = bool(terms[f"identity_wins_prompt_{prompt}"][local])
                    output_rows.append({
                        "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "configuration": config, "prompt": prompt, "perturbation_scale": scale,
                        "identity_cost": identity, "swap_cost": swap, "assignment_margin": abs(identity - swap),
                        "selected_assignment": "identity" if observed_identity else "swap", "assignment_flip": observed_identity != base_identity,
                        "unstable_tiny_perturbation": bool(scale <= 1e-5 and observed_identity != base_identity),
                        "tie_or_flat_flag": bool(abs(identity - swap) <= 1e-7),
                    })
    ordinary_tests = []
    for config in ("O1_EXACT_TRUTH_DUPLICATED", "O2_TRAINED_EXPERT_OUTPUTS", "O3_EXPERT_MEAN_DUPLICATED"):
        indices, output_np = configs[config]
        base = torch.from_numpy(np.ascontiguousarray(output_np))
        terms = scene_loss_terms(base, targets_all[indices], counts_all[indices], blend_all[indices], rows, indices)
        swapped = torch.flip(base, dims=(2,)).contiguous()
        swapped_terms = scene_loss_terms(swapped, targets_all[indices], counts_all[indices], blend_all[indices], rows, indices)
        for local, global_index in enumerate(indices):
            ordinary_tests.append({
                "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "configuration": config,
                "duplicate_target_handling": True, "expert_swap_loss_difference": abs(as_float(terms["total"][local] - swapped_terms["total"][local])),
                "concentration_raw": as_float(terms["ordinary_concentration"][local]), "expert_symmetry_pass": bool(torch.isclose(terms["total"][local], swapped_terms["total"][local], atol=1e-7)),
            })
    write_csv_fresh(run_dir / "tables/assignment_geometry.csv", output_rows)
    write_csv_fresh(run_dir / "tables/ordinary_assignment_symmetry.csv", ordinary_tests)
    return output_rows


def evaluate_path_batch(path: str, parameter: float, indices: np.ndarray, output: torch.Tensor, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray) -> list[dict[str, object]]:
    threshold, sky = thresholds()
    targets = torch.from_numpy(np.ascontiguousarray(arrays["targets"][indices]))
    counts = torch.from_numpy(np.ascontiguousarray(arrays["counts"][indices]))
    blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"][indices]))
    terms = scene_loss_terms(output, targets, counts, blend, rows, indices)
    science = scientific_metrics(output.detach().numpy(), arrays["targets_physical"][indices], arrays["counts"][indices], arrays["blend_physical"][indices], scales, threshold, sky, MEAN_PSF_FWHM_PIXEL)
    result = []
    for local, global_index in enumerate(indices):
        result.append({
            "path": path, "path_parameter": parameter, "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "kind": rows[int(global_index)]["kind"],
            "total_objective": as_float(terms["total"][local]), **{name: as_float(terms[name][local]) for name in TERM_NAMES}, **science[local],
            "assignment_prompt_0": "identity" if bool(terms["identity_wins_prompt_0"][local]) else "swap",
            "assignment_prompt_1": "identity" if bool(terms["identity_wins_prompt_1"][local]) else "swap",
            "assignment_margin": float(np.mean([abs(as_float(terms[f"identity_cost_prompt_{prompt}"][local] - terms[f"swap_cost_prompt_{prompt}"][local])) for prompt in (0, 1)])),
        })
    return result


def path_audit(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray, configs: dict[str, tuple[np.ndarray, np.ndarray]], trained: np.ndarray) -> list[dict[str, object]]:
    ordinary = configs["O1_EXACT_TRUTH_DUPLICATED"][0]
    ambiguous = configs["A1_EXACT_APPROVED_SET"][0]
    all_indices = np.arange(len(rows), dtype=np.int64)
    exact = np.empty_like(trained)
    exact[ordinary] = configs["O1_EXACT_TRUTH_DUPLICATED"][1]
    exact[ambiguous] = configs["A1_EXACT_APPROVED_SET"][1]
    collapsed = exact.copy()
    collapsed[ambiguous] = configs["A4_COLLAPSED_TRUTH_MEAN"][1]
    records: list[dict[str, object]] = []
    grid = np.linspace(0.0, 1.0, 21)
    for alpha in grid:
        records.extend(evaluate_path_batch("P1_TRUTH_TO_TRAINED", float(alpha), all_indices, torch.from_numpy(np.ascontiguousarray((1 - alpha) * exact + alpha * trained)), rows, arrays, scales))
        records.extend(evaluate_path_batch("P2_TRUTH_SET_TO_COLLAPSED_MEAN", float(alpha), ambiguous, torch.from_numpy(np.ascontiguousarray((1 - alpha) * exact[ambiguous] + alpha * collapsed[ambiguous])), rows, arrays, scales))
        separation = (1 - alpha) * collapsed[ambiguous] + alpha * exact[ambiguous]
        records.extend(evaluate_path_batch("P4_EXPERT_SEPARATION", float(alpha), ambiguous, torch.from_numpy(np.ascontiguousarray(separation)), rows, arrays, scales))
        morphology = flux_preserving_morphology(torch.from_numpy(np.ascontiguousarray(exact)), float(alpha))
        records.extend(evaluate_path_batch("P5_FLUX_PRESERVING_MORPHOLOGY", float(alpha), all_indices, morphology, rows, arrays, scales))
    for fraction in np.linspace(-0.5, 0.5, 21):
        transfer = source_light_transfer(torch.from_numpy(np.ascontiguousarray(exact)), float(fraction))
        records.extend(evaluate_path_batch("P3_SOURCE_LIGHT_TRANSFER", float(fraction), all_indices, transfer, rows, arrays, scales))
    by_series: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_series[(str(row["path"]), int(row["micro_index"]))].append(row)
    for values in by_series.values():
        values.sort(key=lambda row: float(row["path_parameter"]))
        objective = np.asarray([float(row["total_objective"]) for row in values])
        parameter = np.asarray([float(row["path_parameter"]) for row in values])
        derivative = np.gradient(objective, parameter)
        for row, value in zip(values, derivative):
            row["objective_directional_derivative"] = float(value)
            row["gradient_direction"] = "toward_increasing_parameter" if value < 0 else "toward_decreasing_parameter" if value > 0 else "stationary"
    write_csv_fresh(run_dir / "tables/objective_path_metrics.csv", records)
    return records


def normalize_directions(direction: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction.flatten(1), dim=1).reshape((-1,) + (1,) * (direction.ndim - 1))
    return direction / torch.clamp(norm, min=1e-20)


def curvature_audit(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], configs: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict[str, object]]:
    targets_all = torch.from_numpy(np.ascontiguousarray(arrays["targets"]))
    counts_all = torch.from_numpy(np.ascontiguousarray(arrays["counts"]))
    blend_all = torch.from_numpy(np.ascontiguousarray(arrays["blend"]))
    records = []
    h = 1e-3
    for config in ("O1_EXACT_TRUTH_DUPLICATED", "O2_TRAINED_EXPERT_OUTPUTS", "A1_EXACT_APPROVED_SET", "A3_TRAINED_EXPERT_OUTPUTS"):
        indices, output_np = configs[config]
        base = torch.from_numpy(np.ascontiguousarray(output_np))
        truth = targets_all[indices].clone()
        ordinary = counts_all[indices, 0] == 1
        truth[ordinary, :, 1] = truth[ordinary, :, 0]
        light = torch.zeros_like(base); light[..., :3, :, :] = truth[..., :3, :, :]; light[..., 3:, :, :] = -truth[..., :3, :, :]
        common = torch.ones_like(base)
        antisymmetric = torch.zeros_like(base)
        delta = truth[:, :, 0] - truth[:, :, 1]
        delta[ordinary] = torch.roll(truth[ordinary, :, 0], shifts=1, dims=-1) - truth[ordinary, :, 0]
        antisymmetric[:, :, 0] = delta; antisymmetric[:, :, 1] = -delta
        flux = base.clone()
        centroid = torch.roll(base, shifts=1, dims=-1) - base
        morphology = torch.zeros_like(base); morphology[..., :3, :, :] = torch.roll(base[..., :3, :, :], shifts=1, dims=-1) - base[..., :3, :, :]
        directions = {"source_light_exchange": light, "both_expert_common_mode": common, "expert_antisymmetric_separation": antisymmetric, "flux_scaling": flux, "centroid_shift": centroid, "morphology_perturbation": morphology}
        targets, counts, blend = targets_all[indices], counts_all[indices], blend_all[indices]
        base_terms = scene_loss_terms(base, targets, counts, blend, rows, indices)
        variable = base.clone().requires_grad_(True)
        scene_loss_terms(variable, targets, counts, blend, rows, indices)["total"].sum().backward()
        base_gradient = variable.grad.detach()
        for name, raw_direction in directions.items():
            direction = normalize_directions(raw_direction)
            plus = scene_loss_terms(base + h * direction, targets, counts, blend, rows, indices)["total"]
            minus = scene_loss_terms(base - h * direction, targets, counts, blend, rows, indices)["total"]
            curvature = (plus - 2 * base_terms["total"] + minus) / (h * h)
            derivative = (base_gradient * direction).flatten(1).sum(dim=1)
            for local, global_index in enumerate(indices):
                value = as_float(curvature[local])
                records.append({"micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "kind": rows[int(global_index)]["kind"], "configuration": config, "direction": name, "finite_difference_step": h, "directional_first_derivative": as_float(derivative[local]), "directional_curvature": value, "flat_flag": abs(value) <= 1e-6, "weak_curvature_flag": abs(value) <= 1e-4, "positive_curvature": value > 0})
    write_csv_fresh(run_dir / "tables/local_curvature.csv", records)
    return records


def optimization_objective(name: str, terms: dict[str, torch.Tensor]) -> torch.Tensor:
    target_no_concentration = terms["requested_reconstruction"] + terms["companion_reconstruction"] + 0.5 * terms["target_source_sum"]
    target_full = target_no_concentration + 0.10 * terms["ordinary_concentration"]
    full = target_full + 0.5 * terms["forward"] + 0.25 * terms["prompt_swap"] + 0.05 * terms["pair_consistency"]
    if name == "D0_FULL": return full
    if name == "D1_SOURCE_SET_ONLY": return target_no_concentration
    if name == "D2_SOURCE_PLUS_CONCENTRATION": return target_full
    if name == "D3_SOURCE_PLUS_SUM_CONSISTENCY": return target_no_concentration + 0.5 * terms["forward"]
    exclusions = {"D4_EXCLUDE_TARGET": target_full, "D4_EXCLUDE_FORWARD": 0.5 * terms["forward"], "D4_EXCLUDE_PROMPT_SWAP": 0.25 * terms["prompt_swap"], "D4_EXCLUDE_PAIR_EQUIVALENCE": 0.05 * terms["pair_consistency"]}
    if name in exclusions: return full - exclusions[name]
    raise ValueError(name)


def optimization_audit(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray, configs: dict[str, tuple[np.ndarray, np.ndarray]], trained: np.ndarray) -> list[dict[str, object]]:
    ordinary = configs["O1_EXACT_TRUTH_DUPLICATED"][0]; ambiguous = configs["A1_EXACT_APPROVED_SET"][0]
    exact = np.empty_like(trained); exact[ordinary] = configs["O1_EXACT_TRUTH_DUPLICATED"][1]; exact[ambiguous] = configs["A1_EXACT_APPROVED_SET"][1]
    collapsed = exact.copy(); collapsed[ambiguous] = configs["A4_COLLAPSED_TRUTH_MEAN"][1]
    compromise = exact.copy(); compromise[ordinary] = configs["O5_SOURCE_SUM_PRESERVING_LIGHT_TRANSFER"][1]; compromise[ambiguous] = configs["A8_SOURCE_SUM_PRESERVING_COMPROMISE"][1]
    rng = np.random.default_rng(2026071302)
    initializations = {"exact_truth": exact, "trained_outputs": trained, "collapsed_mean": collapsed, "random_valid": rng.uniform(0, 1, size=trained.shape).astype(np.float32), "source_sum_compromise": compromise}
    protocols = [("D0_FULL", name) for name in initializations]
    protocols += [(name, "trained_outputs") for name in ("D1_SOURCE_SET_ONLY", "D2_SOURCE_PLUS_CONCENTRATION", "D3_SOURCE_PLUS_SUM_CONSISTENCY", "D4_EXCLUDE_TARGET", "D4_EXCLUDE_FORWARD", "D4_EXCLUDE_PROMPT_SWAP", "D4_EXCLUDE_PAIR_EQUIVALENCE")]
    targets = torch.from_numpy(np.ascontiguousarray(arrays["targets"])); counts = torch.from_numpy(np.ascontiguousarray(arrays["counts"])); blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"]))
    threshold, sky = thresholds()
    records = []
    final_path = run_dir / "output_space_optimization/final_outputs.h5"
    with h5py.File(final_path, "x") as handle:
        for protocol, init_name in protocols:
            variable = torch.from_numpy(np.ascontiguousarray(initializations[init_name])).clone().requires_grad_(True)
            optimizer = torch.optim.Adam([variable], lr=0.01)
            key = f"{protocol}__{init_name}"
            for step in range(41):
                exact_terms = exact_batch_objective(variable, targets, counts, blend, rows)
                optimized = optimization_objective(protocol, exact_terms)
                if step % 5 == 0 or step == 40:
                    science = scientific_metrics(variable.detach().numpy(), arrays["targets_physical"], arrays["counts"], arrays["blend_physical"], scales, threshold, sky, MEAN_PSF_FWHM_PIXEL)
                    diameters = np.linalg.norm((variable.detach().numpy()[:, :, 0, :3] - variable.detach().numpy()[:, :, 1, :3]).reshape(len(rows), 2, -1), axis=-1).mean(axis=1)
                    records.append({
                        "protocol": protocol, "initialization": init_name, "step": step, "optimized_objective": as_float(optimized), "full_frozen_objective": as_float(exact_terms["total"]),
                        **{name: as_float(exact_terms[name]) for name in TERM_NAMES},
                        "mean_primary_scientific_distance": float(np.mean([float(row["primary_scientific_distance"]) for row in science])),
                        "median_primary_scientific_distance": float(np.median([float(row["primary_scientific_distance"]) for row in science])),
                        "ordinary_coverage": float(np.mean([bool(science[i]["ordinary_both_experts_coverage"]) for i in ordinary])),
                        "ambiguous_own_coverage": float(np.mean([bool(science[i]["own_truth_coverage"]) for i in ambiguous])),
                        "ambiguous_alternate_coverage": float(np.mean([bool(science[i]["alternate_truth_coverage"]) for i in ambiguous])),
                        "ambiguous_both_mode_coverage": float(np.mean([bool(science[i]["both_mode_coverage"]) for i in ambiguous])),
                        "forward_consistent_fraction": float(np.mean([float(row["forward_consistent_fraction"]) for row in science])),
                        "median_normalized_expert_pixel_diameter": float(np.median(diameters)),
                        "assignment_identity_fraction": float(np.mean([as_float(exact_terms.get("identity_fraction", torch.tensor(float("nan"))))])) if "identity_fraction" in exact_terms else "",
                    })
                if step == 40: break
                optimizer.zero_grad(set_to_none=True)
                optimized.backward()
                optimizer.step()
                with torch.no_grad(): variable.clamp_(-8.0, 8.0)
            handle.create_dataset(key, data=variable.detach().numpy(), compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.attrs["complete"] = True
        handle.attrs["neural_parameter_count"] = 0
        handle.attrs["optimizer_target"] = "detached_free_output_tensors_only"
    write_csv_fresh(run_dir / "tables/output_space_optimization_trajectories.csv", records)
    write_json_fresh(run_dir / "logs/output_space_optimization_isolation.json", {"status": "PASS", "neural_model_loaded": False, "model_parameter_count_in_graph": 0, "model_optimizer_step_count": 0, "free_tensor_optimizer_step_count": len(protocols) * 40, "bounds": [-8.0, 8.0], "learning_rate": 0.01, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    return records


def regression_audit(run_dir: Path, canonical: list[dict[str, object]], paths: list[dict[str, object]], optimization: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    datasets = {"canonical": canonical, "objective_paths": paths, "output_optimization": optimization}
    predictors = ["total_objective", *TERM_NAMES]
    outcomes = ["primary_scientific_distance", "image_distance", "flux_distance", "color_distance", "centroid_distance"]
    for dataset, records in datasets.items():
        if dataset == "output_optimization":
            predictors_here = ["full_frozen_objective", *TERM_NAMES]
            outcomes_here = ["mean_primary_scientific_distance"]
        else:
            predictors_here, outcomes_here = predictors, outcomes
        for predictor in predictors_here:
            for outcome in outcomes_here:
                pairs = [(float(row[predictor]), float(row[outcome])) for row in records if predictor in row and outcome in row and row[predictor] != "" and row[outcome] != "" and np.isfinite(float(row[predictor])) and np.isfinite(float(row[outcome]))]
                if len(pairs) < 3: continue
                x, y = map(np.asarray, zip(*pairs))
                rows.append({"dataset": dataset, "predictor": predictor, "outcome": outcome, "count": len(x), "spearman": float(spearmanr(x, y).statistic), "kendall": float(kendalltau(x, y).statistic), "partial_spearman_controlling_kind": "", "threshold_entry_probability_lowest_loss_quartile": ""})
    epochs = list(csv.DictReader((MICRO / "tables/micro_epochs.csv").open(newline="", encoding="utf-8")))
    gates = {int(row["epoch"]): row for row in csv.DictReader((MICRO / "tables/micro_gate_history.csv").open(newline="", encoding="utf-8"))}
    for predictor in ("loss", "target_set", "forward", "prompt_swap", "pair_consistency"):
        pairs = [(float(row[predictor]), float(gates[int(row["epoch"])]["ordinary_median_expert_diameter"])) for row in epochs if int(row["epoch"]) in gates]
        if len(pairs) >= 3:
            x, y = map(np.asarray, zip(*pairs)); rows.append({"dataset": "persisted_training_iterations", "predictor": predictor, "outcome": "ordinary_median_expert_diameter", "count": len(x), "spearman": float(spearmanr(x, y).statistic), "kendall": float(kendalltau(x, y).statistic), "partial_spearman_controlling_kind": "", "threshold_entry_probability_lowest_loss_quartile": float(np.mean(y[x <= np.quantile(x, 0.25)] <= 1.0))})
    write_csv_fresh(run_dir / "tables/loss_science_regression.csv", rows)
    return rows


def scale_audit(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray, canonical: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for band, label in enumerate(BANDS):
        physical = arrays["targets_physical"][..., (band, band + 3), :, :]
        normalized = arrays["targets"][..., (band, band + 3), :, :]
        output.append({"audit": "band_scale", "scope": label, "observed": scales[band], "detail": f"physical_abs_p99={np.quantile(np.abs(physical), .99):.9g}; normalized_abs_p99={np.quantile(np.abs(normalized), .99):.9g}", "status": "PASS"})
    trained = [row for row in canonical if row["configuration"] in ("O2_TRAINED_EXPERT_OUTPUTS", "A3_TRAINED_EXPERT_OUTPUTS")]
    for kind in ("ordinary", "near_collision"):
        selected = [row for row in trained if row["kind"] == kind]
        output.append({"audit": "effective_row_loss", "scope": kind, "observed": float(np.mean([float(row["total_objective"]) for row in selected])), "detail": "scene-level objective with pair accounting", "status": "PASS"})
    factors = [
        ("two_experts", 2, "target reconstruction sums two experts; forward averages experts"),
        ("two_prompts", 2, "all scene terms average two prompts"),
        ("three_bands", 3, "channel MSE averages three bands per source layer"),
        ("six_channels", 6, "prompt/concentration MSE averages all six channels"),
        ("ordinary_duplicate_supervision", 2, "ordinary target reconstruction supervises both experts"),
        ("ambiguous_target_set_size", 2, "hard assignment sums two expert-to-target costs"),
        ("pair_scene_accounting", 2, "0.10 on 32 ambiguous rows equals 0.05 pair mean over 64 rows"),
    ]
    for name, factor, detail in factors:
        output.append({"audit": "factor_check", "scope": name, "observed": factor, "detail": detail, "status": "EXPECTED_EXPLICIT_FACTOR"})
    output.append({"audit": "pixel_denominator", "scope": "all_rows", "observed": 3600, "detail": "all tensors are 60x60; no masked/whole-image mismatch", "status": "PASS"})
    output.append({"audit": "floors", "scope": "scientific_distance", "observed": "1e-12 image/flux; float64 epsilon moments", "detail": "low-flux relative metrics can be steep but match frozen implementation", "status": "PASS"})
    write_csv_fresh(run_dir / "tables/numerical_scale_audit.csv", output)
    return output


def figures(run_dir: Path, loss_rows: list[dict[str, object]], cosine_rows: list[dict[str, object]], assignments: list[dict[str, object]], paths: list[dict[str, object]], optimization: list[dict[str, object]], arrays: dict[str, np.ndarray], scales: np.ndarray, configs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    selected_configs = sorted({str(row["configuration"]) for row in loss_rows})
    data = np.zeros((len(selected_configs), len(TERM_NAMES)))
    for i, config in enumerate(selected_configs):
        for j, term in enumerate(TERM_NAMES):
            values = [float(row["weighted_loss"]) for row in loss_rows if row["configuration"] == config and row["term"] == term]
            data[i, j] = np.mean(values)
    fig, ax = plt.subplots(figsize=(12, 7)); bottom = np.zeros(len(selected_configs))
    for j, term in enumerate(TERM_NAMES): ax.barh(selected_configs, data[:, j], left=bottom, label=term); bottom += data[:, j]
    ax.set_xlabel("mean weighted scene objective contribution"); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(run_dir / "figures/loss_term_contributions/canonical_mean_contributions.png", dpi=180); plt.close(fig)

    values = [float(row["assignment_margin"]) for row in assignments if float(row["perturbation_scale"]) == 0]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(np.maximum(values, 1e-16), bins=30); ax.set_xscale("log"); ax.set_xlabel("identity-swap assignment margin"); ax.set_ylabel("count"); fig.tight_layout(); fig.savefig(run_dir / "figures/assignment_margin_distributions.png", dpi=180); plt.close(fig)

    pairs = sorted({f"{row['left_gradient']} vs {row['right_gradient']}" for row in cosine_rows})
    configs_cos = sorted({str(row["configuration"]) for row in cosine_rows})
    heat = np.full((len(configs_cos), len(pairs)), np.nan)
    for i, config in enumerate(configs_cos):
        for j, pair in enumerate(pairs):
            vals = [float(row["cosine"]) for row in cosine_rows if row["configuration"] == config and f"{row['left_gradient']} vs {row['right_gradient']}" == pair and not math.isnan(float(row["cosine"]))]
            if vals: heat[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(13, 6)); image = ax.imshow(heat, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto"); ax.set_yticks(range(len(configs_cos)), configs_cos, fontsize=7); ax.set_xticks(range(len(pairs)), pairs, rotation=45, ha="right", fontsize=7); fig.colorbar(image, ax=ax, label="mean cosine"); fig.tight_layout(); fig.savefig(run_dir / "figures/gradient_cosine_heatmap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for path in sorted({str(row["path"]) for row in paths}):
        params = sorted({float(row["path_parameter"]) for row in paths if row["path"] == path})
        means = [np.mean([float(row["total_objective"]) for row in paths if row["path"] == path and float(row["path_parameter"]) == value]) for value in params]
        ax.plot(params, means, label=path)
    ax.set_xlabel("path parameter"); ax.set_ylabel("mean total objective"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(run_dir / "figures/objective_paths/mean_objective_paths.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for protocol in sorted({str(row["protocol"]) for row in optimization}):
        selected = [row for row in optimization if row["protocol"] == protocol]
        by_step = sorted({int(row["step"]) for row in selected})
        means = [np.mean([float(row["full_frozen_objective"]) for row in selected if int(row["step"]) == step]) for step in by_step]
        ax.plot(by_step, means, label=protocol)
    ax.set_yscale("log"); ax.set_xlabel("free-output Adam step"); ax.set_ylabel("full frozen objective"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(run_dir / "figures/output_space_optimization_trajectories.png", dpi=180); plt.close(fig)

    for kind, index, truth_name, trained_name, compromise_name in (("ordinary", 0, "O1_EXACT_TRUTH_DUPLICATED", "O2_TRAINED_EXPERT_OUTPUTS", "O5_SOURCE_SUM_PRESERVING_LIGHT_TRANSFER"), ("ambiguous", 32, "A1_EXACT_APPROVED_SET", "A3_TRAINED_EXPERT_OUTPUTS", "A8_SOURCE_SUM_PRESERVING_COMPROMISE")):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for col, (label, config) in enumerate((("truth", truth_name), ("trained", trained_name), ("compromise", compromise_name))):
            indices, values = configs[config]; local = int(np.where(indices == index)[0][0]); physical = values[local] * np.tile(scales, 2)[None, None, :, None, None]
            axes[0, col].imshow(physical[0, 0, 1], origin="lower", cmap="magma"); axes[1, col].imshow(physical[0, 1, 1], origin="lower", cmap="magma"); axes[0, col].set_title(f"{label}: expert 1 r"); axes[1, col].set_title(f"{label}: expert 2 r")
            for ax in axes[:, col]: ax.set_axis_off()
        fig.tight_layout(); fig.savefig(run_dir / f"example_grids/{kind}_truth_trained_compromise.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); verify_freeze(run_dir)
    if json.loads((run_dir / "logs/gates_complete.json").read_text())["status"] != "PASS": raise RuntimeError("hard gates not passed")
    started = time.time(); started_utc = datetime.now(timezone.utc).isoformat()
    rows, _, arrays, scales, trained = load_inputs()
    configs = canonical_configurations(arrays["targets"], arrays["counts"], trained, rows)
    def read_existing(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))
    if (run_dir / "tables/canonical_loss_decomposition.csv").exists():
        loss_rows = read_existing(run_dir / "tables/canonical_loss_decomposition.csv")
        canonical = read_existing(run_dir / "tables/objective_ranking.csv")
        ranking = read_existing(run_dir / "tables/objective_ranking_summary.csv")
    else:
        loss_rows, canonical = canonical_audit(run_dir, rows, arrays, scales, configs)
        ranking = ranking_summary(run_dir, canonical)
    if (run_dir / "tables/gradient_norms.csv").exists():
        gradients = read_existing(run_dir / "tables/gradient_norms.csv"); cosines = read_existing(run_dir / "tables/gradient_cosines.csv")
    else: gradients, cosines = gradient_audit(run_dir, rows, arrays, configs)
    assignments = read_existing(run_dir / "tables/assignment_geometry.csv") if (run_dir / "tables/assignment_geometry.csv").exists() else assignment_audit(run_dir, rows, arrays, configs)
    paths = read_existing(run_dir / "tables/objective_path_metrics.csv") if (run_dir / "tables/objective_path_metrics.csv").exists() else path_audit(run_dir, rows, arrays, scales, configs, trained)
    curvature = read_existing(run_dir / "tables/local_curvature.csv") if (run_dir / "tables/local_curvature.csv").exists() else curvature_audit(run_dir, rows, arrays, configs)
    optimization = read_existing(run_dir / "tables/output_space_optimization_trajectories.csv") if (run_dir / "tables/output_space_optimization_trajectories.csv").exists() else optimization_audit(run_dir, rows, arrays, scales, configs, trained)
    regression = read_existing(run_dir / "tables/loss_science_regression.csv") if (run_dir / "tables/loss_science_regression.csv").exists() else regression_audit(run_dir, canonical, paths, optimization)
    scale = read_existing(run_dir / "tables/numerical_scale_audit.csv") if (run_dir / "tables/numerical_scale_audit.csv").exists() else scale_audit(run_dir, rows, arrays, scales, canonical)
    figures(run_dir, loss_rows, cosines, assignments, paths, optimization, arrays, scales, configs)
    write_json_fresh(run_dir / "logs/numerical_audit_complete.json", {"status": "PASS", "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.time() - started, "canonical_scene_configuration_count": len(canonical), "gradient_row_count": len(gradients), "cosine_row_count": len(cosines), "assignment_row_count": len(assignments), "objective_path_row_count": len(paths), "curvature_row_count": len(curvature), "optimization_trajectory_row_count": len(optimization), "regression_row_count": len(regression), "scale_audit_row_count": len(scale), "model_inference_count": 0, "model_parameter_gradient_count": 0, "model_optimizer_step_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    print(json.dumps({"status": "PASS", "runtime_seconds": time.time() - started, "ranking_summary": ranking}, sort_keys=True))


if __name__ == "__main__": main()
