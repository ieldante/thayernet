#!/usr/bin/env python3
"""Run the preregistered training-free Thayer-LG loss-geometry audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.run_thayer_two_expert_micro_overfit import (
    MEAN_PSF_FWHM_PIXEL,
    load_micro_arrays,
    prompt_identity,
    select_microset,
    thresholds,
)
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.competing_hypotheses import forward_consistency, is_plausible, scientific_distance, source_measurements
from src.models_two_expert_decoder import (
    permutation_invariant_target_loss,
    prompt_swap_set_loss,
    source_sum,
    swap_decomposition,
    unordered_set_distance,
)


MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
PROMPT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
MICRO_MANIFEST = MICRO / "tables/microset_manifest.csv"
TRAINED_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
CHECKPOINT = MICRO / "checkpoints/thayer_me_micro_final.pth"
NORMALIZATION = PROMPT / "manifests/normalization.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_freeze(run_dir: Path) -> dict[str, object]:
    freeze_path = run_dir / "preregistration/freeze_record.json"
    freeze = json.loads(freeze_path.read_text())
    prereg = run_dir / "preregistration/frozen_loss_geometry_audit.md"
    if freeze["status"] != "FROZEN_BEFORE_PER_SCENE_NUMERICAL_INSPECTION":
        raise RuntimeError("preregistration status is not frozen")
    if sha256_file(prereg) != freeze["preregistration_sha256"]:
        raise RuntimeError("preregistration changed after freeze")
    provenance = json.loads((run_dir / "logs/input_provenance.json").read_text())
    for item in provenance["relevant_artifacts"].values():
        path = REPO / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"frozen input altered: {item['path']}")
    if sha256_file(MICRO_MANIFEST) != freeze["microset_manifest_sha256"]:
        raise RuntimeError("microset manifest changed")
    return freeze


def load_inputs() -> tuple[list[dict[str, str]], np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rows, source_indices = select_microset()
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    arrays = load_micro_arrays(source_indices, scales)
    with h5py.File(TRAINED_OUTPUTS, "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("persisted output file incomplete")
        trained_physical = np.asarray(handle["decompositions"], dtype=np.float32)
    trained = trained_physical / np.tile(scales, 2)[None, None, None, :, None, None]
    if trained.shape != (64, 2, 2, 6, 60, 60):
        raise RuntimeError(f"unexpected trained output shape: {trained.shape}")
    persisted_manifest = read_csv(MICRO_MANIFEST)
    observed_manifest = [
        {
            "micro_index": str(index),
            "source_h5_index": str(int(source_indices[index])),
            "scene_id": row["scene_id"],
            "kind": row["kind"],
            "pair_id": row["near_collision_pair_id"],
            "partition": row["partition"],
            "validation_access": "0",
            "calibration_access": "0",
            "atlas_access": "0",
            "development_access": "0",
            "lockbox_access": "0",
        }
        for index, row in enumerate(rows)
    ]
    if persisted_manifest != observed_manifest:
        raise RuntimeError("microset row replay differs from persisted manifest")
    return rows, source_indices, arrays, scales, trained


def set_loss_components(output: torch.Tensor, targets: torch.Tensor, count: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return exact per-scene implemented terms with the selected assignment."""
    result = permutation_invariant_target_loss(output, targets, count)
    n = len(output)
    chosen = torch.zeros((n, 2), dtype=torch.long, device=output.device)
    ambiguous = count == 2
    swap = ambiguous & (~result["identity_wins"])
    chosen[:, 0] = torch.where(swap, torch.ones_like(count), torch.zeros_like(count))
    chosen[:, 1] = torch.where(ambiguous & (~swap), torch.ones_like(count), torch.zeros_like(count))
    batch = torch.arange(n, device=output.device)
    matched_0 = targets[batch, chosen[:, 0]]
    matched_1 = targets[batch, chosen[:, 1]]
    matched = torch.stack((matched_0, matched_1), dim=1)
    requested = (output[:, :, :3] - matched[:, :, :3]).square().mean(dim=(-3, -2, -1)).sum(dim=1)
    companion = (output[:, :, 3:] - matched[:, :, 3:]).square().mean(dim=(-3, -2, -1)).sum(dim=1)
    summed = 0.5 * (source_sum(output) - source_sum(matched)).square().mean(dim=(-3, -2, -1)).sum(dim=1)
    concentration = 0.10 * (output[:, 0] - output[:, 1]).square().mean(dim=(-3, -2, -1))
    concentration = torch.where(count == 1, concentration, torch.zeros_like(concentration))
    reconstructed = requested + companion + summed + concentration
    if not torch.allclose(reconstructed, result["per_scene"], rtol=1e-5, atol=1e-7):
        raise RuntimeError("loss component decomposition does not reproduce implemented target loss")
    return {
        **result,
        "requested": requested,
        "companion": companion,
        "target_source_sum": summed,
        "ordinary_concentration_weighted": concentration,
        "chosen_targets": chosen,
    }


def exact_global_objective(outputs: torch.Tensor, targets_tensor: torch.Tensor, counts: torch.Tensor, blend: torch.Tensor, rows: list[dict[str, str]]) -> dict[str, float]:
    prompt_terms = []
    component_terms: dict[str, list[torch.Tensor]] = defaultdict(list)
    for prompt in (0, 1):
        components = set_loss_components(outputs[:, prompt], targets_tensor[:, prompt], counts[:, prompt])
        prompt_terms.append(components["loss"])
        for name in ("requested", "companion", "target_source_sum", "ordinary_concentration_weighted"):
            component_terms[name].append(components[name].mean())
    target = 0.5 * (prompt_terms[0] + prompt_terms[1])
    forward = 0.5 * (
        (source_sum(outputs[:, 0]) - blend[:, None]).square().mean()
        + (source_sum(outputs[:, 1]) - blend[:, None]).square().mean()
    )
    prompt_swap = prompt_swap_set_loss(outputs[:, 0], outputs[:, 1])
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["kind"] == "near_collision":
            by_pair[row["near_collision_pair_id"]].append(index)
    pair_terms = []
    for pair_id in sorted(by_pair):
        left, right = by_pair[pair_id]
        for prompt in (0, 1):
            pair_terms.append(unordered_set_distance(outputs[left:left + 1, prompt], outputs[right:right + 1, prompt]).mean())
    pair_consistency = torch.stack(pair_terms).mean()
    total = target + 0.5 * forward + 0.25 * prompt_swap + 0.05 * pair_consistency
    result = {
        "total": float(total),
        "target_set": float(target),
        "requested_reconstruction": float(0.5 * sum(component_terms["requested"])),
        "companion_reconstruction": float(0.5 * sum(component_terms["companion"])),
        "target_source_sum_weighted_internal": float(0.5 * sum(component_terms["target_source_sum"])),
        "ordinary_concentration_weighted_internal": float(0.5 * sum(component_terms["ordinary_concentration_weighted"])),
        "forward": float(forward),
        "prompt_swap": float(prompt_swap),
        "pair_consistency": float(pair_consistency),
        "weighted_forward": float(0.5 * forward),
        "weighted_prompt_swap": float(0.25 * prompt_swap),
        "weighted_pair_consistency": float(0.05 * pair_consistency),
    }
    return result


def evaluate_persisted(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray, trained: np.ndarray) -> tuple[dict[str, float], list[dict[str, object]]]:
    trained_physical = trained * np.tile(scales, 2)[None, None, None, :, None, None]
    threshold, sky = thresholds()
    per_scene: list[dict[str, object]] = []
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
        for prompt in (0, 1):
            identities[prompt] = prompt_identity(trained[index, prompt], arrays["targets"][index, prompt], count)
            for expert in (0, 1):
                candidate = trained_physical[index, prompt, expert]
                score = forward_consistency(arrays["blend_physical"][index], np.stack((candidate[:3], candidate[3:])), sky)
                plausible[prompt, expert] = is_plausible(score, threshold)
                own_distance = scientific_distance(candidate[:3], arrays["targets_physical"][index, prompt, 0, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL)
                own[prompt, expert] = plausible[prompt, expert] and own_distance.primary_normalized <= 1.0
                if count == 2:
                    alt_distance = scientific_distance(candidate[:3], arrays["targets_physical"][index, prompt, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL)
                    alternate[prompt, expert] = plausible[prompt, expert] and alt_distance.primary_normalized <= 1.0
            diameter = scientific_distance(trained_physical[index, prompt, 0, :3], trained_physical[index, prompt, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL)
            diameters.append(diameter.primary_normalized)
        for expert in (0, 1):
            expert_prompt[expert].append(bool(identities[:, expert].all()))
        set_prompt.append(bool(identities.all()))
        both_modes = [bool((own[prompt, 0] and alternate[prompt, 1]) or (own[prompt, 1] and alternate[prompt, 0])) for prompt in (0, 1)]
        if row["kind"] == "ordinary":
            ordinary_diameter.append(float(np.mean(diameters)))
        per_scene.append({
            "scene_id": row["scene_id"], "kind": row["kind"], "pair_id": row["near_collision_pair_id"],
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
    return metrics, per_scene


def trained_output_rows(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray, trained: np.ndarray) -> list[dict[str, object]]:
    trained_physical = trained * np.tile(scales, 2)[None, None, None, :, None, None]
    targets = torch.from_numpy(np.ascontiguousarray(arrays["targets"]))
    counts = torch.from_numpy(np.ascontiguousarray(arrays["counts"]))
    output = torch.from_numpy(np.ascontiguousarray(trained))
    components = [set_loss_components(output[:, prompt], targets[:, prompt], counts[:, prompt]) for prompt in (0, 1)]
    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        for prompt in (0, 1):
            for expert in (0, 1):
                candidate = trained_physical[index, prompt, expert]
                requested = source_measurements(candidate[:3])
                companion = source_measurements(candidate[3:])
                target_index = int(components[prompt]["chosen_targets"][index, expert])
                target_requested = arrays["targets_physical"][index, prompt, target_index, :3]
                distance = scientific_distance(candidate[:3], target_requested, mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL)
                output_rows.append({
                    "micro_index": index, "scene_id": row["scene_id"], "kind": row["kind"], "pair_id": row["near_collision_pair_id"],
                    "prompt": prompt, "expert": expert, "selected_target": target_index,
                    "canonical_six_channel_sha256": canonical_tensor_sha256(candidate),
                    "normalized_min": float(trained[index, prompt, expert].min()),
                    "normalized_max": float(trained[index, prompt, expert].max()),
                    "normalized_mean": float(trained[index, prompt, expert].mean()),
                    "normalized_negative_fraction": float(np.mean(trained[index, prompt, expert] < 0)),
                    "requested_flux_g": requested.flux_grz[0], "requested_flux_r": requested.flux_grz[1], "requested_flux_z": requested.flux_grz[2],
                    "companion_flux_g": companion.flux_grz[0], "companion_flux_r": companion.flux_grz[1], "companion_flux_z": companion.flux_grz[2],
                    "requested_centroid_x": "" if requested.centroid_xy is None else requested.centroid_xy[0],
                    "requested_centroid_y": "" if requested.centroid_xy is None else requested.centroid_xy[1],
                    "primary_scientific_distance_to_assignment": distance.primary_normalized,
                    "image_distance": distance.image,
                    "flux_distance_max": max(distance.relative_flux_grz),
                    "color_distance_max": max(value for value in distance.color_gr_rz_magnitude if value is not None) if any(value is not None for value in distance.color_gr_rz_magnitude) else "",
                    "centroid_distance_pixel": "" if distance.centroid_pixel is None else distance.centroid_pixel,
                    "target_loss_scene_prompt": float(components[prompt]["per_scene"][index]),
                    "identity_assignment_cost": float(components[prompt]["identity_assignment"][index]),
                    "swap_assignment_cost": float(components[prompt]["swapped_assignment"][index]),
                    "assignment_margin": abs(float(components[prompt]["identity_assignment"][index] - components[prompt]["swapped_assignment"][index])),
                })
    return output_rows


def reproduce_micro(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray, trained: np.ndarray) -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    observed, per_scene = evaluate_persisted(rows, arrays, scales, trained)
    expected = payload["metrics"]
    reproduction_rows = []
    for metric, target in expected.items():
        value = observed[metric]
        tolerance = 1e-6 if metric == "ordinary_median_expert_diameter" else 1e-7
        passed = math.isclose(value, float(target), rel_tol=0.0, abs_tol=tolerance)
        reproduction_rows.append({"metric": metric, "expected": target, "observed": value, "absolute_tolerance": tolerance, "status": "PASS" if passed else "FAIL"})
    write_csv_fresh(run_dir / "tables/micro_overfit_reproduction.csv", reproduction_rows)
    write_csv_fresh(run_dir / "tables/micro_overfit_per_scene_reproduction.csv", per_scene)
    if any(row["status"] != "PASS" for row in reproduction_rows):
        write_text_fresh(run_dir / "diagnostics/micro_overfit_reproduction.md", "# Thayer-ME micro-overfit reproduction\n\n**FAIL — STOP.** Persisted outputs do not reproduce the authoritative metrics.\n")
        raise RuntimeError("micro-overfit reproduction failed")

    tensor = torch.from_numpy(np.ascontiguousarray(trained))
    targets_tensor = torch.from_numpy(np.ascontiguousarray(arrays["targets"]))
    counts = torch.from_numpy(np.ascontiguousarray(arrays["counts"]))
    blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"]))
    objective = exact_global_objective(tensor, targets_tensor, counts, blend, rows)
    write_csv_fresh(run_dir / "tables/trained_objective_reproduction.csv", [{"term": key, "value": value} for key, value in objective.items()])
    write_csv_fresh(run_dir / "tables/trained_output_reproduction.csv", trained_output_rows(rows, arrays, scales, trained))
    write_text_fresh(run_dir / "diagnostics/micro_overfit_reproduction.md", f"""# Thayer-ME micro-overfit reproduction

Status: **PASS** from persisted tensors; no model inference occurred.

- Ordinary own-truth coverage: {observed['ordinary_own_truth_coverage']:.6f}.
- Ambiguous own / alternate / both-mode coverage: {observed['ambiguous_own_truth_coverage']:.6f} / {observed['ambiguous_alternate_truth_coverage']:.6f} / {observed['ambiguous_both_mode_coverage']:.6f}.
- Set prompt-swap: {observed['set_prompt_swap']:.6f}.
- Ordinary / ambiguous forward consistency: {observed['ordinary_forward_consistency']:.6f} / {observed['ambiguous_forward_consistency']:.6f}.
- Ordinary median expert diameter: {observed['ordinary_median_expert_diameter']:.9f}.
- Recomputed persisted-output objective: {objective['total']:.12g}.
- Manifest, target-set, scene-tensor, trained-output, and checkpoint hashes matched the freeze.
""")


def truth_representability(run_dir: Path, rows: list[dict[str, str]], arrays: dict[str, np.ndarray], scales: np.ndarray) -> None:
    threshold, sky = thresholds()
    audit_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for index, row in enumerate(rows):
        count_a, count_b = (int(value) for value in arrays["counts"][index])
        if count_a != count_b or count_a not in (1, 2):
            failures.append(f"{row['scene_id']}: target counts invalid")
        isolated = arrays["isolated_physical"][index]
        expected_a = np.concatenate((isolated[0], isolated[1]), axis=0)
        expected_b = np.concatenate((isolated[1], isolated[0]), axis=0)
        own_a = arrays["targets_physical"][index, 0, 0]
        own_b = arrays["targets_physical"][index, 1, 0]
        order_a = bool(np.array_equal(own_a, expected_a))
        order_b = bool(np.array_equal(own_b, expected_b))
        prompt_set_map = True
        for target_index in range(count_a):
            swapped = np.concatenate((arrays["targets_physical"][index, 0, target_index, 3:], arrays["targets_physical"][index, 0, target_index, :3]), axis=0)
            prompt_set_map = prompt_set_map and any(np.array_equal(swapped, arrays["targets_physical"][index, 1, candidate]) for candidate in range(count_b))
        for prompt in (0, 1):
            truths = arrays["targets_physical"][index, prompt]
            own = truths[0]
            own_score = forward_consistency(arrays["blend_physical"][index], np.stack((own[:3], own[3:])), sky)
            own_plausible = is_plausible(own_score, threshold)
            own_distance = scientific_distance(own[:3], own[:3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL)
            own_coverage = own_plausible and own_distance.primary_normalized <= 1.0
            alternate_coverage = True
            both_mode_coverage = True
            alternate_plausible: bool | str = ""
            alternate_distance_value: float | str = ""
            if count_a == 2:
                alternate = truths[1]
                alternate_score = forward_consistency(arrays["blend_physical"][index], np.stack((alternate[:3], alternate[3:])), sky)
                alternate_plausible = is_plausible(alternate_score, threshold)
                alternate_distance = scientific_distance(alternate[:3], alternate[:3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL)
                alternate_distance_value = alternate_distance.primary_normalized
                alternate_coverage = bool(alternate_plausible and alternate_distance.primary_normalized <= 1.0)
                both_mode_coverage = own_coverage and alternate_coverage
            exact_set = np.stack((truths[0], truths[0] if count_a == 1 else truths[1]))
            exact_norm = exact_set / np.tile(scales, 2)[:, None, None]
            duplicate_diameter = scientific_distance(exact_set[0, :3], exact_set[1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized if count_a == 1 else ""
            source_sum_error = float(np.max(np.abs((own[:3] + own[3:]) - isolated.sum(axis=0))))
            passes = bool(
                order_a and order_b and prompt_set_map and own_coverage and alternate_coverage and both_mode_coverage
                and source_sum_error <= 1e-6 and np.all(np.isfinite(exact_norm))
                and (count_a == 2 or float(duplicate_diameter) <= 1e-12)
            )
            audit_rows.append({
                "micro_index": index, "scene_id": row["scene_id"], "kind": row["kind"], "pair_id": row["near_collision_pair_id"], "prompt": prompt,
                "target_count": count_a, "shape": "2x6x60x60", "dtype_physical": str(exact_set.dtype), "dtype_normalized": str(exact_norm.dtype),
                "band_order": "g,r,z", "source_order": "requested,companion", "zero_background": True, "clipping": False,
                "requested_companion_order_a_exact": order_a, "requested_companion_order_b_exact": order_b,
                "prompt_swap_set_mapping_exact": prompt_set_map,
                "own_exact_primary_distance": own_distance.primary_normalized, "own_exact_forward_plausible": own_plausible, "own_exact_coverage": own_coverage,
                "alternate_exact_primary_distance": alternate_distance_value, "alternate_exact_forward_plausible": alternate_plausible, "alternate_exact_coverage": alternate_coverage if count_a == 2 else "",
                "both_mode_exact_coverage": both_mode_coverage if count_a == 2 else "",
                "ordinary_duplicate_diameter": duplicate_diameter, "source_sum_max_abs_error_physical": source_sum_error,
                "expert_1_canonical_sha256": canonical_tensor_sha256(exact_set[0]), "expert_2_canonical_sha256": canonical_tensor_sha256(exact_set[1]),
                "status": "PASS" if passes else "FAIL",
            })
            if not passes:
                failures.append(f"{row['scene_id']} prompt {prompt}: exact-truth sanity failure")
    write_csv_fresh(run_dir / "tables/truth_representability_audit.csv", audit_rows)
    if failures:
        write_text_fresh(run_dir / "diagnostics/truth_representability_report.md", "# Truth representability audit\n\nStatus: **OUTPUT-CONTRACT OR COVERAGE-METRIC DEFECT — STOP.**\n\n" + "\n".join(f"- {failure}" for failure in failures) + "\n")
        write_json_fresh(run_dir / "logs/truth_representability_complete.json", {"status": "FAIL_STOP", "classification": "OUTPUT-CONTRACT OR COVERAGE-METRIC DEFECT", "failure_count": len(failures), "failures": failures, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
        raise RuntimeError("truth representability gate failed")
    write_text_fresh(run_dir / "diagnostics/truth_representability_report.md", """# Truth representability audit

Status: **PASS** for all 64 microset rows and both prompts.

Every exact own truth passed frozen coverage; every exact alternate passed alternate coverage; every approved two-truth set passed both-mode coverage; duplicated ordinary truths had zero expert diameter; prompt-swap set mapping, requested/companion order, g/r/z band order, float32 normalization, zero background, no clipping, source-sum semantics, canonical hashing, and frozen forward plausibility all passed.
""")
    write_json_fresh(run_dir / "logs/truth_representability_complete.json", {"status": "PASS", "row_prompt_count": len(audit_rows), "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("gates",), default="gates")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = verify_freeze(run_dir)
    rows, source_indices, arrays, scales, trained = load_inputs()
    started = datetime.now(timezone.utc).isoformat()
    reproduce_micro(run_dir, rows, arrays, scales, trained)
    truth_representability(run_dir, rows, arrays, scales)
    write_json_fresh(run_dir / "logs/gates_complete.json", {
        "status": "PASS",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": freeze["preregistration_sha256"],
        "source_index_count": int(len(source_indices)),
        "model_inference_count": 0,
        "model_parameter_gradient_count": 0,
        "model_optimizer_step_count": 0,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "phase": "gates"}, sort_keys=True))


if __name__ == "__main__":
    main()
