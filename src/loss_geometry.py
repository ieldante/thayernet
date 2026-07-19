"""Training-free utilities for the frozen Thayer-ME output-loss audit."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch

from src.competing_hypotheses import forward_consistency, is_plausible, scientific_distance
from src.models_two_expert_decoder import source_sum, swap_decomposition, unordered_set_distance


TOP_LEVEL_WEIGHTS = {
    "requested_reconstruction": 1.0,
    "companion_reconstruction": 1.0,
    "target_source_sum": 0.5,
    "ordinary_concentration": 0.10,
    "forward": 0.5,
    "prompt_swap": 0.25,
    "pair_consistency": 0.10,
}


def canonical_configurations(
    targets: np.ndarray,
    counts: np.ndarray,
    trained: np.ndarray,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Construct all preregistered canonical normalized output configurations."""

    if targets.shape != trained.shape or targets.shape[:3] != (len(rows), 2, 2):
        raise ValueError("targets/trained must be equal N x prompt x expert tensors")
    ordinary = np.asarray([i for i, row in enumerate(rows) if row["kind"] == "ordinary"], dtype=np.int64)
    ambiguous = np.asarray([i for i, row in enumerate(rows) if row["kind"] == "near_collision"], dtype=np.int64)
    if not np.all(counts[ordinary] == 1) or not np.all(counts[ambiguous] == 2):
        raise ValueError("row kind and target count disagree")

    exact = targets.copy()
    exact[ordinary, :, 1] = exact[ordinary, :, 0]
    trained_mean = np.repeat(trained.mean(axis=2, keepdims=True), 2, axis=2)
    truth_mean = np.repeat(targets.mean(axis=2, keepdims=True), 2, axis=2)

    light_transfer = exact.copy()
    requested = light_transfer[ordinary, :, :, :3].copy()
    light_transfer[ordinary, :, :, :3] = 0.75 * requested
    light_transfer[ordinary, :, :, 3:] += 0.25 * requested

    allocation_compromise = exact.copy()
    total = allocation_compromise[ambiguous, :, :, :3] + allocation_compromise[ambiguous, :, :, 3:]
    allocation_compromise[ambiguous, :, :, :3] = 0.5 * total
    allocation_compromise[ambiguous, :, :, 3:] = 0.5 * total

    own_duplicate = np.repeat(targets[:, :, 0:1], 2, axis=2)
    alternate_duplicate = np.repeat(targets[:, :, 1:2], 2, axis=2)
    zero = np.zeros_like(trained)
    swapped_exact = exact[:, :, ::-1].copy()

    return {
        "O1_EXACT_TRUTH_DUPLICATED": (ordinary, exact[ordinary]),
        "O2_TRAINED_EXPERT_OUTPUTS": (ordinary, trained[ordinary]),
        "O3_EXPERT_MEAN_DUPLICATED": (ordinary, trained_mean[ordinary]),
        "O4_ZERO_OUTPUT": (ordinary, zero[ordinary]),
        "O5_SOURCE_SUM_PRESERVING_LIGHT_TRANSFER": (ordinary, light_transfer[ordinary]),
        "A1_EXACT_APPROVED_SET": (ambiguous, exact[ambiguous]),
        "A2_SWAPPED_EXACT_SET": (ambiguous, swapped_exact[ambiguous]),
        "A3_TRAINED_EXPERT_OUTPUTS": (ambiguous, trained[ambiguous]),
        "A4_COLLAPSED_TRUTH_MEAN": (ambiguous, truth_mean[ambiguous]),
        "A5_TRAINED_EXPERT_MEAN_DUPLICATED": (ambiguous, trained_mean[ambiguous]),
        "A6_OWN_TRUTH_DUPLICATED": (ambiguous, own_duplicate[ambiguous]),
        "A7_ALTERNATE_TRUTH_DUPLICATED": (ambiguous, alternate_duplicate[ambiguous]),
        "A8_SOURCE_SUM_PRESERVING_COMPROMISE": (ambiguous, allocation_compromise[ambiguous]),
    }


def _mean_square(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).square().mean(dim=(-3, -2, -1))


def _target_details(
    hypotheses: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact chosen target assignments and unweighted component vectors."""

    n = len(hypotheses)
    c00 = _decomposition_cost(hypotheses[:, 0], targets[:, 0])
    c10 = _decomposition_cost(hypotheses[:, 1], targets[:, 0])
    c01 = _decomposition_cost(hypotheses[:, 0], targets[:, 1])
    c11 = _decomposition_cost(hypotheses[:, 1], targets[:, 1])
    identity = c00 + c11
    swapped = c01 + c10
    identity_wins = identity <= swapped
    ambiguous = counts == 2
    chosen = torch.zeros((n, 2), dtype=torch.long, device=hypotheses.device)
    chosen[:, 0] = torch.where(ambiguous & (~identity_wins), torch.ones_like(counts), torch.zeros_like(counts))
    chosen[:, 1] = torch.where(ambiguous & identity_wins, torch.ones_like(counts), torch.zeros_like(counts))
    batch = torch.arange(n, device=hypotheses.device)
    matched = torch.stack((targets[batch, chosen[:, 0]], targets[batch, chosen[:, 1]]), dim=1)
    requested = _mean_square(hypotheses[:, :, :3], matched[:, :, :3]).sum(dim=1)
    companion = _mean_square(hypotheses[:, :, 3:], matched[:, :, 3:]).sum(dim=1)
    summed = _mean_square(source_sum(hypotheses), source_sum(matched)).sum(dim=1)
    concentration = _mean_square(hypotheses[:, 0], hypotheses[:, 1])
    concentration = torch.where(counts == 1, concentration, torch.zeros_like(concentration))
    return {
        "requested_reconstruction": requested,
        "companion_reconstruction": companion,
        "target_source_sum": summed,
        "ordinary_concentration": concentration,
        "identity_cost": identity,
        "swap_cost": swapped,
        "identity_wins": identity_wins,
        "chosen_targets": chosen,
        "matched_targets": matched,
    }


def _decomposition_cost(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    requested = _mean_square(predicted[:, :3], target[:, :3])
    companion = _mean_square(predicted[:, 3:], target[:, 3:])
    summed = _mean_square(predicted[:, :3] + predicted[:, 3:], target[:, :3] + target[:, 3:])
    return requested + companion + 0.5 * summed


def scene_loss_terms(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    blend: torch.Tensor,
    rows: Sequence[Mapping[str, str]],
    global_indices: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Return differentiable scene-level raw and weighted frozen terms."""

    if outputs.ndim != 6 or outputs.shape[1:4] != (2, 2, 6):
        raise ValueError("outputs must be N x prompt x expert x 6 x H x W")
    n = len(outputs)
    if targets.shape != outputs.shape or counts.shape != (n, 2) or blend.shape[0] != n:
        raise ValueError("loss inputs disagree")
    details = [_target_details(outputs[:, prompt], targets[:, prompt], counts[:, prompt]) for prompt in (0, 1)]
    terms: dict[str, torch.Tensor] = {}
    for name in ("requested_reconstruction", "companion_reconstruction", "target_source_sum", "ordinary_concentration"):
        terms[name] = 0.5 * (details[0][name] + details[1][name])
    terms["forward"] = 0.5 * (
        (source_sum(outputs[:, 0]) - blend[:, None]).square().mean(dim=(-4, -3, -2, -1))
        + (source_sum(outputs[:, 1]) - blend[:, None]).square().mean(dim=(-4, -3, -2, -1))
    )
    terms["prompt_swap"] = unordered_set_distance(swap_decomposition(outputs[:, 0]), outputs[:, 1])
    terms["pair_consistency"] = torch.zeros(n, dtype=outputs.dtype, device=outputs.device)
    local_by_pair: dict[str, list[int]] = defaultdict(list)
    for local, global_index in enumerate(global_indices):
        row = rows[int(global_index)]
        if row["kind"] == "near_collision":
            local_by_pair[row["near_collision_pair_id"]].append(local)
    for pair_id, members in local_by_pair.items():
        if len(members) != 2:
            raise ValueError(f"ambiguous subset does not contain both members of {pair_id}")
        left, right = members
        value = 0.5 * (
            unordered_set_distance(outputs[left:left + 1, 0], outputs[right:right + 1, 0])[0]
            + unordered_set_distance(outputs[left:left + 1, 1], outputs[right:right + 1, 1])[0]
        )
        mask = torch.zeros(n, dtype=outputs.dtype, device=outputs.device)
        mask[left] = 1.0
        mask[right] = 1.0
        terms["pair_consistency"] = terms["pair_consistency"] + mask * value
    total = torch.zeros(n, dtype=outputs.dtype, device=outputs.device)
    for name, weight in TOP_LEVEL_WEIGHTS.items():
        terms[f"weighted_{name}"] = weight * terms[name]
        total = total + terms[f"weighted_{name}"]
    terms["total"] = total
    terms["identity_cost_prompt_0"] = details[0]["identity_cost"]
    terms["swap_cost_prompt_0"] = details[0]["swap_cost"]
    terms["identity_cost_prompt_1"] = details[1]["identity_cost"]
    terms["swap_cost_prompt_1"] = details[1]["swap_cost"]
    terms["identity_wins_prompt_0"] = details[0]["identity_wins"]
    terms["identity_wins_prompt_1"] = details[1]["identity_wins"]
    terms["chosen_targets_prompt_0"] = details[0]["chosen_targets"]
    terms["chosen_targets_prompt_1"] = details[1]["chosen_targets"]
    return terms


def exact_batch_objective(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    blend: torch.Tensor,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, torch.Tensor]:
    """Return the original full-microset objective without scene-accounting rescale."""

    indices = list(range(len(outputs)))
    terms = scene_loss_terms(outputs, targets, counts, blend, rows, indices)
    ordinary = torch.as_tensor([row["kind"] == "ordinary" for row in rows], device=outputs.device)
    ambiguous = ~ordinary
    result = {name: terms[name].mean() for name in TOP_LEVEL_WEIGHTS if name != "pair_consistency"}
    result["pair_consistency"] = terms["pair_consistency"][ambiguous].mean()
    result["total"] = (
        result["requested_reconstruction"]
        + result["companion_reconstruction"]
        + 0.5 * result["target_source_sum"]
        + 0.10 * result["ordinary_concentration"]
        + 0.5 * result["forward"]
        + 0.25 * result["prompt_swap"]
        + 0.05 * result["pair_consistency"]
    )
    return result


def differentiable_scientific_surrogate(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    mean_psf_fwhm_pixel: float,
) -> torch.Tensor:
    """Max-threshold surrogate matched by the frozen loss assignment."""

    per_prompt = []
    yy, xx = torch.meshgrid(
        torch.arange(outputs.shape[-2], dtype=outputs.dtype, device=outputs.device),
        torch.arange(outputs.shape[-1], dtype=outputs.dtype, device=outputs.device),
        indexing="ij",
    )
    for prompt in (0, 1):
        details = _target_details(outputs[:, prompt], targets[:, prompt], counts[:, prompt])
        predicted = outputs[:, prompt, :, :3]
        truth = details["matched_targets"][:, :, :3]
        image = torch.linalg.vector_norm((predicted - truth).flatten(2), dim=2) / (
            0.5 * (torch.linalg.vector_norm(predicted.flatten(2), dim=2) + torch.linalg.vector_norm(truth.flatten(2), dim=2)) + 1e-12
        ) / 0.25
        flux_pred = predicted.sum(dim=(-2, -1))
        flux_true = truth.sum(dim=(-2, -1))
        relative_flux = torch.abs(flux_pred - flux_true) / (torch.abs(0.5 * (flux_pred + flux_true)) + 1e-12) / 0.20
        safe_pred = torch.clamp(flux_pred, min=1e-8)
        safe_true = torch.clamp(flux_true, min=1e-8)
        colors_pred = torch.stack((-2.5 * torch.log10(safe_pred[..., 0] / safe_pred[..., 1]), -2.5 * torch.log10(safe_pred[..., 1] / safe_pred[..., 2])), dim=-1)
        colors_true = torch.stack((-2.5 * torch.log10(safe_true[..., 0] / safe_true[..., 1]), -2.5 * torch.log10(safe_true[..., 1] / safe_true[..., 2])), dim=-1)
        colors = torch.abs(colors_pred - colors_true) / 0.20
        weight_pred = torch.relu(predicted.sum(dim=-3))
        weight_true = torch.relu(truth.sum(dim=-3))
        sum_pred = weight_pred.sum(dim=(-2, -1)) + 1e-12
        sum_true = weight_true.sum(dim=(-2, -1)) + 1e-12
        pred_x = (weight_pred * xx).sum(dim=(-2, -1)) / sum_pred
        pred_y = (weight_pred * yy).sum(dim=(-2, -1)) / sum_pred
        true_x = (weight_true * xx).sum(dim=(-2, -1)) / sum_true
        true_y = (weight_true * yy).sum(dim=(-2, -1)) / sum_true
        centroid = torch.sqrt((pred_x - true_x).square() + (pred_y - true_y).square() + 1e-20) / mean_psf_fwhm_pixel / 0.5
        components = torch.cat((image.unsqueeze(-1), relative_flux, colors, centroid.unsqueeze(-1)), dim=-1)
        per_prompt.append(components.amax(dim=(-2, -1)))
    return torch.stack(per_prompt, dim=1).amax(dim=1)


def scientific_metrics(
    outputs_normalized: np.ndarray,
    targets_physical: np.ndarray,
    counts: np.ndarray,
    blend_physical: np.ndarray,
    scales: np.ndarray,
    thresholds: object,
    sky: np.ndarray,
    mean_psf_fwhm_pixel: float,
) -> list[dict[str, object]]:
    """Evaluate frozen scientific and forward metrics for one configuration batch."""

    physical = outputs_normalized * np.tile(scales, 2)[None, None, None, :, None, None]
    result = []
    for index in range(len(physical)):
        count = int(counts[index, 0])
        distances: list[list[list[object]]] = []
        plausible = np.zeros((2, 2), dtype=bool)
        forward_global = np.zeros((2, 2), dtype=np.float64)
        for prompt in (0, 1):
            prompt_distances = []
            for expert in (0, 1):
                score = forward_consistency(blend_physical[index], np.stack((physical[index, prompt, expert, :3], physical[index, prompt, expert, 3:])), sky)
                plausible[prompt, expert] = is_plausible(score, thresholds)
                forward_global[prompt, expert] = score.global_chi_square_mean
                prompt_distances.append([
                    scientific_distance(physical[index, prompt, expert, :3], targets_physical[index, prompt, target_index, :3], mean_psf_fwhm_pixel=mean_psf_fwhm_pixel)
                    for target_index in range(count)
                ])
            distances.append(prompt_distances)
        selected = []
        own_covered = []
        alternate_covered = []
        both_covered = []
        for prompt in (0, 1):
            if count == 1:
                assignment = (0, 0)
            else:
                identity = distances[prompt][0][0].primary_normalized + distances[prompt][1][1].primary_normalized
                swapped = distances[prompt][0][1].primary_normalized + distances[prompt][1][0].primary_normalized
                assignment = (0, 1) if identity <= swapped else (1, 0)
            selected.extend(distances[prompt][expert][assignment[expert]] for expert in (0, 1))
            own = [plausible[prompt, expert] and distances[prompt][expert][0].primary_normalized <= 1.0 for expert in (0, 1)]
            own_covered.append(any(own))
            if count == 2:
                alt = [plausible[prompt, expert] and distances[prompt][expert][1].primary_normalized <= 1.0 for expert in (0, 1)]
                alternate_covered.append(any(alt))
                both_covered.append((own[0] and alt[1]) or (own[1] and alt[0]))
        color_values = [value for distance in selected for value in distance.color_gr_rz_magnitude if value is not None]
        centroid_values = [distance.centroid_pixel for distance in selected if distance.centroid_pixel is not None]
        result.append({
            "primary_scientific_distance": max(distance.primary_normalized for distance in selected),
            "image_distance": max(distance.image for distance in selected),
            "flux_distance": max(value for distance in selected for value in distance.relative_flux_grz),
            "color_distance": max(color_values) if color_values else float("nan"),
            "centroid_distance": max(centroid_values) if centroid_values else float("nan"),
            "forward_consistency_score": float(forward_global.mean()),
            "forward_consistent_fraction": float(plausible.mean()),
            "own_truth_coverage": bool(all(own_covered)),
            "alternate_truth_coverage": bool(all(alternate_covered)) if count == 2 else False,
            "both_mode_coverage": bool(all(both_covered)) if count == 2 else False,
            "ordinary_both_experts_coverage": bool(all(distances[prompt][expert][0].primary_normalized <= 1.0 and plausible[prompt, expert] for prompt in (0, 1) for expert in (0, 1))) if count == 1 else False,
        })
    return result


def source_light_transfer(base: torch.Tensor, fraction: float) -> torch.Tensor:
    """Move one source layer into the other while preserving their sum."""

    output = base.clone()
    if fraction >= 0:
        moved = fraction * output[..., :3, :, :].clone()
        output[..., :3, :, :] -= moved
        output[..., 3:, :, :] += moved
    else:
        moved = (-fraction) * output[..., 3:, :, :].clone()
        output[..., 3:, :, :] -= moved
        output[..., :3, :, :] += moved
    return output


def flux_preserving_morphology(base: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blend requested layers with a one-pixel roll while preserving band flux."""

    output = base.clone()
    requested = output[..., :3, :, :]
    mixed = (1.0 - alpha) * requested + alpha * torch.roll(requested, shifts=1, dims=-1)
    original_flux = requested.sum(dim=(-2, -1), keepdim=True)
    mixed_flux = mixed.sum(dim=(-2, -1), keepdim=True)
    scale = torch.where(mixed_flux.abs() > 1e-12, original_flux / mixed_flux, torch.ones_like(mixed_flux))
    output[..., :3, :, :] = mixed * scale
    return output
