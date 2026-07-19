"""Differentiable scientific-alignment objective for the Thayer-SA micro campaign."""

from __future__ import annotations

from dataclasses import dataclass

import torch


IMAGE_THRESHOLD = 0.25
FLUX_THRESHOLD = 0.20
COLOR_THRESHOLD_MAG = 0.20
CENTROID_THRESHOLD_PSF = 0.50
IMAGE_FLOOR = 1e-12
FLUX_FLOOR = 1e-12
POSITIVE_FLUX_FLOOR = 1e-12
CENTROID_FLOOR = 1e-12
SMOOTHMAX_TEMPERATURE = 0.005
LAMBDA_SCIENCE = 1.0
ORDINARY_CONCENTRATION_WEIGHT = 1.0


@dataclass(frozen=True)
class ScientificComponents:
    """Threshold-normalized differentiable requested-source distances."""

    image: torch.Tensor
    flux_grz: torch.Tensor
    color_gr_rz: torch.Tensor
    centroid: torch.Tensor

    def stacked(self) -> torch.Tensor:
        return torch.cat(
            (
                self.image.unsqueeze(-1),
                self.flux_grz,
                self.color_gr_rz,
                self.centroid.unsqueeze(-1),
            ),
            dim=-1,
        )


def _physical(source: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if source.shape[-3] != 3 or scales.shape != (3,):
        raise ValueError("source must end in 3xHxW and scales must have shape (3,)")
    view = (1,) * (source.ndim - 3) + (3, 1, 1)
    return source * scales.to(dtype=source.dtype, device=source.device).view(view)


def _symmetric_image_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.linalg.vector_norm((predicted - target).flatten(-3), dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted.flatten(-3), dim=-1)
    target_norm = torch.linalg.vector_norm(target.flatten(-3), dim=-1)
    return difference / (0.5 * (predicted_norm + target_norm) + IMAGE_FLOOR)


def scientific_components(
    predicted_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    scales: torch.Tensor,
    mean_psf_fwhm_pixel: float,
) -> ScientificComponents:
    """Match the frozen physical image, flux, color, and centroid geometry."""

    if predicted_normalized.shape != target_normalized.shape:
        raise ValueError("predicted and target sources must have equal shape")
    if predicted_normalized.ndim < 3 or predicted_normalized.shape[-3] != 3:
        raise ValueError("sources must end in 3xHxW")
    if mean_psf_fwhm_pixel <= 0:
        raise ValueError("mean PSF FWHM must be positive")

    predicted = _physical(predicted_normalized, scales)
    target = _physical(target_normalized.detach(), scales)
    image = _symmetric_image_distance(predicted, target) / IMAGE_THRESHOLD

    flux_predicted = predicted.sum(dim=(-2, -1))
    flux_target = target.sum(dim=(-2, -1))
    flux_denominator = torch.abs(0.5 * (flux_predicted + flux_target)) + FLUX_FLOOR
    flux = torch.abs(flux_predicted - flux_target) / flux_denominator / FLUX_THRESHOLD

    safe_predicted = torch.clamp(flux_predicted, min=POSITIVE_FLUX_FLOOR)
    safe_target = torch.clamp(flux_target, min=POSITIVE_FLUX_FLOOR)
    colors_predicted = torch.stack(
        (
            -2.5 * torch.log10(safe_predicted[..., 0] / safe_predicted[..., 1]),
            -2.5 * torch.log10(safe_predicted[..., 1] / safe_predicted[..., 2]),
        ),
        dim=-1,
    )
    colors_target = torch.stack(
        (
            -2.5 * torch.log10(safe_target[..., 0] / safe_target[..., 1]),
            -2.5 * torch.log10(safe_target[..., 1] / safe_target[..., 2]),
        ),
        dim=-1,
    )
    color = torch.abs(colors_predicted - colors_target) / COLOR_THRESHOLD_MAG

    weight_predicted = torch.relu(predicted.sum(dim=-3))
    weight_target = torch.relu(target.sum(dim=-3))
    yy, xx = torch.meshgrid(
        torch.arange(predicted.shape[-2], dtype=predicted.dtype, device=predicted.device),
        torch.arange(predicted.shape[-1], dtype=predicted.dtype, device=predicted.device),
        indexing="ij",
    )
    total_predicted = weight_predicted.sum(dim=(-2, -1)) + CENTROID_FLOOR
    total_target = weight_target.sum(dim=(-2, -1)) + CENTROID_FLOOR
    x_predicted = (weight_predicted * xx).sum(dim=(-2, -1)) / total_predicted
    y_predicted = (weight_predicted * yy).sum(dim=(-2, -1)) / total_predicted
    x_target = (weight_target * xx).sum(dim=(-2, -1)) / total_target
    y_target = (weight_target * yy).sum(dim=(-2, -1)) / total_target
    squared = (x_predicted - x_target).square() + (y_predicted - y_target).square()
    centroid_pixel = torch.sqrt(squared + CENTROID_FLOOR**2) - CENTROID_FLOOR
    centroid = centroid_pixel / float(mean_psf_fwhm_pixel) / CENTROID_THRESHOLD_PSF
    return ScientificComponents(image=image, flux_grz=flux, color_gr_rz=color, centroid=centroid)


def smoothmax(values: torch.Tensor, temperature: float = SMOOTHMAX_TEMPERATURE) -> torch.Tensor:
    """Zero-anchored log-mean-exp smooth maximum over the final dimension."""

    if values.shape[-1] < 1 or temperature <= 0:
        raise ValueError("smoothmax requires components and positive temperature")
    count = values.shape[-1]
    offset = torch.log(torch.as_tensor(float(count), dtype=values.dtype, device=values.device))
    return temperature * (torch.logsumexp(values / temperature, dim=-1) - offset)


def scientific_surrogate(
    predicted_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    scales: torch.Tensor,
    mean_psf_fwhm_pixel: float,
) -> torch.Tensor:
    return smoothmax(
        scientific_components(
            predicted_normalized,
            target_normalized,
            scales,
            mean_psf_fwhm_pixel,
        ).stacked()
    )


def normalized_source_reconstruction(
    predicted_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Squared physical symmetric image distance, scaled to the frozen limit."""

    predicted = _physical(predicted_normalized, scales)
    target = _physical(target_normalized.detach(), scales)
    return (_symmetric_image_distance(predicted, target) / IMAGE_THRESHOLD).square()


def pairwise_cost(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scales: torch.Tensor,
    mean_psf_fwhm_pixel: float,
) -> dict[str, torch.Tensor]:
    """Corrected expert-to-full-decomposition target cost."""

    if predicted.shape != target.shape or predicted.shape[-3] != 6:
        raise ValueError("decompositions must be equal and end in 6xHxW")
    requested = normalized_source_reconstruction(predicted[..., :3, :, :], target[..., :3, :, :], scales)
    companion = normalized_source_reconstruction(predicted[..., 3:, :, :], target[..., 3:, :, :], scales)
    science = scientific_surrogate(
        predicted[..., :3, :, :],
        target[..., :3, :, :],
        scales,
        mean_psf_fwhm_pixel,
    )
    return {
        "requested_reconstruction": requested,
        "companion_reconstruction": companion,
        "science": science,
        "total": requested + companion + LAMBDA_SCIENCE * science,
    }


def corrected_objective(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    scales: torch.Tensor,
    mean_psf_fwhm_pixel: float,
) -> dict[str, torch.Tensor]:
    """Balanced prompt/scene objective with hard two-permutation set matching."""

    if outputs.ndim != 6 or outputs.shape[1:4] != (2, 2, 6):
        raise ValueError("outputs must be N x prompt x expert x 6 x H x W")
    if targets.shape != outputs.shape or counts.shape != outputs.shape[:2]:
        raise ValueError("targets and counts must match output scene/prompt axes")
    if not bool(torch.all((counts == 1) | (counts == 2))):
        raise ValueError("target counts must be one or two")

    scene_totals = []
    scene_requested = []
    scene_companion = []
    scene_science = []
    scene_concentration = []
    identity_margins = []
    identity_wins = []
    for prompt in (0, 1):
        costs = [[pairwise_cost(outputs[:, prompt, expert], targets[:, prompt, target], scales, mean_psf_fwhm_pixel) for target in (0, 1)] for expert in (0, 1)]
        identity = 0.5 * (costs[0][0]["total"] + costs[1][1]["total"])
        swapped = 0.5 * (costs[0][1]["total"] + costs[1][0]["total"])
        choose_identity = identity <= swapped
        ambiguous_total = torch.minimum(identity, swapped)
        ordinary_total = 0.5 * (costs[0][0]["total"] + costs[1][0]["total"])
        concentration = scientific_surrogate(
            outputs[:, prompt, 0, :3], outputs[:, prompt, 1, :3], scales, mean_psf_fwhm_pixel
        )
        concentration = torch.where(counts[:, prompt] == 1, concentration, torch.zeros_like(concentration))
        scene_totals.append(torch.where(counts[:, prompt] == 2, ambiguous_total, ordinary_total + ORDINARY_CONCENTRATION_WEIGHT * concentration))

        for key, destination in (("requested_reconstruction", scene_requested), ("companion_reconstruction", scene_companion), ("science", scene_science)):
            identity_term = 0.5 * (costs[0][0][key] + costs[1][1][key])
            swapped_term = 0.5 * (costs[0][1][key] + costs[1][0][key])
            ambiguous_term = torch.where(choose_identity, identity_term, swapped_term)
            ordinary_term = 0.5 * (costs[0][0][key] + costs[1][0][key])
            destination.append(torch.where(counts[:, prompt] == 2, ambiguous_term, ordinary_term))
        scene_concentration.append(concentration)
        identity_margins.append(swapped - identity)
        identity_wins.append(choose_identity)

    per_scene = 0.5 * (scene_totals[0] + scene_totals[1])
    return {
        "total": per_scene.mean(),
        "per_scene": per_scene,
        "requested_reconstruction": 0.5 * (scene_requested[0] + scene_requested[1]),
        "companion_reconstruction": 0.5 * (scene_companion[0] + scene_companion[1]),
        "science": 0.5 * (scene_science[0] + scene_science[1]),
        "ordinary_concentration": 0.5 * (scene_concentration[0] + scene_concentration[1]),
        "identity_margin": torch.stack(identity_margins, dim=1),
        "identity_wins": torch.stack(identity_wins, dim=1),
    }
