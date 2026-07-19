"""Paired-prompt data and diagnostics for the frozen Family-E1 model.

This module deliberately defines no neural architecture, optimizer, loss, or
physical output map.  It only assembles paired views and measures whether the
unchanged :mod:`src.family_e1` model responds to their prompt difference.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def build_paired_prompt_examples(
    blend: np.ndarray,
    prompt_a: np.ndarray,
    prompt_b: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    *,
    band_scales: Sequence[float],
) -> dict[str, np.ndarray]:
    """Duplicate each observation and exchange source roles with the prompt."""
    arrays = (blend, prompt_a, prompt_b, source_a, source_b)
    if any(value.ndim != 4 for value in arrays):
        raise ValueError("all paired-prompt arrays must be rank four")
    if blend.shape[1] != 3 or prompt_a.shape[1] != 1 or prompt_b.shape[1] != 1:
        raise ValueError("blend/prompts must have 3/1/1 channels")
    if source_a.shape != blend.shape or source_b.shape != blend.shape:
        raise ValueError("source shapes must match blend")
    if prompt_a.shape[0] != len(blend) or prompt_b.shape[0] != len(blend):
        raise ValueError("prompt batch lengths must match blend")
    if prompt_a.shape[-2:] != blend.shape[-2:] or prompt_b.shape[-2:] != blend.shape[-2:]:
        raise ValueError("prompt spatial shapes must match blend")
    scales = np.asarray(tuple(band_scales), dtype=np.float32)
    if scales.shape != (3,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("band_scales must contain three finite positive values")
    normalized = np.asarray(blend, dtype=np.float32) / scales[None, :, None, None]
    input_a = np.concatenate((normalized, np.asarray(prompt_a, dtype=np.float32)), axis=1)
    input_b = np.concatenate((normalized, np.asarray(prompt_b, dtype=np.float32)), axis=1)
    return {
        "model_input": np.ascontiguousarray(np.concatenate((input_a, input_b), axis=0)),
        "observed": np.ascontiguousarray(np.concatenate((blend, blend), axis=0), dtype=np.float32),
        "requested": np.ascontiguousarray(np.concatenate((source_a, source_b), axis=0), dtype=np.float32),
        "companion": np.ascontiguousarray(np.concatenate((source_b, source_a), axis=0), dtype=np.float32),
    }


def _soft_centroids(images: Tensor, epsilon: float = 1.0e-6) -> Tensor:
    height, width = images.shape[-2:]
    y = torch.arange(height, dtype=images.dtype, device=images.device).reshape(1, 1, height, 1)
    x = torch.arange(width, dtype=images.dtype, device=images.device).reshape(1, 1, 1, width)
    flux = images.sum(dim=(-2, -1))
    cx = (images * x).sum(dim=(-2, -1)) / (flux + epsilon)
    cy = (images * y).sum(dim=(-2, -1)) / (flux + epsilon)
    return torch.stack((cx, cy), dim=-1)


def _colors(images: Tensor, epsilon: float = 1.0e-6) -> Tensor:
    flux = images.sum(dim=(-2, -1))
    return torch.stack(
        (
            torch.log(flux[:, 0] + epsilon) - torch.log(flux[:, 1] + epsilon),
            torch.log(flux[:, 1] + epsilon) - torch.log(flux[:, 2] + epsilon),
        ),
        dim=-1,
    )


def _positive_template_leakage(prediction: Tensor, own: Tensor, other: Tensor) -> Tensor:
    """Companion fraction from a nonnegative two-template least-squares proxy."""
    flat_prediction = prediction.flatten(2).reshape(-1, prediction.shape[-2] * prediction.shape[-1])
    flat_own = own.flatten(2).reshape_as(flat_prediction)
    flat_other = other.flatten(2).reshape_as(flat_prediction)
    epsilon = torch.finfo(prediction.dtype).eps
    own_own = torch.sum(flat_own * flat_own, dim=1) + epsilon
    own_other = torch.sum(flat_own * flat_other, dim=1)
    other_other = torch.sum(flat_other * flat_other, dim=1) + epsilon
    own_prediction = torch.sum(flat_own * flat_prediction, dim=1)
    other_prediction = torch.sum(flat_other * flat_prediction, dim=1)
    determinant = torch.clamp(own_own * other_other - own_other.square(), min=epsilon)
    own_coefficient = torch.clamp(
        (other_other * own_prediction - own_other * other_prediction) / determinant,
        min=0.0,
    )
    other_coefficient = torch.clamp(
        (own_own * other_prediction - own_other * own_prediction) / determinant,
        min=0.0,
    )
    return other_coefficient / (own_coefficient + other_coefficient + epsilon)


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> float:
    denominator_value = float(denominator.detach().cpu())
    if denominator_value == 0.0:
        return 0.0 if float(numerator.detach().cpu()) == 0.0 else float("inf")
    return float((numerator / denominator).detach().cpu())


def paired_prediction_metrics(
    predicted_requested: Tensor,
    requested_target: Tensor,
    companion_target: Tensor,
    *,
    scene_count: int,
    band_scales: Sequence[float],
) -> dict[str, float]:
    """Measure identity, leakage, and paired response for requested predictions."""
    if scene_count <= 0 or len(predicted_requested) != 2 * scene_count:
        raise ValueError("paired batches must contain A views followed by B views")
    if predicted_requested.shape != requested_target.shape or predicted_requested.shape != companion_target.shape:
        raise ValueError("prediction and target shapes must match")
    scales = torch.as_tensor(tuple(band_scales), dtype=predicted_requested.dtype, device=predicted_requested.device)
    scales = scales.reshape(1, 3, 1, 1)
    prediction = predicted_requested / scales
    requested = requested_target / scales
    companion = companion_target / scales

    own_mse = torch.mean((prediction - requested).square(), dim=(1, 2, 3))
    wrong_mse = torch.mean((prediction - companion).square(), dim=(1, 2, 3))
    identity = own_mse < wrong_mse
    pair_correct = identity[:scene_count] & identity[scene_count:]
    requested_error = torch.mean(torch.abs(prediction - requested))
    leakage = torch.mean(_positive_template_leakage(prediction, requested, companion))

    prediction_a, prediction_b = prediction[:scene_count], prediction[scene_count:]
    truth_a, truth_b = requested[:scene_count], requested[scene_count:]
    predicted_l1 = torch.mean(torch.abs(prediction_a - prediction_b))
    truth_l1 = torch.mean(torch.abs(truth_a - truth_b))
    predicted_flux = prediction_a.sum(dim=(-2, -1))
    predicted_flux_b = prediction_b.sum(dim=(-2, -1))
    truth_flux = truth_a.sum(dim=(-2, -1))
    truth_flux_b = truth_b.sum(dim=(-2, -1))
    flux_difference = torch.mean(torch.abs(predicted_flux - predicted_flux_b))
    truth_flux_difference = torch.mean(torch.abs(truth_flux - truth_flux_b))
    centroid_difference = torch.mean(
        torch.linalg.vector_norm(_soft_centroids(prediction_a) - _soft_centroids(prediction_b), dim=-1)
    )
    truth_centroid_difference = torch.mean(
        torch.linalg.vector_norm(_soft_centroids(truth_a) - _soft_centroids(truth_b), dim=-1)
    )
    color_difference = torch.mean(torch.abs(_colors(prediction_a) - _colors(prediction_b)))
    truth_color_difference = torch.mean(torch.abs(_colors(truth_a) - _colors(truth_b)))
    cosine = torch.mean(
        F.cosine_similarity(prediction_a.flatten(1), prediction_b.flatten(1), dim=1, eps=1.0e-12)
    )
    truth_cosine = torch.mean(
        F.cosine_similarity(truth_a.flatten(1), truth_b.flatten(1), dim=1, eps=1.0e-12)
    )
    return {
        "prompt_identity": float(identity.float().mean().detach().cpu()),
        "prompt_swap": float(pair_correct.float().mean().detach().cpu()),
        "requested_source_error": float(requested_error.detach().cpu()),
        "companion_leakage": float(leakage.detach().cpu()),
        "identity_margin_mse": float((wrong_mse - own_mse).mean().detach().cpu()),
        "cross_prompt_l1_difference": float(predicted_l1.detach().cpu()),
        "truth_cross_prompt_l1_difference": float(truth_l1.detach().cpu()),
        "cross_prompt_l1_response_ratio": _safe_ratio(predicted_l1, truth_l1),
        "cross_prompt_flux_difference": float(flux_difference.detach().cpu()),
        "truth_cross_prompt_flux_difference": float(truth_flux_difference.detach().cpu()),
        "cross_prompt_flux_response_ratio": _safe_ratio(flux_difference, truth_flux_difference),
        "cross_prompt_centroid_difference_pixels": float(centroid_difference.detach().cpu()),
        "truth_cross_prompt_centroid_difference_pixels": float(truth_centroid_difference.detach().cpu()),
        "cross_prompt_centroid_response_ratio": _safe_ratio(centroid_difference, truth_centroid_difference),
        "cross_prompt_color_difference": float(color_difference.detach().cpu()),
        "truth_cross_prompt_color_difference": float(truth_color_difference.detach().cpu()),
        "cross_prompt_color_response_ratio": _safe_ratio(color_difference, truth_color_difference),
        "cross_prompt_cosine_similarity": float(cosine.detach().cpu()),
        "truth_cross_prompt_cosine_similarity": float(truth_cosine.detach().cpu()),
    }


def activation_pair_metrics(activation: Tensor, *, scene_count: int) -> dict[str, float | bool]:
    """Quantify the prompt-specific component of a paired feature tensor."""
    if scene_count <= 0 or len(activation) != 2 * scene_count:
        raise ValueError("activation must contain A views followed by B views")
    first = activation[:scene_count].detach().float().flatten(1)
    second = activation[scene_count:].detach().float().flatten(1)
    difference = first - second
    prompt_norm = torch.sqrt(torch.mean(difference.square()))
    feature_norm = 0.5 * (
        torch.sqrt(torch.mean(first.square())) + torch.sqrt(torch.mean(second.square()))
    )
    modulation = prompt_norm / torch.clamp(feature_norm, min=1.0e-30)
    first_centered = first - first.mean(dim=1, keepdim=True)
    second_centered = second - second.mean(dim=1, keepdim=True)
    correlation = torch.mean(
        F.cosine_similarity(first_centered, second_centered, dim=1, eps=1.0e-12)
    )
    cosine = torch.mean(F.cosine_similarity(first, second, dim=1, eps=1.0e-12))
    signal_power = torch.mean(difference.square())
    shared_power = 0.5 * (torch.mean(first.square()) + torch.mean(second.square()))
    information_proxy = 0.5 * torch.log1p(signal_power / torch.clamp(shared_power, min=1.0e-30))
    modulation_value = float(modulation.cpu())
    correlation_value = float(correlation.cpu())
    return {
        "activation_norm": float(feature_norm.cpu()),
        "prompt_activation_norm": float(prompt_norm.cpu()),
        "feature_modulation": modulation_value,
        "cross_correlation": correlation_value,
        "pair_cosine_similarity": float(cosine.cpu()),
        "mutual_information_proxy_nats": float(information_proxy.cpu()),
        "numerically_indistinguishable": modulation_value <= 1.0e-6 and correlation_value >= 0.999999,
    }
