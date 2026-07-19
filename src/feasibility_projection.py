"""Direct projection into the frozen Thayer scientific feasibility region.

The module is deliberately model-free.  It operates on detached six-channel
source decompositions and the unchanged requested-source scientific metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.competing_hypotheses import scientific_distance
from src.scientific_alignment import scientific_components


GRID_SIZE = 1025
INTERIOR_LIMIT = 0.95
BISECTION_STEPS = 40
FEASIBILITY_TOLERANCE = 1e-7
MEAN_PSF_FWHM_PIXEL = float(np.mean([0.86, 0.81, 0.77]) / 0.2)


@dataclass(frozen=True)
class ConstraintRatios:
    image: float
    flux_g: float
    flux_r: float
    flux_z: float
    color_gr: float
    color_rz: float
    centroid: float
    finite: bool
    nonnegative: bool

    def scientific_values(self) -> tuple[float, ...]:
        return (
            self.image,
            self.flux_g,
            self.flux_r,
            self.flux_z,
            self.color_gr,
            self.color_rz,
            self.centroid,
        )

    @property
    def maximum(self) -> float:
        return float(max(self.scientific_values()))

    def feasible(self, limit: float = 1.0, tolerance: float = FEASIBILITY_TOLERANCE) -> bool:
        return bool(self.finite and self.nonnegative and self.maximum <= limit + tolerance)


def output_contract_valid(value: np.ndarray, tolerance: float = FEASIBILITY_TOLERANCE) -> bool:
    array = np.asarray(value)
    return bool(array.shape == (6, 60, 60) and np.all(np.isfinite(array)) and np.min(array) >= -tolerance)


def constraint_ratios(
    decomposition: np.ndarray,
    target: np.ndarray,
    *,
    mean_psf_fwhm_pixel: float = MEAN_PSF_FWHM_PIXEL,
) -> ConstraintRatios:
    """Return the exact frozen requested-source ratios plus output validity."""

    candidate = np.asarray(decomposition, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    finite = bool(candidate.shape == (6, 60, 60) and truth.shape == candidate.shape and np.all(np.isfinite(candidate)))
    if not finite:
        return ConstraintRatios(*(float("inf"),) * 7, finite=False, nonnegative=False)
    distance = scientific_distance(candidate[:3], truth[:3], mean_psf_fwhm_pixel=mean_psf_fwhm_pixel)
    colors = tuple(float("inf") if value is None else float(value / 0.20) for value in distance.color_gr_rz_magnitude)
    centroid = float("inf") if distance.centroid_psf is None else float(distance.centroid_psf / 0.50)
    return ConstraintRatios(
        image=float(distance.image / 0.25),
        flux_g=float(distance.relative_flux_grz[0] / 0.20),
        flux_r=float(distance.relative_flux_grz[1] / 0.20),
        flux_z=float(distance.relative_flux_grz[2] / 0.20),
        color_gr=colors[0],
        color_rz=colors[1],
        centroid=centroid,
        finite=True,
        nonnegative=bool(np.min(candidate) >= -FEASIBILITY_TOLERANCE),
    )


def normalized_correction(projected: np.ndarray, candidate: np.ndarray) -> float:
    projected64 = np.asarray(projected, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    denominator = float(np.linalg.norm(candidate64.ravel())) + 1e-12
    return float(np.linalg.norm((projected64 - candidate64).ravel()) / denominator)


def _alpha_value(candidate: np.ndarray, truth: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * candidate + alpha * truth


def _intervals(mask: np.ndarray, alphas: np.ndarray) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            stop = index if value and index == len(mask) - 1 else index - 1
            intervals.append((float(alphas[start]), float(alphas[stop])))
            start = None
    return intervals


def homotopy_projection(
    candidate: np.ndarray,
    truth: np.ndarray,
    *,
    grid_size: int = GRID_SIZE,
    interior_limit: float = INTERIOR_LIMIT,
    bisection_steps: int = BISECTION_STEPS,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    """Find the earliest feasible truth homotopy point and move to fixed slack."""

    if grid_size < 3 or bisection_steps < 1 or not 0 < interior_limit < 1:
        raise ValueError("invalid frozen homotopy settings")
    candidate64 = np.asarray(candidate, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    alphas = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    rows: list[dict[str, object]] = []
    ratios: list[ConstraintRatios] = []
    for alpha in alphas:
        ratio = constraint_ratios(_alpha_value(candidate64, truth64, float(alpha)), truth64)
        ratios.append(ratio)
        rows.append({
            "alpha": float(alpha),
            "image": ratio.image,
            "flux_g": ratio.flux_g,
            "flux_r": ratio.flux_r,
            "flux_z": ratio.flux_z,
            "color_gr": ratio.color_gr,
            "color_rz": ratio.color_rz,
            "centroid": ratio.centroid,
            "maximum": ratio.maximum,
            "finite": ratio.finite,
            "nonnegative": ratio.nonnegative,
            "feasible": ratio.feasible(1.0),
            "interior": ratio.feasible(interior_limit),
        })
    feasible = np.asarray([ratio.feasible(1.0) for ratio in ratios], dtype=bool)
    interior = np.asarray([ratio.feasible(interior_limit) for ratio in ratios], dtype=bool)
    if not feasible[-1] or not interior[-1]:
        raise RuntimeError("exact truth failed the guaranteed homotopy anchor")
    first_grid = int(np.flatnonzero(feasible)[0])
    low = float(alphas[max(first_grid - 1, 0)])
    high = float(alphas[first_grid])
    if first_grid == 0:
        boundary = 0.0
    else:
        for _ in range(bisection_steps):
            midpoint = 0.5 * (low + high)
            if constraint_ratios(_alpha_value(candidate64, truth64, midpoint), truth64).feasible(1.0):
                high = midpoint
            else:
                low = midpoint
        boundary = high
    first_interior = int(np.flatnonzero(interior)[0])
    low_i = float(alphas[max(first_interior - 1, 0)])
    high_i = float(alphas[first_interior])
    if first_interior == 0:
        interior_alpha = 0.0
    else:
        for _ in range(bisection_steps):
            midpoint = 0.5 * (low_i + high_i)
            if constraint_ratios(_alpha_value(candidate64, truth64, midpoint), truth64).feasible(interior_limit):
                high_i = midpoint
            else:
                low_i = midpoint
        interior_alpha = min(1.0, high_i + 1e-8)
    projected = _alpha_value(candidate64, truth64, interior_alpha).astype(np.float32)
    final_ratio = constraint_ratios(projected, truth64)
    if not final_ratio.feasible(interior_limit):
        projected = truth64.astype(np.float32)
        interior_alpha = 1.0
        final_ratio = constraint_ratios(projected, truth64)
    values = np.asarray([ratio.scientific_values() for ratio in ratios], dtype=np.float64)
    monotone = bool(np.all(np.diff(values, axis=0) <= 1e-8))
    feasible_monotone = bool(np.all(np.diff(feasible.astype(np.int8)) >= 0))
    metric_names = ("image", "flux_g", "flux_r", "flux_z", "color_gr", "color_rz", "centroid")
    entry = {}
    for metric_index, name in enumerate(metric_names):
        hits = np.flatnonzero(values[:, metric_index] <= 1.0 + FEASIBILITY_TOLERANCE)
        entry[name] = float(alphas[hits[0]]) if len(hits) else float("nan")
    summary: dict[str, object] = {
        "boundary_alpha": boundary,
        "interior_alpha": interior_alpha,
        "correction_norm": normalized_correction(projected, candidate64),
        "final_max_ratio": final_ratio.maximum,
        "scientific_ratios_monotone": monotone,
        "feasibility_monotone": feasible_monotone,
        "feasible_intervals": _intervals(feasible, alphas),
        "path_nonmonotonicity": not feasible_monotone,
        "per_constraint_entry_alpha": entry,
        "first_limiting_metric": max(entry, key=lambda name: entry[name] if np.isfinite(entry[name]) else float("inf")),
    }
    return projected, summary, rows


def scientific_ratio_tensor(
    outputs_normalized: torch.Tensor,
    targets_normalized: torch.Tensor,
    scales: torch.Tensor,
    mean_psf_fwhm_pixel: float = MEAN_PSF_FWHM_PIXEL,
) -> torch.Tensor:
    return scientific_components(outputs_normalized[..., :3, :, :], targets_normalized[..., :3, :, :], scales, mean_psf_fwhm_pixel).stacked()


def augmented_lagrangian_refinement(
    start_normalized: torch.Tensor,
    candidate_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    scales: torch.Tensor,
    *,
    iterations: int = 80,
    learning_rate: float = 2e-4,
    dual_update_interval: int = 10,
    interior_limit: float = INTERIOR_LIMIT,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Frozen P1 projected primal-dual refinement from the feasible P0 point."""

    variable = start_normalized.detach().clone().to(dtype=torch.float32, device="cpu").requires_grad_(True)
    candidate = candidate_normalized.detach().to(dtype=torch.float32, device="cpu")
    target = target_normalized.detach().to(dtype=torch.float32, device="cpu")
    scales_cpu = scales.detach().to(dtype=torch.float32, device="cpu")
    dual = torch.zeros((*variable.shape[:-3], 7), dtype=torch.float32)
    optimizer = torch.optim.Adam([variable], lr=learning_rate, weight_decay=0.0)
    penalties = (10.0, 30.0, 100.0, 300.0)
    trajectory: list[dict[str, float]] = []
    for iteration in range(1, iterations + 1):
        penalty = penalties[min((iteration - 1) // max(iterations // len(penalties), 1), len(penalties) - 1)]
        optimizer.zero_grad(set_to_none=True)
        ratios = scientific_ratio_tensor(variable, target, scales_cpu)
        violation = torch.relu(ratios - interior_limit)
        distance = (variable - candidate).square().mean(dim=(-3, -2, -1))
        objective = (distance + (dual * violation).sum(dim=-1) + 0.5 * penalty * violation.square().sum(dim=-1)).mean()
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("nonfinite P1 objective")
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(min=0.0)
            if iteration % dual_update_interval == 0:
                current = torch.relu(scientific_ratio_tensor(variable, target, scales_cpu) - interior_limit)
                dual.add_(penalty * current)
        if iteration == 1 or iteration % 10 == 0 or iteration == iterations:
            with torch.no_grad():
                maximum = float(scientific_ratio_tensor(variable, target, scales_cpu).max().cpu())
                trajectory.append({"iteration": float(iteration), "objective": float(objective.detach().cpu()), "max_ratio": maximum, "penalty": penalty})

    # Fixed feasibility restoration: bisect every invalid result toward its P0
    # anchor.  This is part of P1 and is independent of scene identity.
    with torch.no_grad():
        result = variable.detach().clone()
        maximum = scientific_ratio_tensor(result, target, scales_cpu).amax(dim=-1)
        invalid = maximum > interior_limit + 1e-6
        if bool(invalid.any()):
            low = torch.zeros_like(maximum)
            high = torch.ones_like(maximum)
            for _ in range(BISECTION_STEPS):
                midpoint = 0.5 * (low + high)
                view = midpoint.view(*midpoint.shape, 1, 1, 1)
                trial = view * start_normalized + (1.0 - view) * result
                trial.clamp_(min=0.0)
                okay = scientific_ratio_tensor(trial, target, scales_cpu).amax(dim=-1) <= interior_limit
                high = torch.where(okay, midpoint, high)
                low = torch.where(okay, low, midpoint)
            view = high.view(*high.shape, 1, 1, 1)
            result = view * start_normalized + (1.0 - view) * result
            result.clamp_(min=0.0)
    return result, trajectory
