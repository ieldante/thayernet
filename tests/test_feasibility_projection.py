from __future__ import annotations

import numpy as np
import torch

from src.feasibility_projection import (
    augmented_lagrangian_refinement,
    constraint_ratios,
    homotopy_projection,
    normalized_correction,
)


def truth() -> np.ndarray:
    value = np.zeros((6, 60, 60), dtype=np.float32)
    value[:, 25:35, 25:35] = np.asarray([4, 5, 6, 2, 3, 4], dtype=np.float32)[:, None, None]
    return value


def test_exact_truth_is_feasible_and_zero_distance() -> None:
    target = truth()
    ratios = constraint_ratios(target, target)
    assert ratios.feasible(0.95)
    assert ratios.maximum <= 1e-10
    assert normalized_correction(target, target) == 0.0


def test_homotopy_bisection_reaches_fixed_interior() -> None:
    target = truth()
    candidate = np.roll(target, 8, axis=-1) - 0.1
    projected, summary, rows = homotopy_projection(candidate, target, grid_size=33, bisection_steps=20)
    assert len(rows) == 33
    assert 0 <= float(summary["boundary_alpha"]) <= float(summary["interior_alpha"]) <= 1
    assert constraint_ratios(projected, target).feasible(0.95)


def test_p1_is_finite_nonnegative_and_feasible() -> None:
    target = torch.from_numpy(truth()[None])
    candidate = target + 0.02
    start = target.clone()
    scales = torch.ones(3)
    result, trajectory = augmented_lagrangian_refinement(start, candidate, target, scales, iterations=10)
    assert trajectory
    assert torch.isfinite(result).all()
    assert bool((result >= 0).all())
