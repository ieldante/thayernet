from __future__ import annotations

import torch

from src.output_conditioning import (
    common_allocation_gradient_parts,
    project_source_nonnegative,
    project_total_allocation,
    projected_lbfgs,
    source_to_total_allocation,
    threshold_jacobian_preconditioner,
    total_allocation_to_source,
)


def test_coordinate_roundtrip_exact() -> None:
    source = torch.rand(2, 2, 2, 6, 5, 4, dtype=torch.float64)
    total, allocation = source_to_total_allocation(source)
    decoded = total_allocation_to_source(total, allocation)
    assert torch.allclose(source, decoded, rtol=1e-15, atol=1e-15)


def test_projection_is_finite_nonnegative_and_idempotent() -> None:
    source = torch.tensor([[[[-1.0, float("nan")], [2.0, float("inf")]]]]).repeat(1, 6, 1, 1)
    first, stats = project_source_nonnegative(source)
    second, _ = project_source_nonnegative(first)
    assert torch.isfinite(first).all()
    assert (first >= 0).all()
    assert torch.equal(first, second)
    assert stats.clipped_fraction > 0
    total, allocation = source_to_total_allocation(first)
    total2, allocation2, _ = project_total_allocation(total, allocation)
    assert torch.allclose(first, total_allocation_to_source(total2, allocation2), rtol=1e-15, atol=1e-15)


def test_common_allocation_parts_reconstruct_and_are_orthogonal() -> None:
    gradient = torch.randn(4, 6, 3, 3, dtype=torch.float64)
    common, allocation = common_allocation_gradient_parts(gradient)
    assert torch.allclose(common + allocation, gradient)
    assert abs(float((common * allocation).sum())) < 1e-10


def test_preconditioner_caps() -> None:
    jacobian = torch.tensor([0.0, 1e-8, 1.0, 100.0], dtype=torch.float64)
    value = threshold_jacobian_preconditioner(jacobian)
    assert torch.isfinite(value).all()
    assert float(value.min()) >= 0.1
    assert float(value.max()) <= 10.0


def test_projected_lbfgs_isolated_quadratic() -> None:
    variable = torch.tensor([-1.0, 3.0], dtype=torch.float64, requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return ((value - torch.tensor([1.0, 2.0], dtype=value.dtype)) ** 2).sum()

    def project_(value: torch.Tensor) -> None:
        value.clamp_(min=0.0)

    result = projected_lbfgs(variable, objective, project_, max_iterations=20, trust_rms=1.0)
    assert result.value < 1e-10
    assert torch.allclose(variable, torch.tensor([1.0, 2.0], dtype=torch.float64), atol=1e-5)
    assert result.gradient_evaluations <= 400
