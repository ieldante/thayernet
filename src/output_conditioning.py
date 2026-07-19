"""Frozen physical output coordinates for the Thayer-OC audit.

This module contains no model code.  Every optimizer target is a detached
physical-output tensor or its exactly equivalent total/allocation coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch


@dataclass(frozen=True)
class ProjectionStats:
    clipped_fraction: float
    nonfinite_fraction: float


def source_to_total_allocation(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [..., 6, H, W] requested/companion layers as (T, D)."""

    if source.ndim < 3 or source.shape[-3] != 6:
        raise ValueError("source must end in 6xHxW")
    requested = source[..., :3, :, :]
    companion = source[..., 3:, :, :]
    return requested + companion, 0.5 * (requested - companion)


def total_allocation_to_source(total: torch.Tensor, allocation: torch.Tensor) -> torch.Tensor:
    """Decode (T, D) with S_req=0.5T+D and S_comp=0.5T-D."""

    if total.shape != allocation.shape or total.ndim < 3 or total.shape[-3] != 3:
        raise ValueError("total and allocation must have equal ...x3xHxW shapes")
    return torch.cat((0.5 * total + allocation, 0.5 * total - allocation), dim=-3)


def project_source_nonnegative(source: torch.Tensor) -> tuple[torch.Tensor, ProjectionStats]:
    """Frozen target-independent finite/nonnegative physical-layer projection."""

    finite = torch.isfinite(source)
    nonfinite_fraction = float((~finite).to(torch.float64).mean().detach().cpu())
    sanitized = torch.where(finite, source, torch.zeros_like(source))
    clipped_fraction = float((sanitized < 0).to(torch.float64).mean().detach().cpu())
    return torch.clamp_min(sanitized, 0.0), ProjectionStats(clipped_fraction, nonfinite_fraction)


def project_total_allocation(total: torch.Tensor, allocation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, ProjectionStats]:
    """Project only after physical decoding, then exactly re-encode."""

    projected, stats = project_source_nonnegative(total_allocation_to_source(total, allocation))
    projected_total, projected_allocation = source_to_total_allocation(projected)
    return projected_total, projected_allocation, stats


def common_allocation_gradient_parts(gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal raw-source projections onto common and allocation subspaces."""

    if gradient.shape[-3] != 6:
        raise ValueError("gradient must end in 6xHxW")
    requested = gradient[..., :3, :, :]
    companion = gradient[..., 3:, :, :]
    common_half = 0.5 * (requested + companion)
    allocation_half = 0.5 * (requested - companion)
    common = torch.cat((common_half, common_half), dim=-3)
    allocation = torch.cat((allocation_half, -allocation_half), dim=-3)
    return common, allocation


def threshold_jacobian_preconditioner(
    jacobian: torch.Tensor,
    floor: float = 1e-8,
    minimum: float = 0.1,
    maximum: float = 10.0,
) -> torch.Tensor:
    """Frozen median-normalized inverse local surrogate-Jacobian magnitude."""

    if floor <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid preconditioner floor/caps")
    magnitude = jacobian.detach().abs()
    positive = magnitude[magnitude > 0]
    reference = positive.median() if positive.numel() else torch.ones((), dtype=magnitude.dtype, device=magnitude.device)
    return torch.clamp(reference / (magnitude + floor), min=minimum, max=maximum)


@dataclass
class LBFGSResult:
    value: float
    objective_evaluations: int
    gradient_evaluations: int
    accepted_steps: int
    line_search_failures: int


def projected_lbfgs(
    variable: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    project_: Callable[[torch.Tensor], None],
    *,
    max_objective_evaluations: int = 401,
    max_gradient_evaluations: int = 400,
    max_iterations: int = 120,
    history_size: int = 5,
    armijo_c1: float = 1e-4,
    line_search_shrink: float = 0.5,
    line_search_trials: int = 8,
    trust_rms: float = 0.01,
    tolerance: float = 1e-8,
    wall_time_seconds: float = 600.0,
    callback: Callable[[int, int, torch.Tensor, torch.Tensor, float], None] | None = None,
) -> LBFGSResult:
    """Limited-memory BFGS with Armijo backtracking and projected accepted steps.

    The implementation deliberately exposes objective/gradient counts so the
    campaign can enforce a matched, preregistered evaluation budget.
    """

    if not variable.requires_grad:
        raise ValueError("variable must require gradients")
    if history_size < 1 or max_iterations < 1 or trust_rms <= 0:
        raise ValueError("invalid L-BFGS controls")

    def value_gradient() -> tuple[torch.Tensor, torch.Tensor]:
        if variable.grad is not None:
            variable.grad = None
        value = objective(variable)
        value.backward()
        return value.detach(), variable.grad.detach().clone()

    value, gradient = value_gradient()
    objective_evaluations = 1
    gradient_evaluations = 1
    accepted = 0
    failures = 0
    s_history: list[torch.Tensor] = []
    y_history: list[torch.Tensor] = []
    rho_history: list[torch.Tensor] = []
    if callback is not None:
        callback(objective_evaluations, gradient_evaluations, variable, gradient, float(value))

    started = time.monotonic()
    for _ in range(max_iterations):
        if time.monotonic() - started >= wall_time_seconds:
            break
        if objective_evaluations >= max_objective_evaluations or gradient_evaluations >= max_gradient_evaluations:
            break
        if float(torch.linalg.vector_norm(gradient).cpu()) <= tolerance:
            break
        q = gradient.clone()
        alphas: list[torch.Tensor] = []
        for s, y, rho in reversed(list(zip(s_history, y_history, rho_history))):
            alpha = rho * torch.dot(s.flatten(), q.flatten())
            alphas.append(alpha)
            q = q - alpha * y
        if s_history:
            sy = torch.dot(s_history[-1].flatten(), y_history[-1].flatten())
            yy = torch.dot(y_history[-1].flatten(), y_history[-1].flatten())
            q = q * torch.clamp(sy / torch.clamp(yy, min=1e-20), min=1e-6, max=1e6)
        for (s, y, rho), alpha in zip(zip(s_history, y_history, rho_history), reversed(alphas)):
            beta = rho * torch.dot(y.flatten(), q.flatten())
            q = q + s * (alpha - beta)
        direction = -q
        directional = torch.dot(gradient.flatten(), direction.flatten())
        if not bool(torch.isfinite(directional)) or float(directional.cpu()) >= 0:
            direction = -gradient
            directional = -torch.dot(gradient.flatten(), gradient.flatten())
            s_history.clear(); y_history.clear(); rho_history.clear()
        rms = torch.sqrt(direction.square().mean())
        if float(rms.cpu()) > trust_rms:
            direction = direction * (trust_rms / rms)
            directional = torch.dot(gradient.flatten(), direction.flatten())

        old = variable.detach().clone()
        old_value = value
        step_size = 1.0
        found = False
        for _trial in range(line_search_trials):
            if objective_evaluations >= max_objective_evaluations:
                break
            with torch.no_grad():
                variable.copy_(old + step_size * direction)
                project_(variable)
            trial_value = objective(variable).detach()
            objective_evaluations += 1
            if bool(torch.isfinite(trial_value)) and float(trial_value.cpu()) <= float((old_value + armijo_c1 * step_size * directional).cpu()):
                value = trial_value
                found = True
                break
            step_size *= line_search_shrink
        if not found:
            with torch.no_grad():
                variable.copy_(old)
            failures += 1
            break

        if gradient_evaluations >= max_gradient_evaluations:
            break
        _, new_gradient = value_gradient()
        objective_evaluations += 1
        gradient_evaluations += 1
        s = variable.detach() - old
        y = new_gradient - gradient
        sy = torch.dot(s.flatten(), y.flatten())
        if bool(torch.isfinite(sy)) and float(sy.cpu()) > 1e-12:
            if len(s_history) == history_size:
                s_history.pop(0); y_history.pop(0); rho_history.pop(0)
            s_history.append(s.clone()); y_history.append(y.clone()); rho_history.append(1.0 / sy)
        gradient = new_gradient
        accepted += 1
        if callback is not None:
            callback(objective_evaluations, gradient_evaluations, variable, gradient, float(value))

    return LBFGSResult(float(value.cpu()), objective_evaluations, gradient_evaluations, accepted, failures)
