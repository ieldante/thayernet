"""Deterministic optimizer and identifiability diagnostics for Model 9."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
from scipy.stats import qmc
from scipy.stats import chi2
import torch
from torch import Tensor

from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_structured import (
    FAMILY_BULGE_DISK,
    FAMILY_SERSIC,
    FrozenSolverProtocol,
    SolverInputs,
    canonicalize_parameters,
    likelihood_components,
    observed_flux_reference,
    parameter_bounds,
    parameter_names,
    parameter_scales,
    parameter_sha256,
    parameters_per_source,
    render_pair,
    validate_parameters,
    whitened_residual_vector,
)


@dataclass(frozen=True)
class JacobianDiagnostics:
    singular_values: np.ndarray
    rank: int
    active_parameter_count: int
    null_space_dimension: int
    null_space_basis: np.ndarray
    condition_number: float
    gauss_newton_hessian: np.ndarray
    hessian_eigenvalues: np.ndarray
    hessian_condition_number: float
    rank_tolerance: float
    active_mask: np.ndarray


@dataclass(frozen=True)
class LocalFitDiagnostics:
    objective: float
    chi_square: float
    gradient_norm: float
    jacobian: np.ndarray
    jacobian_diagnostics: JacobianDiagnostics
    canonical_parameters: np.ndarray
    symmetries: tuple[str, ...]


@dataclass(frozen=True)
class MultiStartEndpoint:
    start_index: int
    initialization: np.ndarray
    parameters: np.ndarray
    canonical_parameters: np.ndarray
    initialization_sha256: str
    parameter_sha256: str
    success: bool
    status: int
    message: str
    nfev: int
    njev: int | None
    cost: float
    likelihood_objective: float
    chi_square: float
    optimality: float
    gradient_norm: float
    requested_sha256: str
    companion_sha256: str
    recomposed_sha256: str
    symmetries: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "initialization": self.initialization.tolist(),
            "parameters": self.parameters.tolist(),
            "canonical_parameters": self.canonical_parameters.tolist(),
            "initialization_sha256": self.initialization_sha256,
            "parameter_sha256": self.parameter_sha256,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "nfev": self.nfev,
            "njev": self.njev,
            "cost": self.cost,
            "likelihood_objective": self.likelihood_objective,
            "chi_square": self.chi_square,
            "optimality": self.optimality,
            "gradient_norm": self.gradient_norm,
            "requested_sha256": self.requested_sha256,
            "companion_sha256": self.companion_sha256,
            "recomposed_sha256": self.recomposed_sha256,
            "symmetries": list(self.symmetries),
        }


@dataclass(frozen=True)
class SolutionGeometry:
    acceptable_endpoint_indices: tuple[int, ...]
    solution_class_by_endpoint: dict[int, int]
    distinct_solution_classes: int
    requested_image_diameter: float
    companion_image_diameter: float
    flux_allocation_diameter: float
    morphology_parameter_diameter: float
    prompt_identity_consistent: bool


CLASSIFICATIONS = (
    "UNIQUE",
    "NEAR_UNIQUE",
    "PARTIALLY_IDENTIFIABLE",
    "NON_IDENTIFIABLE",
    "OUT_OF_SUPPORT",
    "OPTIMIZATION_UNRESOLVED",
    "NUMERICALLY_UNSTABLE",
    "INVALID_CONTRACT",
)


def configure_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def residual_jacobian(
    parameters: Sequence[float] | Tensor,
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> np.ndarray:
    """Float64 autograd Jacobian of whitened signed residuals."""

    value = torch.as_tensor(parameters, dtype=torch.float64, device="cpu").clone().requires_grad_(True)

    def function(candidate: Tensor) -> Tensor:
        return whitened_residual_vector(candidate, inputs, protocol)

    try:
        jacobian = torch.autograd.functional.jacobian(
            function,
            value,
            vectorize=True,
            strategy="forward-mode",
        )
    except RuntimeError:
        jacobian = torch.autograd.functional.jacobian(
            function,
            value,
            vectorize=True,
            strategy="reverse-mode",
        )
    return np.asarray(jacobian.detach().cpu(), dtype=np.float64)


def analyze_jacobian(
    jacobian: np.ndarray,
    *,
    active_mask: Sequence[bool] | None = None,
    parameter_scales: Sequence[float] | None = None,
) -> JacobianDiagnostics:
    """Symmetry-corrected SVD, null space, condition, and GN Hessian."""

    full = np.asarray(jacobian, dtype=np.float64)
    if full.ndim != 2 or not np.isfinite(full).all():
        raise ValueError("jacobian must be a finite matrix")
    parameter_count = full.shape[1]
    mask = (
        np.ones(parameter_count, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    if mask.shape != (parameter_count,):
        raise ValueError("active mask has wrong shape")
    active = full[:, mask]
    if parameter_scales is None:
        scales = np.ones(parameter_count, dtype=np.float64)
    else:
        scales = np.asarray(parameter_scales, dtype=np.float64)
        if scales.shape != (parameter_count,) or np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ValueError("parameter scales must be finite and positive")
    scaled = active * scales[mask][None]
    if scaled.shape[1] == 0:
        singular_values = np.empty(0, dtype=np.float64)
        rank = 0
        tolerance = 0.0
        active_null = np.empty((0, 0), dtype=np.float64)
        condition = 1.0
    else:
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        maximum = float(singular_values[0]) if singular_values.size else 0.0
        tolerance = max(scaled.shape) * np.finfo(np.float64).eps * maximum
        rank = int(np.sum(singular_values > tolerance))
        active_null = null_space(scaled, rcond=(tolerance / maximum if maximum > 0 else None)).T
        condition = (
            float(singular_values[0] / singular_values[-1])
            if rank == scaled.shape[1] and singular_values[-1] > 0
            else math.inf
        )
    null_dimension = int(scaled.shape[1] - rank)
    if active_null.shape != (null_dimension, scaled.shape[1]):
        raise RuntimeError("null-space implementation returned an inconsistent basis")
    expanded_null = np.zeros((null_dimension, parameter_count), dtype=np.float64)
    if null_dimension:
        expanded_null[:, mask] = active_null * scales[mask][None]
        norms = np.linalg.norm(expanded_null, axis=1, keepdims=True)
        expanded_null /= np.maximum(norms, np.finfo(np.float64).tiny)
    hessian = scaled.T @ scaled
    eigenvalues = np.linalg.eigvalsh(hessian) if hessian.size else np.empty(0, dtype=np.float64)
    positive = eigenvalues[eigenvalues > tolerance**2]
    hessian_condition = (
        float(positive[-1] / positive[0])
        if null_dimension == 0 and positive.size == scaled.shape[1] and positive.size
        else (1.0 if scaled.shape[1] == 0 else math.inf)
    )
    return JacobianDiagnostics(
        singular_values=singular_values,
        rank=rank,
        active_parameter_count=int(mask.sum()),
        null_space_dimension=null_dimension,
        null_space_basis=expanded_null,
        condition_number=condition,
        gauss_newton_hessian=hessian,
        hessian_eigenvalues=eigenvalues,
        hessian_condition_number=hessian_condition,
        rank_tolerance=tolerance,
        active_mask=mask,
    )


def local_fit_diagnostics(
    parameters: Sequence[float],
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> LocalFitDiagnostics:
    value = np.asarray(parameters, dtype=np.float64)
    canonical, active, symmetries = canonicalize_parameters(value, inputs, protocol)
    jacobian = residual_jacobian(value, inputs, protocol)
    scales = parameter_scales(inputs, protocol)
    diagnostics = analyze_jacobian(jacobian, active_mask=active, parameter_scales=scales)
    tensor = torch.as_tensor(value, dtype=torch.float64)
    residual = whitened_residual_vector(tensor, inputs, protocol).detach().cpu().numpy()
    gradient = (jacobian * scales[None]).T @ residual
    components = likelihood_components(tensor, inputs, protocol)
    return LocalFitDiagnostics(
        objective=float(components.likelihood_total.detach().cpu()),
        chi_square=float(components.chi_square.detach().cpu()),
        gradient_norm=float(np.linalg.norm(gradient)),
        jacobian=jacobian,
        jacobian_diagnostics=diagnostics,
        canonical_parameters=canonical,
        symmetries=symmetries,
    )


def finite_difference_directional_check(
    parameters: Sequence[float],
    direction: Sequence[float],
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    step: float = 1.0e-6,
) -> dict[str, float]:
    value = np.asarray(parameters, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    if value.shape != vector.shape or not np.isfinite(vector).all():
        raise ValueError("direction must be finite and match parameters")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("direction must be nonzero")
    vector /= norm
    validate_parameters(value + step * vector, inputs, protocol, tolerance=0.0)
    validate_parameters(value - step * vector, inputs, protocol, tolerance=0.0)
    jacobian = residual_jacobian(value, inputs, protocol)
    analytic = jacobian @ vector
    plus = whitened_residual_vector(torch.as_tensor(value + step * vector, dtype=torch.float64), inputs, protocol)
    minus = whitened_residual_vector(torch.as_tensor(value - step * vector, dtype=torch.float64), inputs, protocol)
    finite = ((plus - minus) / (2.0 * step)).detach().cpu().numpy()
    absolute = float(np.linalg.norm(analytic - finite))
    relative = absolute / max(float(np.linalg.norm(analytic)), float(np.linalg.norm(finite)), 1.0e-15)
    return {
        "absolute_error": absolute,
        "relative_error": relative,
        "analytic_norm": float(np.linalg.norm(analytic)),
        "finite_difference_norm": float(np.linalg.norm(finite)),
    }


def deterministic_starts(
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    count: int | None = None,
) -> np.ndarray:
    """Observation-only Sobol schedule spanning flux allocation and morphology."""

    count = protocol.starts_per_family if count is None else int(count)
    if count < 1:
        raise ValueError("at least one start is required")
    lower, upper = parameter_bounds(inputs, protocol)
    dimension = lower.size
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=protocol.optimizer_seed)
    if count & (count - 1) == 0:
        unit = sampler.random_base2(m=int(math.log2(count)))
    else:
        unit = sampler.random(n=count)
    finite_upper = upper.copy()
    per_source = parameters_per_source(inputs.family)
    reference = observed_flux_reference(inputs).detach().cpu().numpy().astype(np.float64)
    finite_upper[:3] = 2.0 * reference
    finite_upper[per_source : per_source + 3] = 2.0 * reference
    starts = lower[None] + unit * (finite_upper - lower)[None]
    fractions = np.asarray((0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95), dtype=np.float64)
    for start_index in range(count):
        for band in range(3):
            fraction = fractions[(start_index + 3 * band) % len(fractions)]
            starts[start_index, band] = fraction * reference[band]
            starts[start_index, per_source + band] = (1.0 - fraction) * reference[band]
    if inputs.family == FAMILY_SERSIC:
        n_patterns = ((1.0, 1.0), (1.0, 4.0), (4.0, 1.0), (4.0, 4.0))
        for start_index in range(count):
            left_n, right_n = n_patterns[start_index % len(n_patterns)]
            starts[start_index, 3] = left_n
            starts[start_index, per_source + 3] = right_n
    else:
        bt_patterns = (
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.1, 0.1),
            (0.5, 0.5),
            (0.1, 0.5),
            (0.5, 0.1),
        )
        for start_index in range(min(count, len(bt_patterns))):
            left_bt, right_bt = bt_patterns[start_index]
            starts[start_index, 9:12] = left_bt
            starts[start_index, per_source + 9 : per_source + 12] = right_bt
    if not np.all(starts >= lower[None]) or not np.all(starts <= upper[None]):
        raise RuntimeError("deterministic start escaped frozen bounds")
    return starts


def _objective_numpy(parameters: np.ndarray, inputs: SolverInputs, protocol: FrozenSolverProtocol) -> np.ndarray:
    tensor = torch.as_tensor(parameters, dtype=torch.float64)
    return np.asarray(
        whitened_residual_vector(tensor, inputs, protocol).detach().cpu(),
        dtype=np.float64,
    )


def multi_start_optimize(
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    starts: np.ndarray | None = None,
) -> list[MultiStartEndpoint]:
    """Run every deterministic bounded start and retain favorable or unfavorable results."""

    inputs.validate()
    protocol.validate()
    configure_determinism(protocol.optimizer_seed)
    lower, upper = parameter_bounds(inputs, protocol)
    start_values = deterministic_starts(inputs, protocol) if starts is None else np.asarray(starts, dtype=np.float64)
    if start_values.ndim != 2 or start_values.shape[1] != lower.size:
        raise ValueError("starts have the wrong shape")
    endpoints: list[MultiStartEndpoint] = []
    for start_index, initialization in enumerate(start_values):
        validate_parameters(initialization, inputs, protocol)

        def function(value: np.ndarray) -> np.ndarray:
            return _objective_numpy(value, inputs, protocol)

        def jacobian(value: np.ndarray) -> np.ndarray:
            return residual_jacobian(value, inputs, protocol)

        result = least_squares(
            function,
            initialization,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=protocol.ftol,
            xtol=protocol.xtol,
            gtol=protocol.gtol,
            max_nfev=protocol.max_nfev,
            verbose=0,
        )
        selected = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
        canonical, _, symmetries = canonicalize_parameters(selected, inputs, protocol)
        tensor = torch.as_tensor(selected, dtype=torch.float64)
        components = likelihood_components(tensor, inputs, protocol)
        pair = render_pair(tensor, inputs, protocol)
        local_jacobian = residual_jacobian(selected, inputs, protocol)
        residual = _objective_numpy(selected, inputs, protocol)
        scales = parameter_scales(inputs, protocol)
        gradient_norm = float(np.linalg.norm((local_jacobian * scales[None]).T @ residual))
        endpoints.append(
            MultiStartEndpoint(
                start_index=start_index,
                initialization=np.asarray(initialization, dtype=np.float64).copy(),
                parameters=selected,
                canonical_parameters=canonical,
                initialization_sha256=parameter_sha256(initialization),
                parameter_sha256=parameter_sha256(canonical),
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                nfev=int(result.nfev),
                njev=(None if result.njev is None else int(result.njev)),
                cost=float(result.cost),
                likelihood_objective=float(components.likelihood_total.detach().cpu()),
                chi_square=float(components.chi_square.detach().cpu()),
                optimality=float(result.optimality),
                gradient_norm=gradient_norm,
                requested_sha256=canonical_tensor_sha256(pair.requested),
                companion_sha256=canonical_tensor_sha256(pair.companion),
                recomposed_sha256=canonical_tensor_sha256(pair.recomposed_sources),
                symmetries=symmetries,
            )
        )
    return endpoints


def _relative_image_distance(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(left - right) / scale)


def _centroid(image: np.ndarray) -> np.ndarray | None:
    weight = np.maximum(np.asarray(image, dtype=np.float64).sum(axis=0), 0.0)
    total = float(weight.sum())
    if total <= 0:
        return None
    yy, xx = np.indices(weight.shape, dtype=np.float64)
    return np.asarray((np.sum(weight * xx) / total, np.sum(weight * yy) / total))


def solution_geometry(
    endpoints: Sequence[MultiStartEndpoint],
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> SolutionGeometry:
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    objectives = np.asarray([endpoint.likelihood_objective for endpoint in endpoints], dtype=np.float64)
    best = float(np.min(objectives))
    tolerance = protocol.objective_accept_atol + protocol.objective_accept_rtol * max(abs(best), 1.0)
    acceptable = tuple(int(i) for i, value in enumerate(objectives) if value <= best + tolerance)
    rendered: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in acceptable:
        pair = render_pair(torch.as_tensor(endpoints[index].parameters, dtype=torch.float64), inputs, protocol)
        rendered[index] = (
            np.asarray(pair.requested.detach().cpu(), dtype=np.float64),
            np.asarray(pair.companion.detach().cpu(), dtype=np.float64),
        )
    representatives: list[int] = []
    classes: dict[int, int] = {}
    for index in acceptable:
        assigned = None
        for class_index, representative in enumerate(representatives):
            req_distance = _relative_image_distance(rendered[index][0], rendered[representative][0])
            comp_distance = _relative_image_distance(rendered[index][1], rendered[representative][1])
            if max(req_distance, comp_distance) <= protocol.endpoint_image_rtol:
                assigned = class_index
                break
        if assigned is None:
            representatives.append(index)
            assigned = len(representatives) - 1
        classes[index] = assigned
    requested_diameter = 0.0
    companion_diameter = 0.0
    flux_diameter = 0.0
    morphology_diameter = 0.0
    scales = parameter_scales(inputs, protocol)
    per_source = parameters_per_source(inputs.family)
    flux_scale = np.tile(observed_flux_reference(inputs).detach().cpu().numpy(), 2)
    for left_position, left_index in enumerate(acceptable):
        for right_index in acceptable[left_position + 1 :]:
            requested_diameter = max(requested_diameter, _relative_image_distance(rendered[left_index][0], rendered[right_index][0]))
            companion_diameter = max(companion_diameter, _relative_image_distance(rendered[left_index][1], rendered[right_index][1]))
            left_parameters = endpoints[left_index].canonical_parameters
            right_parameters = endpoints[right_index].canonical_parameters
            flux_indices = np.asarray((0, 1, 2, per_source, per_source + 1, per_source + 2))
            flux_diameter = max(
                flux_diameter,
                float(np.linalg.norm((left_parameters[flux_indices] - right_parameters[flux_indices]) / np.maximum(flux_scale, np.finfo(np.float64).tiny))),
            )
            morphology_indices = np.asarray([index for index in range(2 * per_source) if index not in set(flux_indices)])
            morphology_diameter = max(
                morphology_diameter,
                float(np.linalg.norm((left_parameters[morphology_indices] - right_parameters[morphology_indices]) / scales[morphology_indices])),
            )
    identity_consistent = True
    requested_center = np.asarray(inputs.requested_center_xy, dtype=np.float64)
    companion_center = np.asarray(inputs.companion_center_xy, dtype=np.float64)
    for index in acceptable:
        req_centroid = _centroid(rendered[index][0])
        comp_centroid = _centroid(rendered[index][1])
        if req_centroid is None or comp_centroid is None:
            identity_consistent = False
            continue
        identity_consistent &= bool(
            np.linalg.norm(req_centroid - requested_center) <= np.linalg.norm(req_centroid - companion_center)
            and np.linalg.norm(comp_centroid - companion_center) <= np.linalg.norm(comp_centroid - requested_center)
        )
    return SolutionGeometry(
        acceptable_endpoint_indices=acceptable,
        solution_class_by_endpoint=classes,
        distinct_solution_classes=len(representatives),
        requested_image_diameter=requested_diameter,
        companion_image_diameter=companion_diameter,
        flux_allocation_diameter=flux_diameter,
        morphology_parameter_diameter=morphology_diameter,
        prompt_identity_consistent=identity_consistent,
    )


def deterministic_replay(
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    starts: np.ndarray,
) -> dict[str, Any]:
    first = multi_start_optimize(inputs, protocol, starts=starts)
    second = multi_start_optimize(inputs, protocol, starts=starts)
    first_records = [endpoint.record() for endpoint in first]
    second_records = [endpoint.record() for endpoint in second]
    first_payload = json.dumps(first_records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    second_payload = json.dumps(second_records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    first_hash = hashlib.sha256(first_payload.encode("utf-8")).hexdigest()
    second_hash = hashlib.sha256(second_payload.encode("utf-8")).hexdigest()
    return {
        "status": "PASS" if first_hash == second_hash else "FAIL",
        "first_sha256": first_hash,
        "second_sha256": second_hash,
        "endpoint_count": len(first),
        "exact_record_match": first_records == second_records,
    }


def boundary_contact_flags(
    parameters: Sequence[float],
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> dict[str, Any]:
    """Report physical-bound contacts and invalid zero-source collapse."""

    value = np.asarray(parameters, dtype=np.float64)
    lower, upper = parameter_bounds(inputs, protocol)
    scales = parameter_scales(inputs, protocol)
    names = parameter_names(inputs.family)
    tolerance = 1.0e-8 * scales
    lower_contacts = [names[index] for index in range(value.size) if value[index] <= lower[index] + tolerance[index]]
    upper_contacts = [
        names[index]
        for index in range(value.size)
        if np.isfinite(upper[index]) and value[index] >= upper[index] - tolerance[index]
    ]
    per_source = parameters_per_source(inputs.family)
    reference_total = float(np.sum(observed_flux_reference(inputs).detach().cpu().numpy()))
    threshold = protocol.invalid_zero_flux_fraction * max(reference_total, np.finfo(np.float64).tiny)
    source_fluxes = (float(np.sum(value[:3])), float(np.sum(value[per_source : per_source + 3])))
    return {
        "lower_contacts": lower_contacts,
        "upper_contacts": upper_contacts,
        "requested_total_flux": source_fluxes[0],
        "companion_total_flux": source_fluxes[1],
        "invalid_zero_flux_collapse": bool(source_fluxes[0] <= threshold or source_fluxes[1] <= threshold),
        "zero_flux_threshold": threshold,
    }


def structural_model_is_acceptable(
    endpoint: MultiStartEndpoint,
    diagnostics: LocalFitDiagnostics,
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> tuple[bool, float, int]:
    observation_count = int(inputs.observed.numel())
    degrees_of_freedom = max(
        1,
        observation_count - diagnostics.jacobian_diagnostics.active_parameter_count,
    )
    threshold = float(chi2.ppf(protocol.model_acceptance_quantile, degrees_of_freedom))
    return bool(endpoint.chi_square <= threshold), threshold, degrees_of_freedom


def classify_identifiability(
    endpoints: Sequence[MultiStartEndpoint],
    geometry: SolutionGeometry | None,
    diagnostics: LocalFitDiagnostics | None,
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    contract_valid: bool = True,
    numerically_stable: bool = True,
) -> str:
    """Apply the frozen mutually exclusive flux-free classification rules."""

    if not contract_valid:
        return "INVALID_CONTRACT"
    successful = [endpoint for endpoint in endpoints if endpoint.success]
    if not successful or geometry is None or diagnostics is None:
        return "OPTIMIZATION_UNRESOLVED"
    if not numerically_stable or not np.isfinite(diagnostics.gradient_norm):
        return "NUMERICALLY_UNSTABLE"
    best = min(successful, key=lambda endpoint: endpoint.likelihood_objective)
    acceptable, _, _ = structural_model_is_acceptable(best, diagnostics, inputs, protocol)
    if not acceptable:
        return "OUT_OF_SUPPORT"
    local = diagnostics.jacobian_diagnostics
    image_diameter = max(geometry.requested_image_diameter, geometry.companion_image_diameter)
    allocation_distinct = (
        image_diameter > protocol.unique_image_diameter
        or geometry.flux_allocation_diameter > protocol.unique_flux_allocation_diameter
        or geometry.morphology_parameter_diameter > protocol.unique_morphology_diameter
    )
    multiple = geometry.distinct_solution_classes > 1 or local.null_space_dimension > 0
    if multiple:
        return "NON_IDENTIFIABLE" if allocation_distinct else "PARTIALLY_IDENTIFIABLE"
    boundary = boundary_contact_flags(best.parameters, inputs, protocol)
    strict = (
        local.rank == local.active_parameter_count
        and local.null_space_dimension == 0
        and local.condition_number <= protocol.maximum_condition_number
        and diagnostics.gradient_norm <= protocol.acceptable_gradient_norm
        and image_diameter <= protocol.unique_image_diameter
        and geometry.flux_allocation_diameter <= protocol.unique_flux_allocation_diameter
        and geometry.morphology_parameter_diameter <= protocol.unique_morphology_diameter
        and geometry.prompt_identity_consistent
        and not boundary["invalid_zero_flux_collapse"]
    )
    if strict:
        return "UNIQUE"
    return "NEAR_UNIQUE"
