"""Paired-observation extension of the frozen Model-9 structural solver.

One physical parameter vector is rendered through two independently declared
known PSFs.  The module deliberately has no catalog, isolated-source, or truth
input surface.  Parameter support, initialization, symmetry handling, and all
uniqueness thresholds are inherited unchanged from the authoritative
single-observation implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
import torch
from torch import Tensor

from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_optimizer import (
    LocalFitDiagnostics,
    MultiStartEndpoint,
    SolutionGeometry,
    analyze_jacobian,
    boundary_contact_flags,
    configure_determinism,
    deterministic_starts,
)
from src.model9_structured import (
    FrozenSolverProtocol,
    RenderedPair,
    SolverInputs,
    assert_no_oracle_information,
    canonicalize_parameters,
    expanded_noise_sigma,
    input_provenance_trace,
    likelihood_components,
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
class JointSolverInputs:
    """Exactly two observations sharing source parameters and coordinates."""

    observation_a: SolverInputs
    observation_2: SolverInputs
    condition: str

    def validate(self) -> None:
        if self.condition not in {"S2", "P2"}:
            raise ValueError("joint condition must be S2 or P2")
        self.observation_a.validate()
        self.observation_2.validate()
        left = self.observation_a
        right = self.observation_2
        if left.family != right.family:
            raise ValueError("both observations must use one structural family")
        if left.requested_center_xy != right.requested_center_xy:
            raise ValueError("requested coordinate differs across observations")
        if left.companion_center_xy != right.companion_center_xy:
            raise ValueError("companion coordinate differs across observations")
        if left.observed.shape != right.observed.shape:
            raise ValueError("observation geometry differs across observations")

    @property
    def family(self) -> str:
        return self.observation_a.family

    @property
    def reference(self) -> SolverInputs:
        """Observation A fixes starts/scales identically for S2 and P2."""

        return self.observation_a

    @property
    def observation_pixel_count(self) -> int:
        return int(self.observation_a.observed.numel() + self.observation_2.observed.numel())


@dataclass(frozen=True)
class JointObjectiveComponents:
    likelihood_total: Tensor
    likelihood_by_observation: Tensor
    likelihood_by_observation_band: Tensor
    chi_square: Tensor
    chi_square_by_observation: Tensor
    chi_square_by_observation_band: Tensor
    signed_residuals: tuple[Tensor, Tensor]


def render_joint(
    parameters: Tensor,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> tuple[RenderedPair, RenderedPair]:
    inputs.validate()
    return (
        render_pair(parameters, inputs.observation_a, protocol),
        render_pair(parameters, inputs.observation_2, protocol),
    )


def joint_likelihood_components(
    parameters: Tensor,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> JointObjectiveComponents:
    inputs.validate()
    parts = (
        likelihood_components(parameters, inputs.observation_a, protocol),
        likelihood_components(parameters, inputs.observation_2, protocol),
    )
    likelihood_by_band = torch.stack(tuple(part.likelihood_by_band for part in parts))
    chi_by_band = []
    for part, observation in zip(parts, (inputs.observation_a, inputs.observation_2)):
        sigma = expanded_noise_sigma(observation.noise_sigma, observation.observed).to(parameters)
        chi_by_band.append((part.signed_residual / sigma).square().sum(dim=(-2, -1)))
    chi_by_band_tensor = torch.stack(tuple(chi_by_band))
    return JointObjectiveComponents(
        likelihood_total=likelihood_by_band.sum(),
        likelihood_by_observation=likelihood_by_band.sum(dim=1),
        likelihood_by_observation_band=likelihood_by_band,
        chi_square=chi_by_band_tensor.sum(),
        chi_square_by_observation=chi_by_band_tensor.sum(dim=1),
        chi_square_by_observation_band=chi_by_band_tensor,
        signed_residuals=(parts[0].signed_residual, parts[1].signed_residual),
    )


def joint_whitened_residual_vector(
    parameters: Tensor,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> Tensor:
    inputs.validate()
    return torch.cat(
        (
            whitened_residual_vector(parameters, inputs.observation_a, protocol),
            whitened_residual_vector(parameters, inputs.observation_2, protocol),
        )
    )


def joint_residual_jacobian(
    parameters: Sequence[float] | Tensor,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> np.ndarray:
    value = torch.as_tensor(parameters, dtype=torch.float64, device="cpu").clone().requires_grad_(True)

    def function(candidate: Tensor) -> Tensor:
        return joint_whitened_residual_vector(candidate, inputs, protocol)

    try:
        jacobian = torch.autograd.functional.jacobian(
            function, value, vectorize=True, strategy="forward-mode"
        )
    except RuntimeError:
        jacobian = torch.autograd.functional.jacobian(
            function, value, vectorize=True, strategy="reverse-mode"
        )
    return np.asarray(jacobian.detach().cpu(), dtype=np.float64)


def joint_deterministic_starts(
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    count: int | None = None,
) -> np.ndarray:
    """Use Observation A so S2 and P2 receive byte-identical starts."""

    inputs.validate()
    return deterministic_starts(inputs.reference, protocol, count=count)


def joint_local_fit_diagnostics(
    parameters: Sequence[float],
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> LocalFitDiagnostics:
    value = np.asarray(parameters, dtype=np.float64)
    canonical, active, symmetries = canonicalize_parameters(value, inputs.reference, protocol)
    jacobian = joint_residual_jacobian(value, inputs, protocol)
    scales = parameter_scales(inputs.reference, protocol)
    diagnostics = analyze_jacobian(jacobian, active_mask=active, parameter_scales=scales)
    residual = joint_whitened_residual_vector(
        torch.as_tensor(value, dtype=torch.float64), inputs, protocol
    ).detach().cpu().numpy()
    gradient = (jacobian * scales[None]).T @ residual
    components = joint_likelihood_components(torch.as_tensor(value, dtype=torch.float64), inputs, protocol)
    return LocalFitDiagnostics(
        objective=float(components.likelihood_total.detach().cpu()),
        chi_square=float(components.chi_square.detach().cpu()),
        gradient_norm=float(np.linalg.norm(gradient)),
        jacobian=jacobian,
        jacobian_diagnostics=diagnostics,
        canonical_parameters=canonical,
        symmetries=symmetries,
    )


def _objective_numpy(
    parameters: np.ndarray,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> np.ndarray:
    return np.asarray(
        joint_whitened_residual_vector(
            torch.as_tensor(parameters, dtype=torch.float64), inputs, protocol
        ).detach().cpu(),
        dtype=np.float64,
    )


def _paired_tensor_hash(values: Sequence[Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"thayer-model9-joint-paired-tensor-v1\0")
    for value in values:
        digest.update(canonical_tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stacked_render_hash(pairs: tuple[RenderedPair, RenderedPair], field: str) -> str:
    return _paired_tensor_hash(tuple(getattr(pair, field) for pair in pairs))


def joint_multi_start_optimize(
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    starts: np.ndarray | None = None,
) -> list[MultiStartEndpoint]:
    inputs.validate()
    protocol.validate()
    configure_determinism(protocol.optimizer_seed)
    lower, upper = parameter_bounds(inputs.reference, protocol)
    start_values = (
        joint_deterministic_starts(inputs, protocol)
        if starts is None
        else np.asarray(starts, dtype=np.float64)
    )
    if start_values.ndim != 2 or start_values.shape[1] != lower.size:
        raise ValueError("starts have the wrong shape")
    endpoints: list[MultiStartEndpoint] = []
    for start_index, initialization in enumerate(start_values):
        validate_parameters(initialization, inputs.reference, protocol)
        result = least_squares(
            lambda value: _objective_numpy(value, inputs, protocol),
            initialization,
            jac=lambda value: joint_residual_jacobian(value, inputs, protocol),
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
        canonical, _, symmetries = canonicalize_parameters(selected, inputs.reference, protocol)
        tensor = torch.as_tensor(selected, dtype=torch.float64)
        components = joint_likelihood_components(tensor, inputs, protocol)
        pairs = render_joint(tensor, inputs, protocol)
        jacobian = joint_residual_jacobian(selected, inputs, protocol)
        residual = _objective_numpy(selected, inputs, protocol)
        scales = parameter_scales(inputs.reference, protocol)
        gradient_norm = float(np.linalg.norm((jacobian * scales[None]).T @ residual))
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
                requested_sha256=_stacked_render_hash(pairs, "requested"),
                companion_sha256=_stacked_render_hash(pairs, "companion"),
                recomposed_sha256=_paired_tensor_hash(
                    tuple(pair.recomposed_sources for pair in pairs)
                ),
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


def joint_solution_geometry(
    endpoints: Sequence[MultiStartEndpoint],
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> SolutionGeometry:
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    objectives = np.asarray([endpoint.likelihood_objective for endpoint in endpoints], dtype=np.float64)
    best = float(np.min(objectives))
    tolerance = protocol.objective_accept_atol + protocol.objective_accept_rtol * max(abs(best), 1.0)
    acceptable = tuple(int(i) for i, value in enumerate(objectives) if value <= best + tolerance)
    rendered: dict[int, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {}
    for index in acceptable:
        pairs = render_joint(torch.as_tensor(endpoints[index].parameters, dtype=torch.float64), inputs, protocol)
        rendered[index] = tuple(
            (
                np.asarray(pair.requested.detach().cpu(), dtype=np.float64),
                np.asarray(pair.companion.detach().cpu(), dtype=np.float64),
            )
            for pair in pairs
        )  # type: ignore[assignment]
    representatives: list[int] = []
    classes: dict[int, int] = {}
    for index in acceptable:
        assigned = None
        for class_index, representative in enumerate(representatives):
            distances = []
            for observation_index in (0, 1):
                distances.extend(
                    (
                        _relative_image_distance(
                            rendered[index][observation_index][0],
                            rendered[representative][observation_index][0],
                        ),
                        _relative_image_distance(
                            rendered[index][observation_index][1],
                            rendered[representative][observation_index][1],
                        ),
                    )
                )
            if max(distances) <= protocol.endpoint_image_rtol:
                assigned = class_index
                break
        if assigned is None:
            representatives.append(index)
            assigned = len(representatives) - 1
        classes[index] = assigned
    requested_diameter = companion_diameter = flux_diameter = morphology_diameter = 0.0
    scales = parameter_scales(inputs.reference, protocol)
    per_source = parameters_per_source(inputs.family)
    flux_indices = np.asarray((0, 1, 2, per_source, per_source + 1, per_source + 2))
    flux_scale = np.tile(scales[:3], 2)
    morphology_indices = np.asarray(
        [index for index in range(2 * per_source) if index not in set(flux_indices)]
    )
    for left_position, left_index in enumerate(acceptable):
        for right_index in acceptable[left_position + 1 :]:
            for observation_index in (0, 1):
                requested_diameter = max(
                    requested_diameter,
                    _relative_image_distance(
                        rendered[left_index][observation_index][0],
                        rendered[right_index][observation_index][0],
                    ),
                )
                companion_diameter = max(
                    companion_diameter,
                    _relative_image_distance(
                        rendered[left_index][observation_index][1],
                        rendered[right_index][observation_index][1],
                    ),
                )
            left = endpoints[left_index].canonical_parameters
            right = endpoints[right_index].canonical_parameters
            flux_diameter = max(
                flux_diameter,
                float(np.linalg.norm((left[flux_indices] - right[flux_indices]) / flux_scale)),
            )
            morphology_diameter = max(
                morphology_diameter,
                float(np.linalg.norm((left[morphology_indices] - right[morphology_indices]) / scales[morphology_indices])),
            )
    identity_consistent = True
    requested_center = np.asarray(inputs.reference.requested_center_xy, dtype=np.float64)
    companion_center = np.asarray(inputs.reference.companion_center_xy, dtype=np.float64)
    for index in acceptable:
        for observation_index in (0, 1):
            req_centroid = _centroid(rendered[index][observation_index][0])
            comp_centroid = _centroid(rendered[index][observation_index][1])
            if req_centroid is None or comp_centroid is None:
                identity_consistent = False
                continue
            identity_consistent &= bool(
                np.linalg.norm(req_centroid - requested_center)
                <= np.linalg.norm(req_centroid - companion_center)
                and np.linalg.norm(comp_centroid - companion_center)
                <= np.linalg.norm(comp_centroid - requested_center)
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


def joint_structural_model_is_acceptable(
    endpoint: MultiStartEndpoint,
    diagnostics: LocalFitDiagnostics,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> tuple[bool, float, int]:
    degrees_of_freedom = max(
        1, inputs.observation_pixel_count - diagnostics.jacobian_diagnostics.active_parameter_count
    )
    threshold = float(chi2.ppf(protocol.model_acceptance_quantile, degrees_of_freedom))
    return bool(endpoint.chi_square <= threshold), threshold, degrees_of_freedom


def classify_joint_identifiability(
    endpoints: Sequence[MultiStartEndpoint],
    geometry: SolutionGeometry | None,
    diagnostics: LocalFitDiagnostics | None,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    contract_valid: bool = True,
    numerically_stable: bool = True,
) -> str:
    if not contract_valid:
        return "INVALID_CONTRACT"
    successful = [endpoint for endpoint in endpoints if endpoint.success]
    if not successful or geometry is None or diagnostics is None:
        return "OPTIMIZATION_UNRESOLVED"
    if not numerically_stable or not np.isfinite(diagnostics.gradient_norm):
        return "NUMERICALLY_UNSTABLE"
    best = min(successful, key=lambda endpoint: endpoint.likelihood_objective)
    acceptable, _, _ = joint_structural_model_is_acceptable(best, diagnostics, inputs, protocol)
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
    boundary = boundary_contact_flags(best.parameters, inputs.reference, protocol)
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
    return "UNIQUE" if strict else "NEAR_UNIQUE"


def joint_input_provenance_trace(
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
) -> list[dict[str, Any]]:
    inputs.validate()
    trace: list[dict[str, Any]] = []
    for label, observation in (
        ("observation_a", inputs.observation_a),
        ("observation_2", inputs.observation_2),
    ):
        for entry in input_provenance_trace(observation, protocol):
            item = dict(entry)
            item["input"] = f"{label}.{item['input']}"
            if item["provenance"] == "frozen_blended_observation":
                item["provenance"] = (
                    "authoritative_original_blended_observation"
                    if label == "observation_a"
                    else "authoritative_forward_simulator_paired_observation"
                )
            trace.append(item)
    trace.append(
        {
            "input": "shared_parameter_constraint",
            "provenance": "one_parameter_vector_rendered_through_both_known_psfs",
            "sha256": hashlib.sha256(b"model9-joint-shared-parameters-v1").hexdigest(),
            "oracle": False,
        }
    )
    return trace


def joint_oracle_information_audit(
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    extra_named_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extras = {} if extra_named_inputs is None else dict(extra_named_inputs)
    try:
        assert_no_oracle_information(extras)
        trace = joint_input_provenance_trace(inputs, protocol)
    except ValueError as error:
        return {"status": "FAIL", "reason": str(error), "trace": []}
    return {
        "status": "PASS",
        "reason": "both inference observations are permitted and one shared parameter vector is enforced",
        "trace": trace,
        "extra_input_names": sorted(extras),
    }


def deterministic_joint_replay(
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    starts: np.ndarray,
) -> dict[str, Any]:
    first = joint_multi_start_optimize(inputs, protocol, starts=starts)
    second = joint_multi_start_optimize(inputs, protocol, starts=starts)
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
