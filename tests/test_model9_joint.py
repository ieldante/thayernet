"""Synthetic-only compatibility gates for paired-observation Model 9."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model9_joint import (
    JointSolverInputs,
    classify_joint_identifiability,
    joint_deterministic_starts,
    joint_input_provenance_trace,
    joint_likelihood_components,
    joint_local_fit_diagnostics,
    joint_multi_start_optimize,
    joint_oracle_information_audit,
    joint_residual_jacobian,
    joint_solution_geometry,
    joint_whitened_residual_vector,
)
from src.model9_optimizer import residual_jacobian
from src.model9_structured import (
    FrozenSolverProtocol,
    SolverInputs,
    gaussian_psf_kernels,
    likelihood_components,
    render_pair,
)
from src.model9_synthetic import separated_sersic_fixture


def paired_fixture(*, diverse: bool, signed_noise: float = 0.0) -> tuple[JointSolverInputs, np.ndarray]:
    protocol = FrozenSolverProtocol()
    fixture = separated_sersic_fixture(protocol, shape=(17, 17), signed_noise_amplitude=0.0)
    first = fixture.inputs
    psf = gaussian_psf_kernels((3.8, 2.7, 1.7), size=9) if diverse else first.psf.clone()
    provisional = SolverInputs(
        observed=torch.zeros_like(first.observed),
        requested_center_xy=first.requested_center_xy,
        companion_center_xy=first.companion_center_xy,
        psf=psf,
        noise_sigma=first.noise_sigma,
        family=first.family,
    )
    pair = render_pair(torch.as_tensor(fixture.generating_parameters), provisional, protocol)
    yy, xx = torch.meshgrid(
        torch.arange(17, dtype=torch.float64),
        torch.arange(17, dtype=torch.float64),
        indexing="ij",
    )
    pattern = torch.stack((torch.sin(xx), torch.cos(yy), torch.sin(xx - yy)))
    second = SolverInputs(
        observed=pair.recomposed_sources.detach() + signed_noise * pattern,
        requested_center_xy=first.requested_center_xy,
        companion_center_xy=first.companion_center_xy,
        psf=psf,
        noise_sigma=first.noise_sigma,
        family=first.family,
    )
    return JointSolverInputs(first, second, "P2" if diverse else "S2"), fixture.generating_parameters


def test_joint_inputs_enforce_shared_coordinates_and_family() -> None:
    inputs, _ = paired_fixture(diverse=True)
    inputs.validate()
    bad = SolverInputs(
        observed=inputs.observation_2.observed,
        requested_center_xy=(1.0, 2.0),
        companion_center_xy=inputs.observation_2.companion_center_xy,
        psf=inputs.observation_2.psf,
        noise_sigma=inputs.observation_2.noise_sigma,
        family=inputs.observation_2.family,
    )
    with pytest.raises(ValueError, match="requested coordinate"):
        JointSolverInputs(inputs.observation_a, bad, "P2").validate()


def test_joint_objective_is_exact_sum_of_observation_objectives() -> None:
    inputs, parameters = paired_fixture(diverse=True, signed_noise=0.02)
    tensor = torch.as_tensor(parameters)
    joint = joint_likelihood_components(tensor, inputs, FrozenSolverProtocol())
    separate = torch.stack(
        (
            likelihood_components(tensor, inputs.observation_a, FrozenSolverProtocol()).likelihood_total,
            likelihood_components(tensor, inputs.observation_2, FrozenSolverProtocol()).likelihood_total,
        )
    )
    assert torch.equal(joint.likelihood_by_observation, separate)
    torch.testing.assert_close(joint.likelihood_total, separate.sum(), rtol=1e-15, atol=1e-12)
    assert joint.likelihood_by_observation_band.shape == (2, 3)


def test_joint_residual_uses_one_parameter_vector_not_independent_parameters() -> None:
    inputs, parameters = paired_fixture(diverse=True)
    protocol = FrozenSolverProtocol()
    residual = joint_whitened_residual_vector(torch.as_tensor(parameters), inputs, protocol)
    jacobian = joint_residual_jacobian(parameters, inputs, protocol)
    assert residual.numel() == 2 * inputs.observation_a.observed.numel()
    assert jacobian.shape == (residual.numel(), parameters.size)
    assert jacobian.shape[1] == 14


def test_identical_psf_duplicate_adds_no_noise_free_algebraic_rank() -> None:
    inputs, parameters = paired_fixture(diverse=False)
    protocol = FrozenSolverProtocol()
    single = residual_jacobian(parameters, inputs.observation_a, protocol)
    joint = joint_residual_jacobian(parameters, inputs, protocol)
    np.testing.assert_allclose(joint[: single.shape[0]], single, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(joint[single.shape[0] :], single, rtol=0.0, atol=0.0)
    assert np.linalg.matrix_rank(joint) == np.linalg.matrix_rank(single)
    np.testing.assert_allclose(
        np.linalg.svd(joint, compute_uv=False),
        np.sqrt(2.0) * np.linalg.svd(single, compute_uv=False),
        rtol=5e-12,
        atol=1e-12,
    )


def test_diverse_psf_changes_second_forward_operator_block() -> None:
    inputs, parameters = paired_fixture(diverse=True)
    jacobian = joint_residual_jacobian(parameters, inputs, FrozenSolverProtocol())
    split = inputs.observation_a.observed.numel()
    assert not np.allclose(jacobian[:split], jacobian[split:], rtol=1e-8, atol=1e-10)


def test_same_starts_are_used_for_s2_and_p2() -> None:
    same, _ = paired_fixture(diverse=False)
    diverse, _ = paired_fixture(diverse=True)
    protocol = FrozenSolverProtocol()
    np.testing.assert_array_equal(
        joint_deterministic_starts(same, protocol),
        joint_deterministic_starts(diverse, protocol),
    )


def test_joint_autograd_matches_finite_difference_direction() -> None:
    inputs, parameters = paired_fixture(diverse=True, signed_noise=0.01)
    protocol = FrozenSolverProtocol()
    direction = np.sin(np.arange(parameters.size) + 1.0)
    direction /= np.linalg.norm(direction)
    step = 1e-6
    analytic = joint_residual_jacobian(parameters, inputs, protocol) @ direction
    plus = joint_whitened_residual_vector(torch.as_tensor(parameters + step * direction), inputs, protocol)
    minus = joint_whitened_residual_vector(torch.as_tensor(parameters - step * direction), inputs, protocol)
    finite = ((plus - minus) / (2.0 * step)).numpy()
    relative = np.linalg.norm(analytic - finite) / max(np.linalg.norm(analytic), np.linalg.norm(finite))
    assert relative < 2e-7


def test_joint_oracle_audit_and_trace() -> None:
    inputs, _ = paired_fixture(diverse=True)
    protocol = FrozenSolverProtocol()
    audit = joint_oracle_information_audit(inputs, protocol)
    assert audit["status"] == "PASS"
    assert len(joint_input_provenance_trace(inputs, protocol)) == 17
    failed = joint_oracle_information_audit(
        inputs, protocol, extra_named_inputs={"true_per_source_flux": [1.0, 2.0]}
    )
    assert failed["status"] == "FAIL"


def test_joint_optimizer_recovers_exact_fixture_and_classifies_unique() -> None:
    inputs, parameters = paired_fixture(diverse=True)
    protocol = FrozenSolverProtocol()
    starts = np.asarray([parameters * np.asarray([1.05 if index < 3 or 7 <= index < 10 else 1.0 for index in range(14)])])
    endpoints = joint_multi_start_optimize(inputs, protocol, starts=starts)
    assert endpoints[0].success
    local = joint_local_fit_diagnostics(endpoints[0].parameters, inputs, protocol)
    geometry = joint_solution_geometry(endpoints, inputs, protocol)
    classification = classify_joint_identifiability(
        endpoints, geometry, local, inputs, protocol
    )
    assert endpoints[0].chi_square < 1e-16
    assert local.jacobian_diagnostics.null_space_dimension == 0
    assert classification == "UNIQUE"
