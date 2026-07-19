from __future__ import annotations

import math

import galsim
import numpy as np
import pytest
import torch

from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_optimizer import (
    MultiStartEndpoint,
    analyze_jacobian,
    boundary_contact_flags,
    classify_identifiability,
    deterministic_replay,
    deterministic_starts,
    finite_difference_directional_check,
    local_fit_diagnostics,
    multi_start_optimize,
    residual_jacobian,
    solution_geometry,
)
from src.model9_galsim_adapter import sample_galsim_psf_kernels
from src.model9_structured import (
    FAMILY_BULGE_DISK,
    FAMILY_SERSIC,
    FrozenSolverProtocol,
    canonicalize_parameters,
    conservation_error,
    convolve_psf,
    gaussian_psf_kernels,
    input_provenance_trace,
    likelihood_components,
    normalize_psf,
    observed_flux_reference,
    oracle_information_audit,
    parameter_bounds,
    parameter_names,
    parameters_per_source,
    psf_normalization_error,
    render_bulge_disk_source,
    render_pair,
    render_sersic_source,
    signed_residual,
    validate_parameters,
)
from src.model9_synthetic import (
    boundary_bulge_disk_parameters,
    coincident_ambiguous_sersic_fixture,
    delta_psf,
    fixture_suite,
    separated_bulge_disk_fixture,
    separated_sersic_fixture,
)


PROTOCOL = FrozenSolverProtocol()


def _perturbed_sersic_start(parameters: np.ndarray, inputs) -> np.ndarray:
    start = parameters.copy()
    start[:3] *= np.asarray((0.8, 1.1, 0.9))
    start[7:10] *= np.asarray((1.2, 0.9, 1.1))
    start[4] *= 1.1
    start[11] *= 0.9
    start[5] = 0.75
    start[12] = 0.62
    start[6] += 0.05
    start[13] -= 0.04
    lower, upper = parameter_bounds(inputs, PROTOCOL)
    return np.clip(start, lower + 1.0e-6, upper - 1.0e-6)


def test_protocol_freezes_authoritative_level_4_and_5_support() -> None:
    PROTOCOL.validate()
    assert PROTOCOL.sersic_n_bounds == (0.5, 6.0)
    assert PROTOCOL.half_light_radius_bounds_arcsec == (0.03, 3.0)
    assert PROTOCOL.axis_ratio_bounds == (0.1, 1.0)
    assert PROTOCOL.starts_per_family == 16
    assert PROTOCOL.max_nfev == 500
    assert parameters_per_source(FAMILY_SERSIC) == 7
    assert parameters_per_source(FAMILY_BULGE_DISK) == 12
    assert "requested_flux_g" in parameter_names(FAMILY_SERSIC)
    assert "companion_bt_z" in parameter_names(FAMILY_BULGE_DISK)


@pytest.mark.parametrize("fixture", fixture_suite(PROTOCOL), ids=lambda item: item.name)
def test_all_synthetic_fixture_renderers_are_physical(fixture) -> None:
    parameters = torch.as_tensor(fixture.generating_parameters, dtype=torch.float64).clone().requires_grad_(True)
    pair = render_pair(parameters, fixture.inputs, PROTOCOL)
    assert torch.isfinite(pair.requested).all()
    assert torch.isfinite(pair.companion).all()
    assert torch.all(pair.requested >= 0)
    assert torch.all(pair.companion >= 0)
    per_source = parameters_per_source(fixture.inputs.family)
    assert torch.allclose(pair.requested.sum(dim=(-2, -1)), parameters[:3], rtol=2e-13, atol=2e-13)
    assert torch.allclose(pair.companion.sum(dim=(-2, -1)), parameters[per_source : per_source + 3], rtol=2e-13, atol=2e-13)
    loss = pair.requested.square().mean() + pair.companion.square().mean()
    loss.backward()
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()


def test_sersic_renderer_matches_independent_galsim_reference() -> None:
    shape = (41, 41)
    center = (20.0, 20.0)
    for n, threshold in ((1.0, 0.005), (4.0, 0.05)):
        parameters = torch.tensor((100.0, 110.0, 120.0, n, 0.8, 0.7, 0.4), dtype=torch.float64)
        rendered = render_sersic_source(parameters, center, delta_psf(), shape, PROTOCOL)[0].detach().numpy()
        rendered /= rendered.sum()
        profile = (
            galsim.Exponential(half_light_radius=0.8, flux=1.0)
            if n == 1.0
            else galsim.DeVaucouleurs(half_light_radius=0.8, flux=1.0)
        )
        profile = profile.shear(q=0.7, beta=0.4 * galsim.radians)
        reference = np.asarray(profile.drawImage(nx=41, ny=41, scale=0.2).array, dtype=np.float64)
        reference /= reference.sum()
        relative_error = float(np.linalg.norm(rendered - reference) / np.linalg.norm(reference))
        assert relative_error < threshold


def test_renderer_is_finite_at_every_support_extreme() -> None:
    psf = gaussian_psf_kernels()
    for local in (
        (1.0, 2.0, 3.0, 0.5, 0.03, 0.1, 0.0),
        (1.0, 2.0, 3.0, 6.0, 3.0, 1.0, math.pi),
    ):
        rendered = render_sersic_source(
            torch.tensor(local, dtype=torch.float64),
            (8.0, 8.0),
            psf,
            (17, 17),
            PROTOCOL,
        )
        assert torch.isfinite(rendered).all()
        assert torch.all(rendered >= 0)
        assert torch.allclose(rendered.sum(dim=(-2, -1)), torch.tensor(local[:3], dtype=torch.float64))


def test_bulge_disk_boundaries_quotient_to_active_component() -> None:
    disk_only, bulge_only = boundary_bulge_disk_parameters()
    center = (10.0, 10.0)
    psf = gaussian_psf_kernels()
    disk_render = render_bulge_disk_source(torch.as_tensor(disk_only), center, psf, (21, 21), PROTOCOL)
    disk_sersic = render_sersic_source(
        torch.as_tensor(np.concatenate((disk_only[:3], (1.0,), disk_only[3:6]))),
        center,
        psf,
        (21, 21),
        PROTOCOL,
    )
    bulge_render = render_bulge_disk_source(torch.as_tensor(bulge_only), center, psf, (21, 21), PROTOCOL)
    bulge_sersic = render_sersic_source(
        torch.as_tensor(np.concatenate((bulge_only[:3], (4.0,), bulge_only[6:9]))),
        center,
        psf,
        (21, 21),
        PROTOCOL,
    )
    assert torch.allclose(disk_render, disk_sersic, rtol=2e-13, atol=2e-13)
    assert torch.allclose(bulge_render, bulge_sersic, rtol=2e-13, atol=2e-13)


def test_psf_normalization_and_delta_convolution() -> None:
    raw = gaussian_psf_kernels() * torch.tensor((2.0, 3.0, 4.0), dtype=torch.float64)[:, None, None]
    normalized = normalize_psf(raw)
    assert float(psf_normalization_error(raw)) < 1.0e-15
    assert torch.allclose(normalized.sum(dim=(-2, -1)), torch.ones(3, dtype=torch.float64))
    images = torch.arange(3 * 9 * 9, dtype=torch.float64).reshape(3, 9, 9)
    assert torch.equal(convolve_psf(images, delta_psf()), images)
    with pytest.raises(ValueError):
        normalize_psf(torch.full((3, 4, 4), 1.0))
    invalid = raw.clone()
    invalid[0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        normalize_psf(invalid)


def test_galsim_psf_adapter_and_convolved_round_trip() -> None:
    galsim_psfs = tuple(galsim.Gaussian(fwhm=value, flux=1.0) for value in (0.60, 0.56, 0.52))
    kernels = sample_galsim_psf_kernels(galsim_psfs, kernel_size=21)
    assert kernels.shape == (3, 21, 21)
    assert torch.allclose(kernels.sum(dim=(-2, -1)), torch.ones(3, dtype=torch.float64))
    parameters = torch.tensor((100.0, 110.0, 120.0, 4.0, 0.8, 0.7, 0.4), dtype=torch.float64)
    rendered = render_sersic_source(parameters, (20.0, 20.0), kernels, (41, 41), PROTOCOL)[0].detach().numpy()
    rendered /= rendered.sum()
    profile = galsim.DeVaucouleurs(half_light_radius=0.8, flux=1.0).shear(
        q=0.7, beta=0.4 * galsim.radians
    )
    reference = np.asarray(
        galsim.Convolve(profile, galsim_psfs[0]).drawImage(nx=41, ny=41, scale=0.2).array,
        dtype=np.float64,
    )
    reference /= reference.sum()
    relative_error = float(np.linalg.norm(rendered - reference) / np.linalg.norm(reference))
    assert relative_error < 0.02


def test_signed_residual_is_algebraic_signed_closure() -> None:
    fixture = separated_sersic_fixture(PROTOCOL, signed_noise_amplitude=0.25)
    pair = render_pair(torch.as_tensor(fixture.generating_parameters), fixture.inputs, PROTOCOL)
    residual = signed_residual(fixture.inputs.observed, pair)
    assert torch.any(residual < 0)
    assert torch.any(residual > 0)
    assert float(conservation_error(fixture.inputs.observed, pair)) < 1.0e-13


@pytest.mark.parametrize("factory", (separated_sersic_fixture, separated_bulge_disk_fixture))
def test_autograd_jacobian_matches_centered_finite_difference(factory) -> None:
    fixture = factory(PROTOCOL, shape=(15, 15))
    rng = np.random.default_rng(2026071519 + len(fixture.generating_parameters))
    direction = rng.normal(size=len(fixture.generating_parameters))
    check = finite_difference_directional_check(
        fixture.generating_parameters,
        direction,
        fixture.inputs,
        PROTOCOL,
        step=2.0e-6,
    )
    assert check["relative_error"] < 2.0e-6


def test_likelihood_logs_components_independently() -> None:
    fixture = separated_sersic_fixture(PROTOCOL, signed_noise_amplitude=0.1)
    components = likelihood_components(
        torch.as_tensor(fixture.generating_parameters),
        fixture.inputs,
        PROTOCOL,
    )
    assert components.likelihood_by_band.shape == (3,)
    assert torch.allclose(components.likelihood_total, components.likelihood_by_band.sum())
    sigma = fixture.inputs.noise_sigma[:, None, None]
    assert torch.allclose(components.chi_square, (components.signed_residual / sigma).square().sum())


def test_flux_bounds_use_only_observation_and_noise() -> None:
    fixture = separated_sersic_fixture(PROTOCOL)
    lower, upper = parameter_bounds(fixture.inputs, PROTOCOL)
    assert np.all(lower[[0, 1, 2, 7, 8, 9]] == 0)
    assert np.all(np.isinf(upper[[0, 1, 2, 7, 8, 9]]))
    validate_parameters(fixture.generating_parameters, fixture.inputs, PROTOCOL)


def test_deterministic_start_schedule_spans_every_physical_coordinate() -> None:
    for fixture in (separated_sersic_fixture(PROTOCOL), separated_bulge_disk_fixture(PROTOCOL)):
        first = deterministic_starts(fixture.inputs, PROTOCOL)
        second = deterministic_starts(fixture.inputs, PROTOCOL)
        assert np.array_equal(first, second)
        assert first.shape == (16, 2 * parameters_per_source(fixture.inputs.family))
        assert np.unique(first[:, 0]).size >= 8
        assert np.ptp(first, axis=0).min() > 0
        if fixture.inputs.family == FAMILY_SERSIC:
            assert set(first[:, 3]) == {1.0, 4.0}
            assert set(first[:, 10]) == {1.0, 4.0}
        else:
            assert {0.0, 1.0}.issubset(set(first[:, 9]))
            assert {0.0, 1.0}.issubset(set(first[:, 21]))


def test_multistart_optimizer_recovers_a_nontruth_initialized_synthetic_scene() -> None:
    fixture = separated_sersic_fixture(PROTOCOL, shape=(15, 15), use_delta_psf=True)
    start = _perturbed_sersic_start(fixture.generating_parameters, fixture.inputs)
    endpoint = multi_start_optimize(fixture.inputs, PROTOCOL, starts=start[None])[0]
    assert endpoint.success
    assert endpoint.nfev <= PROTOCOL.max_nfev
    assert endpoint.chi_square < 1.0e-18
    assert endpoint.gradient_norm < 1.0e-6
    assert np.allclose(endpoint.parameters, fixture.generating_parameters, rtol=1.0e-8, atol=1.0e-8)


def test_optimizer_replay_is_bitwise_deterministic() -> None:
    fixture = separated_sersic_fixture(PROTOCOL, shape=(15, 15), use_delta_psf=True)
    start = _perturbed_sersic_start(fixture.generating_parameters, fixture.inputs)
    replay = deterministic_replay(fixture.inputs, PROTOCOL, starts=start[None])
    assert replay["status"] == "PASS"
    assert replay["exact_record_match"]
    assert replay["first_sha256"] == replay["second_sha256"]


def test_analytic_identifiable_and_ambiguous_flux_fixtures() -> None:
    separated = separated_sersic_fixture(PROTOCOL, shape=(15, 15))
    ambiguous = coincident_ambiguous_sersic_fixture(PROTOCOL, shape=(15, 15))
    flux_columns = np.asarray((0, 1, 2, 7, 8, 9))
    separated_jacobian = residual_jacobian(separated.generating_parameters, separated.inputs, PROTOCOL)
    separated_diagnostics = analyze_jacobian(separated_jacobian[:, flux_columns])
    assert separated_diagnostics.rank == 6
    assert separated_diagnostics.null_space_dimension == 0
    assert np.isfinite(separated_diagnostics.condition_number)
    ambiguous_jacobian = residual_jacobian(ambiguous.generating_parameters, ambiguous.inputs, PROTOCOL)
    ambiguous_diagnostics = analyze_jacobian(ambiguous_jacobian[:, flux_columns])
    assert ambiguous_diagnostics.rank == 3
    assert ambiguous_diagnostics.null_space_dimension == 3
    assert math.isinf(ambiguous_diagnostics.condition_number)
    assert np.max(np.abs(ambiguous_jacobian[:, flux_columns] @ ambiguous_diagnostics.null_space_basis.T)) < 1.0e-13


def test_full_local_rank_hessian_and_null_space_diagnostics() -> None:
    separated = separated_sersic_fixture(PROTOCOL, shape=(15, 15))
    full = local_fit_diagnostics(separated.generating_parameters, separated.inputs, PROTOCOL)
    diagnostics = full.jacobian_diagnostics
    assert diagnostics.rank == diagnostics.active_parameter_count == 14
    assert diagnostics.null_space_dimension == 0
    assert np.isfinite(diagnostics.condition_number)
    assert np.isfinite(diagnostics.hessian_condition_number)
    assert diagnostics.hessian_eigenvalues.shape == (14,)
    ambiguous = coincident_ambiguous_sersic_fixture(PROTOCOL, shape=(15, 15))
    ambiguous_full = local_fit_diagnostics(ambiguous.generating_parameters, ambiguous.inputs, PROTOCOL)
    assert ambiguous_full.jacobian_diagnostics.null_space_dimension >= 3
    assert math.isinf(ambiguous_full.jacobian_diagnostics.condition_number)


def test_symmetry_canonicalization_removes_only_known_gauges() -> None:
    fixture = separated_sersic_fixture(PROTOCOL)
    angle_shifted = fixture.generating_parameters.copy()
    angle_shifted[6] += math.pi
    canonical, active, symmetries = canonicalize_parameters(angle_shifted, fixture.inputs, PROTOCOL)
    assert canonical[6] == pytest.approx(fixture.generating_parameters[6])
    assert active.all()
    assert "angle_period_pi" in symmetries
    circular = fixture.generating_parameters.copy()
    circular[5] = 1.0
    circular[6] = 1.4
    canonical, active, symmetries = canonicalize_parameters(circular, fixture.inputs, PROTOCOL)
    assert canonical[6] == 0.0
    assert not active[6]
    assert any("circular_angle_gauge" in item for item in symmetries)
    zero = fixture.generating_parameters.copy()
    zero[:3] = 0.0
    canonical, active, symmetries = canonicalize_parameters(zero, fixture.inputs, PROTOCOL)
    assert not active[3:7].any()
    assert any("zero_flux_morphology_gauge" in item for item in symmetries)


def test_bulge_disk_boundary_symmetry_masks_inactive_shapes() -> None:
    fixture = separated_bulge_disk_fixture(PROTOCOL)
    parameters = fixture.generating_parameters.copy()
    parameters[9:12] = 0.0
    canonical, active, symmetries = canonicalize_parameters(parameters, fixture.inputs, PROTOCOL)
    assert np.allclose(canonical[6:9], (0.3, 0.7, 0.0))
    assert not active[6:9].any()
    assert any("zero_bulge_shape_gauge" in item for item in symmetries)


def test_global_solution_geometry_detects_exact_flux_allocation_family() -> None:
    fixture = coincident_ambiguous_sersic_fixture(PROTOCOL, shape=(15, 15))
    alternative = fixture.generating_parameters.copy()
    alternative[:3] += 10.0
    alternative[7:10] -= 10.0
    endpoints = multi_start_optimize(
        fixture.inputs,
        PROTOCOL,
        starts=np.stack((fixture.generating_parameters, alternative)),
    )
    geometry = solution_geometry(endpoints, fixture.inputs, PROTOCOL)
    assert len(endpoints) == 2
    assert geometry.distinct_solution_classes == 2
    assert geometry.requested_image_diameter > 0
    assert geometry.companion_image_diameter > 0
    assert geometry.flux_allocation_diameter > 0
    diagnostics = local_fit_diagnostics(endpoints[0].parameters, fixture.inputs, PROTOCOL)
    assert classify_identifiability(endpoints, geometry, diagnostics, fixture.inputs, PROTOCOL) == "NON_IDENTIFIABLE"


def test_frozen_classifier_accepts_the_exact_identifiable_fixture() -> None:
    fixture = separated_sersic_fixture(PROTOCOL, shape=(15, 15), use_delta_psf=True)
    endpoints = multi_start_optimize(
        fixture.inputs,
        PROTOCOL,
        starts=fixture.generating_parameters[None],
    )
    geometry = solution_geometry(endpoints, fixture.inputs, PROTOCOL)
    diagnostics = local_fit_diagnostics(endpoints[0].parameters, fixture.inputs, PROTOCOL)
    assert classify_identifiability(endpoints, geometry, diagnostics, fixture.inputs, PROTOCOL) == "UNIQUE"
    flags = boundary_contact_flags(endpoints[0].parameters, fixture.inputs, PROTOCOL)
    assert not flags["invalid_zero_flux_collapse"]
    assert classify_identifiability(
        endpoints,
        geometry,
        diagnostics,
        fixture.inputs,
        PROTOCOL,
        contract_valid=False,
    ) == "INVALID_CONTRACT"


def test_oracle_audit_traces_every_input_and_rejects_negative_controls() -> None:
    fixture = separated_sersic_fixture(PROTOCOL)
    trace = input_provenance_trace(fixture.inputs, PROTOCOL)
    assert len(trace) == 8
    assert all(not item["oracle"] for item in trace)
    assert {item["input"] for item in trace} == {
        "observed",
        "requested_center_xy",
        "companion_center_xy",
        "psf",
        "noise_sigma",
        "image_geometry",
        "structural_family_and_bounds",
        "source_flux_support_and_start_scale",
    }
    assert oracle_information_audit(fixture.inputs, PROTOCOL)["status"] == "PASS"
    failed = oracle_information_audit(
        fixture.inputs,
        PROTOCOL,
        extra_named_inputs={"true_per_source_flux": np.ones(3)},
    )
    assert failed["status"] == "FAIL"


def test_canonical_tensor_hash_is_replay_stable_for_renderer_outputs() -> None:
    fixture = separated_bulge_disk_fixture(PROTOCOL)
    first = render_pair(torch.as_tensor(fixture.generating_parameters), fixture.inputs, PROTOCOL)
    second = render_pair(torch.as_tensor(fixture.generating_parameters), fixture.inputs, PROTOCOL)
    assert canonical_tensor_sha256(first.requested) == canonical_tensor_sha256(second.requested)
    assert canonical_tensor_sha256(first.companion) == canonical_tensor_sha256(second.companion)
    changed = fixture.generating_parameters.copy()
    changed[0] += 1.0e-3
    third = render_pair(torch.as_tensor(changed), fixture.inputs, PROTOCOL)
    assert canonical_tensor_sha256(first.requested) != canonical_tensor_sha256(third.requested)
