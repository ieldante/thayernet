#!/usr/bin/env python3
"""Synthetic-only engineering validator for Thayer-Model-9-Preparation-v0."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import galsim
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_galsim_adapter import sample_galsim_psf_kernels
from src.model9_optimizer import (
    analyze_jacobian,
    classify_identifiability,
    deterministic_replay,
    finite_difference_directional_check,
    local_fit_diagnostics,
    multi_start_optimize,
    residual_jacobian,
    solution_geometry,
)
from src.model9_structured import (
    FrozenSolverProtocol,
    conservation_error,
    gaussian_psf_kernels,
    oracle_information_audit,
    parameter_bounds,
    psf_normalization_error,
    render_bulge_disk_source,
    render_pair,
    render_sersic_source,
    signed_residual,
)
from src.model9_synthetic import (
    boundary_bulge_disk_parameters,
    coincident_ambiguous_sersic_fixture,
    fixture_suite,
    separated_bulge_disk_fixture,
    separated_sersic_fixture,
)


def _finite_or_string(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ("inf" if value > 0 else "-inf")


def _galsim_renderer_errors(protocol: FrozenSolverProtocol) -> dict[str, float]:
    output: dict[str, float] = {}
    for n in (1.0, 4.0):
        parameters = torch.tensor((100.0, 110.0, 120.0, n, 0.8, 0.7, 0.4), dtype=torch.float64)
        from src.model9_synthetic import delta_psf

        rendered = render_sersic_source(parameters, (20.0, 20.0), delta_psf(), (41, 41), protocol)[0]
        rendered_array = np.asarray(rendered.detach().cpu(), dtype=np.float64)
        rendered_array /= rendered_array.sum()
        profile = (
            galsim.Exponential(half_light_radius=0.8, flux=1.0)
            if n == 1.0
            else galsim.DeVaucouleurs(half_light_radius=0.8, flux=1.0)
        ).shear(q=0.7, beta=0.4 * galsim.radians)
        reference = np.asarray(profile.drawImage(nx=41, ny=41, scale=0.2).array, dtype=np.float64)
        reference /= reference.sum()
        output[f"n_{n:g}_relative_l2"] = float(np.linalg.norm(rendered_array - reference) / np.linalg.norm(reference))
    return output


def validate() -> dict[str, Any]:
    protocol = FrozenSolverProtocol()
    protocol.validate()
    physical_rows = []
    physical_pass = True
    for fixture in fixture_suite(protocol):
        parameters = torch.as_tensor(fixture.generating_parameters, dtype=torch.float64)
        pair = render_pair(parameters, fixture.inputs, protocol)
        count = len(parameters) // 2
        source_parameter_count = 7 if fixture.inputs.family == "sersic" else 12
        requested_error = float(torch.max(torch.abs(pair.requested.sum(dim=(-2, -1)) - parameters[:3])))
        companion_error = float(
            torch.max(
                torch.abs(
                    pair.companion.sum(dim=(-2, -1))
                    - parameters[source_parameter_count : source_parameter_count + 3]
                )
            )
        )
        closure = float(conservation_error(fixture.inputs.observed, pair))
        nonnegative = bool(torch.all(pair.requested >= 0) and torch.all(pair.companion >= 0))
        finite = bool(torch.isfinite(pair.requested).all() and torch.isfinite(pair.companion).all())
        residual = signed_residual(fixture.inputs.observed, pair)
        row = {
            "fixture": fixture.name,
            "family": fixture.inputs.family,
            "requested_flux_error": requested_error,
            "companion_flux_error": companion_error,
            "closure_error": closure,
            "nonnegative_sources": nonnegative,
            "finite_sources": finite,
            "signed_residual_min": float(residual.min()),
            "signed_residual_max": float(residual.max()),
            "requested_sha256": canonical_tensor_sha256(pair.requested),
            "companion_sha256": canonical_tensor_sha256(pair.companion),
        }
        physical_rows.append(row)
        physical_pass &= nonnegative and finite and max(requested_error, companion_error, closure) < 1.0e-10

    gradient_rows = []
    gradient_pass = True
    for fixture in (
        separated_sersic_fixture(protocol, shape=(15, 15)),
        separated_bulge_disk_fixture(protocol, shape=(15, 15)),
    ):
        rng = np.random.default_rng(protocol.optimizer_seed + len(fixture.generating_parameters))
        check = finite_difference_directional_check(
            fixture.generating_parameters,
            rng.normal(size=len(fixture.generating_parameters)),
            fixture.inputs,
            protocol,
            step=2.0e-6,
        )
        gradient_rows.append({"fixture": fixture.name, **check})
        gradient_pass &= check["relative_error"] < 2.0e-6

    optimizer_fixture = separated_sersic_fixture(protocol, shape=(15, 15), use_delta_psf=True)
    start = optimizer_fixture.generating_parameters.copy()
    start[:3] *= np.asarray((0.8, 1.1, 0.9))
    start[7:10] *= np.asarray((1.2, 0.9, 1.1))
    start[4] *= 1.1
    start[11] *= 0.9
    start[5], start[12], start[6], start[13] = 0.75, 0.62, start[6] + 0.05, start[13] - 0.04
    lower, upper = parameter_bounds(optimizer_fixture.inputs, protocol)
    start = np.clip(start, lower + 1.0e-6, upper - 1.0e-6)
    endpoint = multi_start_optimize(optimizer_fixture.inputs, protocol, starts=start[None])[0]
    replay = deterministic_replay(optimizer_fixture.inputs, protocol, starts=start[None])
    optimizer_pass = bool(
        endpoint.success
        and endpoint.chi_square < 1.0e-18
        and endpoint.gradient_norm < 1.0e-6
        and np.allclose(endpoint.parameters, optimizer_fixture.generating_parameters, rtol=1.0e-8, atol=1.0e-8)
    )

    separated = separated_sersic_fixture(protocol, shape=(15, 15))
    ambiguous = coincident_ambiguous_sersic_fixture(protocol, shape=(15, 15))
    flux_columns = np.asarray((0, 1, 2, 7, 8, 9))
    identifiable = analyze_jacobian(
        residual_jacobian(separated.generating_parameters, separated.inputs, protocol)[:, flux_columns]
    )
    nonidentifiable = analyze_jacobian(
        residual_jacobian(ambiguous.generating_parameters, ambiguous.inputs, protocol)[:, flux_columns]
    )
    full_identifiable = local_fit_diagnostics(separated.generating_parameters, separated.inputs, protocol)
    rank_pass = bool(
        identifiable.rank == 6
        and identifiable.null_space_dimension == 0
        and nonidentifiable.rank == 3
        and nonidentifiable.null_space_dimension == 3
        and full_identifiable.jacobian_diagnostics.rank == 14
        and full_identifiable.jacobian_diagnostics.null_space_dimension == 0
    )
    identifiable_geometry = solution_geometry([endpoint], optimizer_fixture.inputs, protocol)
    identifiable_local = local_fit_diagnostics(endpoint.parameters, optimizer_fixture.inputs, protocol)
    identifiable_classification = classify_identifiability(
        [endpoint], identifiable_geometry, identifiable_local, optimizer_fixture.inputs, protocol
    )
    ambiguous_alternative = ambiguous.generating_parameters.copy()
    ambiguous_alternative[:3] += 10.0
    ambiguous_alternative[7:10] -= 10.0
    ambiguous_endpoints = multi_start_optimize(
        ambiguous.inputs,
        protocol,
        starts=np.stack((ambiguous.generating_parameters, ambiguous_alternative)),
    )
    ambiguous_geometry = solution_geometry(ambiguous_endpoints, ambiguous.inputs, protocol)
    ambiguous_local = local_fit_diagnostics(ambiguous_endpoints[0].parameters, ambiguous.inputs, protocol)
    ambiguous_classification = classify_identifiability(
        ambiguous_endpoints, ambiguous_geometry, ambiguous_local, ambiguous.inputs, protocol
    )
    classification_pass = (
        identifiable_classification == "UNIQUE"
        and ambiguous_classification == "NON_IDENTIFIABLE"
    )

    disk_only, bulge_only = boundary_bulge_disk_parameters()
    psf = gaussian_psf_kernels()
    disk_bd = render_bulge_disk_source(torch.as_tensor(disk_only), (10.0, 10.0), psf, (21, 21), protocol)
    disk_sersic = render_sersic_source(
        torch.as_tensor(np.concatenate((disk_only[:3], (1.0,), disk_only[3:6]))),
        (10.0, 10.0),
        psf,
        (21, 21),
        protocol,
    )
    bulge_bd = render_bulge_disk_source(torch.as_tensor(bulge_only), (10.0, 10.0), psf, (21, 21), protocol)
    bulge_sersic = render_sersic_source(
        torch.as_tensor(np.concatenate((bulge_only[:3], (4.0,), bulge_only[6:9]))),
        (10.0, 10.0),
        psf,
        (21, 21),
        protocol,
    )
    boundary_errors = {
        "disk_only_max_abs": float(torch.max(torch.abs(disk_bd - disk_sersic))),
        "bulge_only_max_abs": float(torch.max(torch.abs(bulge_bd - bulge_sersic))),
    }
    symmetry_pass = max(boundary_errors.values()) < 1.0e-12

    galsim_psfs = tuple(galsim.Gaussian(fwhm=value) for value in (0.60, 0.56, 0.52))
    sampled_psf = sample_galsim_psf_kernels(galsim_psfs, kernel_size=21)
    psf_error = float(psf_normalization_error(sampled_psf))
    psf_pass = psf_error < 1.0e-15
    renderer_errors = _galsim_renderer_errors(protocol)
    renderer_pass = renderer_errors["n_1_relative_l2"] < 0.005 and renderer_errors["n_4_relative_l2"] < 0.05

    oracle = oracle_information_audit(separated.inputs, protocol)
    oracle_negative_control = oracle_information_audit(
        separated.inputs,
        protocol,
        extra_named_inputs={"true_per_source_flux": np.ones(3)},
    )
    oracle_pass = oracle["status"] == "PASS" and oracle_negative_control["status"] == "FAIL"

    checks = {
        "physical_renderers": physical_pass,
        "gradient_finite_differences": gradient_pass,
        "optimizer_convergence": optimizer_pass,
        "deterministic_replay": replay["status"] == "PASS",
        "analytic_rank_fixtures": rank_pass,
        "classification_logic": classification_pass,
        "symmetry_boundaries": symmetry_pass,
        "psf_normalization": psf_pass,
        "independent_galsim_renderer": renderer_pass,
        "oracle_information_audit": oracle_pass,
    }
    return {
        "campaign": "Thayer-Model-9-Preparation-v0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "protocol": asdict(protocol),
        "physical_fixtures": physical_rows,
        "gradient_checks": gradient_rows,
        "optimizer": {
            "success": endpoint.success,
            "nfev": endpoint.nfev,
            "chi_square": endpoint.chi_square,
            "gradient_norm": endpoint.gradient_norm,
            "max_parameter_error": float(np.max(np.abs(endpoint.parameters - optimizer_fixture.generating_parameters))),
        },
        "replay": replay,
        "analytic_fixtures": {
            "separated_flux_rank": identifiable.rank,
            "separated_flux_null": identifiable.null_space_dimension,
            "separated_flux_condition": _finite_or_string(identifiable.condition_number),
            "coincident_flux_rank": nonidentifiable.rank,
            "coincident_flux_null": nonidentifiable.null_space_dimension,
            "coincident_flux_condition": _finite_or_string(nonidentifiable.condition_number),
            "full_separated_rank": full_identifiable.jacobian_diagnostics.rank,
            "full_separated_null": full_identifiable.jacobian_diagnostics.null_space_dimension,
            "full_separated_condition": _finite_or_string(full_identifiable.jacobian_diagnostics.condition_number),
            "full_separated_hessian_condition": _finite_or_string(full_identifiable.jacobian_diagnostics.hessian_condition_number),
            "separated_classification": identifiable_classification,
            "coincident_classification": ambiguous_classification,
            "coincident_solution_classes": ambiguous_geometry.distinct_solution_classes,
        },
        "boundary_symmetry_errors": boundary_errors,
        "psf_normalization_error": psf_error,
        "renderer_reference_errors": renderer_errors,
        "oracle_audit": oracle,
        "oracle_negative_control": oracle_negative_control,
        "access_counts": {
            "frozen_scientific_scenes": 0,
            "isolated_source_arrays": 0,
            "development": 0,
            "atlas": 0,
            "lockbox": 0,
            "neural_training_steps": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args()
    result = validate()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.json_output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
