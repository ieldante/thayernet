"""Exhaustive synthetic-only fixtures for the Model-9 preparation gate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from src.model9_structured import (
    FAMILY_BULGE_DISK,
    FAMILY_SERSIC,
    FrozenSolverProtocol,
    SolverInputs,
    gaussian_psf_kernels,
    render_pair,
)


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    inputs: SolverInputs
    generating_parameters: np.ndarray
    analytic_expectation: str


def delta_psf(size: int = 3, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if size < 1 or size % 2 != 1:
        raise ValueError("delta PSF size must be positive and odd")
    kernel = torch.zeros((3, size, size), dtype=dtype)
    kernel[:, size // 2, size // 2] = 1.0
    return kernel


def _render_observation(
    family: str,
    parameters: np.ndarray,
    requested_center_xy: tuple[float, float],
    companion_center_xy: tuple[float, float],
    psf: torch.Tensor,
    noise_sigma: torch.Tensor,
    shape: tuple[int, int],
    protocol: FrozenSolverProtocol,
    *,
    signed_noise_amplitude: float,
) -> SolverInputs:
    blank = torch.zeros((3, *shape), dtype=torch.float64)
    provisional = SolverInputs(
        observed=blank,
        requested_center_xy=requested_center_xy,
        companion_center_xy=companion_center_xy,
        psf=psf,
        noise_sigma=noise_sigma,
        family=family,
    )
    pair = render_pair(torch.as_tensor(parameters, dtype=torch.float64), provisional, protocol)
    yy, xx = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float64),
        torch.arange(shape[1], dtype=torch.float64),
        indexing="ij",
    )
    patterns = torch.stack(
        (
            torch.sin(0.37 * xx + 0.19 * yy),
            torch.cos(0.23 * xx - 0.31 * yy),
            torch.sin(0.17 * xx - 0.29 * yy + 0.4),
        )
    )
    noise = signed_noise_amplitude * patterns
    return SolverInputs(
        observed=pair.recomposed_sources.detach() + noise,
        requested_center_xy=requested_center_xy,
        companion_center_xy=companion_center_xy,
        psf=psf,
        noise_sigma=noise_sigma,
        family=family,
    )


def separated_sersic_fixture(
    protocol: FrozenSolverProtocol | None = None,
    *,
    shape: tuple[int, int] = (21, 21),
    signed_noise_amplitude: float = 0.0,
    use_delta_psf: bool = False,
) -> SyntheticFixture:
    protocol = FrozenSolverProtocol() if protocol is None else protocol
    centers = ((6.5, 9.0), (14.0, 11.5))
    parameters = np.asarray(
        [
            120.0, 160.0, 140.0, 1.0, 0.55, 0.72, 0.30,
            80.0, 95.0, 110.0, 2.5, 0.72, 0.58, 1.05,
        ],
        dtype=np.float64,
    )
    psf = delta_psf() if use_delta_psf else gaussian_psf_kernels()
    sigma = torch.asarray((0.5, 0.6, 0.7), dtype=torch.float64)
    inputs = _render_observation(
        FAMILY_SERSIC,
        parameters,
        centers[0],
        centers[1],
        psf,
        sigma,
        shape,
        protocol,
        signed_noise_amplitude=signed_noise_amplitude,
    )
    return SyntheticFixture("separated_sersic", inputs, parameters, "locally_identifiable_flux_templates")


def coincident_ambiguous_sersic_fixture(
    protocol: FrozenSolverProtocol | None = None,
    *,
    shape: tuple[int, int] = (21, 21),
) -> SyntheticFixture:
    protocol = FrozenSolverProtocol() if protocol is None else protocol
    center = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    left = np.asarray([70.0, 90.0, 110.0, 1.7, 0.65, 0.68, 0.45], dtype=np.float64)
    right = np.asarray([50.0, 40.0, 30.0, 1.7, 0.65, 0.68, 0.45], dtype=np.float64)
    parameters = np.concatenate((left, right))
    inputs = _render_observation(
        FAMILY_SERSIC,
        parameters,
        center,
        center,
        gaussian_psf_kernels(),
        torch.asarray((0.5, 0.5, 0.5), dtype=torch.float64),
        shape,
        protocol,
        signed_noise_amplitude=0.0,
    )
    return SyntheticFixture("coincident_ambiguous_sersic", inputs, parameters, "three_flux_allocation_null_directions")


def separated_bulge_disk_fixture(
    protocol: FrozenSolverProtocol | None = None,
    *,
    shape: tuple[int, int] = (23, 23),
    signed_noise_amplitude: float = 0.0,
) -> SyntheticFixture:
    protocol = FrozenSolverProtocol() if protocol is None else protocol
    centers = ((7.0, 10.0), (15.5, 12.0))
    parameters = np.asarray(
        [
            130.0, 155.0, 170.0,
            0.80, 0.72, 0.25,
            0.32, 0.86, 0.60,
            0.20, 0.35, 0.50,
            95.0, 105.0, 115.0,
            0.68, 0.61, 1.15,
            0.28, 0.74, 0.85,
            0.65, 0.45, 0.25,
        ],
        dtype=np.float64,
    )
    inputs = _render_observation(
        FAMILY_BULGE_DISK,
        parameters,
        centers[0],
        centers[1],
        gaussian_psf_kernels(),
        torch.asarray((0.6, 0.7, 0.8), dtype=torch.float64),
        shape,
        protocol,
        signed_noise_amplitude=signed_noise_amplitude,
    )
    return SyntheticFixture("separated_bulge_disk", inputs, parameters, "interior_level_5_fixture")


def boundary_bulge_disk_parameters() -> tuple[np.ndarray, np.ndarray]:
    disk_only = np.asarray(
        [100.0, 110.0, 120.0, 0.75, 0.64, 0.2, 0.25, 0.9, 1.1, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    bulge_only = np.asarray(
        [100.0, 110.0, 120.0, 0.75, 0.64, 0.2, 0.25, 0.9, 1.1, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    return disk_only, bulge_only


def fixture_suite(protocol: FrozenSolverProtocol | None = None) -> tuple[SyntheticFixture, ...]:
    protocol = FrozenSolverProtocol() if protocol is None else protocol
    return (
        separated_sersic_fixture(protocol),
        coincident_ambiguous_sersic_fixture(protocol),
        separated_bulge_disk_fixture(protocol),
    )

