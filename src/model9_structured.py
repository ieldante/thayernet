"""Differentiable flux-free structured galaxy renderer for Model 9.

This module contains no dataset loader and accepts no isolated-source truth.
All source fluxes are physical, free, nonnegative parameters.  The signed
residual is observation minus the two rendered astronomical source layers and
is evaluated under an explicit noise likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.canonical_tensor_hash import canonical_tensor_sha256


BANDS = ("g", "r", "z")
FAMILY_SERSIC = "sersic"
FAMILY_BULGE_DISK = "bulge_disk"
SUPPORTED_FAMILIES = (FAMILY_SERSIC, FAMILY_BULGE_DISK)


@dataclass(frozen=True)
class FrozenSolverProtocol:
    """Numerical and physical settings to freeze before scientific execution."""

    pixel_scale_arcsec: float = 0.2
    oversample: int = 4
    sersic_n_bounds: tuple[float, float] = (0.5, 6.0)
    half_light_radius_bounds_arcsec: tuple[float, float] = (0.03, 3.0)
    axis_ratio_bounds: tuple[float, float] = (0.1, 1.0)
    angle_bounds_radians: tuple[float, float] = (0.0, math.pi)
    bulge_fraction_bounds: tuple[float, float] = (0.0, 1.0)
    flux_initialization_total_multiplier: float = 1.0
    psf_kernel_size: int = 31
    starts_per_family: int = 16
    optimizer_seed: int = 2026071519
    max_nfev: int = 500
    ftol: float = 1.0e-10
    xtol: float = 1.0e-10
    gtol: float = 1.0e-10
    objective_accept_atol: float = 1.0e-8
    objective_accept_rtol: float = 1.0e-8
    endpoint_image_rtol: float = 1.0e-6
    symmetry_tolerance: float = 1.0e-10
    acceptable_gradient_norm: float = 1.0e-5
    maximum_condition_number: float = 1.0e6
    unique_image_diameter: float = 1.0e-3
    unique_flux_allocation_diameter: float = 1.0e-3
    unique_morphology_diameter: float = 1.0e-3
    invalid_zero_flux_fraction: float = 1.0e-8
    model_acceptance_quantile: float = 0.99

    def validate(self) -> None:
        if self.pixel_scale_arcsec <= 0:
            raise ValueError("pixel scale must be positive")
        if self.pixel_scale_arcsec != 0.2 or self.oversample != 4:
            raise ValueError("the preparation freeze requires 0.2 arcsec pixels and 4x integration")
        if self.sersic_n_bounds != (0.5, 6.0):
            raise ValueError("Level-4 Sersic support must remain [0.5, 6]")
        if self.half_light_radius_bounds_arcsec != (0.03, 3.0):
            raise ValueError("Level-4/5 HLR support must remain [0.03, 3] arcsec")
        if self.axis_ratio_bounds != (0.1, 1.0):
            raise ValueError("Level-4/5 axis-ratio support must remain [0.1, 1]")
        if self.flux_initialization_total_multiplier != 1.0:
            raise ValueError("flux starts must use one observation/noise reference total")
        if self.psf_kernel_size != 31:
            raise ValueError("the preparation freeze requires 31x31 PSF kernels")
        if self.starts_per_family != 16:
            raise ValueError("the preparation freeze requires 16 starts per family")
        if self.optimizer_seed != 2026071519:
            raise ValueError("the preparation optimizer seed changed")
        if self.max_nfev != 500:
            raise ValueError("the preparation freeze requires max_nfev=500")
        if (self.ftol, self.xtol, self.gtol) != (1.0e-10, 1.0e-10, 1.0e-10):
            raise ValueError("optimizer tolerances changed after preparation")
        if self.maximum_condition_number != 1.0e6:
            raise ValueError("the frozen condition-number rule changed")
        if (
            self.unique_image_diameter,
            self.unique_flux_allocation_diameter,
            self.unique_morphology_diameter,
        ) != (1.0e-3, 1.0e-3, 1.0e-3):
            raise ValueError("the frozen uniqueness diameters changed")


@dataclass(frozen=True)
class SolverInputs:
    """Complete inference input surface; deliberately excludes hidden truth."""

    observed: Tensor
    requested_center_xy: tuple[float, float]
    companion_center_xy: tuple[float, float]
    psf: Tensor
    noise_sigma: Tensor
    family: str

    def validate(self) -> None:
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError(f"unsupported structural family: {self.family}")
        if self.observed.ndim != 3 or self.observed.shape[0] != 3:
            raise ValueError("observed must have shape (3,H,W)")
        if self.psf.ndim != 3 or self.psf.shape[0] != 3:
            raise ValueError("psf must have shape (3,K,K)")
        if self.psf.shape[1] != self.psf.shape[2] or self.psf.shape[1] % 2 != 1:
            raise ValueError("PSF kernels must be square and odd-sized")
        if not torch.isfinite(self.observed).all():
            raise ValueError("observed contains non-finite values")
        if not torch.isfinite(self.psf).all() or torch.any(self.psf < 0):
            raise ValueError("PSF must be finite and nonnegative")
        if len(self.requested_center_xy) != 2 or len(self.companion_center_xy) != 2:
            raise ValueError("source centers must be two-dimensional")
        centers = np.asarray((self.requested_center_xy, self.companion_center_xy), dtype=np.float64)
        if not np.isfinite(centers).all():
            raise ValueError("source centers must be finite")
        sigma = expanded_noise_sigma(self.noise_sigma, self.observed)
        if not torch.isfinite(sigma).all() or torch.any(sigma <= 0):
            raise ValueError("noise sigma must be finite and positive")
        sums = self.psf.sum(dim=(-2, -1))
        if torch.any(sums <= 0):
            raise ValueError("every PSF band must have positive mass")


@dataclass(frozen=True)
class RenderedPair:
    requested: Tensor
    companion: Tensor

    @property
    def recomposed_sources(self) -> Tensor:
        return self.requested + self.companion


@dataclass(frozen=True)
class ObjectiveComponents:
    likelihood_total: Tensor
    likelihood_by_band: Tensor
    chi_square: Tensor
    signed_residual: Tensor


def expanded_noise_sigma(noise_sigma: Tensor, observed: Tensor) -> Tensor:
    sigma = torch.as_tensor(noise_sigma, dtype=observed.dtype, device=observed.device)
    if sigma.ndim == 0:
        return sigma.expand_as(observed)
    if sigma.shape == (3,):
        return sigma[:, None, None].expand_as(observed)
    if sigma.shape == observed.shape:
        return sigma
    raise ValueError("noise_sigma must be scalar, three-band, or observation-shaped")


def normalize_psf(psf: Tensor) -> Tensor:
    """Return explicitly band-normalized nonnegative odd PSF kernels."""

    value = torch.as_tensor(psf)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("psf must have shape (3,K,K)")
    if value.shape[-1] != value.shape[-2] or value.shape[-1] % 2 != 1:
        raise ValueError("psf must use odd square kernels")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError("psf must be finite and nonnegative")
    mass = value.sum(dim=(-2, -1), keepdim=True)
    if torch.any(mass <= 0):
        raise ValueError("every PSF band must have positive mass")
    return value / mass


def psf_normalization_error(psf: Tensor) -> Tensor:
    normalized = normalize_psf(psf)
    return torch.max(torch.abs(normalized.sum(dim=(-2, -1)) - 1.0))


def _sersic_bn(n: Tensor) -> Tensor:
    """Differentiable Ciotti-Bertin expansion, accurate on frozen n support."""

    return (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n.square())
        + 131.0 / (1148175.0 * n.pow(3))
        - 2194697.0 / (30690717750.0 * n.pow(4))
    )


def _subpixel_grid(
    height: int,
    width: int,
    oversample: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    # Coordinates are zero-indexed pixel coordinates; integer values are pixel
    # centers. Subpixels integrate each pixel over [i-0.5, i+0.5].
    yy = (torch.arange(height * oversample, dtype=dtype, device=device) + 0.5) / oversample - 0.5
    xx = (torch.arange(width * oversample, dtype=dtype, device=device) + 0.5) / oversample - 0.5
    return torch.meshgrid(yy, xx, indexing="ij")


def sersic_density_image(
    n: Tensor,
    half_light_radius_arcsec: Tensor,
    axis_ratio: Tensor,
    angle_radians: Tensor,
    center_xy: Sequence[float],
    image_shape: tuple[int, int],
    protocol: FrozenSolverProtocol,
) -> Tensor:
    """Pixel-integrated, unit-total infinite-profile elliptical Sersic density."""

    protocol.validate()
    height, width = image_shape
    yy, xx = _subpixel_grid(
        height,
        width,
        protocol.oversample,
        dtype=n.dtype,
        device=n.device,
    )
    center = torch.as_tensor(center_xy, dtype=n.dtype, device=n.device)
    dx = (xx - center[0]) * protocol.pixel_scale_arcsec
    dy = (yy - center[1]) * protocol.pixel_scale_arcsec
    cosine = torch.cos(angle_radians)
    sine = torch.sin(angle_radians)
    major = cosine * dx + sine * dy
    minor = -sine * dx + cosine * dy
    radius = torch.sqrt(
        torch.clamp(axis_ratio * major.square() + minor.square() / axis_ratio, min=0.0)
    )
    floor = torch.sqrt(torch.as_tensor(torch.finfo(n.dtype).eps, dtype=n.dtype, device=n.device))
    radius = torch.clamp(radius, min=floor * half_light_radius_arcsec)
    bn = _sersic_bn(n)
    log_total = (
        math.log(2.0 * math.pi)
        + torch.log(n)
        + 2.0 * torch.log(half_light_radius_arcsec)
        + torch.lgamma(2.0 * n)
        - 2.0 * n * torch.log(bn)
    )
    exponent = torch.pow(radius / half_light_radius_arcsec, 1.0 / n)
    density = torch.exp(-bn * exponent - log_total)
    integrated = density.reshape(
        height,
        protocol.oversample,
        width,
        protocol.oversample,
    ).mean(dim=(1, 3))
    return integrated * protocol.pixel_scale_arcsec**2


def convolve_psf(images: Tensor, psf: Tensor) -> Tensor:
    """Bandwise differentiable same-size PSF convolution."""

    if images.ndim != 3 or images.shape[0] != 3:
        raise ValueError("images must have shape (3,H,W)")
    kernel = normalize_psf(psf.to(dtype=images.dtype, device=images.device))
    padding = kernel.shape[-1] // 2
    return F.conv2d(
        images.unsqueeze(0),
        kernel.unsqueeze(1),
        padding=padding,
        groups=3,
    ).squeeze(0)


def _stamp_flux_normalize(images: Tensor, fluxes: Tensor) -> Tensor:
    mass = images.sum(dim=(-2, -1), keepdim=True)
    tiny = torch.finfo(images.dtype).tiny
    if torch.any(mass <= tiny):
        raise RuntimeError("rendered template has nonpositive stamp mass")
    return images / mass * fluxes[:, None, None]


def render_sersic_source(
    source_parameters: Tensor,
    center_xy: Sequence[float],
    psf: Tensor,
    image_shape: tuple[int, int],
    protocol: FrozenSolverProtocol,
) -> Tensor:
    """Render [flux_g,flux_r,flux_z,n,HLR,q,angle]."""

    if source_parameters.shape != (7,):
        raise ValueError("Sersic source parameters must have length 7")
    fluxes = source_parameters[:3]
    density = sersic_density_image(
        source_parameters[3],
        source_parameters[4],
        source_parameters[5],
        source_parameters[6],
        center_xy,
        image_shape,
        protocol,
    )
    convolved = convolve_psf(density.expand(3, -1, -1), psf)
    return _stamp_flux_normalize(convolved, fluxes)


def render_bulge_disk_source(
    source_parameters: Tensor,
    center_xy: Sequence[float],
    psf: Tensor,
    image_shape: tuple[int, int],
    protocol: FrozenSolverProtocol,
) -> Tensor:
    """Render free-flux disk(n=1)+bulge(n=4) with band-specific B/T."""

    if source_parameters.shape != (12,):
        raise ValueError("bulge+disk source parameters must have length 12")
    fluxes = source_parameters[:3]
    one = torch.ones((), dtype=source_parameters.dtype, device=source_parameters.device)
    four = 4.0 * one
    disk = sersic_density_image(
        one,
        source_parameters[3],
        source_parameters[4],
        source_parameters[5],
        center_xy,
        image_shape,
        protocol,
    )
    bulge = sersic_density_image(
        four,
        source_parameters[6],
        source_parameters[7],
        source_parameters[8],
        center_xy,
        image_shape,
        protocol,
    )
    bulge_fraction = source_parameters[9:12]
    intrinsic = (
        (1.0 - bulge_fraction)[:, None, None] * disk[None]
        + bulge_fraction[:, None, None] * bulge[None]
    )
    convolved = convolve_psf(intrinsic, psf)
    return _stamp_flux_normalize(convolved, fluxes)


def parameters_per_source(family: str) -> int:
    if family == FAMILY_SERSIC:
        return 7
    if family == FAMILY_BULGE_DISK:
        return 12
    raise ValueError(f"unsupported structural family: {family}")


def parameter_names(family: str) -> tuple[str, ...]:
    if family == FAMILY_SERSIC:
        local = ("flux_g", "flux_r", "flux_z", "n", "hlr", "q", "theta")
    elif family == FAMILY_BULGE_DISK:
        local = (
            "flux_g",
            "flux_r",
            "flux_z",
            "disk_hlr",
            "disk_q",
            "disk_theta",
            "bulge_hlr",
            "bulge_q",
            "bulge_theta",
            "bt_g",
            "bt_r",
            "bt_z",
        )
    else:
        raise ValueError(f"unsupported structural family: {family}")
    return tuple(f"{identity}_{name}" for identity in ("requested", "companion") for name in local)


def render_pair(
    parameters: Tensor,
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> RenderedPair:
    inputs.validate()
    protocol.validate()
    count = parameters_per_source(inputs.family)
    if parameters.shape != (2 * count,):
        raise ValueError(f"expected {2 * count} parameters for {inputs.family}")
    image_shape = (int(inputs.observed.shape[-2]), int(inputs.observed.shape[-1]))
    psf = normalize_psf(inputs.psf.to(dtype=parameters.dtype, device=parameters.device))
    centers = (inputs.requested_center_xy, inputs.companion_center_xy)
    outputs = []
    for source_index in (0, 1):
        local = parameters[source_index * count : (source_index + 1) * count]
        if inputs.family == FAMILY_SERSIC:
            output = render_sersic_source(local, centers[source_index], psf, image_shape, protocol)
        else:
            output = render_bulge_disk_source(local, centers[source_index], psf, image_shape, protocol)
        outputs.append(output)
    return RenderedPair(outputs[0], outputs[1])


def signed_residual(observed: Tensor, pair: RenderedPair) -> Tensor:
    """Algebraic signed noise closure; never an astronomical source layer."""

    if observed.shape != pair.requested.shape or observed.shape != pair.companion.shape:
        raise ValueError("observed and rendered source shapes must match")
    return observed - pair.requested - pair.companion


def conservation_error(observed: Tensor, pair: RenderedPair) -> Tensor:
    residual = signed_residual(observed, pair)
    return torch.max(torch.abs(pair.requested + pair.companion + residual - observed))


def likelihood_components(
    parameters: Tensor,
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> ObjectiveComponents:
    pair = render_pair(parameters, inputs, protocol)
    residual = signed_residual(inputs.observed.to(parameters), pair)
    sigma = expanded_noise_sigma(inputs.noise_sigma, inputs.observed).to(parameters)
    whitened = residual / sigma
    chi_by_band = whitened.square().sum(dim=(-2, -1))
    log_normalization = 2.0 * torch.log(sigma).sum(dim=(-2, -1))
    by_band = 0.5 * (chi_by_band + log_normalization)
    return ObjectiveComponents(by_band.sum(), by_band, chi_by_band.sum(), residual)


def whitened_residual_vector(
    parameters: Tensor,
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> Tensor:
    pair = render_pair(parameters, inputs, protocol)
    residual = signed_residual(inputs.observed.to(parameters), pair)
    sigma = expanded_noise_sigma(inputs.noise_sigma, inputs.observed).to(parameters)
    return (residual / sigma).reshape(-1)


def observed_flux_reference(inputs: SolverInputs) -> Tensor:
    """Observation/noise-only scale used for finite nonnegative flux bounds."""

    positive = torch.clamp(inputs.observed, min=0.0).sum(dim=(-2, -1))
    sigma = expanded_noise_sigma(inputs.noise_sigma, inputs.observed)
    noise_scale = torch.sqrt(torch.sum(sigma.square(), dim=(-2, -1)))
    return torch.maximum(positive, noise_scale)


def parameter_bounds(
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonnegative-unbounded flux support and frozen morphology bounds."""

    inputs.validate()
    protocol.validate()
    if inputs.family == FAMILY_SERSIC:
        local_lower = np.asarray(
            [0.0, 0.0, 0.0, protocol.sersic_n_bounds[0], protocol.half_light_radius_bounds_arcsec[0], protocol.axis_ratio_bounds[0], protocol.angle_bounds_radians[0]],
            dtype=np.float64,
        )
        local_upper = np.asarray(
            [np.inf, np.inf, np.inf, protocol.sersic_n_bounds[1], protocol.half_light_radius_bounds_arcsec[1], protocol.axis_ratio_bounds[1], protocol.angle_bounds_radians[1]],
            dtype=np.float64,
        )
    else:
        local_lower = np.asarray(
            [0.0, 0.0, 0.0, 0.03, 0.1, 0.0, 0.03, 0.1, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        local_upper = np.asarray(
            [np.inf, np.inf, np.inf, 3.0, 1.0, math.pi, 3.0, 1.0, math.pi, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )
    return np.tile(local_lower, 2), np.tile(local_upper, 2)


def parameter_scales(
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> np.ndarray:
    """Dimensionless diagnostic scales without imposing a flux upper bound."""

    lower, upper = parameter_bounds(inputs, protocol)
    scales = upper - lower
    reference = observed_flux_reference(inputs).detach().cpu().numpy().astype(np.float64)
    count = parameters_per_source(inputs.family)
    scales[:3] = reference
    scales[count : count + 3] = reference
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError("invalid observation-derived parameter scales")
    return scales


def validate_parameters(
    parameters: Sequence[float],
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    tolerance: float = 0.0,
) -> None:
    value = np.asarray(parameters, dtype=np.float64)
    lower, upper = parameter_bounds(inputs, protocol)
    if value.shape != lower.shape:
        raise ValueError(f"expected parameters with shape {lower.shape}")
    if not np.isfinite(value).all():
        raise ValueError("parameters must be finite")
    if np.any(value < lower - tolerance) or np.any(value > upper + tolerance):
        raise ValueError("parameters are outside frozen physical support")


def canonical_angle(angle: float) -> float:
    wrapped = float(angle % math.pi)
    return 0.0 if abs(wrapped - math.pi) <= 1.0e-15 else wrapped


def canonicalize_parameters(
    parameters: Sequence[float],
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Quotient known output gauges and coincident-coordinate label symmetry."""

    value = np.asarray(parameters, dtype=np.float64).copy()
    count = parameters_per_source(inputs.family)
    for source_index in (0, 1):
        offset = source_index * count
        if inputs.family == FAMILY_SERSIC:
            value[offset + 6] = canonical_angle(value[offset + 6])
        else:
            value[offset + 5] = canonical_angle(value[offset + 5])
            value[offset + 8] = canonical_angle(value[offset + 8])
    validate_parameters(value, inputs, protocol, tolerance=1.0e-9)
    active = np.ones_like(value, dtype=bool)
    symmetries: list[str] = ["angle_period_pi"]
    tol = protocol.symmetry_tolerance
    for source_index in (0, 1):
        offset = source_index * count
        local = value[offset : offset + count]
        total_flux = float(np.sum(local[:3]))
        if inputs.family == FAMILY_SERSIC:
            local[6] = canonical_angle(local[6])
            if abs(local[5] - 1.0) <= tol:
                local[6] = 0.0
                active[offset + 6] = False
                symmetries.append(f"source_{source_index}_circular_angle_gauge")
            if total_flux <= tol:
                local[3:7] = np.asarray([2.0, 0.3, 0.7, 0.0])
                active[offset + 3 : offset + 7] = False
                symmetries.append(f"source_{source_index}_zero_flux_morphology_gauge")
        else:
            local[5] = canonical_angle(local[5])
            local[8] = canonical_angle(local[8])
            if abs(local[4] - 1.0) <= tol:
                local[5] = 0.0
                active[offset + 5] = False
                symmetries.append(f"source_{source_index}_circular_disk_angle_gauge")
            if abs(local[7] - 1.0) <= tol:
                local[8] = 0.0
                active[offset + 8] = False
                symmetries.append(f"source_{source_index}_circular_bulge_angle_gauge")
            for band in range(3):
                if local[band] <= tol:
                    local[9 + band] = 0.5
                    active[offset + 9 + band] = False
                    symmetries.append(f"source_{source_index}_{BANDS[band]}_zero_flux_bt_gauge")
            bt = local[9:12]
            if np.all(bt <= tol):
                local[6:9] = np.asarray([0.3, 0.7, 0.0])
                active[offset + 6 : offset + 9] = False
                symmetries.append(f"source_{source_index}_zero_bulge_shape_gauge")
            if np.all(bt >= 1.0 - tol):
                local[3:6] = np.asarray([0.3, 0.7, 0.0])
                active[offset + 3 : offset + 6] = False
                symmetries.append(f"source_{source_index}_zero_disk_shape_gauge")
            if total_flux <= tol:
                local[3:12] = np.asarray([0.3, 0.7, 0.0, 0.3, 0.7, 0.0, 0.5, 0.5, 0.5])
                active[offset + 3 : offset + 12] = False
                symmetries.append(f"source_{source_index}_zero_flux_morphology_gauge")
    center_distance = float(
        np.linalg.norm(np.asarray(inputs.requested_center_xy) - np.asarray(inputs.companion_center_xy))
    )
    if center_distance <= tol:
        first = value[:count].copy()
        second = value[count:].copy()
        if tuple(np.round(second, 14)) < tuple(np.round(first, 14)):
            value[:count], value[count:] = second, first
            first_active = active[:count].copy()
            active[:count], active[count:] = active[count:].copy(), first_active
        symmetries.append("coincident_prompt_component_label_symmetry")
    return value, active, tuple(dict.fromkeys(symmetries))


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_sha256(parameters: Sequence[float]) -> str:
    value = np.asarray(parameters, dtype=np.dtype("<f8"), order="C")
    digest = hashlib.sha256()
    digest.update(b"thayer-model9-physical-parameters-float64-v1\0")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


PROHIBITED_ORACLE_TOKENS = (
    "isolated",
    "truth",
    "true",
    "true_source",
    "true_flux",
    "morphology_label",
    "bulge_label",
    "bulge_fraction_truth",
    "source_mask",
    "hidden_component",
    "catalog_parameter",
)


def assert_no_oracle_information(named_inputs: Mapping[str, Any]) -> None:
    for name in named_inputs:
        lowered = name.lower()
        if any(token in lowered for token in PROHIBITED_ORACLE_TOKENS):
            raise ValueError(f"prohibited oracle input: {name}")


def input_provenance_trace(
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
) -> list[dict[str, Any]]:
    """Machine-readable trace for every permitted solver input."""

    inputs.validate()
    protocol.validate()
    entries = [
        {
            "input": "observed",
            "provenance": "frozen_blended_observation",
            "sha256": canonical_tensor_sha256(inputs.observed),
            "oracle": False,
        },
        {
            "input": "requested_center_xy",
            "provenance": "frozen_requested_coordinate_prompt",
            "sha256": _json_sha256(list(map(float, inputs.requested_center_xy))),
            "oracle": False,
        },
        {
            "input": "companion_center_xy",
            "provenance": "frozen_task_coordinate_contract",
            "sha256": _json_sha256(list(map(float, inputs.companion_center_xy))),
            "oracle": False,
        },
        {
            "input": "psf",
            "provenance": "known_frozen_psf",
            "sha256": canonical_tensor_sha256(normalize_psf(inputs.psf)),
            "oracle": False,
        },
        {
            "input": "noise_sigma",
            "provenance": "declared_noise_convention",
            "sha256": canonical_tensor_sha256(expanded_noise_sigma(inputs.noise_sigma, inputs.observed)),
            "oracle": False,
        },
        {
            "input": "image_geometry",
            "provenance": "observation_shape_and_known_pixel_scale",
            "sha256": _json_sha256({"shape": list(inputs.observed.shape), "pixel_scale_arcsec": protocol.pixel_scale_arcsec}),
            "oracle": False,
        },
        {
            "input": "structural_family_and_bounds",
            "provenance": "frozen_level_4_or_5_support",
            "sha256": _json_sha256({"family": inputs.family, "n": protocol.sersic_n_bounds, "hlr": protocol.half_light_radius_bounds_arcsec, "q": protocol.axis_ratio_bounds, "bt": protocol.bulge_fraction_bounds}),
            "oracle": False,
        },
        {
            "input": "source_flux_support_and_start_scale",
            "provenance": "nonnegative_unbounded_support_with_observation_noise_initialization_scale",
            "sha256": _json_sha256({"lower": 0.0, "upper": "unbounded", "initial_total": observed_flux_reference(inputs).detach().cpu().tolist()}),
            "oracle": False,
        },
    ]
    assert_no_oracle_information({entry["input"]: None for entry in entries})
    return entries


def oracle_information_audit(
    inputs: SolverInputs,
    protocol: FrozenSolverProtocol,
    *,
    extra_named_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extras = {} if extra_named_inputs is None else dict(extra_named_inputs)
    try:
        assert_no_oracle_information(extras)
        trace = input_provenance_trace(inputs, protocol)
    except ValueError as error:
        return {"status": "FAIL", "reason": str(error), "trace": []}
    return {
        "status": "PASS",
        "reason": "all solver inputs are permitted and observation-derived where required",
        "trace": trace,
        "extra_input_names": sorted(extras),
    }


def gaussian_psf_kernels(
    fwhm_pixels: Sequence[float] = (2.5, 2.2, 2.0),
    size: int = 9,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Create normalized synthetic three-band Gaussian PSFs for fixtures."""

    if size < 3 or size % 2 != 1:
        raise ValueError("synthetic PSF size must be odd and at least three")
    if len(fwhm_pixels) != 3:
        raise ValueError("three FWHM values are required")
    coordinate = torch.arange(size, dtype=dtype) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    kernels = []
    for fwhm in fwhm_pixels:
        if fwhm <= 0:
            raise ValueError("FWHM must be positive")
        sigma = float(fwhm) / math.sqrt(8.0 * math.log(2.0))
        kernels.append(torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma**2)))
    return normalize_psf(torch.stack(kernels))
