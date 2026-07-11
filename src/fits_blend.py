"""Auditable flux-space blending for band-first scientific FITS arrays.

This module is deliberately separate from :mod:`src.blend`, which operates on
display-oriented, channel-last RGB arrays and clips their values.  Here every
cube has shape ``(band, y, x)`` and contains linear survey fluxes.  Negative
sky-subtracted values are valid and are never clipped.

The only extraction policy implemented in version 1 subtracts a robust,
per-band constant background estimated from an explicit background mask, then
zeros pixels outside an explicit source mask.  This prevents duplication of a
whole contaminant cutout and its constant background.  It cannot remove the
contaminant coadd's noise realization inside the source mask.  That limitation
is recorded in every result and is enforced by the strict scientific gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


GENERATOR_VERSION = "fits-blend-v1.1-strict-grz-psf-gate"
DEFAULT_BAND_ORDER = ("g", "r", "z")
_PSF_POLICIES = {"not_verified", "caller_verified_compatible"}
_GENERATOR_ESTABLISHES_NOISE_FREE_SOURCE = False
_GENERATOR_INDEPENDENTLY_VERIFIES_PSF = False


class ScientificValidityError(RuntimeError):
    """Raised when a requested scientific guarantee is not established."""


@dataclass(frozen=True)
class BlendTransform:
    """Complete geometric and photometric transform for one contaminant.

    ``shift_xy`` uses continuous image pixels: positive x moves right and
    positive y moves down.  ``flux_scales`` follows the declared band order.
    The seed and RNG algorithm are retained even after sampled values are
    materialized so a manifest records both provenance and exact parameters.
    """

    sample_seed: int
    shift_xy: tuple[float, float]
    flux_scales: tuple[float, float, float]
    rng_algorithm: str = "explicit_parameters"

    def __post_init__(self) -> None:
        _validate_seed(self.sample_seed)
        _validate_xy(self.shift_xy, "shift_xy", require_in_frame=None)
        scales = np.asarray(self.flux_scales, dtype=np.float64)
        if scales.shape != (3,):
            raise ValueError("flux_scales must contain one value for each grz band")
        if not np.isfinite(scales).all() or np.any(scales < 0.0):
            raise ValueError("flux_scales must be finite and nonnegative")
        if not self.rng_algorithm:
            raise ValueError("rng_algorithm must be nonempty")


@dataclass(frozen=True)
class FitsBlendResult:
    """Immutable-by-convention output of :func:`blend_fits_cutouts`.

    The NumPy arrays are marked read-only before return so their recorded
    hashes continue to describe them.  ``source_only_contaminant`` is already
    background-subtracted, per-band scaled, shifted, and placed on the target
    grid; it is the exact additive contribution to ``blend``.
    """

    target: np.ndarray
    source_only_contaminant: np.ndarray
    blend: np.ndarray
    target_coordinate_xy: tuple[float, float]
    contaminant_coordinate_xy: tuple[float, float]
    affected_mask: np.ndarray
    core_mask: np.ndarray
    target_source_id: str
    target_group_id: str
    contaminant_source_id: str
    contaminant_group_id: str
    metadata: dict[str, Any]


def array_sha256(array: np.ndarray) -> str:
    """Return a canonical SHA-256 including dtype, shape, and array bytes."""
    value = np.asarray(array)
    if value.dtype == np.bool_:
        canonical = np.ascontiguousarray(value, dtype=np.uint8)
        dtype_name = "bool"
    elif np.issubdtype(value.dtype, np.floating):
        dtype = np.dtype(f"<f{value.dtype.itemsize}")
        canonical = np.ascontiguousarray(value, dtype=dtype)
        dtype_name = f"float{value.dtype.itemsize * 8}"
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = np.dtype(f"<i{value.dtype.itemsize}")
        canonical = np.ascontiguousarray(value, dtype=dtype)
        dtype_name = f"int{value.dtype.itemsize * 8}"
    elif np.issubdtype(value.dtype, np.unsignedinteger):
        dtype = np.dtype(f"<u{value.dtype.itemsize}")
        canonical = np.ascontiguousarray(value, dtype=dtype)
        dtype_name = f"uint{value.dtype.itemsize * 8}"
    else:
        raise TypeError(f"unsupported dtype for hashing: {value.dtype}")

    descriptor = json.dumps(
        {"dtype": dtype_name, "shape": list(value.shape), "order": "C"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def sample_blend_transform(
    sample_seed: int,
    *,
    shift_x_range: tuple[float, float],
    shift_y_range: tuple[float, float],
    flux_scale_ranges: Sequence[tuple[float, float]],
) -> BlendTransform:
    """Sample and fully materialize a replayable transform with PCG64.

    Exact sampled values are stored in the returned transform.  Replay uses
    those values rather than relying on future NumPy releases to reproduce a
    random-number stream from the seed alone.
    """
    seed = _validate_seed(sample_seed)
    x_low, x_high = _validate_range(shift_x_range, "shift_x_range")
    y_low, y_high = _validate_range(shift_y_range, "shift_y_range")
    if len(flux_scale_ranges) != 3:
        raise ValueError("flux_scale_ranges must contain three (low, high) pairs")
    scale_ranges = [
        _validate_range(bounds, f"flux_scale_ranges[{index}]")
        for index, bounds in enumerate(flux_scale_ranges)
    ]
    if any(low < 0.0 for low, _ in scale_ranges):
        raise ValueError("flux scale lower bounds must be nonnegative")

    rng = np.random.Generator(np.random.PCG64(seed))
    shift_xy = (
        float(rng.uniform(x_low, x_high)),
        float(rng.uniform(y_low, y_high)),
    )
    flux_scales = tuple(
        float(rng.uniform(low, high)) for low, high in scale_ranges
    )
    return BlendTransform(
        sample_seed=seed,
        shift_xy=shift_xy,
        flux_scales=flux_scales,  # type: ignore[arg-type]
        rng_algorithm="numpy.random.PCG64",
    )


def blend_fits_cutouts(
    target: np.ndarray,
    contaminant: np.ndarray,
    *,
    source_mask: np.ndarray,
    background_mask: np.ndarray,
    target_coordinate_xy: tuple[float, float],
    contaminant_coordinate_xy: tuple[float, float],
    target_source_id: str,
    target_group_id: str,
    contaminant_source_id: str,
    contaminant_group_id: str,
    transform: BlendTransform,
    core_mask: np.ndarray | None = None,
    band_order: Sequence[str] = DEFAULT_BAND_ORDER,
    psf_policy: str = "not_verified",
    allow_source_mask_at_border: bool = False,
) -> FitsBlendResult:
    """Extract and add one contaminant source to a target coadd cutout.

    No stochastic noise, PSF convolution, normalization, or clipping occurs.
    The target is preserved exactly and the returned relation is therefore
    bitwise reproducible as ``blend == target + source_only_contaminant``.

    A source mask and a disjoint background mask are required.  They must come
    from the separately audited isolation pipeline; this function does not
    silently infer source support from the pixels it is asked to blend.
    """
    target_array = _validate_fits_cube(target, "target")
    contaminant_array = _validate_fits_cube(contaminant, "contaminant")
    if target_array.shape != contaminant_array.shape:
        raise ValueError("target and contaminant must have identical shapes")

    bands = _validate_band_order(band_order)
    _, height, width = target_array.shape
    source = _validate_mask(source_mask, (height, width), "source_mask")
    background = _validate_mask(
        background_mask, (height, width), "background_mask"
    )
    if np.any(source & background):
        raise ValueError("source_mask and background_mask must be disjoint")
    if not allow_source_mask_at_border and _touches_border(source):
        raise ValueError(
            "source_mask touches a cutout border; source-only extraction may be "
            "truncated, so reject it or opt in explicitly"
        )
    if np.all(source):
        raise ValueError("source_mask cannot cover the entire contaminant cutout")

    if core_mask is None:
        core = source.copy()
        core_mask_origin = "source_mask_fallback"
    else:
        core = _validate_mask(core_mask, (height, width), "core_mask")
        if np.any(core & ~source):
            raise ValueError("core_mask must be a subset of source_mask")
        core_mask_origin = "caller_supplied"

    _validate_identifier(target_source_id, "target_source_id")
    _validate_identifier(target_group_id, "target_group_id")
    _validate_identifier(contaminant_source_id, "contaminant_source_id")
    _validate_identifier(contaminant_group_id, "contaminant_group_id")
    target_xy = _validate_xy(
        target_coordinate_xy,
        "target_coordinate_xy",
        require_in_frame=(height, width),
    )
    contaminant_input_xy = _validate_xy(
        contaminant_coordinate_xy,
        "contaminant_coordinate_xy",
        require_in_frame=(height, width),
    )
    if psf_policy not in _PSF_POLICIES:
        choices = ", ".join(sorted(_PSF_POLICIES))
        raise ValueError(f"psf_policy must be one of: {choices}")

    work_dtype = np.result_type(
        target_array.dtype, contaminant_array.dtype, np.float32
    )
    target_work = np.array(target_array, dtype=work_dtype, copy=True, order="C")
    contaminant_work = np.array(
        contaminant_array, dtype=work_dtype, copy=True, order="C"
    )

    # A constant per-band estimator is deliberately simple and auditable.  It
    # removes neither spatial sky residuals nor coadd noise inside source_mask.
    background_levels = np.median(contaminant_work[:, background], axis=1)
    source_only_unscaled = (
        contaminant_work - background_levels[:, np.newaxis, np.newaxis]
    ) * source[np.newaxis, :, :]
    scales = np.asarray(transform.flux_scales, dtype=work_dtype)
    scaled_source = source_only_unscaled * scales[:, np.newaxis, np.newaxis]

    dx, dy = transform.shift_xy
    placed_source = _translate_bilinear_no_wrap(scaled_source, dx=dx, dy=dy)
    placed_support = _translate_bilinear_no_wrap(
        source.astype(np.float64), dx=dx, dy=dy
    )
    placed_core_support = _translate_bilinear_no_wrap(
        core.astype(np.float64), dx=dx, dy=dy
    )
    affected = placed_support > 0.0
    placed_core = placed_core_support > 0.0
    if not affected.any():
        raise ValueError("shift places the contaminant source entirely out of frame")
    if np.any(placed_core & ~affected):
        raise AssertionError("translated core mask escaped translated source mask")
    if not np.any(placed_source != 0.0):
        raise ValueError(
            "extracted and transformed contaminant has no nonzero flux; reject "
            "the source or transform rather than generating an empty blend"
        )

    # This is the defining flux-space invariant.  Do not clip either operand or
    # the sum: negative values and values above display ranges remain physical.
    blended = target_work + placed_source
    placed_contaminant_xy = (
        contaminant_input_xy[0] + float(dx),
        contaminant_input_xy[1] + float(dy),
    )
    input_support_weight = float(source.sum())
    placed_support_weight = float(placed_support.sum(dtype=np.float64))
    support_retained_fraction = placed_support_weight / input_support_weight
    edge_truncated = not math.isclose(
        placed_support_weight,
        input_support_weight,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )

    hashes = {
        "target_sha256": array_sha256(target_work),
        "contaminant_input_sha256": array_sha256(contaminant_work),
        "source_mask_input_sha256": array_sha256(source),
        "background_mask_input_sha256": array_sha256(background),
        "core_mask_input_sha256": array_sha256(core),
        "source_only_contaminant_sha256": array_sha256(placed_source),
        "blend_sha256": array_sha256(blended),
        "affected_mask_sha256": array_sha256(affected),
        "core_mask_sha256": array_sha256(placed_core),
    }
    psf_verified = psf_policy == "caller_verified_compatible"
    metadata: dict[str, Any] = {
        "generator": "src.fits_blend.blend_fits_cutouts",
        "generator_version": GENERATOR_VERSION,
        "array_convention": "band_y_x",
        "shape": list(target_work.shape),
        "dtype": str(target_work.dtype),
        "band_order": list(bands),
        "source_ids": {
            "target": target_source_id,
            "contaminant": contaminant_source_id,
        },
        "group_ids": {
            "target": target_group_id,
            "contaminant": contaminant_group_id,
        },
        "coordinates_xy_pixels": {
            "target": list(target_xy),
            "contaminant_input": list(contaminant_input_xy),
            "contaminant_placed": list(placed_contaminant_xy),
        },
        "transform": {
            "sample_seed": int(transform.sample_seed),
            "rng_algorithm": transform.rng_algorithm,
            "sampled_values_recorded_explicitly": True,
            "shift_xy_pixels": [float(dx), float(dy)],
            "interpolation": "bilinear_forward_splat_no_wrap_v1",
            "flux_scales_by_band": {
                band: float(scale)
                for band, scale in zip(bands, transform.flux_scales, strict=True)
            },
        },
        "source_extraction": {
            "method": "explicit_mask_constant_median_background_v1",
            "source_mask_origin": "caller_supplied_audited_mask",
            "core_mask_origin": core_mask_origin,
            "background_mask_origin": "caller_supplied_disjoint_mask",
            "background_estimator": "per_band_median",
            "background_levels_by_band": {
                band: float(level)
                for band, level in zip(bands, background_levels, strict=True)
            },
            "background_pixel_count": int(background.sum()),
            "constant_contaminant_background_removed": True,
            "spatial_background_residual_removed": False,
            "noise_free_source_established": False,
            "contaminant_coadd_noise_inside_mask_retained": True,
            "limitation": (
                "A single resampled coadd cannot identify the source's noiseless "
                "flux separately from the coadd noise realization inside the "
                "source mask. No denoising or unsupported noise subtraction is "
                "performed."
            ),
        },
        "target_policy": {
            "target_background_preserved": True,
            "target_noise_preserved": True,
            "target_psf_preserved": True,
        },
        "psf": {
            "policy": psf_policy,
            "convolution_applied": False,
            "caller_asserted_compatible": psf_verified,
            "independently_verified_by_generator": False,
        },
        "optional_effects": {
            "synthetic_noise_added": False,
            "psf_convolution_applied": False,
            "clipping_applied": False,
            "normalization_applied": False,
        },
        "mask_policy": {
            "affected_mask_source": "translated_input_source_mask",
            "affected_mask_uses_prediction": False,
            "affected_mask_uses_target": False,
            "allow_source_mask_at_border": bool(allow_source_mask_at_border),
        },
        "edge_truncation": {
            "source_support_weight_before_shift": input_support_weight,
            "source_support_weight_after_shift": placed_support_weight,
            "source_support_retained_fraction": support_retained_fraction,
            "edge_truncated": edge_truncated,
            "scaled_source_flux_before_shift_by_band": {
                band: float(value)
                for band, value in zip(
                    bands,
                    scaled_source.sum(axis=(1, 2), dtype=np.float64),
                    strict=True,
                )
            },
            "placed_source_flux_by_band": {
                band: float(value)
                for band, value in zip(
                    bands,
                    placed_source.sum(axis=(1, 2), dtype=np.float64),
                    strict=True,
                )
            },
        },
        "scientific_gate": {
            "manifest_ready_if_noise_free_source_required": False,
            "reason": "coadd noise remains inside the extracted source mask",
        },
        "hashes": hashes,
    }

    result = FitsBlendResult(
        target=_mark_read_only(target_work),
        source_only_contaminant=_mark_read_only(placed_source),
        blend=_mark_read_only(blended),
        target_coordinate_xy=target_xy,
        contaminant_coordinate_xy=placed_contaminant_xy,
        affected_mask=_mark_read_only(affected),
        core_mask=_mark_read_only(placed_core),
        target_source_id=target_source_id,
        target_group_id=target_group_id,
        contaminant_source_id=contaminant_source_id,
        contaminant_group_id=contaminant_group_id,
        metadata=metadata,
    )
    audit_fits_blend(result)
    return result


def audit_fits_blend(
    result: FitsBlendResult,
    *,
    require_noise_free_source: bool = False,
    require_psf_compatibility: bool = False,
) -> dict[str, bool]:
    """Verify algebraic, spatial, replay-hash, and scientific invariants.

    The ordinary audit establishes what this generator actually guarantees.
    Setting either ``require_*`` argument turns an unestablished scientific
    assumption into a hard failure suitable for a manifest-generation gate.
    """
    target = np.asarray(result.target)
    source = np.asarray(result.source_only_contaminant)
    blend = np.asarray(result.blend)
    affected = np.asarray(result.affected_mask)
    core = np.asarray(result.core_mask)
    if target.shape != source.shape or target.shape != blend.shape:
        raise AssertionError("target, source-only contaminant, and blend shapes differ")
    if affected.shape != target.shape[-2:] or core.shape != affected.shape:
        raise AssertionError("mask dimensions do not match the image grid")
    if not np.array_equal(blend, target + source):
        raise AssertionError("flux-addition equality failed")
    if np.any(source[:, ~affected] != 0.0):
        raise AssertionError("source-only contaminant leaked outside affected_mask")
    if not np.array_equal(blend[:, ~affected], target[:, ~affected]):
        raise AssertionError("target background changed outside affected_mask")
    if np.any(core & ~affected):
        raise AssertionError("core_mask is not contained in affected_mask")

    expected_hashes = result.metadata.get("hashes", {})
    actual_hashes = {
        "target_sha256": array_sha256(target),
        "source_only_contaminant_sha256": array_sha256(source),
        "blend_sha256": array_sha256(blend),
        "affected_mask_sha256": array_sha256(affected),
        "core_mask_sha256": array_sha256(core),
    }
    for key, value in actual_hashes.items():
        if expected_hashes.get(key) != value:
            raise AssertionError(f"recorded hash mismatch for {key}")

    extraction = result.metadata["source_extraction"]
    effects = result.metadata["optional_effects"]
    psf = result.metadata["psf"]
    if effects["synthetic_noise_added"]:
        raise AssertionError("version 1 must not add unexplained synthetic noise")
    if effects["clipping_applied"]:
        raise AssertionError("scientific flux arrays must not be clipped")
    if not extraction["constant_contaminant_background_removed"]:
        raise AssertionError("constant contaminant background was not removed")
    if bool(extraction["noise_free_source_established"]) != (
        _GENERATOR_ESTABLISHES_NOISE_FREE_SOURCE
    ):
        raise AssertionError("metadata claims unsupported noise-free source semantics")
    if bool(psf["independently_verified_by_generator"]) != (
        _GENERATOR_INDEPENDENTLY_VERIFIES_PSF
    ):
        raise AssertionError("metadata claims unsupported independent PSF verification")
    if require_noise_free_source and not _GENERATOR_ESTABLISHES_NOISE_FREE_SOURCE:
        raise ScientificValidityError(
            "noise-free source extraction is not established from a single coadd"
        )
    if require_psf_compatibility and not _GENERATOR_INDEPENDENTLY_VERIFIES_PSF:
        raise ScientificValidityError(
            "target/contaminant PSF compatibility has not been independently "
            "established; a caller assertion is not a scientific certificate"
        )

    return {
        "flux_addition_exact": True,
        "target_background_preserved_outside_mask": True,
        "no_rectangular_cutout_leakage": True,
        "no_double_constant_background": True,
        "no_synthetic_noise_added": True,
        "no_hidden_clipping": True,
        "band_order_recorded": len(result.metadata["band_order"]) == target.shape[0],
        "affected_mask_prediction_independent": not result.metadata["mask_policy"][
            "affected_mask_uses_prediction"
        ],
        "hashes_match": True,
        "noise_free_source_established": _GENERATOR_ESTABLISHES_NOISE_FREE_SOURCE,
        "psf_compatibility_asserted": bool(psf["caller_asserted_compatible"]),
        "psf_compatibility_independently_verified": (
            _GENERATOR_INDEPENDENTLY_VERIFIES_PSF
        ),
    }


def replay_fits_blend(
    target: np.ndarray,
    contaminant: np.ndarray,
    *,
    source_mask: np.ndarray,
    background_mask: np.ndarray,
    metadata: dict[str, Any],
    core_mask: np.ndarray | None = None,
) -> FitsBlendResult:
    """Replay one blend from recorded metadata and verify all output hashes."""
    if metadata.get("generator_version") != GENERATOR_VERSION:
        raise ValueError(
            "generator version mismatch; exact replay requires the recorded version"
        )
    transform_metadata = metadata["transform"]
    bands = tuple(metadata["band_order"])
    scale_mapping = transform_metadata["flux_scales_by_band"]
    transform = BlendTransform(
        sample_seed=int(transform_metadata["sample_seed"]),
        shift_xy=tuple(transform_metadata["shift_xy_pixels"]),  # type: ignore[arg-type]
        flux_scales=tuple(float(scale_mapping[band]) for band in bands),  # type: ignore[arg-type]
        rng_algorithm=str(transform_metadata["rng_algorithm"]),
    )
    coordinates = metadata["coordinates_xy_pixels"]
    source_ids = metadata["source_ids"]
    group_ids = metadata["group_ids"]
    replayed = blend_fits_cutouts(
        target,
        contaminant,
        source_mask=source_mask,
        background_mask=background_mask,
        core_mask=core_mask,
        target_coordinate_xy=tuple(coordinates["target"]),  # type: ignore[arg-type]
        contaminant_coordinate_xy=tuple(coordinates["contaminant_input"]),  # type: ignore[arg-type]
        target_source_id=str(source_ids["target"]),
        target_group_id=str(group_ids["target"]),
        contaminant_source_id=str(source_ids["contaminant"]),
        contaminant_group_id=str(group_ids["contaminant"]),
        transform=transform,
        band_order=bands,
        psf_policy=str(metadata["psf"]["policy"]),
        allow_source_mask_at_border=bool(
            metadata["mask_policy"]["allow_source_mask_at_border"]
        ),
    )
    expected = metadata["hashes"]
    actual = replayed.metadata["hashes"]
    for key in (
        "target_sha256",
        "contaminant_input_sha256",
        "source_mask_input_sha256",
        "background_mask_input_sha256",
        "core_mask_input_sha256",
        "source_only_contaminant_sha256",
        "blend_sha256",
        "affected_mask_sha256",
        "core_mask_sha256",
    ):
        if expected.get(key) != actual.get(key):
            raise ScientificValidityError(f"exact replay hash mismatch for {key}")
    return replayed


def _translate_bilinear_no_wrap(
    array: np.ndarray, *, dx: float, dy: float
) -> np.ndarray:
    """Translate last two axes by forward bilinear splatting without wrapping."""
    value = np.asarray(array)
    if value.ndim < 2 or not np.issubdtype(value.dtype, np.floating):
        raise TypeError("translated array must be a floating array with >=2 axes")
    if not math.isfinite(dx) or not math.isfinite(dy):
        raise ValueError("dx and dy must be finite")

    x_floor = math.floor(dx)
    y_floor = math.floor(dy)
    x_fraction = float(dx - x_floor)
    y_fraction = float(dy - y_floor)
    output = np.zeros_like(value)
    x_terms = ((x_floor, 1.0 - x_fraction), (x_floor + 1, x_fraction))
    y_terms = ((y_floor, 1.0 - y_fraction), (y_floor + 1, y_fraction))
    for x_offset, x_weight in x_terms:
        if x_weight == 0.0:
            continue
        for y_offset, y_weight in y_terms:
            weight = x_weight * y_weight
            if weight == 0.0:
                continue
            output += _translate_integer_no_wrap(
                value, dx=x_offset, dy=y_offset
            ) * np.asarray(weight, dtype=value.dtype)
    return output


def _translate_integer_no_wrap(
    array: np.ndarray, *, dx: int, dy: int
) -> np.ndarray:
    output = np.zeros_like(array)
    height, width = array.shape[-2:]
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return output
    destination_x0 = source_x0 + dx
    destination_x1 = source_x1 + dx
    destination_y0 = source_y0 + dy
    destination_y1 = source_y1 + dy
    output[..., destination_y0:destination_y1, destination_x0:destination_x1] = (
        array[..., source_y0:source_y1, source_x0:source_x1]
    )
    return output


def _validate_fits_cube(array: np.ndarray, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(array):
        raise TypeError(f"{name} must be a plain array; resolve its mask explicitly")
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"{name} must have band-first shape (3, height, width)")
    if value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError(f"{name} spatial dimensions must be positive")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must have a floating FITS-array dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite pixels")
    return value


def _validate_mask(
    mask: np.ndarray, shape: tuple[int, int], name: str
) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape or value.dtype != np.bool_:
        raise TypeError(f"{name} must be a boolean array with shape {shape}")
    if not value.any():
        raise ValueError(f"{name} must contain at least one selected pixel")
    return np.array(value, dtype=bool, copy=True, order="C")


def _validate_band_order(band_order: Sequence[str]) -> tuple[str, str, str]:
    bands = tuple(str(band).strip().lower() for band in band_order)
    if bands != DEFAULT_BAND_ORDER:
        raise ValueError(
            f"band_order must be the canonical survey order {DEFAULT_BAND_ORDER!r}"
        )
    return bands  # type: ignore[return-value]


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a nonempty string")


def _validate_xy(
    xy: tuple[float, float],
    name: str,
    *,
    require_in_frame: tuple[int, int] | None,
) -> tuple[float, float]:
    if len(xy) != 2:
        raise ValueError(f"{name} must contain (x, y)")
    x, y = float(xy[0]), float(xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    if require_in_frame is not None:
        height, width = require_in_frame
        if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
            raise ValueError(f"{name} must lie inside the input image")
    return x, y


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("sample_seed must be an integer")
    value = int(seed)
    if not 0 <= value < 2**64:
        raise ValueError("sample_seed must be in [0, 2**64)")
    return value


def _validate_range(
    bounds: tuple[float, float], name: str
) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain (low, high)")
    low, high = float(bounds[0]), float(bounds[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{name} must be finite with low <= high")
    return low, high


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _mark_read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array
