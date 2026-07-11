#!/usr/bin/env python3
"""Study global, invertible normalization using accepted DR10 train sources only.

This command reads split metadata for role validation, but opens FITS pixels
only for rows whose role is ``train`` and whose source decision is
``accepted_clean_source``.  It never opens or visualizes lockbox pixels.  Every
normalization uses fixed training-derived per-band parameters; no cutout is
independently rescaled, clipped, or converted to display RGB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


STUDY_VERSION = "dr10_normalization_study_v1"
SAMPLING_VERSION = "immutable_source_group_pixel_hash_v1"
BANDS = ("g", "r", "z")
ROUNDTRIP_RTOL = 1.0e-10
ROUNDTRIP_ATOL = 1.0e-12
WCS_ALIGNMENT_TOLERANCE_ARCSEC = 1.0e-4
WCS_ALIGNMENT_ROWS_PER_CHUNK = 64
OFFICIAL_NANOMAGGY_IVAR_UNITS = {
    "1/(nanomaggies)^2perpixel",
    "1/(nanomaggy)^2perpixel",
    "1/(nanomaggies)^2",
    "1/(nanomaggy)^2",
    "1/nanomaggies^2",
    "1/nanomaggy^2",
}
MIN_IVAR_POSITIVE_SAMPLES_PER_BAND = 100
ROLES = {
    "train",
    "validation",
    "calibration",
    "development_test",
    "future_lockbox",
}


@dataclass(frozen=True)
class NormalizationSpec:
    method: str
    band: str
    scale: float = 1.0
    softening: float = 1.0
    asinh_normalizer: float = 1.0
    low: float = 0.0
    high: float = 1.0
    fit_sample_count: int = 0
    fit_scope: str = "accepted_train_sources_only"


@dataclass
class GrzCube:
    data: np.ndarray
    header: fits.Header
    wcs: WCS
    hdu_index: int
    image_type: str
    pixel_hash: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_pixel_hash(array: np.ndarray) -> str:
    """Match the FITS auditor's canonical decoded-pixel hash exactly."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def parse_bands(value: str) -> tuple[str, ...]:
    tokens = [item.lower() for item in re.findall(r"[A-Za-z]+", value)]
    if len(tokens) == 1 and len(tokens[0]) > 1:
        return tuple(tokens[0])
    return tuple(tokens)


def read_grz_cube(
    path: Path, *, expected_image_type: str = "IMAGE", hdu_index: int | None = None
) -> GrzCube:
    resolved = path.expanduser().resolve()
    if "lockbox" in str(resolved).lower():
        raise ValueError(f"Refusing lockbox-like pixel path: {resolved}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(resolved, mode="readonly", memmap=False, checksum=True) as hdul:
            hdul.verify("exception")
            expected_type = expected_image_type.strip().upper()
            if expected_type not in {"IMAGE", "INVVAR"}:
                raise ValueError(f"Unsupported expected IMAGETYP={expected_type!r}")
            if hdu_index is not None:
                if hdu_index < 0 or hdu_index >= len(hdul):
                    raise ValueError(f"HDU index {hdu_index} is out of range for {resolved}")
                selected_index = hdu_index
                selected_hdu = hdul[hdu_index]
                if selected_hdu.data is None:
                    raise ValueError(f"HDU {hdu_index} has no image data in {resolved}")
            else:
                matches = [
                    (index, hdu)
                    for index, hdu in enumerate(hdul)
                    if hdu.data is not None
                    and str(hdu.header.get("IMAGETYP", "")).strip().upper()
                    == expected_type
                ]
                if len(matches) != 1:
                    inventory = [
                        {
                            "hdu": index,
                            "imagetyp": str(hdu.header.get("IMAGETYP", "")).strip(),
                            "shape": None if hdu.data is None else list(hdu.data.shape),
                        }
                        for index, hdu in enumerate(hdul)
                    ]
                    raise ValueError(
                        f"Expected exactly one IMAGETYP={expected_type} HDU in "
                        f"{resolved}; inventory={inventory}"
                    )
                selected_index, selected_hdu = matches[0]
            image_type = str(selected_hdu.header.get("IMAGETYP", "")).strip().upper()
            if image_type != expected_type:
                raise ValueError(
                    f"HDU {selected_index} IMAGETYP={image_type!r}; expected {expected_type}"
                )
            array = np.asarray(selected_hdu.data).copy()
            if array.ndim != 3 or array.shape[0] != 3:
                raise ValueError(f"Expected (3,H,W) data in {resolved}; got {array.shape}")
            header = selected_hdu.header.copy()
            bands = parse_bands(str(header.get("BANDS", "")))
            if bands != BANDS:
                raise ValueError(f"Expected explicit BANDS=grz in {resolved}; got {bands}")
            wcs = WCS(header).celestial
            if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
                raise ValueError(f"Invalid celestial WCS in {resolved}")
            return GrzCube(
                data=array,
                header=header,
                wcs=wcs,
                hdu_index=int(selected_index),
                image_type=image_type,
                pixel_hash=semantic_pixel_hash(array),
            )


def transform_flux(
    values: np.ndarray,
    spec: NormalizationSpec,
    ivar: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if spec.method == "fixed_per_band_scale":
        if not np.isfinite(spec.scale) or spec.scale <= 0:
            raise ValueError("fixed scale must be positive")
        return array / spec.scale
    if spec.method == "robust_signed_asinh":
        if spec.softening <= 0 or spec.asinh_normalizer <= 0:
            raise ValueError("asinh parameters must be positive")
        return np.arcsinh(array / spec.softening) / spec.asinh_normalizer
    if spec.method == "global_percentile":
        if not spec.high > spec.low:
            raise ValueError("global percentile high must exceed low")
        return 2.0 * (array - spec.low) / (spec.high - spec.low) - 1.0
    if spec.method == "variance_aware":
        if ivar is None:
            raise ValueError("variance-aware transform requires verified IVAR")
        weight = np.asarray(ivar, dtype=np.float64)
        if weight.shape != array.shape:
            raise ValueError("IVAR shape differs from flux shape")
        valid = np.isfinite(array) & np.isfinite(weight) & (weight > 0)
        result = np.full(array.shape, np.nan, dtype=np.float64)
        result[valid] = array[valid] * np.sqrt(weight[valid])
        return result
    raise ValueError(f"Unknown normalization method: {spec.method}")


def inverse_flux(
    normalized: np.ndarray,
    spec: NormalizationSpec,
    ivar: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(normalized, dtype=np.float64)
    if spec.method == "fixed_per_band_scale":
        return array * spec.scale
    if spec.method == "robust_signed_asinh":
        return spec.softening * np.sinh(array * spec.asinh_normalizer)
    if spec.method == "global_percentile":
        return (array + 1.0) * 0.5 * (spec.high - spec.low) + spec.low
    if spec.method == "variance_aware":
        if ivar is None:
            raise ValueError("variance-aware inverse requires the same verified IVAR")
        weight = np.asarray(ivar, dtype=np.float64)
        if weight.shape != array.shape:
            raise ValueError("IVAR shape differs from normalized shape")
        valid = np.isfinite(array) & np.isfinite(weight) & (weight > 0)
        result = np.full(array.shape, np.nan, dtype=np.float64)
        result[valid] = array[valid] / np.sqrt(weight[valid])
        return result
    raise ValueError(f"Unknown normalization method: {spec.method}")


def fit_specs(samples_by_band: dict[str, np.ndarray]) -> dict[tuple[str, str], NormalizationSpec]:
    specs: dict[tuple[str, str], NormalizationSpec] = {}
    for band in BANDS:
        values = np.asarray(samples_by_band[band], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 100:
            raise ValueError(f"Insufficient finite train pixels for band {band}: {values.size}")
        absolute = np.abs(values)
        fixed_scale = float(np.percentile(absolute, 99.5))
        median = float(np.median(values))
        robust_scale = 1.4826 * float(np.median(np.abs(values - median)))
        high_absolute = float(np.percentile(absolute, 99.9))
        positive_floor = max(np.finfo(np.float64).eps, high_absolute * 1e-12)
        fixed_scale = max(fixed_scale, positive_floor)
        softening = max(robust_scale, positive_floor)
        normalizer = max(float(np.arcsinh(high_absolute / softening)), 1.0)
        low, high = map(float, np.percentile(values, [0.1, 99.9]))
        if not high > low:
            high = low + positive_floor
        specs[("fixed_per_band_scale", band)] = NormalizationSpec(
            "fixed_per_band_scale", band, scale=fixed_scale, fit_sample_count=values.size
        )
        specs[("robust_signed_asinh", band)] = NormalizationSpec(
            "robust_signed_asinh",
            band,
            softening=softening,
            asinh_normalizer=normalizer,
            fit_sample_count=values.size,
        )
        specs[("global_percentile", band)] = NormalizationSpec(
            "global_percentile",
            band,
            low=low,
            high=high,
            fit_sample_count=values.size,
        )
    return specs


def select_train_rows(
    split_rows: Sequence[dict[str, str]], split_manifest_parent: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    role_counts: dict[str, int] = {role: 0 for role in sorted(ROLES)}
    group_roles: dict[str, set[str]] = {}
    path_roles: dict[str, set[str]] = {}
    train: list[dict[str, Any]] = []
    for row in split_rows:
        role = str(row.get("role", "")).strip()
        if role not in ROLES:
            raise ValueError(f"Unknown or missing role in split manifest: {role!r}")
        role_counts[role] += 1
        group_id = str(row.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("Split manifest row lacks group_id")
        group_roles.setdefault(group_id, set()).add(role)
        raw_path = str(row.get("fits_path") or row.get("path") or "").strip()
        if not raw_path:
            raise ValueError("Split manifest row lacks fits_path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = split_manifest_parent / path
        path_key = str(path.resolve())
        path_roles.setdefault(path_key, set()).add(role)
        if role != "train":
            # Crucially, no existence, stat, FITS, image, or visualization access here.
            continue
        if row.get("source_quality_decision") != "accepted_clean_source":
            raise ValueError("Train row is not an accepted clean source")
        if "lockbox" in path_key.lower():
            raise ValueError(f"Train row uses lockbox-like path: {path_key}")
        train.append({**row, "_resolved_fits_path": path_key})
    leaking_groups = {group: roles for group, roles in group_roles.items() if len(roles) != 1}
    leaking_paths = {path: roles for path, roles in path_roles.items() if len(roles) != 1}
    if leaking_groups:
        raise ValueError(f"Cross-role source groups in split manifest: {leaking_groups}")
    if leaking_paths:
        raise ValueError(f"The same FITS path appears across roles: {leaking_paths}")
    empty_roles = sorted(role for role, count in role_counts.items() if count == 0)
    if empty_roles:
        raise ValueError(f"Five-way split has empty roles: {empty_roles}")
    if not train:
        raise ValueError("No accepted train sources")
    return sorted(train, key=lambda row: (str(row.get("group_id")), str(row.get("source_id")))), role_counts


def quality_index(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        raw = str(row.get("path") or row.get("fits_path") or "").strip()
        if not raw:
            continue
        key = str(Path(raw).expanduser().resolve())
        if key in result and result[key] != row:
            raise ValueError(f"Ambiguous FITS quality rows for {key}")
        result[key] = row
    return result


def validate_train_quality(
    train_rows: Sequence[dict[str, Any]], quality_rows: Sequence[dict[str, str]]
) -> str:
    index = quality_index(quality_rows)
    units: set[str] = set()
    for row in train_rows:
        path = str(row["_resolved_fits_path"])
        quality = index.get(path)
        if quality is None:
            raise ValueError(f"Train FITS lacks an audit-quality row: {path}")
        for field in (
            "frozen_fits_semantics_valid",
            "manifest_request_semantics_valid",
            "fits_structure_valid",
            "band_order_valid",
            "wcs_valid",
        ):
            if not truthy(quality.get(field)):
                raise ValueError(f"Train FITS failed {field}: {path}")
        expected_hash = str(row.get("pixel_hash", "")).strip().lower()
        audited_hash = str(quality.get("pixel_hash", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"Train split row lacks a valid audited pixel hash: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", audited_hash):
            raise ValueError(f"FITS quality row lacks a valid decoded-pixel hash: {path}")
        if expected_hash != audited_hash:
            raise ValueError(f"Split/audit pixel hash mismatch: {path}")
        preallocation_pixel_hash = str(
            row.get("preallocation_pixel_hash", "")
        ).strip().lower()
        if not truthy(row.get("preallocation_hash_revalidation_pass")):
            raise ValueError(
                f"Train split row lacks a passing preallocation hash gate: {path}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", preallocation_pixel_hash):
            raise ValueError(
                f"Train split row lacks preallocation decoded-pixel hash: {path}"
            )
        if preallocation_pixel_hash != audited_hash:
            raise ValueError(
                f"Preallocation/audit decoded-pixel hash mismatch: {path}"
            )
        audited_file_hash = str(quality.get("file_sha256", "")).strip().lower()
        split_file_hash = str(row.get("preallocation_file_sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", audited_file_hash):
            raise ValueError(f"FITS quality row lacks a valid full-file SHA-256: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", split_file_hash):
            raise ValueError(f"Train split row lacks a valid full-file SHA-256: {path}")
        if split_file_hash != audited_file_hash:
            raise ValueError(f"Split/audit full-file SHA-256 mismatch: {path}")
        row["_audited_file_sha256"] = audited_file_hash
        unit_source = str(quality.get("unit_source", "")).strip()
        unit_value = str(quality.get("unit_value", "")).strip()
        if unit_source == "unresolved" or not unit_value:
            raise ValueError(f"Train FITS has unresolved flux units: {path}")
        units.add(unit_value)
    if len(units) != 1:
        raise ValueError(f"Train FITS units are not uniform: {sorted(units)}")
    return next(iter(units))


def verify_current_pixel_hash(
    cube: GrzCube, expected_hash: str, path: Path
) -> None:
    expected = str(expected_hash).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Missing canonical expected pixel hash for {path}")
    if cube.pixel_hash != expected:
        raise ValueError(f"Current decoded pixels differ from audited split hash: {path}")


def deterministic_sample(
    values: np.ndarray, count: int, token: str
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values).ravel()
    finite_indices = np.flatnonzero(np.isfinite(flat))
    if not finite_indices.size:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    count = min(count, finite_indices.size)
    digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
    offset = digest % finite_indices.size
    positions = (offset + np.linspace(0, finite_indices.size - 1, count, dtype=np.int64)) % finite_indices.size
    indices = finite_indices[positions]
    return np.asarray(flat[indices], dtype=np.float64), indices


def immutable_sampling_token(
    source_id: str, group_id: str, pixel_hash: str, band: str
) -> str:
    if not source_id or not group_id:
        raise ValueError("Sampling token requires stable source and group IDs")
    if not re.fullmatch(r"[0-9a-f]{64}", pixel_hash):
        raise ValueError("Sampling token requires a canonical decoded-pixel hash")
    if band not in BANDS:
        raise ValueError(f"Unknown sampling band: {band}")
    return f"{SAMPLING_VERSION}|{source_id}|{group_id}|{pixel_hash}|{band}"


def cap_samples(chunks: Sequence[np.ndarray], maximum: int) -> np.ndarray:
    values = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    if values.size > maximum:
        indices = np.linspace(0, values.size - 1, maximum, dtype=np.int64)
        values = values[indices]
    return np.asarray(values, dtype=np.float64)


def canonical_unit(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def _optional_hdu_index(value: Any, field: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        result = int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer HDU index") from exc
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def load_verified_ivar_index(
    path: Path | None,
    *,
    allowed_science_paths: set[str],
    science_flux_unit: str,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_csv(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        science_path = str(row.get("science_fits_path") or row.get("path") or "").strip()
        ivar_path = str(row.get("ivar_fits_path") or "").strip()
        if not science_path or not ivar_path:
            raise ValueError("Verified IVAR manifest requires science_fits_path and ivar_fits_path")
        key = str(Path(science_path).expanduser().resolve())
        if key not in allowed_science_paths:
            raise ValueError(
                "IVAR manifest must be train-only; refusing non-train science path: "
                f"{key}"
            )
        if not truthy(row.get("semantics_verified")) or not truthy(
            row.get("wcs_alignment_verified") or row.get("alignment_verified")
        ):
            raise ValueError(
                "Every supplied IVAR row must explicitly verify semantics and WCS alignment"
            )
        declared_science_unit = str(row.get("science_flux_unit") or "").strip()
        if canonical_unit(declared_science_unit) != canonical_unit(science_flux_unit):
            raise ValueError(
                f"IVAR manifest science unit {declared_science_unit!r} does not match "
                f"audited unit {science_flux_unit!r}"
            )
        ivar_unit = str(row.get("ivar_unit") or row.get("unit") or "").strip()
        if canonical_unit(ivar_unit) not in OFFICIAL_NANOMAGGY_IVAR_UNITS:
            raise ValueError(
                f"Unsupported IVAR unit {ivar_unit!r}; expected documented inverse "
                "nanomaggy-squared coadd units"
            )
        ivar_pixel_hash = str(row.get("ivar_pixel_hash") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", ivar_pixel_hash):
            raise ValueError("Every IVAR row requires a canonical ivar_pixel_hash")
        if key in result:
            raise ValueError(f"Duplicate IVAR manifest entry: {key}")
        result[key] = {
            **row,
            "_resolved_ivar_path": str(Path(ivar_path).expanduser().resolve()),
            "_science_hdu": _optional_hdu_index(row.get("science_hdu"), "science_hdu"),
            "_ivar_hdu": _optional_hdu_index(row.get("ivar_hdu"), "ivar_hdu"),
            "_ivar_unit": ivar_unit,
            "_ivar_pixel_hash": ivar_pixel_hash,
        }
    return result


def wcs_aligned(science: GrzCube, ivar: GrzCube) -> bool:
    if science.data.shape != ivar.data.shape:
        return False
    height, width = science.data.shape[1:]
    # Verify every pixel center, not only corners or a diagonal. Row chunks keep
    # this optional brick-product check memory-bounded for larger cutouts.
    for row_start in range(0, height, WCS_ALIGNMENT_ROWS_PER_CHUNK):
        row_stop = min(row_start + WCS_ALIGNMENT_ROWS_PER_CHUNK, height)
        grid_y, grid_x = np.mgrid[row_start:row_stop, 0:width]
        x, y = grid_x.ravel(), grid_y.ravel()
        s_ra, s_dec = science.wcs.pixel_to_world_values(x, y)
        i_ra, i_dec = ivar.wcs.pixel_to_world_values(x, y)
        s_ra = np.asarray(s_ra)
        s_dec = np.asarray(s_dec)
        i_ra = np.asarray(i_ra)
        i_dec = np.asarray(i_dec)
        delta_ra = (s_ra - i_ra + 180.0) % 360.0 - 180.0
        separation_arcsec = np.hypot(
            delta_ra * np.cos(np.deg2rad(s_dec)), s_dec - i_dec
        ) * 3600.0
        if not np.all(np.isfinite(separation_arcsec)):
            return False
        if np.max(separation_arcsec) >= WCS_ALIGNMENT_TOLERANCE_ARCSEC:
            return False
    return True


def verify_header_unit(cube: GrzCube, expected_unit: str, label: str) -> None:
    """Reject a FITS BUNIT that contradicts the audited/documented unit."""
    header_unit = str(cube.header.get("BUNIT", "")).strip()
    if header_unit and canonical_unit(header_unit) != canonical_unit(expected_unit):
        raise ValueError(
            f"{label} BUNIT {header_unit!r} conflicts with declared unit "
            f"{expected_unit!r}"
        )


def study_rows(
    specs: dict[tuple[str, str], NormalizationSpec],
    samples: dict[str, np.ndarray],
    ivar_flux_samples: dict[str, np.ndarray],
    ivar_samples: dict[str, np.ndarray],
    input_flux_unit: str,
    ivar_unit: str = "",
    ivar_domain_counts: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("fixed_per_band_scale", "robust_signed_asinh", "global_percentile"):
        for band in BANDS:
            spec = specs[(method, band)]
            values = samples[band]
            normalized = transform_flux(values, spec)
            replay = inverse_flux(normalized, spec)
            difference = np.abs(replay - values)
            denom = np.maximum(np.abs(values), np.finfo(np.float64).tiny)
            roundtrip_pass = bool(
                np.isfinite(normalized).all()
                and np.isfinite(replay).all()
                and np.allclose(
                    replay,
                    values,
                    rtol=ROUNDTRIP_RTOL,
                    atol=ROUNDTRIP_ATOL,
                )
            )
            rows.append(
                {
                    **asdict(spec),
                    "input_flux_unit": input_flux_unit,
                    "parameter_flux_unit": input_flux_unit,
                    "normalized_unit": "dimensionless",
                    "ivar_unit": "not_used",
                    "status": "available",
                    "input_negative_fraction": float(np.mean(values < 0)),
                    "normalized_finite_fraction": float(np.mean(np.isfinite(normalized))),
                    "normalized_p001": float(np.percentile(normalized, 0.1)),
                    "normalized_p01": float(np.percentile(normalized, 1.0)),
                    "normalized_median": float(np.median(normalized)),
                    "normalized_p99": float(np.percentile(normalized, 99.0)),
                    "normalized_p999": float(np.percentile(normalized, 99.9)),
                    "roundtrip_max_abs_error": float(np.max(difference)),
                    "roundtrip_max_rel_error": float(np.max(difference / denom)),
                    "roundtrip_rtol": ROUNDTRIP_RTOL,
                    "roundtrip_atol": ROUNDTRIP_ATOL,
                    "roundtrip_pass": roundtrip_pass,
                    "hidden_clipping": False,
                    "negative_flux_supported": True,
                    "global_parameters_only": True,
                    "physical_flux_recoverable": True,
                    "cross_band_color_recoverable_with_parameters": True,
                    "relative_source_order_monotonic": True,
                    "relative_flux_ratio_linear_in_model_space": method == "fixed_per_band_scale",
                }
            )
    for band in BANDS:
        flux = ivar_flux_samples.get(band, np.array([], dtype=float))
        weight = ivar_samples.get(band, np.array([], dtype=float))
        sampled_total, positive_count = (ivar_domain_counts or {}).get(
            band, (int(flux.size), int(flux.size))
        )
        positive_fraction = (
            float(positive_count / sampled_total) if sampled_total else float("nan")
        )
        spec = NormalizationSpec("variance_aware", band, fit_sample_count=int(flux.size))
        if positive_count < MIN_IVAR_POSITIVE_SAMPLES_PER_BAND or flux.size < MIN_IVAR_POSITIVE_SAMPLES_PER_BAND:
            rows.append(
                {
                    **asdict(spec),
                    "input_flux_unit": input_flux_unit,
                    "parameter_flux_unit": input_flux_unit,
                    "normalized_unit": "dimensionless_snr_like",
                    "ivar_unit": "unavailable",
                    "status": (
                        "unavailable_no_verified_ivar"
                        if sampled_total == 0
                        else "unavailable_insufficient_positive_ivar_domain"
                    ),
                    "ivar_sampled_domain_count": sampled_total,
                    "ivar_positive_domain_count": positive_count,
                    "ivar_positive_domain_fraction": positive_fraction,
                    "minimum_positive_ivar_samples_required": (
                        MIN_IVAR_POSITIVE_SAMPLES_PER_BAND
                    ),
                    "hidden_clipping": False,
                    "negative_flux_supported": True,
                    "global_parameters_only": False,
                    "physical_flux_recoverable": "requires_same_positive_ivar",
                    "cross_band_color_recoverable_with_parameters": "requires_same_ivar",
                    "relative_source_order_monotonic": False,
                    "relative_flux_ratio_linear_in_model_space": False,
                }
            )
            continue
        normalized = transform_flux(flux, spec, weight)
        replay = inverse_flux(normalized, spec, weight)
        valid = np.isfinite(normalized) & np.isfinite(replay)
        difference = np.abs(replay[valid] - flux[valid])
        denom = np.maximum(np.abs(flux[valid]), np.finfo(np.float64).tiny)
        roundtrip_pass = bool(
            valid.any()
            and np.allclose(
                replay[valid],
                flux[valid],
                rtol=ROUNDTRIP_RTOL,
                atol=ROUNDTRIP_ATOL,
            )
        )
        rows.append(
            {
                **asdict(spec),
                "input_flux_unit": input_flux_unit,
                "parameter_flux_unit": input_flux_unit,
                "normalized_unit": "dimensionless_snr_like",
                "ivar_unit": ivar_unit,
                "status": "available_verified_ivar_positive_domain",
                "ivar_sampled_domain_count": sampled_total,
                "ivar_positive_domain_count": positive_count,
                "ivar_positive_domain_fraction": positive_fraction,
                "minimum_positive_ivar_samples_required": (
                    MIN_IVAR_POSITIVE_SAMPLES_PER_BAND
                ),
                "input_negative_fraction": float(np.mean(flux < 0)),
                "normalized_finite_fraction": float(np.mean(np.isfinite(normalized))),
                "normalized_p001": float(np.percentile(normalized[valid], 0.1)),
                "normalized_p01": float(np.percentile(normalized[valid], 1.0)),
                "normalized_median": float(np.median(normalized[valid])),
                "normalized_p99": float(np.percentile(normalized[valid], 99.0)),
                "normalized_p999": float(np.percentile(normalized[valid], 99.9)),
                "roundtrip_max_abs_error": float(np.max(difference)),
                "roundtrip_max_rel_error": float(np.max(difference / denom)),
                "roundtrip_rtol": ROUNDTRIP_RTOL,
                "roundtrip_atol": ROUNDTRIP_ATOL,
                "roundtrip_pass": roundtrip_pass,
                "hidden_clipping": False,
                "negative_flux_supported": True,
                "global_parameters_only": False,
                "physical_flux_recoverable": "requires_same_positive_ivar",
                "cross_band_color_recoverable_with_parameters": "requires_same_ivar",
                "relative_source_order_monotonic": False,
                "relative_flux_ratio_linear_in_model_space": False,
            }
        )
    return rows


def validate_recommendation_gate(rows: Sequence[dict[str, Any]]) -> None:
    """Fail before recommendation if a global transform is nonfinite/noninvertible."""
    global_methods = {
        "fixed_per_band_scale",
        "robust_signed_asinh",
        "global_percentile",
    }
    expected = {(method, band) for method in global_methods for band in BANDS}
    observed: set[tuple[str, str]] = set()
    failures: list[str] = []
    for row in rows:
        method, band = str(row.get("method")), str(row.get("band"))
        if method not in global_methods:
            continue
        observed.add((method, band))
        finite_fraction = safe_float(row.get("normalized_finite_fraction"))
        if row.get("status") != "available":
            failures.append(f"{method}/{band}: unavailable")
        if not np.isfinite(finite_fraction) or finite_fraction != 1.0:
            failures.append(
                f"{method}/{band}: normalized finite fraction={finite_fraction}"
            )
        if not truthy(row.get("roundtrip_pass")):
            failures.append(f"{method}/{band}: round-trip tolerance failed")
        if truthy(row.get("hidden_clipping")):
            failures.append(f"{method}/{band}: hidden clipping reported")
    missing = expected - observed
    failures.extend(f"{method}/{band}: missing study row" for method, band in sorted(missing))
    if failures:
        raise ValueError(
            "Normalization recommendation gate failed: " + "; ".join(failures)
        )


def write_csv_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure_exclusive(fig: plt.Figure, path: Path) -> None:
    with path.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_example_figure(
    cube: np.ndarray,
    source_id: str,
    specs: dict[tuple[str, str], NormalizationSpec],
    study_by_key: dict[tuple[str, str], dict[str, Any]],
    path: Path,
) -> None:
    methods = ("fixed_per_band_scale", "robust_signed_asinh", "global_percentile")
    fig, axes = plt.subplots(len(methods), len(BANDS), figsize=(10, 9), squeeze=False)
    for row_index, method in enumerate(methods):
        for band_index, band in enumerate(BANDS):
            axis = axes[row_index, band_index]
            transformed = transform_flux(cube[band_index], specs[(method, band)])
            metrics = study_by_key[(method, band)]
            vmin = float(metrics["normalized_p01"])
            vmax = float(metrics["normalized_p99"])
            if not vmax > vmin:
                vmax = vmin + 1.0
            axis.imshow(
                transformed,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.set_title(f"{method}\n{band} band", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        f"TRAIN ONLY – normalization display example – {source_id}\n"
        "Global train-derived display limits; model arrays are not clipped",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save_figure_exclusive(fig, path)


def prepare_outputs(run_dir: Path) -> dict[str, Path]:
    root = run_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for child in ("diagnostics", "tables", "figures"):
        (root / child).mkdir(exist_ok=True)
    outputs = {
        "report": root / "diagnostics/fits_normalization_study.md",
        "statistics": root / "tables/normalization_statistics.csv",
        "examples": root / "figures/normalization_examples",
    }
    for key, path in outputs.items():
        if path.exists():
            raise FileExistsError(f"Refusing existing normalization output: {path}")
    outputs["examples"].mkdir(exist_ok=False)
    return outputs


def report_text(
    train_count: int,
    role_counts: dict[str, int],
    split_hash: str,
    sample_counts: dict[str, int],
    ivar_available: bool,
    example_count: int,
    input_flux_unit: str,
    fits_quality_hash: str,
    ivar_manifest_hash: str,
    ivar_unit: str,
) -> str:
    return f"""# FITS normalization study

Generated: `{utc_now()}`  
Study version: `{STUDY_VERSION}`  
Sampling version: `{SAMPLING_VERSION}`  
Split-manifest SHA-256: `{split_hash}`  
FITS-quality SHA-256: `{fits_quality_hash}`  
IVAR-manifest SHA-256: `{ivar_manifest_hash or 'not supplied'}`

## Data boundary

Statistics were learned from `{train_count}` sources with role `train` and
decision `accepted_clean_source` only. Role counts from metadata were
`{json.dumps(role_counts, sort_keys=True)}`. Pixel reads and the `{example_count}`
example figures were restricted to train paths before any FITS file was opened.
No validation, calibration, development-test, or future-lockbox pixels were
read or visualized. Non-train split metadata was used only to verify group/path
role disjointness; FITS-quality rows were dereferenced only for train paths.

Per-band deterministic sample counts were
`{json.dumps(sample_counts, sort_keys=True)}`. Percentiles are therefore
reproducible bounded-sample estimates, not falsely labeled exact full-corpus
quantiles. The audit-established, uniform input flux unit is
`{input_flux_unit}`; learned scales are expressed in that native unit.
Sampling tokens use immutable source ID, group ID, decoded-pixel hash, band,
and sampling version; absolute filesystem paths do not affect selected pixels.

## Compared transforms

1. `fixed_per_band_scale`: `y = x / s_b`, with `s_b` the train-only 99.5th
   percentile of `|x|`. This is linear, globally fixed, and exactly invertible.
2. `robust_signed_asinh`: `y = asinh(x / a_b) / c_b`, where `a_b` is the
   train-only robust scale and `c_b` maps the 99.9th percentile of `|x|` near
   unity. Its inverse is `x = a_b sinh(c_b y)`.
3. `global_percentile`: a fixed affine map from train-only 0.1/99.9
   percentiles to -1/+1. Values outside that interval are **not clipped**.
4. `variance_aware`: `{('evaluated only on the reported finite positive-IVAR domain of explicitly verified, WCS-aligned products' if ivar_available else 'not evaluated because no explicitly verified IVAR manifest was supplied')}`.
   Verified IVAR unit: `{ivar_unit or 'not available'}`.

The three global transforms retain negative sky-subtracted flux and use one
stored set of parameters per band, never per-cutout parameters. Thus sources
remain on a common scale and physical flux and colors can be recovered exactly
(within floating-point tolerance). Variance-aware scaling is invertible only
where the same finite positive IVAR is present; its lost-domain fraction is
reported and it is unavailable with fewer than
`{MIN_IVAR_POSITIVE_SAMPLES_PER_BAND}` positive samples per band. Signed asinh is monotonic but intentionally
compresses ratios in model space; ratios and colors must be interpreted after
inverse transformation.

## Initial recommendation

All nine global method/band rows passed the finite-output and round-trip gate
(`rtol={ROUNDTRIP_RTOL}`, `atol={ROUNDTRIP_ATOL}`) before this recommendation
was written.

Use the globally fitted `robust_signed_asinh` transform as the initial model
normalization. It accommodates the observed signed background while reducing
the leverage of bright cores, remains monotonic, and has an explicit inverse.
Retain the parameters in the training manifest and compute all scientific
metrics after conversion back to FITS flux space. This is an **initial
recommendation**, not a final choice; reconstruction and calibration tests on
validation/calibration roles must compare it with the fixed linear baseline.

Normalization and visualization remain separate. The example PNGs use global
train-derived display limits and grayscale panels; those rendered pixels are
never model inputs or scientific arrays.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--fits-quality", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ivar-manifest", type=Path)
    parser.add_argument("--pixels-per-source-band", type=int, default=4096)
    parser.add_argument("--max-samples-per-band", type=int, default=2_000_000)
    parser.add_argument("--examples", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pixels_per_source_band < 100 or args.max_samples_per_band < 100:
        raise SystemExit("Sampling limits must be at least 100")
    if args.examples < 0 or args.examples > 50:
        raise SystemExit("--examples must be in [0, 50]")
    split_path = args.source_split_manifest.expanduser().resolve()
    split_rows = read_csv(split_path)
    train_rows, role_counts = select_train_rows(split_rows, split_path.parent)
    quality_path = args.fits_quality.expanduser().resolve()
    quality_rows = read_csv(quality_path)
    input_flux_unit = validate_train_quality(train_rows, quality_rows)
    train_paths = {str(row["_resolved_fits_path"]) for row in train_rows}
    ivar_index = load_verified_ivar_index(
        args.ivar_manifest,
        allowed_science_paths=train_paths,
        science_flux_unit=input_flux_unit,
    )
    if ivar_index:
        missing = [
            row["_resolved_fits_path"]
            for row in train_rows
            if row["_resolved_fits_path"] not in ivar_index
        ]
        if missing:
            raise ValueError(f"Verified IVAR manifest lacks train paths: {missing[:5]}")

    split_hash = file_sha256(split_path)
    fits_quality_hash = file_sha256(quality_path)
    ivar_manifest_hash = (
        file_sha256(args.ivar_manifest.expanduser().resolve())
        if args.ivar_manifest is not None
        else ""
    )
    ivar_units = {str(row["_ivar_unit"]) for row in ivar_index.values()}
    if len(ivar_units) > 1:
        raise ValueError(f"IVAR manifest units are not uniform: {sorted(ivar_units)}")
    ivar_unit = next(iter(ivar_units), "")

    chunks: dict[str, list[np.ndarray]] = {band: [] for band in BANDS}
    ivar_flux_chunks: dict[str, list[np.ndarray]] = {band: [] for band in BANDS}
    ivar_chunks: dict[str, list[np.ndarray]] = {band: [] for band in BANDS}
    ivar_domain_counts: dict[str, list[int]] = {band: [0, 0] for band in BANDS}
    examples = train_rows[: args.examples]
    example_paths = {str(row["_resolved_fits_path"]) for row in examples}
    example_cubes: dict[str, np.ndarray] = {}
    training_input_descriptors: list[str] = []
    for row in train_rows:
        science_path = Path(str(row["_resolved_fits_path"]))
        if file_sha256(science_path) != str(row["_audited_file_sha256"]):
            raise ValueError(
                f"Train FITS full-file SHA-256 changed after audit/split: {science_path}"
            )
        ivar_metadata = ivar_index.get(str(science_path.resolve()))
        science = read_grz_cube(
            science_path,
            expected_image_type="IMAGE",
            hdu_index=(None if ivar_metadata is None else ivar_metadata["_science_hdu"]),
        )
        if file_sha256(science_path) != str(row["_audited_file_sha256"]):
            raise ValueError(
                f"Train FITS changed during normalization read: {science_path}"
            )
        expected_science_hash = str(row.get("pixel_hash", "")).strip().lower()
        verify_current_pixel_hash(science, expected_science_hash, science_path)
        verify_header_unit(science, input_flux_unit, "science IMAGE")
        manifest_science_hash = (
            str(ivar_metadata.get("science_pixel_hash") or "").strip().lower()
            if ivar_metadata is not None
            else ""
        )
        if manifest_science_hash and manifest_science_hash != science.pixel_hash:
            raise ValueError(f"IVAR manifest science hash mismatch: {science_path}")
        source_identity = str(row.get("source_id") or row.get("dr8_id") or "").strip()
        group_identity = str(row.get("group_id") or "")
        if not source_identity:
            raise ValueError(
                f"Train row lacks immutable source_id/dr8_id identity: {science_path}"
            )
        training_input_descriptors.append(
            f"{source_identity}|{group_identity}|{science.pixel_hash}|hdu={science.hdu_index}"
        )
        if str(science_path) in example_paths:
            example_cubes[str(science_path)] = science.data
        ivar_cube: GrzCube | None = None
        if ivar_metadata is not None:
            ivar_cube = read_grz_cube(
                Path(str(ivar_metadata["_resolved_ivar_path"])),
                expected_image_type="INVVAR",
                hdu_index=ivar_metadata["_ivar_hdu"],
            )
            if ivar_cube.pixel_hash != str(ivar_metadata["_ivar_pixel_hash"]):
                raise ValueError(f"Current IVAR pixels differ from manifest hash: {science_path}")
            verify_header_unit(ivar_cube, ivar_unit, "INVVAR")
            if not wcs_aligned(science, ivar_cube):
                raise ValueError(f"Science/IVAR WCS or shape mismatch: {science_path}")
            if np.any(np.isfinite(ivar_cube.data) & (ivar_cube.data < 0)):
                raise ValueError(f"Negative values in verified IVAR: {science_path}")
        for index, band in enumerate(BANDS):
            values, indices = deterministic_sample(
                science.data[index],
                args.pixels_per_source_band,
                immutable_sampling_token(
                    source_identity, group_identity, science.pixel_hash, band
                ),
            )
            chunks[band].append(values)
            if ivar_cube is not None:
                flux_flat = np.asarray(science.data[index]).ravel()
                ivar_flat = np.asarray(ivar_cube.data[index]).ravel()
                ivar_domain_counts[band][0] += int(indices.size)
                valid_indices = indices[
                    np.isfinite(ivar_flat[indices]) & (ivar_flat[indices] > 0)
                ]
                ivar_domain_counts[band][1] += int(valid_indices.size)
                if valid_indices.size:
                    ivar_flux_chunks[band].append(
                        np.asarray(flux_flat[valid_indices], dtype=np.float64)
                    )
                    ivar_chunks[band].append(
                        np.asarray(ivar_flat[valid_indices], dtype=np.float64)
                    )

    samples = {
        band: cap_samples(chunks[band], args.max_samples_per_band) for band in BANDS
    }
    ivar_flux_samples = {
        band: cap_samples(ivar_flux_chunks[band], args.max_samples_per_band)
        for band in BANDS
    }
    ivar_samples = {
        band: cap_samples(ivar_chunks[band], args.max_samples_per_band)
        for band in BANDS
    }
    specs = fit_specs(samples)
    statistics = study_rows(
        specs,
        samples,
        ivar_flux_samples,
        ivar_samples,
        input_flux_unit,
        ivar_unit,
        {
            band: (counts[0], counts[1])
            for band, counts in ivar_domain_counts.items()
        },
    )
    validate_recommendation_gate(statistics)
    training_input_set_sha256 = hashlib.sha256(
        "\n".join(sorted(training_input_descriptors)).encode("utf-8")
    ).hexdigest()
    for row in statistics:
        row.update(
            study_version=STUDY_VERSION,
            sampling_version=SAMPLING_VERSION,
            sampling_token_fields="source_id|group_id|decoded_pixel_hash|band",
            split_manifest_sha256=split_hash,
            fits_quality_sha256=fits_quality_hash,
            ivar_manifest_sha256=ivar_manifest_hash or "not_supplied",
            training_input_set_sha256=training_input_set_sha256,
            wcs_alignment_grid="every_pixel_center_chunked",
            wcs_alignment_tolerance_arcsec=WCS_ALIGNMENT_TOLERANCE_ARCSEC,
            recommendation_gate_pass=True,
        )
    outputs = prepare_outputs(args.run_dir)
    write_csv_exclusive(outputs["statistics"], statistics)

    study_by_key = {
        (str(row["method"]), str(row["band"])): row
        for row in statistics
        if row["status"] == "available"
    }
    for index, row in enumerate(examples, start=1):
        cube = example_cubes[str(row["_resolved_fits_path"])]
        source_id = str(row.get("source_id") or row.get("catalog_row_index") or index)
        make_example_figure(
            cube,
            source_id,
            specs,
            study_by_key,
            outputs["examples"] / f"train_normalization_example_{index:03d}.png",
        )
    report = report_text(
        train_count=len(train_rows),
        role_counts=role_counts,
        split_hash=split_hash,
        sample_counts={band: int(samples[band].size) for band in BANDS},
        ivar_available=bool(ivar_index),
        example_count=len(examples),
        input_flux_unit=input_flux_unit,
        fits_quality_hash=fits_quality_hash,
        ivar_manifest_hash=ivar_manifest_hash,
        ivar_unit=ivar_unit,
    )
    with outputs["report"].open("x", encoding="utf-8") as handle:
        handle.write(report.rstrip() + "\n")
    print(
        json.dumps(
            {
                "study_version": STUDY_VERSION,
                "train_source_count": len(train_rows),
                "role_counts": role_counts,
                "verified_ivar": bool(ivar_index),
                "training_input_set_sha256": training_input_set_sha256,
                "recommendation_gate_pass": True,
                "lockbox_pixels_accessed": False,
                "lockbox_visualized": False,
                "run_dir": str(args.run_dir.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
