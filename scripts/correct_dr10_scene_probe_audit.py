#!/usr/bin/env python3
"""Append-only scientific correction for the DR10 scene-probe audit.

This command writes only ``*_v2`` audit products. It preserves the initial
tables and figures as superseded evidence, revalidates every input role/hash,
uses independent catalog-centered morphology apertures with empirical blank
nulls, and keeps central-only extraction blocked for summed scene models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import astropy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

try:
    from scripts import audit_dr10_scene_triplets as base
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import audit_dr10_scene_triplets as base


EXPECTED_LAYERS = {
    "observed": "ls-dr10-south",
    "model": "ls-dr10-model",
    "residual": "ls-dr10-resid",
}
EXPECTED_SHAPE = (3, 256, 256)
EXPECTED_BANDS = "grz"
EXPECTED_SCALE = 0.262
CLOSURE_RMSE_NOISE_MAX = 0.01
CLOSURE_L1_MAX = 0.001
CLOSURE_P9999_NOISE_MAX = 0.1
CLOSURE_PEAK_RELATIVE_MAX = 5e-5
UNIT_TEXT = "nanomaggies per coadd pixel"
OFFICIAL_FILES = "https://www.legacysurvey.org/dr10/files/"
OFFICIAL_CATALOGS = "https://www.legacysurvey.org/dr10/catalogs/"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str) -> None:
    write_exclusive(path, value.encode("utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def save_figure(path: Path, figure: plt.Figure, dpi: int = 130) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", metadata={"Software": "correct_dr10_scene_probe_audit.py"})
    plt.close(figure)


def bool_int(value: Any) -> int:
    return int(bool(value))


def text_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_query(row: dict[str, str], source: dict[str, str]) -> tuple[bool, str]:
    query = parse_qs(urlparse(row["request_url"]).query)
    expected = {
        "layer": row["layer"],
        "bands": "grz",
        "size": "256",
        "pixscale": "0.262000",
    }
    errors: list[str] = []
    for key, value in expected.items():
        if query.get(key, [""])[0] != value:
            errors.append(f"{key}_mismatch")
    for key in ("ra", "dec"):
        try:
            if abs(float(query[key][0]) - float(source[key])) > 5e-10:
                errors.append(f"{key}_mismatch")
        except Exception:
            errors.append(f"{key}_missing")
    return not errors, ";".join(errors)


def validate_inputs(
    run_dir: Path,
    foundation_run: Path,
    sources: list[dict[str, str]],
    grouped: dict[str, dict[str, dict[str, str]]],
    catalog_manifest: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        source_id = source["source_id"]
        for product in ("observed", "model", "residual"):
            manifest = grouped[source_id][product]
            path = Path(manifest["relative_path"])
            raw_path = Path(manifest["raw_response_path"])
            header_path = Path(manifest["response_headers_path"])
            actual_hash = sha256_file(path)
            raw_hash = sha256_file(raw_path)
            header_hash = sha256_file(header_path)
            query_valid, query_error = validate_query(manifest, source)
            layer_valid = manifest["layer"] == EXPECTED_LAYERS[product]
            product_valid = manifest["product"] == product
            hash_valid = actual_hash == manifest["sha256"]
            raw_valid = raw_hash == manifest["raw_response_sha256"] and raw_hash == actual_hash
            headers_valid = header_hash == manifest["response_headers_sha256"]
            foundation_match = ""
            if product == "observed":
                foundation_match = bool_int(actual_hash == source["foundation_fov_sha256"])
            passed = all((query_valid, layer_valid, product_valid, hash_valid, raw_valid, headers_valid))
            rows.append(
                {
                    "source_id": source_id,
                    "product": product,
                    "expected_layer": EXPECTED_LAYERS[product],
                    "manifest_layer": manifest["layer"],
                    "layer_role_valid": bool_int(layer_valid and product_valid),
                    "request_parameters_valid": bool_int(query_valid),
                    "request_parameter_error": query_error,
                    "validated_path": str(path),
                    "manifest_sha256": manifest["sha256"],
                    "actual_sha256": actual_hash,
                    "file_hash_valid": bool_int(hash_valid),
                    "raw_response_path": str(raw_path),
                    "raw_response_hash_valid": bool_int(raw_valid),
                    "response_headers_path": str(header_path),
                    "response_headers_hash_valid": bool_int(headers_valid),
                    "observed_matches_foundation_fov_hash": foundation_match,
                    "input_integrity_pass": bool_int(passed),
                }
            )
        catalog = catalog_manifest[source_id]
        catalog_path = Path(catalog["relative_path"])
        raw_catalog = Path(catalog["raw_response_path"])
        header_path = Path(catalog["response_headers_path"])
        actual = sha256_file(catalog_path)
        raw_actual = sha256_file(raw_catalog)
        header_actual = sha256_file(header_path)
        parsed = urlparse(catalog["request_url"])
        query = parse_qs(parsed.query)
        bounds_valid = False
        try:
            bounds_valid = (
                parsed.path.endswith("/viewer/ls-dr10-south/cat.fits")
                and float(query["ralo"][0]) <= float(source["ra"]) <= float(query["rahi"][0])
                and float(query["declo"][0]) <= float(source["dec"]) <= float(query["dechi"][0])
            )
        except Exception:
            bounds_valid = False
        with fits.open(catalog_path, mode="readonly", memmap=False, checksum=True) as hdul:
            data = hdul[1].data
            names = {name.lower(): name for name in data.names}
            coords = SkyCoord(
                np.asarray(data[names["ra"]], dtype=float) * u.deg,
                np.asarray(data[names["dec"]], dtype=float) * u.deg,
            )
            target = SkyCoord(float(source["ra"]) * u.deg, float(source["dec"]) * u.deg)
            separations = target.separation(coords).arcsec
            nearest = int(np.argmin(separations))
            central = data[nearest]
            catalog_semantics_valid = (
                len(data) == int(catalog["row_count"])
                and nearest == int(catalog["central_row_index"])
                and abs(float(separations[nearest]) - float(catalog["central_separation_arcsec"])) < 1e-8
                and int(central[names["release"]]) == int(catalog["release"])
                and int(central[names["brickid"]]) == int(catalog["brickid"])
                and text_value(central[names["brickname"]]) == catalog["brickname"]
                and int(central[names["objid"]]) == int(catalog["objid"])
                and text_value(central[names["type"]]) == catalog["type"]
                and all(
                    abs(float(central[names[f"psfsize_{band}"]]) - float(catalog[f"psfsize_{band}_arcsec"])) < 1e-7
                    for band in base.BANDS
                )
            )
        passed = (
            actual == catalog["sha256"]
            and raw_actual == catalog["raw_response_sha256"]
            and actual == raw_actual
            and header_actual == catalog["response_headers_sha256"]
            and bounds_valid
            and catalog_semantics_valid
        )
        rows.append(
            {
                "source_id": source_id,
                "product": "official_dr10_catalog_box",
                "expected_layer": "ls-dr10-south catalog",
                "manifest_layer": "ls-dr10-south catalog",
                "layer_role_valid": 1,
                "request_parameters_valid": bool_int(bounds_valid),
                "request_parameter_error": "" if bounds_valid else "catalog_box_bounds_or_layer_invalid",
                "validated_path": str(catalog_path),
                "manifest_sha256": catalog["sha256"],
                "actual_sha256": actual,
                "file_hash_valid": bool_int(actual == catalog["sha256"]),
                "raw_response_path": str(raw_catalog),
                "raw_response_hash_valid": bool_int(raw_actual == actual),
                "response_headers_path": str(header_path),
                "response_headers_hash_valid": bool_int(header_actual == catalog["response_headers_sha256"]),
                "observed_matches_foundation_fov_hash": "",
                "catalog_row_identity_and_psf_revalidated": bool_int(catalog_semantics_valid),
                "input_integrity_pass": bool_int(passed),
            }
        )
    if not all(row["input_integrity_pass"] for row in rows):
        raise RuntimeError("input role/hash validation failed")
    return rows


def corrected_alignment(
    source: dict[str, str],
    products: dict[str, base.Product],
    integrity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row = base.alignment_row(source, products)
    requested = np.asarray([float(source["ra"]), float(source["dec"])])
    expected_semantics: list[bool] = []
    details: dict[str, Any] = {}
    for product, item in products.items():
        header = item.header
        band_order = "".join(text_value(header.get(f"BAND{i}", "")).lower() for i in range(3))
        center = np.asarray(item.wcs.all_world2pix(requested[0], requested[1], 0), dtype=float)
        scales = np.abs(
            np.asarray([value.to_value("deg") for value in item.wcs.proj_plane_pixel_scales()])
            * 3600.0
        )
        valid = (
            item.data.shape == EXPECTED_SHAPE
            and text_value(header.get("BANDS", "")).lower() == EXPECTED_BANDS
            and band_order == EXPECTED_BANDS
            and np.allclose(center, [127.5, 127.5], rtol=0, atol=0.05)
            and np.allclose(scales, [EXPECTED_SCALE, EXPECTED_SCALE], rtol=0, atol=1e-10)
            and text_value(header.get("CTYPE1", "")) == "RA---TAN"
            and text_value(header.get("CTYPE2", "")) == "DEC--TAN"
        )
        expected_semantics.append(valid)
        details[f"{product}_requested_center_offset_pixels"] = float(np.linalg.norm(center - [127.5, 127.5]))
        details[f"{product}_expected_semantics_valid"] = bool_int(valid)
    unit_headers = [text_value(products[name].header.get("BUNIT", "")).lower() for name in EXPECTED_LAYERS]
    explicit_units = [value for value in unit_headers if value]
    normalized = {value.replace("nanomaggy", "nanomaggies").replace("/pixel", " per coadd pixel") for value in explicit_units}
    units_resolved = (not explicit_units) or len(normalized) == 1
    integrity_pass = all(
        item["input_integrity_pass"]
        for item in integrity_rows
        if item["source_id"] == source["source_id"] and item["product"] in EXPECTED_LAYERS
    )
    corrected_pass = bool(row["alignment_pass"] and all(expected_semantics) and units_resolved and integrity_pass)
    row.update(
        **details,
        expected_shape="3x256x256",
        expected_band_order="grz",
        expected_pixel_scale_arcsec=EXPECTED_SCALE,
        requested_coordinate_center_valid=bool_int(all(expected_semantics)),
        normalized_unit_semantics_valid=bool_int(units_resolved),
        input_roles_and_hashes_valid=bool_int(integrity_pass),
        alignment_pass_v2=bool_int(corrected_pass),
        supersedes="tables/scene_triplet_alignment.csv",
    )
    if not corrected_pass:
        raise RuntimeError(f"corrected alignment failed for {source['source_id']}")
    return row


def corrected_closure(source: dict[str, str], products: dict[str, base.Product]) -> list[dict[str, Any]]:
    image = products["observed"].data
    model = products["model"].data
    residual = products["residual"].data
    rows: list[dict[str, Any]] = []
    for index, band in enumerate(base.BANDS):
        finite = np.isfinite(image[index]) & np.isfinite(model[index]) & np.isfinite(residual[index])
        closure = image[index] - model[index] - residual[index]
        values = closure[finite]
        obs = image[index][finite]
        resid = residual[index][finite]
        coverage = float(finite.mean())
        rmse = float(np.sqrt(np.mean(values**2))) if values.size else float("nan")
        noise = base.robust_scale(resid)
        rmse_ratio = rmse / noise if noise > 0 else float("inf")
        l1 = float(np.sum(np.abs(values)) / np.sum(np.abs(obs))) if np.sum(np.abs(obs)) > 0 else float("inf")
        p9999_ratio = float(np.percentile(np.abs(values), 99.99) / noise) if noise > 0 and values.size else float("inf")
        peak_obs = float(np.max(np.abs(obs))) if obs.size else 0.0
        max_abs = float(np.max(np.abs(values))) if values.size else float("nan")
        peak_relative = max_abs / peak_obs if peak_obs > 0 else float("inf")
        obs_total = float(np.sum(obs))
        valid = (
            bool(finite.all())
            and rmse_ratio <= CLOSURE_RMSE_NOISE_MAX
            and l1 <= CLOSURE_L1_MAX
            and p9999_ratio <= CLOSURE_P9999_NOISE_MAX
            and peak_relative <= CLOSURE_PEAK_RELATIVE_MAX
        )
        rows.append(
            {
                "source_id": source["source_id"],
                "catalog_row_index": source["catalog_row_index"],
                "band": band,
                "maximum_absolute_closure": max_abs,
                "mean_closure": float(np.mean(values)) if values.size else float("nan"),
                "closure_rmse": rmse,
                "relative_total_flux_closure": float(np.sum(values) / obs_total) if obs_total != 0 else float("nan"),
                "absolute_l1_relative_closure": l1,
                "residual_robust_noise": noise,
                "closure_rmse_to_residual_noise": rmse_ratio,
                "closure_p99_99_to_residual_noise": p9999_ratio,
                "maximum_closure_to_peak_observed": peak_relative,
                "finite_pixel_count": int(finite.sum()),
                "total_pixel_count": int(finite.size),
                "finite_pixel_coverage": coverage,
                "all_pixels_jointly_finite": bool_int(finite.all()),
                "closure_rmse_noise_max": CLOSURE_RMSE_NOISE_MAX,
                "closure_l1_max": CLOSURE_L1_MAX,
                "closure_p99_99_noise_max": CLOSURE_P9999_NOISE_MAX,
                "closure_peak_relative_max": CLOSURE_PEAK_RELATIVE_MAX,
                "closure_valid_v2": bool_int(valid),
                "supersedes": "tables/scene_triplet_closure.csv",
            }
        )
    return rows


def catalog_arrays(record: base.SourceRecord) -> tuple[dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    names = {name.lower(): name for name in record.catalog_data.dtype.names or ()}
    ra = np.asarray(record.catalog_data[names["ra"]], dtype=float)
    dec = np.asarray(record.catalog_data[names["dec"]], dtype=float)
    x, y = record.products["observed"].wcs.all_world2pix(ra, dec, 0)
    return names, np.asarray(x), np.asarray(y), (x >= -0.5) & (x <= 255.5) & (y >= -0.5) & (y <= 255.5)


def association(record: base.SourceRecord) -> tuple[bool, list[str], float, float]:
    reasons: list[str] = []
    if record.catalog_central_separation_arcsec > 1.0:
        reasons.append("nearest_dr10_catalog_component_gt_1arcsec")
    offsets: list[float] = []
    for label, detection in (("observed", record.observed_detection), ("model", record.model_detection)):
        if detection.central_index is None:
            reasons.append(f"no_{label}_central_detection")
            offsets.append(float("nan"))
            continue
        obj = detection.objects[detection.central_index]
        offset = float(math.hypot(float(obj["x"]) - record.catalog_target_x, float(obj["y"]) - record.catalog_target_y))
        offsets.append(offset)
        if offset > 6.0:
            reasons.append(f"{label}_segmentation_centroid_gt_6px")
    in_frame = -0.5 <= record.catalog_target_x <= 255.5 and -0.5 <= record.catalog_target_y <= 255.5
    if not in_frame:
        reasons.append("catalog_component_out_of_frame")
    names, xcat, ycat, catalog_in_frame = catalog_arrays(record)
    primary = np.asarray(record.catalog_data[names["brick_primary"]], dtype=bool)
    types = np.asarray([text_value(value) for value in record.catalog_data[names["type"]]])
    other = catalog_in_frame & primary & (types != "DUP")
    other[record.catalog_central_index] = False
    if np.any(other):
        nearest_other = float(
            np.min(
                np.hypot(
                    xcat[other] - record.catalog_target_x,
                    ycat[other] - record.catalog_target_y,
                )
            )
        )
        if nearest_other < 2.0 * float(np.max(record.psf_fwhm_pixels)):
            reasons.append("catalog_neighbor_within_2fwhm_target_attribution_ambiguous")
    return not reasons, reasons, offsets[0], offsets[1]


def independent_central_mask(record: base.SourceRecord) -> tuple[np.ndarray, float, str]:
    valid, reasons, _obs_offset, _model_offset = association(record)
    shape = record.products["observed"].data.shape[-2:]
    if not valid:
        return np.zeros(shape, dtype=bool), float("nan"), ";".join(reasons)
    names, xcat, ycat, in_frame = catalog_arrays(record)
    central = record.catalog_data[record.catalog_central_index]
    shape_r = float(central[names["shape_r"]])
    radius_arcsec = min(10.0, max(2.5 * float(np.max(record.psf_fwhm_arcsec)), 2.0 * max(shape_r, 0.0)))
    radius_pixels = radius_arcsec / base.PIXEL_SCALE
    yy, xx = np.indices(shape)
    center_distance = np.hypot(xx - record.catalog_target_x, yy - record.catalog_target_y)
    mask = center_distance <= radius_pixels
    other = np.where(in_frame & (np.arange(len(in_frame)) != record.catalog_central_index))[0]
    if len(other):
        nearest_other = np.full(shape, np.inf)
        for index in other:
            nearest_other = np.minimum(nearest_other, np.hypot(xx - xcat[index], yy - ycat[index]))
        mask &= center_distance <= nearest_other
    return mask, radius_pixels, ""


def translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a mask by integer pixels with no wraparound."""
    height, width = mask.shape
    output = np.zeros_like(mask, dtype=bool)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return output
    output[src_y0 + dy : src_y1 + dy, src_x0 + dx : src_x1 + dx] = mask[src_y0:src_y1, src_x0:src_x1]
    return output


def blank_apertures(
    record: base.SourceRecord,
    central_mask: np.ndarray,
    count: int = 6,
) -> list[tuple[float, float, np.ndarray]]:
    """Return exact translated central footprints that avoid dilated sources."""
    if not central_mask.any():
        return []
    shape = record.products["observed"].data.shape[-2:]
    names, xcat, ycat, in_frame = catalog_arrays(record)
    psf_clearance = max(2, int(math.ceil(2.0 * float(np.max(record.psf_fwhm_pixels)))))
    detected = (record.observed_detection.segmentation > 0) | (record.model_detection.segmentation > 0)
    exclusion = ndimage.binary_dilation(detected, iterations=psf_clearance)
    yy, xx = np.indices(shape)
    primary = np.asarray(record.catalog_data[names["brick_primary"]], dtype=bool)
    types = np.asarray([text_value(value) for value in record.catalog_data[names["type"]]])
    shape_r = np.asarray(record.catalog_data[names["shape_r"]], dtype=float)
    for index in np.where(in_frame & primary & (types != "DUP"))[0]:
        catalog_radius = max(
            float(psf_clearance),
            2.0 * max(float(shape_r[index]), 0.0) / base.PIXEL_SCALE,
        )
        exclusion |= np.hypot(xx - xcat[index], yy - ycat[index]) <= catalog_radius
    clearance = ndimage.distance_transform_edt(~exclusion)
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-216, 217, 8):
        for dx in range(-216, 217, 8):
            if dx == 0 and dy == 0:
                continue
            candidate = translate_mask(central_mask, dx, dy)
            if int(candidate.sum()) != int(central_mask.sum()) or np.any(candidate & exclusion):
                continue
            score = float(np.min(clearance[candidate]))
            candidates.append((score, dx, dy))
    candidates.sort(reverse=True)
    selected: list[tuple[float, float, np.ndarray]] = []
    selected_exclusion = np.zeros(shape, dtype=bool)
    for _score, dx, dy in candidates:
        candidate = translate_mask(central_mask, dx, dy)
        if np.any(candidate & selected_exclusion):
            continue
        selected.append((record.catalog_target_x + dx, record.catalog_target_y + dy, candidate))
        selected_exclusion |= ndimage.binary_dilation(candidate, iterations=psf_clearance)
        if len(selected) >= count:
            break
    return selected


def vector_gradient_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    gx_a = ndimage.sobel(a, axis=-1, mode="nearest")
    gy_a = ndimage.sobel(a, axis=-2, mode="nearest")
    gx_b = ndimage.sobel(b, axis=-1, mode="nearest")
    gy_b = ndimage.sobel(b, axis=-2, mode="nearest")
    denominator = float(
        np.sqrt(
            np.sum(gx_a[mask] ** 2 + gy_a[mask] ** 2)
            * np.sum(gx_b[mask] ** 2 + gy_b[mask] ** 2)
        )
    )
    return float(np.sum(gx_a[mask] * gx_b[mask] + gy_a[mask] * gy_b[mask]) / denominator) if denominator > 0 else float("nan")


def asymmetry(array: np.ndarray, mask: np.ndarray, center_x: float, center_y: float) -> float:
    rotated = base.rotate_180_about(array, center_x, center_y)
    rotated_mask = base.rotate_180_about(mask.astype(float), center_x, center_y) >= 0.5
    valid = mask & rotated_mask & np.isfinite(array) & np.isfinite(rotated)
    denominator = float(np.sum(np.abs(array[valid]) + np.abs(rotated[valid])))
    return float(np.sum(np.abs(array[valid] - rotated[valid])) / denominator) if denominator > 0 else float("nan")


def corrected_morphology(
    record: base.SourceRecord,
    manual: dict[str, str],
) -> tuple[list[dict[str, Any]], np.ndarray, list[tuple[float, float, np.ndarray]], float]:
    valid, reasons, obs_offset, model_offset = association(record)
    mask, radius, mask_error = independent_central_mask(record)
    blanks = blank_apertures(record, mask)
    rows: list[dict[str, Any]] = []
    if not valid or not mask.any() or len(blanks) < 2:
        status = ";".join(reasons + ([mask_error] if mask_error else []) + (["insufficient_blank_apertures"] if len(blanks) < 2 else []))
        for band in base.BANDS:
            rows.append(
                {
                    "source_id": record.source["source_id"],
                    "catalog_row_index": record.source["catalog_row_index"],
                    "band": band,
                    "association_valid": 0,
                    "association_reasons": status,
                    "observed_segmentation_centroid_offset_px": obs_offset,
                    "model_segmentation_centroid_offset_px": model_offset,
                    "central_mask_available": 0,
                    "manual_classification": manual["classification"],
                    "metric_status": "unavailable_invalid_central_association",
                    "supersedes": "tables/residual_morphology_metrics.csv",
                }
            )
        return rows, mask, blanks, radius

    yy, xx = np.indices(mask.shape)
    radial = np.hypot(xx - record.catalog_target_x, yy - record.catalog_target_y)
    core = mask & (radial <= 0.5 * radius)
    halo = mask & (radial > 0.5 * radius)
    arrays = {
        "observed": record.products["observed"].data,
        "model": record.products["model"].data,
        "residual": record.products["residual"].data,
    }
    for band_index, band in enumerate(base.BANDS):
        blank_union = np.logical_or.reduce([item[2] for item in blanks])
        backgrounds = {key: float(np.median(value[band_index][blank_union])) for key, value in arrays.items()}
        observed = arrays["observed"][band_index] - backgrounds["observed"]
        model = arrays["model"][band_index] - backgrounds["model"]
        residual = arrays["residual"][band_index] - backgrounds["residual"]
        blank_fluxes = np.asarray([float(np.sum(residual[item[2]])) for item in blanks])
        blank_correlations = np.asarray([base.pearson(residual[item[2]], observed[item[2]]) for item in blanks])
        blank_gradient = np.asarray([vector_gradient_correlation(residual, observed, item[2]) for item in blanks])
        blank_asymmetry = np.asarray([asymmetry(residual, item[2], item[0], item[1]) for item in blanks])
        blank_power = np.asarray([float(np.mean(residual[item[2]] ** 2)) for item in blanks])
        blank_core_power: list[float] = []
        blank_halo_power: list[float] = []
        for blank_x, blank_y, _blank_mask in blanks:
            dx = int(round(blank_x - record.catalog_target_x))
            dy = int(round(blank_y - record.catalog_target_y))
            translated_core = translate_mask(core, dx, dy)
            translated_halo = translate_mask(halo, dx, dy)
            blank_core_power.append(float(np.mean(residual[translated_core] ** 2)))
            blank_halo_power.append(float(np.mean(residual[translated_halo] ** 2)))
        blank_flux_scale = base.robust_scale(blank_fluxes)
        central_flux = float(np.sum(residual[mask]))
        observed_flux = float(np.sum(observed[mask]))
        central_corr = base.pearson(residual[mask], observed[mask])
        central_gradient = vector_gradient_correlation(residual, observed, mask)
        central_asym = asymmetry(residual, mask, record.catalog_target_x, record.catalog_target_y)
        central_values = residual[mask]
        absolute = float(np.sum(np.abs(central_values)))
        core_power = float(np.mean(residual[core] ** 2)) if core.any() else float("nan")
        halo_power = float(np.mean(residual[halo] ** 2)) if halo.any() else float("nan")
        rows.append(
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "band": band,
                "association_valid": 1,
                "association_reasons": "",
                "observed_segmentation_centroid_offset_px": obs_offset,
                "model_segmentation_centroid_offset_px": model_offset,
                "central_mask_available": 1,
                "central_mask_definition": "catalog-centered circular aperture clipped by nearest-catalog-component Voronoi boundary; attribution rejected for catalog neighbors within 2 PSF FWHM",
                "central_mask_radius_pixels": radius,
                "central_mask_radius_arcsec": radius * base.PIXEL_SCALE,
                "central_mask_pixel_count": int(mask.sum()),
                "central_mask_sha256": array_hash(mask.astype(np.uint8)),
                "blank_aperture_count": len(blanks),
                "background_convention": "per-product median over exact translated central footprints with zero overlap against dilated observed/model detections and catalog-source exclusion regions; raw arrays retained",
                "observed_central_flux": observed_flux,
                "model_central_aperture_flux": float(np.sum(model[mask])),
                "residual_central_flux": central_flux,
                "residual_to_observed_flux_fraction": central_flux / observed_flux if observed_flux != 0 else float("nan"),
                "blank_residual_flux_median": float(np.median(blank_fluxes)),
                "blank_residual_flux_robust_scale": blank_flux_scale,
                "central_residual_flux_null_zscore": (central_flux - float(np.median(blank_fluxes))) / blank_flux_scale if blank_flux_scale > 0 else float("nan"),
                "residual_observed_spatial_correlation": central_corr,
                "blank_spatial_correlation_median": float(np.nanmedian(blank_correlations)),
                "central_minus_blank_spatial_correlation": central_corr - float(np.nanmedian(blank_correlations)),
                "residual_observed_gradient_vector_correlation": central_gradient,
                "blank_gradient_vector_correlation_median": float(np.nanmedian(blank_gradient)),
                "central_minus_blank_gradient_correlation": central_gradient - float(np.nanmedian(blank_gradient)),
                "positive_central_residual_pixel_fraction": float(np.mean(central_values > 0)),
                "negative_central_residual_pixel_fraction": float(np.mean(central_values < 0)),
                "positive_significant_residual_pixel_fraction_3sigma": float(np.mean(central_values > 3 * record.noises[band_index])),
                "negative_significant_residual_pixel_fraction_3sigma": float(np.mean(central_values < -3 * record.noises[band_index])),
                "positive_central_absolute_flux_fraction": float(np.sum(np.clip(central_values, 0, None)) / absolute) if absolute > 0 else float("nan"),
                "negative_central_absolute_flux_fraction": float(np.sum(np.clip(-central_values, 0, None)) / absolute) if absolute > 0 else float("nan"),
                "residual_asymmetry_180": central_asym,
                "blank_asymmetry_median": float(np.nanmedian(blank_asymmetry)),
                "blank_corrected_residual_asymmetry": central_asym - float(np.nanmedian(blank_asymmetry)),
                "residual_core_power_per_pixel": core_power,
                "residual_halo_power_per_pixel": halo_power,
                "blank_residual_power_per_pixel_median": float(np.median(blank_power)),
                "blank_core_power_per_pixel_median": float(np.median(blank_core_power)),
                "blank_halo_power_per_pixel_median": float(np.median(blank_halo_power)),
                "residual_core_excess_power_per_pixel": core_power - float(np.median(blank_core_power)),
                "residual_halo_excess_power_per_pixel": halo_power - float(np.median(blank_halo_power)),
                "correlated_noise_caveat": "blank-aperture empirical correction used; no independence or white-noise assumption",
                "manual_classification": manual["classification"],
                "metric_status": "measured_with_empirical_blank_null",
                "supersedes": "tables/residual_morphology_metrics.csv",
            }
        )
    return rows, mask, blanks, radius


def corrected_radial_profiles(
    record: base.SourceRecord,
    mask: np.ndarray,
    blanks: list[tuple[float, float, np.ndarray]],
) -> list[dict[str, Any]]:
    """Build association-aware, background-subtracted one-pixel annular profiles."""
    valid, reasons, _obs_offset, _model_offset = association(record)
    arrays = {
        "observed": record.products["observed"].data,
        "model": record.products["model"].data,
        "residual": record.products["residual"].data,
    }
    rows: list[dict[str, Any]] = []
    if not valid or not mask.any() or len(blanks) < 2:
        status = ";".join(reasons + (["insufficient_exact_blank_apertures"] if len(blanks) < 2 else []))
        for product in arrays:
            for band in base.BANDS:
                rows.append(
                    {
                        "source_id": record.source["source_id"],
                        "catalog_row_index": record.source["catalog_row_index"],
                        "product": product,
                        "band": band,
                        "association_valid": 0,
                        "association_reasons": status,
                        "profile_status": "unavailable_invalid_central_association",
                        "supersedes": "tables/radial_profiles.csv",
                    }
                )
        return rows

    blank_union = np.logical_or.reduce([item[2] for item in blanks])
    yy, xx = np.indices(mask.shape)
    radial = np.hypot(xx - record.catalog_target_x, yy - record.catalog_target_y)
    maximum_bin = int(math.ceil(float(np.max(radial[mask]))))
    mask_hash = array_hash(mask.astype(np.uint8))
    for product, cube in arrays.items():
        for band_index, band in enumerate(base.BANDS):
            background = float(np.median(cube[band_index][blank_union]))
            blank_rms = base.robust_scale(cube[band_index][blank_union] - background)
            plane = cube[band_index] - background
            for radius_inner in range(maximum_bin):
                annulus = mask & (radial >= radius_inner) & (radial < radius_inner + 1)
                values = plane[annulus]
                rows.append(
                    {
                        "source_id": record.source["source_id"],
                        "catalog_row_index": record.source["catalog_row_index"],
                        "product": product,
                        "band": band,
                        "association_valid": 1,
                        "association_reasons": "",
                        "central_mask_sha256": mask_hash,
                        "radius_inner_pixels": radius_inner,
                        "radius_outer_pixels": radius_inner + 1,
                        "radius_midpoint_arcsec": (radius_inner + 0.5) * base.PIXEL_SCALE,
                        "annulus_pixel_count": int(values.size),
                        "annulus_mean_nanomaggies_per_pixel": float(np.mean(values)) if values.size else float("nan"),
                        "annulus_median_nanomaggies_per_pixel": float(np.median(values)) if values.size else float("nan"),
                        "annulus_sum_nanomaggies": float(np.sum(values)) if values.size else float("nan"),
                        "blank_background_median_nanomaggies_per_pixel": background,
                        "blank_robust_scale_nanomaggies_per_pixel": blank_rms,
                        "profile_status": "measured_exact_mask_and_source_free_blank_background" if values.size else "empty_annulus",
                        "profile_convention": "catalog-centered one-pixel annuli intersected with the independently defined central mask",
                        "supersedes": "tables/radial_profiles.csv",
                    }
                )
    return rows


def corrected_components(
    record: base.SourceRecord,
    closure_valid: bool,
) -> dict[str, Any]:
    names, xcat, ycat, in_frame = catalog_arrays(record)
    primary = np.asarray(record.catalog_data[names["brick_primary"]], dtype=bool)
    types = np.asarray([text_value(value) for value in record.catalog_data[names["type"]]])
    ref_cat = np.asarray([text_value(value) for value in record.catalog_data[names["ref_cat"]]])
    usable = in_frame & primary & (types != "DUP")
    central = record.catalog_data[record.catalog_central_index]
    central_fluxes = np.asarray([float(central[names[f"flux_{band}"]]) for band in base.BANDS])
    neighbor_mask = usable.copy()
    neighbor_mask[record.catalog_central_index] = False
    ratios: dict[str, float] = {}
    for band_index, band in enumerate(base.BANDS):
        neighbor_flux = np.asarray(record.catalog_data[names[f"flux_{band}"]][neighbor_mask], dtype=float)
        ratios[band] = float(np.sum(np.abs(neighbor_flux)) / abs(central_fluxes[band_index])) if central_fluxes[band_index] != 0 else float("inf")
    gaia_g = np.asarray(record.catalog_data[names["gaia_phot_g_mean_mag"]], dtype=float)
    bright = usable & np.isin(ref_cat, ["GE", "T2"]) & np.isfinite(gaia_g) & (gaia_g < 13)
    medium = usable & np.isin(ref_cat, ["GE", "T2"]) & np.isfinite(gaia_g) & (gaia_g < 16)
    large = usable & (ref_cat == "L3")
    valid, reasons, obs_offset, model_offset = association(record)
    raw_model = record.products["model"].data
    outer = base.outer_mask(raw_model.shape[-2:])
    pedestal = [float(np.median(raw_model[i][outer])) for i in range(3)]
    pedestal_ratio = [abs(pedestal[i]) / record.noises[i] for i in range(3)]
    observed_metrics = base.detection_component_metrics(record.observed_detection)
    model_metrics = base.detection_component_metrics(record.model_detection)
    return {
        "source_id": record.source["source_id"],
        "catalog_row_index": record.source["catalog_row_index"],
        "observed_component_count": observed_metrics["component_count"],
        "modeled_component_count": model_metrics["component_count"],
        "official_catalog_primary_components_in_frame": int(usable.sum()),
        "central_component_identity": f"release={int(central[names['release']])};brickid={int(central[names['brickid']])};objid={int(central[names['objid']])}",
        "central_component_type": text_value(central[names["type"]]),
        "central_component_ref_cat": text_value(central[names["ref_cat"]]),
        "central_component_catalog_separation_arcsec": record.catalog_central_separation_arcsec,
        "central_association_valid": bool_int(valid),
        "central_association_reasons": ";".join(reasons),
        "observed_central_centroid_offset_px": obs_offset,
        "model_central_centroid_offset_px": model_offset,
        "modeled_neighbor_count_catalog_primary": int(neighbor_mask.sum()),
        "neighbor_to_target_flux_ratio_g": ratios["g"],
        "neighbor_to_target_flux_ratio_r": ratios["r"],
        "neighbor_to_target_flux_ratio_z": ratios["z"],
        "neighbor_flux_ratio_source": "official DR10 catalog per-band nanomaggy fluxes; absolute neighbor sums",
        "catalog_bright_star_component_count_g_lt_13": int(bright.sum()),
        "catalog_medium_bright_star_component_count_g_lt_16": int(medium.sum()),
        "catalog_large_galaxy_component_count_ref_cat_l3": int(large.sum()),
        "raw_model_outer_median_g": pedestal[0],
        "raw_model_outer_median_r": pedestal[1],
        "raw_model_outer_median_z": pedestal[2],
        "raw_model_outer_pedestal_max_noise_ratio": float(max(pedestal_ratio)),
        "background_component_status": "no explicit background row type in official source catalog; raw model pedestal measured independently",
        "scene_triplet_closure_valid": bool_int(closure_valid),
        "pixel_segmentation_candidate_association_valid": bool_int(valid),
        "central_only_model_isolation_reliable": 0,
        "central_only_model_isolation_reasons": "summed_scene_model_has_no_per_source_component_planes;overlapping_tractor_profiles_not_quantitatively_bounded",
        "whole_scene_model_prohibited_as_contaminant": 1,
        "official_catalog_source": OFFICIAL_CATALOGS,
        "supersedes": "tables/scene_component_audit.csv",
    }


def apodized_weight(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape)
    distance = ndimage.distance_transform_edt(mask)
    return np.clip(distance / 2.0, 0, 1)


def edge_gradient_ratio(cube: np.ndarray, support: np.ndarray) -> float:
    if not support.any():
        return float("nan")
    boundary = support ^ ndimage.binary_erosion(support)
    inner = ndimage.binary_erosion(support, iterations=2)
    values: list[float] = []
    for plane in cube:
        gradient = np.hypot(
            ndimage.sobel(plane, axis=-1, mode="nearest"),
            ndimage.sobel(plane, axis=-2, mode="nearest"),
        )
        b = float(np.mean(gradient[boundary])) if boundary.any() else float("nan")
        i = float(np.mean(gradient[inner])) if inner.any() else float("nan")
        values.append(b / i if np.isfinite(i) and i > 0 else float("nan"))
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def color(flux_a: float, flux_b: float) -> float:
    return -2.5 * math.log10(flux_a / flux_b) if flux_a > 0 and flux_b > 0 else float("nan")


def corrected_extraction(
    record: base.SourceRecord,
    central_mask: np.ndarray,
    blanks: list[tuple[float, float, np.ndarray]],
    manual: dict[str, str],
    component_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    shape = central_mask.shape
    blank_union = np.logical_or.reduce([item[2] for item in blanks]) if blanks else base.outer_mask(shape)
    raw_obs = record.products["observed"].data
    raw_model = record.products["model"].data
    raw_resid = record.products["residual"].data
    backgrounds = {
        "observed": np.asarray([np.median(raw_obs[i][blank_union]) for i in range(3)]),
        "model": np.asarray([np.median(raw_model[i][blank_union]) for i in range(3)]),
        "residual": np.asarray([np.median(raw_resid[i][blank_union]) for i in range(3)]),
    }
    observed_sub = raw_obs - backgrounds["observed"][:, None, None]
    model_mask = base.central_mask(record.model_detection)
    if component_row["central_association_valid"]:
        model_support = ndimage.binary_dilation(model_mask, iterations=max(2, int(math.ceil(np.max(record.psf_fwhm_pixels)))))
    else:
        model_support = np.zeros(shape, dtype=bool)
    support_a = central_mask
    support_b = model_support
    support_c = central_mask | model_support
    weight_a = apodized_weight(support_a)
    weight_b = apodized_weight(support_b)
    weight_c = apodized_weight(support_c)
    a = observed_sub * weight_a
    b = (raw_model - backgrounds["model"][:, None, None]) * weight_b
    c = (
        (raw_resid - backgrounds["residual"][:, None, None])
        + (raw_model - backgrounds["model"][:, None, None]) * weight_b
    ) * weight_c
    arrays = {"A": a, "B": b, "C": c}
    supports = {"A": support_a, "B": support_b, "C": support_c}
    reference = np.sum(observed_sub[:, central_mask], axis=1) if central_mask.any() else np.full(3, np.nan)
    ref_gr = color(float(reference[0]), float(reference[1]))
    ref_rz = color(float(reference[1]), float(reference[2]))
    outer = np.zeros(shape, dtype=bool)
    outer[:8] = outer[-8:] = True
    outer[:, :8] = outer[:, -8:] = True
    rows: list[dict[str, Any]] = []
    definitions = {
        "A": "catalog-apertured observed proxy; contains coadd noise and residual background",
        "B": "segmented summed-scene-model proxy; not an official per-source Tractor render",
        "C": "background-subtracted residual plus background-subtracted B proxy before outer support application",
    }
    for option in ("A", "B", "C"):
        cube = arrays[option]
        support = supports[option]
        flux = np.sum(cube, axis=(1, 2))
        gr = color(float(flux[0]), float(flux[1]))
        rz = color(float(flux[1]), float(flux[2]))
        correlations = [base.pearson(cube[i][central_mask], observed_sub[i][central_mask]) if central_mask.any() else float("nan") for i in range(3)]
        finite_corr = np.asarray(correlations)[np.isfinite(correlations)]
        absolute = float(np.sum(np.abs(cube)))
        x_y = np.column_stack(np.nonzero(support)) if support.any() else np.empty((0, 2))
        margin = (
            float(min(np.min(x_y[:, 0]), np.min(x_y[:, 1]), 255 - np.max(x_y[:, 0]), 255 - np.max(x_y[:, 1])))
            if len(x_y)
            else float("nan")
        )
        rows.append(
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "option": option,
                "array_semantics": definitions[option],
                "support_pixel_count": int(support.sum()),
                "support_sha256": array_hash(support.astype(np.uint8)),
                "flux_g": float(flux[0]),
                "flux_r": float(flux[1]),
                "flux_z": float(flux[2]),
                "flux_preservation_g_vs_catalog_aperture_observed": float(flux[0] / reference[0]) if reference[0] != 0 else float("nan"),
                "flux_preservation_r_vs_catalog_aperture_observed": float(flux[1] / reference[1]) if reference[1] != 0 else float("nan"),
                "flux_preservation_z_vs_catalog_aperture_observed": float(flux[2] / reference[2]) if reference[2] != 0 else float("nan"),
                "color_g_minus_r": gr,
                "color_r_minus_z": rz,
                "color_error_g_minus_r_vs_observed_aperture": gr - ref_gr if np.isfinite(gr) and np.isfinite(ref_gr) else float("nan"),
                "color_error_r_minus_z_vs_observed_aperture": rz - ref_rz if np.isfinite(rz) and np.isfinite(ref_rz) else float("nan"),
                "morphology_correlation_with_observed_aperture": float(np.mean(finite_corr)) if finite_corr.size else float("nan"),
                "metric_dependency_warning": "A is derived from observed reference; C algebraically shares residual; metrics are descriptive not independent validation",
                "neighbor_to_target_flux_ratio_g_catalog": component_row["neighbor_to_target_flux_ratio_g"],
                "neighbor_to_target_flux_ratio_r_catalog": component_row["neighbor_to_target_flux_ratio_r"],
                "neighbor_to_target_flux_ratio_z_catalog": component_row["neighbor_to_target_flux_ratio_z"],
                "edge_artifact_gradient_ratio_spatial_axes": edge_gradient_ratio(cube, support),
                "rectangular_cutout_border_leakage_fraction": float(np.sum(np.abs(cube[:, outer])) / absolute) if absolute > 0 else float("nan"),
                "support_min_frame_margin_pixels": margin,
                "contains_coadd_noise_realization": bool_int(option in {"A", "C"}),
                "would_add_second_noise_if_used_as_contaminant": bool_int(option in {"A", "C"}),
                "manual_morphology_classification": manual["classification"],
                "subpixel_shift_experiment_performed": 0,
                "psf_shape_matching_validated": 0,
                "central_only_official_source_render_available": 0,
                "suitable_as_contaminant_for_single_noise_contract": 0,
                "suitable_as_target_array": 0,
                "target_role_note": "recommended target is the unchanged full observed scene not any segmented extraction option",
                "decision_reason": (
                    "contains_second_coadd_noise_realization"
                    if option in {"A", "C"}
                    else "summed_scene_model_not_per_source_render;psf_and_subpixel_shift_unvalidated"
                ),
                "supersedes": "tables/source_extraction_options.csv",
            }
        )
    return rows, arrays, supports


def corrected_psf(record: base.SourceRecord) -> list[dict[str, Any]]:
    cross_band = float(np.max(record.psf_fwhm_arcsec) - np.min(record.psf_fwhm_arcsec))
    rows: list[dict[str, Any]] = []
    for index, band in enumerate(base.BANDS):
        rows.append(
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "band": band,
                "psf_fwhm_arcsec": float(record.psf_fwhm_arcsec[index]),
                "psf_fwhm_pixels": float(record.psf_fwhm_pixels[index]),
                "within_source_cross_band_range_arcsec": cross_band,
                "catalog_match_separation_arcsec": record.catalog_central_separation_arcsec,
                "value_source": "official ls-dr10-south catalog PSFSIZE column",
                "corresponding_local_map_template": "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/<AAA>/<brick>/legacysurvey-<brick>-psfsize-<band>.fits.fz",
                "direct_source_addition_psf_consistent": 0,
                "pairing_by_scalar_fwhm_feasibility": "exploratory screening only; full PSF shape and interpolation response unvalidated",
                "convolution_to_common_broader_psf_feasibility": "feasible in principle after obtaining validated full PSF kernels; not established by scalar FWHM",
                "forward_rendering_feasibility": "preferred in principle from intrinsic Tractor parameters at target PSF; unavailable from an already convolved model cutout",
                "deconvolution_implemented": 0,
                "supersedes": "tables/psf_audit.csv",
            }
        )
    return rows


def corrected_scene_figure(
    record: base.SourceRecord,
    mask: np.ndarray,
    blanks: list[tuple[float, float, np.ndarray]],
    manual: dict[str, str],
    closure_valid: bool,
    path: Path,
) -> None:
    figure, axes = plt.subplots(4, 5, figsize=(19, 14), constrained_layout=True)
    cubes = [record.products["observed"].data, record.products["model"].data, record.products["residual"].data]
    titles = ["Observed", "Tractor scene model", "Residual"]
    for row, (cube, title) in enumerate(zip(cubes, titles, strict=True)):
        for index, band in enumerate(base.BANDS):
            axes[row, index].imshow(base.display_plane(cube[index]), origin="lower", cmap="gray", vmin=0, vmax=1)
            axes[row, index].set_title(f"{title} {band}")
        axes[row, 3].imshow(base.display_rgb(cube), origin="lower")
        axes[row, 3].set_title(f"{title} RGB (display only)")
        axes[row, 4].imshow(base.display_rgb(cube), origin="lower")
        if mask.any():
            axes[row, 4].contour(mask.astype(float), [0.5], colors=["cyan"], linewidths=0.8)
        axes[row, 4].plot(record.catalog_target_x, record.catalog_target_y, "+", color="yellow", ms=10)
        requested_x, requested_y = record.products["observed"].wcs.all_world2pix(float(record.source["ra"]), float(record.source["dec"]), 0)
        axes[row, 4].plot(requested_x, requested_y, "x", color="magenta", ms=8)
        axes[row, 4].set_title("independent aperture; catalog +; requested ×")
    for index, band in enumerate(base.BANDS):
        axes[3, index].imshow(
            base.signed_residual(record.products["residual"].data[index], record.noises[index]),
            origin="lower",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        if mask.any():
            axes[3, index].contour(mask.astype(float), [0.5], colors=["black"], linewidths=0.6)
        axes[3, index].set_title(f"Robust signed residual {band}; aperture")
    axes[2, 4].clear()
    axes[2, 4].imshow(base.display_rgb(record.products["residual"].data), origin="lower")
    for x, y, blank in blanks:
        axes[2, 4].contour(blank.astype(float), [0.5], colors=["white"], linewidths=0.5, alpha=0.8)
    axes[2, 4].set_title("Residual RGB; exact source-free blank footprints")
    valid, reasons, obs_offset, model_offset = association(record)
    radial_axis = axes[3, 3]
    if valid and mask.any() and len(blanks) >= 2:
        yy, xx = np.indices(mask.shape)
        radial = np.hypot(xx - record.catalog_target_x, yy - record.catalog_target_y)
        maximum_bin = int(math.ceil(float(np.max(radial[mask]))))
        blank_union = np.logical_or.reduce([item[2] for item in blanks])
        product_styles = {
            "observed": ("black", "-"),
            "model": ("tab:orange", "--"),
            "residual": ("tab:purple", ":"),
        }
        band_alpha = {"g": 0.45, "r": 0.75, "z": 1.0}
        for product, cube in (
            ("observed", record.products["observed"].data),
            ("model", record.products["model"].data),
            ("residual", record.products["residual"].data),
        ):
            color, linestyle = product_styles[product]
            for band_index, band in enumerate(base.BANDS):
                background = float(np.median(cube[band_index][blank_union]))
                plane = cube[band_index] - background
                profile = []
                radii = []
                for radius_inner in range(maximum_bin):
                    annulus = mask & (radial >= radius_inner) & (radial < radius_inner + 1)
                    if annulus.any():
                        radii.append((radius_inner + 0.5) * base.PIXEL_SCALE)
                        profile.append(float(np.mean(plane[annulus])))
                radial_axis.plot(
                    radii,
                    profile,
                    color=color,
                    linestyle=linestyle,
                    alpha=band_alpha[band],
                    linewidth=1.0,
                    label=f"{product} {band}",
                )
        radial_axis.axhline(0, color="0.6", linewidth=0.6)
        radial_axis.set_yscale("symlog", linthresh=max(1e-5, float(np.median(record.noises))))
        radial_axis.set_xlabel("radius (arcsec)")
        radial_axis.set_ylabel("mean nanomaggies/pixel")
        radial_axis.legend(fontsize=5, ncol=3, loc="best")
        radial_axis.set_title("Association-aware radial profiles")
    else:
        radial_axis.axis("off")
        radial_axis.text(
            0,
            1,
            "Radial profile unavailable:\n" + (";".join(reasons) or "insufficient exact blanks"),
            va="top",
            wrap=True,
        )
    axes[3, 4].axis("off")
    axes[3, 4].text(
        0,
        1,
        "\n".join(
            [
                f"source: {record.source['source_id']}",
                f"manual class: {manual['classification']}",
                f"association valid: {valid}",
                f"association reasons: {';'.join(reasons) or 'none'}",
                f"observed/model centroid offsets: {obs_offset:.2f} / {model_offset:.2f} px",
                f"closure valid all bands: {closure_valid}",
                "Model is evaluated as a prediction not truth.",
                "All stretches/RGB are display-only.",
            ]
        ),
        va="top",
        fontsize=9,
        wrap=True,
    )
    for axis in axes.ravel():
        if axis is radial_axis and valid and mask.any() and len(blanks) >= 2:
            continue
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Corrected DR10 triplet audit — {record.source['source_id']}", fontsize=16)
    save_figure(path, figure)


def corrected_extraction_figure(
    record: base.SourceRecord,
    arrays: dict[str, np.ndarray],
    supports: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for column, option in enumerate(("A", "B", "C")):
        axes[0, column].imshow(base.display_rgb(arrays[option]), origin="lower")
        axes[0, column].set_title(f"{option} diagnostic proxy")
        axes[1, column].imshow(base.display_rgb(arrays[option]), origin="lower")
        if supports[option].any():
            axes[1, column].contour(supports[option].astype(float), [0.5], colors=["cyan"], linewidths=0.8)
        axes[1, column].set_title(f"{option} actual support")
    axes[0, 3].imshow(base.display_rgb(record.products["observed"].data), origin="lower")
    axes[0, 3].set_title("Unchanged observed target candidate")
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0,
        1,
        "A/C contain coadd noise.\nB is a summed-scene segmentation proxy.\n"
        "No option passes as a contaminant.\nSubpixel shifting and full-PSF matching are unvalidated.",
        va="top",
        fontsize=11,
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Corrected extraction comparison — {record.source['source_id']}", fontsize=15)
    save_figure(path, figure)


def checkpoint_integrity(foundation_run: Path) -> list[dict[str, Any]]:
    rows = read_csv(foundation_run / "tables" / "checkpoint_inventory_after.csv")
    output: list[dict[str, Any]] = []
    for row in rows:
        lowered = row["path"].lower()
        if "lockbox" in lowered or "sealed" in lowered:
            raise RuntimeError("refusing sealed/lockbox checkpoint path")
        path = Path(row["path"])
        actual_hash = sha256_file(path)
        stat = path.stat()
        valid = (
            actual_hash == row["sha256"]
            and stat.st_size == int(row["size_bytes"])
            and stat.st_mtime_ns == int(row["mtime_ns"])
        )
        output.append(
            {
                "path": row["path"],
                "foundation_sha256": row["sha256"],
                "current_sha256": actual_hash,
                "foundation_size_bytes": row["size_bytes"],
                "current_size_bytes": stat.st_size,
                "foundation_mtime_ns": row["mtime_ns"],
                "current_mtime_ns": stat.st_mtime_ns,
                "identity_unchanged": bool_int(valid),
            }
        )
    if not all(row["identity_unchanged"] for row in output):
        raise RuntimeError("checkpoint integrity failure")
    return output


def corrected_manual_review_rows(
    manual_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in manual_rows:
        corrected = dict(row)
        corrected["visual_evidence_path"] = (
            f"figures/scene_probe_v2/scene_triplet_v2_{int(row['engineering_rank']):02d}_{row['source_id']}.png"
        )
        corrected["scalar_evidence_summary"] = (
            "classification is manual and is not assigned from scalar thresholds; "
            "consult residual_morphology_metrics_v2.csv where association is valid"
        )
        corrected["prior_v1_scalar_evidence_superseded"] = 1
        corrected["v2_review_basis"] = (
            "full individual observed/model/residual visual review; v2 metrics are supporting evidence only"
        )
        corrected["automatic_classification"] = 0
        output.append(corrected)
    return output


def output_inventory_rows(
    run_dir: Path,
    foundation_run: Path,
    generated_paths: list[Path],
) -> list[dict[str, Any]]:
    dependency_paths = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        run_dir / "manifests" / "engineering_sources_20.csv",
        run_dir / "manifests" / "scene_triplet_download_manifest.csv",
        run_dir / "manifests" / "official_catalog_download_manifest.csv",
        run_dir / "tables" / "manual_morphology_review.csv",
        foundation_run / "tables" / "checkpoint_inventory_after.csv",
    ]
    rows: list[dict[str, Any]] = []
    timestamp = utc_now()
    seen: set[Path] = set()
    for scope, paths in (("input_dependency", dependency_paths), ("generated_v2_output", generated_paths)):
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            stat = resolved.stat()
            rows.append(
                {
                    "inventory_generated_utc": timestamp,
                    "scope": scope,
                    "role": (
                        "analysis_code_or_manifest"
                        if scope == "input_dependency"
                        else ("diagnostic_figure" if resolved.suffix.lower() == ".png" else "tabular_report_or_log")
                    ),
                    "path": str(resolved),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(resolved),
                    "inventory_self_excluded": 1,
                    "replay_note": "output_inventory_v2.csv is excluded because a file cannot contain its own stable hash",
                }
            )
    return rows


def reports(
    run_dir: Path,
    alignment_rows: list[dict[str, Any]],
    closure_rows: list[dict[str, Any]],
    morph_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    psf_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, str]],
    figure_dir: Path,
) -> dict[str, str]:
    closure_pass = sum(
        all(row["closure_valid_v2"] for row in closure_rows if row["source_id"] == source_id)
        for source_id in {row["source_id"] for row in closure_rows}
    )
    associations = sum(row["central_association_valid"] for row in component_rows)
    measured_bands = sum(row["metric_status"] == "measured_with_empirical_blank_null" for row in morph_rows)
    classes: dict[str, int] = {}
    for row in manual_rows:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
    by_band = {
        band: np.asarray([float(row["psf_fwhm_arcsec"]) for row in psf_rows if row["band"] == band])
        for band in base.BANDS
    }
    audit = f"""# DR10 scene-triplet alignment and additivity audit — authoritative v2

Generated: `{utc_now()}`

This append-only correction supersedes the initial unsuffixed numerical tables;
the original files remain preserved. Every validated FITS/catalog file, raw
response, response-header record, request role, URL parameter, and SHA-256 was
rechecked before analysis.

- Alignment: **{sum(row['alignment_pass_v2'] for row in alignment_rows)}/20**.
  Exact shape `(3,256,256)`, `grz`, requested coordinate at the center,
  0.262 arcsec/pixel, CRPIX/CRVAL, canonical CD/PC matrix, full-grid celestial
  WCS, and common documented units all pass.
- Additivity: **{closure_pass}/20** sources pass all bands. Five fail
  source-dependent closure far beyond float32/resampling tolerance.
- Unit: viewer headers omit `BUNIT`; official DR10 image-stack documentation
  establishes `{UNIT_TEXT}` for image/model and therefore residual.

Closure gates jointly require complete finite coverage, RMSE ≤
{CLOSURE_RMSE_NOISE_MAX} residual-noise sigma, L1 closure ≤ {CLOSURE_L1_MAX},
99.99th-percentile closure ≤ {CLOSURE_P9999_NOISE_MAX} sigma, and maximum
closure ≤ {CLOSURE_PEAK_RELATIVE_MAX} of observed peak amplitude. Closure
tests service additivity only; it does not validate Tractor morphology.

Authoritative tables:

- `tables/scene_triplet_alignment_v2.csv`
- `tables/scene_triplet_closure_v2.csv`
- `tables/input_integrity_v2.csv`
"""
    morphology = f"""# Morphology-in-residual audit — authoritative v2

Generated: `{utc_now()}`

Central association is valid for **{associations}/20** sources; invalid cases
are explicitly unavailable rather than measured on displaced segments.
`{measured_bands}` source-band records use catalog-centered apertures clipped
by catalog-component Voronoi boundaries, with target attribution rejected when
another primary catalog component lies within two PSF FWHM. Residual
flux/correlation/gradient, sign fractions, asymmetry, and core/halo power are
compared with exact translated central footprints having zero overlap with
dilated detected/catalog-source exclusion regions. This empirical null handles correlated coadd
noise more honestly than subtracting N×sigma², though it cannot make residual
and observed statistically independent.

Manual review of all 20 examples was separate from scalar calculation:
`{json.dumps(classes, sort_keys=True)}`. The Tractor model is a prediction, not
ground truth.

- `tables/residual_morphology_metrics_v2.csv`
- `tables/manual_morphology_review_v2.csv`
- `tables/radial_profiles_v2.csv`
- `{figure_dir}/scene_triplet_v2_*.png`
"""
    components = f"""# Scene-component audit — authoritative v2

Generated: `{utc_now()}`

Source detection uses a shared observed-noise scale on image and model layers;
official DR10 catalog rows provide central identity and physical per-band
neighbor fluxes. Raw model outer medians are measured before any background
subtraction. Bright-star candidates use the official G<13 GE/T2 definition;
large-galaxy components use `REF_CAT=L3`.

**0/20** central-only sources are declared reliably isolated. A SEP segment of
a summed scene prediction is not a per-source Tractor render and cannot bound
overlapping analytic wings. Consequently no whole-scene model and no segmented
model proxy is approved as a contaminant stamp.

- `tables/scene_component_audit_v2.csv`
"""
    psf = f"""# PSF audit — authoritative v2

Generated: `{utc_now()}`

Official `PSFSIZE_G/R/Z` ranges are g {np.min(by_band['g']):.3f}–{np.max(by_band['g']):.3f},
r {np.min(by_band['r']):.3f}–{np.max(by_band['r']):.3f}, and z
{np.min(by_band['z']):.3f}–{np.max(by_band['z']):.3f} arcsec. Direct addition
would mix source-dependent and band-dependent PSFs.

Scalar FWHM can screen possible pairings but cannot validate PSF wings,
ellipticity, or interpolation response. Convolution to a common broader PSF is
feasible in principle only after full local PSF kernels and moment tests are
available. Forward-rendering intrinsic Tractor profiles at the target PSF is
preferred; an already convolved model cutout cannot supply that without
deconvolution. No deconvolution was implemented.

- Official path: `south/coadd/<AAA>/<brick>/legacysurvey-<brick>-psfsize-<band>.fits.fz`
- `tables/psf_audit_v2.csv`
"""
    extraction = f"""# Source-extraction options — authoritative v2

Generated: `{utc_now()}`

- A preserves observed morphology but adds a second coadd-noise/background
  realization when used as a contaminant.
- B is low-noise but is only a segment of the summed parametric scene model;
  central source wings and neighbor overlap are not separable.
- C preserves residual structure but again carries coadd noise and fit errors;
  it also depends on the invalid B proxy.

Every proxy is measured with its own support and correct spatial gradient axes.
The measurements are descriptive; A and C share the observed/residual algebra
with their reference and are not independent validation. Subpixel shifts were
not tested, full-PSF matching is unvalidated, and **0/60 contaminant options
pass**. The only defensible target representation is the unchanged observed
scene, but no contaminant representation passes this probe.

- `tables/source_extraction_options_v2.csv`
- `{figure_dir}/source_extraction_v2_*.png`
"""
    return {
        "scene_triplet_audit_v2.md": audit,
        "morphology_residual_audit_v2.md": morphology,
        "scene_component_audit_v2.md": components,
        "psf_audit_v2.md": psf,
        "source_extraction_options_v2.md": extraction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--foundation-run", type=Path, required=True)
    parser.add_argument("--correction-id", default="v2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.run_dir, args.foundation_run):
        if "lockbox" in str(path).lower() or "sealed" in str(path).lower():
            raise SystemExit("refusing lockbox/sealed path")
    sources, grouped = base.load_products(args.run_dir)
    catalog_manifest = base.load_catalog_manifest(args.run_dir)
    manual_rows = read_csv(args.run_dir / "tables" / "manual_morphology_review.csv")
    manual_v2_rows = corrected_manual_review_rows(manual_rows)
    manual = {row["source_id"]: row for row in manual_rows}
    if len(manual) != 20 or set(manual) != {row["source_id"] for row in sources}:
        raise RuntimeError("manual review is not complete and one-to-one")
    allowed = {
        "model retains morphology sufficiently",
        "model moderately simplifies morphology",
        "model omits important target structure",
        "model fit is unusable",
    }
    if any(row["classification"] not in allowed or row["automatic_classification"] != "0" for row in manual_rows):
        raise RuntimeError("manual classification contract violation")

    integrity = validate_inputs(args.run_dir, args.foundation_run, sources, grouped, catalog_manifest)
    checkpoints = checkpoint_integrity(args.foundation_run)
    figure_dir = args.run_dir / "figures" / f"scene_probe_{args.correction_id}"
    figure_dir.mkdir(parents=True, exist_ok=False)
    alignment_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    morph_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    psf_rows: list[dict[str, Any]] = []
    records: list[base.SourceRecord] = []
    masks: dict[str, np.ndarray] = {}
    blanks_by_source: dict[str, list[tuple[float, float, np.ndarray]]] = {}

    for rank, source in enumerate(sources, start=1):
        source_id = source["source_id"]
        products = {
            product: base.open_product(Path(grouped[source_id][product]["relative_path"]))
            for product in EXPECTED_LAYERS
        }
        alignment_rows.append(corrected_alignment(source, products, integrity))
        source_closure = corrected_closure(source, products)
        closure_rows.extend(source_closure)
        closure_valid = all(row["closure_valid_v2"] for row in source_closure)
        record = base.load_source_record(source, grouped[source_id], catalog_manifest[source_id])
        records.append(record)
        source_morph, mask, blanks, _radius = corrected_morphology(record, manual[source_id])
        morph_rows.extend(source_morph)
        radial_rows.extend(corrected_radial_profiles(record, mask, blanks))
        masks[source_id] = mask
        blanks_by_source[source_id] = blanks
        component = corrected_components(record, closure_valid)
        component_rows.append(component)
        source_extraction, arrays, supports = corrected_extraction(
            record, mask, blanks, manual[source_id], component
        )
        extraction_rows.extend(source_extraction)
        psf_rows.extend(corrected_psf(record))
        corrected_scene_figure(
            record,
            mask,
            blanks,
            manual[source_id],
            closure_valid,
            figure_dir / f"scene_triplet_v2_{rank:02d}_{source_id}.png",
        )
        corrected_extraction_figure(
            record,
            arrays,
            supports,
            figure_dir / f"source_extraction_v2_{rank:02d}_{source_id}.png",
        )
        print(f"[{rank:02d}/20] corrected {source_id}", flush=True)

    tables = {
        "input_integrity_v2.csv": integrity,
        "checkpoint_integrity_v2.csv": checkpoints,
        "scene_triplet_alignment_v2.csv": alignment_rows,
        "scene_triplet_closure_v2.csv": closure_rows,
        "residual_morphology_metrics_v2.csv": morph_rows,
        "radial_profiles_v2.csv": radial_rows,
        "manual_morphology_review_v2.csv": manual_v2_rows,
        "scene_component_audit_v2.csv": component_rows,
        "source_extraction_options_v2.csv": extraction_rows,
        "psf_audit_v2.csv": psf_rows,
    }
    for filename, rows in tables.items():
        write_csv(args.run_dir / "tables" / filename, rows)

    report_texts = reports(
        args.run_dir,
        alignment_rows,
        closure_rows,
        morph_rows,
        component_rows,
        psf_rows,
        manual_v2_rows,
        figure_dir,
    )
    for filename, value in report_texts.items():
        write_text(args.run_dir / "diagnostics" / filename, value)

    environment = {
        "generated_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "astropy": astropy.__version__,
        "scipy": scipy.__version__,
        "sep": sep.__version__,
        "matplotlib": matplotlib.__version__,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "base_script": str(Path(base.__file__).resolve()),
        "base_script_sha256": sha256_file(Path(base.__file__).resolve()),
        "manual_review_input_sha256": sha256_file(args.run_dir / "tables" / "manual_morphology_review.csv"),
        "scene_manifest_sha256": sha256_file(args.run_dir / "manifests" / "scene_triplet_download_manifest.csv"),
        "catalog_manifest_sha256": sha256_file(args.run_dir / "manifests" / "official_catalog_download_manifest.csv"),
        "engineering_sources_manifest_sha256": sha256_file(args.run_dir / "manifests" / "engineering_sources_20.csv"),
        "git_branch": subprocess.run(["git", "branch", "--show-current"], check=True, text=True, capture_output=True).stdout.strip(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip(),
        "superseded_products_preserved": [
            "tables/scene_triplet_alignment.csv",
            "tables/scene_triplet_closure.csv",
            "tables/residual_morphology_metrics.csv",
            "tables/scene_component_audit.csv",
            "tables/source_extraction_options.csv",
            "tables/psf_audit.csv",
            "figures/scene_probe_v1/",
        ],
    }
    write_text(
        args.run_dir / "logs" / "scene_probe_correction_v2_environment.json",
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
    )
    summary = {
        "alignment_pass": sum(row["alignment_pass_v2"] for row in alignment_rows),
        "closure_pass_sources": sum(
            all(row["closure_valid_v2"] for row in closure_rows if row["source_id"] == source["source_id"])
            for source in sources
        ),
        "valid_central_associations": sum(row["central_association_valid"] for row in component_rows),
        "measured_radial_profile_rows": sum(row["profile_status"].startswith("measured") for row in radial_rows),
        "central_only_isolation_reliable": sum(row["central_only_model_isolation_reliable"] for row in component_rows),
        "contaminant_options_pass": sum(row["suitable_as_contaminant_for_single_noise_contract"] for row in extraction_rows),
        "checkpoint_integrity_pass": all(row["identity_unchanged"] for row in checkpoints),
        "figure_count_v2": len(list(figure_dir.glob("*.png"))),
    }
    write_text(
        args.run_dir / "logs" / "scene_probe_correction_v2_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    generated_paths = [args.run_dir / "tables" / filename for filename in tables]
    generated_paths.extend(args.run_dir / "diagnostics" / filename for filename in report_texts)
    generated_paths.extend(
        [
            args.run_dir / "logs" / "scene_probe_correction_v2_environment.json",
            args.run_dir / "logs" / "scene_probe_correction_v2_summary.json",
        ]
    )
    generated_paths.extend(sorted(figure_dir.glob("*.png")))
    write_csv(
        args.run_dir / "tables" / "output_inventory_v2.csv",
        output_inventory_rows(args.run_dir, args.foundation_run, generated_paths),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
