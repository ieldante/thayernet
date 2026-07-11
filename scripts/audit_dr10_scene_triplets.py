#!/usr/bin/env python3
"""Audit matched DR10 observed/model/residual engineering scene triplets.

The command is analysis-only: it creates new audit tables and diagnostic
figures, never trains a model, never creates blend manifests, and never reads a
role split. The Tractor layer is evaluated as a model prediction, not truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage


BANDS = ("g", "r", "z")
PIXEL_SCALE = 0.262
DETECTION_SIGMA = 2.5
MINAREA = 8
DEBLEND_NTHRESH = 32
DEBLEND_CONT = 0.005
CLOSURE_NOISE_RATIO_MAX = 0.01
CLOSURE_L1_RELATIVE_MAX = 0.001
UNIT_TEXT = "nanomaggies per coadd pixel"
OFFICIAL_UNIT_URL = "https://www.legacysurvey.org/dr10/files/#image-stacks-southcoadd"
OFFICIAL_CATALOG_URL = "https://www.legacysurvey.org/dr10/catalogs/"
OFFICIAL_PSF_URL = "https://www.legacysurvey.org/dr10/files/"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_text_exclusive(path: Path, text: str) -> None:
    write_exclusive(path, text.encode("utf-8"))


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing headerless empty table: {path}")
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


def save_figure_exclusive(path: Path, figure: plt.Figure, dpi: int = 130) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing existing figure: {path}")
    figure.savefig(path, dpi=dpi, bbox_inches="tight", metadata={"Software": "audit_dr10_scene_triplets.py"})
    plt.close(figure)


def robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    median = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
    return scale


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan")
    xx = np.asarray(x[keep], dtype=np.float64)
    yy = np.asarray(y[keep], dtype=np.float64)
    xx -= np.mean(xx)
    yy -= np.mean(yy)
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return float(np.sum(xx * yy) / denom) if denom > 0 else float("nan")


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def text_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def bool_int(value: Any) -> int:
    return int(bool(value))


@dataclass
class Product:
    path: Path
    data: np.ndarray
    header: fits.Header
    wcs: WCS


@dataclass
class Detection:
    objects: np.ndarray
    segmentation: np.ndarray
    detection_image: np.ndarray
    central_index: int | None
    target_x: float
    target_y: float


@dataclass
class SourceRecord:
    source: dict[str, str]
    products: dict[str, Product]
    catalog_path: Path
    catalog_data: np.ndarray
    catalog_central_index: int
    catalog_central_separation_arcsec: float
    catalog_target_x: float
    catalog_target_y: float
    observed_subtracted: np.ndarray
    model_subtracted: np.ndarray
    residual_subtracted: np.ndarray
    noises: np.ndarray
    observed_detection: Detection
    model_detection: Detection
    psf_fwhm_arcsec: np.ndarray
    psf_fwhm_pixels: np.ndarray


def load_products(run_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    source_path = run_dir / "manifests" / "engineering_sources_20.csv"
    manifest_path = run_dir / "manifests" / "scene_triplet_download_manifest.csv"
    with source_path.open(newline="", encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(sources) != 20 or len(records) != 60:
        raise RuntimeError("expected 20 sources and 60 triplet manifest rows")
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in records:
        if row["status"] != "downloaded_valid":
            raise RuntimeError(f"non-valid triplet row: {row}")
        grouped.setdefault(row["source_id"], {})[row["product"]] = row
    expected = {"observed", "model", "residual"}
    if any(set(grouped.get(source["source_id"], {})) != expected for source in sources):
        raise RuntimeError("triplet manifest is not one-to-one by source/product")
    return sources, grouped


def load_catalog_manifest(run_dir: Path) -> dict[str, dict[str, str]]:
    path = run_dir / "manifests" / "official_catalog_download_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or any(row["status"] != "downloaded_valid" for row in rows):
        raise RuntimeError("official catalog manifest is incomplete")
    result = {row["source_id"]: row for row in rows}
    if len(result) != 20:
        raise RuntimeError("official catalog manifest has duplicate sources")
    return result


def open_product(path: Path) -> Product:
    with fits.open(path, mode="readonly", memmap=False, checksum=True) as hdul:
        hdul.verify("exception")
        if len(hdul) != 1 or hdul[0].data is None:
            raise ValueError(f"unexpected FITS structure: {path}")
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    return Product(path=path, data=data, header=header, wcs=WCS(header, naxis=2))


def canonical_matrix(wcs: WCS) -> np.ndarray:
    return np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)


def full_grid_wcs_separation_arcsec(a: WCS, b: WCS, shape: tuple[int, int]) -> float:
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    ra_a, dec_a = a.all_pix2world(xx.ravel(), yy.ravel(), 0)
    ra_b, dec_b = b.all_pix2world(xx.ravel(), yy.ravel(), 0)
    coord_a = SkyCoord(ra_a * u.deg, dec_a * u.deg)
    coord_b = SkyCoord(ra_b * u.deg, dec_b * u.deg)
    return float(np.max(coord_a.separation(coord_b).arcsec))


def alignment_row(source: dict[str, str], products: dict[str, Product]) -> dict[str, Any]:
    order = ("observed", "model", "residual")
    headers = [products[name].header for name in order]
    shapes = [products[name].data.shape for name in order]
    band_orders = [
        "".join(text_value(header.get(f"BAND{i}", "")).lower() for i in range(3))
        for header in headers
    ]
    crpix = [(float(h["CRPIX1"]), float(h["CRPIX2"])) for h in headers]
    crval = [(float(h["CRVAL1"]), float(h["CRVAL2"])) for h in headers]
    matrices = [canonical_matrix(products[name].wcs) for name in order]
    scales = [
        np.abs(np.asarray([q.to_value("deg") for q in products[name].wcs.proj_plane_pixel_scales()]))
        * 3600.0
        for name in order
    ]
    bunits = [text_value(h.get("BUNIT", "")) for h in headers]
    ctype = [(text_value(h.get("CTYPE1", "")), text_value(h.get("CTYPE2", ""))) for h in headers]
    world_model = full_grid_wcs_separation_arcsec(
        products["observed"].wcs, products["model"].wcs, shapes[0][-2:]
    )
    world_resid = full_grid_wcs_separation_arcsec(
        products["observed"].wcs, products["residual"].wcs, shapes[0][-2:]
    )
    shape_equal = len(set(shapes)) == 1
    bands_equal = len(set(band_orders)) == 1 and band_orders[0] == "grz"
    crpix_equal = all(np.array_equal(crpix[0], value) for value in crpix[1:])
    crval_equal = all(np.array_equal(crval[0], value) for value in crval[1:])
    matrix_equal = all(np.array_equal(matrices[0], value) for value in matrices[1:])
    ctype_equal = len(set(ctype)) == 1
    scale_equal = all(np.allclose(scales[0], value, rtol=0, atol=1e-12) for value in scales[1:])
    unit_headers_equal = len(set(bunits)) == 1
    documented_units_resolved = unit_headers_equal and all(value == "" for value in bunits)
    wcs_equal = world_model <= 1e-7 and world_resid <= 1e-7
    aligned = all(
        (
            shape_equal,
            bands_equal,
            crpix_equal,
            crval_equal,
            matrix_equal,
            ctype_equal,
            scale_equal,
            unit_headers_equal,
            documented_units_resolved,
            wcs_equal,
        )
    )
    return {
        "source_id": source["source_id"],
        "catalog_row_index": source["catalog_row_index"],
        "ra": source["ra"],
        "dec": source["dec"],
        "observed_shape": "x".join(map(str, shapes[0])),
        "model_shape": "x".join(map(str, shapes[1])),
        "residual_shape": "x".join(map(str, shapes[2])),
        "shape_identical": bool_int(shape_equal),
        "observed_band_order": band_orders[0],
        "model_band_order": band_orders[1],
        "residual_band_order": band_orders[2],
        "band_order_identical": bool_int(bands_equal),
        "crpix_identical": bool_int(crpix_equal),
        "crval_identical": bool_int(crval_equal),
        "canonical_cd_or_pc_matrix_identical": bool_int(matrix_equal),
        "ctype_identical": bool_int(ctype_equal),
        "pixel_scale_identical": bool_int(scale_equal),
        "pixel_scale_x_arcsec": float(scales[0][0]),
        "pixel_scale_y_arcsec": float(scales[0][1]),
        "observed_bunit": bunits[0],
        "model_bunit": bunits[1],
        "residual_bunit": bunits[2],
        "unit_headers_identical": bool_int(unit_headers_equal),
        "unit_headers_absent": bool_int(all(value == "" for value in bunits)),
        "resolved_common_unit": UNIT_TEXT,
        "unit_resolution_source": OFFICIAL_UNIT_URL,
        "max_observed_model_world_separation_arcsec": world_model,
        "max_observed_residual_world_separation_arcsec": world_resid,
        "full_grid_wcs_identical": bool_int(wcs_equal),
        "observed_survey_version": f"{headers[0].get('SURVEY', '')}/{headers[0].get('VERSION', '')}",
        "model_survey_version": f"{headers[1].get('SURVEY', '')}/{headers[1].get('VERSION', '')}",
        "residual_survey_version": f"{headers[2].get('SURVEY', '')}/{headers[2].get('VERSION', '')}",
        "alignment_pass": bool_int(aligned),
    }


def closure_rows(source: dict[str, str], products: dict[str, Product]) -> list[dict[str, Any]]:
    image = products["observed"].data
    model = products["model"].data
    residual = products["residual"].data
    rows: list[dict[str, Any]] = []
    for band_index, band in enumerate(BANDS):
        finite = np.isfinite(image[band_index]) & np.isfinite(model[band_index]) & np.isfinite(residual[band_index])
        closure = image[band_index] - model[band_index] - residual[band_index]
        values = closure[finite]
        obs = image[band_index][finite]
        resid = residual[band_index][finite]
        rmse = float(np.sqrt(np.mean(values * values))) if values.size else float("nan")
        residual_noise = robust_scale(resid)
        noise_ratio = rmse / residual_noise if residual_noise > 0 else float("inf")
        obs_total = float(np.sum(obs))
        signed_relative = float(np.sum(values) / obs_total) if obs_total != 0 else float("nan")
        l1_relative = float(np.sum(np.abs(values)) / np.sum(np.abs(obs))) if np.sum(np.abs(obs)) > 0 else float("nan")
        valid = (
            finite.size == image[band_index].size
            and noise_ratio <= CLOSURE_NOISE_RATIO_MAX
            and l1_relative <= CLOSURE_L1_RELATIVE_MAX
        )
        rows.append(
            {
                "source_id": source["source_id"],
                "catalog_row_index": source["catalog_row_index"],
                "band": band,
                "maximum_absolute_closure": float(np.max(np.abs(values))) if values.size else float("nan"),
                "mean_closure": float(np.mean(values)) if values.size else float("nan"),
                "closure_rmse": rmse,
                "relative_total_flux_closure": signed_relative,
                "absolute_l1_relative_closure": l1_relative,
                "residual_robust_noise": residual_noise,
                "closure_rmse_to_residual_noise": noise_ratio,
                "finite_pixel_count": int(values.size),
                "total_pixel_count": int(image[band_index].size),
                "finite_pixel_coverage": float(values.size / image[band_index].size),
                "closure_noise_ratio_max": CLOSURE_NOISE_RATIO_MAX,
                "closure_l1_relative_max": CLOSURE_L1_RELATIVE_MAX,
                "closure_valid": bool_int(valid),
            }
        )
    return rows


def observed_background_and_noise(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subtracted: list[np.ndarray] = []
    backgrounds: list[np.ndarray] = []
    noises: list[float] = []
    for plane in cube:
        clean = np.ascontiguousarray(plane.astype(np.float32))
        background = sep.Background(clean, bw=32, bh=32, fw=3, fh=3)
        back = np.asarray(background.back(), dtype=np.float64)
        rms = np.asarray(background.rms(), dtype=np.float64)
        valid_rms = rms[np.isfinite(rms) & (rms > 0)]
        noise = float(np.median(valid_rms)) if valid_rms.size else robust_scale(clean - back)
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("unable to derive positive observed noise")
        backgrounds.append(back)
        subtracted.append(np.asarray(plane, dtype=np.float64) - back)
        noises.append(noise)
    return np.asarray(subtracted), np.asarray(backgrounds), np.asarray(noises)


def outer_mask(shape: tuple[int, int], width: int = 24) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    mask[width:-width, width:-width] = False
    return mask


def subtract_outer_median(cube: np.ndarray) -> np.ndarray:
    mask = outer_mask(cube.shape[-2:])
    return np.asarray([plane - np.median(plane[mask]) for plane in cube], dtype=np.float64)


def detect(cube_subtracted: np.ndarray, noises: np.ndarray, target_x: float, target_y: float) -> Detection:
    standardized = [plane / noise for plane, noise in zip(cube_subtracted, noises, strict=True)]
    detection = np.asarray(np.sum(standardized, axis=0) / math.sqrt(len(BANDS)), dtype=np.float32)
    objects, segmentation = sep.extract(
        np.ascontiguousarray(detection),
        DETECTION_SIGMA,
        minarea=MINAREA,
        deblend_nthresh=DEBLEND_NTHRESH,
        deblend_cont=DEBLEND_CONT,
        clean=True,
        segmentation_map=True,
    )
    central: int | None = None
    ix = int(np.clip(math.floor(target_x + 0.5), 0, detection.shape[1] - 1))
    iy = int(np.clip(math.floor(target_y + 0.5), 0, detection.shape[0] - 1))
    label = int(segmentation[iy, ix])
    if 1 <= label <= len(objects):
        central = label - 1
    elif len(objects):
        distance = np.hypot(objects["x"] - target_x, objects["y"] - target_y)
        nearest = int(np.argmin(distance))
        if float(distance[nearest]) <= 12.0:
            central = nearest
    return Detection(
        objects=objects,
        segmentation=segmentation,
        detection_image=detection,
        central_index=central,
        target_x=target_x,
        target_y=target_y,
    )


def load_source_record(
    source: dict[str, str],
    manifests: dict[str, dict[str, str]],
    catalog_manifest: dict[str, str],
) -> SourceRecord:
    products = {name: open_product(Path(row["relative_path"])) for name, row in manifests.items()}
    catalog_path = Path(catalog_manifest["relative_path"])
    with fits.open(catalog_path, mode="readonly", memmap=False, checksum=True) as hdul:
        catalog_data = np.asarray(hdul[1].data)
    central_index = int(catalog_manifest["central_row_index"])
    central = catalog_data[central_index]
    names = {name.lower(): name for name in catalog_data.dtype.names or ()}
    central_ra = float(central[names["ra"]])
    central_dec = float(central[names["dec"]])
    target_x, target_y = products["observed"].wcs.all_world2pix(central_ra, central_dec, 0)
    observed_subtracted, _background, noises = observed_background_and_noise(products["observed"].data)
    model_subtracted = subtract_outer_median(products["model"].data)
    residual_subtracted = subtract_outer_median(products["residual"].data)
    observed_detection = detect(observed_subtracted, noises, float(target_x), float(target_y))
    model_detection = detect(model_subtracted, noises, float(target_x), float(target_y))
    psf = np.asarray(
        [float(catalog_manifest[f"psfsize_{band}_arcsec"]) for band in BANDS],
        dtype=np.float64,
    )
    return SourceRecord(
        source=source,
        products=products,
        catalog_path=catalog_path,
        catalog_data=catalog_data,
        catalog_central_index=central_index,
        catalog_central_separation_arcsec=float(catalog_manifest["central_separation_arcsec"]),
        catalog_target_x=float(target_x),
        catalog_target_y=float(target_y),
        observed_subtracted=observed_subtracted,
        model_subtracted=model_subtracted,
        residual_subtracted=residual_subtracted,
        noises=noises,
        observed_detection=observed_detection,
        model_detection=model_detection,
        psf_fwhm_arcsec=psf,
        psf_fwhm_pixels=psf / PIXEL_SCALE,
    )


def central_mask(detection: Detection) -> np.ndarray:
    if detection.central_index is None:
        return np.zeros(detection.segmentation.shape, dtype=bool)
    return detection.segmentation == (detection.central_index + 1)


def detection_component_metrics(detection: Detection) -> dict[str, Any]:
    result: dict[str, Any] = {
        "component_count": int(len(detection.objects)),
        "central_present": bool_int(detection.central_index is not None),
        "central_index": "" if detection.central_index is None else int(detection.central_index),
        "central_centroid_x": "",
        "central_centroid_y": "",
        "central_centroid_offset_from_catalog_px": "",
        "central_area_pixels": 0,
        "central_mask_touches_border": 0,
        "neighbor_count": max(0, int(len(detection.objects)) - bool_int(detection.central_index is not None)),
        "nearest_neighbor_distance_px": "",
        "total_neighbor_to_target_detection_flux_ratio": "",
        "nearest_neighbor_to_target_detection_flux_ratio": "",
    }
    if detection.central_index is None:
        return result
    index = detection.central_index
    obj = detection.objects[index]
    mask = central_mask(detection)
    x = float(obj["x"])
    y = float(obj["y"])
    target_flux = abs(float(obj["flux"]))
    others = [i for i in range(len(detection.objects)) if i != index]
    result.update(
        central_centroid_x=x,
        central_centroid_y=y,
        central_centroid_offset_from_catalog_px=float(math.hypot(x - detection.target_x, y - detection.target_y)),
        central_area_pixels=int(mask.sum()),
        central_mask_touches_border=bool_int(
            mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        ),
    )
    if others:
        distances = np.asarray(
            [math.hypot(float(detection.objects[i]["x"]) - x, float(detection.objects[i]["y"]) - y) for i in others]
        )
        fluxes = np.asarray([abs(float(detection.objects[i]["flux"])) for i in others])
        nearest_position = int(np.argmin(distances))
        result.update(
            nearest_neighbor_distance_px=float(distances[nearest_position]),
            total_neighbor_to_target_detection_flux_ratio=(
                float(np.sum(fluxes) / target_flux) if target_flux > 0 else float("inf")
            ),
            nearest_neighbor_to_target_detection_flux_ratio=(
                float(fluxes[nearest_position] / target_flux) if target_flux > 0 else float("inf")
            ),
        )
    return result


def catalog_scene_metrics(record: SourceRecord) -> dict[str, Any]:
    data = record.catalog_data
    names = {name.lower(): name for name in data.dtype.names or ()}
    ra = np.asarray(data[names["ra"]], dtype=float)
    dec = np.asarray(data[names["dec"]], dtype=float)
    x, y = record.products["observed"].wcs.all_world2pix(ra, dec, 0)
    in_frame = (x >= -0.5) & (x <= 255.5) & (y >= -0.5) & (y <= 255.5)
    ref_cat = np.asarray([text_value(v) for v in data[names["ref_cat"]]])
    types = np.asarray([text_value(v) for v in data[names["type"]]])
    gaia_g = np.asarray(data[names["gaia_phot_g_mean_mag"]], dtype=float)
    bright = in_frame & np.isin(ref_cat, ["GE", "T2"]) & np.isfinite(gaia_g) & (gaia_g < 13)
    medium = in_frame & np.isin(ref_cat, ["GE", "T2"]) & np.isfinite(gaia_g) & (gaia_g < 16)
    large = in_frame & (ref_cat == "L3")
    central = data[record.catalog_central_index]
    second_sep = float("nan")
    coords = SkyCoord(ra * u.deg, dec * u.deg)
    target = SkyCoord(float(record.source["ra"]) * u.deg, float(record.source["dec"]) * u.deg)
    separations = np.sort(target.separation(coords).arcsec)
    if len(separations) > 1:
        second_sep = float(separations[1])
    return {
        "official_catalog_components_in_frame": int(in_frame.sum()),
        "central_catalog_release": int(central[names["release"]]),
        "central_catalog_brickid": int(central[names["brickid"]]),
        "central_catalog_brickname": text_value(central[names["brickname"]]),
        "central_catalog_objid": int(central[names["objid"]]),
        "central_catalog_type": text_value(central[names["type"]]),
        "central_catalog_ref_cat": text_value(central[names["ref_cat"]]),
        "central_catalog_separation_from_requested_arcsec": record.catalog_central_separation_arcsec,
        "second_catalog_source_separation_from_requested_arcsec": second_sep,
        "catalog_bright_star_component_count": int(bright.sum()),
        "catalog_medium_bright_star_component_count": int(medium.sum()),
        "catalog_large_galaxy_component_count": int(large.sum()),
        "catalog_psf_component_count": int(np.sum(in_frame & (types == "PSF"))),
    }


def curve_of_growth_stability(record: SourceRecord, mask: np.ndarray) -> tuple[float, float, float]:
    if not mask.any():
        return float("nan"), float("nan"), float("nan")
    first = ndimage.binary_dilation(mask, iterations=max(1, int(math.ceil(np.median(record.psf_fwhm_pixels)))))
    second = ndimage.binary_dilation(first, iterations=max(1, int(math.ceil(np.median(record.psf_fwhm_pixels)))))
    total_first = float(np.sum(np.abs(record.model_subtracted[:, first])))
    total_second = float(np.sum(np.abs(record.model_subtracted[:, second])))
    stability = abs(total_second - total_first) / total_second if total_second > 0 else float("inf")
    yy, xx = np.nonzero(mask)
    cx = float(record.model_detection.objects[record.model_detection.central_index]["x"]) if record.model_detection.central_index is not None else record.catalog_target_x
    cy = float(record.model_detection.objects[record.model_detection.central_index]["y"]) if record.model_detection.central_index is not None else record.catalog_target_y
    radius95 = float(np.percentile(np.hypot(xx - cx, yy - cy), 95)) if len(xx) else float("nan")
    return stability, radius95, total_second


def scene_component_row(record: SourceRecord, triplet_closure_valid: bool) -> dict[str, Any]:
    observed = detection_component_metrics(record.observed_detection)
    model = detection_component_metrics(record.model_detection)
    catalog = catalog_scene_metrics(record)
    model_mask = central_mask(record.model_detection)
    stability, radius95, _flux = curve_of_growth_stability(record, model_mask)
    outer = outer_mask(model_mask.shape)
    pedestal_ratios = [abs(float(np.median(record.model_subtracted[i][outer]))) / record.noises[i] for i in range(3)]
    reasons: list[str] = []
    if not model["central_present"]:
        reasons.append("no_central_model_detection")
    if model["central_present"] and float(model["central_centroid_offset_from_catalog_px"]) > 3.0:
        reasons.append("model_centroid_offset_gt_3px")
    if record.catalog_central_separation_arcsec > 1.0:
        reasons.append("dr10_catalog_match_gt_1arcsec")
    if model["central_mask_touches_border"]:
        reasons.append("central_model_mask_touches_border")
    nearest = model["nearest_neighbor_distance_px"]
    safe_radius = radius95 + 2.0 * float(np.max(record.psf_fwhm_pixels)) if np.isfinite(radius95) else float("inf")
    if nearest != "" and float(nearest) <= safe_radius:
        reasons.append("modeled_neighbor_within_profile_safety_radius")
    if np.isfinite(stability) and stability > 0.05:
        reasons.append("model_curve_of_growth_not_stable_to_5pct")
    if not np.isfinite(stability):
        reasons.append("model_curve_of_growth_unavailable")
    if catalog["second_catalog_source_separation_from_requested_arcsec"] < 2.0 * float(np.max(record.psf_fwhm_arcsec)):
        reasons.append("catalog_component_ambiguity_within_2fwhm")
    reliable = not reasons
    row: dict[str, Any] = {
        "source_id": record.source["source_id"],
        "catalog_row_index": record.source["catalog_row_index"],
        "observed_component_count": observed.pop("component_count"),
        "modeled_component_count": model.pop("component_count"),
        **{f"observed_{key}": value for key, value in observed.items()},
        **{f"model_{key}": value for key, value in model.items()},
        **catalog,
        "modeled_neighbor_count": model.get("neighbor_count", 0),
        "neighbor_to_target_flux_ratio": model.get("total_neighbor_to_target_detection_flux_ratio", ""),
        "model_central_radius95_px": radius95,
        "model_curve_of_growth_fractional_change": stability,
        "model_outer_pedestal_max_noise_ratio": float(max(pedestal_ratios)),
        "explicit_background_catalog_component_count": 0,
        "scene_triplet_closure_valid": bool_int(triplet_closure_valid),
        "central_only_model_isolation_reliable": bool_int(reliable),
        "central_only_model_isolation_reasons": ";".join(reasons),
        "catalog_source": OFFICIAL_CATALOG_URL,
    }
    return row


def rotate_180_about(array: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
    yy, xx = np.indices(array.shape, dtype=np.float64)
    sample_x = 2.0 * center_x - xx
    sample_y = 2.0 * center_y - yy
    return ndimage.map_coordinates(array, [sample_y, sample_x], order=1, mode="constant", cval=np.nan)


def radial_profile_rows(record: SourceRecord, maximum_radius: int = 64) -> list[dict[str, Any]]:
    yy, xx = np.indices(record.observed_subtracted.shape[-2:])
    cx = record.catalog_target_x
    cy = record.catalog_target_y
    if record.observed_detection.central_index is not None:
        obj = record.observed_detection.objects[record.observed_detection.central_index]
        if math.hypot(float(obj["x"]) - cx, float(obj["y"]) - cy) <= 6.0:
            cx, cy = float(obj["x"]), float(obj["y"])
    radius = np.hypot(xx - cx, yy - cy)
    rows: list[dict[str, Any]] = []
    arrays = {
        "observed": record.observed_subtracted,
        "model": record.model_subtracted,
        "residual": record.residual_subtracted,
    }
    for integer_radius in range(maximum_radius):
        annulus = (radius >= integer_radius) & (radius < integer_radius + 1)
        for product, cube in arrays.items():
            for band_index, band in enumerate(BANDS):
                values = cube[band_index][annulus]
                rows.append(
                    {
                        "source_id": record.source["source_id"],
                        "band": band,
                        "product": product,
                        "radius_inner_pixels": integer_radius,
                        "radius_outer_pixels": integer_radius + 1,
                        "radius_mid_arcsec": (integer_radius + 0.5) * PIXEL_SCALE,
                        "finite_pixel_count": int(np.isfinite(values).sum()),
                        "mean_flux_per_pixel": float(np.nanmean(values)),
                        "median_flux_per_pixel": float(np.nanmedian(values)),
                        "rms_flux_per_pixel": float(np.sqrt(np.nanmean(values * values))),
                    }
                )
    return rows


def flux_radii(record: SourceRecord, mask: np.ndarray) -> tuple[float, float]:
    if not mask.any():
        return float("nan"), float("nan")
    yy, xx = np.indices(mask.shape)
    cx, cy = record.catalog_target_x, record.catalog_target_y
    radius = np.hypot(xx - cx, yy - cy)
    flux = np.clip(record.observed_detection.detection_image, 0, None) * mask
    order = np.argsort(radius[mask])
    radii = radius[mask][order]
    values = flux[mask][order]
    cumulative = np.cumsum(values)
    if not len(cumulative) or cumulative[-1] <= 0:
        return float("nan"), float("nan")
    return (
        float(radii[np.searchsorted(cumulative, 0.5 * cumulative[-1])]),
        float(radii[np.searchsorted(cumulative, 0.9 * cumulative[-1])]),
    )


def morphology_rows(record: SourceRecord) -> list[dict[str, Any]]:
    mask = central_mask(record.observed_detection)
    if not mask.any():
        return [
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "band": band,
                "central_mask_available": 0,
                "metric_status": "unavailable_no_central_observed_mask",
            }
            for band in BANDS
        ]
    r50, r90 = flux_radii(record, mask)
    yy, xx = np.indices(mask.shape)
    radius = np.hypot(xx - record.catalog_target_x, yy - record.catalog_target_y)
    core = mask & (radius <= r50) if np.isfinite(r50) else np.zeros_like(mask)
    halo = mask & (radius > r50) & (radius <= r90) if np.isfinite(r50) and np.isfinite(r90) else np.zeros_like(mask)
    rows: list[dict[str, Any]] = []
    for band_index, band in enumerate(BANDS):
        observed = record.observed_subtracted[band_index]
        model = record.model_subtracted[band_index]
        residual = record.residual_subtracted[band_index]
        observed_flux = float(np.sum(observed[mask]))
        residual_flux = float(np.sum(residual[mask]))
        gx_obs = ndimage.sobel(observed, axis=1, mode="nearest")
        gy_obs = ndimage.sobel(observed, axis=0, mode="nearest")
        gx_res = ndimage.sobel(residual, axis=1, mode="nearest")
        gy_res = ndimage.sobel(residual, axis=0, mode="nearest")
        vector_denom = float(
            np.sqrt(
                np.sum((gx_obs[mask] ** 2 + gy_obs[mask] ** 2))
                * np.sum((gx_res[mask] ** 2 + gy_res[mask] ** 2))
            )
        )
        vector_corr = (
            float(np.sum(gx_obs[mask] * gx_res[mask] + gy_obs[mask] * gy_res[mask]) / vector_denom)
            if vector_denom > 0
            else float("nan")
        )
        magnitude_corr = pearson(
            np.hypot(gx_obs, gy_obs)[mask], np.hypot(gx_res, gy_res)[mask]
        )
        rotated = rotate_180_about(residual, record.catalog_target_x, record.catalog_target_y)
        valid_asym = mask & np.isfinite(rotated)
        asym_denom = float(np.sum(np.abs(residual[valid_asym]) + np.abs(rotated[valid_asym])))
        asymmetry = (
            float(np.sum(np.abs(residual[valid_asym] - rotated[valid_asym])) / asym_denom)
            if asym_denom > 0
            else float("nan")
        )
        central_values = residual[mask]
        abs_total = float(np.sum(np.abs(central_values)))
        noise = float(record.noises[band_index])
        core_power = float(np.sum(residual[core] ** 2)) if core.any() else float("nan")
        halo_power = float(np.sum(residual[halo] ** 2)) if halo.any() else float("nan")
        rows.append(
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "band": band,
                "central_mask_available": 1,
                "central_mask_pixel_count": int(mask.sum()),
                "core_pixel_count": int(core.sum()),
                "halo_pixel_count": int(halo.sum()),
                "observed_central_flux": observed_flux,
                "model_central_mask_flux": float(np.sum(model[mask])),
                "residual_central_flux": residual_flux,
                "residual_to_observed_flux_fraction": residual_flux / observed_flux if observed_flux != 0 else float("nan"),
                "residual_observed_spatial_correlation": pearson(residual[mask], observed[mask]),
                "residual_observed_gradient_vector_correlation": vector_corr,
                "residual_observed_edge_magnitude_correlation": magnitude_corr,
                "positive_central_residual_pixel_fraction": float(np.mean(central_values > 0)),
                "negative_central_residual_pixel_fraction": float(np.mean(central_values < 0)),
                "positive_significant_residual_pixel_fraction_3sigma": float(np.mean(central_values > 3 * noise)),
                "negative_significant_residual_pixel_fraction_3sigma": float(np.mean(central_values < -3 * noise)),
                "positive_central_absolute_flux_fraction": float(np.sum(np.clip(central_values, 0, None)) / abs_total) if abs_total > 0 else float("nan"),
                "negative_central_absolute_flux_fraction": float(np.sum(np.clip(-central_values, 0, None)) / abs_total) if abs_total > 0 else float("nan"),
                "residual_asymmetry_180": asymmetry,
                "observed_flux_radius50_pixels": r50,
                "observed_flux_radius90_pixels": r90,
                "residual_core_power": core_power,
                "residual_halo_power": halo_power,
                "residual_core_noise_subtracted_power": core_power - int(core.sum()) * noise**2 if core.any() else float("nan"),
                "residual_halo_noise_subtracted_power": halo_power - int(halo.sum()) * noise**2 if halo.any() else float("nan"),
                "residual_core_to_observed_power_fraction": core_power / float(np.sum(observed[core] ** 2)) if core.any() and np.sum(observed[core] ** 2) > 0 else float("nan"),
                "residual_halo_to_observed_power_fraction": halo_power / float(np.sum(observed[halo] ** 2)) if halo.any() and np.sum(observed[halo] ** 2) > 0 else float("nan"),
                "residual_noise_sigma": noise,
                "metric_status": "measured",
            }
        )
    return rows


def display_plane(plane: np.ndarray) -> np.ndarray:
    finite = plane[np.isfinite(plane)]
    if not finite.size:
        return np.zeros(plane.shape)
    median = float(np.median(finite))
    scale = float(np.percentile(np.abs(finite - median), 99.5))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    transformed = np.arcsinh((plane - median) / max(0.1 * scale, 1e-30)) / np.arcsinh(10.0)
    return np.clip((transformed + 0.15) / 1.15, 0, 1)


def display_rgb(cube: np.ndarray) -> np.ndarray:
    channels = {band: display_plane(cube[i]) for i, band in enumerate(BANDS)}
    return np.stack([channels["z"], channels["r"], channels["g"]], axis=-1)


def signed_residual(plane: np.ndarray, noise: float) -> np.ndarray:
    value = np.arcsinh(plane / max(3 * noise, 1e-30)) / np.arcsinh(5.0)
    return np.clip(value, -1, 1)


def support_weight(mask: np.ndarray, psf_pixels: float) -> tuple[np.ndarray, np.ndarray]:
    if not mask.any():
        return np.zeros(mask.shape), np.zeros(mask.shape, dtype=bool)
    iterations = max(2, int(math.ceil(psf_pixels)))
    support = ndimage.binary_dilation(mask, iterations=iterations)
    distance_inside = ndimage.distance_transform_edt(support)
    weight = np.clip(distance_inside / 2.0, 0, 1)
    return weight, support


def option_arrays(record: SourceRecord) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    obs_mask = central_mask(record.observed_detection)
    model_mask = central_mask(record.model_detection)
    common_mask = obs_mask | model_mask
    weight, support = support_weight(common_mask, float(np.max(record.psf_fwhm_pixels)))
    model_weight, model_support = support_weight(model_mask, float(np.max(record.psf_fwhm_pixels)))
    option_a = record.observed_subtracted * weight
    option_b = record.model_subtracted * model_weight
    option_c = (record.residual_subtracted + option_b) * weight
    return {"A": option_a, "B": option_b, "C": option_c}, {
        "support": support,
        "weight": weight,
        "model_support": model_support,
        "obs_mask": obs_mask,
        "model_mask": model_mask,
    }


def color_from_flux(flux1: float, flux2: float) -> float:
    return -2.5 * math.log10(flux1 / flux2) if flux1 > 0 and flux2 > 0 else float("nan")


def boundary_artifact(array: np.ndarray, support: np.ndarray) -> float:
    if not support.any():
        return float("nan")
    boundary = support ^ ndimage.binary_erosion(support)
    inner = ndimage.binary_erosion(support, iterations=2)
    gradient = np.sqrt(
        ndimage.sobel(array, axis=1, mode="nearest") ** 2
        + ndimage.sobel(array, axis=0, mode="nearest") ** 2
    )
    boundary_value = float(np.mean(gradient[:, boundary])) if boundary.any() else float("nan")
    inner_value = float(np.mean(gradient[:, inner])) if inner.any() else float("nan")
    return boundary_value / inner_value if np.isfinite(inner_value) and inner_value > 0 else float("nan")


def extraction_rows(record: SourceRecord, isolation_reliable: bool) -> list[dict[str, Any]]:
    options, metadata = option_arrays(record)
    support = metadata["support"]
    outer_frame = np.zeros(support.shape, dtype=bool)
    outer_frame[:8] = True
    outer_frame[-8:] = True
    outer_frame[:, :8] = True
    outer_frame[:, -8:] = True
    reference_flux = np.sum(record.observed_subtracted[:, support], axis=1) if support.any() else np.full(3, np.nan)
    reference_gr = color_from_flux(float(reference_flux[0]), float(reference_flux[1]))
    reference_rz = color_from_flux(float(reference_flux[1]), float(reference_flux[2]))
    yy, xx = np.nonzero(support)
    margin = float(min(np.min(xx), np.min(yy), 255 - np.max(xx), 255 - np.max(yy))) if len(xx) else float("nan")
    rows: list[dict[str, Any]] = []
    descriptions = {
        "A": ("segmented observed source", "contains source/coadd noise and residual background"),
        "B": ("segmented Tractor model source", "low-noise parametric scene-model estimate"),
        "C": ("model-assisted observed source", "central model estimate plus observed-minus-model residual"),
    }
    for option, cube in options.items():
        flux = np.sum(cube, axis=(1, 2))
        gr = color_from_flux(float(flux[0]), float(flux[1]))
        rz = color_from_flux(float(flux[1]), float(flux[2]))
        correlations = [pearson(cube[i][support], record.observed_subtracted[i][support]) if support.any() else float("nan") for i in range(3)]
        gradient_correlations: list[float] = []
        for band_index in range(3):
            grad_option = np.hypot(
                ndimage.sobel(cube[band_index], axis=1), ndimage.sobel(cube[band_index], axis=0)
            )
            grad_observed = np.hypot(
                ndimage.sobel(record.observed_subtracted[band_index], axis=1),
                ndimage.sobel(record.observed_subtracted[band_index], axis=0),
            )
            gradient_correlations.append(
                pearson(grad_option[support], grad_observed[support]) if support.any() else float("nan")
            )
        abs_total = float(np.sum(np.abs(cube)))
        border_leakage = float(np.sum(np.abs(cube[:, outer_frame])) / abs_total) if abs_total > 0 else float("nan")
        off_core = support & ~metadata["obs_mask"]
        background_leakage = (
            float(np.median([robust_scale(cube[i][off_core]) / record.noises[i] for i in range(3)]))
            if off_core.any()
            else float("nan")
        )
        second_noise = option in {"A", "C"}
        suitability = (
            option == "B"
            and isolation_reliable
            and np.isfinite(margin)
            and margin >= 8
            and border_leakage == 0
        )
        rows.append(
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "option": option,
                "option_name": descriptions[option][0],
                "array_semantics": descriptions[option][1],
                "flux_g": float(flux[0]),
                "flux_r": float(flux[1]),
                "flux_z": float(flux[2]),
                "flux_preservation_g": float(flux[0] / reference_flux[0]) if reference_flux[0] != 0 else float("nan"),
                "flux_preservation_r": float(flux[1] / reference_flux[1]) if reference_flux[1] != 0 else float("nan"),
                "flux_preservation_z": float(flux[2] / reference_flux[2]) if reference_flux[2] != 0 else float("nan"),
                "color_g_minus_r": gr,
                "color_r_minus_z": rz,
                "color_error_g_minus_r": gr - reference_gr if np.isfinite(gr) and np.isfinite(reference_gr) else float("nan"),
                "color_error_r_minus_z": rz - reference_rz if np.isfinite(rz) and np.isfinite(reference_rz) else float("nan"),
                "morphology_spatial_correlation_mean": finite_mean(correlations),
                "morphology_gradient_correlation_mean": finite_mean(gradient_correlations),
                "background_leakage_robust_sigma_ratio": background_leakage,
                "edge_artifact_gradient_ratio": boundary_artifact(cube, support),
                "rectangular_cutout_border_leakage_fraction": border_leakage,
                "support_min_frame_margin_pixels": margin,
                "contains_coadd_noise_realization": bool_int(second_noise),
                "would_add_second_noise_if_used_as_contaminant": bool_int(second_noise),
                "native_source_psf_retained": 1,
                "subpixel_shift_suitability_without_further_validation": bool_int(suitability),
                "central_model_isolation_reliable": bool_int(isolation_reliable),
                "option_suitable_for_single_noise_contract": bool_int(suitability),
            }
        )
    return rows


def psf_rows(record: SourceRecord) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    within_source_range = float(np.max(record.psf_fwhm_arcsec) - np.min(record.psf_fwhm_arcsec))
    for index, band in enumerate(BANDS):
        rows.append(
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "band": band,
                "psf_fwhm_arcsec": float(record.psf_fwhm_arcsec[index]),
                "psf_fwhm_pixels": float(record.psf_fwhm_pixels[index]),
                "within_source_cross_band_range_arcsec": within_source_range,
                "catalog_match_separation_arcsec": record.catalog_central_separation_arcsec,
                "value_source": "official ls-dr10-south catalog box PSFSIZE column",
                "official_product_documentation": OFFICIAL_PSF_URL,
                "direct_addition_psf_consistent_without_pairing_or_convolution": 0,
            }
        )
    return rows


def figure_triplet(record: SourceRecord, out_path: Path, closure_valid: bool, isolation_reliable: bool) -> None:
    figure, axes = plt.subplots(4, 5, figsize=(19, 14), constrained_layout=True)
    obs_mask = central_mask(record.observed_detection)
    model_mask = central_mask(record.model_detection)
    cubes = [record.observed_subtracted, record.model_subtracted, record.residual_subtracted]
    titles = ["Observed", "Tractor model", "Residual (image-model)"]
    for row, (cube, title) in enumerate(zip(cubes, titles, strict=True)):
        for band_index, band in enumerate(BANDS):
            axes[row, band_index].imshow(display_plane(cube[band_index]), origin="lower", cmap="gray", vmin=0, vmax=1)
            axes[row, band_index].set_title(f"{title} {band}")
        axes[row, 3].imshow(display_rgb(cube), origin="lower")
        axes[row, 3].set_title(f"{title} RGB (display only)")
        axes[row, 4].imshow(display_rgb(cube), origin="lower")
        mask = obs_mask if row != 1 else model_mask
        if mask.any():
            axes[row, 4].contour(mask.astype(float), levels=[0.5], colors=["#00ffff"], linewidths=0.8)
        axes[row, 4].plot(record.catalog_target_x, record.catalog_target_y, "+", color="yellow", ms=10, mew=1.5)
        axes[row, 4].set_title("central mask + DR10 catalog position")
    for band_index, band in enumerate(BANDS):
        axes[3, band_index].imshow(
            signed_residual(record.residual_subtracted[band_index], record.noises[band_index]),
            origin="lower",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axes[3, band_index].set_title(f"Robust signed residual {band} (±3σ scale)")
    ax = axes[3, 3]
    yy, xx = np.indices(obs_mask.shape)
    radius = np.hypot(xx - record.catalog_target_x, yy - record.catalog_target_y)
    for band_index, band in enumerate(BANDS):
        for product, cube, style in (
            ("obs", record.observed_subtracted, "-"),
            ("model", record.model_subtracted, "--"),
            ("resid", record.residual_subtracted, ":"),
        ):
            means = [np.mean(cube[band_index][(radius >= r) & (radius < r + 1)]) for r in range(40)]
            ax.plot(np.arange(40) * PIXEL_SCALE, means, style, label=f"{product}-{band}", alpha=0.8)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_yscale("symlog", linthresh=max(float(np.min(record.noises)), 1e-6))
    ax.set_xlabel("radius (arcsec)")
    ax.set_ylabel("mean nanomaggies/pixel")
    ax.set_title("Radial profiles")
    ax.legend(fontsize=6, ncol=3)
    axes[3, 4].axis("off")
    axes[3, 4].text(
        0,
        1,
        "\n".join(
            [
                f"source: {record.source['source_id']}",
                f"requested RA/Dec: {record.source['ra']}, {record.source['dec']}",
                f"catalog match: {record.catalog_central_separation_arcsec:.3f} arcsec",
                f"PSF FWHM g/r/z: {record.psf_fwhm_arcsec[0]:.3f} / {record.psf_fwhm_arcsec[1]:.3f} / {record.psf_fwhm_arcsec[2]:.3f} arcsec",
                f"closure valid (all bands): {closure_valid}",
                f"central-only model isolation reliable: {isolation_reliable}",
                "All RGB/stretches are display-only.",
                "The Tractor model is not treated as truth.",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
    )
    for axis in axes.ravel():
        axis.set_xticks([]) if axis is not ax else None
        axis.set_yticks([]) if axis is not ax else None
    figure.suptitle(f"DR10 scene triplet audit — {record.source['source_id']}", fontsize=16)
    save_figure_exclusive(out_path, figure)


def figure_extraction(record: SourceRecord, out_path: Path, isolation_reliable: bool) -> None:
    options, metadata = option_arrays(record)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    names = {
        "A": "A: segmented observed",
        "B": "B: segmented Tractor model",
        "C": "C: model-assisted observed",
    }
    for column, option in enumerate(("A", "B", "C")):
        axes[0, column].imshow(display_rgb(options[option]), origin="lower")
        axes[0, column].set_title(names[option])
        axes[1, column].imshow(display_rgb(options[option]), origin="lower")
        axes[1, column].contour(metadata["support"].astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
        axes[1, column].set_title("support boundary")
    axes[0, 3].imshow(display_rgb(record.observed_subtracted), origin="lower")
    axes[0, 3].set_title("Observed reference (display only)")
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0,
        1,
        "\n".join(
            [
                f"source: {record.source['source_id']}",
                f"central-only model isolation reliable: {isolation_reliable}",
                "A and C contain a coadd-noise realization.",
                "B is low-noise but parametric and already PSF-convolved.",
                "No panel is a ground-truth source image.",
            ]
        ),
        va="top",
        fontsize=11,
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Source-extraction comparison — {record.source['source_id']}", fontsize=15)
    save_figure_exclusive(out_path, figure)


def figure_review_montage(records: list[SourceRecord], path: Path) -> None:
    figure, axes = plt.subplots(len(records), 5, figsize=(15, 3 * len(records)), constrained_layout=True)
    if len(records) == 1:
        axes = axes[np.newaxis, :]
    for row, record in enumerate(records):
        mask = central_mask(record.observed_detection)
        axes[row, 0].imshow(display_rgb(record.observed_subtracted), origin="lower")
        axes[row, 0].set_ylabel(record.source["source_id"], fontsize=9)
        axes[row, 0].set_title("Observed RGB" if row == 0 else "")
        axes[row, 1].imshow(display_rgb(record.model_subtracted), origin="lower")
        axes[row, 1].set_title("Model RGB" if row == 0 else "")
        axes[row, 2].imshow(
            signed_residual(record.residual_subtracted[1], record.noises[1]),
            origin="lower",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axes[row, 2].set_title("Signed residual r" if row == 0 else "")
        axes[row, 3].imshow(display_rgb(record.observed_subtracted), origin="lower")
        if mask.any():
            axes[row, 3].contour(mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
        axes[row, 3].set_title("Observed + mask" if row == 0 else "")
        difference = record.observed_subtracted - record.model_subtracted
        axes[row, 4].imshow(display_rgb(difference), origin="lower")
        axes[row, 4].set_title("Observed − model" if row == 0 else "")
        for column in range(5):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle("Manual morphology review montage (display-only stretches)", fontsize=15)
    save_figure_exclusive(path, figure, dpi=150)


def psf_summary(psf_rows_all: list[dict[str, Any]]) -> dict[str, Any]:
    by_band = {band: np.asarray([row["psf_fwhm_arcsec"] for row in psf_rows_all if row["band"] == band]) for band in BANDS}
    source_vectors: dict[str, np.ndarray] = {}
    for source_id in {row["source_id"] for row in psf_rows_all}:
        source_vectors[source_id] = np.asarray(
            [next(row["psf_fwhm_arcsec"] for row in psf_rows_all if row["source_id"] == source_id and row["band"] == band) for band in BANDS]
        )
    vectors = list(source_vectors.values())
    compatible_pairs = 0
    total_pairs = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            total_pairs += 1
            if float(np.max(np.abs(vectors[i] - vectors[j]))) <= 0.1:
                compatible_pairs += 1
    summary: dict[str, Any] = {
        "pairing_tolerance_arcsec_all_bands": 0.1,
        "compatible_source_pairs": compatible_pairs,
        "total_source_pairs": total_pairs,
        "compatible_pair_fraction": compatible_pairs / total_pairs,
        "common_gaussian_target_fwhm_arcsec": float(max(np.max(values) for values in by_band.values()) + 0.1),
    }
    for band, values in by_band.items():
        summary[f"{band}_minimum_arcsec"] = float(np.min(values))
        summary[f"{band}_maximum_arcsec"] = float(np.max(values))
        summary[f"{band}_mean_arcsec"] = float(np.mean(values))
        summary[f"{band}_std_arcsec"] = float(np.std(values, ddof=1))
    return summary


def markdown_reports(
    run_dir: Path,
    alignment_rows: list[dict[str, Any]],
    closure: list[dict[str, Any]],
    morphology: list[dict[str, Any]],
    components: list[dict[str, Any]],
    extraction: list[dict[str, Any]],
    psf: list[dict[str, Any]],
    psf_stats: dict[str, Any],
    figure_dir: Path,
) -> dict[str, str]:
    aligned_count = sum(row["alignment_pass"] for row in alignment_rows)
    closure_sources = {
        source_id: all(row["closure_valid"] for row in closure if row["source_id"] == source_id)
        for source_id in {row["source_id"] for row in closure}
    }
    closure_count = sum(closure_sources.values())
    reliable_count = sum(row["central_only_model_isolation_reliable"] for row in components)
    audit = f"""# DR10 scene-triplet alignment and additivity audit

Generated: `{utc_now()}`  
Execution: CPU-only Astropy/NumPy/SEP; no training or blend-manifest generation.

## Result

- Pixel alignment: **{aligned_count}/20 pass**. Shape, `grz` order, CRPIX/CRVAL,
  canonical CD/PC matrix, pixel scale, and a full 256×256 WCS grid agree.
- Additivity: **{closure_count}/20 sources pass all three bands**; {20 - closure_count}
  fail the numerical closure gate.
- Viewer cutout `BUNIT` is absent for all products. The common numerical unit is
  resolved from the [official DR10 image-stack documentation]({OFFICIAL_UNIT_URL})
  as `{UNIT_TEXT}`. Product identity is taken from the preserved request URL,
  because non-WCS SURVEY/VERSION cards differ between observed and model/residual.

## Closure rule

`closure = image - model - residual` is evaluated in float64 on jointly finite
pixels, per band. A band passes only when finite coverage is 1.0, closure RMSE
is at most `{CLOSURE_NOISE_RATIO_MAX}` of the residual robust background scale,
and L1 closure is at most `{CLOSURE_L1_RELATIVE_MAX}` of observed L1 flux. Both
signed total-flux closure and L1 normalization are retained because a
sky-subtracted signed total can approach zero.

Alignment proves common pixel coordinates; closure tests service additivity.
Neither test establishes that the Tractor prediction is astrophysical truth.

## Tables and figures

- `tables/scene_triplet_alignment.csv`
- `tables/scene_triplet_closure.csv`
- Individual scene sheets: `{figure_dir}/scene_triplet_*.png`
"""
    morphology_measured = sum(row.get("metric_status") == "measured" for row in morphology)
    morph_report = f"""# Morphology-in-residual audit

Generated: `{utc_now()}`

Per-band metrics were measured for `{morphology_measured}` of 60 source-band
records using an observed-derived central segmentation associated with the
nearest official DR10 catalog component. The table reports signed residual
flux, residual/observed fraction, spatial and gradient correlations, positive
and negative fractions, 180-degree asymmetry, and raw/noise-subtracted core and
halo power. Core and halo are defined by the observed detection-image r50 and
r90 radii inside the central mask.

These scalars support—but do not assign—the required manual morphology class.
In particular, residual/observed correlation is not independent because the
observed image algebraically contains the residual.

- Metrics: `tables/residual_morphology_metrics.csv`
- Profiles: `tables/radial_profiles.csv`
- Manual-review montages: `{figure_dir}/manual_review_montage_page*.png`
"""
    component_report = f"""# DR10 scene-component audit

Generated: `{utc_now()}`

Observed and model components use the same observed-noise-normalized SEP
detection (`{DETECTION_SIGMA} sigma`, minarea `{MINAREA}`, deblend thresholds
`{DEBLEND_NTHRESH}`/`{DEBLEND_CONT}`). The official catalog boxes independently
record Tractor component identities, morphological types, and reference-catalog
flags. `GE`/`T2` with G<13 is counted as an official bright-star candidate and
`L3` as an SGA large-galaxy component.

Only **{reliable_count}/20** summed model cutouts meet every conservative
central-only isolation condition. Pixel segmentation of a summed scene model
is not a true per-source decomposition; overlapping analytic profiles remain
inseparable without official per-source forward rendering. The whole model
cutout must never be used as one contaminant stamp.

- Table: `tables/scene_component_audit.csv`
"""
    psf_report = f"""# DR10 PSF audit

Generated: `{utc_now()}`

The friendly foundation catalog and viewer-cutout headers contain no usable
PSF FWHM. Official DR10 catalog boxes provide `PSFSIZE_G/R/Z`, documented as
weighted-average FWHM in arcseconds; the corresponding exact local product is
`south/coadd/<AAA>/<brick>/legacysurvey-<brick>-psfsize-<band>.fits.fz` in the
[official DR10 file tree]({OFFICIAL_PSF_URL}).

Ranges are g `{psf_stats['g_minimum_arcsec']:.3f}–{psf_stats['g_maximum_arcsec']:.3f}`,
r `{psf_stats['r_minimum_arcsec']:.3f}–{psf_stats['r_maximum_arcsec']:.3f}`,
and z `{psf_stats['z_minimum_arcsec']:.3f}–{psf_stats['z_maximum_arcsec']:.3f}`
arcsec. Only {psf_stats['compatible_source_pairs']}/{psf_stats['total_source_pairs']}
source pairs match within 0.1 arcsec in all bands.

Direct source addition would generally mix PSFs. Pairing is sparse; Gaussian
broadening to a common FWHM is computationally feasible but scalar FWHM does
not validate wings or ellipticity. Forward rendering intrinsic Tractor
profiles at the target PSF is preferable, but cannot be done from an already
convolved model cutout. No deconvolution was implemented.

- Table: `tables/psf_audit.csv`
"""
    option_counts = {
        option: sum(row["option_suitable_for_single_noise_contract"] for row in extraction if row["option"] == option)
        for option in ("A", "B", "C")
    }
    extraction_report = f"""# Source-extraction option audit

Generated: `{utc_now()}`

- A (segmented observed) preserves observed structure but carries source/coadd
  noise and residual background. As a contaminant it creates a second local
  noise realization. Suitable count: {option_counts['A']}/20.
- B (segmented Tractor model) is low-noise but parametric, already
  PSF-convolved, and only conditionally separable from neighbors. Suitable
  count under all pixel-isolation checks: {option_counts['B']}/20.
- C (central model estimate plus residual) preserves model-missed observed
  structure, but the residual also carries coadd noise and neighbor/sky fit
  errors. As a contaminant it creates a second noise realization. Suitable
  count: {option_counts['C']}/20.

The decision is based on noise accounting, component isolation, flux/color and
morphology metrics, boundaries, and PSF semantics—not visual attractiveness.

- Metrics/decision table: `tables/source_extraction_options.csv`
- Comparative sheets: `{figure_dir}/source_extraction_*.png`
"""
    return {
        "scene_triplet_audit.md": audit,
        "morphology_residual_audit.md": morph_report,
        "scene_component_audit.md": component_report,
        "psf_audit.md": psf_report,
        "source_extraction_options.md": extraction_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "lockbox" in str(args.run_dir).lower() or "sealed" in str(args.run_dir).lower():
        raise SystemExit("refusing lockbox/sealed path")
    sources, triplet_manifests = load_products(args.run_dir)
    catalog_manifests = load_catalog_manifest(args.run_dir)
    figure_dir = args.run_dir / "figures" / f"scene_probe_{args.analysis_id}"
    figure_dir.mkdir(parents=True, exist_ok=False)

    alignment: list[dict[str, Any]] = []
    closure: list[dict[str, Any]] = []
    morphology: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    extraction: list[dict[str, Any]] = []
    psf: list[dict[str, Any]] = []
    radial: list[dict[str, Any]] = []
    loaded_records: list[SourceRecord] = []

    for rank, source in enumerate(sources, start=1):
        source_id = source["source_id"]
        products = {
            name: open_product(Path(row["relative_path"]))
            for name, row in triplet_manifests[source_id].items()
        }
        alignment_record = alignment_row(source, products)
        alignment.append(alignment_record)
        if not alignment_record["alignment_pass"]:
            raise RuntimeError(f"fail-closed pixel alignment failure for {source_id}")
        closure_for_source = closure_rows(source, products)
        closure.extend(closure_for_source)
        closure_valid = all(row["closure_valid"] for row in closure_for_source)
        record = load_source_record(
            source,
            triplet_manifests[source_id],
            catalog_manifests[source_id],
        )
        loaded_records.append(record)
        component = scene_component_row(record, closure_valid)
        components.append(component)
        isolation_reliable = bool(component["central_only_model_isolation_reliable"])
        morphology.extend(morphology_rows(record))
        extraction.extend(extraction_rows(record, isolation_reliable))
        psf.extend(psf_rows(record))
        radial.extend(radial_profile_rows(record))
        figure_triplet(
            record,
            figure_dir / f"scene_triplet_{rank:02d}_{source_id}.png",
            closure_valid,
            isolation_reliable,
        )
        figure_extraction(
            record,
            figure_dir / f"source_extraction_{rank:02d}_{source_id}.png",
            isolation_reliable,
        )
        print(f"[{rank:02d}/20] audited {source_id}", flush=True)

    for page, start in enumerate(range(0, 20, 5), start=1):
        figure_review_montage(
            loaded_records[start : start + 5],
            figure_dir / f"manual_review_montage_page{page:02d}.png",
        )

    tables = {
        "scene_triplet_alignment.csv": alignment,
        "scene_triplet_closure.csv": closure,
        "residual_morphology_metrics.csv": morphology,
        "scene_component_audit.csv": components,
        "psf_audit.csv": psf,
        "source_extraction_options.csv": extraction,
        "radial_profiles.csv": radial,
    }
    for filename, rows in tables.items():
        write_csv_exclusive(args.run_dir / "tables" / filename, rows)
    psf_stats = psf_summary(psf)
    write_text_exclusive(
        args.run_dir / "diagnostics" / "psf_summary.json",
        json.dumps(psf_stats, indent=2, sort_keys=True) + "\n",
    )
    reports = markdown_reports(
        args.run_dir,
        alignment,
        closure,
        morphology,
        components,
        extraction,
        psf,
        psf_stats,
        figure_dir,
    )
    for filename, report in reports.items():
        write_text_exclusive(args.run_dir / "diagnostics" / filename, report)

    protocol = {
        "generated_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_id": args.analysis_id,
        "parameters": {
            "bands": BANDS,
            "pixel_scale_arcsec": PIXEL_SCALE,
            "detection_sigma": DETECTION_SIGMA,
            "minarea": MINAREA,
            "deblend_nthresh": DEBLEND_NTHRESH,
            "deblend_cont": DEBLEND_CONT,
            "closure_noise_ratio_max": CLOSURE_NOISE_RATIO_MAX,
            "closure_l1_relative_max": CLOSURE_L1_RELATIVE_MAX,
        },
        "input_manifest_sha256": {
            "triplets": sha256_file(args.run_dir / "manifests" / "scene_triplet_download_manifest.csv"),
            "official_catalogs": sha256_file(args.run_dir / "manifests" / "official_catalog_download_manifest.csv"),
            "sources": sha256_file(args.run_dir / "manifests" / "engineering_sources_20.csv"),
        },
        "output_sha256": {
            filename: sha256_file(args.run_dir / "tables" / filename) for filename in tables
        },
        "figure_count": len(list(figure_dir.glob("*.png"))),
        "git_branch": subprocess.run(["git", "branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
    }
    write_text_exclusive(
        args.run_dir / "logs" / "scene_probe_analysis_protocol.json",
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
    )
    summary = {
        "alignment_pass_sources": sum(row["alignment_pass"] for row in alignment),
        "closure_pass_sources": sum(
            all(row["closure_valid"] for row in closure if row["source_id"] == source["source_id"])
            for source in sources
        ),
        "central_isolation_reliable_sources": sum(
            row["central_only_model_isolation_reliable"] for row in components
        ),
        "figure_count": protocol["figure_count"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
