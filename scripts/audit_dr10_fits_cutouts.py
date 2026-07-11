#!/usr/bin/env python3
"""Audit DR10 g,r,z FITS cutouts and classify source isolation quality.

This command is deliberately append-only.  It reads cutouts on CPU, never
changes their pixels, refuses lockbox-like inputs, and creates every table,
report, and figure with exclusive-create semantics.  Source detection is a
transparent SEP heuristic intended to build a clean pilot library; it is not a
claim of astrophysical source type or calibrated photometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sep
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


AUDIT_VERSION = "dr10_fits_audit_v3_complete_candidates"
ISOLATION_RULE_VERSION = "dr10_source_isolation_v3_all_neighbor_scan"
FITS_SUFFIXES = (".fits", ".fit", ".fits.fz")
OFFICIAL_IMAGE_UNIT = "nanomaggies per pixel"
SUCCESSFUL_DOWNLOAD_STATUSES = {"downloaded_valid", "resume_validated_skip"}
TERMINAL_REJECTION_STATUSES = {
    "content_rejected",
    "http_client_error",
    "validation_rejected",
}
INCOMPLETE_DOWNLOAD_STATUSES = {
    "cancelled",
    "circuit_open",
    "dry_run",
    "existing_file_refused",
    "failed",
}


@dataclass(frozen=True)
class AuditSettings:
    expected_bands: tuple[str, ...] = ("g", "r", "z")
    expected_size: int = 256
    expected_layer: str = "ls-dr10-south"
    expected_survey: str = "DECaLS"
    expected_version: str = "DR10-south"
    expected_pixel_scale_arcsec: float = 0.262
    pixel_scale_rtol: float = 5e-4
    pixel_scale_atol: float = 1e-6
    minimum_per_band_finite_fraction: float = 0.99
    minimum_per_band_nonzero_fraction: float = 0.01
    expected_documented_unit: str = OFFICIAL_IMAGE_UNIT
    maximum_request_center_offset_px: float = 0.05
    detection_sigma: float = 2.5
    minarea: int = 8
    deblend_nthresh: int = 32
    deblend_cont: float = 0.005
    central_max_offset_px: float = 12.0
    neighbor_review_distance_px: float = 32.0
    neighbor_review_flux_ratio: float = 0.10
    clear_blend_distance_px: float = 12.0
    clear_blend_flux_ratio: float = 0.50
    full_frame_area_fraction: float = 0.50
    extreme_sigma: float = 20.0
    # Real source pixels are expected to be many sky-MADs above the background;
    # only a very large frame fraction is an artifact-review trigger.
    extreme_fraction_review: float = 0.10
    reject_finite_fraction: float = 0.95
    review_finite_fraction: float = 0.999
    compact_a_max_px: float = 2.2
    compact_axis_ratio_min: float = 0.75
    compact_peak_fraction_min: float = 0.15
    contact_sheet_size: int = 16
    max_contact_sheet_files: int = 64
    histogram_samples_per_file: int = 512
    histogram_max_samples_per_band: int = 200_000


@dataclass
class FitsPayload:
    cube_native_order: np.ndarray
    cube_grz_order: np.ndarray
    header: fits.Header
    bands_header: str
    inferred_bands: tuple[str, ...]
    band_axis_original: int
    hdu_layout_json: str
    wcs: WCS | None
    wcs_valid: bool
    wcs_warnings: str
    pixel_scale_arcsec: float
    pixel_scales_arcsec: tuple[float, float]
    center_ra: float
    center_dec: float
    survey: str
    version: str
    per_band_finite_fraction: tuple[float, ...]
    per_band_nonzero_fraction: tuple[float, ...]
    checksum_keyword_present: bool
    datasum_keyword_present: bool


@dataclass
class VisualRecord:
    path: str
    source_id: str
    rgb: np.ndarray | None
    detection_image: np.ndarray | None
    object_x: np.ndarray
    object_y: np.ndarray
    central_index: int | None
    requested_x: float
    requested_y: float
    decision_hint: str
    error: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bands(value: str) -> tuple[str, ...]:
    tokens = [item.lower() for item in re.findall(r"[A-Za-z]+", value)]
    if len(tokens) == 1 and len(tokens[0]) > 1:
        return tuple(tokens[0])
    return tuple(tokens)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_pixel_hash(array: np.ndarray) -> str:
    """Hash shape, dtype, and exact C-order decoded pixel bytes."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def native_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def bool_int(value: Any) -> int:
    return int(bool(value))


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_csv_exclusive(
    path: Path, rows: Sequence[dict[str, Any]], preferred_fields: Sequence[str]
) -> None:
    fields = list(preferred_fields)
    seen = set(fields)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def save_figure_exclusive(fig: plt.Figure, path: Path, dpi: int = 160) -> None:
    with path.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def is_fits_path(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in FITS_SUFFIXES)


def discover_inputs(paths: Sequence[Path], recursive: bool, limit: int | None) -> list[Path]:
    found: dict[str, Path] = {}
    for raw in paths:
        path = raw.expanduser().resolve()
        if path.is_file():
            if not is_fits_path(path):
                raise ValueError(f"Input is not FITS-like: {path}")
            found[str(path)] = path
        elif path.is_dir():
            iterator: Iterable[Path] = path.rglob("*") if recursive else path.iterdir()
            for candidate in iterator:
                if candidate.is_file() and is_fits_path(candidate):
                    resolved = candidate.resolve()
                    found[str(resolved)] = resolved
        else:
            raise FileNotFoundError(path)
    result = [found[key] for key in sorted(found)]
    if limit is not None:
        result = result[:limit]
    if not result:
        raise ValueError("No FITS files were discovered")
    return result


def reject_lockbox_inputs(paths: Sequence[Path], manifest_rows: Iterable[dict[str, str]]) -> None:
    for path in paths:
        if "lockbox" in str(path).lower():
            raise ValueError(f"Refusing lockbox-like path: {path}")
    for row in manifest_rows:
        for key, value in row.items():
            if key.lower() in {"split", "partition", "role", "split_role"} and (
                "lockbox" in str(value).lower()
            ):
                raise ValueError("Refusing manifest row assigned to a lockbox")


def read_manifest(path: Path | None) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[dict[str, str]]]:
    if path is None:
        return {}, {}, []
    resolved_manifest = path.expanduser().resolve()
    with resolved_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_path: dict[str, dict[str, str]] = {}
    basename_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        relative = row.get("relative_path") or row.get("path") or row.get("fits_path")
        if relative:
            candidate = Path(relative)
            if not candidate.is_absolute():
                candidate = resolved_manifest.parent / candidate
            by_path[str(candidate.resolve())] = row
            basename_rows.setdefault(Path(relative).name, []).append(row)
    by_basename = {
        name: candidates[-1]
        for name, candidates in basename_rows.items()
        if len({json.dumps(item, sort_keys=True) for item in candidates}) == 1
    }
    return by_path, by_basename, rows


def read_source_manifest(path: Path) -> list[dict[str, str]]:
    resolved = path.expanduser().resolve()
    with resolved.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"source manifest is empty: {resolved}")
    return rows


def _source_group_id(row: dict[str, str]) -> str:
    return str(row.get("group_id") or row.get("provisional_group_id") or "").strip()


def validate_candidate_download_coverage(
    source_rows: Sequence[dict[str, str]],
    manifest_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    """Require one completed latest downloader outcome for every candidate.

    Append-only download manifests can contain several historical attempts. The
    last row for each source is authoritative, and every authoritative row must
    belong to the same completed invocation. Terminal semantic/HTTP failures
    are returned for explicit source-library rejection; cancelled, circuit-open,
    dry-run, or otherwise incomplete outcomes abort the production audit.
    """
    candidates: dict[str, dict[str, str]] = {}
    candidate_catalog_rows: set[str] = set()
    for row in source_rows:
        source_id = str(row.get("source_id", "")).strip()
        catalog_row = str(row.get("catalog_row_index", "")).strip()
        group_id = _source_group_id(row)
        if not source_id or not catalog_row or not group_id:
            raise ValueError(
                "source manifest rows require source_id, catalog_row_index, and "
                "group_id/provisional_group_id"
            )
        if source_id in candidates:
            raise ValueError(f"duplicate source_id in source manifest: {source_id}")
        if catalog_row in candidate_catalog_rows:
            raise ValueError(
                f"duplicate catalog_row_index in source manifest: {catalog_row}"
            )
        ra = safe_float(row.get("ra"))
        dec = safe_float(row.get("dec"))
        if not (np.isfinite(ra) and np.isfinite(dec)):
            raise ValueError(f"source manifest has nonfinite coordinate: {source_id}")
        candidates[source_id] = row
        candidate_catalog_rows.add(catalog_row)

    latest: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            raise ValueError("download manifest row lacks source_id")
        latest[source_id] = row

    candidate_ids = set(candidates)
    latest_ids = set(latest)
    if candidate_ids != latest_ids:
        missing = sorted(candidate_ids - latest_ids)[:5]
        extra = sorted(latest_ids - candidate_ids)[:5]
        raise ValueError(
            "candidate/download source sets differ: "
            f"missing_count={len(candidate_ids - latest_ids)} examples={missing}; "
            f"extra_count={len(latest_ids - candidate_ids)} examples={extra}"
        )

    run_ids = {str(row.get("run_id", "")).strip() for row in latest.values()}
    if "" in run_ids or len(run_ids) != 1:
        raise ValueError(
            "latest candidate outcomes do not belong to one completed downloader "
            f"invocation: run_ids={sorted(run_ids)}"
        )
    run_id = next(iter(run_ids))

    terminal_rows: list[dict[str, str]] = []
    for source_id, candidate in candidates.items():
        outcome = latest[source_id]
        comparisons = {
            "catalog_row_index": (
                str(candidate.get("catalog_row_index", "")).strip(),
                str(outcome.get("catalog_row_index", "")).strip(),
            ),
            "group_id": (
                _source_group_id(candidate),
                str(outcome.get("group_id", "")).strip(),
            ),
        }
        for label, (expected, actual) in comparisons.items():
            if not expected or expected != actual:
                raise ValueError(
                    f"candidate/download {label} mismatch for {source_id}: "
                    f"{expected!r} != {actual!r}"
                )
        for coordinate in ("ra", "dec"):
            expected = safe_float(candidate.get(coordinate))
            actual = safe_float(outcome.get(coordinate))
            if not (
                np.isfinite(expected)
                and np.isfinite(actual)
                and np.isclose(expected, actual, rtol=0.0, atol=1e-8)
            ):
                raise ValueError(
                    f"candidate/download {coordinate} mismatch for {source_id}: "
                    f"{expected!r} != {actual!r}"
                )

        status = str(outcome.get("status", "")).strip()
        if status in SUCCESSFUL_DOWNLOAD_STATUSES:
            continue
        if status in TERMINAL_REJECTION_STATUSES:
            if not str(outcome.get("error", "")).strip():
                raise ValueError(
                    f"terminal downloader outcome lacks explicit error: {source_id}"
                )
            terminal_rows.append(outcome)
            continue
        if status in INCOMPLETE_DOWNLOAD_STATUSES:
            raise ValueError(
                f"candidate has incomplete downloader outcome {status!r}: {source_id}"
            )
        raise ValueError(
            f"candidate has unknown downloader outcome {status!r}: {source_id}"
        )
    return terminal_rows, run_id


def terminal_rejection_decision(
    row: dict[str, str], manifest_parent: Path
) -> dict[str, Any]:
    status = str(row.get("status", "")).strip()
    error = str(row.get("error", "")).strip()
    relative = str(row.get("relative_path", "")).strip()
    intended_path = Path(relative) if relative else Path("missing_download_path")
    if not intended_path.is_absolute():
        intended_path = manifest_parent / intended_path
    reason = f"download_terminal_{status}"
    return {
        "path": str(intended_path.resolve()),
        "filename": Path(relative).name if relative else "",
        "source_id": str(row.get("source_id", "")).strip(),
        "group_id": str(row.get("group_id", "")).strip(),
        "catalog_row_index": str(row.get("catalog_row_index", "")).strip(),
        "decision": "rejected_for_source_library_use",
        "primary_reason": reason,
        "all_reasons": reason,
        "rejection_reasons": reason,
        "manual_review_reasons": "",
        "rule_version": ISOLATION_RULE_VERSION,
        "exact_duplicate_group_id": "",
        "download_status": status,
        "download_error": error,
        "download_run_id": str(row.get("run_id", "")).strip(),
    }


def metadata_for_path(
    path: Path,
    by_path: dict[str, dict[str, str]],
    by_basename: dict[str, dict[str, str]],
) -> dict[str, str]:
    return dict(by_path.get(str(path.resolve())) or by_basename.get(path.name) or {})


def validate_manifest_alignment(
    inputs: Sequence[Path],
    by_path: dict[str, dict[str, str]],
    by_basename: dict[str, dict[str, str]],
    manifest_rows: Sequence[dict[str, str]],
    *,
    manifest_parent: Path,
    require_all_successful_rows: bool,
) -> None:
    """Fail closed unless every audited FITS has one successful hashed record."""
    input_paths = {str(path.resolve()) for path in inputs}
    for path in inputs:
        row = metadata_for_path(path, by_path, by_basename)
        if not row:
            raise ValueError(f"FITS has no download-manifest row: {path}")
        if row.get("status") not in SUCCESSFUL_DOWNLOAD_STATUSES:
            raise ValueError(
                f"FITS latest manifest status is not validated success: "
                f"{path} status={row.get('status')!r}"
            )
        expected_hash = str(row.get("sha256", "")).strip().lower()
        if len(expected_hash) != 64:
            raise ValueError(f"FITS manifest SHA-256 is missing or malformed: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"FITS/manifest SHA-256 mismatch for {path}: "
                f"manifest={expected_hash} actual={actual_hash}"
            )

    if not require_all_successful_rows:
        return
    successful_paths: set[str] = set()
    for row in manifest_rows:
        if row.get("status") not in SUCCESSFUL_DOWNLOAD_STATUSES:
            continue
        relative = row.get("relative_path") or row.get("path") or row.get("fits_path")
        if not relative:
            raise ValueError("successful manifest row has no FITS path")
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = manifest_parent / candidate
        successful_paths.add(str(candidate.resolve()))
    missing_files = successful_paths - input_paths
    if missing_files:
        preview = sorted(missing_files)[:5]
        raise ValueError(
            f"{len(missing_files)} successful manifest FITS are absent from audit inputs; "
            f"examples={preview}"
        )


def normalized_unit(value: str) -> str:
    return " ".join(value.strip().lower().split())


def require_documented_unit(value: str, settings: AuditSettings) -> str:
    canonical = normalized_unit(value)
    expected = normalized_unit(settings.expected_documented_unit)
    if canonical != expected:
        raise ValueError(
            "production FITS audit requires explicit documented unit "
            f"{settings.expected_documented_unit!r}; received {value!r}"
        )
    return settings.expected_documented_unit


def validate_manifest_request_semantics(
    inputs: Sequence[Path],
    by_path: dict[str, dict[str, str]],
    by_basename: dict[str, dict[str, str]],
    settings: AuditSettings,
) -> None:
    """Require source identity and frozen request semantics for every FITS."""
    expected_bands = "".join(settings.expected_bands)
    expected_shape = f"{len(settings.expected_bands)}x{settings.expected_size}x{settings.expected_size}"
    seen_source_ids: set[str] = set()
    seen_catalog_rows: set[str] = set()
    for path in inputs:
        row = metadata_for_path(path, by_path, by_basename)
        required = ("source_id", "catalog_row_index", "ra", "dec", "group_id")
        missing = [name for name in required if not str(row.get(name, "")).strip()]
        if missing:
            raise ValueError(f"manifest metadata missing {missing} for {path}")
        source_id = str(row["source_id"]).strip()
        catalog_row = str(row["catalog_row_index"]).strip()
        if source_id in seen_source_ids:
            raise ValueError(f"manifest source_id is repeated in audit inputs: {source_id}")
        if catalog_row in seen_catalog_rows:
            raise ValueError(
                f"manifest catalog_row_index is repeated in audit inputs: {catalog_row}"
            )
        seen_source_ids.add(source_id)
        seen_catalog_rows.add(catalog_row)

        try:
            parameters = json.loads(row.get("request_parameters_json", ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid request_parameters_json for {path}: {exc}") from exc
        if not isinstance(parameters, dict):
            raise ValueError(f"request parameters are not an object for {path}")
        if parameters.get("layer") != settings.expected_layer:
            raise ValueError(
                f"request layer mismatch for {path}: {parameters.get('layer')!r}"
            )
        if str(parameters.get("bands", "")).lower() != expected_bands:
            raise ValueError(
                f"request bands mismatch for {path}: {parameters.get('bands')!r}"
            )
        try:
            request_size = int(parameters["size"])
            request_scale = float(parameters["pixscale"])
            request_ra = float(parameters["ra"])
            request_dec = float(parameters["dec"])
            manifest_ra = float(row["ra"])
            manifest_dec = float(row["dec"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"non-numeric frozen request metadata for {path}: {exc}") from exc
        if request_size != settings.expected_size:
            raise ValueError(
                f"request size mismatch for {path}: {request_size} != {settings.expected_size}"
            )
        if not np.isclose(
            request_scale,
            settings.expected_pixel_scale_arcsec,
            rtol=settings.pixel_scale_rtol,
            atol=settings.pixel_scale_atol,
        ):
            raise ValueError(
                f"request pixel scale mismatch for {path}: {request_scale}"
            )
        if not (
            np.isclose(request_ra, manifest_ra, rtol=0.0, atol=1e-8)
            and np.isclose(request_dec, manifest_dec, rtol=0.0, atol=1e-8)
        ):
            raise ValueError(
                f"request and manifest coordinates disagree for {path}: "
                f"request=({request_ra},{request_dec}) manifest=({manifest_ra},{manifest_dec})"
            )
        if str(row.get("fits_shape", "")).lower() != expected_shape:
            raise ValueError(
                f"manifest FITS shape mismatch for {path}: "
                f"{row.get('fits_shape')!r} != {expected_shape!r}"
            )
        if str(row.get("bands_header", "")).lower() != expected_bands:
            raise ValueError(
                f"manifest BANDS mismatch for {path}: {row.get('bands_header')!r}"
            )


def output_paths_for_run(run_dir: Path) -> dict[str, Path]:
    run_dir = run_dir.expanduser().resolve()
    return {
        "quality": run_dir / "tables/fits_quality_metrics.csv",
        "bands": run_dir / "tables/fits_band_statistics.csv",
        "artifacts": run_dir / "tables/fits_artifact_candidates.csv",
        "duplicates": run_dir / "tables/exact_duplicate_groups.csv",
        "isolation": run_dir / "tables/source_isolation_metrics.csv",
        "decisions": run_dir / "tables/source_quality_decisions.csv",
        "fits_report": run_dir / "diagnostics/fits_data_audit.md",
        "isolation_report": run_dir / "diagnostics/source_isolation_protocol.md",
        "fits_figures": run_dir / "figures/fits_contact_sheets",
        "band_figures": run_dir / "figures/band_distribution_plots",
        "isolation_figures": run_dir / "figures/isolation_contact_sheets",
    }


def preflight_output_paths(run_dir: Path) -> dict[str, Path]:
    """Check final destinations without creating anything."""
    outputs = output_paths_for_run(run_dir)
    for key, path in outputs.items():
        if key.endswith("figures"):
            if path.exists():
                raise FileExistsError(f"Refusing existing figure directory: {path}")
        elif path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    return outputs


def prepare_output_paths(run_dir: Path) -> dict[str, Path]:
    """Create final destinations only after the complete CPU audit succeeds."""
    outputs = preflight_output_paths(run_dir)
    resolved = run_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for child in ("tables", "diagnostics", "figures"):
        (resolved / child).mkdir(exist_ok=True)
    for key in ("fits_figures", "band_figures", "isolation_figures"):
        outputs[key].mkdir(exist_ok=False)
    return outputs


def deterministic_sample_indices(total: int, maximum: int) -> set[int]:
    """Evenly sample zero-based indices with a strict deterministic bound."""
    if total < 0 or maximum < 0:
        raise ValueError("sample sizes cannot be negative")
    count = min(total, maximum)
    if count == 0:
        return set()
    if count == 1:
        return {0}
    indices = (np.arange(count, dtype=np.int64) * (total - 1)) // (count - 1)
    result = {int(value) for value in indices}
    if len(result) != count:
        raise RuntimeError("deterministic sample index construction produced duplicates")
    return result


def retention_plan(
    total: int, settings: AuditSettings
) -> tuple[set[int], set[int]]:
    visual_indices = deterministic_sample_indices(total, settings.max_contact_sheet_files)
    histogram_file_cap = max(
        1,
        settings.histogram_max_samples_per_band
        // settings.histogram_samples_per_file,
    )
    histogram_indices = deterministic_sample_indices(total, histogram_file_cap)
    return visual_indices, histogram_indices


def _hdu_layout(hdul: fits.HDUList) -> str:
    rows = []
    for index, hdu in enumerate(hdul):
        data = hdu.data
        rows.append(
            {
                "index": index,
                "name": hdu.name,
                "class": type(hdu).__name__,
                "shape": list(data.shape) if data is not None else None,
                "dtype": str(data.dtype) if data is not None else None,
                "bunit": str(hdu.header.get("BUNIT", "")),
            }
        )
    return json.dumps(rows, sort_keys=True)


def _bands_from_header(header: fits.Header, n_planes: int) -> tuple[str, ...]:
    value = str(header.get("BANDS", "")).strip()
    if value:
        parsed = parse_bands(value)
        if len(parsed) == n_planes:
            return parsed
    indexed = tuple(
        str(header.get(f"BAND{index}", "")).strip().lower()
        for index in range(n_planes)
    )
    if all(indexed):
        return indexed
    return ()


def read_fits_payload(
    path: Path,
    expected_bands: tuple[str, ...],
    *,
    settings: AuditSettings | None = None,
) -> FitsPayload:
    settings = settings or AuditSettings(expected_bands=expected_bands)
    if settings.expected_bands != expected_bands:
        raise ValueError("settings expected_bands disagrees with function argument")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fits.open(path, mode="readonly", memmap=False, checksum=True) as hdul:
            hdul.verify("exception")
            layout = _hdu_layout(hdul)
            image_hdus = [hdu for hdu in hdul if hdu.data is not None]
            if not image_hdus:
                raise ValueError("no image HDU")

            header: fits.Header
            band_axis = 0
            if len(image_hdus) == 1 and np.asarray(image_hdus[0].data).ndim == 3:
                image = np.asarray(image_hdus[0].data)
                matching_axes = [
                    axis for axis, size in enumerate(image.shape) if size == len(expected_bands)
                ]
                if not matching_axes:
                    raise ValueError(
                        f"no axis has expected band count {len(expected_bands)}; shape={image.shape}"
                    )
                band_axis = 0 if 0 in matching_axes else matching_axes[0]
                cube = np.moveaxis(image, band_axis, 0)
                header = image_hdus[0].header.copy()
                inferred = _bands_from_header(header, cube.shape[0])
                bands_header = str(header.get("BANDS", "")).strip()
            elif all(np.asarray(hdu.data).ndim == 2 for hdu in image_hdus):
                if len(image_hdus) != len(expected_bands):
                    raise ValueError(
                        f"found {len(image_hdus)} two-dimensional image HDUs; "
                        f"expected {len(expected_bands)}"
                    )
                shapes = {tuple(np.asarray(hdu.data).shape) for hdu in image_hdus}
                if len(shapes) != 1:
                    raise ValueError(f"2-D image HDU shapes differ: {sorted(shapes)}")
                cube = np.stack([np.asarray(hdu.data) for hdu in image_hdus], axis=0)
                inferred = tuple(hdu.name.strip().lower() for hdu in image_hdus)
                header = image_hdus[0].header.copy()
                bands_header = "".join(inferred)
                band_axis = 0
            else:
                shapes = [tuple(np.asarray(hdu.data).shape) for hdu in image_hdus]
                raise ValueError(f"unsupported mixed image HDU layout: {shapes}")

            expected_shape = (
                len(expected_bands),
                settings.expected_size,
                settings.expected_size,
            )
            if tuple(cube.shape) != expected_shape:
                raise ValueError(
                    f"unexpected frozen cutout shape {tuple(cube.shape)}; "
                    f"expected {expected_shape}"
                )
            if band_axis != 0:
                raise ValueError(
                    f"noncanonical frozen band axis {band_axis}; expected axis 0"
                )
            if not inferred:
                raise ValueError("band order is absent from BANDS/BANDi/EXTNAME metadata")
            bands_keyword = parse_bands(str(header.get("BANDS", "")))
            indexed_bands = tuple(
                str(header.get(f"BAND{index}", "")).strip().lower()
                for index in range(len(expected_bands))
            )
            if bands_keyword != expected_bands or indexed_bands != expected_bands:
                raise ValueError(
                    "frozen band metadata mismatch: "
                    f"BANDS={bands_keyword!r}, BAND0..={indexed_bands!r}, "
                    f"expected={expected_bands!r}"
                )
            if inferred != expected_bands:
                raise ValueError(
                    f"noncanonical native band order {inferred!r}; expected {expected_bands!r}"
                )
            canonical = np.ascontiguousarray(cube)

            survey = str(header.get("SURVEY", "")).strip()
            version = str(header.get("VERSION", "")).strip()
            if survey != settings.expected_survey or version != settings.expected_version:
                raise ValueError(
                    "frozen provenance mismatch: "
                    f"SURVEY={survey!r}, VERSION={version!r}; expected "
                    f"{settings.expected_survey!r}, {settings.expected_version!r}"
                )
            bunit = str(header.get("BUNIT", "")).strip()
            if bunit and normalized_unit(bunit) != normalized_unit(
                settings.expected_documented_unit
            ):
                raise ValueError(
                    f"FITS BUNIT={bunit!r} conflicts with documented unit "
                    f"{settings.expected_documented_unit!r}"
                )

            required_wcs = (
                "CTYPE1",
                "CTYPE2",
                "CRVAL1",
                "CRVAL2",
                "CRPIX1",
                "CRPIX2",
                "CD1_1",
                "CD1_2",
                "CD2_1",
                "CD2_2",
            )
            missing_wcs = [key for key in required_wcs if key not in header]
            if missing_wcs:
                raise ValueError(
                    f"frozen celestial WCS keywords are missing: {missing_wcs}"
                )

            wcs_obj: WCS | None = None
            wcs_valid = False
            pixel_scale = float("nan")
            pixel_scales = (float("nan"), float("nan"))
            center_ra = float("nan")
            center_dec = float("nan")
            try:
                wcs_obj = WCS(header).celestial
                if wcs_obj.pixel_n_dim == 2 and wcs_obj.world_n_dim == 2:
                    x_center = (canonical.shape[2] - 1) / 2.0
                    y_center = (canonical.shape[1] - 1) / 2.0
                    center_ra, center_dec = map(
                        float, wcs_obj.pixel_to_world_values(x_center, y_center)
                    )
                    scales = np.asarray(proj_plane_pixel_scales(wcs_obj), dtype=float) * 3600.0
                    if scales.shape == (2,) and np.all(np.isfinite(scales)) and np.all(scales > 0):
                        pixel_scales = (float(scales[0]), float(scales[1]))
                        pixel_scale = float(np.mean(scales))
                        wcs_valid = bool(np.isfinite(center_ra) and np.isfinite(center_dec))
            except Exception as exc:
                caught.append(
                    warnings.WarningMessage(
                        message=RuntimeWarning(f"WCS error: {type(exc).__name__}: {exc}"),
                        category=RuntimeWarning,
                        filename=str(path),
                        lineno=0,
                    )
                )

            if not wcs_valid:
                raise ValueError("missing or invalid celestial WCS")
            if not all(
                np.isclose(
                    scale,
                    settings.expected_pixel_scale_arcsec,
                    rtol=settings.pixel_scale_rtol,
                    atol=settings.pixel_scale_atol,
                )
                for scale in pixel_scales
            ):
                raise ValueError(
                    f"pixel scales {pixel_scales!r} arcsec do not match frozen "
                    f"{settings.expected_pixel_scale_arcsec}"
                )

            per_band_finite = tuple(
                float(np.isfinite(plane).mean()) for plane in canonical
            )
            per_band_nonzero = tuple(
                float(np.mean(np.isfinite(plane) & (plane != 0)))
                for plane in canonical
            )
            if any(
                value < settings.minimum_per_band_finite_fraction
                for value in per_band_finite
            ):
                raise ValueError(
                    "per-band finite fractions violate frozen minimum "
                    f"{settings.minimum_per_band_finite_fraction}: {per_band_finite}"
                )
            if any(
                value < settings.minimum_per_band_nonzero_fraction
                for value in per_band_nonzero
            ):
                raise ValueError(
                    "per-band nonzero fractions violate frozen minimum "
                    f"{settings.minimum_per_band_nonzero_fraction}: {per_band_nonzero}"
                )

            warning_text = " | ".join(str(item.message) for item in caught)
            return FitsPayload(
                cube_native_order=np.asarray(cube),
                cube_grz_order=np.asarray(canonical),
                header=header,
                bands_header=bands_header,
                inferred_bands=inferred,
                band_axis_original=band_axis,
                hdu_layout_json=layout,
                wcs=wcs_obj,
                wcs_valid=wcs_valid,
                wcs_warnings=warning_text,
                pixel_scale_arcsec=pixel_scale,
                pixel_scales_arcsec=pixel_scales,
                center_ra=center_ra,
                center_dec=center_dec,
                survey=survey,
                version=version,
                per_band_finite_fraction=per_band_finite,
                per_band_nonzero_fraction=per_band_nonzero,
                checksum_keyword_present="CHECKSUM" in header,
                datasum_keyword_present="DATASUM" in header,
            )


def robust_statistics(plane: np.ndarray, extreme_sigma: float) -> dict[str, Any]:
    total = int(plane.size)
    finite_mask = np.isfinite(plane)
    finite = np.asarray(plane[finite_mask], dtype=np.float64)
    result: dict[str, Any] = {
        "total_pixels": total,
        "finite_pixels": int(finite.size),
        "finite_fraction": float(finite.size / total),
        "nan_fraction": float(np.isnan(plane).sum() / total),
        "posinf_fraction": float(np.isposinf(plane).sum() / total),
        "neginf_fraction": float(np.isneginf(plane).sum() / total),
    }
    if not finite.size:
        result.update(
            min="",
            max="",
            median="",
            robust_scale="",
            negative_fraction="",
            zero_fraction="",
            extreme_fraction="",
            background_like_fraction="",
            p001="",
            p01="",
            p99="",
            p999="",
        )
        return result
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_scale = 1.4826 * mad
    if robust_scale > 0 and np.isfinite(robust_scale):
        deviations = np.abs(finite - median)
        extreme_fraction = float(np.mean(deviations > extreme_sigma * robust_scale))
        background_like_fraction = float(np.mean(deviations <= robust_scale))
    else:
        extreme_fraction = float(np.mean(finite != median))
        background_like_fraction = float(np.mean(finite == median))
    percentiles = np.percentile(finite, [0.1, 1.0, 99.0, 99.9])
    result.update(
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        median=median,
        robust_scale=robust_scale,
        negative_fraction=float(np.mean(finite < 0)),
        zero_fraction=float(np.mean(finite == 0)),
        extreme_fraction=extreme_fraction,
        background_like_fraction=background_like_fraction,
        p001=float(percentiles[0]),
        p01=float(percentiles[1]),
        p99=float(percentiles[2]),
        p999=float(percentiles[3]),
    )
    return result


def requested_pixel(
    payload: FitsPayload,
    metadata: dict[str, str],
    *,
    require_manifest_coordinate: bool = False,
) -> tuple[float, float, str, float, float]:
    height, width = payload.cube_grz_order.shape[1:]
    fallback_x = (width - 1) / 2.0
    fallback_y = (height - 1) / 2.0
    ra = safe_float(metadata.get("ra"))
    dec = safe_float(metadata.get("dec"))
    if payload.wcs_valid and payload.wcs is not None and np.isfinite(ra) and np.isfinite(dec):
        try:
            x, y = map(float, payload.wcs.world_to_pixel_values(ra, dec))
            if np.isfinite(x) and np.isfinite(y):
                return x, y, "manifest_world_coordinate", ra, dec
        except Exception:
            pass
    if require_manifest_coordinate:
        raise ValueError(
            "validated manifest RA/Dec could not be transformed through the FITS WCS"
        )
    return fallback_x, fallback_y, "geometric_cutout_center", payload.center_ra, payload.center_dec


def detection_and_isolation(
    cube: np.ndarray,
    requested_x: float,
    requested_y: float,
    settings: AuditSettings,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int | None, np.ndarray]:
    n_bands, height, width = cube.shape
    standardized: list[np.ndarray] = []
    background_subtracted: list[np.ndarray] = []
    invalid_mask = ~np.all(np.isfinite(cube), axis=0)
    per_band_noise: list[float] = []
    for plane in cube:
        finite = plane[np.isfinite(plane)]
        fill = float(np.median(finite)) if finite.size else 0.0
        clean = native_float32(np.where(np.isfinite(plane), plane, fill))
        box = max(8, min(32, height // 4, width // 4))
        background = sep.Background(clean, bw=box, bh=box, fw=3, fh=3)
        subtracted = native_float32(clean - background.back())
        rms_map = np.asarray(background.rms(), dtype=np.float64)
        rms_values = rms_map[np.isfinite(rms_map) & (rms_map > 0)]
        noise = float(np.median(rms_values)) if rms_values.size else 0.0
        if not np.isfinite(noise) or noise <= 0:
            finite_sub = subtracted[np.isfinite(subtracted)]
            mad = float(np.median(np.abs(finite_sub - np.median(finite_sub)))) if finite_sub.size else 0.0
            noise = 1.4826 * mad
        if not np.isfinite(noise) or noise <= 0:
            noise = 1.0
        background_subtracted.append(subtracted)
        standardized.append(native_float32(subtracted / noise))
        per_band_noise.append(noise)

    detection = native_float32(np.sum(standardized, axis=0) / math.sqrt(n_bands))
    objects, segmentation = sep.extract(
        detection,
        settings.detection_sigma,
        mask=invalid_mask,
        minarea=settings.minarea,
        deblend_nthresh=settings.deblend_nthresh,
        deblend_cont=settings.deblend_cont,
        clean=True,
        segmentation_map=True,
    )
    n_objects = int(len(objects))
    central_index: int | None = None
    nearest_to_prompt = float("nan")
    in_frame = 0 <= requested_x < width and 0 <= requested_y < height
    if n_objects and in_frame:
        distances = np.hypot(objects["x"] - requested_x, objects["y"] - requested_y)
        nearest_candidate = int(np.argmin(distances))
        nearest_to_prompt = float(distances[nearest_candidate])
        ix = int(np.clip(round(requested_x), 0, width - 1))
        iy = int(np.clip(round(requested_y), 0, height - 1))
        label_at_prompt = int(segmentation[iy, ix])
        if 1 <= label_at_prompt <= n_objects:
            central_index = label_at_prompt - 1
        elif nearest_to_prompt <= settings.central_max_offset_px:
            central_index = nearest_candidate

    labels_on_border = np.unique(
        np.concatenate(
            [segmentation[0], segmentation[-1], segmentation[:, 0], segmentation[:, -1]]
        )
    )
    labels_on_border = labels_on_border[labels_on_border > 0]
    edge_source_count = int(labels_on_border.size)
    full_frame = False
    largest_area_fraction = 0.0
    for label in range(1, n_objects + 1):
        object_mask = segmentation == label
        fraction = float(object_mask.mean())
        largest_area_fraction = max(largest_area_fraction, fraction)
        touches = (
            bool(object_mask[0].any()),
            bool(object_mask[-1].any()),
            bool(object_mask[:, 0].any()),
            bool(object_mask[:, -1].any()),
        )
        if fraction >= settings.full_frame_area_fraction or all(touches):
            full_frame = True

    result: dict[str, Any] = {
        "detected_source_count": n_objects,
        "requested_x": requested_x,
        "requested_y": requested_y,
        "requested_coordinate_in_frame": bool_int(in_frame),
        "central_source_present": bool_int(central_index is not None),
        "central_object_index": central_index if central_index is not None else "",
        "central_centroid_x": "",
        "central_centroid_y": "",
        "central_centroid_offset_px": "",
        "central_area_pixels": "",
        "central_area_fraction": "",
        "central_a_px": "",
        "central_b_px": "",
        "central_axis_ratio": "",
        "central_peak_fraction": "",
        "nearest_neighbor_distance_px": "",
        "neighbor_to_target_detection_flux_ratio": "",
        "central_mask_touches_border": 0,
        "edge_touching_source_count": edge_source_count,
        "any_source_mask_touches_border": bool_int(edge_source_count > 0),
        "largest_source_area_fraction": largest_area_fraction,
        "full_frame_object": bool_int(full_frame),
        "likely_stellar_candidate": 0,
        "likely_stellar_neighbor_contamination": 0,
        "likely_preexisting_blend": 0,
        "clear_preexisting_blend": 0,
        "qualifying_review_neighbor_count": 0,
        "qualifying_clear_blend_neighbor_count": 0,
        "max_neighbor_detection_flux_ratio_within_review_radius": "",
        "closest_clear_blend_neighbor_distance_px": "",
        "blank_cutout": bool_int(n_objects == 0 and float(np.max(detection)) < settings.detection_sigma),
        "detection_peak_sigma": float(np.max(detection)),
        "nearest_detection_to_prompt_px": nearest_to_prompt,
        "sep_background_rms_by_band_json": json.dumps(per_band_noise),
    }
    for band in settings.expected_bands:
        result[f"central_flux_{band}"] = ""
        result[f"neighbor_flux_{band}"] = ""
        result[f"neighbor_to_target_flux_ratio_{band}"] = ""

    if central_index is not None:
        label = central_index + 1
        central_mask = segmentation == label
        x = float(objects[central_index]["x"])
        y = float(objects[central_index]["y"])
        offset = float(math.hypot(x - requested_x, y - requested_y))
        area = int(central_mask.sum())
        touches_border = bool(
            central_mask[0].any()
            or central_mask[-1].any()
            or central_mask[:, 0].any()
            or central_mask[:, -1].any()
        )
        a = float(objects[central_index]["a"])
        b = float(objects[central_index]["b"])
        axis_ratio = b / a if a > 0 else float("nan")
        central_detection_values = detection[central_mask]
        positive_sum = float(np.sum(np.clip(central_detection_values, 0, None)))
        peak_fraction = (
            float(np.max(central_detection_values)) / positive_sum
            if positive_sum > 0 and central_detection_values.size
            else float("nan")
        )
        result.update(
            central_centroid_x=x,
            central_centroid_y=y,
            central_centroid_offset_px=offset,
            central_area_pixels=area,
            central_area_fraction=float(area / (height * width)),
            central_a_px=a,
            central_b_px=b,
            central_axis_ratio=axis_ratio,
            central_peak_fraction=peak_fraction,
            central_mask_touches_border=bool_int(touches_border),
            likely_stellar_candidate=bool_int(
                a <= settings.compact_a_max_px
                and np.isfinite(axis_ratio)
                and axis_ratio >= settings.compact_axis_ratio_min
                and np.isfinite(peak_fraction)
                and peak_fraction >= settings.compact_peak_fraction_min
            ),
        )
        central_fluxes = []
        for band, plane in zip(settings.expected_bands, background_subtracted, strict=True):
            flux = float(np.sum(np.asarray(plane, dtype=np.float64)[central_mask]))
            central_fluxes.append(flux)
            result[f"central_flux_{band}"] = flux

        other_indices = [index for index in range(n_objects) if index != central_index]
        if other_indices:
            distances = np.asarray(
                [
                    math.hypot(
                        float(objects[index]["x"]) - x,
                        float(objects[index]["y"]) - y,
                    )
                    for index in other_indices
                ],
                dtype=np.float64,
            )
            neighbor_index = other_indices[int(np.argmin(distances))]
            neighbor_distance = float(np.min(distances))
            neighbor_mask = segmentation == (neighbor_index + 1)
            central_detection_flux = abs(float(objects[central_index]["flux"]))
            detection_ratios = np.asarray(
                [
                    (
                        abs(float(objects[index]["flux"]))
                        / central_detection_flux
                        if central_detection_flux > 0
                        else float("inf")
                    )
                    for index in other_indices
                ],
                dtype=np.float64,
            )
            detection_ratio = float(
                detection_ratios[other_indices.index(neighbor_index)]
            )
            result["nearest_neighbor_distance_px"] = neighbor_distance
            result["neighbor_to_target_detection_flux_ratio"] = detection_ratio
            for band_index, (band, plane) in enumerate(
                zip(settings.expected_bands, background_subtracted, strict=True)
            ):
                neighbor_flux = float(np.sum(np.asarray(plane, dtype=np.float64)[neighbor_mask]))
                target_flux = central_fluxes[band_index]
                ratio = abs(neighbor_flux) / abs(target_flux) if target_flux != 0 else float("inf")
                result[f"neighbor_flux_{band}"] = neighbor_flux
                result[f"neighbor_to_target_flux_ratio_{band}"] = ratio

            review_mask = (
                (distances <= settings.neighbor_review_distance_px)
                & (detection_ratios >= settings.neighbor_review_flux_ratio)
            )
            clear_mask = (
                (distances <= settings.clear_blend_distance_px)
                & (detection_ratios >= settings.clear_blend_flux_ratio)
            )
            result["likely_preexisting_blend"] = bool_int(bool(review_mask.any()))
            result["clear_preexisting_blend"] = bool_int(bool(clear_mask.any()))
            result["qualifying_review_neighbor_count"] = int(review_mask.sum())
            result["qualifying_clear_blend_neighbor_count"] = int(clear_mask.sum())
            within_review_radius = distances <= settings.neighbor_review_distance_px
            if within_review_radius.any():
                result["max_neighbor_detection_flux_ratio_within_review_radius"] = float(
                    np.max(detection_ratios[within_review_radius])
                )
            if clear_mask.any():
                result["closest_clear_blend_neighbor_distance_px"] = float(
                    np.min(distances[clear_mask])
                )

            stellar_neighbor = False
            for position, object_index in enumerate(other_indices):
                if not review_mask[position]:
                    continue
                neighbor_a = float(objects[object_index]["a"])
                neighbor_b = float(objects[object_index]["b"])
                neighbor_axis_ratio = (
                    neighbor_b / neighbor_a if neighbor_a > 0 else float("nan")
                )
                object_mask = segmentation == (object_index + 1)
                values = detection[object_mask]
                positive = float(np.sum(np.clip(values, 0, None)))
                neighbor_peak_fraction = (
                    float(np.max(values)) / positive
                    if positive > 0 and values.size
                    else float("nan")
                )
                if (
                    neighbor_a <= settings.compact_a_max_px
                    and np.isfinite(neighbor_axis_ratio)
                    and neighbor_axis_ratio >= settings.compact_axis_ratio_min
                    and np.isfinite(neighbor_peak_fraction)
                    and neighbor_peak_fraction >= settings.compact_peak_fraction_min
                ):
                    stellar_neighbor = True
                    break
            result["likely_stellar_neighbor_contamination"] = bool_int(
                stellar_neighbor
            )

    return result, detection, objects, central_index, segmentation


def display_channel(plane: np.ndarray) -> np.ndarray:
    finite = plane[np.isfinite(plane)]
    if not finite.size:
        return np.zeros(plane.shape, dtype=np.float32)
    median = float(np.median(finite))
    scale = float(np.percentile(np.abs(finite - median), 99.5))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    value = np.arcsinh((np.where(np.isfinite(plane), plane, median) - median) / (0.1 * scale))
    value /= np.arcsinh(10.0)
    return np.clip((value + 0.15) / 1.15, 0.0, 1.0).astype(np.float32)


def display_rgb(cube: np.ndarray, bands: tuple[str, ...]) -> np.ndarray:
    lookup = {band: display_channel(plane) for band, plane in zip(bands, cube, strict=True)}
    # Display-only mapping: z->R, r->G, g->B.  It never enters a scientific table.
    return np.stack([lookup["z"], lookup["r"], lookup["g"]], axis=-1)


def audit_one(
    path: Path,
    metadata: dict[str, str],
    settings: AuditSettings,
    documented_unit: str,
    *,
    retain_visual: bool = True,
    retain_histogram_samples: bool = True,
    strict_source_association: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], VisualRecord, list[np.ndarray]]:
    stat = path.stat()
    source_id = metadata.get("source_id") or metadata.get("dr8_id") or path.stem
    group_id = metadata.get("group_id") or metadata.get("provisional_group_id") or ""
    base = {
        "path": str(path),
        "filename": path.name,
        "source_id": source_id,
        "group_id": group_id,
        "catalog_row_index": metadata.get("catalog_row_index", ""),
        "file_size_bytes": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
        "file_sha256": file_sha256(path),
        "audit_version": AUDIT_VERSION,
    }
    quality: dict[str, Any] = {
        **base,
        "fits_open_valid": 0,
        "fits_structure_valid": 0,
        "hdu_count": "",
        "hdu_layout_json": "",
        "data_shape": "",
        "expected_data_shape": (
            f"{len(settings.expected_bands)}x{settings.expected_size}x{settings.expected_size}"
        ),
        "data_dtype": "",
        "band_axis_original": "",
        "bands_header": "",
        "inferred_band_order": "",
        "expected_band_order": "".join(settings.expected_bands),
        "band_order_valid": 0,
        "wcs_valid": 0,
        "wcs_center_ra": "",
        "wcs_center_dec": "",
        "pixel_scale_arcsec": "",
        "pixel_scales_arcsec_json": "",
        "expected_pixel_scale_arcsec": settings.expected_pixel_scale_arcsec,
        "wcs_warnings": "",
        "survey_header": "",
        "version_header": "",
        "expected_survey": settings.expected_survey,
        "expected_version": settings.expected_version,
        "per_band_finite_fraction_json": "",
        "per_band_nonzero_fraction_json": "",
        "minimum_per_band_finite_fraction": settings.minimum_per_band_finite_fraction,
        "minimum_per_band_nonzero_fraction": settings.minimum_per_band_nonzero_fraction,
        "frozen_fits_semantics_valid": 0,
        "manifest_request_semantics_valid": bool_int(strict_source_association),
        "bunit_header": "",
        "documented_unit": documented_unit,
        "unit_value": "",
        "unit_source": "unresolved",
        "checksum_keyword_present": 0,
        "datasum_keyword_present": 0,
        "finite_fraction_all_bands": 0.0,
        "nan_inf_fraction_all_bands": 1.0,
        "negative_fraction_all_bands": "",
        "zero_fraction_all_bands": "",
        "max_extreme_fraction": "",
        "pixel_hash": "",
        "central_source_present": 0,
        "central_centroid_offset_px": "",
        "border_truncation": 0,
        "blank_cutout": 0,
        "full_frame_object": 0,
        "missing_band_behavior": "unavailable",
        "possible_artifact": 1,
        "audit_error": "",
    }
    band_rows: list[dict[str, Any]] = []
    empty_isolation = {
        **base,
        "detection_status": "unavailable",
        "detection_error": "",
        "requested_coordinate_source": "",
        "requested_ra": metadata.get("ra", ""),
        "requested_dec": metadata.get("dec", ""),
        "pixel_scale_arcsec": "",
        "detected_source_count": 0,
        "central_source_present": 0,
        "blank_cutout": 0,
        "full_frame_object": 0,
    }
    visual = VisualRecord(
        path=str(path),
        source_id=str(source_id),
        rgb=None,
        detection_image=None,
        object_x=np.array([], dtype=float),
        object_y=np.array([], dtype=float),
        central_index=None,
        requested_x=float("nan"),
        requested_y=float("nan"),
        decision_hint="rejected_for_source_library_use",
    )
    samples: list[np.ndarray] = [np.array([], dtype=np.float32) for _ in settings.expected_bands]

    try:
        payload = read_fits_payload(
            path, settings.expected_bands, settings=settings
        )
        quality.update(
            fits_open_valid=1,
            fits_structure_valid=1,
            hdu_layout_json=payload.hdu_layout_json,
            data_shape="x".join(map(str, payload.cube_native_order.shape)),
            data_dtype=str(payload.cube_native_order.dtype),
            band_axis_original=payload.band_axis_original,
            bands_header=payload.bands_header,
            inferred_band_order="".join(payload.inferred_bands),
            band_order_valid=bool_int(payload.inferred_bands == settings.expected_bands),
            wcs_valid=bool_int(payload.wcs_valid),
            wcs_center_ra=payload.center_ra,
            wcs_center_dec=payload.center_dec,
            pixel_scale_arcsec=payload.pixel_scale_arcsec,
            pixel_scales_arcsec_json=json.dumps(payload.pixel_scales_arcsec),
            wcs_warnings=payload.wcs_warnings,
            survey_header=payload.survey,
            version_header=payload.version,
            per_band_finite_fraction_json=json.dumps(
                payload.per_band_finite_fraction
            ),
            per_band_nonzero_fraction_json=json.dumps(
                payload.per_band_nonzero_fraction
            ),
            frozen_fits_semantics_valid=1,
            bunit_header=str(payload.header.get("BUNIT", "")).strip(),
            checksum_keyword_present=bool_int(payload.checksum_keyword_present),
            datasum_keyword_present=bool_int(payload.datasum_keyword_present),
            pixel_hash=semantic_pixel_hash(payload.cube_native_order),
            missing_band_behavior="all_expected_bands_present",
        )
        quality["hdu_count"] = len(json.loads(payload.hdu_layout_json))
        if quality["bunit_header"]:
            quality["unit_value"] = quality["bunit_header"]
            quality["unit_source"] = "fits_header_BUNIT"
        elif documented_unit:
            quality["unit_value"] = documented_unit
            quality["unit_source"] = "explicit_cli_documentation"

        all_finite = []
        all_negative = []
        all_zero = []
        extreme_fractions = []
        for index, (band, plane) in enumerate(
            zip(settings.expected_bands, payload.cube_grz_order, strict=True)
        ):
            stats = robust_statistics(plane, settings.extreme_sigma)
            row = {
                **base,
                "band": band,
                "band_index_canonical": index,
                "band_index_original": payload.inferred_bands.index(band),
                "pixel_hash": semantic_pixel_hash(plane),
                "frozen_finite_gate_pass": bool_int(
                    payload.per_band_finite_fraction[index]
                    >= settings.minimum_per_band_finite_fraction
                ),
                "frozen_nonzero_gate_pass": bool_int(
                    payload.per_band_nonzero_fraction[index]
                    >= settings.minimum_per_band_nonzero_fraction
                ),
                **stats,
            }
            band_rows.append(row)
            all_finite.append(stats["finite_fraction"])
            if stats["negative_fraction"] != "":
                all_negative.append(float(stats["negative_fraction"]))
                all_zero.append(float(stats["zero_fraction"]))
                extreme_fractions.append(float(stats["extreme_fraction"]))
            finite = np.asarray(plane[np.isfinite(plane)], dtype=np.float32)
            if retain_histogram_samples and finite.size:
                step = max(1, finite.size // settings.histogram_samples_per_file)
                samples[index] = finite[::step][: settings.histogram_samples_per_file]

        total_pixels = payload.cube_grz_order.size
        finite_count = int(np.isfinite(payload.cube_grz_order).sum())
        finite_fraction = float(finite_count / total_pixels)
        finite_values = payload.cube_grz_order[np.isfinite(payload.cube_grz_order)]
        quality.update(
            finite_fraction_all_bands=finite_fraction,
            nan_inf_fraction_all_bands=1.0 - finite_fraction,
            negative_fraction_all_bands=(
                float(np.mean(finite_values < 0)) if finite_values.size else ""
            ),
            zero_fraction_all_bands=(
                float(np.mean(finite_values == 0)) if finite_values.size else ""
            ),
            max_extreme_fraction=max(extreme_fractions) if extreme_fractions else "",
        )

        request_x, request_y, coordinate_source, request_ra, request_dec = requested_pixel(
            payload,
            metadata,
            require_manifest_coordinate=strict_source_association,
        )
        frame_center_x = (payload.cube_grz_order.shape[2] - 1) / 2.0
        frame_center_y = (payload.cube_grz_order.shape[1] - 1) / 2.0
        request_center_offset = float(
            math.hypot(request_x - frame_center_x, request_y - frame_center_y)
        )
        if (
            strict_source_association
            and request_center_offset > settings.maximum_request_center_offset_px
        ):
            raise ValueError(
                "manifest requested coordinate is not at the frozen cutout center: "
                f"offset={request_center_offset:.9g} pixels"
            )
        isolation, detection, objects, central_index, _segmentation = detection_and_isolation(
            payload.cube_grz_order, request_x, request_y, settings
        )
        isolation_row = {
            **base,
            "detection_status": "success",
            "detection_error": "",
            "requested_coordinate_source": coordinate_source,
            "requested_ra": request_ra,
            "requested_dec": request_dec,
            "requested_coordinate_center_offset_px": request_center_offset,
            "pixel_scale_arcsec": payload.pixel_scale_arcsec,
            **isolation,
        }
        if isolation.get("central_centroid_offset_px") != "" and np.isfinite(
            payload.pixel_scale_arcsec
        ):
            isolation_row["central_centroid_offset_arcsec"] = float(
                isolation["central_centroid_offset_px"]
            ) * payload.pixel_scale_arcsec
        else:
            isolation_row["central_centroid_offset_arcsec"] = ""

        quality.update(
            central_source_present=isolation["central_source_present"],
            central_centroid_offset_px=isolation["central_centroid_offset_px"],
            border_truncation=isolation["central_mask_touches_border"],
            blank_cutout=isolation["blank_cutout"],
            full_frame_object=isolation["full_frame_object"],
        )
        possible_artifact = bool(
            not payload.wcs_valid
            or quality["unit_source"] == "unresolved"
            or finite_fraction < settings.review_finite_fraction
            or bool(isolation["blank_cutout"])
            or bool(isolation["full_frame_object"])
            or bool(isolation["central_mask_touches_border"])
            or (extreme_fractions and max(extreme_fractions) > settings.extreme_fraction_review)
        )
        quality["possible_artifact"] = bool_int(possible_artifact)
        if retain_visual:
            visual = VisualRecord(
                path=str(path),
                source_id=str(source_id),
                rgb=display_rgb(payload.cube_grz_order, settings.expected_bands),
                detection_image=detection,
                object_x=np.asarray(objects["x"], dtype=float),
                object_y=np.asarray(objects["y"], dtype=float),
                central_index=central_index,
                requested_x=request_x,
                requested_y=request_y,
                decision_hint="pending",
            )
        return quality, band_rows, isolation_row, visual, samples
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        quality["audit_error"] = error
        empty_isolation["detection_error"] = error
        visual.error = error
        for index, band in enumerate(settings.expected_bands):
            band_rows.append(
                {
                    **base,
                    "band": band,
                    "band_index_canonical": index,
                    "audit_status": "unavailable",
                    "audit_error": error,
                }
            )
        return quality, band_rows, empty_isolation, visual, samples


def duplicate_rows(quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in quality_rows:
        pixel_hash = str(row.get("pixel_hash", ""))
        if pixel_hash:
            grouped.setdefault(pixel_hash, []).append(row)
    result: list[dict[str, Any]] = []
    duplicate_number = 0
    for pixel_hash, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        duplicate_number += 1
        group_id = f"exact_pixels_{duplicate_number:06d}_{pixel_hash[:12]}"
        for member in sorted(members, key=lambda item: str(item["path"])):
            member["exact_duplicate_group_id"] = group_id
            result.append(
                {
                    "exact_duplicate_group_id": group_id,
                    "pixel_hash": pixel_hash,
                    "group_size": len(members),
                    "path": member["path"],
                    "source_id": member["source_id"],
                    "group_id": member["group_id"],
                    "file_sha256": member["file_sha256"],
                }
            )
    for row in quality_rows:
        row.setdefault("exact_duplicate_group_id", "")
    return result


def decide_quality(
    quality: dict[str, Any], isolation: dict[str, Any], settings: AuditSettings
) -> dict[str, Any]:
    reject: list[str] = []
    review: list[str] = []
    if not bool(quality.get("frozen_fits_semantics_valid")):
        reject.append("frozen_fits_semantics_not_verified")
    if not bool(quality.get("manifest_request_semantics_valid")):
        reject.append("manifest_request_semantics_not_verified")
    if not bool(quality.get("fits_structure_valid")):
        reject.append("invalid_fits_structure_or_data_layout")
    if not bool(quality.get("band_order_valid")):
        reject.append("unverified_or_noncanonical_grz_band_order")
    if not bool(quality.get("wcs_valid")):
        reject.append("invalid_or_missing_celestial_wcs")
    finite_fraction = safe_float(quality.get("finite_fraction_all_bands"))
    if not np.isfinite(finite_fraction) or finite_fraction < settings.reject_finite_fraction:
        reject.append("finite_fraction_below_reject_threshold")
    elif finite_fraction < settings.review_finite_fraction:
        review.append("finite_fraction_below_review_threshold")
    if bool(isolation.get("blank_cutout")):
        reject.append("blank_cutout")
    if not bool(isolation.get("central_source_present")):
        reject.append("central_source_not_detected")
    if bool(isolation.get("full_frame_object")):
        reject.append("full_frame_object")
    if bool(isolation.get("central_mask_touches_border")):
        reject.append("central_source_mask_touches_border")
    offset = safe_float(isolation.get("central_centroid_offset_px"))
    if np.isfinite(offset) and offset > settings.central_max_offset_px:
        reject.append("central_centroid_offset_above_threshold")
    distance = safe_float(isolation.get("nearest_neighbor_distance_px"))
    ratio = safe_float(isolation.get("neighbor_to_target_detection_flux_ratio"))
    clear_flag_value = isolation.get("clear_preexisting_blend")
    clear_blend = (
        bool(clear_flag_value)
        if clear_flag_value not in (None, "")
        else bool(isolation.get("likely_preexisting_blend"))
        and np.isfinite(distance)
        and np.isfinite(ratio)
        and distance <= settings.clear_blend_distance_px
        and ratio >= settings.clear_blend_flux_ratio
    )
    if clear_blend:
        reject.append("clear_preexisting_blend")
    elif bool(isolation.get("likely_preexisting_blend")):
        review.append("possible_preexisting_blend")
    if bool(isolation.get("likely_stellar_candidate")):
        review.append("compact_psf_like_stellar_candidate")
    if bool(isolation.get("likely_stellar_neighbor_contamination")):
        review.append("compact_psf_like_neighbor_contamination")
    # A peripheral, noncentral segmentation touching the deliberately large
    # 256-pixel frame edge is retained as a diagnostic, not a source-library
    # review trigger. The 100-source engineering set showed this on 91% of
    # frames while no central mask touched a border; it measures added context,
    # not contamination of the requested central object.
    extreme = safe_float(quality.get("max_extreme_fraction"))
    if np.isfinite(extreme) and extreme > settings.extreme_fraction_review:
        review.append("extreme_value_fraction_above_review_threshold")
    if quality.get("unit_source") == "unresolved":
        review.append("flux_unit_unresolved")
    if quality.get("exact_duplicate_group_id"):
        review.append("exact_pixel_duplicate_requires_group_handling")
    if quality.get("audit_error"):
        reject.append("audit_exception")

    reject = list(dict.fromkeys(reject))
    review = [item for item in dict.fromkeys(review) if item not in reject]
    if reject:
        decision = "rejected_for_source_library_use"
        reasons = reject + review
    elif review:
        decision = "manual_review"
        reasons = review
    else:
        decision = "accepted_clean_source"
        reasons = []
    return {
        "path": quality["path"],
        "filename": quality["filename"],
        "source_id": quality["source_id"],
        "group_id": quality["group_id"],
        "catalog_row_index": quality["catalog_row_index"],
        "decision": decision,
        "primary_reason": reasons[0] if reasons else "all_fixed_rules_passed",
        "all_reasons": ";".join(reasons) if reasons else "all_fixed_rules_passed",
        "rejection_reasons": ";".join(reject),
        "manual_review_reasons": ";".join(review),
        "rule_version": ISOLATION_RULE_VERSION,
        "exact_duplicate_group_id": quality.get("exact_duplicate_group_id", ""),
    }


def artifact_rows(
    quality_rows: Sequence[dict[str, Any]],
    isolation_by_path: dict[str, dict[str, Any]],
    settings: AuditSettings,
) -> list[dict[str, Any]]:
    rows = []
    for quality in quality_rows:
        isolation = isolation_by_path[quality["path"]]
        flags = {
            "flag_invalid_fits": not bool(quality.get("fits_structure_valid")),
            "flag_frozen_fits_semantics": not bool(
                quality.get("frozen_fits_semantics_valid")
            ),
            "flag_manifest_request_semantics": not bool(
                quality.get("manifest_request_semantics_valid")
            ),
            "flag_band_semantics": not bool(quality.get("band_order_valid")),
            "flag_invalid_wcs": not bool(quality.get("wcs_valid")),
            "flag_unit_unresolved": quality.get("unit_source") == "unresolved",
            "flag_nonfinite_pixels": safe_float(quality.get("nan_inf_fraction_all_bands")) > 0,
            "flag_extreme_values": (
                safe_float(quality.get("max_extreme_fraction"))
                > settings.extreme_fraction_review
            ),
            "flag_blank_cutout": bool(isolation.get("blank_cutout")),
            "flag_full_frame_object": bool(isolation.get("full_frame_object")),
            "flag_border_truncation": bool(isolation.get("central_mask_touches_border")),
            "flag_preexisting_blend": bool(isolation.get("likely_preexisting_blend")),
            "flag_stellar_candidate": bool(isolation.get("likely_stellar_candidate")),
            "flag_exact_duplicate": bool(quality.get("exact_duplicate_group_id")),
        }
        reasons = [name.removeprefix("flag_") for name, value in flags.items() if value]
        if reasons:
            rows.append(
                {
                    "path": quality["path"],
                    "filename": quality["filename"],
                    "source_id": quality["source_id"],
                    "group_id": quality["group_id"],
                    "catalog_row_index": quality["catalog_row_index"],
                    "artifact_reasons": ";".join(reasons),
                    "exact_duplicate_group_id": quality.get("exact_duplicate_group_id", ""),
                    **{key: bool_int(value) for key, value in flags.items()},
                }
            )
    return rows


def flush_contact_sheet(records: Sequence[VisualRecord], path: Path) -> None:
    rows = cols = int(math.ceil(math.sqrt(len(records))))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), records, strict=False):
        axis.axis("on")
        if record.rgb is None:
            axis.text(0.5, 0.5, "INVALID FITS", ha="center", va="center", transform=axis.transAxes)
            axis.set_facecolor("0.9")
        else:
            axis.imshow(record.rgb, origin="lower", interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{record.source_id}\n{Path(record.path).name[:34]}", fontsize=7)
    fig.suptitle("DR10 display-only signed-asinh z/r/g contact sheet", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    save_figure_exclusive(fig, path)


def flush_isolation_sheet(records: Sequence[VisualRecord], path: Path) -> None:
    rows = cols = int(math.ceil(math.sqrt(len(records))))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), records, strict=False):
        axis.axis("on")
        if record.detection_image is None:
            axis.text(0.5, 0.5, "DETECTION UNAVAILABLE", ha="center", va="center", transform=axis.transAxes)
            axis.set_facecolor("0.9")
        else:
            finite = record.detection_image[np.isfinite(record.detection_image)]
            low, high = (np.percentile(finite, [5, 99.5]) if finite.size else (0.0, 1.0))
            if high <= low:
                high = low + 1.0
            axis.imshow(
                record.detection_image,
                origin="lower",
                cmap="gray",
                vmin=low,
                vmax=high,
                interpolation="nearest",
            )
            if record.object_x.size:
                axis.scatter(record.object_x, record.object_y, s=35, facecolors="none", edgecolors="cyan", linewidths=0.8)
            if record.central_index is not None:
                axis.scatter(
                    [record.object_x[record.central_index]],
                    [record.object_y[record.central_index]],
                    s=70,
                    facecolors="none",
                    edgecolors="lime",
                    linewidths=1.3,
                )
            if np.isfinite(record.requested_x) and np.isfinite(record.requested_y):
                axis.scatter([record.requested_x], [record.requested_y], marker="+", s=65, c="red", linewidths=1.1)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(str(record.source_id), fontsize=7)
    fig.suptitle("SEP detection: cyan=all, green=central, red=requested coordinate", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    save_figure_exclusive(fig, path)


def band_distribution_figures(
    sample_chunks: dict[str, list[np.ndarray]], output_dir: Path, max_samples: int
) -> None:
    for band, chunks in sample_chunks.items():
        values = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        if values.size > max_samples:
            indices = np.linspace(0, values.size - 1, max_samples, dtype=np.int64)
            values = values[indices]
        fig, axis = plt.subplots(figsize=(7, 4.5))
        if values.size:
            lo, hi = np.percentile(values, [0.1, 99.9])
            clipped = values[(values >= lo) & (values <= hi)]
            if clipped.size and hi > lo:
                axis.hist(clipped, bins=120, color={"g": "green", "r": "firebrick", "z": "purple"}.get(band, "steelblue"), alpha=0.75)
                axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
                axis.set_title(f"Band {band}: finite values within global 0.1–99.9 percentiles")
            else:
                axis.text(0.5, 0.5, "Constant finite sample", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.text(0.5, 0.5, "No finite samples", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel("FITS pixel value (not visualization-normalized)")
        axis.set_ylabel("Sample count")
        axis.grid(alpha=0.2)
        save_figure_exclusive(fig, output_dir / f"band_{band}_distribution.png")


def fits_report(
    quality_rows: Sequence[dict[str, Any]],
    band_rows: Sequence[dict[str, Any]],
    artifact_candidates: Sequence[dict[str, Any]],
    duplicate_groups: Sequence[dict[str, Any]],
    settings: AuditSettings,
    documented_unit: str,
) -> str:
    valid = sum(bool(row.get("fits_structure_valid")) for row in quality_rows)
    wcs_valid = sum(bool(row.get("wcs_valid")) for row in quality_rows)
    band_valid = sum(bool(row.get("band_order_valid")) for row in quality_rows)
    blank = sum(bool(row.get("blank_cutout")) for row in quality_rows)
    full_frame = sum(bool(row.get("full_frame_object")) for row in quality_rows)
    unique_duplicate_groups = len({row["exact_duplicate_group_id"] for row in duplicate_groups})
    return f"""# DR10 FITS data audit

Generated: `{utc_now()}`
Audit version: `{AUDIT_VERSION}`
Execution: CPU-only NumPy/Astropy/SEP; no model inference or training.

## Scope and semantics

Audited `{len(quality_rows)}` files. Structural/data-layout valid: `{valid}`;
celestial-WCS valid: `{wcs_valid}`; exact expected band order
`{''.join(settings.expected_bands)}`: `{band_valid}`. Blank candidates: `{blank}`;
full-frame candidates: `{full_frame}`; artifact/semantic candidates:
`{len(artifact_candidates)}`; exact-pixel duplicate groups:
`{unique_duplicate_groups}`.

These products are **resampled DR10 coadd cutouts**, not raw detector exposures.
The pixel arrays were never clipped, sky-filled, inpainted, or converted to
RGB. Negative sky-subtracted values were retained and summarized. Contact
sheets use a separate, display-only signed-asinh z/r/g rendering and must not be
used as scientific arrays.

## Units and bands

Frozen production semantics require exactly `{settings.expected_size} ×
{settings.expected_size}` pixels, `BANDS` and `BAND0/1/2` equal to
`{''.join(settings.expected_bands)}`, `SURVEY={settings.expected_survey}`,
`VERSION={settings.expected_version}`, and both WCS pixel scales equal to
`{settings.expected_pixel_scale_arcsec}` arcsec/pixel. Each band independently
requires finite fraction at least `{settings.minimum_per_band_finite_fraction}`
and finite-nonzero fraction at least
`{settings.minimum_per_band_nonzero_fraction}`. The same size/layer/band/scale
semantics are independently checked against each download-manifest request.

The official documented image unit was explicitly supplied as
`{documented_unit}`. `BUNIT`, if present, must agree. No unit is inferred from
pixel appearance. Official image-stack documentation:
https://www.legacysurvey.org/dr10/files/#image-stacks-southcoadd

## Statistical definitions

- Robust scale: `1.4826 × median(|x - median(x)|)` over finite pixels.
- Extreme: `|x - median| > {settings.extreme_sigma:g} × robust_scale`.
- Negative and zero fractions use finite pixels as denominator.
- NaN and ±Inf fractions use all pixels as denominator and remain distinct.
- Pixel hashes cover decoded array shape, dtype, and exact C-order bytes; they
  are not perceptual similarity hashes.
- Contact-sheet retention is a deterministic evenly spaced sample bounded at
  `{settings.max_contact_sheet_files}` files. Histogram retention is bounded at
  `{settings.histogram_max_samples_per_band}` values per band.
- Blank and full-frame classifications use the deterministic SEP protocol
  documented separately. Questionable sources are flagged, never deleted.

Tables contain `{len(band_rows)}` per-band rows. FITS `CHECKSUM`/`DATASUM`
keyword presence is reported separately from the external file SHA-256.
"""


def isolation_report(
    decisions: Sequence[dict[str, Any]], settings: AuditSettings
) -> str:
    counts: dict[str, int] = {}
    for row in decisions:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    terminal_downloads = [
        row for row in decisions if str(row.get("download_status", "")).strip()
    ]
    return f"""# DR10 source-isolation protocol

Generated: `{utc_now()}`
Rule version: `{ISOLATION_RULE_VERSION}`
SEP version: `{sep.__version__}`

## Deterministic detection

For each g,r,z plane, nonfinite pixels are masked for detection and replaced
only in a temporary detection copy by the finite median. SEP estimates a mesh
background (`bw=bh=min(32, image_dimension/4)`, floor 8). Each background-
subtracted band is divided by its median positive SEP RMS; the detection image
is their sum divided by `sqrt(3)`. Detection uses threshold
`{settings.detection_sigma:g} sigma`, minimum area `{settings.minarea}` pixels,
`deblend_nthresh={settings.deblend_nthresh}`, and
`deblend_cont={settings.deblend_cont:g}`.

The requested position comes from the required, one-to-one download-manifest
RA/Dec transformed by the independently validated FITS WCS. Production audits
fail closed rather than falling back to an unidentified geometric center. The
segmentation label under the requested pixel wins; otherwise the nearest
centroid is central only within `{settings.central_max_offset_px:g}` pixels.
Fluxes are sums of the
per-band, SEP-background-subtracted pixels in that object's deblended
segmentation mask. They are deterministic aperture-like diagnostics, not
catalog-calibrated total fluxes.

## Fixed source-library decisions

`rejected_for_source_library_use` is assigned for invalid FITS/grz/WCS
semantics, finite fraction below
`{settings.reject_finite_fraction:g}`, blank/no central source, central mask on
the border, a full-frame object, excessive centroid offset, or a clear close
blend (neighbor within `{settings.clear_blend_distance_px:g}` pixels with
detection-flux ratio at least `{settings.clear_blend_flux_ratio:g}`).

`manual_review` is assigned, absent a rejection, for finite fraction below
`{settings.review_finite_fraction:g}`, unresolved units, extreme-value rate
above `{settings.extreme_fraction_review:g}`, a neighbor within
`{settings.neighbor_review_distance_px:g}` pixels with ratio at least
`{settings.neighbor_review_flux_ratio:g}`, a compact PSF-like central candidate,
a qualifying compact PSF-like neighbor, or an exact-pixel duplicate. Every
detected neighbor is evaluated for the blend and compact-neighbor flags; the
nearest-neighbor distance and flux fields remain separate diagnostics. A
noncentral peripheral mask touching the raw frame edge remains reported but is
not itself a review trigger; central border contact remains a rejection.
Everything else is
`accepted_clean_source`.

Compact/stellar and pre-existing-blend fields are explicitly heuristic
candidates. No PSF model or external star/galaxy classifier is used, so they
must not be interpreted as definitive astrophysical labels. Sources are never
inpainted or deleted, and every non-accept decision retains explicit reasons.

The candidate source manifest is reconciled one-to-one against the latest
outcome for every source, and all latest outcomes must share one completed
downloader run ID. `{len(terminal_downloads)}` candidates ended in an explicit
terminal download or FITS-semantic failure and were rejected before SEP rather
than silently omitted. Their downloader status and verbatim error are retained
in the decision table. Cancelled, circuit-open, dry-run, existing-file-refused,
unknown, mixed-run, missing, or extra outcomes abort the audit instead of being
classified as scientific source-quality decisions.

Decision counts: `{json.dumps(counts, sort_keys=True)}`.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="FITS files or directories")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--bands", default="grz")
    parser.add_argument("--documented-unit", required=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--detection-sigma", type=float, default=2.5)
    parser.add_argument("--minarea", type=int, default=8)
    parser.add_argument("--central-max-offset", type=float, default=12.0)
    parser.add_argument("--contact-sheet-size", type=int, default=16)
    parser.add_argument("--max-contact-sheet-files", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_bands = parse_bands(args.bands)
    if expected_bands != ("g", "r", "z"):
        raise SystemExit("This foundation audit requires exact --bands grz")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.detection_sigma <= 0 or args.minarea < 1 or args.central_max_offset <= 0:
        raise SystemExit("detection settings must be positive")
    if args.contact_sheet_size < 1 or args.contact_sheet_size > 64:
        raise SystemExit("--contact-sheet-size must be in [1, 64]")
    if args.max_contact_sheet_files < 1 or args.max_contact_sheet_files > 256:
        raise SystemExit("--max-contact-sheet-files must be in [1, 256]")

    frozen = AuditSettings(expected_bands=expected_bands)
    requested_decision_settings = (
        float(args.detection_sigma),
        int(args.minarea),
        float(args.central_max_offset),
    )
    frozen_decision_settings = (
        frozen.detection_sigma,
        frozen.minarea,
        frozen.central_max_offset_px,
    )
    if requested_decision_settings != frozen_decision_settings:
        raise SystemExit(
            "production source-decision thresholds are frozen and cannot be "
            f"overridden: requested={requested_decision_settings}, "
            f"frozen={frozen_decision_settings}"
        )
    settings = AuditSettings(
        expected_bands=expected_bands,
        contact_sheet_size=args.contact_sheet_size,
        max_contact_sheet_files=args.max_contact_sheet_files,
    )
    try:
        documented_unit = require_documented_unit(args.documented_unit, settings)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    inputs = discover_inputs(args.inputs, args.recursive, args.limit)
    by_path, by_basename, manifest_rows = read_manifest(args.download_manifest)
    source_rows = read_source_manifest(args.source_manifest)
    reject_lockbox_inputs(inputs, [*manifest_rows, *source_rows])
    terminal_download_rows, completed_download_run_id = (
        validate_candidate_download_coverage(source_rows, manifest_rows)
    )
    validate_manifest_alignment(
        inputs,
        by_path,
        by_basename,
        manifest_rows,
        manifest_parent=args.download_manifest.expanduser().resolve().parent,
        require_all_successful_rows=args.limit is None,
    )
    validate_manifest_request_semantics(inputs, by_path, by_basename, settings)
    # Collision checking is read-only. Final directories are intentionally not
    # created until the complete CPU audit and decision pass has succeeded.
    preflight_output_paths(args.run_dir)
    visual_indices, histogram_indices = retention_plan(len(inputs), settings)

    quality_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    isolation_rows: list[dict[str, Any]] = []
    visuals: list[VisualRecord] = []
    sample_chunks: dict[str, list[np.ndarray]] = {band: [] for band in expected_bands}

    for zero_index, path in enumerate(inputs):
        index = zero_index + 1
        metadata = metadata_for_path(path, by_path, by_basename)
        quality, bands, isolation, visual, samples = audit_one(
            path,
            metadata,
            settings,
            documented_unit,
            retain_visual=zero_index in visual_indices,
            retain_histogram_samples=zero_index in histogram_indices,
            strict_source_association=True,
        )
        quality_rows.append(quality)
        band_rows.extend(bands)
        isolation_rows.append(isolation)
        if zero_index in visual_indices:
            visuals.append(visual)
        for band, values in zip(expected_bands, samples, strict=True):
            if values.size:
                sample_chunks[band].append(values)
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(inputs),
                    "path": str(path),
                    "fits_valid": bool(quality["fits_structure_valid"]),
                    "central_source": bool(isolation.get("central_source_present")),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    duplicate_group_rows = duplicate_rows(quality_rows)
    isolation_by_path = {row["path"]: row for row in isolation_rows}
    decisions = [
        decide_quality(row, isolation_by_path[row["path"]], settings)
        for row in quality_rows
    ]
    decisions.extend(
        terminal_rejection_decision(
            row, args.download_manifest.expanduser().resolve().parent
        )
        for row in terminal_download_rows
    )
    decision_by_path = {row["path"]: row["decision"] for row in decisions}
    for visual in visuals:
        visual.decision_hint = decision_by_path[visual.path]
    artifacts = artifact_rows(quality_rows, isolation_by_path, settings)

    if len(visuals) > settings.max_contact_sheet_files:
        raise RuntimeError("visual retention exceeded the configured hard bound")
    for chunks in sample_chunks.values():
        if sum(len(chunk) for chunk in chunks) > settings.histogram_max_samples_per_band:
            raise RuntimeError("histogram retention exceeded the configured hard bound")

    output_paths = prepare_output_paths(args.run_dir)

    for start in range(0, len(visuals), settings.contact_sheet_size):
        page = start // settings.contact_sheet_size + 1
        chunk = visuals[start : start + settings.contact_sheet_size]
        flush_contact_sheet(chunk, output_paths["fits_figures"] / f"fits_contact_sheet_{page:04d}.png")
        flush_isolation_sheet(chunk, output_paths["isolation_figures"] / f"isolation_contact_sheet_{page:04d}.png")
    band_distribution_figures(
        sample_chunks,
        output_paths["band_figures"],
        settings.histogram_max_samples_per_band,
    )

    common = [
        "path", "filename", "source_id", "group_id", "catalog_row_index",
        "file_size_bytes", "file_mtime_ns", "file_sha256",
    ]
    write_csv_exclusive(output_paths["quality"], quality_rows, common)
    write_csv_exclusive(output_paths["bands"], band_rows, common + ["band"])
    write_csv_exclusive(output_paths["artifacts"], artifacts, common[:5] + ["artifact_reasons"])
    write_csv_exclusive(
        output_paths["duplicates"],
        duplicate_group_rows,
        ["exact_duplicate_group_id", "pixel_hash", "group_size", "path", "source_id", "group_id", "file_sha256"],
    )
    write_csv_exclusive(output_paths["isolation"], isolation_rows, common)
    write_csv_exclusive(
        output_paths["decisions"],
        decisions,
        [
            "path", "filename", "source_id", "group_id", "catalog_row_index",
            "decision", "primary_reason", "all_reasons", "rejection_reasons",
            "manual_review_reasons", "rule_version", "exact_duplicate_group_id",
        ],
    )
    write_text_exclusive(
        output_paths["fits_report"],
        fits_report(
            quality_rows,
            band_rows,
            artifacts,
            duplicate_group_rows,
            settings,
            documented_unit,
        ),
    )
    write_text_exclusive(
        output_paths["isolation_report"], isolation_report(decisions, settings)
    )

    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision["decision"]] = counts.get(decision["decision"], 0) + 1
    print(
        json.dumps(
            {
                "audit_version": AUDIT_VERSION,
                "input_count": len(inputs),
                "candidate_count": len(source_rows),
                "terminal_download_rejection_count": len(terminal_download_rows),
                "completed_download_run_id": completed_download_run_id,
                "decision_counts": counts,
                "run_dir": str(args.run_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
