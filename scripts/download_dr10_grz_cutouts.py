#!/usr/bin/env python3
"""Respectful, resumable downloader for Legacy Surveys DR10 FITS cutouts.

The downloader is deliberately conservative: request starts are globally rate
limited, validated destination files are never replaced, downloads land in a
unique ``.part`` path, and only structurally valid FITS files are renamed to
their final name.  Manifests and event logs are append-only.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import email.utils
import errno
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import uuid
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

ENDPOINT = "https://www.legacysurvey.org/viewer/fits-cutout"
DEFAULT_USER_AGENT = (
    "Brown-Thayer-Select-DR10-Foundation/1.0 "
    "(scientific research; respectful automated cutout retrieval)"
)
MANIFEST_FIELDS = [
    "event_utc",
    "run_id",
    "source_id",
    "catalog_row_index",
    "ra",
    "dec",
    "group_id",
    "status",
    "attempts",
    "http_status",
    "elapsed_seconds",
    "bytes",
    "sha256",
    "relative_path",
    "request_url",
    "request_parameters_json",
    "fits_shape",
    "bands_header",
    "finite_fraction",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_token(value: object, limit: int = 80) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return (token or "unknown")[:limit]


def output_filename(row: dict[str, str]) -> str:
    source_id = _safe_token(row.get("source_id") or row.get("dr8_id") or "unknown")
    row_index = _safe_token(row.get("catalog_row_index") or "na")
    ra = float(row["ra"])
    dec = float(row["dec"])
    return f"row{row_index}_{source_id}_ra{ra:.7f}_dec{dec:+.7f}_grz.fits"


@dataclass(frozen=True)
class FitsValidation:
    valid: bool
    shape: str = ""
    bands_header: str = ""
    finite_fraction: float = float("nan")
    error: str = ""
    per_band_finite_fraction: tuple[float, ...] = ()
    per_band_nonzero_fraction: tuple[float, ...] = ()
    pixel_scale_arcsec: float = float("nan")
    survey: str = ""
    version: str = ""
    center_ra: float = float("nan")
    center_dec: float = float("nan")
    center_offset_pixels: float = float("nan")


class DownloadCancelled(Exception):
    """Internal cooperative-cancellation signal for one in-flight download."""


class TerminalDownloadError(Exception):
    """A deterministic request/content failure that must not be retried."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def sanitized_argv(argv: list[str]) -> list[str]:
    """Return reproducible argv while redacting conventional secret options."""
    secret_names = {
        "--api-key",
        "--authorization",
        "--password",
        "--secret",
        "--token",
    }
    result: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        option, separator, _option_value = value.partition("=")
        normalized = option.strip().lower()
        if normalized in secret_names:
            if separator:
                result.append(f"{option}=<redacted>")
            else:
                result.append(value)
                redact_next = True
            continue
        result.append(value)
    return result


def validate_fits(
    path: Path,
    expected_size: int,
    expected_bands: str,
    expected_pixscale: float = 0.262,
    expected_ra: float | None = None,
    expected_dec: float | None = None,
    maximum_center_offset_pixels: float = 0.05,
) -> FitsValidation:
    """Validate structure and basic semantics without altering pixel values."""
    try:
        with path.open("rb") as handle:
            signature = handle.read(80)
        if not signature.startswith(b"SIMPLE"):
            return FitsValidation(False, error="file does not start with FITS SIMPLE")

        with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
            hdul.verify("exception")
            if len(hdul) < 1 or hdul[0].data is None:
                return FitsValidation(False, error="missing primary image data")
            data = np.asarray(hdul[0].data)
            expected_shape = (len(expected_bands), expected_size, expected_size)
            if data.shape != expected_shape:
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    error=f"unexpected data shape {data.shape}; expected {expected_shape}",
                )
            bands_header = str(hdul[0].header.get("BANDS", "")).strip().lower()
            if bands_header != expected_bands.lower():
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    error=f"BANDS={bands_header!r}; expected {expected_bands!r}",
                )
            header = hdul[0].header
            expected_band_headers = tuple(expected_bands.lower())
            actual_band_headers = tuple(
                str(header.get(f"BAND{index}", "")).strip().lower()
                for index in range(len(expected_bands))
            )
            if actual_band_headers != expected_band_headers:
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    error=(
                        f"BAND0.. headers={actual_band_headers!r}; "
                        f"expected {expected_band_headers!r}"
                    ),
                )
            survey = str(header.get("SURVEY", "")).strip()
            version = str(header.get("VERSION", "")).strip()
            if survey != "DECaLS" or version != "DR10-south":
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    survey=survey,
                    version=version,
                    error=(
                        f"unexpected provenance SURVEY={survey!r}, VERSION={version!r}; "
                        "expected DECaLS DR10-south"
                    ),
                )
            per_band_finite = tuple(float(np.isfinite(band).mean()) for band in data)
            per_band_nonzero = tuple(
                float(np.mean(np.isfinite(band) & (band != 0))) for band in data
            )
            finite_fraction = float(np.isfinite(data).mean())
            if any(value < 0.99 for value in per_band_finite):
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    finite_fraction=finite_fraction,
                    per_band_finite_fraction=per_band_finite,
                    per_band_nonzero_fraction=per_band_nonzero,
                    survey=survey,
                    version=version,
                    error=f"per-band finite fractions below 0.99: {per_band_finite}",
                )
            if any(value < 0.01 for value in per_band_nonzero):
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    finite_fraction=finite_fraction,
                    per_band_finite_fraction=per_band_finite,
                    per_band_nonzero_fraction=per_band_nonzero,
                    survey=survey,
                    version=version,
                    error=f"blank or missing band; nonzero fractions: {per_band_nonzero}",
                )
            required_wcs = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")
            missing = [key for key in required_wcs if key not in header]
            if missing:
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    finite_fraction=finite_fraction,
                    per_band_finite_fraction=per_band_finite,
                    per_band_nonzero_fraction=per_band_nonzero,
                    survey=survey,
                    version=version,
                    error=f"missing WCS keys: {','.join(missing)}",
                )
            if not all(key in header for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    finite_fraction=finite_fraction,
                    per_band_finite_fraction=per_band_finite,
                    per_band_nonzero_fraction=per_band_nonzero,
                    survey=survey,
                    version=version,
                    error="missing celestial CD matrix",
                )
            scale_x = 3600.0 * float(np.hypot(header["CD1_1"], header["CD2_1"]))
            scale_y = 3600.0 * float(np.hypot(header["CD1_2"], header["CD2_2"]))
            pixel_scale = 0.5 * (scale_x + scale_y)
            if not (
                np.isclose(scale_x, expected_pixscale, rtol=5e-4, atol=1e-6)
                and np.isclose(scale_y, expected_pixscale, rtol=5e-4, atol=1e-6)
            ):
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    finite_fraction=finite_fraction,
                    per_band_finite_fraction=per_band_finite,
                    per_band_nonzero_fraction=per_band_nonzero,
                    pixel_scale_arcsec=pixel_scale,
                    survey=survey,
                    version=version,
                    error=(
                        f"pixel scale ({scale_x:.9g},{scale_y:.9g}) arcsec; "
                        f"expected {expected_pixscale}"
                    ),
                )
            if (expected_ra is None) != (expected_dec is None):
                return FitsValidation(
                    False,
                    shape="x".join(map(str, data.shape)),
                    bands_header=bands_header,
                    finite_fraction=finite_fraction,
                    pixel_scale_arcsec=pixel_scale,
                    survey=survey,
                    version=version,
                    error="expected RA and Dec must be supplied together",
                )
            center_ra = float("nan")
            center_dec = float("nan")
            center_offset_pixels = float("nan")
            if expected_ra is not None and expected_dec is not None:
                requested_ra = float(expected_ra)
                requested_dec = float(expected_dec)
                if not (
                    np.isfinite(requested_ra)
                    and np.isfinite(requested_dec)
                    and 0.0 <= requested_ra <= 360.0
                    and -90.0 <= requested_dec <= 90.0
                ):
                    return FitsValidation(
                        False,
                        shape="x".join(map(str, data.shape)),
                        bands_header=bands_header,
                        finite_fraction=finite_fraction,
                        pixel_scale_arcsec=pixel_scale,
                        survey=survey,
                        version=version,
                        error=(
                            f"invalid requested coordinate RA={requested_ra}, "
                            f"Dec={requested_dec}"
                        ),
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wcs = WCS(header).celestial
                x_center = (data.shape[2] - 1) / 2.0
                y_center = (data.shape[1] - 1) / 2.0
                world_ra, world_dec = wcs.pixel_to_world_values(x_center, y_center)
                center_ra = float(np.asarray(world_ra))
                center_dec = float(np.asarray(world_dec))
                delta_ra = np.deg2rad(
                    (center_ra - requested_ra + 180.0) % 360.0 - 180.0
                )
                center_dec_rad = np.deg2rad(center_dec)
                requested_dec_rad = np.deg2rad(requested_dec)
                haversine = (
                    np.sin((center_dec_rad - requested_dec_rad) / 2.0) ** 2
                    + np.cos(center_dec_rad)
                    * np.cos(requested_dec_rad)
                    * np.sin(delta_ra / 2.0) ** 2
                )
                separation_arcsec = 2.0 * np.arcsin(
                    np.sqrt(np.clip(haversine, 0.0, 1.0))
                ) * (180.0 / np.pi) * 3600.0
                center_offset_pixels = float(separation_arcsec / pixel_scale)
                if not (
                    np.isfinite(center_ra)
                    and np.isfinite(center_dec)
                    and np.isfinite(center_offset_pixels)
                    and center_offset_pixels <= maximum_center_offset_pixels
                ):
                    return FitsValidation(
                        False,
                        shape="x".join(map(str, data.shape)),
                        bands_header=bands_header,
                        finite_fraction=finite_fraction,
                        per_band_finite_fraction=per_band_finite,
                        per_band_nonzero_fraction=per_band_nonzero,
                        pixel_scale_arcsec=pixel_scale,
                        survey=survey,
                        version=version,
                        center_ra=center_ra,
                        center_dec=center_dec,
                        center_offset_pixels=center_offset_pixels,
                        error=(
                            f"WCS center offset {center_offset_pixels:.9g} pixels exceeds "
                            f"{maximum_center_offset_pixels} for requested "
                            f"RA={requested_ra}, Dec={requested_dec}"
                        ),
                    )
            return FitsValidation(
                True,
                shape="x".join(map(str, data.shape)),
                bands_header=bands_header,
                finite_fraction=finite_fraction,
                per_band_finite_fraction=per_band_finite,
                per_band_nonzero_fraction=per_band_nonzero,
                pixel_scale_arcsec=pixel_scale,
                survey=survey,
                version=version,
                center_ra=center_ra,
                center_dec=center_dec,
                center_offset_pixels=center_offset_pixels,
            )
    except Exception as exc:  # validation must report arbitrary corrupt inputs
        return FitsValidation(False, error=f"{type(exc).__name__}: {exc}")


class GlobalRateLimiter:
    """Guarantee a minimum interval between request starts across workers."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self._condition = threading.Condition()
        self._next_start = 0.0

    def wait(self, cancel_event: threading.Event | None = None) -> bool:
        """Reserve the next request start, returning false if cancelled."""
        with self._condition:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return False
                delay = max(0.0, self._next_start - time.monotonic())
                if delay <= 0.0:
                    self._next_start = time.monotonic() + self.interval_seconds
                    return True
                # Release the condition while waiting so a 429 response can
                # extend the shared deadline before another worker starts.
                poll = min(delay, 0.25) if cancel_event is not None else delay
                self._condition.wait(timeout=poll)

    def defer_all(self, delay_seconds: float) -> None:
        """Apply Retry-After to every worker, not just the one seeing 429."""
        with self._condition:
            self._next_start = max(
                self._next_start, time.monotonic() + max(0.0, delay_seconds)
            )
            self._condition.notify_all()

    def wake_all(self) -> None:
        """Wake rate-limit waiters so they can observe cancellation promptly."""
        with self._condition:
            self._condition.notify_all()


class FailureCircuitBreaker:
    """Stop new requests after sustained terminal failures across sources."""

    def __init__(self, threshold: int) -> None:
        if threshold < 1:
            raise ValueError("circuit-breaker threshold must be positive")
        self.threshold = threshold
        self._consecutive_failures = 0
        self._open = False
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> bool:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.threshold:
                self._open = True
            return self._open


def atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically move ``source`` while refusing an existing destination."""
    if os.uname().sysname == "Darwin":
        # renamex_np(..., RENAME_EXCL) is the macOS no-replace primitive.
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error_number, os.strerror(error_number), destination)
        return
    # POSIX hard-link creation is atomic and fails if destination exists. The
    # temporary directory entry is then retired only after the final name is
    # safely linked; downloaded scientific content is never replaced.
    os.link(source, destination)
    os.unlink(source)


class AppendOnlyCsv:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._lock = threading.Lock()

    def ensure_created(self) -> None:
        """Create a header-only table if absent; never truncate an existing one."""
        with self._lock:
            if self.path.exists():
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("x", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                # Another worker/process won the exclusive-create race.
                return

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            is_new = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
                if is_new:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in self.fieldnames})
                handle.flush()
                os.fsync(handle.fileno())


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def exclusive_json(path: Path, payload: dict[str, Any]) -> Path:
    """Create JSON once; choose a timestamped sibling rather than overwrite."""
    destination = path
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        destination = path.with_name(f"{path.stem}.resume_{stamp}{path.suffix}")
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return destination


def read_sources(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ra", "dec"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"input manifest is missing columns: {sorted(missing)}")
        rows: list[dict[str, str]] = []
        for row in reader:
            ra, dec = float(row["ra"]), float(row["dec"])
            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_request_url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{endpoint}?{urlencode(params)}"


def _base_record(
    row: dict[str, str], run_id: str, url: str, params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "event_utc": utc_now(),
        "run_id": run_id,
        "source_id": row.get("source_id") or row.get("dr8_id") or "",
        "catalog_row_index": row.get("catalog_row_index", ""),
        "ra": row["ra"],
        "dec": row["dec"],
        "group_id": row.get("provisional_group_id") or row.get("group_id") or "",
        "request_url": url,
        "request_parameters_json": json.dumps(params, sort_keys=True),
    }


def download_one(
    row: dict[str, str],
    *,
    output_dir: Path,
    endpoint: str,
    layer: str,
    bands: str,
    pixscale: float,
    size: int,
    timeout: tuple[float, float],
    max_retries: int,
    backoff_base: float,
    backoff_cap: float,
    user_agent: str,
    resume: bool,
    limiter: GlobalRateLimiter,
    circuit_breaker: FailureCircuitBreaker,
    cancel_event: threading.Event,
    event_log: JsonlLogger,
    run_id: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "ra": f"{float(row['ra']):.10f}",
        "dec": f"{float(row['dec']):.10f}",
        "layer": layer,
        "bands": bands,
        "pixscale": f"{pixscale:.6f}",
        "size": size,
    }
    url = build_request_url(endpoint, params)
    record = _base_record(row, run_id, url, params)
    filename = output_filename(row)
    destination = output_dir / filename
    record["relative_path"] = filename

    if destination.exists():
        validation = validate_fits(
            destination,
            size,
            bands,
            expected_pixscale=pixscale,
            expected_ra=float(row["ra"]),
            expected_dec=float(row["dec"]),
        )
        if resume and validation.valid:
            record.update(
                status="resume_validated_skip",
                attempts=0,
                http_status="",
                elapsed_seconds=0.0,
                bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                fits_shape=validation.shape,
                bands_header=validation.bands_header,
                finite_fraction=validation.finite_fraction,
                wcs_center_ra=validation.center_ra,
                wcs_center_dec=validation.center_dec,
                wcs_center_offset_pixels=validation.center_offset_pixels,
                error="",
            )
            event_log.append({**record, "event": "resume_skip"})
            return record
        record.update(
            status="existing_file_refused",
            attempts=0,
            http_status="",
            elapsed_seconds=0.0,
            bytes=destination.stat().st_size,
            sha256="",
            fits_shape=validation.shape,
            bands_header=validation.bands_header,
            finite_fraction=(
                validation.finite_fraction if np.isfinite(validation.finite_fraction) else ""
            ),
            error=(
                "destination already exists; use --resume for validated files; "
                f"validation={validation.error or 'valid'}"
            ),
        )
        event_log.append({**record, "event": "existing_refused"})
        return record

    if circuit_breaker.is_open():
        record.update(
            status="circuit_open",
            attempts=0,
            http_status="",
            elapsed_seconds=0.0,
            bytes="",
            sha256="",
            fits_shape="",
            bands_header="",
            finite_fraction="",
            error="global circuit breaker opened after sustained transient failures",
        )
        event_log.append({**record, "event": "request_suppressed_circuit_open"})
        return record

    start = time.monotonic()
    last_error = ""
    last_status: int | str = ""
    attempts_made = 0
    terminal_status = ""
    headers = {"User-Agent": user_agent, "Accept": "application/fits,application/octet-stream"}

    for attempt in range(1, max_retries + 1):
        if cancel_event.is_set():
            terminal_status = "cancelled"
            last_error = "download cancelled before request start"
            break
        if circuit_breaker.is_open():
            terminal_status = "circuit_open"
            last_error = "global circuit breaker opened before retry"
            break
        if not limiter.wait(cancel_event):
            terminal_status = "cancelled"
            last_error = "download cancelled while waiting for request slot"
            break
        if cancel_event.is_set():
            terminal_status = "cancelled"
            last_error = "download cancelled before request start"
            break
        if circuit_breaker.is_open():
            terminal_status = "circuit_open"
            last_error = "global circuit breaker opened before request start"
            break
        attempts_made = attempt
        event_log.append(
            {
                **record,
                "event": "request_start",
                "event_utc": utc_now(),
                "attempt": attempt,
            }
        )
        part = destination.with_name(
            f"{destination.name}.part.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            if cancel_event.is_set():
                raise DownloadCancelled("download cancelled before HTTP request")
            with requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=True,
            ) as response:
                last_status = response.status_code
                content_type = response.headers.get("Content-Type", "").lower()
                if response.status_code == 429:
                    delay = retry_after_seconds(response.headers.get("Retry-After"))
                    if delay is None:
                        delay = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                    delay += random.uniform(0.0, min(1.0, delay * 0.1))
                    limiter.defer_all(delay)
                    last_error = f"HTTP 429; Retry-After delay {delay:.3f}s"
                    event_log.append(
                        {
                            **record,
                            "event": "rate_limited",
                            "event_utc": utc_now(),
                            "attempt": attempt,
                            "http_status": 429,
                            "sleep_seconds": delay,
                        }
                    )
                    if attempt < max_retries:
                        if cancel_event.wait(delay):
                            raise DownloadCancelled(
                                "download cancelled during Retry-After"
                            )
                        continue
                    break
                if 400 <= response.status_code < 500:
                    raise TerminalDownloadError(
                        "http_client_error",
                        f"nonretryable HTTP {response.status_code}",
                    )
                if response.status_code >= 500:
                    raise requests.HTTPError(
                        f"server returned HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()
                if "text/html" in content_type or "application/json" in content_type:
                    raise TerminalDownloadError(
                        "content_rejected",
                        f"refusing error-like Content-Type {content_type!r}",
                    )

                with part.open("xb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if cancel_event.is_set():
                            raise DownloadCancelled(
                                "download cancelled while streaming response"
                            )
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())

            if cancel_event.is_set():
                raise DownloadCancelled("download cancelled before FITS validation")
            validation = validate_fits(
                part,
                size,
                bands,
                expected_pixscale=pixscale,
                expected_ra=float(row["ra"]),
                expected_dec=float(row["dec"]),
            )
            if not validation.valid:
                failed_part = part.with_name(part.name.replace(".part.", ".invalid."))
                part.rename(failed_part)
                raise TerminalDownloadError(
                    "validation_rejected",
                    f"downloaded FITS failed validation: {validation.error}",
                )
            if cancel_event.is_set():
                raise DownloadCancelled("download cancelled before atomic finalization")
            try:
                atomic_rename_noreplace(part, destination)
            except FileExistsError:
                retained = part.with_name(part.name.replace(".part.", ".race_retained."))
                part.rename(retained)
                raise FileExistsError(
                    "destination appeared during download; retained validated part without overwrite"
                )
            elapsed = time.monotonic() - start
            record.update(
                event_utc=utc_now(),
                status="downloaded_valid",
                attempts=attempt,
                http_status=last_status,
                elapsed_seconds=f"{elapsed:.6f}",
                bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                fits_shape=validation.shape,
                bands_header=validation.bands_header,
                finite_fraction=f"{validation.finite_fraction:.12g}",
                wcs_center_ra=validation.center_ra,
                wcs_center_dec=validation.center_dec,
                wcs_center_offset_pixels=validation.center_offset_pixels,
                error="",
            )
            event_log.append({**record, "event": "download_success"})
            circuit_breaker.record_success()
            return record
        except DownloadCancelled as exc:
            terminal_status = "cancelled"
            last_error = str(exc)
            event_log.append(
                {
                    **record,
                    "event": "attempt_cancelled",
                    "event_utc": utc_now(),
                    "attempt": attempt,
                    "http_status": last_status,
                    "error": last_error,
                }
            )
            break
        except TerminalDownloadError as exc:
            terminal_status = exc.status
            last_error = str(exc)
            event_log.append(
                {
                    **record,
                    "event": "terminal_request_outcome",
                    "event_utc": utc_now(),
                    "attempt": attempt,
                    "http_status": last_status,
                    "status": terminal_status,
                    "error": last_error,
                }
            )
            # A deterministic client/content outcome demonstrates that the
            # service is responsive and must not contribute to the transient
            # service-failure circuit.
            circuit_breaker.record_success()
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            event_log.append(
                {
                    **record,
                    "event": "attempt_failed",
                    "event_utc": utc_now(),
                    "attempt": attempt,
                    "http_status": last_status,
                    "error": last_error,
                }
            )
            if attempt < max_retries:
                delay = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                delay += random.uniform(0.0, min(1.0, delay * 0.1))
                if cancel_event.wait(delay):
                    terminal_status = "cancelled"
                    last_error = "download cancelled during exponential backoff"
                    break

    final_status = terminal_status or "failed"
    record.update(
        event_utc=utc_now(),
        status=final_status,
        attempts=attempts_made,
        http_status=last_status,
        elapsed_seconds=f"{time.monotonic() - start:.6f}",
        bytes="",
        sha256="",
        fits_shape="",
        bands_header="",
        finite_fraction="",
        error=last_error,
    )
    event_log.append(
        {
            **record,
            "event": (
                "download_failed" if final_status == "failed" else "download_terminal"
            ),
        }
    )
    if final_status == "failed":
        opened = circuit_breaker.record_failure()
        if opened:
            event_log.append(
                {
                    **record,
                    "event": "circuit_opened",
                    "event_utc": utc_now(),
                    "threshold": circuit_breaker.threshold,
                }
            )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--layer", default="ls-dr10-south")
    parser.add_argument("--bands", default="grz")
    parser.add_argument("--pixscale", type=float, default=0.262)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--backoff-cap", type=float, default=60.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--circuit-breaker-failures", type=int, default=5)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.request_interval < 1.0:
        raise SystemExit("--request-interval must be >= 1.0 seconds")
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    positive_values = {
        "max_retries": args.max_retries,
        "size": args.size,
        "pixscale": args.pixscale,
        "checkpoint_every": args.checkpoint_every,
        "connect_timeout": args.connect_timeout,
        "read_timeout": args.read_timeout,
        "backoff_base": args.backoff_base,
        "backoff_cap": args.backoff_cap,
        "circuit_breaker_failures": args.circuit_breaker_failures,
    }
    if any(value <= 0 for value in positive_values.values()):
        raise SystemExit(
            "size, pixscale, retries, checkpoints, timeouts, and backoff settings "
            "must be positive"
        )
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    rows = read_sources(args.input_manifest, args.limit)
    if not rows:
        raise SystemExit("no finite-coordinate rows found in input manifest")
    run_id = datetime.now(timezone.utc).strftime("download_%Y%m%d_%H%M%S_%f")
    request_template = {
        "endpoint": args.endpoint,
        "layer": args.layer,
        "bands": args.bands,
        "pixscale": args.pixscale,
        "size": args.size,
        "request_interval": args.request_interval,
        "workers": args.workers,
        "connect_timeout": args.connect_timeout,
        "read_timeout": args.read_timeout,
        "max_retries": args.max_retries,
        "backoff_base": args.backoff_base,
        "backoff_cap": args.backoff_cap,
        "checkpoint_every": args.checkpoint_every,
        "circuit_breaker_failures": args.circuit_breaker_failures,
        "user_agent": args.user_agent,
        "limit": args.limit,
        "resume": args.resume,
        "dry_run": args.dry_run,
    }
    run_argv = sanitized_argv(list(sys.argv))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "configuration": request_template,
                    "python_executable": sys.executable,
                    "argv": run_argv,
                },
                indent=2,
            )
        )
        for row in rows:
            params = {
                "ra": f"{float(row['ra']):.10f}",
                "dec": f"{float(row['dec']):.10f}",
                "layer": args.layer,
                "bands": args.bands,
                "pixscale": f"{args.pixscale:.6f}",
                "size": args.size,
            }
            print(build_request_url(args.endpoint, params))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    manifest = AppendOnlyCsv(args.output_dir / "download_manifest.csv", MANIFEST_FIELDS)
    failures = AppendOnlyCsv(args.output_dir / "failed_downloads.csv", MANIFEST_FIELDS)
    manifest.ensure_created()
    failures.ensure_created()
    events = JsonlLogger(args.output_dir / "logs" / "download_events.jsonl")
    input_manifest_path = args.input_manifest.expanduser().resolve()
    input_manifest_sha256 = sha256_file(input_manifest_path)
    downloader_source_path = Path(__file__).resolve()
    downloader_source_sha256 = sha256_file(downloader_source_path)
    events.append(
        {
            "event": "run_start",
            "event_utc": utc_now(),
            "run_id": run_id,
            "input_manifest": str(input_manifest_path),
            "input_sha256": input_manifest_sha256,
            "downloader_source": str(downloader_source_path),
            "downloader_source_sha256": downloader_source_sha256,
            "python_executable": sys.executable,
            "argv": run_argv,
            "row_count": len(rows),
            "configuration": request_template,
        }
    )
    limiter = GlobalRateLimiter(args.request_interval)
    circuit_breaker = FailureCircuitBreaker(args.circuit_breaker_failures)
    stop_event = threading.Event()
    records: list[dict[str, Any]] = []
    started = time.monotonic()

    def task(row: dict[str, str]) -> dict[str, Any]:
        return download_one(
            row,
            output_dir=args.output_dir,
            endpoint=args.endpoint,
            layer=args.layer,
            bands=args.bands,
            pixscale=args.pixscale,
            size=args.size,
            timeout=(args.connect_timeout, args.read_timeout),
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            backoff_cap=args.backoff_cap,
            user_agent=args.user_agent,
            resume=args.resume,
            limiter=limiter,
            circuit_breaker=circuit_breaker,
            cancel_event=stop_event,
            event_log=events,
            run_id=run_id,
        )

    executor = ThreadPoolExecutor(max_workers=args.workers)
    future_to_index: dict[Future[dict[str, Any]], int] = {}
    row_iterator = iter(enumerate(rows))
    input_exhausted = False
    completed = 0
    success_statuses = {"downloaded_valid", "resume_validated_skip"}

    def fill_slots() -> None:
        nonlocal input_exhausted
        while len(future_to_index) < args.workers and not stop_event.is_set():
            try:
                index, row = next(row_iterator)
            except StopIteration:
                input_exhausted = True
                return
            future_to_index[executor.submit(task, row)] = index

    def retain_result(future: Future[dict[str, Any]]) -> None:
        nonlocal completed
        record = future.result()
        manifest.append(record)
        if record["status"] not in success_statuses:
            failures.append(record)
        records.append(record)
        completed += 1

    def write_checkpoint_if_due() -> None:
        if completed % args.checkpoint_every != 0 and completed != len(rows):
            return
        counts: dict[str, int] = {}
        for item in records:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        checkpoint = {
            "event_utc": utc_now(),
            "run_id": run_id,
            "completed": completed,
            "total": len(rows),
            "status_counts": counts,
            "elapsed_seconds": time.monotonic() - started,
        }
        checkpoint_path = args.output_dir / "logs" / (
            f"progress_checkpoint_{completed:07d}_{uuid.uuid4().hex[:8]}.json"
        )
        exclusive_json(checkpoint_path, checkpoint)
        print(json.dumps(checkpoint, sort_keys=True), flush=True)

    def stop_executor() -> list[Future[dict[str, Any]]]:
        stop_event.set()
        limiter.wake_all()
        pending = list(future_to_index)
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        return pending

    try:
        fill_slots()
        while future_to_index:
            done, _pending = wait(
                tuple(future_to_index), return_when=FIRST_COMPLETED
            )
            for future in sorted(done, key=lambda item: future_to_index[item]):
                future_to_index.pop(future)
                retain_result(future)
                write_checkpoint_if_due()
            fill_slots()
        if not input_exhausted:
            raise RuntimeError("bounded executor stopped before consuming input rows")
    except KeyboardInterrupt:
        pending = stop_executor()
        # Retain any in-flight result that completed during orderly shutdown.
        for future in sorted(
            pending, key=lambda item: future_to_index.get(item, len(rows))
        ):
            if future.done() and not future.cancelled():
                retain_result(future)
        counts: dict[str, int] = {}
        for item in records:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        interrupted_summary = {
            "event_utc": utc_now(),
            "run_id": run_id,
            "interrupted": True,
            "completed": completed,
            "total": len(rows),
            "status_counts": counts,
            "elapsed_seconds": time.monotonic() - started,
            "input_manifest": str(input_manifest_path),
            "input_sha256": input_manifest_sha256,
            "downloader_source": str(downloader_source_path),
            "downloader_source_sha256": downloader_source_sha256,
            "python_executable": sys.executable,
            "argv": run_argv,
            "configuration": request_template,
        }
        interrupted_path = args.output_dir / (
            f"download_summary.interrupted_{run_id}.json"
        )
        exclusive_json(interrupted_path, interrupted_summary)
        events.append({**interrupted_summary, "event": "run_interrupted"})
        return 130
    except BaseException:
        stop_executor()
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)

    counts = {}
    for item in records:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    summary = {
        "event_utc": utc_now(),
        "run_id": run_id,
        "input_manifest": str(input_manifest_path),
        "input_sha256": input_manifest_sha256,
        "downloader_source": str(downloader_source_path),
        "downloader_source_sha256": downloader_source_sha256,
        "python_executable": sys.executable,
        "argv": run_argv,
        "configuration": request_template,
        "attempted_rows": len(rows),
        "status_counts": counts,
        "elapsed_seconds": time.monotonic() - started,
        "total_validated_bytes": sum(
            int(item.get("bytes") or 0)
            for item in records
            if item["status"] in {"downloaded_valid", "resume_validated_skip"}
        ),
    }
    summary_path = exclusive_json(args.output_dir / "download_summary.json", summary)
    events.append(
        {
            "event": "run_complete",
            "event_utc": utc_now(),
            "run_id": run_id,
            "summary_path": str(summary_path),
            "status_counts": counts,
        }
    )
    return 1 if any(status not in success_statuses for status in counts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
