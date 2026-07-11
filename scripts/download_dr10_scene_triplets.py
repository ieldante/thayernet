#!/usr/bin/env python3
"""Download immutable matched DR10 observed/model/residual engineering cutouts.

This downloader is intentionally narrow: it selects the same 20 sources used by
the foundation field-of-view study, creates a brand-new run directory, records
complete request/response provenance, and refuses to replace any destination.
It does not create blending manifests or access role/split/lockbox products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS


ENDPOINT = "https://www.legacysurvey.org/viewer/fits-cutout"
LAYERS = (
    ("observed", "ls-dr10-south"),
    ("model", "ls-dr10-model"),
    ("residual", "ls-dr10-resid"),
)
USER_AGENT = (
    "Brown-DR10-Scene-Triplet-Probe/1.0 "
    "(scientific research; provenance-preserving retrieval)"
)
MANIFEST_FIELDS = (
    "event_utc",
    "source_id",
    "catalog_row_index",
    "ra",
    "dec",
    "product",
    "layer",
    "request_url",
    "final_url",
    "http_status",
    "status",
    "attempt",
    "elapsed_seconds",
    "bytes",
    "sha256",
    "relative_path",
    "response_headers_path",
    "response_headers_sha256",
    "raw_response_path",
    "raw_response_sha256",
    "content_type",
    "fits_shape",
    "bands",
    "band0",
    "band1",
    "band2",
    "survey",
    "version",
    "imagetyp",
    "bunit",
    "finite_fraction",
    "crpix1",
    "crpix2",
    "crval1",
    "crval2",
    "cd1_1",
    "cd1_2",
    "cd2_1",
    "cd2_2",
    "pixel_scale_x_arcsec",
    "pixel_scale_y_arcsec",
    "error",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


class AppendOnlyCsv:
    def __init__(self, path: Path, fields: tuple[str, ...]) -> None:
        self.path = path
        self.fields = fields
        write_csv_exclusive(path, [], fields)

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


class AppendOnlyJsonl:
    def __init__(self, path: Path) -> None:
        self.path = path
        write_exclusive(path, b"")

    def append(self, payload: dict[str, Any]) -> None:
        with self.path.open("ab") as handle:
            handle.write(json_bytes(payload).replace(b"\n", b" ").rstrip() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


class RequestLimiter:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.last_start = 0.0

    def wait(self) -> None:
        delay = self.interval_seconds - (time.monotonic() - self.last_start)
        if delay > 0:
            time.sleep(delay)
        self.last_start = time.monotonic()


def safe_component(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)


def read_engineering_sources(foundation_run: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source_manifest = foundation_run / "manifests" / "dr10_engineering_sources.csv"
    fov_table = foundation_run / "tables" / "field_of_view_metrics.csv"
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        first_twenty = list(csv.DictReader(handle))[:20]
    with fov_table.open(newline="", encoding="utf-8") as handle:
        fov_rows = [row for row in csv.DictReader(handle) if row["size_pixels"] == "256"]
    fov_by_id = {row["source_id"]: row for row in fov_rows}
    expected_ids = [row["source_id"] for row in first_twenty]
    if len(first_twenty) != 20 or len(fov_rows) != 20 or set(expected_ids) != set(fov_by_id):
        raise RuntimeError("foundation 20-source FOV subset is not one-to-one")
    rows: list[dict[str, str]] = []
    for rank, source in enumerate(first_twenty, start=1):
        fov = fov_by_id[source["source_id"]]
        for key in ("catalog_row_index", "ra", "dec"):
            if key == "catalog_row_index":
                equal = source[key] == fov[key]
            else:
                equal = abs(float(source[key]) - float(fov[key])) < 1e-12
            if not equal:
                raise RuntimeError(f"foundation source mismatch for {source['source_id']} {key}")
        prior_path = Path(fov["relative_path"])
        if not prior_path.is_absolute():
            prior_path = Path.cwd() / prior_path
        if not prior_path.is_file():
            raise RuntimeError(f"missing foundation FOV cutout: {prior_path}")
        actual_hash = sha256_file(prior_path)
        if actual_hash != fov["sha256"]:
            raise RuntimeError(f"foundation FOV hash mismatch: {prior_path}")
        rows.append(
            {
                "engineering_rank": str(rank),
                "source_id": source["source_id"],
                "catalog_row_index": source["catalog_row_index"],
                "ra": source["ra"],
                "dec": source["dec"],
                "brickid_dr8": source.get("brickid", ""),
                "objid_dr8": source.get("objid", ""),
                "morphology_stratum": source.get("morphology_stratum", ""),
                "foundation_fov_path": str(prior_path.resolve()),
                "foundation_fov_sha256": actual_hash,
            }
        )
    provenance = {
        "selection_rule": "first 20 rows of dr10_engineering_sources.csv; identical to unique size=256 rows in field_of_view_metrics.csv",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "fov_table": str(fov_table.resolve()),
        "fov_table_sha256": sha256_file(fov_table),
        "source_count": len(rows),
    }
    return rows, provenance


def filename_for(row: dict[str, str]) -> str:
    return (
        f"rank{int(row['engineering_rank']):02d}_row{row['catalog_row_index']}_"
        f"{safe_component(row['source_id'])}_ra{float(row['ra']):.10f}_"
        f"dec{float(row['dec']):+.10f}_grz.fits"
    )


def validate_product(path: Path, expected_ra: float, expected_dec: float) -> dict[str, Any]:
    with path.open("rb") as handle:
        if not handle.read(80).startswith(b"SIMPLE"):
            raise ValueError("response does not begin with FITS SIMPLE")
    with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
        hdul.verify("exception")
        if len(hdul) != 1 or hdul[0].data is None:
            raise ValueError("expected exactly one populated PrimaryHDU")
        data = np.asarray(hdul[0].data)
        header = hdul[0].header
        if data.shape != (3, 256, 256):
            raise ValueError(f"unexpected shape {data.shape}")
        bands = str(header.get("BANDS", "")).strip().lower()
        band_headers = tuple(str(header.get(f"BAND{i}", "")).strip().lower() for i in range(3))
        if bands != "grz" or band_headers != ("g", "r", "z"):
            raise ValueError(f"unexpected band semantics BANDS={bands!r} BAND0..2={band_headers!r}")
        wcs = WCS(header, naxis=2)
        if not wcs.has_celestial:
            raise ValueError("missing celestial WCS")
        center = wcs.all_world2pix([[expected_ra, expected_dec]], 0)[0]
        expected_center = np.array([127.5, 127.5])
        if float(np.linalg.norm(center - expected_center)) > 0.05:
            raise ValueError(f"requested coordinate is off center: {center.tolist()}")
        scales = np.abs(
            np.asarray(
                [scale.to_value("deg") for scale in wcs.proj_plane_pixel_scales()],
                dtype=float,
            )
            * 3600.0
        )
        if not np.allclose(scales, [0.262, 0.262], rtol=0, atol=1e-8):
            raise ValueError(f"unexpected pixel scale {scales.tolist()}")
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
        result = {
            "fits_shape": "x".join(str(v) for v in data.shape),
            "bands": bands,
            "band0": band_headers[0],
            "band1": band_headers[1],
            "band2": band_headers[2],
            "survey": str(header.get("SURVEY", "")).strip(),
            "version": str(header.get("VERSION", "")).strip(),
            "imagetyp": str(header.get("IMAGETYP", "")).strip(),
            "bunit": str(header.get("BUNIT", "")).strip(),
            "finite_fraction": float(np.isfinite(data).mean()),
            "crpix1": float(header["CRPIX1"]),
            "crpix2": float(header["CRPIX2"]),
            "crval1": float(header["CRVAL1"]),
            "crval2": float(header["CRVAL2"]),
            "cd1_1": float(matrix[0, 0]),
            "cd1_2": float(matrix[0, 1]),
            "cd2_1": float(matrix[1, 0]),
            "cd2_2": float(matrix[1, 1]),
            "pixel_scale_x_arcsec": float(scales[0]),
            "pixel_scale_y_arcsec": float(scales[1]),
        }
        return result


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout.rstrip("\n")

    return {
        "branch": run("git", "branch", "--show-current"),
        "head": run("git", "rev-parse", "HEAD"),
        "status_porcelain_v1": run("git", "status", "--porcelain=v1", "--untracked-files=all"),
    }


def download_one(
    session: requests.Session,
    row: dict[str, str],
    product: str,
    layer: str,
    layer_dir: Path,
    timeout: tuple[float, float],
    max_attempts: int,
    limiter: RequestLimiter,
    events: AppendOnlyJsonl,
) -> dict[str, Any]:
    params = {
        "ra": f"{float(row['ra']):.10f}",
        "dec": f"{float(row['dec']):.10f}",
        "layer": layer,
        "bands": "grz",
        "pixscale": "0.262000",
        "size": "256",
    }
    request_url = f"{ENDPOINT}?{urlencode(params)}"
    destination = layer_dir / filename_for(row)
    if destination.exists():
        raise FileExistsError(f"refusing existing destination: {destination}")
    record: dict[str, Any] = {
        "event_utc": utc_now(),
        "source_id": row["source_id"],
        "catalog_row_index": row["catalog_row_index"],
        "ra": row["ra"],
        "dec": row["dec"],
        "product": product,
        "layer": layer,
        "request_url": request_url,
        "relative_path": str(destination),
        "status": "failed",
    }
    start = time.monotonic()
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        raw_path = layer_dir / "raw_responses" / (
            f"{destination.name}.attempt{attempt}.{uuid.uuid4().hex}.response"
        )
        try:
            limiter.wait()
            events.append(
                {
                    "event": "request_start",
                    "event_utc": utc_now(),
                    "source_id": row["source_id"],
                    "product": product,
                    "layer": layer,
                    "attempt": attempt,
                    "request_url": request_url,
                }
            )
            with session.get(ENDPOINT, params=params, timeout=timeout, stream=True) as response:
                record.update(
                    attempt=attempt,
                    http_status=response.status_code,
                    final_url=response.url,
                    content_type=response.headers.get("Content-Type", ""),
                )
                headers_payload = {
                    "captured_utc": utc_now(),
                    "request_method": "GET",
                    "request_url": request_url,
                    "final_url": response.url,
                    "status_code": response.status_code,
                    "reason": response.reason,
                    "headers": dict(response.headers),
                    "request_headers": {
                        key: value
                        for key, value in response.request.headers.items()
                        if key.lower() not in {"authorization", "cookie", "proxy-authorization"}
                    },
                }
                header_path = layer_dir / "headers" / f"{destination.stem}.headers.attempt{attempt}.json"
                write_exclusive(header_path, json_bytes(headers_payload))
                record["response_headers_path"] = str(header_path)
                record["response_headers_sha256"] = sha256_file(header_path)
                with raw_path.open("xb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                record["raw_response_path"] = str(raw_path)
                record["raw_response_sha256"] = sha256_file(raw_path)
                if response.status_code != 200:
                    terminal = 400 <= response.status_code < 500 and response.status_code != 429
                    raise RuntimeError(
                        f"HTTP {response.status_code}; raw response retained at {raw_path}; "
                        f"terminal={terminal}"
                    )
            validation = validate_product(raw_path, float(row["ra"]), float(row["dec"]))
            os.link(raw_path, destination)
            record.update(
                status="downloaded_valid",
                elapsed_seconds=f"{time.monotonic() - start:.6f}",
                bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                error="",
                **validation,
            )
            events.append({"event": "downloaded_valid", **record})
            return record
        except BaseException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if raw_path.exists():
                record["raw_response_path"] = str(raw_path)
                record["raw_response_sha256"] = sha256_file(raw_path)
            events.append(
                {
                    "event": "attempt_failed",
                    "event_utc": utc_now(),
                    "source_id": row["source_id"],
                    "product": product,
                    "layer": layer,
                    "attempt": attempt,
                    "request_url": request_url,
                    "response_headers_path": record.get("response_headers_path", ""),
                    "response_headers_sha256": record.get("response_headers_sha256", ""),
                    "raw_response_path": record.get("raw_response_path", ""),
                    "raw_response_sha256": record.get("raw_response_sha256", ""),
                    "error": last_error,
                }
            )
            if destination.exists():
                raise
            status = int(record.get("http_status") or 0)
            if 400 <= status < 500 and status != 429:
                break
            if attempt < max_attempts:
                time.sleep(min(30.0, 2.0**attempt))
    record.update(
        status="failed",
        elapsed_seconds=f"{time.monotonic() - start:.6f}",
        bytes="",
        sha256="",
        error=last_error,
    )
    events.append({"event": "download_failed", **record})
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for value in (args.foundation_run, args.output_run):
        lowered = str(value).lower()
        if "lockbox" in lowered or "sealed" in lowered:
            raise SystemExit("refusing a lockbox/sealed path")
    if args.request_interval < 1.0:
        raise SystemExit("--request-interval must be at least 1 second")
    if args.output_run.exists():
        raise SystemExit(f"output run already exists; refusing overwrite: {args.output_run}")
    sources, source_provenance = read_engineering_sources(args.foundation_run)
    args.output_run.mkdir(parents=True, exist_ok=False)
    for child in ("manifests", "logs", "tables", "diagnostics", "figures", "reports", "downloads"):
        (args.output_run / child).mkdir(exist_ok=False)
    layer_dirs = {
        product: args.output_run / "downloads" / f"{product}_{args.timestamp}"
        for product, _layer in LAYERS
    }
    for layer_dir in layer_dirs.values():
        for child in (
            layer_dir,
            layer_dir / "headers",
            layer_dir / "failures",
            layer_dir / "raw_responses",
        ):
            child.mkdir(exist_ok=False)

    source_fields = tuple(sources[0].keys())
    source_path = args.output_run / "manifests" / "engineering_sources_20.csv"
    write_csv_exclusive(source_path, sources, source_fields)
    start_payload = {
        "event": "campaign_download_start",
        "event_utc": utc_now(),
        "argv": sys.argv,
        "python_executable": sys.executable,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "foundation_run": str(args.foundation_run.resolve()),
        "output_run": str(args.output_run.resolve()),
        "timestamp": args.timestamp,
        "source_provenance": source_provenance,
        "source_manifest_copy": str(source_path.resolve()),
        "source_manifest_copy_sha256": sha256_file(source_path),
        "endpoint": ENDPOINT,
        "layers": dict((product, layer) for product, layer in LAYERS),
        "bands": "grz",
        "size": 256,
        "pixscale_arcsec": 0.262,
        "request_interval_seconds": args.request_interval,
        "timeouts_seconds": [args.connect_timeout, args.read_timeout],
        "max_attempts": args.max_attempts,
        "git": git_snapshot(),
    }
    start_path = args.output_run / "logs" / "download_run_start.json"
    write_exclusive(start_path, json_bytes(start_payload))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/fits,application/octet-stream"})
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manifest_path = args.output_run / "manifests" / "scene_triplet_download_manifest.csv"
    failure_path = args.output_run / "manifests" / "scene_triplet_download_failures.csv"
    manifest = AppendOnlyCsv(manifest_path, MANIFEST_FIELDS)
    failure_manifest = AppendOnlyCsv(failure_path, MANIFEST_FIELDS)
    events = AppendOnlyJsonl(args.output_run / "logs" / "download_events.jsonl")
    limiter = RequestLimiter(args.request_interval)
    for product, layer in LAYERS:
        for index, row in enumerate(sources, start=1):
            record = download_one(
                session,
                row,
                product,
                layer,
                layer_dirs[product],
                (args.connect_timeout, args.read_timeout),
                args.max_attempts,
                limiter,
                events,
            )
            records.append(record)
            manifest.append(record)
            if record["status"] != "downloaded_valid":
                failures.append(record)
                failure_manifest.append(record)
            print(
                f"[{len(records):02d}/60] {product:<8} {index:02d}/20 "
                f"{row['source_id']}: {record['status']}",
                flush=True,
            )
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['product']}:{record['status']}"
        counts[key] = counts.get(key, 0) + 1
    finish_payload = {
        "event": "campaign_download_finish",
        "event_utc": utc_now(),
        "counts": counts,
        "record_count": len(records),
        "failure_count": len(failures),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "failures": str(failure_path.resolve()),
        "failures_sha256": sha256_file(failure_path),
        "git": git_snapshot(),
    }
    write_exclusive(args.output_run / "logs" / "download_run_finish.json", json_bytes(finish_payload))
    if len(records) != 60 or failures:
        print(json.dumps(finish_payload, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(finish_payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
