#!/usr/bin/env python3
"""Download official DR10 catalog boxes for the 20 scene-probe sources.

The catalog boxes supply documented PSFSIZE_G/R/Z values and source/component
metadata without consulting the sealed split. Every response and header is
retained; destinations are created with no-replace hard links.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u


ENDPOINT = "https://www.legacysurvey.org/viewer/ls-dr10-south/cat.fits"
FIELDS = (
    "event_utc",
    "source_id",
    "catalog_row_index",
    "ra",
    "dec",
    "request_url",
    "final_url",
    "http_status",
    "status",
    "elapsed_seconds",
    "bytes",
    "sha256",
    "relative_path",
    "raw_response_path",
    "raw_response_sha256",
    "response_headers_path",
    "response_headers_sha256",
    "row_count",
    "central_row_index",
    "central_separation_arcsec",
    "release",
    "brickid",
    "brickname",
    "objid",
    "type",
    "ref_cat",
    "ref_id",
    "psfsize_g_arcsec",
    "psfsize_r_arcsec",
    "psfsize_z_arcsec",
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


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class CsvLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore").writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def validate_catalog(path: Path, source_ra: float, source_dec: float) -> dict[str, Any]:
    with path.open("rb") as handle:
        if not handle.read(80).startswith(b"SIMPLE"):
            raise ValueError("response is not FITS")
    with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
        hdul.verify("exception")
        tables = [hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU)]
        if len(tables) != 1 or tables[0].data is None:
            raise ValueError("expected one populated binary table")
        data = tables[0].data
        names = {name.lower(): name for name in data.names}
        required = {"ra", "dec", "psfsize_g", "psfsize_r", "psfsize_z"}
        missing = required - names.keys()
        if missing:
            raise ValueError(f"catalog missing columns: {sorted(missing)}")
        if len(data) == 0:
            raise ValueError("empty catalog box")
        coords = SkyCoord(
            np.asarray(data[names["ra"]], dtype=float) * u.deg,
            np.asarray(data[names["dec"]], dtype=float) * u.deg,
        )
        target = SkyCoord(source_ra * u.deg, source_dec * u.deg)
        separations = target.separation(coords).arcsec
        index = int(np.argmin(separations))
        row = data[index]

        def get(name: str, default: Any = "") -> Any:
            actual = names.get(name.lower())
            return default if actual is None else row[actual]

        result = {
            "row_count": len(data),
            "central_row_index": index,
            "central_separation_arcsec": float(separations[index]),
            "release": int(get("release", 0)),
            "brickid": int(get("brickid", 0)),
            "brickname": text(get("brickname")),
            "objid": int(get("objid", 0)),
            "type": text(get("type")),
            "ref_cat": text(get("ref_cat")),
            "ref_id": text(get("ref_id")),
            "psfsize_g_arcsec": float(get("psfsize_g", np.nan)),
            "psfsize_r_arcsec": float(get("psfsize_r", np.nan)),
            "psfsize_z_arcsec": float(get("psfsize_z", np.nan)),
        }
        if result["central_separation_arcsec"] > 3.0:
            raise ValueError(
                f"nearest official DR10 source is {result['central_separation_arcsec']:.3f} arcsec away"
            )
        for band in "grz":
            value = result[f"psfsize_{band}_arcsec"]
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid PSFSIZE_{band.upper()}={value}")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--request-interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "lockbox" in str(args.run_dir).lower() or "sealed" in str(args.run_dir).lower():
        raise SystemExit("refusing lockbox/sealed path")
    source_path = args.run_dir / "manifests" / "engineering_sources_20.csv"
    with source_path.open(newline="", encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))
    if len(sources) != 20:
        raise SystemExit("expected exactly 20 engineering sources")
    output_dir = args.run_dir / "downloads" / f"official_catalog_{args.timestamp}"
    output_dir.mkdir(exist_ok=False)
    for name in ("headers", "raw_responses"):
        (output_dir / name).mkdir(exist_ok=False)
    manifest = CsvLog(args.run_dir / "manifests" / "official_catalog_download_manifest.csv")
    failures = CsvLog(args.run_dir / "manifests" / "official_catalog_download_failures.csv")
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Brown-DR10-Scene-Probe/1.0 (official catalog/PSF audit)",
            "Accept": "application/fits,application/octet-stream",
        }
    )
    failed = 0
    last_start = 0.0
    for rank, source in enumerate(sources, start=1):
        ra = float(source["ra"])
        dec = float(source["dec"])
        half_dec = (256 * 0.262 / 2 + 3.0) / 3600.0
        half_ra = half_dec / max(math.cos(math.radians(dec)), 0.1)
        params = {
            "ralo": f"{ra - half_ra:.10f}",
            "rahi": f"{ra + half_ra:.10f}",
            "declo": f"{dec - half_dec:.10f}",
            "dechi": f"{dec + half_dec:.10f}",
        }
        request_url = f"{ENDPOINT}?{urlencode(params)}"
        filename = f"rank{rank:02d}_{source['source_id']}_dr10_catalog.fits"
        destination = output_dir / filename
        if destination.exists():
            raise FileExistsError(f"refusing existing destination: {destination}")
        raw_path = output_dir / "raw_responses" / f"{filename}.{uuid.uuid4().hex}.response"
        header_path = output_dir / "headers" / f"{filename}.headers.json"
        row: dict[str, Any] = {
            "event_utc": utc_now(),
            "source_id": source["source_id"],
            "catalog_row_index": source["catalog_row_index"],
            "ra": source["ra"],
            "dec": source["dec"],
            "request_url": request_url,
            "status": "failed",
            "relative_path": str(destination),
        }
        delay = args.request_interval - (time.monotonic() - last_start)
        if delay > 0:
            time.sleep(delay)
        last_start = time.monotonic()
        start = time.monotonic()
        try:
            with session.get(ENDPOINT, params=params, timeout=(15.0, 180.0), stream=True) as response:
                row.update(http_status=response.status_code, final_url=response.url)
                headers = {
                    "captured_utc": utc_now(),
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
                write_exclusive(
                    header_path,
                    (json.dumps(headers, indent=2, sort_keys=True) + "\n").encode(),
                )
                row.update(
                    response_headers_path=str(header_path),
                    response_headers_sha256=sha256_file(header_path),
                )
                with raw_path.open("xb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                row.update(
                    raw_response_path=str(raw_path),
                    raw_response_sha256=sha256_file(raw_path),
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
            validation = validate_catalog(raw_path, ra, dec)
            os.link(raw_path, destination)
            row.update(
                status="downloaded_valid",
                elapsed_seconds=f"{time.monotonic() - start:.6f}",
                bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                error="",
                **validation,
            )
        except BaseException as exc:
            failed += 1
            row.update(
                status="failed",
                elapsed_seconds=f"{time.monotonic() - start:.6f}",
                error=f"{type(exc).__name__}: {exc}",
            )
            failures.append(row)
        manifest.append(row)
        print(f"[{rank:02d}/20] {source['source_id']}: {row['status']}", flush=True)
    print(json.dumps({"records": len(sources), "failures": failed}, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
