#!/usr/bin/env python3
"""Fail-closed, append-only final provenance snapshot for the DR10 campaign.

This script is intentionally a finalization step, not a general-purpose audit.
It creates exactly two files, both with exclusive-create semantics:

* ``logs/input_provenance.json``
* ``tables/checkpoint_inventory_after.csv``

All input, Git, source-file, and checkpoint checks are performed twice before
either output is created.  The script never stages files and never replaces or
removes an existing path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "runs"
DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "data/catalogs/galaxy_zoo_desi/gz_desi_deep_learning_catalog_friendly.parquet"
)
DEFAULT_SMOKE_FITS = (
    PROJECT_ROOT
    / "data/dr10_grz_cutouts/manual_smoke/ra190.1086_dec1.2005_grz_256.fits"
)
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "outputs/checkpoints"
EXPECTED_BRANCH = "thayer-select"
EXPECTED_HEAD = "f0f36a3fac01ca2b09b989f7cac107d5687e6af9"
EXPECTED_CATALOG_MD5 = "114785d00c4d4f2208185bee73dd08b8"
EXPECTED_CATALOG_SHA256 = (
    "90a78648e1b1aa7642e7cb03c32fc932bdf5c04e2c42ed6bb722aa1ecbd9f1c8"
)
EXPECTED_SMOKE_SHA256 = (
    "fe7beb655ecff5bfea211d78df2dcdb4e6290e2834c5341ebc2ee917a4b974af"
)
CAMPAIGN_FILE_ROOTS = ("scripts/", "src/", "tests/", "docs/")
CHECKPOINT_FIELDS = ("path", "sha256", "size_bytes", "mtime_ns")
AFTER_CHECKPOINT_FIELDS = (
    "path",
    "sha256",
    "size_bytes",
    "mtime_ns",
    "before_sha256",
    "before_size_bytes",
    "before_mtime_ns",
    "identity_unchanged",
)
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceError(RuntimeError):
    """A fail-closed provenance gate did not pass."""


def _run_git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = os.fsdecode(result.stderr).strip()
        raise ProvenanceError(
            f"git {' '.join(arguments)} failed with code {result.returncode}: {stderr}"
        )
    return result


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="surrogateescape")


def parse_porcelain_v1_z(payload: bytes) -> list[dict[str, str]]:
    """Parse ``git status --porcelain=v1 -z`` without losing path spaces."""
    tokens = payload.split(b"\0")
    records: list[dict[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        if len(token) < 4 or token[2:3] != b" ":
            raise ProvenanceError(f"malformed Git porcelain record: {token!r}")
        status_code = _decode(token[:2])
        record = {"status": status_code, "path": _decode(token[3:])}
        if "R" in status_code or "C" in status_code:
            if index >= len(tokens) or not tokens[index]:
                raise ProvenanceError("rename/copy Git status record lacks its source path")
            record["original_path"] = _decode(tokens[index])
            index += 1
        records.append(record)
    return records


def validate_provenance_caveats(values: Sequence[str]) -> list[str]:
    """Validate caveats while preserving every accepted string verbatim."""
    caveats: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ProvenanceError("provenance caveats must contain non-whitespace text")
        if "\x00" in value:
            raise ProvenanceError("provenance caveats cannot contain NUL bytes")
        caveats.append(value)
    return caveats


def _safe_regular_file(path: Path, description: str) -> os.stat_result:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise ProvenanceError(f"missing {description}: {path}") from exc
    if stat.S_ISLNK(details.st_mode):
        raise ProvenanceError(f"{description} cannot be a symbolic link: {path}")
    if not stat.S_ISREG(details.st_mode):
        raise ProvenanceError(f"{description} is not a regular file: {path}")
    return details


def hash_regular_file(path: Path, algorithms: Sequence[str] = ("sha256",)) -> dict[str, Any]:
    """Hash a stable regular file and fail if its stat changes during reading."""
    before = _safe_regular_file(path, "file to hash")
    digests = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for digest in digests.values():
                digest.update(block)
    after = path.stat(follow_symlinks=False)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ProvenanceError(f"file changed while it was being hashed: {path}")
    result: dict[str, Any] = {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }
    result.update({name: digest.hexdigest() for name, digest in digests.items()})
    return result


def _project_relative(path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ProvenanceError(f"path is outside the project root: {path}") from exc


def _resolve_argument(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _normalize_checkpoint_row(row: dict[str, Any], origin: str) -> dict[str, Any]:
    missing = [field for field in CHECKPOINT_FIELDS if field not in row]
    if missing:
        raise ProvenanceError(f"{origin} checkpoint row lacks fields: {missing}")
    checkpoint_path = str(row["path"])
    pure_path = PurePosixPath(checkpoint_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or checkpoint_path == ".":
        raise ProvenanceError(f"unsafe checkpoint path in {origin}: {checkpoint_path!r}")
    sha256 = str(row["sha256"]).lower()
    if not HEX_64.fullmatch(sha256):
        raise ProvenanceError(f"invalid checkpoint SHA-256 in {origin}: {sha256!r}")
    try:
        size_bytes = int(row["size_bytes"])
        mtime_ns = int(row["mtime_ns"])
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"non-integer checkpoint metadata in {origin}") from exc
    if size_bytes < 0 or mtime_ns < 0:
        raise ProvenanceError(f"negative checkpoint metadata in {origin}")
    return {
        "path": pure_path.as_posix(),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
    }


def _rows_by_unique_path(
    rows: Iterable[dict[str, Any]], origin: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = _normalize_checkpoint_row(raw_row, origin)
        if row["path"] in indexed:
            raise ProvenanceError(
                f"duplicate checkpoint path in {origin}: {row['path']}"
            )
        indexed[row["path"]] = row
    return indexed


def compare_checkpoint_inventories(
    before_rows: Iterable[dict[str, Any]],
    current_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require identical path sets, bytes, sizes, and nanosecond mtimes."""
    before = _rows_by_unique_path(before_rows, "before inventory")
    current = _rows_by_unique_path(current_rows, "current inventory")
    missing = sorted(set(before) - set(current))
    extra = sorted(set(current) - set(before))
    if missing or extra:
        raise ProvenanceError(
            "checkpoint path set changed; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    changed: list[str] = []
    after_rows: list[dict[str, Any]] = []
    for path in sorted(before):
        old = before[path]
        new = current[path]
        differences = [
            field
            for field in ("sha256", "size_bytes", "mtime_ns")
            if old[field] != new[field]
        ]
        if differences:
            changed.append(f"{path} ({','.join(differences)})")
        after_rows.append(
            {
                **new,
                "before_sha256": old["sha256"],
                "before_size_bytes": old["size_bytes"],
                "before_mtime_ns": old["mtime_ns"],
                "identity_unchanged": not differences,
            }
        )
    if changed:
        raise ProvenanceError(
            "checkpoint integrity comparison failed: " + "; ".join(changed)
        )
    return after_rows


def read_checkpoint_inventory(path: Path) -> list[dict[str, Any]]:
    _safe_regular_file(path, "before checkpoint inventory")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(CHECKPOINT_FIELDS).issubset(
            reader.fieldnames
        ):
            raise ProvenanceError(
                f"checkpoint inventory has incompatible columns: {reader.fieldnames}"
            )
        return list(reader)


def inventory_checkpoints(checkpoint_dir: Path) -> list[dict[str, Any]]:
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise ProvenanceError(
            f"checkpoint root must be a real directory: {checkpoint_dir}"
        )
    rows: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(checkpoint_dir, followlinks=False):
        dirnames.sort()
        filenames.sort()
        base = Path(directory)
        for dirname in dirnames:
            candidate = base / dirname
            if candidate.is_symlink():
                raise ProvenanceError(
                    f"symbolic link found in checkpoint inventory: {candidate}"
                )
        for filename in filenames:
            path = base / filename
            if path.is_symlink():
                raise ProvenanceError(
                    f"symbolic link found in checkpoint inventory: {path}"
                )
            digest = hash_regular_file(path)
            rows.append(
                {
                    "path": _project_relative(path),
                    "sha256": digest["sha256"],
                    "size_bytes": digest["size_bytes"],
                    "mtime_ns": digest["mtime_ns"],
                }
            )
    if not rows:
        raise ProvenanceError(f"checkpoint inventory is empty: {checkpoint_dir}")
    return sorted(rows, key=lambda row: row["path"])


def serialize_checkpoint_csv(rows: Sequence[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=AFTER_CHECKPOINT_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def validate_smoke_fits(path: Path) -> dict[str, Any]:
    """Strictly reopen the pinned manual smoke FITS without altering it."""
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as exc:
        raise ProvenanceError("Astropy is required to validate the smoke FITS") from exc

    try:
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            hdul.verify("exception")
            if len(hdul) != 1 or hdul[0].data is None:
                raise ProvenanceError("smoke FITS must contain one primary image HDU")
            shape = tuple(int(value) for value in hdul[0].data.shape)
            header = hdul[0].header
            bands = str(header.get("BANDS", "")).strip().lower()
            band_headers = [
                str(header.get(f"BAND{index}", "")).strip().lower()
                for index in range(3)
            ]
            wcs = WCS(header)
            if shape != (3, 256, 256):
                raise ProvenanceError(f"unexpected smoke FITS shape: {shape}")
            if bands != "grz" or band_headers != ["g", "r", "z"]:
                raise ProvenanceError(
                    f"unexpected smoke FITS band semantics: {bands}, {band_headers}"
                )
            if not wcs.has_celestial or wcs.pixel_n_dim < 2:
                raise ProvenanceError("smoke FITS lacks a usable celestial WCS")
            return {
                "valid": True,
                "hdu_count": len(hdul),
                "shape": list(shape),
                "dtype": str(hdul[0].data.dtype),
                "bands": bands,
                "band_headers": band_headers,
                "survey": header.get("SURVEY"),
                "version": header.get("VERSION"),
                "bunit": header.get("BUNIT"),
                "celestial_wcs": True,
                "wcs_pixel_dimensions": int(wcs.pixel_n_dim),
            }
    except ProvenanceError:
        raise
    except Exception as exc:
        raise ProvenanceError(f"smoke FITS validation failed: {exc}") from exc


def capture_inputs(
    catalog: Path,
    smoke_fits: Path,
    expected_catalog_md5: str,
    expected_catalog_sha256: str,
    expected_smoke_sha256: str,
) -> dict[str, Any]:
    catalog_identity = hash_regular_file(catalog, ("md5", "sha256"))
    smoke_identity = hash_regular_file(smoke_fits, ("sha256",))
    if catalog_identity["md5"] != expected_catalog_md5.lower():
        raise ProvenanceError(
            "pinned catalog MD5 mismatch: "
            f"{catalog_identity['md5']} != {expected_catalog_md5.lower()}"
        )
    if catalog_identity["sha256"] != expected_catalog_sha256.lower():
        raise ProvenanceError(
            "pinned catalog SHA-256 mismatch: "
            f"{catalog_identity['sha256']} != {expected_catalog_sha256.lower()}"
        )
    if smoke_identity["sha256"] != expected_smoke_sha256.lower():
        raise ProvenanceError(
            "manual smoke FITS SHA-256 mismatch: "
            f"{smoke_identity['sha256']} != {expected_smoke_sha256.lower()}"
        )
    return {
        "catalog": {
            "path": _project_relative(catalog),
            "absolute_path": str(catalog),
            **catalog_identity,
            "pinned_source": "Galaxy Zoo DESI Zenodo record 8360385",
            "pinned_version": "1.0.1",
            "expected_md5": expected_catalog_md5.lower(),
            "expected_sha256": expected_catalog_sha256.lower(),
            "identity_pass": True,
        },
        "manual_smoke_fits": {
            "path": _project_relative(smoke_fits),
            "absolute_path": str(smoke_fits),
            **smoke_identity,
            "expected_sha256": expected_smoke_sha256.lower(),
            "identity_pass": True,
            "structural_validation": validate_smoke_fits(smoke_fits),
        },
    }


def _git_ignore_evidence(paths: Sequence[Path]) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for path in paths:
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise ProvenanceError(f"ignore probe is outside project: {path}") from exc
        result = _run_git("check-ignore", "-v", relative, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise ProvenanceError(f"required data/output path is not Git-ignored: {relative}")
        evidence.append(
            {
                "path": relative,
                "git_check_ignore": _decode(result.stdout).strip(),
            }
        )
    return evidence


def hash_campaign_files(
    status_records: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in status_records:
        path_text = record["path"]
        if not path_text.startswith(CAMPAIGN_FILE_ROOTS):
            continue
        path = PROJECT_ROOT / path_text
        identity = hash_regular_file(path)
        rows.append(
            {
                "path": path_text,
                "git_status": record["status"],
                **identity,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def capture_repository(
    expected_branch: str,
    expected_head: str,
    ignore_probe_paths: Sequence[Path],
) -> dict[str, Any]:
    branch = _decode(_run_git("branch", "--show-current").stdout).strip()
    head = _decode(_run_git("rev-parse", "HEAD").stdout).strip()
    if branch != expected_branch:
        raise ProvenanceError(f"branch mismatch: {branch!r} != {expected_branch!r}")
    if head != expected_head:
        raise ProvenanceError(f"HEAD mismatch: {head!r} != {expected_head!r}")

    staged = _run_git("diff", "--cached", "--quiet", "--exit-code", check=False)
    if staged.returncode == 1:
        raise ProvenanceError("Git index contains staged changes")
    if staged.returncode != 0:
        raise ProvenanceError(
            f"could not verify the Git index (exit {staged.returncode})"
        )

    porcelain = _run_git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all"
    ).stdout
    records = parse_porcelain_v1_z(porcelain)
    tracked_changes = [record for record in records if record["status"] != "??"]
    if tracked_changes:
        raise ProvenanceError(
            "tracked worktree/index changes violate the append-only campaign: "
            + json.dumps(tracked_changes, ensure_ascii=False, sort_keys=True)
        )
    short_branch = _decode(
        _run_git("status", "--short", "--branch", "--untracked-files=all").stdout
    )
    return {
        "branch": branch,
        "expected_branch": expected_branch,
        "head": head,
        "expected_head": expected_head,
        "index_has_staged_changes": False,
        "tracked_worktree_changes": [],
        "status_porcelain_v1": records,
        "status_porcelain_v1_z_sha256": hashlib.sha256(porcelain).hexdigest(),
        "status_short_branch": short_branch,
        "git_ignore_checks": _git_ignore_evidence(ignore_probe_paths),
        "campaign_source_test_doc_files_from_git_status": hash_campaign_files(records),
    }


def capture_checkpoint_integrity(
    before_csv: Path, checkpoint_dir: Path
) -> dict[str, Any]:
    before_rows = read_checkpoint_inventory(before_csv)
    current_rows = inventory_checkpoints(checkpoint_dir)
    after_rows = compare_checkpoint_inventories(before_rows, current_rows)
    return {
        "before_inventory": {
            "path": _project_relative(before_csv),
            **hash_regular_file(before_csv),
        },
        "checkpoint_root": _project_relative(checkpoint_dir),
        "checkpoint_count": len(after_rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in after_rows),
        "comparison_policy": (
            "Fail unless path set, SHA-256, size_bytes, and mtime_ns exactly match "
            "the before inventory. SHA-256 and size establish byte identity; mtime_ns "
            "is an additional historical-integrity gate."
        ),
        "integrity_pass": True,
        "after_rows": after_rows,
    }


def _default_dataset_roots() -> list[Path]:
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.is_dir():
        raise ProvenanceError(f"data root is missing: {data_dir}")
    return sorted(
        (path for path in data_dir.iterdir() if path.name != ".gitkeep"),
        key=lambda path: path.name,
    )


def _summarize_dataset_root(path: Path) -> dict[str, Any]:
    data_dir = (PROJECT_ROOT / "data").resolve()
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(data_dir)
    except ValueError as exc:
        raise ProvenanceError(f"dataset root is outside data/: {path}") from exc
    root_details = path.lstat()
    if stat.S_ISLNK(root_details.st_mode):
        raise ProvenanceError(f"dataset root cannot be a symbolic link: {path}")

    if stat.S_ISREG(root_details.st_mode):
        return {
            "path": _project_relative(path),
            "absolute_path": str(resolved),
            "kind": "file",
            "root_mtime_ns": root_details.st_mtime_ns,
            "regular_file_count": 1,
            "directory_count": 0,
            "symbolic_link_count": 0,
            "other_entry_count": 0,
            "total_regular_file_bytes": root_details.st_size,
            "latest_member_mtime_ns": root_details.st_mtime_ns,
            "content_hashing_performed": False,
        }
    if not stat.S_ISDIR(root_details.st_mode):
        raise ProvenanceError(f"dataset root is neither file nor directory: {path}")

    file_count = 0
    directory_count = 1
    symlink_count = 0
    other_count = 0
    total_bytes = 0
    latest_mtime = root_details.st_mtime_ns
    for directory, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames.sort()
        filenames.sort()
        base = Path(directory)
        for name in dirnames:
            details = (base / name).lstat()
            latest_mtime = max(latest_mtime, details.st_mtime_ns)
            if stat.S_ISLNK(details.st_mode):
                symlink_count += 1
            elif stat.S_ISDIR(details.st_mode):
                directory_count += 1
            else:
                other_count += 1
        for name in filenames:
            details = (base / name).lstat()
            latest_mtime = max(latest_mtime, details.st_mtime_ns)
            if stat.S_ISLNK(details.st_mode):
                symlink_count += 1
            elif stat.S_ISREG(details.st_mode):
                file_count += 1
                total_bytes += details.st_size
            else:
                other_count += 1
    return {
        "path": _project_relative(path),
        "absolute_path": str(resolved),
        "kind": "directory",
        "root_mtime_ns": root_details.st_mtime_ns,
        "regular_file_count": file_count,
        "directory_count": directory_count,
        "symbolic_link_count": symlink_count,
        "other_entry_count": other_count,
        "total_regular_file_bytes": total_bytes,
        "latest_member_mtime_ns": latest_mtime,
        "content_hashing_performed": False,
    }


def enumerate_dataset_roots(values: Sequence[str]) -> list[dict[str, Any]]:
    roots = [_resolve_argument(value) for value in values] if values else _default_dataset_roots()
    unique: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise ProvenanceError(f"dataset root does not exist: {root}")
        unique[str(root.resolve(strict=True))] = root
    return [
        _summarize_dataset_root(unique[key])
        for key in sorted(unique)
    ]


def _package_snapshot() -> dict[str, Any]:
    package_names = (
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-image",
        "h5py",
        "requests",
        "tqdm",
        "PyYAML",
        "astropy",
        "pyarrow",
        "sep",
        "photutils",
        "torch",
        "torchvision",
    )
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    availability = {
        name: importlib.util.find_spec(name) is not None
        for name in ("astropy", "pyarrow", "sep", "photutils", "torch")
    }
    torch_runtime: dict[str, Any] = {"available": availability["torch"]}
    if availability["torch"]:
        try:
            import torch

            torch_runtime.update(
                {
                    "version": torch.__version__,
                    "mps_built": bool(torch.backends.mps.is_built()),
                    "mps_available": bool(torch.backends.mps.is_available()),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "neural_inference_or_training_performed_by_finalizer": False,
                }
            )
        except Exception as exc:
            torch_runtime["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "package_versions": versions,
        "scientific_package_availability": availability,
        "pytorch_runtime": torch_runtime,
    }


def _disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _validate_run_dir(run_dir: Path) -> tuple[Path, Path, Path]:
    runs_root = DEFAULT_RUN_DIR.resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ProvenanceError(f"run directory must be an existing real directory: {run_dir}")
    try:
        run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ProvenanceError(f"run directory is outside outputs/runs: {run_dir}") from exc
    if not run_dir.name.startswith("dr10_foundation_"):
        raise ProvenanceError(f"unexpected campaign run directory name: {run_dir.name}")
    logs_dir = run_dir / "logs"
    tables_dir = run_dir / "tables"
    if not logs_dir.is_dir() or logs_dir.is_symlink():
        raise ProvenanceError(f"missing real logs directory: {logs_dir}")
    if not tables_dir.is_dir() or tables_dir.is_symlink():
        raise ProvenanceError(f"missing real tables directory: {tables_dir}")
    return logs_dir, tables_dir, tables_dir / "checkpoint_inventory_before.csv"


def _exclusive_write(path: Path, content: str) -> None:
    """Create and fsync a file without any overwrite or cleanup behavior."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ProvenanceError(f"refusing to overwrite existing output: {path}") from exc
    encoded = content.encode("utf-8")
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        raise ProvenanceError(
            f"exclusive write failed; file is retained for forensic inspection: {path}: {exc}"
        ) from exc


def _require_outputs_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise ProvenanceError(
            "exclusive finalization outputs already exist; no overwrite is permitted: "
            + ", ".join(existing)
        )


def _assert_reverified(label: str, first: Any, second: Any) -> None:
    if first != second:
        raise ProvenanceError(f"{label} changed between initial audit and final recheck")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing dr10_foundation run")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--smoke-fits", default=str(DEFAULT_SMOKE_FITS))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--expected-branch", default=EXPECTED_BRANCH)
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    parser.add_argument("--expected-catalog-md5", default=EXPECTED_CATALOG_MD5)
    parser.add_argument("--expected-catalog-sha256", default=EXPECTED_CATALOG_SHA256)
    parser.add_argument("--expected-smoke-sha256", default=EXPECTED_SMOKE_SHA256)
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="Existing path under data/ to enumerate; repeatable (default: data/* roots)",
    )
    parser.add_argument(
        "--provenance-caveat",
        action="append",
        default=[],
        help="Historical provenance limitation recorded verbatim; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    arguments = build_parser().parse_args(argv)
    caveats = validate_provenance_caveats(arguments.provenance_caveat)

    run_dir = _resolve_argument(arguments.run_dir)
    catalog = _resolve_argument(arguments.catalog)
    smoke_fits = _resolve_argument(arguments.smoke_fits)
    checkpoint_dir = _resolve_argument(arguments.checkpoint_dir)
    logs_dir, tables_dir, before_csv = _validate_run_dir(run_dir)
    provenance_json = logs_dir / "input_provenance.json"
    after_csv = tables_dir / "checkpoint_inventory_after.csv"
    _require_outputs_absent((provenance_json, after_csv))

    dataset_roots = enumerate_dataset_roots(arguments.dataset_root)
    ignore_probes = [catalog, smoke_fits, run_dir / "logs/input_provenance.json"]
    ignore_probes.extend(
        PROJECT_ROOT / root["path"] / ".provenance_gitignore_probe"
        if root["kind"] == "directory"
        else PROJECT_ROOT / root["path"]
        for root in dataset_roots
    )

    # Phase 1: establish a complete, passing candidate snapshot.
    initial_repository = capture_repository(
        arguments.expected_branch, arguments.expected_head, ignore_probes
    )
    initial_inputs = capture_inputs(
        catalog,
        smoke_fits,
        arguments.expected_catalog_md5,
        arguments.expected_catalog_sha256,
        arguments.expected_smoke_sha256,
    )
    initial_checkpoints = capture_checkpoint_integrity(before_csv, checkpoint_dir)

    # Environment inspection is read-only and occurs before the final recheck.
    package_snapshot = _package_snapshot()

    # Phase 2: repeat all mutable gates and all hashes before creating any file.
    final_repository = capture_repository(
        arguments.expected_branch, arguments.expected_head, ignore_probes
    )
    final_inputs = capture_inputs(
        catalog,
        smoke_fits,
        arguments.expected_catalog_md5,
        arguments.expected_catalog_sha256,
        arguments.expected_smoke_sha256,
    )
    final_checkpoints = capture_checkpoint_integrity(before_csv, checkpoint_dir)
    _assert_reverified("repository/source snapshot", initial_repository, final_repository)
    _assert_reverified("required input snapshot", initial_inputs, final_inputs)
    _assert_reverified(
        "checkpoint inventory", initial_checkpoints, final_checkpoints
    )
    _require_outputs_absent((provenance_json, after_csv))

    after_csv_text = serialize_checkpoint_csv(final_checkpoints["after_rows"])
    after_csv_sha256 = hashlib.sha256(after_csv_text.encode("utf-8")).hexdigest()
    finished_utc = datetime.now(timezone.utc).isoformat()
    finished_local = datetime.now().astimezone().isoformat()
    finalizer_elapsed = time.monotonic() - started_monotonic
    provenance = {
        "schema_name": "thayer_select_dr10_input_provenance",
        "schema_version": 1,
        "result": "PASS",
        "append_only_semantics": {
            "outputs_created_exclusively": [
                _project_relative(provenance_json.parent) + "/input_provenance.json",
                _project_relative(after_csv.parent) + "/checkpoint_inventory_after.csv",
            ],
            "overwrite_permitted": False,
            "staging_performed": False,
            "deletion_performed": False,
            "two_phase_reverification_pass": True,
        },
        "campaign": {
            "run_directory": _project_relative(run_dir),
            "absolute_run_directory": str(run_dir),
            "invocation": shlex.join([sys.executable, str(Path(__file__)), *(argv or sys.argv[1:])]),
            "working_directory": str(Path.cwd()),
            "finalizer_started_utc": started_utc,
            "snapshot_finished_utc": finished_utc,
            "snapshot_finished_local": finished_local,
            "finalizer_runtime_seconds": finalizer_elapsed,
        },
        "repository_final": final_repository,
        "required_inputs_final": final_inputs,
        "dataset_roots_final": dataset_roots,
        "checkpoint_integrity": {
            key: value
            for key, value in final_checkpoints.items()
            if key != "after_rows"
        }
        | {
            "after_inventory": {
                "path": _project_relative(after_csv.parent)
                + "/checkpoint_inventory_after.csv",
                "sha256_of_serialized_csv": after_csv_sha256,
                "row_count": len(final_checkpoints["after_rows"]),
            }
        },
        "campaign_provenance_caveats_verbatim": [
            {"ordinal": index, "text": caveat, "source": "--provenance-caveat"}
            for index, caveat in enumerate(caveats, start=1)
        ],
        "runtime_final": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "shell": os.environ.get("SHELL"),
            **package_snapshot,
            "disk": _disk_snapshot(PROJECT_ROOT),
        },
        "scope_notes": {
            "dataset_root_content_hashing": (
                "Dataset roots are enumerated by filesystem metadata only. The two "
                "required inputs are fully hashed; every checkpoint and every Git-status "
                "campaign source/test/doc file is fully hashed."
            ),
            "model_execution": "No neural training or inference is performed by this script.",
        },
    }
    provenance_text = json.dumps(
        provenance, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"

    # No file exists before this point. Both writes use O_EXCL and never clean up.
    _exclusive_write(after_csv, after_csv_text)
    _exclusive_write(provenance_json, provenance_text)
    print(
        json.dumps(
            {
                "result": "PASS",
                "input_provenance": str(provenance_json),
                "checkpoint_inventory_after": str(after_csv),
                "checkpoint_count": final_checkpoints["checkpoint_count"],
                "campaign_file_hash_count": len(
                    final_repository["campaign_source_test_doc_files_from_git_status"]
                ),
                "provenance_caveat_count": len(caveats),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvenanceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
