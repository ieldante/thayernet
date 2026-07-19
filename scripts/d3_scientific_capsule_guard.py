#!/usr/bin/env python3
"""Exact-path and small-payload guard for the Thayer-D3C capsule campaign."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


class CapsuleAccessViolation(RuntimeError):
    """Raised when capsule construction attempts an unregistered access."""


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _scalar_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_scalar_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_scalar_count(item) for item in value)
    return 1


def _rank(value: Any) -> int:
    if isinstance(value, dict):
        return max((_rank(item) for item in value.values()), default=0)
    if not isinstance(value, (list, tuple)):
        return 0
    if not value:
        return 1
    return 1 + max(_rank(item) for item in value)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def validate_small_payload(
    value: Any,
    *,
    max_scalars: int = 64,
    allow_mapping_rank2: bool = False,
) -> tuple[int, int]:
    """Return scalar count and rank or reject an oversized/nonfinite payload."""

    scalar_count = _scalar_count(value)
    rank = _rank(value)
    if scalar_count > max_scalars:
        raise CapsuleAccessViolation(
            f"small-payload scalar limit exceeded: {scalar_count}>{max_scalars}"
        )
    if rank > (2 if allow_mapping_rank2 else 1):
        raise CapsuleAccessViolation(f"small-payload rank limit exceeded: {rank}")
    if not _finite(value):
        raise CapsuleAccessViolation("nonfinite scientific metadata payload")
    return scalar_count, rank


class ExactPathGuard:
    """Permit only registered regular files and predeclared metadata keys."""

    def __init__(
        self,
        repo: Path,
        allowed_paths: Iterable[Path],
        log_path: Path,
        *,
        max_scalars: int = 64,
    ) -> None:
        self.repo = repo.resolve()
        self.allowed = {path.resolve() for path in allowed_paths}
        self.log_path = log_path.resolve()
        self.max_scalars = max_scalars
        self.events: list[dict[str, Any]] = []

    def _record(self, **event: Any) -> None:
        self.events.append({"timestamp_utc": _utcnow(), **event})

    def _require(self, path: Path, operation: str) -> Path:
        resolved = path.resolve()
        if resolved not in self.allowed:
            self._record(
                operation=operation,
                path=str(resolved),
                decision="BLOCK",
                reason="exact_path_not_allowlisted",
            )
            raise CapsuleAccessViolation(f"nonallowlisted exact path: {resolved}")
        if not resolved.is_file():
            self._record(
                operation=operation,
                path=str(resolved),
                decision="BLOCK",
                reason="not_regular_file",
            )
            raise CapsuleAccessViolation(f"not a regular file: {resolved}")
        return resolved

    @staticmethod
    def sha256_unlogged(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def file_metadata(self, path: Path, *, role: str) -> dict[str, Any]:
        resolved = self._require(path, "file_metadata")
        result = {
            "path": str(resolved.relative_to(self.repo)),
            "bytes": resolved.stat().st_size,
            "sha256": self.sha256_unlogged(resolved),
            "role": role,
            "deserialized": False,
        }
        self._record(
            operation="file_metadata",
            path=result["path"],
            decision="ALLOW",
            role=role,
            bytes=result["bytes"],
            sha256=result["sha256"],
            scientific_array_deserialized=False,
        )
        return result

    def read_text(self, path: Path, *, role: str) -> str:
        resolved = self._require(path, "read_text")
        text = resolved.read_text(encoding="utf-8")
        self._record(
            operation="read_text",
            path=str(resolved.relative_to(self.repo)),
            decision="ALLOW",
            role=role,
            bytes=len(text.encode("utf-8")),
            scientific_array_deserialized=False,
        )
        return text

    def read_json_fields(
        self,
        path: Path,
        *,
        fields: Iterable[str],
        role: str,
        allow_mapping_rank2: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved = self._require(path, "read_json_fields")
        requested = tuple(fields)
        if not requested:
            raise CapsuleAccessViolation("metadata JSON fields must be predeclared")
        raw = resolved.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise CapsuleAccessViolation("metadata JSON root must be an object")
        missing = [field for field in requested if field not in payload]
        if missing:
            raise CapsuleAccessViolation(f"missing predeclared metadata keys: {missing}")
        selected = {field: payload[field] for field in requested}
        scalar_count, rank = validate_small_payload(
            selected,
            max_scalars=self.max_scalars,
            allow_mapping_rank2=allow_mapping_rank2,
        )
        metadata = {
            "path": str(resolved.relative_to(self.repo)),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "requested_keys": list(requested),
            "root_key_count": len(payload),
            "selected_scalar_count": scalar_count,
            "selected_rank": rank,
            "scientific_array_deserialized": False,
        }
        self._record(
            operation="read_json_fields",
            path=metadata["path"],
            decision="ALLOW",
            role=role,
            keys=list(requested),
            scalar_count=scalar_count,
            rank=rank,
            scientific_array_deserialized=False,
        )
        return selected, metadata

    def blocked_probe(self, path: Path, *, reason: str) -> bool:
        try:
            self._require(path, "blocked_probe")
        except CapsuleAccessViolation:
            self.events[-1]["probe_reason"] = reason
            return True
        self._record(
            operation="blocked_probe",
            path=str(path.resolve()),
            decision="FAIL_OPEN",
            probe_reason=reason,
        )
        return False

    def write_log_fresh(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("x", encoding="utf-8", newline="\n") as handle:
            for event in self.events:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
