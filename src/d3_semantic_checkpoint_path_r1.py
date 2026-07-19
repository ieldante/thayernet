"""Resolve semantic-state checkpoint references inside one execution root.

Trajectory and checkpoint-inventory records store checkpoint paths relative to
the candidate execution root.  Semantic-state persistence uses the same
anchor.  Canonical containment is checked before a checkpoint is opened so a
relative traversal or an absolute path outside that root fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


class SemanticCheckpointPathError(ValueError):
    """Raised when a semantic-state checkpoint reference violates its root."""


@dataclass(frozen=True)
class ResolvedSemanticCheckpoint:
    """A canonical checkpoint path and its execution-root-relative record."""

    path: Path
    relative_path: Path
    execution_root: Path


def resolve_semantic_checkpoint_path(
    checkpoint_path: PathLike,
    authorized_execution_root: PathLike,
    *,
    absolute_paths_supported: bool = True,
    must_exist: bool = True,
) -> ResolvedSemanticCheckpoint:
    """Resolve a checkpoint using the authoritative execution-root contract.

    Relative checkpoint references are anchored at ``authorized_execution_root``.
    Absolute references are supported only when explicitly enabled and when the
    canonical target remains contained by that root.
    """

    raw_checkpoint = Path(checkpoint_path)
    raw_root = Path(authorized_execution_root)
    if not str(raw_checkpoint):
        raise SemanticCheckpointPathError("checkpoint path must not be empty")

    try:
        execution_root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise SemanticCheckpointPathError(
            f"authorized execution root does not exist: {raw_root}"
        ) from exc
    if not execution_root.is_dir():
        raise SemanticCheckpointPathError(
            f"authorized execution root is not a directory: {execution_root}"
        )
    if raw_checkpoint.is_absolute() and not absolute_paths_supported:
        raise SemanticCheckpointPathError(
            f"absolute checkpoint paths are not supported: {raw_checkpoint}"
        )

    anchored = (
        raw_checkpoint
        if raw_checkpoint.is_absolute()
        else execution_root / raw_checkpoint
    )
    resolved = anchored.resolve(strict=False)

    try:
        relative = resolved.relative_to(execution_root)
    except ValueError as exc:
        raise SemanticCheckpointPathError(
            "checkpoint path escapes the authorized execution root: "
            f"{raw_checkpoint} -> {resolved}"
        ) from exc

    if must_exist and not resolved.is_file():
        raise SemanticCheckpointPathError(
            f"checkpoint path does not exist or is not a file: {resolved}"
        )
    return ResolvedSemanticCheckpoint(
        path=resolved,
        relative_path=relative,
        execution_root=execution_root,
    )


__all__ = [
    "ResolvedSemanticCheckpoint",
    "SemanticCheckpointPathError",
    "resolve_semantic_checkpoint_path",
]
