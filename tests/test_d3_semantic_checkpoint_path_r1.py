"""Regressions for semantic-state checkpoint path resolution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.d3_semantic_checkpoint_path_r1 import (
    SemanticCheckpointPathError,
    resolve_semantic_checkpoint_path,
)


def _checkpoint(
    root: Path, relative: str = "checkpoints/evaluation_step_0000.pt"
) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint-bytes")
    return path


def test_relative_checkpoint_is_anchored_at_candidate_execution_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution_root = tmp_path / "candidate_001/execution_root"
    checkpoint = _checkpoint(execution_root)
    repository_cwd = tmp_path / "repository"
    repository_cwd.mkdir()
    monkeypatch.chdir(repository_cwd)

    result = resolve_semantic_checkpoint_path(
        Path("checkpoints/evaluation_step_0000.pt"), execution_root
    )

    assert result.path == checkpoint.resolve()
    assert result.relative_path == Path("checkpoints/evaluation_step_0000.pt")
    assert result.path != (repository_cwd / result.relative_path).resolve()


def test_nested_relative_checkpoint_is_supported(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution_root"
    checkpoint = _checkpoint(execution_root, "checkpoints/nested/step.pt")

    result = resolve_semantic_checkpoint_path(
        "checkpoints/nested/step.pt", execution_root
    )

    assert result.path == checkpoint.resolve()
    assert result.relative_path == Path("checkpoints/nested/step.pt")


def test_contained_absolute_checkpoint_is_supported(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution_root"
    checkpoint = _checkpoint(execution_root)

    result = resolve_semantic_checkpoint_path(checkpoint.resolve(), execution_root)

    assert result.path == checkpoint.resolve()
    assert result.relative_path == Path("checkpoints/evaluation_step_0000.pt")


def test_absolute_checkpoint_can_be_disabled(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution_root"
    checkpoint = _checkpoint(execution_root)

    with pytest.raises(SemanticCheckpointPathError, match="absolute.*not supported"):
        resolve_semantic_checkpoint_path(
            checkpoint, execution_root, absolute_paths_supported=False
        )


def test_traversal_escape_is_rejected(tmp_path: Path) -> None:
    execution_root = tmp_path / "campaign/candidate/execution_root"
    _checkpoint(execution_root)
    outside = tmp_path / "campaign/outside-campaign/state.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(SemanticCheckpointPathError, match="escapes"):
        resolve_semantic_checkpoint_path(
            "../../../outside-campaign/state.json", execution_root
        )


def test_absolute_wrong_root_is_rejected(tmp_path: Path) -> None:
    execution_root = tmp_path / "candidate/execution_root"
    _checkpoint(execution_root)
    wrong_root = tmp_path / "repository/checkpoints/evaluation_step_0000.pt"
    wrong_root.parent.mkdir(parents=True)
    wrong_root.write_bytes(b"wrong-root")

    with pytest.raises(SemanticCheckpointPathError, match="escapes"):
        resolve_semantic_checkpoint_path(wrong_root, execution_root)


def test_symlink_escape_is_rejected_after_canonicalization(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution_root"
    execution_root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    link = execution_root / "checkpoints/linked.pt"
    link.parent.mkdir()
    link.symlink_to(outside)

    with pytest.raises(SemanticCheckpointPathError, match="escapes"):
        resolve_semantic_checkpoint_path(
            link.relative_to(execution_root), execution_root
        )


def test_missing_checkpoint_is_rejected_without_parent_creation(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution_root"
    execution_root.mkdir()
    missing = execution_root / "checkpoints/new/step.pt"

    with pytest.raises(SemanticCheckpointPathError, match="does not exist"):
        resolve_semantic_checkpoint_path(
            missing.relative_to(execution_root), execution_root
        )

    assert not missing.parent.exists()


def test_resolution_does_not_mutate_existing_checkpoint(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution_root"
    checkpoint = _checkpoint(execution_root)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    first = resolve_semantic_checkpoint_path(checkpoint, execution_root)
    second = resolve_semantic_checkpoint_path(first.relative_path, execution_root)

    assert first == second
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_atomic_semantic_manifest_and_parent_creation_remain_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.d3_control_policy import SemanticCandidate
    from src.d3_state_machine import SemanticStateAdapter

    monkeypatch.delenv("D3_PRECREATED_OUTPUT_TREE", raising=False)
    execution_root = tmp_path / "candidate/execution_root"
    checkpoint = _checkpoint(execution_root)
    resolved = resolve_semantic_checkpoint_path(
        checkpoint.relative_to(execution_root), execution_root
    )
    state_root = execution_root / "semantic_states"
    adapter = SemanticStateAdapter(state_root)
    payload = json.dumps(
        {
            "state": "initial",
            "checkpoint": str(resolved.relative_path),
            "checkpoint_sha256": hashlib.sha256(
                resolved.path.read_bytes()
            ).hexdigest(),
        },
        sort_keys=True,
    ).encode("utf-8")

    adapter.persist(
        SemanticCandidate(
            state="initial",
            evaluation_index=0,
            step_index=0,
            payload=payload,
            scalar_metrics={"objective": 1.0},
            optimizer_state_sha256="0" * 64,
            assignment={"prompt_a": "expert_1", "prompt_b": "expert_2"},
            event={"code": "CONTINUE"},
            terminal_status="CONTINUE",
            objective=None,
            distance_to_d1=None,
            semantic_members=("prompt_a.expert_1.requested",),
        )
    )

    assert state_root.is_dir()
    assert (state_root / "semantic_state_payloads").is_dir()
    assert adapter.manifest_path.is_file()
    assert not list(state_root.glob(".manifest-revision-*.tmp"))
    with pytest.raises(FileExistsError, match="already reached"):
        adapter.persist(
            SemanticCandidate(
                state="initial",
                evaluation_index=1,
                step_index=1,
                payload=payload,
                scalar_metrics={"objective": 1.0},
                optimizer_state_sha256="0" * 64,
                assignment={"prompt_a": "expert_1", "prompt_b": "expert_2"},
                event={"code": "CONTINUE"},
                terminal_status="CONTINUE",
                objective=None,
                distance_to_d1=None,
                semantic_members=("prompt_a.expert_1.requested",),
            )
        )


def test_repository_cwd_does_not_affect_resolution(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution_root"
    checkpoint = _checkpoint(execution_root)
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        first = resolve_semantic_checkpoint_path(
            "checkpoints/evaluation_step_0000.pt", execution_root
        )
        os.chdir(execution_root)
        second = resolve_semantic_checkpoint_path(
            "checkpoints/evaluation_step_0000.pt", execution_root
        )
    finally:
        os.chdir(original_cwd)

    assert first.path == second.path == checkpoint.resolve()
