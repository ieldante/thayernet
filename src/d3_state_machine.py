"""Append-only semantic artifact adapter for the Thayer-D3 policy engine."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from src.d3_control_policy import PolicyContractError, SEMANTIC_STATES, SemanticCandidate


SELECTION_STATES = frozenset(("lowest_objective", "closest_to_d1"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SemanticStateAdapter:
    """Persist semantic dummy payloads without silent omission or overwrite."""

    def __init__(self, root: Path, allow_existing_root: bool = False):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=allow_existing_root)
        self.payload_root = self.root / "semantic_state_payloads"
        self.payload_root.mkdir(exist_ok=False)
        self.manifest_path = self.root / "d3_state_machine_manifest.json"
        self.manifest: dict[str, Any] = {
            "schema_version": "thayer-d3-semantic-state-manifest-v3",
            "revision": 0,
            "states": {
                state: {
                    "status": "not_reached",
                    "reason": "campaign_not_finalized",
                    "terminal_campaign_status": "UNSET",
                    "last_eligible_evaluation_index": -1,
                    "occurrences": [],
                    "selected": None,
                }
                for state in SEMANTIC_STATES
            },
            "finalized": False,
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        revision = int(self.manifest["revision"]) + 1
        self.manifest["revision"] = revision
        temporary = self.root / f".manifest-revision-{revision:06d}.tmp"
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.manifest_path)

    @staticmethod
    def _validate(candidate: SemanticCandidate) -> None:
        if candidate.state not in SEMANTIC_STATES:
            raise PolicyContractError("state.semantic_persistence", f"unknown semantic state: {candidate.state}")
        if candidate.evaluation_index < 0 or candidate.step_index < 0:
            raise PolicyContractError("state.semantic_persistence", "negative state index")
        if len(candidate.payload) == 0:
            raise PolicyContractError("state.semantic_persistence", "empty state payload")
        if len(candidate.semantic_members) == 0 or any(not isinstance(value, str) or "." not in value for value in candidate.semantic_members):
            raise PolicyContractError("state.semantic_persistence", "semantic member names are required")
        if candidate.state == "lowest_objective" and (candidate.objective is None or not math.isfinite(candidate.objective)):
            raise PolicyContractError("state.semantic_persistence", "lowest_objective requires a finite objective")
        if candidate.state == "closest_to_d1" and (candidate.distance_to_d1 is None or not math.isfinite(candidate.distance_to_d1)):
            raise PolicyContractError("state.semantic_persistence", "closest_to_d1 requires a finite distance")

    @staticmethod
    def _is_better(state: str, record: Mapping[str, Any], selected: Mapping[str, Any] | None) -> bool:
        if selected is None:
            return True
        metric = "objective" if state == "lowest_objective" else "distance_to_d1"
        candidate_value = float(record[metric])
        selected_value = float(selected[metric])
        if candidate_value < selected_value:
            return True
        if candidate_value > selected_value:
            return False
        candidate_index = int(record["evaluation_index"])
        selected_index = int(selected["evaluation_index"])
        if candidate_index < selected_index:
            return True
        if candidate_index > selected_index:
            return False
        return str(record["payload_sha256"]) < str(selected["payload_sha256"])

    def persist(self, candidate: SemanticCandidate) -> dict[str, Any]:
        """Persist one reached state using exclusive payload creation."""

        if self.manifest["finalized"] is True:
            raise FileExistsError("semantic state manifest already finalized")
        self._validate(candidate)
        entry = self.manifest["states"][candidate.state]
        if candidate.state not in SELECTION_STATES and entry["status"] == "reached":
            raise FileExistsError(f"single-occurrence semantic state already reached: {candidate.state}")
        payload_hash = sha256_bytes(candidate.payload)
        filename = f"{candidate.state}__eval_{candidate.evaluation_index:06d}__{payload_hash[:12]}.bin"
        relative = Path("semantic_state_payloads") / filename
        path = self.root / relative
        with path.open("xb") as handle:
            handle.write(candidate.payload)
            handle.flush()
            os.fsync(handle.fileno())
        record = {
            "state": candidate.state,
            "evaluation_index": candidate.evaluation_index,
            "step_index": candidate.step_index,
            "path": str(relative),
            "bytes": len(candidate.payload),
            "payload_sha256": payload_hash,
            "scalar_metrics": dict(candidate.scalar_metrics),
            "optimizer_state_sha256": candidate.optimizer_state_sha256,
            "assignment": dict(candidate.assignment),
            "event": dict(candidate.event),
            "terminal_status": candidate.terminal_status,
            "objective": candidate.objective,
            "distance_to_d1": candidate.distance_to_d1,
            "semantic_members": list(candidate.semantic_members),
        }
        entry["status"] = "reached"
        entry["reason"] = None
        entry["terminal_campaign_status"] = candidate.terminal_status
        entry["last_eligible_evaluation_index"] = candidate.evaluation_index
        entry["occurrences"].append(record)
        if candidate.state in SELECTION_STATES:
            if self._is_better(candidate.state, record, entry["selected"]):
                entry["selected"] = record
        else:
            entry["selected"] = record
        self._write_manifest()
        return record

    def finalize(self, terminal_campaign_status: str, last_eligible_evaluation_index: int, reasons: Mapping[str, str]) -> dict[str, Any]:
        """Write explicit not-reached records and freeze the final manifest."""

        if self.manifest["finalized"] is True:
            raise FileExistsError("semantic state manifest already finalized")
        if self.manifest["states"]["final"]["status"] != "reached":
            raise PolicyContractError("state.semantic_persistence", "final state must be reached before finalization")
        for state in SEMANTIC_STATES:
            entry = self.manifest["states"][state]
            if entry["status"] == "not_reached":
                entry["reason"] = reasons.get(state, "trigger_not_observed")
                entry["terminal_campaign_status"] = terminal_campaign_status
                entry["last_eligible_evaluation_index"] = last_eligible_evaluation_index
        self.manifest["finalized"] = True
        self.manifest["terminal_campaign_status"] = terminal_campaign_status
        self.manifest["last_eligible_evaluation_index"] = last_eligible_evaluation_index
        self._write_manifest()
        return self.manifest


def replay_manifest(root: Path) -> dict[str, Any]:
    """Validate a finalized manifest and every reached payload hash."""

    root = root.resolve()
    manifest_path = root / "d3_state_machine_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "thayer-d3-semantic-state-manifest-v3":
        raise PolicyContractError("persistence.artifact_integrity", "state manifest schema mismatch")
    if manifest.get("finalized") is not True:
        raise PolicyContractError("persistence.artifact_integrity", "state manifest is not finalized")
    if set(manifest["states"]) != set(SEMANTIC_STATES):
        raise PolicyContractError("state.semantic_persistence", "semantic state set mismatch")
    reached = 0
    not_reached = 0
    for state in SEMANTIC_STATES:
        entry = manifest["states"][state]
        if entry["status"] == "reached":
            reached += 1
            if entry["selected"] is None:
                raise PolicyContractError("state.semantic_persistence", f"selected state record missing: {state}")
            for occurrence in entry["occurrences"]:
                path = root / occurrence["path"]
                if not path.is_file() or path.stat().st_size != occurrence["bytes"] or sha256_file(path) != occurrence["payload_sha256"]:
                    raise PolicyContractError("persistence.artifact_integrity", f"payload replay mismatch: {state}")
                if not occurrence["semantic_members"]:
                    raise PolicyContractError("state.semantic_persistence", f"semantic member names missing: {state}")
        elif entry["status"] == "not_reached":
            not_reached += 1
            for field in ("reason", "terminal_campaign_status", "last_eligible_evaluation_index"):
                if entry.get(field) is None:
                    raise PolicyContractError("state.semantic_persistence", f"not-reached field missing: {state}.{field}")
        else:
            raise PolicyContractError("state.semantic_persistence", f"unknown state status: {state}")
    return {
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "reached_state_count": reached,
        "not_reached_state_count": not_reached,
        "revision": manifest["revision"],
    }
