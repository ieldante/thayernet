"""Fail-closed Thayer-CL output-contract and ordering checks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from src.canonical_tensor_hash import canonical_tensor_sha256


REPO = Path(__file__).resolve().parents[1]
RUN = (
    Path(os.environ["THAYER_CL_RUN_DIR"]).resolve()
    if "THAYER_CL_RUN_DIR" in os.environ
    else sorted((REPO / "outputs/runs").glob("thayer_capacity_ladder_*"))[-1]
)
FP = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_parameter_count(dec2: int, dec1: int) -> int:
    def block(in_channels: int, out_channels: int) -> int:
        return 9 * in_channels * out_channels + 9 * out_channels * out_channels + 6 * out_channels

    return block(96, dec2) + block(dec2 + 16, dec1) + 6 * dec1 + 6


def test_preregistration_precedes_per_scene_audit() -> None:
    freeze = json.loads((RUN / "preregistration/freeze_record.json").read_text())
    prereg = RUN / "preregistration/contract_compliant_decoder_capacity_ladder.md"
    started = RUN / "logs/per_scene_audit_started.json"
    assert freeze["per_scene_tensor_load_count"] == 0
    assert sha256(prereg) == freeze["preregistration_sha256"]
    assert prereg.stat().st_mtime_ns <= started.stat().st_mtime_ns


def test_thayer_fp_reproduction_passed() -> None:
    rows = read_csv(RUN / "tables/thayer_fp_reproduction.csv")
    assert len(rows) == 24
    assert all(row["pass"] == "True" for row in rows)


def test_projected_targets_are_finite_nonnegative_and_roundtrip() -> None:
    scales = np.asarray([611.9199829101562, 1805.8800048828125, 1854.199951171875], dtype=np.float32)
    path = FP / "projection_targets/projected_target_sets_final.h5"
    with h5py.File(path, "r") as handle:
        normalized = np.asarray(handle["targets_normalized"], dtype=np.float32)
        physical = np.asarray(handle["targets_physical"], dtype=np.float32)
    rebuilt = normalized * np.tile(scales, 2)[None, None, None, :, None, None]
    assert np.all(np.isfinite(normalized))
    assert np.all(np.isfinite(physical))
    assert float(physical.min()) >= 0
    assert np.array_equal(rebuilt, physical)


def test_projected_target_canonical_hashes_reproduce() -> None:
    rows = read_csv(FP / "tables/projected_target_hashes_final.csv")
    with h5py.File(FP / "projection_targets/projected_target_sets_final.h5", "r") as handle:
        physical = handle["targets_physical"]
        for row in rows:
            value = np.asarray(physical[int(row["scene"]), int(row["prompt"]), int(row["target_slot"])], dtype=np.float32)
            assert canonical_tensor_sha256(value) == row["canonical_sha256"]


def test_historical_identity_mapping_preserves_negative_sign() -> None:
    with h5py.File(FP / "micro_overfit/final_outputs.h5", "r") as handle:
        physical = np.asarray(handle["decompositions_physical"], dtype=np.float32)
    assert float(physical.min()) < 0
    assert float(np.mean(physical < 0)) > 0.43


def test_mapping_uniqueness_gate_failed_closed() -> None:
    rows = read_csv(RUN / "tables/output_mapping_uniqueness_audit.csv")
    selected = [row for row in rows if row["defined_by_frozen_contract"] == "True" and row["mathematically_admissible_for_new_contract"] == "True"]
    admissible = [row for row in rows if row["mathematically_admissible_for_new_contract"] == "True"]
    assert selected == []
    assert {row["mapping"] for row in admissible} == {
        "relu_as_neural_head_activation",
        "square_as_neural_head_mapping",
        "absolute_value_as_neural_head_mapping",
    }


def test_capacity_counts_were_frozen_without_model_construction() -> None:
    expected = {
        (32, 16): (46470, 165612),
        (80, 40): (176646, 425964),
        (160, 80): (554886, 1182444),
        (224, 112): (1002630, 2077932),
    }
    for widths, (expert, total) in expected.items():
        assert expert_parameter_count(*widths) == expert
        assert 72672 + 2 * expert == total


def test_stop_record_is_synchronous_and_blocks_ladder_outputs() -> None:
    stop_path = RUN / "logs/fail_closed_stop.json"
    complete_path = RUN / "logs/contract_audit_complete.json"
    stop = json.loads(stop_path.read_text())
    assert stop["reason"] == "NO_UNIQUE_CONTRACT_COMPLIANT_OUTPUT_MAPPING"
    assert stop["model_construction_count"] == 0
    assert stop["neural_optimizer_step_count"] == 0
    assert stop["capacity_ladder_authorized"] is False
    assert stop_path.stat().st_mtime_ns <= complete_path.stat().st_mtime_ns
    assert not any((RUN / "conditions").iterdir())
    assert not any((RUN / "checkpoints").iterdir())
    assert not any((RUN / "micro_overfit").iterdir())


def test_protected_access_and_input_leakage_are_zero() -> None:
    stop = json.loads((RUN / "logs/fail_closed_stop.json").read_text())
    provenance = json.loads((RUN / "logs/input_provenance.json").read_text())
    assert stop["atlas_access_count"] == 0
    assert stop["development_access_count"] == 0
    assert stop["lockbox_access_count"] == 0
    assert provenance["staged_index"] == []


def test_historical_checkpoints_remain_unchanged() -> None:
    before = read_csv(RUN / "tables/checkpoint_inventory_before.csv")
    authoritative = read_csv(FP / "tables/checkpoint_inventory_after.csv")
    assert len(before) == len(authoritative)
    assert {row["path"] for row in before} == {row["path"] for row in authoritative}
    for row in before:
        path = REPO / row["path"]
        assert path.is_file()
        assert sha256(path) == row["sha256"]
