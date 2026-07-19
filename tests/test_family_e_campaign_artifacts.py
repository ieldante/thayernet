from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/runs/thayer_family_e_v0_20260714_195256"
EXPECTED_PREREG = (
    "256bffe3bc53b572b7596bba844f0afdbf4abf3c4cb1d8906fc0ad08663d8881"
)
UPSTREAM = {
    "training": REPO
    / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_training_scene_manifest.csv",
    "validation": REPO
    / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_r_validation_scene_manifest.csv",
    "calibration": REPO
    / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/manifests/v2_natural_calibration_scene_manifest.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector = pd.read_csv(RUN / f"manifests/{partition}_manifest.csv")
    upstream = pd.read_csv(UPSTREAM[partition])
    return selector, upstream.iloc[selector.upstream_index.astype(int)].reset_index(
        drop=True
    )


def test_preregistration_and_selector_hashes_are_frozen() -> None:
    assert (
        sha256(
            RUN
            / "preregistration/family_e_nonnegative_flux_conserving_eligibility.md"
        )
        == EXPECTED_PREREG
    )
    expected = {
        "training": "4a8768eaa70e1d3f5f7a29fd4035e994c9c6f1494d3553e6ac0f805c8e911bc1",
        "validation": "bc5c65ffab19baea38e37edcb4d5dabd15bae1c0266b7dfdaa749eba5c6c464d",
        "calibration": "70326c1835726677e5d98c50323329f919bcd405f0f379420987fcd97e20fa0c",
    }
    for partition, digest in expected.items():
        assert sha256(RUN / f"manifests/{partition}_manifest.csv") == digest


def test_source_partitions_and_pairs_are_disjoint() -> None:
    frames = {key: selected(key)[1] for key in UPSTREAM}
    groups = {
        key: set(frame.source_a_group) | set(frame.source_b_group)
        for key, frame in frames.items()
    }
    pairs = {
        key: set(zip(frame.source_a_group, frame.source_b_group))
        for key, frame in frames.items()
    }
    assert len(frames["training"]) == 10_000
    assert len(frames["validation"]) == 2_000
    assert len(frames["calibration"]) == 2_000
    for left, right in (
        ("training", "validation"),
        ("training", "calibration"),
        ("validation", "calibration"),
    ):
        assert not groups[left] & groups[right]
        assert not pairs[left] & pairs[right]
    assert all(not frame.duplicated(["source_a_group", "source_b_group"]).any() for frame in frames.values())


def test_connected_source_group_oof_assignment() -> None:
    selector, frame = selected("training")
    assert selector.oof_fold.value_counts().sort_index().tolist() == [2000] * 5
    fold_groups = {}
    for fold in range(5):
        rows = frame.loc[selector.oof_fold.eq(fold)]
        fold_groups[fold] = set(rows.source_a_group) | set(rows.source_b_group)
    for left in range(5):
        for right in range(left + 1, 5):
            assert not fold_groups[left] & fold_groups[right]


def test_fail_closed_decision_precedes_model_and_outputs() -> None:
    decision = json.loads((RUN / "reports/frozen_core_decision.json").read_text())
    stop = json.loads((RUN / "training/stop_record.json").read_text())
    assert decision["outcome"] == "DATA_OR_IMPLEMENTATION_FAILURE"
    assert not decision["thayer_audit_v1_authorized"]
    assert not stop["model_constructed"]
    assert stop["checkpoints_written"] == 0
    assert not list((RUN / "checkpoints").glob("*"))
