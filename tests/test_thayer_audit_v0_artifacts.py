import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from src.direct_catalog_safety_auditor import (
    PostAuditSafetyNetwork,
    PreAuditQueryNetwork,
    SCALAR_FEATURE_NAMES,
    trainable_parameter_count,
)


REPO = Path(__file__).resolve().parents[1]
RUN = Path(os.environ["THAYER_AUDIT_RUN_DIR"]).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(name: str) -> pd.DataFrame:
    return pd.read_csv(RUN / f"episodes/{name}_manifest.csv", dtype=str, keep_default_na=False)


def test_preregistration_precedes_episodes_and_is_intact():
    record = json.loads((RUN / "logs/preregistration_complete.json").read_text())
    assert record["status"] == "FROZEN_BEFORE_EPISODES_OR_FITTING"
    assert sha256_file(REPO / record["path"]) == record["sha256"]


def test_episode_schema_and_target_feature_exclusion():
    expected = {
        "pre_training": (4, False),
        "pre_validation": (4, False),
        "pre_calibration": (4, False),
        "post_training": (10, True),
        "post_validation": (10, True),
        "policy_validation": (10, True),
        "policy_calibration": (10, True),
    }
    for name, (channels, has_scalar) in expected.items():
        with h5py.File(RUN / f"episodes/{name}.h5", "r") as handle:
            assert set(handle) == ({"image", "label", "scalar", "catastrophic"} if has_scalar else {"image", "label"})
            assert handle["image"].shape[1:] == (channels, 60, 60)
            if has_scalar:
                assert handle["scalar"].shape[1] == len(SCALAR_FEATURE_NAMES)
        frame = manifest(name)
        assert (frame.truth_derived_inference_feature_count.astype(int) == 0).all()
        assert (frame.family_metadata_input == "False").all()


def test_oof_provenance_and_source_group_leakage():
    base = pd.read_csv(
        REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/development_scene_definitions.csv",
        dtype=str,
        keep_default_na=False,
    )
    used = set(base.loc[base.partition.isin(("training", "validation")), "source_a_group"])
    used |= set(base.loc[base.partition.isin(("training", "validation")), "source_b_group"])
    training = pd.concat((manifest("pre_training"), manifest("post_training")), ignore_index=True)
    assert (training.base_prediction_provenance == "OUT_OF_HISTORICAL_BASE_FOLD").all()
    assert not (set(training.source_a_group) | set(training.source_b_group)) & used
    validation = pd.concat((manifest("pre_validation"), manifest("post_validation")), ignore_index=True)
    calibration = manifest("pre_calibration")
    tg = set(training.source_a_group) | set(training.source_b_group)
    vg = set(validation.source_a_group) | set(validation.source_b_group)
    cg = set(calibration.source_a_group) | set(calibration.source_b_group)
    assert not tg & vg and not tg & cg and not vg & cg


def test_architectures_and_mps_execution():
    assert trainable_parameter_count(PreAuditQueryNetwork()) == 28_307
    assert trainable_parameter_count(PostAuditSafetyNetwork()) == 155_209
    assert torch.backends.mps.is_available()
    assert PreAuditQueryNetwork().to("mps")(torch.zeros(1, 4, 60, 60, device="mps")).device.type == "mps"


def test_calibration_threshold_policy_and_held_family_contracts():
    calibrators = json.loads((RUN / "calibration/calibrators.json").read_text())
    assert calibrators["pre_temperature"] > 0 and calibrators["post_temperature"] > 0
    threshold = json.loads((RUN / "thresholds/frozen_post_audit_threshold.json").read_text())
    assert threshold["status"] == "NO_FEASIBLE_THRESHOLD_FAIL_CLOSED"
    assert threshold["calibration_policy"]["accepted_coverage"] == 0.0
    held = pd.read_csv(RUN / "family_holdout/results.csv")
    assert held.status.iloc[0] == "UNRESOLVED_ONE_ELIGIBLE_FAMILY"
    bootstrap = pd.read_csv(RUN / "bootstrap/source_group_bootstrap_replicates.csv")
    assert len(bootstrap) == 300


def test_atlas_is_post_freeze_diagnostic_only():
    record = json.loads((RUN / "logs/atlas_diagnostic_complete.json").read_text())
    assert record["status"] == "PASS"
    assert record["post_freeze_only"] and not record["atlas_selection_use"]
    assert record["development_outcome_access_count"] == 0
    assert record["final_lockbox_outcome_access_count"] == 0


def test_csv_schema_and_historical_checkpoint_hashes():
    for path in RUN.rglob("*.csv"):
        pd.read_csv(path)
    before = pd.read_csv(RUN / "tables/checkpoint_inventory_before.csv", keep_default_na=False)
    assert all((REPO / row.relative_path).is_file() and sha256_file(REPO / row.relative_path) == row.sha256 for row in before.itertuples(index=False))


def test_readme_staged_index_and_access_markers():
    provenance = json.loads((RUN / "logs/input_provenance.json").read_text())
    assert sha256_file(REPO / "README.md") == provenance["readme_sha256"]
    assert provenance["development_outcome_access_count"] == 0
    assert provenance["final_lockbox_outcome_access_count"] == 0
    assert provenance["atlas_selection_access_count"] == 0
