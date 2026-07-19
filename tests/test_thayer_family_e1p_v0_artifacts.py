from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / os.environ.get(
    "THAYER_FAMILY_E1P_RUN",
    "outputs/runs/thayer_family_e1p_v0_20260714_225228",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_preregistration_and_architecture_remained_frozen() -> None:
    freeze = json.loads((RUN / "logs/preregistration_complete.json").read_text())
    preregistration = REPO / freeze["path"]
    assert freeze["status"] == "FROZEN_BEFORE_MODEL_CONSTRUCTION_OR_FITTING"
    assert freeze["model_construction_count"] == 0
    assert freeze["optimizer_construction_count"] == 0
    assert freeze["training_tensor_load_count"] == 0
    assert sha256_file(preregistration) == freeze["sha256"]
    architecture = json.loads((RUN / "architecture/unchanged_architecture_contract.json").read_text())
    assert architecture["trainable_parameters"] == 1_162_662
    assert architecture["architecture_changes"] == 0
    assert architecture["parameter_changes"] == 0
    assert architecture["source_mapping"] == "in_forward_relu"


def test_only_the_two_paired_microsets_ran_and_identity_failed() -> None:
    results = pd.read_csv(RUN / "tables/micro_overfit_results.csv").set_index("condition")
    assert set(results.index) == {"difficult_one_scene", "mixed_eight_scene"}
    assert results.loc["difficult_one_scene", "scenes"] == 1
    assert results.loc["mixed_eight_scene", "scenes"] == 8
    assert results.loc["difficult_one_scene", "prompt_identity"] == 0.5
    assert results.loc["mixed_eight_scene", "prompt_identity"] == 0.5625
    assert not results.identity_pass.any()
    assert results.physical_pass.all()
    assert results.unchanged_micro_reconstruction_pass.all()
    status = json.loads((RUN / "logs/micro_overfit_complete.json").read_text())
    assert status["status"] == "FAIL"
    assert status["full_training_authorized"] is False
    assert status["validation_access_count"] == 0
    assert status["calibration_access_count"] == 0
    assert status["oof_outputs"] == 0
    assert status["safety_labels"] == 0
    assert status["auditor_models"] == 0


def test_paired_manifests_have_identical_observations_and_distinct_prompts() -> None:
    for condition, rows in (("difficult_one_scene", 1), ("mixed_eight_scene", 8)):
        manifest = pd.read_csv(RUN / f"manifests/{condition}_paired_scene_manifest.csv")
        assert len(manifest) == rows
        assert manifest.observation_sha256.nunique() == rows
        assert (manifest.prompt_a_sha256 != manifest.prompt_b_sha256).all()
        assert (manifest.prompt_l1_difference > 0).all()


def test_every_encoder_level_was_traced_without_numerical_prompt_loss() -> None:
    trace = pd.read_csv(RUN / "tables/layerwise_prompt_trace.csv")
    required = {
        "enc0_first",
        "enc0_second",
        "down0",
        "enc1",
        "down1",
        "enc2",
        "down2",
        "enc3",
    }
    for condition in ("difficult_one_scene", "mixed_eight_scene"):
        final = trace[(trace.condition == condition) & (trace.phase == "final")]
        assert required <= set(final.layer)
        assert (final.gradient_wrt_prompt_input_rms > 0).all()
        assert not final.numerically_indistinguishable.any()


def test_only_fresh_micro_states_exist() -> None:
    states = sorted((RUN / "micro_overfit").glob("*_final_state.pth"))
    assert [path.name for path in states] == [
        "difficult_one_scene_final_state.pth",
        "mixed_eight_scene_final_state.pth",
    ]
    assert not (RUN / "training").exists()
    assert not (RUN / "validation").exists()
    assert not (RUN / "calibration").exists()
    assert not (RUN / "oof_outputs").exists()
    assert not (RUN / "safety_labels").exists()
