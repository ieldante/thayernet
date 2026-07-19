from __future__ import annotations

import csv
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run() -> Path:
    value = os.environ.get("THAYER_FAMILY_E1_RUN")
    assert value, "THAYER_FAMILY_E1_RUN is required"
    path = (REPO / value).resolve()
    assert path.parent == (REPO / "outputs/runs").resolve()
    assert path.name.startswith("thayer_family_e1_v0_")
    return path


def test_preregistration_preceded_model_work() -> None:
    path = run()
    freeze = json.loads((path / "logs/preregistration_complete.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_MODEL_CONSTRUCTION_OR_FITTING"
    assert freeze["model_construction_count"] == 0
    assert freeze["optimizer_construction_count"] == 0
    assert freeze["training_tensor_load_count"] == 0


def test_architecture_and_objective_passed() -> None:
    path = run()
    architecture = json.loads((path / "architecture/architecture_manifest.json").read_text())
    objective = json.loads((path / "objective_audit/objective_alignment_summary.json").read_text())
    assert architecture["trainable_parameters"] == 1_162_662
    assert architecture["architecture_variants_constructed"] == 1
    assert architecture["source_mapping"] == "in_forward_relu"
    assert objective["status"] == "PASS"
    assert objective["truth_stationary_minimum"] is True
    assert objective["compromise_beats_truth"] is False


def test_micro_stop_is_authoritative() -> None:
    path = run()
    stop = json.loads((path / "logs/micro_overfit_complete.json").read_text())
    assert stop == {
        "cpu_fallback": False,
        "difficult_pass": False,
        "full_training_authorized": False,
        "mixed_eight_pass": False,
        "ordinary_pass": True,
        "status": "FAIL",
    }
    with (path / "tables/micro_overfit_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["condition"] == "ordinary_one_scene" and rows[0]["pass"] == "True"
    assert rows[1]["pass"] == "False" and rows[2]["pass"] == "False"


def test_no_prohibited_downstream_artifacts_after_stop() -> None:
    path = run()
    assert not (path / "logs/primary_training_complete.json").exists()
    assert not (path / "logs/oof_fold_training_complete.json").exists()
    assert not list((path / "checkpoints").glob("*.pth"))
    assert not list((path / "oof_outputs").glob("*.h5"))
    assert not list((path / "safety_labels").glob("*.csv"))
    assert not list((path / "bootstrap").glob("*replicate*.csv"))


def test_source_groups_and_folds_are_leak_free() -> None:
    path = run()
    with (path / "tables/source_group_leakage_tests.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["pass"] == "True" for row in rows)
    assert next(row for row in rows if row["check"] == "maximum_cross_fold_group_overlap")["observed"] == "0"


def test_physical_contract_and_mps() -> None:
    path = run()
    physical = json.loads((path / "physical_contract/mps_physical_preflight.json").read_text())
    assert physical["status"] == "PASS"
    assert physical["device"] == "mps"
    assert physical["cpu_fallback"] is False
    assert physical["source_negative_fraction"] == 0.0
    assert physical["conservation_error"] <= physical["conservation_tolerance"]
