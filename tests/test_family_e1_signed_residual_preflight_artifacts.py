from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUN = (
    REPO
    / "outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340"
)
EXPECTED_PREREG = (
    "be546f7f1aa2ec04f1a76f84bc5305c87521d5b89331c681dc3cdf18a5293d3b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_hash_and_boundary() -> None:
    path = RUN / "preregistration/signed_noise_residual_physical_contract_preflight.md"
    record = json.loads(
        (
            RUN
            / "preregistration/signed_noise_residual_physical_contract_preflight.sha256.json"
        ).read_text()
    )
    assert sha256(path) == EXPECTED_PREREG == record["sha256"]
    assert not record["source_tensors_loaded_in_campaign"]
    assert not record["model_constructed"]
    assert not record["optimizer_constructed"]


def test_all_partition_gates_pass() -> None:
    result = json.loads((RUN / "physical_contract/full_preflight.json").read_text())
    assert result["status"] == "SIGNED_NOISE_RESIDUAL_CONTRACT_PASS"
    assert result["synthetic_mps"]["status"] == "PASS"
    assert result["all_frozen_target_representability_gates_pass"]
    assert [row["episodes"] for row in result["partitions"]] == [10000, 2000, 2000]
    for row in result["partitions"]:
        assert row["status"] == "PASS"
        assert row["mapped_source_negative_count"] == 0
        assert row["target_negative_count"] == 0
        assert row["residual_negative_count"] > 0
        assert row["residual_positive_count"] > 0
        assert (
            row["requested_roundtrip_max_abs_error"]
            <= row["source_roundtrip_tolerance"]
        )
        assert (
            row["companion_roundtrip_max_abs_error"]
            <= row["source_roundtrip_tolerance"]
        )
        assert (
            row["float32_conservation_max_abs_error"]
            <= row["float32_conservation_tolerance"]
        )
        assert (
            row["float64_conservation_max_abs_error"]
            <= row["float64_conservation_tolerance"]
        )


def test_authorization_is_narrow_and_no_model_artifact_exists() -> None:
    decision = json.loads((RUN / "reports/frozen_core_decision.json").read_text())
    assert decision["outcome"] == "SIGNED_NOISE_RESIDUAL_CONTRACT_PASS"
    assert decision["next_campaign_authorized"]
    assert (
        decision["authorized_next_campaign"]
        == "Thayer-Family-E1-v0 — Nonnegative-Source Signed-Residual Model Eligibility"
    )
    assert not decision["thayer_audit_v1_authorized"]
    assert not decision["model_constructed"]
    assert decision["checkpoints_written"] == 0
    assert decision["reconstructions_written"] == 0
    assert decision["safety_labels_generated"] == 0


def test_gate_table_has_no_failure() -> None:
    gates = pd.read_csv(RUN / "tables/gate_results.csv")
    assert len(gates) == 23
    assert set(gates.status) == {"PASS"}
