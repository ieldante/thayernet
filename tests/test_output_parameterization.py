"""Definition and campaign-artifact tests for fixed-L0 Thayer-OP mappings."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.run_thayer_output_parameterization_micro import (
    hard_physical_set_loss,
    physical_direct_cost,
)
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.models_two_expert_decoder import parameter_count, warm_start_condition_c_encoder
from src.output_parameterization import (
    INITIAL_PHYSICAL_EPSILON,
    MAPPINGS,
    NUMERICAL_ZERO_TOLERANCE,
    ROUNDTRIP_PHYSICAL_ATOL,
    MappedThayerMixtureExperts,
    apply_output_mapping,
    decoder_parameter_count,
    encoder_tensor_sha256,
    freeze_encoder,
    initial_raw_bias,
    mapping_derivative,
    raw_inverse_witness,
)


REPO = Path(__file__).resolve().parents[1]
RUN = (
    Path(os.environ["THAYER_OP_RUN_DIR"]).resolve()
    if "THAYER_OP_RUN_DIR" in os.environ
    else (sorted((REPO / "outputs/runs").glob("thayer_output_parameterization_*"))[-1] if list((REPO / "outputs/runs").glob("thayer_output_parameterization_*")) else None)
)
FP = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"
CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
SCALES = torch.tensor([611.9199829101562, 1805.8800048828125, 1854.199951171875], dtype=torch.float32)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_mapping_definitions_are_exactly_the_three_preregistered_choices() -> None:
    assert MAPPINGS == ("relu", "square", "absolute")
    raw = torch.tensor([-2.0, 0.0, 3.0])
    assert torch.equal(apply_output_mapping(raw, "relu"), torch.tensor([0.0, 0.0, 3.0]))
    assert torch.equal(apply_output_mapping(raw, "square"), torch.tensor([4.0, 0.0, 9.0]))
    assert torch.equal(apply_output_mapping(raw, "absolute"), torch.tensor([2.0, 0.0, 3.0]))
    try:
        apply_output_mapping(raw, "softplus")
    except ValueError:
        pass
    else:
        raise AssertionError("a fourth mapping was accepted")


def test_target_inverse_witnesses_roundtrip_nonnegative_values() -> None:
    target = torch.tensor([0.0, NUMERICAL_ZERO_TOLERANCE, 0.1, 18.5], dtype=torch.float32)
    for mapping in MAPPINGS:
        witness = raw_inverse_witness(target, mapping)
        rebuilt = apply_output_mapping(witness, mapping)
        physical_error = torch.abs(rebuilt - target) * SCALES.max()
        assert bool(torch.all(torch.isfinite(witness)))
        assert bool(torch.all(rebuilt >= 0))
        assert float(physical_error.max()) <= ROUNDTRIP_PHYSICAL_ATOL


def test_initial_mapped_outputs_match_across_mappings() -> None:
    outputs = []
    for mapping in MAPPINGS:
        raw = torch.full((2, 6, 4, 4), initial_raw_bias(mapping), dtype=torch.float32)
        outputs.append(apply_output_mapping(raw, mapping))
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    assert torch.equal(outputs[0], torch.full_like(outputs[0], INITIAL_PHYSICAL_EPSILON))


def test_framework_boundary_gradients_are_finite_and_positive_support_is_usable() -> None:
    for mapping in MAPPINGS:
        zero = torch.tensor([0.0], requires_grad=True)
        apply_output_mapping(zero, mapping).sum().backward()
        assert bool(torch.isfinite(zero.grad).all())
        positive_target = torch.tensor([0.1], dtype=torch.float32)
        witness = raw_inverse_witness(positive_target, mapping)
        derivative = mapping_derivative(witness, mapping)
        assert bool(torch.isfinite(derivative).all())
        assert float(torch.abs(derivative).min()) > 0.0


def test_l0_topology_parameter_count_and_encoder_freeze() -> None:
    hashes = []
    for mapping in MAPPINGS:
        model = MappedThayerMixtureExperts(mapping, SCALES)
        warm_start_condition_c_encoder(model, CONDITION_C)
        freeze_encoder(model)
        assert decoder_parameter_count(model) == (46470, 46470)
        assert parameter_count(model) == 165612
        assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
        assert all(parameter.requires_grad for parameter in model.expert_1.parameters())
        assert all(parameter.requires_grad for parameter in model.expert_2.parameters())
        hashes.append(encoder_tensor_sha256(model))
    assert len(set(hashes)) == 1


def test_mapped_model_forward_is_physical_and_nonnegative() -> None:
    blend = torch.zeros((1, 3, 60, 60), dtype=torch.float32)
    prompt = torch.zeros((1, 1, 60, 60), dtype=torch.float32)
    for mapping in MAPPINGS:
        model = MappedThayerMixtureExperts(mapping, SCALES)
        output = model.forward_outputs(blend, prompt)
        assert output.raw_normalized.shape == (1, 2, 6, 60, 60)
        assert torch.equal(
            output.physical,
            output.mapped_normalized * SCALES.repeat(2).view(1, 1, 6, 1, 1),
        )
        assert float(output.physical.min()) >= 0.0


def test_physical_hard_assignment_is_permutation_invariant() -> None:
    scale6 = SCALES.repeat(2)
    targets = torch.rand((2, 2, 6, 4, 4), generator=torch.Generator().manual_seed(7))
    outputs = targets.clone()
    loss_identity, wins_identity, _ = hard_physical_set_loss(outputs, targets, scale6)
    loss_swap, wins_swap, _ = hard_physical_set_loss(outputs[:, [1, 0]], targets, scale6)
    assert float(loss_identity) == 0.0
    assert float(loss_swap) == 0.0
    assert bool(torch.all(wins_identity))
    assert bool(torch.all(~wins_swap))
    assert torch.equal(
        physical_direct_cost(outputs[:, 0], targets[:, 0], scale6),
        torch.zeros(2),
    )


def test_projected_target_canonical_hash_contract_is_unchanged() -> None:
    rows = read_csv(FP / "tables/projected_target_hashes_final.csv")
    with h5py.File(FP / "projection_targets/projected_target_sets_final.h5", "r") as handle:
        physical = handle["targets_physical"]
        for row in rows[::31]:
            sample = np.asarray(
                physical[int(row["scene"]), int(row["prompt"]), int(row["target_slot"])],
                dtype=np.float32,
            )
            assert canonical_tensor_sha256(sample) == row["canonical_sha256"]


def test_campaign_preregistration_and_preflight_artifacts_when_available() -> None:
    if RUN is None or not (RUN / "logs/preflight_complete.json").is_file():
        return
    freeze = json.loads((RUN / "preregistration/freeze_record.json").read_text())
    prereg = RUN / "preregistration/fixed_l0_output_parameterization.md"
    assert sha256(prereg) == freeze["preregistration_sha256"]
    preflight = json.loads((RUN / "logs/preflight_complete.json").read_text())
    if preflight["status"] != "PASS" and "THAYER_OP_RUN_DIR" not in os.environ:
        return
    assert preflight["status"] == "PASS"
    assert set(preflight["eligible_mappings"]) == set(MAPPINGS)
    assert preflight["stop_rule_self_tests_passed"] is True
    assert preflight["initial_outputs_matched"] is True
    assert all(row["pass"] == "True" for row in read_csv(RUN / "tables/mapping_representability.csv"))
    assert all(row["pass"] == "True" for row in read_csv(RUN / "tables/stop_rule_self_tests.csv"))
    assert all(row["pass"] == "True" for row in read_csv(RUN / "tables/synthetic_fit_summary.csv"))


def test_campaign_scene_isolation_encoder_identity_and_physical_outputs_when_available() -> None:
    if RUN is None or not (RUN / "logs/micro_campaign_complete.json").is_file():
        return
    complete = json.loads((RUN / "logs/micro_campaign_complete.json").read_text())
    assert complete["mps_only"] is True
    assert complete["fallback"] is False
    assert complete["unique_scene_input_load_count"] == 8
    assert complete["remaining_56_microset_scene_input_load_count"] == 0
    frozen = read_csv(RUN / "tables/frozen_row_selection.csv")
    assert [int(row["micro_index"]) for row in frozen] == [0, 8, 16, 24, 32, 40, 48, 56]
    encoder_rows = read_csv(RUN / "tables/condition_encoder_hashes.csv")
    assert encoder_rows
    assert all(row["byte_identical"] == "True" for row in encoder_rows)
    assert len({row["reference_encoder_hash"] for row in encoder_rows}) == 1
    for path in RUN.glob("one_scene/*_outputs.h5"):
        with h5py.File(path, "r") as handle:
            physical = np.asarray(handle["physical"], dtype=np.float32)
            assert np.all(np.isfinite(physical))
            assert float(physical.min()) >= 0.0
            assert bool(handle.attrs["complete"])
    for path in RUN.glob("eight_scene/*_outputs.h5"):
        with h5py.File(path, "r") as handle:
            assert handle["physical"].shape[0] == 8
            assert float(np.min(handle["physical"][:])) >= 0.0
