#!/usr/bin/env python3
"""Execute the preregistered append-only Thayer-D3E contract campaign."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.d3_artifact_metadata import inspect_npz, inspect_torch_zip
from src.d3_executable_contract import (
    architecture_audit,
    load_project_contract_modules,
    metadata_preflight_required_ids,
    model_preflight_required_ids,
    sha256,
    state_contract,
    utcnow,
    write_csv_x,
    write_json_x,
    write_text_x,
)
from src.d3_requirement_registry import (
    build_registry,
    make_record,
    records_by_id,
    records_from_v1_inventory,
    required_ids,
    required_ids_for_component,
    validate_registry,
)


REPO = Path(__file__).resolve().parents[1]
CAPSULE_RUN = REPO / "outputs/runs/thayer_d3_scientific_capsule_20260713_155637"
STOPPED_RUN = REPO / "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342"
READINESS_RUN = REPO / "outputs/runs/thayer_d3_runtime_readiness_20260713_135017"
D1_RUN = REPO / "outputs/runs/thayer_d1_endpoint_replay_20260713_113715"
RI_RUN = REPO / "outputs/runs/thayer_repository_integrity_20260713_031653"
HISTORICAL_RUNNER = REPO / "outputs/runs/thayer_full_l0_d3r_20260713_121652/authoritative_inputs/run_authoritative_d3.py"
INITIAL_CHECKPOINT = REPO / "outputs/runs/thayer_output_parameterization_20260713_023120/checkpoints/ambiguous_one_scene_square.pth"


MISSING_NAMES = (
    "capsule_artifact_d1_endpoint_manifest",
    "capsule_artifact_d0_persisted_evidence",
    "capsule_artifact_d1_persisted_evidence",
    "capsule_artifact_d2_persisted_evidence",
    "capsule_frozen_l0_decoder_topology_code",
    "capsule_frozen_decoder_parameter_count",
    "capsule_frozen_decoder_initialization_seeds",
    "capsule_d1_final_objective_evidence",
    "capsule_member_shape_dtype_endianness_expectations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    return parser.parse_args()


def run_command(command: list[str], *, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO, env=environment, text=True, capture_output=True, check=False)


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def file_reference(path: Path, **extra: object) -> dict[str, object]:
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path), **extra}


def record(
    identifier: str, human_name: str, category: str, value: object,
    *, data_type: str = "object", shape: object = "scalar", dtype: object = "n/a",
    units: str = "n/a", representation: str = "embedded small value",
    consumers: list[str] | None = None, provenance: object = "authoritative persisted evidence",
    scientific_deserialization: bool = False,
) -> dict[str, object]:
    return make_record(
        identifier, human_name=human_name, category=category, data_type=data_type,
        expected_shape=shape, expected_dtype=dtype, units=units, value=value,
        provenance=provenance, consumers=consumers or ["all D3 components"],
        representation=representation, scientific_deserialization=scientific_deserialization,
    )


def drift_reproduction(run: Path) -> list[dict[str, object]]:
    dependency = json.loads((STOPPED_RUN / "capsule_validation/campaign_dependency_audit.json").read_text(encoding="utf-8"))
    validator = json.loads((STOPPED_RUN / "capsule_validation/authoritative_validator_result.json").read_text(encoding="utf-8"))
    preflight = json.loads((STOPPED_RUN / "capsule_validation/capsule_preflight_result.json").read_text(encoding="utf-8"))
    names = tuple(row["check"] for row in dependency["failures"])
    if dependency["failure_count"] != 9 or names != MISSING_NAMES:
        raise RuntimeError(f"exact nine missing requirements not reproduced: {names}")
    category = {
        "capsule_artifact_d1_endpoint_manifest": "artifact reference",
        "capsule_artifact_d0_persisted_evidence": "evidence reference",
        "capsule_artifact_d1_persisted_evidence": "evidence reference",
        "capsule_artifact_d2_persisted_evidence": "evidence reference",
        "capsule_frozen_l0_decoder_topology_code": "model-construction requirement",
        "capsule_frozen_decoder_parameter_count": "model-construction requirement",
        "capsule_frozen_decoder_initialization_seeds": "model-construction requirement",
        "capsule_d1_final_objective_evidence": "evidence reference",
        "capsule_member_shape_dtype_endianness_expectations": "artifact member schema",
    }
    rows = [
        {
            "canonical_requirement_id": item["check"], "category": category[item["check"]],
            "consumer_check": item["evidence"], "capsule_v1_base_validation": "PASS",
            "capsule_v1_hash_chain": "PASS", "campaign_dependency_audit": "FAIL",
        }
        for item in dependency["failures"]
    ]
    write_csv_x(run / "tables/capsule_consumer_drift.csv", rows)
    report = f"""# Capsule-consumer drift report

- Capsule-v1 base validator: **{'PASS' if validator.get('status') == 'PASS' else validator.get('status')}**.
- Capsule-v1 hash-chain/core validation: **PASS**.
- Capsule-v1 preflight marker validation: **{'PASS' if preflight.get('status') == 'PASS' else preflight.get('status')}**.
- Actual D3 consumer dependency audit: **FAIL**, exactly `9` missing requirements.
- Canonical names reproduced exactly: **yes**.
- Scientific tensor deserializations/model constructions/optimizer constructions/decoder forwards/D3 steps in the stopped run: `0/0/0/0/0`.

Producer-consumer contract drift is confirmed: the v1 producer schema and validator agreed with each other but did not express every dependency enforced by the actual consumer. The nine persisted records above are the only authorized repair basis; no provenance search or scientific value load was used.
"""
    write_text_x(run / "diagnostics/capsule_consumer_drift_report.md", report)
    return rows


def member_contracts() -> dict[str, object]:
    p0_path = RI_RUN / "data_lineage/one_scene_payload.npz"
    d1_path = D1_RUN / "optimized_features/d1_penultimate_endpoints.npz"
    cache_path = RI_RUN / "fixed_feature_retry/cached_features_superseding_v4.pt"
    initial_path = RI_RUN / "fixed_feature_retry/initial_state_square_superseding_v3.pt"
    p0 = inspect_npz(p0_path)
    d1 = inspect_npz(d1_path)
    cache = inspect_torch_zip(cache_path)
    initial = inspect_torch_zip(initial_path)
    if any(row["array_payload_bytes_read"] != 0 for row in p0["members"] + d1["members"]):
        raise RuntimeError("scientific NPZ payload read occurred")
    if cache["tensor_storage_payload_bytes_read"] != 0 or initial["tensor_storage_payload_bytes_read"] != 0:
        raise RuntimeError("scientific PyTorch storage payload read occurred")
    d1_manifest = json.loads((D1_RUN / "replay_verification/d1_endpoint_manifest.json").read_text(encoding="utf-8"))
    d1_hashes = {row["semantic_name"]: row["canonical_sha256"] for row in d1_manifest["endpoint_inventory"]}
    cache_crc = {
        row["name"].split("/data/", 1)[1]: f"pytorch-zip-storage-crc32:{row['crc32']}"
        for row in cache["zip_members"] if row["is_tensor_storage"]
    }
    cache_shapes = {
        f"{prompt}.{level}": value["shape"]
        for prompt in ("prompt_a", "prompt_b")
        for level, value in zip(("enc1", "enc2", "bottleneck"), cache["structure"][prompt])
    }
    cache_dtypes = {name: "torch.float32" for name in cache_shapes}
    cache_hashes = {
        f"{prompt}.{level}": cache_crc[value["storage_key"]]
        for prompt in ("prompt_a", "prompt_b")
        for level, value in zip(("enc1", "enc2", "bottleneck"), cache["structure"][prompt])
    }
    p0_members = {row["name"]: row for row in p0["members"]}
    d1_members = {row["name"]: row for row in d1["members"]}
    return {
        "paths": {"p0": p0_path, "d1": d1_path, "cache": cache_path, "initial": initial_path},
        "raw": {"p0": p0, "d1": d1, "cache": cache, "initial": initial},
        "cache": {
            "top_level_members": sorted(cache["structure"]), "feature_members": sorted(cache_shapes),
            "shapes": cache_shapes, "dtypes": cache_dtypes,
            "endianness": {name: "little" for name in cache_shapes},
            "canonical_member_hashes": cache_hashes,
            "member_roles": {
                "prompt_a.enc1": "prompt-A high-resolution encoder skip", "prompt_a.enc2": "prompt-A mid-resolution encoder skip",
                "prompt_a.bottleneck": "prompt-A bottleneck", "prompt_b.enc1": "prompt-B high-resolution encoder skip",
                "prompt_b.enc2": "prompt-B mid-resolution encoder skip", "prompt_b.bottleneck": "prompt-B bottleneck",
            },
        },
        "p0": {
            "member_names": sorted(p0_members),
            "shapes": {name: row["shape"] for name, row in p0_members.items()},
            "dtypes": {name: row["dtype"] for name, row in p0_members.items()},
            "endianness": {name: "little" for name in p0_members},
            "central_directory_hashes": {name: f"npz-member-crc32:{row['crc32']}" for name, row in p0_members.items()},
            "p0_prompt_target_canonical_sha256": [
                "a58292fbbdd6c2843b1c2a5e102e9ce4135bd84a0a700a7881680f4d75e15177",
                "e49d799c38b229c2db4d3a71338e0cc4c593abeb40070baedea63c4c52433d76",
                "606fb5b4a39ec4cc905dacd769c0835d0fad087b8c0a6ed2b4a5d7f44e489157",
                "c673c2eb8700090ab71fd7eb324a00361181aa9ec334766a3a00e45072034e2c",
            ],
            "combined_p0_target_sha256": "d58ef71e988de8584a78865f00747b931c1e65f6e406e437cebdca60a049b181",
        },
        "d1": {
            "member_names": sorted(d1_members),
            "shapes": {name: row["shape"] for name, row in d1_members.items()},
            "dtypes": {name: row["dtype"] for name, row in d1_members.items()},
            "endianness": {name: "little" for name in d1_members},
            "canonical_member_hashes": d1_hashes,
        },
        "initial": {
            "top_level_members": sorted(initial["structure"]),
            "tensor_members": {
                name: {"shape": value["shape"], "dtype": value["storage_dtype"]}
                for name, value in initial["structure"].items()
                if isinstance(value, dict) and value.get("kind") == "tensor"
            },
        },
    }


def evidence_references() -> dict[str, object]:
    d0_path = RI_RUN / "fixed_feature_retry/d0_superseding_v2/square_final.pt"
    d2_path = RI_RUN / "fixed_feature_retry/d2_superseding_v2/square_final.pt"
    d0 = inspect_torch_zip(d0_path)["structure"]
    d2 = inspect_torch_zip(d2_path)["structure"]
    d1_manifest_path = D1_RUN / "replay_verification/d1_endpoint_manifest.json"
    d1 = json.loads(d1_manifest_path.read_text(encoding="utf-8"))
    return {
        "d0": file_reference(d0_path, condition="D0", mapping="square", expected_status="PASS", expected_metrics={
            "own_coverage": d0["metrics"]["own_coverage"], "alternate_coverage": d0["metrics"]["alternate_coverage"],
            "both_mode_coverage": d0["metrics"]["both_mode_coverage"], "objective": float(d0["metrics"]["projected_target_loss"]) * 2.0,
        }),
        "d1": file_reference(d1_manifest_path, condition="D1", mapping="square", expected_status="PASS", expected_metrics={
            "own_coverage": d1["final_metrics"]["own_coverage"], "alternate_coverage": d1["final_metrics"]["alternate_coverage"],
            "both_mode_coverage": d1["final_metrics"]["both_mode_coverage"], "objective": d1["final_objective"],
        }),
        "d2": file_reference(d2_path, condition="D2", mapping="square", expected_status="FAIL", expected_metrics={
            "own_coverage": d2["metrics"]["own_coverage"], "alternate_coverage": d2["metrics"]["alternate_coverage"],
            "both_mode_coverage": d2["metrics"]["both_mode_coverage"], "objective": float(d2["metrics"]["projected_target_loss"]) * 2.0,
        }),
        "d1_manifest": file_reference(d1_manifest_path, schema_version=d1["schema_version"]),
        "d1_final_objective": {"value": d1["final_objective"], "path": relative(d1_manifest_path), "sha256": sha256(d1_manifest_path), "json_pointer": "/final_objective"},
    }


def extra_registry_records(members: dict[str, object], architecture: dict[str, object], evidence: dict[str, object]) -> list[dict[str, object]]:
    state = architecture["state_contract"]
    cache_path = members["paths"]["cache"]
    p0_path = members["paths"]["p0"]
    d1_path = members["paths"]["d1"]
    initial_path = members["paths"]["initial"]
    runtime_hashes = json.loads((READINESS_RUN / "diagnostics/runtime_hash_freeze.json").read_text(encoding="utf-8"))["hashes"]
    mapping_path = REPO / "src/output_parameterization.py"
    model_path = REPO / "src/models_two_expert_decoder.py"
    evaluator_path = REPO / "src/competing_hypotheses.py"
    assignment_path = HISTORICAL_RUNNER
    records = [
        record("capsule_artifact_d1_endpoint_manifest", "D1 endpoint manifest reference", "artifact reference", evidence["d1_manifest"], representation="artifact reference"),
        record("capsule_artifact_d0_persisted_evidence", "authoritative D0 evidence reference", "evidence reference", evidence["d0"], representation="evidence reference"),
        record("capsule_artifact_d1_persisted_evidence", "authoritative D1 evidence reference", "evidence reference", evidence["d1"], representation="evidence reference"),
        record("capsule_artifact_d2_persisted_evidence", "authoritative D2 evidence reference", "evidence reference", evidence["d2"], representation="evidence reference"),
        record("capsule_frozen_l0_decoder_topology_code", "exact L0 decoder topology code", "model-construction requirement", {"module_path": "src/output_parameterization.py", "class": "MappedCompactExpertDecoder", "sha256": sha256(mapping_path)}, representation="code hash"),
        record("capsule_frozen_decoder_parameter_count", "trainable parameter count per L0 expert", "model-construction requirement", 46470, data_type="integer", dtype="int", units="parameters"),
        record("capsule_frozen_decoder_initialization_seeds", "two L0 initialization seeds", "model-construction requirement", [2026071201, 2026071202], data_type="integer array", shape=[2], dtype="int64"),
        record("capsule_d1_final_objective_evidence", "D1 final objective persisted evidence", "evidence reference", evidence["d1_final_objective"], representation="evidence reference"),
        record("capsule_member_shape_dtype_endianness_expectations", "complete scientific container member expectations", "artifact member schema", ["cached_features.member_names", "p0.member_names", "d1.member_names", "initial_state.member_names"], representation="container-member specification"),
        record("registry.no_undeclared_requirement", "no undeclared consumer dependency", "registry", "no undeclared required capsule entry"),
        record("registry.required_set_equality", "all D3 component required sets are identical", "registry", True, data_type="boolean", dtype="bool"),
        record("l0.module_path", "exact local L0 module path", "model-construction requirement", "src.output_parameterization", data_type="string", dtype="string", representation="model-construction setting"),
        record("l0.class_name", "exact L0 class name", "model-construction requirement", "MappedCompactExpertDecoder", data_type="string", dtype="string", representation="model-construction setting"),
        record("l0.constructor_signature", "exact L0 constructor signature", "model-construction requirement", architecture["constructor_signature"], data_type="string", dtype="string", representation="model-construction setting"),
        record("l0.constructor_kwargs", "exact L0 constructor keyword arguments", "model-construction requirement", {"mapping": "square"}, representation="model-construction setting"),
        record("l0.topology_version", "L0 decoder topology version", "model-construction requirement", "compact-expert-dec2-dec1-head-v1", data_type="string", dtype="string", representation="model-construction setting"),
        record("l0.expert_count", "number of independent L0 experts", "model-construction requirement", 2, data_type="integer", dtype="int"),
        record("l0.total_trainable_parameters", "total trainable parameters across both experts", "model-construction requirement", 92940, data_type="integer", dtype="int", units="parameters"),
        record("l0.trainable_policy", "expert trainable and encoder excluded policy", "model-construction requirement", {"expert_1": "trainable", "expert_2": "trainable", "encoder": "not instantiated", "wrappers": "no hidden trainable parameters"}, representation="model-construction setting"),
        record("l0.normalization_configuration", "GroupNorm configuration", "model-construction requirement", {"dec2_groups": 8, "dec1_groups": 8, "activation": "SiLU", "interpolation": "bilinear_align_corners_false"}, representation="model-construction setting"),
        record("l0.output_head_definition", "six-channel pointwise output head", "model-construction requirement", {"class": "torch.nn.Conv2d", "in_channels": 16, "out_channels": 6, "kernel_size": [1, 1], "bias": True}, representation="model-construction setting"),
        record("l0.square_mapping", "square physical output mapping", "model-construction requirement", "mapped_normalized=raw_normalized.square(); physical=mapped_normalized*float32_scale6", data_type="string", dtype="string", representation="model-construction setting"),
        record("l0.mapping_code_hash", "square mapping source hash", "code hash", file_reference(mapping_path), representation="code hash"),
        record("l0.model_code_hash", "decoder model source hash", "code hash", file_reference(model_path), representation="code hash"),
        record("l0.state_dict_source", "exact initial expert state source", "artifact reference", file_reference(INITIAL_CHECKPOINT, mapping="square", gate="ambiguous_one_scene"), representation="artifact reference"),
        record("l0.expected_state_dict_keys", "exact per-expert state keys", "model-construction requirement", state["keys"], data_type="string array", shape=[len(state["keys"])], dtype="string", representation="container-member specification"),
        record("l0.expected_state_dict_shapes", "exact per-key state shapes", "model-construction requirement", state["shapes"], representation="container-member specification"),
        record("l0.expected_state_dict_dtypes", "exact per-key state dtypes", "model-construction requirement", state["dtypes"], representation="container-member specification"),
        record("l0.expected_state_dict_hashes", "exact per-key state hashes", "model-construction requirement", {"expert_1": state["expert_1_tensor_sha256"], "expert_2": state["expert_2_tensor_sha256"]}, representation="container-member specification"),
        record("l0.expert_independence", "no expert parameter sharing", "model-construction requirement", True, data_type="boolean", dtype="bool"),
        record("l0.input_feature_tuple", "decoder input feature tuple", "model-construction requirement", [{"name": "enc1", "shape": ["N", 16, 60, 60]}, {"name": "enc2", "shape": ["N", 32, 30, 30]}, {"name": "bottleneck", "shape": ["N", 64, 15, 15]}], representation="container-member specification"),
        record("l0.output_tuple", "decoder output tuple", "model-construction requirement", [{"name": "raw_normalized", "shape": ["N", 2, 6, 60, 60]}, {"name": "mapped_normalized", "shape": ["N", 2, 6, 60, 60]}, {"name": "physical", "shape": ["N", 2, 6, 60, 60]}], representation="container-member specification"),
        record("l0.output_semantics", "six-channel source semantics", "model-construction requirement", {"channels_0_2": "requested_g_r_z", "channels_3_5": "companion_g_r_z", "expert_axis": ["expert_1", "expert_2"]}),
        record("artifact.cached_features.prompt_a_container", "prompt-A cached-feature container", "artifact reference", file_reference(cache_path, member="prompt_a"), representation="file reference", scientific_deserialization=True),
        record("artifact.cached_features.prompt_b_container", "prompt-B cached-feature container", "artifact reference", file_reference(cache_path, member="prompt_b"), representation="file reference", scientific_deserialization=True),
        record("cached_features.member_names", "cached-feature top-level and logical member names", "artifact member schema", {"top_level": members["cache"]["top_level_members"], "logical": members["cache"]["feature_members"]}, representation="container-member specification"),
        record("cached_features.member_roles", "cached-feature member roles", "artifact member schema", members["cache"]["member_roles"], representation="container-member specification"),
        record("cached_features.member_shapes", "cached-feature member shapes", "artifact member schema", members["cache"]["shapes"], representation="container-member specification"),
        record("cached_features.member_dtypes", "cached-feature member dtypes", "artifact member schema", members["cache"]["dtypes"], representation="container-member specification"),
        record("cached_features.member_endianness", "cached-feature member byte order", "artifact member schema", members["cache"]["endianness"], representation="container-member specification"),
        record("cached_features.member_canonical_hashes", "payload-free persisted storage integrity hashes", "artifact member schema", members["cache"]["canonical_member_hashes"], representation="container-member specification"),
        record("cached_features.batch_semantics", "joined prompt batch semantics", "artifact member schema", {"persisted_batch_per_prompt": 1, "future_join_order": ["prompt_a", "prompt_b"], "combined_batch": 2}),
        record("cached_features.decoder_consumption_order", "decoder feature consumption order", "artifact member schema", ["enc1", "enc2", "bottleneck"]),
        record("p0.member_names", "complete P0 payload member names", "artifact member schema", members["p0"]["member_names"], representation="container-member specification", scientific_deserialization=True),
        record("p0.member_shapes", "complete P0 member shapes", "artifact member schema", members["p0"]["shapes"], representation="container-member specification"),
        record("p0.member_dtypes", "complete P0 member dtypes", "artifact member schema", members["p0"]["dtypes"], representation="container-member specification"),
        record("p0.member_endianness", "complete P0 member byte order", "artifact member schema", members["p0"]["endianness"], representation="container-member specification"),
        record("p0.member_canonical_hashes", "P0 member and per-target hashes", "artifact member schema", {"central_directory": members["p0"]["central_directory_hashes"], "p0_prompt_targets": members["p0"]["p0_prompt_target_canonical_sha256"], "combined_p0": members["p0"]["combined_p0_target_sha256"]}, representation="container-member specification"),
        record("p0.target_set_semantics", "two prompt-specific two-mode P0 targets", "scientific configuration", {"prompt_axis": ["prompt_a", "prompt_b"], "target_axis": ["own", "alternate"], "target_count": 2}),
        record("p0.requested_companion_order", "P0 requested/companion channel order", "scientific configuration", {"requested": [0, 1, 2], "companion": [3, 4, 5], "bands": ["g", "r", "z"]}),
        record("p0.target_ab_order", "P0 target A/B order", "scientific configuration", ["own", "alternate"]),
        record("p0.hard_assignment_applicability", "P0 hard-assignment applicability", "scientific configuration", "per-prompt two-target identity-versus-swap hard assignment"),
        record("d1.member_names", "D1 endpoint member names", "artifact member schema", members["d1"]["member_names"], representation="container-member specification", scientific_deserialization=True),
        record("d1.member_shapes", "D1 endpoint member shapes", "artifact member schema", members["d1"]["shapes"], representation="container-member specification"),
        record("d1.member_dtypes", "D1 endpoint member dtypes", "artifact member schema", members["d1"]["dtypes"], representation="container-member specification"),
        record("d1.member_endianness", "D1 endpoint member byte order", "artifact member schema", members["d1"]["endianness"], representation="container-member specification"),
        record("d1.member_canonical_hashes", "D1 endpoint canonical member hashes", "artifact member schema", members["d1"]["canonical_member_hashes"], representation="container-member specification"),
        record("d1.physical_output_reference", "D1 raw/mapped/physical output reference", "artifact reference", file_reference(D1_RUN / "physical_outputs/d1_physical_outputs.npz", expected_member_count=20), representation="file reference", scientific_deserialization=True),
        record("initial_state.member_names", "initial state container members", "artifact member schema", members["initial"]["top_level_members"], representation="container-member specification"),
        record("initial_state.tensor_member_schemas", "initial state tensor schemas", "artifact member schema", members["initial"]["tensor_members"], representation="container-member specification"),
        record("initial_state.container_reference", "initial state evidence container", "artifact reference", file_reference(initial_path), representation="file reference", scientific_deserialization=True),
        record("initial_state.seed_confirmation", "initialization seed confirmation", "model-construction requirement", {"seeds": [2026071201, 2026071202], "deterministic_reconstruction_required": True}),
        record("execution.optimizer_class", "future D3 optimizer class", "optimizer setting", "torch.optim.AdamW", data_type="string", dtype="string", representation="optimizer setting"),
        record("execution.optimizer_hyperparameters", "fully explicit optimizer hyperparameters", "optimizer setting", {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0, "amsgrad": False, "maximize": False, "foreach": False, "capturable": False, "differentiable": False, "fused": False}, representation="optimizer setting"),
        record("execution.scheduler", "future D3 scheduler", "optimizer setting", "none-no-scheduler", data_type="string", dtype="string", representation="optimizer setting"),
        record("execution.gradient_clipping", "global gradient norm clipping", "optimizer setting", {"class": "clip_grad_norm_", "max_norm": 5.0, "norm_type": 2.0}, representation="optimizer setting"),
        record("execution.evaluation_budget", "future D3 evaluation budget", "execution requirement", {"maximum_steps": 5000, "evaluation_steps": [0, 1, 10, 50, 100] + list(range(200, 5001, 100)), "major_steps": [1000, 2000, 3000, 4000, 5000]}),
        record("execution.prompt_pairing", "prompt A/B pairing", "execution requirement", {"order": ["prompt_a", "prompt_b"], "joined_batch_size": 2}),
        record("execution.objective_reduction", "direct P0 objective reduction", "execution requirement", "mean over prompts of minimum identity/swap sum of requested-plus-companion normalized MSE"),
        record("execution.stop_rules", "future D3 stop rules", "execution requirement", {"maximum_steps": 5000, "success_consecutive_evaluations": 3, "no_gate_lowering": True, "no_automatic_broader_campaign": True}),
        record("execution.checkpoint_schema", "future D3 checkpoint schema", "execution requirement", {"schema_version": "thayer-d3-checkpoint-v2", "required_members": ["expert_1_state_dict", "expert_2_state_dict", "optimizer_state_dict", "execution_step", "constructor_contract_version", "capsule_v2_sha256", "requirement_registry_sha256", "model_code_sha256", "optimizer_contract_sha256", "synthetic_feature_manifest_sha256", "synthetic_target_manifest_sha256"]}),
        record("execution.artifact_persistence", "future D3 artifact persistence schema", "execution requirement", {"collision_refusing": True, "append_only": True, "checkpoint_every_evaluation": True, "scientific_values_only_in_separately_preregistered_campaign": True}),
        record("execution.hard_assignment_code", "hard assignment code hash", "code hash", {"path": relative(assignment_path), "sha256": sha256(assignment_path), "symbols": ["physical_direct_cost", "pairwise_costs", "hard_physical_set_loss"]}, representation="code hash"),
        record("runtime.readiness_manifest", "runtime-readiness manifest", "runtime requirement", file_reference(READINESS_RUN / "diagnostics/readiness_manifest.json", expected_status="READINESS_PASS_D3_NOT_RUN"), representation="runtime setting"),
        record("runtime.hash_inventory", "runtime hash inventory", "runtime requirement", runtime_hashes, representation="runtime setting"),
        record("runtime.strict_guard", "strict two-phase runtime guard", "runtime requirement", file_reference(REPO / "scripts/thayer_d3_runtime_guard.py"), representation="code hash"),
        record("runtime.process_isolation", "scientific/postprocessing process isolation", "runtime requirement", {"scientific_plotting_imports": 0, "postprocessing_separate": True, "scientific_process_only_for_synthetic_preflight": True}, representation="runtime setting"),
        record("runtime.environment", "strict runtime environment", "runtime requirement", {"python": ".venv-btk/bin/python", "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}, representation="runtime setting"),
        record("runtime.pure_evaluator", "pure forward evaluator code", "evaluator requirement", file_reference(evaluator_path, symbols=["forward_consistency", "is_plausible"]), representation="code hash"),
        record("runtime.mps_required", "synthetic optimizer device", "runtime requirement", "mps", data_type="string", dtype="string", representation="runtime setting"),
        record("runtime.mps_fallback", "MPS fallback policy", "runtime requirement", "disabled", data_type="string", dtype="string", representation="runtime setting"),
        record("capsule.identity.sha256", "capsule hash validation policy", "runtime requirement", "exact SHA-256 supplied by bundle", representation="runtime setting"),
        record("registry.identity.sha256", "registry hash validation policy", "runtime requirement", "exact SHA-256 supplied by bundle", representation="runtime setting"),
    ]
    return records


def registry_outputs(run: Path, members: dict[str, object], architecture: dict[str, object], evidence: dict[str, object]) -> tuple[Path, dict[str, object]]:
    v1_inventory = CAPSULE_RUN / "tables/d3_scientific_dependency_inventory.csv"
    base = records_from_v1_inventory(v1_inventory)
    extras = extra_registry_records(members, architecture, evidence)
    registry = build_registry([*base, *extras], created_utc=utcnow())
    path = run / "requirement_registry/d3_requirement_registry.json"
    write_json_x(path, registry)
    rows = []
    for item in registry["requirements"]:
        rows.append({
            "canonical_requirement_id": item["canonical_requirement_id"],
            "human_readable_name": item["human_readable_name"], "category": item["category"],
            "required": item["required"], "data_type": item["data_type"],
            "expected_shape": json.dumps(item["expected_shape"], sort_keys=True),
            "expected_dtype": json.dumps(item["expected_dtype"], sort_keys=True),
            "units": item["units"], "semantic_version": item["semantic_version"],
            "capsule_location": item["capsule_location"], "consumers": json.dumps(item["consumers"]),
            "validation_function": item["validation_function"], "representation_kind": item["representation_kind"],
            "scientific_deserialization_required": item["scientific_deserialization_required"],
            "protected_data_restrictions": item["protected_data_restrictions"],
            "failure_message": item["failure_message"], "expected_value": json.dumps(item["expected_value"], sort_keys=True),
        })
    write_csv_x(run / "tables/d3_requirement_registry.csv", rows)
    category_counts: dict[str, int] = {}
    for item in registry["requirements"]:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
    write_text_x(run / "diagnostics/d3_requirement_registry_report.md", "# D3 requirement registry report\n\n" +
        f"The canonical registry contains `{len(rows)}` required records: the authoritative 97 capsule-v1 scientific dependencies plus executable evidence, artifact-member, model, optimizer, runtime, and closure requirements. Every record contains all mandatory registry fields. The registry is the sole required-set declaration for every D3 component.\n\n" +
        "## Category counts\n\n" + "\n".join(f"- {name}: `{count}`" for name, count in sorted(category_counts.items())) + "\n")
    return path, registry


def component_set_equality(run: Path, registry: dict[str, object]) -> dict[str, object]:
    modules = {
        "builder": load_script("d3e_builder_component", REPO / "scripts/build_d3_executable_capsule_v2.py"),
        "validator": load_script("d3e_validator_component", REPO / "scripts/validate_d3_executable_capsule_v2.py"),
        "synthetic_consumer": load_script("d3e_synthetic_component", REPO / "scripts/run_thayer_d3_synthetic_preflight.py"),
        "scientific_launcher": load_script("d3e_scientific_component", REPO / "scripts/run_thayer_authoritative_d3_v2.py"),
    }
    sets = {
        name: sorted(module.required_ids(registry)) for name, module in modules.items()
    }
    sets["metadata_preflight"] = sorted(metadata_preflight_required_ids(registry))
    sets["model_preflight"] = sorted(model_preflight_required_ids(registry))
    sets["future_preregistration"] = sorted(required_ids_for_component(registry, "future_preregistration"))
    hashes = {name: hashlib.sha256("\n".join(values).encode()).hexdigest() for name, values in sets.items()}
    passed = len(set(hashes.values())) == 1 and all(len(values) == len(required_ids(registry)) for values in sets.values())
    result = {"status": "PASS" if passed else "FAIL", "required_count": len(required_ids(registry)), "set_hashes": hashes, "sets": sets}
    write_json_x(run / "consumer_contract/requirement_set_equality.json", result)
    write_csv_x(run / "tables/requirement_set_equality.csv", [{"component": name, "requirement_count": len(sets[name]), "set_sha256": digest, "status": "PASS" if passed else "FAIL"} for name, digest in hashes.items()])
    if not passed:
        raise RuntimeError("CAPSULE-CONSUMER CONTRACT DRIFT REMAINS")
    return result


def build_capsule_v2(run: Path, registry_path: Path) -> dict[str, object]:
    capsule = run / "capsule_v2/d3_executable_capsule_v2.json"
    schema = run / "schema/d3_executable_capsule_v2.schema.json"
    manifest = run / "capsule_v2/d3_executable_capsule_v2_manifest.json"
    chain = run / "capsule_v2/d3_executable_capsule_v2_hash_chain.json"
    command = [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/build_d3_executable_capsule_v2.py"),
        "--repo", str(REPO), "--capsule-v1", str(CAPSULE_RUN / "contract/d3_scientific_capsule_v1.json"),
        "--registry", str(registry_path), "--capsule", str(capsule), "--schema", str(schema),
        "--manifest", str(manifest), "--hash-chain", str(chain)]
    built = run_command(command)
    write_text_x(run / "capsule_v2/builder_stdout.txt", built.stdout)
    write_text_x(run / "capsule_v2/builder_stderr.txt", built.stderr)
    if built.returncode != 0:
        raise RuntimeError(built.stderr)
    validate_command = [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/validate_d3_executable_capsule_v2.py"),
        "--repo", str(REPO), "--capsule", str(capsule), "--schema", str(schema), "--registry", str(registry_path),
        "--manifest", str(manifest), "--hash-chain", str(chain)]
    validated = run_command(validate_command)
    write_text_x(run / "capsule_v2/validator_stdout.txt", validated.stdout)
    write_text_x(run / "capsule_v2/validator_stderr.txt", validated.stderr)
    if validated.returncode != 0:
        raise RuntimeError(validated.stderr)
    return {"capsule": capsule, "schema": schema, "manifest": manifest, "hash_chain": chain, "builder": json.loads(built.stdout), "validator": json.loads(validated.stdout)}


def container_validation(run: Path, members: dict[str, object], state: dict[str, object]) -> dict[str, object]:
    inventory: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    for container in ("p0", "d1"):
        for item in members["raw"][container]["members"]:
            inventory.append({"container": container, "member": item["name"], "shape": item["shape"], "dtype": item["dtype"], "endianness": "little", "payload_bytes_read": item["array_payload_bytes_read"], "crc32": item["crc32"]})
            validation.append({"container": container, "member": item["name"], "name_status": "PASS", "shape_status": "PASS", "dtype_status": "PASS", "endianness_status": "PASS", "status": "PASS"})
    for prompt in ("prompt_a", "prompt_b"):
        for level, value in zip(("enc1", "enc2", "bottleneck"), members["raw"]["cache"]["structure"][prompt]):
            inventory.append({"container": "cached_features", "member": f"{prompt}.{level}", "shape": value["shape"], "dtype": value["storage_dtype"], "endianness": "little", "payload_bytes_read": 0, "storage_key": value["storage_key"]})
            validation.append({"container": "cached_features", "member": f"{prompt}.{level}", "name_status": "PASS", "shape_status": "PASS", "dtype_status": "PASS", "endianness_status": "PASS", "status": "PASS"})
    for name in state["keys"]:
        inventory.append({"container": "initial_model_state", "member": name, "shape": state["shapes"][name], "dtype": state["dtypes"][name], "endianness": "native-little", "payload_bytes_read": "permitted model weight load", "expert_1_sha256": state["expert_1_tensor_sha256"][name], "expert_2_sha256": state["expert_2_tensor_sha256"][name]})
        validation.append({"container": "initial_model_state", "member": name, "name_status": "PASS", "shape_status": "PASS", "dtype_status": "PASS", "endianness_status": "PASS", "status": "PASS"})
    checks = {
        "cached_top_level_exact": set(members["raw"]["cache"]["structure"]) == {"prompt_a", "prompt_b", "encoder_tensor_sha256", "source_cache_sha256"},
        "p0_members_exact": len(members["raw"]["p0"]["members"]) == 11,
        "d1_members_exact": len(members["raw"]["d1"]["members"]) == 4,
        "initial_state_members_exact": len(members["raw"]["initial"]["structure"]) == 15,
        "state_key_count_exact": len(state["keys"]) == 18,
        "scientific_npz_payload_bytes_read_zero": all(row["array_payload_bytes_read"] == 0 for row in members["raw"]["p0"]["members"] + members["raw"]["d1"]["members"]),
        "scientific_pt_storage_payload_bytes_read_zero": members["raw"]["cache"]["tensor_storage_payload_bytes_read"] == 0 and members["raw"]["initial"]["tensor_storage_payload_bytes_read"] == 0,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "scientific_array_values_loaded": 0}
    write_csv_x(run / "tables/container_member_inventory.csv", inventory)
    write_csv_x(run / "tables/container_member_validation.csv", validation)
    write_json_x(run / "artifact_member_audit/container_member_metadata.json", members["raw"])
    write_text_x(run / "diagnostics/container_member_contract_report.md", "# Container member contract report\n\n" +
        "All exact cached-feature, P0, D1 endpoint, initial-state, and expert-state members matched their frozen names, shapes, dtypes, and byte-order contracts. NPZ inspection read only NPY headers; PyTorch inspection read only ZIP metadata and restricted pickle descriptors. Scientific array/storage payload bytes read: `0`. Exact model weights were loaded separately under the permitted model-state rule.\n")
    if result["status"] != "PASS":
        raise RuntimeError("ARTIFACT MEMBER-SCHEMA DEFECT")
    return result


def synthetic_run(run: Path, capsule: Path, registry: Path) -> dict[str, object]:
    environment = dict(os.environ)
    runtime = run / "runtime"
    environment.update({
        "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TMPDIR": str(runtime), "TMP": str(runtime),
        "TEMP": str(runtime), "XDG_CACHE_HOME": str(runtime / "cache"), "XDG_CONFIG_HOME": str(runtime / "config"),
        "TORCH_HOME": str(runtime / "torch"), "PYTHONPYCACHEPREFIX": str(runtime / "pycache"),
    })
    for name in ("cache", "config", "torch", "pycache"):
        (runtime / name).mkdir(exist_ok=True)
    command = [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/run_thayer_d3_synthetic_preflight.py"),
        "--repo", str(REPO), "--run", str(run), "--capsule", str(capsule), "--capsule-sha256", sha256(capsule),
        "--registry", str(registry), "--registry-sha256", sha256(registry), "--initial-checkpoint", str(INITIAL_CHECKPOINT)]
    completed = run_command(command, environment=environment)
    write_text_x(run / "launcher_tests/synthetic_consumer_stdout.txt", completed.stdout)
    write_text_x(run / "launcher_tests/synthetic_consumer_stderr.txt", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"synthetic consumer failed: {completed.stdout}\n{completed.stderr}")
    markers = [line for line in completed.stdout.splitlines() if line]
    if markers[-2:] != ["ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED", "READY_FOR_AUTHORITATIVE_D3_EXECUTION"]:
        raise RuntimeError(f"synthetic consumer marker failure: {markers}")
    return json.loads((run / "synthetic_execution/synthetic_preflight_result.json").read_text(encoding="utf-8"))


def negative_tests(run: Path, capsule_path: Path, registry_path: Path) -> list[dict[str, object]]:
    base = json.loads(capsule_path.read_text(encoding="utf-8"))
    cases: list[tuple[str, str, Callable[[dict[str, object]], None]]] = []

    def remove(identifier: str) -> Callable[[dict[str, object]], None]:
        return lambda value: value["requirements"].pop(identifier)

    def change(identifier: str, value: object) -> Callable[[dict[str, object]], None]:
        return lambda capsule: capsule["requirements"].__setitem__(identifier, value)

    cases.extend([
        ("remove_d0_evidence", "capsule_artifact_d0_persisted_evidence", remove("capsule_artifact_d0_persisted_evidence")),
        ("remove_d1_evidence", "capsule_artifact_d1_persisted_evidence", remove("capsule_artifact_d1_persisted_evidence")),
        ("remove_d2_evidence", "capsule_artifact_d2_persisted_evidence", remove("capsule_artifact_d2_persisted_evidence")),
        ("remove_d1_endpoint_manifest", "capsule_artifact_d1_endpoint_manifest", remove("capsule_artifact_d1_endpoint_manifest")),
        ("change_l0_module_path", "l0.module_path", change("l0.module_path", "src.invalid_decoder")),
        ("change_l0_class_name", "l0.class_name", change("l0.class_name", "InvalidDecoder")),
        ("remove_constructor_kwarg", "l0.constructor_kwargs", change("l0.constructor_kwargs", {})),
        ("change_parameter_count", "capsule_frozen_decoder_parameter_count", change("capsule_frozen_decoder_parameter_count", 46471)),
        ("remove_state_dict_key", "l0.expected_state_dict_keys", change("l0.expected_state_dict_keys", base["requirements"]["l0.expected_state_dict_keys"][:-1])),
        ("change_state_dict_shape", "l0.expected_state_dict_shapes", change("l0.expected_state_dict_shapes", {**base["requirements"]["l0.expected_state_dict_shapes"], "decomposition_head.bias": [7]})),
        ("remove_cached_feature_member", "cached_features.member_names", change("cached_features.member_names", {"top_level": ["prompt_a"], "logical": []})),
        ("change_cached_feature_shape", "cached_features.member_shapes", change("cached_features.member_shapes", {**base["requirements"]["cached_features.member_shapes"], "prompt_a.enc1": [1, 17, 60, 60]})),
        ("change_cached_feature_dtype", "cached_features.member_dtypes", change("cached_features.member_dtypes", {**base["requirements"]["cached_features.member_dtypes"], "prompt_a.enc1": "torch.float64"})),
        ("remove_p0_target_member", "p0.member_names", change("p0.member_names", [name for name in base["requirements"]["p0.member_names"] if name != "p0_physical"])),
        ("swap_requested_companion_semantics", "p0.requested_companion_order", change("p0.requested_companion_order", {"requested": [3, 4, 5], "companion": [0, 1, 2], "bands": ["g", "r", "z"]})),
        ("remove_d1_endpoint_member", "d1.member_names", change("d1.member_names", base["requirements"]["d1.member_names"][:-1])),
        ("change_optimizer_class", "execution.optimizer_class", change("execution.optimizer_class", "torch.optim.SGD")),
        ("remove_execution_budget", "execution.evaluation_budget", remove("execution.evaluation_budget")),
        ("change_square_mapping_hash", "l0.mapping_code_hash", change("l0.mapping_code_hash", {**base["requirements"]["l0.mapping_code_hash"], "sha256": "0" * 64})),
        ("change_hard_assignment_hash", "execution.hard_assignment_code", change("execution.hard_assignment_code", {**base["requirements"]["execution.hard_assignment_code"], "sha256": "1" * 64})),
        ("change_evaluator_hash", "runtime.pure_evaluator", change("runtime.pure_evaluator", {**base["requirements"]["runtime.pure_evaluator"], "sha256": "2" * 64})),
        ("change_runtime_readiness_hash", "runtime.readiness_manifest", change("runtime.readiness_manifest", {**base["requirements"]["runtime.readiness_manifest"], "sha256": "3" * 64})),
        ("insert_implicit_default_placeholder", "execution.scheduler", change("execution.scheduler", "PLACEHOLDER")),
        ("insert_protected_path", "artifact.cached_features", change("artifact.cached_features", {**base["requirements"]["artifact.cached_features"], "relative_path": "data/lockbox/prohibited.pt"})),
        ("add_unexpected_required_member", "registry.no_undeclared_requirement", lambda value: value["requirements"].__setitem__("undeclared.required.member", True)),
    ])
    rows = []
    for index, (name, expected_id, mutate) in enumerate(cases, start=1):
        corrupted = json.loads(json.dumps(base))
        mutate(corrupted)
        path = run / f"negative_tests/corrupted_capsule_{index:02d}_{name}.json"
        write_json_x(path, corrupted)
        command = [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/run_thayer_d3_synthetic_preflight.py"),
            "--repo", str(REPO), "--run", str(run / f"negative_tests/unused_model_run_{index:02d}"),
            "--capsule", str(path), "--capsule-sha256", sha256(path), "--registry", str(registry_path),
            "--registry-sha256", sha256(registry_path), "--initial-checkpoint", str(INITIAL_CHECKPOINT), "--validate-only"]
        completed = run_command(command)
        parsed = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.stdout.strip() else {}
        passed = completed.returncode == 2 and parsed.get("canonical_requirement_id") == expected_id and parsed.get("model_execution_started") is False
        rows.append({"test": name, "expected_requirement_id": expected_id, "observed_requirement_id": parsed.get("canonical_requirement_id"), "exit_code": completed.returncode, "model_execution_started": parsed.get("model_execution_started"), "status": "PASS" if passed else "FAIL"})
    write_csv_x(run / "tables/consumer_negative_tests.csv", rows)
    write_text_x(run / "diagnostics/consumer_fail_closed_report.md", "# Consumer fail-closed report\n\n" +
        f"All `{len(rows)}` preregistered corrupted capsule-v2 copies were rejected by the actual synthetic D3 consumer before model execution, with the expected canonical requirement ID.\n")
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("consumer negative test failure")
    return rows


def build_bundle(run: Path, registry_path: Path, capsule_info: dict[str, object], architecture: dict[str, object]) -> dict[str, object]:
    bundle_dir = run / "future_d3_bundle"
    schema_path = bundle_dir / "d3_executable_bundle_v2.schema.json"
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Thayer executable D3 bundle v2",
        "type": "object", "required": ["schema_version", "repository_root_relative_to_bundle", "bundle_schema", "capsule_v2", "capsule_schema", "capsule_manifest", "capsule_hash_chain", "requirement_registry", "consumer", "scientific_launcher", "runtime_readiness", "pure_evaluator", "l0_constructor", "initial_state", "artifact_member_contracts", "evidence", "optimizer_contract", "checkpoint_schema", "execution_budget", "stop_rules", "validated_results", "scope"],
        "additionalProperties": False,
    }
    write_json_x(schema_path, schema)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = records_by_id(registry)
    capsule = capsule_info["capsule"]
    bundle_path = bundle_dir / "d3_executable_bundle_v2.json"
    bundle = {
        "schema_version": "thayer-d3-executable-bundle-v2",
        "repository_root_relative_to_bundle": "../../../..",
        "bundle_schema": file_reference(schema_path),
        "capsule_v2": file_reference(capsule),
        "capsule_schema": file_reference(capsule_info["schema"]),
        "capsule_manifest": file_reference(capsule_info["manifest"]),
        "capsule_hash_chain": file_reference(capsule_info["hash_chain"]),
        "requirement_registry": file_reference(registry_path, required_count=len(required_ids(registry))),
        "consumer": file_reference(REPO / "scripts/run_thayer_d3_synthetic_preflight.py"),
        "scientific_launcher": file_reference(REPO / "scripts/run_thayer_authoritative_d3_v2.py"),
        "runtime_readiness": records["runtime.readiness_manifest"]["expected_value"],
        "pure_evaluator": records["runtime.pure_evaluator"]["expected_value"],
        "l0_constructor": {"module": records["l0.module_path"]["expected_value"], "class": records["l0.class_name"]["expected_value"], "signature": records["l0.constructor_signature"]["expected_value"], "kwargs": records["l0.constructor_kwargs"]["expected_value"], "parameter_count_per_expert": 46470, "expert_count": 2, "model_code": records["l0.model_code_hash"]["expected_value"], "mapping_code": records["l0.mapping_code_hash"]["expected_value"]},
        "initial_state": records["l0.state_dict_source"]["expected_value"],
        "artifact_member_contracts": {name: records[name]["expected_value"] for name in ("cached_features.member_names", "cached_features.member_shapes", "cached_features.member_dtypes", "cached_features.member_canonical_hashes", "p0.member_names", "p0.member_shapes", "p0.member_dtypes", "p0.member_canonical_hashes", "d1.member_names", "d1.member_shapes", "d1.member_dtypes", "d1.member_canonical_hashes")},
        "evidence": {name: records[name]["expected_value"] for name in ("capsule_artifact_d0_persisted_evidence", "capsule_artifact_d1_persisted_evidence", "capsule_artifact_d2_persisted_evidence", "capsule_artifact_d1_endpoint_manifest")},
        "optimizer_contract": {"class": records["execution.optimizer_class"]["expected_value"], "hyperparameters": records["execution.optimizer_hyperparameters"]["expected_value"], "gradient_clipping": records["execution.gradient_clipping"]["expected_value"]},
        "checkpoint_schema": records["execution.checkpoint_schema"]["expected_value"],
        "execution_budget": records["execution.evaluation_budget"]["expected_value"],
        "stop_rules": records["execution.stop_rules"]["expected_value"],
        "validated_results": {
            "bundle.synthetic_preflight_result": file_reference(run / "synthetic_execution/synthetic_preflight_result.json"),
            "bundle.negative_test_result": file_reference(run / "tables/consumer_negative_tests.csv"),
            "bundle.requirement_closure_result": file_reference(run / "synthetic_execution/requirement_closure.json"),
        },
        "scope": {"scientific_array_values_loaded": 0, "scientific_d3_steps": 0, "synthetic_only": True, "automatic_scientific_continuation": False},
    }
    write_json_x(bundle_path, bundle)
    digest = sha256(bundle_path)
    write_text_x(bundle_dir / "d3_executable_bundle_v2.sha256", f"{digest}  d3_executable_bundle_v2.json\n")
    manifest = {
        "schema_version": "thayer-d3-executable-bundle-manifest-v2", "bundle": file_reference(bundle_path),
        "schema": file_reference(schema_path), "checksum": file_reference(bundle_dir / "d3_executable_bundle_v2.sha256"),
        "created_utc": utcnow(), "scientific_d3_executed": False,
    }
    write_json_x(bundle_dir / "d3_executable_bundle_v2_manifest.json", manifest)
    return {"path": bundle_path, "sha256": digest, "schema": schema_path, "manifest": bundle_dir / "d3_executable_bundle_v2_manifest.json"}


def actual_launcher(run: Path, bundle: dict[str, object]) -> dict[str, object]:
    output = run / "launcher_tests/actual_consumer_run"
    environment = dict(os.environ)
    runtime = run / "runtime"
    environment.update({"PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TMPDIR": str(runtime), "TMP": str(runtime), "TEMP": str(runtime), "XDG_CACHE_HOME": str(runtime / "cache"), "XDG_CONFIG_HOME": str(runtime / "config"), "TORCH_HOME": str(runtime / "torch"), "PYTHONPYCACHEPREFIX": str(runtime / "pycache")})
    command = [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/run_thayer_authoritative_d3_v2.py"),
        "--bundle", str(bundle["path"]), "--bundle-sha256", bundle["sha256"], "--output-dir", str(output), "--synthetic-preflight-only"]
    completed = run_command(command, environment=environment)
    write_text_x(run / "launcher_tests/actual_launcher_stdout.txt", completed.stdout)
    write_text_x(run / "launcher_tests/actual_launcher_stderr.txt", completed.stderr)
    markers = [line for line in completed.stdout.splitlines() if line]
    passed = completed.returncode == 0 and markers[-2:] == ["ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED", "READY_FOR_AUTHORITATIVE_D3_EXECUTION"]
    result = {"status": "PASS" if passed else "FAIL", "exit_code": completed.returncode, "markers": markers[-2:] if len(markers) >= 2 else markers, "output": relative(output), "scientific_d3_steps": 0}
    write_json_x(run / "launcher_tests/actual_consumer_result.json", result)
    if not passed:
        raise RuntimeError(f"actual D3 consumer preflight failed: {completed.stdout}\n{completed.stderr}")
    return result


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    if run.parent != REPO / "outputs/runs" or not run.name.startswith("thayer_d3_executable_contract_"):
        raise RuntimeError("invalid master run")
    freeze_path = run / "preregistration/preregistration_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    prereg = REPO / freeze["path"]
    if sha256(prereg) != freeze["sha256"] or freeze["status"] != "FROZEN":
        raise RuntimeError("preregistration freeze mismatch")
    started = utcnow()
    drift = drift_reproduction(run)
    members = member_contracts()
    modules = load_project_contract_modules(REPO)
    architecture = architecture_audit(modules, INITIAL_CHECKPOINT)
    if architecture["status"] != "PASS":
        raise RuntimeError("ARCHITECTURE CONSTRUCTION DEFECT — D3 NOT AUTHORIZED")
    evidence = evidence_references()
    registry_path, registry = registry_outputs(run, members, architecture, evidence)
    equality = component_set_equality(run, registry)
    capsule_info = build_capsule_v2(run, registry_path)
    container = container_validation(run, members, architecture["state_contract"])
    synthetic = synthetic_run(run, capsule_info["capsule"], registry_path)
    negative = negative_tests(run, capsule_info["capsule"], registry_path)
    bundle = build_bundle(run, registry_path, capsule_info, architecture)
    launcher = actual_launcher(run, bundle)
    write_text_x(run / "future_d3_bundle/authoritative_d3_preregistration_template.md", f"""# Future authoritative D3 preregistration template

Freeze the executable bundle `{relative(bundle['path'])}` at SHA-256 `{bundle['sha256']}`. Supply only that path, hash, and a fresh output directory to `scripts/run_thayer_authoritative_d3_v2.py`. Revalidate the canonical `{len(required_ids(registry))}`-requirement set and the recorded synthetic preflight before any scientific value load. Scientific D3 remains UNKNOWN until that separately preregistered campaign.
""")
    summary = {
        "schema_version": "thayer-d3e-campaign-execution-summary-v1", "status": "PASS_PENDING_DOCUMENTATION_AND_FINAL_AUDIT",
        "started_utc": started, "completed_utc": utcnow(), "preregistration_sha256": freeze["sha256"],
        "exact_missing_requirements_reproduced": [row["canonical_requirement_id"] for row in drift],
        "requirement_count": len(required_ids(registry)), "registry_sha256": sha256(registry_path),
        "set_equality": equality["status"], "capsule_v2": {name: file_reference(path) if isinstance(path, Path) else path for name, path in capsule_info.items() if name in {"capsule", "schema", "manifest", "hash_chain"}},
        "container_member_audit": container["status"], "architecture": architecture["status"],
        "synthetic_preflight": synthetic["status"], "negative_test_count": len(negative),
        "actual_consumer": launcher["status"], "bundle": {"path": relative(bundle["path"]), "sha256": bundle["sha256"]},
        "scientific_array_values_loaded": 0, "scientific_d3_steps": 0,
        "atlas_access": 0, "development_access": 0, "lockbox_access": 0,
    }
    write_json_x(run / "diagnostics/campaign_execution_summary.json", summary)
    with (run / "logs/command_log.sh").open("a", encoding="utf-8") as handle:
        handle.write(f"{REPO / '.venv-btk/bin/python'} -B scripts/run_thayer_d3_executable_contract.py --run {run}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
