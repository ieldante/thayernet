#!/usr/bin/env python3
"""Bundle-v3-aware scientific worker for synthetic and authoritative D3."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.d3_execution_bridge_v4 import (  # noqa: E402
    IntegrationRequirementFailure,
    sha256_file,
    validate_bridge,
)


torch: Any = None
np: Any = None
F: Any = None
MappedCompactExpertDecoder: Any = None
canonical_tensor_sha256: Any = None
forward_consistency: Any = None
is_plausible: Any = None
scientific_distance: Any = None
PlausibilityThresholds: Any = None
policy: Any = None
control: Any = None
SemanticStateAdapter: Any = None
SemanticCandidate: Any = None
replay_manifest: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-v4", type=Path, required=True)
    parser.add_argument("--bridge-v4-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict-runtime-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("synthetic_integration_preflight", "authoritative_scientific_d3"),
    )
    return parser.parse_args()


def write_json_x(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, allow_nan=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_csv(path: Path, row: Mapping[str, Any], fields: list[str]) -> None:
    exists = path.exists()
    with path.open("a" if exists else "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))
        handle.flush()
        os.fsync(handle.fileno())


def tensor_sha256(value: Any) -> str:
    cpu = value.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(experts: tuple[Any, Any]) -> str:
    digest = hashlib.sha256()
    for label, expert in zip(("expert_1", "expert_2"), experts):
        for name, value in sorted(expert.state_dict().items()):
            digest.update(f"{label}.{name}".encode("utf-8"))
            digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def load_runtime_modules() -> None:
    global torch, np, F, MappedCompactExpertDecoder, canonical_tensor_sha256
    global forward_consistency, is_plausible, scientific_distance, PlausibilityThresholds
    global policy, control, SemanticStateAdapter, SemanticCandidate, replay_manifest

    import numpy as numpy_module
    import torch as torch_module
    from torch.nn import functional as functional_module
    from src.canonical_tensor_hash import canonical_tensor_sha256 as canonical_hash
    from src.competing_hypotheses import (
        PlausibilityThresholds as Thresholds,
        forward_consistency as forward_evaluator,
        is_plausible as plausibility_evaluator,
        scientific_distance as distance_evaluator,
    )
    from src.output_parameterization import MappedCompactExpertDecoder as Decoder
    import src.d3_policy_engine as policy_module
    import src.d3_control_policy as control_module
    from src.d3_state_machine import (
        SemanticStateAdapter as StateAdapter,
        replay_manifest as replay_state_manifest,
    )
    from src.d3_control_policy import SemanticCandidate as Candidate

    torch = torch_module
    np = numpy_module
    F = functional_module
    MappedCompactExpertDecoder = Decoder
    canonical_tensor_sha256 = canonical_hash
    forward_consistency = forward_evaluator
    is_plausible = plausibility_evaluator
    scientific_distance = distance_evaluator
    PlausibilityThresholds = Thresholds
    policy = policy_module
    control = control_module
    SemanticStateAdapter = StateAdapter
    SemanticCandidate = Candidate
    replay_manifest = replay_state_manifest

    # Eager optimizer imports and scratch lifecycle are completed before the
    # strict runtime phase begins.
    __import__("torch._dynamo")
    try:
        import torch.distributed.nn.jit.instantiator as jit_instantiator

        jit_instantiator._TEMP_DIR._finalizer.detach()
    except (ImportError, AttributeError):
        pass


def validate_requirements_and_policies(context: dict[str, Any], output: Path) -> dict[str, Any]:
    from src.d3_policy_preflight import READINESS_MARKERS, run_launcher_preflight
    from src.d3_requirement_registry import validate_capsule_requirements, validate_registry
    from src.d3_policy_registry import validate_policy_registry

    chain = context["chain"]
    validate_registry(chain["registry"])
    accessed = validate_capsule_requirements(chain["capsule_v2"], chain["registry"])
    if len(accessed) != 180:
        raise IntegrationRequirementFailure("D3I-WORKER-REQUIREMENT-CLOSURE", "180 requirements not consumed")
    policy_registry_path = REPO / chain["bundle_v3"]["policy_registry"]["path"]
    policy_registry = json.loads(policy_registry_path.read_text(encoding="utf-8"))
    validate_policy_registry(policy_registry, verify_implementation=True)
    if len(policy_registry["canonical_policy_ids"]) != 16:
        raise IntegrationRequirementFailure("D3I-WORKER-POLICY-COUNT", "16 policies not validated")
    if os.environ.get("D3_V4_SYNTHETIC_REPLAY_ONLY") == "1":
        preflight_dir = output / "replay_verification/synthetic_policy_preflight"
    elif os.environ.get("D3_V4_REPLAY_ONLY") == "1":
        preflight_dir = output / "replay_verification/policy_preflight"
    elif context["worker_received"]["mode"] == "authoritative_scientific_d3":
        preflight_dir = output / "policy_preflight/worker_science"
    else:
        preflight_dir = output / "policy_preflight/worker_synthetic"
    run_launcher_preflight(
        context["chain"]["bundle_v3_path"],
        context["bridge"]["authorities"]["bundle_v3"]["sha256"],
        preflight_dir,
        REPO,
    )
    result = {
        "requirement_count": len(accessed),
        "policy_count": len(policy_registry["canonical_policy_ids"]),
        "markers": list(READINESS_MARKERS),
        "status": "PASS",
    }
    return result


def construct_experts(v2: dict[str, Any], device: Any, *, load_state: bool) -> tuple[Any, Any]:
    constructor = v2["l0_constructor"]
    if constructor["class"] != "MappedCompactExpertDecoder" or constructor["kwargs"] != {"mapping": "square"}:
        raise IntegrationRequirementFailure("D3I-MODEL-CONSTRUCTOR", "constructor contract mismatch")
    experts = (MappedCompactExpertDecoder("square"), MappedCompactExpertDecoder("square"))
    if any(sum(parameter.numel() for parameter in expert.parameters()) != 46470 for expert in experts):
        raise IntegrationRequirementFailure("D3I-MODEL-PARAMETER-COUNT", "expert parameter count mismatch")
    if load_state:
        checkpoint_path = REPO / v2["initial_state"]["path"]
        if sha256_file(checkpoint_path) != v2["initial_state"]["sha256"]:
            raise IntegrationRequirementFailure("D3I-INITIAL-STATE-SHA", "initial state mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        for index, expert in enumerate(experts, 1):
            prefix = f"expert_{index}."
            expert.load_state_dict(
                {name[len(prefix) :]: value for name, value in state.items() if name.startswith(prefix)},
                strict=True,
            )
    for expert in experts:
        expert.to(device)
        expert.train()
    ids = [id(parameter) for expert in experts for parameter in expert.parameters()]
    if len(ids) != len(set(ids)):
        raise IntegrationRequirementFailure("D3I-MODEL-SHARED-PARAMETER", "expert parameters share identity")
    return experts


def construct_optimizer(experts: tuple[Any, Any], v2: dict[str, Any]) -> Any:
    contract = v2["optimizer_contract"]
    if contract["class"] != "torch.optim.AdamW":
        raise IntegrationRequirementFailure("D3I-OPTIMIZER-CLASS", "optimizer class mismatch")
    parameters = [parameter for expert in experts for parameter in expert.parameters()]
    kwargs = dict(contract["hyperparameters"])
    optimizer = torch.optim.AdamW(parameters, **kwargs)
    optimizer_ids = [id(value) for group in optimizer.param_groups for value in group["params"]]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(map(id, parameters)):
        raise IntegrationRequirementFailure("D3I-OPTIMIZER-MEMBERSHIP", "optimizer membership mismatch")
    return optimizer


def penultimate(expert: Any, features: tuple[Any, Any, Any]) -> Any:
    enc1, enc2, bottleneck = features
    up2 = F.interpolate(bottleneck, size=(30, 30), mode="bilinear", align_corners=False)
    dec2 = expert.dec2(torch.cat((up2, enc2), dim=1))
    up1 = F.interpolate(dec2, size=(60, 60), mode="bilinear", align_corners=False)
    return expert.dec1(torch.cat((up1, enc1), dim=1))


def decode(experts: tuple[Any, Any], features: tuple[Any, Any, Any], scale6: Any) -> tuple[Any, ...]:
    pen1 = penultimate(experts[0], features)
    pen2 = penultimate(experts[1], features)
    raw = torch.stack(
        (experts[0].decomposition_head(pen1), experts[1].decomposition_head(pen2)), dim=1
    )
    mapped = raw.square()
    physical = mapped * scale6.view(1, 1, 6, 1, 1)
    return raw, mapped, physical, pen1, pen2


def direct_cost(predicted: Any, target: Any, scale6: Any) -> Any:
    residual = (predicted - target) / scale6.view(1, 6, 1, 1)
    return residual[:, :3].square().mean(dim=(1, 2, 3)) + residual[:, 3:].square().mean(dim=(1, 2, 3))


def hard_assignment(outputs: Any, targets: Any, scale6: Any) -> tuple[Any, ...]:
    c00 = direct_cost(outputs[:, 0], targets[:, 0], scale6)
    c01 = direct_cost(outputs[:, 0], targets[:, 1], scale6)
    c10 = direct_cost(outputs[:, 1], targets[:, 0], scale6)
    c11 = direct_cost(outputs[:, 1], targets[:, 1], scale6)
    identity = c00 + c11
    swap = c01 + c10
    return torch.minimum(identity, swap).mean(), identity <= swap, swap - identity, (c00, c01, c10, c11)


def synthetic_features(device: Any) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260713)
    values = (
        torch.randn((2, 16, 60, 60), generator=generator) * 0.05,
        torch.randn((2, 32, 30, 30), generator=generator) * 0.05,
        torch.randn((2, 64, 15, 15), generator=generator) * 0.05,
    )
    return tuple(value.to(device) for value in values)


def run_synthetic(context: dict[str, Any], output: Path) -> dict[str, Any]:
    if os.environ.get("D3_V4_SYNTHETIC_REPLAY_ONLY") == "1":
        return replay_synthetic(context, output)
    v2 = context["chain"]["bundle_v2"]
    device = torch.device("cpu")
    experts = construct_experts(v2, device, load_state=True)
    optimizer = construct_optimizer(experts, v2)
    features = synthetic_features(device)
    scale6 = torch.ones(6, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, initial, _, _ = decode(experts, features, scale6)
    targets = initial.detach().clone()
    targets[:, 1] = torch.flip(targets[:, 1], dims=(-1,)) + 1e-5
    before = {name: value.detach().clone() for index, expert in enumerate(experts, 1) for name, value in ((f"expert_{index}.{n}", p) for n, p in expert.named_parameters())}
    optimizer.zero_grad(set_to_none=True)
    raw, mapped, physical, pen1, pen2 = decode(experts, features, scale6)
    loss, wins, margin, _ = hard_assignment(physical, targets, scale6)
    loss.backward()
    clip = v2["optimizer_contract"]["gradient_clipping"]
    torch.nn.utils.clip_grad_norm_([p for e in experts for p in e.parameters()], clip["max_norm"], norm_type=clip["norm_type"])
    optimizer.step()
    with torch.no_grad():
        replay_raw, replay_mapped, replay_physical, _, _ = decode(experts, features, scale6)
    updates = []
    for index, expert in enumerate(experts, 1):
        for name, parameter in expert.named_parameters():
            updates.append(float((parameter.detach() - before[f"expert_{index}.{name}"]).norm()))
    checkpoint = output / "synthetic_preflight/synthetic_worker_checkpoint.pt"
    with checkpoint.open("xb") as handle:
        torch.save(
            {
                "schema_version": "thayer-d3i-synthetic-checkpoint-v1",
                "expert_1_state_dict": experts[0].state_dict(),
                "expert_2_state_dict": experts[1].state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "physical_sha256": tensor_sha256(replay_physical),
                "bridge_sha256": context["bridge_sha256"],
            },
            handle,
        )
    # Exercise policy engine and state adapter on the actual worker path.
    state_root = output / "synthetic_preflight/semantic_states"
    adapter = SemanticStateAdapter(state_root)
    for state, index in (("initial", 0), ("one_step", 1), ("lowest_objective", 1), ("closest_to_d1", 1), ("final", 1)):
        candidate = SemanticCandidate(
            state=state,
            evaluation_index=index,
            step_index=index,
            payload=json.dumps({"state": state, "checkpoint": str(checkpoint.relative_to(output))}, sort_keys=True).encode(),
            scalar_metrics={"objective": float(loss), "synthetic": True},
            optimizer_state_sha256=canonical_json_sha256_safe(optimizer.state_dict()),
            assignment={"prompt_a": bool(wins[0]), "prompt_b": bool(wins[1])},
            event={"code": "SYNTHETIC_WORKER_EVENT"},
            terminal_status="SYNTHETIC_PASS",
            objective=float(loss) if state == "lowest_objective" else None,
            distance_to_d1=0.0 if state == "closest_to_d1" else None,
            semantic_members=("prompt_a.expert_1.requested", "prompt_b.expert_2.companion"),
        )
        adapter.persist(candidate)
    adapter.finalize("SYNTHETIC_PASS", 1, {})
    policy_event = policy.evaluate_runtime_safety({name: False for name, _ in control.STOP_PRECEDENCE})
    assignment_event = policy.evaluate_assignment_diagnostic("stable_identity" if bool(wins[0]) else "stable_swap")
    square_event = policy.evaluate_square_mapping_diagnostic("usable_derivatives")
    result = {
        "status": "PASS",
        "mode": "synthetic_integration_preflight",
        "model_constructions": 2,
        "optimizer_constructions": 1,
        "decoder_forwards": 4,
        "backward_passes": 1,
        "optimizer_steps": 1,
        "loss": float(loss.detach()),
        "finite": bool(torch.isfinite(replay_physical).all()),
        "nonnegative": bool((replay_physical >= 0).all()),
        "nonzero_update_count": sum(value > 0 for value in updates),
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "output_sha256": tensor_sha256(replay_physical),
        "policy_engine_sha256": context["bridge"]["launchers"]["policy_engine"]["sha256"],
        "policy_decisions": [asdict(policy_event), asdict(assignment_event), asdict(square_event)],
        "scientific_array_values_loaded": 0,
        "bundle_v3_sha256": context["bridge"]["authorities"]["bundle_v3"]["sha256"],
    }
    write_json_x(output / "synthetic_preflight/scientific_worker_result.json", result)
    print("BUNDLE_V3_PROPAGATED_TO_SCIENTIFIC_WORKER", flush=True)
    return result


def canonical_json_sha256_safe(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if torch is not None and isinstance(item, torch.Tensor):
            return {"tensor_sha256": tensor_sha256(item), "shape": list(item.shape)}
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return repr(item)
    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def replay_synthetic(context: dict[str, Any], output: Path) -> dict[str, Any]:
    v2 = context["chain"]["bundle_v2"]
    checkpoint_path = output / "synthetic_preflight/synthetic_worker_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    experts = construct_experts(v2, torch.device("cpu"), load_state=False)
    experts[0].load_state_dict(checkpoint["expert_1_state_dict"], strict=True)
    experts[1].load_state_dict(checkpoint["expert_2_state_dict"], strict=True)
    features = synthetic_features(torch.device("cpu"))
    with torch.no_grad():
        _, _, physical, _, _ = decode(experts, features, torch.ones(6))
    status = "PASS" if tensor_sha256(physical) == checkpoint["physical_sha256"] else "FAIL"
    result = {
        "status": status,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "expected_output_sha256": checkpoint["physical_sha256"],
        "actual_output_sha256": tensor_sha256(physical),
        "state_manifest": replay_manifest(output / "synthetic_preflight/semantic_states"),
    }
    write_json_x(output / "synthetic_preflight/synthetic_checkpoint_replay.json", result)
    if status != "PASS":
        raise IntegrationRequirementFailure("D3I-SYNTHETIC-REPLAY", "synthetic replay mismatch")
    return result


def scientific_paths(context: dict[str, Any]) -> dict[str, Path]:
    v2 = context["chain"]["bundle_v2"]
    capsule = context["chain"]["capsule_v1"]
    refs = capsule["scientific_artifact_references"]
    paths = {name: REPO / record["relative_path"] for name, record in refs.items()}
    paths.update(
        {
            "model_state": REPO / v2["initial_state"]["path"],
            "d1_manifest": REPO / v2["evidence"]["capsule_artifact_d1_endpoint_manifest"]["path"],
            "d0": REPO / v2["evidence"]["capsule_artifact_d0_persisted_evidence"]["path"],
            "d2": REPO / v2["evidence"]["capsule_artifact_d2_persisted_evidence"]["path"],
        }
    )
    return paths


def validate_scientific_files(context: dict[str, Any], paths: dict[str, Path]) -> list[dict[str, Any]]:
    v2 = context["chain"]["bundle_v2"]
    capsule = context["chain"]["capsule_v1"]
    expected: dict[str, str] = {
        name: record["sha256"] for name, record in capsule["scientific_artifact_references"].items()
    }
    expected.update(
        {
            "model_state": v2["initial_state"]["sha256"],
            "d1_manifest": v2["evidence"]["capsule_artifact_d1_endpoint_manifest"]["sha256"],
            "d0": v2["evidence"]["capsule_artifact_d0_persisted_evidence"]["sha256"],
            "d2": v2["evidence"]["capsule_artifact_d2_persisted_evidence"]["sha256"],
        }
    )
    rows = []
    for name, path in paths.items():
        actual = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "role": name,
                "path": str(path.relative_to(REPO)),
                "expected_sha256": expected[name],
                "actual_sha256": actual,
                "status": "PASS" if actual == expected[name] else "FAIL",
                "deserialized": False,
            }
        )
    if any(row["status"] != "PASS" for row in rows):
        raise IntegrationRequirementFailure("D3I-SCIENTIFIC-ARTIFACT-SHA", "scientific input hash mismatch")
    return rows


def load_scientific_assets(context: dict[str, Any], output: Path) -> dict[str, Any]:
    paths = scientific_paths(context)
    rows = validate_scientific_files(context, paths)
    capsule = context["chain"]["capsule_v1"]
    v2 = context["chain"]["bundle_v2"]
    cache = torch.load(paths["cached_features"], map_location="cpu", weights_only=True)
    with np.load(paths["p0_target_set"], allow_pickle=False) as handle:
        p0 = {name: np.asarray(handle[name]) for name in handle.files}
    initial = torch.load(paths["initial_decoder_state"], map_location="cpu", weights_only=True)
    model_state = torch.load(paths["model_state"], map_location="cpu", weights_only=True)
    with np.load(paths["d1_endpoint"], allow_pickle=False) as handle:
        d1 = {name: np.asarray(handle[name]) for name in handle.files}
    d1_manifest = json.loads(paths["d1_manifest"].read_text(encoding="utf-8"))
    d0 = torch.load(paths["d0"], map_location="cpu", weights_only=True)
    d2 = torch.load(paths["d2"], map_location="cpu", weights_only=True)
    loaded_members = (
        len(cache.get("prompt_a", ()))
        + len(cache.get("prompt_b", ()))
        + len(p0)
        + len(initial)
        + len(model_state)
        + len(d1)
        + len(d1_manifest)
        + len(d0)
        + len(d2)
    )
    for row in rows:
        row["deserialized"] = True
    write_csv_x(output / "tables/scientific_tensor_load_inventory.csv", rows)
    write_json_x(
        output / "authoritative_inputs/scientific_load_summary.json",
        {
            "container_count": len(rows),
            "member_count": loaded_members,
            "containers": [str(path.relative_to(REPO)) for path in paths.values()],
            "status": "PASS",
        },
    )
    # Contract member checks are evaluated after values are loaded, never before freeze.
    contracts = v2["artifact_member_contracts"]
    feature_shapes = {
        f"{prompt}.{name}": list(value.shape)
        for prompt in ("prompt_a", "prompt_b")
        for name, value in zip(("enc1", "enc2", "bottleneck"), cache[prompt])
    }
    expected_shapes = contracts["cached_features.member_shapes"]
    if feature_shapes != expected_shapes:
        raise IntegrationRequirementFailure("D3I-CACHED-FEATURE-SHAPE", "cached feature shapes mismatch")
    if set(d1) != set(contracts["d1.member_names"]):
        raise IntegrationRequirementFailure("D3I-D1-MEMBER-NAMES", "D1 endpoint member mismatch")
    for name in d1:
        if list(d1[name].shape) != contracts["d1.member_shapes"][name] or str(d1[name].dtype) != contracts["d1.member_dtypes"][name]:
            raise IntegrationRequirementFailure("D3I-D1-MEMBER-CONTRACT", f"D1 member mismatch: {name}")
        if canonical_tensor_sha256(d1[name]) != contracts["d1.member_canonical_hashes"][name]:
            raise IntegrationRequirementFailure("D3I-D1-CANONICAL-HASH", f"D1 hash mismatch: {name}")
    return {
        "paths": paths,
        "cache": cache,
        "p0": p0,
        "initial": initial,
        "model_state": model_state,
        "d1": d1,
        "d1_manifest": d1_manifest,
        "d0": d0,
        "d2": d2,
        "load_rows": rows,
        "member_count": loaded_members,
    }


def assemble_d1(values: dict[str, Any]) -> tuple[Any, Any]:
    return (
        torch.from_numpy(np.stack((values["penultimate_prompt_a_expert_1"], values["penultimate_prompt_b_expert_1"]))),
        torch.from_numpy(np.stack((values["penultimate_prompt_a_expert_2"], values["penultimate_prompt_b_expert_2"]))),
    )


def thresholds_from_capsule(capsule: dict[str, Any]) -> Any:
    values = capsule["forward_plausibility"]["thresholds"]
    return PlausibilityThresholds(
        global_chi_square_mean=values["global_chi_square_mean"],
        per_band_chi_square_mean=tuple(values["per_band_chi_square_mean"][band] for band in ("g", "r", "z")),
        absolute_relative_flux_residual=values["absolute_relative_flux_residual"],
        calibration_count=capsule["forward_plausibility"]["calibration_scene_count"],
    )


def weighted_mse(left: Any, right: Any, scales: Any) -> float:
    return float(np.mean(((left - right) / scales.reshape(3, 1, 1)) ** 2))


def scientific_metrics(physical: Any, assets: dict[str, Any], capsule: dict[str, Any]) -> dict[str, Any]:
    value = physical.detach().to("cpu").numpy()
    truth = assets["p0"]["truth_physical"].astype(np.float64)
    observed = assets["p0"]["blend_physical"].astype(np.float64)
    sky = capsule["observation_configuration"]["scientific_sky_vector"]["values"]
    mean_psf = capsule["observation_configuration"]["mean_psf_fwhm_pixel"]
    thresholds = thresholds_from_capsule(capsule)
    distances = np.zeros((2, 2, 2), dtype=np.float64)
    plausible = np.zeros((2, 2), dtype=bool)
    forward_rows = []
    for prompt_index in (0, 1):
        for expert_index in (0, 1):
            layers = np.stack((value[prompt_index, expert_index, :3], value[prompt_index, expert_index, 3:]))
            forward = forward_consistency(observed, layers, sky)
            plausible[prompt_index, expert_index] = is_plausible(forward, thresholds)
            forward_rows.append(asdict(forward))
            for target_index in (0, 1):
                distances[prompt_index, expert_index, target_index] = scientific_distance(
                    value[prompt_index, expert_index, :3],
                    truth[prompt_index, target_index, :3],
                    mean_psf_fwhm_pixel=mean_psf,
                    image_floor=capsule["numerical_tolerances"]["image_floor"],
                    flux_floor=capsule["numerical_tolerances"]["flux_floor"],
                ).primary_normalized
    own = all(any(plausible[p, e] and distances[p, e, 0] <= 1.0 for e in (0, 1)) for p in (0, 1))
    alternate = all(any(plausible[p, e] and distances[p, e, 1] <= 1.0 for e in (0, 1)) for p in (0, 1))
    both = all(
        (plausible[p, 0] and plausible[p, 1])
        and (
            (distances[p, 0, 0] <= 1.0 and distances[p, 1, 1] <= 1.0)
            or (distances[p, 1, 0] <= 1.0 and distances[p, 0, 1] <= 1.0)
        )
        for p in (0, 1)
    )
    identities = []
    scales = np.asarray(capsule["observation_configuration"]["normalization_scale_grz"], dtype=np.float64)
    for prompt_index in (0, 1):
        for expert_index in (0, 1):
            requested = value[prompt_index, expert_index, :3]
            requested_cost = min(weighted_mse(requested, truth[prompt_index, target, :3], scales) for target in (0, 1))
            companion_cost = min(weighted_mse(requested, truth[prompt_index, target, 3:], scales) for target in (0, 1))
            identities.append(requested_cost < companion_cost)
    diameter = max(
        scientific_distance(value[p, 0, :3], value[p, 1, :3], mean_psf_fwhm_pixel=mean_psf).primary_normalized
        for p in (0, 1)
    )
    return {
        "own_coverage": float(own),
        "alternate_coverage": float(alternate),
        "both_mode_coverage": float(both),
        "set_prompt_swap": float(all(identities)),
        "forward_consistency": float(bool(plausible.all())),
        "forward_plausible_fraction": float(plausible.mean()),
        "expert_diameter": float(diameter),
        "max_own_distance": float(np.max(np.min(distances[:, :, 0], axis=1))),
        "max_alternate_distance": float(np.max(np.min(distances[:, :, 1], axis=1))),
        "forward_rows": forward_rows,
    }


def selected_errors(physical: Any, targets: Any, wins: Any, scale6: Any) -> dict[str, float]:
    predicted = physical.detach().cpu()
    aligned = torch.empty_like(predicted)
    for prompt_index in (0, 1):
        if bool(wins[prompt_index]):
            aligned[prompt_index, 0] = targets[prompt_index, 0]
            aligned[prompt_index, 1] = targets[prompt_index, 1]
        else:
            aligned[prompt_index, 0] = targets[prompt_index, 1]
            aligned[prompt_index, 1] = targets[prompt_index, 0]
    normalized = (predicted - aligned.cpu()) / scale6.cpu().view(1, 1, 6, 1, 1)
    return {
        "requested_source_mse": float(normalized[:, :, :3].square().mean()),
        "companion_source_mse": float(normalized[:, :, 3:].square().mean()),
        "z_band_error": float(normalized[:, :, (2, 5)].square().mean()),
        "image_mse": float(normalized.square().mean()),
        "flux_error": float((predicted.sum((-1, -2)) - aligned.sum((-1, -2))).abs().mean()),
    }


def control_reproduction(assets: dict[str, Any], v2: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    rows = []
    records = {
        "D0": (assets["d0"], v2["evidence"]["capsule_artifact_d0_persisted_evidence"]),
        "D1": (assets["d1_manifest"], v2["evidence"]["capsule_artifact_d1_persisted_evidence"]),
        "D2": (assets["d2"], v2["evidence"]["capsule_artifact_d2_persisted_evidence"]),
    }
    for condition, (payload, expected) in records.items():
        metrics = payload["final_metrics"] if condition == "D1" else payload["metrics"]
        objective = float(payload["final_objective"]) if condition == "D1" else float(metrics["projected_target_loss"] * 2.0)
        actual = {
            "own_coverage": float(metrics["own_coverage"]),
            "alternate_coverage": float(metrics["alternate_coverage"]),
            "both_mode_coverage": float(metrics["both_mode_coverage"]),
            "objective": objective,
        }
        passed = all(abs(actual[key] - float(expected["expected_metrics"][key])) <= 1e-12 for key in actual)
        rows.append({"condition": condition, **actual, "status": "PASS" if passed else "FAIL"})
    write_csv_x(output / "tables/d0_d1_d2_reproduction.csv", rows)
    if any(row["status"] != "PASS" for row in rows):
        raise IntegrationRequirementFailure("D3I-D0-D1-D2-REPRODUCTION", "persisted controls differ")
    return rows


def build_guard(output: Path, runtime: Path, exact_reads: list[Path]) -> Any:
    from scripts.thayer_d3_runtime_guard import GuardPolicy, TwoPhaseGuard

    guard = TwoPhaseGuard(
        GuardPolicy(
            repository_root=REPO,
            fresh_run_root=output,
            runtime_root=runtime,
            access_log=output / "access_guard/scientific_access_log.jsonl",
            blocked_log=output / "access_guard/scientific_blocked_access_log.jsonl",
            exact_read_files=tuple(exact_reads),
            strict_write_roots=(output,),
            strict_atomic_roots=(output,),
            bootstrap_write_roots=(runtime,),
            bootstrap_read_roots=(REPO / ".venv-btk",),
        )
    )
    guard.install()
    guard.transition("strict")
    return guard


def persist_state(adapter: Any, state: str, eval_index: int, step: int, row: dict[str, Any], checkpoint: Path, terminal: str) -> None:
    payload = json.dumps(
        {"state": state, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "metrics": row},
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    adapter.persist(
        SemanticCandidate(
            state=state,
            evaluation_index=eval_index,
            step_index=step,
            payload=payload,
            scalar_metrics={key: value for key, value in row.items() if isinstance(value, (int, float, bool))},
            optimizer_state_sha256=str(row["optimizer_state_sha256"]),
            assignment={"prompt_a": row["assignment_prompt_a"], "prompt_b": row["assignment_prompt_b"]},
            event={"code": terminal},
            terminal_status=terminal,
            objective=float(row["objective"]) if state == "lowest_objective" else None,
            distance_to_d1=float(row["d1_feature_distance"]) if state == "closest_to_d1" else None,
            semantic_members=("prompt_a.expert_1.requested", "prompt_b.expert_2.companion"),
        )
    )


def run_authoritative(context: dict[str, Any], output: Path, runtime: Path) -> dict[str, Any]:
    if os.environ.get("D3_V4_REPLAY_ONLY") == "1":
        return replay_authoritative(context, output, runtime)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise IntegrationRequirementFailure("D3I-RUNTIME-MPS-FALLBACK", "MPS fallback must equal 0")
    if not torch.backends.mps.is_available():
        raise IntegrationRequirementFailure("D3I-RUNTIME-MPS", "MPS unavailable")
    v2 = context["chain"]["bundle_v2"]
    capsule = context["chain"]["capsule_v1"]
    paths = scientific_paths(context)
    pre_rows = validate_scientific_files(context, paths)
    write_csv_x(output / "tables/scientific_artifact_preload_hashes.csv", pre_rows)
    exact_reads = list(paths.values()) + [
        context["chain"]["bundle_v3_path"], context["chain"]["bundle_v2_path"],
        context["chain"]["capsule_v1_path"], context["chain"]["runtime_path"],
        context["chain"]["registry_path"], Path(context["bridge_path"]),
    ]
    guard = build_guard(output, runtime, exact_reads)
    assets = load_scientific_assets(context, output)
    controls = control_reproduction(assets, v2, output)
    device = torch.device("mps")
    features = tuple(torch.cat((left, right), dim=0).to(device) for left, right in zip(assets["cache"]["prompt_a"], assets["cache"]["prompt_b"]))
    targets = torch.from_numpy(assets["p0"]["p0_physical"].astype(np.float32)).to(device)
    scales = torch.tensor(capsule["observation_configuration"]["normalization_scale_grz"], dtype=torch.float32, device=device)
    scale6 = scales.repeat(2)
    feature_hashes_before = [canonical_tensor_sha256(value.detach().cpu()) for value in features]
    target_hashes_before = [canonical_tensor_sha256(targets[p, t].detach().cpu()) for p in (0, 1) for t in (0, 1)]
    d1_cpu = assemble_d1(assets["d1"])
    d1 = tuple(value.to(device) for value in d1_cpu)

    experts = construct_experts(v2, device, load_state=True)
    initial_state_digest = state_sha256(experts)
    with torch.no_grad():
        raw0, mapped0, physical0, pen10, pen20 = decode(experts, features, scale6)
        loss0, wins0, margin0, _ = hard_assignment(physical0, targets, scale6)
    initial_ref = assets["initial"]
    alignment_checks = {
        "raw_exact": torch.equal(raw0.cpu(), initial_ref["raw_normalized"]),
        "mapped_exact": torch.equal(mapped0.cpu(), initial_ref["mapped_normalized"]),
        "physical_exact": torch.equal(physical0.cpu(), initial_ref["physical"]),
        "penultimate_expert_1_exact": torch.equal(pen10.cpu(), initial_ref["penultimate_expert_1"]),
        "penultimate_expert_2_exact": torch.equal(pen20.cpu(), initial_ref["penultimate_expert_2"]),
        "objective_exact": float(loss0.cpu()) == float(initial_ref["target_loss"]),
        "parameter_counts": all(sum(p.numel() for p in expert.parameters()) == 46470 for expert in experts),
        "no_shared_parameters": len({id(p) for e in experts for p in e.parameters()}) == len([p for e in experts for p in e.parameters()]),
        "encoder_constructed": False,
    }
    write_json_x(output / "initial_state/alignment.json", {"status": "PASS" if all(alignment_checks.values()) else "FAIL", "checks": alignment_checks, "initial_objective": float(loss0.cpu())})
    if not all(alignment_checks.values()):
        raise IntegrationRequirementFailure("D3I-INITIAL-ALIGNMENT", "initial D3 alignment mismatch")

    optimizer = construct_optimizer(experts, v2)
    named = [(f"expert_{i}.{name}", parameter) for i, expert in enumerate(experts, 1) for name, parameter in expert.named_parameters()]
    parameters = [parameter for _, parameter in named]
    before = {name: parameter.detach().clone() for name, parameter in named}
    optimizer.zero_grad(set_to_none=True)
    raw1, mapped1, physical1, pen11, pen21 = decode(experts, features, scale6)
    one_loss, one_wins, one_margin, _ = hard_assignment(physical1, targets, scale6)
    one_loss.backward()
    gradients = {name: float(parameter.grad.detach().norm().cpu()) if parameter.grad is not None else 0.0 for name, parameter in named}
    clip = v2["optimizer_contract"]["gradient_clipping"]
    torch.nn.utils.clip_grad_norm_(parameters, clip["max_norm"], norm_type=clip["norm_type"])
    optimizer.step()
    updates = {name: float((parameter.detach() - before[name]).norm().cpu()) for name, parameter in named}
    with torch.no_grad():
        _, _, post_one, _, _ = decode(experts, features, scale6)
    one_trace = {
        "status": "PASS",
        "finite_nonzero_expert_1_gradient": any(value > 0 and math.isfinite(value) for name, value in gradients.items() if name.startswith("expert_1.")),
        "finite_nonzero_expert_2_gradient": any(value > 0 and math.isfinite(value) for name, value in gradients.items() if name.startswith("expert_2.")),
        "final_heads_updated": any(value > 0 for name, value in updates.items() if "decomposition_head" in name),
        "non_final_blocks_updated": any(value > 0 for name, value in updates.items() if "decomposition_head" not in name),
        "features_immutable": feature_hashes_before == [canonical_tensor_sha256(value.detach().cpu()) for value in features],
        "targets_immutable": target_hashes_before == [canonical_tensor_sha256(targets[p, t].detach().cpu()) for p in (0, 1) for t in (0, 1)],
        "finite_outputs": bool(torch.isfinite(post_one).all()),
        "nonnegative_outputs": bool((post_one >= 0).all()),
        "objective": float(one_loss.detach().cpu()),
        "gradient_norm_expert_1": math.sqrt(sum(value * value for name, value in gradients.items() if name.startswith("expert_1."))),
        "gradient_norm_expert_2": math.sqrt(sum(value * value for name, value in gradients.items() if name.startswith("expert_2."))),
        "final_head_update_norm": math.sqrt(sum(value * value for name, value in updates.items() if "decomposition_head" in name)),
        "non_final_update_norm": math.sqrt(sum(value * value for name, value in updates.items() if "decomposition_head" not in name)),
    }
    one_trace["status"] = "PASS" if all(value for key, value in one_trace.items() if key not in {"status", "objective", "gradient_norm_expert_1", "gradient_norm_expert_2", "final_head_update_norm", "non_final_update_norm"}) else "FAIL"
    write_json_x(output / "one_step_trace/one_step_trace.json", one_trace)
    write_csv_x(output / "tables/one_step_parameter_trace.csv", [{"parameter": name, "gradient_norm": gradients[name], "update_norm": updates[name], "role": "final_head" if "decomposition_head" in name else "decoder_body"} for name, _ in named])
    if one_trace["status"] != "PASS":
        raise IntegrationRequirementFailure("D3I-ONE-STEP", "one-step trace failed")

    # The authoritative trajectory starts again from the exact initial state.
    experts = construct_experts(v2, device, load_state=True)
    optimizer = construct_optimizer(experts, v2)
    named = [(f"expert_{i}.{name}", parameter) for i, expert in enumerate(experts, 1) for name, parameter in expert.named_parameters()]
    parameters = [parameter for _, parameter in named]
    adapter = SemanticStateAdapter(output / "semantic_states")
    evaluation_steps = set(v2["execution_budget"]["evaluation_steps"])
    maximum_steps = int(v2["execution_budget"]["maximum_steps"])
    trajectory_path = output / "decoder_training/trajectory.csv"
    policy_log = output / "decoder_training/policy_event_trajectory.jsonl"
    rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    last_gradients = {name: 0.0 for name, _ in named}
    last_updates = {name: 0.0 for name, _ in named}
    prior_physical = None
    prior_assignments = None
    inactivity = {"expert_1": 0, "expert_2": 0}
    collapse_streak = 0
    success_streak = 0
    assignment_flips = 0
    first_seen = {"own": False, "alternate": False, "both": False}
    terminal_event = "BUDGET_EXHAUSTED"
    terminal_step = maximum_steps
    terminal_policy = None
    eval_index = -1

    for step in range(maximum_steps + 1):
        if step > 0:
            before = {name: parameter.detach().clone() for name, parameter in named}
            optimizer.zero_grad(set_to_none=True)
            raw, mapped, physical, pen1, pen2 = decode(experts, features, scale6)
            loss, wins, margin, costs = hard_assignment(physical, targets, scale6)
            loss.backward()
            last_gradients = {name: float(parameter.grad.detach().norm().cpu()) if parameter.grad is not None else 0.0 for name, parameter in named}
            torch.nn.utils.clip_grad_norm_(parameters, clip["max_norm"], norm_type=clip["norm_type"])
            optimizer.step()
            last_updates = {name: float((parameter.detach() - before[name]).norm().cpu()) for name, parameter in named}
        if step not in evaluation_steps:
            continue
        eval_index += 1
        with torch.no_grad():
            raw, mapped, physical, pen1, pen2 = decode(experts, features, scale6)
            loss, wins, margin, costs = hard_assignment(physical, targets, scale6)
        science = scientific_metrics(physical, assets, capsule)
        errors = selected_errors(physical, targets, wins, scale6)
        current_flat = torch.cat((pen1.detach().reshape(-1), pen2.detach().reshape(-1))).cpu()
        d1_flat = torch.cat((d1[0].detach().reshape(-1), d1[1].detach().reshape(-1))).cpu()
        d1_distance = float(torch.linalg.vector_norm(current_flat - d1_flat))
        gradient_expert = {
            label: math.sqrt(sum(value * value for name, value in last_gradients.items() if name.startswith(label + ".")))
            for label in ("expert_1", "expert_2")
        }
        update_expert = {
            label: math.sqrt(sum(value * value for name, value in last_updates.items() if name.startswith(label + ".")))
            for label in ("expert_1", "expert_2")
        }
        physical_change = {
            "expert_1": 0.0 if prior_physical is None else float(torch.linalg.vector_norm(physical[:, 0].detach().cpu() - prior_physical[:, 0])),
            "expert_2": 0.0 if prior_physical is None else float(torch.linalg.vector_norm(physical[:, 1].detach().cpu() - prior_physical[:, 1])),
        }
        assignment = (bool(wins[0]), bool(wins[1]))
        if prior_assignments is not None:
            assignment_flips += sum(a != b for a, b in zip(assignment, prior_assignments))
        prior_assignments = assignment
        normalized = physical.detach().cpu() / scale6.cpu().view(1, 1, 6, 1, 1)
        same_distances = []
        for expert_index in (0, 1):
            same_distances.extend(
                (
                    float(torch.sqrt(torch.mean((normalized[0, expert_index, :3] - normalized[1, expert_index, :3]) ** 2))),
                    float(torch.sqrt(torch.mean((normalized[0, expert_index, 3:] - normalized[1, expert_index, 3:]) ** 2))),
                )
            )
        canonical_swap = float(torch.sqrt(torch.mean((normalized[0, :, :3] - normalized[1, :, 3:]) ** 2)))
        activity_decisions = []
        for label in ("expert_1", "expert_2"):
            decision = policy.evaluate_expert_activity(
                control.ExpertMetrics(
                    expert_id=label,
                    evaluation_index=eval_index,
                    learning_rate=0.0 if step == 0 else v2["optimizer_contract"]["hyperparameters"]["lr"],
                    optimizer_member=True,
                    parameter_finite=all(bool(torch.isfinite(parameter).all()) for name, parameter in named if name.startswith(label + ".")),
                    gradient_finite=math.isfinite(gradient_expert[label]),
                    raw_output_finite=bool(torch.isfinite(raw).all()),
                    physical_output_finite=bool(torch.isfinite(physical).all()),
                    gradient_norm=gradient_expert[label],
                    parameter_update_norm=update_expert[label],
                    physical_output_change_norm=physical_change[label],
                    frozen_parameter_count=0,
                ),
                inactivity[label],
            )
            inactivity[label] = decision.inactivity_streak
            activity_decisions.append(decision)
        death_decision = policy.evaluate_expert_death(activity_decisions)
        prompt_decision = policy.evaluate_prompt_collapse(
            control.PromptMetrics(
                evaluation_index=eval_index,
                prompt_pair_complete=True,
                expert_1_same_requested_distance=same_distances[0],
                expert_1_same_companion_distance=same_distances[1],
                expert_2_same_requested_distance=same_distances[2],
                expert_2_same_companion_distance=same_distances[3],
                canonical_source_swap_distance=canonical_swap,
                set_permutation_match=canonical_swap <= control.NUMERICAL_ZERO,
                ordinary_scene_concentration=False,
            ),
            collapse_streak,
        )
        collapse_streak = prompt_decision.collapse_streak
        assignment_mode = "stable_identity" if assignment == (True, True) else ("stable_swap" if assignment == (False, False) else "prompt_inconsistent")
        assignment_decision = policy.evaluate_assignment_diagnostic(assignment_mode)
        zero_fraction = float((raw.detach().cpu() == 0).float().mean())
        square_decision = policy.evaluate_square_mapping_diagnostic("high_zero_gradient" if zero_fraction == 1.0 else "usable_derivatives")
        safety_events = {name: False for name, _ in control.STOP_PRECEDENCE}
        safety_events.update(
            {
                "NONFINITE": not bool(torch.isfinite(physical).all()) or not math.isfinite(float(loss)),
                "MPS_FALLBACK": any(parameter.device.type != "mps" for parameter in parameters),
                "PHYSICAL_NEGATIVE_OUTPUT": bool((physical < 0).any()),
                "CACHED_FEATURE_MUTATION": feature_hashes_before != [canonical_tensor_sha256(value.detach().cpu()) for value in features],
                "OPTIMIZER_CONTRACT_VIOLATION": set(id(p) for p in parameters) != {id(value) for group in optimizer.param_groups for value in group["params"]},
                "EXPERT_DEAD": death_decision.terminal,
                "PROMPT_COLLAPSE": prompt_decision.terminal,
            }
        )
        safety_decision = policy.evaluate_runtime_safety(safety_events)
        gates = science["own_coverage"] == science["alternate_coverage"] == science["both_mode_coverage"] == 1.0 and science["set_prompt_swap"] == 1.0 and science["forward_consistency"] == 1.0
        success_streak, success_decision = policy.evaluate_success_gate(gates, safety_decision.terminal or death_decision.terminal or prompt_decision.terminal, success_streak)
        budget_decision = policy.evaluate_budget_exhaustion(step, safety_decision.terminal or death_decision.terminal or prompt_decision.terminal or success_decision.terminal)
        safety_events["SUCCESS_GATE"] = success_decision.terminal
        safety_events["BUDGET_EXHAUSTED"] = budget_decision.terminal
        stop_decision = policy.select_terminal_event(safety_events)
        optimizer_sha = canonical_json_sha256_safe(optimizer.state_dict())
        row = {
            "evaluation_index": eval_index,
            "step": step,
            "objective": float(loss.cpu()),
            **{key: value for key, value in science.items() if key != "forward_rows"},
            **errors,
            "assignment_prompt_a": "identity" if assignment[0] else "swap",
            "assignment_prompt_b": "identity" if assignment[1] else "swap",
            "assignment_margin_prompt_a": float(margin[0].cpu()),
            "assignment_margin_prompt_b": float(margin[1].cpu()),
            "assignment_flip_count": assignment_flips,
            "gradient_norm_expert_1": gradient_expert["expert_1"],
            "gradient_norm_expert_2": gradient_expert["expert_2"],
            "update_norm_expert_1": update_expert["expert_1"],
            "update_norm_expert_2": update_expert["expert_2"],
            "d1_feature_distance": d1_distance,
            "raw_minimum": float(raw.min().cpu()),
            "raw_maximum": float(raw.max().cpu()),
            "physical_minimum": float(physical.min().cpu()),
            "physical_negative_fraction": float((physical < 0).float().mean().cpu()),
            "finite_output_fraction": float(torch.isfinite(physical).float().mean().cpu()),
            "square_zero_gradient_fraction": zero_fraction,
            "expert_1_activity": activity_decisions[0].state,
            "expert_2_activity": activity_decisions[1].state,
            "prompt_collapse_state": prompt_decision.state,
            "success_streak": success_streak,
            "selected_terminal_event": stop_decision.event_code or "CONTINUE",
            "optimizer_state_sha256": optimizer_sha,
            "physical_sha256": tensor_sha256(physical),
        }
        checkpoint = output / f"checkpoints/evaluation_step_{step:04d}.pt"
        with checkpoint.open("xb") as handle:
            torch.save(
                {
                    "schema_version": "thayer-d3-checkpoint-v4",
                    "step": step,
                    "expert_1_state_dict": {name: value.detach().cpu() for name, value in experts[0].state_dict().items()},
                    "expert_2_state_dict": {name: value.detach().cpu() for name, value in experts[1].state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": row,
                    "physical": physical.detach().cpu(),
                    "penultimate_expert_1": pen1.detach().cpu(),
                    "penultimate_expert_2": pen2.detach().cpu(),
                    "bridge_sha256": context["bridge_sha256"],
                    "policy_engine_sha256": context["bridge"]["launchers"]["policy_engine"]["sha256"],
                },
                handle,
            )
        row["checkpoint"] = str(checkpoint.relative_to(output))
        if not rows:
            trajectory_fields = list(row)
        append_csv(trajectory_path, row, trajectory_fields)
        rows.append(row)
        checkpoint_rows.append({"step": step, "path": str(checkpoint.relative_to(output)), "sha256": sha256_file(checkpoint), "physical_sha256": row["physical_sha256"]})
        append_jsonl(policy_log, {"step": step, "evaluation_index": eval_index, "activity": [asdict(item) for item in activity_decisions], "death": asdict(death_decision), "prompt": asdict(prompt_decision), "assignment": asdict(assignment_decision), "square": asdict(square_decision), "safety": asdict(safety_decision), "success": asdict(success_decision), "budget": asdict(budget_decision), "stop": asdict(stop_decision), "policy_engine_sha256": context["bridge"]["launchers"]["policy_engine"]["sha256"]})
        first_own = science["own_coverage"] == 1.0 and not first_seen["own"]
        first_alt = science["alternate_coverage"] == 1.0 and not first_seen["alternate"]
        first_both = science["both_mode_coverage"] == 1.0 and not first_seen["both"]
        trigger = policy.semantic_state_triggers({"step_index": step, "optimizer_step_completed": step == 1, "eligible_objective": True, "eligible_distance_to_d1": True, "first_own_coverage": first_own, "first_alternate_coverage": first_alt, "first_both_mode_coverage": first_both, "success": success_decision.terminal, "terminal_failure": stop_decision.terminal and stop_decision.event_code not in {"SUCCESS_GATE", "BUDGET_EXHAUSTED"}, "budget_exhausted": budget_decision.terminal, "final": stop_decision.terminal})
        first_seen["own"] |= first_own
        first_seen["alternate"] |= first_alt
        first_seen["both"] |= first_both
        for state in trigger.details["states"]:
            persist_state(adapter, state, eval_index, step, row, checkpoint.relative_to(output), stop_decision.event_code or "CONTINUE")
        print(json.dumps({"step": step, "objective": row["objective"], "coverage": [science["own_coverage"], science["alternate_coverage"], science["both_mode_coverage"]], "forward": science["forward_consistency"], "d1_distance": d1_distance, "event": row["selected_terminal_event"]}, sort_keys=True), flush=True)
        prior_physical = physical.detach().cpu()
        if stop_decision.terminal:
            terminal_event = str(stop_decision.event_code)
            terminal_step = step
            terminal_policy = asdict(stop_decision)
            break

    if not rows:
        raise IntegrationRequirementFailure("D3I-TRAJECTORY-EMPTY", "no trajectory evaluations")
    adapter.finalize(terminal_event, eval_index, {})
    replay_manifest(output / "semantic_states")
    write_json_x(output / "checkpoints/checkpoint_inventory.json", checkpoint_rows)
    scientific_success = terminal_event == "SUCCESS_GATE"
    optimization_decision = policy.evaluate_optimization_diagnostic(
        informative_gradients=any(row["gradient_norm_expert_1"] > control.NUMERICAL_ZERO and row["gradient_norm_expert_2"] > control.NUMERICAL_ZERO for row in rows[1:]),
        meaningful_feature_movement=any(row["d1_feature_distance"] != rows[0]["d1_feature_distance"] for row in rows[1:]),
        scientific_success=scientific_success,
    )
    implementation_failure = terminal_event in {name for name, _ in control.STOP_PRECEDENCE[:10]}
    outcome_evidence = control.OutcomeEvidence(
        implementation_or_contract_failure=implementation_failure,
        authoritative_trajectory_exists=True,
        full_scientific_success=scientific_success,
        optimization_barrier_supported=(optimization_decision.status == "INFORMATIVE_NO_SUCCESS" and not implementation_failure),
        capacity_barrier_supported=False,
        hard_assignment_barrier_supported=False,
        square_mapping_barrier_supported=False,
        evidence_consistent=True,
        capacity_relies_on_tangent=False,
        tangent_protocol_passed=False,
    )
    outcome_decision = policy.map_scientific_outcome(outcome_evidence)
    outcome = outcome_decision.status
    write_json_x(output / "diagnostics/scientific_outcome.json", {"outcome": outcome, "evidence": asdict(outcome_evidence), "decision": asdict(outcome_decision), "policy_engine_sha256": context["bridge"]["launchers"]["policy_engine"]["sha256"]})
    authorization_context = control.AuthorizationContext(
        outcome=outcome,
        authoritative_d3_scientific_failure=not scientific_success,
        d0_authoritative_pass=controls[0]["status"] == "PASS",
        d1_authoritative_pass=controls[1]["status"] == "PASS",
        prompt_gate_pass=rows[-1]["set_prompt_swap"] == 1.0,
        forward_gate_pass=rows[-1]["forward_consistency"] == 1.0,
        fresh_process_replay_pass=False,
        contract_unchanged=True,
        contract_integrity=not implementation_failure,
        code_runtime_loss_mapping_assignment_defect=implementation_failure,
        tangent_evidence_used=False,
        tangent_protocol_passed=False,
    )
    preliminary_auth = policy.authorize_downstream(authorization_context)
    write_json_x(output / "diagnostics/preliminary_downstream_authorization.json", {"authorization": preliminary_auth.status, "context": asdict(authorization_context), "final_after_replay": False})
    np.savez_compressed(output / "postprocessing_inputs/selected_outputs.npz", initial=assets["initial"]["physical"].numpy(), final=torch.load(output / rows[-1]["checkpoint"], map_location="cpu", weights_only=True)["physical"].numpy())
    summary = {
        "status": "SCIENTIFIC_CLOSURE_PENDING_REPLAY",
        "terminal_event": terminal_event,
        "terminal_step": terminal_step,
        "terminal_policy": terminal_policy,
        "outcome": outcome,
        "preliminary_authorization": preliminary_auth.status,
        "scientific_container_count": len(assets["load_rows"]),
        "scientific_member_count": assets["member_count"],
        "model_constructions": 4,
        "optimizer_constructions": 2,
        "decoder_forwards": terminal_step + len(rows) + 3,
        "optimizer_steps": terminal_step + 1,
        "one_step_trace": one_trace,
        "control_reproduction": controls,
        "initial_state_sha256": initial_state_digest,
        "evaluation_rows": len(rows),
        "own_coverage_ever": any(row["own_coverage"] == 1.0 for row in rows),
        "alternate_coverage_ever": any(row["alternate_coverage"] == 1.0 for row in rows),
        "both_mode_coverage_ever": any(row["both_mode_coverage"] == 1.0 for row in rows),
        "prompt_swap_final": rows[-1]["set_prompt_swap"],
        "forward_consistency_final": rows[-1]["forward_consistency"],
        "expert_death": terminal_event == "EXPERT_DEAD",
        "prompt_collapse": terminal_event == "PROMPT_COLLAPSE",
        "initial_z_band_error": rows[0]["z_band_error"],
        "final_z_band_error": rows[-1]["z_band_error"],
        "initial_d1_distance": rows[0]["d1_feature_distance"],
        "minimum_d1_distance": min(row["d1_feature_distance"] for row in rows),
        "features_unchanged": feature_hashes_before == [canonical_tensor_sha256(value.detach().cpu()) for value in features],
        "targets_unchanged": target_hashes_before == [canonical_tensor_sha256(targets[p, t].detach().cpu()) for p in (0, 1) for t in (0, 1)],
        "atlas_access": 0,
        "development_access": 0,
        "lockbox_access": 0,
        "broader_scene_access": 0,
        "policy_engine_sha256": context["bridge"]["launchers"]["policy_engine"]["sha256"],
        "bridge_sha256": context["bridge_sha256"],
        "guard_snapshot": guard.snapshot(),
    }
    write_json_x(output / "decoder_training/trajectory_summary.json", summary)
    guard.transition("shutdown")
    return summary


def replay_authoritative(context: dict[str, Any], output: Path, runtime: Path) -> dict[str, Any]:
    v2 = context["chain"]["bundle_v2"]
    inventory = json.loads((output / "checkpoints/checkpoint_inventory.json").read_text(encoding="utf-8"))
    state_result = replay_manifest(output / "semantic_states")
    selected = []
    for row in inventory:
        if row["step"] in {inventory[0]["step"], inventory[-1]["step"]}:
            selected.append(row)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    assets = load_scientific_assets_for_replay(context)
    capsule = context["chain"]["capsule_v1"]
    features = tuple(torch.cat((left, right), dim=0).to(device) for left, right in zip(assets["cache"]["prompt_a"], assets["cache"]["prompt_b"]))
    scales = torch.tensor(capsule["observation_configuration"]["normalization_scale_grz"], dtype=torch.float32, device=device).repeat(2)
    replay_rows = []
    for row in selected:
        checkpoint = torch.load(output / row["path"], map_location="cpu", weights_only=True)
        experts = construct_experts(v2, device, load_state=False)
        experts[0].load_state_dict(checkpoint["expert_1_state_dict"], strict=True)
        experts[1].load_state_dict(checkpoint["expert_2_state_dict"], strict=True)
        with torch.no_grad():
            _, _, physical, _, _ = decode(experts, features, scales)
        replay_rows.append({"step": row["step"], "expected_physical_sha256": row["physical_sha256"], "actual_physical_sha256": tensor_sha256(physical), "status": "PASS" if tensor_sha256(physical) == row["physical_sha256"] else "FAIL"})
    status = "PASS" if all(row["status"] == "PASS" for row in replay_rows) else "FAIL"
    write_csv_x(output / "replay_verification/scientific_checkpoint_replay.csv", replay_rows)
    result = {"status": status, "semantic_state_replay": state_result, "selected_checkpoint_count": len(replay_rows), "rows": replay_rows}
    write_json_x(output / "replay_verification/replay_summary.json", result)
    if status != "PASS":
        raise IntegrationRequirementFailure("D3I-SCIENTIFIC-REPLAY", "scientific replay mismatch")
    summary = json.loads((output / "decoder_training/trajectory_summary.json").read_text(encoding="utf-8"))
    auth_context = control.AuthorizationContext(
        outcome=summary["outcome"],
        authoritative_d3_scientific_failure=summary["outcome"] != "L0_FULL_DECODER_SUCCESS",
        d0_authoritative_pass=True,
        d1_authoritative_pass=True,
        prompt_gate_pass=summary["prompt_swap_final"] == 1.0,
        forward_gate_pass=summary["forward_consistency_final"] == 1.0,
        fresh_process_replay_pass=True,
        contract_unchanged=True,
        contract_integrity=summary["outcome"] != "IMPLEMENTATION_OR_CONTRACT_FAILURE",
        code_runtime_loss_mapping_assignment_defect=summary["outcome"] == "IMPLEMENTATION_OR_CONTRACT_FAILURE",
        tangent_evidence_used=False,
        tangent_protocol_passed=False,
    )
    auth = policy.authorize_downstream(auth_context)
    write_json_x(output / "diagnostics/downstream_authorization.json", {"authorization": auth.status, "decision": asdict(auth), "context": asdict(auth_context), "policy_engine_sha256": context["bridge"]["launchers"]["policy_engine"]["sha256"]})
    return result


def load_scientific_assets_for_replay(context: dict[str, Any]) -> dict[str, Any]:
    paths = scientific_paths(context)
    cache = torch.load(paths["cached_features"], map_location="cpu", weights_only=True)
    return {"cache": cache}


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    runtime = args.strict_runtime_root.resolve()
    if not output.is_dir() or not runtime.is_dir():
        raise IntegrationRequirementFailure("D3I-WORKER-FRESH-ROOT", "output/runtime root absent")
    context = validate_bridge(
        repo=REPO,
        bridge_path=args.bridge_v4.resolve(),
        bridge_sha256=args.bridge_v4_sha256,
        require_frozen=args.mode == "authoritative_scientific_d3",
    )
    context["bridge_path"] = str(args.bridge_v4.resolve())
    context["worker_received"] = {
        "bridge_path": str(args.bridge_v4.resolve()),
        "bridge_sha256": args.bridge_v4_sha256,
        "output_root": str(output),
        "runtime_root": str(runtime),
        "mode": args.mode,
        "bundle_v3_sha256": context["bridge"]["authorities"]["bundle_v3"]["sha256"],
        "policy_engine_sha256": context["bridge"]["launchers"]["policy_engine"]["sha256"],
        "base_bundle_v2_sha256": context["bridge"]["authorities"]["base_bundle_v2"]["sha256"],
    }
    load_runtime_modules()
    validation = validate_requirements_and_policies(context, output)
    if os.environ.get("D3_V4_REPLAY_ONLY") != "1" and os.environ.get("D3_V4_SYNTHETIC_REPLAY_ONLY") != "1":
        target = output / (
            "synthetic_preflight/worker_received_arguments.json"
            if args.mode == "synthetic_integration_preflight"
            else "runtime/orchestrator/scientific_worker_received_arguments.json"
        )
        write_json_x(target, {**context["worker_received"], "validation": validation})
    if args.mode == "synthetic_integration_preflight":
        result = run_synthetic(context, output)
    else:
        result = run_authoritative(context, output, runtime)
    print(json.dumps({"worker_status": result["status"], "mode": args.mode}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IntegrationRequirementFailure as exc:
        print(json.dumps({"status": "REJECTED", "canonical_integration_requirement_id": exc.requirement_id, "message": exc.message}, sort_keys=True), flush=True)
        raise SystemExit(2)
