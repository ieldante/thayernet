#!/usr/bin/env python3
"""Fresh guarded replay of a synthetic-only executable D3 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.d3_executable_contract import (
    SCALE6,
    construct_exact_experts,
    decode_experts,
    install_runtime_guard,
    load_production_loss_functions,
    load_project_contract_modules,
    sha256,
    state_sha256,
    tensor_sha256,
    utcnow,
    write_json_x,
)
from src.d3_requirement_registry import validate_capsule_requirements, validate_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--capsule-sha256", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    run = args.run.resolve()
    exact_reads = [
        args.capsule.resolve(), args.registry.resolve(), args.checkpoint.resolve(),
        run / "synthetic_inputs/synthetic_features.npz",
        run / "synthetic_inputs/synthetic_targets.npz",
        repo / "src/models_probabilistic_unet.py", repo / "src/models_two_expert_decoder.py",
        repo / "src/output_parameterization.py", repo / "src/competing_hypotheses.py",
        repo / "outputs/runs/thayer_repository_integrity_20260713_031653/independent_oracles/reference_implementation.py",
        repo / "outputs/runs/thayer_full_l0_d3r_20260713_121652/authoritative_inputs/run_authoritative_d3.py",
        repo / "outputs/runs/thayer_output_parameterization_20260713_023120/checkpoints/ambiguous_one_scene_square.pth",
    ]
    guard = install_runtime_guard(repo, run, exact_reads)
    checks = {
        "capsule_hash": sha256(args.capsule) == args.capsule_sha256,
        "registry_hash": sha256(args.registry) == args.registry_sha256,
        "checkpoint_hash": sha256(args.checkpoint) == args.checkpoint_sha256,
        "mps_available": torch.backends.mps.is_available(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"replay input validation failed: {checks}")
    capsule = json.loads(args.capsule.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    validate_registry(registry)
    validate_capsule_requirements(capsule, registry)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    with np.load(run / "synthetic_inputs/synthetic_features.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(run / "synthetic_inputs/synthetic_targets.npz", allow_pickle=False) as archive:
        targets = torch.from_numpy(np.asarray(archive["targets"]))
    features = tuple(
        torch.from_numpy(np.stack((arrays[f"prompt_a_{level}"], arrays[f"prompt_b_{level}"])))
        for level in ("enc1", "enc2", "bottleneck")
    )
    modules = load_project_contract_modules(repo)
    historical_checkpoint = repo / "outputs/runs/thayer_output_parameterization_20260713_023120/checkpoints/ambiguous_one_scene_square.pth"
    experts, _ = construct_exact_experts(modules, historical_checkpoint, torch.device("cpu"))
    left_result = experts[0].load_state_dict(checkpoint["expert_1_state_dict"], strict=True)
    right_result = experts[1].load_state_dict(checkpoint["expert_2_state_dict"], strict=True)
    for expert in experts:
        expert.to("mps")
        expert.train()
    features_mps = tuple(value.to("mps") for value in features)
    targets_mps = targets.to("mps")
    scale6 = SCALE6.to("mps")
    raw, mapped, physical, _ = decode_experts(experts, features_mps, scale6)
    functions = load_production_loss_functions(repo)
    loss, assignment, margin, _ = functions["hard_physical_set_loss"](physical, targets_mps, scale6)
    for expert in experts:
        expert.zero_grad(set_to_none=True)
    loss.backward()
    gradient_hashes = {
        f"expert_{index}.{name}": tensor_sha256(parameter.grad)
        for index, expert in enumerate(experts, start=1)
        for name, parameter in expert.named_parameters()
        if parameter.grad is not None
    }
    observed = {
        "expert_1_state": state_sha256({name: value.detach().cpu() for name, value in experts[0].state_dict().items()}),
        "expert_2_state": state_sha256({name: value.detach().cpu() for name, value in experts[1].state_dict().items()}),
        "output": tensor_sha256(physical),
        "assignment": assignment.detach().cpu().tolist(),
        "loss": float(loss.detach().cpu()),
        "gradient_hashes": gradient_hashes,
    }
    expected = {
        "expert_1_state": state_sha256(checkpoint["expert_1_state_dict"]),
        "expert_2_state": state_sha256(checkpoint["expert_2_state_dict"]),
        "output": checkpoint["post_output_sha256"],
        "assignment": checkpoint["post_assignment"].tolist(),
        "loss": float(checkpoint["post_loss"]),
        "gradient_hashes": checkpoint["post_gradient_sha256"],
    }
    replay_checks = {
        **checks,
        "strict_state_load": not left_result.missing_keys and not left_result.unexpected_keys and not right_result.missing_keys and not right_result.unexpected_keys,
        "state_hashes": observed["expert_1_state"] == expected["expert_1_state"] and observed["expert_2_state"] == expected["expert_2_state"],
        "output_hash": observed["output"] == expected["output"],
        "assignment": observed["assignment"] == expected["assignment"],
        "loss": observed["loss"] == expected["loss"],
        "gradient_hashes": observed["gradient_hashes"] == expected["gradient_hashes"],
        "physical_finite_nonnegative": bool(torch.isfinite(physical).all()) and int((physical < 0).sum().cpu()) == 0,
    }
    result = {
        "schema_version": "thayer-d3-synthetic-checkpoint-replay-v2",
        "status": "PASS" if all(replay_checks.values()) else "FAIL",
        "checks": replay_checks,
        "expected": expected,
        "observed": observed,
        "guard_snapshot": guard.snapshot(),
        "scientific_array_values_loaded": 0,
        "scientific_d3_steps": 0,
        "completed_utc": utcnow(),
    }
    write_json_x(args.output, result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
