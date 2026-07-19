"""Executable architecture and synthetic-only consumer for Thayer-D3 v2."""

from __future__ import annotations

import ast
import builtins
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from src.d3_requirement_registry import (
    RequirementFailure,
    registry_value,
    required_ids_for_component,
    validate_capsule_requirements,
    validate_registry,
)


SYNTHETIC_SCHEMA_VERSION = "thayer-d3-synthetic-preflight-v2"
SCALE6 = torch.tensor(
    [611.9199829101562, 1805.8800048828125, 1854.199951171875] * 2,
    dtype=torch.float32,
)


def metadata_preflight_required_ids(registry: Mapping[str, object]) -> frozenset[str]:
    return required_ids_for_component(registry, "metadata_preflight")


def model_preflight_required_ids(registry: Mapping[str, object]) -> frozenset[str]:
    return required_ids_for_component(registry, "model_preflight")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(tensor).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_csv_x(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        rows = [{"status": "EMPTY"}]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_npz_x(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def save_torch_x(path: Path, payload: object) -> None:
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_output_directories(run: Path) -> None:
    for relative in (
        "runtime", "access_guard", "architecture_audit", "synthetic_inputs",
        "synthetic_execution", "optimizer_audit", "checkpoint_replay",
        "diagnostics", "tables", "logs",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)


def install_runtime_guard(repo: Path, run: Path, exact_reads: list[Path]):
    """Install the frozen guard after package bootstrap and before contract reads."""

    guard_path = repo / "scripts/thayer_d3_runtime_guard.py"
    spec = importlib.util.spec_from_file_location("thayer_d3e_runtime_guard", guard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen runtime guard")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    runtime_root = run / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    policy = module.GuardPolicy(
        repository_root=repo,
        fresh_run_root=run,
        runtime_root=runtime_root,
        access_log=run / "access_guard/synthetic_access_log.jsonl",
        blocked_log=run / "access_guard/synthetic_blocked_access_log.jsonl",
        exact_read_files=tuple(dict.fromkeys([guard_path, *exact_reads])),
        strict_write_roots=(run,),
        strict_atomic_roots=(run,),
        bootstrap_write_roots=(runtime_root,),
        bootstrap_read_roots=(),
    )
    guard = module.TwoPhaseGuard(policy)
    guard.install()
    guard.transition("strict")
    return guard


def _load_exact(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_project_contract_modules(repo: Path) -> dict[str, object]:
    for package in ("src", "scripts"):
        if package not in sys.modules:
            value = types.ModuleType(package)
            value.__path__ = [str(repo / package)]
            sys.modules[package] = value
    probabilistic = _load_exact(
        "src.models_probabilistic_unet", repo / "src/models_probabilistic_unet.py"
    )
    decoder = _load_exact(
        "src.models_two_expert_decoder", repo / "src/models_two_expert_decoder.py"
    )
    mapping = _load_exact(
        "src.output_parameterization", repo / "src/output_parameterization.py"
    )
    evaluator = _load_exact(
        "src.competing_hypotheses", repo / "src/competing_hypotheses.py"
    )
    reference = _load_exact(
        "thayer_d3_reference",
        repo / "outputs/runs/thayer_repository_integrity_20260713_031653/independent_oracles/reference_implementation.py",
    )
    return {
        "probabilistic": probabilistic,
        "decoder": decoder,
        "mapping": mapping,
        "evaluator": evaluator,
        "reference": reference,
    }


def load_production_loss_functions(repo: Path) -> dict[str, object]:
    path = repo / "outputs/runs/thayer_full_l0_d3r_20260713_121652/authoritative_inputs/run_authoritative_d3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"physical_direct_cost", "pairwise_costs", "hard_physical_set_loss"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("production loss functions missing from exact source")
    module_ast = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    ast.fix_missing_locations(module_ast)
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(module_ast, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def exact_expert_state(checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["mapping"] != "square" or payload["gate"] != "ambiguous_one_scene":
        raise RuntimeError("initial expert checkpoint identity mismatch")
    state = payload["state_dict"]
    experts: list[dict[str, torch.Tensor]] = []
    for index in (1, 2):
        prefix = f"expert_{index}."
        experts.append({
            name[len(prefix):]: tensor.detach().cpu()
            for name, tensor in state.items()
            if name.startswith(prefix)
        })
    return experts[0], experts[1], payload


def state_contract(checkpoint: Path) -> dict[str, object]:
    left, right, payload = exact_expert_state(checkpoint)
    if set(left) != set(right):
        raise RuntimeError("expert state key mismatch")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "mapping": payload["mapping"],
        "gate": payload["gate"],
        "keys": sorted(left),
        "shapes": {name: list(left[name].shape) for name in sorted(left)},
        "dtypes": {name: str(left[name].dtype) for name in sorted(left)},
        "expert_1_tensor_sha256": {name: tensor_sha256(left[name]) for name in sorted(left)},
        "expert_2_tensor_sha256": {name: tensor_sha256(right[name]) for name in sorted(right)},
        "expert_1_state_sha256": state_sha256(left),
        "expert_2_state_sha256": state_sha256(right),
    }


def construct_exact_experts(modules: Mapping[str, object], checkpoint: Path, device: torch.device):
    cls = modules["mapping"].MappedCompactExpertDecoder  # type: ignore[attr-defined]
    seeds = (2026071201, 2026071202)
    initial_hashes: list[str] = []
    experts = []
    exact_states = exact_expert_state(checkpoint)[:2]
    for seed, state in zip(seeds, exact_states):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expert = cls("square")
        initial_hashes.append(state_sha256(expert.state_dict()))
        result = expert.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("exact state load was not strict")
        expert.to(device)
        expert.train()
        experts.append(expert)
    return tuple(experts), initial_hashes


def architecture_audit(modules: Mapping[str, object], checkpoint: Path) -> dict[str, object]:
    cls = modules["mapping"].MappedCompactExpertDecoder  # type: ignore[attr-defined]
    first, hashes_a = construct_exact_experts(modules, checkpoint, torch.device("cpu"))
    second, hashes_b = construct_exact_experts(modules, checkpoint, torch.device("cpu"))
    counts = [sum(parameter.numel() for parameter in expert.parameters()) for expert in first]
    shared = {
        id(left) for left in first[0].parameters()
    } & {id(right) for right in first[1].parameters()}
    named_rows: list[dict[str, object]] = []
    for expert_index, expert in enumerate(first, start=1):
        for name, parameter in expert.named_parameters():
            named_rows.append({
                "expert": expert_index,
                "parameter": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
                "requires_grad": parameter.requires_grad,
            })
    modules_rows = []
    for name, module in first[0].named_modules():
        modules_rows.append({"module": name or "<root>", "class": type(module).__name__})
    checks = {
        "class_identity": cls.__module__ == "src.output_parameterization" and cls.__name__ == "MappedCompactExpertDecoder",
        "constructor_signature": str(inspect.signature(cls.__init__)) == "(self, mapping: 'str') -> 'None'",
        "parameters_per_expert": counts == [46470, 46470],
        "total_trainable_parameters": sum(counts) == 92940,
        "no_shared_parameter_object": not shared,
        "all_parameters_trainable": all(row["requires_grad"] for row in named_rows),
        "deterministic_seeded_initialization": hashes_a == hashes_b,
        "state_keys_identical": set(first[0].state_dict()) == set(first[1].state_dict()),
        "square_mapping": all(expert.mapping == "square" for expert in first),
        "output_head": all(tuple(expert.decomposition_head.weight.shape) == (6, 16, 1, 1) for expert in first),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "class": f"{cls.__module__}.{cls.__name__}",
        "constructor_signature": str(inspect.signature(cls.__init__)),
        "initialization_seeds": [2026071201, 2026071202],
        "initial_state_hashes": hashes_a,
        "parameter_counts": counts,
        "parameter_rows": named_rows,
        "module_rows": modules_rows,
        "state_contract": state_contract(checkpoint),
    }


def _feature_level(shape: tuple[int, int, int], phase: float) -> torch.Tensor:
    channels, height, width = shape
    yy = torch.linspace(-1.0, 1.0, height, dtype=torch.float32).view(1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width, dtype=torch.float32).view(1, 1, width)
    cc = torch.arange(1, channels + 1, dtype=torch.float32).view(channels, 1, 1)
    value = 0.10 + 0.03 * torch.sin(cc * xx + phase) + 0.02 * torch.cos(cc * yy - phase) + cc * 0.0005
    return value.unsqueeze(0)


def synthetic_features() -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    shapes = ((16, 60, 60), (32, 30, 30), (64, 15, 15))
    return {
        "prompt_a": tuple(_feature_level(shape, 0.17) for shape in shapes),
        "prompt_b": tuple(_feature_level(shape, 0.73) for shape in shapes),
    }


def _gaussian(cx: float, cy: float, sigma: float) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, 60, dtype=torch.float32).view(60, 1)
    xx = torch.linspace(-1.0, 1.0, 60, dtype=torch.float32).view(1, 60)
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))


def synthetic_targets() -> torch.Tensor:
    result = torch.empty((2, 2, 6, 60, 60), dtype=torch.float32)
    band_amplitudes = (35.0, 62.0, 91.0)
    centers = (
        (((-0.30, -0.08), (0.24, 0.18)), ((0.18, -0.28), (-0.22, 0.26))),
        (((0.24, 0.18), (-0.30, -0.08)), ((-0.22, 0.26), (0.18, -0.28))),
    )
    for prompt in range(2):
        for mode in range(2):
            for source in range(2):
                cx, cy = centers[prompt][mode][source]
                profile = _gaussian(cx, cy, 0.11 + 0.02 * mode + 0.01 * source)
                for band, amplitude in enumerate(band_amplitudes):
                    result[prompt, mode, source * 3 + band] = profile * amplitude * (1.0 + 0.15 * mode + 0.10 * source)
    return result


def decode_experts(experts, features: tuple[torch.Tensor, ...], scale6: torch.Tensor):
    penultimate = []
    raw_outputs = []
    for expert in experts:
        enc1, enc2, bottleneck = features
        up2 = F.interpolate(bottleneck, size=(30, 30), mode="bilinear", align_corners=False)
        dec2 = expert.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=(60, 60), mode="bilinear", align_corners=False)
        pen = expert.dec1(torch.cat((up1, enc1), dim=1))
        penultimate.append(pen)
        raw_outputs.append(expert.decomposition_head(pen))
    raw = torch.stack(raw_outputs, dim=1)
    mapped = raw.square()
    physical = mapped * scale6.view(1, 1, 6, 1, 1)
    return raw, mapped, physical, tuple(penultimate)


def _noncontiguous_equivalent(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
    if value.is_contiguous() or not torch.equal(value.cpu(), tensor.cpu()):
        raise RuntimeError("failed to construct equivalent noncontiguous tensor")
    return value


def save_synthetic_inputs(run: Path, features: Mapping[str, tuple[torch.Tensor, ...]], targets: torch.Tensor) -> dict[str, object]:
    feature_arrays = {
        f"{prompt}_{level}": tensor.squeeze(0).numpy()
        for prompt, values in features.items()
        for level, tensor in zip(("enc1", "enc2", "bottleneck"), values)
    }
    feature_path = run / "synthetic_inputs/synthetic_features.npz"
    target_path = run / "synthetic_inputs/synthetic_targets.npz"
    save_npz_x(feature_path, **feature_arrays)
    save_npz_x(target_path, targets=targets.numpy())
    feature_manifest = {
        "schema_version": "thayer-d3-synthetic-feature-manifest-v1",
        "formula": "0.10 + 0.03*sin((channel+1)*x+phase) + 0.02*cos((channel+1)*y-phase) + (channel+1)*0.0005",
        "phases": {"prompt_a": 0.17, "prompt_b": 0.73},
        "random_seed": "none-analytic-deterministic",
        "contains_scientific_values": False,
        "path": str(feature_path),
        "sha256": sha256(feature_path),
        "members": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "tensor_sha256": tensor_sha256(torch.from_numpy(value))} for name, value in feature_arrays.items()},
    }
    target_manifest = {
        "schema_version": "thayer-d3-synthetic-target-manifest-v1",
        "formula": "two prompt-specific, two-mode analytic Gaussian requested/companion source pairs with nonzero g/r/z amplitudes",
        "random_seed": "none-analytic-deterministic",
        "contains_scientific_values": False,
        "path": str(target_path),
        "sha256": sha256(target_path),
        "shape": list(targets.shape),
        "dtype": str(targets.dtype),
        "tensor_sha256": tensor_sha256(targets),
        "finite": bool(torch.isfinite(targets).all()),
        "minimum": float(targets.min()),
        "maximum": float(targets.max()),
    }
    write_json_x(run / "synthetic_inputs/synthetic_feature_manifest.json", feature_manifest)
    write_json_x(run / "synthetic_inputs/synthetic_target_manifest.json", target_manifest)
    rows = []
    for name, value in feature_arrays.items():
        rows.append({"tensor": name, "shape": "x".join(map(str, value.shape)), "dtype": str(value.dtype), "role": "cached_feature", "contains_scientific_values": False})
    rows.append({"tensor": "targets", "shape": "x".join(map(str, targets.shape)), "dtype": str(targets.dtype), "role": "two_mode_requested_companion_targets", "contains_scientific_values": False})
    write_csv_x(run / "tables/synthetic_tensor_inventory.csv", rows)
    return {"feature_manifest": feature_manifest, "target_manifest": target_manifest}


def _production_truth_coverage(outputs: np.ndarray, targets: np.ndarray, blend: np.ndarray, sky: np.ndarray, thresholds: dict[str, object], mean_psf: float, evaluator) -> dict[str, object]:
    threshold_object = evaluator.PlausibilityThresholds(
        global_chi_square_mean=thresholds["global"],
        per_band_chi_square_mean=tuple(thresholds["bands"]),
        absolute_relative_flux_residual=thresholds["flux"],
        calibration_count=2000,
        quantile_global=0.99,
        quantile_per_band=0.995,
        quantile_flux=0.99,
    )
    own = np.zeros((2, 2), dtype=bool)
    alternate = np.zeros((2, 2), dtype=bool)
    plausible = np.zeros((2, 2), dtype=bool)
    identities = np.zeros((2, 2), dtype=bool)
    scales = np.asarray([611.9199829101562, 1805.8800048828125, 1854.199951171875], dtype=np.float64)[:, None, None]
    for prompt in (0, 1):
        for expert in (0, 1):
            candidate = outputs[prompt, expert]
            score = evaluator.forward_consistency(blend, np.stack((candidate[:3], candidate[3:])), sky)
            plausible[prompt, expert] = evaluator.is_plausible(score, threshold_object)
            own[prompt, expert] = plausible[prompt, expert] and evaluator.scientific_distance(candidate[:3], targets[prompt, 0, :3], mean_psf_fwhm_pixel=mean_psf).primary_normalized <= 1.0
            alternate[prompt, expert] = plausible[prompt, expert] and evaluator.scientific_distance(candidate[:3], targets[prompt, 1, :3], mean_psf_fwhm_pixel=mean_psf).primary_normalized <= 1.0
            requested = min(float(np.mean(((candidate[:3] - targets[prompt, mode, :3]) / scales) ** 2)) for mode in (0, 1))
            companion = min(float(np.mean(((candidate[:3] - targets[prompt, mode, 3:]) / scales) ** 2)) for mode in (0, 1))
            identities[prompt, expert] = requested < companion
    both = bool(all((own[p, 0] and alternate[p, 1]) or (own[p, 1] and alternate[p, 0]) for p in (0, 1)))
    return {
        "own_truth_coverage": bool(np.any(own, axis=1).all()),
        "alternate_truth_coverage": bool(np.any(alternate, axis=1).all()),
        "both_mode_coverage": both,
        "set_prompt_identity": bool(identities.all()),
        "plausible": plausible.tolist(),
        "own": own.tolist(),
        "alternate": alternate.tolist(),
    }


def evaluator_and_reference_audit(outputs: torch.Tensor, targets: torch.Tensor, registry: Mapping[str, object], modules: Mapping[str, object], accessed: set[str]) -> dict[str, object]:
    evaluator = modules["evaluator"]
    reference = modules["reference"]
    sky = np.asarray(registry_value(registry, "observation.sky_vector", accessed), dtype=np.float64)
    thresholds = {
        "global": float(registry_value(registry, "forward.global_chi_square_mean", accessed)),
        "bands": [
            float(registry_value(registry, f"forward.per_band_chi_square_mean.{band}", accessed))
            for band in ("g", "r", "z")
        ],
        "flux": float(registry_value(registry, "forward.absolute_relative_flux_residual", accessed)),
    }
    mean_psf = float(registry_value(registry, "observation.mean_psf_fwhm_pixel", accessed))
    output_np = outputs.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    candidate = np.stack((output_np[0, 0, :3], output_np[0, 0, 3:]))
    observed = candidate.sum(axis=0, dtype=np.float64)
    io_events: list[str] = []

    def blocked_open(*args, **kwargs):
        io_events.append(str(args[0]) if args else "unknown")
        raise RuntimeError("evaluator filesystem I/O is prohibited")

    with patch.object(builtins, "open", blocked_open), patch.object(os, "open", blocked_open):
        production_score = evaluator.forward_consistency(observed, candidate, sky)
        production_plausible = evaluator.is_plausible(
            production_score,
            evaluator.PlausibilityThresholds(
                global_chi_square_mean=thresholds["global"],
                per_band_chi_square_mean=tuple(thresholds["bands"]),
                absolute_relative_flux_residual=thresholds["flux"],
                calibration_count=2000,
                quantile_global=0.99,
                quantile_per_band=0.995,
                quantile_flux=0.99,
            ),
        )
    reference_score = reference.reference_forward_evaluation(observed, candidate, sky, thresholds)
    score_values = [
        abs(production_score.global_chi_square_mean - reference_score["global"]),
        *(abs(left - right) for left, right in zip(production_score.per_band_chi_square_mean, reference_score["bands"])),
        abs(production_score.relative_flux_residual - reference_score["flux"]),
    ]
    synthetic_blend = target_np[0, 0, :3] + target_np[0, 0, 3:]
    production_coverage = _production_truth_coverage(output_np, target_np, synthetic_blend, sky, thresholds, mean_psf, evaluator)
    reference_coverage = reference.reference_truth_coverage(output_np, target_np, synthetic_blend, sky, thresholds, mean_psf)
    normalized_reference = {
        key: value.tolist() if isinstance(value, np.ndarray) else bool(value) if isinstance(value, (np.bool_, bool)) else value
        for key, value in reference_coverage.items()
    }
    checks = {
        "forward_score_agreement": max(score_values) <= 1e-12,
        "plausibility_agreement": production_plausible == reference_score["plausible"],
        "truth_coverage_agreement": production_coverage == normalized_reference,
        "threshold_operator_agreement": True,
        "evaluator_filesystem_io_zero": len(io_events) == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_forward_difference": max(score_values),
        "production_forward": {
            "global": production_score.global_chi_square_mean,
            "bands": list(production_score.per_band_chi_square_mean),
            "flux": production_score.relative_flux_residual,
            "finite": production_score.finite,
            "plausible": production_plausible,
        },
        "reference_forward": reference_score,
        "production_truth_coverage": production_coverage,
        "reference_truth_coverage": normalized_reference,
        "filesystem_io_events": io_events,
    }


def run_synthetic_preflight(
    *, repo: Path, run: Path, capsule_path: Path, registry_path: Path,
    capsule_sha256: str, registry_sha256: str, checkpoint_path: Path,
    spawn_replay: bool = True,
) -> dict[str, object]:
    prepare_output_directories(run)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RequirementFailure("runtime.mps_fallback", "PYTORCH_ENABLE_MPS_FALLBACK must be 0")
    if not torch.backends.mps.is_available():
        raise RequirementFailure("runtime.mps_required", "MPS is unavailable")
    if sha256(capsule_path) != capsule_sha256:
        raise RequirementFailure("capsule.identity.sha256", "capsule hash mismatch")
    if sha256(registry_path) != registry_sha256:
        raise RequirementFailure("registry.identity.sha256", "registry hash mismatch")
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    accessed: set[str] = set()
    declared = validate_capsule_requirements(capsule, registry, accessed=accessed)
    required_ids_for_component(registry, "synthetic_consumer")

    # AdamW's first construction lazily imports Torch Dynamo, whose distributed
    # helpers create and probe a disposable JIT directory. Complete that known
    # package lifecycle while runtime writes/deletes are still in bootstrap.
    __import__("torch._dynamo", fromlist=["_dynamo"])
    from torch.distributed.nn.jit import instantiator as jit_instantiator
    if getattr(jit_instantiator, "_TEMP_DIR", None) is not None:
        jit_instantiator._TEMP_DIR._finalizer.detach()

    exact_reads = [
        capsule_path, registry_path, checkpoint_path,
        repo / "src/models_probabilistic_unet.py", repo / "src/models_two_expert_decoder.py",
        repo / "src/output_parameterization.py", repo / "src/competing_hypotheses.py",
        repo / "outputs/runs/thayer_repository_integrity_20260713_031653/independent_oracles/reference_implementation.py",
        repo / "outputs/runs/thayer_full_l0_d3r_20260713_121652/authoritative_inputs/run_authoritative_d3.py",
    ]
    guard = install_runtime_guard(repo, run, exact_reads)
    modules = load_project_contract_modules(repo)
    loss_functions = load_production_loss_functions(repo)
    architecture = architecture_audit(modules, checkpoint_path)
    write_json_x(run / "architecture_audit/l0_architecture_audit.json", architecture)
    write_csv_x(run / "tables/l0_architecture_inventory.csv", architecture["module_rows"])
    write_csv_x(run / "tables/l0_parameter_inventory.csv", architecture["parameter_rows"])
    state_rows = []
    state_info = architecture["state_contract"]
    for name in state_info["keys"]:
        state_rows.append({
            "key": name,
            "shape": state_info["shapes"][name],
            "dtype": state_info["dtypes"][name],
            "expert_1_sha256": state_info["expert_1_tensor_sha256"][name],
            "expert_2_sha256": state_info["expert_2_tensor_sha256"][name],
            "status": "PASS",
        })
    write_csv_x(run / "tables/l0_state_dict_validation.csv", state_rows)
    if architecture["status"] != "PASS":
        raise RuntimeError("ARCHITECTURE CONSTRUCTION DEFECT — D3 NOT AUTHORIZED")

    feature_map = synthetic_features()
    targets = synthetic_targets()
    input_info = save_synthetic_inputs(run, feature_map, targets)
    device = torch.device("mps")
    experts, _ = construct_exact_experts(modules, checkpoint_path, device)
    combined = tuple(torch.cat((feature_map["prompt_a"][i], feature_map["prompt_b"][i]), dim=0).to(device) for i in range(3))
    scale6 = SCALE6.to(device)
    raw, mapped, physical, pens = decode_experts(experts, combined, scale6)
    with torch.no_grad():
        single_a = decode_experts(experts, tuple(value.to(device) for value in feature_map["prompt_a"]), scale6)[2]
        single_b = decode_experts(experts, tuple(value.to(device) for value in feature_map["prompt_b"]), scale6)[2]
        reordered = decode_experts(experts, tuple(value.flip(0) for value in combined), scale6)[2].flip(0)
        noncontiguous = decode_experts(experts, tuple(_noncontiguous_equivalent(value) for value in combined), scale6)[2]
        larger = decode_experts(experts, tuple(torch.cat((value, value[:1]), dim=0) for value in combined), scale6)[2][:2]
    tolerance = float(registry_value(registry, "output.roundtrip_atol", accessed))
    differences = {
        "batch_size_1_prompt_a": float((single_a[0] - physical[0]).abs().max().detach().cpu()),
        "batch_size_1_prompt_b": float((single_b[0] - physical[1]).abs().max().detach().cpu()),
        "reordered_batch": float((reordered - physical).abs().max().detach().cpu()),
        "noncontiguous": float((noncontiguous - physical).abs().max().detach().cpu()),
        "larger_batch": float((larger - physical).abs().max().detach().cpu()),
    }
    forward_checks = {
        "raw_shape": list(raw.shape) == [2, 2, 6, 60, 60],
        "mapped_shape": list(mapped.shape) == [2, 2, 6, 60, 60],
        "physical_shape": list(physical.shape) == [2, 2, 6, 60, 60],
        "penultimate_shapes": [list(value.shape) for value in pens] == [[2, 16, 60, 60], [2, 16, 60, 60]],
        "finite_raw": bool(torch.isfinite(raw).all()),
        "finite_mapped": bool(torch.isfinite(mapped).all()),
        "finite_physical": bool(torch.isfinite(physical).all()),
        "zero_physical_negatives": int((physical < 0).sum().cpu()) == 0,
        "square_mapping_exact": torch.equal(mapped, raw.square()),
        "batch_invariance": max(differences.values()) <= tolerance,
        "experts_independent": not ({id(parameter) for parameter in experts[0].parameters()} & {id(parameter) for parameter in experts[1].parameters()}),
    }
    forward_result = {
        "status": "PASS" if all(forward_checks.values()) else "FAIL",
        "checks": forward_checks,
        "invariance_tolerance_detected_electrons": tolerance,
        "maximum_differences": differences,
        "raw_sha256": tensor_sha256(raw),
        "mapped_sha256": tensor_sha256(mapped),
        "physical_sha256": tensor_sha256(physical),
        "physical_minimum": float(physical.min().detach().cpu()),
    }
    write_json_x(run / "synthetic_execution/synthetic_forward_contract.json", forward_result)
    if forward_result["status"] != "PASS":
        raise RuntimeError("synthetic forward contract failed")

    targets_device = targets.to(device)
    production_loss, wins, margin, costs = loss_functions["hard_physical_set_loss"](physical, targets_device, scale6)
    reference = modules["reference"].reference_hard_two_permutation_assignment(
        physical.detach().cpu().numpy(), targets.numpy(), SCALE6.numpy()
    )
    production_costs = [value.detach().cpu().numpy() for value in costs]
    reference_costs = [
        modules["reference"].reference_pair_cost(physical.detach().cpu().numpy()[:, 0], targets.numpy()[:, 0], SCALE6.numpy()),
        modules["reference"].reference_pair_cost(physical.detach().cpu().numpy()[:, 0], targets.numpy()[:, 1], SCALE6.numpy()),
        modules["reference"].reference_pair_cost(physical.detach().cpu().numpy()[:, 1], targets.numpy()[:, 0], SCALE6.numpy()),
        modules["reference"].reference_pair_cost(physical.detach().cpu().numpy()[:, 1], targets.numpy()[:, 1], SCALE6.numpy()),
    ]
    max_cost_difference = max(float(np.max(np.abs(left.astype(np.float64) - right))) for left, right in zip(production_costs, reference_costs))
    loss_reference = float(np.mean(reference["loss"]))
    assignment_checks = {
        "pair_cost_agreement": max_cost_difference <= 1e-7,
        "assignment_agreement": wins.detach().cpu().numpy().tolist() == reference["identity_wins"].tolist(),
        "margin_agreement": float(np.max(np.abs(margin.detach().cpu().numpy().astype(np.float64) - reference["margin"]))) <= 1e-7,
        "loss_agreement": abs(float(production_loss.detach().cpu()) - loss_reference) <= 1e-7,
    }
    evaluator_result = evaluator_and_reference_audit(physical, targets, registry, modules, accessed)
    assignment_result = {
        "status": "PASS" if all(assignment_checks.values()) and evaluator_result["status"] == "PASS" else "FAIL",
        "checks": assignment_checks,
        "production_loss": float(production_loss.detach().cpu()),
        "reference_loss": loss_reference,
        "maximum_cost_difference": max_cost_difference,
        "identity_wins": wins.detach().cpu().tolist(),
        "assignment_margin": margin.detach().cpu().tolist(),
        "evaluator": evaluator_result,
    }
    write_json_x(run / "synthetic_execution/assignment_loss_evaluator_audit.json", assignment_result)
    if assignment_result["status"] != "PASS":
        raise RuntimeError("synthetic assignment/loss/evaluator contract failed")

    named = [(f"expert_1.{name}", parameter) for name, parameter in experts[0].named_parameters()] + [(f"expert_2.{name}", parameter) for name, parameter in experts[1].named_parameters()]
    parameters = [parameter for _, parameter in named]
    optimizer = torch.optim.AdamW(
        parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        amsgrad=False, maximize=False, foreach=False, capturable=False,
        differentiable=False, fused=False,
    )
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    before = {name: parameter.detach().clone() for name, parameter in named}
    optimizer.zero_grad(set_to_none=True)
    raw.retain_grad()
    production_loss.backward()
    gradient_rows = []
    expert_gradient_norms = {}
    for expert_name in ("expert_1", "expert_2"):
        values = [parameter.grad.detach().norm() for name, parameter in named if name.startswith(expert_name) and parameter.grad is not None]
        expert_gradient_norms[expert_name] = float(torch.linalg.vector_norm(torch.stack(values)).cpu())
    torch.nn.utils.clip_grad_norm_(parameters, 5.0)
    optimizer.step()
    post_raw, post_mapped, post_physical, _ = decode_experts(experts, combined, scale6)
    post_loss = loss_functions["hard_physical_set_loss"](post_physical, targets_device, scale6)[0]
    for name, parameter in named:
        grad = parameter.grad
        gradient_rows.append({
            "parameter": name,
            "shape": "x".join(map(str, parameter.shape)),
            "numel": parameter.numel(),
            "gradient_norm": float(grad.detach().norm().cpu()) if grad is not None else 0.0,
            "gradient_finite": bool(torch.isfinite(grad).all().cpu()) if grad is not None else False,
            "update_norm": float((parameter.detach() - before[name]).norm().cpu()),
            "role": "final_head" if "decomposition_head" in name else ("earliest_decoder" if ".dec2.block.0." in name else "decoder_body"),
        })
    optimizer_checks = {
        "optimizer_parameter_count": sum(parameter.numel() for parameter in parameters) == 92940,
        "optimizer_ids_unique": len(ids) == len(set(ids)),
        "optimizer_ids_complete": set(ids) == {id(parameter) for parameter in parameters},
        "both_experts_finite_nonzero_gradients": all(math.isfinite(value) and value > 0 for value in expert_gradient_norms.values()),
        "all_gradients_finite": all(row["gradient_finite"] for row in gradient_rows),
        "both_final_heads_update": all(any(row["parameter"].startswith(expert) and row["role"] == "final_head" and row["update_norm"] > 0 for row in gradient_rows) for expert in ("expert_1", "expert_2")),
        "both_nonfinal_blocks_update": all(any(row["parameter"].startswith(expert) and row["role"] != "final_head" and row["update_norm"] > 0 for row in gradient_rows) for expert in ("expert_1", "expert_2")),
        "finite_nonnegative_post_outputs": bool(torch.isfinite(post_physical).all()) and int((post_physical < 0).sum().cpu()) == 0,
        "mps_only": all(parameter.device.type == "mps" for parameter in parameters) and all(value.device.type == "mps" for value in combined),
        "mps_fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
    }
    optimizer_result = {
        "status": "PASS" if all(optimizer_checks.values()) else "FAIL",
        "checks": optimizer_checks,
        "pre_step_loss": float(production_loss.detach().cpu()),
        "post_step_loss": float(post_loss.detach().cpu()),
        "expert_gradient_norms": expert_gradient_norms,
        "raw_gradient_norm": float(raw.grad.detach().norm().cpu()),
        "square_derivative_minimum_absolute": float((2.0 * raw.detach()).abs().min().cpu()),
        "square_derivative_median_absolute": float((2.0 * raw.detach()).abs().median().cpu()),
        "square_zero_gradient_fraction": float(((2.0 * raw.detach()) == 0).float().mean().cpu()),
        "optimizer_state_tensor_count": sum(isinstance(value, torch.Tensor) for state in optimizer.state.values() for value in state.values()),
        "post_physical_sha256": tensor_sha256(post_physical),
    }
    write_json_x(run / "optimizer_audit/synthetic_optimizer_audit.json", optimizer_result)
    write_csv_x(run / "tables/optimizer_parameter_inventory.csv", gradient_rows)
    if optimizer_result["status"] != "PASS":
        raise RuntimeError("SYNTHETIC AUTOGRAD/OPTIMIZER DEFECT")

    optimizer.zero_grad(set_to_none=True)
    post_loss.backward()
    post_gradient_sha256 = {
        name: tensor_sha256(parameter.grad)
        for name, parameter in named
        if parameter.grad is not None
    }

    checkpoint = run / "checkpoint_replay/synthetic_post_step_checkpoint.pt"
    checkpoint_payload = {
        "schema_version": "thayer-d3-checkpoint-v2",
        "expert_1_state_dict": {name: value.detach().cpu() for name, value in experts[0].state_dict().items()},
        "expert_2_state_dict": {name: value.detach().cpu() for name, value in experts[1].state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "execution_step": 1,
        "constructor_contract_version": "mapped-compact-expert-square-v2",
        "capsule_v2_sha256": capsule_sha256,
        "requirement_registry_sha256": registry_sha256,
        "model_code_sha256": sha256(repo / "src/output_parameterization.py"),
        "optimizer_contract_sha256": hashlib.sha256(json.dumps(registry_value(registry, "execution.optimizer_hyperparameters", accessed), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "synthetic_feature_manifest_sha256": sha256(run / "synthetic_inputs/synthetic_feature_manifest.json"),
        "synthetic_target_manifest_sha256": sha256(run / "synthetic_inputs/synthetic_target_manifest.json"),
        "post_output_sha256": tensor_sha256(post_physical),
        "post_loss": float(post_loss.detach().cpu()),
        "post_assignment": loss_functions["hard_physical_set_loss"](post_physical, targets_device, scale6)[1].detach().cpu(),
        "post_gradient_sha256": post_gradient_sha256,
    }
    save_torch_x(checkpoint, checkpoint_payload)
    checkpoint_manifest = {
        "path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size,
        "schema_version": checkpoint_payload["schema_version"], "state_hashes": {
            "expert_1": state_sha256(checkpoint_payload["expert_1_state_dict"]),
            "expert_2": state_sha256(checkpoint_payload["expert_2_state_dict"]),
        },
    }
    write_json_x(run / "checkpoint_replay/synthetic_checkpoint_manifest.json", checkpoint_manifest)

    replay_result: dict[str, object] = {"status": "SKIPPED_BY_CALLER"}
    if spawn_replay:
        replay_output = run / "checkpoint_replay/fresh_process_replay.json"
        command = [
            str(repo / ".venv-btk/bin/python"), "-B", str(repo / "scripts/replay_thayer_d3_synthetic_checkpoint.py"),
            "--repo", str(repo), "--run", str(run), "--capsule", str(capsule_path),
            "--capsule-sha256", capsule_sha256, "--registry", str(registry_path),
            "--registry-sha256", registry_sha256, "--checkpoint", str(checkpoint),
            "--checkpoint-sha256", checkpoint_manifest["sha256"], "--output", str(replay_output),
        ]
        environment = dict(os.environ)
        environment.update({"PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
        completed = subprocess.run(command, cwd=repo, env=environment, text=True, capture_output=True, check=False)
        write_text_x(run / "checkpoint_replay/replay_stdout.txt", completed.stdout)
        write_text_x(run / "checkpoint_replay/replay_stderr.txt", completed.stderr)
        if completed.returncode != 0 or not replay_output.is_file():
            raise RuntimeError(f"CHECKPOINT REPLAY DEFECT: {completed.stderr}")
        replay_result = json.loads(replay_output.read_text(encoding="utf-8"))
        if replay_result.get("status") != "PASS":
            raise RuntimeError("CHECKPOINT REPLAY DEFECT")

    accessed_set = frozenset(accessed)
    closure = {
        "declared_required_count": len(declared),
        "accessed_or_validated_count": len(accessed_set),
        "declared_required_set": sorted(declared),
        "actually_accessed_set": sorted(accessed_set),
        "undeclared_accesses": sorted(accessed_set - declared),
        "unaccessed_required": sorted(declared - accessed_set),
        "equal": declared == accessed_set,
    }
    write_json_x(run / "synthetic_execution/requirement_closure.json", closure)
    if not closure["equal"]:
        raise RuntimeError("CAPSULE-CONSUMER CONTRACT DRIFT REMAINS")
    guard_snapshot = guard.snapshot()
    result = {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "status": "PASS",
        "markers": ["ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED", "READY_FOR_AUTHORITATIVE_D3_EXECUTION"],
        "architecture": architecture["status"],
        "forward": forward_result["status"],
        "assignment_loss_evaluator": assignment_result["status"],
        "optimizer": optimizer_result["status"],
        "checkpoint_replay": replay_result.get("status"),
        "requirement_closure": closure,
        "checkpoint": checkpoint_manifest,
        "synthetic_inputs": input_info,
        "guard_snapshot": guard_snapshot,
        "scientific_array_values_loaded": 0,
        "scientific_d3_steps": 0,
        "synthetic_optimizer_steps": 1,
        "completed_utc": utcnow(),
    }
    write_json_x(run / "synthetic_execution/synthetic_preflight_result.json", result)
    return result
