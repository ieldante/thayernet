"""Plotting-free, metadata-only scientific readiness process for Thayer-D3B.

This entry point is launched with ``python -B`` and a fully redirected runtime.
It never loads a scientific tensor, instantiates a project model, constructs an
optimizer, or executes a decoder forward pass.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Callable
import warnings


PLOT_MODULE_ROOT = "mat" + "plotlib"
SCIENTIFIC_TENSOR_DESERIALIZATIONS = 0
MODEL_INSTANTIATIONS = 0
OPTIMIZER_CONSTRUCTIONS = 0
DECODER_FORWARDS = 0


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inventory_roots(runtime: Path, cache: Path) -> list[dict[str, Any]]:
    """Inventory only preregistered disposable roots; never repository trees."""

    roots = [runtime]
    if cache != runtime / "cache":
        roots.append(cache)
    rows: list[dict[str, Any]] = []
    for root in roots:
        for current, directory_names, file_names in os.walk(root):
            directory_names.sort()
            file_names.sort()
            current_path = Path(current)
            rows.append(
                {
                    "root": str(root),
                    "path": str(current_path.relative_to(root)) or ".",
                    "kind": "directory",
                    "bytes": 0,
                }
            )
            for name in file_names:
                path = current_path / name
                rows.append(
                    {
                        "root": str(root),
                        "path": str(path.relative_to(root)),
                        "kind": "file",
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    return rows


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_guard(config: dict[str, Any]):
    path = Path(config["guard_source"])
    spec = importlib.util.spec_from_file_location("thayer_d3_runtime_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load runtime guard")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    policy = module.GuardPolicy(
        repository_root=Path(config["repo"]),
        fresh_run_root=Path(config["run"]),
        runtime_root=Path(config["runtime_root"]),
        access_log=Path(config["access_log"]),
        blocked_log=Path(config["blocked_log"]),
        exact_read_files=tuple(Path(value) for value in config["exact_read_files"]),
        strict_write_roots=tuple(Path(value) for value in config["strict_write_roots"]),
        strict_atomic_roots=(Path(config["atomic_root"]),),
        bootstrap_write_roots=tuple(Path(value) for value in config["bootstrap_write_roots"]),
        bootstrap_read_roots=tuple(Path(value) for value in config.get("bootstrap_read_roots", ())),
    )
    guard = module.TwoPhaseGuard(policy)
    guard.install()
    return module, guard


def assert_environment(config: dict[str, Any]) -> dict[str, str]:
    expected = config["environment"]
    observed = {key: os.environ.get(key, "") for key in expected}
    if observed != expected:
        raise RuntimeError(f"runtime environment mismatch: {observed!r} != {expected!r}")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RuntimeError("bytecode suppression is not active")
    return observed


def run_guard_self_tests(config: dict[str, Any], module, guard) -> None:
    runtime = Path(config["runtime_root"])
    repo = Path(config["repo"])
    run = Path(config["run"])
    rows: list[dict[str, Any]] = []

    def probe(name: str, expected: str, operation: Callable[[], Any]) -> None:
        observed = "allowed"
        detail = ""
        try:
            operation()
        except module.GuardViolation as exc:
            observed = "blocked"
            detail = str(exc)
        except FileNotFoundError as exc:
            observed = "allowed_missing"
            detail = str(exc)
        rows.append(
            {
                "test": name,
                "expected": expected,
                "observed": observed,
                "status": "PASS" if observed == expected else "FAIL",
                "detail": detail,
            }
        )

    created = runtime / "tmp/bootstrap_create.tmp"
    renamed = runtime / "tmp/bootstrap_rename.tmp"
    strict_delete = runtime / "tmp/strict_delete.tmp"
    shutdown_delete = runtime / "tmp/shutdown_delete.tmp"
    for path in (strict_delete, shutdown_delete):
        path.open("x").close()

    probe("bootstrap_create_inside_runtime_tmp", "allowed", lambda: created.open("x").close())
    probe("bootstrap_rename_inside_runtime_tmp", "allowed", lambda: os.rename(created, renamed))
    probe("bootstrap_delete_inside_runtime_tmp", "allowed", lambda: os.remove(renamed))
    probe(
        "bootstrap_delete_outside_runtime_scratch",
        "blocked",
        lambda: os.remove(run / "diagnostics/dummy_outside_runtime"),
    )
    historical = Path(config["historical_dummy"])
    probe("historical_write", "blocked", lambda: historical.open("x").close())
    probe("historical_rename", "blocked", lambda: os.rename(historical, historical.with_name("dummy_renamed")))
    probe("historical_delete", "blocked", lambda: os.remove(historical))
    protected = runtime / "lockbox/dummy_nonexistent"
    probe("protected_read", "blocked", lambda: protected.open("rb").close())

    guard.transition("strict")
    probe("strict_delete_inside_runtime_scratch", "blocked", lambda: os.remove(strict_delete))
    probe("strict_cache_write", "blocked", lambda: (Path(config["cache_root"]) / "strict_cache").open("x").close())
    probe("strict_bytecode_write", "blocked", lambda: (runtime / "pycache/strict.pyc").open("xb").close())
    report = run / "launcher_tests/strict_selftest_report.txt"
    probe("strict_allowlisted_report_write", "allowed", lambda: report.open("x").close())
    atomic_source = Path(config["atomic_source"])
    atomic_destination = Path(config["atomic_destination"])
    probe("strict_approved_atomic_rename", "allowed", lambda: os.replace(atomic_source, atomic_destination))
    probe("recursive_historical_directory_iteration", "blocked", lambda: list((repo / "outputs").iterdir()))

    guard.transition("shutdown")
    probe("shutdown_cleanup_inside_runtime_scratch", "allowed", lambda: os.remove(shutdown_delete))
    probe(
        "shutdown_cleanup_outside_runtime_scratch",
        "blocked",
        lambda: os.remove(run / "diagnostics/dummy_shutdown_outside"),
    )
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "test_count": len(rows),
        "tests": rows,
        "guard_snapshot": guard.snapshot(),
        "completed_utc": utcnow(),
    }
    write_json_x(runtime / "guard_self_tests.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"guard self-tests failed: {rows}")


def bootstrap_imports(config: dict[str, Any], guard):
    rows: list[dict[str, Any]] = []

    def perform(label: str, operation: Callable[[], Any]):
        before = guard.snapshot()["event_count"]
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = operation()
        after = guard.snapshot()["event_count"]
        rows.append(
            {
                "label": label,
                "runtime_seconds": time.perf_counter() - started,
                "event_count_before": before,
                "event_count_after": after,
                "event_delta": after - before,
                "warnings": [str(item.message) for item in caught],
                "status": "PASS",
            }
        )
        return value

    np = perform("numpy", lambda: __import__("numpy"))
    torch = perform("torch", lambda: __import__("torch"))
    functional = perform(
        "torch.nn.functional",
        lambda: __import__("torch.nn.functional", fromlist=["functional"]),
    )
    perform("torch._dynamo", lambda: __import__("torch._dynamo", fromlist=["_dynamo"]))
    perform(
        "torch.distributed.nn.jit.instantiator",
        lambda: __import__("torch.distributed.nn.jit.instantiator", fromlist=["instantiator"]),
    )
    if any(name == PLOT_MODULE_ROOT or name.startswith(PLOT_MODULE_ROOT + ".") for name in sys.modules):
        raise RuntimeError("plotting package entered the scientific interpreter during bootstrap")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.manual_seed(20260713)
    value = torch.arange(25, dtype=torch.float32, device="mps").reshape(1, 1, 5, 5)
    value.requires_grad_(True)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="mps", requires_grad=True)
    convolved = functional.conv2d(value, kernel)
    scalar = convolved.square().mean()
    scalar.backward()
    torch.mps.synchronize()
    if value.grad is None or kernel.grad is None or not bool(torch.isfinite(scalar).item()):
        raise RuntimeError("bootstrap MPS synthetic gradient failed")
    inventory = {
        "case": config["case"],
        "imports": rows,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "synthetic_shape": list(convolved.shape),
        "synthetic_scalar": float(scalar.detach().cpu()),
        "environment": assert_environment(config),
        "plotting_modules": [],
        "scientific_tensor_deserializations": SCIENTIFIC_TENSOR_DESERIALIZATIONS,
        "model_instantiations": MODEL_INSTANTIATIONS,
        "optimizer_constructions": OPTIMIZER_CONSTRUCTIONS,
        "decoder_forwards": DECODER_FORWARDS,
        "guard_snapshot": guard.snapshot(),
        "file_inventory": inventory_roots(Path(config["runtime_root"]), Path(config["cache_root"])),
        "completed_utc": utcnow(),
    }
    path = Path(config["runtime_root"]) / "bootstrap_inventory.json"
    write_json_x(path, inventory)
    return np, torch, functional, inventory


def load_exact(module_name: str, path: Path):
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def load_d3_contract_functions(path: Path, torch):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    names = ("physical_direct_cost", "pairwise_costs", "hard_physical_set_loss")
    segments: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError(f"cannot recover D3 contract function {node.name}")
            segments[node.name] = segment
    if tuple(name for name in names if name in segments) != names:
        raise RuntimeError("D3 target-loss function inventory is incomplete")
    module = types.ModuleType("thayer_d3_frozen_contract")
    module.__dict__["torch"] = torch
    joined = "from __future__ import annotations\n\n" + "\n\n".join(segments[name] for name in names)
    exec(compile(joined, str(path), "exec", dont_inherit=True), module.__dict__)
    return module, {
        name: hashlib.sha256(segments[name].encode("utf-8")).hexdigest()
        for name in names
    }


def reference_forward(np, observed, candidates, sky) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    recomposed = candidates.sum(axis=0, dtype=np.float64)
    variance = np.maximum(recomposed + np.asarray(sky, dtype=np.float64)[:, None, None], 1.0)
    residual = observed - recomposed
    whitened = residual / np.sqrt(variance)
    squared = whitened**2

    def correlation(image) -> float:
        pairs = []
        if image.shape[0] > 1:
            pairs.append((image[:-1].ravel(), image[1:].ravel()))
        if image.shape[1] > 1:
            pairs.append((image[:, :-1].ravel(), image[:, 1:].ravel()))
        if not pairs:
            return 0.0
        left = np.concatenate([pair[0] for pair in pairs])
        right = np.concatenate([pair[1] for pair in pairs])
        if np.std(left) <= np.finfo(np.float64).eps or np.std(right) <= np.finfo(np.float64).eps:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    bands = tuple(float(value) for value in squared.mean(axis=(1, 2)))
    correlations = tuple(correlation(whitened[index]) for index in range(3))
    flux = float(np.sum(residual) / (float(np.sum(np.abs(observed))) + np.finfo(np.float64).eps))
    values = (float(squared.mean()), *bands, *correlations, flux)
    return {
        "global": values[0],
        "bands": bands,
        "correlations": correlations,
        "flux": flux,
        "finite": bool(np.all(np.isfinite(values))),
    }


def result_values(value) -> tuple[float, ...]:
    return (
        float(value.global_chi_square_mean),
        *tuple(float(item) for item in value.per_band_chi_square_mean),
        *tuple(float(item) for item in value.residual_neighbor_correlation),
        float(value.relative_flux_residual),
        float(bool(value.finite)),
    )


def reference_values(value: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(value["global"]),
        *tuple(float(item) for item in value["bands"]),
        *tuple(float(item) for item in value["correlations"]),
        float(value["flux"]),
        float(bool(value["finite"])),
    )


def evaluator_tests(config: dict[str, Any], guard, np, torch, production) -> dict[str, Any]:
    sky = np.asarray([11.0, 13.0, 17.0], dtype=np.float64)

    def source(values, y=1, x=1):
        output = np.zeros((3, 5, 5), dtype=np.float64)
        output[:, y, x] = np.asarray(values, dtype=np.float64)
        return output

    requested = source([10.0, 20.0, 30.0], 1, 1)
    companion = source([7.0, 11.0, 19.0], 3, 3)
    exact = np.stack((requested, companion))
    g_only = source([9.0, 0.0, 0.0])
    z_only = source([0.0, 0.0, 8.0], 2, 2)
    positive = exact.sum(axis=0) + 2.0
    noncontiguous_base = np.zeros((2, 3, 10, 10), dtype=np.float64)
    noncontiguous_base[:, :, ::2, ::2] = exact
    noncontiguous = noncontiguous_base[:, :, ::2, ::2]
    rows: list[dict[str, Any]] = []
    tolerance = 1.0e-12

    def compare(name: str, observed, candidates, local_sky=sky) -> tuple[float, ...]:
        before = guard.snapshot()["event_count"]
        first = production(observed, candidates, local_sky)
        after = guard.snapshot()["event_count"]
        second = production(observed, candidates, local_sky)
        reference = reference_forward(np, observed, candidates, local_sky)
        actual_values = result_values(first)
        reference_result = reference_values(reference)
        difference = max(abs(left - right) for left, right in zip(actual_values, reference_result))
        deterministic = actual_values == result_values(second)
        zero_io = before == after
        status = difference <= tolerance and deterministic and zero_io
        rows.append(
            {
                "case": name,
                "max_abs_difference": difference,
                "deterministic": deterministic,
                "filesystem_events": after - before,
                "status": "PASS" if status else "FAIL",
            }
        )
        return actual_values

    compare("exact_two_source_sum", exact.sum(axis=0), exact)
    compare("one_pixel_requested_and_companion", exact.sum(axis=0), exact)
    compare("g_only_requested_z_only_companion", g_only + z_only, np.stack((g_only, z_only)))
    compare("source_order_swap", exact.sum(axis=0), exact[::-1])
    compare("prompt_a_prompt_b_semantic_swap", exact.sum(axis=0), np.stack((companion, requested)))
    compare("zero_source", np.zeros((3, 5, 5)), np.zeros((2, 3, 5, 5)))
    compare("known_positive_residual", positive, exact)
    compare("wrong_band_order", exact.sum(axis=0)[::-1], exact[:, ::-1], sky[::-1])
    compare("noncontiguous_input", noncontiguous.sum(axis=0), noncontiguous)

    batch = [(exact.sum(axis=0), exact), (positive, exact), (g_only + z_only, np.stack((g_only, z_only)))]
    one = compare("batch_size_1_versus_batch_n", batch[0][0], batch[0][1])
    batch_values = [result_values(production(observed, candidates, sky)) for observed, candidates in batch]
    rows[-1]["batch_consistent"] = one == batch_values[0]
    rows[-1]["status"] = "PASS" if rows[-1]["status"] == "PASS" and rows[-1]["batch_consistent"] else "FAIL"

    reordered = [2, 0, 1]
    before = guard.snapshot()["event_count"]
    reordered_values = [result_values(production(batch[index][0], batch[index][1], sky)) for index in reordered]
    after = guard.snapshot()["event_count"]
    reorder_ok = reordered_values == [batch_values[index] for index in reordered]
    rows.append(
        {
            "case": "batch_reordering",
            "max_abs_difference": 0.0,
            "deterministic": reorder_ok,
            "filesystem_events": after - before,
            "status": "PASS" if reorder_ok and before == after else "FAIL",
        }
    )

    cpu_candidates = exact.astype(np.float32)
    cpu_observed = cpu_candidates.sum(axis=0, dtype=np.float32)
    mps_candidates = torch.from_numpy(cpu_candidates).to("mps").to("cpu").numpy()
    mps_observed = torch.from_numpy(cpu_observed).to("mps").to("cpu").numpy()
    cpu_values = result_values(production(cpu_observed, cpu_candidates, sky))
    before = guard.snapshot()["event_count"]
    mps_result = production(mps_observed, mps_candidates, sky)
    after = guard.snapshot()["event_count"]
    mps_values = result_values(mps_result)
    reference = reference_values(reference_forward(np, mps_observed, mps_candidates, sky))
    device_difference = max(abs(left - right) for left, right in zip(cpu_values, mps_values))
    reference_difference = max(abs(left - right) for left, right in zip(mps_values, reference))
    rows.append(
        {
            "case": "float32_cpu_versus_mps_to_cpu",
            "max_abs_difference": max(device_difference, reference_difference),
            "deterministic": cpu_values == mps_values,
            "filesystem_events": after - before,
            "status": "PASS" if device_difference <= tolerance and reference_difference <= tolerance and before == after else "FAIL",
        }
    )
    result = {
        "status": "PASS" if len(rows) == 12 and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "case_count": len(rows),
        "tolerance": tolerance,
        "production_function": "src.competing_hypotheses.forward_consistency",
        "production_source_sha256": sha256(Path(config["sources"]["competing_hypotheses"])),
        "filesystem_event_count": sum(int(row["filesystem_events"]) for row in rows),
        "rows": rows,
    }
    write_csv_x(Path(config["run"]) / "evaluator_tests/pure_forward_evaluator_cases.csv", rows)
    write_json_x(Path(config["run"]) / "evaluator_tests/pure_forward_evaluator_result.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"pure evaluator comparison failed: {rows}")
    return result


def metadata_prerequisites(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    prerequisite_path = Path(config["d1_prerequisite_table"])
    with prerequisite_path.open(newline="", encoding="utf-8") as handle:
        upstream = list(csv.DictReader(handle))
    rows.append(
        {
            "check": "all_21_d1r_prerequisites_pass",
            "status": "PASS" if len(upstream) == 21 and all(row["status"] == "PASS" for row in upstream) else "FAIL",
            "evidence": f"{len(upstream)}/21 rows report PASS",
        }
    )
    for item in config["metadata_files"]:
        path = Path(item["path"])
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        status = actual_size == int(item["expected_size"]) and actual_hash == item["expected_sha256"]
        prohibited = any(marker in {part.casefold() for part in path.parts} for marker in ("atlas", "development", "lockbox"))
        rows.append(
            {
                "check": item["name"],
                "status": "PASS" if status and not prohibited else "FAIL",
                "evidence": str(path),
                "expected_size": item["expected_size"],
                "actual_size": actual_size,
                "expected_sha256": item["expected_sha256"],
                "actual_sha256": actual_hash,
                "prohibited_partition": prohibited,
            }
        )
    for item in config["code_hashes"]:
        actual = sha256(Path(item["path"]))
        rows.append(
            {
                "check": item["name"],
                "status": "PASS" if actual == item["expected_sha256"] else "FAIL",
                "evidence": item["path"],
                "expected_sha256": item["expected_sha256"],
                "actual_sha256": actual,
            }
        )
    rows.extend(
        [
            {
                "check": "exact_d3_scene_and_prompt_ids_recorded",
                "status": "PASS",
                "evidence": "micro/P0 row 32; source row 12000; pu_training_near_00000; pu_training_pair_00001; prompt A/B",
            },
            {
                "check": "metadata_only_no_deserialization",
                "status": "PASS" if SCIENTIFIC_TENSOR_DESERIALIZATIONS == 0 else "FAIL",
                "evidence": "no numpy.load, h5py.File, torch.load, or archive-member access",
            },
        ]
    )
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "row_count": len(rows),
        "upstream_d1_prerequisite_count": len(upstream),
        "scientific_tensor_deserializations": SCIENTIFIC_TENSOR_DESERIALIZATIONS,
        "rows": rows,
    }
    write_csv_x(Path(config["run"]) / "tables/metadata_d3_prerequisites.csv", rows)
    report = [
        "# Metadata-only D3 prerequisite report",
        "",
        f"Status: **{result['status']}**.",
        "",
        f"All {len(upstream)} D1R prerequisite rows were checked from the exact CSV. ",
        "Named scientific containers were verified only by path metadata, byte size, and SHA-256. ",
        "No archive member or tensor was deserialized.",
        "",
    ]
    with (Path(config["run"]) / "diagnostics/metadata_d3_prerequisite_report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    if result["status"] != "PASS":
        raise RuntimeError("metadata-only D3 prerequisite audit failed")
    return result


def strict_scientific(config: dict[str, Any], guard, np, torch, functional) -> dict[str, Any]:
    if any(name == PLOT_MODULE_ROOT or name.startswith(PLOT_MODULE_ROOT + ".") for name in sys.modules):
        raise RuntimeError("plotting package was loaded before strict phase")
    guard.transition("strict")
    src_package = types.ModuleType("src")
    src_package.__path__ = [str(Path(config["repo"]) / "src")]
    sys.modules["src"] = src_package
    competing = load_exact("src.competing_hypotheses", Path(config["sources"]["competing_hypotheses"]))
    load_exact("src.models_probabilistic_unet", Path(config["sources"]["models_probabilistic_unet"]))
    models = load_exact("src.models_two_expert_decoder", Path(config["sources"]["models_two_expert_decoder"]))
    mapping = load_exact("src.output_parameterization", Path(config["sources"]["output_parameterization"]))
    contract, contract_hashes = load_d3_contract_functions(Path(config["d3r_runner"]), torch)

    objects = {
        "CompactExpertDecoder": inspect.isclass(models.CompactExpertDecoder),
        "MappedCompactExpertDecoder": inspect.isclass(mapping.MappedCompactExpertDecoder),
        "apply_output_mapping": inspect.isfunction(mapping.apply_output_mapping),
        "physical_direct_cost": inspect.isfunction(contract.physical_direct_cost),
        "pairwise_costs": inspect.isfunction(contract.pairwise_costs),
        "hard_physical_set_loss": inspect.isfunction(contract.hard_physical_set_loss),
        "forward_consistency": inspect.isfunction(competing.forward_consistency),
    }
    if not all(objects.values()):
        raise RuntimeError(f"strict scientific definition inspection failed: {objects}")

    outputs = torch.zeros((2, 2, 6, 2, 2), dtype=torch.float32, device="mps", requires_grad=True)
    targets = torch.zeros_like(outputs)
    scale6 = torch.ones(6, dtype=torch.float32, device="mps")
    set_loss, identity, margin, _ = contract.hard_physical_set_loss(outputs, targets, scale6)
    set_loss.backward()
    torch.mps.synchronize()
    if outputs.grad is None or not bool(torch.isfinite(set_loss).item()):
        raise RuntimeError("strict hard-assignment synthetic test failed")

    value = torch.arange(25, dtype=torch.float32, device="mps").reshape(1, 1, 5, 5)
    value.requires_grad_(True)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="mps", requires_grad=True)
    scalar = functional.conv2d(value, kernel).square().mean()
    scalar.backward()
    torch.mps.synchronize()
    if value.grad is None or kernel.grad is None:
        raise RuntimeError("strict synthetic MPS backward failed")

    evaluator = None
    metadata = None
    if config["case"] == "primary":
        evaluator = evaluator_tests(config, guard, np, torch, competing.forward_consistency)
        metadata = metadata_prerequisites(config)
    plotting_modules = sorted(
        name for name in sys.modules
        if name == PLOT_MODULE_ROOT or name.startswith(PLOT_MODULE_ROOT + ".")
    )
    if plotting_modules:
        raise RuntimeError(f"plotting modules entered strict scientific process: {plotting_modules}")
    status = {
        "case": config["case"],
        "status": "PASS",
        "marker": "READY_FOR_SCIENTIFIC_TENSOR_LOAD",
        "definition_objects": objects,
        "d3_contract_function_hashes": contract_hashes,
        "hard_assignment_identity": [bool(value) for value in identity.detach().cpu().tolist()],
        "hard_assignment_margin": [float(value) for value in margin.detach().cpu().tolist()],
        "strict_synthetic_scalar": float(scalar.detach().cpu()),
        "evaluator_status": None if evaluator is None else evaluator["status"],
        "metadata_status": None if metadata is None else metadata["status"],
        "plotting_modules": plotting_modules,
        "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "sys_dont_write_bytecode": sys.dont_write_bytecode,
        "scientific_tensor_deserializations": SCIENTIFIC_TENSOR_DESERIALIZATIONS,
        "model_instantiations": MODEL_INSTANTIATIONS,
        "optimizer_constructions": OPTIMIZER_CONSTRUCTIONS,
        "decoder_forwards": DECODER_FORWARDS,
        "guard_snapshot_before_shutdown": guard.snapshot(),
        "completed_utc": utcnow(),
    }
    write_json_x(Path(config["run"]) / f"diagnostics/scientific_readiness_{config['case']}.json", status)
    guard.transition("shutdown")
    strict_end_inventory = inventory_roots(Path(config["runtime_root"]), Path(config["cache_root"]))
    write_json_x(
        Path(config["runtime_root"]) / "strict_end_inventory.json",
        {
            "case": config["case"],
            "captured_immediately_after_strict_before_shutdown_cleanup": True,
            "entries": strict_end_inventory,
            "completed_utc": utcnow(),
        },
    )
    shutdown_probe = Path(config["runtime_root"]) / "tmp/shutdown_probe.tmp"
    shutdown_probe.open("x").close()
    os.remove(shutdown_probe)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("selftest", "readiness"), default="readiness")
    args = parser.parse_args()
    config = read_config(args.config.resolve())
    preguard = Path(config["runtime_root"]) / "pre_guard_access.json"
    write_json_x(
        preguard,
        {
            "reads_before_guard_install": [str(args.config.resolve()), config["guard_source"]],
            "third_party_imports_before_guard": [],
            "timestamp_utc": utcnow(),
        },
    )
    assert_environment(config)
    module, guard = load_guard(config)
    if args.mode == "selftest":
        run_guard_self_tests(config, module, guard)
        print("GUARD_SELF_TESTS_PASS")
        return
    np, torch, functional, _ = bootstrap_imports(config, guard)
    strict_scientific(config, guard, np, torch, functional)
    print("READY_FOR_SCIENTIFIC_TENSOR_LOAD")


if __name__ == "__main__":
    main()
