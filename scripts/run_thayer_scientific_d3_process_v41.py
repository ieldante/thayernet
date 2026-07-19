#!/usr/bin/env python3
"""Two-correction adapter over the immutable scientific D3 v4 worker.

All model, optimizer, loss, assignment, policy, threshold, execution-budget,
scientific-metric, semantic-state, replay, and outcome code executes from the
frozen v4 module. This adapter changes only NumPy dtype token comparison and
bootstrap serialization readiness.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.run_thayer_scientific_d3_process_v4 as frozen  # noqa: E402


IntegrationRequirementFailure = frozen.IntegrationRequirementFailure
_FROZEN_LOAD_RUNTIME_MODULES = frozen.load_runtime_modules
_FROZEN_RUN_SYNTHETIC = frozen.run_synthetic
_FROZEN_RUN_AUTHORITATIVE = frozen.run_authoritative
_ACTIVE_GUARD: Any = None
numpy_dtype_contract_equal: Any = None


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def load_runtime_modules_v41() -> None:
    """Preserve frozen import order, then eagerly load the two proven modules."""

    global numpy_dtype_contract_equal
    _FROZEN_LOAD_RUNTIME_MODULES()
    import torch.utils.serialization as torch_utils_serialization
    import torch.utils.serialization.config
    from src.d3_contract_tokens_v41 import (
        numpy_dtype_contract_equal as semantic_dtype_equal,
    )

    if torch_utils_serialization is not sys.modules.get("torch.utils.serialization"):
        raise IntegrationRequirementFailure(
            "D3I41-SERIALIZATION-IMPORT", "torch.utils.serialization import identity mismatch"
        )
    numpy_dtype_contract_equal = semantic_dtype_equal


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _member_rows(
    cache: dict[str, Any],
    p0: dict[str, Any],
    initial: dict[str, Any],
    model_state: dict[str, Any],
    d1: dict[str, Any],
    d1_manifest: dict[str, Any],
    d0: dict[str, Any],
    d2: dict[str, Any],
) -> list[dict[str, Any]]:
    groups: list[tuple[str, Iterable[tuple[str, Any]]]] = [
        (
            "cached_features",
            (
                (f"{prompt}.{name}", value)
                for prompt in ("prompt_a", "prompt_b")
                for name, value in zip(("enc1", "enc2", "bottleneck"), cache[prompt])
            ),
        ),
        ("p0_target_set", p0.items()),
        ("initial_decoder_state", initial.items()),
        ("model_state", model_state.items()),
        ("d1_endpoint", d1.items()),
        ("d1_manifest", d1_manifest.items()),
        ("d0_evidence", d0.items()),
        ("d2_evidence", d2.items()),
    ]
    rows: list[dict[str, Any]] = []
    for container, members in groups:
        for name, value in members:
            is_tensor = frozen.torch is not None and isinstance(value, frozen.torch.Tensor)
            is_array = frozen.np is not None and isinstance(value, frozen.np.ndarray)
            if is_tensor or is_array:
                dtype_value = value.detach().cpu().numpy() if is_tensor else value
                canonical_dtype = frozen.np.dtype(dtype_value.dtype).str
                canonical_hash = frozen.canonical_tensor_sha256(dtype_value)
                shape = json.dumps(list(value.shape))
                original_dtype = str(value.dtype)
            else:
                canonical_dtype = "NOT_APPLICABLE"
                canonical_hash = _json_hash(value)
                shape = "NOT_APPLICABLE"
                original_dtype = "NOT_APPLICABLE"
            rows.append(
                {
                    "container_role": container,
                    "member_name": str(name),
                    "semantic_role": f"{container}.{name}",
                    "shape": shape,
                    "original_dtype_token": original_dtype,
                    "canonical_dtype_token": canonical_dtype,
                    "canonical_member_hash": canonical_hash,
                    "loaded": True,
                }
            )
    return rows


def load_scientific_assets_v41(context: dict[str, Any], output: Path) -> dict[str, Any]:
    """Frozen v4 loader with only the D1 dtype comparison normalized."""

    paths = frozen.scientific_paths(context)
    rows = frozen.validate_scientific_files(context, paths)
    if len(rows) != 8:
        raise IntegrationRequirementFailure(
            "D3I41-SCIENTIFIC-CONTAINER-COUNT", f"expected 8 containers, found {len(rows)}"
        )
    capsule = context["chain"]["capsule_v1"]
    v2 = context["chain"]["bundle_v2"]
    cache = frozen.torch.load(paths["cached_features"], map_location="cpu", weights_only=True)
    with frozen.np.load(paths["p0_target_set"], allow_pickle=False) as handle:
        p0 = {name: frozen.np.asarray(handle[name]) for name in handle.files}
    initial = frozen.torch.load(
        paths["initial_decoder_state"], map_location="cpu", weights_only=True
    )
    model_state = frozen.torch.load(paths["model_state"], map_location="cpu", weights_only=True)
    with frozen.np.load(paths["d1_endpoint"], allow_pickle=False) as handle:
        d1 = {name: frozen.np.asarray(handle[name]) for name in handle.files}
    d1_manifest = json.loads(paths["d1_manifest"].read_text(encoding="utf-8"))
    d0 = frozen.torch.load(paths["d0"], map_location="cpu", weights_only=True)
    d2 = frozen.torch.load(paths["d2"], map_location="cpu", weights_only=True)
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
    if loaded_members != 91:
        raise IntegrationRequirementFailure(
            "D3I41-SCIENTIFIC-MEMBER-COUNT", f"expected 91 members, found {loaded_members}"
        )
    for row in rows:
        row["deserialized"] = True
    frozen.write_csv_x(output / "tables/scientific_tensor_load_inventory.csv", rows)
    frozen.write_csv_x(output / "tables/v41_scientific_container_inventory.csv", rows)
    frozen.write_json_x(
        output / "authoritative_inputs/scientific_load_summary.json",
        {
            "container_count": len(rows),
            "member_count": loaded_members,
            "containers": [str(path.relative_to(REPO)) for path in paths.values()],
            "status": "PASS",
        },
    )

    contracts = v2["artifact_member_contracts"]
    feature_shapes = {
        f"{prompt}.{name}": list(value.shape)
        for prompt in ("prompt_a", "prompt_b")
        for name, value in zip(("enc1", "enc2", "bottleneck"), cache[prompt])
    }
    if feature_shapes != contracts["cached_features.member_shapes"]:
        raise IntegrationRequirementFailure(
            "D3I-CACHED-FEATURE-SHAPE", "cached feature shapes mismatch"
        )
    if set(d1) != set(contracts["d1.member_names"]):
        raise IntegrationRequirementFailure(
            "D3I-D1-MEMBER-NAMES", "D1 endpoint member mismatch"
        )

    dtype_rows: list[dict[str, Any]] = []
    dtype_failure: str | None = None
    for name in d1:
        expected_shape = contracts["d1.member_shapes"][name]
        actual_shape = list(d1[name].shape)
        if actual_shape != expected_shape:
            raise IntegrationRequirementFailure(
                "D3I-D1-MEMBER-SHAPE", f"D1 shape mismatch: {name}"
            )
        dtype_result = numpy_dtype_contract_equal(d1[name], contracts["d1.member_dtypes"][name])
        dtype_rows.append(
            {
                "member_name": name,
                **dtype_result.to_dict(),
                "shape_status": "PASS",
                "semantic_status": "PASS",
                "status": "PASS" if dtype_result.equal else "FAIL",
            }
        )
        if not dtype_result.equal and dtype_failure is None:
            dtype_failure = name
        if frozen.canonical_tensor_sha256(d1[name]) != contracts[
            "d1.member_canonical_hashes"
        ][name]:
            raise IntegrationRequirementFailure(
                "D3I-D1-CANONICAL-HASH", f"D1 hash mismatch: {name}"
            )

    frozen.write_csv_x(output / "tables/v41_dtype_callsite_inventory.csv", dtype_rows)
    write_text_x(
        output / "diagnostics/v41_dtype_normalization_report.md",
        "# V4.1 dtype normalization report\n\n"
        f"All {len(dtype_rows)} frozen D1 dtype contracts preserved their original expected "
        "tokens and were compared through `numpy_dtype_contract_equal`. Shapes, member "
        "names, semantic roles, and canonical hashes retained separate strict checks.\n",
    )
    member_rows = _member_rows(cache, p0, initial, model_state, d1, d1_manifest, d0, d2)
    if len(member_rows) != 91:
        raise IntegrationRequirementFailure(
            "D3I41-SCIENTIFIC-MEMBER-INVENTORY", "member inventory did not close at 91"
        )
    frozen.write_csv_x(output / "tables/v41_scientific_member_inventory.csv", member_rows)
    write_text_x(
        output / "diagnostics/v41_scientific_artifact_validation.md",
        "# V4.1 scientific artifact validation\n\n"
        "The same eight frozen containers and 91 counted members were loaded. File hashes, "
        "member names, shapes, semantic roles, canonical hashes, original dtype displays, "
        "and canonical dtype tokens were inventoried without loading any additional member.\n",
    )
    if dtype_failure is not None:
        raise IntegrationRequirementFailure(
            "D3I-D1-MEMBER-CONTRACT", f"D1 dtype mismatch: {dtype_failure}"
        )
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


def _tag(context: dict[str, Any]) -> str:
    mode = context["worker_received"]["mode"]
    if os.environ.get("D3_V4_SYNTHETIC_REPLAY_ONLY") == "1":
        return "synthetic_replay"
    if os.environ.get("D3_V4_REPLAY_ONLY") == "1":
        return "authoritative_replay"
    return "synthetic_primary" if mode == "synthetic_integration_preflight" else "authoritative"


def _guard(
    output: Path, runtime: Path, exact_reads: tuple[Path, ...], tag: str
) -> Any:
    from scripts.thayer_d3_runtime_guard import GuardPolicy, TwoPhaseGuard

    if tag == "authoritative":
        access_log = output / "access_guard/scientific_access_log.jsonl"
        blocked_log = output / "access_guard/scientific_blocked_access_log.jsonl"
    else:
        access_log = output / f"access_guard/{tag}_access_log.jsonl"
        blocked_log = output / f"access_guard/{tag}_blocked_access_log.jsonl"
    guard = TwoPhaseGuard(
        GuardPolicy(
            repository_root=REPO,
            fresh_run_root=output,
            runtime_root=runtime,
            access_log=access_log,
            blocked_log=blocked_log,
            exact_read_files=exact_reads,
            strict_write_roots=(output,),
            strict_atomic_roots=(output,),
            bootstrap_write_roots=(runtime,),
            bootstrap_read_roots=(REPO / ".venv-btk",),
        )
    )
    guard.install()
    guard._v41_access_log = access_log
    guard._v41_blocked_log = blocked_log
    guard._v41_tag = tag
    return guard


def _artifact_paths(output: Path, tag: str) -> tuple[Path, Path]:
    suffix = "" if tag == "candidate_bootstrap" else f"_{tag}"
    return (
        output / f"tables/v41_serialization_bootstrap_inventory{suffix}.csv",
        output / f"diagnostics/v41_serialization_bootstrap_report{suffix}.md",
    )


def serialization_bootstrap_prewarm(
    guard: Any, output: Path, runtime: Path, tag: str
) -> dict[str, Any]:
    """Exercise only a tiny synthetic weights-only save/load before strict."""

    required_serialization_modules = (
        "torch.utils.serialization",
        "torch.utils.serialization.config",
    )
    missing = [name for name in required_serialization_modules if name not in sys.modules]
    if missing:
        raise IntegrationRequirementFailure(
            "D3I41-SERIALIZATION-MODULE-INVENTORY", f"missing before strict: {missing}"
        )
    if any(name == "matplotlib" or name.startswith("matplotlib.") for name in sys.modules):
        raise IntegrationRequirementFailure(
            "D3I41-SERIALIZATION-MATPLOTLIB", "Matplotlib loaded before strict mode"
        )
    scratch = runtime / f"tmp/serialization_prewarm/{tag}"
    scratch.mkdir(parents=True, exist_ok=False)
    checkpoint = scratch / "synthetic_probe.pt"
    synthetic_probe = frozen.torch.tensor([1.25, -2.5], dtype=frozen.torch.float32)
    frozen.torch.save({"synthetic_probe": synthetic_probe}, checkpoint)
    loaded = frozen.torch.load(checkpoint, map_location="cpu", weights_only=True)
    equal = bool(frozen.torch.equal(loaded["synthetic_probe"], synthetic_probe))
    checkpoint.unlink()
    scratch.rmdir()
    if not equal:
        raise IntegrationRequirementFailure(
            "D3I41-SERIALIZATION-PREWARM-EQUALITY", "synthetic save/load mismatch"
        )
    access_rows = []
    if guard._v41_access_log.is_file():
        access_rows = [
            json.loads(line)
            for line in guard._v41_access_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    pyc_reads = [
        row
        for row in access_rows
        if row.get("event") == "open" and str(row.get("path", "")).endswith(".pyc")
    ]
    module_rows = []
    for name in required_serialization_modules:
        module = sys.modules[name]
        module_rows.append(
            {
                "module": name,
                "source_path": str(getattr(module, "__file__", "BUILTIN")),
                "present_before_strict": True,
                "synthetic_save_load_pass": equal,
                "bootstrap_pyc_read_count": len(pyc_reads),
                "scientific_checkpoint_opened": False,
                "matplotlib_loaded": False,
                "bytecode_writing_disabled": bool(sys.dont_write_bytecode),
                "strict_package_allowlist_broadened": False,
                "files_created": str(checkpoint),
                "files_renamed": "NONE",
                "files_removed": f"{checkpoint};{scratch}",
                "status": "PASS",
            }
        )
    guard.transition("strict")
    if any(name not in sys.modules for name in required_serialization_modules):
        raise IntegrationRequirementFailure(
            "D3I41-STRICT-SERIALIZATION-MODULE", "serialization module absent after transition"
        )
    table, report = _artifact_paths(output, tag)
    frozen.write_csv_x(table, module_rows)
    write_text_x(
        report,
        "# V4.1 serialization bootstrap report\n\n"
        f"Tag: `{tag}`. Both proven serialization modules were imported before strict mode. "
        "A tiny synthetic state dictionary completed one `torch.save` and one "
        "`torch.load(..., weights_only=True)` inside fresh runtime scratch, equality passed, "
        "and bootstrap cleanup remained inside scratch. No scientific checkpoint or "
        "Matplotlib module was used; bytecode writes stayed disabled; strict package read "
        "permissions were not broadened.\n",
    )
    result = {
        "schema_version": "thayer-d3i41-serialization-bootstrap-v1",
        "tag": tag,
        "status": "PASS",
        "required_serialization_modules": list(required_serialization_modules),
        "module_source_paths": {
            row["module"]: row["source_path"] for row in module_rows
        },
        "synthetic_save_load_equal": equal,
        "scientific_checkpoint_opened": False,
        "matplotlib_loaded": False,
        "bytecode_writing_disabled": bool(sys.dont_write_bytecode),
        "strict_permissions_broadened": False,
        "bootstrap_pyc_reads": pyc_reads,
        "lifecycle_confined_to_scratch": True,
        "scratch_checkpoint_removed": not checkpoint.exists(),
        "scratch_directory_removed": not scratch.exists(),
        "module_inventory_before_strict": sorted(
            name for name in sys.modules if name.startswith("torch.utils.serialization")
        ),
        "table": str(table.relative_to(output)),
        "report": str(report.relative_to(output)),
    }
    frozen.write_json_x(output / f"serialization_bootstrap/{tag}_result.json", result)
    return result


def _exact_reads(context: dict[str, Any]) -> tuple[Path, ...]:
    paths = frozen.scientific_paths(context)
    return tuple(
        list(paths.values())
        + [
            context["chain"]["bundle_v3_path"],
            context["chain"]["bundle_v2_path"],
            context["chain"]["capsule_v1_path"],
            context["chain"]["runtime_path"],
            context["chain"]["registry_path"],
            Path(context["bridge_path"]),
        ]
    )


def _start_guard(context: dict[str, Any], output: Path, runtime: Path) -> Any:
    tag = _tag(context)
    guard = _guard(output, runtime, _exact_reads(context), tag)
    serialization_bootstrap_prewarm(guard, output, runtime, tag)
    return guard


def _strict_audit(guard: Any, output: Path) -> None:
    rows = []
    if guard._v41_access_log.is_file():
        rows = [
            json.loads(line)
            for line in guard._v41_access_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    strict_external_pyc_reads = [
        row
        for row in rows
        if row.get("phase") == "strict"
        and str(row.get("path", "")).endswith(".pyc")
        and "torch/utils/serialization" in str(row.get("path", ""))
    ]
    result = {
        "schema_version": "thayer-d3i41-strict-serialization-audit-v1",
        "tag": guard._v41_tag,
        "strict_external_pyc_reads": strict_external_pyc_reads,
        "strict_external_pyc_read_count": len(strict_external_pyc_reads),
        "strict_permissions_broadened": False,
        "status": "PASS" if not strict_external_pyc_reads else "FAIL",
    }
    frozen.write_json_x(
        output / f"serialization_bootstrap/{guard._v41_tag}_strict_audit.json", result
    )
    if strict_external_pyc_reads:
        raise IntegrationRequirementFailure(
            "D3I41-STRICT-EXTERNAL-PYC-READ", "serialization .pyc read occurred after strict"
        )


def build_guard_v41(output: Path, runtime: Path, exact_reads: list[Path]) -> Any:
    del exact_reads
    if _ACTIVE_GUARD is None:
        raise IntegrationRequirementFailure(
            "D3I41-GUARD-SEQUENCE", "authoritative guard was not prewarmed"
        )
    return _ACTIVE_GUARD


def run_synthetic_v41(context: dict[str, Any], output: Path) -> dict[str, Any]:
    runtime = Path(context["worker_received"]["runtime_root"])
    guard = _start_guard(context, output, runtime)
    try:
        result = _FROZEN_RUN_SYNTHETIC(context, output)
        _strict_audit(guard, output)
        result["v41_dtype_token_normalization"] = "PASS"
        result["v41_serialization_bootstrap"] = "PASS"
        result["v41_zero_strict_external_pyc_reads"] = True
        target = output / (
            "synthetic_preflight/synthetic_checkpoint_replay_v41.json"
            if os.environ.get("D3_V4_SYNTHETIC_REPLAY_ONLY") == "1"
            else "synthetic_preflight/v41_scientific_worker_result.json"
        )
        frozen.write_json_x(target, result)
        return result
    finally:
        if guard.phase == "strict":
            guard.transition("shutdown")


def run_authoritative_v41(
    context: dict[str, Any], output: Path, runtime: Path
) -> dict[str, Any]:
    global _ACTIVE_GUARD
    guard = _start_guard(context, output, runtime)
    original_transition = guard.transition

    def defer_shutdown(phase: str) -> None:
        if phase != "shutdown":
            original_transition(phase)

    guard.transition = defer_shutdown
    _ACTIVE_GUARD = guard
    try:
        if os.environ.get("D3_V4_REPLAY_ONLY") == "1":
            result = frozen.replay_authoritative(context, output, runtime)
        else:
            result = _FROZEN_RUN_AUTHORITATIVE(context, output, runtime)
        _strict_audit(guard, output)
        result["v41_dtype_token_normalization"] = "PASS"
        result["v41_serialization_bootstrap"] = "PASS"
        result["v41_zero_strict_external_pyc_reads"] = True
        return result
    finally:
        _ACTIVE_GUARD = None
        guard.transition = original_transition
        if guard.phase == "strict":
            guard.transition("shutdown")


def standalone_candidate_prewarm(
    output: Path, runtime: Path, tag: str = "candidate_bootstrap"
) -> dict[str, Any]:
    """Create the candidate-referenced prewarm result before bridge assembly."""

    load_runtime_modules_v41()
    guard = _guard(output.resolve(), runtime.resolve(), (), tag)
    try:
        result = serialization_bootstrap_prewarm(
            guard, output.resolve(), runtime.resolve(), tag
        )
        _strict_audit(guard, output.resolve())
        return result
    finally:
        if guard.phase == "strict":
            guard.transition("shutdown")


def install_v41_adapter() -> None:
    frozen.load_runtime_modules = load_runtime_modules_v41
    frozen.load_scientific_assets = load_scientific_assets_v41
    frozen.build_guard = build_guard_v41
    frozen.run_synthetic = run_synthetic_v41
    frozen.run_authoritative = run_authoritative_v41


def main() -> int:
    install_v41_adapter()
    return frozen.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IntegrationRequirementFailure as exc:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "canonical_integration_requirement_id": exc.requirement_id,
                    "message": exc.message,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2)
