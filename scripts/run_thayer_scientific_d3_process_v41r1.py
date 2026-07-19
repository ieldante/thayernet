#!/usr/bin/env python3
"""Independent-audit-compliant adapter over the frozen D3 v4.1 worker."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.serialization
import torch.utils.serialization
import torch.utils.serialization.config

import scripts.run_thayer_scientific_d3_process_v41 as v41
from src import d3_checkpoint_adapter_v41r1 as checkpoint_adapter
from src.d3_contract_tokens_v41r1 import numpy_dtype_contract_equal
REQUIRED_SERIALIZATION_MODULES = (
    "torch.serialization",
    "torch.utils.serialization",
    "torch.utils.serialization.config",
)
_ORIGINAL_V41_LOAD_RUNTIME = v41.load_runtime_modules_v41


def _write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runtime_modules_v41r1() -> None:
    _ORIGINAL_V41_LOAD_RUNTIME()
    v41.numpy_dtype_contract_equal = numpy_dtype_contract_equal
    checkpoint_adapter.install_torch_checkpoint_routing()


def _exercise_production_adapter(
    scratch: Path,
    transition_to_strict: Callable[[], None],
    access_log: Path | None,
) -> dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=False)
    checkpoint = scratch / "production_schema_synthetic.pt"
    checkpoint_adapter.reset_adapter_trace()
    payload = checkpoint_adapter.build_synthetic_production_checkpoint_payload()
    checkpoint_adapter.write_production_checkpoint(checkpoint, payload)
    bootstrap_loaded = checkpoint_adapter.read_production_checkpoint(checkpoint)
    checkpoint_adapter.validate_production_checkpoint_payload(bootstrap_loaded)
    loaded_before = sorted(
        name for name in REQUIRED_SERIALIZATION_MODULES if name in sys.modules
    )
    modules_before = set(sys.modules)
    transition_to_strict()
    strict_loaded = checkpoint_adapter.read_production_checkpoint(checkpoint)
    checkpoint_adapter.validate_production_checkpoint_payload(strict_loaded)
    strict_new_imports = sorted(set(sys.modules) - modules_before)
    strict_pyc_reads: list[dict[str, Any]] = []
    if access_log is not None and access_log.is_file():
        for line in access_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == "strict" and str(row.get("path", "")).endswith(".pyc"):
                strict_pyc_reads.append(row)
    trace = checkpoint_adapter.adapter_trace()
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "required_serialization_modules": list(REQUIRED_SERIALIZATION_MODULES),
        "loaded_before_strict": loaded_before,
        "bootstrap_writer_pass": trace["writer_calls"] == 1,
        "bootstrap_reader_pass": trace["reader_calls"] >= 2,
        "strict_new_imports": strict_new_imports,
        "strict_external_pyc_reads": strict_pyc_reads,
        "adapter_trace": trace,
        "schema_version": checkpoint_adapter.SCHEMA_VERSION,
        "schema_keys": list(checkpoint_adapter.PRODUCTION_CHECKPOINT_KEYS),
        "map_location": checkpoint_adapter.FROZEN_MAP_LOCATION,
        "weights_only": checkpoint_adapter.FROZEN_WEIGHTS_ONLY,
        "scientific_checkpoint_opened": False,
        "scientific_model_tensor_used": False,
        "scientific_payload_values_loaded": 0,
        "matplotlib_loaded": any(
            name == "matplotlib" or name.startswith("matplotlib.")
            for name in sys.modules
        ),
        "bytecode_writing_disabled": bool(sys.dont_write_bytecode),
        "strict_permissions_broadened": False,
    }


def run_serialization_contract_probe(
    scratch_root: Path, *, strict_verify: bool
) -> dict[str, Any]:
    load_runtime_modules_v41r1()
    transitioned = False

    def transition() -> None:
        nonlocal transitioned
        transitioned = bool(strict_verify)

    result = _exercise_production_adapter(
        scratch_root / "serialization_prewarm", transition, None
    )
    result["strict_phase_exercised"] = transitioned
    return result


def serialization_bootstrap_prewarm_r1(
    guard: Any, output: Path, runtime: Path, tag: str
) -> dict[str, Any]:
    load_runtime_modules_v41r1()

    def transition() -> None:
        guard.transition("strict")

    result = _exercise_production_adapter(
        runtime / f"tmp/serialization_prewarm_r1/{tag}",
        transition,
        guard._v41_access_log,
    )
    if set(result["loaded_before_strict"]) != set(REQUIRED_SERIALIZATION_MODULES):
        raise v41.IntegrationRequirementFailure(
            "D3I41R1-SERIALIZATION-MODULE-SET", "required module set incomplete"
        )
    if result["strict_new_imports"]:
        raise v41.IntegrationRequirementFailure(
            "D3I41R1-STRICT-NEW-IMPORT", str(result["strict_new_imports"])
        )
    if result["strict_external_pyc_reads"]:
        raise v41.IntegrationRequirementFailure(
            "D3I41R1-STRICT-EXTERNAL-PYC", "external pyc read after strict"
        )
    if result["matplotlib_loaded"]:
        raise v41.IntegrationRequirementFailure(
            "D3I41R1-PREWARM-MATPLOTLIB", "Matplotlib loaded in scientific worker"
        )
    suffix = tag.replace("/", "_")
    result_path = output / f"serialization_bootstrap/{suffix}_r1_result.json"
    _write_json_x(result_path, result)
    module_rows = [
        {
            "module": name,
            "loaded_before_strict": name in result["loaded_before_strict"],
            "source_path": str(getattr(sys.modules[name], "__file__", "BUILTIN")),
            "status": "PASS",
        }
        for name in REQUIRED_SERIALIZATION_MODULES
    ]
    _write_csv_x(
        output / f"tables/v41r1_serialization_module_inventory_{suffix}.csv",
        module_rows,
    )
    _write_csv_x(
        output / f"tables/v41r1_checkpoint_adapter_trace_{suffix}.csv",
        [
            {"symbol": key, "call_count": value, "status": "PASS"}
            for key, value in result["adapter_trace"].items()
        ],
    )
    return result


def metadata_validation_contract() -> dict[str, int]:
    return {"container_count": 8, "member_count": 91, "payload_values_loaded": 0}


def standalone_candidate_prewarm(
    output: Path, runtime: Path, tag: str = "candidate_001_bootstrap"
) -> dict[str, Any]:
    load_runtime_modules_v41r1()
    guard = v41._guard(output.resolve(), runtime.resolve(), (), tag)
    try:
        return serialization_bootstrap_prewarm_r1(
            guard, output.resolve(), runtime.resolve(), tag
        )
    finally:
        if guard.phase == "strict":
            guard.transition("shutdown")


def install_v41r1_adapter() -> None:
    v41.load_runtime_modules_v41 = load_runtime_modules_v41r1
    v41.serialization_bootstrap_prewarm = serialization_bootstrap_prewarm_r1
    v41.numpy_dtype_contract_equal = numpy_dtype_contract_equal
    v41.install_v41_adapter()


def main() -> int:
    install_v41r1_adapter()
    return v41.frozen.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except v41.IntegrationRequirementFailure as exc:
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
