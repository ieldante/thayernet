"""Create and preregister a fresh Thayer-D3B readiness run.

This entry point intentionally uses only the Python standard library.  It must
finish before any campaign process imports NumPy, PyTorch, Matplotlib, or a
project module.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
D3R = REPO / "outputs/runs/thayer_full_l0_d3r_20260713_121652"
D1R = REPO / "outputs/runs/thayer_d1_endpoint_replay_20260713_113715"
RI = REPO / "outputs/runs/thayer_repository_integrity_20260713_031653"
OP = REPO / "outputs/runs/thayer_output_parameterization_20260713_023120"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.rstrip("\n")


def metadata(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path.relative_to(REPO)),
        "absolute_path": str(path),
        "bytes": info.st_size,
        "mode": stat.filemode(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "sha256": sha256(path),
    }


def authoritative_files() -> list[Path]:
    return [
        D3R / "reports/final_report.md",
        D3R / "diagnostics/final_correctness_audit.json",
        D3R / "diagnostics/final_access_log_manifest.json",
        D3R / "diagnostics/preload_frozen_input_audit.json",
        D3R / "preregistration/authoritative_square_full_l0_d3.md",
        D3R / "access_guard/access_log.jsonl",
        D3R / "access_guard/blocked_access_log.jsonl",
        D3R / "access_guard/path_guard.py",
        D3R / "access_guard/test_path_guard.py",
        D3R / "logs/fail_closed_stop.json",
        D3R / "authoritative_inputs/run_authoritative_d3.py",
        D3R / "tables/checkpoint_inventory_before.csv",
        D3R / "tables/checkpoint_inventory_after.csv",
        D1R / "reports/final_report.md",
        D1R / "replay_verification/d1_endpoint_manifest.json",
        D1R / "diagnostics/final_correctness_audit_superseding_v2.json",
        D1R / "diagnostics/final_access_log_manifest.json",
        D1R / "tables/downstream_d3_prerequisite_check.csv",
        D1R / "tables/d1_endpoint_inventory.csv",
        D1R / "optimized_features/d1_penultimate_endpoints.npz",
        D1R / "physical_outputs/d1_physical_outputs.npz",
        D1R / "frozen_heads/d1_frozen_heads.npz",
        D1R / "authoritative_inputs/p0_targets.npz",
        RI / "reports/final_report.md",
        RI / "diagnostics/final_correctness_audit_superseding_v2.json",
        RI / "access_guard/path_guard.py",
        RI / "access_guard/test_path_guard.py",
        RI / "independent_oracles/reference_implementation.py",
        RI / "independent_oracles/run_independent_oracles.py",
        RI / "code_inventory/local_import_graph_v3.json",
        RI / "code_inventory/code_hashes_v3.csv",
        RI / "code_inventory/execution_module_inventory_v3.csv",
        RI / "tables/production_reference_comparison_final_recheck.csv",
        RI / "fixed_feature_retry/cached_features_superseding_v4.pt",
        RI / "data_lineage/one_scene_payload.npz",
        RI / "fixed_feature_retry/initial_state_square_superseding_v3.pt",
        RI / "fixed_feature_retry/d0_superseding_v2/square_final.pt",
        RI / "fixed_feature_retry/d1_superseding_v2/square_final.pt",
        RI / "fixed_feature_retry/d2_superseding_v2/square_final.pt",
        OP / "checkpoints/ambiguous_one_scene_square.pth",
        REPO / "src/canonical_tensor_hash.py",
        REPO / "src/competing_hypotheses.py",
        REPO / "src/models_two_expert_decoder.py",
        REPO / "src/output_parameterization.py",
        REPO / "scripts/thayer_d3_runtime_guard.py",
        REPO / "scripts/run_thayer_d3_readiness.py",
        REPO / "scripts/postprocess_thayer_d3_readiness.py",
        REPO / "docs/d1_endpoint_persistence.md",
        REPO / "docs/feature_endpoint_artifact_contract.md",
        REPO / "docs/fixed_feature_decoder_audit.md",
        REPO / "docs/full_l0_fixed_feature_d3.md",
        REPO / "docs/cached_encoder_feature_contract.md",
        REPO / "docs/repository_integrity_audit.md",
        REPO / "docs/allowlisted_file_access_contract.md",
        REPO / "docs/authoritative_full_l0_d3.md",
    ]


def runtime_state(run: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in (
        "runtime/tmp",
        "runtime/cache",
        "runtime/config",
        "runtime/matplotlib",
        "runtime/torch",
        "runtime/pycache",
        "runtime/postprocess_tmp",
    ):
        path = run / relative
        info = path.stat()
        rows.append(
            {
                "path": relative,
                "mode": stat.filemode(info.st_mode),
                "uid": info.st_uid,
                "gid": info.st_gid,
                "initial_entry_count": sum(1 for _ in path.iterdir()),
                "initially_empty": not any(path.iterdir()),
            }
        )
    return rows


def incident_rows() -> list[dict[str, Any]]:
    base = "outputs/runs/thayer_full_l0_d3r_20260713_121652/access_guard"
    return [
        {
            "attempted_operation": "os.remove",
            "exact_path": f"{base}/matplotlib_d3r/fontlist-v390.json.matplotlib-lock",
            "path_classification": "fresh D3R run / Matplotlib disposable cache lock",
            "package_module": "matplotlib font manager",
            "call_stack": "UNRESOLVED_NOT_PERSISTED",
            "historical": False,
            "protected": False,
            "repository_source": False,
            "new_run_output": True,
            "system_cache": False,
            "temporary_runtime_state": True,
            "could_change_scientific_data": False,
            "escaped_d3r_fresh_run": False,
            "succeeded": False,
            "evidence": "access_guard/access_log.jsonl:12031; guard logs path eligibility then raises unconditionally",
        },
        {
            "attempted_operation": "os.remove",
            "exact_path": f"{base}/tmp/mat1pmql",
            "path_classification": "fresh D3R run / PyTorch tempfile probe",
            "package_module": "torch.distributed.nn.jit.instantiator via tempfile usable-directory probe",
            "call_stack": "UNRESOLVED_NOT_PERSISTED",
            "historical": False,
            "protected": False,
            "repository_source": False,
            "new_run_output": True,
            "system_cache": False,
            "temporary_runtime_state": True,
            "could_change_scientific_data": False,
            "escaped_d3r_fresh_run": False,
            "succeeded": False,
            "evidence": "access_guard/access_log.jsonl:13507 and logs/fail_closed_stop.json",
        },
    ]


def make_preregistration(run: Path, files: list[dict[str, Any]], environment: dict[str, str]) -> str:
    input_lines = "\n".join(
        f"- `{item['path']}` — `{item['bytes']}` bytes — SHA-256 `{item['sha256']}`"
        for item in files
    )
    env_lines = "\n".join(f"- `{key}={value}`" for key, value in environment.items())
    process_lines = []
    for case, root_name in (
        ("cold_1", "cold_1"),
        ("cold_2", "cold_2"),
        ("warm_1", "cold_1"),
        ("after_shutdown", "after_shutdown"),
        ("postprocess_isolated", "postprocess_isolated"),
    ):
        root = run / "runtime" / root_name
        process_lines.append(
            f"- `{case}`: `TMPDIR={root / 'tmp'}`, `MPLCONFIGDIR={root / 'matplotlib'}`, "
            f"`XDG_CACHE_HOME={root / 'cache'}`, `XDG_CONFIG_HOME={root / 'config'}`, "
            f"`TORCH_HOME={root / 'torch'}`, `PYTHONPYCACHEPREFIX={root / 'pycache'}`."
        )
    process_environment_lines = "\n".join(process_lines)
    return f"""# Thayer-D3B Runtime Bootstrap Readiness Preregistration

## Identity and freeze

- Campaign: `Thayer-D3B` (`Thayer D3 Bootstrap`).
- Fresh run: `{run.name}`.
- Purpose: metadata-only and synthetic-runtime readiness; this is not D3.
- Third-party imports before this document was written: `0` in the campaign process.
- Scientific tensor deserializations: prohibited.
- Model, decoder, encoder, optimizer, scientific forward, JVP, and VJP construction: prohibited.

## Exact authoritative metadata inputs

{input_lines}

The NPZ/PT/PTH files above may be opened only for byte hashing and stat metadata.
No array member or tensor payload may be deserialized.

## Frozen bootstrap environment

{env_lines}

Every directory is inside the fresh run.  No cache may be redirected into
source, data, outputs outside this run, or a historical run.

The exact process roots are frozen as follows; `TMP` and `TEMP` equal each
listed `TMPDIR`, while backend, seed, and fallback settings equal the values
above:

{process_environment_lines}

## Two-phase guard

Bootstrap phase permits exact package/environment reads and create, same-tree
rename, and deletion only under the designated runtime scratch.  It blocks
historical writes, protected-path access, recursive repository traversal, and
all lifecycle operations outside scratch.

Strict scientific phase permits exact allowlisted reads and preregistered
fresh-run report/test/log writes.  It permits only prospectively designated
same-directory atomic artifact renames.  It blocks every deletion, cache write,
historical write, nonallowlisted read, recursive traversal, Matplotlib import,
plotting, and new bootstrap activity.  Any strict deletion attempt is terminal.

A shutdown phase may delete only under runtime scratch after the readiness
status and access log are flushed.  Any shutdown deletion elsewhere is terminal.

## Frozen import order and synthetic operations

1. NumPy.
2. PyTorch.
3. `torch.nn.functional`, `torch._dynamo`, and
   `torch.distributed.nn.jit.instantiator`.
4. Matplotlib with Agg, bootstrap only.
5. Strict phase exact modules: `src.competing_hypotheses`,
   `src.models_two_expert_decoder`, and `src.output_parameterization`.

Bootstrap triggers are a tiny synthetic MPS tensor, functional convolution,
scalar backward, synchronization, and a synthetic Agg render.  Strict phase
repeats a tiny functional PyTorch forward/backward.  No optimizer or neural
module instance may be constructed.

## Pure evaluator tests

The production function is `src.competing_hypotheses.forward_consistency` with
the observed blend, candidate source layers, and sky electrons passed directly.
An independently written reference must agree on: exact two-source sum,
one-pixel sources, g-only versus z-only, source-order swap, prompt swap, zero
source, known positive residual, wrong band order, noncontiguous input, and
batch-size-one versus batch-size-N evaluation.  Calls must be deterministic and
produce zero filesystem events.

## Readiness sequence and marker

The guarded launcher must run environment setup, bootstrap guard, third-party
bootstrap, bootstrap inventory freeze, strict transition, exact project imports,
import-separation checks, pure evaluator tests, metadata-only prerequisites, and
final log flush.  It may emit exactly `READY_FOR_SCIENTIFIC_TENSOR_LOAD` and exit;
it may not enter D3.

## Frozen gates

Readiness requires the exact Part P gates: preregistration order, incident
classification, all guard tests, scratch-confined bootstrap lifecycle, zero
strict deletes/cache writes/nonallowlisted reads/protected access, no strict
Matplotlib import, isolated postprocessing, cold/warm/shutdown passes, pure
evaluator agreement with zero I/O, all 21 metadata prerequisites, exact marker,
zero scientific deserialization, zero model/optimizer construction, and
unchanged historical checkpoints.  Gates will not be weakened after execution.

## Terminal rules

Any unmet requirement is `READINESS FAIL`.  On failure, identify exactly one
remaining metadata/runtime/evaluator blocker and recommend one metadata-only
correction.  On pass, freeze runtime and launcher/guard/evaluator hashes and
authorize only one separately preregistered D3 campaign.  Do not run D3 here.
"""


def main() -> None:
    started = utcnow()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / "outputs/runs" / f"thayer_d3_runtime_readiness_{stamp}"
    run.mkdir(parents=False, exist_ok=False)
    directories = [
        "access_guard",
        "runtime/tmp",
        "runtime/cache",
        "runtime/config",
        "runtime/matplotlib",
        "runtime/torch",
        "runtime/pycache",
        "runtime/postprocess_tmp",
        "diagnostics",
        "tables",
        "figures",
        "logs",
        "reports",
        "preregistration",
        "import_tests",
        "evaluator_tests",
        "launcher_tests",
        "metadata_checks",
    ]
    for relative in directories:
        (run / relative).mkdir(parents=True, exist_ok=False)

    paths = authoritative_files()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing authoritative files: {missing}")
    files = [metadata(path) for path in paths]
    usage = shutil.disk_usage(REPO)
    environment = {
        "TMPDIR": "<PROCESS_RUNTIME>/tmp",
        "TMP": "<PROCESS_RUNTIME>/tmp",
        "TEMP": "<PROCESS_RUNTIME>/tmp",
        "MPLCONFIGDIR": "<PROCESS_RUNTIME>/matplotlib",
        "XDG_CACHE_HOME": "<PROCESS_RUNTIME>/cache",
        "XDG_CONFIG_HOME": "<PROCESS_RUNTIME>/config",
        "TORCH_HOME": "<PROCESS_RUNTIME>/torch",
        "PYTHONPYCACHEPREFIX": "<PROCESS_RUNTIME>/pycache",
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "20260713",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    snapshot = {
        "campaign": "Thayer-D3B",
        "campaign_start_utc": started,
        "snapshot_utc": utcnow(),
        "branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short").splitlines(),
        "staged_index": git("diff", "--cached", "--name-only").splitlines(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "exact_environment_path": str(REPO / ".venv-btk"),
        "filesystem_free_bytes": usage.free,
        "filesystem_total_bytes": usage.total,
        "authoritative_artifacts": files,
        "runtime_directories": runtime_state(run),
        "third_party_imports": 0,
        "scientific_array_deserializations": 0,
    }
    snapshot_lines = [
        "# Standard-Library-Only Environment Snapshot",
        "",
        f"- Campaign start: `{started}`",
        f"- Branch / HEAD: `{snapshot['branch']}` / `{snapshot['git_head']}`",
        f"- Python: `{sys.executable}`",
        f"- Expected environment: `{REPO / '.venv-btk'}`",
        f"- Free disk bytes: `{usage.free}`",
        f"- Staged paths: `{len(snapshot['staged_index'])}`",
        f"- Third-party imports: `0`",
        f"- Scientific array deserializations: `0`",
        "",
        "Exact artifact metadata and runtime-directory ownership are frozen in",
        "`logs/input_provenance.json`.",
        "",
    ]
    write_text_x(run / "diagnostics/environment_snapshot_stdlib_only.md", "\n".join(snapshot_lines))
    write_json_x(run / "logs/input_provenance.json", snapshot)
    contract = """# Thayer-D3B Campaign Contract

This fresh run is metadata-only and synthetic-runtime-only. It does not run D3.
No scene, target, cached feature, endpoint, checkpoint tensor, or model state may
be deserialized. No encoder, decoder, model, optimizer, scientific model
forward, JVP, or VJP may be constructed or executed. Historical paths are
immutable. Atlas, development, lockbox, ordinary, eight-scene, and full-
microset inputs are prohibited. Writes are fresh and collision-refusing.

Bootstrap lifecycle operations are confined to `runtime/`. Strict phase
deletion, cache writes, Matplotlib imports, plotting, broad search, and
nonallowlisted reads are terminal. Passing readiness does not automatically
continue into D3.
"""
    write_text_x(run / "diagnostics/campaign_contract.md", contract)

    preregistration = make_preregistration(run, files, environment)
    prereg_path = run / "preregistration/d3_runtime_bootstrap_readiness.md"
    write_text_x(prereg_path, preregistration)
    prereg_hash = sha256(prereg_path)
    write_json_x(
        run / "preregistration/preregistration_freeze.json",
        {
            "path": str(prereg_path.relative_to(REPO)),
            "sha256": prereg_hash,
            "frozen_utc": utcnow(),
            "third_party_imports_before_freeze": 0,
            "scientific_tensor_deserializations_before_freeze": 0,
        },
    )

    rows = incident_rows()
    write_csv_x(run / "tables/d3r_bootstrap_incident.csv", rows)
    incident_report = f"""# D3R Bootstrap Incident Report

The authoritative D3R runner encountered two `os.remove` attempts. Both exact
paths were inside the D3R fresh run: a Matplotlib font-cache lock and a PyTorch
temporary-directory probe file. The D3R guard logged each path as eligible for
fresh-run writes and then unconditionally raised on removal, so neither deletion
succeeded and both files were preserved.

Neither target was historical, protected, repository source, or scientific
data. Neither attempt escaped the D3R fresh run, and neither could change a
scientific artifact. The persisted access and blocked logs contain no call-stack
field, so both exact call stacks are `UNRESOLVED_NOT_PERSISTED`; this campaign
therefore assumes both lifecycle operations may recur and captures independent
stacks under a stricter disposable sandbox.

- Incident rows: `{len(rows)}`
- Preregistration SHA-256: `{prereg_hash}`
"""
    write_text_x(run / "diagnostics/d3r_bootstrap_incident_report.md", incident_report)
    write_json_x(
        run / "runtime/bootstrap_environment.json",
        {
            "environment": environment,
            "process_environments": {
                case: {
                    **{
                        "TMPDIR": str(run / "runtime" / root_name / "tmp"),
                        "TMP": str(run / "runtime" / root_name / "tmp"),
                        "TEMP": str(run / "runtime" / root_name / "tmp"),
                        "MPLCONFIGDIR": str(run / "runtime" / root_name / "matplotlib"),
                        "XDG_CACHE_HOME": str(run / "runtime" / root_name / "cache"),
                        "XDG_CONFIG_HOME": str(run / "runtime" / root_name / "config"),
                        "TORCH_HOME": str(run / "runtime" / root_name / "torch"),
                        "PYTHONPYCACHEPREFIX": str(run / "runtime" / root_name / "pycache"),
                    },
                    "MPLBACKEND": "Agg",
                    "PYTHONHASHSEED": "20260713",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                }
                for case, root_name in (
                    ("cold_1", "cold_1"),
                    ("cold_2", "cold_2"),
                    ("warm_1", "cold_1"),
                    ("after_shutdown", "after_shutdown"),
                    ("postprocess_isolated", "postprocess_isolated"),
                )
            },
            "directories": snapshot["runtime_directories"],
            "preregistration_sha256": prereg_hash,
            "recorded_utc": utcnow(),
        },
    )
    print(run)


if __name__ == "__main__":
    main()
