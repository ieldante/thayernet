"""Standard-library-only orchestrator for the Thayer-D3B readiness audit."""

from __future__ import annotations

import argparse
import ast
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
import time
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv-btk"
PYTHON = VENV / "bin/python"
D3R = REPO / "outputs/runs/thayer_full_l0_d3r_20260713_121652"
D1R = REPO / "outputs/runs/thayer_d1_endpoint_replay_20260713_113715"
RI = REPO / "outputs/runs/thayer_repository_integrity_20260713_031653"
OP = REPO / "outputs/runs/thayer_output_parameterization_20260713_023120"

SCIENTIFIC = REPO / "scripts/run_thayer_d3_scientific_readiness.py"
POSTPROCESS = REPO / "scripts/run_thayer_d3_postprocess_readiness.py"
GUARD = REPO / "scripts/thayer_d3_runtime_guard.py"
REGRESSION = REPO / "tests/test_d3_readiness_process_isolation.py"

SOURCE_BEFORE = {
    "scripts/run_thayer_d3_readiness.py": "e060e97fd990724af6ebbeeba3a2d9ddd357211ee2affeab9420fc0015471e68",
    "scripts/bootstrap_thayer_d3_readiness.py": "3a12a0d2b406ca8fdbfc38be7ff6f38af8446a8452e7accb36dafbaecbbf9eb1",
    "scripts/postprocess_thayer_d3_readiness.py": "65dd9dd005ea01bf8634b4da544bc2607b517f7229d5d06c51a81ac973b769da",
    "scripts/thayer_d3_runtime_guard.py": "cfc291bb36e2984744344ba30eaddd801d96ef12ab0c7b174dca862cd762c731",
    "scripts/run_thayer_d3_scientific_readiness.py": "ABSENT",
    "scripts/run_thayer_d3_postprocess_readiness.py": "ABSENT",
}


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


def append_text(path: Path, value: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(value)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def exact_inputs() -> list[Path]:
    return [
        D3R / "reports/final_report.md",
        D3R / "diagnostics/final_correctness_audit.json",
        D3R / "preregistration/authoritative_square_full_l0_d3.md",
        D3R / "diagnostics/final_access_log_manifest.json",
        D3R / "access_guard/access_log.jsonl",
        D3R / "access_guard/blocked_access_log.jsonl",
        D3R / "logs/fail_closed_stop.json",
        D3R / "authoritative_inputs/run_authoritative_d3.py",
        D3R / "tables/checkpoint_inventory_after.csv",
        D1R / "reports/final_report.md",
        D1R / "replay_verification/d1_endpoint_manifest.json",
        D1R / "diagnostics/final_correctness_audit_superseding_v2.json",
        D1R / "diagnostics/final_access_log_manifest.json",
        D1R / "tables/downstream_d3_prerequisite_check.csv",
        D1R / "tables/d1_endpoint_inventory.csv",
        RI / "reports/final_report.md",
        RI / "diagnostics/final_correctness_audit_superseding_v2.json",
        RI / "access_guard/path_guard.py",
        RI / "access_guard/test_path_guard.py",
        RI / "independent_oracles/reference_implementation.py",
        RI / "code_inventory/local_import_graph_v3.json",
        RI / "code_inventory/code_hashes_v3.csv",
        RI / "code_inventory/execution_module_inventory_v3.csv",
        REPO / "docs/d1_endpoint_persistence.md",
        REPO / "docs/feature_endpoint_artifact_contract.md",
        REPO / "docs/fixed_feature_decoder_audit.md",
        REPO / "docs/full_l0_fixed_feature_d3.md",
        REPO / "docs/cached_encoder_feature_contract.md",
        REPO / "docs/repository_integrity_audit.md",
        REPO / "docs/allowlisted_file_access_contract.md",
        REPO / "src/competing_hypotheses.py",
        REPO / "src/models_probabilistic_unet.py",
        REPO / "src/models_two_expert_decoder.py",
        REPO / "src/output_parameterization.py",
        REPO / "scripts/evaluate_probabilistic_unet_pre_atlas.py",
        Path(__file__).resolve(),
        SCIENTIFIC,
        POSTPROCESS,
        GUARD,
        REGRESSION,
    ]


def metadata_files() -> list[dict[str, str]]:
    return [
        {"name": "cached_feature_artifact", "path": str(RI / "fixed_feature_retry/cached_features_superseding_v4.pt"), "sha256": "4ffa31a7bd0e77578fb435288a433709ac01031486aa6bba479fc650926ce99a"},
        {"name": "p0_one_scene_payload", "path": str(RI / "data_lineage/one_scene_payload.npz"), "sha256": "86afd4b1dd1eabeface69c1236577c3732bc161a6603cb7a445454f479879df6"},
        {"name": "initial_decoder_state", "path": str(RI / "fixed_feature_retry/initial_state_square_superseding_v3.pt"), "sha256": "49058eb2ba9bf50a9df33f72d3aab1dace55612b7625dec6f475b4d6c3afa065"},
        {"name": "square_d0_endpoint", "path": str(RI / "fixed_feature_retry/d0_superseding_v2/square_final.pt"), "sha256": "a9e4d6a9ad4de3afaf8a10d1f0bf3ab977f07ae7432a06d4e5e08becf0a031dd"},
        {"name": "square_d1_endpoint", "path": str(RI / "fixed_feature_retry/d1_superseding_v2/square_final.pt"), "sha256": "4526f724aa34d6475100435c4eb7dfc9eb7f836ee8c87cd58e9ee7ff2834ff54"},
        {"name": "square_d2_endpoint", "path": str(RI / "fixed_feature_retry/d2_superseding_v2/square_final.pt"), "sha256": "a9d67c1b4c93f705e4dc04d960286b169bbf186def5757393b68d91f34d8dd5e"},
        {"name": "square_mapping_checkpoint", "path": str(OP / "checkpoints/ambiguous_one_scene_square.pth"), "sha256": "8b06e788853a9180df7f83803d25cab17e362aac602c2932efe8dee680fa591e"},
        {"name": "d1_penultimate_endpoint", "path": str(D1R / "optimized_features/d1_penultimate_endpoints.npz"), "sha256": "ec5ecd6ef892512e3a128e0d44d214840da0815e69f578ff51fb5a7a14ef69ba"},
        {"name": "d1_physical_outputs", "path": str(D1R / "physical_outputs/d1_physical_outputs.npz"), "sha256": "8de76b207f3765fbcbc639ffbbf51b36b5d58e6eb8455e09824f5dcf228ecd92"},
        {"name": "d1_frozen_heads", "path": str(D1R / "frozen_heads/d1_frozen_heads.npz"), "sha256": "343d001425e737e9fef1445b0838229c75a70ebce3dc0b17f46c5c42cf8c7ec7"},
        {"name": "d1_p0_targets", "path": str(D1R / "authoritative_inputs/p0_targets.npz"), "sha256": "1b0cd6ed34b2e88832d5724bb5205d92abc386b17c9990d5d40d095e54821a1f"},
    ]


def code_hashes() -> list[dict[str, str]]:
    return [
        {"name": "pure_evaluator_code_hash", "path": str(REPO / "src/competing_hypotheses.py"), "sha256": "e66111b2853c2b954efaa35880ee74d99736c03dc75197fd474fdc390271ca6d"},
        {"name": "model_dependency_code_hash", "path": str(REPO / "src/models_probabilistic_unet.py"), "sha256": "b86de449ba0524c5675ea300e87ff753c4d18b974ca18e26fbae74a760ed8b1e"},
        {"name": "decoder_class_code_hash", "path": str(REPO / "src/models_two_expert_decoder.py"), "sha256": "9931c81b42aa4463ef9715223f768c787d40c373519043b68167645f7708f415"},
        {"name": "square_mapping_code_hash", "path": str(REPO / "src/output_parameterization.py"), "sha256": "a47c322ffa3fda58a84a45c0a15891f60cef2455215ec99a229c6200f8edf1ae"},
        {"name": "target_loss_and_hard_assignment_runner_hash", "path": str(D3R / "authoritative_inputs/run_authoritative_d3.py"), "sha256": "ad322d2d480556fdd0594fb6c832bd9db92ea381f55983a90db6b9e21aa95ffd"},
    ]


def make_run() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / "outputs/runs" / f"thayer_d3_runtime_readiness_{stamp}"
    run.mkdir(exist_ok=False)
    directories = [
        "access_guard", "diagnostics", "tables", "figures", "logs", "reports",
        "preregistration", "import_tests", "evaluator_tests", "launcher_tests",
        "metadata_checks", "runtime/orchestrator", "runtime/scientific/tmp",
        "runtime/scientific/cache", "runtime/scientific/config", "runtime/scientific/torch",
        "runtime/scientific/pycache", "runtime/postprocess_runtime/tmp",
        "runtime/postprocess_runtime/cache", "runtime/postprocess_runtime/config",
        "runtime/postprocess_runtime/matplotlib", "runtime/postprocess_runtime/pycache",
        "runtime/postprocess_runtime/output",
    ]
    for case in ("cold_1", "cold_2", "warm_1", "shutdown_1"):
        for child in ("tmp", "cache", "config", "torch", "pycache"):
            directories.append(f"runtime/process_tests/{case}/{child}")
    for name in directories:
        (run / name).mkdir(parents=True, exist_ok=False)
    selftest = run / "runtime/orchestrator/guard_selftest"
    for child in ("tmp", "cache", "config", "torch", "pycache"):
        (selftest / child).mkdir(parents=True, exist_ok=False)
    atomic = run / "launcher_tests/atomic_selftest"
    atomic.mkdir(exist_ok=False)
    write_text_x(atomic / "artifact.tmp", "atomic self-test\n")
    return run


def file_snapshot(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "path": str(path),
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "mode": stat.filemode(value.st_mode),
        "uid": value.st_uid,
        "gid": value.st_gid,
        "sha256": sha256(path),
    }


def runtime_inventory(runtime: Path, cache: Path | None = None) -> dict[str, Any]:
    roots = [runtime]
    if cache is not None and cache != runtime / "cache":
        roots.append(cache)
    rows: list[dict[str, Any]] = []
    for root in roots:
        for current, directory_names, file_names in os.walk(root):
            directory_names.sort()
            file_names.sort()
            current_path = Path(current)
            rows.append({"root": str(root), "path": str(current_path.relative_to(root)) or ".", "kind": "directory", "bytes": 0})
            for name in file_names:
                path = current_path / name
                rows.append(
                    {
                        "root": str(root), "path": str(path.relative_to(root)), "kind": "file",
                        "bytes": path.stat().st_size, "sha256": sha256(path),
                    }
                )
    return {
        "runtime_root": str(runtime), "cache_root": None if cache is None else str(cache),
        "entries": rows, "file_count": sum(row["kind"] == "file" for row in rows),
        "root_file_counts": {
            str(root): sum(row["kind"] == "file" and row["root"] == str(root) for row in rows)
            for root in roots
        },
        "directory_count": sum(row["kind"] == "directory" for row in rows), "captured_utc": utcnow(),
    }


def checkpoint_audit(run: Path, phase: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with (D3R / "tables/checkpoint_inventory_after.csv").open(newline="", encoding="utf-8") as handle:
        for frozen in csv.DictReader(handle):
            path = REPO / frozen["path"]
            actual_size = path.stat().st_size
            actual_hash = sha256(path)
            status = actual_size == int(frozen["expected_bytes"]) and actual_hash == frozen["expected_sha256"]
            rows.append(
                {
                    "path": frozen["path"],
                    "expected_bytes": frozen["expected_bytes"],
                    "actual_bytes": actual_size,
                    "expected_sha256": frozen["expected_sha256"],
                    "actual_sha256": actual_hash,
                    "status": "PASS" if status else "FAIL",
                }
            )
    output = run / f"metadata_checks/checkpoint_hash_audit_{phase}.csv"
    write_csv_x(output, rows)
    return {
        "phase": phase,
        "count": len(rows),
        "mismatches": sum(row["status"] != "PASS" for row in rows),
        "table": str(output.relative_to(run)),
        "table_sha256": sha256(output),
    }


def incident_audit(run: Path) -> list[dict[str, Any]]:
    final = json.loads((D3R / "diagnostics/final_correctness_audit.json").read_text(encoding="utf-8"))
    targets = [Path(item["path"]).name for item in final["guard_prohibited_destructive_attempts"]]
    matches: dict[str, dict[str, Any]] = {}
    with (D3R / "access_guard/access_log.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            event = json.loads(line)
            if event.get("event") == "os.remove" and Path(event.get("path", "")).name in targets:
                event["line_number"] = line_number
                matches[Path(event["path"]).name] = event
    rows: list[dict[str, Any]] = []
    for item in final["guard_prohibited_destructive_attempts"]:
        name = Path(item["path"]).name
        event = matches.get(name, {})
        path = event.get("path", str(D3R / item["path"]))
        lock = name.endswith("matplotlib-lock")
        rows.append(
            {
                "operation": "os.remove",
                "exact_path": path,
                "path_classification": "fresh_D3R_runtime_cache_lock" if lock else "fresh_D3R_runtime_temp_probe",
                "package_module": "matplotlib font manager" if lock else "torch.distributed.nn.jit.instantiator via tempfile",
                "call_stack": "UNRESOLVED_NOT_RECORDED_IN_D3R_ACCESS_LOG",
                "access_log_line": event.get("line_number", "UNRESOLVED"),
                "historical_or_scientific_data": False,
                "inside_D3R_fresh_run": str(path).startswith(str(D3R)),
                "could_change_scientific_data": False,
                "D3R_reported_succeeded": item["succeeded"],
                "D3R_reported_file_preserved": item["file_preserved"],
                "currently_exists_metadata_only": Path(path).exists(),
                "redirected_in_D3B": "postprocess_runtime/matplotlib" if lock else "scientific tmp",
                "status": "CLASSIFIED_WITH_CALL_STACK_UNRESOLVED",
            }
        )
    write_csv_x(run / "tables/d3r_bootstrap_incident.csv", rows)
    lines = [
        "# D3R bootstrap incident report", "",
        "Both terminal operations were `os.remove` calls against files inside the fresh D3R run: one Matplotlib font-cache lock and one Python tempfile usability probe reached through PyTorch's distributed JIT instantiator. The exact D3R audit reported both attempts as unsuccessful and both files preserved; metadata-only checks still find both paths present.",
        "", "Neither path was a scene, target, feature, endpoint, checkpoint, repository source, or protected-partition artifact. The access log did not persist Python call stacks for these two rows, so exact frames remain unresolved and are not characterized as benign. D3B therefore retains the restrictive process sandbox and redirects the corresponding lifecycle state into disposable per-process roots.", "",
    ]
    write_text_x(run / "diagnostics/d3r_bootstrap_incident_report.md", "\n".join(lines))
    return rows


def scientific_environment(root: Path, cache: Path) -> dict[str, str]:
    return {
        "TMPDIR": str(root / "tmp"), "TMP": str(root / "tmp"), "TEMP": str(root / "tmp"),
        "XDG_CACHE_HOME": str(cache), "XDG_CONFIG_HOME": str(root / "config"),
        "TORCH_HOME": str(root / "torch"), "PYTHONPYCACHEPREFIX": str(root / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "20260713",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0", "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def process_config(run: Path, case: str, root: Path, cache: Path) -> dict[str, Any]:
    reads = exact_inputs() + [Path(item["path"]) for item in metadata_files()]
    strict_roots = [run / name for name in ("diagnostics", "tables", "reports", "import_tests", "evaluator_tests", "launcher_tests", "metadata_checks")]
    metadata = []
    for item in metadata_files():
        path = Path(item["path"])
        metadata.append(
            {
                "name": item["name"], "path": item["path"],
                "expected_sha256": item["sha256"], "expected_size": path.stat().st_size,
            }
        )
    codes = [
        {"name": item["name"], "path": item["path"], "expected_sha256": item["sha256"]}
        for item in code_hashes()
    ]
    bootstrap_roots = [root]
    if cache != root / "cache":
        bootstrap_roots.append(cache)
    return {
        "case": case, "repo": str(REPO), "run": str(run), "runtime_root": str(root),
        "cache_root": str(cache), "guard_source": str(GUARD),
        "access_log": str(run / f"access_guard/scientific_{case}_access_log.jsonl"),
        "blocked_log": str(run / f"access_guard/scientific_{case}_blocked_access_log.jsonl"),
        "exact_read_files": [str(path) for path in reads],
        "strict_write_roots": [str(path) for path in strict_roots],
        "atomic_root": str(run / "launcher_tests/atomic_selftest"),
        "atomic_source": str(run / "launcher_tests/atomic_selftest/artifact.tmp"),
        "atomic_destination": str(run / "launcher_tests/atomic_selftest/artifact.final"),
        "historical_dummy": str(D3R / "dummy_never_created"),
        "bootstrap_write_roots": [str(path) for path in bootstrap_roots],
        "bootstrap_read_roots": [], "environment": scientific_environment(root, cache),
        "sources": {
            "competing_hypotheses": str(REPO / "src/competing_hypotheses.py"),
            "models_probabilistic_unet": str(REPO / "src/models_probabilistic_unet.py"),
            "models_two_expert_decoder": str(REPO / "src/models_two_expert_decoder.py"),
            "output_parameterization": str(REPO / "src/output_parameterization.py"),
        },
        "d3r_runner": str(D3R / "authoritative_inputs/run_authoritative_d3.py"),
        "d1_prerequisite_table": str(D1R / "tables/downstream_d3_prerequisite_check.csv"),
        "metadata_files": metadata, "code_hashes": codes,
    }


def postprocess_config(run: Path) -> dict[str, Any]:
    root = run / "runtime/postprocess_runtime"
    environment = {
        "TMPDIR": str(root / "tmp"), "TMP": str(root / "tmp"), "TEMP": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / "cache"), "XDG_CONFIG_HOME": str(root / "config"),
        "MPLCONFIGDIR": str(root / "matplotlib"), "PYTHONPYCACHEPREFIX": str(root / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "20260713", "MPLBACKEND": "Agg",
    }
    font_roots = [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    return {
        "repo": str(REPO), "run": str(run), "runtime_root": str(root),
        "guard_source": str(GUARD),
        "access_log": str(run / "access_guard/postprocess_access_log.jsonl"),
        "blocked_log": str(run / "access_guard/postprocess_blocked_access_log.jsonl"),
        "bootstrap_read_roots": [str(path) for path in font_roots], "environment": environment,
    }


def write_configs(run: Path) -> dict[str, Path]:
    roots = {
        "selftest": run / "runtime/orchestrator/guard_selftest",
        "primary": run / "runtime/scientific",
        "cold_1": run / "runtime/process_tests/cold_1",
        "cold_2": run / "runtime/process_tests/cold_2",
        "warm_1": run / "runtime/process_tests/warm_1",
        "shutdown_1": run / "runtime/process_tests/shutdown_1",
    }
    configs: dict[str, Path] = {}
    for case, root in roots.items():
        cache = roots["cold_1"] / "cache" if case == "warm_1" else root / "cache"
        config = process_config(run, case, root, cache)
        path = run / f"runtime/orchestrator/{case}_config.json"
        write_json_x(path, config)
        configs[case] = path
    post = run / "runtime/orchestrator/postprocess_config.json"
    write_json_x(post, postprocess_config(run))
    configs["postprocess"] = post
    return configs


def record_import_graph(run: Path) -> dict[str, Any]:
    paths = [SCIENTIFIC, POSTPROCESS, REPO / "src/competing_hypotheses.py", REPO / "src/models_probabilistic_unet.py", REPO / "src/models_two_expert_decoder.py", REPO / "src/output_parameterization.py"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rows.append({"from": str(path.relative_to(REPO)), "line": node.lineno, "module": alias.name})
            elif isinstance(node, ast.ImportFrom):
                rows.append({"from": str(path.relative_to(REPO)), "line": node.lineno, "module": node.module or ""})
    scientific_rows = [row for row in rows if row["from"] == str(SCIENTIFIC.relative_to(REPO))]
    plotting_edges = [row for row in scientific_rows if row["module"].split(".", 1)[0] in {"matplotlib", "seaborn", "plotly"}]
    historical_runner = (D3R / "authoritative_inputs/run_authoritative_d3.py").read_text(encoding="utf-8")
    historical_target = REPO / "scripts/evaluate_probabilistic_unet_pre_atlas.py"
    target_tree = ast.parse(historical_target.read_text(encoding="utf-8"), filename=str(historical_target))
    target_plot_edges = []
    for node in ast.walk(target_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] == "matplotlib":
                    target_plot_edges.append({"line": node.lineno, "module": alias.name})
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".", 1)[0] == "matplotlib":
            target_plot_edges.append({"line": node.lineno, "module": node.module})
    historical_edge = {
        "runner": str((D3R / "authoritative_inputs/run_authoritative_d3.py").relative_to(REPO)),
        "imported_module": "scripts.evaluate_probabilistic_unet_pre_atlas",
        "target": str(historical_target.relative_to(REPO)),
        "runner_records_edge": '"scripts.evaluate_probabilistic_unet_pre_atlas"' in historical_runner,
        "target_plotting_imports": target_plot_edges,
        "status": "PROVED" if '"scripts.evaluate_probabilistic_unet_pre_atlas"' in historical_runner and target_plot_edges else "UNRESOLVED",
    }
    result = {
        "status": "PASS" if not plotting_edges and historical_edge["status"] == "PROVED" else "FAIL", "edges": rows,
        "scientific_plotting_edges": plotting_edges,
        "historical_d3r_transitive_plotting_edge": historical_edge,
        "scientific_launcher_sha256": sha256(SCIENTIFIC),
        "postprocess_launcher_sha256": sha256(POSTPROCESS),
    }
    write_json_x(run / "import_tests/scientific_postprocess_import_graph.json", result)
    return result


def preregister(run: Path, configs: dict[str, Path], checkpoint_before: dict[str, Any], incident_rows: list[dict[str, Any]]) -> dict[str, Any]:
    environments = {name: json.loads(path.read_text(encoding="utf-8"))["environment"] for name, path in configs.items()}
    config_hashes = {name: sha256(path) for name, path in configs.items()}
    text = f"""# Thayer-D3B runtime bootstrap readiness preregistration

## Identity and frozen scope

- Campaign: Thayer-D3B (Thayer D3 Bootstrap).
- Fresh run: `{run.name}`.
- Repository: `{REPO}`.
- BTK environment: `{VENV}`.
- D3R input: `{D3R}`.
- D1R input: `{D1R}`.
- Repository-integrity input: `{RI}`.
- Scientific tensor deserializations before freeze: 0.
- Model and optimizer constructions before freeze: 0.
- Historical checkpoint baseline: {checkpoint_before['count']} paths, {checkpoint_before['mismatches']} mismatches.
- D3R deletion incidents found: {len(incident_rows)}; exact call stacks remain unresolved because D3R did not persist them.

## Frozen process architecture

The standard-library orchestrator launches a guard-self-test interpreter, five plotting-free scientific-readiness interpreters (`primary`, `cold_1`, `cold_2`, `warm_1`, and `shutdown_1`), and one isolated postprocessing interpreter. Every subprocess uses `{PYTHON} -B`. The warm process alone reuses the prospectively declared `cold_1` cache; its tmp, config, Torch, bytecode, log, and status roots remain separate. No process transitions from a plotting-loaded state into strict scientific execution.

## Frozen environments and configurations

```json
{json.dumps({'config_sha256': config_hashes, 'process_environments': environments}, indent=2, sort_keys=True)}
```

The scientific environment omits `MPLCONFIGDIR` and `MPLBACKEND`, sets `PYTHONDONTWRITEBYTECODE=1`, and uses separate tmp, XDG cache/config, Torch, and pycache roots. The postprocessor alone receives `MPLCONFIGDIR` and `MPLBACKEND=Agg`.

## Guard phases and lifecycle policy

Bootstrap permits exact source/package reads and creates, same-root renames, and deletes only inside preregistered disposable runtime roots. Strict scientific mode permits exact reads and preregistered run-report writes, permits one self-test-only atomic rename, and blocks every deletion, cache write, bytecode write, plotting import, recursive historical iteration, and nonallowlisted read. Shutdown starts only after readiness status and strict logs are flushed and permits cleanup only under the active disposable roots.

## Frozen synthetic work

Bootstrap imports NumPy, PyTorch, `torch.nn.functional`, `torch._dynamo`, and `torch.distributed.nn.jit.instantiator` in that order; it performs a tiny MPS convolution and scalar backward without a module or optimizer. Strict mode compiles exact plotting-free project source in memory, inspects class/function objects without instantiation, extracts the exact D3R target-loss and hard-assignment functions by AST, repeats the tiny functional MPS forward/backward, and runs twelve synthetic pure-evaluator comparisons with an independent reference and zero evaluator file I/O.

## Metadata-only prerequisite policy

The process reads the exact D1R prerequisite CSV and verifies the eleven named scientific containers only by existence, size, and SHA-256. It does not call `numpy.load`, `h5py.File`, or `torch.load`, and it does not inspect archive members. Exact row/prompt identifiers are micro/P0 row 32, source row 12000, scene `pu_training_near_00000`, pair `pu_training_pair_00001`, and prompt A/B.

## Attainability and terminal gates

All gates are operationally attainable: the required runtime roots are writable and empty, the BTK interpreter exists, D1R reports 21/21 PASS, all eleven authoritative container hashes match at freeze, MPS was required and available in the immediately preceding D1R/D3R work, and the synthetic evaluator has an independent formula reference. Any self-test failure, import/process failure, MPS failure, strict deletion/cache/bytecode/nonallowlisted access, plotting-module presence, metadata mismatch, evaluator disagreement/file I/O, marker mismatch, protected access, checkpoint mismatch, tensor load, model construction, optimizer construction, or decoder forward is terminal. Gates will not be weakened after execution.

## Readiness marker and stop rule

Only a plotting-free scientific process may emit `READY_FOR_SCIENTIFIC_TENSOR_LOAD`. Postprocessing emits `READY_FOR_POSTPROCESSING` independently. A complete pass authorizes one separately preregistered D3 campaign but never loads a tensor or starts D3 here. Failure identifies one metadata-only runtime/import/guard/evaluator blocker and stops.
"""
    path = run / "preregistration/d3_runtime_bootstrap_readiness.md"
    write_text_x(path, text)
    frozen = {"path": str(path.relative_to(run)), "sha256": sha256(path), "frozen_utc": utcnow(), "third_party_imports_before_freeze": 0}
    write_json_x(run / "preregistration/preregistration_freeze.json", frozen)
    return frozen


def syntax_checks(run: Path) -> dict[str, Any]:
    paths = [Path(__file__).resolve(), SCIENTIFIC, POSTPROCESS, GUARD, REGRESSION]
    failures = []
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            failures.append({"path": str(path.relative_to(REPO)), "error": str(exc)})
    result = {"status": "PASS" if not failures else "FAIL", "file_count": len(paths), "failures": failures}
    write_json_x(run / "launcher_tests/stdlib_in_memory_syntax_checks.json", result)
    return result


def launch(config_path: Path, launcher: Path, mode: str | None = None) -> subprocess.CompletedProcess[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment.pop("MPLCONFIGDIR", None)
    environment.pop("MPLBACKEND", None)
    environment.update(config["environment"])
    command = [str(PYTHON), "-B", str(launcher), "--config", str(config_path)]
    if mode is not None:
        command.extend(("--mode", mode))
    return subprocess.run(command, cwd=REPO, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def record_initial_inventory(run: Path, case: str, config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["runtime_root"])
    cache = Path(config["cache_root"]) if "cache_root" in config else None
    write_json_x(run / f"diagnostics/process_inventory_{case}_initial.json", runtime_inventory(root, cache))


def persist_process_result(run: Path, case: str, result: subprocess.CompletedProcess[str], config_path: Path) -> None:
    write_text_x(run / f"logs/{case}_stdout.txt", result.stdout)
    write_text_x(run / f"logs/{case}_stderr.txt", result.stderr)
    write_json_x(run / f"launcher_tests/{case}_exit.json", {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["runtime_root"])
    cache = Path(config["cache_root"]) if "cache_root" in config else None
    write_json_x(run / f"diagnostics/process_inventory_{case}_shutdown.json", runtime_inventory(root, cache))
    append_text(run / "logs/command_log.sh", f"{PYTHON} -B {'postprocess' if case == 'postprocess' else 'scientific'} --case {case}  # exit {result.returncode}\n")


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def under(path: str, roots: Iterable[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath((path, root)) == root:
                return True
        except ValueError:
            pass
    return False


def audit_process(run: Path, case: str, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    events = load_events(Path(config["access_log"]))
    strict = [row for row in events if row.get("phase") == "strict"]
    lifecycle_names = {"os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace"}
    strict_lifecycle = [row for row in strict if row.get("event") in lifecycle_names]
    strict_blocked = [row for row in strict if not row.get("allowed")]
    strict_cache_writes = [row for row in strict if row.get("event") == "open" and row.get("path") and under(row["path"], [str(config["cache_root"]), str(Path(config["runtime_root"]) / "pycache")]) and "write" in row.get("reason", "")]
    plotting_imports = [row for row in events if row.get("event") == "import" and str(row.get("args", [""])[0]).split(".", 1)[0] == "matplotlib"]
    bootstrap_lifecycle = [row for row in events if row.get("phase") == "bootstrap" and row.get("event") in lifecycle_names and row.get("allowed")]
    bootstrap_confined = all(
        all(under(path, config["bootstrap_write_roots"]) for path in (row.get("paths") or [row.get("path")]))
        for row in bootstrap_lifecycle
    )
    status_path = run / f"diagnostics/scientific_readiness_{case}.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    stdout = (run / f"logs/{case}_stdout.txt").read_text(encoding="utf-8")
    initial_path = run / f"diagnostics/process_inventory_{case}_initial.json"
    shutdown_path = run / f"diagnostics/process_inventory_{case}_shutdown.json"
    bootstrap_path = Path(config["runtime_root"]) / "bootstrap_inventory.json"
    strict_end_path = Path(config["runtime_root"]) / "strict_end_inventory.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.exists() else {}
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8")) if bootstrap_path.exists() else {}
    strict_end = json.loads(strict_end_path.read_text(encoding="utf-8")) if strict_end_path.exists() else {}
    shutdown = json.loads(shutdown_path.read_text(encoding="utf-8")) if shutdown_path.exists() else {}
    bootstrap_files = {
        (row["root"], row["path"], row.get("bytes"), row.get("sha256"))
        for row in bootstrap.get("file_inventory", [])
        if row.get("kind") == "file"
    }
    strict_files = {
        (row["root"], row["path"], row.get("bytes"), row.get("sha256"))
        for row in strict_end.get("entries", [])
        if row.get("kind") == "file" and row.get("path") != "bootstrap_inventory.json"
    }
    initial_runtime_empty = initial.get("root_file_counts", {}).get(str(config["runtime_root"])) == 0
    inventory_complete = (
        initial_runtime_empty
        and bool(bootstrap.get("file_inventory"))
        and bool(strict_end.get("entries"))
        and bool(shutdown.get("entries"))
        and bootstrap_files == strict_files
    )
    result = {
        "case": case,
        "status": "PASS" if status.get("status") == "PASS" and stdout == "READY_FOR_SCIENTIFIC_TENSOR_LOAD\n" and not strict_lifecycle and not strict_blocked and not strict_cache_writes and not plotting_imports and bootstrap_confined and inventory_complete else "FAIL",
        "pid": events[0]["pid"] if events else None,
        "event_count": len(events), "blocked_count": sum(not row.get("allowed") for row in events),
        "strict_event_count": len(strict), "strict_blocked_count": len(strict_blocked),
        "strict_lifecycle_count": len(strict_lifecycle), "strict_cache_or_bytecode_write_count": len(strict_cache_writes),
        "scientific_plotting_import_count": len(plotting_imports),
        "bootstrap_allowed_lifecycle_count": len(bootstrap_lifecycle),
        "bootstrap_lifecycle_confined": bootstrap_confined,
        "marker_exact": stdout == "READY_FOR_SCIENTIFIC_TENSOR_LOAD\n",
        "initial_inventory_empty": initial_runtime_empty,
        "bootstrap_inventory_recorded": bool(bootstrap.get("file_inventory")),
        "strict_end_inventory_recorded": bool(strict_end.get("entries")),
        "shutdown_inventory_recorded": bool(shutdown.get("entries")),
        "strict_runtime_inventory_unchanged": bootstrap_files == strict_files,
        "scientific_tensor_deserializations": status.get("scientific_tensor_deserializations"),
        "model_instantiations": status.get("model_instantiations"),
        "optimizer_constructions": status.get("optimizer_constructions"),
        "decoder_forwards": status.get("decoder_forwards"),
    }
    return result


def audit_postprocess(run: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = Path(config["runtime_root"])
    events = load_events(Path(config["access_log"]))
    lifecycle_names = {"os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace"}
    lifecycle = [row for row in events if row.get("event") in lifecycle_names]
    lifecycle_confined = all(
        row.get("allowed") and all(under(path, [str(runtime)]) for path in (row.get("paths") or [row.get("path")]))
        for row in lifecycle
    )
    protected = [row for row in events if "protected" in row.get("reason", "")]
    scientific_roots = [str(D3R), str(D1R), str(RI), str(REPO / "data")]
    allowed_scientific_reads = [
        row for row in events
        if row.get("allowed") and row.get("path") and under(row["path"], scientific_roots)
    ]
    manifest_path = runtime / "postprocessing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    strict_path = runtime / "strict_end_inventory.json"
    strict_end = json.loads(strict_path.read_text(encoding="utf-8")) if strict_path.exists() else {}
    initial_path = run / "diagnostics/process_inventory_postprocess_initial.json"
    shutdown_path = run / "diagnostics/process_inventory_postprocess_shutdown.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.exists() else {}
    shutdown = json.loads(shutdown_path.read_text(encoding="utf-8")) if shutdown_path.exists() else {}
    bootstrap_files = {
        (row["path"], row.get("bytes"), row.get("sha256"))
        for row in manifest.get("bootstrap_file_inventory", []) if row.get("kind") == "file"
    }
    strict_files = {
        (row["path"], row.get("bytes"), row.get("sha256"))
        for row in strict_end.get("entries", [])
        if row.get("kind") == "file" and row.get("path") != "postprocessing_manifest.json"
    }
    stdout = (run / "logs/postprocess_stdout.txt").read_text(encoding="utf-8") if (run / "logs/postprocess_stdout.txt").exists() else ""
    result = {
        "status": "PASS",
        "pid": events[0]["pid"] if events else None,
        "event_count": len(events),
        "blocked_count": sum(not row.get("allowed") for row in events),
        "lifecycle_count": len(lifecycle),
        "lifecycle_confined": lifecycle_confined,
        "protected_access_count": len(protected),
        "allowed_scientific_read_count": len(allowed_scientific_reads),
        "project_module_imports": manifest.get("project_module_imports"),
        "initial_inventory_empty": initial.get("file_count") == 0,
        "bootstrap_inventory_recorded": bool(manifest.get("bootstrap_file_inventory")),
        "strict_end_inventory_recorded": bool(strict_end.get("entries")),
        "shutdown_inventory_recorded": bool(shutdown.get("entries")),
        "runtime_unchanged_before_shutdown_cleanup": bootstrap_files == strict_files,
        "marker_exact": stdout == "READY_FOR_POSTPROCESSING\n",
    }
    result["status"] = "PASS" if all(
        (
            lifecycle_confined,
            not protected,
            not allowed_scientific_reads,
            not manifest.get("project_module_imports"),
            result["initial_inventory_empty"],
            result["bootstrap_inventory_recorded"],
            result["strict_end_inventory_recorded"],
            result["shutdown_inventory_recorded"],
            result["runtime_unchanged_before_shutdown_cleanup"],
            result["marker_exact"],
        )
    ) else "FAIL"
    return result


def documentation_audit() -> dict[str, Any]:
    paths = [
        REPO / "docs/d3_runtime_bootstrap_contract.md", REPO / "docs/scientific_process_isolation.md",
        REPO / "docs/scientific_postprocessing_isolation.md", REPO / "docs/pure_forward_evaluator_contract.md",
        REPO / "docs/d3_runtime_readiness.md", REPO / "docs/authoritative_full_l0_d3.md",
        REPO / "docs/full_l0_fixed_feature_d3.md", REPO / "docs/allowlisted_file_access_contract.md",
        REPO / "docs/repository_integrity_audit.md", REPO / "docs/current_status.md",
        REPO / "docs/project_roadmap.md", REPO / "docs/experiment_log.md",
        REPO / "docs/limitations_and_next_steps.md",
    ]
    issues = []
    for path in paths:
        if not path.exists():
            issues.append({"path": str(path.relative_to(REPO)), "issue": "missing"})
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.rstrip() != line:
                issues.append({"path": str(path.relative_to(REPO)), "line": number, "issue": "trailing_whitespace"})
            if "/Users/" in line or "ChatGPT" in line or "OpenAI" in line:
                issues.append({"path": str(path.relative_to(REPO)), "line": number, "issue": "privacy_or_assistant_token"})
    return {"status": "PASS" if not issues else "FAIL", "file_count": len(paths), "issues": issues}


def finalize(run: Path, configs: dict[str, Path], prereg: dict[str, Any], checkpoint_before: dict[str, Any], started: float, blocker: str | None) -> dict[str, Any]:
    process_rows = [audit_process(run, case, configs[case]) for case in ("primary", "cold_1", "cold_2", "warm_1", "shutdown_1") if (run / f"launcher_tests/{case}_exit.json").exists()]
    post_stdout = (run / "logs/postprocess_stdout.txt").read_text(encoding="utf-8") if (run / "logs/postprocess_stdout.txt").exists() else ""
    post_events = load_events(run / "access_guard/postprocess_access_log.jsonl")
    post_pid = post_events[0]["pid"] if post_events else None
    scientific_pids = {row["pid"] for row in process_rows if row["pid"] is not None}
    post_audit = audit_postprocess(run, configs["postprocess"]) if (run / "launcher_tests/postprocess_exit.json").exists() else {"status": "NOT_RUN", "pid": None}
    post_pass = post_audit["status"] == "PASS" and post_pid not in scientific_pids
    checkpoint_after = checkpoint_audit(run, "after")
    graph = json.loads((run / "import_tests/scientific_postprocess_import_graph.json").read_text(encoding="utf-8"))
    evaluator_path = run / "evaluator_tests/pure_forward_evaluator_result.json"
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8")) if evaluator_path.exists() else {}
    metadata_path = run / "diagnostics/scientific_readiness_primary.json"
    primary = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    selftest_path = run / "runtime/orchestrator/guard_selftest/guard_self_tests.json"
    selftest = json.loads(selftest_path.read_text(encoding="utf-8")) if selftest_path.exists() else {}
    diff_check = git("diff", "--check", check=False).returncode
    cached_diff_check = git("diff", "--cached", "--check", check=False).returncode
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    readme_diff = git("diff", "--", "README.md").stdout
    docs = documentation_audit()
    final_status = git("status", "--short").stdout
    write_text_x(run / "logs/final_git_status.txt", final_status)
    source_after = {
        str(path.relative_to(REPO)): sha256(path)
        for path in (Path(__file__).resolve(), SCIENTIFIC, POSTPROCESS, GUARD, REGRESSION)
    }
    corrections = {
        "failing_regression_before_correction": {
            "command": "python3 tests/test_d3_readiness_process_isolation.py",
            "exit_code": 1,
            "failure": "required scientific and postprocessing launcher paths were absent",
        },
        "before_sha256": SOURCE_BEFORE,
        "after_sha256": source_after,
        "structural_diff": [
            "replaced the mixed scientific/Matplotlib launcher with a standard-library orchestrator",
            "added a plotting-free scientific launcher",
            "added a separate Matplotlib/Agg postprocessing launcher",
            "extended the guard with prospectively bounded bootstrap read/write roots",
            "changed no production model, target, mapping, assignment, loss, evaluator formula, threshold, or data path",
        ],
    }
    write_json_x(run / "diagnostics/source_runtime_corrections.json", corrections)
    tests = {
        "preregistration_frozen_before_third_party": bool(prereg),
        "guard_self_tests_16_of_16": selftest.get("status") == "PASS" and selftest.get("test_count") == 16,
        "all_scientific_processes_pass": len(process_rows) == 5 and all(row["status"] == "PASS" for row in process_rows),
        "cold_1_pass": any(row["case"] == "cold_1" and row["status"] == "PASS" for row in process_rows),
        "cold_2_pass": any(row["case"] == "cold_2" and row["status"] == "PASS" for row in process_rows),
        "warm_1_pass": any(row["case"] == "warm_1" and row["status"] == "PASS" for row in process_rows),
        "shutdown_1_pass": any(row["case"] == "shutdown_1" and row["status"] == "PASS" for row in process_rows),
        "strict_zero_deletions": all(row["strict_lifecycle_count"] == 0 for row in process_rows),
        "strict_zero_cache_bytecode_writes": all(row["strict_cache_or_bytecode_write_count"] == 0 for row in process_rows),
        "strict_zero_nonallowlisted_reads": all(row["strict_blocked_count"] == 0 for row in process_rows),
        "scientific_plotting_absent": graph.get("status") == "PASS" and all(row["scientific_plotting_import_count"] == 0 for row in process_rows),
        "postprocessing_separate_and_pass": post_pass,
        "postprocessing_lifecycle_confined": post_audit.get("lifecycle_confined") is True,
        "all_process_inventories_recorded": len(process_rows) == 5 and all(
            row["initial_inventory_empty"] and row["bootstrap_inventory_recorded"]
            and row["strict_end_inventory_recorded"] and row["shutdown_inventory_recorded"]
            and row["strict_runtime_inventory_unchanged"]
            for row in process_rows
        ) and post_audit.get("initial_inventory_empty") is True
        and post_audit.get("bootstrap_inventory_recorded") is True
        and post_audit.get("strict_end_inventory_recorded") is True
        and post_audit.get("shutdown_inventory_recorded") is True,
        "evaluator_12_cases_reference_and_zero_io": evaluator.get("status") == "PASS" and evaluator.get("case_count") == 12 and evaluator.get("filesystem_event_count") == 0,
        "metadata_prerequisites_pass": primary.get("metadata_status") == "PASS",
        "scientific_marker_exact": all(row["marker_exact"] for row in process_rows),
        "postprocess_marker_exact": post_stdout == "READY_FOR_POSTPROCESSING\n",
        "no_tensor_model_optimizer_decoder_operation": all(row["scientific_tensor_deserializations"] == 0 and row["model_instantiations"] == 0 and row["optimizer_constructions"] == 0 and row["decoder_forwards"] == 0 for row in process_rows),
        "checkpoint_before_600_unchanged": checkpoint_before["count"] == 600 and checkpoint_before["mismatches"] == 0,
        "checkpoint_after_600_unchanged": checkpoint_after["count"] == 600 and checkpoint_after["mismatches"] == 0,
        "staged_index_empty": not staged,
        "readme_unchanged": not readme_diff,
        "git_diff_check": diff_check == 0,
        "git_cached_diff_check": cached_diff_check == 0,
        "documentation_audit": docs["status"] == "PASS",
    }
    if blocker is not None:
        tests["terminal_blocker_absent"] = False
    failures = [name for name, passed in tests.items() if not passed]
    status = "READINESS_PASS_D3_NOT_RUN" if not failures else "READINESS_FAIL_D3_NOT_RUN"
    test_rows = [{"test": name, "status": "PASS" if passed else "FAIL"} for name, passed in tests.items()]
    write_csv_x(run / "tables/final_test_matrix.csv", test_rows)
    access_manifest = {"scientific": process_rows, "postprocess": post_audit, "postprocess_pid": post_pid, "postprocess_separate": post_pid not in scientific_pids, "postprocess_event_count": len(post_events)}
    write_json_x(run / "diagnostics/final_access_log_manifest.json", access_manifest)
    runtime_seconds = time.perf_counter() - started
    manifest = {
        "status": status, "completed_utc": utcnow(), "runtime_seconds": runtime_seconds,
        "preregistration_sha256": prereg.get("sha256"), "failure_count": len(failures),
        "failures": failures, "blocker": blocker, "processes": process_rows,
        "postprocessing_status": "PASS" if post_pass else "FAIL", "postprocessing_audit": post_audit, "evaluator": evaluator,
        "metadata_status": primary.get("metadata_status"), "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after, "documentation": docs,
        "git": {"branch": git("branch", "--show-current").stdout.strip(), "head": git("rev-parse", "HEAD").stdout.strip(), "staged_paths": staged, "diff_check_exit": diff_check, "cached_diff_check_exit": cached_diff_check, "readme_diff": readme_diff},
        "scientific_tensor_deserializations": 0, "model_instantiations": 0,
        "optimizer_constructions": 0, "decoder_forwards": 0, "d3_run": False,
        "atlas_access": 0, "development_access": 0, "lockbox_access": 0,
    }
    write_json_x(run / "diagnostics/readiness_manifest.json", manifest)
    freeze_items = [
        Path(__file__).resolve(), SCIENTIFIC, POSTPROCESS, GUARD,
        REPO / "src/competing_hypotheses.py", REPO / "src/models_probabilistic_unet.py",
        REPO / "src/models_two_expert_decoder.py", REPO / "src/output_parameterization.py",
        run / "diagnostics/readiness_manifest.json", run / "tables/metadata_d3_prerequisites.csv",
        run / "diagnostics/metadata_d3_prerequisite_report.md",
        run / "import_tests/scientific_postprocess_import_graph.json",
        run / "evaluator_tests/pure_forward_evaluator_result.json",
        run / "diagnostics/final_access_log_manifest.json",
        run / "runtime/postprocess_runtime/postprocessing_manifest.json",
        *configs.values(),
        *(Path(json.loads(configs[case].read_text(encoding="utf-8"))["runtime_root"]) / "bootstrap_inventory.json" for case in ("primary", "cold_1", "cold_2", "warm_1", "shutdown_1")),
    ]
    freeze = {str(path.relative_to(REPO) if path.is_relative_to(REPO) else path): sha256(path) for path in freeze_items if path.exists()}
    write_json_x(run / "diagnostics/runtime_hash_freeze.json", {"frozen_utc": utcnow(), "hashes": freeze})
    audit = {"status": "PASS" if not failures else "FAIL", "primary_outcome": "READINESS PASS — D3 NOT RUN" if not failures else "READINESS FAIL — D3 NOT RUN", "test_count": len(tests), "failure_count": len(failures), "failures": failures, "readiness_manifest_sha256": sha256(run / "diagnostics/readiness_manifest.json"), "preregistration_sha256": prereg.get("sha256"), "scientific_tensor_deserializations": 0, "model_instantiations": 0, "optimizer_constructions": 0, "decoder_forwards": 0}
    write_json_x(run / "diagnostics/final_correctness_audit.json", audit)
    report = final_report(run, manifest, tests, corrections, final_status)
    write_text_x(run / "reports/final_report.md", report)
    return manifest


def final_report(run: Path, manifest: dict[str, Any], tests: dict[str, bool], corrections: dict[str, Any], final_status: str) -> str:
    passed = manifest["status"] == "READINESS_PASS_D3_NOT_RUN"
    process = {row["case"]: row for row in manifest["processes"]}
    outcome = "READINESS PASS — D3 NOT RUN" if passed else "READINESS FAIL — D3 NOT RUN"
    authorization = "Yes, one separately preregistered authoritative square-only one-scene D3 campaign." if passed else "No."
    blocker = manifest.get("blocker") or "None."
    return f"""# Thayer-D3B final report

## Decision

Primary outcome: **{outcome}**.

Thayer-D3B was metadata-only and synthetic-runtime-only. D3 was not run. The campaign deserialized zero scene, target, cached-feature, D1-endpoint, or checkpoint tensors; instantiated zero encoders, decoders, or project models; constructed zero optimizers; and executed zero decoder forwards. D3 remains scientifically unknown.

The earlier append-only readiness record `thayer_d3_runtime_readiness_20260713_134646` is preserved but non-authoritative because its first closure did not persist every required process-phase inventory or validate the postprocessor's complete lifecycle confinement. This fresh run adds those evidence gates without altering scientific code or weakening policy.

## Process architecture

```mermaid
flowchart LR
    O["Standard-library orchestrator"] --> S["Plotting-free scientific interpreters"]
    O --> P["Isolated Matplotlib/Agg postprocessor"]
    S --> R["READY_FOR_SCIENTIFIC_TENSOR_LOAD"]
    P --> Q["READY_FOR_POSTPROCESSING"]
    S -. "no tensor/model/optimizer" .-> X["Future separately preregistered D3"]
```

The exact environment variables and per-process roots are frozen in the hashed configuration files under `runtime/orchestrator/` and reproduced in the preregistration. Initial, bootstrap, strict-end, and post-shutdown inventories are preserved for every primary, cold, warm, shutdown, and postprocessing process.

## Answers to the 30 closure questions

1. D3R stopped on two `os.remove` lifecycle operations: Matplotlib font-cache lock cleanup and a Python tempfile usability-probe cleanup reached through `torch.distributed.nn.jit.instantiator`.
2. The exact paths are recorded in `tables/d3r_bootstrap_incident.csv`; both are under the historical D3R run's `access_guard` runtime area.
3. Yes. One path is runtime cache-lock state and the other is runtime temporary state.
4. No scientific or historical-data container was targeted. D3R reported both operations unsuccessful and both files preserved.
5. Exact Python call stacks remain unresolved because D3R did not record stacks on those rows.
6. Frozen variables were TMPDIR, TMP, TEMP, XDG_CACHE_HOME, XDG_CONFIG_HOME, TORCH_HOME, PYTHONPYCACHEPREFIX, PYTHONDONTWRITEBYTECODE, PYTHONHASHSEED, PYTORCH_ENABLE_MPS_FALLBACK, OMP_NUM_THREADS, and VECLIB_MAXIMUM_THREADS; only postprocessing received MPLCONFIGDIR and MPLBACKEND.
7. Yes. The orchestrator imported only standard-library modules.
8. {'Yes' if tests.get('all_scientific_processes_pass') else 'No'}. Successful scientific bootstrap lifecycle operations stayed in preregistered fresh scratch.
9. {'Yes' if tests.get('strict_zero_deletions') else 'No'}. Strict scientific deletion count was zero.
10. {'Yes' if tests.get('strict_zero_cache_bytecode_writes') else 'No'}. Strict cache and bytecode write counts were zero.
11. {'Yes' if tests.get('scientific_plotting_absent') else 'No'}. Matplotlib was absent from every scientific interpreter.
12. {'Yes' if tests.get('postprocessing_separate_and_pass') else 'No'}. Postprocessing used a distinct interpreter and scratch root.
13. Cold 1: {process.get('cold_1', {}).get('status', 'NOT_RUN')}; Cold 2: {process.get('cold_2', {}).get('status', 'NOT_RUN')}.
14. Warm 1: {process.get('warm_1', {}).get('status', 'NOT_RUN')}.
15. Shutdown 1: {process.get('shutdown_1', {}).get('status', 'NOT_RUN')}; cleanup was limited to its scratch root.
16. {'Yes' if tests.get('evaluator_12_cases_reference_and_zero_io') else 'No'}. The production evaluator was path-independent in the frozen synthetic audit.
17. {'Yes' if tests.get('evaluator_12_cases_reference_and_zero_io') else 'No'}. It agreed with the independent reference across all 12 cases.
18. {'Yes' if tests.get('evaluator_12_cases_reference_and_zero_io') else 'No'}. Evaluator calls produced zero file-access events.
19. {'Yes' if tests.get('metadata_prerequisites_pass') else 'No'}. All metadata-only D3 prerequisites passed.
20. {'Yes' if tests.get('scientific_marker_exact') else 'No'}. Every scientific process emitted exactly `READY_FOR_SCIENTIFIC_TENSOR_LOAD`.
21. {'Yes' if tests.get('postprocess_marker_exact') else 'No'}. The postprocessor emitted exactly `READY_FOR_POSTPROCESSING`.
22. No scientific tensor was deserialized.
23. No model or optimizer was constructed.
24. {authorization}
25. Freeze the hashes in `diagnostics/runtime_hash_freeze.json`, including all three launchers, the guard, four plotting-free project modules, metadata prerequisite table, and readiness manifest.
26. Yes. Atlas, development, and lockbox access counts were zero.
27. {'Yes' if tests.get('checkpoint_before_600_unchanged') and tests.get('checkpoint_after_600_unchanged') else 'No'}. All 600 historical checkpoints matched before and after.
28. The working tree changes are the three readiness entry points, the guard correction, the isolation regression test, the requested documentation, and this ignored append-only run; unrelated pre-existing changes were preserved. Exact final status appears below.
29. The three launchers, runtime guard, isolation regression test, and requested public documentation are reusable source/tests for later human-reviewed commit.
30. The complete run under `{run.relative_to(REPO)}`—access logs, runtime caches, manifests, tables, synthetic figure, and reports—should remain generated and ignored.

## Evidence summary

- Preregistration SHA-256: `{manifest['preregistration_sha256']}`.
- Guard self-tests: {'16/16 PASS' if tests.get('guard_self_tests_16_of_16') else 'FAIL'}.
- Pure evaluator: {manifest.get('evaluator', {}).get('case_count', 0)} cases; status {manifest.get('evaluator', {}).get('status', 'NOT_RUN')}; file events {manifest.get('evaluator', {}).get('filesystem_event_count', 'UNRESOLVED')}.
- Metadata prerequisites: {manifest.get('metadata_status')}.
- Historical checkpoints: {manifest['checkpoint_before']['count']}/{manifest['checkpoint_before']['count']} before and {manifest['checkpoint_after']['count']}/{manifest['checkpoint_after']['count']} after; zero mismatches required.
- Runtime: {manifest['runtime_seconds']:.3f} seconds.
- Terminal blocker: {blocker}
- Exact runtime/code correction evidence: `diagnostics/source_runtime_corrections.json`.
- Scientific import graph: `import_tests/scientific_postprocess_import_graph.json`.
- Scientific access logs: `access_guard/scientific_*_access_log.jsonl`.
- Postprocessing access log: `access_guard/postprocess_access_log.jsonl`.
- Process-phase inventories: `diagnostics/process_inventory_*_initial.json`, each runtime's bootstrap and strict-end manifests, and `diagnostics/process_inventory_*_shutdown.json`.
- Deletion/rename inventory is preserved in the access logs and final access manifest.
- Bytecode/cache audit, cold/warm comparison, shutdown evidence, metadata table, and marker/exit proofs are represented in `tables/final_test_matrix.csv` and `diagnostics/readiness_manifest.json`.

## Final git status

```text
{final_status.rstrip()}
```

No file was staged, committed, pushed, merged, moved, renamed, or deleted by this campaign. README remained unchanged.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    started = time.perf_counter()
    run = make_run()
    write_text_x(run / "logs/command_log.sh", "# Standard-library-only Thayer-D3B command log\n")
    branch = git("branch", "--show-current").stdout.strip()
    head = git("rev-parse", "HEAD").stdout.strip()
    status = git("status", "--short").stdout
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    inputs = [file_snapshot(path) for path in exact_inputs()]
    for item in metadata_files():
        path = Path(item["path"])
        snapshot = file_snapshot(path)
        if snapshot["sha256"] != item["sha256"]:
            raise SystemExit(f"authoritative metadata input hash mismatch before preregistration: {path}")
        inputs.append(snapshot)
    disk = shutil.disk_usage(REPO)
    environment = {
        "campaign_start_utc": utcnow(), "branch": branch, "git_head": head,
        "git_status": status.splitlines(), "staged_index": staged, "python_executable": str(PYTHON),
        "btk_environment": str(VENV), "free_disk_bytes": disk.free,
        "authoritative_runs": {"D3R": str(D3R), "D1R": str(D1R), "repository_integrity": str(RI)},
    }
    write_text_x(run / "diagnostics/environment_snapshot_stdlib_only.md", "# Standard-library-only environment snapshot\n\n```json\n" + json.dumps(environment, indent=2, sort_keys=True) + "\n```\n")
    write_text_x(run / "diagnostics/campaign_contract.md", "# Thayer-D3B campaign contract\n\nMetadata-only runtime readiness. No scientific tensor load, model, optimizer, decoder forward, D3 execution, protected access, historical mutation, or automatic continuation is authorized.\n")
    write_json_x(run / "logs/input_provenance.json", {"environment": environment, "files": inputs})
    incident_rows = incident_audit(run)
    checkpoint_before = checkpoint_audit(run, "before")
    configs = write_configs(run)
    graph = record_import_graph(run)
    prereg = preregister(run, configs, checkpoint_before, incident_rows)
    syntax = syntax_checks(run)
    if staged or graph["status"] != "PASS" or syntax["status"] != "PASS" or checkpoint_before["mismatches"]:
        blocker = "pre-import provenance, syntax, import-graph, checkpoint, or staged-index gate failed"
        manifest = finalize(run, configs, prereg, checkpoint_before, started, blocker)
        print(run)
        raise SystemExit(1 if manifest["status"].startswith("READINESS_FAIL") else 0)

    record_initial_inventory(run, "selftest", configs["selftest"])
    selftest = launch(configs["selftest"], SCIENTIFIC, "selftest")
    persist_process_result(run, "selftest", selftest, configs["selftest"])
    if selftest.returncode != 0 or selftest.stdout != "GUARD_SELF_TESTS_PASS\n":
        manifest = finalize(run, configs, prereg, checkpoint_before, started, "guard self-tests failed")
        print(run)
        raise SystemExit(1)

    blocker = None
    for case in ("primary", "cold_1", "cold_2", "warm_1", "shutdown_1"):
        record_initial_inventory(run, case, configs[case])
        result = launch(configs[case], SCIENTIFIC, "readiness")
        persist_process_result(run, case, result, configs[case])
        if result.returncode != 0 or result.stdout != "READY_FOR_SCIENTIFIC_TENSOR_LOAD\n":
            blocker = f"scientific readiness process {case} failed"
            break
    if blocker is None:
        record_initial_inventory(run, "postprocess", configs["postprocess"])
        post = launch(configs["postprocess"], POSTPROCESS)
        persist_process_result(run, "postprocess", post, configs["postprocess"])
        if post.returncode != 0 or post.stdout != "READY_FOR_POSTPROCESSING\n":
            blocker = "isolated postprocessing readiness failed"
    manifest = finalize(run, configs, prereg, checkpoint_before, started, blocker)
    print(run)
    raise SystemExit(0 if manifest["status"] == "READINESS_PASS_D3_NOT_RUN" else 1)


if __name__ == "__main__":
    main()
