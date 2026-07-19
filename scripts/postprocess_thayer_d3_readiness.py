"""Isolated postprocessing probe for Thayer-D3B readiness outputs only."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]


def load_guard():
    path = REPO / "scripts/thayer_d3_runtime_guard.py"
    spec = importlib.util.spec_from_file_location("thayer_d3_postprocess_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load postprocess guard")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--input-table", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    input_table = args.input_table.resolve()
    runtime = args.runtime_root.resolve()
    if run.parent != REPO / "outputs/runs" or not run.name.startswith("thayer_d3_runtime_readiness_"):
        raise SystemExit("invalid readiness run")
    if not input_table.is_relative_to(run / "tables"):
        raise SystemExit("postprocessing input must be a new-run table")
    if runtime.exists():
        raise SystemExit("postprocessing runtime collision")
    runtime.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "cache", "config", "matplotlib", "torch", "pycache", "postprocess_tmp"):
        (runtime / name).mkdir(exist_ok=False)
    environment = {
        "TMPDIR": str(runtime / "tmp"),
        "TMP": str(runtime / "tmp"),
        "TEMP": str(runtime / "tmp"),
        "MPLCONFIGDIR": str(runtime / "matplotlib"),
        "XDG_CACHE_HOME": str(runtime / "cache"),
        "XDG_CONFIG_HOME": str(runtime / "config"),
        "TORCH_HOME": str(runtime / "torch"),
        "PYTHONPYCACHEPREFIX": str(runtime / "pycache"),
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "20260713",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    os.environ.update(environment)
    frozen = json.loads((run / "runtime/bootstrap_environment.json").read_text(encoding="utf-8"))["process_environments"]["postprocess_isolated"]
    observed = {key: os.environ.get(key) for key in frozen}
    if observed != frozen:
        raise SystemExit("postprocessing environment differs from preregistration")
    module = load_guard()
    policy = module.GuardPolicy(
        repository_root=REPO,
        fresh_run_root=run,
        runtime_root=runtime,
        access_log=run / "access_guard/postprocess_access_log.jsonl",
        blocked_log=run / "access_guard/postprocess_blocked_access_log.jsonl",
        exact_read_files=(input_table, Path(__file__).resolve(), REPO / "scripts/thayer_d3_runtime_guard.py"),
        strict_write_roots=(),
        strict_atomic_roots=(),
    )
    guard = module.TwoPhaseGuard(policy)
    guard.install()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with input_table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    passed = sum(row.get("status") == "PASS" for row in rows)
    failed = len(rows) - passed
    figure, axis = plt.subplots(figsize=(3.0, 2.0))
    axis.bar(("PASS", "FAIL"), (passed, failed), color=("#2a7f62", "#b54b4b"))
    axis.set_ylabel("metadata checks")
    figure.tight_layout()
    output = runtime / "postprocess_tmp/metadata_status.png"
    figure.savefig(output)
    plt.close(figure)
    with (runtime / "postprocess_status.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "PASS",
                "input": str(input_table.relative_to(run)),
                "output": str(output.relative_to(run)),
                "row_count": len(rows),
                "scientific_input_reads": 0,
                "scientific_output_mutations": 0,
                "environment": environment,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
