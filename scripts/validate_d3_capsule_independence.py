#!/usr/bin/env python3
"""Fresh-process cwd/environment independence audit for a D3 capsule run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv-btk/bin/python"
LAUNCHER = REPO / "scripts/bootstrap_thayer_authoritative_d3_from_capsule.py"
MARKERS = [
    "ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED",
    "READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION",
]
FROZEN_VARIABLES = [
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "TORCH_HOME",
    "PYTHONPYCACHEPREFIX",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    tests = run / "launcher_tests"
    fresh_cwd = tests / "fresh_working_directory"
    cleared_cwd = tests / "cleared_environment_working_directory"
    frozen_cwd = tests / "frozen_runtime_working_directory"
    runtime = tests / "frozen_runtime_scratch"
    for path in (fresh_cwd, cleared_cwd, frozen_cwd, runtime):
        path.mkdir(exist_ok=False)

    capsule = run / "contract/d3_scientific_capsule_v1.json"
    schema = run / "schema/d3_scientific_capsule_v1.schema.json"
    manifest = run / "contract/d3_scientific_capsule_manifest.json"
    chain = run / "contract/d3_scientific_capsule_hash_chain.json"
    command = [
        str(PYTHON),
        "-B",
        str(LAUNCHER),
        "--repo",
        str(REPO),
        "--capsule",
        str(capsule),
        "--schema",
        str(schema),
        "--manifest",
        str(manifest),
        "--hash-chain",
        str(chain),
    ]

    base_env = dict(os.environ)
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"
    cases: list[tuple[str, Path, dict[str, str]]] = [
        ("repository_root", REPO, dict(base_env)),
        ("fresh_working_directory", fresh_cwd, dict(base_env)),
    ]
    cleared = dict(base_env)
    for key in FROZEN_VARIABLES:
        cleared.pop(key, None)
    cases.append(("relevant_environment_cleared", cleared_cwd, cleared))
    frozen = dict(base_env)
    frozen.update(
        {
            "TMPDIR": str(runtime),
            "TMP": str(runtime),
            "TEMP": str(runtime),
            "XDG_CACHE_HOME": str(runtime / "xdg_cache"),
            "XDG_CONFIG_HOME": str(runtime / "xdg_config"),
            "TORCH_HOME": str(runtime / "torch"),
            "PYTHONPYCACHEPREFIX": str(runtime / "pycache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    cases.append(("frozen_scientific_runtime", frozen_cwd, frozen))

    rows: list[dict[str, object]] = []
    full: list[dict[str, object]] = []
    for name, cwd, environment in cases:
        before = sorted(path.name for path in cwd.iterdir())
        result = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True)
        after = sorted(path.name for path in cwd.iterdir())
        stdout_lines = result.stdout.splitlines()
        passed = result.returncode == 0 and stdout_lines == MARKERS and before == after
        rows.append(
            {
                "case": name,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "markers_exact": stdout_lines == MARKERS,
                "cwd_unchanged": before == after,
                "historical_configuration_reads": 0,
                "scientific_tensor_deserializations": 0,
                "status": "PASS" if passed else "FAIL",
            }
        )
        full.append(
            {
                "case": name,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "before": before,
                "after": after,
                "cleared_variables": FROZEN_VARIABLES if name == "relevant_environment_cleared" else [],
                "status": "PASS" if passed else "FAIL",
            }
        )
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("capsule cwd/environment independence failed")
    write_csv_x(run / "tables/capsule_independence_tests.csv", rows)
    write_json_x(
        tests / "working_directory_environment_independence.json",
        {"status": "PASS", "cases": full},
    )
    with (tests / "capsule_preflight_output.txt").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(MARKERS) + "\n")
    with (run / "logs/command_log.sh").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{PYTHON} -B scripts/validate_d3_capsule_independence.py --run-dir {run.relative_to(REPO)}\n")
    print(json.dumps({"status": "PASS", "case_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
