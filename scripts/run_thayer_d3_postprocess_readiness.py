"""Isolated synthetic postprocessing readiness process for Thayer-D3B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(runtime: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for current, directory_names, file_names in os.walk(runtime):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        rows.append({"path": str(current_path.relative_to(runtime)) or ".", "kind": "directory", "bytes": 0})
        for name in file_names:
            path = current_path / name
            rows.append(
                {
                    "path": str(path.relative_to(runtime)),
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def load_guard(config: dict[str, Any]):
    guard_path = Path(config["guard_source"])
    spec = importlib.util.spec_from_file_location("thayer_d3_runtime_guard_postprocess", guard_path)
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
        exact_read_files=(guard_path,),
        strict_write_roots=(),
        strict_atomic_roots=(),
        bootstrap_write_roots=(Path(config["runtime_root"]),),
        bootstrap_read_roots=tuple(Path(value) for value in config["bootstrap_read_roots"]),
    )
    guard = module.TwoPhaseGuard(policy)
    guard.install()
    return guard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    runtime = Path(config["runtime_root"])
    write_json_x(
        runtime / "pre_guard_access.json",
        {
            "reads_before_guard_install": [str(args.config.resolve()), config["guard_source"]],
            "third_party_imports_before_guard": [],
            "timestamp_utc": utcnow(),
        },
    )
    expected = config["environment"]
    observed = {key: os.environ.get(key, "") for key in expected}
    if observed != expected:
        raise RuntimeError(f"postprocessing environment mismatch: {observed!r} != {expected!r}")
    guard = load_guard(config)
    before = guard.snapshot()["event_count"]
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot

        figure, axis = pyplot.subplots(figsize=(3.0, 2.0))
        axis.plot([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
        axis.set_title("Synthetic readiness")
        output = runtime / "output/synthetic_readiness.png"
        figure.savefig(output)
        pyplot.close(figure)
    after = guard.snapshot()["event_count"]
    manifest = {
        "status": "PASS",
        "marker": "READY_FOR_POSTPROCESSING",
        "runtime_seconds": time.perf_counter() - started,
        "event_count_before": before,
        "event_count_after": after,
        "event_delta": after - before,
        "warnings": [str(item.message) for item in caught],
        "matplotlib_version": matplotlib.__version__,
        "backend": matplotlib.get_backend(),
        "output": str(output),
        "scientific_input_reads": 0,
        "project_module_imports": sorted(name for name in sys.modules if name == "src" or name.startswith("src.")),
        "bootstrap_file_inventory": inventory(runtime),
        "completed_utc": utcnow(),
    }
    if manifest["project_module_imports"]:
        raise RuntimeError(f"project modules entered postprocessing: {manifest['project_module_imports']}")
    write_json_x(runtime / "postprocessing_manifest.json", manifest)
    guard.transition("shutdown")
    write_json_x(
        runtime / "strict_end_inventory.json",
        {
            "captured_before_shutdown_cleanup": True,
            "entries": inventory(runtime),
            "completed_utc": utcnow(),
        },
    )
    probe = runtime / "tmp/shutdown_probe.tmp"
    probe.open("x").close()
    os.remove(probe)
    print("READY_FOR_POSTPROCESSING")


if __name__ == "__main__":
    main()
