#!/usr/bin/env python3
"""Isolated, manifest-only postprocessing worker for Thayer-D3I."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    return parser.parse_args()


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def validate_manifest(path: Path) -> tuple[dict[str, Any], Path, list[Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "thayer-d3i-postprocessing-input-v1":
        raise RuntimeError("D3I-POST-MANIFEST-SCHEMA")
    run = Path(manifest["run"]).resolve()
    if not run.is_dir() or path.resolve().parent != (run / "postprocessing_inputs").resolve():
        raise RuntimeError("D3I-POST-MANIFEST-RUN")
    inputs = []
    for record in manifest.get("inputs", []):
        candidate = Path(record["path"]).resolve()
        try:
            candidate.relative_to(run)
        except ValueError as exc:
            raise RuntimeError("D3I-POST-INPUT-OUTSIDE-RUN") from exc
        lowered = str(candidate).casefold()
        if any(token in lowered for token in ("cached_features", "one_scene_payload", "d1_penultimate_endpoints", "initial_state_square")):
            raise RuntimeError("D3I-POST-ORIGINAL-SCIENTIFIC-INPUT")
        if not candidate.is_file() or sha256_file(candidate) != record["sha256"]:
            raise RuntimeError("D3I-POST-INPUT-SHA")
        inputs.append(candidate)
    if not inputs:
        raise RuntimeError("D3I-POST-INPUT-EMPTY")
    return manifest, run, inputs


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for row in raw:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        rows.append(converted)
    return rows


def save_line(plt: Any, figures: Path, name: str, x: list[float], series: list[tuple[str, list[float]]], ylabel: str) -> str:
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for label, values in series:
        axis.plot(x, values, marker="o", markersize=2.5, linewidth=1.2, label=label)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if len(series) > 1:
        axis.legend(frameon=False, fontsize=8)
    target = figures / f"{name}.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return str(target)


def main() -> int:
    args = parse_args()
    manifest, run, inputs = validate_manifest(args.input_manifest.resolve())
    if os.environ.get("MPLBACKEND") != "Agg":
        raise RuntimeError("D3I-POST-MPLBACKEND")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figures = run / "figures"
    created: list[str] = []
    if manifest["mode"] == "synthetic":
        result = json.loads(inputs[0].read_text(encoding="utf-8"))
        figure, axis = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
        axis.bar(["loss", "updates"], [float(result["loss"]), float(result["nonzero_update_count"])])
        axis.set_title("V4 synthetic full-stack execution")
        target = figures / "synthetic_full_stack.png"
        figure.savefig(target, dpi=160)
        plt.close(figure)
        created.append(str(target))
    else:
        trajectory_path = next(path for path in inputs if path.name == "trajectory.csv")
        arrays_path = next(path for path in inputs if path.name == "selected_outputs.npz")
        rows = load_rows(trajectory_path)
        x = [float(row["step"]) for row in rows]
        created.extend(
            (
                save_line(plt, figures, "objective_coverage_curves", x, [("objective", [float(r["objective"]) for r in rows]), ("own", [float(r["own_coverage"]) for r in rows]), ("alternate", [float(r["alternate_coverage"]) for r in rows]), ("both", [float(r["both_mode_coverage"]) for r in rows])], "value"),
                save_line(plt, figures, "assignment_margin_trajectory", x, [("prompt A", [float(r["assignment_margin_prompt_a"]) for r in rows]), ("prompt B", [float(r["assignment_margin_prompt_b"]) for r in rows])], "swap minus identity cost"),
                save_line(plt, figures, "z_band_trajectory", x, [("z-band error", [float(r["z_band_error"]) for r in rows])], "normalized MSE"),
                save_line(plt, figures, "gradient_update_trajectory", x, [("gradient expert 1", [float(r["gradient_norm_expert_1"]) for r in rows]), ("gradient expert 2", [float(r["gradient_norm_expert_2"]) for r in rows]), ("update expert 1", [float(r["update_norm_expert_1"]) for r in rows]), ("update expert 2", [float(r["update_norm_expert_2"]) for r in rows])], "norm"),
                save_line(plt, figures, "expert_activity", x, [("expert diameter", [float(r["expert_diameter"]) for r in rows])], "primary normalized distance"),
                save_line(plt, figures, "prompt_collapse", x, [("prompt swap", [float(r["set_prompt_swap"]) for r in rows])], "gate"),
                save_line(plt, figures, "d1_feature_distance", x, [("D1 distance", [float(r["d1_feature_distance"]) for r in rows])], "L2 distance"),
                save_line(plt, figures, "forward_consistency", x, [("forward gate", [float(r["forward_consistency"]) for r in rows]), ("plausible fraction", [float(r["forward_plausible_fraction"]) for r in rows])], "gate/fraction"),
            )
        )
        with np.load(arrays_path, allow_pickle=False) as arrays:
            initial = np.asarray(arrays["initial"])
            final = np.asarray(arrays["final"])
        figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        for column, (label, array) in enumerate((("initial", initial), ("final", final))):
            for expert in (0, 1):
                image = array[0, expert, :3].sum(axis=0)
                axes[column, expert * 2].imshow(image, origin="lower", cmap="magma")
                axes[column, expert * 2].set_title(f"{label} expert {expert + 1} requested")
                companion = array[0, expert, 3:].sum(axis=0)
                axes[column, expert * 2 + 1].imshow(companion, origin="lower", cmap="magma")
                axes[column, expert * 2 + 1].set_title(f"{label} expert {expert + 1} companion")
        for axis in axes.ravel():
            axis.set_axis_off()
        target = figures / "selected_output_grids.png"
        figure.savefig(target, dpi=160)
        plt.close(figure)
        created.append(str(target))
        summary = json.loads(next(path for path in inputs if path.name == "trajectory_summary.json").read_text(encoding="utf-8"))
        figure, axis = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
        axis.axis("off")
        axis.text(0.02, 0.90, f"Outcome: {summary['outcome']}", fontsize=13, weight="bold")
        axis.text(0.02, 0.66, f"Terminal event: {summary['terminal_event']} at step {summary['terminal_step']}")
        axis.text(0.02, 0.46, f"Coverage ever (own/alternate/both): {summary['own_coverage_ever']} / {summary['alternate_coverage_ever']} / {summary['both_mode_coverage_ever']}")
        axis.text(0.02, 0.26, f"D1 distance: {summary['initial_d1_distance']:.6g} -> {summary['minimum_d1_distance']:.6g}")
        target = figures / "outcome_evidence_panel.png"
        figure.savefig(target, dpi=160)
        plt.close(figure)
        created.append(str(target))
    result = {
        "status": "PASS",
        "mode": manifest["mode"],
        "input_manifest": str(args.input_manifest.resolve()),
        "input_count": len(inputs),
        "created_figures": [str(Path(path).relative_to(run)) for path in created],
        "original_scientific_input_count": 0,
        "matplotlib_process_pid": os.getpid(),
    }
    write_json_x(run / ("synthetic_preflight/postprocessing_result.json" if manifest["mode"] == "synthetic" else "diagnostics/postprocessing_result.json"), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
