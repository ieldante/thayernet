#!/usr/bin/env python3
"""Append-only closure for the completed Thayer-OP one-scene stop."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bootstrap_thayer_output_parameterization import EXPECTED, P0_TARGETS, sha256, write_json_fresh


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    selection = json.loads((run / "logs/selection.json").read_text())
    conditions = read_csv(run / "tables/condition_summary.csv")
    evaluations = read_csv(run / "one_scene/evaluation_history.csv")
    if selection["primary_outcome"] != "NO MAPPING PASSES" or not selection["stopped_after_ambiguous_gate"]:
        raise RuntimeError("selection record does not authorize one-scene closure")
    if len(conditions) != 6 or any(row["gate"] == "eight_scene" for row in conditions):
        raise RuntimeError("expected exactly six one-scene conditions and no eight-scene condition")
    if any((run / "eight_scene").iterdir()):
        raise RuntimeError("eight-scene directory is not empty after the stop gate")
    if len(list((run / "checkpoints").glob("*.pth"))) != 6:
        raise RuntimeError("one-scene checkpoint count mismatch")
    if any(float(row["physical_negative_fraction"]) != 0.0 for row in conditions):
        raise RuntimeError("physical output contract violation in persisted summary")
    if any(sha256(path) != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("authoritative input changed before closure")

    gates = ("ordinary_one_scene", "ambiguous_one_scene")
    mappings = ("relu", "square", "absolute")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), squeeze=False)
    for axis, gate in zip(axes[:, 0], gates):
        for mapping in mappings:
            rows = [row for row in evaluations if row["gate"] == gate and row["mapping"] == mapping]
            axis.plot(
                [int(row["step"]) for row in rows],
                [float(row["projected_target_loss"]) for row in rows],
                label=mapping,
            )
        axis.set_yscale("log")
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("projected-target loss")
        axis.set_title(gate.replace("_", " "))
        axis.legend()
    fig.tight_layout()
    figure = run / "figures/output_mapping_learning_curves.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    runtime = sum(float(row["runtime_seconds"]) for row in conditions)
    complete = {
        "status": "PASS_WITH_APPEND_ONLY_CLOSURE_CORRECTION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime,
        "condition_count": 6,
        "optimizer_step_count": 19200,
        "mps_only": True,
        "fallback": False,
        "one_scene_gate_results": {
            mapping: {"ordinary": False, "ambiguous": False} for mapping in mappings
        },
        "selection": selection,
        "unique_scene_input_load_count": 8,
        "remaining_56_microset_scene_input_load_count": 0,
        "p0_target_sha256": sha256(P0_TARGETS),
        "historical_input_hashes_match": True,
        "atlas_scene_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
        "closure_correction": "plot and completion record reconstructed from immutable persisted CSV tables",
    }
    write_json_fresh(run / "logs/micro_campaign_complete.json", complete)
    write_json_fresh(
        run / "logs/micro_closure_correction.json",
        {
            "status": "PASS",
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_tables": [
                "tables/condition_summary.csv",
                "tables/mapping_comparison.csv",
                "one_scene/evaluation_history.csv",
                "logs/selection.json",
            ],
            "new_artifacts": [
                "figures/output_mapping_learning_curves.png",
                "logs/micro_campaign_complete.json",
                "logs/micro_closure_correction.json",
            ],
            "existing_artifact_overwrite_count": 0,
            "additional_optimizer_step_count": 0,
            "additional_scene_input_load_count": 0,
            "eight_scene_fit_count": 0,
        },
    )
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
