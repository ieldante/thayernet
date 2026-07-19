#!/usr/bin/env python3
"""Add complete partial-correlation and coverage-entry summaries to Thayer-LG."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.audit_thayer_loss_geometry import sha256_file, write_csv_fresh, write_json_fresh


PREDICTORS = ("total_objective", "requested_reconstruction", "companion_reconstruction", "target_source_sum", "ordinary_concentration", "forward", "prompt_swap", "pair_consistency")
OUTCOMES = ("primary_scientific_distance", "image_distance", "flux_distance", "color_distance", "centroid_distance")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def partial_rank(x: np.ndarray, y: np.ndarray, kind: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    design = np.column_stack((np.ones(len(kind)), kind))
    residual_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(residual_x) == 0 or np.std(residual_y) == 0:
        return float("nan")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = __import__("json").loads((run_dir / "preregistration/freeze_record.json").read_text())
    if sha256_file(run_dir / "preregistration/frozen_loss_geometry_audit.md") != freeze["preregistration_sha256"]:
        raise RuntimeError("preregistration changed")
    if not (run_dir / "logs/numerical_audit_complete.json").is_file():
        raise RuntimeError("numerical audit is incomplete")
    tables = run_dir / "tables"
    datasets = {"canonical": read(tables / "objective_ranking.csv"), "objective_paths": read(tables / "objective_path_metrics.csv")}
    output = []
    for dataset, records in datasets.items():
        for scope in ("all", "ordinary", "near_collision"):
            selected = records if scope == "all" else [row for row in records if row["kind"] == scope]
            for predictor in PREDICTORS:
                coverage = np.asarray([
                    float(row["ordinary_both_experts_coverage"] == "True") if row["kind"] == "ordinary" else float(row["both_mode_coverage"] == "True")
                    for row in selected
                ])
                predictor_values = np.asarray([float(row[predictor]) for row in selected])
                lowest = predictor_values <= np.quantile(predictor_values, 0.25)
                coverage_probability = float(np.mean(coverage[lowest])) if np.any(lowest) else float("nan")
                for outcome in OUTCOMES:
                    triples = [(float(row[predictor]), float(row[outcome]), 0.0 if row["kind"] == "ordinary" else 1.0) for row in selected if row[outcome] != "" and math.isfinite(float(row[outcome]))]
                    if len(triples) < 3: continue
                    x, y, kind = (np.asarray(values) for values in zip(*triples))
                    output.append({
                        "dataset": dataset, "scope": scope, "predictor": predictor, "outcome": outcome, "count": len(x),
                        "spearman": float(spearmanr(x, y).statistic), "kendall": float(kendalltau(x, y).statistic),
                        "partial_spearman_controlling_ordinary_ambiguous": partial_rank(x, y, kind) if scope == "all" else "",
                        "coverage_entry_probability_lowest_predictor_quartile": coverage_probability,
                    })
    write_csv_fresh(tables / "loss_science_regression_full.csv", output)
    write_json_fresh(run_dir / "logs/loss_science_regression_supplement_complete.json", {"status": "PASS", "row_count": len(output), "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    print(len(output))


if __name__ == "__main__": main()
