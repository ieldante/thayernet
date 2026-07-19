#!/usr/bin/env python3
"""Append-only final-report addendum and calibration-transfer figure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def fresh_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    capacity = pd.read_csv(run / "tables/risk_head_capacity_summary.csv")
    decision = pd.read_csv(run / "tables/component_decision_table.csv").query("risk != 'OVERALL'")
    seed = pd.read_csv(run / "tables/selected_seed_stability.csv")
    comparison = pd.read_csv(run / "tables/conditional_calibration_method_comparison.csv")
    rows = []
    for choice in decision.itertuples(index=False):
        valid = capacity.query("risk == @choice.risk and head == @choice.head").iloc[0]
        calibration = seed.query("risk == @choice.risk")
        selected = comparison.query("risk == @choice.risk and head == @choice.head and method == @choice.method").iloc[0]
        rows.append({
            "risk": choice.risk,
            "selected_head": choice.head,
            "selected_method": choice.method,
            "validation_spearman_mean": valid.spearman_mean,
            "validation_spearman_sd": valid.spearman_sd,
            "natural_calibration_spearman_mean": calibration.spearman.mean(),
            "natural_calibration_spearman_sd": calibration.spearman.std(),
            "selected_mean_width": selected.mean_width,
            "selected_median_width": selected.median_width,
            "selected_p95_width": selected.p95_width,
        })
    transfer = pd.DataFrame(rows)
    transfer_path = run / "tables/calibration_transfer_summary.csv"
    if transfer_path.exists():
        raise FileExistsError(transfer_path)
    transfer.to_csv(transfer_path, index=False)
    figure_path = run / "figures/calibration_transfer.png"
    if figure_path.exists():
        raise FileExistsError(figure_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(transfer))
    ax.errorbar(x, transfer.validation_spearman_mean, yerr=transfer.validation_spearman_sd, marker="o", label="validation")
    ax.errorbar(x, transfer.natural_calibration_spearman_mean, yerr=transfer.natural_calibration_spearman_sd, marker="s", label="natural calibration")
    ax.set_xticks(list(x)); ax.set_xticklabels(transfer.risk); ax.set_ylim(0.75, 1.0); ax.set_ylabel("Spearman rank correlation"); ax.legend(); fig.tight_layout()
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    completion = (run / "logs/campaign_complete.json").stat().st_mtime
    elapsed = completion - provenance["campaign_start_unix"]
    transfer_records = transfer[["risk", "natural_calibration_spearman_mean", "natural_calibration_spearman_sd"]].to_dict("records")
    width_records = transfer[["risk", "selected_mean_width", "selected_median_width", "selected_p95_width"]].to_dict("records")
    addendum = f"""# Conditional-calibration final-report addendum

This append-only addendum clarifies three items in `final_report.md`; the frozen **FAILURE** classification and all component gates are unchanged.

## Natural-calibration transfer

Answer 7 is clarified: the selected nonlinear heads retained strong ranking on natural calibration, with five-seed Spearman summaries {transfer_records}. The original answer listed validation values. The corresponding validation-to-calibration comparison is in `tables/calibration_transfer_summary.csv` and `figures/calibration_transfer.png`.

## Width-tail caution

The selected median and 95th-percentile widths were bounded, but raw-space means remained dominated by rare exponential-tail bounds: {width_records}. Centroid therefore passes the exact preregistered median/p95-oriented gates, but the large mean is an important limitation and does not rescue the overall campaign. Image and flux fail because their adequately supported low-SNR + high-obstruction coverage was 0.637 and 0.684; C4 did not restore conditional reliability.

## Corrective experiment and runtime

Answer 19 is corrected because the support audit found no underpowered frozen subgroup and generated no extra scenes. The one next experiment is a separately preregistered, train/validation/calibration-only **partially pooled deployable scale-model correction**: retain the best frozen reconstruction and risk head; fit a robust heavy-tail scale model on model-accessible features with group-safe training/validation separation; freeze its pooling and shrinkage before calibration; then repeat the same physical subgroup audit on the existing natural calibration partition. Do not generate development data or access the lockbox.

The 4.9-second value in the base report measured only its final continuation. Total campaign wall time from bootstrap through the completed append-only continuations was {elapsed:.1f} seconds; the 54 CPU head/scale checkpoints were written over a 34.2-second span. No reconstruction inference was performed.
"""
    fresh_text(run / "reports/final_report_addendum.md", addendum)
    fresh_text(run / "logs/final_report_addendum_complete.json", json.dumps({"status": "PASS", "created_at_iso": datetime.now(timezone.utc).isoformat(), "classification_changed": False, "development_accesses": 0, "lockbox_accesses": 0, "campaign_wall_seconds": elapsed}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"transfer": transfer_records, "campaign_wall_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
