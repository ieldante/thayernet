#!/usr/bin/env python3
"""Compare Atlas ambiguity evidence with deterministic validation controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_deblenders_on_ambiguity_atlas import (  # noqa: E402
    NORMALIZATION,
    infer_model,
    load_models,
    load_thresholds,
    require_mps,
)
from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy  # noqa: E402
from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene  # noqa: E402
from src.competing_hypotheses import empirical_ambiguity_witness, forward_consistency  # noqa: E402

CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
CONTROL_COUNT = 25


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text_fresh(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def auc(positive: list[float], negative: list[float]) -> float:
    comparisons = [1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative]
    return float(np.mean(comparisons))


def higher_quantile(values: list[float], quantile: float) -> float:
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(values, quantile, interpolation="higher"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "tables/ambiguity_evidence_baselines.csv"
    if output.exists():
        raise FileExistsError(output)
    device = require_mps()
    models = load_models(device)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    thresholds = load_thresholds(run_dir)
    sky = json.loads((run_dir / "calibration/forward_consistency_thresholds.json").read_text())["sky_electrons_grz"]
    definitions = read_csv(run_dir / "manifests/fresh_validation_scene_definitions.csv")[:CONTROL_COUNT]
    catalog, _ = load_catsim_catalog(CATALOG)
    controls = []
    observation_records = []
    for definition in definitions:
        spec = SceneSpec(
            scene_id=definition["scene_id"],
            catalog_rows=(int(definition["target_catalog_row"]), int(definition["contaminant_catalog_row"])),
            positions_arcsec=(
                (float(definition["target_x_arcsec"]), float(definition["target_y_arcsec"])),
                (float(definition["contaminant_x_arcsec"]), float(definition["contaminant_y_arcsec"])),
            ),
            source_selection_seed=int(definition["source_selection_seed"]),
            position_seed=int(definition["position_seed"]),
            noise_seed=int(definition["noise_seed"]),
        )
        noiseless = render_fixed_scene(catalog, spec, add_noise="none")
        noisy = render_fixed_scene(catalog, spec, add_noise="all")
        xy = [(float(row["x_peak"]), float(row["y_peak"])) for row in noisy.catalog]
        controls.append({"scene_id": definition["scene_id"], "blend": noisy.blend, "truth": noiseless.isolated})
        for source_index in (0, 1):
            observation_records.append(
                {
                    "scene_id": definition["scene_id"],
                    "source_index": source_index,
                    "blend": noisy.blend,
                    "prompt": gaussian_prompt_numpy(*xy[source_index]),
                }
            )
    blends = np.stack([row["blend"] for row in observation_records]).astype(np.float32)
    prompts = np.stack([row["prompt"] for row in observation_records]).astype(np.float32)
    predictions = {}
    confidences = {}
    for family, model in models.items():
        prediction, confidence, _ = infer_model(family, model, blends, prompts, scales, device)
        predictions[family] = prediction
        confidences[family] = confidence
    control_rows: list[dict[str, object]] = []
    for scene_index, control in enumerate(controls):
        requested = {}
        consistency = {}
        for family in models:
            layers = np.stack([predictions[family][2 * scene_index], predictions[family][2 * scene_index + 1]])
            requested[family] = layers[0]
            consistency[family] = forward_consistency(control["blend"], layers, sky)
        witness = empirical_ambiguity_witness(
            requested,
            consistency,
            thresholds,
            mean_psf_fwhm_pixel=(0.86 + 0.81 + 0.77) / (3 * 0.2),
            artifact_audit_passed=True,
        )
        control_rows.append(
            {
                "evaluation_class": "VALIDATION_CONTROL_NO_CONSTRUCTED_NEAR_COLLISION",
                "scene_id": control["scene_id"],
                "atlas_label": 0,
                "plausible_set_size": len(witness.retained_candidate_ids),
                "diameter_score": witness.primary_diameter,
                "forward_residual_score": min(value.global_chi_square_mean for value in consistency.values()),
                "self_confidence_unsafe_score": -float(confidences["THAYER_SELECT_R1_RECONSTRUCTION_ONLY"][2 * scene_index]),
            }
        )
    write_csv_fresh(run_dir / "tables/matched_control_ambiguity_scores.csv", control_rows)

    atlas_witness = {
        (row["pair_id"], row["side"]): row
        for row in read_csv(run_dir / "tables/model_candidate_witness_inventory.csv")
        if row["regime"] == "noisy_observation"
    }
    atlas_decompositions = [
        row
        for row in read_csv(run_dir / "tables/candidate_decomposition_inventory.csv")
        if row["regime"] == "noisy_observation"
    ]
    atlas_outputs = [
        row
        for row in read_csv(run_dir / "tables/candidate_output_inventory.csv")
        if row["regime"] == "noisy_observation"
        and row["requested_source"] == "True"
        and row["family_id_provenance_only"] == "THAYER_SELECT_R1_RECONSTRUCTION_ONLY"
    ]
    atlas_confidence = {(row["pair_id"], row["side"]): float(row["r1_private_recoverability_diagnostic"]) for row in atlas_outputs}
    atlas_rows = []
    for key, witness in sorted(atlas_witness.items()):
        residuals = [
            float(row["forward_global_chi_square_mean"])
            for row in atlas_decompositions
            if (row["pair_id"], row["side"]) == key
        ]
        diameter = 0.0 if witness["model_candidate_primary_diameter"] == "NOT_APPLICABLE" else float(witness["model_candidate_primary_diameter"])
        atlas_rows.append(
            {
                "evaluation_class": "FROZEN_ATLAS",
                "scene_id": f"{key[0]}:{key[1]}",
                "atlas_label": 1,
                "plausible_set_size": int(witness["plausible_candidate_count"]),
                "diameter_score": diameter,
                "forward_residual_score": min(residuals),
                "self_confidence_unsafe_score": -atlas_confidence[key],
            }
        )
    write_csv_fresh(run_dir / "tables/atlas_ambiguity_scores.csv", atlas_rows)
    metrics = []
    for metric in ("diameter_score", "plausible_set_size", "forward_residual_score", "self_confidence_unsafe_score"):
        positive = [float(row[metric]) for row in atlas_rows]
        negative = [float(row[metric]) for row in control_rows]
        threshold = higher_quantile(negative, 0.95)
        recall = float(np.mean(np.asarray(positive) > threshold))
        false_positive = float(np.mean(np.asarray(negative) > threshold))
        metrics.append(
            {
                "metric": metric,
                "atlas_count": len(positive),
                "control_count": len(negative),
                "auroc": auc(positive, negative),
                "frozen_control_95th_percentile": threshold,
                "atlas_recall_at_frozen_threshold": recall,
                "control_false_positive_rate": false_positive,
            }
        )
    write_csv_fresh(output, metrics)
    by_metric = {row["metric"]: row for row in metrics}
    diameter_pass = bool(
        float(by_metric["diameter_score"]["auroc"]) > float(by_metric["forward_residual_score"]["auroc"])
        and float(by_metric["diameter_score"]["auroc"]) > float(by_metric["self_confidence_unsafe_score"]["auroc"])
        and float(by_metric["diameter_score"]["atlas_recall_at_frozen_threshold"]) >= 0.5
    )
    report = f"""# Ambiguity-evidence baseline comparison

Status: **{'DIAMETER_BASELINE_PASS' if diameter_pass else 'DIAMETER_BASELINE_FAIL'}**.

The frozen 50 Atlas observations were compared with 25 deterministic fresh
validation controls that have no constructed near-collision witness. This does
not prove that controls are unique. The control 95th percentile was frozen
before computing Atlas recall.

- Diameter AUROC: {float(by_metric['diameter_score']['auroc']):.4f}; recall at frozen control threshold: {float(by_metric['diameter_score']['atlas_recall_at_frozen_threshold']):.4f}.
- Forward-residual AUROC: {float(by_metric['forward_residual_score']['auroc']):.4f}.
- R1 self-confidence AUROC: {float(by_metric['self_confidence_unsafe_score']['auroc']):.4f}.

All three candidates share one architecture cluster, so even a positive result
would remain model-specific feasibility evidence rather than cross-deblender
validation.
"""
    write_text_fresh(run_dir / "diagnostics/ambiguity_evidence_baseline_report.md", report)


if __name__ == "__main__":
    main()
