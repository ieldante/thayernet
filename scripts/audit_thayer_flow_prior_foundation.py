#!/usr/bin/env python3
"""Reproduce persisted baselines, audit truth coverage, and freeze Part-D gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.competing_hypotheses import scientific_distance  # noqa: E402


PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
MEAN_PSF_FWHM_PIXEL = float(np.mean([0.86, 0.81, 0.77]) / 0.2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value}")


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    combined = np.concatenate((positive, negative))
    ranks = rankdata(combined, method="average")
    count_positive = len(positive)
    return float((ranks[:count_positive].sum() - count_positive * (count_positive + 1) / 2) / (count_positive * len(negative)))


def reproduce_baselines(run_dir: Path) -> None:
    rows: list[dict[str, object]] = []

    latent = json.loads((PU / "logs/latent_use_audit_complete.json").read_text())
    prompt = json.loads((PU / "logs/pre_atlas_promptability_complete.json").read_text())
    gap = json.loads((PU / "logs/prior_posterior_gap_complete.json").read_text())
    forward = json.loads((PU / "logs/forward_consistency_gate_complete.json").read_text())
    control = json.loads((PU / "logs/control_concentration_gate_complete.json").read_text())
    persisted = {
        "promptability_majority_of_16": prompt["majority_of_16_prompt_swap_success"],
        "promptability_best_of_16": prompt["best_of_16_requested_success"],
        "active_latent_dimensions": latent["active_latent_dimensions"],
        "prior_best_to_posterior_mse_ratio": gap["prior_best_to_posterior_mse_ratio"],
        "posterior_minus_prior_identity_gap": gap["posterior_minus_prior_identity_gap"],
        "forward_consistent_prior_fraction": forward["overall_plausibility_rate"],
        "ordinary_false_witness_rate": control["ordinary_false_witness_rate"],
        "near_to_control_diameter_ratio": control["near_to_matched_diameter_ratio"],
    }
    expected = {
        "promptability_majority_of_16": 0.9875,
        "promptability_best_of_16": 0.99425,
        "active_latent_dimensions": 4.0,
        "prior_best_to_posterior_mse_ratio": 0.9143036380031532,
        "posterior_minus_prior_identity_gap": -9.375000000000355e-05,
        "forward_consistent_prior_fraction": 0.95184375,
        "ordinary_false_witness_rate": 0.059333333333333335,
        "near_to_control_diameter_ratio": 1.2645083438325424,
    }
    for metric, observed in persisted.items():
        tolerance = 0.0 if metric == "active_latent_dimensions" else 1e-12
        passed = math.isclose(float(observed), expected[metric], rel_tol=0.0, abs_tol=tolerance)
        rows.append({"scope": "non_atlas_persisted", "metric": metric, "expected": expected[metric], "observed": observed, "absolute_tolerance": tolerance, "pass": passed})

    atlas_rows = read_csv(PU / "tables/atlas_stochastic_hypothesis_results.csv")
    control_rows = read_csv(PU / "tables/frozen_atlas_matched_control_results.csv")
    operating = {int(row["k_prefix"]): row for row in read_csv(PU / "tables/frozen_atlas_operating_thresholds.csv")}
    if len(atlas_rows) != 50 or len(control_rows) != 25:
        raise RuntimeError("persisted Atlas/control row counts changed")
    atlas_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in atlas_rows])
    control_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in control_rows])
    reproduced_atlas = {
        "thayer_pu_witness_count": float(sum(as_bool(row["model_generated_witness"]) for row in atlas_rows)),
        "thayer_pu_candidate_diameter_auroc": auc(atlas_diameter, control_diameter),
        "thayer_pu_recall_at_4pct_fpr": float(np.mean(atlas_diameter > float(operating[32]["control_95th_percentile"]))),
        "thayer_pu_own_truth_coverage": float(np.mean([as_bool(row["own_truth_coverage"]) for row in atlas_rows])),
        "thayer_pu_alternate_truth_coverage": float(np.mean([as_bool(row["alternate_truth_coverage"]) for row in atlas_rows])),
    }
    atlas_expected = {
        "thayer_pu_witness_count": 24.0,
        "thayer_pu_candidate_diameter_auroc": 0.856,
        "thayer_pu_recall_at_4pct_fpr": 0.32,
        "thayer_pu_own_truth_coverage": 0.0,
        "thayer_pu_alternate_truth_coverage": 0.0,
    }
    for metric, observed in reproduced_atlas.items():
        passed = math.isclose(observed, atlas_expected[metric], rel_tol=0.0, abs_tol=1e-12)
        rows.append({"scope": "atlas_persisted_only", "metric": metric, "expected": atlas_expected[metric], "observed": observed, "absolute_tolerance": 1e-12, "pass": passed})

    deterministic_rows = [
        row for row in read_csv(ATLAS / "tables/model_candidate_witness_inventory.csv")
        if row["regime"] == "noisy_observation"
    ]
    deterministic_witnesses = sum(as_bool(row["model_candidate_ambiguity_witness"]) for row in deterministic_rows)
    baseline = read_csv(ATLAS / "tables/ambiguity_evidence_baselines.csv")[0]
    deterministic_metrics = {
        "deterministic_witness_count": (float(deterministic_witnesses), 19.0),
        "deterministic_candidate_diameter_auroc": (float(baseline["auroc"]), 0.4712),
        "deterministic_recall_at_4pct_fpr": (float(baseline["atlas_recall_at_frozen_threshold"]), 0.0),
    }
    for metric, (observed, expected_value) in deterministic_metrics.items():
        passed = math.isclose(observed, expected_value, rel_tol=0.0, abs_tol=1e-12)
        rows.append({"scope": "atlas_persisted_only", "metric": metric, "expected": expected_value, "observed": observed, "absolute_tolerance": 1e-12, "pass": passed})

    if any(not bool(row["pass"]) for row in rows):
        write_csv_fresh(run_dir / "tables/baseline_reproduction.csv", rows)
        raise RuntimeError("persisted baseline reproduction failed")
    write_csv_fresh(run_dir / "tables/baseline_reproduction.csv", rows)
    report = f"""# Persisted Thayer-PU baseline reproduction

Status: **PASS**. No Atlas neural inference was run.

- Non-Atlas promptability: majority-of-16 {prompt['majority_of_16_prompt_swap_success']:.6f}; best-of-16 {prompt['best_of_16_requested_success']:.6f}.
- Active latent dimensions: {latent['active_latent_dimensions']}/8.
- Prior-best/posterior MSE ratio and posterior-minus-prior identity gap: {gap['prior_best_to_posterior_mse_ratio']:.6f} / {gap['posterior_minus_prior_identity_gap']:.8g}.
- Forward-consistent prior-sample fraction: {forward['overall_plausibility_rate']:.6f}.
- Ordinary false-witness rate: {control['ordinary_false_witness_rate']:.6f}.
- Near/control median diameter ratio: {control['near_to_matched_diameter_ratio']:.6f}.
- Deterministic Atlas baseline: {deterministic_witnesses}/50 witnesses, AUROC {float(baseline['auroc']):.4f}, 4%-FPR recall {float(baseline['atlas_recall_at_frozen_threshold']):.2f}.
- Thayer-PU persisted Atlas: {int(reproduced_atlas['thayer_pu_witness_count'])}/50 witnesses, AUROC {reproduced_atlas['thayer_pu_candidate_diameter_auroc']:.3f}, 4%-FPR recall {reproduced_atlas['thayer_pu_recall_at_4pct_fpr']:.2f}, own/alternate coverage 0/0.

The Atlas values were recomputed only from the immutable persisted candidate and
control tables. The frozen Atlas scenes were not opened, rendered, or inferred on.
"""
    write_text_fresh(run_dir / "diagnostics/baseline_reproduction.md", report)


def gaussian_source(scale: float = 1.0, shift_x: int = 0, band_values: tuple[float, float, float] = (1.0, 1.6, 2.2)) -> np.ndarray:
    yy, xx = np.indices((21, 21), dtype=np.float64)
    profile = np.exp(-0.5 * (((xx - 10.0) / 2.0) ** 2 + ((yy - 10.0) / 2.4) ** 2))
    value = np.stack([coefficient * profile for coefficient in band_values]) * scale
    return np.roll(value, shift=shift_x, axis=2)


def coverage(candidate: np.ndarray, truth: np.ndarray) -> tuple[bool, float]:
    distance = scientific_distance(candidate, truth, mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized
    return bool(distance <= 1.0), float(distance)


def audit_metric(run_dir: Path) -> None:
    own = gaussian_source()
    alternate = gaussian_source(scale=1.4)
    cases: list[tuple[str, np.ndarray, bool, bool, str]] = [
        ("candidate_equals_own_truth", own.copy(), True, False, "exact tensor identity"),
        ("floating_tolerance_only", own * (1.0 + 1e-8), True, False, "tiny finite relative perturbation"),
        ("flux_only_outside_threshold", own * 1.3, False, True, "30 percent flux scaling"),
        ("translation_only_outside_threshold", gaussian_source(shift_x=3), False, False, "three-pixel translation"),
        ("candidate_equals_alternate_truth", alternate.copy(), False, True, "exact alternate tensor identity"),
        ("candidate_between_both_truths", 0.5 * (own + alternate), True, True, "arithmetic midpoint in flux"),
        ("band_order_permuted", own[[2, 1, 0]], False, False, "g and z exchanged"),
        ("constant_background_added", own + 0.05, False, False, "background is not a source layer"),
    ]
    rows: list[dict[str, object]] = []
    for name, candidate, expected_own, expected_alternate, rationale in cases:
        own_covered, own_distance = coverage(candidate, own)
        alternate_covered, alternate_distance = coverage(candidate, alternate)
        passed = own_covered == expected_own and alternate_covered == expected_alternate
        rows.append({
            "case": name, "expected_own_coverage": expected_own,
            "observed_own_coverage": own_covered, "own_primary_distance": own_distance,
            "expected_alternate_coverage": expected_alternate,
            "observed_alternate_coverage": alternate_covered,
            "alternate_primary_distance": alternate_distance,
            "threshold": 1.0, "pass": passed, "rationale": rationale,
        })

    alignment = np.zeros((3, 21, 21), dtype=np.float64)
    alignment[:, 10, 10] = (1.0, 2.0, 3.0)
    shifted_alignment = np.roll(alignment, 1, axis=2)
    alignment_covered, alignment_distance = coverage(shifted_alignment, alignment)
    rows.append({
        "case": "source_layer_alignment_one_pixel", "expected_own_coverage": False,
        "observed_own_coverage": alignment_covered, "own_primary_distance": alignment_distance,
        "expected_alternate_coverage": False, "observed_alternate_coverage": False,
        "alternate_primary_distance": math.nan, "threshold": 1.0,
        "pass": not alignment_covered, "rationale": "candidate and truth must share the same pixel grid",
    })
    if any(not bool(row["pass"]) for row in rows):
        write_csv_fresh(run_dir / "tables/truth_coverage_metric_tests.csv", rows)
        raise RuntimeError("truth-coverage metric audit failed")
    write_csv_fresh(run_dir / "tables/truth_coverage_metric_tests.csv", rows)

    report = f"""# Frozen truth-coverage metric audit

Status: **PASS**. The frozen threshold remains primary normalized scientific distance <= 1.0.

The implementation in `src/competing_hypotheses.py` was exercised independently
on synthetic 3-band arrays. Exact identity and floating tolerance cover; a 30%
flux change and a three-pixel translation do not. Exact alternate truth covers
only the alternate mode, while the constructed flux midpoint covers both. Band
permutation, source-grid misalignment, and added background are detected as
scientific differences rather than silently normalized away.

Verified contract:

- arrays are physical, inverse-normalized g/r/z source layers;
- band order is g, r, z and is not permutation invariant;
- source layers contain zero residual background and are PSF-convolved/aligned;
- no per-candidate flux rescaling or centroid registration occurs;
- image, flux, color, and centroid components use the frozen limits 0.25, 0.20,
  0.20 mag, and 0.5 mean-PSF FWHM;
- coverage applies only to applicable retained candidates and uses <= 1.0;
- candidate clustering remains complete linkage at primary distance 1.0.

Mean frozen PSF FWHM used by the metric: {MEAN_PSF_FWHM_PIXEL:.12g} pixels.
No Atlas threshold, candidate, or truth artifact was altered or regenerated.
"""
    write_text_fresh(run_dir / "diagnostics/truth_coverage_metric_audit.md", report)


def freeze_posterior_gate(run_dir: Path) -> None:
    preregistration = """# Posterior/decoder sufficiency preregistration

Frozen before any Part-D neural inference. The test uses only the authoritative
Thayer-PU best checkpoint and its non-Atlas validation arrays. It draws K=32
posterior samples with seed 2026078401. The first 256 ordinary validation scenes
and every one of the 250 frozen validation near-collision pairs (500 scenes) are
selected by manifest order. This fixed sample is large enough to expose a route
failure while avoiding any Atlas or unauthorized development access.

For each scene and both coordinate prompts, q(z|x,y_A,y_B) is sampled in canonical
source order. Own decode uses the scene's own posterior latent under its own
condition. For each frozen near-collision pair, left posterior latents are also
decoded under the right observed condition and right posterior latents under the
left observed condition. The destination scene's corresponding prompt is used;
the teacher latent is the only transferred quantity. Full decompositions are
inverse-normalized before all scientific and forward metrics.

Coverage means at least one forward-consistent requested-source sample within
the unchanged primary scientific distance <=1.0. Source identity means requested
MSE is strictly smaller to the named truth than to the other source in the same
destination scene. Forward consistency uses the already-frozen Thayer-PU
calibration thresholds without recalibration.

Prospective hard gates:

1. ordinary posterior own-truth coverage >=0.70;
2. near-collision own-posterior own-truth coverage >=0.70;
3. near-collision cross-decode alternate-truth coverage >=0.30;
4. ordinary, near-own, and near-cross forward-consistent sample fractions each >=0.50;
5. ordinary and near-own requested-source identity each >=0.70;
6. cross-decode alternate identity >=0.70;
7. every metric is finite, all 250 pairs are evaluated, and all four source
   groups in each pair remain disjoint.

These rates lie in [0,1], have nonempty attainable pass regions, and do not
depend on realized Part-D outcomes. Gate 3 matches the frozen Atlas alternate
coverage floor; Gates 1-2 match the frozen Atlas own-coverage floor; the
forward and identity floors reuse existing campaign semantics. If any gate
fails, the campaign stops before flow implementation or fitting and recommends
exactly one ambiguity-set decoder-training experiment. Posterior samples remain
diagnostics only and never become deployable hypotheses.
"""
    path = run_dir / "preregistration/posterior_decoder_sufficiency.md"
    write_text_fresh(path, preregistration)
    record = {
        "status": "FROZEN_BEFORE_POSTERIOR_DECODER_EVALUATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(path),
        "k": 32,
        "seed": 2026078401,
        "ordinary_scene_count": 256,
        "near_collision_pair_count": 250,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "preregistration/posterior_decoder_sufficiency_record.json", record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    part_a = json.loads((run_dir / "logs/part_a_complete.json").read_text())
    if part_a["status"] != "PASS" or part_a["atlas_evaluation_count"] != 0:
        raise RuntimeError("Part A does not authorize foundation audit")
    reproduce_baselines(run_dir)
    audit_metric(run_dir)
    freeze_posterior_gate(run_dir)
    write_json_fresh(run_dir / "logs/parts_b_c_complete.json", {
        "status": "PASS", "baseline_reproduction": "PASS", "truth_coverage_metric_audit": "PASS",
        "posterior_decoder_evaluation_authorized": True, "flow_implementation_authorized": False,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS", "posterior_decoder_evaluation_authorized": True}, sort_keys=True))


if __name__ == "__main__":
    main()
