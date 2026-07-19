#!/usr/bin/env python3
"""Freeze the Thayer-SA objective and attainable gates before official preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preregistration(frozen_at: str, provenance: dict[str, object]) -> str:
    return f"""# Preregistration: Thayer-SA scientific-alignment micro-overfit

Frozen at UTC `{frozen_at}` after the persisted Thayer-LG diagnosis reproduced and before the official surrogate audit, official output-space preflight, assignment audit, or neural fitting.

## Scope and immutable inputs

Thayer-SA tests only whether the unchanged 165,612-parameter Thayer-ME architecture can memorize the frozen 64-row training microset under a corrected scientific objective. Architecture, shared prompted encoder, independent decoders, expert seeds `2026071201/2026071202`, training seed `2026071250`, microset manifest `{provenance['microset_manifest']['sha256']}`, target sets, prompt mapping, source-layer semantics, normalization, truth-coverage implementation, scientific thresholds, and forward evaluation contract are immutable. Atlas, validation, calibration, development, and lockbox access are prohibited. No full-data training follows a pass in this run.

## Differentiable scientific distance

All quantities are computed after inverse normalization into physical g/r/z detected-electron source layers. The image component is the frozen symmetric L2 image distance divided by `0.25`, using floor `1e-12`. Each g/r/z total-flux relative error uses the frozen symmetric denominator and floor `1e-12`, then divides by `0.20`. Physical flux-derived g-r and r-z magnitudes use positivity floor `1e-12` and divide absolute color error by `0.20 mag`; display RGB is forbidden. The centroid uses nonnegative summed-band weights, floor `1e-12`, and divides displacement by the fixed mean PSF FWHM and `0.50`. Target tensors are detached; gradients through target values are forbidden. Prediction-dependent portions of the frozen symmetric denominators remain differentiable by design.

The seven normalized components are combined by the zero-anchored log-mean-exp smooth maximum `tau * (logsumexp(v/tau) - log(7))`, with temperature `tau=0.005`. Exact truth must have value at most `1e-6` and gradient norm at most `1e-5`.

## Corrected objective and balancing

For each expert/target pair, `C = L_requested + L_companion + L_science`. Requested and companion reconstruction are squared frozen image-normalized distances; `lambda_science=1.0`. Ambiguous rows use the exact hard minimum over identity and swap, averaging across the two experts. Ordinary rows supervise both experts to the one truth and add weight `1.0` times their threshold-normalized requested-source scientific distance. Prompts are averaged, experts are averaged, and the 32 ordinary and 32 ambiguous rows enter with equal per-row weight. There is no factor from expert, prompt, or target-set cardinality.

Forward loss, source-sum loss, prompt-swap loss, pair-equivalence loss, generic diversity, adversarial, perceptual, uncertainty, likelihood, target-aware separation, and scene-recomposition regularizers are absent. Forward, source sum, prompt swap, and pair consistency are evaluation-only.

## Official surrogate and weight gates

Canonical truth, trained Thayer-ME output, collapsed mean, and source-sum-preserving wrong allocation are compared with the exact metric. Required Spearman is at least `0.95`, Kendall at least `0.90`, and threshold-side agreement at least `0.98`; truth must rank best and approved sets must outrank every compromise. Flux, translation, color, and morphology perturbations must activate their intended components. A one-threshold component violation must contribute between `0.90` and `1.05`. At exact truth no term may have a nonzero gradient above `1e-5`; a zero total gradient is stationary and has component share zero. At compromise, raw/weighted terms, band contributions, and ordinary/ambiguous weights are descriptive and cannot change weights.

## Official CPU output-space preflight

Free output tensors are detached from all model parameters and optimized on CPU float32 with Adam, seed `2026071303`, learning rate `1e-4`, no weight decay, 400 updates, and no clipping-based gate relaxation. Initializations are exact truth, persisted trained Thayer-ME output, collapsed truth mean, source-sum-preserving wrong allocation, and deterministic random bounded within the frozen target minimum/maximum. Exact truth must remain within `1e-6` loss, `1e-5` tensor RMS, and full frozen coverage. From trained/compromise/collapsed starts, corrected loss and mean scientific distance must each fall by at least 10%, and ordinary/own/alternate/both-mode coverage must each enter at least `0.90`. Random output must reduce loss and scientific distance by at least 20%. Any failure is `CORRECTED OBJECTIVE STILL MISALIGNED` and stops the campaign before assignment or neural fitting.

## Assignment audit if preflight passes

The hard two-permutation rule remains unchanged. Audit exact-truth and truth-to-compromise paths, deterministic Gaussian perturbations at `1e-7` through `1e-4`, identity/swap margins, assignment flips, and gradient jumps near ties. Exact-truth median absolute margin must exceed `0.10`; perturbations through `1e-5` may flip at most 5% of non-tied exact-truth assignments. Ties at deliberately collapsed outputs are reported but do not alone fail; instability along covered truth neighborhoods does. Failure stops before neural fitting and recommends one separate smooth-assignment campaign.

## Neural micro-overfit if authorized

Use the exact Thayer-ME architecture and Condition-C encoder warm start, independent expert initialization seeds, full phase-2 trainability contract, AdamW, batch size 8, learning rate `1e-3`, weight decay 0, seed `2026071250`, no augmentation, and at most 400 epochs. MPS is mandatory and CPU fallback is forbidden. Checkpoints are selected by lowest corrected microset objective, then truth-coverage hierarchy, without post-hoc gate changes.

Ordinary both-expert own-truth coverage, ambiguous own coverage, alternate coverage, both-mode coverage, expert-1 prompt identity, expert-2 prompt identity, set prompt identity, ordinary forward consistency, ambiguous forward consistency, and source-sum consistency must each be at least `0.90`; median ordinary expert diameter must be at most `1.0`. Exact truth proves every gate attainable: all rate gates can attain `1.0` and diameter can attain `0.0`; on 32-row groups, `0.90` corresponds to at least 29/32. A nonzero substantial improvement with one remaining gate is PARTIAL SUCCESS. Zero truth coverage, divergent ordinary experts, unusable forward behavior, or prompt collapse is FAILURE.

## Decisions

SUCCESS authorizes only a separate preregistered full non-Atlas campaign. PARTIAL recommends exactly one focused correction. FAILURE distinguishes corrected-objective/output optimization from assignment geometry or neural optimization without adding capacity or loosening thresholds. Atlas, development, and lockbox accesses remain zero under every outcome; all historical checkpoints remain immutable.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    provenance = json.loads((run_dir / "logs/input_provenance.json").read_text())
    reproduction = json.loads((run_dir / "logs/loss_geometry_reproduction.json").read_text())
    if reproduction["status"] != "PASS" or not reproduction["reproduced_before_preregistration"]:
        raise RuntimeError("loss-geometry reproduction gate failed")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint exists before preregistration")

    attainable = [
        {"gate": "rate_gates", "domain": "multiples of 1/32", "threshold": ">=0.90", "nearest_passing_value": 29 / 32, "exact_truth_value": 1.0, "attainable": True},
        {"gate": "ordinary_expert_diameter", "domain": "nonnegative real", "threshold": "<=1.0", "nearest_passing_value": 1.0, "exact_truth_value": 0.0, "attainable": True},
        {"gate": "truth_surrogate", "domain": "nonnegative numerical", "threshold": "<=1e-6", "nearest_passing_value": 1e-6, "exact_truth_value": 0.0, "attainable": True},
        {"gate": "forward_evaluation", "domain": "multiples of 1/32", "threshold": ">=0.90", "nearest_passing_value": 29 / 32, "exact_truth_value": 1.0, "attainable": True},
        {"gate": "prompt_identity", "domain": "multiples of 1/32", "threshold": ">=0.90", "nearest_passing_value": 29 / 32, "exact_truth_value": 1.0, "attainable": True},
    ]
    if not all(row["attainable"] for row in attainable):
        raise RuntimeError("unattainable preregistered gate")
    write_csv_fresh(run_dir / "tables/preregistered_gate_attainability.csv", attainable)
    frozen_at = datetime.now(timezone.utc).isoformat()
    prereg_path = run_dir / "preregistration/scientific_alignment_micro_overfit.md"
    write_text_fresh(prereg_path, preregistration(frozen_at, provenance))
    record = {
        "status": "FROZEN_BEFORE_OFFICIAL_PREFLIGHT_OR_NEURAL_FIT",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": sha256_file(prereg_path),
        "input_provenance_sha256": sha256_file(run_dir / "logs/input_provenance.json"),
        "loss_geometry_reproduction_sha256": sha256_file(run_dir / "tables/loss_geometry_reproduction.csv"),
        "scientific_alignment_implementation_sha256": sha256_file(REPO / "src/scientific_alignment.py"),
        "gate_attainability_sha256": sha256_file(run_dir / "tables/preregistered_gate_attainability.csv"),
        "neural_optimizer_step_count": 0,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "preregistration/freeze_record.json", record)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
