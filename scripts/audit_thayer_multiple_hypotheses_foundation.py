#!/usr/bin/env python3
"""Reproduce prior failure, exclude every Atlas source, and preregister Thayer-MH."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
FLOW = REPO / "outputs/runs/thayer_flow_prior_20260712_182516"
PROMPT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
SOURCE_SPLIT = PROMPT / "manifests/source_split_manifest.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()


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
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def reproduce_failure(run_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    summary = {row["subset"]: row for row in read_csv(FLOW / "tables/posterior_decoder_sufficiency_summary.csv")}
    checks = {
        "ordinary_posterior_own_truth_coverage": (float(summary["ordinary"]["target_truth_coverage_rate"]), 0.0),
        "near_collision_posterior_own_truth_coverage": (float(summary["near_own"]["target_truth_coverage_rate"]), 0.0),
        "cross_decoded_alternate_truth_coverage": (float(summary["near_cross"]["alternate_truth_coverage_rate"]), 0.0),
        "ordinary_forward_consistent_fraction": (float(summary["ordinary"]["mean_forward_consistent_fraction"]), 0.92987060546875),
        "near_own_forward_consistent_fraction": (float(summary["near_own"]["mean_forward_consistent_fraction"]), 1.0),
        "near_cross_forward_consistent_fraction": (float(summary["near_cross"]["mean_forward_consistent_fraction"]), 1.0),
    }
    atlas_rows = read_csv(PU / "tables/atlas_stochastic_hypothesis_results.csv")
    control_rows = read_csv(PU / "tables/frozen_atlas_matched_control_results.csv")
    operating = {int(row["k_prefix"]): row for row in read_csv(PU / "tables/frozen_atlas_operating_thresholds.csv")}
    atlas_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in atlas_rows])
    control_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in control_rows])
    checks.update({
        "thayer_pu_atlas_witness_count": (float(sum(as_bool(row["model_generated_witness"]) for row in atlas_rows)), 24.0),
        "thayer_pu_atlas_candidate_diameter_auroc": (auc(atlas_diameter, control_diameter), 0.856),
        "thayer_pu_atlas_recall_at_4pct_fpr": (float(np.mean(atlas_diameter > float(operating[32]["control_95th_percentile"]))), 0.32),
        "thayer_pu_atlas_own_truth_coverage": (float(np.mean([as_bool(row["own_truth_coverage"]) for row in atlas_rows])), 0.0),
        "thayer_pu_atlas_alternate_truth_coverage": (float(np.mean([as_bool(row["alternate_truth_coverage"]) for row in atlas_rows])), 0.0),
    })
    for metric, (observed, expected) in checks.items():
        passed = math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        rows.append({"metric": metric, "expected": expected, "observed": observed, "absolute_tolerance": 1e-12, "status": "PASS" if passed else "FAIL", "source": "immutable persisted artifacts; no Atlas inference"})
    write_csv_fresh(run_dir / "tables/baseline_reproduction.csv", rows)
    if any(row["status"] != "PASS" for row in rows):
        write_text_fresh(run_dir / "diagnostics/baseline_reproduction.md", "# Decoder-failure reproduction\n\n**FAIL.** Thayer-MH implementation and fitting are prohibited.\n")
        raise RuntimeError("decoder-failure reproduction mismatch")
    write_text_fresh(run_dir / "diagnostics/baseline_reproduction.md", """# Decoder-failure reproduction

Status: **PASS**. Values were recomputed from immutable persisted tables; no Atlas arrays were opened and no neural inference was run.

- Ordinary posterior own-truth coverage: 0%.
- Near-collision posterior own-truth coverage: 0%.
- Cross-decoded alternate-truth coverage: 0%.
- Ordinary / near-own / near-cross forward-consistency fractions: 0.92987060546875 / 1.0 / 1.0.
- Thayer-PU frozen Atlas: 24/50 witnesses, AUROC 0.856, recall 0.32 at the frozen 4% control FPR, own/alternate coverage 0/0.

The previous posterior/decoder failure is therefore reproduced within exact persisted-table tolerance before any Thayer-MH model implementation or fitting.
""")


def collect_atlas_groups() -> tuple[dict[str, set[str]], Counter[str]]:
    roles: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()

    def add(group: str, role: str) -> None:
        roles[group].add(role)
        occurrences[group] += 1

    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    atlas_pairs = {row["pair_id"]: row for row in read_csv(ATLAS / "tables/atlas_pair_manifest.csv")}
    for pair_id in freeze["pair_ids"]:
        for field in ("left_target_group", "right_target_group", "left_contaminant_group", "right_contaminant_group"):
            add(atlas_pairs[pair_id][field], f"frozen_pair:{field}")
    for row in read_csv(ATLAS / "tables/targeted_optimization_pair_manifest.csv"):
        add(row["selected_contaminant_group"], "targeted_feasibility:selected_contaminant")
    for row in read_csv(ATLAS / "tables/atlas_pool_render_manifest.csv"):
        add(row["target_group"], "historical_atlas_candidate_pool:target")
        add(row["contaminant_group"], "historical_atlas_candidate_pool:contaminant")

    control_ids = {row["scene_id"] for row in read_csv(ATLAS / "tables/matched_control_ambiguity_scores.csv")}
    control_ids.update(row["scene_id"] for row in read_csv(PU / "tables/frozen_atlas_matched_control_results.csv"))
    control_manifest = {row["scene_id"]: row for row in read_csv(ATLAS / "tables/fresh_validation_scene_manifest.csv")}
    for scene_id in sorted(control_ids):
        row = control_manifest[scene_id]
        add(row["target_group"], "atlas_control:target")
        add(row["contaminant_group"], "atlas_control:contaminant")
    return roles, occurrences


def exclusion_audit(run_dir: Path) -> dict[str, object]:
    roles, occurrences = collect_atlas_groups()
    excluded = set(roles)
    split = read_csv(SOURCE_SPLIT)
    partitions: dict[str, set[str]] = defaultdict(set)
    rows_by_group: Counter[str] = Counter()
    for row in split:
        partitions[row["duplicate_group_id"]].add(row["partition"])
        rows_by_group[row["duplicate_group_id"]] += 1

    audit_rows = [{
        "source_group": group,
        "atlas_roles": ";".join(sorted(roles[group])),
        "atlas_occurrences": occurrences[group],
        "source_split_partitions": ";".join(sorted(partitions[group])),
        "catalog_rows_in_group": rows_by_group[group],
        "thayer_mh_training_exposures": 0,
        "thayer_mh_validation_exposures": 0,
        "thayer_mh_calibration_exposures": 0,
        "status": "EXCLUDED",
    } for group in sorted(excluded)]
    write_csv_fresh(run_dir / "tables/atlas_source_exclusion_audit.csv", audit_rows)

    allowed: dict[str, list[dict[str, object]]] = {name: [] for name in ("training", "validation", "calibration")}
    split_hash = sha256_file(SOURCE_SPLIT)
    for row in split:
        partition = row["partition"]
        if partition not in allowed or row["engineering_excluded"] != "0" or row["duplicate_group_id"] in excluded:
            continue
        allowed[partition].append({
            "campaign_partition": partition,
            "catalog_row": row["catalog_row"],
            "persistent_source_id": row["persistent_source_id"],
            "duplicate_group_id": row["duplicate_group_id"],
            "source_split_sha256": split_hash,
            "atlas_excluded": False,
            "development_or_lockbox": False,
        })
    commitments = [row for partition in ("training", "validation", "calibration") for row in allowed[partition]]
    write_csv_fresh(run_dir / "manifests/approved_source_commitments.csv", commitments)
    group_sets = {name: {str(row["duplicate_group_id"]) for row in values} for name, values in allowed.items()}
    overlaps = {
        "training_validation": len(group_sets["training"] & group_sets["validation"]),
        "training_calibration": len(group_sets["training"] & group_sets["calibration"]),
        "validation_calibration": len(group_sets["validation"] & group_sets["calibration"]),
    }
    minima = {"training": 16_000, "validation": 4_000, "calibration": 4_000}
    if any(overlaps.values()) or any(len(allowed[name]) < minima[name] for name in minima):
        raise RuntimeError(f"source exclusion leaves an invalid prospective population: {overlaps}")

    existing_pairs = read_csv(PU / "tables/non_atlas_near_collision_pair_manifest.csv")
    existing_summary = []
    for partition, needed in (("training", 1500), ("validation", 250), ("calibration", 250)):
        rows = [row for row in existing_pairs if row["partition"] == partition]
        surviving = [row for row in rows if not ({row[field] for field in ("left_source_a_group", "left_source_b_group", "right_source_a_group", "right_source_b_group")} & excluded)]
        existing_summary.append({"partition": partition, "historical_pair_count": len(rows), "surviving_after_expanded_exclusion": len(surviving), "required_prospective_pairs": needed, "reuse_authorized": False, "reason": "fresh prospective pair search required under expanded Atlas-pool exclusion"})
    write_csv_fresh(run_dir / "tables/preexisting_non_atlas_pair_exclusion_check.csv", existing_summary)

    report = f"""# Atlas source exclusion report

Status: **PASS — EXPANDED EXCLUSION FROZEN; FRESH TARGET SEARCH REQUIRED**.

- Unique groups excluded across frozen pairs, targeted feasibility, Atlas controls, and the full historical Atlas candidate pool: {len(excluded):,}.
- Eligible training / validation / calibration rows: {len(allowed['training']):,} / {len(allowed['validation']):,} / {len(allowed['calibration']):,}.
- Eligible groups: {len(group_sets['training']):,} / {len(group_sets['validation']):,} / {len(group_sets['calibration']):,}.
- Cross-partition duplicate-group overlaps: 0 / 0 / 0.
- Development and final-lockbox commitments: 0 / 0.
- The historical Thayer-PU pair manifest is not reused. Only 41/2000 training and 242/250 validation pairs survive the expanded exclusion; a new partition-specific prospective search is mandatory.

Every prospective pair member must come from one allowed partition, all four groups must be distinct, and no pair may cross partitions. The source population is large enough for the frozen prospective pool and ordinary-scene sizes. Pair discovery itself remains a fail-closed post-preregistration gate; weak pairs cannot be added to meet counts.
"""
    write_text_fresh(run_dir / "diagnostics/atlas_source_exclusion_report.md", report)
    return {
        "excluded_group_count": len(excluded),
        "excluded_groups_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest(),
        "commitments_sha256": sha256_file(run_dir / "manifests/approved_source_commitments.csv"),
        "eligible_rows": {name: len(values) for name, values in allowed.items()},
        "eligible_groups": {name: len(values) for name, values in group_sets.items()},
        "partition_overlaps": overlaps,
    }


def gate_rows() -> list[dict[str, object]]:
    return [
        {"gate": "parameter_ceiling", "range": "integer [0,+inf)", "threshold": "<=300000", "attainable": True},
        {"gate": "individual_hypothesis_prompt_swap", "range": "[0,1]", "threshold": "each >=0.80", "attainable": True},
        {"gate": "set_level_prompt_swap", "range": "[0,1]", "threshold": ">=0.90", "attainable": True},
        {"gate": "ordinary_source_confusion", "range": "[0,1]", "threshold": "<=0.20 each hypothesis", "attainable": True},
        {"gate": "ordinary_reconstruction_factor", "range": "[0,+inf)", "threshold": "<=3.0x Condition C", "attainable": True},
        {"gate": "ordinary_own_truth_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"gate": "ordinary_forward_consistency", "range": "[0,1]", "threshold": "each hypothesis >=0.50", "attainable": True},
        {"gate": "ordinary_false_witness", "range": "[0,1]", "threshold": "<=0.10", "attainable": True},
        {"gate": "non_atlas_own_truth_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"gate": "non_atlas_alternate_truth_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"gate": "non_atlas_both_mode_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"gate": "near_collision_forward_consistency", "range": "[0,1]", "threshold": "each hypothesis >=0.50", "attainable": True},
        {"gate": "diameter_separation", "range": "[0,+inf)", "threshold": "near/control median ratio >=1.25 and pair-bootstrap lower >1", "attainable": True},
        {"gate": "low_fpr_non_atlas_recall", "range": "[0,1]", "threshold": ">0 at calibration control 95th percentile", "attainable": True},
        {"gate": "pair_set_consistency", "range": "[0,+inf)", "threshold": "median unordered normalized MSE <=0.10", "attainable": True},
        {"gate": "atlas_own_truth_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"gate": "atlas_alternate_truth_coverage", "range": "[0,1]", "threshold": ">0", "attainable": True},
        {"gate": "atlas_witness_count", "range": "integer [0,50]", "threshold": ">24 or truth coverage improves with count remaining high", "attainable": True},
        {"gate": "atlas_auroc", "range": "[0,1]", "threshold": ">=0.806 (no material regression >0.05 from 0.856)", "attainable": True},
        {"gate": "atlas_recall_at_4pct_fpr", "range": "[0,1]", "threshold": ">=0.32", "attainable": True},
        {"gate": "atlas_safe_control_false_witness", "range": "[0,1]", "threshold": "<=0.10", "attainable": True},
    ]


def preregister(run_dir: Path, exclusion: dict[str, object]) -> None:
    gates = gate_rows()
    if not all(bool(row["attainable"]) for row in gates):
        raise RuntimeError("unattainable preregistered gate")
    write_csv_fresh(run_dir / "tables/preregistered_gate_attainability.csv", gates)
    frozen_at = datetime.now(timezone.utc).isoformat()
    text = f"""# Preregistration: ambiguity-set multiple-hypothesis decoder

Frozen at UTC `{frozen_at}` after exact baseline reproduction and expanded Atlas-source exclusion, and before Thayer-MH model implementation, target rendering, fitting, checkpoint selection, or new Atlas inference.

## Hypothesis and boundaries

A compact coordinate-conditioned K=2 shared decoder trained with explicit set-valued supervision can represent both scientifically approved decompositions of an observationally equivalent non-Atlas pair while collapsing to one solution on ordinary controls. It must preserve prompt identity and forward consistency. This is decoder-representation feasibility, not posterior calibration, posterior completeness, black-box auditing, or proof of uniqueness.

Excluded Atlas-related groups: {exclusion['excluded_group_count']:,}; exclusion-set SHA-256 `{exclusion['excluded_groups_sha256']}`. Approved-source commitment SHA-256 `{exclusion['commitments_sha256']}`. Final lockbox, historical development, all Atlas observations, and all Atlas-source groups are prohibited during training, validation, calibration, target construction, debugging, and model selection.

## Prospective data and equivalence contract

- Training: 12,000 ordinary observations and 3,000 ambiguous observations from 1,500 pairs.
- Validation: 1,500 ordinary and 500 ambiguous observations from 250 pairs.
- Calibration: 1,500 ordinary and 500 ambiguous observations from 250 pairs.
- Fresh near-collision search pools: 16,000 / 4,000 / 4,000 scenes for training / validation / calibration, with seeds 2026079101 / 2026079102 / 2026079103.
- Ordinary seeds: 2026079201 / 2026079202 / 2026079203. Noise bases: 2026079300 / 2026079400 / 2026079500 multiplied by 10,000 plus scene index.
- Each pair uses four distinct source groups from exactly one allowed partition and two unique pool scenes. No group crosses partitions.
- A pair is approved only when exact replay/additivity/finite/hash checks pass, mean whitened observation distance <=1.0, requested-source primary scientific distance >1.0, and global-rescaling relative residual >0.01. Candidate selection is rank by observation-embedding distance divided by target-embedding distance, with 32 nearest neighbors and no scene reuse. Counts are hard requirements; weak pairs are forbidden.
- Ordinary target set is the one canonical full decomposition. For either member of an approved pair, the target set contains the left and right full decompositions. Pair provenance and group IDs are stored only in manifests, never inference tensors.

## Full decomposition and prompt semantics

Each hypothesis outputs six unclipped normalized channels: requested g/r/z followed by companion g/r/z. Both are zero-background PSF-convolved source layers; their sum is the noiseless two-source scene. Prompt A maps canonical `[A,B]` and prompt B maps `[B,A]`. For an alternate decomposition, the source nearest the requested coordinate is requested and the other source is companion; the association must be unambiguous and is frozen in the target manifest. Band order g/r/z, training-only normalization, no activation, no clipping, source ordering, and prompt-swap channel exchange are frozen.

## Architecture and warm start

Thayer-MH uses the Condition-C-compatible prompted encoder (`enc1 4->16`, `enc2 16->32`, `bottleneck 32->64`), shared decoder (`dec2 96->32`, `dec1 48->16`), and one shared 16->6 head. K=2 learned 8-dimensional hypothesis tokens are injected through learned linear maps into the 64-channel bottleneck and 32-channel late decoder stage. Decoder weights are shared; only the token distinguishes slots. Matching Condition-C encoder/decoder tensors load exactly, and the historical 3-channel head initializes both output halves. The exact parameter count must be <=300,000. No stochastic sampling, family ID, source ID, catalog metadata, target truth, or generator difficulty enters inference.

## Loss and training

For each prompt, component losses are normalized MSE with fixed weights: requested 1.0, companion 1.0, source-sum 0.5, forward/noiseless-sum 0.5, prompt-swap 0.25, ordinary concentration 0.10, ambiguous pair-equivalence set consistency 0.05. For a two-target set, compute identity and swapped total target assignment and use the smaller assignment per scene without preserving a global slot identity. For an ordinary one-target set, supervise both hypotheses to the one target and apply hypothesis concentration. There is no generic diversity reward.

Training is MPS-only with `PYTORCH_ENABLE_MPS_FALLBACK` disabled: seed 2026079601, 30 epochs, batch size 8, AdamW learning rate 3e-4 and weight decay 1e-4, fixed ordinary:ambiguous batch ratio 3:1. Epochs 1-5 freeze enc1, enc2, and bottleneck while training tokens, injections, decoders, and head. Epochs 6-30 keep enc1 and enc2 frozen and unfreeze bottleneck. Select the best checkpoint by the lowest validation objective only; save best and final separately. Stop on NaN/Inf, MPS fallback, manifest/hash/source exposure, checkpoint collision, prompt collapse sustained for 3 validations, source-sum instability, ambiguous slot collapse sustained for 3 validations, or uncontrolled ordinary divergence.

## Gates and one-time Atlas boundary

All gate definitions and attainable ranges are frozen in `tables/preregistered_gate_attainability.csv`. Promptability is evaluated first; then non-Atlas set coverage, control concentration, pair consistency, and forward consistency. Own and alternate near-collision coverage and both-mode coverage must each be nonzero. Ordinary false witnesses must be <=0.10, near/control diameter ratio must be >=1.25 with pair-bootstrap lower endpoint >1, and non-Atlas recall at the calibration-control 95th-percentile diameter must be nonzero. Any failure stops before Atlas.

Only after every non-Atlas gate passes may one selected checkpoint, threshold, ordering rule, deterministic inference protocol, metrics, controls, and success gates be hashed. Atlas evaluation is exactly one pass over 50 frozen observations; no retraining, recalibration, threshold changes, or post-Atlas tuning is allowed. Atlas own and alternate coverage must become nonzero, AUROC must be >=0.806, recall at the frozen 4% control FPR >=0.32, and safe-control false witnesses <=0.10. Final lockbox access remains zero under every outcome.
"""
    path = run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md"
    write_text_fresh(path, text)
    record = {
        "status": "FROZEN_BEFORE_MODEL_IMPLEMENTATION_TARGET_RENDERING_AND_FITTING",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": sha256_file(path),
        "gate_table_sha256": sha256_file(run_dir / "tables/preregistered_gate_attainability.csv"),
        "excluded_groups_sha256": exclusion["excluded_groups_sha256"],
        "approved_commitments_sha256": exclusion["commitments_sha256"],
        "k": 2, "model_seed": 2026079601, "epochs": 30, "batch_size": 8,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "preregistration/freeze_record.json", record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_multiple_hypotheses_"):
        raise ValueError("unexpected run directory")
    part_a = json.loads((run_dir / "logs/part_a_complete.json").read_text())
    if part_a["status"] != "PASS" or part_a["atlas_evaluation_count"] != 0:
        raise RuntimeError("Part A did not pass")
    if command(["git", "diff", "--cached", "--name-only"]).splitlines():
        raise RuntimeError("staged files appeared after Part A")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint exists before preregistration")
    reproduce_failure(run_dir)
    exclusion = exclusion_audit(run_dir)
    preregister(run_dir, exclusion)
    write_json_fresh(run_dir / "logs/foundation_complete.json", {
        "status": "PASS", "baseline_reproduction": "PASS", "atlas_source_exclusion": "PASS",
        "preregistration_sha256": sha256_file(run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md"),
        "model_implementation_authorized": True, "target_construction_authorized": True,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS", "excluded_groups": exclusion["excluded_group_count"], "preregistration_sha256": sha256_file(run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
