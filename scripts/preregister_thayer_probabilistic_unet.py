#!/usr/bin/env python3
"""Audit Atlas exclusions and freeze the Thayer-PU design before model fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
PROMPT_RUN = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
SOURCE_SPLIT = PROMPT_RUN / "manifests/source_split_manifest.csv"
TRAIN_ORDINARY = 12_000
TRAIN_NEAR_OBSERVATIONS = 4_000
TRAIN_NEAR_PAIRS = 2_000
VALID_ORDINARY = 1_500
VALID_NEAR_OBSERVATIONS = 500
VALID_NEAR_PAIRS = 250
CALIB_ORDINARY = 1_500
CALIB_NEAR_OBSERVATIONS = 500
CALIB_NEAR_PAIRS = 250


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


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


def require_run(path: Path) -> Path:
    run_dir = path.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_probabilistic_unet_"):
        raise ValueError("unexpected run directory")
    if json.loads((run_dir / "logs/part_a_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("Part A did not pass")
    hash_rows = read_csv(run_dir / "tables/canonical_hash_tests.csv")
    if len(hash_rows) != 11 or any(row["status"] != "PASS" for row in hash_rows):
        raise RuntimeError("canonical hash gate did not pass")
    if command(["git", "diff", "--cached", "--name-only"]).splitlines():
        raise RuntimeError("staged files appeared after Part A")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("model checkpoint exists before preregistration")
    return run_dir


def atlas_groups() -> tuple[set[str], dict[str, set[str]], Counter[str]]:
    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    pairs = {row["pair_id"]: row for row in read_csv(ATLAS / "tables/atlas_pair_manifest.csv")}
    targeted = {row["source_pair_id"]: row for row in read_csv(ATLAS / "tables/targeted_optimization_pair_manifest.csv")}
    roles: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    for pair_id in freeze["pair_ids"]:
        pair = pairs[pair_id]
        for field in ("left_target_group", "right_target_group", "left_contaminant_group", "right_contaminant_group"):
            group = pair[field]
            roles[group].add(field)
            occurrences[group] += 1
        group = targeted[pair_id]["selected_contaminant_group"]
        roles[group].add("targeted_feasibility_contaminant")
        occurrences[group] += 1
    return set(roles), roles, occurrences


def source_exclusion_audit(run_dir: Path) -> dict[str, object]:
    excluded, roles, occurrences = atlas_groups()
    split = read_csv(SOURCE_SPLIT)
    partitions_by_group: dict[str, set[str]] = defaultdict(set)
    rows_by_group: Counter[str] = Counter()
    for row in split:
        partitions_by_group[row["duplicate_group_id"]].add(row["partition"])
        rows_by_group[row["duplicate_group_id"]] += 1

    condition_scenes = read_csv(PROMPT_RUN / "manifests/development_scene_definitions.csv")
    historical: Counter[tuple[str, str]] = Counter()
    for row in condition_scenes:
        if row["partition"] in {"training", "validation"}:
            for field in ("source_a_group", "source_b_group"):
                historical[(row[field], row["partition"])] += 1

    audit_rows = []
    for group in sorted(excluded):
        audit_rows.append({
            "source_group": group,
            "atlas_roles": ";".join(sorted(roles[group])),
            "atlas_or_targeted_occurrences": occurrences[group],
            "source_split_partition": ";".join(sorted(partitions_by_group[group])),
            "catalog_rows_in_group": rows_by_group[group],
            "condition_c_training_exposures": historical[(group, "training")],
            "condition_c_validation_exposures": historical[(group, "validation")],
            "thayer_pu_training_exposures": 0,
            "thayer_pu_validation_exposures": 0,
            "thayer_pu_calibration_exposures": 0,
            "thayer_pu_near_collision_exposures": 0,
            "status": "EXCLUDED",
        })
    audit_path = run_dir / "tables/atlas_source_exclusion_audit.csv"
    if audit_path.exists():
        existing = read_csv(audit_path)
        if (
            len(existing) != len(audit_rows)
            or {row["source_group"] for row in existing} != excluded
            or any(row["status"] != "EXCLUDED" for row in existing)
        ):
            raise RuntimeError("existing Atlas exclusion audit is incomplete or inconsistent")
    else:
        write_csv_fresh(audit_path, audit_rows)

    allowed: dict[str, list[dict[str, str]]] = {"training": [], "validation": [], "calibration": []}
    source_split_sha256 = sha256_file(SOURCE_SPLIT)
    for row in split:
        partition = row["partition"]
        if partition not in allowed:
            continue
        if row["engineering_excluded"] != "0" or row["duplicate_group_id"] in excluded:
            continue
        allowed[partition].append({
            "campaign_partition": partition,
            "catalog_row": row["catalog_row"],
            "persistent_source_id": row["persistent_source_id"],
            "duplicate_group_id": row["duplicate_group_id"],
            "source_split_sha256": source_split_sha256,
            "atlas_excluded": False,
            "development_or_lockbox": False,
        })
    commitments = [row for partition in ("training", "validation", "calibration") for row in allowed[partition]]
    write_csv_fresh(run_dir / "manifests/approved_source_commitments.csv", commitments)
    group_sets = {key: {row["duplicate_group_id"] for row in rows} for key, rows in allowed.items()}
    overlaps = {
        "training_validation": len(group_sets["training"] & group_sets["validation"]),
        "training_calibration": len(group_sets["training"] & group_sets["calibration"]),
        "validation_calibration": len(group_sets["validation"] & group_sets["calibration"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"partition source-group overlap: {overlaps}")
    required_rows = {"training": 2, "validation": 2, "calibration": 2}
    if any(len(allowed[key]) < value for key, value in required_rows.items()):
        raise RuntimeError("insufficient approved source population")
    if any(row["duplicate_group_id"] in excluded for row in commitments):
        raise RuntimeError("Atlas group leaked into source commitments")

    historical_exposed = sum(bool(int(row["condition_c_training_exposures"]) or int(row["condition_c_validation_exposures"])) for row in audit_rows)
    report = f"""# Atlas source exclusion report

Status: **PASS — ZERO PROSPECTIVE ATLAS SOURCE EXPOSURE**.

- Frozen Atlas plus targeted-feasibility source groups: {len(excluded)}.
- Those groups with historical Condition-C train/validation exposure: {historical_exposed}.
- Eligible training rows/groups after exclusion: {len(allowed['training']):,} / {len(group_sets['training']):,}.
- Eligible validation rows/groups after exclusion: {len(allowed['validation']):,} / {len(group_sets['validation']):,}.
- Eligible calibration rows/groups after exclusion: {len(allowed['calibration']):,} / {len(group_sets['calibration']):,}.
- Cross-partition duplicate-group overlaps: 0 / 0 / 0.
- Proposed Thayer-PU train/validation/calibration and near-collision exposure: 0 for every excluded group.
- Historical development and final-lockbox scene access: 0 / 0.

Historical Condition-C exposure is inventory context only. All Atlas targets,
companions, and targeted-optimization selected contaminants are excluded from
new ordinary scenes, non-Atlas collision pools, training, validation, and
calibration. The approved commitment file contains only the three permitted
source partitions and explicitly omits development and sealed-lockbox rows.
"""
    write_text_fresh(run_dir / "diagnostics/atlas_source_exclusion_report.md", report)
    return {
        "excluded_group_count": len(excluded),
        "excluded_groups_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest(),
        "condition_c_exposed_excluded_groups": historical_exposed,
        "eligible_rows": {key: len(value) for key, value in allowed.items()},
        "eligible_groups": {key: len(value) for key, value in group_sets.items()},
        "partition_group_overlaps": overlaps,
        "commitments_sha256": sha256_file(run_dir / "manifests/approved_source_commitments.csv"),
    }


def gate_rows() -> list[dict[str, object]]:
    return [
        {"gate": "parameter_ceiling", "metric_range": "[0,+inf)", "threshold": "<=600000", "attainable": True, "reason": "finite compact architecture"},
        {"gate": "prompt_swap_majority", "metric_range": "[0,1]", "threshold": ">=0.80", "attainable": True, "reason": "Condition C historical reference 0.98"},
        {"gate": "individual_prior_identity", "metric_range": "[0,1]", "threshold": ">=0.70", "attainable": True, "reason": "strictly inside probability range"},
        {"gate": "best_of_16_identity", "metric_range": "[0,1]", "threshold": ">=0.90", "attainable": True, "reason": "strictly inside probability range"},
        {"gate": "prompt_output_collapse", "metric_range": "[0,1]", "threshold": "<=0.10", "attainable": True, "reason": "zero collapse is possible"},
        {"gate": "reconstruction_factor", "metric_range": "[0,+inf)", "threshold": "<=3.0x Condition C", "attainable": True, "reason": "warm-started backbone supplies a finite reference"},
        {"gate": "active_latent_dimensions", "metric_range": "integer [0,8]", "threshold": ">=2", "attainable": True, "reason": "two is below latent dimension eight"},
        {"gate": "decoder_latent_sensitivity", "metric_range": "[0,+inf)", "threshold": ">=0.02 normalized truth diameter", "attainable": True, "reason": "latent injection is nonconstant"},
        {"gate": "prior_posterior_best_mse_gap", "metric_range": "[0,+inf)", "threshold": "prior best-of-16 <=2.0x posterior", "attainable": True, "reason": "equality is attainable when q and p match"},
        {"gate": "forward_consistency_rate", "metric_range": "[0,1]", "threshold": ">=0.50", "attainable": True, "reason": "truth decomposition rate is one by construction"},
        {"gate": "plausible_samples_per_scene", "metric_range": "integer [0,16]", "threshold": "median >=4", "attainable": True, "reason": "four is below K=16"},
        {"gate": "control_false_witness_rate", "metric_range": "[0,1]", "threshold": "<=0.10", "attainable": True, "reason": "zero false witnesses is possible"},
        {"gate": "near_collision_diameter_ratio", "metric_range": "[0,+inf)", "threshold": "median ratio >=1.25 and bootstrap lower >1", "attainable": True, "reason": "unbounded positive ratio with nonzero control denominator"},
        {"gate": "atlas_witness_count", "metric_range": "integer [0,50]", "threshold": ">=30 and >19", "attainable": True, "reason": "30 lies inside finite count range"},
        {"gate": "atlas_candidate_diameter_auroc", "metric_range": "[0,1]", "threshold": ">=0.60 and CI lower >0.5", "attainable": True, "reason": "perfect separation gives one"},
        {"gate": "atlas_recall_at_4pct_fpr", "metric_range": "[0,1]", "threshold": ">=0.10", "attainable": True, "reason": "strictly inside probability range"},
        {"gate": "atlas_safe_control_false_witness", "metric_range": "[0,1]", "threshold": "<=0.10", "attainable": True, "reason": "zero is possible"},
        {"gate": "own_truth_coverage", "metric_range": "[0,1]", "threshold": ">=0.70", "attainable": True, "reason": "strictly inside probability range"},
        {"gate": "alternate_truth_coverage", "metric_range": "[0,1]", "threshold": ">=0.30", "attainable": True, "reason": "strictly inside probability range"},
    ]


def preregistration(run_dir: Path, exclusion: dict[str, object]) -> None:
    gates = gate_rows()
    if not all(row["attainable"] for row in gates):
        raise RuntimeError("unattainable metric gate")
    write_csv_fresh(run_dir / "tables/preregistered_gate_attainability.csv", gates)
    frozen_at = datetime.now(timezone.utc).isoformat()
    text = f"""# Preregistration: Thayer-PU prompted probabilistic U-Net

Frozen at UTC `{frozen_at}` before model implementation, data rendering, fitting,
checkpoint selection, or any new Atlas inference.

## Scientific hypothesis and boundaries

A compact coordinate-conditioned conditional latent-variable model warm-started
from the successful Condition-C representation can sample multiple requested-
source decompositions for one blend. Prior samples should concentrate on ordinary
uniquely recoverable controls and expand on independently generated non-Atlas
near-collision observations while retaining prompt identity and exact forward
consistency. If every non-Atlas gate passes, the same frozen sampling protocol
will be evaluated once on Atlas v0.

This is multi-hypothesis feasibility, not posterior certification, formal Bayesian
correctness, a black-box audit, catalog admission, or novelty of VAE deblending.
Posterior samples are training diagnostics only and never inference evidence.

## Source isolation and data sizes

- Excluded Atlas/targeted-feasibility groups: {exclusion['excluded_group_count']};
  exclusion-set SHA-256 `{exclusion['excluded_groups_sha256']}`.
- Approved commitments SHA-256 `{exclusion['commitments_sha256']}`.
- Train: {TRAIN_ORDINARY:,} ordinary observations plus {TRAIN_NEAR_OBSERVATIONS:,}
  observations from {TRAIN_NEAR_PAIRS:,} independent training-only near-collision pairs.
- Validation: {VALID_ORDINARY:,} ordinary plus {VALID_NEAR_OBSERVATIONS:,} observations
  from {VALID_NEAR_PAIRS:,} validation-only pairs.
- Calibration: {CALIB_ORDINARY:,} ordinary plus {CALIB_NEAR_OBSERVATIONS:,} observations
  from {CALIB_NEAR_PAIRS:,} calibration-only pairs.
- Every observation supports both coordinate prompts. Duplicate groups never cross
  train, validation, or calibration. Atlas groups, development, and lockbox are prohibited.

Near-collision pools are generated independently per partition from only approved
groups and fresh seeds. Scenes use two disjoint source groups, randomized midpoint,
separation uniform on [0.6, 3.0] arcsec, fixed g/r/z PSF/noise contracts, no clipping,
and exact replay. Candidate pairs require four mutually disjoint groups, mean
noise-whitened noiseless blend MSE <=1.0, requested-target primary scientific
distance >1.0 under the pre-existing metric, and rejection of trivial global
rescaling. Parameters are scientific one-noise-unit/one-scientific-unit limits,
not selected from Atlas outcomes. Pool search is fixed at 32 neighbors with
deterministic rank `blend_distance / target_primary`; first disjoint valid pairs win.

## Model and warm start

Working name: **Thayer-PU** (Thayer Probabilistic U-Net). Latent dimension is 8;
there is no latent-size sweep. Total parameters must not exceed 600,000.

The deterministic path is the exact 4-channel Condition-C compact U-Net: normalized
g/r/z blend plus unit-peak sigma-2-pixel Gaussian coordinate prompt. All matching
encoder, bottleneck, and decoder tensors are loaded from Condition C and inventoried
by name, shape, and SHA-256. Its earliest prompt-sensitive input block stays frozen
throughout. The output head expands from 3 to 6 channels and is initialized by
copying the Condition-C requested-source head into both halves.

The scene-level prior `p(z|blend)` sees only the three normalized observed blend
channels and is independent of prompt and truth. The training-only posterior
`q(z|blend,source_A,source_B)` sees the blend plus both normalized truth layers in
canonical source-A/source-B manifest order, never prompt order. Prior and posterior
are separate modules and APIs. Both emit diagonal Gaussian mean/log variance.
Latent z is linearly injected only at the 64-channel bottleneck.

The decoder emits six linear, unclipped normalized channels. Channels 0:3 are the
requested source; 3:6 are the companion. Inverse normalization multiplies g/r/z by
the frozen train-only scales. Band order is g,r,z; background is exactly zero;
no positivity projection or clipping is applied. Prompt A means `[A,B]`; prompt B
means `[B,A]`. Matched latent samples must swap the two layers.

## Objective and training

Loss weights are frozen: requested MSE 1.0, companion MSE 1.0, source-sum MSE 0.5,
matched-latent prompt-swap MSE 0.25, best-of-four prior full-decomposition MSE 0.10,
and KL 0.001. Raw diagonal-Gaussian KL uses free bits 0.05 nats per dimension;
beta ramps linearly from 0 at epoch 1 to 1 at epoch 10 and remains one. Diversity
has no independent reward. No adversarial, GAN, or perceptual loss is allowed.

Training uses MPS only, AdamW (learning rate 0.001, weight decay 0.0001), batch size
8, 30 epochs, seed 2026077501, and validation-only minimum total frozen objective
with first exact tie. Best and final checkpoints are separate. Epochs 1-3 train
only prior, posterior, latent injection, decoder adaptation, and decomposition head;
epochs 4-30 additionally unfreeze bottleneck and late decoder blocks. Enc1, enc2,
and the earliest prompt-sensitive enc1 block stay frozen. A smaller batch is allowed
only after a documented pre-step MPS OOM and must be fixed thereafter.

Stop immediately on NaN/Inf, MPS fallback, collision, manifest/hash/exposure failure,
raw total KL above 100 nats for two epochs, fewer than one raw active dimension for
three consecutive epochs after epoch 10, or validation source-sum MSE above ten
times epoch-1 for two epochs.

## Non-Atlas gates

Active means mean raw posterior-vs-prior KL >=0.02 nats for a latent dimension.
Latent-use PASS requires >=2 active dimensions, finite nondegenerate prior standard
deviations (all median std in [0.05,3.0]), decoder output change >=0.02 of truth
diameter between fixed z=-1 and z=+1 probes, and nonzero prior pairwise distance.
Posterior-only diversity cannot pass this gate.

With K=16 prior samples per validation scene and one posterior diagnostic sample,
promptability PASS requires majority-of-K paired prompt-swap >=0.80, individual
prior-sample requested identity >=0.70, best-of-16 identity >=0.90, prompt-output
collapse <=0.10, median signed requested-minus-alternate MSE <0, no inversion in
any band-majority subgroup, and mean whole-image requested MSE <=3.0 times Condition C.

The prior/posterior gap passes only if prior best-of-16 requested MSE <=2.0 times
posterior requested MSE and prior individual identity is no more than 0.15 below
posterior identity. Posterior results are reported separately.

Forward consistency uses the exact frozen observation contract. Calibration truth
decompositions establish a lower sanity bound; the deployed plausibility tolerance
is the larger of their 99th percentile and the 95th percentile of calibration-only
Condition-C full-decomposition scores. No validation or Atlas value enters it.
PASS requires >=0.50 of all prior samples plausible, median >=4 plausible samples
per K=16 scene, and >=75% of scenes with at least one plausible sample.

Scientific clusters use complete linkage with every pair inside a cluster at or
below primary distance 1.0; clusters are formed only after forward filtering.
Matched ordinary controls are selected calibration-blind by nearest separation and
source-flux-ratio from the same validation partition. Control-concentration PASS
requires ordinary false witnesses <=0.10, near-collision median scientific diameter
>=1.25 times matched ordinary (2,000 pair-cluster bootstrap lower ratio >1.0), near
median plausible-set size >=4, and retained near identity >=0.70. Equal diversity
everywhere is uncontrolled stochasticity; negligible diversity everywhere is collapse.

## Frozen one-time Atlas protocol and gates

Atlas access remains blocked unless every preceding gate passes. Then freeze the
selected checkpoint, K=32, seeds 2026077600..2026077631, same scene-level z under
both prompts, calibration tolerance, clustering, scientific metrics, matched frozen
controls, and authoritative 4% control-FPR threshold. Use prior samples only; no
truth-guided resampling, posterior samples, rejection by truth, tuning, or second pass.

Atlas witness PASS requires >=30/50 model-generated witnesses (and >19/50), candidate-
diameter AUROC >=0.60 with 2,000 pair-cluster bootstrap 95% lower endpoint >0.5,
recall at the authoritative 4% control FPR >=0.10, safe-control false witnesses
<=0.10, own-truth coverage >=0.70, alternate-truth coverage >=0.30, and Atlas prior
forward-consistency rate >=0.50. Sample-efficiency uses prefixes K=1,2,4,8,16,32 of
the same frozen sequence. No gate changes follow Atlas results.

Overall SUCCESS requires every non-Atlas pass, Atlas witness pass, zero exposure,
and zero development/lockbox access. PARTIAL SUCCESS requires valid stochastic
hypotheses and improved witnesses but insufficient low-FPR discrimination. Collapse,
promptability failure, uncontrolled diversity, forward inconsistency, or no witness
improvement is FAILURE. Only SUCCESS may authorize a separate auditor campaign.
"""
    prereg_path = run_dir / "preregistration/prompted_probabilistic_unet.md"
    write_text_fresh(prereg_path, text)
    digest = sha256_file(prereg_path)
    write_json_fresh(run_dir / "preregistration/freeze_record.json", {
        "status": "FROZEN_BEFORE_MODEL_IMPLEMENTATION_DATA_RENDERING_OR_FITTING",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": digest,
        "gate_attainability_sha256": sha256_file(run_dir / "tables/preregistered_gate_attainability.csv"),
        "source_exclusion_audit_sha256": sha256_file(run_dir / "tables/atlas_source_exclusion_audit.csv"),
        "source_commitments_sha256": exclusion["commitments_sha256"],
        "canonical_hash_tests_sha256": sha256_file(run_dir / "tables/canonical_hash_tests.csv"),
        "model_checkpoint_count": 0,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    print(digest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = require_run(args.run_dir)
    exclusion = source_exclusion_audit(run_dir)
    preregistration(run_dir, exclusion)


if __name__ == "__main__":
    main()
