#!/usr/bin/env python3
"""Create the append-only master run for competing-hypothesis recoverability.

This bootstrap performs provenance and preregistration only. It does not open,
render, evaluate, or copy historical development or sealed-lockbox scenes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
SOURCE_SPLIT = FOUNDATION / "manifests/btk_engineering_source_groups.csv"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
EXPECTED_SPLIT_HASH = "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27"
EXPECTED_CATALOG_HASH = "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46"
CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
R0 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518/checkpoints/r0_best.pth"
R1 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518/checkpoints/r1_best.pth"
EXPECTED_CANDIDATES = {
    CONDITION_C: "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
    R0: "e8007205452a77df084caab309fc6c91d23898bd0cbd1f58f7ff6de911b30a6a",
    R1: "6637c10fd940b7a853a9e2abd1aef2c371988f31f264c1bf433ec3b161a51750",
}
SUBDIRECTORIES = (
    "diagnostics",
    "tables",
    "figures",
    "logs",
    "reports",
    "preregistration",
    "manifests",
    "candidate_outputs",
    "decompositions",
    "atlas",
    "embeddings",
    "features",
    "models",
    "calibration",
    "rotations",
    "example_grids",
    "paper_figures",
)
AUTHORITATIVE_RUNS = (
    "thayer_select_btk_foundation_20260711_152613",
    "thayer_select_prompt_ablation_20260711_164329",
    "thayer_select_recoverability_20260711_191518",
    "thayer_select_recoverability_seed_replication_20260711_203115",
    "thayer_select_frozen_head_ablation_20260711_220756",
    "thayer_select_hierarchical_feasibility_20260712_010729",
    "thayer_select_scale_correction_20260712_024957",
    "thayer_select_shape_constrained_quantile_20260712_033406",
    "thayer_select_conditional_calibration_20260712_021556",
    "thayer_select_observability_distillation_20260712_035843",
    "thayer_select_psf_conditioning_20260712_043442",
    "thayer_competing_hypotheses_20260712_131111",
)

HISTORICAL_CHECKPOINT_BASELINE = (
    REPO
    / "outputs/runs/thayer_competing_hypotheses_20260712_131111"
    / "tables/checkpoint_inventory_after.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.rstrip()


def write_text_fresh(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_fraction(key: str) -> float:
    value = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return value / float(16**16)


def package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "scipy",
        "pandas",
        "astropy",
        "torch",
        "blending-toolkit",
        "galsim",
        "sep",
        "scikit-learn",
        "scarlet",
        "scarlet2",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def inventory_checkpoints(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if run_dir in path.parents:
            continue
        stat = path.stat()
        digest = sha256_file(path)
        expected = EXPECTED_CANDIDATES.get(path)
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "sha256": digest,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "candidate_checkpoint": path in EXPECTED_CANDIDATES or path == R1,
                "verified_expected_hash": expected is not None and digest == expected,
            }
        )
        if expected is not None and digest != expected:
            raise RuntimeError(f"Altered candidate checkpoint: {path}")
    return rows


def verify_historical_checkpoint_baseline(rows: list[dict[str, object]]) -> None:
    if not HISTORICAL_CHECKPOINT_BASELINE.exists():
        raise RuntimeError("Historical checkpoint baseline is unavailable")
    with HISTORICAL_CHECKPOINT_BASELINE.open(newline="", encoding="utf-8") as handle:
        baseline = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
    current = {str(row["path"]): str(row["sha256"]) for row in rows}
    missing = sorted(set(baseline).difference(current))
    altered = sorted(path for path, digest in baseline.items() if current.get(path) != digest)
    if missing or altered:
        raise RuntimeError(f"Historical checkpoint integrity failure: missing={missing}, altered={altered}")


def prior_anchor_hashes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_name in AUTHORITATIVE_RUNS:
        root = REPO / "outputs/runs" / run_name
        anchors = sorted(root.glob("**/campaign_file_hashes.csv"))
        anchors += sorted(root.glob("**/final_correctness_audit*.json"))
        anchors += sorted(root.glob("reports/final_report.md"))
        if not anchors:
            rows.append({"run": run_name, "anchor": "MISSING", "sha256": "MISSING"})
            continue
        for path in dict.fromkeys(anchors):
            rows.append(
                {
                    "run": run_name,
                    "anchor": str(path.relative_to(REPO)),
                    "sha256": sha256_file(path),
                }
            )
    return rows


def source_commitments(run_dir: Path) -> dict[str, int]:
    rows: list[dict[str, object]] = []
    original_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    with SOURCE_SPLIT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source in reader:
            partition = source["partition"]
            original_counts[partition] += 1
            if partition == "training":
                role = (
                    "audit_evaluation"
                    if stable_fraction("competing-hypotheses-audit-v1|" + source["duplicate_group_id"]) < 0.10
                    else "training"
                )
            elif partition in {"validation", "calibration"}:
                role = partition
            else:
                continue
            role_counts[role] += 1
            rows.append(
                {
                    "catalog_row": source["catalog_row"],
                    "galtileid": source["galtileid"],
                    "persistent_source_id": source["persistent_source_id"],
                    "duplicate_group_id": source["duplicate_group_id"],
                    "campaign_role": role,
                    "source_split_sha256": EXPECTED_SPLIT_HASH,
                    "allocation_rule": "existing validation/calibration retained; deterministic 10% group-hash holdout from existing training for feasibility audit",
                }
            )
    fields = [
        "catalog_row",
        "galtileid",
        "persistent_source_id",
        "duplicate_group_id",
        "campaign_role",
        "source_split_sha256",
        "allocation_rule",
    ]
    write_csv_fresh(run_dir / "manifests/campaign_source_partition_commitments.csv", rows, fields)
    payload = {
        "approved_campaign_role_counts": dict(sorted(role_counts.items())),
        "original_partition_counts": dict(sorted(original_counts.items())),
        "development_rows_committed": 0,
        "sealed_lockbox_rows_committed": 0,
        "development_and_lockbox_policy": "labels counted for exclusion audit only; identities omitted from campaign commitments; no scenes opened, sampled, or rendered",
        "source_split_sha256": EXPECTED_SPLIT_HASH,
    }
    write_json_fresh(run_dir / "manifests/source_partition_summary.json", payload)
    return dict(role_counts)


def family_inventory(run_dir: Path) -> None:
    normalization = "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
    rows = [
        {
            "family_id": "THAYER_SELECT_CONDITION_C",
            "family_cluster": "THAYER_COMPACT_PROMPTED_UNET",
            "status": "COMPATIBLE_AUXILIARY",
            "architecture_algorithm": "compact prompted U-Net",
            "parameter_count": 119091,
            "training_target": "requested noiseless source; whole-image normalized MSE",
            "prompt_support": "Gaussian coordinate prompt",
            "input_normalization": normalization,
            "output_units": "detected electrons per pixel after frozen inverse scale",
            "source_count_assumptions": "queried one source at a time",
            "checkpoint_or_config_sha256": sha256_file(CONDITION_C),
            "reconstruct_every_requested_source": True,
            "full_scene_decomposition": "by querying every declared source; residual background fixed to zero",
            "deterministic_replay": "expected in eval/no_grad; prospective replay test required",
            "primary_distinct_family": True,
            "reason": "Frozen compatible candidate; family cluster shared with R0/R1.",
        },
        {
            "family_id": "THAYER_SELECT_R0",
            "family_cluster": "THAYER_COMPACT_PROMPTED_UNET",
            "status": "COMPATIBLE_AUXILIARY",
            "architecture_algorithm": "same compact prompted U-Net as Condition C",
            "parameter_count": 119091,
            "training_target": "masked requested-source reconstruction across valid/null/ambiguous query mixture",
            "prompt_support": "Gaussian coordinate prompt",
            "input_normalization": normalization,
            "output_units": "detected electrons per pixel after frozen inverse scale",
            "source_count_assumptions": "queried one source at a time",
            "checkpoint_or_config_sha256": sha256_file(R0),
            "reconstruct_every_requested_source": True,
            "full_scene_decomposition": "by querying every declared source; residual background fixed to zero",
            "deterministic_replay": "expected in eval/no_grad; prospective replay test required",
            "primary_distinct_family": False,
            "reason": "Separately trained control, but not a distinct architecture family.",
        },
        {
            "family_id": "THAYER_SELECT_R1_RECONSTRUCTION_ONLY",
            "family_cluster": "THAYER_COMPACT_PROMPTED_UNET",
            "status": "AUXILIARY_PENDING_REPLAY",
            "architecture_algorithm": "compact prompted U-Net plus private heads; only reconstruction exported",
            "parameter_count": 123368,
            "training_target": "requested source plus unstable uncertainty/actionability heads",
            "prompt_support": "Gaussian coordinate prompt",
            "input_normalization": normalization,
            "output_units": "detected electrons per pixel after frozen inverse scale",
            "source_count_assumptions": "queried one source at a time",
            "checkpoint_or_config_sha256": sha256_file(R1),
            "reconstruct_every_requested_source": True,
            "full_scene_decomposition": "by querying every declared source; private heads excluded",
            "deterministic_replay": "prospective replay test required",
            "primary_distinct_family": False,
            "reason": "Private confidence/latent outputs prohibited; same primary family cluster.",
        },
        {
            "family_id": "SEP_CLASSICAL_CONTROL",
            "family_cluster": "SEP_SEGMENTATION",
            "status": "INCOMPATIBLE_NO_VALIDATED_ADAPTER",
            "architecture_algorithm": "SEP detection/segmentation",
            "parameter_count": 0,
            "training_target": "none",
            "prompt_support": "not yet validated",
            "input_normalization": "observed electrons with known sky variance",
            "output_units": "would be detected electrons per pixel",
            "source_count_assumptions": "detections may merge or miss requested source",
            "checkpoint_or_config_sha256": "sep-" + package_versions()["sep"],
            "reconstruct_every_requested_source": False,
            "full_scene_decomposition": "not yet validated",
            "deterministic_replay": "expected",
            "primary_distinct_family": False,
            "reason": "Installed, but no truth-free prompt-to-segment/source-layer adapter is validated.",
        },
        {
            "family_id": "LEGACY_DIRECT_AND_RESIDUAL_UNETS",
            "family_cluster": "LEGACY_GALAXY10_RGB_UNETS",
            "status": "INCOMPATIBLE_INPUT_CONTRACT",
            "architecture_algorithm": "direct/residual compact U-Nets and ResUNet",
            "parameter_count": "various",
            "training_target": "Galaxy10 RGB target or residual",
            "prompt_support": "none",
            "input_normalization": "legacy RGB contract",
            "output_units": "legacy normalized RGB, not BTK electrons",
            "source_count_assumptions": "fixed target convention",
            "checkpoint_or_config_sha256": "see checkpoint_inventory_before.csv",
            "reconstruct_every_requested_source": False,
            "full_scene_decomposition": False,
            "deterministic_replay": "not relevant until compatible",
            "primary_distinct_family": False,
            "reason": "No hidden-truth-free map into the frozen BTK g/r/z source-layer contract.",
        },
        {
            "family_id": "SCARLET_OR_SCARLET2",
            "family_cluster": "SCARLET",
            "status": "NOT_INSTALLED",
            "architecture_algorithm": "scarlet/scarlet2",
            "parameter_count": 0,
            "training_target": "optimization",
            "prompt_support": "unknown",
            "input_normalization": "not evaluated",
            "output_units": "not evaluated",
            "source_count_assumptions": "not evaluated",
            "checkpoint_or_config_sha256": "NOT_INSTALLED",
            "reconstruct_every_requested_source": False,
            "full_scene_decomposition": False,
            "deterministic_replay": "not evaluated",
            "primary_distinct_family": False,
            "reason": "Dependency absent; campaign does not depend on installation.",
        },
        {
            "family_id": "PROSPECTIVE_PROMPTED_RESUNET",
            "family_cluster": "PROMPTED_RESUNET",
            "status": "RECOMMENDED_PROSPECTIVE_ADDITION",
            "architecture_algorithm": "prompted compact residual-block U-Net",
            "parameter_count": "not frozen",
            "training_target": "requested source reconstruction",
            "prompt_support": "required",
            "input_normalization": normalization,
            "output_units": "must be detected electrons after frozen inverse scale",
            "source_count_assumptions": "must query every requested source",
            "checkpoint_or_config_sha256": "NOT_TRAINED",
            "reconstruct_every_requested_source": False,
            "full_scene_decomposition": False,
            "deterministic_replay": "required before admission",
            "primary_distinct_family": False,
            "reason": "Recommended single addition before any cross-family claim; no architecture search.",
        },
    ]
    fields = list(rows[0])
    write_csv_fresh(run_dir / "tables/deblender_family_inventory.csv", rows, fields)
    report = """# Deblender compatibility report

Status: **CROSS_FAMILY_BLOCKED**.

The frozen BTK contract is three PSF-convolved g/r/z source layers in detected
electrons per pixel, queried by the same Gaussian coordinate prompt and summed
without clipping or duplicated background. Condition C, R0, and the
reconstruction component of R1 can be mapped into that contract. They share
one compact prompted-U-Net family cluster, so they are auxiliary candidates,
not three meaningfully distinct primary families.

SEP 1.4.1 is installed, but a truth-free prompt-to-segment adapter that
reconstructs every requested source and a complete decomposition has not been
validated. Legacy RGB U-Nets are incompatible in prompt semantics, units, and
source contract. scarlet/scarlet2 are absent.

Therefore the campaign may continue with controlled Atlas construction,
forward-consistency checks, and finite-candidate ambiguity witnesses. It may
not claim leave-one-deblender-family-out or model-agnostic auditing. The one
recommended prospective addition is a compact prompted ResUNet trained once
under the frozen BTK normalization and output contract; this is not permission
for a broad architecture search.
"""
    write_text_fresh(run_dir / "diagnostics/deblender_compatibility_report.md", report)


def preregistration_text(role_counts: dict[str, int], frozen_at: str) -> str:
    return f"""# Competing-Hypothesis Recoverability preregistration

Status: **FROZEN_BEFORE_NEW_CANDIDATE_INFERENCE_OR_TRAINING**  
Frozen at: `{frozen_at}`  
Working project title: **Don't Even Try**

## Scope decision and family definitions

Only one meaningfully distinct compatible family cluster is currently
available: `THAYER_COMPACT_PROMPTED_UNET`, represented by frozen Condition C,
R0, and reconstruction-only R1 controls. SEP has no validated source-layer
adapter, legacy RGB models have incompatible contracts, and scarlet is absent.
Consequently, Atlas and finite-candidate ambiguity feasibility are active;
leave-one-family-out training, held-out-family evaluation, and model-agnostic
claims are blocked until at least three genuinely distinct compatible families
exist. Absence of a witness will never be treated as proof of uniqueness.

## Source commitments and scene counts

The immutable parent split is SHA-256 `{EXPECTED_SPLIT_HASH}`. Historical
`development_test` and `sealed_lockbox` identities are omitted from campaign
commitments and their scenes are never opened, sampled, rendered, normalized,
or inspected. Existing training groups are deterministically divided by group
hash into campaign training and a prospective 10% feasibility-audit holdout;
existing validation and calibration groups retain their roles. Approved row
counts are `{json.dumps(role_counts, sort_keys=True)}`.

Frozen requested scene counts are 30,000 noiseless TRAIN/SEARCH scenes, 2,000
fresh validation scenes, and 3,000 fresh calibration scenes. If implementation
or resource validation cannot support these counts, the run stops before
claims; it does not silently substitute a smaller analysis. Atlas v0 uses
two-source scenes only. Targets and contaminants always come from the same
campaign role and duplicate group is the atomic isolation unit.

## Candidate-output contract

Every candidate is a float source-layer tensor `(K,3,60,60)` with g/r/z band
order, 0.2 arcsec pixels, PSF-convolved noiseless detected electrons per pixel,
no clipping, and no sky/background duplicated into any layer. Its full
decomposition is the sum of its K layers plus an explicitly zero background.
The requested layer is selected only by the declared prompt-to-source mapping.
Family/checkpoint/path metadata is provenance only and is prohibited from any
auditor tensor. Each output stores scene ID, candidate ID, requested layer,
decomposition hash, measurements, runtime, finite status, and configuration
hash. Expected-deterministic models must replay byte-identically or within
`1e-6` maximum absolute float32 error.

## Frozen forward model and measurement distance

The forward model is BTK 1.0.9 / GalSim 2.8.4 LSST g/r/z rendering as frozen in
`src/btk_scene.py`: source layers are already PSF-convolved, so recomposition is
their unclipped sum. Observation noise is source Poisson plus one zero-mean sky
Poisson realization. Per-pixel variance for candidate consistency is
`max(recomposed_noiseless + sky_electrons_per_pixel, 1.0)` separately by band,
where the exact surveycodex mean sky level is stored in scene metadata.
Residual is `observed - recomposed`; whitened residual is residual divided by
the square root of that variance. The primary consistency score is the mean
squared whitened residual. Per-band means, 8-neighbor residual correlation,
and relative total-flux residual are mandatory diagnostics. Truth is never an
input to this score.

Plausibility thresholds are fit without candidate outcome errors: use the
calibration distribution of the known-truth full decomposition under the same
observation/noise contract. Freeze the finite-sample conservative 99th
percentile of the global score, 99.5th percentile per band, and 99th percentile
of absolute relative flux residual. A candidate passes only if all applicable
limits pass. The threshold procedure, not its future numeric values, is frozen
here; calibration may not use audit-evaluation families or scenes.

## Scientific distances and empirical witness

For requested layers A and B, define image distance as
`||A-B||_2 / (0.5*(||A||_2+||B||_2)+training_flux_floor)`. Per-band flux
distance is absolute flux difference divided by the absolute mean flux plus a
training-only floor. Colors are AB-like `-2.5 log10(F1/F2)` and are not
applicable for non-positive flux. Centroids use nonnegative band-summed source
weights after subtracting only the training-frozen zero floor; nonpositive
total flux is not applicable. Shape distance is diagnostic until its validity
gate passes.

Frozen component limits are image distance 0.25, any-band relative flux 0.20,
either color 0.20 mag, or centroid distance 0.5 mean-PSF FWHM. The primary
diameter is the maximum applicable component divided by its limit. An empirical
ambiguity witness requires at least two forward-consistent decompositions with
primary diameter greater than 1.0 and a passed unit/clipping/translation/
serialization/background artifact audit. It certifies ambiguity only within
the finite candidate family and frozen forward model. No witness never
certifies uniqueness.

## Ambiguity Atlas

Route 1 generates exactly 30,000 noiseless training scenes, embeds raw
noise-whitened pixels (PCA only if fit on training), and rejects same-group
pairs and global rescalings. Route 2 prospectively performs bounded
counterfactual optimization over contaminant choice from a finite approved
pool, position, flux scale, and orientation using exact BTK/GalSim rendering.
Both routes preserve seeds, bounds, replay hashes, and route-specific results.

A valid pair has different requested-source duplicate groups, exact replay,
mean squared whitened blend difference at most 0.25, primary truth diameter
greater than 1.0, and passed numerical plus visual artifact audit. The initial
Atlas freezes only with at least 25 genuine pairs. At least one frozen
deblender must then show either confidence inversion, essentially identical
outputs on divergent truths, or a scientifically unsafe output on the set.

## Failure labels and black-box inputs

Labels retain positive/negative/not-applicable semantics for QUERY_NULL,
QUERY_AMBIGUOUS, SOURCE_CONFUSION, CATASTROPHIC_IMAGE, CATASTROPHIC_FLUX,
CATASTROPHIC_CENTROID, COLOR_UNSAFE, SHAPE_UNSAFE, ATLAS_NON_IDENTIFIABLE, and
SAFE_CANDIDATE. Truth is allowed only to form these labels and evaluation
metrics.

If family compatibility is later reopened prospectively, deployable auditor
inputs are limited to observed blend, coordinate prompt, candidate requested
layer, candidate full decomposition, blend-minus-recomposition residual,
forward score, candidate measurements, plausible-set diameter, and legitimate
observational metadata. Target truth, family/checkpoint/path/architecture
identity, private activations, gradients, training loss, true errors, source
IDs, true SNR/obstruction, and generator variables are forbidden. Frozen
ablations A0--A5 follow the campaign brief. A compact two-stream CNN with
masked failure-specific heads and five seeds is the only allowed primary
auditor; no broad search is allowed.

## Calibration, coverage, intervals, and success gates

If cross-family work becomes attainable in a future preregistered extension,
each family is excluded from training, model selection, and calibration;
thresholds freeze on seen-family validation/calibration and the held-out family
is evaluated once. Accepted coverages are 95, 90, 80, 70, and 50%. Confidence
intervals are source-duplicate-group cluster bootstraps with 2,000 resamples,
and families are macro-averaged. Random rejection and oracle ranking bounds are
reported.

Atlas feasibility passes only with at least 25 valid replayable pairs, all
distance/divergence/artifact gates, and at least one deblender failure. Witness
feasibility passes only if diameter beats both self-confidence and forward
residual in training/validation selection and calibration-frozen evaluation,
is stable across five seeds where a stochastic component exists, and has
useful recall at a frozen 5% false-positive rate. Cross-family audit is
currently mathematically unattainable because fewer than three distinct
families exist and is therefore not an active success gate. Its future 80%
coverage relative false-safe reduction must be selected from training-only
prevalence before held-out-family evaluation and may never be changed post hoc.

## Critical ablations and correctness stops

Required future ablations are candidate shuffling, removal of forward residual,
candidate, blend, prompt, or diameter, same-family seeds only, removal of
distinct families, family-ID leakage, normalization/border probes, held-out
failure severity, and separate Atlas evaluation. Any target leakage, family ID
in tensors, group overlap, calibration reuse, candidate/prompt misalignment,
forward/noise formula failure, post-evaluation tuning, historical development
access, lockbox access, historical-checkpoint mutation, staged-index mutation,
or path collision is a fail-closed stop.

## Attainability audit

- Source counts exceed the requested scene counts without replacement at the
  duplicate-group level; exact scene reuse is not required.
- The forward score has finite variance floor and defined calibration
  quantiles.
- The witness thresholds can be passed or failed by finite arrays and do not
  require inaccessible metadata.
- The Atlas minimum is finite and both construction routes use approved
  training/validation sources only.
- Atlas and witness gates are independently attainable with the current single
  family cluster plus optimized decompositions.
- Cross-family transfer is explicitly blocked, not assigned an impossible
  success requirement.
- No final-development, lockbox, survey, uniqueness, or model-agnostic claim is
  authorized by this preregistration.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", help="Local YYYYMMDD_HHMMSS; defaults to current local time")
    parser.add_argument(
        "--campaign-prefix",
        choices=("thayer_competing_hypotheses", "thayer_ambiguity_atlas_v0"),
        default="thayer_competing_hypotheses",
    )
    args = parser.parse_args()
    timestamp = args.timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / "outputs/runs" / f"{args.campaign_prefix}_{timestamp}"
    if run_dir.exists():
        raise FileExistsError(f"Output-path collision: {run_dir}")
    staged = command("git", "diff", "--cached", "--name-only")
    if staged:
        raise RuntimeError(f"Unexpected staged changes:\n{staged}")
    split_hash = sha256_file(SOURCE_SPLIT)
    catalog_hash = sha256_file(CATALOG)
    if split_hash != EXPECTED_SPLIT_HASH:
        raise RuntimeError("Altered source split")
    if catalog_hash != EXPECTED_CATALOG_HASH:
        raise RuntimeError("Altered source catalog")
    run_dir.mkdir(parents=True)
    for subdirectory in SUBDIRECTORIES:
        (run_dir / subdirectory).mkdir()

    started = datetime.now(timezone.utc).isoformat()
    checkpoints = inventory_checkpoints(run_dir)
    verify_historical_checkpoint_baseline(checkpoints)
    checkpoint_fields = ["path", "sha256", "bytes", "mtime_ns", "candidate_checkpoint", "verified_expected_hash"]
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", checkpoints, checkpoint_fields)
    prior = prior_anchor_hashes()
    write_csv_fresh(run_dir / "tables/authoritative_prior_run_hashes.csv", prior, ["run", "anchor", "sha256"])
    roles = source_commitments(run_dir)
    family_inventory(run_dir)

    usage = shutil.disk_usage(REPO)
    try:
        import torch

        mps = {"built": torch.backends.mps.is_built(), "available": torch.backends.mps.is_available()}
    except Exception as error:  # pragma: no cover - diagnostic only
        mps = {"built": False, "available": False, "error": repr(error)}
    git_status = command("git", "status", "--porcelain=v2")
    environment = f"""# Environment snapshot

- Campaign start UTC: `{started}`
- Local timestamp token: `{timestamp}`
- Repository: `{REPO}`
- Branch: `{command('git', 'branch', '--show-current')}`
- Git HEAD: `{command('git', 'rev-parse', 'HEAD')}`
- Staged index: empty (required)
- Worktree status at start:

```text
{git_status or '(clean)'}
```

- Python: `{sys.version.replace(os.linesep, ' ')}`
- Platform: `{platform.platform()}`
- Package versions: `{json.dumps(package_versions(), sort_keys=True)}`
- MPS: `{json.dumps(mps, sort_keys=True)}`
- Source catalog SHA-256: `{catalog_hash}`
- Source split SHA-256: `{split_hash}`
- Historical checkpoint count: `{len(checkpoints)}`
- Free disk bytes: `{usage.free}`
- Total disk bytes: `{usage.total}`

The worktree was dirty but unstaged at campaign start. Those pre-existing files
are preserved and excluded from this append-only output root. No historical
development or lockbox scene was opened by this bootstrap.
"""
    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", environment)
    contract = """# Campaign contract

This run is append-only and collision-refusing. Historical checkpoints, runs,
development scenes, and lockbox scenes are immutable and out of scope. CPU is
used for provenance, manifests, simulation, optimization, calibration, and
reports. Neural training/inference must use MPS and must stop if MPS is absent.
No CPU neural fallback is permitted.

The active scientific scope is controlled Ambiguity Atlas and finite-candidate
recoverability feasibility. Cross-family auditing is blocked because fewer
than three meaningfully distinct compatible families are available. No absence
of a competing candidate is evidence of uniqueness, and no result can be
described as a mathematical or final-survey certificate.
"""
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", contract)

    forward_files = [REPO / "src/btk_scene.py", Path(__file__).resolve()]
    forward_rows = [
        {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in forward_files
    ]
    write_csv_fresh(run_dir / "tables/simulator_forward_model_hashes.csv", forward_rows, ["path", "sha256"])
    provenance = {
        "campaign_started_utc": started,
        "run_dir": str(run_dir.relative_to(REPO)),
        "branch": command("git", "branch", "--show-current"),
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status_porcelain_v2": git_status.splitlines(),
        "staged_index_empty": True,
        "source_catalog": {"path": str(CATALOG.relative_to(REPO)), "sha256": catalog_hash},
        "source_split": {"path": str(SOURCE_SPLIT.relative_to(REPO)), "sha256": split_hash},
        "candidate_checkpoint_hashes": {
            str(path.relative_to(REPO)): sha256_file(path) for path in (CONDITION_C, R0, R1)
        },
        "historical_checkpoint_inventory": "tables/checkpoint_inventory_before.csv",
        "authoritative_prior_run_hashes": "tables/authoritative_prior_run_hashes.csv",
        "simulator_forward_model_hashes": "tables/simulator_forward_model_hashes.csv",
        "development_scenes_opened": 0,
        "lockbox_scenes_opened": 0,
        "historical_outputs_modified": 0,
        "free_disk_bytes": usage.free,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)

    frozen_at = datetime.now(timezone.utc).isoformat()
    prereg_name = (
        "ambiguity_atlas_v0.md"
        if args.campaign_prefix == "thayer_ambiguity_atlas_v0"
        else "competing_hypothesis_recoverability.md"
    )
    prereg_path = run_dir / "preregistration" / prereg_name
    write_text_fresh(prereg_path, preregistration_text(roles, frozen_at))
    prereg_hash = sha256_file(prereg_path)
    write_json_fresh(
        run_dir / "preregistration/freeze_record.json",
        {
            "status": "FROZEN_BEFORE_NEW_CANDIDATE_INFERENCE_OR_TRAINING",
            "frozen_at_utc": frozen_at,
            "preregistration_path": str(prereg_path.relative_to(run_dir)),
            "preregistration_sha256": prereg_hash,
            "candidate_inference_count_at_freeze": 0,
            "auditor_training_count_at_freeze": 0,
            "development_scene_access_count": 0,
            "lockbox_scene_access_count": 0,
        },
    )
    write_json_fresh(
        run_dir / "logs/bootstrap_complete.json",
        {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "preregistration_sha256": prereg_hash},
    )
    print(json.dumps({"run_dir": str(run_dir), "preregistration_sha256": prereg_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
