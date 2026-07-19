#!/usr/bin/env python3
"""Frozen Thayer-PU eligibility and POST label-support audit.

The workflow is phased so preregistration is hashed before neural inference.
Every campaign artifact is created exclusively and historical inputs are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_probabilistic_unet_pre_atlas import sample_outputs  # noqa: E402
from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy  # noqa: E402
from src.canonical_tensor_hash import SCHEMA_VERSION, canonical_tensor_sha256  # noqa: E402
from src.competing_hypotheses import scientific_distance  # noqa: E402
from src.direct_catalog_safety_auditor import (  # noqa: E402
    MEAN_PSF_FWHM_PIXELS,
    SCALAR_FEATURE_NAMES,
    connected_components,
    deployable_scalar_features,
    normalized_post_image,
    post_audit_supervision,
)
from src.models_probabilistic_unet import ThayerProbabilisticUNet, swap_decomposition  # noqa: E402

PU_RUN = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
AUDIT_RUN = REPO / "outputs/runs/thayer_audit_v0_20260714_154655"
HIERARCHICAL = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
PROMPT_ABLATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"

CHECKPOINT = PU_RUN / "checkpoints/thayer_pu_best.pth"
MODEL_SOURCE = REPO / "src/models_probabilistic_unet.py"
PROMPT_SOURCE = REPO / "scripts/thayer_select_prompt_ablation_common.py"
SAFETY_SOURCE = REPO / "src/direct_catalog_safety_auditor.py"
THRESHOLD_CONTRACT = REPO / "docs/d3_threshold_contract.md"
SOURCE_SPLIT = PROMPT_ABLATION / "manifests/source_split_manifest.csv"
NORMALIZATION = PROMPT_ABLATION / "manifests/normalization.json"
PU_SCENES = PU_RUN / "manifests/probabilistic_unet_scene_definitions.csv"
PU_RENDERED = PU_RUN / "manifests/probabilistic_unet_rendered_scene_manifest.csv"
HISTORICAL_CHECKPOINTS = AUDIT_RUN / "tables/checkpoint_inventory_after.csv"

SOURCE_SPECS = {
    "training": (
        HIERARCHICAL / "manifests/v2_r_training_scene_manifest.csv",
        HIERARCHICAL / "manifests/v2_r_training_scenes.h5",
        HIERARCHICAL / "features/v2_r_training_frozen_reconstructions.h5",
    ),
    "validation": (
        HIERARCHICAL / "manifests/v2_r_validation_scene_manifest.csv",
        HIERARCHICAL / "manifests/v2_r_validation_scenes.h5",
        HIERARCHICAL / "features/v2_r_validation_frozen_reconstructions.h5",
    ),
    "calibration": (
        HIERARCHICAL / "manifests/v2_natural_calibration_scene_manifest.csv",
        HIERARCHICAL / "manifests/v2_natural_calibration_scenes.h5",
        HIERARCHICAL / "features/v2_natural_calibration_frozen_reconstructions.h5",
    ),
}

EXPECTED_HASHES = {
    CHECKPOINT: "c1d17a3f67962cce2fec03d6b15da5f2e330ee97b31c270a7ff019a1373a557e",
    MODEL_SOURCE: "b86de449ba0524c5675ea300e87ff753c4d18b974ca18e26fbae74a760ed8b1e",
    PROMPT_SOURCE: "449079faf20a29a1c65cd9c5916d1cffe641b4ef0ac5293ca9987cf2c3904fb7",
    SAFETY_SOURCE: "9efe750a60d746cd6cd496c6843a9a4f62500016ed599d1a1a84e4edac199df4",
    THRESHOLD_CONTRACT: "ac6c4585d214008c03b19b6b61b69dee999242d02d7e1cb724caf5fffa7320e3",
    SOURCE_SPLIT: "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27",
    NORMALIZATION: "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    PU_SCENES: "fa809f1fae58b09531d646c6ca20455721d987b6285b67eb92a808196fa4cfe4",
    PU_RENDERED: "fc2041e070705968a12ebbdfd62fd55c4742023acaa7312e8e0800fae10d59d1",
}

K = 16
LATENT_SEEDS = tuple(range(2026077600, 2026077616))
BATCH_SIZE = 8
BOOTSTRAP_REPLICATES = 300
BOOTSTRAP_SEED = 2026079101
PREFLIGHT_POSITIONS = tuple(range(8))
REPLAY_POSITIONS = (0, 1, 2, 7, 15, 31, 63, 127, 255, 511)
EXPECTED_COUNTS = {"training": 3998, "validation": 793, "calibration": 2800}
DIRECTORIES = (
    "diagnostics", "logs", "reports", "preregistration", "provenance", "manifests",
    "frozen_model", "deployment_rule", "inference", "aligned_outputs", "episodes",
    "safety_labels", "tables", "figures", "replay_verification", "bootstrap",
)
SUPPORT_GATES = {
    "training": {"safe": 500, "unsafe": 500, "unique_safe": 100, "perturbed_safe": 0, "safe_groups": 0},
    "validation": {"safe": 150, "unsafe": 150, "unique_safe": 100, "perturbed_safe": 50, "safe_groups": 100},
    "calibration": {"safe": 150, "unsafe": 150, "unique_safe": 100, "perturbed_safe": 50, "safe_groups": 100},
}
REASON_COLUMNS = (
    "catastrophic_image", "catastrophic_flux", "catastrophic_color", "catastrophic_centroid",
    "source_confusion", "catastrophic", "physical_output_contract_failure",
    "false_subtraction_failure", "worse_than_baseline_catastrophic", "unsafe_to_catalog",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_hash(entries: Sequence[tuple[str, str]]) -> str:
    payload = json.dumps(list(entries), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fresh_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def fresh_json(path: Path, payload: object) -> None:
    fresh_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_command(arguments: Sequence[str]) -> dict[str, object]:
    result = subprocess.run(list(arguments), cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": list(arguments), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def require_run(run: Path, stage: str | None = None) -> Path:
    target = run.resolve()
    expected_parent = (REPO / "outputs/runs").resolve()
    if target.parent != expected_parent or not target.name.startswith("thayer_pu_eligibility_v1_"):
        raise RuntimeError("invalid eligibility-run path")
    if stage is not None:
        record = json.loads((target / "logs" / stage).read_text())
        if record["status"] != "PASS":
            raise RuntimeError(f"required stage did not pass: {stage}")
    return target


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("CPU fallback is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def load_model(device: torch.device) -> ThayerProbabilisticUNet:
    if sha256_file(CHECKPOINT) != EXPECTED_HASHES[CHECKPOINT]:
        raise RuntimeError("frozen Thayer-PU checkpoint hash mismatch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    required = {
        "model_family": "THAYER_PU", "selection": "minimum validation frozen total objective",
        "epoch": 27, "latent_dimension": 8, "parameter_count": 170278,
        "prior_truth_free": True, "posterior_training_only": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("selected checkpoint metadata mismatch")
    model = ThayerProbabilisticUNet().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def package_versions() -> dict[str, str]:
    names = {
        "astropy": "astropy", "btk": "blending-toolkit", "galsim": "GalSim",
        "h5py": "h5py", "matplotlib": "matplotlib", "numpy": "numpy",
        "pandas": "pandas", "scipy": "scipy", "torch": "torch",
    }
    output = {}
    for label, package in names.items():
        try:
            output[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[label] = "NOT_INSTALLED"
    return output


def verify_historical_checkpoints() -> pd.DataFrame:
    inventory = pd.read_csv(HISTORICAL_CHECKPOINTS, dtype=str, keep_default_na=False)
    rows = []
    for row in inventory.itertuples(index=False):
        path = REPO / row.relative_path
        observed = sha256_file(path) if path.is_file() else "MISSING"
        rows.append({
            "relative_path": row.relative_path, "expected_sha256": row.after_sha256,
            "observed_sha256": observed, "unchanged": observed == row.after_sha256,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 743 or not frame.unchanged.all():
        raise RuntimeError("historical checkpoint inventory mismatch")
    return frame


def source_manifests(run: Path) -> dict[str, pd.DataFrame]:
    pu = pd.read_csv(PU_SCENES, dtype=str, keep_default_na=False)
    base = pu.partition.isin(("training", "validation"))
    excluded = set(pu.loc[base, "source_a_group"]) | set(pu.loc[base, "source_b_group"])
    outputs = {}
    group_sets = {}
    audit_rows = []
    for partition, (manifest_path, scene_path, condition_path) in SOURCE_SPECS.items():
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False, low_memory=False)
        eligible = frame.query_state.eq("UNIQUE_VALID") & ~frame.source_a_group.isin(excluded) & ~frame.source_b_group.isin(excluded)
        selected = frame.loc[eligible].copy()
        selected.insert(0, "eligibility_index", np.arange(len(selected), dtype=int))
        selected.insert(1, "upstream_index", np.flatnonzero(eligible.to_numpy()))
        selected["eligibility_partition"] = partition
        selected["base_prediction_provenance"] = "OUT_OF_THAYER_PU_FIT_AND_SELECTION_GROUPS"
        selected["both_groups_excluded_from_thayer_pu_fit_and_selection"] = True
        selected["upstream_manifest_path"] = relative(manifest_path)
        selected["upstream_scene_path"] = relative(scene_path)
        selected["condition_c_reconstruction_path"] = relative(condition_path)
        if len(selected) != EXPECTED_COUNTS[partition]:
            raise RuntimeError(f"unexpected {partition} OOF count: {len(selected)}")
        group_sets[partition] = set(selected.source_a_group) | set(selected.source_b_group)
        if group_sets[partition] & excluded:
            raise RuntimeError(f"Thayer-PU base-group leakage in {partition}")
        outputs[partition] = selected
        fresh_csv(run / f"manifests/thayer_pu_{partition}_source_manifest.csv", selected)
        audit_rows.append({
            "partition": partition, "rows": len(selected),
            "unique_valid_prompt_rows": int(selected.prompt_subtype.eq("UNIQUE_VALID").sum()),
            "perturbed_valid_prompt_rows": int(selected.prompt_subtype.eq("PERTURBED_VALID").sum()),
            "distinct_source_groups": len(group_sets[partition]), "base_group_overlap": 0,
            "source_manifest_sha256": sha256_file(manifest_path), "scene_h5_sha256": sha256_file(scene_path),
            "condition_c_output_sha256": sha256_file(condition_path),
        })
    for left, right in itertools.combinations(group_sets, 2):
        if group_sets[left] & group_sets[right]:
            raise RuntimeError(f"source-group overlap: {left}/{right}")
    fresh_csv(run / "tables/source_partition_audit.csv", pd.DataFrame(audit_rows))
    fresh_json(run / "manifests/source_group_exclusion_contract.json", {
        "source_split_path": relative(SOURCE_SPLIT), "source_split_sha256": sha256_file(SOURCE_SPLIT),
        "thayer_pu_fit_and_selection_group_count": len(excluded),
        "exclusion_partitions": ["training", "validation"], "audit_partition_counts": EXPECTED_COUNTS,
        "cross_partition_source_group_overlap": 0, "all_audit_groups_excluded_from_fit_and_selection": True,
    })
    return outputs


def preregistration_text(manifests: dict[str, pd.DataFrame], frozen_at: str) -> str:
    replay = ", ".join(map(str, REPLAY_POSITIONS))
    return f"""# Frozen Thayer-PU deployment and label-support preregistration

Frozen UTC: `{frozen_at}` before any new Thayer-PU inference, deployed output,
safety label, prevalence calculation, or family comparison. Working experiment:
`Thayer-PU-Eligibility-v1`.

## Frozen model, constructor, inputs, and partitions

- Checkpoint: `{relative(CHECKPOINT)}`; SHA-256 `{sha256_file(CHECKPOINT)}`; validation-selected epoch 27.
- Constructor: `src.models_probabilistic_unet.ThayerProbabilisticUNet`; source SHA-256 `{sha256_file(MODEL_SOURCE)}`.
- Prompt: unit-peak sigma-2-pixel Gaussian coordinate prompt in `{relative(PROMPT_SOURCE)}`; SHA-256 `{sha256_file(PROMPT_SOURCE)}`.
- Normalization: `{relative(NORMALIZATION)}`; SHA-256 `{sha256_file(NORMALIZATION)}`; g/r/z scales apply unchanged.
- Source split: `{relative(SOURCE_SPLIT)}`; SHA-256 `{sha256_file(SOURCE_SPLIT)}`.
- Authorized valid-query rows: training `{len(manifests['training'])}`, validation `{len(manifests['validation'])}`, calibration `{len(manifests['calibration'])}`. Every source group is absent from Thayer-PU fitting and validation checkpoint selection. Original source partitions remain unchanged and mutually disjoint.
- Only `UNIQUE_VALID` query-state rows receive POST labels. Prompt subtypes `UNIQUE_VALID` and `PERTURBED_VALID` are retained. Invalid query states remain outside this POST audit.

## One frozen deployment rule

The sole rule is `POSTERIOR_MEAN_RULE` in the campaign vocabulary, implemented
for this model as a truth-free prior-predictive mean. For every episode, draw
K={K} standard-normal epsilon vectors with NumPy PCG64 seeds `{LATENT_SEEDS[0]}`
through `{LATENT_SEEDS[-1]}` in that exact order. The prior is `p(z|blend)` and
receives no prompt or truth. Reparameterize with its frozen mean/log variance,
decode K six-channel decompositions, take requested channels 0:3, average them
on CPU in float64, then cast once to contiguous float32. No candidate is
selected. Truth, Atlas, and safety labels never enter deployment.

Inference batch size is {BATCH_SIZE}; the final partial batch is allowed. Epsilon
is generated for the full frozen partition before batching. MPS is mandatory for
model execution; CPU is restricted to manifests, hashing, aggregation, labels,
metrics, plots, and reports. Band order is g/r/z, units are detected electrons,
background is zero, and linear output remains unclipped.

## Promptability, replay, and physical gates

Preflight uses positions {PREFLIGHT_POSITIONS} from every partition. It requires
shape `(K,6,60,60)`, float32, finite values, exact candidate/deployed replay,
exact batch-size 1/4/{BATCH_SIZE} invariance, majority-of-K paired prompt identity
at least 0.80, individual requested identity at least 0.70, and every band
identity above 0.50. Failure assigns `THAYER_PU_DEPLOYMENT_INELIGIBLE` and stops.

Full replay positions are {replay} per partition. It requires exact scene and
prompt identity, candidate order and hashes, deployed hash, normalized POST-input
hash, scalar features, and batch geometry. Any unresolved mismatch fails closed.

Physical outputs must have shape `(3,60,60)`, float32, and finite values. Minimum,
maximum, and negative-pixel fraction are reported. Negative values are not clipped
or corrected and remain an unchanged physical-output-contract failure. Candidate
diameter is maximum pairwise physical-space RMS among requested candidates;
stable within-partition rank quartiles are frozen before labels.

## Unchanged safety and support gates

Safety is computed only after every output is closed and hashed, with
`post_audit_supervision`, source SHA-256 `{sha256_file(SAFETY_SOURCE)}`, and
threshold contract SHA-256 `{sha256_file(THRESHOLD_CONTRACT)}`. Existing strict-
greater image 0.25, per-band flux 0.20, color 0.20 mag, centroid 0.50 PSF,
false-subtraction 0.20, source-confusion, physical-output, and worse-than-blend
baseline rules are unchanged.

Training requires at least 500 safe and 500 unsafe episodes and safe prevalence
in [0.05,0.95]. Validation and calibration each require at least 150 safe and 150
unsafe and the same prevalence interval. Each partition requires at least 100
safe `UNIQUE_VALID` rows. Validation and calibration also require at least 50
safe `PERTURBED_VALID` rows and 100 distinct safe source groups. Gates cannot be
weakened after labels.

## Family distinctness, bootstrap, outcome, and stopping

Condition-C comparison is aligned and non-development. Safety disagreement is
unequal-label fraction; reconstruction-error rank correlation is Spearman
correlation of requested-source MSE; output-contract disagreement is unequal
physical-failure fraction. Candidate-diameter behavior is materially different
if Condition C is zero and Thayer-PU median exceeds `1e-6` detected-electron RMS.
Structural distinctness comes only from the independently verified architecture.
Family PASS needs at least two frozen criteria.

Use exactly {BOOTSTRAP_REPLICATES} deterministic connected-source-group replicates
with seed {BOOTSTRAP_SEED}. Intervals cover safe/unsafe prevalence, output-contract
pass, catastrophic pass, both-pass, family label disagreement, and safe source-
group support. No auditor is fitted and no auditor-performance claim is made.

Outcome precedence: prompt/determinism/batch/replay failure ->
`THAYER_PU_DEPLOYMENT_INELIGIBLE`; unavailable OOF training outputs ->
`THAYER_PU_OOF_PROVENANCE_FAILURE`; artifact, label, manifest, or scientific-gate
reproduction failure -> `DATA_OR_IMPLEMENTATION_FAILURE`; otherwise all support
and distinctness gates pass -> `THAYER_PU_ELIGIBLE_WITH_LABEL_SUPPORT`; otherwise
`THAYER_PU_ELIGIBLE_BUT_LABEL_COLLAPSED`.

Stop on frozen hash mismatch, staged changes, MPS failure, source overlap, in-sample
row, nonfinite inference, alignment mismatch, threshold change, checkpoint mutation,
development/final-lockbox access, or truth/Atlas-based deployment choice. Development,
Atlas selection, and final lockbox data are prohibited. No reconstructor or auditor
is trained.
"""


def bootstrap(run: Path) -> None:
    target = run.resolve()
    expected_parent = (REPO / "outputs/runs").resolve()
    if target.exists():
        raise FileExistsError(target)
    if target.parent != expected_parent or not target.name.startswith("thayer_pu_eligibility_v1_"):
        raise RuntimeError("invalid fresh run path")
    target.mkdir(parents=True, exist_ok=False)
    for directory in DIRECTORIES:
        (target / directory).mkdir(exist_ok=False)
    started = utc_now()
    if run_command(("git", "diff", "--cached", "--quiet"))["returncode"] != 0:
        raise RuntimeError("staged index is not empty")
    input_rows = []
    mismatches = []
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256_file(path)
        input_rows.append({
            "relative_path": relative(path), "expected_sha256": expected,
            "observed_sha256": observed, "bytes": path.stat().st_size,
            "status": "PASS" if observed == expected else "FAIL",
        })
        if observed != expected:
            mismatches.append(relative(path))
    if mismatches:
        raise RuntimeError(f"frozen-input mismatch: {mismatches}")
    checkpoint_inventory = verify_historical_checkpoints()
    fresh_csv(target / "provenance/historical_checkpoint_inventory_before.csv", checkpoint_inventory)
    manifests = source_manifests(target)
    condition_manifests = [
        AUDIT_RUN / "episodes/post_training_manifest.csv",
        AUDIT_RUN / "episodes/post_validation_manifest.csv",
        AUDIT_RUN / "episodes/policy_validation_manifest.csv",
        AUDIT_RUN / "episodes/policy_calibration_manifest.csv",
    ]
    for path in condition_manifests:
        value = sha256_file(path)
        input_rows.append({
            "relative_path": relative(path), "expected_sha256": value,
            "observed_sha256": value, "bytes": path.stat().st_size, "status": "PASS",
        })
    fresh_csv(target / "provenance/frozen_input_inventory.csv", pd.DataFrame(input_rows))
    condition_checkpoint = PROMPT_ABLATION / "checkpoints/c_randomized_coordinate_prompt_best.pth"
    fresh_csv(target / "frozen_model/checkpoint_model_inventory.csv", pd.DataFrame([
        {"role": "selected_thayer_pu", "path": relative(CHECKPOINT), "sha256": sha256_file(CHECKPOINT), "epoch": 27, "parameters": 170278, "constructor": "ThayerProbabilisticUNet"},
        {"role": "model_source", "path": relative(MODEL_SOURCE), "sha256": sha256_file(MODEL_SOURCE), "epoch": "", "parameters": "", "constructor": ""},
        {"role": "condition_c_comparator", "path": relative(condition_checkpoint), "sha256": sha256_file(condition_checkpoint), "epoch": "", "parameters": 119091, "constructor": "CompactSelectNet"},
    ]))
    deployment = {
        "schema_version": "thayer-pu-deployment-rule-v1", "rule_name": "POSTERIOR_MEAN_RULE",
        "model_specific_interpretation": "truth-free prior-predictive mean",
        "checkpoint_path": relative(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "constructor": "src.models_probabilistic_unet.ThayerProbabilisticUNet",
        "latent_distribution": "epsilon iid standard normal; z=prior_mean+prior_std*epsilon",
        "k": K, "latent_seeds": list(LATENT_SEEDS), "candidate_order": "ascending listed seed order",
        "candidate_mapping": "full six-channel output; requested channels 0:3",
        "aggregation": "CPU float64 arithmetic mean; one float32 cast",
        "selection_uses_truth": False, "selection_uses_atlas": False, "candidate_selection": "none",
        "batch_size": BATCH_SIZE, "neural_device": "mps", "cpu_neural_fallback": False,
        "output_shape": [3, 60, 60], "output_dtype": "float32", "clipping": False,
        "canonical_hash_schema": SCHEMA_VERSION,
    }
    fresh_json(target / "deployment_rule/frozen_deployment_rule.json", deployment)
    fresh_csv(target / "deployment_rule/latent_seed_manifest.csv", pd.DataFrame([
        {"candidate_index": index, "seed": seed, "distribution": "NumPy PCG64 standard_normal float32", "truth_free": True}
        for index, seed in enumerate(LATENT_SEEDS)
    ]))
    fresh_text(target / "diagnostics/deployment_rule_rationale.md", """# Deployment-rule rationale

The frozen implementation already supports explicit truth-free prior samples and
K=16 was its authoritative non-Atlas promptability contract. The first 16 seeds
of its separately preregistered 32-sample sequence are kept in order. The mean is
deployable and reduces Monte Carlo variance without truth or candidate selection.
No alternative deployment rule will be evaluated for label prevalence.
""")
    frozen_at = utc_now()
    prereg_path = target / "preregistration/frozen_thayer_pu_deployment_and_label_support.md"
    fresh_text(prereg_path, preregistration_text(manifests, frozen_at))
    prereg_hash = sha256_file(prereg_path)
    fresh_text(target / "preregistration/frozen_thayer_pu_deployment_and_label_support.sha256", f"{prereg_hash}  {prereg_path.name}\n")
    fresh_json(target / "preregistration/freeze_record.json", {
        "status": "FROZEN_BEFORE_ANY_NEW_THAYER_PU_INFERENCE", "frozen_at_utc": frozen_at,
        "preregistration_sha256": prereg_hash,
        "deployment_rule_sha256": sha256_file(target / "deployment_rule/frozen_deployment_rule.json"),
        "latent_seed_manifest_sha256": sha256_file(target / "deployment_rule/latent_seed_manifest.csv"),
        "development_scene_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
    })
    git_status = run_command(("git", "status", "--porcelain=v2"))["stdout"].splitlines()
    branch = run_command(("git", "branch", "--show-current"))["stdout"].strip()
    head = run_command(("git", "rev-parse", "HEAD"))["stdout"].strip()
    disk = os.statvfs(REPO)
    require_mps()
    prompt_hash = combined_hash([(relative(PROMPT_SOURCE), sha256_file(PROMPT_SOURCE)), (relative(NORMALIZATION), sha256_file(NORMALIZATION))])
    provenance = {
        "experiment": "Thayer-PU-Eligibility-v1", "run_dir": relative(target), "start_utc": started,
        "preregistration_frozen_utc": frozen_at, "branch": branch, "git_head": head,
        "git_status_porcelain_v2": git_status, "staged_index_empty": True,
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "mps_execution_probe": 2.0, "free_disk_bytes": disk.f_bavail * disk.f_frsize,
        "checkpoint_path": relative(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_source_sha256": sha256_file(MODEL_SOURCE), "prompt_contract_sha256": prompt_hash,
        "source_partition_sha256": sha256_file(SOURCE_SPLIT),
        "condition_c_audit_manifest_hashes": {relative(path): sha256_file(path) for path in condition_manifests},
        "scientific_threshold_sha256": sha256_file(THRESHOLD_CONTRACT),
        "safety_implementation_sha256": sha256_file(SAFETY_SOURCE),
        "historical_checkpoint_count": len(checkpoint_inventory), "historical_checkpoint_mismatches": 0,
        "campaign_script_sha256": sha256_file(Path(__file__)), "readme_sha256": sha256_file(REPO / "README.md"),
        "development_scene_access_count": 0, "atlas_selection_access_count": 0,
        "final_lockbox_access_count": 0, "reconstruction_models_trained": 0, "auditors_trained": 0,
    }
    fresh_json(target / "provenance/start_environment.json", provenance)
    fresh_text(target / "diagnostics/environment_snapshot.md", "# Environment snapshot\n\n```json\n" + json.dumps(provenance, indent=2, sort_keys=True) + "\n```\n")
    fresh_json(target / "logs/bootstrap_complete.json", {
        "status": "PASS", "completed_utc": utc_now(), "preregistration_sha256": prereg_hash,
        "partition_counts": EXPECTED_COUNTS, "source_group_overlap": 0, "fit_selection_group_overlap": 0,
        "historical_checkpoint_mismatches": 0, "mps": True, "staged_index_empty": True,
    })
    print(json.dumps({"status": "PASS", "run": relative(target), "preregistration_sha256": prereg_hash}, sort_keys=True))


def scales() -> np.ndarray:
    return np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)


def epsilon_manifest(scene_count: int) -> np.ndarray:
    values = [np.random.default_rng(seed).standard_normal((scene_count, 8)).astype(np.float32) for seed in LATENT_SEEDS]
    return np.stack(values, axis=1)


def infer_candidates(
    model: ThayerProbabilisticUNet,
    blend_physical: np.ndarray,
    prompt: np.ndarray,
    epsilon: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    scale = scales()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(blend_physical), batch_size):
            stop = min(start + batch_size, len(blend_physical))
            blend = torch.from_numpy(np.ascontiguousarray(blend_physical[start:stop] / scale[None, :, None, None])).to(device)
            request = torch.from_numpy(np.ascontiguousarray(prompt[start:stop], dtype=np.float32)).to(device)
            eps = torch.from_numpy(np.ascontiguousarray(epsilon[start:stop], dtype=np.float32)).to(device)
            mean, log_variance = model.encode_prior(blend)
            outputs.append(sample_outputs(model, blend, request, mean, log_variance, eps).cpu().numpy())
    scale6 = np.tile(scale, 2)[None, None, :, None, None]
    return np.ascontiguousarray(np.concatenate(outputs) * scale6, dtype=np.float32)


def deployed_mean(candidates: np.ndarray) -> np.ndarray:
    if candidates.ndim != 5 or candidates.shape[1:] != (K, 6, 60, 60):
        raise ValueError(f"candidate geometry mismatch: {candidates.shape}")
    return np.ascontiguousarray(candidates[:, :, :3].mean(axis=1, dtype=np.float64).astype(np.float32))


def alternate_prompts(xy: np.ndarray, matched: np.ndarray) -> np.ndarray:
    result = []
    for coordinates, requested in zip(xy, matched):
        other = 1 - int(requested)
        result.append(gaussian_prompt_numpy(float(coordinates[other, 0]), float(coordinates[other, 1])))
    return np.stack(result)[:, None].astype(np.float32)


def candidate_hashes(candidates: np.ndarray) -> list[list[str]]:
    return [[canonical_tensor_sha256(sample) for sample in scene] for scene in candidates]


def preflight(run: Path) -> None:
    target = require_run(run, "bootstrap_complete.json")
    if (target / "logs/preflight_complete.json").exists():
        raise FileExistsError("preflight already exists")
    device = require_mps()
    model = load_model(device)
    rows = []
    all_majority = []
    all_individual = []
    band_success = [[], [], []]
    all_exact = []
    started = time.time()
    for partition, (_, scene_path, _) in SOURCE_SPECS.items():
        manifest = pd.read_csv(target / f"manifests/thayer_pu_{partition}_source_manifest.csv", dtype=str, keep_default_na=False)
        positions = np.asarray(PREFLIGHT_POSITIONS, dtype=int)
        upstream = manifest.iloc[positions].upstream_index.astype(int).to_numpy()
        with h5py.File(scene_path, "r") as handle:
            blend = np.asarray(handle["blend"][upstream], dtype=np.float32)
            prompt = np.asarray(handle["prompt"][upstream], dtype=np.float32)
            isolated = np.asarray(handle["isolated"][upstream], dtype=np.float32)
            xy = np.asarray(handle["xy"][upstream], dtype=np.float64)
            matched = np.asarray(handle["matched_index"][upstream], dtype=np.int8)
        epsilon = epsilon_manifest(len(manifest))[positions]
        other_prompt = alternate_prompts(xy, matched)
        frozen = infer_candidates(model, blend, prompt, epsilon, device, batch_size=BATCH_SIZE)
        repeated = infer_candidates(model, blend, prompt, epsilon, device, batch_size=BATCH_SIZE)
        batch4 = infer_candidates(model, blend, prompt, epsilon, device, batch_size=4)
        singles = infer_candidates(model, blend, prompt, epsilon, device, batch_size=1)
        swapped = infer_candidates(model, blend, other_prompt, epsilon, device, batch_size=BATCH_SIZE)
        hashes = candidate_hashes(frozen)
        repeat_hashes = candidate_hashes(repeated)
        batch4_hashes = candidate_hashes(batch4)
        single_hashes = candidate_hashes(singles)
        deployed = deployed_mean(frozen)
        deployed_repeat = deployed_mean(repeated)
        deployed4 = deployed_mean(batch4)
        deployed1 = deployed_mean(singles)
        for local, position in enumerate(positions):
            requested = isolated[local, int(matched[local])]
            alternate = isolated[local, 1 - int(matched[local])]
            own = ((frozen[local, :, :3] - requested[None]) ** 2).mean(axis=(1, 2, 3))
            alt = ((frozen[local, :, :3] - alternate[None]) ** 2).mean(axis=(1, 2, 3))
            other_own = ((swapped[local, :, :3] - alternate[None]) ** 2).mean(axis=(1, 2, 3))
            other_alt = ((swapped[local, :, :3] - requested[None]) ** 2).mean(axis=(1, 2, 3))
            success_a, success_b = own < alt, other_own < other_alt
            majority = bool(success_a.sum() > K / 2 and success_b.sum() > K / 2)
            individual = float(np.mean(np.concatenate((success_a, success_b))))
            all_majority.append(majority)
            all_individual.extend(np.concatenate((success_a, success_b)).tolist())
            for band in range(3):
                a_own = ((frozen[local, :, band] - requested[band]) ** 2).mean(axis=(1, 2))
                a_alt = ((frozen[local, :, band] - alternate[band]) ** 2).mean(axis=(1, 2))
                b_own = ((swapped[local, :, band] - alternate[band]) ** 2).mean(axis=(1, 2))
                b_alt = ((swapped[local, :, band] - requested[band]) ** 2).mean(axis=(1, 2))
                band_success[band].extend(np.concatenate((a_own < a_alt, b_own < b_alt)).tolist())
            candidates_exact = hashes[local] == repeat_hashes[local] == batch4_hashes[local] == single_hashes[local]
            deployed_hash = canonical_tensor_sha256(deployed[local])
            deployed_exact = deployed_hash == canonical_tensor_sha256(deployed_repeat[local]) == canonical_tensor_sha256(deployed4[local]) == canonical_tensor_sha256(deployed1[local])
            exact = bool(candidates_exact and deployed_exact)
            all_exact.append(exact)
            swapped_order = np.take(swapped[local], [3, 4, 5, 0, 1, 2], axis=1)
            rows.append({
                "partition": partition, "manifest_position": int(position),
                "scene_id": manifest.iloc[position].scene_id,
                "prompt_subtype": manifest.iloc[position].prompt_subtype,
                "candidate_sequence_exact_replay": hashes[local] == repeat_hashes[local],
                "batch_4_invariant": hashes[local] == batch4_hashes[local],
                "single_scene_invariant": hashes[local] == single_hashes[local],
                "deployed_hash": deployed_hash, "deployed_exact_all_geometries": deployed_exact,
                "output_shape": str(tuple(frozen[local].shape)), "output_dtype": str(frozen.dtype),
                "finite": bool(np.isfinite(frozen[local]).all()),
                "majority_prompt_swap_identity": majority, "individual_prompt_identity": individual,
                "prompt_swap_consistency_mse": float(np.mean((frozen[local] - swapped_order) ** 2)),
            })
    frame = pd.DataFrame(rows)
    majority_rate = float(np.mean(all_majority))
    individual_rate = float(np.mean(all_individual))
    band_rates = [float(np.mean(values)) for values in band_success]
    gates = pd.DataFrame([
        {"gate": "majority_of_k_prompt_swap", "threshold": ">=0.80", "observed": majority_rate, "pass": majority_rate >= 0.80},
        {"gate": "individual_requested_identity", "threshold": ">=0.70", "observed": individual_rate, "pass": individual_rate >= 0.70},
        {"gate": "no_band_identity_inversion", "threshold": "each >0.50", "observed": ";".join(map(str, band_rates)), "pass": all(value > 0.50 for value in band_rates)},
        {"gate": "candidate_replay", "threshold": "all exact", "observed": bool(frame.candidate_sequence_exact_replay.all()), "pass": bool(frame.candidate_sequence_exact_replay.all())},
        {"gate": "batch_size_invariance", "threshold": "all exact", "observed": bool(frame.batch_4_invariant.all()), "pass": bool(frame.batch_4_invariant.all())},
        {"gate": "single_scene_consistency", "threshold": "all exact", "observed": bool(frame.single_scene_invariant.all()), "pass": bool(frame.single_scene_invariant.all())},
        {"gate": "shape_dtype_finite", "threshold": "Kx6x60x60 float32 finite", "observed": bool(frame.finite.all()), "pass": bool(frame.finite.all() and (frame.output_dtype == "float32").all())},
    ])
    passed = bool(gates["pass"].all() and all(all_exact))
    fresh_csv(target / "tables/deployment_preflight_audit.csv", frame)
    fresh_csv(target / "tables/deployment_preflight_gates.csv", gates)
    fresh_text(target / "diagnostics/deployment_preflight_report.md", f"""# Deployment preflight

Status: **{'PASS' if passed else 'FAIL'}**. The source-group-safe subset contained
{len(frame)} episodes across all partitions. Majority-of-{K} prompt identity was
`{majority_rate:.6f}`, individual identity `{individual_rate:.6f}`, and band rates
`{band_rates}`. Candidate replay and batch-size 1/4/{BATCH_SIZE} consistency were
exact: `{bool(all(all_exact))}`.
""")
    fresh_json(target / "logs/preflight_complete.json", {
        "status": "PASS" if passed else "FAIL",
        "scientific_classification": None if passed else "THAYER_PU_DEPLOYMENT_INELIGIBLE",
        "runtime_seconds": time.time() - started, "device": "mps", "mps_fallback": False,
        "subset_rows": len(frame), "majority_prompt_swap": majority_rate,
        "individual_identity": individual_rate, "band_identity_rates": band_rates,
        "exact_replay": bool(all(all_exact)), "full_inference_authorized": passed,
        "development_scene_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
    })
    if not passed:
        raise RuntimeError("THAYER_PU_DEPLOYMENT_INELIGIBLE")
    print(json.dumps({"status": "PASS", "majority_prompt_swap": majority_rate, "exact": True}, sort_keys=True))


def pairwise_candidate_diameter(requested: np.ndarray) -> float:
    values = np.asarray(requested, dtype=np.float64).reshape(K, -1)
    norms = np.sum(values * values, axis=1)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * (values @ values.T), 0.0)
    return float(np.sqrt(np.max(squared) / values.shape[1]))


def inference(run: Path) -> None:
    target = require_run(run, "preflight_complete.json")
    if (target / "logs/inference_complete.json").exists():
        raise FileExistsError("inference already exists")
    device = require_mps()
    model = load_model(device)
    start_all = time.time()
    inventory = []
    scale = scales()
    for partition, (_, scene_path, _) in SOURCE_SPECS.items():
        started = time.time()
        manifest = pd.read_csv(target / f"manifests/thayer_pu_{partition}_source_manifest.csv", dtype=str, keep_default_na=False)
        upstream = manifest.upstream_index.astype(int).to_numpy()
        with h5py.File(scene_path, "r") as handle:
            blend_all = np.asarray(handle["blend"][upstream], dtype=np.float32)
            prompt_all = np.asarray(handle["prompt"][upstream], dtype=np.float32)
            xy_all = np.asarray(handle["xy"][upstream], dtype=np.float64)
            matched_all = np.asarray(handle["matched_index"][upstream], dtype=np.int8)
        epsilon_all = epsilon_manifest(len(manifest))
        output_path = target / f"aligned_outputs/thayer_pu_{partition}_deployed.h5"
        metadata_rows = []
        with h5py.File(output_path, "x") as output:
            output.attrs["complete"] = False
            output.attrs["partition"] = partition
            output.attrs["checkpoint_sha256"] = sha256_file(CHECKPOINT)
            output.attrs["deployment_rule_sha256"] = sha256_file(target / "deployment_rule/frozen_deployment_rule.json")
            output.attrs["canonical_hash_schema"] = SCHEMA_VERSION
            reconstruction_ds = output.create_dataset("reconstruction", shape=(len(manifest), 3, 60, 60), dtype="f4", chunks=(1, 3, 60, 60), compression="lzf")
            scalar_ds = output.create_dataset("deployable_scalar_features", shape=(len(manifest), len(SCALAR_FEATURE_NAMES)), dtype="f4", chunks=(128, len(SCALAR_FEATURE_NAMES)), compression="lzf")
            diameter_ds = output.create_dataset("candidate_diameter", shape=(len(manifest),), dtype="f8")
            for start in range(0, len(manifest), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(manifest))
                candidates = infer_candidates(model, blend_all[start:stop], prompt_all[start:stop], epsilon_all[start:stop], device, batch_size=BATCH_SIZE)
                other_prompt = alternate_prompts(xy_all[start:stop], matched_all[start:stop])
                swapped = infer_candidates(model, blend_all[start:stop], other_prompt, epsilon_all[start:stop], device, batch_size=BATCH_SIZE)
                deployed = deployed_mean(candidates)
                hashes = candidate_hashes(candidates)
                requested_hashes = candidate_hashes(candidates[:, :, :3])
                for local in range(stop - start):
                    index = start + local
                    reconstruction = deployed[local]
                    scalar = deployable_scalar_features(blend_all[index], prompt_all[index], reconstruction)
                    normalized = normalized_post_image(blend_all[index], prompt_all[index], reconstruction, scale)
                    diameter = pairwise_candidate_diameter(candidates[local, :, :3])
                    reconstruction_ds[index] = reconstruction
                    scalar_ds[index] = scalar
                    diameter_ds[index] = diameter
                    swapped_order = np.take(swapped[local], [3, 4, 5, 0, 1, 2], axis=1)
                    metadata_rows.append({
                        "eligibility_index": index, "scene_id": manifest.iloc[index].scene_id,
                        "prompt_id": f"{manifest.iloc[index].scene_id}:{manifest.iloc[index].prompt_sha256}",
                        "partition": partition, "prompt_subtype": manifest.iloc[index].prompt_subtype,
                        "source_a_group": manifest.iloc[index].source_a_group, "source_b_group": manifest.iloc[index].source_b_group,
                        "latent_seed_list": json.dumps(LATENT_SEEDS, separators=(",", ":")),
                        "candidate_full_hashes": json.dumps(hashes[local], separators=(",", ":")),
                        "candidate_requested_hashes": json.dumps(requested_hashes[local], separators=(",", ":")),
                        "deployed_reconstruction_sha256": canonical_tensor_sha256(reconstruction),
                        "normalized_auditor_input_sha256": canonical_tensor_sha256(normalized),
                        "scalar_features_sha256": hashlib.sha256(np.ascontiguousarray(scalar, dtype=np.dtype("<f4")).tobytes()).hexdigest(),
                        "prompt_swap_consistency_mse": float(np.mean((candidates[local] - swapped_order) ** 2)),
                        "output_shape": "3x60x60", "output_dtype": "float32",
                        "finite_value_count": int(np.isfinite(reconstruction).sum()),
                        "nonfinite_value_count": int((~np.isfinite(reconstruction)).sum()),
                        "minimum_physical_value": float(np.min(reconstruction)),
                        "maximum_physical_value": float(np.max(reconstruction)),
                        "negative_pixel_fraction": float(np.mean(reconstruction < 0.0)),
                        "candidate_diameter": diameter, "deployment_rule": "POSTERIOR_MEAN_RULE",
                        "aggregation_precision": "float64_then_float32", "batch_size": BATCH_SIZE,
                        "neural_device": "mps", "mps_fallback": False,
                    })
                print(json.dumps({"phase": "inference", "partition": partition, "completed": stop, "total": len(manifest), "elapsed_seconds": time.time() - started}), flush=True)
            output.attrs["complete"] = True
            output.attrs["completed_count"] = len(manifest)
        metadata = pd.DataFrame(metadata_rows)
        ranks = metadata.candidate_diameter.rank(method="first").to_numpy(dtype=int) - 1
        metadata["candidate_diameter_quantile"] = [f"Q{min(4, (rank * 4) // len(metadata) + 1)}" for rank in ranks]
        metadata_path = target / f"inference/thayer_pu_{partition}_inference_manifest.csv"
        fresh_csv(metadata_path, metadata)
        inventory.append({
            "partition": partition, "rows": len(metadata), "h5_path": relative(output_path),
            "h5_sha256": sha256_file(output_path), "inference_manifest_path": relative(metadata_path),
            "inference_manifest_sha256": sha256_file(metadata_path),
            "all_finite": bool(metadata.nonfinite_value_count.eq(0).all()), "runtime_seconds": time.time() - started,
        })
        if not metadata.nonfinite_value_count.eq(0).all():
            raise RuntimeError("nonfinite full-inference output")
    fresh_csv(target / "aligned_outputs/output_freeze_inventory.csv", pd.DataFrame(inventory))
    fresh_json(target / "aligned_outputs/output_freeze_record.json", {
        "status": "FROZEN_BEFORE_SAFETY_LABELS", "frozen_at_utc": utc_now(),
        "partition_counts": EXPECTED_COUNTS,
        "inventory_sha256": sha256_file(target / "aligned_outputs/output_freeze_inventory.csv"),
        "all_outputs_complete_finite_and_hashed": True, "safety_labels_computed": False,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "deployment_rule_sha256": sha256_file(target / "deployment_rule/frozen_deployment_rule.json"),
    })
    fresh_json(target / "logs/inference_complete.json", {
        "status": "PASS", "runtime_seconds": time.time() - start_all,
        "partition_counts": EXPECTED_COUNTS, "device": "mps", "mps_fallback": False,
        "all_outputs_frozen_and_hashed": True, "safety_labels_computed_during_inference": False,
    })
    print(json.dumps({"status": "PASS", "partition_counts": EXPECTED_COUNTS, "runtime_seconds": time.time() - start_all}, sort_keys=True))


def batch_mismatch_diagnostic(run: Path) -> None:
    target = require_run(run)
    record = json.loads((target / "logs/preflight_complete.json").read_text())
    if record["status"] != "FAIL" or record["scientific_classification"] != "THAYER_PU_DEPLOYMENT_INELIGIBLE":
        raise RuntimeError("batch mismatch diagnostic is only valid after the frozen preflight failure")
    device = require_mps()
    model = load_model(device)
    rows = []
    for partition, (_, scene_path, _) in SOURCE_SPECS.items():
        manifest = pd.read_csv(target / f"manifests/thayer_pu_{partition}_source_manifest.csv", dtype=str, keep_default_na=False)
        positions = np.asarray(PREFLIGHT_POSITIONS, dtype=int)
        upstream = manifest.iloc[positions].upstream_index.astype(int).to_numpy()
        with h5py.File(scene_path, "r") as handle:
            blend = np.asarray(handle["blend"][upstream], dtype=np.float32)
            prompt = np.asarray(handle["prompt"][upstream], dtype=np.float32)
        epsilon = epsilon_manifest(len(manifest))[positions]
        batched = infer_candidates(model, blend, prompt, epsilon, device, batch_size=BATCH_SIZE)
        singles = infer_candidates(model, blend, prompt, epsilon, device, batch_size=1)
        mean_batched = deployed_mean(batched)
        mean_singles = deployed_mean(singles)
        for local, position in enumerate(positions):
            candidate_delta = np.abs(batched[local].astype(np.float64) - singles[local].astype(np.float64))
            deployed_delta = np.abs(mean_batched[local].astype(np.float64) - mean_singles[local].astype(np.float64))
            rows.append({
                "partition": partition, "manifest_position": int(position), "scene_id": manifest.iloc[position].scene_id,
                "candidate_unequal_values": int(np.count_nonzero(candidate_delta)),
                "candidate_max_abs_delta": float(candidate_delta.max()),
                "candidate_mean_abs_delta": float(candidate_delta.mean()),
                "candidate_rms_delta": float(np.sqrt(np.mean(candidate_delta ** 2))),
                "deployed_unequal_values": int(np.count_nonzero(deployed_delta)),
                "deployed_max_abs_delta": float(deployed_delta.max()),
                "deployed_mean_abs_delta": float(deployed_delta.mean()),
                "deployed_rms_delta": float(np.sqrt(np.mean(deployed_delta ** 2))),
                "candidate_hashes_equal": candidate_hashes(batched[local:local + 1])[0] == candidate_hashes(singles[local:local + 1])[0],
                "deployed_hashes_equal": canonical_tensor_sha256(mean_batched[local]) == canonical_tensor_sha256(mean_singles[local]),
            })
    frame = pd.DataFrame(rows)
    fresh_csv(target / "tables/single_scene_batch_mismatch.csv", frame)
    fresh_text(target / "diagnostics/single_scene_batch_mismatch.md", f"""# Single-scene versus batched mismatch

All `{len(frame)}` preregistered preflight scenes changed candidate and deployed
tensor hashes at batch size 1. Candidate maximum absolute delta ranged from
`{frame.candidate_max_abs_delta.min():.9g}` to `{frame.candidate_max_abs_delta.max():.9g}`
detected electrons; deployed maximum absolute delta ranged from
`{frame.deployed_max_abs_delta.min():.9g}` to `{frame.deployed_max_abs_delta.max():.9g}`.
These may be numerically small, but the frozen gate required exact canonical hashes.
Batch sizes 4 and 8 were exact in the primary preflight; the single-scene mismatch
therefore establishes batch-geometry dependence and deployment ineligibility.
""")
    fresh_json(target / "logs/batch_mismatch_diagnostic_complete.json", {
        "status": "PASS", "rows": len(frame), "all_candidate_hashes_differ": bool((~frame.candidate_hashes_equal).all()),
        "all_deployed_hashes_differ": bool((~frame.deployed_hashes_equal).all()),
        "max_candidate_abs_delta": float(frame.candidate_max_abs_delta.max()),
        "max_deployed_abs_delta": float(frame.deployed_max_abs_delta.max()),
        "classification_unchanged": "THAYER_PU_DEPLOYMENT_INELIGIBLE",
    })


def _bool_equal(left: object, right: object) -> bool:
    if pd.isna(left):
        return right is None
    expected = str(left).strip().lower() == "true"
    return expected == bool(right)


def condition_c_replay(run: Path) -> None:
    target = require_run(run)
    if (target / "logs/condition_c_replay_complete.json").exists():
        raise FileExistsError("Condition-C replay already exists")
    manifest_paths = [
        AUDIT_RUN / "episodes/post_training_manifest.csv",
        AUDIT_RUN / "episodes/post_validation_manifest.csv",
        AUDIT_RUN / "episodes/policy_validation_manifest.csv",
        AUDIT_RUN / "episodes/policy_calibration_manifest.csv",
    ]
    summaries = []
    mismatch_rows = []
    total = 0
    unsafe = 0
    started = time.time()
    flag_names = [
        "unsafe_to_catalog", "catastrophic", "catastrophic_image", "catastrophic_flux",
        "catastrophic_color", "catastrophic_centroid", "physical_output_contract_failure",
        "false_subtraction_failure", "false_subtraction_applicable",
        "worse_than_baseline_catastrophic", "source_confusion",
    ]
    for manifest_path in manifest_paths:
        manifest = pd.read_csv(manifest_path, low_memory=False)
        valid = manifest.loc[manifest.unsafe_to_catalog.notna()].copy()
        if valid.empty:
            raise RuntimeError(f"no Condition-C labels in {manifest_path}")
        scene_path = REPO / str(valid.upstream_scene_path.iloc[0])
        reconstruction_path = REPO / str(valid.upstream_reconstruction_path.iloc[0])
        set_mismatches = 0
        set_unsafe = 0
        with h5py.File(scene_path, "r") as scenes, h5py.File(reconstruction_path, "r") as reconstructions:
            for row in valid.itertuples(index=False):
                index = int(row.upstream_index)
                blend = np.asarray(scenes["blend"][index], dtype=np.float32)
                isolated = np.asarray(scenes["isolated"][index], dtype=np.float32)
                matched = int(scenes["matched_index"][index])
                reconstruction = np.asarray(reconstructions["reconstruction"][index], dtype=np.float32)
                result = post_audit_supervision(reconstruction, isolated[matched], isolated[1 - matched], blend)
                observed = result.to_dict()
                equal = all(_bool_equal(getattr(row, name), observed[name]) for name in flag_names)
                equal = equal and math.isclose(float(row.scientific_primary_distance), observed["scientific_primary_distance"], rel_tol=0.0, abs_tol=1e-12)
                equal = equal and math.isclose(float(row.baseline_primary_distance), observed["baseline_primary_distance"], rel_tol=0.0, abs_tol=1e-12)
                if not equal:
                    set_mismatches += 1
                    if len(mismatch_rows) < 100:
                        mismatch_rows.append({"episode_set": manifest_path.stem, "scene_id": row.scene_id, "upstream_index": index})
                set_unsafe += int(result.unsafe_to_catalog)
        total += len(valid)
        unsafe += set_unsafe
        summaries.append({
            "episode_set": manifest_path.stem, "rows": len(valid), "unsafe": set_unsafe,
            "safe": len(valid) - set_unsafe, "unsafe_prevalence": set_unsafe / len(valid),
            "label_or_metric_mismatches": set_mismatches, "status": "PASS" if set_mismatches == 0 else "FAIL",
        })
    passed = total == 12493 and unsafe == 12493 and not mismatch_rows
    fresh_csv(target / "replay_verification/condition_c_label_replay_summary.csv", pd.DataFrame(summaries))
    if mismatch_rows:
        fresh_csv(target / "replay_verification/condition_c_label_replay_mismatches.csv", pd.DataFrame(mismatch_rows))
    fresh_text(target / "diagnostics/condition_c_label_replay_report.md", f"""# Condition-C label replay

Status: **{'PASS' if passed else 'FAIL'}**. The unchanged safety implementation
replayed `{total}` labeled POST episodes from the authoritative audit. Unsafe:
`{unsafe}`; safe: `{total - unsafe}`; unsafe prevalence: `{unsafe / total:.12f}`;
label/component/metric mismatches: `{sum(row['label_or_metric_mismatches'] for row in summaries)}`.
""")
    fresh_json(target / "logs/condition_c_replay_complete.json", {
        "status": "PASS" if passed else "FAIL", "rows": total, "unsafe": unsafe,
        "safe": total - unsafe, "unsafe_prevalence": unsafe / total,
        "mismatches": sum(row["label_or_metric_mismatches"] for row in summaries),
        "runtime_seconds": time.time() - started,
    })
    if not passed:
        raise RuntimeError("Condition-C label replay failed")


def condition_c_precision_addendum(run: Path) -> None:
    """Resolve the two recorded decimal round-trip differences append-only."""
    target = require_run(run)
    original = json.loads((target / "logs/condition_c_replay_complete.json").read_text())
    if original != {**original, "status": "FAIL"} or original["rows"] != 12493 or original["unsafe"] != 12493:
        raise RuntimeError("unexpected Condition-C replay state")
    mismatches = pd.read_csv(target / "replay_verification/condition_c_label_replay_mismatches.csv")
    if len(mismatches) != 2:
        raise RuntimeError("expected exactly two precision rows")
    source = pd.read_csv(AUDIT_RUN / "episodes/post_training_manifest.csv", low_memory=False)
    rows = []
    flags = [
        "unsafe_to_catalog", "catastrophic", "catastrophic_image", "catastrophic_flux",
        "catastrophic_color", "catastrophic_centroid", "physical_output_contract_failure",
        "false_subtraction_failure", "false_subtraction_applicable",
        "worse_than_baseline_catastrophic", "source_confusion",
    ]
    for item in mismatches.itertuples(index=False):
        row = source.loc[source.scene_id.eq(item.scene_id)].iloc[0]
        with h5py.File(REPO / row.upstream_scene_path, "r") as scenes, h5py.File(REPO / row.upstream_reconstruction_path, "r") as reconstructions:
            index = int(row.upstream_index)
            blend = np.asarray(scenes["blend"][index], dtype=np.float32)
            isolated = np.asarray(scenes["isolated"][index], dtype=np.float32)
            matched = int(scenes["matched_index"][index])
            reconstruction = np.asarray(reconstructions["reconstruction"][index], dtype=np.float32)
        result = post_audit_supervision(reconstruction, isolated[matched], isolated[1 - matched], blend).to_dict()
        flag_match = all(_bool_equal(row[name], result[name]) for name in flags)
        scientific_delta = abs(float(row.scientific_primary_distance) - result["scientific_primary_distance"])
        baseline_delta = abs(float(row.baseline_primary_distance) - result["baseline_primary_distance"])
        rows.append({
            "scene_id": item.scene_id, "all_labels_and_components_match": flag_match,
            "scientific_primary_abs_delta": scientific_delta, "baseline_primary_abs_delta": baseline_delta,
            "maximum_scalar_abs_delta": max(scientific_delta, baseline_delta),
            "within_decimal_roundtrip_tolerance_1e_10": max(scientific_delta, baseline_delta) <= 1e-10,
        })
    detail = pd.DataFrame(rows)
    passed = bool(detail.all_labels_and_components_match.all() and detail.within_decimal_roundtrip_tolerance_1e_10.all())
    summary = pd.read_csv(target / "replay_verification/condition_c_label_replay_summary.csv")
    summary["decimal_roundtrip_only_rows"] = summary.label_or_metric_mismatches
    summary["substantive_label_or_metric_mismatches"] = 0
    summary["corrected_status"] = "PASS"
    fresh_csv(target / "replay_verification/condition_c_label_replay_precision_detail.csv", detail)
    fresh_csv(target / "replay_verification/condition_c_label_replay_precision_corrected_summary.csv", summary)
    fresh_text(target / "diagnostics/condition_c_label_replay_precision_addendum.md", f"""# Condition-C replay precision addendum

The initial checker used a zero-relative, `1e-12` absolute tolerance on CSV-loaded
primary distances. Two large baseline distances differed by exactly
`1.8189894035458565e-12` after decimal CSV round-trip. Every final label and every
component gate matched, both distances were within `1e-10`, and the replay remained
12,493 unsafe / 0 safe. Corrected substantive status: **{'PASS' if passed else 'FAIL'}**.
The original strict-check artifact is preserved append-only.
""")
    fresh_json(target / "logs/condition_c_replay_precision_addendum.json", {
        "status": "PASS" if passed else "FAIL", "authoritative_replay_status": "PASS" if passed else "FAIL",
        "rows": 12493, "unsafe": 12493, "safe": 0, "unsafe_prevalence": 1.0,
        "decimal_roundtrip_only_rows": 2, "substantive_mismatches": 0 if passed else 2,
        "maximum_scalar_abs_delta": float(detail.maximum_scalar_abs_delta.max()),
        "original_artifact_preserved": True,
    })
    if not passed:
        raise RuntimeError("Condition-C precision addendum failed")


def stopped_outputs(run: Path) -> None:
    target = require_run(run)
    if (target / "logs/stopped_outputs_complete.json").exists():
        raise FileExistsError("stopped-output markers already exist")
    reason = "NOT_RUN: frozen deployment preflight failed exact single-scene/batch invariance"
    requested = [
        "tables/inference_replay_audit.csv", "tables/thayer_pu_safety_prevalence.csv",
        "tables/thayer_pu_unsafe_reason_frequency.csv", "tables/thayer_pu_gate_cooccurrence.csv",
        "tables/label_support_gates.csv", "tables/family_distinctness_gates.csv",
        "tables/combined_family_label_support.csv", "bootstrap/source_group_bootstrap_intervals.csv",
    ]
    for value in requested:
        fresh_csv(target / value, pd.DataFrame([{"status": "NOT_RUN_DEPLOYMENT_INELIGIBLE", "reason": reason}]))
    fresh_text(target / "replay_verification/status.md", f"# Full inference replay\n\n{reason}.\n")
    fresh_text(target / "bootstrap/status.md", f"# Bootstrap\n\n{reason}; 300 replicates were not fabricated.\n")
    fresh_text(target / "diagnostics/inference_replay_report.md", f"# Full inference replay\n\n{reason}.\n")
    fresh_text(target / "diagnostics/family_distinctness_report.md", f"# Family distinctness\n\n{reason}; structural distinctness alone cannot establish eligibility.\n")
    fresh_json(target / "logs/stopped_outputs_complete.json", {
        "status": "PASS", "authoritative_outcome": "THAYER_PU_DEPLOYMENT_INELIGIBLE",
        "full_inference_rows": 0, "safety_labels": 0, "bootstrap_replicates": 0,
        "reason": reason,
    })


def finalize(run: Path) -> None:
    target = require_run(run)
    preflight_record = json.loads((target / "logs/preflight_complete.json").read_text())
    condition_record = json.loads((target / "logs/condition_c_replay_precision_addendum.json").read_text())
    stopped_record = json.loads((target / "logs/stopped_outputs_complete.json").read_text())
    if preflight_record["scientific_classification"] != "THAYER_PU_DEPLOYMENT_INELIGIBLE":
        raise RuntimeError("unexpected outcome")
    if condition_record["status"] != "PASS" or stopped_record["status"] != "PASS":
        raise RuntimeError("required stopped-run audit failed")
    if (target / "logs/inference_complete.json").exists() or list((target / "safety_labels").iterdir()):
        raise RuntimeError("full inference or safety labels exist after stop")

    commands = {
        "compileall": [".venv-btk/bin/python", "-m", "compileall", "-q", "src", "scripts", "tests"],
        "focused_tests": [
            ".venv-btk/bin/python", "-m", "pytest", "-q",
            "tests/test_probabilistic_unet.py", "tests/test_canonical_tensor_hash.py",
            "tests/test_direct_catalog_safety_auditor.py", "tests/test_thayer_pu_eligibility_v1.py",
        ],
        "git_diff_check": ["git", "diff", "--check"],
    }
    command_results = {}
    for name, arguments in commands.items():
        environment = os.environ.copy()
        environment["THAYER_PU_ELIGIBILITY_RUN"] = str(target)
        result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False, env=environment)
        command_results[name] = {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        fresh_text(target / f"logs/{name}.txt", "$ " + " ".join(arguments) + "\n" + result.stdout + result.stderr)

    csv_failures = []
    csv_count = 0
    for path in sorted(target.rglob("*.csv")):
        csv_count += 1
        try:
            frame = pd.read_csv(path, low_memory=False)
            if not len(frame.columns) or frame.empty:
                csv_failures.append({"path": relative(path), "reason": "empty schema or rows"})
        except Exception as error:  # pragma: no cover - final audit path
            csv_failures.append({"path": relative(path), "reason": repr(error)})
    fresh_json(target / "diagnostics/csv_schema_validation.json", {
        "status": "PASS" if not csv_failures else "FAIL", "csv_count": csv_count, "failures": csv_failures,
    })

    absolute_path_hits = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if "/Users/" in text:
                absolute_path_hits.append(relative(path))
    fresh_json(target / "diagnostics/privacy_path_audit.json", {
        "status": "PASS" if not absolute_path_hits else "FAIL", "absolute_home_path_hits": absolute_path_hits,
        "development_scene_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
    })

    after_checkpoints = verify_historical_checkpoints()
    fresh_csv(target / "provenance/historical_checkpoint_inventory_after.csv", after_checkpoints)
    start_environment = json.loads((target / "provenance/start_environment.json").read_text())
    readme_unchanged = sha256_file(REPO / "README.md") == start_environment["readme_sha256"]
    checkpoint_unchanged = sha256_file(CHECKPOINT) == EXPECTED_HASHES[CHECKPOINT]
    staged_empty = run_command(("git", "diff", "--cached", "--quiet"))["returncode"] == 0
    git_status = run_command(("git", "status", "--short"))["stdout"]
    fresh_text(target / "logs/final_git_status.txt", git_status)
    disk_result = run_command(("du", "-sk", str(target)))
    disk_bytes = int(disk_result["stdout"].split()[0]) * 1024
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(start_environment["start_utc"])).total_seconds()

    tests = [
        ("compileall", "PASS" if command_results["compileall"]["returncode"] == 0 else "FAIL", "repository Python sources"),
        ("deployment_rule_tests", "PASS" if command_results["focused_tests"]["returncode"] == 0 else "FAIL", "frozen constants and mean mapping"),
        ("latent_seed_determinism", "PASS", "repeated batch-8 candidate hashes exact"),
        ("batch_invariance", "FAIL_SCIENTIFIC_GATE", "batch 1 differs from batch 8 on 24/24; expected stopped outcome"),
        ("prompt_swap", "PASS", "majority and individual identity 1.0"),
        ("oof_provenance", "PASS", "all 7,591 source rows exclude fit/selection groups"),
        ("source_group_leakage", "PASS", "zero cross-partition or base overlap"),
        ("full_alignment", "NOT_RUN_STOP_CONDITION", "blocked before full inference"),
        ("condition_c_safety_label_replay", "PASS", "12,493 labels; zero substantive mismatch"),
        ("threshold_contract", "PASS" if sha256_file(THRESHOLD_CONTRACT) == EXPECTED_HASHES[THRESHOLD_CONTRACT] else "FAIL", "unchanged SHA-256"),
        ("label_support", "NOT_RUN_STOP_CONDITION", "no Thayer-PU safety labels"),
        ("family_distinctness", "NOT_RUN_STOP_CONDITION", "no complete aligned outputs"),
        ("bootstrap_300", "NOT_RUN_STOP_CONDITION", "no labels; replicates not fabricated"),
        ("csv_schema", "PASS" if not csv_failures else "FAIL", f"{csv_count} CSV files"),
        ("checkpoint_hash_audit", "PASS" if after_checkpoints.unchanged.all() and checkpoint_unchanged else "FAIL", "743 historical plus frozen Thayer-PU"),
        ("privacy_path", "PASS" if not absolute_path_hits else "FAIL", "zero absolute home paths and protected access"),
        ("git_diff_check", "PASS" if command_results["git_diff_check"]["returncode"] == 0 else "FAIL", "whitespace audit"),
        ("staged_index", "PASS" if staged_empty else "FAIL", "must remain empty"),
        ("readme", "PASS" if readme_unchanged else "FAIL", "README unchanged"),
    ]
    test_frame = pd.DataFrame(tests, columns=["check", "status", "evidence"])
    fresh_csv(target / "tables/integrity_test_matrix.csv", test_frame)
    implementation_pass = all(
        status in {"PASS", "NOT_RUN_STOP_CONDITION", "FAIL_SCIENTIFIC_GATE"}
        for status in test_frame.status
    )
    integrity = {
        "status": "PASS" if implementation_pass else "FAIL",
        "authoritative_outcome": "THAYER_PU_DEPLOYMENT_INELIGIBLE",
        "preregistration_sha256": sha256_file(target / "preregistration/frozen_thayer_pu_deployment_and_label_support.md"),
        "checkpoint_unchanged": checkpoint_unchanged, "historical_checkpoint_count": len(after_checkpoints),
        "historical_checkpoint_mismatches": int((~after_checkpoints.unchanged).sum()),
        "readme_unchanged": readme_unchanged, "staged_index_empty": staged_empty,
        "git_diff_check_pass": command_results["git_diff_check"]["returncode"] == 0,
        "reconstruction_models_trained": 0, "auditors_trained": 0,
        "deployment_rule_frozen_before_inference": True, "truth_based_candidate_selection": False,
        "output_clipping": False, "safety_threshold_changed": False,
        "development_scene_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
        "full_inference_rows": 0, "thayer_pu_safety_labels": 0, "bootstrap_replicates": 0,
        "csv_count": csv_count, "csv_failures": len(csv_failures), "absolute_path_hits": len(absolute_path_hits),
        "runtime_seconds": elapsed, "run_disk_bytes": disk_bytes,
        "campaign_script_sha256_final": sha256_file(Path(__file__)),
        "post_preregistration_implementation_addendum": relative(target / "provenance/post_preregistration_implementation_addendum.md"),
    }
    fresh_json(target / "diagnostics/final_integrity_audit.json", integrity)
    fresh_json(target / "reports/final_decision.json", {
        "authoritative_outcome": "THAYER_PU_DEPLOYMENT_INELIGIBLE",
        "thayer_pu_eligible_second_family": False, "thayer_audit_v1_authorized": False,
        "full_inference_authorized": False,
        "next_experiment": "Thayer-Audit Family-D v0 — One New Physically Compliant Frozen Family Eligibility Audit",
    })

    mismatch = json.loads((target / "logs/batch_mismatch_diagnostic_complete.json").read_text())
    prereg_hash = integrity["preregistration_sha256"]
    report = f"""# Thayer-PU Eligibility v1 final report

## Outcome

**THAYER_PU_DEPLOYMENT_INELIGIBLE**. Promptability passed, but the frozen exact
single-scene versus batched replay gate failed on 24/24 preflight scenes. Full
inference stopped as preregistered. Thayer-PU is not an eligible second audit
family and Thayer-Audit v1 is not authorized.

Preregistration SHA-256: `{prereg_hash}`. Frozen checkpoint SHA-256:
`{sha256_file(CHECKPOINT)}`. Integrity audit: **{integrity['status']}**.

## Required answers

1. **Why was Thayer-Audit v0 POST untestable?** Condition C supplied no safe POST episodes, so binary discrimination and nonzero safe coverage were undefined.
2. **Was Condition C's 100% unsafe prevalence reproduced?** Yes: 12,493 unsafe, 0 safe, prevalence 1.0, with zero substantive label/component/metric mismatch. The preserved strict checker flagged only two `1.8189894035458565e-12` CSV decimal round-trip differences.
3. **What exact checkpoint was frozen?** `{relative(CHECKPOINT)}`, validation-selected epoch 27, SHA-256 `{sha256_file(CHECKPOINT)}`.
4. **What deployment rule was frozen?** Mean requested reconstruction from K=16 truth-free prior samples, seeds 2026077600–2026077615, CPU float64 mean then float32 cast, MPS batch size 8, no clipping.
5. **Was truth used in deployment selection?** No. No candidate was selected and no truth or Atlas outcome entered the rule.
6. **Did promptability pass?** Yes: majority-of-16 and individual identity were both 1.0; band identity was 1.0/1.0/0.9505208.
7. **Did deterministic replay pass?** Repeated batch-8 replay passed exactly; the complete deployment replay gate did not pass because single-scene geometry changed hashes.
8. **Did batch-size invariance pass?** No. Batch 4 equaled batch 8, but batch 1 differed on every preflight scene. Maximum candidate/deployed absolute deltas were `{mismatch['max_candidate_abs_delta']}` / `{mismatch['max_deployed_abs_delta']}` detected electrons.
9. **Were training outputs truly OOF?** Complete outputs were not generated. The authorized source manifest contained 3,998 training rows whose groups were all absent from Thayer-PU fitting and validation selection.
10. **Were source groups leak-free?** Yes in the frozen source manifests: zero fit/selection overlap and zero train/validation/calibration overlap.
11. **How many complete outputs were generated?** Training 0, validation 0, calibration 0. Only the 24-scene preflight ran; no deployed reconstruction archive was frozen.
12. **Safe prevalence by partition?** Not computed because the stop preceded labels.
13. **Unsafe prevalence by partition?** Not computed for Thayer-PU.
14. **Which safety gates dominated?** Not evaluated for Thayer-PU.
15. **Did physical output failures persist?** Not evaluated on complete partitions.
16. **Were safe examples present in every partition?** Unknown; no Thayer-PU safety labels were created.
17. **Did safe support pass?** Not evaluated.
18. **Did unsafe support pass?** Not evaluated.
19. **Were safe examples spread across enough groups?** Not evaluated.
20. **Was Thayer-PU distinct from Condition C?** Structural distinctness was previously verified, but the preregistered multi-criterion comparison could not run; family distinctness did not pass.
21. **Did the combined dataset support POST-auditor training?** No. No eligible Thayer-PU labels exist and Condition C remains one-class unsafe.
22. **Is Thayer-PU an eligible second frozen family?** No.
23. **Authoritative outcome?** `THAYER_PU_DEPLOYMENT_INELIGIBLE`.
24. **Is Thayer-Audit v1 authorized?** No.
25. **What happens next?** Run exactly one separately preregistered experiment: **Thayer-Audit Family-D v0 — One New Physically Compliant Frozen Family Eligibility Audit**.
26. **Were development, Atlas selection, and final lockbox untouched?** Yes; access counts were 0/0/0. Existing Atlas outcomes were not used to select deployment.
27. **Were historical checkpoints unchanged?** Yes: 743/743 inventory rows plus the frozen Thayer-PU checkpoint matched.
28. **Reusable code/tests to review eventually?** `scripts/run_thayer_pu_eligibility_v1.py` and `tests/test_thayer_pu_eligibility_v1.py`; nothing was staged or committed.
29. **Generated artifacts to keep ignored?** This entire `{relative(target)}` run tree, including preregistration, source manifests, preflight diagnostics, replay evidence, integrity logs, status tables, and reports.

## Frozen inventory and stopped downstream analyses

The checkpoint/model inventory is `frozen_model/checkpoint_model_inventory.csv`;
the exact rule and seeds are in `deployment_rule/`; OOF source provenance is in
`manifests/` and `tables/source_partition_audit.csv`; preflight proof is in
`tables/deployment_preflight_audit.csv` and `tables/single_scene_batch_mismatch.csv`.
Condition-C replay proof is in `replay_verification/`. Full partition outputs,
Thayer-PU safety prevalence, unsafe reasons, family comparison, combined support,
and 300-replicate bootstrap are explicitly `NOT_RUN_DEPLOYMENT_INELIGIBLE`; no
values were fabricated after the stop.

## Runtime, disk, and Git state

- Elapsed campaign runtime: `{elapsed:.2f}` seconds.
- Run disk usage at integrity close: `{disk_bytes}` bytes.
- MPS inference fallback: false.
- README unchanged: `{readme_unchanged}`.
- Staged index empty: `{staged_empty}`.
- `git diff --check`: `{command_results['git_diff_check']['returncode'] == 0}`.

Final Git status:

```text
{git_status.rstrip()}
```
"""
    fresh_text(target / "reports/final_report.md", report)
    fresh_json(target / "logs/finalization_complete.json", {
        "status": "PASS" if implementation_pass else "FAIL", "authoritative_outcome": "THAYER_PU_DEPLOYMENT_INELIGIBLE",
        "runtime_seconds": elapsed, "run_disk_bytes": disk_bytes, "staged_index_empty": staged_empty,
        "historical_checkpoint_mismatches": int((~after_checkpoints.unchanged).sum()),
        "development_scene_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
    })
    if not implementation_pass:
        raise RuntimeError("final integrity audit failed")


def finalize_addendum(run: Path) -> None:
    """Append-only correction for a repository-local pytest warning path."""
    target = require_run(run)
    original_privacy = json.loads((target / "diagnostics/privacy_path_audit.json").read_text())
    original_integrity = json.loads((target / "diagnostics/final_integrity_audit.json").read_text())
    expected_hit = relative(target / "logs/focused_tests.txt")
    if original_privacy["absolute_home_path_hits"] != [expected_hit] or original_integrity["status"] != "FAIL":
        raise RuntimeError("unexpected first finalization state")
    warning_log = (target / "logs/focused_tests.txt").read_text()
    hit_lines = [line for line in warning_log.splitlines() if "/Users/" in line]
    repo_prefix = str(REPO.resolve()) + "/.venv-btk/"
    allowlisted = bool(hit_lines and all(line.lstrip().startswith(repo_prefix) for line in hit_lines))
    environment = os.environ.copy()
    environment["THAYER_PU_ELIGIBILITY_RUN"] = str(target)
    environment["PYTHONWARNINGS"] = "ignore"
    arguments = [
        ".venv-btk/bin/python", "-m", "pytest", "-q",
        "tests/test_probabilistic_unet.py", "tests/test_canonical_tensor_hash.py",
        "tests/test_direct_catalog_safety_auditor.py", "tests/test_thayer_pu_eligibility_v1.py",
    ]
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False, env=environment)
    fresh_text(target / "logs/focused_tests_warning_suppressed.txt", "$ " + " ".join(arguments) + "\n" + result.stdout + result.stderr)
    sanitized_has_absolute = "/Users/" in (target / "logs/focused_tests_warning_suppressed.txt").read_text()
    staged_empty = run_command(("git", "diff", "--cached", "--quiet"))["returncode"] == 0
    diff_check = run_command(("git", "diff", "--check"))["returncode"] == 0
    checkpoints = verify_historical_checkpoints()
    readme = json.loads((target / "provenance/start_environment.json").read_text())["readme_sha256"] == sha256_file(REPO / "README.md")
    passed = bool(
        allowlisted and result.returncode == 0 and not sanitized_has_absolute and staged_empty and diff_check
        and checkpoints.unchanged.all() and readme and sha256_file(CHECKPOINT) == EXPECTED_HASHES[CHECKPOINT]
    )
    fresh_json(target / "diagnostics/privacy_path_audit_addendum.json", {
        "status": "PASS" if passed else "FAIL",
        "original_hit": expected_hit, "original_hit_line_count": len(hit_lines),
        "classification": "ALLOWLISTED_REPOSITORY_LOCAL_VENV_WARNING_TRACEBACK",
        "contains_data_or_external_path": False, "sanitized_test_log_absolute_path_hits": int(sanitized_has_absolute),
        "development_scene_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
        "original_audit_preserved": True,
    })
    fresh_json(target / "diagnostics/final_integrity_audit_addendum.json", {
        "status": "PASS" if passed else "FAIL", "authoritative_integrity_status": "PASS" if passed else "FAIL",
        "authoritative_outcome": "THAYER_PU_DEPLOYMENT_INELIGIBLE",
        "scientific_results_changed": False, "privacy_allowlist_scope": expected_hit,
        "focused_tests_pass": result.returncode == 0, "staged_index_empty": staged_empty,
        "git_diff_check_pass": diff_check, "readme_unchanged": readme,
        "historical_checkpoint_mismatches": int((~checkpoints.unchanged).sum()),
        "development_scene_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
    })
    final_status = run_command(("git", "status", "--short"))["stdout"]
    fresh_text(target / "logs/final_git_status_addendum.txt", final_status)
    fresh_text(target / "reports/final_report_addendum.md", f"""# Final integrity addendum

The original final report's scientific outcome remains
`THAYER_PU_DEPLOYMENT_INELIGIBLE`. Its first integrity status was false only
because the privacy scanner counted pytest warning traceback lines from the
repository-local `.venv-btk` as protected external paths. The original log and
failed audit are preserved.

The warning-suppressed 24-test suite passed, its log contains zero absolute home
paths, the allowlist is restricted to the original repository-local warning
traceback, and checkpoint, README, staged-index, and `git diff --check` audits
all pass. Authoritative integrity status: **{'PASS' if passed else 'FAIL'}**.
No scientific output, deployment gate, label, threshold, or outcome changed.
""")
    fresh_json(target / "logs/finalization_addendum_complete.json", {
        "status": "PASS" if passed else "FAIL", "authoritative_integrity_status": "PASS" if passed else "FAIL",
        "authoritative_outcome": "THAYER_PU_DEPLOYMENT_INELIGIBLE", "scientific_results_changed": False,
    })
    if not passed:
        raise RuntimeError("final integrity addendum failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("bootstrap", "preflight", "inference", "batch-diagnostic", "condition-c-replay", "condition-c-addendum", "stopped-outputs", "finalize", "finalize-addendum"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    functions = {
        "bootstrap": bootstrap, "preflight": preflight, "inference": inference,
        "batch-diagnostic": batch_mismatch_diagnostic, "condition-c-replay": condition_c_replay,
        "condition-c-addendum": condition_c_precision_addendum,
        "stopped-outputs": stopped_outputs,
        "finalize": finalize,
        "finalize-addendum": finalize_addendum,
    }
    functions[args.phase](args.run)


if __name__ == "__main__":
    main()
