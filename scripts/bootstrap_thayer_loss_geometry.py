#!/usr/bin/env python3
"""Create and preregister the append-only Thayer-LG frozen audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
PROMPT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
ATLAS_CONTRACT = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"

SCENES = MH / "manifests/probabilistic_unet_training_scenes.h5"
TARGETS = MH / "target_sets/thayer_mh_training_target_sets.h5"
DEFINITIONS = MH / "manifests/probabilistic_unet_scene_definitions.csv"
MICRO_MANIFEST = MICRO / "tables/microset_manifest.csv"
TRAINED_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
ME_CHECKPOINT = MICRO / "checkpoints/thayer_me_micro_final.pth"
MH_BEST = MH / "checkpoints/thayer_mh_best.pth"
MH_FINAL = MH / "checkpoints/thayer_mh_final.pth"
NORMALIZATION = PROMPT / "manifests/normalization.json"
FORWARD_THRESHOLDS = PU / "manifests/forward_consistency_thresholds.json"
NOISE_CONTRACT = ATLAS_CONTRACT / "manifests/fixed_noise_contract.json"

SUBDIRECTORIES = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "microset", "objective_paths", "gradients", "output_space_optimization",
    "example_grids", "figures/loss_term_contributions", "figures/objective_paths",
)

FROZEN_WEIGHTS = {
    "requested_source_reconstruction": 1.0,
    "companion_source_reconstruction": 1.0,
    "source_sum_inside_decomposition_cost": 0.5,
    "ordinary_concentration_inside_target_loss": 0.10,
    "target_set_top_level": 1.0,
    "forward_consistency_top_level": 0.5,
    "prompt_swap_top_level": 0.25,
    "pair_equivalence_top_level": 0.05,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = str(path.relative_to(REPO)).encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
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


def verify_authoritative_inputs() -> None:
    if command(["git", "diff", "--cached", "--name-only"]):
        raise RuntimeError("staged index is nonempty")

    me_provenance = json.loads((ME / "logs/input_provenance.json").read_text())
    expected: dict[Path, str] = {
        SCENES: "d6ca6f1cbcb136a075f0216460e5f6b2dcd5fefbb63894803b86069df4e5f48d",
        TARGETS: "7fc92222ff2d980c4beb787b961fa7bdaf3130c055ce842dc8fd5f600c29c19a",
        DEFINITIONS: "2dc7e17ef83bc02ed59bd0947c7b8f403e1ab7f933a2d753f61c2bc77d81ba64",
        MICRO_MANIFEST: "9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085",
        TRAINED_OUTPUTS: "612d02fc72686ccde704ebf56bdb10c30af28fde3d20939afebac3fa34553446",
        ME_CHECKPOINT: "f96648ebd990dceaab89b6e10e518701fd9126a17df45cc7eb2795968a4a5757",
        MH_BEST: me_provenance["thayer_mh_checkpoints"]["best"],
        MH_FINAL: me_provenance["thayer_mh_checkpoints"]["final"],
        NORMALIZATION: "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
        FORWARD_THRESHOLDS: sha256_file(FORWARD_THRESHOLDS),
        NOISE_CONTRACT: "3ce4435330da83eace363ceee3856612e100f43b63d2493aed7441992494ec7b",
    }
    micro_frozen = json.loads((MICRO / "logs/microset_frozen_before_fit.json").read_text())
    if expected[SCENES] != micro_frozen["scene_tensor_sha256"] or expected[TARGETS] != micro_frozen["target_tensor_sha256"]:
        raise RuntimeError("authoritative microset commitments disagree")
    if expected[MICRO_MANIFEST] != micro_frozen["manifest_sha256"]:
        raise RuntimeError("authoritative microset manifest commitment disagrees")
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"frozen input altered: {path.relative_to(REPO)}")

    mh_audit = json.loads((MH / "diagnostics/final_correctness_audit.json").read_text())
    me_audit = json.loads((ME / "diagnostics/final_correctness_audit_superseding.json").read_text())
    if mh_audit["failure_count"] or me_audit["failure_count"]:
        raise RuntimeError("authoritative correctness audit unresolved")
    if any(me_audit[key] != 0 for key in ("atlas_evaluation_count", "development_scene_access_count", "lockbox_scene_access_count")):
        raise RuntimeError("protected data-access boundary is not zero")

    expected_inventory = read_csv(ME / "tables/checkpoint_inventory_after.csv")
    for row in expected_inventory:
        path = REPO / row["path"]
        expected_digest = row.get("after_sha256", row.get("observed_sha256", ""))
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise RuntimeError(f"historical checkpoint altered: {row['path']}")
    current = sorted((REPO / "outputs/runs").rglob("*.pth"))
    expected_paths = {str((REPO / row["path"]).resolve()) for row in expected_inventory}
    expected_paths.add(str(ME_CHECKPOINT.resolve()))
    if {str(path.resolve()) for path in current} != expected_paths:
        raise RuntimeError("historical checkpoint set differs from the frozen 574 plus Thayer-ME micro checkpoint")


def checkpoint_inventory() -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "status": "PASS",
        }
        for path in sorted((REPO / "outputs/runs").rglob("*.pth"))
    ]


def source_inventory() -> list[dict[str, object]]:
    rows = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def preregistration_text(started: str, micro_hash: str) -> str:
    return f"""# Frozen Loss-Geometry Audit preregistration

Working name: **Thayer-LG (Thayer Loss Geometry)**  
Frozen at UTC: `{started}`  
Audited microset manifest SHA-256: `{micro_hash}`

## Scientific boundary

This is a training-free audit of persisted Thayer-ME microset inputs, approved target sets, and trained outputs. Neural parameters are immutable and may not receive gradients or optimizer steps. Automatic differentiation is allowed only with respect to detached output tensors. Atlas inference, validation/calibration evaluation, historical development evaluation, and final-lockbox access are forbidden. Results cannot be used to alter a loss weight, target set, source-layer semantic, or coverage threshold in this campaign.

## Frozen rows and contracts

The audited rows are exactly the 64 rows in the immutable Thayer-ME microset manifest: the first 32 training ordinary rows selected by the original selector and both observations of the first 16 lexicographically sorted approved training pairs. Both prompts are retained. Outputs are float32 N x 2-expert x 6-channel x 60 x 60 tensors; channels 0:3 are requested g/r/z and 3:6 are companion g/r/z. Values are unclipped normalized source layers, inverted with the frozen training-only g/r/z scales. Background is exactly zero.

The frozen objective is the implementation in `src/models_two_expert_decoder.py` as invoked by `scripts/run_thayer_two_expert_micro_overfit.py`: decomposition cost = requested MSE + companion MSE + 0.5 source-sum MSE; ordinary scenes supervise both experts to target zero and add 0.10 expert concentration; ambiguous scenes use hard min(identity, swap); prompt losses are averaged across A/B; total = target + 0.5 forward + 0.25 prompt-swap + 0.05 paired-observation set equivalence. No additional regularizer or target-aware separation term exists. Weights are immutable.

For scene-level accounting, paired-observation equivalence is assigned to both members of each ambiguous pair with effective coefficient 0.10; its mean over all 64 rows therefore equals the original 0.05 pair-average contribution. Aggregate reproduction also evaluates the original batch-equivalent formula directly.

## Canonical configurations

Ordinary: O1 exact target duplicated; O2 persisted trained experts; O3 trained expert mean duplicated; O4 all-zero; O5 exact target duplicated after transferring 25% of requested light into the companion layer while preserving the source sum. Ambiguous: A1 exact approved set in stored order; A2 the same set with expert slots reversed; A3 persisted trained experts; A4 both experts equal the pixelwise mean of the approved set; A5 both equal the persisted expert mean; A6 own target duplicated; A7 alternate target duplicated; A8 each exact decomposition is replaced by a 50/50 requested/companion allocation at fixed source sum. The corresponding prompt-specific target tensors define prompt A/B mappings.

## Distances and coverage

The primary scientific distance is the frozen `scientific_distance` maximum of image NRMSE/0.25, per-band relative flux/0.20, valid g-r and r-z color differences/0.20, and centroid displacement in mean-PSF units/0.5. Coverage requires primary distance <= 1.0 and frozen forward plausibility. Image, flux, color, centroid, and forward metrics are reported separately. The differentiable scientific surrogate uses the maximum of the same image, relative-flux, valid-color, and centroid components with epsilons fixed by the implementation; at nondifferentiable ties PyTorch's deterministic subgradient is accepted.

## Paths and perturbations

All interpolation grids are frozen to 21 equally spaced points from 0 through 1. Truth-to-trained is affine. Truth-set-to-collapsed moves each expert toward the approved-set mean. Source-light transfer uses 21 fractions from -0.5 through 0.5; positive values move requested light to companion and negative values move companion light to requested, always preserving the sum. Expert separation moves duplicated set mean toward the two exact targets. Flux-preserving morphology mixes each requested source with its one-pixel positive-x roll and rescales every band to its original total when the denominator magnitude exceeds 1e-12.

Assignment perturbations use deterministic Gaussian directions with seed 2026071301 and scales 1e-7, 1e-6, 1e-5, and 1e-4 normalized units. Finite-difference gradient checks use central step 1e-4. Directional curvature uses central steps 1e-3 and reports `(L(x+h d)-2L(x)+L(x-h d))/h^2` for unit-L2 directions. A flat-direction flag is absolute curvature <= 1e-6; weak curvature is <= 1e-4. Gradient cosine uses denominator floor 1e-20; two zero gradients are reported as undefined, not aligned.

Potential dominance is flagged when one weighted term supplies >= 0.75 of the sum of individual weighted gradient L2 norms, or when one band/layer supplies >= 0.75. A gradient conflict is negative cosine; severe conflict is <= -0.5. Assignment instability is any slot flip under perturbation <= 1e-5 or any interpolation assignment margin <= 1e-7.

## Output-space optimization

All neural state is absent from the optimization graph. Free output tensors use CPU float32 Adam, seed 2026071302, learning rate 0.01, 40 updates, logging every 5 updates, and elementwise diagnostic bounds [-8, 8] applied after every step. D0 runs from exact truth, persisted trained outputs, collapsed/duplicated mean, uniform random [0,1], and the source-sum-preserving compromise. D1 target reconstruction/set matching only; D2 target plus ordinary concentration (identical to the exact implemented target loss, reported separately for clarity); D3 target plus 0.5 forward; and D4 removes each of target, forward, prompt-swap, and pair-equivalence once. D1-D4 start from persisted trained outputs. These are diagnostics, never model fitting or future-loss selection.

## Decision gates

Any frozen-input mismatch stops the campaign. Any exact truth that cannot be constructed, hashed, mapped across prompts, pass its named coverage check, satisfy ordinary duplication concentration, or satisfy frozen forward plausibility stops interpretation and is classified as OUTPUT-CONTRACT DEFECT or COVERAGE-METRIC DEFECT. Otherwise the primary category is selected exactly from the ten user-specified categories. OBJECTIVE MISALIGNMENT requires direct lower frozen objective for a compromise/trained output than approved truth or a truth-started D0 trajectory that leaves coverage while lowering objective. LOSS-SCALE DOMINANCE, GRADIENT CONFLICT, and PERMUTATION-MATCHING PATHOLOGY require the thresholds above. OPTIMIZATION/NETWORK BOTTLENECK requires truth representability, truth objective optimality, and direct optimization reaching truth while the neural microfit failed. MIXED CAUSE is used only when multiple categories have direct evidence. Exactly one future experiment will be recommended and not run.

## Numerical and access guarantees

CSV floats retain at least 10 significant digits. Equality reproduction tolerances are absolute 1e-7 for persisted aggregate rates and 1e-6 for persisted expert diameter; tensor reconstruction tolerance is max absolute 1e-6 physical electrons and 1e-7 normalized units where exact arithmetic is expected. SHA-256 is used for files and canonical little-endian float32 CHW tensors. No loss weight or threshold may be tuned after inspecting results. Access counters for Atlas, development, and lockbox remain zero.
"""


def main() -> None:
    verify_authoritative_inputs()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_loss_geometry_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    checkpoints = checkpoint_inventory()
    sources = source_inventory()
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", checkpoints)
    write_csv_fresh(run_dir / "tables/source_code_hashes_before.csv", sources)

    import btk
    import galsim
    import h5py
    import numpy
    import scipy
    import torch

    started = datetime.now(timezone.utc).isoformat()
    disk = shutil.disk_usage(REPO)
    mps = {
        "built": bool(torch.backends.mps.is_built()),
        "available": bool(torch.backends.mps.is_available()),
        "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1",
    }
    if mps["fallback_enabled"]:
        raise RuntimeError("MPS fallback environment is prohibited")

    loss_files = [REPO / "src/models_two_expert_decoder.py", REPO / "scripts/run_thayer_two_expert_micro_overfit.py"]
    coverage_files = [REPO / "src/competing_hypotheses.py", REPO / "scripts/run_thayer_two_expert_micro_overfit.py"]
    forward_files = [REPO / "src/competing_hypotheses.py", REPO / "src/btk_scene.py", FORWARD_THRESHOLDS, NOISE_CONTRACT]
    relevant = {
        "thayer_mh_best_checkpoint": MH_BEST,
        "thayer_mh_final_checkpoint": MH_FINAL,
        "thayer_me_micro_checkpoint": ME_CHECKPOINT,
        "microset_manifest": MICRO_MANIFEST,
        "training_scene_tensor": SCENES,
        "training_target_set": TARGETS,
        "scene_definitions": DEFINITIONS,
        "trained_expert_outputs": TRAINED_OUTPUTS,
        "source_layer_contract": REPO / "docs/multi_hypothesis_source_contract.md",
        "prompt_swap_contract": REPO / "docs/thayer_select_prompt_semantics.md",
        "forward_consistency_contract": REPO / "docs/forward_consistency_contract.md",
        "truth_coverage_contract": REPO / "docs/latent_truth_coverage.md",
        "ambiguity_set_contract": REPO / "docs/ambiguity_set_supervision.md",
        "expert_specialization_contract": REPO / "docs/expert_specialization_contract.md",
        "normalization": NORMALIZATION,
        "forward_thresholds": FORWARD_THRESHOLDS,
        "noise_contract": NOISE_CONTRACT,
    }
    provenance = {
        "campaign": "Frozen Loss-Geometry Audit",
        "working_experiment_name": "Thayer-LG",
        "campaign_started_utc": started,
        "run_dir": str(run_dir.relative_to(REPO)),
        "branch": command(["git", "branch", "--show-current"]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_v2": command(["git", "status", "--porcelain=v2", "--untracked-files=all"]).splitlines(),
        "staged_index": command(["git", "diff", "--cached", "--name-status"]).splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": numpy.__version__, "torch": torch.__version__, "btk": btk.__version__,
            "galsim": galsim.__version__, "h5py": h5py.__version__, "scipy": scipy.__version__,
            "pandas": package_version("pandas"), "matplotlib": package_version("matplotlib"),
        },
        "mps": mps,
        "free_disk_bytes": disk.free,
        "relevant_artifacts": {name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for name, path in relevant.items()},
        "loss_implementation": {"files": [str(path.relative_to(REPO)) for path in loss_files], "combined_sha256": combined_hash(loss_files)},
        "loss_weight_configuration": FROZEN_WEIGHTS,
        "loss_weight_configuration_sha256": sha256_json(FROZEN_WEIGHTS),
        "truth_coverage_implementation": {"files": [str(path.relative_to(REPO)) for path in coverage_files], "combined_sha256": combined_hash(coverage_files)},
        "forward_model": {"files": [str(path.relative_to(REPO)) for path in forward_files], "combined_sha256": combined_hash(forward_files)},
        "historical_checkpoint_count": len(checkpoints),
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)

    prereg_path = run_dir / "preregistration/frozen_loss_geometry_audit.md"
    write_text_fresh(prereg_path, preregistration_text(started, sha256_file(MICRO_MANIFEST)))
    prereg_hash = sha256_file(prereg_path)
    frozen_at = datetime.now(timezone.utc).isoformat()
    write_json_fresh(run_dir / "preregistration/freeze_record.json", {
        "status": "FROZEN_BEFORE_PER_SCENE_NUMERICAL_INSPECTION",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": prereg_hash,
        "input_provenance_sha256": sha256_file(run_dir / "logs/input_provenance.json"),
        "microset_manifest_sha256": sha256_file(MICRO_MANIFEST),
        "loss_implementation_sha256": provenance["loss_implementation"]["combined_sha256"],
        "loss_weight_configuration_sha256": provenance["loss_weight_configuration_sha256"],
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })

    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", f"""# Thayer-LG environment snapshot

- Campaign start UTC: `{started}`
- Preregistration freeze UTC: `{frozen_at}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python: `{sys.version.splitlines()[0]}`
- Torch / BTK / GalSim: `{torch.__version__}` / `{btk.__version__}` / `{galsim.__version__}`
- MPS built / available: `{mps['built']}` / `{mps['available']}`; fallback disabled.
- Free disk at start: `{disk.free}` bytes.
- Staged index: empty.
- Historical checkpoints: {len(checkpoints)} hashed and matched to the frozen inventory.
- Atlas / development / lockbox access: 0 / 0 / 0.
""")
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", """# Thayer-LG campaign contract

Thayer-LG is a training-free, append-only audit of the exact frozen Thayer-ME objective on persisted microset tensors and outputs. It asks whether approved truth decompositions receive favorable objective values and gradients relative to forward-consistent compromises. Neural weights are immutable; only detached free output tensors may be differentiated or optimized. Exact-truth representability is a hard gate before any loss interpretation. No Atlas, validation, calibration, historical development, or lockbox row may be opened. No result authorizes a loss-weight change or follow-on training inside this campaign.
""")
    write_json_fresh(run_dir / "logs/part_a_b_complete.json", {
        "status": "PASS_AND_PREREGISTERED",
        "run_dir": str(run_dir.relative_to(REPO)),
        "preregistration_sha256": prereg_hash,
        "frozen_at_utc": frozen_at,
        "historical_checkpoint_count": len(checkpoints),
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    print(run_dir)


if __name__ == "__main__":
    main()
