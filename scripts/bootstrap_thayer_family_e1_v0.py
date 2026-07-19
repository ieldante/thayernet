#!/usr/bin/env python3
"""Create and freeze the append-only Thayer-Family-E1-v0 master run."""
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
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs/runs"
PREFLIGHT = REPO / "outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340"
FAMILY_E = REPO / "outputs/runs/thayer_family_e_v0_20260714_195256"
HIERARCHICAL = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
THAYER_PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340/checkpoints/thayer_pu_best.pth"
EXPECTED = {
    "preflight_report": "28c3d91501616d8c250873bdd445199282f933f0536514586b3bc75f1d8821f2",
    "preflight_physical": "83bfc71a8efef88e9cf76b771b10e4f60e0e34c4a1c8bb87821c4c2f1cf9cc62",
    "preflight_doc": "31f9e3816b1b6290124bd37ad190155aa22fff97dc6afb8ae3d4e16994ed9d69",
    "preflight_prereg": "be546f7f1aa2ec04f1a76f84bc5305c87521d5b89331c681dc3cdf18a5293d3b",
    "training_selector": "4a8768eaa70e1d3f5f7a29fd4035e994c9c6f1494d3553e6ac0f805c8e911bc1",
    "validation_selector": "bc5c65ffab19baea38e37edcb4d5dabd15bae1c0266b7dfdaa749eba5c6c464d",
    "calibration_selector": "70326c1835726677e5d98c50323329f919bcd405f0f379420987fcd97e20fa0c",
    "training_manifest": "6c20d846709987c96c3d27c586756f1f48d75904a9e285ffd48d9b0a7b047ac3",
    "validation_manifest": "acdb4071cb0c3b2eb67e9d9f26f0dd43f0ea76872efd137c2e067386cdf82413",
    "calibration_manifest": "7fbfa02ce5d73ceefd4ce6478b5c6ea8b87de8745536ca4c6d4aff9ac348c74f",
    "training_h5": "a9efead2293b47afca61c1a156ac0fed9cdd4bc1c5920e197a581022c5fa0f22",
    "validation_h5": "5a29100a96a1c01d657e91e68430809a68794fe647fee20012ca4d542933ab17",
    "calibration_h5": "99392093cc096b467bcee840e9af88f8600d620130422a536893a8f35a705b10",
    "prompt": "449079faf20a29a1c65cd9c5916d1cffe641b4ef0ac5293ca9987cf2c3904fb7",
    "source_layer": "3fbd8c019a0489106ec0be8efc1cbe0a152c36fd022e928673813c9bab74303f",
    "threshold": "ac6c4585d214008c03b19b6b61b69dee999242d02d7e1cb724caf5fffa7320e3",
    "safety": "9efe750a60d746cd6cd496c6843a9a4f62500016ed599d1a1a84e4edac199df4",
    "normalization": "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    "condition_c": "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
    "thayer_pu": "c1d17a3f67962cce2fec03d6b15da5f2e330ee97b31c270a7ff019a1373a557e",
    "readme": "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1",
}
DIRECTORIES = (
    "diagnostics", "logs", "reports", "preregistration", "provenance",
    "manifests", "architecture", "physical_contract", "objective_audit",
    "micro_overfit", "training", "validation", "calibration", "inference",
    "replay_verification", "oof_outputs", "episodes", "safety_labels",
    "tables", "figures", "bootstrap", "checkpoints", "family_comparison",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def command(args: list[str]) -> str:
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed: {args}: {result.stderr}")
    return result.stdout


def verify(path: Path, expected: str, name: str) -> dict[str, object]:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"authoritative input mismatch for {name}: {observed} != {expected}")
    return {"name": name, "path": relative(path), "sha256": observed, "status": "PASS"}


def package_versions() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    for package in ("numpy", "pandas", "h5py", "scipy", "matplotlib", "astropy", "blending-toolkit", "GalSim"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "NOT_INSTALLED"
    return result


def preregistration(now_utc: str) -> str:
    return f"""# Thayer-Family-E1-v0 preregistration

Frozen UTC: `{now_utc}`

Status: **FROZEN BEFORE MODEL CONSTRUCTION, MODEL-SOURCE IMPORT, TRAINING-TENSOR LOAD, OR FITTING**

## Hypothesis and boundaries

The sole hypothesis is that one compact coordinate-conditioned reconstruction model with nonnegative requested and companion source outputs and a derived signed residual can create meaningful safe and unsafe support under the unchanged Thayer-Audit v0 gates. This campaign trains exactly one architecture family. It does not train a POST auditor, tune from safety labels, access development, use Atlas for selection, or access the final lockbox.

The authoritative signed-residual preflight outcome must remain exactly `SIGNED_NOISE_RESIDUAL_CONTRACT_PASS`. Requested and companion are catalog-source layers; the signed residual is observational noise/background closure and is not a catalog source.

## Frozen data

Use the exact Family-E selectors: training 10,000 (`{EXPECTED['training_selector']}`), validation 2,000 (`{EXPECTED['validation_selector']}`), and calibration 2,000 (`{EXPECTED['calibration_selector']}`). The upstream manifests and HDF5 tensors retain their authoritative hashes. Cross-partition source-group overlap, source-pair overlap, and duplicate source pairs are zero. Prompt support is training 8,515 `UNIQUE_VALID` plus 1,485 `PERTURBED_VALID`, validation 1,429 plus 571, and calibration 1,400 plus 600. No source/group ID or target-derived difficulty variable is an input.

Five connected-source-group folds from the immutable training selector are used. Each contains 2,000 episodes; no source group crosses folds. Fold models exclude both source groups of every held episode and use identical settings. OOF fold seed is `2026071501`.

## Inputs and normalization

Input is exactly normalized observed g/r/z followed by the stored unit-peak sigma-2-pixel Gaussian coordinate prompt: four channels. Frozen positive g/r/z scales are `611.9199829101562, 1805.8800048828125, 1854.199951171875`. Observations and targets are never clipped or offset. Targets are used only for supervised fitting and evaluation.

## Sole architecture

One compact coordinate U-Net is frozen. Encoder widths are `24, 48, 96, 128`. Each stage has two bias-free 3x3 convolutions, GroupNorm(8), and SiLU. Between stages a bias-free stride-2 3x3 convolution changes width and is followed by GroupNorm(8) and SiLU. The mirrored decoder bilinearly upsamples to each skip shape, applies a bias-free 3x3 width-changing convolution plus GroupNorm(8)/SiLU, concatenates the skip, then applies two bias-free 3x3 convolutions with GroupNorm(8)/SiLU. The biased 1x1 head maps 24 channels to six normalized logits ordered requested g/r/z then companion g/r/z. Head bias initializes to `0.01`; other convolutions use Kaiming-normal initialization, GroupNorm scale one/bias zero. BatchNorm, attention, transformers, recurrence, latent variables, stochastic sampling, experts, and variants are prohibited. The hard ceiling is 3,000,000 trainable parameters; expected exact count is 1,162,662.

## Sole physical mapping

Inside `forward`, and nowhere afterward:

- `P_req = S * ReLU(R_req)`;
- `P_comp = S * ReLU(R_comp)`;
- `P_noise = O - P_req - P_comp`.

The same mapped source tensors enter loss, metrics, persistence, hashing, labels, and deployment. There is no post-forward clipping, positive floor, observation transformation, truth-based rescaling, softplus, square, absolute-value, or simplex alternative. Float32 conservation tolerance is `1e-5 * max(1, max(abs(O)), max(abs(P_req)), max(abs(P_comp)), max(abs(P_noise)))`; float64 reference tolerance is the same scale times `1e-10`.

## Frozen objective

All sources are divided by the positive band scales for objective evaluation. The total is the nonnegative weighted sum:

1. requested normalized-pixel L1 mean, weight `1.0`;
2. companion normalized-pixel L1 mean, weight `1.0`;
3. mean per-source/per-band absolute relative flux error with denominator `abs(truth flux)+1e-6`, weight `0.25`;
4. mean per-source/per-band soft-centroid Euclidean error divided by 60, with prediction and truth centroids each using its own `flux+1e-6` denominator, weight `0.10`;
5. mean per-source absolute g/r and r/z log-flux-ratio error with `log(flux+1e-6)`, weight `0.10`.

No residual-target, source-sum, adversarial, perceptual, uncertainty, auditor, safety-label, D1/D3 endpoint, or Atlas loss is used. Exact truth must be a stationary global minimum. Equal allocation, swapping, 0.8 requested scaling, and averaged source fixtures must not beat truth. Gradients must be finite and nonzero away from optimum; ReLU inactive-gradient fractions are measured; exact zero target pixels remain representable.

## Objective and micro gates

Objective alignment runs on synthetic fixtures and the frozen ordinary/difficult/mixed crops before fitting. Any compromise below truth stops the campaign.

Micro fitting uses MPS AdamW with learning rate `3e-3`, weight decay `1e-4`, batch equal to the augmented microset, gradient clip `5.0`, and no scheduler. Selectors are ordinary index 16, difficult index 6, and mixed indices `[0,3,5,6,18,51,73,81]`. Each scene is trained with its stored requested prompt and a deterministically reconstructed companion-coordinate sigma-2 prompt with swapped targets. Budgets are 1,500 / 2,000 / 3,000 updates. Pass requires at least 95% total-objective reduction, at least 80% reduction in each requested and companion L1, requested identity closer than inversion on at least 90% of prompt views, exact zero source-negative/nonfinite fractions, conservation within tolerance, and updates to the head and first decoder up-convolution. Ordinary and mixed-eight must pass; difficult-only failure is reported but is not a stop.

## Full optimization and checkpoint selection

Seeds are `2026071501, 2026071502, 2026071503`; `2026071501` is the preregistered primary seed. MPS only. AdamW learning rate `3e-4`, weight decay `1e-4`; batch 16; maximum 40 epochs; patience 8; gradient clip 5.0; CosineAnnealingLR with `T_max=40, eta_min=0`, stepped after each epoch. Training order is seeded by seed and epoch.

Checkpoint selection is lexicographic and validation-only: lowest requested-source validation L1, then companion-source validation L1, then validation flux surrogate, then validation centroid surrogate. Safety labels, safe prevalence, calibration outcomes, Atlas, development, and lockbox never select a checkpoint or primary seed. Persist every requested loss/diagnostic per epoch.

## Freeze, OOF, deployment, and replay

Freeze one selected checkpoint per seed. Freeze architecture, physical mapping, inference source, prompt constructor, fixed-batch executor, normalization, and canonical hash implementation. Training labels use only genuine five-fold OOF outputs. Validation and calibration use the frozen full-fit primary checkpoint.

Deployment uses fixed neural batch size 16. Short chunks receive explicit zero dummy rows which are discarded. Canonical hashes use `thayer-per-sample-tensor-sha256-v1`, CHW little-endian contiguous float32 with its versioned header. Replay at least 100 OOF training, 100 validation, 100 calibration episodes and every physical edge fixture. Exact tensor/hash replay, batch consistency, prompt/source order, shapes/dtypes, zero source negatives, and conservation are mandatory.

## Unchanged labels and support gates

After output freeze, apply `src/direct_catalog_safety_auditor.py` at hash `{EXPECTED['safety']}` with unchanged image, flux, color, centroid, confusion, catastrophic, false-subtraction, worse-than-baseline, and source-output rules. Requested and companion source layers must both be finite/nonnegative; the signed residual is excluded from catalog-source nonnegativity. Invalid queries retain label `-1` (none are selected here).

Training gates: safe >=500, unsafe >=500, safe prevalence in [0.05,0.95], >=100 distinct safe and unsafe source groups, and >=100 safe `UNIQUE_VALID`. Validation/calibration gates: safe and unsafe each >=150, prevalence in [0.05,0.95], >=50 distinct safe and unsafe groups, >=100 safe `UNIQUE_VALID`, and >=50 safe `PERTURBED_VALID`. Scientific gates additionally require 100% source-output contract pass, >=10% catastrophic pass in validation/calibration, >=5% joint safe, prompt-swap >=0.90, and no systematic inversion.

## Family comparison and bootstrap

Compare aligned non-development Family-E1 outputs with frozen Condition C and repaired Thayer-PU. Distinctness requires safety disagreement >=0.10, reconstruction-error Spearman <=0.90, or a materially different failure profile. Family identity is never a future auditor input.

Run exactly 300 deterministic connected-source-group bootstrap replicates with seed `2026071599` for safe prevalence, catastrophic-pass, flux-pass, output-contract-pass, joint-safe, false-subtraction, safe-source-group count, and family disagreement. Do not claim subgroup-conditional guarantees.

## Outcomes and authorization

Assign exactly one: `FAMILY_E1_ELIGIBLE_WITH_LABEL_SUPPORT`, `FAMILY_E1_PHYSICALLY_VALID_BUT_LABEL_COLLAPSED`, `FAMILY_E1_SCIENTIFIC_PARTIAL`, `FAMILY_E1_RECONSTRUCTION_FAILURE`, or `DATA_OR_IMPLEMENTATION_FAILURE`. Only the eligible outcome authorizes one separately preregistered `Thayer-Audit v1 — Multi-Family POST Auditor`. No auditor is trained here. Every other outcome receives exactly one next experiment.

## Integrity

One architecture, one mapping, no safety selection, no clipping, no truth deployment, genuine OOF, leak-free source groups, and zero development/Atlas-selection/lockbox access are mandatory. The preflight, Condition C, Thayer-PU, historical checkpoints, README, and staged index remain unchanged. Nothing is staged, committed, pushed, merged, deleted, moved, renamed, or historically overwritten.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    local_now = datetime.now().astimezone()
    timestamp = args.timestamp or local_now.strftime("%Y%m%d_%H%M%S")
    run = RUNS / f"thayer_family_e1_v0_{timestamp}"
    run.mkdir(parents=False, exist_ok=False)
    for name in DIRECTORIES:
        (run / name).mkdir(exist_ok=False)

    staged = command(["git", "diff", "--cached", "--name-status"])
    if staged.strip():
        raise RuntimeError("staged index must be empty")
    report = PREFLIGHT / "reports/final_report.md"
    report_text = report.read_text(encoding="utf-8")
    if "**SIGNED_NOISE_RESIDUAL_CONTRACT_PASS**" not in report_text:
        raise RuntimeError("authoritative preflight outcome is not exact PASS")

    inputs = [
        verify(report, EXPECTED["preflight_report"], "preflight_report"),
        verify(PREFLIGHT / "diagnostics/physical_contract.md", EXPECTED["preflight_physical"], "preflight_physical_contract"),
        verify(PREFLIGHT / "preregistration/signed_noise_residual_physical_contract_preflight.md", EXPECTED["preflight_prereg"], "preflight_preregistration"),
        verify(REPO / "docs/family_e1_signed_noise_residual_preflight.md", EXPECTED["preflight_doc"], "preflight_documentation"),
        verify(FAMILY_E / "manifests/training_manifest.csv", EXPECTED["training_selector"], "training_selector"),
        verify(FAMILY_E / "manifests/validation_manifest.csv", EXPECTED["validation_selector"], "validation_selector"),
        verify(FAMILY_E / "manifests/calibration_manifest.csv", EXPECTED["calibration_selector"], "calibration_selector"),
        verify(HIERARCHICAL / "manifests/v2_r_training_scene_manifest.csv", EXPECTED["training_manifest"], "training_upstream_manifest"),
        verify(HIERARCHICAL / "manifests/v2_r_validation_scene_manifest.csv", EXPECTED["validation_manifest"], "validation_upstream_manifest"),
        verify(HIERARCHICAL / "manifests/v2_natural_calibration_scene_manifest.csv", EXPECTED["calibration_manifest"], "calibration_upstream_manifest"),
        verify(HIERARCHICAL / "manifests/v2_r_training_scenes.h5", EXPECTED["training_h5"], "training_h5"),
        verify(HIERARCHICAL / "manifests/v2_r_validation_scenes.h5", EXPECTED["validation_h5"], "validation_h5"),
        verify(HIERARCHICAL / "manifests/v2_natural_calibration_scenes.h5", EXPECTED["calibration_h5"], "calibration_h5"),
        verify(REPO / "scripts/thayer_select_prompt_ablation_common.py", EXPECTED["prompt"], "prompt_contract"),
        verify(REPO / "docs/physical_source_output_contract.md", EXPECTED["source_layer"], "source_layer_contract"),
        verify(REPO / "docs/d3_threshold_contract.md", EXPECTED["threshold"], "scientific_threshold_contract"),
        verify(REPO / "src/direct_catalog_safety_auditor.py", EXPECTED["safety"], "safety_label_implementation"),
        verify(REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json", EXPECTED["normalization"], "normalization"),
        verify(CONDITION_C, EXPECTED["condition_c"], "condition_c_checkpoint"),
        verify(THAYER_PU, EXPECTED["thayer_pu"], "thayer_pu_checkpoint"),
        verify(REPO / "README.md", EXPECTED["readme"], "readme"),
    ]

    before = FAMILY_E / "tables/checkpoint_inventory_before.csv"
    inventory_rows = list(csv.DictReader(before.open(encoding="utf-8")))
    verified_inventory: list[dict[str, object]] = []
    for row in inventory_rows:
        path = REPO / row["relative_path"]
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != row["expected_sha256"]:
            raise RuntimeError(f"historical checkpoint mismatch: {path}")
        verified_inventory.append({
            "relative_path": row["relative_path"],
            "expected_sha256": row["expected_sha256"],
            "observed_sha256": observed,
            "unchanged": True,
        })
    inventory_path = run / "tables/checkpoint_inventory_before.csv"
    with inventory_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verified_inventory[0]))
        writer.writeheader(); writer.writerows(verified_inventory)

    # New run contains immutable byte-for-byte selector references.
    for partition in ("training", "validation", "calibration"):
        shutil.copyfile(FAMILY_E / f"manifests/{partition}_manifest.csv", run / f"manifests/{partition}_manifest.csv")
    git_status = command(["git", "status", "--short", "--branch"])
    branch = command(["git", "branch", "--show-current"]).strip()
    git_head = command(["git", "rev-parse", "HEAD"]).strip()
    probe = None
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        probe = float(torch.ones(2, device="mps").sum().cpu())
    if probe != 2.0:
        raise RuntimeError("MPS unavailable or execution probe failed")
    usage = shutil.disk_usage(REPO)
    start_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    provenance = {
        "campaign": "Thayer-Family-E1-v0",
        "working_experiment": "Thayer-Family-E1-v0",
        "run_dir": relative(run),
        "campaign_start_local": local_now.isoformat(),
        "campaign_start_utc": start_utc,
        "branch": branch,
        "git_head": git_head,
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "staged_index_empty": True,
        "authoritative_preflight_outcome": "SIGNED_NOISE_RESIDUAL_CONTRACT_PASS",
        "authoritative_inputs": inputs,
        "historical_checkpoint_count": len(verified_inventory),
        "historical_checkpoint_inventory_sha256": sha256_file(inventory_path),
        "filesystem_free_bytes": usage.free,
        "development_access_count": 0,
        "atlas_selection_access_count": 0,
        "final_lockbox_access_count": 0,
    }
    fresh_json(run / "logs/input_provenance.json", provenance)
    fresh_text(run / "diagnostics/environment_snapshot.md", f"""# Family-E1 environment snapshot

- Start local / UTC: `{local_now.isoformat()}` / `{start_utc}`
- Branch / Git HEAD: `{branch}` / `{git_head}`
- Staged index empty: `true`
- Historical checkpoints verified: `{len(verified_inventory)}`
- MPS built / available / probe: `{torch.backends.mps.is_built()} / {torch.backends.mps.is_available()} / {probe}`
- Free disk bytes: `{usage.free}`
- Development / Atlas-selection / lockbox access: `0 / 0 / 0`

## Package versions

```json
{json.dumps(package_versions(), indent=2, sort_keys=True)}
```

## Initial Git status

```text
{git_status.rstrip()}
```
""")
    fresh_text(run / "diagnostics/campaign_contract.md", """# Thayer-Family-E1-v0 campaign contract

Status: **FROZEN MASTER-RUN BOUNDARY** before model construction or fitting.

The campaign uses one 4-channel coordinate U-Net, one six-channel in-forward ReLU source mapping, and one derived signed residual. Requested and companion alone are catalog sources. It uses exact source-safe training/validation/calibration selectors, MPS for all neural work, CPU only for metadata/labels/bootstrap/reports, validation-only checkpoint selection, genuine five-fold OOF training outputs, fixed-batch deterministic deployment, unchanged Thayer-Audit gates, and append-only collision-refusing artifacts. No auditor is trained. Development, Atlas selection, and lockbox access remain zero. Historical artifacts, Condition C, Thayer-PU, README, and the staged index remain unchanged.
""")

    prereg_path = run / "preregistration/family_e1_nonnegative_source_signed_residual_model.md"
    fresh_text(prereg_path, preregistration(start_utc))
    prereg_hash = sha256_file(prereg_path)
    fresh_text(prereg_path.with_suffix(".sha256"), f"{prereg_hash}  {prereg_path.name}\n")
    fresh_json(run / "logs/preregistration_complete.json", {
        "status": "FROZEN_BEFORE_MODEL_CONSTRUCTION_OR_FITTING",
        "path": relative(prereg_path), "sha256": prereg_hash, "frozen_utc": start_utc,
        "model_construction_count": 0, "optimizer_construction_count": 0,
        "training_tensor_load_count": 0,
    })
    print(json.dumps({"run": relative(run), "preregistration_sha256": prereg_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
