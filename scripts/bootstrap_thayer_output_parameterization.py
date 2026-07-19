#!/usr/bin/env python3
"""Create the metadata-only Thayer-OP run and freeze its preregistration."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs/runs"
PREFIX = "thayer_output_parameterization_"

ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
FP = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"
CL = REPO / "outputs/runs/thayer_capacity_ladder_20260713_013132"
PROMPT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
ATLAS_NOISE = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/manifests/fixed_noise_contract.json"

MICROSET = ME / "diagnostics/micro_overfit_20260712_203540/tables/microset_manifest.csv"
P0_TARGETS = FP / "projection_targets/projected_target_sets_final.h5"
P0_HASHES = FP / "tables/projected_target_hashes_final.csv"
CONDITION_C = PROMPT / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PROMPT / "manifests/normalization.json"
FORWARD_THRESHOLDS = PU / "manifests/forward_consistency_thresholds.json"
TRAINING_SCENES = MH / "manifests/probabilistic_unet_training_scenes.h5"
TRAINING_TARGETS = MH / "target_sets/thayer_mh_training_target_sets.h5"

EXPECTED = {
    MICROSET: "9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085",
    P0_TARGETS: "d58ef71e988de8584a78865f00747b931c1e65f6e406e437cebdca60a049b181",
    P0_HASHES: "b45e26f95f7ecf7fc117f3b4660901224060fd2a9ef9eecd1a408ba5e693a65b",
    CONDITION_C: "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
    NORMALIZATION: "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    FORWARD_THRESHOLDS: "a479a94bc1940b5fa146bc1a3eda3aeee6c931c90f25cc3a2108197486833e0a",
    TRAINING_SCENES: "d6ca6f1cbcb136a075f0216460e5f6b2dcd5fefbb63894803b86069df4e5f48d",
    TRAINING_TARGETS: "7fc92222ff2d980c4beb787b961fa7bdaf3130c055ce842dc8fd5f600c29c19a",
    ATLAS_NOISE: "3ce4435330da83eace363ceee3856612e100f43b63d2493aed7441992494ec7b",
    REPO / "src/models_two_expert_decoder.py": "9931c81b42aa4463ef9715223f768c787d40c373519043b68167645f7708f415",
    REPO / "src/coordinate_prompt.py": "a45b47ab2fb078d7624a4389a7634a6b1d898ade95d8de10899b6da590d82732",
    REPO / "src/prompt_semantics.py": "c604301d15e9254f6f2e72858710a2b8d035a466382d46281853f5065afac536",
    REPO / "src/scientific_alignment.py": "62c0f1f7704a50a66b16c0044df7e140b3fae71563f1fa7db895f1d260655b07",
    REPO / "src/competing_hypotheses.py": "e66111b2853c2b954efaa35880ee74d99736c03dc75197fd474fdc390271ca6d",
    REPO / "src/canonical_tensor_hash.py": "65566c01c5e6a76bc35e638423562180f370edb7b5b8bc5a3931ae2ca994bb6e",
    REPO / "scripts/train_thayer_feasibility_projection_micro.py": "30764623541a5bbb54e45aa96daa169ec9dc5c5399b637601a86ab4709d3e7e8",
    REPO / "scripts/run_thayer_two_expert_micro_overfit.py": "69a4e862fdff54f4de7dc2564f35ccced29528317a266882d710b28b92ac7ec2",
    REPO / "docs/multi_hypothesis_source_contract.md": "dc3a78b65b2eda17b71887c7616189a24fbf1f367c8fb61014d6e291a2e02128",
    REPO / "docs/latent_truth_coverage.md": "4d2d53ea7ef77c09b263ee90dec50b7138b0dbe07f2b16150113c0041e589d97",
}

CONTRACT_ROLES = {
    "microset_manifest": MICROSET,
    "p0_projected_target_tensor": P0_TARGETS,
    "p0_canonical_hash_table": P0_HASHES,
    "condition_c_encoder_checkpoint": CONDITION_C,
    "l0_encoder_decoder_topology": REPO / "src/models_two_expert_decoder.py",
    "output_mapping_implementation": REPO / "src/output_parameterization.py",
    "prompt_generation": REPO / "src/coordinate_prompt.py",
    "prompt_semantics": REPO / "src/prompt_semantics.py",
    "hard_assignment": REPO / "scripts/train_thayer_feasibility_projection_micro.py",
    "source_layer_contract": REPO / "docs/multi_hypothesis_source_contract.md",
    "scientific_thresholds": REPO / "src/scientific_alignment.py",
    "truth_coverage_implementation": REPO / "src/competing_hypotheses.py",
    "truth_coverage_contract": REPO / "docs/latent_truth_coverage.md",
    "canonical_tensor_hash": REPO / "src/canonical_tensor_hash.py",
    "inverse_normalization": NORMALIZATION,
    "forward_consistency_thresholds": FORWARD_THRESHOLDS,
    "fixed_forward_noise_metadata": ATLAS_NOISE,
}

SUBDIRS = (
    "diagnostics",
    "tables",
    "figures",
    "logs",
    "reports",
    "preregistration",
    "output_contract",
    "synthetic_preflight",
    "one_scene",
    "eight_scene",
    "checkpoints",
    "example_grids",
)

ONE_ORDINARY = 0
ONE_AMBIGUOUS = 32
EIGHT_SCENES = (0, 8, 16, 24, 32, 40, 48, 56)
EXPERT_SEEDS = (2026071201, 2026071202)
TRAINING_SEED = 2026071250
MAPPINGS = ("relu", "square", "absolute")
INITIAL_EPSILON = 9.999999406318238e-08
NUMERICAL_ZERO_TOLERANCE = 1e-7
PHYSICAL_NEGATIVE_TOLERANCE = 0.0
ROUNDTRIP_PHYSICAL_ATOL = 0.00390625
OPTIMIZER_STEPS = 3200
MICROBATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 8
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, check=True, text=True, capture_output=True).stdout.rstrip()


def write_text_fresh(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def checkpoint_inventory() -> list[dict[str, object]]:
    paths = sorted(
        path
        for path in (REPO / "outputs").rglob("*")
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
    )
    return [
        {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def relevant_source_inventory() -> list[dict[str, object]]:
    candidates = sorted(
        set(CONTRACT_ROLES.values())
        | {
            REPO / "scripts/bootstrap_thayer_output_parameterization.py",
            REPO / "scripts/run_thayer_output_parameterization_preflight.py",
            REPO / "scripts/run_thayer_output_parameterization_micro.py",
            REPO / "scripts/finalize_thayer_output_parameterization.py",
            REPO / "tests/test_output_parameterization.py",
        }
    )
    return [
        {
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in candidates
        if path.is_file()
    ]


def selected_rows() -> list[dict[str, object]]:
    rows = read_csv(MICROSET)
    if len(rows) != 64:
        raise RuntimeError("frozen microset manifest does not contain 64 rows")
    expected = {
        0: ("pu_training_ordinary_00000", "ordinary", "0"),
        8: ("pu_training_ordinary_00008", "ordinary", "8"),
        16: ("pu_training_ordinary_00016", "ordinary", "16"),
        24: ("pu_training_ordinary_00024", "ordinary", "24"),
        32: ("pu_training_near_00000", "near_collision", "12000"),
        40: ("pu_training_near_00008", "near_collision", "12008"),
        48: ("pu_training_near_00016", "near_collision", "12016"),
        56: ("pu_training_near_00024", "near_collision", "12024"),
    }
    output = []
    for index in EIGHT_SCENES:
        row = rows[index]
        observed = (row["scene_id"], row["kind"], row["source_h5_index"])
        if observed != expected[index]:
            raise RuntimeError(f"frozen row mismatch at micro index {index}: {observed}")
        output.append(
            {
                "micro_index": index,
                "scene_id": row["scene_id"],
                "kind": row["kind"],
                "pair_id": row["pair_id"],
                "source_h5_index": row["source_h5_index"],
                "one_scene_ordinary": index == ONE_ORDINARY,
                "one_scene_ambiguous": index == ONE_AMBIGUOUS,
                "eight_scene": True,
            }
        )
    return output


def preregistration(started: str, row_table_hash: str, mapping_hash: str) -> str:
    return f"""# Fixed-L0 Output-Parameterization preregistration

Working name: **Thayer-OP (Thayer Output Parameterization)**  
Frozen at UTC: `{started}`  
Microset manifest SHA-256: `{EXPECTED[MICROSET]}`  
Frozen P0 target-file SHA-256: `{EXPECTED[P0_TARGETS]}`  
Frozen P0 canonical-hash table SHA-256: `{EXPECTED[P0_HASHES]}`  
Frozen row-selection table SHA-256: `{row_table_hash}`  
Output-mapping implementation SHA-256: `{mapping_hash}`

## Scope and protected boundary

This is a fixed-L0, training-only output-parameterization campaign. It compares exactly ReLU, square, and absolute value inside the model forward path. It is not a decoder-capacity ladder, 64-row fit, full-data campaign, Atlas evaluation, development evaluation, auditor campaign, or lockbox evaluation. The remaining 56 microset scene inputs are prohibited. Historical runs and checkpoints are read-only. New artifacts are timestamped, campaign-local, append-only, and collision-refusing.

Atlas, validation, calibration, development, and lockbox arrays are forbidden. Truth, target indices, pair IDs, projection metadata, and scientific constraints may not enter the model input. The only inference tensors are normalized g/r/z blends and the frozen Condition-C Gaussian coordinate prompt.

## Frozen rows and targets

The one-scene ordinary row is micro index `0`, `pu_training_ordinary_00000`, source HDF5 index `0`. The one-scene ambiguous observation is micro index `32`, `pu_training_near_00000`, pair `pu_training_pair_00001`, source HDF5 index `12000`. The eight-scene set is micro indices `{list(EIGHT_SCENES)}`: ordinary `0,8,16,24` and ambiguous `32,40,48,56`, in that exact order. The exact selection is hashed in `tables/frozen_row_selection.csv`.

Every fit uses the exact final P0 tensor and per-sample canonical hashes. No target, threshold, source-layer semantic, prompt rule, hard assignment, or truth-coverage definition may change. The P0 tensor may be read in full only for the mapping representability audit. Scene arrays may be loaded only for the eight frozen source indices.

## Fixed L0 architecture and encoder isolation

Every mapping uses the exact Condition-C shared encoder `4->16->32->64`, the exact L0 decoder blocks `96->32` and `48->16`, and two independent 46,470-parameter expert decoders seeded `{EXPERT_SEEDS[0]}` and `{EXPERT_SEEDS[1]}`. The encoder checkpoint hash is `{EXPECTED[CONDITION_C]}`. Every encoder parameter has `requires_grad=False`, the encoder remains in `eval()` throughout, no encoder tensor enters an optimizer, and encoder state is byte-hashed before and after every condition. Only the two L0 expert decoders train. Decoder topology, width, depth, GroupNorm, SiLU, bilinear skips, 1x1 head, expert count, and parameter count are identical across mappings.

## Exact output mappings and physical path

- M0 ReLU: `physical_normalized = relu(raw)`.
- M1 Square: `physical_normalized = raw ** 2`.
- M2 Absolute: `physical_normalized = abs(raw)`.

The mapped normalized tensor passes through the single frozen positive-scale multiplication to become the physical detected-electron tensor. Training loss, hard assignment, truth coverage, prompt swap, forward consistency, source sum, hashes, and saved outputs consume that exact physical tensor. The reconstruction loss weights physical residuals by the frozen positive per-band scales, preserving the prior dimensionless direct requested-plus-companion MSE without creating a second source-value path. Training on raw output, evaluation-only clamping, detached value-changing postprocessing, or a different train/evaluation mapping is prohibited.

The dtype is float32; channel order is requested g/r/z then companion g/r/z; source order is requested then companion; spatial shape is 60x60; zero background is exact. Numerical-zero tolerance is `{NUMERICAL_ZERO_TOLERANCE}` normalized source units, physical negative tolerance is `{PHYSICAL_NEGATIVE_TOLERANCE}` detected electrons, nonfinite tolerance is zero values, and frozen physical round-trip tolerance is `{ROUNDTRIP_PHYSICAL_ATOL}` detected electrons. Any physical value below zero or any nonfinite value triggers synchronous termination before another optimizer step.

## Matched initialization

`initial_physical_epsilon` is `{INITIAL_EPSILON}` in the normalized source layer, derived from the frozen `1e-7` exact-arithmetic numerical-zero contract. It maps to per-band physical values g/r/z `0.0000611920 / 0.0001805880 / 0.0001854200` detected electrons. Every final 1x1 convolution weight is initialized to zero. ReLU and absolute biases are `{INITIAL_EPSILON}`; square bias is `sqrt({INITIAL_EPSILON})`. Earlier decoder tensors use the same two frozen expert seeds for every mapping. This makes the initial mapped tensor byte-identical across mappings.

## Representability, gradient, and stop-rule gates

For every P0 target, ReLU and absolute use raw witness `target`; square uses `sqrt(target)`. Each witness must map to finite, nonnegative, correctly shaped output within `{ROUNDTRIP_PHYSICAL_ATOL}` physical electrons, reproduce deterministically under the canonical hash, and show no positive-target saturation. Exact zero, near-zero, sparse positive, constant positive, high-value, and z-extrema cases are mandatory.

Gradient audits cover initialization, numerical zero, low positive output `1e-6`, median positive P0 value, high P0 value, and z-band extrema. Framework subgradient zero at the exact ReLU boundary or absolute-value cusp is reported but does not disqualify a mapping. A mapping is ineligible only if the derivative is nonfinite or unusable over a material portion of strictly positive P0 support.

Before fitting, isolated sentinels must prove synchronous stopping for negative physical output, NaN, Inf, target-hash mismatch, and simulated MPS fallback. Each expected incident must be written before the injected path terminates; its local status must be failed; optimizer-step count may not advance; and no checkpoint may be promoted. Every self-test must pass.

## Synthetic output-head preflight

With the encoder bypassed, train only a zero-weight L0 16-to-6 1x1 head on MPS using a deterministic 4x4 one-hot spatial basis. Cases are zero, constant positive, sparse positive, the central 4x4 crop of P0 row 0/prompt 0/slot 0, and a 4x4 crop centered on the globally maximal z-channel P0 value under a deterministic first-index tie rule. Each case uses AdamW, learning rate `0.03`, weight decay zero, and exactly `500` steps. A pixel receives one independent weight and the common bias, so the approximate two-parameter Adam displacement bound is `2 * 0.03 * 500 = 30` normalized units, above the frozen P0 maximum near 18.41. Nonzero cases require at least 95% loss reduction; the zero case must remain within `1e-12` normalized MSE. Gradients and outputs must remain finite and physical negatives must remain zero.

## Common neural compute contract

Every real fit uses MPS-only AdamW, learning rate `{LEARNING_RATE}`, weight decay `0`, no scheduler, gradient-norm clipping `{GRADIENT_CLIP_NORM}`, microbatch `{MICROBATCH_SIZE}`, effective batch `{EFFECTIVE_BATCH_SIZE}`, accumulation `1`, exactly `{OPTIMIZER_STEPS}` optimizer steps, and exactly `{OPTIMIZER_STEPS * EFFECTIVE_BATCH_SIZE}` scene presentations. One-scene batches repeat only the one frozen observation eight times. Eight-scene batches present the eight frozen observations once in the fixed row order. There is no shuffle, augmentation, early success stop, extra seed, mapping-specific schedule, or extra optimization. Evaluate at steps 0, 1, every 100 steps, and step 3200. MPS memory probes must pass for all eligible mappings before a real fit.

## One-scene gates

Each gate starts from a fresh matched initialization. The ordinary gate requires zero physical negatives at every used forward, both experts covering the approved ordinary scientific region on both prompts, set-level coverage 100%, median expert diameter <=1.0, set prompt identity 100%, finite forward evaluation, finite gradients, and no dead expert. A failure remains reportable but cannot be selected.

The ambiguous gate uses the same single ambiguous observation and requires zero physical negatives, own coverage 100%, alternate coverage 100%, both-mode coverage 100%, both experts active, no mode collapse, set prompt identity 100%, and finite forward evaluation for both decompositions. If every mapping fails, the campaign stops before eight-scene fitting and concludes that mapping alone is insufficient.

## Eight-scene gate and selection

Only mappings passing both one-scene gates may fit the frozen eight-scene set. The remaining 56 scene rows stay unopened. Report ordinary, own, alternate, both-mode coverage; ordinary expert diameter; prompt swap; forward and source-sum consistency; dimensionless projected-target loss; z-channel projected-target MSE; zero/stagnant derivative fraction; raw magnitude; and mapping boundary/cusp activation.

A mapping is selectable only if representability, self-tests, synthetic fits, both one-scene gates, prompt fidelity, finite behavior, and zero negatives all pass. Among selectable mappings: (1) maximize the minimum of eight-scene ordinary/own/alternate/both-mode coverage; (2) minimize physical projected-target loss; (3) minimize ordinary expert diameter; (4) minimize derivative-stagnation fraction; (5) minimize z-band target error; (6) prefer ReLU only on an exact remaining tie. Forward consistency is evaluation-only and never the first selection metric. No mapping may be selected if all remain at zero both-mode coverage.

## Decisions and authorization

The primary outcome is exactly one of RELU SELECTED, SQUARE SELECTED, ABSOLUTE SELECTED, MULTIPLE MAPPINGS EQUIVALENT, or NO MAPPING PASSES. A practical tie means all preceding lexicographic quantities are exactly equal at their recorded float64 values; the ReLU rule still freezes one later-ladder mapping. One selected mapping authorizes only a separate decoder-capacity-ladder campaign. No width ladder runs here. If no mapping passes, recommend exactly one isolated diagnostic based on dead gradients, hard assignment, frozen encoder representation, or expert-decoder optimization. If one-scene passes but eight-scene fails, aggregation within the microset is the blocker and only the smallest isolated follow-up may be recommended.
"""


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = RUNS / f"{PREFIX}{stamp}"
    run.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRS:
        (run / name).mkdir(exist_ok=False)

    mismatches = []
    frozen_inputs = {}
    for path, expected in EXPECTED.items():
        observed = sha256(path) if path.is_file() else "MISSING"
        frozen_inputs[str(path.relative_to(REPO))] = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": expected,
            "sha256": observed,
        }
        if observed != expected:
            mismatches.append(
                {"path": str(path.relative_to(REPO)), "expected": expected, "observed": observed}
            )

    rows = selected_rows()
    write_csv_fresh(run / "tables/frozen_row_selection.csv", rows)
    row_hash = sha256(run / "tables/frozen_row_selection.csv")

    checkpoints = checkpoint_inventory()
    write_csv_fresh(run / "tables/checkpoint_inventory_before.csv", checkpoints)
    source_rows = relevant_source_inventory()
    write_csv_fresh(run / "tables/source_code_hashes_before.csv", source_rows)

    packages = {}
    for name in ("torch", "numpy", "h5py", "matplotlib", "scipy", "btk", "astropy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    try:
        import torch

        mps_built = bool(torch.backends.mps.is_built())
        mps_available = bool(torch.backends.mps.is_available())
        torch_version = torch.__version__
    except Exception as exc:  # pragma: no cover - environment snapshot only.
        mps_built = False
        mps_available = False
        torch_version = f"unavailable: {exc}"

    contract_hashes = {
        role: {
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for role, path in CONTRACT_ROLES.items()
    }
    git_status = command("git", "status", "--short").splitlines()
    staged = command("git", "diff", "--cached", "--name-status").splitlines()
    provenance = {
        "campaign": "Thayer-OP Fixed-L0 Physical Output-Parameterization",
        "campaign_started_utc": started,
        "branch": command("git", "branch", "--show-current"),
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status_short": git_status,
        "staged_index": staged,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_version": torch_version,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "free_disk_bytes": shutil.disk_usage(REPO).free,
        "microset_manifest_sha256": sha256(MICROSET),
        "p0_projected_target_file_sha256": sha256(P0_TARGETS),
        "p0_projected_target_hash_table_sha256": sha256(P0_HASHES),
        "condition_c_checkpoint_sha256": sha256(CONDITION_C),
        "frozen_contract_hashes": contract_hashes,
        "frozen_inputs": frozen_inputs,
        "frozen_input_mismatches": mismatches,
        "historical_checkpoint_count": len(checkpoints),
        "relevant_source_file_count": len(source_rows),
        "frozen_row_selection_sha256": row_hash,
        "expert_initialization_seeds": list(EXPERT_SEEDS),
        "training_seed": TRAINING_SEED,
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
        "per_scene_tensor_load_count_before_preregistration": 0,
        "p0_tensor_load_count_before_preregistration": 0,
        "neural_model_construction_count_before_preregistration": 0,
        "optimizer_step_count_before_preregistration": 0,
    }
    write_json_fresh(run / "logs/input_provenance.json", provenance)

    environment = f"""# Environment snapshot

- Campaign start UTC: `{started}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python / PyTorch: `{sys.version.split()[0]}` / `{torch_version}`
- MPS built / available: `{mps_built}` / `{mps_available}`
- Historical checkpoint files inventoried: `{len(checkpoints)}`
- Relevant source files hashed: `{len(source_rows)}`
- Frozen-input mismatches: `{len(mismatches)}`
- Staged index entries: `{len(staged)}`
- Free disk bytes: `{provenance['free_disk_bytes']}`

The complete package, git-status, contract-hash, checkpoint, source-code, and protected-data records are in `logs/input_provenance.json` and the inventory tables. No P0 tensor or per-scene tensor was opened while creating this snapshot.
"""
    write_text_fresh(run / "diagnostics/environment_snapshot.md", environment)

    contract = f"""# Thayer-OP campaign contract

This campaign holds the exact L0 shared encoder and two 46,470-parameter expert decoder topologies fixed. Only the in-forward raw-to-nonnegative mapping changes among ReLU, square, and absolute value. The exact P0 targets, hard assignment, prompt, source layers, thresholds, truth coverage, normalization, optimizer, step budget, and scene ordering are common.

The encoder is frozen in evaluation mode and byte-audited around every condition. Neural fitting is MPS-only with no fallback. CPU is reserved for hashes, audits, metrics, plots, and reports. Physical negatives and nonfinite values stop synchronously. Atlas, development, lockbox, the remaining 56 scene rows, a fourth mapping, capacity changes, stage, commit, push, merge, delete, overwrite, and historical mutation are prohibited.

Frozen-input mismatches: `{len(mismatches)}`. MPS available at bootstrap: `{mps_available}`.
"""
    write_text_fresh(run / "diagnostics/campaign_contract.md", contract)

    mapping_contract = {
        "mappings": {
            "relu": "torch.relu(raw)",
            "square": "raw.square()",
            "absolute": "torch.abs(raw)",
        },
        "dtype": "float32",
        "band_order": ["requested_g", "requested_r", "requested_z", "companion_g", "companion_r", "companion_z"],
        "source_order": ["requested", "companion"],
        "spatial_shape": [60, 60],
        "zero_background": True,
        "initial_physical_epsilon_normalized": INITIAL_EPSILON,
        "numerical_zero_tolerance_normalized": NUMERICAL_ZERO_TOLERANCE,
        "physical_negative_tolerance_electrons": PHYSICAL_NEGATIVE_TOLERANCE,
        "roundtrip_physical_atol_electrons": ROUNDTRIP_PHYSICAL_ATOL,
        "expert_initialization_seeds": list(EXPERT_SEEDS),
        "training_seed": TRAINING_SEED,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "microbatch_size": MICROBATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation": 1,
        "optimizer_steps": OPTIMIZER_STEPS,
        "scene_presentations": OPTIMIZER_STEPS * EFFECTIVE_BATCH_SIZE,
        "mapping_code_sha256": sha256(REPO / "src/output_parameterization.py"),
    }
    write_json_fresh(run / "output_contract/mapping_contract.json", mapping_contract)

    attainability = [
        {"gate": "all_p0_targets_representable", "threshold": "100%", "attainable": True, "proof": "nonnegative target is its ReLU/abs witness and sqrt(target) is the square witness"},
        {"gate": "physical_nonnegativity", "threshold": "minimum >= 0", "attainable": True, "proof": "all three mappings have codomain [0,infinity)"},
        {"gate": "finite_mapping_and_gradient", "threshold": "100% finite", "attainable": True, "proof": "finite float32 targets and finite positive inverse witnesses"},
        {"gate": "usable_gradient_on_positive_support", "threshold": "no material dead support", "attainable": True, "proof": "all frozen P0 entries are strictly positive and each positive inverse witness has nonzero derivative"},
        {"gate": "stop_rule_self_tests", "threshold": "5/5", "attainable": True, "proof": "controlled sentinel paths deterministically trigger each guard before an optimizer step"},
        {"gate": "synthetic_head_fits", "threshold": "all five cases", "attainable": True, "proof": "one-hot basis gives an exact witness and 2*0.03*500=30 common raw displacement exceeds the 18.41 P0 maximum"},
        {"gate": "ordinary_one_scene_truth_coverage", "threshold": "100%", "attainable": True, "proof": "both frozen P0 ordinary slots already lie inside the approved scientific region"},
        {"gate": "ordinary_one_scene_diameter", "threshold": "<=1.0", "attainable": True, "proof": "the two frozen P0 ordinary slots satisfy set coverage and concentration"},
        {"gate": "ambiguous_one_scene_both_mode", "threshold": "100%", "attainable": True, "proof": "the two P0 slots represent the own and alternate approved modes"},
        {"gate": "prompt_fidelity", "threshold": "100%", "attainable": True, "proof": "the frozen prompt-ordered P0 targets retain requested/companion identity"},
        {"gate": "eight_scene_minimum_coverage", "threshold": ">0 for selection", "attainable": True, "proof": "the exact eight P0 target sets have full frozen target coverage"},
        {"gate": "selection_tie_breaker", "threshold": "deterministic", "attainable": True, "proof": "exact lexicographic metrics followed by ReLU on an exact remaining tie"},
    ]
    write_csv_fresh(run / "tables/preregistered_gate_attainability.csv", attainability)

    prereg_path = run / "preregistration/fixed_l0_output_parameterization.md"
    write_text_fresh(
        prereg_path,
        preregistration(
            started,
            row_hash,
            sha256(REPO / "src/output_parameterization.py"),
        ),
    )
    prereg_hash = sha256(prereg_path)
    frozen_at = datetime.now(timezone.utc).isoformat()
    freeze_status = "FROZEN_BEFORE_PER_SCENE_LOAD"
    if mismatches:
        freeze_status = "FAIL_CLOSED_INPUT_MISMATCH"
    elif not mps_available:
        freeze_status = "FAIL_CLOSED_MPS_UNAVAILABLE"
    freeze = {
        "campaign": "Thayer-OP",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": prereg_hash,
        "frozen_row_selection_sha256": row_hash,
        "gate_attainability_sha256": sha256(run / "tables/preregistered_gate_attainability.csv"),
        "mapping_contract_sha256": sha256(run / "output_contract/mapping_contract.json"),
        "mapping_code_sha256": sha256(REPO / "src/output_parameterization.py"),
        "frozen_input_mismatch_count": len(mismatches),
        "per_scene_tensor_load_count": 0,
        "p0_tensor_load_count": 0,
        "neural_model_construction_count": 0,
        "optimizer_step_count": 0,
        "status": freeze_status,
    }
    write_json_fresh(run / "preregistration/freeze_record.json", freeze)
    write_json_fresh(
        run / "logs/bootstrap_complete.json",
        {
            "run_dir": str(run.relative_to(REPO)),
            "preregistration_sha256": prereg_hash,
            "frozen_at_utc": frozen_at,
            "frozen_input_mismatches": mismatches,
            "mps_available": mps_available,
            "status": "PASS" if freeze_status == "FROZEN_BEFORE_PER_SCENE_LOAD" else "FAIL",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run),
                "preregistration_sha256": prereg_hash,
                "mismatches": mismatches,
                "mps_available": mps_available,
                "status": freeze_status,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
