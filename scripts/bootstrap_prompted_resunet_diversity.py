#!/usr/bin/env python3
"""Create and preregister the Prompted ResUNet candidate-diversity campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from astropy.table import Table


REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
PROMPT_RUN = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PROMPT_RUN / "manifests/source_split_manifest.csv"
CONDITION_C = PROMPT_RUN / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PROMPT_RUN / "manifests/normalization.json"
SUBDIRECTORIES = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "manifests", "checkpoints", "candidate_outputs", "atlas_evaluation",
    "example_grids", "paper_figures",
)
TRAIN_COUNT = 10_000
VALIDATION_COUNT = 1_500
SCENE_SEED_BASE = 2026077400
NOISE_SEED_BASE = 2026078400
TRAINING_SEED = 2026077301
EXPECTED_PARAMETER_COUNT = 199_219
CONDITION_C_PARAMETER_COUNT = 119_091
EXPECTED_SPLIT_SHA256 = "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27"
EXPECTED_CATALOG_SHA256 = "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46"
EXPECTED_CONDITION_C_SHA256 = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, cwd=REPO, check=True, text=True, capture_output=True)
    return result.stdout.strip()


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


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_run() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_prompted_resunet_diversity_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir(exist_ok=False)
    return run_dir


def verify_hard_gates(run_dir: Path) -> dict[str, object]:
    staged = command(["git", "diff", "--cached", "--name-only"]).splitlines()
    if staged:
        raise RuntimeError(f"Staged files are prohibited: {staged}")
    expected_inputs = {
        CATALOG: EXPECTED_CATALOG_SHA256,
        SOURCE_SPLIT: EXPECTED_SPLIT_SHA256,
        CONDITION_C: EXPECTED_CONDITION_C_SHA256,
    }
    for path, expected in expected_inputs.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Frozen input altered: {path.relative_to(REPO)}")

    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    pair_manifest = ATLAS / "tables/atlas_pair_manifest.csv"
    visual_audit = ATLAS / "tables/atlas_initial_visual_audit.csv"
    if freeze["status"] != "FROZEN_INITIAL_ATLAS_PASS" or freeze["pair_count"] != 25:
        raise RuntimeError("Authoritative Atlas freeze is unresolved")
    if sha256_file(pair_manifest) != freeze["numerical_manifest_sha256"]:
        raise RuntimeError("Frozen Atlas pair manifest changed")
    if sha256_file(visual_audit) != freeze["visual_audit_sha256"]:
        raise RuntimeError("Frozen Atlas visual audit changed")
    pairs = {row["pair_id"]: row for row in read_csv(pair_manifest)}
    definitions = read_csv(ATLAS / "manifests/atlas_pool_scene_definitions.csv")
    sys.path.insert(0, str(REPO))
    from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene

    catalog, _ = load_catsim_catalog(CATALOG)
    array_rows = []
    for pair_id in freeze["pair_ids"]:
        row = pairs[pair_id]
        path = ATLAS / f"atlas/{pair_id}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            checks = {}
            for side in ("left", "right"):
                definition = definitions[int(row[f"{side}_pool_index"])]
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
                replay = render_fixed_scene(catalog, spec, add_noise="none")
                stored_blend = np.asarray(arrays[f"{side}_blend"], dtype=np.float32)
                stored_isolated = np.asarray(arrays[f"{side}_isolated"], dtype=np.float32)
                checks[f"{side}_blend_float32_replay"] = np.array_equal(
                    np.asarray(replay.blend, dtype=np.float32), stored_blend
                )
                checks[f"{side}_isolated_float32_replay"] = np.array_equal(
                    np.asarray(replay.isolated, dtype=np.float32), stored_isolated
                )
        if not all(checks.values()):
            raise RuntimeError(f"Frozen Atlas array mismatch: {pair_id}")
        array_rows.append({"pair_id": pair_id, "path": str(path.relative_to(REPO)), "sha256": sha256_file(path), **checks})
    write_csv_fresh(run_dir / "tables/atlas_artifact_hashes_before.csv", array_rows)

    historical_reference = read_csv(ATLAS / "tables/checkpoint_inventory_before.csv")
    inventory = []
    for row in historical_reference:
        path = REPO / row["path"]
        observed = sha256_file(path)
        status = "PASS" if observed == row["sha256"] else "FAIL"
        inventory.append({**row, "observed_sha256": observed, "status": status})
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", inventory)
    if len(inventory) != 556 or any(row["status"] != "PASS" for row in inventory):
        raise RuntimeError("Historical checkpoint inventory changed")

    source_code = []
    for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            source_code.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv_fresh(run_dir / "tables/source_code_hashes_before.csv", source_code)
    return {
        "atlas_freeze": freeze,
        "atlas_pair_manifest_sha256": sha256_file(pair_manifest),
        "atlas_visual_audit_sha256": sha256_file(visual_audit),
        "historical_checkpoint_count": len(inventory),
        "staged_index_empty": True,
    }


def atlas_exposure_and_scenes(run_dir: Path) -> dict[str, object]:
    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    pair_rows = {row["pair_id"]: row for row in read_csv(ATLAS / "tables/atlas_pair_manifest.csv")}
    targeted_rows = {row["source_pair_id"]: row for row in read_csv(ATLAS / "tables/targeted_optimization_pair_manifest.csv")}
    roles: dict[str, set[str]] = defaultdict(set)
    pair_occurrences: Counter[str] = Counter()
    for pair_id in freeze["pair_ids"]:
        row = pair_rows[pair_id]
        for field in ("left_target_group", "right_target_group", "left_contaminant_group", "right_contaminant_group"):
            group = row[field]
            roles[group].add(field)
            pair_occurrences[group] += 1
        targeted = targeted_rows[pair_id]
        group = targeted["selected_contaminant_group"]
        roles[group].add("targeted_optimization_selected_contaminant_group")
        pair_occurrences[group] += 1
    excluded = set(roles)

    historical_scenes = read_csv(PROMPT_RUN / "manifests/development_scene_definitions.csv")
    exposure: dict[tuple[str, str], int] = Counter()
    for row in historical_scenes:
        if row["partition"] not in {"training", "validation"}:
            continue
        for field in ("source_a_group", "source_b_group"):
            exposure[(row[field], row["partition"])] += 1

    split = read_csv(SOURCE_SPLIT)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    group_rows: Counter[str] = Counter()
    for row in split:
        split_by_group[row["duplicate_group_id"]].add(row["partition"])
        group_rows[row["duplicate_group_id"]] += 1
    audit_rows = []
    for group in sorted(excluded):
        partitions = sorted(split_by_group[group])
        audit_rows.append({
            "source_group": group,
            "atlas_roles": ";".join(sorted(roles[group])),
            "atlas_or_targeted_occurrences": pair_occurrences[group],
            "historical_split_partition": ";".join(partitions),
            "catalog_rows_in_group": group_rows[group],
            "condition_c_training_scene_exposures": exposure[(group, "training")],
            "condition_c_validation_scene_exposures": exposure[(group, "validation")],
            "condition_c_ever_exposed": bool(exposure[(group, "training")] or exposure[(group, "validation")]),
            "resunet_training_excluded": True,
            "resunet_validation_excluded": True,
        })
    write_csv_fresh(run_dir / "tables/atlas_source_exposure_audit.csv", audit_rows)

    table = Table.read(CATALOG)
    allowed: dict[str, list[dict[str, str]]] = {"training": [], "validation": []}
    for row in split:
        partition = row["partition"]
        if partition in allowed and row["engineering_excluded"] == "0" and row["duplicate_group_id"] not in excluded:
            allowed[partition].append(row)
    allowed_groups = {partition: len({row["duplicate_group_id"] for row in rows}) for partition, rows in allowed.items()}
    if len(allowed["training"]) < 10_000 or len(allowed["validation"]) < 1_500:
        raise RuntimeError(f"Atlas exclusions leave insufficient populations: {allowed_groups}")
    if {row["duplicate_group_id"] for row in allowed["training"]} & {row["duplicate_group_id"] for row in allowed["validation"]}:
        raise RuntimeError("Training/validation source groups overlap")

    scene_rows: list[dict[str, object]] = []
    global_index = 0
    for partition, count in (("training", TRAIN_COUNT), ("validation", VALIDATION_COUNT)):
        pool = allowed[partition]
        for local_index in range(count):
            scene_seed = SCENE_SEED_BASE + global_index
            rng = np.random.default_rng(scene_seed)
            selected_indices = rng.choice(len(pool), size=2, replace=False)
            selected = [pool[int(index)] for index in selected_indices]
            if selected[0]["duplicate_group_id"] == selected[1]["duplicate_group_id"]:
                candidates = [row for row in pool if row["duplicate_group_id"] != selected[0]["duplicate_group_id"]]
                selected[1] = candidates[int(rng.integers(0, len(candidates)))]
            target_index = int(rng.integers(0, 2))
            separation = float(rng.uniform(0.8, 3.2))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            midpoint = rng.uniform(-0.35, 0.35, size=2)
            offset = 0.5 * separation * np.asarray([np.cos(angle), np.sin(angle)])
            positions = np.stack((midpoint - offset, midpoint + offset))
            catalog_rows = [int(row["catalog_row"]) for row in selected]
            size = [max(float(table["a_b"][index]), float(table["a_d"][index])) for index in catalog_rows]
            scene_rows.append({
                "scene_id": f"resunet_{partition}_{local_index:05d}",
                "partition": partition,
                "partition_index": local_index,
                "scene_seed": scene_seed,
                "noise_seed": NOISE_SEED_BASE + global_index,
                "source_a_row": catalog_rows[0],
                "source_b_row": catalog_rows[1],
                "source_a_id": selected[0]["persistent_source_id"],
                "source_b_id": selected[1]["persistent_source_id"],
                "source_a_group": selected[0]["duplicate_group_id"],
                "source_b_group": selected[1]["duplicate_group_id"],
                "target_index": target_index,
                "target_source_group": selected[target_index]["duplicate_group_id"],
                "alternate_source_group": selected[1 - target_index]["duplicate_group_id"],
                "source_a_x_arcsec": float(positions[0, 0]),
                "source_a_y_arcsec": float(positions[0, 1]),
                "source_b_x_arcsec": float(positions[1, 0]),
                "source_b_y_arcsec": float(positions[1, 1]),
                "separation_arcsec": separation,
                "source_a_size_arcsec": size[0],
                "source_b_size_arcsec": size[1],
                "requested_source_rule": "seeded uniform A/B",
                "position_rule": "symmetric around seeded midpoint",
                "noise_rule": "one explicit BTK add_noise=all realization",
            })
            global_index += 1
    write_csv_fresh(run_dir / "manifests/resunet_scene_definitions.csv", scene_rows)
    manifest_hash = sha256_file(run_dir / "manifests/resunet_scene_definitions.csv")

    historical_exposed = sum(bool(row["condition_c_ever_exposed"]) for row in audit_rows)
    report = f"""# Atlas source-exposure audit

Status: **PASS — ALL ATLAS GROUPS EXCLUDED FROM RESUNET FITTING**.

- Frozen initial Atlas pairs audited: 25.
- Distinct source groups across frozen pairs and targeted feasibility pairs: {len(excluded)}.
- Groups seen in historical Condition-C training or validation scenes: {historical_exposed}.
- Remaining eligible training rows/groups: {len(allowed['training']):,} / {allowed_groups['training']:,}.
- Remaining eligible validation rows/groups: {len(allowed['validation']):,} / {allowed_groups['validation']:,}.
- Fresh ResUNet scenes: {TRAIN_COUNT:,} training and {VALIDATION_COUNT:,} validation.
- Training/validation group overlap: 0.
- Development and sealed-lockbox scene access: 0 / 0.

Historical Condition-C exposure is reported as development-benchmark context; it does not invalidate Atlas v0. Every group in a frozen Atlas pair, including the selected contaminant in each targeted feasibility pair, is prospectively excluded from both ResUNet training and validation.
"""
    write_text_fresh(run_dir / "diagnostics/atlas_source_exposure_report.md", report)
    write_json_fresh(run_dir / "manifests/source_exclusion_freeze.json", {
        "status": "FROZEN_PASS",
        "excluded_group_count": len(excluded),
        "excluded_groups_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest(),
        "training_remaining_rows": len(allowed["training"]),
        "validation_remaining_rows": len(allowed["validation"]),
        "training_remaining_groups": allowed_groups["training"],
        "validation_remaining_groups": allowed_groups["validation"],
        "scene_manifest_sha256": manifest_hash,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    return {
        "excluded_groups": excluded,
        "excluded_group_count": len(excluded),
        "condition_c_exposed_group_count": historical_exposed,
        "training_remaining_rows": len(allowed["training"]),
        "validation_remaining_rows": len(allowed["validation"]),
        "scene_manifest_sha256": manifest_hash,
    }


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def write_environment(run_dir: Path, hard: dict[str, object], exposure: dict[str, object]) -> None:
    import astropy
    import btk
    import galsim
    import h5py
    import scipy
    import skimage
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    disk = shutil.disk_usage(REPO)
    packages = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "astropy": astropy.__version__,
        "h5py": h5py.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "btk": btk.__version__,
        "galsim": galsim.__version__,
        "surveycodex": package_version("surveycodex"),
    }
    environment = f"""# Prompted ResUNet diversity environment snapshot

- Campaign start UTC: {datetime.now(timezone.utc).isoformat()}
- Branch: `{command(['git', 'branch', '--show-current'])}`
- Git HEAD: `{command(['git', 'rev-parse', 'HEAD'])}`
- Staged index: empty
- MPS built / available / probe: `{torch.backends.mps.is_built()}` / `{torch.backends.mps.is_available()}` / `PASS`
- BTK / GalSim: `{btk.__version__}` / `{galsim.__version__}`
- Free disk at start: {disk.free} bytes
- Frozen Atlas pair-manifest hash: `{hard['atlas_pair_manifest_sha256']}`
- Frozen source-split hash: `{EXPECTED_SPLIT_SHA256}`
- Frozen normalization hash: `{sha256_file(NORMALIZATION)}`
- Condition-C checkpoint hash: `{EXPECTED_CONDITION_C_SHA256}`
- Historical checkpoint count: {hard['historical_checkpoint_count']}
- Atlas-excluded groups: {exposure['excluded_group_count']}

Package versions are recorded in `logs/input_provenance.json`. Neural training and inference are MPS-only; manifests, hashes, metrics, bootstraps, and figures are CPU-only.
"""
    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", environment)
    git_status = command(["git", "status", "--porcelain=v2", "--untracked-files=all"]).splitlines()
    provenance = {
        "run_dir": str(run_dir.relative_to(REPO)),
        "campaign_started_utc": datetime.now(timezone.utc).isoformat(),
        "branch": command(["git", "branch", "--show-current"]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_v2": git_status,
        "staged_index_empty": True,
        "packages": packages,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_probe": "PASS",
        "disk_free_bytes": disk.free,
        "source_catalog": {"path": str(CATALOG.relative_to(REPO)), "sha256": sha256_file(CATALOG)},
        "source_split": {"path": str(SOURCE_SPLIT.relative_to(REPO)), "sha256": sha256_file(SOURCE_SPLIT)},
        "source_layer_contract": {"path": "src/btk_scene.py", "sha256": sha256_file(REPO / "src/btk_scene.py")},
        "prompt_implementation": {"path": "scripts/thayer_select_prompt_ablation_common.py", "sha256": sha256_file(REPO / "scripts/thayer_select_prompt_ablation_common.py")},
        "normalization": {"path": str(NORMALIZATION.relative_to(REPO)), "sha256": sha256_file(NORMALIZATION)},
        "condition_c_checkpoint": {"path": str(CONDITION_C.relative_to(REPO)), "sha256": sha256_file(CONDITION_C)},
        "atlas_pair_manifest": {"path": str((ATLAS / "tables/atlas_pair_manifest.csv").relative_to(REPO)), "sha256": hard["atlas_pair_manifest_sha256"]},
        "atlas_freeze_record": {"path": str((ATLAS / "manifests/atlas_initial_freeze_record.json").relative_to(REPO)), "sha256": sha256_file(ATLAS / "manifests/atlas_initial_freeze_record.json")},
        "atlas_metric_and_witness_code": [
            {"path": "src/competing_hypotheses.py", "sha256": sha256_file(REPO / "src/competing_hypotheses.py")},
            {"path": "scripts/evaluate_deblenders_on_ambiguity_atlas.py", "sha256": sha256_file(REPO / "scripts/evaluate_deblenders_on_ambiguity_atlas.py")},
            {"path": "scripts/evaluate_ambiguity_evidence_baselines.py", "sha256": sha256_file(REPO / "scripts/evaluate_ambiguity_evidence_baselines.py")},
        ],
        "historical_checkpoint_inventory": "tables/checkpoint_inventory_before.csv",
        "all_source_code_hashes": "tables/source_code_hashes_before.csv",
        "atlas_artifact_hashes": "tables/atlas_artifact_hashes_before.csv",
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)


def write_contract_and_preregistration(run_dir: Path, exposure: dict[str, object]) -> None:
    contract = """# Prompted ResUNet candidate-diversity campaign contract

This is a prospective architecture-diversity feasibility campaign. Atlas v0 already passed; its same-family operational witness failed. The campaign adds exactly one compact, freshly initialized prompted ResUNet under the frozen g/r/z plus Gaussian-prompt interface and requested-source zero-background output contract.

The frozen Atlas, targeted feasibility pairs, thresholds, controls, witness semantics, Condition C, all historical checkpoints, historical development scenes, and the final lockbox are immutable. Atlas groups are excluded from ResUNet fitting. Neural work is MPS-only with no CPU fallback. No auditor, uncertainty head, recoverability head, catalog policy, warm start, Condition-C weight import, Atlas tuning, or post-Atlas tuning is authorized. All outputs are fresh and collision-refusing.
"""
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", contract)

    prereg = f"""# Preregistration: prompted ResUNet candidate diversity

Frozen before the corrected campaign model implementation, scene rendering, or
model fitting. An untrained rejected scaffold from the stopped predecessor run
is present in the source tree; it has incompatible dimensions, produced no
checkpoint or scientific output, and is not this campaign's model.

## Scientific hypothesis

A compact residual-block encoder-decoder, initialized and trained independently under the exact Condition-C source-layer contract, will produce candidate differences that materially exceed same-family seed differences while retaining promptability and forward consistency. Useful diversity must add valid model-candidate ambiguity witnesses and improve frozen candidate-diameter detection; raw disagreement caused by error or output artifacts is not success.

## Frozen data and exclusions

- Catalog and source split hashes are frozen in `logs/input_provenance.json`.
- All {exposure['excluded_group_count']} source groups appearing in the 25 frozen Atlas pairs or their targeted feasibility pairs are excluded from both ResUNet training and validation.
- Fresh manifests contain {TRAIN_COUNT:,} training and {VALIDATION_COUNT:,} validation two-source scenes with group-disjoint approved training/validation partitions.
- Requested source is seeded uniform A/B. Positions are symmetric about a seeded midpoint, separation is uniform on [0.8, 3.2] arcsec, and one explicit BTK `add_noise=all` realization is used.
- Historical development and final lockbox scenes remain inaccessible. Access counts must stay 0/0.

## Exact architecture

Input is a four-channel 60x60 tensor: normalized g/r/z blend plus a unit-peak Gaussian coordinate prompt with sigma 2 pixels. Output is three unconstrained linear channels representing the requested noiseless g/r/z source on zero background.

Every encoder/decoder transform is a predeclared residual block: 3x3 convolution, GroupNorm, SiLU, 3x3 convolution, GroupNorm, residual/1x1 projection, then SiLU. Convolutions followed by normalization have no bias. Downsampling uses stride-2 residual blocks; upsampling is bilinear followed by skip concatenation and a residual fusion block.

```text
4x60x60 -> RB(4,16) -> RBs2(16,32) -> RBs2(32,64)
         -> RB(64,64)
         -> up+skip32 -> RB(96,32)
         -> up+skip16 -> RB(48,16) -> 1x1 linear head -> 3x60x60
```

Exact expected trainable parameter count is {EXPECTED_PARAMETER_COUNT:,}, below the frozen 350,000 preferred ceiling and 500,000 absolute ceiling. Condition C has {CONDITION_C_PARAMETER_COUNT:,}; the expected ratio is {EXPECTED_PARAMETER_COUNT / CONDITION_C_PARAMETER_COUNT:.6f}. No Condition-C blocks, encoder/decoder weights, or historical checkpoints may be loaded. Initialization is fresh Kaiming-normal for convolutions, zero convolution bias, GroupNorm scale one/bias zero.

## Training

- Loss: whole-image normalized MSE, identical in meaning to the promptability baseline.
- Optimizer: Adam, learning rate 0.001; no weight decay.
- Scheduler: cosine annealing over exactly 20 epochs.
- Batch size: 8. A smaller frozen value is permitted only after a documented MPS out-of-memory event before any optimizer step; otherwise no change.
- Seed: {TRAINING_SEED} for initialization, minibatch order, and Torch/NumPy/Python.
- Checkpoint selection: minimum validation MSE only; first epoch wins exact ties. Best and final are stored separately.
- Stop on non-finite values/gradients, MPS fallback, manifest/replay mismatch, checkpoint collision, Atlas exposure, or instability defined as validation loss above 10 times epoch-1 loss for two consecutive epochs.

## Pre-Atlas validation gate

Condition C and ResUNet are evaluated on the same fresh non-Atlas validation scenes. ResUNet must satisfy all:

1. finite predictions and finite stable reconstruction metrics;
2. prompt-swap success at least 0.80 (both A/B queries closer to their requested truth than the alternate truth);
3. output-collapse rate at most 0.10 (swapped-output distance below 10% of truth distance);
4. mean whole-image MSE at most 3.0 times Condition C on identical scenes;
5. at least 0.75 of individual queries closer to requested than alternate truth;
6. no source-identity inversion signal: prompt-swap failure at most 0.20 and median signed requested-versus-alternate MSE advantage below zero.

Report whole/source-region MSE, MAE, PSNR, SSIM, per-band flux error, centroid error, prompt perturbation sensitivity, collapse, and confusion. Failure stops before any ResUNet Atlas inference.

## Candidate contract and leakage gate

Both families use identical dimensions, g/r/z order, frozen inverse normalization, electron-per-pixel units, no clipping, zero residual background, prompt alignment, and two-query full decomposition. Candidate hashes and recomposition are recorded. Trivial family leakage is tested with dynamic range, border mean/variance, clipping frequency, zero fraction, total-flux scale, and edge/interior ratios. Any deterministic contract defect is corrected before Atlas evaluation; scientific outputs are never rescaled to increase agreement. A family-ID classifier is not trained.

## One-pass Atlas evaluation

After implementation, training, checkpoint selection, promptability, contract alignment, and threshold/hash revalidation, the selected ResUNet is evaluated exactly once on the 50 frozen noisy Atlas observations and the 25 frozen matched controls. Atlas labels/results may not affect model selection or fitting. One candidate per requested source and a two-query decomposition are saved with hashes, runtime, finite audit, and frozen forward-consistency scores.

## Frozen analyses and gates

Scientific distance uses the existing frozen image (0.25), per-band flux (0.20), color (0.20 mag), and centroid (0.5 mean-PSF-FWHM) limits; valid size/shape distances are descriptive. Same-family reference is the scene-aligned Condition-C/R0/R1 pairwise output distance already in the authoritative Atlas. Bootstrap intervals use 2,000 deterministic resamples clustered by Atlas pair/source groups; 95% percentile intervals are reported.

Architecture-diversity PASS requires compatibility/promptability, median ResUNet-versus-Condition-C primary distance at least 1.25 times the median same-family distance, the 95% cluster-bootstrap lower bound on that median ratio above 1.0, and no trivial contract leakage or catastrophic reconstruction degradation.

Witness-improvement PASS requires at least 25/50 model-candidate witnesses versus 19/50, at least six paired net additions, one-sided exact paired sign probability at most 0.05, a 95% paired cluster-bootstrap improvement interval with lower endpoint above zero, forward consistency for every added witness, and bounded controls.

Diameter PASS requires candidate-diameter AUROC at least 0.60 with its 95% cluster-bootstrap interval lower endpoint above 0.5, recall at the frozen 4% control-FPR threshold at least 0.10 and nonzero, observed control FPR at that frozen threshold at most 0.08, and no family artifact explanation. The historical reference is AUROC 0.4712 and recall 0.

Overall SUCCESS requires promptability, architecture diversity, witness improvement, diameter, zero leakage, and 0/0 development/lockbox access. PARTIAL SUCCESS means genuine diversity or witness gain without diameter PASS. FAILURE means same-cluster behavior, error/artifact-driven diversity, no witness improvement, or non-informative diameter. No threshold changes follow Atlas results.

## Authorized interpretation

SUCCESS authorizes one third genuinely distinct family, not an auditor. PARTIAL SUCCESS preserves ResUNet and recommends one third classical constrained or fundamentally different family. FAILURE recommends explicit multi-hypothesis generation or posterior sampling, not another deterministic U-Net variant. No result establishes model-agnostic transfer.
"""
    path = run_dir / "preregistration/prompted_resunet_candidate_diversity.md"
    write_text_fresh(path, prereg)
    frozen_at = datetime.now(timezone.utc).isoformat()
    digest = sha256_file(path)
    write_json_fresh(run_dir / "preregistration/freeze_record.json", {
        "status": "FROZEN_BEFORE_CORRECTED_MODEL_IMPLEMENTATION_RENDERING_OR_FITTING",
        "frozen_at_utc": frozen_at,
        "preregistration_path": str(path.relative_to(run_dir)),
        "preregistration_sha256": digest,
        "scene_manifest_sha256": exposure["scene_manifest_sha256"],
        "atlas_evaluation_count_at_freeze": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })


def main() -> None:
    run_dir = create_run()
    try:
        hard = verify_hard_gates(run_dir)
        exposure = atlas_exposure_and_scenes(run_dir)
        write_environment(run_dir, hard, exposure)
        write_contract_and_preregistration(run_dir, exposure)
        write_json_fresh(run_dir / "logs/bootstrap_complete.json", {
            "status": "PASS",
            "run_dir": str(run_dir.relative_to(REPO)),
            "preregistration_sha256": sha256_file(run_dir / "preregistration/prompted_resunet_candidate_diversity.md"),
            "scene_manifest_sha256": exposure["scene_manifest_sha256"],
            "development_scene_access_count": 0,
            "lockbox_scene_access_count": 0,
        })
    except Exception as error:
        write_json_fresh(run_dir / "logs/bootstrap_failure.json", {"status": "FAIL", "error": repr(error)})
        raise
    print(run_dir.relative_to(REPO))


if __name__ == "__main__":
    main()
