#!/usr/bin/env python3
"""Execute, calibrate, audit, and finalize the preregistered Thayer-Audit v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import CompactSelectNet, gaussian_prompt_numpy
from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene
from src.direct_catalog_safety_auditor import (
    PRE_CLASSES,
    PRE_TO_INDEX,
    QUERY_TO_PRE,
    SCALAR_FEATURE_NAMES,
    PostAuditSafetyNetwork,
    PreAuditQueryNetwork,
    binary_metrics,
    connected_components,
    deployable_scalar_features,
    fit_binary_temperature,
    fit_multiclass_temperature,
    inverse_frequency_class_weights,
    multiclass_metrics,
    normalized_post_image,
    normalized_pre_image,
    percentile_interval,
    policy_metrics,
    post_audit_supervision,
    select_fail_closed_threshold,
    sigmoid,
    softmax,
    threshold_constraints,
    trainable_parameter_count,
)


HIERARCHICAL = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
PROMPT_ABLATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PHASE2 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
THAYER_PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
PROMPTED_RESUNET = REPO / "outputs/runs/thayer_prompted_resunet_diversity_20260712_153956"
CONDITION_C = PROMPT_ABLATION / "checkpoints/c_randomized_coordinate_prompt_best.pth"
CONDITION_C_SCENES = PROMPT_ABLATION / "manifests/development_scene_definitions.csv"
NORMALIZATION = PROMPT_ABLATION / "manifests/normalization.json"
CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"

SEEDS = (2026071501, 2026071502, 2026071503)
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
BOOTSTRAP_REPLICATES = 300
FAMILY = "THAYER_SELECT_CONDITION_C"
FAMILY_CLUSTER = "THAYER_COMPACT_PROMPTED_UNET"

EPISODE_SPECS = {
    "pre_training": ("q_training", "pre", True, False),
    "pre_validation": ("q_validation", "pre", False, False),
    "pre_calibration": ("natural_calibration", "pre", False, False),
    "post_training": ("r_training", "post", True, True),
    "post_validation": ("r_validation", "post", False, True),
    "policy_validation": ("q_validation", "policy", False, False),
    "policy_calibration": ("natural_calibration", "policy", False, False),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def command(arguments: list[str]) -> dict[str, object]:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def validate_run(run: Path, required_stage: str | None = None) -> None:
    if run.parent.resolve() != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_audit_v0_"):
        raise RuntimeError("invalid Thayer-Audit run path")
    if not run.is_dir():
        raise FileNotFoundError(run)
    prereg = json.loads((run / "logs/preregistration_complete.json").read_text())
    path = REPO / prereg["path"]
    if prereg["status"] != "FROZEN_BEFORE_EPISODES_OR_FITTING" or sha256_file(path) != prereg["sha256"]:
        raise RuntimeError("preregistration hash mismatch")
    staged = command(["git", "diff", "--cached", "--name-status"])
    if staged["returncode"] != 0 or str(staged["stdout"]).strip():
        raise RuntimeError("staged index is not empty")
    if required_stage is not None:
        record = json.loads((run / "logs" / required_stage).read_text())
        if record["status"] != "PASS":
            raise RuntimeError(f"required stage did not pass: {required_stage}")


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS probe failed")
    return torch.device("mps")


def source_paths(dataset: str) -> tuple[Path, Path, Path]:
    return (
        HIERARCHICAL / f"manifests/v2_{dataset}_scene_manifest.csv",
        HIERARCHICAL / f"manifests/v2_{dataset}_scenes.h5",
        HIERARCHICAL / f"features/v2_{dataset}_frozen_reconstructions.h5",
    )


def selected_indices(dataset: str, training_oof: bool) -> tuple[pd.DataFrame, np.ndarray, str]:
    manifest_path, _, _ = source_paths(dataset)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False, low_memory=False)
    if not training_oof:
        return manifest, np.arange(len(manifest), dtype=int), "FROZEN_PARTITION_OUTPUT"
    base = pd.read_csv(CONDITION_C_SCENES, dtype=str, keep_default_na=False)
    used = set(base.loc[base.partition.isin(("training", "validation")), "source_a_group"])
    used.update(base.loc[base.partition.isin(("training", "validation")), "source_b_group"])
    eligible = ~manifest.source_a_group.isin(used) & ~manifest.source_b_group.isin(used)
    indices = np.flatnonzero(eligible.to_numpy())
    if len(indices) == 0:
        raise RuntimeError(f"no out-of-base-fit rows: {dataset}")
    if manifest.loc[eligible, "source_a_group"].isin(used).any() or manifest.loc[eligible, "source_b_group"].isin(used).any():
        raise RuntimeError("OOF group exclusion failed")
    return manifest, indices, "OUT_OF_HISTORICAL_BASE_FOLD"


def frozen_family_inventory(run: Path) -> pd.DataFrame:
    candidates = [
        {
            "family_id": FAMILY, "family_cluster": FAMILY_CLUSTER, "classification": "primary valid reconstructor",
            "checkpoint_path": relative(CONDITION_C), "checkpoint_sha256": sha256_file(CONDITION_C),
            "promptability": "PASS", "aligned_train_validation_calibration_outputs": True,
            "core_eligible": True, "held_family_eligible": True,
            "reason": "Immutable promptable requested-source output with persisted aligned outputs and source-group-excluded training subset.",
        },
        {
            "family_id": "THAYER_SELECT_R0_R1_CLUSTER", "family_cluster": FAMILY_CLUSTER,
            "classification": "negative/failure-domain family", "checkpoint_path": relative(PHASE2 / "checkpoints/r0_best.pth"),
            "checkpoint_sha256": sha256_file(PHASE2 / "checkpoints/r0_best.pth"), "promptability": "PASS",
            "aligned_train_validation_calibration_outputs": False, "core_eligible": False, "held_family_eligible": False,
            "reason": "Same architecture cluster as Condition C and no complete aligned persisted audit-episode outputs; cannot create family diversity.",
        },
        {
            "family_id": "THAYER_PU", "family_cluster": "THAYER_PROMPTED_PROBABILISTIC_UNET",
            "classification": "stochastic candidate family", "checkpoint_path": relative(THAYER_PU / "checkpoints/thayer_pu_best.pth"),
            "checkpoint_sha256": sha256_file(THAYER_PU / "checkpoints/thayer_pu_best.pth"), "promptability": "PASS",
            "aligned_train_validation_calibration_outputs": False, "core_eligible": False, "held_family_eligible": False,
            "reason": "No complete scene-aligned out-of-fit requested-source outputs under one frozen sampling rule; truth coverage failed and no new outputs are generated merely to increase family count.",
        },
        {
            "family_id": "PROMPTED_RESUNET", "family_cluster": "PROMPTED_RESUNET",
            "classification": "ineligible", "checkpoint_path": "NOT_ADMITTED", "checkpoint_sha256": "NOT_ADMITTED",
            "promptability": "FAIL_0.3947", "aligned_train_validation_calibration_outputs": False,
            "core_eligible": False, "held_family_eligible": False,
            "reason": "Failed its frozen 80% prompt-swap gate before Atlas admission.",
        },
    ]
    frame = pd.DataFrame(candidates)
    write_csv_fresh(run / "tables/frozen_deblender_family_inventory.csv", frame)
    write_text_fresh(run / "diagnostics/frozen_family_eligibility.md", f"""# Frozen deblender-family eligibility

Exactly one family, `{FAMILY}`, is scientifically eligible for the core experiment. R0/R1 share its architecture cluster and cannot count as distinct. Thayer-PU lacks complete aligned out-of-fit outputs under one frozen deployment sampling rule, and the prompted ResUNet failed promptability. The core therefore proceeds with one valid family, while deblender-family generalization and leave-one-family-out evaluation remain **UNRESOLVED**. No family was admitted merely to increase the count.
""")
    return frame


def create_episode(run: Path, name: str, dataset: str, kind: str, training_oof: bool, valid_only: bool, scales: np.ndarray) -> pd.DataFrame:
    manifest, indices, provenance_status = selected_indices(dataset, training_oof)
    if valid_only:
        indices = indices[manifest.iloc[indices].query_state.to_numpy() == "UNIQUE_VALID"]
    manifest_path, scene_path, reconstruction_path = source_paths(dataset)
    output = run / f"episodes/{name}.h5"
    output_manifest = run / f"episodes/{name}_manifest.csv"
    if output.exists() or output_manifest.exists():
        raise FileExistsError(output)
    rows: list[dict[str, object]] = []
    channels = 4 if kind == "pre" else 10
    with h5py.File(scene_path, "r") as scenes, h5py.File(reconstruction_path, "r") as reconstructions, h5py.File(output, "x") as target:
        image_ds = target.create_dataset("image", shape=(len(indices), channels, 60, 60), dtype="f4", chunks=(1, channels, 60, 60), compression="lzf")
        label_ds = target.create_dataset("label", shape=(len(indices),), dtype="i1")
        if kind != "pre":
            scalar_ds = target.create_dataset("scalar", shape=(len(indices), len(SCALAR_FEATURE_NAMES)), dtype="f4", chunks=(128, len(SCALAR_FEATURE_NAMES)), compression="lzf")
            catastrophic_ds = target.create_dataset("catastrophic", shape=(len(indices),), dtype="i1")
        for output_index, upstream_index in enumerate(indices):
            row = manifest.iloc[int(upstream_index)]
            blend = np.asarray(scenes["blend"][upstream_index], dtype=np.float32)
            prompt = np.asarray(scenes["prompt"][upstream_index], dtype=np.float32)
            reconstruction = np.asarray(reconstructions["reconstruction"][upstream_index], dtype=np.float32)
            query_label = PRE_TO_INDEX[QUERY_TO_PRE[row.query_state]]
            supervision: dict[str, object] = {}
            if kind == "pre":
                image_ds[output_index] = normalized_pre_image(blend, prompt, scales)
                label_ds[output_index] = query_label
            else:
                image_ds[output_index] = normalized_post_image(blend, prompt, reconstruction, scales)
                scalar_ds[output_index] = deployable_scalar_features(blend, prompt, reconstruction)
                if row.query_state == "UNIQUE_VALID":
                    matched = int(scenes["matched_index"][upstream_index])
                    if matched not in (0, 1):
                        raise RuntimeError("valid row lacks a matched source")
                    isolated = np.asarray(scenes["isolated"][upstream_index], dtype=np.float32)
                    result = post_audit_supervision(reconstruction, isolated[matched], isolated[1 - matched], blend)
                    supervision = result.to_dict()
                    label_ds[output_index] = int(result.unsafe_to_catalog)
                    catastrophic_ds[output_index] = int(result.catastrophic)
                else:
                    label_ds[output_index] = -1
                    catastrophic_ds[output_index] = -1
            base = {
                "episode_index": output_index, "scene_id": row.scene_id, "partition": row.source_partition,
                "query_state": row.query_state, "pre_audit_label": QUERY_TO_PRE[row.query_state],
                "prompt_id": f"{row.scene_id}:{row.prompt_sha256}", "prompt_subtype": row.prompt_subtype,
                "source_a_id": row.source_a_id, "source_b_id": row.source_b_id,
                "source_a_group": row.source_a_group, "source_b_group": row.source_b_group,
                "matched_source_id": row.matched_source_id, "matched_source_group": row.matched_source_group,
                "deblender_family": FAMILY, "deblender_family_cluster": FAMILY_CLUSTER,
                "deblender_seed": 2026074101, "family_metadata_input": False,
                "upstream_dataset": dataset, "upstream_index": int(upstream_index),
                "upstream_manifest_path": relative(manifest_path), "upstream_scene_path": relative(scene_path),
                "upstream_reconstruction_path": relative(reconstruction_path),
                "blend_sha256": row.blend_sha256, "prompt_sha256": row.prompt_sha256,
                "reconstructor_checkpoint_sha256": sha256_file(CONDITION_C),
                "base_prediction_provenance": provenance_status,
                "both_groups_excluded_from_base_fit_and_selection": bool(training_oof),
                "truth_derived_inference_feature_count": 0,
            }
            rows.append({**base, **supervision})
    frame = pd.DataFrame(rows)
    write_csv_fresh(output_manifest, frame)
    return frame


def prevalence_tables(run: Path, manifests: dict[str, pd.DataFrame]) -> None:
    target_rows = [
        {"stage": "PRE", "target": "VALID", "applicability": "all requests", "definition": "exact UNIQUE_VALID query semantics"},
        {"stage": "PRE", "target": "NULL_OR_WRONG", "applicability": "all requests", "definition": "exact NULL query semantics"},
        {"stage": "PRE", "target": "AMBIGUOUS_OR_UNSUPPORTED", "applicability": "all requests", "definition": "exact AMBIGUOUS query semantics"},
        {"stage": "POST", "target": "UNSAFE_TO_CATALOG", "applicability": "VALID only", "definition": "OR of frozen scientific, physical, false-subtraction, worse-baseline, and confusion failures"},
        {"stage": "POST", "target": "SAFE_TO_CATALOG", "applicability": "VALID only", "definition": "no applicable unsafe component"},
    ]
    write_csv_fresh(run / "tables/audit_target_inventory.csv", pd.DataFrame(target_rows))
    rows: list[dict[str, object]] = []
    for name, frame in manifests.items():
        for label, group in frame.groupby("pre_audit_label", sort=True):
            rows.append({"episode_set": name, "aggregation": "query_class", "group": label, "rows": len(group),
                         "unsafe_prevalence": float(group.unsafe_to_catalog.mean()) if "unsafe_to_catalog" in group and group.unsafe_to_catalog.notna().any() else math.nan})
        if "unsafe_to_catalog" in frame:
            valid = frame[frame.unsafe_to_catalog.notna()].copy()
            rows.append({"episode_set": name, "aggregation": "partition", "group": str(frame.partition.iloc[0]), "rows": len(valid),
                         "unsafe_prevalence": float(valid.unsafe_to_catalog.mean()) if len(valid) else math.nan})
            rows.append({"episode_set": name, "aggregation": "deblender_family", "group": FAMILY, "rows": len(valid),
                         "unsafe_prevalence": float(valid.unsafe_to_catalog.mean()) if len(valid) else math.nan})
            exploded = pd.concat((valid.assign(source_group=valid.source_a_group), valid.assign(source_group=valid.source_b_group)))
            for source_group, subset in exploded.groupby("source_group", sort=True):
                rows.append({"episode_set": name, "aggregation": "source_group", "group": source_group, "rows": len(subset),
                             "unsafe_prevalence": float(subset.unsafe_to_catalog.mean())})
    write_csv_fresh(run / "tables/audit_target_prevalence.csv", pd.DataFrame(rows))


def episodes(run: Path) -> None:
    validate_run(run)
    if (run / "logs/episode_construction_complete.json").exists():
        raise FileExistsError("episodes already constructed")
    started = time.time()
    inventory = frozen_family_inventory(run)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    manifests = {}
    for name, (dataset, kind, training_oof, valid_only) in EPISODE_SPECS.items():
        manifests[name] = create_episode(run, name, dataset, kind, training_oof, valid_only, scales)

    base = pd.read_csv(CONDITION_C_SCENES, dtype=str, keep_default_na=False)
    base_groups = set(base.loc[base.partition.isin(("training", "validation")), "source_a_group"]) | set(base.loc[base.partition.isin(("training", "validation")), "source_b_group"])
    for name in ("pre_training", "post_training"):
        groups = set(manifests[name].source_a_group) | set(manifests[name].source_b_group)
        if groups & base_groups:
            raise RuntimeError(f"in-sample reconstruction output entered {name}")
    train_groups = set(manifests["pre_training"].source_a_group) | set(manifests["pre_training"].source_b_group) | set(manifests["post_training"].source_a_group) | set(manifests["post_training"].source_b_group)
    validation_groups = set(manifests["pre_validation"].source_a_group) | set(manifests["pre_validation"].source_b_group) | set(manifests["post_validation"].source_a_group) | set(manifests["post_validation"].source_b_group)
    calibration_groups = set(manifests["pre_calibration"].source_a_group) | set(manifests["pre_calibration"].source_b_group)
    if train_groups & validation_groups or train_groups & calibration_groups or validation_groups & calibration_groups:
        raise RuntimeError("source-group leakage across auditor partitions")

    prevalence_tables(run, manifests)
    post_train_prevalence = float(manifests["post_training"].unsafe_to_catalog.mean())
    write_csv_fresh(run / "tables/deployable_feature_inventory.csv", pd.DataFrame(
        [{"stage": "PRE", "feature": value, "truth_free": True} for value in ("blend_g", "blend_r", "blend_z", "gaussian_prompt")]
        + [{"stage": "POST_IMAGE", "feature": value, "truth_free": True} for value in ("blend_g", "blend_r", "blend_z", "gaussian_prompt", "reconstruction_g", "reconstruction_r", "reconstruction_z", "residual_g", "residual_r", "residual_z")]
        + [{"stage": "POST_SCALAR", "feature": value, "truth_free": True} for value in SCALAR_FEATURE_NAMES]
    ))
    write_text_fresh(run / "diagnostics/deployable_feature_contract.md", """# Deployable feature contract

PRE-AUDIT contains only the observed normalized g/r/z blend and Gaussian prompt. POST-AUDIT contains only the normalized blend, prompt, frozen proposed g/r/z reconstruction, observation-minus-reconstruction residual, and the 25 frozen scalar diagnostics. Episode HDF5 inference arrays contain no truth, target mask, true error, source/group/family identity, physical difficulty, SNR, obstruction, separation, flux ratio, morphology, generator parameter, gradient, optimizer, D3 trajectory, or outcome field. Labels and provenance remain in separate aligned datasets/manifests.
""")
    write_text_fresh(run / "diagnostics/audit_target_contract.md", """# Audit target contract

PRE-AUDIT retains the three exact query-semantic classes. POST-AUDIT applies only to valid requests and uses the preregistered OR of frozen scientific, physical output, false-subtraction, worse-than-baseline, and confusion failures. Null and ambiguous requests never receive a valid-query safety label.
""")
    episode_rows = []
    for name, frame in manifests.items():
        h5_path = run / f"episodes/{name}.h5"
        csv_path = run / f"episodes/{name}_manifest.csv"
        episode_rows.append({"episode_set": name, "rows": len(frame), "partition": frame.partition.iloc[0],
                             "h5_sha256": sha256_file(h5_path), "manifest_sha256": sha256_file(csv_path),
                             "oof_rows": int((frame.base_prediction_provenance == "OUT_OF_HISTORICAL_BASE_FOLD").sum()),
                             "truth_derived_inference_features": int(frame.truth_derived_inference_feature_count.sum())})
    write_csv_fresh(run / "tables/audit_episode_inventory.csv", pd.DataFrame(episode_rows))
    attainable = post_train_prevalence <= 0.85
    write_csv_fresh(run / "tables/post_label_gate_attainability.csv", pd.DataFrame([{
        "gate": "validation_auprc_at_least_prevalence_plus_0.15", "training_unsafe_prevalence": post_train_prevalence,
        "maximum_possible_absolute_lift": 1.0 - post_train_prevalence, "attainable_from_observed_label_support": attainable,
        "gate_changed": False,
    }]))
    write_json_fresh(run / "logs/episode_construction_complete.json", {
        "status": "PASS", "runtime_seconds": time.time() - started, "episode_sets": {name: len(frame) for name, frame in manifests.items()},
        "pre_training_oof_rows": len(manifests["pre_training"]), "post_training_oof_rows": len(manifests["post_training"]),
        "source_group_leakage_count": 0, "truth_derived_inference_feature_count": 0,
        "eligible_family_count": int(inventory.core_eligible.sum()), "held_family_generalization": "UNRESOLVED",
        "post_auprc_gate_empirically_attainable": bool(attainable), "development_outcomes_accessed": False,
        "final_lockbox_outcomes_accessed": False,
    })
    print(json.dumps({"status": "PASS", "episode_rows": {name: len(frame) for name, frame in manifests.items()}, "post_prevalence": post_train_prevalence}, sort_keys=True))


def load_episode(run: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with h5py.File(run / f"episodes/{name}.h5", "r") as handle:
        image = np.asarray(handle["image"], dtype=np.float32)
        label = np.asarray(handle["label"], dtype=np.int64)
        scalar = np.asarray(handle["scalar"], dtype=np.float32) if "scalar" in handle else None
        catastrophic = np.asarray(handle["catastrophic"], dtype=np.int64) if "catastrophic" in handle else None
    return image, label, scalar, catastrophic


def batches(length: int, *, seed: int | None = None) -> list[np.ndarray]:
    indices = np.arange(length)
    if seed is not None:
        indices = np.random.default_rng(seed).permutation(indices)
    return [indices[start:start + BATCH_SIZE] for start in range(0, length, BATCH_SIZE)]


def predict_model(model: torch.nn.Module, image: np.ndarray, scalar: np.ndarray | None, device: torch.device) -> np.ndarray:
    model.eval(); output = []
    with torch.no_grad():
        for batch in batches(len(image)):
            x = torch.from_numpy(image[batch]).to(device)
            if scalar is None:
                value = model(x)
            else:
                value = model(x, torch.from_numpy(scalar[batch]).to(device))
            output.append(value.detach().cpu().numpy())
    return np.concatenate(output)


def train_seed(task: str, seed: int, train: tuple[np.ndarray, np.ndarray, np.ndarray | None], validation: tuple[np.ndarray, np.ndarray, np.ndarray | None], device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(seed); torch.mps.manual_seed(seed); np.random.seed(seed % (2**32 - 1))
    train_image, train_label, train_scalar = train
    valid_image, valid_label, valid_scalar = validation
    model = PreAuditQueryNetwork().to(device) if task == "pre" else PostAuditSafetyNetwork().to(device)
    ceiling = 100_000 if task == "pre" else 350_000
    if trainable_parameter_count(model) > ceiling:
        raise RuntimeError("auditor parameter ceiling exceeded")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    class_weights = inverse_frequency_class_weights(train_label, 3 if task == "pre" else 2)
    class_weight_tensor = torch.from_numpy(class_weights).to(device)
    curves: list[dict[str, object]] = []
    best_state = None; best_metrics = None; best_key = None; stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); losses = []
        for batch in batches(len(train_image), seed=seed + epoch):
            x = torch.from_numpy(train_image[batch]).to(device)
            y = torch.from_numpy(train_label[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            if task == "pre":
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=class_weight_tensor)
            else:
                logits = model(x, torch.from_numpy(train_scalar[batch]).to(device))
                per_row = F.binary_cross_entropy_with_logits(logits, y.to(torch.float32), reduction="none")
                loss = torch.mean(per_row * class_weight_tensor[y])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite auditor loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        logits = predict_model(model, valid_image, valid_scalar, device)
        if task == "pre":
            metrics = multiclass_metrics(valid_label, softmax(logits))
            key = (float(metrics["macro_f1"]), float(metrics["recall_by_class"][2]), -float(metrics["cross_entropy"]))
        else:
            metrics = binary_metrics(sigmoid(logits), valid_label)
            key = (float(metrics["auprc"]) if np.isfinite(metrics["auprc"]) else -math.inf,
                   float(metrics["auroc"]) if np.isfinite(metrics["auroc"]) else -math.inf,
                   -float(metrics["brier"]))
        curves.append({"task": task, "seed": seed, "epoch": epoch, "training_loss": float(np.mean(losses)),
                       **{key_name: value for key_name, value in metrics.items() if not isinstance(value, (list, dict))}})
        if best_key is None or key > best_key:
            best_key = key; best_metrics = metrics; stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("no auditor checkpoint selected")
    summary = {"task": task, "seed": seed, "best_epoch": best_epoch, "epochs_run": len(curves),
               "class_weights": class_weights.astype(float).tolist(), "parameter_count": trainable_parameter_count(model),
               **{key: value for key, value in best_metrics.items() if not isinstance(value, (list, dict))}}
    if task == "pre":
        summary.update({f"recall_{PRE_CLASSES[index].lower()}": float(best_metrics["recall_by_class"][index]) for index in range(3)})
    return best_state, curves, summary


def train(run: Path) -> None:
    validate_run(run, "episode_construction_complete.json")
    if (run / "logs/training_complete.json").exists():
        raise FileExistsError("training already complete")
    device = require_mps(); started = time.time()
    pre_train_image, pre_train_label, _, _ = load_episode(run, "pre_training")
    pre_valid_image, pre_valid_label, _, _ = load_episode(run, "pre_validation")
    post_train_image, post_train_label, post_train_scalar, _ = load_episode(run, "post_training")
    post_valid_image, post_valid_label, post_valid_scalar, _ = load_episode(run, "post_validation")
    scalar_mean = post_train_scalar.mean(axis=0).astype(np.float32)
    scalar_scale = post_train_scalar.std(axis=0).astype(np.float32)
    scalar_scale[scalar_scale < 1e-6] = 1.0
    post_train_scalar = ((post_train_scalar - scalar_mean) / scalar_scale).astype(np.float32)
    post_valid_scalar = ((post_valid_scalar - scalar_mean) / scalar_scale).astype(np.float32)
    write_json_fresh(run / "features/post_scalar_standardization.json", {
        "feature_names": list(SCALAR_FEATURE_NAMES), "mean": scalar_mean.astype(float).tolist(),
        "scale": scalar_scale.astype(float).tolist(), "fit_partition": "post_training_out_of_historical_base_fold",
    })
    all_curves = []; summaries = []
    for task, training, validation in (
        ("pre", (pre_train_image, pre_train_label, None), (pre_valid_image, pre_valid_label, None)),
        ("post", (post_train_image, post_train_label, post_train_scalar), (post_valid_image, post_valid_label, post_valid_scalar)),
    ):
        for seed in SEEDS:
            state, curves, summary = train_seed(task, seed, training, validation, device)
            checkpoint = run / f"checkpoints/{task}_auditor_seed_{seed}.pth"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            torch.save({"state_dict": state, "task": task, "seed": seed, "selection_partition": "validation",
                        "selection_metrics": summary, "mps_training": True, "reconstruction_parameters_trained": 0}, checkpoint)
            all_curves.extend(curves); summaries.append({**summary, "checkpoint_path": relative(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)})
    write_csv_fresh(run / "tables/auditor_training_curves.csv", pd.DataFrame(all_curves))
    write_csv_fresh(run / "tables/auditor_seed_metrics.csv", pd.DataFrame(summaries))
    parameter_frame = pd.DataFrame([
        {"architecture": "A1_PRE", "parameters": trainable_parameter_count(PreAuditQueryNetwork()), "ceiling": 100_000, "status": "PASS"},
        {"architecture": "A2_POST", "parameters": trainable_parameter_count(PostAuditSafetyNetwork()), "ceiling": 350_000, "status": "PASS"},
    ])
    write_csv_fresh(run / "tables/auditor_architecture_inventory.csv", parameter_frame)
    curves = pd.DataFrame(all_curves)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, task in zip(axes, ("pre", "post")):
        for seed, group in curves[curves.task == task].groupby("seed"):
            axis.plot(group.epoch, group.training_loss, label=str(seed))
        axis.set_title(task.upper()); axis.set_xlabel("epoch"); axis.set_ylabel("weighted training loss"); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(run / "figures/training_curves.png", dpi=170); plt.close(fig)
    write_json_fresh(run / "logs/training_complete.json", {
        "status": "PASS", "runtime_seconds": time.time() - started, "device": "mps", "seeds": list(SEEDS),
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "gradient_clip": GRADIENT_CLIP, "reconstruction_models_trained": 0, "cpu_neural_fallbacks": 0,
        "validation_only_checkpoint_selection": True, "calibration_used_for_selection": False,
        "development_outcomes_accessed": False, "final_lockbox_outcomes_accessed": False,
    })
    print(json.dumps({"status": "PASS", "runtime_seconds": time.time() - started, "seed_checkpoints": len(summaries)}, sort_keys=True))


def scalar_standardization(run: Path, values: np.ndarray) -> np.ndarray:
    payload = json.loads((run / "features/post_scalar_standardization.json").read_text())
    mean = np.asarray(payload["mean"], dtype=np.float32); scale = np.asarray(payload["scale"], dtype=np.float32)
    return ((values - mean) / scale).astype(np.float32)


def ensemble_logits(run: Path, task: str, episode: str, *, return_seed: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    device = require_mps(); image, _, scalar, _ = load_episode(run, episode)
    if scalar is not None:
        scalar = scalar_standardization(run, scalar)
    blocks = []
    for seed in SEEDS:
        model = PreAuditQueryNetwork() if task == "pre" else PostAuditSafetyNetwork()
        payload = torch.load(run / f"checkpoints/{task}_auditor_seed_{seed}.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"]); model.to(device)
        blocks.append(predict_model(model, image, scalar, device))
        del model
    stacked = np.stack(blocks)
    return (stacked.mean(axis=0), stacked) if return_seed else stacked.mean(axis=0)


def reliability_plot(path: Path, probability: np.ndarray, truth: np.ndarray, title: str) -> None:
    score = np.asarray(probability); y = np.asarray(truth); xs=[]; ys=[]; sizes=[]
    edges = np.linspace(0, 1, 11)
    for index in range(10):
        selected = (score >= edges[index]) & (score < edges[index + 1] if index < 9 else score <= edges[index + 1])
        if np.any(selected):
            xs.append(float(score[selected].mean())); ys.append(float(y[selected].mean())); sizes.append(int(selected.sum()))
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--"); ax.scatter(xs, ys, s=np.maximum(20, np.sqrt(sizes) * 8))
    ax.set(xlabel="predicted probability", ylabel="empirical frequency", title=title, xlim=(0, 1), ylim=(0, 1)); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def frozen_b1_metrics(new_validation_truth: np.ndarray, new_calibration_truth: np.ndarray) -> list[dict[str, object]]:
    try:
        from calibrate_hierarchical_feasibility import binary_logits as legacy_binary_logits
        with np.load(HIERARCHICAL / "features/v2_r_validation_features.npz", allow_pickle=True) as validation_features:
            validation_logits = legacy_binary_logits(HIERARCHICAL, validation_features, "catastrophic")
        with np.load(HIERARCHICAL / "features/v2_natural_calibration_features.npz", allow_pickle=True) as calibration_features:
            calibration_logits_all = legacy_binary_logits(HIERARCHICAL, calibration_features, "catastrophic")
        summary = pd.read_csv(HIERARCHICAL / "tables/binary_risk_calibration_summary.csv")
        temperature = float(summary.loc[summary.task == "catastrophic", "temperature"].iloc[0])
        calibration_samples = pd.read_csv(HIERARCHICAL / "features/v4_natural_calibration_samples.csv", keep_default_na=False)
        valid = calibration_samples.query_state.to_numpy() == "UNIQUE_VALID"
        validation = binary_metrics(sigmoid(validation_logits / temperature), new_validation_truth)
        calibration = binary_metrics(sigmoid(calibration_logits_all[valid] / temperature), new_calibration_truth)
        return [{"baseline": "B1_frozen_hierarchical_catastrophic_scalar", "partition": "validation", **validation},
                {"baseline": "B1_frozen_hierarchical_catastrophic_scalar", "partition": "calibration", **calibration}]
    except Exception as error:
        return [{"baseline": "B1_frozen_hierarchical_catastrophic_scalar", "partition": "replay", "status": "UNAVAILABLE", "reason": repr(error)}]


def bootstrap_policy(run: Path, pre_probability: np.ndarray, post_probability: np.ndarray, threshold: float) -> pd.DataFrame:
    _, query, _, _ = load_episode(run, "pre_calibration")
    _, unsafe, _, catastrophic = load_episode(run, "policy_calibration")
    manifest = pd.read_csv(run / "episodes/policy_calibration_manifest.csv", keep_default_na=False)
    components = connected_components(zip(manifest.source_a_group.astype(str), manifest.source_b_group.astype(str)))
    component_ids = np.unique(components); by_component = {value: np.flatnonzero(components == value) for value in component_ids}
    rng = np.random.default_rng(2026071599); rows=[]
    pre_prediction = np.argmax(pre_probability, axis=1)
    unsafe_full = np.where(unsafe < 0, 0, unsafe); catastrophic_full = np.where(catastrophic < 0, 0, catastrophic)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(component_ids, size=len(component_ids), replace=True)
        indices = np.concatenate([by_component[value] for value in sampled])
        pre_metrics = multiclass_metrics(query[indices], pre_probability[indices])
        valid = unsafe[indices] >= 0
        post_metrics = binary_metrics(post_probability[indices][valid], unsafe[indices][valid])
        policy = policy_metrics(query[indices], pre_prediction[indices], post_probability[indices], unsafe_full[indices], catastrophic_full[indices], threshold)
        rows.append({"replicate": replicate, "macro_f1": pre_metrics["macro_f1"],
                     "null_recall": pre_metrics["recall_by_class"][1], "ambiguous_recall": pre_metrics["recall_by_class"][2],
                     "auroc": post_metrics["auroc"], "auprc": post_metrics["auprc"], "brier": post_metrics["brier"], "ece": post_metrics["ece"],
                     "coverage": policy["accepted_coverage"], "accepted_unsafe_rate": policy["accepted_unsafe_rate"],
                     "unsafe_rate_reduction": policy["unsafe_rate_reduction"], "catastrophic_rate_reduction": policy["catastrophic_rate_reduction"],
                     "null_acceptance": policy["null_acceptance"], "ambiguous_acceptance": policy["ambiguous_acceptance"]})
    return pd.DataFrame(rows)


def calibration_and_policy(run: Path) -> None:
    validate_run(run, "training_complete.json")
    if (run / "logs/calibration_complete.json").exists():
        raise FileExistsError("calibration already complete")
    started = time.time()
    pre_validation_logits, pre_seed_logits = ensemble_logits(run, "pre", "pre_validation", return_seed=True)
    pre_calibration_logits = ensemble_logits(run, "pre", "pre_calibration")
    post_validation_logits, post_seed_logits = ensemble_logits(run, "post", "post_validation", return_seed=True)
    post_policy_validation_logits = ensemble_logits(run, "post", "policy_validation")
    post_policy_calibration_logits = ensemble_logits(run, "post", "policy_calibration")
    _, pre_validation_truth, _, _ = load_episode(run, "pre_validation")
    _, pre_calibration_truth, _, _ = load_episode(run, "pre_calibration")
    _, post_validation_truth, _, _ = load_episode(run, "post_validation")
    _, policy_validation_unsafe, _, policy_validation_catastrophic = load_episode(run, "policy_validation")
    _, policy_calibration_unsafe, _, policy_calibration_catastrophic = load_episode(run, "policy_calibration")

    pre_temperature = fit_multiclass_temperature(pre_calibration_logits, pre_calibration_truth)
    valid_calibration = policy_calibration_unsafe >= 0
    post_temperature = fit_binary_temperature(post_policy_calibration_logits[valid_calibration], policy_calibration_unsafe[valid_calibration])
    pre_validation_probability = softmax(pre_validation_logits / pre_temperature)
    pre_calibration_probability = softmax(pre_calibration_logits / pre_temperature)
    post_validation_probability = sigmoid(post_validation_logits / post_temperature)
    post_policy_validation_probability = sigmoid(post_policy_validation_logits / post_temperature)
    post_policy_calibration_probability = sigmoid(post_policy_calibration_logits / post_temperature)

    pre_rows=[]
    for partition, truth, raw_logits, calibrated in (
        ("validation", pre_validation_truth, pre_validation_logits, pre_validation_probability),
        ("calibration", pre_calibration_truth, pre_calibration_logits, pre_calibration_probability),
    ):
        for state, probability in (("raw", softmax(raw_logits)), ("temperature", calibrated)):
            metrics = multiclass_metrics(truth, probability)
            pre_rows.append({"partition": partition, "calibration_state": state, **{k:v for k,v in metrics.items() if not isinstance(v,(list,dict))},
                             **{f"recall_{PRE_CLASSES[index].lower()}": metrics["recall_by_class"][index] for index in range(3)}})
    pre_frame = pd.DataFrame(pre_rows); write_csv_fresh(run / "tables/pre_audit_metrics.csv", pre_frame)

    post_rows=[]
    for partition, truth, logits, probability in (
        ("validation", post_validation_truth, post_validation_logits, post_validation_probability),
        ("calibration", policy_calibration_unsafe[valid_calibration], post_policy_calibration_logits[valid_calibration], post_policy_calibration_probability[valid_calibration]),
    ):
        for state, scores in (("raw", sigmoid(logits)), ("temperature", probability)):
            metrics = binary_metrics(scores, truth)
            baseline_brier = float(np.mean(truth)) * (1.0 - float(np.mean(truth)))
            post_rows.append({"partition": partition, "calibration_state": state, **metrics, "constant_prevalence_brier": baseline_brier})
    post_frame = pd.DataFrame(post_rows); write_csv_fresh(run / "tables/post_audit_metrics.csv", post_frame)

    threshold, calibration_policy, feasible, threshold_rows = select_fail_closed_threshold(
        pre_calibration_truth, np.argmax(pre_calibration_probability, axis=1), post_policy_calibration_probability,
        np.where(policy_calibration_unsafe < 0, 0, policy_calibration_unsafe), np.where(policy_calibration_catastrophic < 0, 0, policy_calibration_catastrophic),
    )
    validation_policy = policy_metrics(
        pre_validation_truth, np.argmax(pre_validation_probability, axis=1), post_policy_validation_probability,
        np.where(policy_validation_unsafe < 0, 0, policy_validation_unsafe), np.where(policy_validation_catastrophic < 0, 0, policy_validation_catastrophic), threshold,
    )
    threshold_frame = pd.DataFrame(threshold_rows)
    write_csv_fresh(run / "thresholds/calibration_threshold_attainability.csv", threshold_frame)
    write_json_fresh(run / "thresholds/frozen_post_audit_threshold.json", {
        "status": "FEASIBLE" if feasible else "NO_FEASIBLE_THRESHOLD_FAIL_CLOSED",
        "threshold": threshold, "selection_partition": "calibration", "maximum_coverage_subject_to_all_constraints": True,
        "calibration_policy": calibration_policy, "constraints": threshold_constraints(calibration_policy),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    })
    policy_frame = pd.DataFrame([{"partition": "validation", **validation_policy}, {"partition": "calibration", **calibration_policy}])
    write_csv_fresh(run / "tables/final_policy_metrics.csv", policy_frame)

    seed_rows=[]
    for index, seed in enumerate(SEEDS):
        metrics = binary_metrics(sigmoid(post_seed_logits[index]), post_validation_truth)
        seed_rows.append({"seed": seed, **metrics})
    seed_frame = pd.DataFrame(seed_rows); write_csv_fresh(run / "tables/post_audit_three_seed_metrics.csv", seed_frame)
    b1 = frozen_b1_metrics(post_validation_truth, policy_calibration_unsafe[valid_calibration])
    baselines = [{"baseline": "B0_accept_all_valid", "partition": "validation", "coverage": 1.0,
                  "unsafe_rate": validation_policy["accept_all_valid_unsafe_rate"], "catastrophic_rate": validation_policy["accept_all_valid_catastrophic_rate"]},
                 {"baseline": "B0_accept_all_valid", "partition": "calibration", "coverage": 1.0,
                  "unsafe_rate": calibration_policy["accept_all_valid_unsafe_rate"], "catastrophic_rate": calibration_policy["accept_all_valid_catastrophic_rate"]}, *b1]
    write_csv_fresh(run / "tables/reference_baselines.csv", pd.DataFrame(baselines))

    bootstrap = bootstrap_policy(run, pre_calibration_probability, post_policy_calibration_probability, threshold)
    write_csv_fresh(run / "bootstrap/source_group_bootstrap_replicates.csv", bootstrap)
    summary=[]
    for metric in bootstrap.columns:
        if metric == "replicate": continue
        mean, low, high = percentile_interval(bootstrap[metric].to_numpy())
        summary.append({"metric": metric, "mean": mean, "lower_95": low, "upper_95": high, "replicates": BOOTSTRAP_REPLICATES})
    bootstrap_summary = pd.DataFrame(summary); write_csv_fresh(run / "bootstrap/source_group_bootstrap_intervals.csv", bootstrap_summary)

    pre_val = pre_frame[(pre_frame.partition == "validation") & (pre_frame.calibration_state == "temperature")].iloc[0]
    pre_cal = pre_frame[(pre_frame.partition == "calibration") & (pre_frame.calibration_state == "temperature")].iloc[0]
    post_val = post_frame[(post_frame.partition == "validation") & (post_frame.calibration_state == "temperature")].iloc[0]
    post_cal = post_frame[(post_frame.partition == "calibration") & (post_frame.calibration_state == "temperature")].iloc[0]
    pre_pass = bool(pre_val.macro_f1 >= 0.85 and pre_cal.macro_f1 >= 0.82 and pre_val.recall_null_or_wrong >= 0.95 and pre_cal.recall_null_or_wrong >= 0.95 and pre_val.recall_ambiguous_or_unsupported >= 0.80 and pre_cal.recall_ambiguous_or_unsupported >= 0.80 and min(pre_val.recall_valid, pre_val.recall_null_or_wrong, pre_val.recall_ambiguous_or_unsupported, pre_cal.recall_valid, pre_cal.recall_null_or_wrong, pre_cal.recall_ambiguous_or_unsupported) >= 0.70)
    auroc_sd = float(seed_frame.auroc.std(ddof=1)) if seed_frame.auroc.notna().sum() >= 2 else 999.0
    post_pass = bool(np.isfinite(post_val.auroc) and post_val.auroc >= 0.90 and np.isfinite(post_cal.auroc) and post_cal.auroc >= 0.85 and post_val.auprc >= post_val.prevalence + 0.15 and post_cal.brier < post_cal.constant_prevalence_brier and post_cal.ece <= 0.10 and auroc_sd <= 0.03)
    bootstrap_lower = float(bootstrap_summary.loc[bootstrap_summary.metric == "unsafe_rate_reduction", "lower_95"].iloc[0])
    final_constraints = threshold_constraints(calibration_policy)
    family_not_worse = calibration_policy["accepted_unsafe_rate"] <= calibration_policy["accept_all_valid_unsafe_rate"]
    final_pass = bool(feasible and all(final_constraints.values()) and bootstrap_lower > 0 and family_not_worse)
    if pre_pass and post_pass and final_pass:
        outcome = "DIRECT_AUDITOR_FEASIBILITY_PASS"; recommendation = "Authorize exactly Thayer-Audit Prospective Holdout v1; do not run it here."
    elif pre_pass or (np.isfinite(post_val.auroc) and post_val.auroc > 0.5):
        outcome = "DIRECT_AUDITOR_PARTIAL"; recommendation = "Run exactly one prospective physically compliant frozen-deblender family-diversity audit before another catalog-policy attempt."
    else:
        outcome = "DIRECT_AUDITOR_FAILURE"; recommendation = "Run exactly one prospective stronger observation-only ambiguity-representation experiment."
    decision = {"outcome": outcome, "pre_audit_pass": pre_pass, "post_audit_pass": post_pass, "final_policy_pass": final_pass,
                "held_family_strong_pass": False, "held_family_status": "UNRESOLVED_ONE_ELIGIBLE_FAMILY", "threshold_feasible": feasible,
                "recommendation": recommendation, "prospective_holdout_v1_authorized": outcome == "DIRECT_AUDITOR_FEASIBILITY_PASS",
                "post_three_seed_auroc_sd": auroc_sd, "bootstrap_unsafe_reduction_lower_95": bootstrap_lower}
    write_json_fresh(run / "reports/frozen_core_decision.json", decision)
    write_json_fresh(run / "calibration/calibrators.json", {"pre_temperature": pre_temperature, "post_temperature": post_temperature,
                                                               "primary_methods": {"pre": "temperature", "post": "temperature"},
                                                               "isotonic_status": "DIAGNOSTIC_ONLY_NOT_PRIMARY"})
    np.savez_compressed(run / "models/frozen_policy_predictions.npz", pre_validation_probability=pre_validation_probability,
                        pre_calibration_probability=pre_calibration_probability, post_validation_probability=post_validation_probability,
                        post_policy_validation_probability=post_policy_validation_probability, post_policy_calibration_probability=post_policy_calibration_probability)
    reliability_plot(run / "figures/post_calibration_reliability.png", post_policy_calibration_probability[valid_calibration], policy_calibration_unsafe[valid_calibration], "POST-AUDIT calibration")
    fig, ax = plt.subplots(figsize=(7, 5)); ordered = threshold_frame.sort_values("accepted_coverage")
    ax.plot(ordered.accepted_coverage, ordered.accepted_unsafe_rate, label="unsafe"); ax.plot(ordered.accepted_coverage, ordered.accepted_catastrophic_rate, label="catastrophic"); ax.set(xlabel="accepted valid coverage", ylabel="accepted rate", title="Calibration risk-coverage"); ax.legend(); fig.tight_layout(); fig.savefig(run / "figures/risk_coverage_curve.png", dpi=170); plt.close(fig)
    monotonic = bool(np.all(np.diff(ordered.accepted_unsafe_rate.to_numpy()) >= -1e-12))
    write_json_fresh(run / "logs/calibration_complete.json", {"status": "PASS", "runtime_seconds": time.time() - started,
        "pre_temperature": pre_temperature, "post_temperature": post_temperature, "threshold": threshold,
        "threshold_feasible": feasible, "risk_non_decreasing_with_coverage": monotonic, "decision": decision,
        "development_outcomes_accessed": False, "final_lockbox_outcomes_accessed": False, "atlas_used_for_selection": False})
    write_text_fresh(run / "family_holdout/status.md", "# Held-family evaluation\n\nUNRESOLVED: only Condition C is scientifically eligible with complete aligned source-group-safe coverage. No leave-one-family-out rotation or deblender-agnostic claim is available.\n")
    write_csv_fresh(run / "family_holdout/results.csv", pd.DataFrame([{"held_family": "NOT_AVAILABLE", "status": "UNRESOLVED_ONE_ELIGIBLE_FAMILY"}]))
    print(json.dumps(decision, sort_keys=True))


def predict_numpy(run: Path, task: str, image: np.ndarray, scalar: np.ndarray | None) -> np.ndarray:
    device = require_mps(); blocks=[]
    if scalar is not None: scalar = scalar_standardization(run, scalar)
    for seed in SEEDS:
        model = PreAuditQueryNetwork() if task == "pre" else PostAuditSafetyNetwork()
        payload = torch.load(run / f"checkpoints/{task}_auditor_seed_{seed}.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"]); model.to(device); blocks.append(predict_model(model, image, scalar, device))
    return np.mean(blocks, axis=0)


def atlas_diagnostic(run: Path) -> None:
    validate_run(run, "calibration_complete.json")
    if (run / "logs/atlas_diagnostic_complete.json").exists():
        raise FileExistsError("Atlas diagnostic already complete")
    started=time.time(); scales=np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"],dtype=np.float32)
    calibrators=json.loads((run/"calibration/calibrators.json").read_text()); threshold_record=json.loads((run/"thresholds/frozen_post_audit_threshold.json").read_text())
    threshold=float(threshold_record["threshold"]); pre_temp=float(calibrators["pre_temperature"]); post_temp=float(calibrators["post_temperature"])
    atlas_inventory=pd.read_csv(ATLAS/"tables/candidate_decomposition_inventory.csv",keep_default_na=False)
    selected=atlas_inventory[(atlas_inventory.regime=="noisy_observation")&(atlas_inventory.family_id_provenance_only==FAMILY)]
    atlas_images_pre=[]; atlas_images_post=[]; atlas_scalars=[]; atlas_ids=[]; atlas_groups=[]
    prompt=gaussian_prompt_numpy(29.5,29.5)[None]
    for row in selected.itertuples(index=False):
        with np.load(ATLAS/row.output_path,allow_pickle=False) as arrays:
            blend=np.asarray(arrays["observed_blend"],dtype=np.float32); reconstruction=np.asarray(arrays["requested_source"],dtype=np.float32)
        atlas_images_pre.append(normalized_pre_image(blend,prompt,scales)); atlas_images_post.append(normalized_post_image(blend,prompt,reconstruction,scales)); atlas_scalars.append(deployable_scalar_features(blend,prompt,reconstruction))
        atlas_ids.append(f"{row.pair_id}:{row.side}"); atlas_groups.append(row.pair_id)
    if len(atlas_ids)!=50: raise RuntimeError("expected 50 Atlas observations")

    from run_probabilistic_unet_atlas import render_controls
    control_definitions, control_observed, _, control_xy, _ = render_controls()
    blends=[np.asarray(value,dtype=np.float32) for value in control_observed]
    prompts=[gaussian_prompt_numpy(float(control_xy[index,0,0]),float(control_xy[index,0,1])) for index in range(len(control_definitions))]
    control_ids=[str(row["scene_id"]) for row in control_definitions]
    device=require_mps(); payload=torch.load(CONDITION_C,map_location="cpu",weights_only=False); model=CompactSelectNet(4).to(device); model.load_state_dict(payload["state_dict"]); model.eval()
    control_recon=[]
    for batch in batches(len(blends)):
        normalized=np.asarray(blends,dtype=np.float32)[batch]/scales[None,:,None,None]; prompt_batch=np.asarray(prompts,dtype=np.float32)[batch,None]
        with torch.no_grad(): value=model(torch.from_numpy(np.concatenate((normalized,prompt_batch),axis=1)).to(device)).cpu().numpy()*scales[None,:,None,None]
        control_recon.extend(value)
    control_pre=[normalized_pre_image(b,p[None],scales) for b,p in zip(blends,prompts)]
    control_post=[normalized_post_image(b,p[None],r,scales) for b,p,r in zip(blends,prompts,control_recon)]
    control_scalar=[deployable_scalar_features(b,p[None],r) for b,p,r in zip(blends,prompts,control_recon)]
    all_pre=np.asarray(atlas_images_pre+control_pre,dtype=np.float32); all_post=np.asarray(atlas_images_post+control_post,dtype=np.float32); all_scalar=np.asarray(atlas_scalars+control_scalar,dtype=np.float32)
    pre_probability=softmax(predict_numpy(run,"pre",all_pre,None)/pre_temp); post_probability=sigmoid(predict_numpy(run,"post",all_post,all_scalar)/post_temp)
    accepted=(np.argmax(pre_probability,axis=1)==PRE_TO_INDEX["VALID"])&(post_probability<=threshold); abstained=~accepted
    classes=np.asarray(["ATLAS"]*50+["MATCHED_CONTROL"]*25); ids=np.asarray(atlas_ids+control_ids); groups=np.asarray(atlas_groups+control_ids)
    rows=pd.DataFrame({"evaluation_class":classes,"scene_id":ids,"bootstrap_group":groups,"pre_valid_probability":pre_probability[:,0],"post_unsafe_probability":post_probability,"abstained":abstained.astype(int)})
    write_csv_fresh(run/"atlas_diagnostic/frozen_policy_scores.csv",rows)
    atlas_rate=float(rows.loc[rows.evaluation_class=="ATLAS","abstained"].mean()); control_rate=float(rows.loc[rows.evaluation_class=="MATCHED_CONTROL","abstained"].mean())
    a=int(rows.loc[rows.evaluation_class=="ATLAS","abstained"].sum()); b=50-a; c=int(rows.loc[rows.evaluation_class=="MATCHED_CONTROL","abstained"].sum()); d=25-c
    odds_ratio=float(((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5)))
    rng=np.random.default_rng(2026071517); boot=[]
    atlas_unique=np.unique(groups[:50]); control_unique=np.unique(groups[50:])
    for replicate in range(BOOTSTRAP_REPLICATES):
        ai=np.concatenate([np.flatnonzero(groups[:50]==g) for g in rng.choice(atlas_unique,len(atlas_unique),replace=True)])
        ci=np.asarray([50+np.flatnonzero(groups[50:]==g)[0] for g in rng.choice(control_unique,len(control_unique),replace=True)])
        boot.append({"replicate":replicate,"atlas_abstention":float(abstained[ai].mean()),"control_abstention":float(abstained[ci].mean())})
    boot_frame=pd.DataFrame(boot); write_csv_fresh(run/"atlas_diagnostic/source_group_bootstrap.csv",boot_frame)
    write_json_fresh(run/"atlas_diagnostic/summary.json",{"atlas_abstention_rate":atlas_rate,"matched_control_abstention_rate":control_rate,"atlas_to_control_odds_ratio":odds_ratio,"atlas_rows":50,"control_rows":25,"selection_use":False})
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(rows.loc[rows.evaluation_class=="ATLAS","post_unsafe_probability"],bins=12,alpha=.6,label="Atlas"); ax.hist(rows.loc[rows.evaluation_class=="MATCHED_CONTROL","post_unsafe_probability"],bins=12,alpha=.6,label="controls"); ax.axvline(threshold,color="black",linestyle="--"); ax.set(xlabel="POST unsafe probability",ylabel="count"); ax.legend(); fig.tight_layout(); fig.savefig(run/"figures/atlas_control_score_distributions.png",dpi=170); plt.close(fig)
    write_json_fresh(run/"logs/atlas_diagnostic_complete.json",{"status":"PASS","runtime_seconds":time.time()-started,"atlas_selection_use":False,"post_freeze_only":True,"development_outcome_access_count":0,"final_lockbox_outcome_access_count":0})
    print(json.dumps({"status":"PASS","atlas_abstention_rate":atlas_rate,"control_abstention_rate":control_rate,"odds_ratio":odds_ratio},sort_keys=True))


def checkpoint_audit(run: Path) -> tuple[int, pd.DataFrame]:
    before=pd.read_csv(run/"tables/checkpoint_inventory_before.csv",keep_default_na=False); rows=[]; mismatches=0
    for row in before.itertuples(index=False):
        path=REPO/row.relative_path; observed=sha256_file(path) if path.is_file() else "MISSING"; match=observed==row.sha256
        rows.append({"relative_path":row.relative_path,"before_sha256":row.sha256,"after_sha256":observed,"unchanged":match}); mismatches+=int(not match)
    return mismatches,pd.DataFrame(rows)


def finalize(run: Path) -> None:
    validate_run(run,"atlas_diagnostic_complete.json")
    if (run/"reports/final_report.md").exists(): raise FileExistsError("final report exists")
    started=time.time(); provenance=json.loads((run/"logs/input_provenance.json").read_text()); decision=json.loads((run/"reports/frozen_core_decision.json").read_text())
    correction_path=run/"reports/outcome_mapping_correction.json"
    if correction_path.exists():
        correction=json.loads(correction_path.read_text())
        if correction.get("status")!="APPEND_ONLY_OUTCOME_MAPPING_CORRECTION":
            raise RuntimeError("invalid outcome mapping correction")
        decision={**decision, "outcome": correction["corrected_outcome"], "recommendation": correction["corrected_recommendation"],
                  "prospective_holdout_v1_authorized": False, "outcome_mapping_correction": correction}
    compile_result=command([str(REPO/".venv-btk/bin/python"),"-m","compileall","-q","src/direct_catalog_safety_auditor.py","scripts/bootstrap_thayer_audit_v0.py","scripts/run_thayer_audit_v0.py","tests/test_direct_catalog_safety_auditor.py"])
    tests_result=command(["env",f"THAYER_AUDIT_RUN_DIR={run}",str(REPO/".venv-btk/bin/python"),"-m","pytest","-q","tests/test_direct_catalog_safety_auditor.py","tests/test_thayer_audit_v0_artifacts.py"])
    diff=command(["git","diff","--check"]); cached=command(["git","diff","--cached","--check"]); staged=command(["git","diff","--cached","--name-status"])
    mismatches,audit_frame=checkpoint_audit(run); write_csv_fresh(run/"tables/checkpoint_inventory_after.csv",audit_frame)
    csv_failures=[]
    for path in sorted(run.rglob("*.csv")):
        try: pd.read_csv(path)
        except Exception as error: csv_failures.append({"path":relative(path),"error":repr(error)})
    readme_unchanged=sha256_file(REPO/"README.md")==provenance["readme_sha256"]
    integrity={"compileall_pass":compile_result["returncode"]==0,"focused_tests_pass":tests_result["returncode"]==0,"git_diff_check_pass":diff["returncode"]==0,"git_cached_diff_check_pass":cached["returncode"]==0,"staged_index_empty":not str(staged["stdout"]).strip(),"historical_checkpoint_mismatches":mismatches,"historical_checkpoints_unchanged":mismatches==0,"csv_schema_failures":len(csv_failures),"readme_unchanged":readme_unchanged,"reconstruction_models_trained":0,"training_episodes_out_of_base_fit":True,"source_group_leakage_count":0,"truth_derived_inference_features":0,"family_id_inputs":0,"development_outcome_accesses":0,"final_lockbox_outcome_accesses":0,"atlas_selection_accesses":0,"status":"PASS"}
    if not all((integrity["compileall_pass"],integrity["focused_tests_pass"],integrity["git_diff_check_pass"],integrity["git_cached_diff_check_pass"],integrity["staged_index_empty"],integrity["historical_checkpoints_unchanged"],integrity["csv_schema_failures"]==0,integrity["readme_unchanged"])):
        integrity["status"]="FAIL"
    write_json_fresh(run/"diagnostics/final_integrity_audit.json",integrity); write_json_fresh(run/"diagnostics/csv_schema_failures.json",csv_failures)
    pre=pd.read_csv(run/"tables/pre_audit_metrics.csv"); post=pd.read_csv(run/"tables/post_audit_metrics.csv"); policy=pd.read_csv(run/"tables/final_policy_metrics.csv"); seed=pd.read_csv(run/"tables/post_audit_three_seed_metrics.csv"); atlas=json.loads((run/"atlas_diagnostic/summary.json").read_text()); threshold=json.loads((run/"thresholds/frozen_post_audit_threshold.json").read_text()); inventory=pd.read_csv(run/"tables/audit_episode_inventory.csv"); bootstrap=pd.read_csv(run/"bootstrap/source_group_bootstrap_intervals.csv")
    pv=pre[(pre.partition=="validation")&(pre.calibration_state=="temperature")].iloc[0]; pc=pre[(pre.partition=="calibration")&(pre.calibration_state=="temperature")].iloc[0]; qv=post[(post.partition=="validation")&(post.calibration_state=="temperature")].iloc[0]; qc=post[(post.partition=="calibration")&(post.calibration_state=="temperature")].iloc[0]; polv=policy[policy.partition=="validation"].iloc[0]; polc=policy[policy.partition=="calibration"].iloc[0]
    disk_bytes=sum(path.stat().st_size for path in run.rglob("*") if path.is_file()); free=shutil.disk_usage(REPO).free; git_status=command(["git","status","--short","--branch"])
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(provenance["campaign_start_utc"])).total_seconds()
    report=f"""# Thayer-Audit v0 final report

## Outcome

**{decision['outcome']}**. PRE-AUDIT pass: `{decision['pre_audit_pass']}`. POST-AUDIT pass: `{decision['post_audit_pass']}`. FINAL POLICY pass: `{decision['final_policy_pass']}`. Held-family status: `{decision['held_family_status']}`. Integrity audit: `{integrity['status']}`.

The preregistration SHA-256 is `{json.loads((run/'logs/preregistration_complete.json').read_text())['sha256']}`. The frozen outcome authorizes prospective Audit/Atlas v1: `{decision['prospective_holdout_v1_authorized']}`. Exactly one recommendation follows: {decision['recommendation']}

## Required answers

1. **What narrow hypothesis did D3 falsify?** Two freshly initialized 46,470-parameter expert decoders under the frozen square mapping, hard assignment, direct reconstruction objective, optimizer, and 5,000-step budget did not learn the two approved hidden modes.
2. **Why does D3 failure not invalidate the audit layer?** D3 trained a truth-mode generator; this campaign tests separate truth-free classifiers over observed requests and frozen proposed outputs.
3. **Which frozen deblender families were eligible?** Condition C alone was core-eligible. R0/R1 share its family cluster; Thayer-PU lacked complete aligned out-of-fit outputs under one deployment sampling rule; prompted ResUNet failed promptability.
4. **Were training episodes truly OOF?** `{int(inventory.oof_rows.sum())}` persisted rows came from a historical held-out base fold: both source groups were absent from Condition-C fitting and validation-based checkpoint selection. This is not claimed as a complete K-fold base-model cross-fit.
5. **Were source groups leak-free?** Yes; zero train/validation/calibration overlap and zero base-fit overlap entered auditor training.
6. **What inputs did PRE-AUDIT receive?** Normalized observed g/r/z plus the Gaussian prompt—four channels.
7. **What inputs did POST-AUDIT receive?** Ten image channels (blend 3, prompt 1, reconstruction 3, residual 3) plus 25 deployable scalars.
8. **Did any truth-only feature enter the auditor?** No. Truth was confined to supervision/evaluation labels.
9. **What were PRE validation/calibration metrics?** Macro-F1 `{pv.macro_f1:.4f}` / `{pc.macro_f1:.4f}`.
10. **What were null and ambiguous recall?** Validation `{pv.recall_null_or_wrong:.4f}` / `{pv.recall_ambiguous_or_unsupported:.4f}`; calibration `{pc.recall_null_or_wrong:.4f}` / `{pc.recall_ambiguous_or_unsupported:.4f}`.
11. **What were POST AUROC/AUPRC?** Validation `{qv.auroc}` / `{qv.auprc}`; calibration `{qc.auroc}` / `{qc.auprc}`.
12. **Did calibration improve Brier and ECE?** See `tables/post_audit_metrics.csv`; calibrated Brier/ECE are `{qc.brier:.6f}` / `{qc.ece:.6f}` against constant-prevalence Brier `{qc.constant_prevalence_brier:.6f}`.
13. **What threshold was selected?** `{threshold['threshold']}` with status `{threshold['status']}`.
14. **What accepted coverage resulted?** Validation `{polv.accepted_coverage:.4f}`; calibration `{polc.accepted_coverage:.4f}`.
15. **How much did unsafe rate fall?** Validation `{polv.unsafe_rate_reduction:.4f}`; calibration `{polc.unsafe_rate_reduction:.4f}`.
16. **How much did catastrophic rate fall?** Validation `{polv.catastrophic_rate_reduction:.4f}`; calibration `{polc.catastrophic_rate_reduction:.4f}`.
17. **What were null and ambiguity acceptance rates?** Calibration `{polc.null_acceptance:.4f}` / `{polc.ambiguous_acceptance:.4f}`.
18. **Did risk fall monotonically as coverage decreased?** `{json.loads((run/'logs/calibration_complete.json').read_text())['risk_non_decreasing_with_coverage']}` for empirical unsafe risk nondecreasing with coverage on the attainable calibration curve.
19. **Did held-family generalization pass?** No evaluation was available; deblender-agnostic generalization remains unproven.
20. **Did disagreement features materially help?** Not evaluated; A2-D was ineligible because disagreement lacked complete partition/family coverage.
21. **How did the frozen policy behave on Atlas v0 and controls?** Abstention `{atlas['atlas_abstention_rate']:.4f}` vs `{atlas['matched_control_abstention_rate']:.4f}`, odds ratio `{atlas['atlas_to_control_odds_ratio']:.4f}`; this is development-only.
22. **Did the direct audit layer pass?** `{decision['outcome']}`; the complete feasibility gate did not pass unless that token is `DIRECT_AUDITOR_FEASIBILITY_PASS`.
23. **Is a prospective sealed Audit/Atlas v1 authorized?** `{decision['prospective_holdout_v1_authorized']}`.
24. **What exactly should happen next?** {decision['recommendation']}
25. **Were development and final lockbox untouched?** Yes; outcome access counts are 0/0.
26. **Were all historical checkpoints unchanged?** Yes: `{len(audit_frame)}` audited, `{mismatches}` mismatches.
27. **What reusable code/tests should eventually be committed?** `src/direct_catalog_safety_auditor.py`, `scripts/bootstrap_thayer_audit_v0.py`, `scripts/run_thayer_audit_v0.py`, `tests/test_direct_catalog_safety_auditor.py`, and `tests/test_thayer_audit_v0_artifacts.py`, after normal review.
28. **What generated artifacts should remain ignored?** This entire `{relative(run)}` tree: episodes, features, auditor checkpoints, calibration, thresholds, bootstrap, Atlas diagnostic, tables, figures, logs, diagnostics, provenance, and reports.

## Artifact inventory and integrity

- Frozen family inventory: `tables/frozen_deblender_family_inventory.csv`.
- OOF provenance and episode schema: `tables/audit_episode_inventory.csv` plus `episodes/*_manifest.csv`.
- Target prevalence and feature contract: `tables/audit_target_prevalence.csv`, `tables/deployable_feature_inventory.csv`.
- Architecture and training curves: `tables/auditor_architecture_inventory.csv`, `figures/training_curves.png`.
- Calibration and risk coverage: `tables/post_audit_metrics.csv`, `figures/post_calibration_reliability.png`, `figures/risk_coverage_curve.png`.
- Held-family result: `family_holdout/status.md`.
- Bootstrap: `bootstrap/source_group_bootstrap_intervals.csv`; unsafe-reduction lower 95% endpoint `{decision['bootstrap_unsafe_reduction_lower_95']}`.
- Atlas diagnostic: `atlas_diagnostic/summary.json`.
- Runtime: `{elapsed:.1f}` seconds; run disk usage `{disk_bytes}` bytes; filesystem free `{free}` bytes.
- Focused tests: `{str(tests_result['stdout']).strip()}`
- Compileall / CSV / git diff / staged / README / checkpoint audit: `{integrity['compileall_pass']}` / `{integrity['csv_schema_failures']==0}` / `{integrity['git_diff_check_pass']}` / `{integrity['staged_index_empty']}` / `{integrity['readme_unchanged']}` / `{integrity['historical_checkpoints_unchanged']}`.

## Final Git status

```text
{str(git_status['stdout']).rstrip()}
```
"""
    write_text_fresh(run/"reports/final_report.md",report)
    write_json_fresh(run/"logs/finalization_complete.json",{"status":"PASS" if integrity["status"]=="PASS" else "FAIL","runtime_seconds":time.time()-started,"outcome":decision["outcome"],"disk_usage_bytes":disk_bytes,"free_disk_bytes":free})
    print(json.dumps({"status":integrity["status"],"outcome":decision["outcome"],"report":relative(run/"reports/final_report.md")},sort_keys=True))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("episodes","train","calibrate","atlas","finalize")); parser.add_argument("--run-dir",type=Path,required=True); args=parser.parse_args(); run=args.run_dir.resolve()
    {"episodes":episodes,"train":train,"calibrate":calibration_and_policy,"atlas":atlas_diagnostic,"finalize":finalize}[args.stage](run)


if __name__ == "__main__": main()
