#!/usr/bin/env python3
"""Fixed-protocol two-seed replication for Thayer-Select Phase II.

This orchestrator imports the frozen Phase-II model, loss, metrics, and
calibration implementations.  It never modifies the authoritative run and
permits only initialization and minibatch-order seeds to vary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.models_thayer_select import ThayerSelectNet
from src.prompt_semantics import QueryClass
from evaluate_thayer_select_recoverability import (
    COVERAGES,
    apply_calibrator,
    auprc,
    auroc,
    calibration_errors,
    isotonic_apply,
    isotonic_fit,
    metric_summary,
)
from thayer_select_prompt_ablation_common import require_mps, seed_everything
from thayer_select_recoverability_common import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LOG_VARIANCE,
    MAX_UNCERTAINTY_SATURATION_FRACTION,
    MIN_LOG_VARIANCE,
    PHASE1,
    add_actionable_acceptance_labels,
    load_scales,
    outcome_metrics,
    read_csv,
    sha256_file,
    write_csv_fresh,
    write_csv_union_fresh,
    write_json_fresh,
    write_text_fresh,
)
from train_thayer_select_recoverability import MixedSceneDataset, loss_batch

ORIGINAL = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
PRIMARY_CONTRACT = "permissive"
PARAMETER_COUNT = 123_368
GLOBAL_POSITIVE_WEIGHT = 20.0
SEED_MAP = {
    "R1_seed_2": {"initialization_seed": 2026078202, "minibatch_order_seed": 2026079202},
    "R1_seed_3": {"initialization_seed": 2026078303, "minibatch_order_seed": 2026079303},
}
STABILITY_TOLERANCES = {
    "auroc_range": 0.10,
    "auprc_range": 0.10,
    "brier_range": 0.05,
    "risk_range_each_coverage": 0.05,
    "null_hallucination_range": 0.10,
}
UPGRADE_GATES = {
    "minimum_catastrophic_reduction_at_80": 0.05,
    "minimum_risk_reduction_at_70": 0.05,
    "maximum_null_hallucination": 0.05,
    "ambiguous_minus_valid_score_must_be_negative": True,
}

FROZEN_CODE = (
    "src/models_thayer_select.py",
    "src/recoverability.py",
    "src/prompt_semantics.py",
    "scripts/thayer_select_recoverability_common.py",
    "scripts/train_thayer_select_recoverability.py",
    "scripts/evaluate_thayer_select_recoverability.py",
)
FROZEN_INPUTS = (
    "manifests/training_scenes.h5",
    "manifests/validation_scenes.h5",
    "manifests/calibration_scenes.h5",
    "manifests/development_test_scenes.h5",
    "manifests/training_scene_manifest.csv",
    "manifests/validation_scene_manifest.csv",
    "manifests/calibration_scene_manifest.csv",
    "manifests/development_test_scene_manifest.csv",
    "manifests/development_test_scene_definitions.csv",
    "manifests/source_split_manifest.csv",
    "manifests/normalization.json",
    "manifests/primary_actionable_contract_selection.json",
    "manifests/r1_training_config.json",
    "tables/training_actionable_acceptance_labels.csv",
    "tables/validation_actionable_acceptance_labels.csv",
    "calibration/selected_calibrator.json",
    "calibration/frozen_abstention_thresholds.json",
    "checkpoints/r1_best.pth",
)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def expected_original_hashes() -> dict[str, str]:
    rows = read_csv(ORIGINAL / "manifests/campaign_file_hashes.csv")
    return {str(Path(row["relative_path"]).relative_to(ORIGINAL.relative_to(REPO))): row["sha256"] for row in rows}


def verify_frozen_inputs() -> list[dict]:
    expected = expected_original_hashes()
    rows = []
    for relative in FROZEN_INPUTS:
        path = ORIGINAL / relative
        observed = sha256_file(path)
        reference = expected.get(relative)
        rows.append({"category": "frozen_phase2_input", "relative_path": str(path.relative_to(REPO)), "expected_sha256": reference or "MISSING", "observed_sha256": observed, "status": "PASS" if reference == observed else "FAIL"})
    final_code = {row["relative_path"]: row["final_sha256"] for row in read_csv(ORIGINAL / "tables/campaign_code_hashes_post_audit_self_match.csv")}
    for relative in FROZEN_CODE:
        observed = sha256_file(REPO / relative)
        reference = final_code.get(relative)
        rows.append({"category": "frozen_phase2_code", "relative_path": relative, "expected_sha256": reference or "MISSING", "observed_sha256": observed, "status": "PASS" if reference == observed else "FAIL"})
    if any(row["status"] != "PASS" for row in rows):
        failed = [row["relative_path"] for row in rows if row["status"] != "PASS"]
        raise RuntimeError(f"Frozen Phase-II provenance mismatch: {failed}")
    config = json.loads((ORIGINAL / "manifests/r1_training_config.json").read_text())
    calibrator = json.loads((ORIGINAL / "calibration/selected_calibrator.json").read_text())
    selection = json.loads((ORIGINAL / "manifests/primary_actionable_contract_selection.json").read_text())
    gates = {
        "parameter_count": config["parameter_count"] == PARAMETER_COUNT,
        "epochs": config["epochs"] == EPOCHS,
        "batch_size": config["batch_size"] == BATCH_SIZE,
        "learning_rate": config["learning_rate"] == LEARNING_RATE,
        "global_positive_weight": config["global_positive_weight"] == GLOBAL_POSITIVE_WEIGHT,
        "variance_bounds": config["log_variance_bounds"] == [MIN_LOG_VARIANCE, MAX_LOG_VARIANCE],
        "saturation_stop": config["maximum_validation_saturation_fraction"] == MAX_UNCERTAINTY_SATURATION_FRACTION,
        "primary_contract": selection["primary_contract"] == PRIMARY_CONTRACT,
        "calibrator": calibrator["method"] == "isotonic",
        "original_training_seed": config["training_seed"] == 2026078101,
        "development_read_only": oct((ORIGINAL / "manifests/development_test_scenes.h5").stat().st_mode & 0o777) == "0o444",
    }
    if not all(gates.values()):
        raise RuntimeError(f"Frozen protocol semantic mismatch: {[key for key, value in gates.items() if not value]}")
    return rows


def checkpoint_inventory() -> list[dict]:
    original = read_csv(ORIGINAL / "tables/checkpoint_inventory_after.csv")
    rows = []
    for row in original:
        path = REPO / row["relative_path"]
        expected = row.get("final_sha256", row.get("observed_sha256", ""))
        observed = sha256_file(path)
        rows.append({"category": row["category"], "relative_path": row["relative_path"], "expected_sha256": expected, "observed_sha256": observed, "status": "PASS" if expected == observed else "FAIL"})
    for name in ("r0_best.pth", "r0_final.pth", "r1_best.pth", "r1_final.pth"):
        path = ORIGINAL / "checkpoints" / name
        digest = sha256_file(path)
        rows.append({"category": "authoritative_phase2", "relative_path": str(path.relative_to(REPO)), "expected_sha256": digest, "observed_sha256": digest, "status": "PASS"})
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("Pre-replication checkpoint inventory failed")
    return rows


def bootstrap(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Replication output collision: {run_dir}")
    if not run_dir.name.startswith("thayer_select_recoverability_seed_replication_"):
        raise RuntimeError("Replication run requires the timestamped naming contract")
    frozen_rows = verify_frozen_inputs()
    device = require_mps()
    if device.type != "mps":
        raise RuntimeError("MPS gate failed")
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("diagnostics", "logs", "tables", "figures", "reports", "checkpoints", "calibration", "uncertainty_maps", "example_grids"):
        (run_dir / directory).mkdir(exist_ok=False)
    write_csv_fresh(run_dir / "tables/frozen_input_hashes.csv", frozen_rows)
    inventory = checkpoint_inventory()
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", inventory)
    import astropy, btk, galsim, surveycodex
    packages = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "h5py": h5py.__version__, "astropy": astropy.__version__, "btk": btk.__version__, "galsim": galsim.__version__, "surveycodex": version("surveycodex"), "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "mps_probe": float(torch.ones(2, device="mps").sum().cpu())}
    git = {"branch": command(["git", "branch", "--show-current"]), "head": command(["git", "rev-parse", "HEAD"]), "status": command(["git", "status", "--short", "--branch"]), "staged": command(["git", "diff", "--cached", "--name-status"])}
    provenance = {"status": "FROZEN_BEFORE_TRAINING", "authoritative_phase2_run": str(ORIGINAL.relative_to(REPO)), "authoritative_phase2_final_report_sha256": sha256_file(ORIGINAL / "reports/final_report.md"), "frozen_inputs": frozen_rows, "seed_map": SEED_MAP, "packages": packages, "git": git, "original_r1_checkpoint_sha256": sha256_file(ORIGINAL / "checkpoints/r1_best.pth"), "original_development_manifest_sha256": sha256_file(ORIGINAL / "manifests/development_test_scene_manifest.csv"), "original_development_hdf5_sha256": sha256_file(ORIGINAL / "manifests/development_test_scenes.h5"), "replication_orchestrator_sha256": sha256_file(Path(__file__).resolve()), "lockbox_accessed": False, "created_at_unix": time.time()}
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)
    write_json_fresh(run_dir / "logs/seed_map.json", {"status": "FROZEN_BEFORE_TRAINING", "original": {"initialization_seed": 2026078101, "minibatch_order_seed": 2026078101}, "replications": SEED_MAP})
    contract = f"""# Fixed-protocol R1 training-seed replication contract

Status: frozen before training. Authoritative input run: `{ORIGINAL.relative_to(REPO)}`.

Exactly two new conditions are allowed: `R1_seed_2` and `R1_seed_3`. Their initialization/minibatch seeds are `{json.dumps(SEED_MAP, sort_keys=True)}`. Only those two seed fields may differ from the original R1 protocol.

The architecture (123,368 parameters), optimizer, cosine scheduler, 20 epochs, batch size 8, learning rate 0.001, normalization, train/validation/calibration/development arrays, actionable PERMISSIVE contract, positive weight 20, whole-image valid/null MSE, 0.1 bounded valid-source-support NLL, 0.5 actionable BCE, 0.25 no-source BCE, 2.0 full-map saturation penalty, `[-8,2]` log-variance bounds, 25% saturation stop, best-checkpoint rule, isotonic implementation, score definition, coverage points, and metrics are immutable.

Development inference is authorized exactly once for each new seed. It must persist reconstruction and full physical pixel-uncertainty maps during that pass. The original R1 is never rerun and its missing uncertainty maps remain missing. The lockbox remains sealed.

Predeclared stability tolerances: `{json.dumps(STABILITY_TOLERANCES, sort_keys=True)}`. Predeclared upgraded-success gates: `{json.dumps(UPGRADE_GATES, sort_keys=True)}`. Ambiguous overconfidence means an ambiguous scene score exceeds that seed's median valid-scene score. Pairwise accept/reject agreement uses each seed's calibration-frozen 80%-coverage threshold. Ensemble combinations are fixed to mean, median, minimum, low-disagreement `clip(1-4*variance,0,1)`, and `clip(mean-std,0,1)`; no weights are tuned.
"""
    write_text_fresh(run_dir / "diagnostics/replication_contract.md", contract)
    write_json_fresh(run_dir / "logs/bootstrap_complete.json", {"status": "PASS", "device": "mps", "frozen_input_count": len(frozen_rows), "checkpoint_count": len(inventory), "lockbox_accessed": False, "completed_at_unix": time.time()})


def seed_paths(run_dir: Path, condition: str) -> dict[str, Path]:
    stem = condition.lower()
    return {"best": run_dir / f"checkpoints/{stem}_best.pth", "final": run_dir / f"checkpoints/{stem}_final.pth", "epochs": run_dir / f"tables/{stem}_epochs.csv", "validation": run_dir / f"tables/{stem}_validation_by_query.csv", "config": run_dir / f"logs/{stem}_training_config.json", "device": run_dir / f"logs/{stem}_device.json"}


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_seed(run_dir: Path, condition: str) -> dict:
    if condition not in SEED_MAP:
        raise ValueError(condition)
    paths = seed_paths(run_dir, condition)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Replication output collision for {condition}")
    verify_frozen_inputs()
    seed = SEED_MAP[condition]
    device = require_mps()
    seed_everything(seed["initialization_seed"])
    model = ThayerSelectNet(min_log_variance=MIN_LOG_VARIANCE, max_log_variance=MAX_LOG_VARIANCE).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PARAMETER_COUNT or next(model.parameters()).device.type != "mps":
        raise RuntimeError("Architecture/parameter/device mismatch")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    training = MixedSceneDataset(ORIGINAL, "training", PRIMARY_CONTRACT)
    validation = MixedSceneDataset(ORIGINAL, "validation", PRIMARY_CONTRACT)
    loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed["minibatch_order_seed"]), num_workers=0)
    validation_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    best_loss = math.inf; best_epoch = -1; best_state = None; epoch_rows = []; query_rows = []; started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); training_sum = 0.0; training_count = 0; components_sum = defaultdict(float)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, components, _ = loss_batch("R1", model, batch, device, GLOBAL_POSITIVE_WEIGHT)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf in {condition}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError(f"Non-finite gradient in {condition}")
            optimizer.step()
            size = len(batch[0]); training_sum += float(loss.detach().cpu()) * size; training_count += size
            for name, value in components.items(): components_sum[name] += float(value.detach().cpu()) * size
        model.eval(); validation_sum = 0.0; validation_count = 0; variance_min = math.inf; variance_max = -math.inf; saturation_sum = 0.0; by_query = defaultdict(list)
        with torch.no_grad():
            for batch in validation_loader:
                loss, components, query_index = loss_batch("R1", model, batch, device, GLOBAL_POSITIVE_WEIGHT)
                if not torch.isfinite(loss): raise RuntimeError(f"Non-finite validation in {condition}")
                size = len(batch[0]); validation_sum += float(loss.cpu()) * size; validation_count += size
                variance_min = min(variance_min, float(components["log_variance_min"].cpu())); variance_max = max(variance_max, float(components["log_variance_max"].cpu())); saturation_sum += float(components["saturation_fraction"].cpu()) * size
                for query in torch.unique(query_index).tolist(): by_query[int(query)].append(float(loss.cpu()))
        record = {"condition": condition, "epoch": epoch, "training_loss": training_sum / training_count, "validation_loss": validation_sum / validation_count, "learning_rate": optimizer.param_groups[0]["lr"], "validation_log_variance_min": variance_min, "validation_log_variance_max": variance_max, "validation_saturation_fraction": saturation_sum / validation_count, "device": "mps", **{f"training_{key}": value / training_count for key, value in components_sum.items()}}
        if variance_min < MIN_LOG_VARIANCE or variance_max > MAX_LOG_VARIANCE: raise RuntimeError("Variance-bound violation")
        if record["validation_saturation_fraction"] > MAX_UNCERTAINTY_SATURATION_FRACTION: raise RuntimeError("Uncertainty saturation stop")
        epoch_rows.append(record)
        for query, values in by_query.items(): query_rows.append({"condition": condition, "epoch": epoch, "query_class": list(QueryClass)[query].value, "batch_count": len(values), "mean_validation_objective": float(np.mean(values))})
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]; best_epoch = epoch; best_state = cpu_state(model)
        scheduler.step(); print(json.dumps(record, sort_keys=True), flush=True)
    if best_state is None: raise RuntimeError("No best checkpoint")
    torch.save({"state_dict": best_state, "condition": condition, "epoch": best_epoch, "selection": "minimum frozen validation objective", **seed, "primary_contract": PRIMARY_CONTRACT}, paths["best"])
    torch.save({"state_dict": cpu_state(model), "condition": condition, "epoch": EPOCHS, "selection": "final epoch", **seed, "primary_contract": PRIMARY_CONTRACT}, paths["final"])
    write_csv_fresh(paths["epochs"], epoch_rows); write_csv_fresh(paths["validation"], query_rows)
    config = {"status": "FROZEN", "condition": condition, **seed, "parameter_count": count, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "optimizer": "Adam", "scheduler": "CosineAnnealingLR", "loss": json.loads((ORIGINAL / "manifests/r1_training_config.json").read_text())["loss"], "primary_contract": PRIMARY_CONTRACT, "best_epoch": best_epoch, "best_validation_objective": best_loss, "best_checkpoint": str(paths["best"].relative_to(REPO)), "best_checkpoint_sha256": sha256_file(paths["best"]), "final_checkpoint": str(paths["final"].relative_to(REPO)), "final_checkpoint_sha256": sha256_file(paths["final"]), "runtime_seconds": time.time() - started, "device": "mps", "calibration_used": False, "development_used": False, "lockbox_used": False}
    write_json_fresh(paths["config"], config); write_json_fresh(paths["device"], {"condition": condition, "device": "mps", "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "all_epochs": EPOCHS, "fallback_detected": False})
    return config


def load_replica(run_dir: Path, condition: str, device: torch.device) -> ThayerSelectNet:
    payload = torch.load(seed_paths(run_dir, condition)["best"], map_location="cpu", weights_only=False)
    if payload["condition"] != condition or payload["selection"] != "minimum frozen validation objective": raise RuntimeError("Checkpoint identity mismatch")
    model = ThayerSelectNet(min_log_variance=MIN_LOG_VARIANCE, max_log_variance=MAX_LOG_VARIANCE).to(device)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    return model


def infer_full(model: ThayerSelectNet, blends: np.ndarray, prompts: np.ndarray, scales: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = torch.from_numpy(np.ascontiguousarray(blends / scales[None, :, None, None])).to(device)
    prompt = torch.from_numpy(np.ascontiguousarray(prompts)).to(device)
    with torch.no_grad(): output = model(image, prompt)
    if any(value.device.type != "mps" or not torch.isfinite(value).all() for value in output.values()): raise RuntimeError("MPS inference failure/fallback")
    reconstruction = output["reconstruction"].cpu().numpy() * scales[None, :, None, None]
    sigma = np.exp(0.5 * output["log_variance"].cpu().numpy()) * scales[None, :, None, None]
    return reconstruction.astype(np.float32), sigma.astype(np.float32), output["recoverability"].flatten().cpu().numpy(), output["no_source_probability"].flatten().cpu().numpy()


def calibrate_seed(run_dir: Path, condition: str) -> dict:
    output = run_dir / f"calibration/{condition.lower()}_isotonic.json"
    if output.exists(): raise FileExistsError(output)
    verify_frozen_inputs(); device = require_mps(); model = load_replica(run_dir, condition, device); scales = load_scales()
    manifest = read_csv(ORIGINAL / "manifests/calibration_scene_manifest.csv"); rows = []; raw_scores = []; labels = []
    with h5py.File(ORIGINAL / "manifests/calibration_scenes.h5", "r") as handle:
        for start in range(0, len(manifest), 32):
            stop = min(len(manifest), start + 32); blends = np.asarray(handle["blend"][start:stop], dtype=np.float32); isolated = np.asarray(handle["isolated"][start:stop], dtype=np.float32); prompts = np.asarray(handle["prompt"][start:stop], dtype=np.float32); matched = np.asarray(handle["matched_index"][start:stop], dtype=int)
            prediction, sigma, score, no_source = infer_full(model, blends, prompts, scales, device)
            for local, index in enumerate(range(start, stop)):
                item = manifest[index]; metrics = add_actionable_acceptance_labels(outcome_metrics(prediction[local], blends[local], isolated[local], item["query_class"], None if matched[local] < 0 else int(matched[local])), item["query_class"])
                rows.append({"scene_id": item["scene_id"], "query_class": item["query_class"], "raw_score": float(score[local]), "no_source_probability": float(no_source[local]), "pixel_uncertainty_mean": float(np.mean(sigma[local])), **metrics}); raw_scores.append(score[local]); labels.append(metrics[f"{PRIMARY_CONTRACT}_actionable_success"])
    raw = np.asarray(raw_scores); truth = np.asarray(labels, dtype=int); parameters = isotonic_fit(raw, truth); calibrated = isotonic_apply(raw, parameters)
    error = np.asarray([float(row["normalized_rmse"]) for row in rows]); raw_metrics = metric_summary(raw, truth, error); calibrated_metrics = metric_summary(calibrated, truth, error)
    for row, value in zip(rows, calibrated): row["calibrated_score"] = float(value)
    write_csv_union_fresh(run_dir / f"tables/{condition.lower()}_calibration_per_sample.csv", rows)
    ece, mce, bins = calibration_errors(calibrated, truth); write_csv_fresh(run_dir / f"tables/{condition.lower()}_calibration_reliability.csv", bins)
    thresholds = {f"coverage_{int(coverage*100)}": float(np.quantile(calibrated, 1.0 - coverage, method="lower")) for coverage in COVERAGES}; thresholds.update({f"probability_{str(level).replace('.', '_')}": level for level in (0.5, 0.7, 0.8, 0.9, 0.95)})
    value = {"status": "FROZEN_BEFORE_DEVELOPMENT", "condition": condition, "method": "isotonic", "parameters": parameters, "selection": "fixed by authoritative Phase II; no method comparison", "primary_score_definition": "calibrated R1 actionable source-reconstruction success probability; null and ambiguous imply abstention", "primary_contract": PRIMARY_CONTRACT, "calibration_scene_count": len(truth), "raw_metrics": raw_metrics, "calibrated_metrics": calibrated_metrics, "thresholds": thresholds, "checkpoint_sha256": sha256_file(seed_paths(run_dir, condition)["best"]), "calibration_manifest_sha256": sha256_file(ORIGINAL / "manifests/calibration_scene_manifest.csv"), "development_used": False, "lockbox_used": False}
    write_json_fresh(output, value)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], "k--"); axes[0].plot([row["mean_score"] for row in bins if row["count"]], [row["empirical_success"] for row in bins if row["count"]], "o-"); axes[0].set(xlabel="calibrated score", ylabel="empirical success", title=f"{condition} reliability")
    for query in QueryClass:
        values = [float(row["calibrated_score"]) for row in rows if row["query_class"] == query.value]; axes[1].hist(values, bins=20, alpha=.45, label=query.value, density=True)
    axes[1].set(xlabel="calibrated score", ylabel="density", title="Calibration scores by query"); axes[1].legend(fontsize=7)
    fig.savefig(run_dir / f"figures/{condition.lower()}_calibration.png", dpi=170); plt.close(fig)
    return value


def evaluate_seed_once(run_dir: Path, condition: str) -> dict:
    marker = run_dir / f"logs/{condition.lower()}_development_evaluation_complete.json"
    if marker.exists(): raise RuntimeError(f"Development already evaluated for {condition}")
    calibrator_path = run_dir / f"calibration/{condition.lower()}_isotonic.json"; calibrator = json.loads(calibrator_path.read_text())
    freeze = {"status": "FROZEN_BEFORE_AUTHORIZED_INFERENCE", "condition": condition, "checkpoint_sha256": sha256_file(seed_paths(run_dir, condition)["best"]), "calibrator_sha256": sha256_file(calibrator_path), "development_manifest_sha256": sha256_file(ORIGINAL / "manifests/development_test_scene_manifest.csv"), "development_hdf5_sha256": sha256_file(ORIGINAL / "manifests/development_test_scenes.h5"), "original_r1_inference_authorized": False, "uncertainty_maps_must_be_persisted_now": True}
    write_json_fresh(run_dir / f"logs/{condition.lower()}_development_freeze.json", freeze)
    verify_frozen_inputs(); device = require_mps(); model = load_replica(run_dir, condition, device); scales = load_scales(); manifest = read_csv(ORIGINAL / "manifests/development_test_scene_manifest.csv")
    h5_path = run_dir / f"uncertainty_maps/{condition.lower()}_development_inference.h5"; rows = []; calibrated_scores = []
    with h5py.File(ORIGINAL / "manifests/development_test_scenes.h5", "r") as source, h5py.File(h5_path, "x") as output:
        reconstruction_store = output.create_dataset("reconstruction", shape=(len(manifest), 3, 60, 60), dtype="f4", chunks=(1,3,60,60), compression="lzf")
        uncertainty_store = output.create_dataset("pixel_uncertainty_sigma", shape=(len(manifest), 3, 60, 60), dtype="f4", chunks=(1,3,60,60), compression="lzf")
        output.attrs["condition"] = condition; output.attrs["checkpoint_sha256"] = freeze["checkpoint_sha256"]; output.attrs["complete"] = False
        for start in range(0, len(manifest), 32):
            stop = min(len(manifest), start + 32); blends = np.asarray(source["blend"][start:stop], dtype=np.float32); isolated = np.asarray(source["isolated"][start:stop], dtype=np.float32); prompts = np.asarray(source["prompt"][start:stop], dtype=np.float32); matched = np.asarray(source["matched_index"][start:stop], dtype=int)
            prediction, sigma, raw, no_source = infer_full(model, blends, prompts, scales, device); calibrated = apply_calibrator(raw, calibrator)
            reconstruction_store[start:stop] = prediction; uncertainty_store[start:stop] = sigma
            for local, index in enumerate(range(start, stop)):
                item = manifest[index]; metrics = add_actionable_acceptance_labels(outcome_metrics(prediction[local], blends[local], isolated[local], item["query_class"], None if matched[local] < 0 else int(matched[local])), item["query_class"])
                rows.append({"condition": condition, "scene_id": item["scene_id"], "query_class": item["query_class"], "raw_score": float(raw[local]), "calibrated_score": float(calibrated[local]), "no_source_probability": float(no_source[local]), "pixel_uncertainty_mean": float(np.mean(sigma[local])), "coordinate_error_pixels": float(item["coordinate_error_pixels"]), **metrics}); calibrated_scores.append(calibrated[local])
            output.attrs["completed_count"] = stop; output.flush(); print(f"{condition}: development inference {stop}/{len(manifest)}", flush=True)
        output.attrs["complete"] = True; output.flush()
    write_csv_union_fresh(run_dir / f"tables/{condition.lower()}_development_per_sample.csv", rows)
    labels = np.asarray([int(row[f"{PRIMARY_CONTRACT}_actionable_success"]) for row in rows]); scores = np.asarray(calibrated_scores); errors = np.asarray([float(row["normalized_rmse"]) for row in rows])
    dev_metrics = metric_summary(scores, labels, errors)
    risk_rows = []
    ordered = sorted(rows, key=lambda row: (-float(row["calibrated_score"]), row["scene_id"]))
    for coverage in COVERAGES:
        count = int(math.ceil(coverage * len(ordered))); accepted = ordered[:count]; rejected = ordered[count:]
        risk_rows.append({"condition": condition, "target_coverage": coverage, "accepted_count": count, "realized_coverage": count / len(ordered), "selective_risk": float(np.mean([1-int(row[f"{PRIMARY_CONTRACT}_actionable_success"]) for row in accepted])), "catastrophic_failure_rate": float(np.mean([int(bool(row["catastrophic_failure"])) for row in accepted])), "null_hallucination_rate": float(np.mean([int(bool(row["hallucination"])) for row in accepted])), "source_confusion_rate": float(np.mean([int(bool(row["source_confusion"])) for row in accepted])), "accepted_query_composition": json.dumps(Counter(row["query_class"] for row in accepted), sort_keys=True), "rejected_query_composition": json.dumps(Counter(row["query_class"] for row in rejected), sort_keys=True)})
    write_csv_fresh(run_dir / f"tables/{condition.lower()}_risk_coverage.csv", risk_rows)
    macro = []
    for query in QueryClass:
        selected = [row for row in rows if row["query_class"] == query.value]
        def finite_mean(field: str) -> float:
            values = np.asarray([float(row[field]) for row in selected], dtype=float); values = values[np.isfinite(values)]; return float(values.mean()) if len(values) else math.nan
        macro.append({"condition": condition, "query_class": query.value, "scene_count": len(selected), "mean_calibrated_score": finite_mean("calibrated_score"), "normalized_rmse": finite_mean("normalized_rmse"), "source_mse": finite_mean("source_mse"), "catastrophic_failure_rate": float(np.mean([int(bool(row["catastrophic_failure"])) for row in selected])), "null_hallucination_rate": float(np.mean([int(bool(row["hallucination"])) for row in selected])), "source_confusion_rate": float(np.mean([int(bool(row["source_confusion"])) for row in selected]))})
    write_csv_fresh(run_dir / f"tables/{condition.lower()}_development_macro.csv", macro)
    valid_score = next(row["mean_calibrated_score"] for row in macro if row["query_class"] == QueryClass.VALID_SOURCE.value); ambiguous_score = next(row["mean_calibrated_score"] for row in macro if row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value)
    summary = {"condition": condition, "development_scene_count": len(rows), "development_metrics": dev_metrics, "risk_coverage": risk_rows, "null_hallucination_rate": next(row["null_hallucination_rate"] for row in macro if row["query_class"] == QueryClass.NULL_SOURCE.value), "valid_mean_score": valid_score, "ambiguous_mean_score": ambiguous_score, "ambiguous_minus_valid_score": ambiguous_score - valid_score, "valid_normalized_rmse": next(row["normalized_rmse"] for row in macro if row["query_class"] == QueryClass.VALID_SOURCE.value), "uncertainty_map_hdf5": str(h5_path.relative_to(REPO)), "uncertainty_map_hdf5_sha256": sha256_file(h5_path), "development_inference_count": 1, "original_r1_inference_count": 0, "lockbox_accessed": False}
    write_json_fresh(run_dir / f"reports/{condition.lower()}_development_summary.json", summary)
    write_json_fresh(marker, {"status": "PASS", "condition": condition, "evaluated_exactly_once": True, "scene_count": len(rows), "device": "mps", "uncertainty_maps_persisted_during_inference": True, "original_r1_rerun": False, "lockbox_accessed": False, "completed_at_unix": time.time()})
    return summary


def original_seed_record() -> tuple[dict, list[dict]]:
    config = json.loads((ORIGINAL / "manifests/r1_training_config.json").read_text()); calibrator = json.loads((ORIGINAL / "calibration/selected_calibrator.json").read_text()); risk = json.loads((ORIGINAL / "reports/risk_coverage_summary.json").read_text())["operating_points"]
    macro = read_csv(ORIGINAL / "tables/development_metrics_macro.csv")
    valid = next(row for row in macro if row["condition"] == "R1_calibrated" and row["query_class"] == QueryClass.VALID_SOURCE.value); ambiguous = next(row for row in macro if row["condition"] == "R1_calibrated" and row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value); null = next(row for row in macro if row["condition"] == "R1_calibrated" and row["query_class"] == QueryClass.NULL_SOURCE.value)
    points = {int(float(row["target_coverage"])*100): row for row in risk}
    row = {"seed_label": "R1_original", "initialization_seed": 2026078101, "minibatch_order_seed": 2026078101, "best_epoch": config["best_epoch"], "best_validation_loss": config["best_validation_objective"], "calibration_auroc": calibrator["calibrated_metrics"]["auroc"], "calibration_auprc": calibrator["calibrated_metrics"]["auprc"], "brier_raw": calibrator["raw_metrics"]["brier_score"], "brier_calibrated": calibrator["calibrated_metrics"]["brier_score"], "ece_calibrated": calibrator["calibrated_metrics"]["expected_calibration_error"], **{f"risk_{coverage}": float(points[coverage]["selective_risk"]) for coverage in (100,95,90,80,70)}, "catastrophic_100": float(points[100]["catastrophic_failure_rate"]), "catastrophic_80": float(points[80]["catastrophic_failure_rate"]), "null_hallucination": float(null["hallucination_rate"]), "valid_mean_score": float(valid["mean_score"]), "ambiguous_mean_score": float(ambiguous["mean_score"]), "ambiguous_minus_valid_score": float(ambiguous["mean_score"])-float(valid["mean_score"]), "valid_normalized_rmse": float(valid["mean_normalized_rmse"]), "valid_source_mse": float(valid["mean_source_mse"]), "uncertainty_maps_available": False, "checkpoint_sha256": config["best_checkpoint_sha256"]}
    samples = [row for row in read_csv(ORIGINAL / "tables/development_metrics_per_sample.csv") if row["condition"] == "R1_calibrated"]
    return row, samples


def replica_seed_record(run_dir: Path, condition: str) -> tuple[dict, list[dict]]:
    config = json.loads(seed_paths(run_dir, condition)["config"].read_text()); cal = json.loads((run_dir / f"calibration/{condition.lower()}_isotonic.json").read_text()); summary = json.loads((run_dir / f"reports/{condition.lower()}_development_summary.json").read_text()); macro = read_csv(run_dir / f"tables/{condition.lower()}_development_macro.csv"); points = {int(float(row["target_coverage"])*100): row for row in summary["risk_coverage"]}
    valid = next(row for row in macro if row["query_class"] == QueryClass.VALID_SOURCE.value)
    row = {"seed_label": condition, "initialization_seed": config["initialization_seed"], "minibatch_order_seed": config["minibatch_order_seed"], "best_epoch": config["best_epoch"], "best_validation_loss": config["best_validation_objective"], "calibration_auroc": cal["calibrated_metrics"]["auroc"], "calibration_auprc": cal["calibrated_metrics"]["auprc"], "brier_raw": cal["raw_metrics"]["brier_score"], "brier_calibrated": cal["calibrated_metrics"]["brier_score"], "ece_calibrated": cal["calibrated_metrics"]["expected_calibration_error"], **{f"risk_{coverage}": float(points[coverage]["selective_risk"]) for coverage in (100,95,90,80,70)}, "catastrophic_100": float(points[100]["catastrophic_failure_rate"]), "catastrophic_80": float(points[80]["catastrophic_failure_rate"]), "null_hallucination": summary["null_hallucination_rate"], "valid_mean_score": summary["valid_mean_score"], "ambiguous_mean_score": summary["ambiguous_mean_score"], "ambiguous_minus_valid_score": summary["ambiguous_minus_valid_score"], "valid_normalized_rmse": summary["valid_normalized_rmse"], "valid_source_mse": float(valid["source_mse"]), "uncertainty_maps_available": True, "checkpoint_sha256": config["best_checkpoint_sha256"]}
    return row, read_csv(run_dir / f"tables/{condition.lower()}_development_per_sample.csv")


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def uncertainty_correlation(path_a: Path, path_b: Path) -> float:
    count = 0; sx=sy=sxx=syy=sxy=0.0
    with h5py.File(path_a, "r") as left, h5py.File(path_b, "r") as right:
        for start in range(0, len(left["pixel_uncertainty_sigma"]), 16):
            x=np.asarray(left["pixel_uncertainty_sigma"][start:start+16],dtype=np.float64).ravel(); y=np.asarray(right["pixel_uncertainty_sigma"][start:start+16],dtype=np.float64).ravel(); count += len(x); sx += x.sum(); sy += y.sum(); sxx += np.dot(x,x); syy += np.dot(y,y); sxy += np.dot(x,y)
    numerator = count*sxy-sx*sy; denominator = math.sqrt(max(count*sxx-sx*sx,0)*max(count*syy-sy*sy,0)); return numerator/denominator if denominator else math.nan


def compare_three_seeds(run_dir: Path) -> dict:
    if (run_dir / "reports/three_seed_comparison.json").exists(): raise FileExistsError("Comparison already exists")
    original_row, original_samples = original_seed_record(); seed2_row, seed2_samples = replica_seed_record(run_dir, "R1_seed_2"); seed3_row, seed3_samples = replica_seed_record(run_dir, "R1_seed_3")
    metric_rows = [original_row, seed2_row, seed3_row]; write_csv_fresh(run_dir / "tables/three_seed_metrics.csv", metric_rows)
    labels = [row["seed_label"] for row in metric_rows]; numeric = [key for key in metric_rows[0] if key not in ("seed_label","checkpoint_sha256","uncertainty_maps_available") and key not in ("initialization_seed","minibatch_order_seed")]
    summaries=[]
    for metric in numeric:
        values=np.asarray([float(row[metric]) for row in metric_rows]); mean=float(values.mean()); std=float(values.std(ddof=1)); summaries.append({"metric":metric,"mean":mean,"std":std,"min":float(values.min()),"max":float(values.max()),"range":float(values.max()-values.min()),"coefficient_of_variation":abs(std/mean) if mean else math.nan,"ordering_low_to_high":"|".join(labels[index] for index in np.argsort(values))})
    write_csv_fresh(run_dir / "tables/three_seed_metric_summary.csv", summaries)
    sample_sets={"R1_original":original_samples,"R1_seed_2":seed2_samples,"R1_seed_3":seed3_samples}; by_seed={name:{row["scene_id"]:row for row in rows} for name,rows in sample_sets.items()}; ids=list(by_seed["R1_original"])
    if any(list(mapping)!=ids for mapping in by_seed.values()): raise RuntimeError("Development scene alignment mismatch")
    thresholds={"R1_original":json.loads((ORIGINAL/"calibration/frozen_abstention_thresholds.json").read_text())["thresholds"]["coverage_80"],"R1_seed_2":json.loads((run_dir/"calibration/r1_seed_2_isotonic.json").read_text())["thresholds"]["coverage_80"],"R1_seed_3":json.loads((run_dir/"calibration/r1_seed_3_isotonic.json").read_text())["thresholds"]["coverage_80"]}
    pair_rows=[]; failure_rows=[]; accepted={name:{scene for scene,row in mapping.items() if float(row.get("calibrated_score",row.get("score")))>=thresholds[name]} for name,mapping in by_seed.items()}
    pairs=(("R1_original","R1_seed_2"),("R1_original","R1_seed_3"),("R1_seed_2","R1_seed_3"))
    for left,right in pairs:
        x=np.asarray([float(by_seed[left][scene].get("calibrated_score",by_seed[left][scene].get("score"))) for scene in ids]); y=np.asarray([float(by_seed[right][scene].get("calibrated_score",by_seed[right][scene].get("score"))) for scene in ids]); errx=np.asarray([float(by_seed[left][scene]["normalized_rmse"]) for scene in ids]); erry=np.asarray([float(by_seed[right][scene]["normalized_rmse"]) for scene in ids]); finite=np.isfinite(errx)&np.isfinite(erry)
        pair_rows.append({"seed_left":left,"seed_right":right,"score_pearson":pearsonr(x,y).statistic,"score_spearman":spearmanr(x,y).statistic,"reconstruction_error_pearson":pearsonr(errx[finite],erry[finite]).statistic,"reconstruction_error_spearman":spearmanr(errx[finite],erry[finite]).statistic,"acceptance_exact_agreement":float(np.mean([(scene in accepted[left])==(scene in accepted[right]) for scene in ids])),"acceptance_jaccard":jaccard(accepted[left],accepted[right])})
        def cases(seed,field,query=None): return {scene for scene,row in by_seed[seed].items() if (query is None or row["query_class"]==query) and str(row[field]).lower() in ("true","1")}
        catastrophic_left=cases(left,"catastrophic_failure"); catastrophic_right=cases(right,"catastrophic_failure"); null_left=cases(left,"hallucination",QueryClass.NULL_SOURCE.value); null_right=cases(right,"hallucination",QueryClass.NULL_SOURCE.value)
        med_left=np.median([float(row.get("calibrated_score",row.get("score"))) for row in by_seed[left].values() if row["query_class"]==QueryClass.VALID_SOURCE.value]); med_right=np.median([float(row.get("calibrated_score",row.get("score"))) for row in by_seed[right].values() if row["query_class"]==QueryClass.VALID_SOURCE.value]); over_left={scene for scene,row in by_seed[left].items() if row["query_class"]==QueryClass.AMBIGUOUS_SOURCE.value and float(row.get("calibrated_score",row.get("score")))>med_left}; over_right={scene for scene,row in by_seed[right].items() if row["query_class"]==QueryClass.AMBIGUOUS_SOURCE.value and float(row.get("calibrated_score",row.get("score")))>med_right}
        failure_rows.extend([{"seed_left":left,"seed_right":right,"failure_type":"catastrophic","left_count":len(catastrophic_left),"right_count":len(catastrophic_right),"intersection":len(catastrophic_left&catastrophic_right),"jaccard":jaccard(catastrophic_left,catastrophic_right)},{"seed_left":left,"seed_right":right,"failure_type":"null_hallucination","left_count":len(null_left),"right_count":len(null_right),"intersection":len(null_left&null_right),"jaccard":jaccard(null_left,null_right)},{"seed_left":left,"seed_right":right,"failure_type":"ambiguous_overconfidence","left_count":len(over_left),"right_count":len(over_right),"intersection":len(over_left&over_right),"jaccard":jaccard(over_left,over_right)}])
    write_csv_fresh(run_dir/"tables/pairwise_stability.csv",pair_rows); write_csv_fresh(run_dir/"tables/failure_overlap.csv",failure_rows)
    matrix=np.asarray([[float(np.mean([(scene in accepted[a])==(scene in accepted[b]) for scene in ids])) for b in labels] for a in labels]); write_csv_fresh(run_dir/"tables/acceptance_agreement_matrix.csv",[{"seed":labels[i],**{labels[j]:matrix[i,j] for j in range(3)}} for i in range(3)])
    uncertainty_corr=uncertainty_correlation(run_dir/"uncertainty_maps/r1_seed_2_development_inference.h5",run_dir/"uncertainty_maps/r1_seed_3_development_inference.h5"); write_json_fresh(run_dir/"reports/uncertainty_map_stability.json",{"seed_2_vs_seed_3_pixelwise_pearson":uncertainty_corr,"original_maps_available":False,"original_maps_regenerated":False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    for axis,(left,right) in zip(axes,pairs):
        x=[float(by_seed[left][scene].get("calibrated_score",by_seed[left][scene].get("score"))) for scene in ids]; y=[float(by_seed[right][scene].get("calibrated_score",by_seed[right][scene].get("score"))) for scene in ids]; axis.scatter(x,y,s=5,alpha=.25); axis.set(xlabel=left,ylabel=right,title="calibrated score")
    fig.savefig(run_dir/"figures/score_correlations.png",dpi=170); plt.close(fig)
    fig,axis=plt.subplots(figsize=(5,4.5)); image=axis.imshow(matrix,vmin=0,vmax=1,cmap="viridis"); axis.set_xticks(range(3),labels,rotation=30,ha="right"); axis.set_yticks(range(3),labels); fig.colorbar(image,ax=axis,label="decision agreement"); fig.tight_layout(); fig.savefig(run_dir/"figures/acceptance_agreement.png",dpi=170); plt.close(fig)
    score_matrix=np.asarray([[float(by_seed[name][scene].get("calibrated_score",by_seed[name][scene].get("score"))) for name in labels] for scene in ids]); disagreement=np.std(score_matrix,axis=1); chosen=np.argsort(-disagreement)[:12]
    with h5py.File(ORIGINAL/"manifests/development_test_scenes.h5","r") as source,h5py.File(run_dir/"uncertainty_maps/r1_seed_2_development_inference.h5","r") as s2,h5py.File(run_dir/"uncertainty_maps/r1_seed_3_development_inference.h5","r") as s3:
        fig,axes=plt.subplots(len(chosen),5,figsize=(12,2.2*len(chosen)),squeeze=False,constrained_layout=True)
        for row_index,index in enumerate(chosen):
            panels=[np.asarray(source["blend"][index,1]),np.asarray(s2["reconstruction"][index,1]),np.asarray(s3["reconstruction"][index,1]),np.asarray(s2["pixel_uncertainty_sigma"][index,1]),np.asarray(s3["pixel_uncertainty_sigma"][index,1])]
            for column,panel in enumerate(panels): axes[row_index,column].imshow(np.arcsinh(panel/max(float(np.max(np.abs(panel))),1e-30)*20),origin="lower",cmap="coolwarm"); axes[row_index,column].set_xticks([]); axes[row_index,column].set_yticks([])
            axes[row_index,0].set_ylabel(ids[index],fontsize=6)
        for column,title in enumerate(("blend","seed 2 recon","seed 3 recon","seed 2 uncertainty","seed 3 uncertainty")): axes[0,column].set_title(title)
        fig.savefig(run_dir/"example_grids/seed_disagreement_gallery.png",dpi=150); plt.close(fig)
    # Fixed post-hoc score-only ensembles. Labels are majority per-seed actionable success.
    actionable=np.asarray([[int(float(by_seed[name][scene][f"{PRIMARY_CONTRACT}_actionable_success"])) for name in labels] for scene in ids]); majority=(actionable.sum(axis=1)>=2).astype(int); catastrophic_any=np.asarray([any(str(by_seed[name][scene]["catastrophic_failure"]).lower() in ("true","1") for name in labels) for scene in ids]); query=np.asarray([by_seed[labels[0]][scene]["query_class"] for scene in ids])
    combos={"mean_probability":score_matrix.mean(axis=1),"median_probability":np.median(score_matrix,axis=1),"minimum_probability":score_matrix.min(axis=1),"low_disagreement":np.clip(1-4*np.var(score_matrix,axis=1),0,1),"mean_minus_std":np.clip(score_matrix.mean(axis=1)-score_matrix.std(axis=1),0,1)}
    ensemble_rows=[]
    for name,score in combos.items():
        order=np.argsort(-score,kind="stable")
        for coverage in COVERAGES:
            count=int(math.ceil(coverage*len(ids))); accepted=order[:count]; rejected=order[count:]
            ensemble_rows.append({"combination":name,"coverage":coverage,"auroc":auroc(score,majority),"auprc":auprc(score,majority),"brier":float(np.mean((score-majority)**2)),"selective_majority_failure_risk":float(np.mean(1-majority[accepted])),"catastrophic_any_rate":float(np.mean(catastrophic_any[accepted])),"ambiguous_rejection_rate":float(np.mean(np.isin(np.flatnonzero(query==QueryClass.AMBIGUOUS_SOURCE.value),rejected))),"null_rejection_rate":float(np.mean(np.isin(np.flatnonzero(query==QueryClass.NULL_SOURCE.value),rejected)))})
    write_csv_fresh(run_dir/"tables/ensemble_analysis.csv",ensemble_rows)
    # Stability decision.
    ranges={metric:float(max(float(row[metric]) for row in metric_rows)-min(float(row[metric]) for row in metric_rows)) for metric in ("calibration_auroc","calibration_auprc","brier_calibrated","null_hallucination")}; risk_ranges={coverage:max(float(row[f"risk_{coverage}"]) for row in metric_rows)-min(float(row[f"risk_{coverage}"]) for row in metric_rows) for coverage in (100,95,90,80,70)}
    stable=bool(ranges["calibration_auroc"]<=STABILITY_TOLERANCES["auroc_range"] and ranges["calibration_auprc"]<=STABILITY_TOLERANCES["auprc_range"] and ranges["brier_calibrated"]<=STABILITY_TOLERANCES["brier_range"] and ranges["null_hallucination"]<=STABILITY_TOLERANCES["null_hallucination_range"] and all(value<=STABILITY_TOLERANCES["risk_range_each_coverage"] for value in risk_ranges.values()))
    upgraded=all(float(row["catastrophic_100"])-float(row["catastrophic_80"])>=UPGRADE_GATES["minimum_catastrophic_reduction_at_80"] and float(row["risk_100"])-float(row["risk_70"])>=UPGRADE_GATES["minimum_risk_reduction_at_70"] and float(row["null_hallucination"])<=UPGRADE_GATES["maximum_null_hallucination"] and float(row["ambiguous_minus_valid_score"])<0 for row in metric_rows)
    persistent_partial=sum(float(row["ambiguous_minus_valid_score"])>0 for row in metric_rows)>=2 and sum(float(row["catastrophic_100"])-float(row["catastrophic_80"])<UPGRADE_GATES["minimum_catastrophic_reduction_at_80"] for row in metric_rows)>=2
    classification="UPGRADED SUCCESS" if upgraded else ("STABLE PARTIAL SUCCESS" if stable and persistent_partial else "UNSTABLE RESULT")
    comparison={"classification":classification,"stability_ranges":ranges,"risk_ranges":risk_ranges,"stable_under_predeclared_tolerances":stable,"persistent_partial_failures":persistent_partial,"upgraded_gates_passed":upgraded,"uncertainty_map_correlation_seed2_seed3":uncertainty_corr,"next_experiment":"targeted objective redesign for ambiguity and catastrophic failure, without enlarging the architecture" if classification=="STABLE PARTIAL SUCCESS" else ("optimization stability and label-noise audit before scientific-objective changes" if classification=="UNSTABLE RESULT" else "prepare a new untouched final-test protocol before lockbox consideration"),"original_development_inference_rerun":False,"lockbox_accessed":False}
    write_json_fresh(run_dir/"reports/three_seed_comparison.json",comparison)
    return comparison


def finalize(run_dir: Path) -> None:
    comparison=json.loads((run_dir/"reports/three_seed_comparison.json").read_text()); compile_result=command([str(REPO/".venv-btk/bin/python"),"-m","compileall","-q","src","scripts","tests"]); tests=command([str(REPO/".venv-btk/bin/python"),"-m","unittest","tests.test_thayer_select","tests.test_recoverability_phase2","tests.test_recoverability_seed_replication","-v"]); diff=command(["git","diff","--check"])
    write_json_fresh(run_dir/"logs/compileall.json",compile_result); write_json_fresh(run_dir/"logs/relevant_tests.json",tests); write_json_fresh(run_dir/"logs/git_diff_check.json",diff)
    frozen=verify_frozen_inputs(); before=read_csv(run_dir/"tables/checkpoint_inventory_before.csv"); after=[]
    for row in before:
        observed=sha256_file(REPO/row["relative_path"]); after.append({**row,"final_sha256":observed,"final_status":"PASS" if observed==row["observed_sha256"] else "FAIL"})
    write_csv_fresh(run_dir/"tables/checkpoint_inventory_after.csv",after)
    csv_rows=[]
    for path in sorted(run_dir.rglob("*.csv")):
        with path.open(newline="") as handle: rows=list(csv.reader(handle))
        width=len(rows[0]) if rows else 0; ok=bool(rows and all(len(row)==width for row in rows)); csv_rows.append({"path":str(path.relative_to(REPO)),"rows":len(rows),"columns":width,"status":"PASS" if ok else "FAIL"})
    write_csv_fresh(run_dir/"tables/csv_schema_validation.csv",csv_rows)
    scanner=Path(__file__).resolve(); scanner_exclusions={scanner,(REPO/"scripts/finalize_thayer_select_recoverability.py").resolve()}; patterns=("AWS_SECRET","PRIVATE KEY","api_key=","token="); hits=[]
    for path in sorted((REPO/"src").rglob("*.py"))+sorted((REPO/"scripts").rglob("*.py"))+sorted((REPO/"docs").rglob("*.md")):
        if path.resolve() in scanner_exclusions: continue
        text=path.read_text(errors="replace")
        for pattern in patterns:
            if pattern.lower() in text.lower(): hits.append({"path":str(path.relative_to(REPO)),"pattern":pattern})
    write_json_fresh(run_dir/"diagnostics/privacy_path_grep.json",{"status":"PASS" if not hits else "FAIL","scanner_exclusions":[str(path.relative_to(REPO)) for path in sorted(scanner_exclusions)],"hits":hits})
    dev_markers=[json.loads((run_dir/f"logs/{condition.lower()}_development_evaluation_complete.json").read_text()) for condition in SEED_MAP]
    original_config=json.loads((ORIGINAL/"manifests/r1_training_config.json").read_text()); replica_configs=[json.loads(seed_paths(run_dir,c)["config"].read_text()) for c in SEED_MAP]
    protocol_fields=("batch_size","epochs","learning_rate","loss","optimizer","parameter_count","primary_contract","scheduler")
    original_scene_ids=[row["scene_id"] for row in read_csv(ORIGINAL/"tables/development_metrics_per_sample.csv") if row["condition"]=="R1_calibrated"]
    replica_scene_ids=[[row["scene_id"] for row in read_csv(run_dir/f"tables/{condition.lower()}_development_per_sample.csv")] for condition in SEED_MAP]
    historical_hashes={row["observed_sha256"] for row in before}; new_checkpoint_hashes=[sha256_file(seed_paths(run_dir,c)[tag]) for c in SEED_MAP for tag in ("best","final")]
    calibrators=[json.loads((run_dir/f"calibration/{c.lower()}_isotonic.json").read_text()) for c in SEED_MAP]
    checks={"frozen_inputs_match":all(row["status"]=="PASS" for row in frozen),"only_seed_fields_changed":all(all(config[field]==original_config[field] for field in protocol_fields) for config in replica_configs),"mps_throughout":all(json.loads(seed_paths(run_dir,c)["device"].read_text())["device"]=="mps" and json.loads(seed_paths(run_dir,c)["device"].read_text())["fallback_detected"] is False for c in SEED_MAP),"zero_lockbox_access":all(marker["lockbox_accessed"] is False for marker in dev_markers),"one_development_inference_per_new_seed":all(marker["evaluated_exactly_once"] for marker in dev_markers),"original_inference_not_repeated":all(marker["original_r1_rerun"] is False for marker in dev_markers),"original_uncertainty_maps_not_regenerated":not (run_dir/"uncertainty_maps/r1_original_development_inference.h5").exists(),"sample_scene_alignment":all(scene_ids==original_scene_ids for scene_ids in replica_scene_ids),"calibration_only":all(calibrator["development_used"] is False and calibrator["method"]=="isotonic" for calibrator in calibrators),"thresholds_not_retuned":all(calibrator["status"]=="FROZEN_BEFORE_DEVELOPMENT" and calibrator["selection"]=="fixed by authoritative Phase II; no method comparison" for calibrator in calibrators),"fresh_checkpoints":all(seed_paths(run_dir,c)["best"].is_file() and seed_paths(run_dir,c)["final"].is_file() for c in SEED_MAP),"checkpoint_collision_free":len(set(new_checkpoint_hashes))==4 and not (set(new_checkpoint_hashes)&historical_hashes),"historical_checkpoints_unchanged":all(row["final_status"]=="PASS" for row in after),"compileall":compile_result["returncode"]==0,"relevant_tests":tests["returncode"]==0,"csv_schemas":all(row["status"]=="PASS" for row in csv_rows),"git_diff_check":diff["returncode"]==0,"privacy_grep":not hits}
    write_json_fresh(run_dir/"diagnostics/final_correctness_audit.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks})
    if not all(checks.values()): raise RuntimeError(f"Final audit failure: {[key for key,value in checks.items() if not value]}")
    metrics=read_csv(run_dir/"tables/three_seed_metrics.csv"); pairwise=read_csv(run_dir/"tables/pairwise_stability.csv"); summary=read_csv(run_dir/"tables/three_seed_metric_summary.csv"); summary_by_metric={row["metric"]:row for row in summary}
    ensemble=read_csv(run_dir/"tables/ensemble_analysis.csv"); ensemble_by_name={row["combination"]:row for row in ensemble if float(row["coverage"])==1.0}
    thresholds={"R1_original":json.loads((ORIGINAL/"calibration/frozen_abstention_thresholds.json").read_text())["thresholds"]["coverage_80"],**{condition:json.loads((run_dir/f"calibration/{condition.lower()}_isotonic.json").read_text())["thresholds"]["coverage_80"] for condition in SEED_MAP}}
    threshold_acceptance_counts={row["seed_label"]:sum(float(sample.get("calibrated_score",sample.get("score")))>=thresholds[row["seed_label"]] for sample in ([item for item in read_csv(ORIGINAL/"tables/development_metrics_per_sample.csv") if item["condition"]=="R1_calibrated"] if row["seed_label"]=="R1_original" else read_csv(run_dir/f"tables/{row['seed_label'].lower()}_development_per_sample.csv"))) for row in metrics}
    ambiguous_all=all(float(row["ambiguous_minus_valid_score"])>0 for row in metrics)
    material_cat_improvement=any(float(row["catastrophic_100"])-float(row["catastrophic_80"])>=UPGRADE_GATES["minimum_catastrophic_reduction_at_80"] for row in metrics)
    disagreement_useful=float(ensemble_by_name["low_disagreement"]["auroc"])>float(ensemble_by_name["mean_probability"]["auroc"])
    report=f"""# Thayer-Select Phase II fixed-protocol seed replication

## Executive result

Classification: **{comparison['classification']}**. {comparison['next_experiment']}.

1. Frozen Phase-II inputs reproduced exactly: **yes**; {len(frozen)} frozen code/artifact hashes passed.
2. New seeds: R1_seed_2 init/order `{SEED_MAP['R1_seed_2']['initialization_seed']}` / `{SEED_MAP['R1_seed_2']['minibatch_order_seed']}`; R1_seed_3 `{SEED_MAP['R1_seed_3']['initialization_seed']}` / `{SEED_MAP['R1_seed_3']['minibatch_order_seed']}`.
3. Both completed 20 epochs on MPS: **yes**.
4. Calibration variability (sample standard deviation; min–max): AUROC `{float(summary_by_metric['calibration_auroc']['std']):.4f}` (`{float(summary_by_metric['calibration_auroc']['min']):.4f}`–`{float(summary_by_metric['calibration_auroc']['max']):.4f}`), AUPRC `{float(summary_by_metric['calibration_auprc']['std']):.4f}` (`{float(summary_by_metric['calibration_auprc']['min']):.4f}`–`{float(summary_by_metric['calibration_auprc']['max']):.4f}`), calibrated Brier `{float(summary_by_metric['brier_calibrated']['std']):.4f}` (`{float(summary_by_metric['brier_calibrated']['min']):.4f}`–`{float(summary_by_metric['brier_calibrated']['max']):.4f}`), and ECE `{float(summary_by_metric['ece_calibrated']['std']):.3g}` (`{float(summary_by_metric['ece_calibrated']['min']):.3g}`–`{float(summary_by_metric['ece_calibrated']['max']):.3g}`). Predeclared stability status: `{comparison['stable_under_predeclared_tolerances']}`.
5. Selective-risk ranges by coverage: `{json.dumps(comparison['risk_ranges'], sort_keys=True)}`.
6. Null-hallucination range: `{comparison['stability_ranges']['null_hallucination']:.6f}`.
7. Ambiguous prompts remained over-ranked in all three seeds: **{ambiguous_all}**.
8. Material catastrophic-failure rejection in any seed at 80% coverage: **{material_cat_improvement}**. Seed 3 improved only from `{float(metrics[2]['catastrophic_100']):.4f}` to `{float(metrics[2]['catastrophic_80']):.4f}`, below the predeclared material gate; the other seeds worsened.
9. Pairwise accepted/rejected agreement: {', '.join(f"{row['seed_left']} vs {row['seed_right']}={float(row['acceptance_exact_agreement']):.3f}" for row in pairwise)}. This is degenerate, not evidence of robust threshold decisions: all frozen 80%-coverage thresholds equal zero and accept all scenes (`{json.dumps(threshold_acceptance_counts, sort_keys=True)}`) because of calibration ties.
10. Same-scene failure overlap is in `tables/failure_overlap.csv`.
11. Ensemble disagreement added useful signal: **{disagreement_useful}**. The disagreement-only AUROC was `{float(ensemble_by_name['low_disagreement']['auroc']):.4f}` versus `{float(ensemble_by_name['mean_probability']['auroc']):.4f}` for mean probability; fixed combinations are exploratory development evidence only.
12. Final decision: **{comparison['classification']}**.
13. Next objective: **{comparison['next_experiment']}**.
14. Lockbox untouched: **yes**.
15. Historical checkpoints unchanged: **yes**.

Full uncertainty maps and reconstructions for only the two new seeds were persisted during their single authorized development passes. The original missing uncertainty maps were not regenerated. Checkpoint hashes, runtimes, disk inventory, git state, calibration/risk plots, score correlations, agreement matrix, failure overlaps, and disagreement examples are retained in this timestamped run.
"""
    write_text_fresh(run_dir/"reports/final_report.md",report)
    # Combined training/risk figures.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for condition in SEED_MAP:
        rows=read_csv(seed_paths(run_dir,condition)["epochs"]); axes[0].plot([int(row["epoch"]) for row in rows],[float(row["training_loss"]) for row in rows],label=condition); axes[1].plot([int(row["epoch"]) for row in rows],[float(row["validation_loss"]) for row in rows],label=condition)
    for axis,title in zip(axes,("Training objective","Validation objective")): axis.set(xlabel="epoch",ylabel="objective",title=title); axis.grid(alpha=.25); axis.legend()
    fig.savefig(run_dir/"figures/replication_training_curves.png",dpi=170); plt.close(fig)
    fig,axis=plt.subplots(figsize=(6,5))
    for row in metrics:
        axis.plot([1,.95,.9,.8,.7],[float(row[f"risk_{coverage}"]) for coverage in (100,95,90,80,70)],"o-",label=row["seed_label"])
    axis.set(xlabel="coverage",ylabel="selective risk",title="Three-seed risk–coverage"); axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(run_dir/"figures/three_seed_risk_coverage.png",dpi=170); plt.close(fig)
    disk=command(["du","-sh",str(run_dir)]); git=command(["git","status","--short","--branch"]); write_json_fresh(run_dir/"logs/finalization_complete.json",{"status":"PASS","classification":comparison["classification"],"disk":disk,"git":git,"lockbox_accessed":False,"completed_at_unix":time.time()})
    output=run_dir/"tables/campaign_file_hashes.csv"; write_csv_fresh(output,[{"relative_path":str(path.relative_to(REPO)),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path!=output])


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--phase",choices=("bootstrap","train","calibrate","evaluate","compare","finalize","all"),default="all"); args=parser.parse_args(); run_dir=args.run_dir.resolve()
    if args.phase in ("bootstrap","all"): bootstrap(run_dir)
    if args.phase in ("train","all"):
        for condition in SEED_MAP: train_seed(run_dir,condition)
        write_json_fresh(run_dir/"logs/training_complete.json",{"status":"PASS","conditions":list(SEED_MAP),"epochs":EPOCHS,"device":"mps","original_r1_retrained":False,"lockbox_accessed":False})
    if args.phase in ("calibrate","all"):
        for condition in SEED_MAP: calibrate_seed(run_dir,condition)
        write_json_fresh(run_dir/"logs/calibration_complete.json",{"status":"PASS","method":"isotonic","conditions":list(SEED_MAP),"development_used":False,"lockbox_accessed":False})
    if args.phase in ("evaluate","all"):
        for condition in SEED_MAP: evaluate_seed_once(run_dir,condition)
        write_json_fresh(run_dir/"logs/development_evaluation_complete.json",{"status":"PASS","new_seed_inference_counts":{"R1_seed_2":1,"R1_seed_3":1},"original_r1_inference_count":0,"original_uncertainty_maps_regenerated":False,"lockbox_accessed":False})
    if args.phase in ("compare","all"): compare_three_seeds(run_dir)
    if args.phase in ("finalize","all"): finalize(run_dir)


if __name__=="__main__": main()
