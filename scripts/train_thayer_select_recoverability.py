#!/usr/bin/env python3
"""Generate empirical labels and train Phase-II R0/R1 on MPS only."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.models_thayer_select import ThayerSelectNet
from src.prompt_semantics import QueryClass
from src.recoverability import bounded_heteroscedastic_gaussian_nll
from thayer_select_prompt_ablation_common import CompactSelectNet, require_mps, seed_everything
from thayer_select_recoverability_common import (
    BATCH_SIZE,
    CONTRACT_BALANCE_RANGE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LOG_VARIANCE,
    MAX_UNCERTAINTY_SATURATION_FRACTION,
    MIN_LOG_VARIANCE,
    PHASE1,
    PRIMARY_CONTRACT_CANDIDATE,
    TEACHER_CHECKPOINT,
    TRAINING_SEED,
    UNCERTAINTY_SATURATION_WEIGHT,
    add_actionable_acceptance_labels,
    load_scales,
    load_teacher,
    model_parameter_counts,
    outcome_metrics,
    read_csv,
    sha256_file,
    write_csv_fresh,
    write_csv_union_fresh,
    write_json_fresh,
)


def teacher_infer(model, blends: np.ndarray, prompts: np.ndarray, scales: np.ndarray, device: torch.device) -> np.ndarray:
    normalized = np.asarray(blends, dtype=np.float32) / scales[None, :, None, None]
    model_input = np.concatenate((normalized, np.asarray(prompts, dtype=np.float32)), axis=1)
    tensor = torch.from_numpy(np.ascontiguousarray(model_input)).to(device)
    if tensor.device.type != "mps":
        raise RuntimeError("Teacher inference device fallback")
    with torch.no_grad():
        prediction = model(tensor)
    if prediction.device.type != "mps" or not torch.isfinite(prediction).all():
        raise RuntimeError("Invalid teacher inference")
    return prediction.detach().cpu().numpy() * scales[None, :, None, None]


def label_partition(run_dir: Path, partition: str, model, scales: np.ndarray, device: torch.device, *, limit: int | None = None) -> list[dict]:
    manifest = read_csv(run_dir / f"manifests/{partition}_scene_manifest.csv")
    if limit is not None:
        by_query = defaultdict(list)
        for index, row in enumerate(manifest):
            by_query[row["query_class"]].append(index)
        per_query = max(1, limit // len(by_query))
        selected = sorted(index for values in by_query.values() for index in values[:per_query])[:limit]
    else:
        selected = list(range(len(manifest)))
    records = []
    with h5py.File(run_dir / f"manifests/{partition}_scenes.h5", "r") as handle:
        for offset in range(0, len(selected), 32):
            indices = selected[offset : offset + 32]
            blends = np.asarray(handle["blend"][indices], dtype=np.float32)
            isolated = np.asarray(handle["isolated"][indices], dtype=np.float32)
            prompts = np.asarray(handle["prompt"][indices], dtype=np.float32)
            matched = np.asarray(handle["matched_index"][indices], dtype=int)
            predictions = teacher_infer(model, blends, prompts, scales, device)
            for batch_index, scene_index in enumerate(indices):
                row = manifest[scene_index]
                metrics = outcome_metrics(predictions[batch_index], blends[batch_index], isolated[batch_index], row["query_class"], None if matched[batch_index] < 0 else int(matched[batch_index]))
                labeled = add_actionable_acceptance_labels(metrics, row["query_class"])
                failure_reasons = [name for name in ("hallucination", "forced_source_selection", "source_confusion", "catastrophic_failure") if bool(labeled.get(name, False))]
                records.append({
                    "scene_id": row["scene_id"], "partition": partition, "query_class": row["query_class"],
                    "teacher_checkpoint_sha256": sha256_file(TEACHER_CHECKPOINT),
                    "metric_implementation_sha256": sha256_file(REPO / "scripts/thayer_select_recoverability_common.py"),
                    "failure_reasons": "|".join(failure_reasons) if failure_reasons else "none",
                    **labeled,
                })
    return records


def generate_teacher_labels(run_dir: Path, device: torch.device) -> str:
    if (run_dir / "logs/teacher_label_generation_complete.json").exists():
        raise RuntimeError("Teacher labels already exist; refusing regeneration")
    scales = load_scales()
    teacher = load_teacher(device)
    label_sets = {}
    for partition in ("training", "validation"):
        rows = label_partition(run_dir, partition, teacher, scales, device)
        label_sets[partition] = rows
        write_csv_union_fresh(run_dir / f"tables/{partition}_teacher_reliability_labels.csv", rows)
    descriptive = []
    for partition in ("validation", "calibration"):
        descriptive.extend(label_partition(run_dir, partition, teacher, scales, device, limit=256))
    summary_rows = []
    for partition, rows in label_sets.items():
        for query in QueryClass:
            selected = [row for row in rows if row["query_class"] == query.value]
            for contract in ("strict", "moderate", "permissive"):
                summary_rows.append({"partition": partition, "query_class": query.value, "contract": contract, "scene_count": len(selected), "success_rate": float(np.mean([int(row[f"{contract}_success"]) for row in selected])) if selected else math.nan})
    write_csv_fresh(run_dir / "tables/teacher_label_balance_by_query.csv", summary_rows)
    descriptive_summary = []
    for partition in ("validation", "calibration"):
        for query in QueryClass:
            selected = [row for row in descriptive if row["partition"] == partition and row["query_class"] == query.value]
            descriptive_summary.append({"partition": partition, "query_class": query.value, "scene_count": len(selected), "mean_normalized_rmse": float(np.nanmean([float(row["normalized_rmse"]) for row in selected])) if selected else math.nan, "hallucination_rate": float(np.mean([int(bool(row["hallucination"])) for row in selected])) if selected else math.nan, "forced_source_selection_rate": float(np.mean([int(bool(row["forced_source_selection"])) for row in selected])) if selected else math.nan})
    write_csv_fresh(run_dir / "tables/frozen_phase1_teacher_descriptive_subset.csv", descriptive_summary)

    combined = label_sets["training"] + label_sets["validation"]
    rates = {contract: float(np.mean([int(row[f"{contract}_success"]) for row in combined])) for contract in ("strict", "moderate", "permissive")}
    low, high = CONTRACT_BALANCE_RANGE
    selected_contract = PRIMARY_CONTRACT_CANDIDATE
    reason = "predeclared intended primary contract; combined training/validation teacher labels are learnable"
    if not low <= rates[selected_contract] <= high:
        eligible = [name for name, rate in rates.items() if low <= rate <= high]
        candidates = eligible or list(rates)
        selected_contract = min(candidates, key=lambda name: abs(rates[name] - 0.5))
        reason = "predeclared imbalance fallback selected closest-to-balanced contract using training/validation labels only"
    selection = {"status": "FROZEN_BEFORE_R1_TRAINING", "primary_contract": selected_contract, "candidate": PRIMARY_CONTRACT_CANDIDATE, "combined_training_validation_success_rates": rates, "allowed_balance_range": [low, high], "reason": reason, "calibration_or_development_used": False, "teacher_checkpoint_sha256": sha256_file(TEACHER_CHECKPOINT)}
    write_json_fresh(run_dir / "manifests/primary_contract_selection.json", selection)

    overlap = Counter(row["failure_reasons"] for row in combined)
    write_csv_fresh(run_dir / "tables/teacher_failure_reason_overlap.csv", [{"failure_reason_combination": key, "count": value} for key, value in sorted(overlap.items())])
    write_json_fresh(run_dir / "logs/teacher_label_generation_complete.json", {"status": "PASS", "device": "mps", "training_labels": len(label_sets["training"]), "validation_labels": len(label_sets["validation"]), "primary_contract": selected_contract, "calibration_labels_used_for_training": 0, "development_labels_used": 0, "lockbox_scenes_used": 0, "completed_at_unix": time.time()})
    return selected_contract


def freeze_actionable_labels(run_dir: Path) -> str:
    """Create append-only actionable labels from preserved oracle outcomes."""

    outputs = {}
    combined = []
    for partition in ("training", "validation"):
        source = read_csv(run_dir / f"tables/{partition}_teacher_reliability_labels.csv")
        rows = []
        for row in source:
            source_query = row["query_class"] in (QueryClass.VALID_SOURCE.value, QueryClass.PERTURBED_VALID.value)
            record = {"scene_id": row["scene_id"], "partition": partition, "query_class": row["query_class"]}
            for contract in ("strict", "moderate", "permissive"):
                record[f"{contract}_actionable_success"] = int(source_query and int(float(row[f"{contract}_success"])) == 1)
            rows.append(record)
        path = run_dir / f"tables/{partition}_actionable_acceptance_labels.csv"
        if not path.exists():
            write_csv_union_fresh(path, rows)
        outputs[partition] = rows
        combined.extend(rows)
    rates = {contract: float(np.mean([row[f"{contract}_actionable_success"] for row in combined])) for contract in ("strict", "moderate", "permissive")}
    low, high = CONTRACT_BALANCE_RANGE
    selected = PRIMARY_CONTRACT_CANDIDATE
    reason = "predeclared moderate actionable label is learnable"
    if not low <= rates[selected] <= high:
        selected = min(rates, key=lambda name: abs(rates[name] - 0.5))
        reason = "all actionable contracts are highly imbalanced; predeclared closest-to-balanced fallback selected using training/validation only"
    selection = {"status": "FROZEN_BEFORE_R1_TRAINING", "supersedes": "manifests/primary_contract_selection.json for global accept/abstain supervision", "primary_contract": selected, "candidate": PRIMARY_CONTRACT_CANDIDATE, "combined_training_validation_actionable_success_rates": rates, "allowed_balance_range": [low, high], "reason": reason, "null_global_acceptance_target": 0, "ambiguous_global_acceptance_target": 0, "null_recognition_target_in_dedicated_head": 1, "calibration_or_development_used": False}
    path = run_dir / "manifests/primary_actionable_contract_selection.json"
    if not path.exists():
        write_json_fresh(path, selection)
    return selected


class MixedSceneDataset(Dataset):
    def __init__(self, run_dir: Path, partition: str, contract: str) -> None:
        if partition not in ("training", "validation"):
            raise ValueError("Training may open only training or validation")
        self.path = run_dir / f"manifests/{partition}_scenes.h5"
        self.manifest = read_csv(run_dir / f"manifests/{partition}_scene_manifest.csv")
        labels = read_csv(run_dir / f"tables/{partition}_actionable_acceptance_labels.csv")
        if [row["scene_id"] for row in labels] != [row["scene_id"] for row in self.manifest]:
            raise RuntimeError("Teacher labels and scenes are misaligned")
        self.labels = np.asarray([float(row[f"{contract}_actionable_success"]) for row in labels], dtype=np.float32)
        self.scales = load_scales()
        self.handle = None

    def __len__(self) -> int:
        return len(self.manifest)

    def _handle(self):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return self.handle

    def __getitem__(self, index: int):
        handle = self._handle()
        blend = np.asarray(handle["blend"][index], dtype=np.float32) / self.scales[:, None, None]
        prompt = np.asarray(handle["prompt"][index], dtype=np.float32)
        matched = int(handle["matched_index"][index])
        query = QueryClass(self.manifest[index]["query_class"])
        if matched >= 0:
            target = np.asarray(handle["isolated"][index, matched], dtype=np.float32) / self.scales[:, None, None]
        else:
            target = np.zeros_like(blend)
        target_defined = float(query is not QueryClass.AMBIGUOUS_SOURCE)
        no_source = float(query is QueryClass.NULL_SOURCE)
        query_index = list(QueryClass).index(query)
        return tuple(torch.from_numpy(np.ascontiguousarray(value)) if isinstance(value, np.ndarray) else torch.tensor(value) for value in (blend, prompt, target, np.float32(target_defined), np.float32(self.labels[index]), np.float32(no_source), np.int64(query_index)))


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def loss_batch(condition: str, model, batch, device: torch.device, global_positive_weight: float = 1.0) -> tuple[torch.Tensor, dict, torch.Tensor]:
    blend, prompt, target, target_defined, recoverable, no_source = [value.to(device) for value in batch[:6]]
    query_index = batch[6]
    if any(value.device.type != "mps" for value in (blend, prompt, target, target_defined, recoverable, no_source)):
        raise RuntimeError("MPS fallback detected")
    defined = target_defined > 0.5
    if condition == "R0":
        prediction = model(torch.cat((blend, prompt), dim=1))
        per_sample = (prediction - target).square().mean(dim=(1, 2, 3))
        reconstruction = per_sample[defined].mean() if torch.any(defined) else prediction.sum() * 0.0
        components = {"reconstruction": reconstruction, "global": prediction.sum() * 0.0, "no_source": prediction.sum() * 0.0, "saturation": prediction.sum() * 0.0}
        return reconstruction, components, query_index
    output = model(blend, prompt)
    per_sample_mse = (output["reconstruction"] - target).square().mean(dim=(1, 2, 3))
    base_mse = per_sample_mse[defined].mean() if torch.any(defined) else output["reconstruction"].sum() * 0.0
    nll_elements = bounded_heteroscedastic_gaussian_nll(output["reconstruction"], target, output["log_variance"], min_log_variance=MIN_LOG_VARIANCE, max_log_variance=MAX_LOG_VARIANCE, reduction="none")
    valid_source = defined & (no_source < 0.5)
    brightness = torch.amax(torch.abs(target), dim=1)
    peaks = torch.amax(brightness, dim=(1, 2), keepdim=True)
    source_support = brightness > 0.01 * peaks
    uncertainty_mask = valid_source[:, None, None, None] & source_support[:, None, :, :]
    uncertainty_mask = uncertainty_mask.expand_as(nll_elements)
    source_nll = nll_elements[uncertainty_mask].mean() if torch.any(uncertainty_mask) else output["log_variance"].sum() * 0.0
    reconstruction = base_mse + 0.1 * source_nll
    global_probability = output["recoverability"].flatten().clamp(1e-6, 1.0 - 1e-6)
    global_loss = -torch.mean(global_positive_weight * recoverable.float() * torch.log(global_probability) + (1.0 - recoverable.float()) * torch.log(1.0 - global_probability))
    no_source_loss = F.binary_cross_entropy(output["no_source_probability"].flatten(), no_source.float())
    margin = 0.05 * (MAX_LOG_VARIANCE - MIN_LOG_VARIANCE)
    saturation = (F.relu((MIN_LOG_VARIANCE + margin) - output["log_variance"]).square() + F.relu(output["log_variance"] - (MAX_LOG_VARIANCE - margin)).square()).mean()
    saturation_fraction = ((output["log_variance"] <= MIN_LOG_VARIANCE + margin) | (output["log_variance"] >= MAX_LOG_VARIANCE - margin)).float().mean()
    total = reconstruction + 0.5 * global_loss + 0.25 * no_source_loss + UNCERTAINTY_SATURATION_WEIGHT * saturation
    components = {"reconstruction": reconstruction, "base_mse": base_mse, "source_nll": source_nll, "global": global_loss, "no_source": no_source_loss, "saturation": saturation, "saturation_fraction": saturation_fraction, "log_variance_min": output["log_variance"].min(), "log_variance_max": output["log_variance"].max()}
    return total, components, query_index


def train_condition(run_dir: Path, condition: str, contract: str, device: torch.device) -> dict:
    paths = {"best": run_dir / f"checkpoints/{condition.lower()}_best.pth", "final": run_dir / f"checkpoints/{condition.lower()}_final.pth", "history": run_dir / f"tables/{condition.lower()}_epochs.csv", "query": run_dir / f"tables/{condition.lower()}_validation_by_query.csv", "config": run_dir / f"manifests/{condition.lower()}_training_config.json"}
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"Checkpoint/output collision for {condition}")
    seed_everything(TRAINING_SEED)
    model = (CompactSelectNet(4) if condition == "R0" else ThayerSelectNet(min_log_variance=MIN_LOG_VARIANCE, max_log_variance=MAX_LOG_VARIANCE)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_data = MixedSceneDataset(run_dir, "training", contract)
    validation_data = MixedSceneDataset(run_dir, "validation", contract)
    positives = float(np.sum(train_data.labels)); negatives = len(train_data) - positives
    global_positive_weight = min(20.0, negatives / max(positives, 1.0)) if condition == "R1" else 1.0
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    best_loss = math.inf; best_epoch = -1; best_state = None; epochs = []; query_rows = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); train_total = 0.0; train_count = 0
        component_sums = defaultdict(float)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, components, _ = loss_batch(condition, model, batch, device, global_positive_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf training loss in {condition}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError(f"Non-finite gradient in {condition}")
            optimizer.step()
            batch_count = len(batch[0]); train_total += float(loss.detach().cpu()) * batch_count; train_count += batch_count
            for key, value in components.items():
                component_sums[key] += float(value.detach().cpu()) * batch_count
        model.eval(); validation_total = 0.0; validation_count = 0; by_query = defaultdict(list); epoch_var_min = math.inf; epoch_var_max = -math.inf; validation_saturation_sum = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                loss, components, query_index = loss_batch(condition, model, batch, device, global_positive_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"NaN/Inf validation loss in {condition}")
                count = len(batch[0]); validation_total += float(loss.detach().cpu()) * count; validation_count += count
                for query_value in torch.unique(query_index).detach().cpu().tolist():
                    by_query[int(query_value)].append(float(loss.detach().cpu()))
                if condition == "R1":
                    epoch_var_min = min(epoch_var_min, float(components["log_variance_min"].detach().cpu()))
                    epoch_var_max = max(epoch_var_max, float(components["log_variance_max"].detach().cpu()))
                    validation_saturation_sum += float(components["saturation_fraction"].detach().cpu()) * count
        record = {"condition": condition, "epoch": epoch, "training_loss": train_total / train_count, "validation_loss": validation_total / validation_count, "learning_rate": optimizer.param_groups[0]["lr"], "device": "mps", **{f"training_{key}": value / train_count for key, value in component_sums.items()}}
        if condition == "R1":
            record.update({"validation_log_variance_min": epoch_var_min, "validation_log_variance_max": epoch_var_max, "validation_saturation_fraction": validation_saturation_sum / validation_count})
            if epoch_var_min < MIN_LOG_VARIANCE or epoch_var_max > MAX_LOG_VARIANCE:
                raise RuntimeError("Uncertainty bound violation")
            if record["validation_saturation_fraction"] > MAX_UNCERTAINTY_SATURATION_FRACTION:
                raise RuntimeError("Uncertainty saturation fraction exceeded frozen stop threshold")
        epochs.append(record)
        for query_value, values in by_query.items():
            query_rows.append({"condition": condition, "epoch": epoch, "query_class": list(QueryClass)[query_value].value, "batch_count": len(values), "mean_validation_objective": float(np.mean(values))})
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]; best_epoch = epoch; best_state = cpu_state(model)
        scheduler.step()
        print(json.dumps(record, sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("No best checkpoint")
    final_payload = {"state_dict": cpu_state(model), "condition": condition, "epoch": EPOCHS, "selection": "final epoch", "training_seed": TRAINING_SEED, "primary_contract": contract}
    best_payload = {"state_dict": best_state, "condition": condition, "epoch": best_epoch, "selection": "minimum frozen validation objective", "validation_loss": best_loss, "training_seed": TRAINING_SEED, "primary_contract": contract}
    if paths["best"].exists() or paths["final"].exists():
        raise FileExistsError("Checkpoint collision")
    torch.save(best_payload, paths["best"]); torch.save(final_payload, paths["final"])
    write_csv_fresh(paths["history"], epochs); write_csv_fresh(paths["query"], query_rows)
    counts = model_parameter_counts()
    config = {"status": "FROZEN", "condition": condition, "architecture": "compact coordinate-conditioned Thayer U-Net" if condition == "R0" else "same compact prompted backbone plus bounded pixel uncertainty, actionable contract-success, and no-source heads", "parameter_count": counts[condition], "parameter_growth_over_R0": counts[condition] - counts["R0"], "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam", "scheduler": "CosineAnnealingLR", "learning_rate": LEARNING_RATE, "training_seed": TRAINING_SEED, "data_order_seed": TRAINING_SEED, "training_scenes": len(train_data), "validation_scenes": len(validation_data), "primary_contract": contract, "global_positive_weight": global_positive_weight, "global_acceptance_semantics": "positive only for valid/perturbed-valid reconstructions passing the selected contract; null and ambiguous target abstention", "ambiguous_reconstruction_treatment": "excluded from pixel reconstruction loss; global actionable target zero", "null_reconstruction_target": "exact zero image; no-source head target one; global actionable target zero", "loss": "masked normalized MSE" if condition == "R0" else f"whole-image valid/null MSE + 0.1 bounded valid-source-support NLL + 0.5 weighted actionable BCE + 0.25 no-source BCE + {UNCERTAINTY_SATURATION_WEIGHT} full-map saturation penalty", "log_variance_bounds": [MIN_LOG_VARIANCE, MAX_LOG_VARIANCE] if condition == "R1" else None, "uncertainty_nll_support": "oracle requested-source >1% peak support for valid and perturbed-valid training/evaluation loss only; never a model input" if condition == "R1" else None, "uncertainty_saturation_margin_fraction": 0.05 if condition == "R1" else None, "uncertainty_saturation_weight": UNCERTAINTY_SATURATION_WEIGHT if condition == "R1" else None, "maximum_validation_saturation_fraction": MAX_UNCERTAINTY_SATURATION_FRACTION if condition == "R1" else None, "best_epoch": best_epoch, "best_validation_objective": best_loss, "best_checkpoint": str(paths["best"].relative_to(REPO)), "best_checkpoint_sha256": sha256_file(paths["best"]), "final_checkpoint": str(paths["final"].relative_to(REPO)), "final_checkpoint_sha256": sha256_file(paths["final"]), "runtime_seconds": time.time() - started, "device": "mps", "teacher_checkpoint_sha256": sha256_file(TEACHER_CHECKPOINT), "oracle_inputs_to_forward_pass": 0, "calibration_scenes_used": 0, "development_scenes_used": 0, "lockbox_scenes_used": 0}
    write_json_fresh(paths["config"], config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "r0", "r1"), default="all")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    preparation = json.loads((run_dir / "logs/data_preparation_complete.json").read_text())
    if preparation.get("status") != "PASS" or preparation.get("lockbox_scenes_accessed") != 0:
        raise RuntimeError("Data preparation gate failed")
    device = require_mps()
    correction_hashes = run_dir / "tables/campaign_code_hashes_superseding_actionable_acceptance.csv"
    if not correction_hashes.exists():
        initial = read_csv(run_dir / "tables/campaign_code_hashes.csv")
        rows = []
        for row in initial:
            path = REPO / row["relative_path"]
            rows.append({"relative_path": row["relative_path"], "bootstrap_sha256": row["sha256"], "superseding_sha256": sha256_file(path), "changed_after_bootstrap": int(row["sha256"] != sha256_file(path)), "reason": "append-only actionable-acceptance correction before R1 training"})
        write_csv_fresh(correction_hashes, rows)
    saturation_hashes = run_dir / "tables/campaign_code_hashes_superseding_uncertainty_saturation.csv"
    if args.phase == "r1" and not saturation_hashes.exists():
        initial = read_csv(run_dir / "tables/campaign_code_hashes.csv")
        rows = []
        for row in initial:
            path = REPO / row["relative_path"]
            rows.append({"relative_path": row["relative_path"], "bootstrap_sha256": row["sha256"], "superseding_sha256": sha256_file(path), "changed_after_bootstrap": int(row["sha256"] != sha256_file(path)), "reason": "append-only uncertainty-saturation correction before R1 restart"})
        write_csv_fresh(saturation_hashes, rows)
    source_support_hashes = run_dir / "tables/campaign_code_hashes_superseding_source_supported_uncertainty.csv"
    if args.phase == "r1" and not source_support_hashes.exists():
        initial = read_csv(run_dir / "tables/campaign_code_hashes.csv")
        rows = []
        for row in initial:
            path = REPO / row["relative_path"]
            rows.append({"relative_path": row["relative_path"], "bootstrap_sha256": row["sha256"], "superseding_sha256": sha256_file(path), "changed_after_bootstrap": int(row["sha256"] != sha256_file(path)), "reason": "append-only source-supported uncertainty correction before second R1 restart"})
        write_csv_fresh(source_support_hashes, rows)
    if (run_dir / "logs/teacher_label_generation_complete.json").exists():
        contract = freeze_actionable_labels(run_dir)
    else:
        generate_teacher_labels(run_dir, device)
        contract = freeze_actionable_labels(run_dir)
    conditions = ("R0", "R1") if args.phase == "all" else (args.phase.upper(),)
    _ = [train_condition(run_dir, condition, contract, device) for condition in conditions]
    configs = []
    for condition in ("R0", "R1"):
        path = run_dir / f"manifests/{condition.lower()}_training_config.json"
        if path.exists():
            configs.append(json.loads(path.read_text()))
    if len(configs) != 2:
        raise RuntimeError("Both R0 and R1 must be frozen before training completion")
    write_json_fresh(run_dir / "logs/training_complete.json", {"status": "PASS", "device": "mps", "all_epochs_completed": all(config["epochs"] == EPOCHS for config in configs), "conditions": ["R0", "R1"], "primary_contract": contract, "optional_seed_replications_run": 0, "optional_seed_replication_reason": "deferred to preserve primary calibration and frozen-development completion", "calibration_scenes_used": 0, "development_scenes_used": 0, "lockbox_scenes_used": 0, "completed_at_unix": time.time()})


if __name__ == "__main__":
    main()
