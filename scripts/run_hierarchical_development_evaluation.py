#!/usr/bin/env python3
"""Generate and evaluate the one-time frozen hierarchical development set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from src.btk_scene import load_catsim_catalog
from src.hierarchical_safety import QueryState, RISK_LIMITS, metric_specific_risks, normalized_policy_violation
from src.models_thayer_select import ThayerSelectNet
from prepare_hierarchical_safety_data import (
    BASE_SEED, CATALOG_PATH, SCHEMA_VERSION, SOURCE_SPLIT, artifact_stem, choose_sources,
    draw_positions, prompt_for_state, render_dataset, replay_audit, source_size,
)
from extract_hierarchical_safety_features import extract_dataset, fit_flux_floors, load_model
from calibrate_hierarchical_safety import (
    ALPHA, CLASS_TO_INDEX, CONFUSION_LIMIT, apply_vector, confusion_logits, query_logits,
    risk_outputs, sigmoid,
)


PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PHASE2 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
CONDITION_C = PHASE1 / "checkpoints/c_randomized_coordinate_prompt_best.pth"
R1_CHECKPOINT = PHASE2 / "checkpoints/r1_best.pth"
NORMALIZATION = PHASE1 / "manifests/normalization.json"
DEVELOPMENT_COUNTS = {
    "natural_unique": 1200, "perturbed_valid": 300, "null": 500, "ambiguous": 500,
    "low_snr": 125, "high_overlap": 125, "equal_flux_similar_size": 125, "confusion_prone": 125,
}
DEVELOPMENT_SEED = 2026071299


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str: return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL; descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle: handle.write(value)


def write_json_fresh(path: Path, value: object) -> None: write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists(): raise FileExistsError(path)
    frame.to_csv(path, index=False)


def development_labels(rng: np.random.Generator) -> list[tuple[QueryState, str, bool]]:
    values = []
    for label, count in DEVELOPMENT_COUNTS.items():
        if label == "null": state, stratum, perturbed = QueryState.NULL, "natural", False
        elif label == "ambiguous": state, stratum, perturbed = QueryState.AMBIGUOUS, "natural", False
        elif label == "perturbed_valid": state, stratum, perturbed = QueryState.UNIQUE_VALID, "natural", True
        elif label == "natural_unique": state, stratum, perturbed = QueryState.UNIQUE_VALID, "natural", False
        else: state, stratum, perturbed = QueryState.UNIQUE_VALID, label, False
        values.extend((state, stratum, perturbed) for _ in range(count))
    rng.shuffle(values); return values


def build_development_definitions(pool: pd.DataFrame, table) -> pd.DataFrame:
    rng = np.random.default_rng(DEVELOPMENT_SEED); records = []
    for index, (state, stratum, perturbed) in enumerate(development_labels(rng)):
        first, second = choose_sources(pool, table, rng, stratum); positions = draw_positions(rng, stratum, state); requested = int(rng.integers(0, 2))
        records.append({
            "scene_id": f"v2_development_{index:05d}", "dataset": "development", "source_partition": "development_test", "dataset_index": index,
            "query_state": state.value, "prompt_subtype": "PERTURBED_VALID" if perturbed else state.value, "sampling_stratum": stratum,
            "inverse_sampling_weight": 1.0, "operational_weight_applicable": 0, "requested_index_for_generation": requested,
            "source_a_row": int(first.catalog_row), "source_b_row": int(second.catalog_row), "source_a_id": first.persistent_source_id, "source_b_id": second.persistent_source_id,
            "source_a_group": first.duplicate_group_id, "source_b_group": second.duplicate_group_id,
            "source_a_x_arcsec": positions[0, 0], "source_a_y_arcsec": positions[0, 1], "source_b_x_arcsec": positions[1, 0], "source_b_y_arcsec": positions[1, 1],
            "scene_seed": DEVELOPMENT_SEED + index, "noise_seed": DEVELOPMENT_SEED + 1_000_000 + index, "prompt_seed": DEVELOPMENT_SEED + 2_000_000 + index,
            "schema_version": SCHEMA_VERSION,
        })
    return pd.DataFrame(records)


def r1_inference(run: Path, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS required for R1 development baseline")
    payload = torch.load(R1_CHECKPOINT, map_location="cpu", weights_only=False); model = ThayerSelectNet(min_log_variance=-8.0, max_log_variance=2.0); model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.eval(); model.to("mps"); reconstruction = []; score = []
    with h5py.File(run / "manifests/v2_development_scenes.h5", "r") as handle, torch.no_grad():
        for start in range(0, len(handle["blend"]), 128):
            blend = np.asarray(handle["blend"][start:start + 128], dtype=np.float32); prompt = np.asarray(handle["prompt"][start:start + 128], dtype=np.float32)
            output = model(torch.from_numpy(np.ascontiguousarray(blend / scales[None, :, None, None])).to("mps"), torch.from_numpy(np.ascontiguousarray(prompt)).to("mps"))
            reconstruction.append(output["reconstruction"].cpu().numpy() * scales[None, :, None, None]); score.append(output["recoverability"].flatten().cpu().numpy())
    values = np.concatenate(reconstruction).astype(np.float32); scores = np.concatenate(score).astype(np.float32)
    path = run / "features/v2_development_r1_outputs.h5"
    with h5py.File(path, "x") as handle:
        handle.create_dataset("reconstruction", data=values, chunks=(1, 3, 60, 60), compression="lzf"); handle.create_dataset("raw_recoverability", data=scores); handle.attrs["checkpoint_sha256"] = sha256_file(R1_CHECKPOINT); handle.attrs["complete"] = True
    return values, scores


def isotonic_apply(raw_score: np.ndarray) -> np.ndarray:
    calibrator = json.loads((PHASE2 / "calibration/selected_calibrator.json").read_text())["parameters"]
    return np.interp(raw_score, np.asarray(calibrator["upper_x"]), np.asarray(calibrator["value"]), left=calibrator["value"][0], right=calibrator["value"][-1])


def color_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred_flux = prediction.sum(axis=(-2, -1), dtype=np.float64); truth_flux = truth.sum(axis=(-2, -1), dtype=np.float64)
    if np.any(pred_flux <= 0) or np.any(truth_flux <= 0): return math.inf
    pred_color = -2.5 * np.log10(pred_flux[:-1] / pred_flux[1:]); truth_color = -2.5 * np.log10(truth_flux[:-1] / truth_flux[1:]); return float(np.max(np.abs(pred_color - truth_color)))


def invalid_outcome(prediction: np.ndarray, blend: np.ndarray, isolated: np.ndarray, state: str) -> dict:
    absolute_flux = float(np.sum(np.abs(prediction))); blend_flux = max(float(np.sum(np.abs(blend))), 1e-30); ratio = absolute_flux / blend_flux; zero_mse = float(np.mean(prediction**2)); source_mse = [float(np.mean((prediction - isolated[index]) ** 2)) for index in range(2)]; preference = int(np.argmin(source_mse))
    if state == "NULL": return {"hidden_flux_ratio": ratio, "hallucination": int(ratio > 0.10), "forced_source": 0, "source_preference": preference}
    forced = int(ratio > 0.10 and min(source_mse) < zero_mse); return {"hidden_flux_ratio": ratio, "hallucination": 0, "forced_source": forced, "source_preference": preference}


def two_way_cluster_ci(frame: pd.DataFrame, accepted: np.ndarray, outcome: np.ndarray, repetitions: int = 300) -> tuple[float, float]:
    groups = sorted(set(frame.source_a_group) | set(frame.source_b_group)); index = {group: i for i, group in enumerate(groups)}; a = frame.source_a_group.map(index).to_numpy(); b = frame.source_b_group.map(index).to_numpy(); rng = np.random.default_rng(20260712991); values = []
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups)); weights = counts[a] * counts[b] * accepted
        if weights.sum() > 0: values.append(float(np.sum(weights * outcome) / np.sum(weights)))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975])) if values else (math.nan, math.nan)


def risk_coverage(per_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = {
        "reconstruction_only": np.zeros(len(per_sample)), "random_rejection": np.random.default_rng(20260712992).random(len(per_sample)),
        "original_monolithic_R1": per_sample.r1_raw_score.to_numpy(), "hierarchical_query_gate": per_sample.query_margin.to_numpy(),
        "complete_hierarchical_policy": per_sample.recoverability_margin.to_numpy(), "oracle_risk_reference": -per_sample.oracle_policy_violation.fillna(np.inf).to_numpy(),
    }
    coverages = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50, 0.20, 0.10, 0.05)
    scopes = {"all": np.ones(len(per_sample), dtype=bool), "unique_valid": per_sample.query_state.eq("UNIQUE_VALID").to_numpy(), "null": per_sample.query_state.eq("NULL").to_numpy(), "ambiguous": per_sample.query_state.eq("AMBIGUOUS").to_numpy()}
    rows = []; bootstrap_rows = []
    for method, score in scores.items():
        for scope, mask in scopes.items():
            indices = np.flatnonzero(mask); ordered = indices[np.argsort(-score[indices], kind="stable")]
            for coverage in coverages:
                count = max(1, int(math.ceil(len(ordered) * coverage))); accepted_indices = ordered[:count]; accepted = np.zeros(len(per_sample), dtype=int); accepted[accepted_indices] = 1
                if scope == "unique_valid": outcome = per_sample.catastrophic_valid.to_numpy(dtype=float)
                elif scope == "null": outcome = per_sample.c_hallucination.to_numpy(dtype=float)
                elif scope == "ambiguous": outcome = per_sample.c_forced_source.to_numpy(dtype=float)
                else: outcome = per_sample.task_aligned_failure.to_numpy(dtype=float)
                risk = float(outcome[accepted_indices].mean()); rows.append({"method": method, "scope": scope, "target_coverage": coverage, "realized_coverage": count / len(ordered), "accepted_count": count, "samples": len(ordered), "accepted_case_risk": risk})
                if scope == "unique_valid" and coverage in (0.95, 0.90, 0.80, 0.70):
                    low, high = two_way_cluster_ci(per_sample.loc[mask].reset_index(drop=True), accepted[mask], outcome[mask]); bootstrap_rows.append({"method": method, "valid_coverage": coverage, "catastrophic_rate": risk, "cluster_ci_low": low, "cluster_ci_high": high, "bootstrap": "two-way source-group pigeonhole", "repetitions": 300})
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def make_gallery(run: Path, per_sample: pd.DataFrame) -> None:
    chosen = pd.concat((per_sample.nlargest(6, "recoverability_margin"), per_sample.nsmallest(6, "recoverability_margin"))).drop_duplicates("scene_id").head(12)
    with h5py.File(run / "manifests/v2_development_scenes.h5", "r") as source, h5py.File(run / "features/v2_development_frozen_reconstructions.h5", "r") as recon:
        fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 2.3 * len(chosen)))
        for row_number, row in enumerate(chosen.itertuples()):
            index = int(row.development_index); blend = np.asarray(source["blend"][index]); prediction = np.asarray(recon["reconstruction"][index]); matched = int(source["matched_index"][index]); truth = np.zeros_like(prediction) if matched < 0 else np.asarray(source["isolated"][index, matched])
            for ax, image, title in zip(axes[row_number], (blend, prediction, truth), ("blend", "Condition C", "requested truth/blank")):
                display = np.arcsinh(np.maximum(image, 0).sum(axis=0) / max(float(np.quantile(np.abs(image), 0.99)), 1e-6)); ax.imshow(display, origin="lower", cmap="gray"); ax.axis("off"); ax.set_title(title, fontsize=8)
            axes[row_number, 0].set_ylabel(f"{row.query_state}\nmargin={row.recoverability_margin:.2f}\naccept={row.full_policy_accept}", fontsize=7)
        fig.tight_layout(); fig.savefig(run / "example_grids/development_accepted_rejected_gallery.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); run = args.run_dir.resolve()
    freeze_path = run / "manifests/hierarchical_policy_freeze.json"; correction_path = run / "manifests/hierarchical_policy_freeze_superseding_nondegeneracy.json"
    if not freeze_path.is_file() or not correction_path.is_file(): raise RuntimeError("Complete policy freeze missing")
    if any((run / path).exists() for path in ("manifests/v2_development_scene_definitions.csv", "manifests/v2_development_scene_manifest.csv", "manifests/v2_development_scenes.h5", "logs/development_evaluation_started.json")): raise FileExistsError("Development path or marker already exists")
    condition_hash = sha256_file(CONDITION_C); r1_hash = sha256_file(R1_CHECKPOINT); split = pd.read_csv(SOURCE_SPLIT); pool = split[(split.partition == "development_test") & (split.engineering_excluded == 0)].copy(); forbidden = set(split.loc[split.partition == "sealed_lockbox", "duplicate_group_id"])
    if set(pool.duplicate_group_id) & forbidden: raise RuntimeError("Lockbox group in development pool")
    from astropy.table import Table
    table = Table.read(CATALOG_PATH, format="fits"); catalog, catalog_hash = load_catsim_catalog(CATALOG_PATH); definitions = build_development_definitions(pool, table); write_csv_fresh(run / "manifests/v2_development_scene_definitions.csv", definitions)
    manifest = render_dataset(run, "development", definitions, catalog, table); write_csv_fresh(run / "manifests/v2_development_scene_manifest.csv", manifest); replay = pd.DataFrame(replay_audit(run, "development", manifest, catalog)); write_csv_fresh(run / "tables/development_replay_audit.csv", replay)
    if not (replay.status == "PASS").all() or len(manifest) != 3000 or manifest.scene_id.nunique() != 3000: raise RuntimeError("Development manifest gate failed")
    checksum = {"definitions_sha256": sha256_file(run / "manifests/v2_development_scene_definitions.csv"), "manifest_sha256": sha256_file(run / "manifests/v2_development_scene_manifest.csv"), "hdf5_sha256": sha256_file(run / "manifests/v2_development_scenes.h5"), "scene_count": 3000, "source_partition": "development_test", "lockbox_used": False}
    write_json_fresh(run / "manifests/development_manifest_freeze.json", checksum)
    for path in (run / "manifests/v2_development_scene_definitions.csv", run / "manifests/v2_development_scene_manifest.csv", run / "manifests/v2_development_scenes.h5", run / "manifests/development_manifest_freeze.json"): os.chmod(path, 0o444)

    write_json_fresh(run / "logs/development_evaluation_started.json", {"status": "STARTED_ONCE", "started_at_unix": time.time(), "policy_freeze_sha256": sha256_file(freeze_path), "development_manifest_sha256": checksum["manifest_sha256"], "attempt": 1, "lockbox_used": False})
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32); floors = np.asarray(json.loads((run / "manifests/risk_flux_floors.json").read_text())["floor_by_band"], dtype=float); model = load_model(); extract_inventory = extract_dataset(run, "development", model, scales, floors); write_json_fresh(run / "tables/development_feature_inventory.json", extract_inventory)
    dev_npz = np.load(run / "features/v2_development_features.npz", allow_pickle=True); samples = pd.read_csv(run / "features/v2_development_samples.csv", keep_default_na=False); if_alignment = dev_npz["scene_id"].astype(str).tolist() == samples.scene_id.tolist() == manifest.scene_id.tolist()
    if not if_alignment: raise RuntimeError("Development feature alignment failed")
    vector = json.loads((run / "calibration/query_vector_scaling.json").read_text())["parameters"]; query_probability = apply_vector(query_logits(run, dev_npz), vector); policy = json.loads(freeze_path.read_text()); thresholds = policy["query_thresholds"]; conformal = json.loads((run / "calibration/risk_split_conformal.json").read_text()); confusion_temperature = json.loads((run / "calibration/confusion_temperature.json").read_text())["temperature"]
    predicted = {}
    for task in ("image", "flux", "centroid"):
        output = risk_outputs(run, dev_npz, task); predicted[f"{task}_median"] = np.maximum(np.expm1(output[:, 0]), 0); predicted[f"{task}_upper"] = np.maximum(np.expm1(output[:, 1] + conformal[task]["offset"]), 0)
    predicted["p_confusion"] = sigmoid(confusion_logits(run, dev_npz) / confusion_temperature)
    query_pass = (query_probability[:, 0] >= thresholds["unique_minimum"]) & (query_probability[:, 1] <= thresholds["null_maximum"]) & (query_probability[:, 2] <= thresholds["ambiguous_maximum"]); limits = RISK_LIMITS["moderate"]
    full_accept = query_pass & (predicted["image_upper"] < limits.image) & (predicted["flux_upper"] < limits.flux) & (predicted["centroid_upper"] < limits.centroid_pixels) & (predicted["p_confusion"] < CONFUSION_LIMIT)
    query_margins = np.column_stack(((query_probability[:, 0] - thresholds["unique_minimum"]) / max(1 - thresholds["unique_minimum"], 1e-12), (thresholds["null_maximum"] - query_probability[:, 1]) / max(thresholds["null_maximum"], 1e-12), (thresholds["ambiguous_maximum"] - query_probability[:, 2]) / max(thresholds["ambiguous_maximum"], 1e-12)))
    all_margins = np.column_stack((query_margins, (limits.image - predicted["image_upper"]) / limits.image, (limits.flux - predicted["flux_upper"]) / limits.flux, (limits.centroid_pixels - predicted["centroid_upper"]) / limits.centroid_pixels, (CONFUSION_LIMIT - predicted["p_confusion"]) / CONFUSION_LIMIT)); query_margin = query_margins.min(axis=1); recoverability_margin = all_margins.min(axis=1)
    r1_reconstruction, r1_raw = r1_inference(run, scales); r1_calibrated = isotonic_apply(r1_raw); r1_threshold = json.loads((PHASE2 / "calibration/frozen_abstention_thresholds.json").read_text())["thresholds"]["coverage_90"]; r1_accept = r1_calibrated >= r1_threshold

    rows = []
    with h5py.File(run / "manifests/v2_development_scenes.h5", "r") as source, h5py.File(run / "features/v2_development_frozen_reconstructions.h5", "r") as c_recon:
        for index, row in manifest.iterrows():
            blend = np.asarray(source["blend"][index]); isolated = np.asarray(source["isolated"][index]); matched = int(source["matched_index"][index]); c_prediction = np.asarray(c_recon["reconstruction"][index]); r_prediction = r1_reconstruction[index]
            base = {"development_index": index, "scene_id": row.scene_id, "query_state": row.query_state, "sampling_stratum": row.sampling_stratum, "source_a_group": row.source_a_group, "source_b_group": row.source_b_group, "p_unique": query_probability[index, 0], "p_null": query_probability[index, 1], "p_ambiguous": query_probability[index, 2], "query_margin": query_margin[index], "recoverability_margin": recoverability_margin[index], "query_gate_accept": int(query_pass[index]), "full_policy_accept": int(full_accept[index]), "r1_raw_score": r1_raw[index], "r1_calibrated_score": r1_calibrated[index], "r1_accept": int(r1_accept[index]), **{name: values[index] for name, values in predicted.items()}}
            if row.query_state == "UNIQUE_VALID":
                requested = isolated[matched]; alternate = isolated[1 - matched]; c_risk = metric_specific_risks(c_prediction, requested, alternate, flux_floor_by_band=floors); r_risk = metric_specific_risks(r_prediction, requested, alternate, flux_floor_by_band=floors); violation = normalized_policy_violation(c_risk, limits); r_violation = normalized_policy_violation(r_risk, limits); catastrophic = int(c_risk["confusion_risk"] or violation >= 2)
                base.update({"c_image_risk": c_risk["image_risk"], "c_flux_risk_g": c_risk["flux_risk_by_band"][0], "c_flux_risk_r": c_risk["flux_risk_by_band"][1], "c_flux_risk_z": c_risk["flux_risk_by_band"][2], "c_flux_risk_max": c_risk["flux_risk_max"], "c_centroid_pixels": c_risk["centroid_risk_pixels"], "c_color_error": color_error(c_prediction, requested), "c_confusion": int(c_risk["confusion_risk"]), "r1_image_risk": r_risk["image_risk"], "r1_flux_risk_max": r_risk["flux_risk_max"], "r1_centroid_pixels": r_risk["centroid_risk_pixels"], "r1_confusion": int(r_risk["confusion_risk"]), "oracle_policy_violation": violation, "r1_policy_violation": r_violation, "catastrophic_valid": catastrophic, "c_hidden_flux_ratio": math.nan, "c_hallucination": 0, "c_forced_source": 0, "r1_hidden_flux_ratio": math.nan, "r1_hallucination": 0, "r1_forced_source": 0})
            else:
                c_out = invalid_outcome(c_prediction, blend, isolated, row.query_state); r_out = invalid_outcome(r_prediction, blend, isolated, row.query_state); base.update({"c_image_risk": math.nan, "c_flux_risk_g": math.nan, "c_flux_risk_r": math.nan, "c_flux_risk_z": math.nan, "c_flux_risk_max": math.nan, "c_centroid_pixels": math.nan, "c_color_error": math.nan, "c_confusion": 0, "r1_image_risk": math.nan, "r1_flux_risk_max": math.nan, "r1_centroid_pixels": math.nan, "r1_confusion": 0, "oracle_policy_violation": math.inf, "r1_policy_violation": math.inf, "catastrophic_valid": 0, "c_hidden_flux_ratio": c_out["hidden_flux_ratio"], "c_hallucination": c_out["hallucination"], "c_forced_source": c_out["forced_source"], "c_source_preference": c_out["source_preference"], "r1_hidden_flux_ratio": r_out["hidden_flux_ratio"], "r1_hallucination": r_out["hallucination"], "r1_forced_source": r_out["forced_source"], "r1_source_preference": r_out["source_preference"]})
            base["task_aligned_failure"] = base["catastrophic_valid"] if row.query_state == "UNIQUE_VALID" else 1
            rows.append(base)
    per_sample = pd.DataFrame(rows); write_csv_fresh(run / "tables/development_per_sample.csv", per_sample)
    curves, bootstrap = risk_coverage(per_sample); write_csv_fresh(run / "tables/development_risk_coverage.csv", curves); write_csv_fresh(run / "tables/development_valid_operating_points.csv", bootstrap)
    macro = []
    for method, accept_column in (("condition_c_reconstruction_only", None), ("original_monolithic_R1", "r1_accept"), ("hierarchical_query_gate_only", "query_gate_accept"), ("complete_hierarchical_policy", "full_policy_accept")):
        accepted = np.ones(len(per_sample), dtype=bool) if accept_column is None else per_sample[accept_column].to_numpy(dtype=int).astype(bool)
        for state, group_indices in per_sample.groupby("query_state").groups.items():
            mask = np.zeros(len(per_sample), dtype=bool); mask[list(group_indices)] = True; selected = mask & accepted; macro.append({"method": method, "query_state": state, "samples": int(mask.sum()), "accepted": int(selected.sum()), "coverage": float(selected.sum() / mask.sum()), "catastrophic_rate_accepted": float(per_sample.loc[selected, "catastrophic_valid"].mean()) if selected.any() and state == "UNIQUE_VALID" else math.nan, "null_false_accept_rate": float(selected.sum() / mask.sum()) if state == "NULL" else math.nan, "ambiguous_false_accept_rate": float(selected.sum() / mask.sum()) if state == "AMBIGUOUS" else math.nan, "exposed_hallucination_rate": float((selected & per_sample.c_hallucination.eq(1).to_numpy()).sum() / mask.sum()) if state == "NULL" else math.nan, "exposed_forced_source_rate": float((selected & per_sample.c_forced_source.eq(1).to_numpy()).sum() / mask.sum()) if state == "AMBIGUOUS" else math.nan})
    write_csv_fresh(run / "tables/development_metrics_macro.csv", pd.DataFrame(macro)); make_gallery(run, per_sample)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("random_rejection", "original_monolithic_R1", "hierarchical_query_gate", "complete_hierarchical_policy", "oracle_risk_reference"):
        data = curves[(curves.method == method) & (curves.scope == "unique_valid")]; ax.plot(data.realized_coverage, data.accepted_case_risk, marker="o", label=method)
    ax.set_xlabel("valid-scene coverage"); ax.set_ylabel("catastrophic valid failure rate"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(run / "figures/class_conditional_valid_risk_coverage.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for state, style in (("NULL", "-"), ("AMBIGUOUS", "--")):
        subset = per_sample[per_sample.query_state == state].sort_values("recoverability_margin", ascending=False); ax.plot(np.arange(1, len(subset) + 1) / len(subset), np.arange(1, len(subset) + 1) / len(per_sample), style, label=state)
    ax.set_xlabel("within-class accepted fraction by hierarchy ranking"); ax.set_ylabel("overall false-accept contribution"); ax.legend(); fig.tight_layout(); fig.savefig(run / "figures/null_ambiguous_false_accept_curves.png", dpi=180); plt.close(fig)
    write_json_fresh(run / "logs/development_evaluation_complete.json", {"status": "PASS", "evaluation_count": 1, "completed_at_unix": time.time(), "development_manifest_sha256": checksum["manifest_sha256"], "policy_freeze_sha256": sha256_file(freeze_path), "condition_c_checkpoint_before": condition_hash, "condition_c_checkpoint_after": sha256_file(CONDITION_C), "r1_checkpoint_before": r1_hash, "r1_checkpoint_after": sha256_file(R1_CHECKPOINT), "sample_alignment": True, "threshold_retuning": False, "lockbox_used": False})
    print(json.dumps({"development_scenes": len(per_sample), "full_policy_accepts": int(full_accept.sum()), "query_gate_accepts": int(query_pass.sum()), "condition_c_hash_unchanged": condition_hash == sha256_file(CONDITION_C)}, sort_keys=True))


if __name__ == "__main__": main()
