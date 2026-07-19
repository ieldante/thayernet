#!/usr/bin/env python3
"""Evaluate the frozen posterior/decoder sufficiency gate before any flow work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_probabilistic_unet_pre_atlas import load_model, prompts, require_mps  # noqa: E402
from src.competing_hypotheses import scientific_distance  # noqa: E402
from src.models_probabilistic_unet import reparameterize  # noqa: E402


PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
ATLAS_NOISE = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/manifests/fixed_noise_contract.json"
K = 32
SEED = 2026078401
BATCH_SIZE = 8
ORDINARY_SCENES = 256
MEAN_PSF_FWHM_PIXEL = float(np.mean([0.86, 0.81, 0.77]) / 0.2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_validation() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        row for row in read_csv(PU / "manifests/probabilistic_unet_scene_definitions.csv")
        if row["partition"] == "validation"
    ]
    with h5py.File(PU / "manifests/probabilistic_unet_validation_scenes.h5", "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("authoritative validation arrays incomplete")
        blend = np.asarray(handle["blend"], dtype=np.float32)
        isolated = np.asarray(handle["isolated"], dtype=np.float32)
        xy = np.asarray(handle["xy"], dtype=np.float64)
    if len(rows) != len(blend) or isolated.shape != (len(rows), 2, 3, 60, 60):
        raise RuntimeError("validation manifest/array mismatch")
    return rows, blend, isolated, xy


def score_candidates(observed: np.ndarray, layers: np.ndarray, sky: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recomposed = layers[:, :, :3] + layers[:, :, 3:]
    variance = np.maximum(recomposed + sky[None, None, :, None, None], 1.0)
    residual = observed[:, None] - recomposed
    whitened_squared = residual**2 / variance
    global_score = whitened_squared.mean(axis=(2, 3, 4))
    band_score = whitened_squared.mean(axis=(3, 4))
    denominator = np.maximum(np.abs(observed).sum(axis=(1, 2, 3)), np.finfo(np.float64).eps)
    flux_score = residual.sum(axis=(2, 3, 4)) / denominator[:, None]
    return global_score, band_score, flux_score


def plausible_mask(observed: np.ndarray, layers: np.ndarray, sky: np.ndarray, thresholds: dict[str, object]) -> np.ndarray:
    global_score, band_score, flux_score = score_candidates(observed, layers, sky)
    return (
        (global_score <= float(thresholds["global_chi_square_mean"]))
        & np.all(band_score <= np.asarray(thresholds["per_band_chi_square_mean"])[None, None, :], axis=2)
        & (np.abs(flux_score) <= float(thresholds["absolute_relative_flux_residual"]))
    )


def sample_latents(mean: torch.Tensor, log_variance: torch.Tensor, epsilon: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(epsilon)).to(mean.device)
    return reparameterize(
        mean[:, None].expand(-1, K, -1),
        log_variance[:, None].expand(-1, K, -1),
        epsilon=value,
    )


def decode(model: torch.nn.Module, blend: torch.Tensor, prompt: torch.Tensor, latent: torch.Tensor) -> np.ndarray:
    batch = len(blend)
    output = model.decode(
        blend[:, None].expand(-1, K, -1, -1, -1).reshape(batch * K, 3, 60, 60),
        prompt[:, None].expand(-1, K, -1, -1, -1).reshape(batch * K, 1, 60, 60),
        latent.reshape(batch * K, 8),
    )
    return output.reshape(batch, K, 6, 60, 60).cpu().numpy()


def scientific_distances(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.asarray([
        scientific_distance(sample, truth, mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized
        for sample in samples
    ], dtype=np.float64)


def summarize_scene(
    *,
    scene: dict[str, str],
    evaluation: str,
    samples_physical: np.ndarray,
    samples_normalized: np.ndarray,
    observed_physical: np.ndarray,
    target_physical: np.ndarray,
    target_normalized: np.ndarray,
    comparison_normalized: np.ndarray,
    sky: np.ndarray,
    thresholds: dict[str, object],
    alternate_physical: np.ndarray | None,
) -> dict[str, object]:
    mask = plausible_mask(observed_physical[None], samples_physical[None], sky, thresholds)[0]
    requested_physical = samples_physical[:, :3]
    requested_normalized = samples_normalized[:, :3]
    target_distances = scientific_distances(requested_physical, target_physical)
    alternate_distances = None if alternate_physical is None else scientific_distances(requested_physical, alternate_physical)
    mse_target = ((requested_normalized - target_normalized[None]) ** 2).mean(axis=(1, 2, 3))
    mse_comparison = ((requested_normalized - comparison_normalized[None]) ** 2).mean(axis=(1, 2, 3))
    applicable = target_distances[mask]
    alternate_applicable = np.asarray([], dtype=np.float64) if alternate_distances is None else alternate_distances[mask]
    return {
        "scene_id": scene["scene_id"],
        "kind": scene["kind"],
        "near_collision_pair_id": scene["near_collision_pair_id"],
        "near_collision_pair_side": scene["near_collision_pair_side"],
        "source_a_group": scene["source_a_group"],
        "source_b_group": scene["source_b_group"],
        "evaluation": evaluation,
        "sample_count": K,
        "plausible_sample_count": int(mask.sum()),
        "forward_consistent_fraction": float(mask.mean()),
        "target_truth_coverage": bool(len(applicable) and float(np.min(applicable)) <= 1.0),
        "alternate_truth_coverage": bool(len(alternate_applicable) and float(np.min(alternate_applicable)) <= 1.0),
        "best_target_scientific_distance": float(np.min(target_distances)),
        "mean_target_scientific_distance": float(np.mean(target_distances)),
        "best_alternate_scientific_distance": math.nan if alternate_distances is None else float(np.min(alternate_distances)),
        "mean_alternate_scientific_distance": math.nan if alternate_distances is None else float(np.mean(alternate_distances)),
        "best_of_k_requested_mse_normalized": float(np.min(mse_target)),
        "mean_requested_mse_normalized": float(np.mean(mse_target)),
        "source_identity_fraction": float(np.mean(mse_target < mse_comparison)),
        "scientific_divergence_from_comparison": math.nan if alternate_distances is None else float(np.mean(alternate_distances)),
        "finite": bool(
            np.all(np.isfinite(samples_physical))
            and np.all(np.isfinite(target_distances))
            and np.all(np.isfinite(mse_target))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    prerequisite = json.loads((run_dir / "logs/parts_b_c_complete.json").read_text())
    if prerequisite["status"] != "PASS" or not prerequisite["posterior_decoder_evaluation_authorized"]:
        raise RuntimeError("Parts B/C block posterior/decoder evaluation")
    prereg_path = run_dir / "preregistration/posterior_decoder_sufficiency.md"
    prereg_record = json.loads((run_dir / "preregistration/posterior_decoder_sufficiency_record.json").read_text())
    if prereg_record["sha256"] != sha256_file(prereg_path):
        raise RuntimeError("posterior/decoder preregistration changed")
    if any(any(run_dir.joinpath(name).iterdir()) for name in ("flow_models", "checkpoints", "prior_samples")):
        raise RuntimeError("flow artifacts exist before posterior/decoder gate")

    device = require_mps()
    model = load_model(PU, device)
    checkpoint_hash_before = sha256_file(PU / "checkpoints/thayer_pu_best.pth")
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    output_scales = np.tile(scales, 2)[None, None, :, None, None]
    sky = np.asarray(json.loads(ATLAS_NOISE.read_text())["sky_electrons_grz"], dtype=np.float64)
    thresholds = json.loads((PU / "manifests/forward_consistency_thresholds.json").read_text())
    rows, blends_physical, isolated_physical, xy = load_validation()
    index_by_pair_side = {
        (row["near_collision_pair_id"], row["near_collision_pair_side"]): index
        for index, row in enumerate(rows) if row["kind"] == "near_collision"
    }
    ordinary_indices = [index for index, row in enumerate(rows) if row["kind"] == "ordinary"][:ORDINARY_SCENES]
    pair_rows = [row for row in read_csv(PU / "tables/non_atlas_near_collision_pair_manifest.csv") if row["partition"] == "validation"]
    if len(ordinary_indices) != ORDINARY_SCENES or len(pair_rows) != 250:
        raise RuntimeError("frozen Part-D sample cardinality mismatch")
    for pair in pair_rows:
        groups = [pair[name] for name in ("left_source_a_group", "left_source_b_group", "right_source_a_group", "right_source_b_group")]
        if len(set(groups)) != 4 or pair["four_groups_disjoint"] != "True":
            raise RuntimeError(f"near-collision source groups not disjoint: {pair['near_collision_pair_id']}")

    rng = np.random.default_rng(SEED)
    scene_rows: list[dict[str, object]] = []
    ordinary_latents: list[np.ndarray] = []
    ordinary_ids: list[str] = []
    near_latents: list[np.ndarray] = []
    near_ids: list[str] = []
    started = time.time()

    with torch.no_grad():
        for start in range(0, len(ordinary_indices), BATCH_SIZE):
            indices = ordinary_indices[start:start + BATCH_SIZE]
            blend_norm_np = blends_physical[indices] / scales[None, :, None, None]
            isolated_norm_np = isolated_physical[indices] / scales[None, None, :, None, None]
            prompt_a_np, prompt_b_np = prompts(xy[indices])
            blend = torch.from_numpy(np.ascontiguousarray(blend_norm_np)).to(device)
            source_a = torch.from_numpy(np.ascontiguousarray(isolated_norm_np[:, 0])).to(device)
            source_b = torch.from_numpy(np.ascontiguousarray(isolated_norm_np[:, 1])).to(device)
            prompt_a = torch.from_numpy(np.ascontiguousarray(prompt_a_np)).to(device)
            prompt_b = torch.from_numpy(np.ascontiguousarray(prompt_b_np)).to(device)
            mean, log_variance = model.encode_posterior(blend, source_a, source_b)
            epsilon = rng.standard_normal((len(indices), K, 8)).astype(np.float32)
            latent = sample_latents(mean, log_variance, epsilon)
            output_a_norm = decode(model, blend, prompt_a, latent)
            output_b_norm = decode(model, blend, prompt_b, latent)
            output_a_physical = output_a_norm * output_scales
            output_b_physical = output_b_norm * output_scales
            for local, index in enumerate(indices):
                scene = rows[index]
                scene_rows.append(summarize_scene(
                    scene=scene, evaluation="ordinary_own_prompt_a",
                    samples_physical=output_a_physical[local], samples_normalized=output_a_norm[local],
                    observed_physical=blends_physical[index], target_physical=isolated_physical[index, 0],
                    target_normalized=isolated_norm_np[local, 0], comparison_normalized=isolated_norm_np[local, 1],
                    sky=sky, thresholds=thresholds, alternate_physical=None,
                ))
                scene_rows.append(summarize_scene(
                    scene=scene, evaluation="ordinary_own_prompt_b",
                    samples_physical=output_b_physical[local], samples_normalized=output_b_norm[local],
                    observed_physical=blends_physical[index], target_physical=isolated_physical[index, 1],
                    target_normalized=isolated_norm_np[local, 1], comparison_normalized=isolated_norm_np[local, 0],
                    sky=sky, thresholds=thresholds, alternate_physical=None,
                ))
                ordinary_latents.append(latent[local].cpu().numpy())
                ordinary_ids.append(scene["scene_id"])
            print(json.dumps({"phase": "ordinary", "completed": min(start + BATCH_SIZE, len(ordinary_indices)), "total": len(ordinary_indices), "elapsed_seconds": time.time() - started}), flush=True)

        for start in range(0, len(pair_rows), BATCH_SIZE // 2):
            batch_pairs = pair_rows[start:start + BATCH_SIZE // 2]
            indices: list[int] = []
            for pair in batch_pairs:
                pair_id = pair["near_collision_pair_id"]
                indices.extend((index_by_pair_side[(pair_id, "left")], index_by_pair_side[(pair_id, "right")]))
            blend_norm_np = blends_physical[indices] / scales[None, :, None, None]
            isolated_norm_np = isolated_physical[indices] / scales[None, None, :, None, None]
            prompt_a_np, _ = prompts(xy[indices])
            blend = torch.from_numpy(np.ascontiguousarray(blend_norm_np)).to(device)
            source_a = torch.from_numpy(np.ascontiguousarray(isolated_norm_np[:, 0])).to(device)
            source_b = torch.from_numpy(np.ascontiguousarray(isolated_norm_np[:, 1])).to(device)
            prompt_a = torch.from_numpy(np.ascontiguousarray(prompt_a_np)).to(device)
            mean, log_variance = model.encode_posterior(blend, source_a, source_b)
            epsilon = rng.standard_normal((len(indices), K, 8)).astype(np.float32)
            latent = sample_latents(mean, log_variance, epsilon)
            own_norm = decode(model, blend, prompt_a, latent)
            swap_order = np.arange(len(indices)).reshape(-1, 2)[:, ::-1].reshape(-1)
            cross_latent = latent[torch.as_tensor(swap_order, device=device)]
            cross_norm = decode(model, blend, prompt_a, cross_latent)
            own_physical = own_norm * output_scales
            cross_physical = cross_norm * output_scales
            for local, index in enumerate(indices):
                alternate_local = int(swap_order[local])
                alternate_index = indices[alternate_local]
                scene = rows[index]
                scene_rows.append(summarize_scene(
                    scene=scene, evaluation="near_collision_own_posterior",
                    samples_physical=own_physical[local], samples_normalized=own_norm[local],
                    observed_physical=blends_physical[index], target_physical=isolated_physical[index, 0],
                    target_normalized=isolated_norm_np[local, 0], comparison_normalized=isolated_norm_np[local, 1],
                    sky=sky, thresholds=thresholds, alternate_physical=isolated_physical[alternate_index, 0],
                ))
                scene_rows.append(summarize_scene(
                    scene=scene, evaluation="near_collision_cross_posterior",
                    samples_physical=cross_physical[local], samples_normalized=cross_norm[local],
                    observed_physical=blends_physical[index], target_physical=isolated_physical[alternate_index, 0],
                    target_normalized=isolated_norm_np[alternate_local, 0], comparison_normalized=isolated_norm_np[local, 0],
                    sky=sky, thresholds=thresholds, alternate_physical=isolated_physical[index, 0],
                ))
                near_latents.append(latent[local].cpu().numpy())
                near_ids.append(scene["scene_id"])
            print(json.dumps({"phase": "near_collision_pairs", "completed": min(start + BATCH_SIZE // 2, len(pair_rows)), "total": len(pair_rows), "elapsed_seconds": time.time() - started}), flush=True)

    write_csv_fresh(run_dir / "tables/posterior_decoder_sufficiency_per_scene.csv", scene_rows)
    latent_path = run_dir / "posterior_samples/part_d_latents_k32.h5"
    with h5py.File(latent_path, "x") as handle:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("ordinary_scene_id", data=np.asarray(ordinary_ids, dtype=object), dtype=string_dtype)
        handle.create_dataset("ordinary_posterior_latent", data=np.asarray(ordinary_latents, dtype=np.float32), compression="lzf")
        handle.create_dataset("near_scene_id", data=np.asarray(near_ids, dtype=object), dtype=string_dtype)
        handle.create_dataset("near_posterior_latent", data=np.asarray(near_latents, dtype=np.float32), compression="lzf")
        handle.attrs["complete"] = True
        handle.attrs["checkpoint_sha256"] = checkpoint_hash_before
        handle.attrs["preregistration_sha256"] = prereg_record["sha256"]
        handle.attrs["seed"] = SEED
        handle.attrs["k"] = K

    subsets = {
        "ordinary": [row for row in scene_rows if row["evaluation"].startswith("ordinary_own")],
        "near_own": [row for row in scene_rows if row["evaluation"] == "near_collision_own_posterior"],
        "near_cross": [row for row in scene_rows if row["evaluation"] == "near_collision_cross_posterior"],
    }
    summary_rows: list[dict[str, object]] = []
    for name, values in subsets.items():
        summary_rows.append({
            "subset": name,
            "evaluation_units": len(values),
            "posterior_samples_per_unit": K,
            "target_truth_coverage_rate": float(np.mean([bool(row["target_truth_coverage"]) for row in values])),
            "alternate_truth_coverage_rate": float(np.mean([bool(row["alternate_truth_coverage"]) for row in values])),
            "mean_forward_consistent_fraction": float(np.mean([float(row["forward_consistent_fraction"]) for row in values])),
            "mean_source_identity_fraction": float(np.mean([float(row["source_identity_fraction"]) for row in values])),
            "median_best_target_scientific_distance": float(np.median([float(row["best_target_scientific_distance"]) for row in values])),
            "mean_best_of_k_requested_mse_normalized": float(np.mean([float(row["best_of_k_requested_mse_normalized"]) for row in values])),
            "mean_requested_mse_normalized": float(np.mean([float(row["mean_requested_mse_normalized"]) for row in values])),
            "all_finite": bool(all(bool(row["finite"]) for row in values)),
        })
    write_csv_fresh(run_dir / "tables/posterior_decoder_sufficiency_summary.csv", summary_rows)
    summary = {row["subset"]: row for row in summary_rows}
    gates = [
        {"gate": "ordinary_posterior_own_truth_coverage", "threshold": ">=0.70", "observed": summary["ordinary"]["target_truth_coverage_rate"], "pass": float(summary["ordinary"]["target_truth_coverage_rate"]) >= 0.70},
        {"gate": "near_own_posterior_own_truth_coverage", "threshold": ">=0.70", "observed": summary["near_own"]["target_truth_coverage_rate"], "pass": float(summary["near_own"]["target_truth_coverage_rate"]) >= 0.70},
        {"gate": "near_cross_posterior_alternate_truth_coverage", "threshold": ">=0.30", "observed": summary["near_cross"]["target_truth_coverage_rate"], "pass": float(summary["near_cross"]["target_truth_coverage_rate"]) >= 0.30},
        {"gate": "ordinary_forward_consistent_fraction", "threshold": ">=0.50", "observed": summary["ordinary"]["mean_forward_consistent_fraction"], "pass": float(summary["ordinary"]["mean_forward_consistent_fraction"]) >= 0.50},
        {"gate": "near_own_forward_consistent_fraction", "threshold": ">=0.50", "observed": summary["near_own"]["mean_forward_consistent_fraction"], "pass": float(summary["near_own"]["mean_forward_consistent_fraction"]) >= 0.50},
        {"gate": "near_cross_forward_consistent_fraction", "threshold": ">=0.50", "observed": summary["near_cross"]["mean_forward_consistent_fraction"], "pass": float(summary["near_cross"]["mean_forward_consistent_fraction"]) >= 0.50},
        {"gate": "ordinary_source_identity", "threshold": ">=0.70", "observed": summary["ordinary"]["mean_source_identity_fraction"], "pass": float(summary["ordinary"]["mean_source_identity_fraction"]) >= 0.70},
        {"gate": "near_own_source_identity", "threshold": ">=0.70", "observed": summary["near_own"]["mean_source_identity_fraction"], "pass": float(summary["near_own"]["mean_source_identity_fraction"]) >= 0.70},
        {"gate": "near_cross_alternate_identity", "threshold": ">=0.70", "observed": summary["near_cross"]["mean_source_identity_fraction"], "pass": float(summary["near_cross"]["mean_source_identity_fraction"]) >= 0.70},
        {"gate": "finite_complete_disjoint_evaluation", "threshold": "all finite; 250 pairs; disjoint groups", "observed": f"finite={all(bool(row['all_finite']) for row in summary_rows)};pairs={len(pair_rows)};disjoint=True", "pass": bool(all(bool(row["all_finite"]) for row in summary_rows) and len(pair_rows) == 250)},
    ]
    write_csv_fresh(run_dir / "tables/posterior_decoder_sufficiency_gates.csv", gates)
    passed = all(bool(row["pass"]) for row in gates)
    if sha256_file(PU / "checkpoints/thayer_pu_best.pth") != checkpoint_hash_before:
        raise RuntimeError("Thayer-PU checkpoint changed during Part D")

    decision = "PASS — FLOW PRIOR CORRECTION JUSTIFIED" if passed else "FAIL — DECODER/POSTERIOR INSUFFICIENT; FLOW PROHIBITED"
    next_experiment = (
        "Proceed to the separately preregistered compact conditional flow-prior fit."
        if passed else
        "Preregister one ambiguity-set decoder-training experiment that presents both non-Atlas near-collision decompositions under each observationally equivalent condition while preserving prompt identity and forward consistency."
    )
    report = f"""# Posterior/decoder sufficiency gate

Decision: **{decision}**.

- Ordinary posterior own-truth coverage: {float(summary['ordinary']['target_truth_coverage_rate']):.6f}.
- Near-collision own-posterior own-truth coverage: {float(summary['near_own']['target_truth_coverage_rate']):.6f}.
- Near-collision cross-decode alternate-truth coverage: {float(summary['near_cross']['target_truth_coverage_rate']):.6f}.
- Forward-consistent fractions ordinary / near-own / near-cross: {float(summary['ordinary']['mean_forward_consistent_fraction']):.6f} / {float(summary['near_own']['mean_forward_consistent_fraction']):.6f} / {float(summary['near_cross']['mean_forward_consistent_fraction']):.6f}.
- Source identity ordinary / near-own / near-cross-alternate: {float(summary['ordinary']['mean_source_identity_fraction']):.6f} / {float(summary['near_own']['mean_source_identity_fraction']):.6f} / {float(summary['near_cross']['mean_source_identity_fraction']):.6f}.
- Median best scientific distance ordinary / near-own / near-cross: {float(summary['ordinary']['median_best_target_scientific_distance']):.6f} / {float(summary['near_own']['median_best_target_scientific_distance']):.6f} / {float(summary['near_cross']['median_best_target_scientific_distance']):.6f}.

The evaluation used K={K}, the frozen seed {SEED}, 256 ordinary scenes, every
one of 250 validation near-collision pairs, the unchanged coverage metric, and
the unchanged forward-consistency thresholds. All four source groups per pair
were disjoint. Neural work ran on MPS with fallback prohibited. Posterior samples
are diagnostic only.

Exact next experiment: {next_experiment}

Atlas inference, unauthorized development data, and the final lockbox were not
accessed. No flow module, checkpoint, prior sample, or training objective was
created. The Thayer-PU checkpoint remained byte-identical.
"""
    write_text_fresh(run_dir / "diagnostics/posterior_decoder_sufficiency.md", report)
    write_json_fresh(run_dir / "logs/posterior_decoder_sufficiency_complete.json", {
        "status": "PASS" if passed else "FAIL",
        "scientific_decision": decision,
        "flow_implementation_authorized": passed,
        "atlas_evaluation_authorized": False,
        "ordinary_coverage": summary["ordinary"]["target_truth_coverage_rate"],
        "near_own_coverage": summary["near_own"]["target_truth_coverage_rate"],
        "near_cross_alternate_coverage": summary["near_cross"]["target_truth_coverage_rate"],
        "checkpoint_sha256_before_after": checkpoint_hash_before,
        "posterior_latents_sha256": sha256_file(latent_path),
        "preregistration_sha256": prereg_record["sha256"],
        "script_sha256": sha256_file(Path(__file__)),
        "runtime_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "mps", "mps_fallback": False,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
        "exact_next_experiment": next_experiment,
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL", "flow_implementation_authorized": passed, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
