#!/usr/bin/env python3
"""Run sequential latent-use and promptability gates before any Atlas access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import CompactSelectNet, gaussian_prompt_numpy  # noqa: E402
from src.canonical_tensor_hash import canonical_tensor_sha256  # noqa: E402
from src.competing_hypotheses import source_measurements  # noqa: E402
from src.models_probabilistic_unet import (  # noqa: E402
    ThayerProbabilisticUNet,
    gaussian_kl_per_dimension,
    reparameterize,
    swap_decomposition,
)


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
K = 16
BATCH_SIZE = 8
LATENT_AUDIT_SCENES = 256
PRIOR_EPSILON_SEED = 2026078201
POSTERIOR_EPSILON_SEED = 2026078202


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


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("MPS fallback is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def require_run(path: Path, phase: str) -> Path:
    run_dir = path.resolve()
    training = json.loads((run_dir / "logs/training_complete.json").read_text())
    if training["status"] != "PASS" or training["epochs_completed"] != 30 or training["atlas_evaluation_count"] != 0:
        raise RuntimeError("training did not complete cleanly")
    if phase == "promptability":
        latent = json.loads((run_dir / "logs/latent_use_audit_complete.json").read_text())
        if latent["status"] != "PASS" or not latent["promptability_authorized"]:
            raise RuntimeError("latent-use gate blocks promptability")
    return run_dir


def load_model(run_dir: Path, device: torch.device) -> ThayerProbabilisticUNet:
    payload = torch.load(run_dir / "checkpoints/thayer_pu_best.pth", map_location="cpu", weights_only=False)
    if payload["model_family"] != "THAYER_PU" or payload["selection"] != "minimum validation frozen total objective":
        raise RuntimeError("selected checkpoint contract mismatch")
    model = ThayerProbabilisticUNet().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def validation_arrays(run_dir: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        row for row in read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv")
        if row["partition"] == "validation"
    ]
    with h5py.File(run_dir / "manifests/probabilistic_unet_validation_scenes.h5", "r") as handle:
        if not bool(handle.attrs["complete"]) or len(handle["blend"]) != len(rows):
            raise RuntimeError("validation arrays incomplete")
        return rows, np.asarray(handle["blend"], dtype=np.float32), np.asarray(handle["isolated"], dtype=np.float32), np.asarray(handle["xy"], dtype=np.float64)


def prompts(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.stack([gaussian_prompt_numpy(float(value[0, 0]), float(value[0, 1])) for value in xy])[:, None]
    b = np.stack([gaussian_prompt_numpy(float(value[1, 0]), float(value[1, 1])) for value in xy])[:, None]
    return a.astype(np.float32), b.astype(np.float32)


def sample_outputs(
    model: ThayerProbabilisticUNet,
    blend: torch.Tensor,
    prompt: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    # epsilon is (B,K,Z), output is (B,K,6,H,W).
    batch_size, sample_count = epsilon.shape[:2]
    latent = reparameterize(
        mean[:, None].expand(-1, sample_count, -1),
        log_variance[:, None].expand(-1, sample_count, -1),
        epsilon=epsilon,
    )
    output = model.decode(
        blend[:, None].expand(-1, sample_count, -1, -1, -1).reshape(batch_size * sample_count, 3, 60, 60),
        prompt[:, None].expand(-1, sample_count, -1, -1, -1).reshape(batch_size * sample_count, 1, 60, 60),
        latent.reshape(batch_size * sample_count, 8),
    )
    return output.reshape(batch_size, sample_count, 6, 60, 60)


def latent_audit(run_dir: Path) -> None:
    device = require_mps()
    model = load_model(run_dir, device)
    rows, blends_physical, isolated_physical, xy = validation_arrays(run_dir)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    ordinary = [index for index, row in enumerate(rows) if row["kind"] == "ordinary"][: LATENT_AUDIT_SCENES // 2]
    near = [index for index, row in enumerate(rows) if row["kind"] == "near_collision"][: LATENT_AUDIT_SCENES // 2]
    indices = np.asarray(ordinary + near, dtype=int)
    blend_all = blends_physical[indices] / scales[None, :, None, None]
    isolated_all = isolated_physical[indices] / scales[None, None, :, None, None]
    prompt_a_all, prompt_b_all = prompts(xy[indices])
    rng_prior = np.random.default_rng(PRIOR_EPSILON_SEED)
    rng_posterior = np.random.default_rng(POSTERIOR_EPSILON_SEED)
    kl_values = []
    prior_means, prior_stds, posterior_means, posterior_stds = [], [], [], []
    prior_pairwise, posterior_pairwise = [], []
    latent_sensitivity, prompt_sensitivity = [], []
    first_example = None
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(indices))
            blend = torch.from_numpy(np.ascontiguousarray(blend_all[start:stop])).to(device)
            source_a = torch.from_numpy(np.ascontiguousarray(isolated_all[start:stop, 0])).to(device)
            source_b = torch.from_numpy(np.ascontiguousarray(isolated_all[start:stop, 1])).to(device)
            prompt_a = torch.from_numpy(np.ascontiguousarray(prompt_a_all[start:stop])).to(device)
            prompt_b = torch.from_numpy(np.ascontiguousarray(prompt_b_all[start:stop])).to(device)
            prior_mean, prior_log_variance = model.encode_prior(blend)
            posterior_mean, posterior_log_variance = model.encode_posterior(blend, source_a, source_b)
            kl = gaussian_kl_per_dimension(posterior_mean, posterior_log_variance, prior_mean, prior_log_variance)
            eps_prior = torch.from_numpy(rng_prior.standard_normal((len(blend), K, 8)).astype(np.float32)).to(device)
            eps_posterior = torch.from_numpy(rng_posterior.standard_normal((len(blend), K, 8)).astype(np.float32)).to(device)
            prior_output = sample_outputs(model, blend, prompt_a, prior_mean, prior_log_variance, eps_prior)
            posterior_output = sample_outputs(model, blend, prompt_a, posterior_mean, posterior_log_variance, eps_posterior)
            for sample_set, destination in ((prior_output, prior_pairwise), (posterior_output, posterior_pairwise)):
                requested = sample_set[:, :, :3]
                distances = []
                for left, right in itertools.combinations(range(K), 2):
                    distances.append((requested[:, left] - requested[:, right]).square().mean(dim=(1, 2, 3)))
                destination.extend(torch.stack(distances, dim=1).mean(dim=1).cpu().numpy().tolist())
            negative = model.decode(blend, prompt_a, -torch.ones((len(blend), 8), device=device))[:, :3]
            positive = model.decode(blend, prompt_a, torch.ones((len(blend), 8), device=device))[:, :3]
            mean_a = model.decode(blend, prompt_a, prior_mean)[:, :3]
            mean_b = model.decode(blend, prompt_b, prior_mean)[:, :3]
            truth_diameter = (source_a - source_b).abs().mean(dim=(1, 2, 3)).clamp_min(1e-12)
            latent_sensitivity.extend(((positive - negative).abs().mean(dim=(1, 2, 3)) / truth_diameter).cpu().numpy().tolist())
            prompt_sensitivity.extend(((mean_a - mean_b).abs().mean(dim=(1, 2, 3)) / truth_diameter).cpu().numpy().tolist())
            kl_values.append(kl.cpu().numpy())
            prior_means.append(prior_mean.cpu().numpy())
            prior_stds.append(torch.exp(0.5 * prior_log_variance).cpu().numpy())
            posterior_means.append(posterior_mean.cpu().numpy())
            posterior_stds.append(torch.exp(0.5 * posterior_log_variance).cpu().numpy())
            if first_example is None:
                interpolation = []
                for value in np.linspace(-2, 2, 9):
                    z = prior_mean[:1].clone()
                    z[:, 0] = prior_mean[:1, 0] + float(value) * torch.exp(0.5 * prior_log_variance[:1, 0])
                    interpolation.append(model.decode(blend[:1], prompt_a[:1], z)[0, :3].cpu().numpy() * scales[:, None, None])
                first_example = np.stack(interpolation)
    kl_array = np.concatenate(kl_values)
    prior_std = np.concatenate(prior_stds)
    posterior_std = np.concatenate(posterior_stds)
    prior_mean_array = np.concatenate(prior_means)
    posterior_mean_array = np.concatenate(posterior_means)
    active = int(np.sum(kl_array.mean(axis=0) >= 0.02))
    median_prior_std = float(np.median(prior_std))
    decoder_sensitivity = float(np.mean(latent_sensitivity))
    pairwise_prior = float(np.mean(prior_pairwise))
    gates = [
        {"gate": "active_latent_dimensions", "threshold": ">=2", "observed": active, "pass": active >= 2},
        {"gate": "prior_std_finite_nondegenerate", "threshold": "all per-dim medians in [0.05,3.0]", "observed": ";".join(f"{value:.6g}" for value in np.median(prior_std, axis=0)), "pass": bool(np.all((np.median(prior_std, axis=0) >= 0.05) & (np.median(prior_std, axis=0) <= 3.0)))},
        {"gate": "decoder_latent_sensitivity", "threshold": ">=0.02 truth diameter", "observed": decoder_sensitivity, "pass": decoder_sensitivity >= 0.02},
        {"gate": "prior_pairwise_distance", "threshold": ">0", "observed": pairwise_prior, "pass": pairwise_prior > 0.0},
    ]
    passed = all(bool(row["pass"]) for row in gates)
    write_csv_fresh(run_dir / "tables/latent_use_gates.csv", gates)
    write_csv_fresh(run_dir / "tables/latent_kl_per_dimension.csv", [
        {
            "latent_dimension": dimension,
            "mean_raw_kl": float(kl_array[:, dimension].mean()),
            "active": bool(kl_array[:, dimension].mean() >= 0.02),
            "prior_mean": float(prior_mean_array[:, dimension].mean()),
            "prior_std_median": float(np.median(prior_std[:, dimension])),
            "posterior_mean": float(posterior_mean_array[:, dimension].mean()),
            "posterior_std_median": float(np.median(posterior_std[:, dimension])),
        }
        for dimension in range(8)
    ])
    summary = [{
        "audit_scene_count": len(indices), "total_raw_kl": float(kl_array.sum(axis=1).mean()),
        "active_latent_dimensions": active, "prior_mean_global": float(prior_mean_array.mean()),
        "prior_std_median": median_prior_std, "posterior_mean_global": float(posterior_mean_array.mean()),
        "posterior_std_median": float(np.median(posterior_std)),
        "prior_posterior_mean_distance": float(np.linalg.norm(prior_mean_array - posterior_mean_array, axis=1).mean()),
        "prior_pairwise_sample_mse": pairwise_prior,
        "posterior_pairwise_sample_mse": float(np.mean(posterior_pairwise)),
        "decoder_latent_sensitivity_ratio": decoder_sensitivity,
        "prompt_sensitivity_ratio": float(np.mean(prompt_sensitivity)),
        "status": "PASS" if passed else "FAIL",
    }]
    write_csv_fresh(run_dir / "tables/latent_use_summary.csv", summary)
    if first_example is None:
        raise RuntimeError("latent interpolation example missing")
    figure, axes = plt.subplots(1, 9, figsize=(18, 2.4), constrained_layout=True)
    scale = float(np.max(np.abs(first_example[:, 1])))
    for index, axis in enumerate(axes):
        axis.imshow(np.arcsinh(first_example[index, 1] / max(scale, 1e-12) * 20), origin="lower", cmap="coolwarm")
        axis.set_title(f"{np.linspace(-2, 2, 9)[index]:+.1f}σ")
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle("Thayer-PU latent dimension 0 interpolation (r band)")
    figure.savefig(run_dir / "example_grids/latent_interpolation_grid.png", dpi=170)
    plt.close(figure)
    report = f"""# Posterior-collapse and latent-use audit

Status: **{'PASS' if passed else 'FAIL — PROMPTABILITY AND ATLAS BLOCKED'}**.

- Audit scenes: {len(indices)} (128 ordinary, 128 non-Atlas near-collision).
- Total raw KL: {summary[0]['total_raw_kl']:.6f} nats.
- Active dimensions: {active}/8 (criterion mean raw KL >=0.02; gate >=2).
- Median prior/posterior std: {median_prior_std:.6f} / {summary[0]['posterior_std_median']:.6f}.
- Prior/posterior mean distance: {summary[0]['prior_posterior_mean_distance']:.6f}.
- Prior/posterior pairwise requested-sample MSE: {pairwise_prior:.6g} / {summary[0]['posterior_pairwise_sample_mse']:.6g}.
- Decoder z sensitivity: {decoder_sensitivity:.6f} truth diameters (gate >=0.02).
- Prompt sensitivity at prior mean: {summary[0]['prompt_sensitivity_ratio']:.6f} truth diameters.

Prior and posterior samples are reported separately. Posterior diversity is not
used as inference-time evidence. This audit accessed only the fresh non-Atlas
validation partition on MPS.
"""
    write_text_fresh(run_dir / "diagnostics/posterior_collapse_and_latent_use.md", report)
    write_json_fresh(run_dir / "logs/latent_use_audit_complete.json", {
        "status": "PASS" if passed else "FAIL", "promptability_authorized": passed,
        "active_latent_dimensions": active, "decoder_latent_sensitivity_ratio": decoder_sensitivity,
        "prior_pairwise_sample_mse": pairwise_prior, "audit_scene_count": len(indices),
        "runtime_seconds": time.time() - started, "device": "mps", "mps_fallback": False,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL", "gates": gates}, sort_keys=True))


def centroid_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    left = source_measurements(prediction).centroid_xy
    right = source_measurements(truth).centroid_xy
    return math.nan if left is None or right is None else float(np.linalg.norm(np.subtract(left, right)))


def promptability_audit(run_dir: Path) -> None:
    device = require_mps()
    model = load_model(run_dir, device)
    condition_payload = torch.load(CONDITION_C, map_location="cpu", weights_only=False)
    condition = CompactSelectNet(4).to(device)
    condition.load_state_dict(condition_payload["state_dict"], strict=True)
    condition.eval()
    for parameter in condition.parameters():
        parameter.requires_grad_(False)
    rows, blends_physical, isolated_physical, xy = validation_arrays(run_dir)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    blend_all = blends_physical / scales[None, :, None, None]
    isolated_all = isolated_physical / scales[None, None, :, None, None]
    prompt_a_all, prompt_b_all = prompts(xy)
    rng_prior = np.random.default_rng(PRIOR_EPSILON_SEED)
    rng_posterior = np.random.default_rng(POSTERIOR_EPSILON_SEED)
    scene_rows: list[dict[str, object]] = []
    sample_hash_rows: list[dict[str, object]] = []
    all_signed, band_success = [], [[], [], []]
    individual_success, best_success, majority_success, collapse_values = [], [], [], []
    prior_mse_values, prior_best_mse_values, posterior_mse_values, condition_mse_values = [], [], [], []
    posterior_identity_values = []
    flux_errors, centroid_errors, swap_consistency_values = [], [], []
    examples = []
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(rows))
            blend = torch.from_numpy(np.ascontiguousarray(blend_all[start:stop])).to(device)
            source_a = torch.from_numpy(np.ascontiguousarray(isolated_all[start:stop, 0])).to(device)
            source_b = torch.from_numpy(np.ascontiguousarray(isolated_all[start:stop, 1])).to(device)
            prompt_a = torch.from_numpy(np.ascontiguousarray(prompt_a_all[start:stop])).to(device)
            prompt_b = torch.from_numpy(np.ascontiguousarray(prompt_b_all[start:stop])).to(device)
            prior_mean, prior_log_variance = model.encode_prior(blend)
            eps_prior = torch.from_numpy(rng_prior.standard_normal((len(blend), K, 8)).astype(np.float32)).to(device)
            output_a = sample_outputs(model, blend, prompt_a, prior_mean, prior_log_variance, eps_prior)
            output_b = sample_outputs(model, blend, prompt_b, prior_mean, prior_log_variance, eps_prior)
            posterior_mean, posterior_log_variance = model.encode_posterior(blend, source_a, source_b)
            eps_posterior = torch.from_numpy(rng_posterior.standard_normal((len(blend), 8)).astype(np.float32)).to(device)
            posterior_z = reparameterize(posterior_mean, posterior_log_variance, epsilon=eps_posterior)
            posterior_pair = model.decode(
                torch.cat((blend, blend)), torch.cat((prompt_a, prompt_b)), torch.cat((posterior_z, posterior_z))
            )
            posterior_a, posterior_b = posterior_pair.chunk(2)
            condition_input = torch.cat((
                torch.cat((blend, prompt_a), dim=1), torch.cat((blend, prompt_b), dim=1)
            ))
            condition_output = condition(condition_input)
            condition_a, condition_b = condition_output.chunk(2)

            requested_a, requested_b = output_a[:, :, :3], output_b[:, :, :3]
            mse_a_own = (requested_a - source_a[:, None]).square().mean(dim=(2, 3, 4))
            mse_a_alt = (requested_a - source_b[:, None]).square().mean(dim=(2, 3, 4))
            mse_b_own = (requested_b - source_b[:, None]).square().mean(dim=(2, 3, 4))
            mse_b_alt = (requested_b - source_a[:, None]).square().mean(dim=(2, 3, 4))
            success_a, success_b = mse_a_own < mse_a_alt, mse_b_own < mse_b_alt
            truth_diameter = (source_a - source_b).abs().mean(dim=(1, 2, 3)).clamp_min(1e-12)
            collapse_ratio = (requested_a - requested_b).abs().mean(dim=(2, 3, 4)) / truth_diameter[:, None]
            swap_consistency = (output_a - swap_decomposition(output_b.reshape(-1, 6, 60, 60)).reshape_as(output_b)).square().mean(dim=(2, 3, 4))
            posterior_own = torch.stack((
                (posterior_a[:, :3] - source_a).square().mean(dim=(1, 2, 3)),
                (posterior_b[:, :3] - source_b).square().mean(dim=(1, 2, 3)),
            ), dim=1)
            posterior_alt = torch.stack((
                (posterior_a[:, :3] - source_b).square().mean(dim=(1, 2, 3)),
                (posterior_b[:, :3] - source_a).square().mean(dim=(1, 2, 3)),
            ), dim=1)
            condition_own = torch.stack((
                (condition_a - source_a).square().mean(dim=(1, 2, 3)),
                (condition_b - source_b).square().mean(dim=(1, 2, 3)),
            ), dim=1)
            for local in range(len(blend)):
                global_index = start + local
                successes_a = success_a[local].cpu().numpy()
                successes_b = success_b[local].cpu().numpy()
                signed = torch.cat((mse_a_own[local] - mse_a_alt[local], mse_b_own[local] - mse_b_alt[local])).cpu().numpy()
                paired_majority = bool(successes_a.sum() > K / 2 and successes_b.sum() > K / 2)
                query_best = [bool(successes_a.any()), bool(successes_b.any())]
                scene_individual = float(np.mean(np.concatenate((successes_a, successes_b))))
                scene_collapse = float((collapse_ratio[local] < 0.1).float().mean().cpu())
                scene_rows.append({
                    "scene_id": rows[global_index]["scene_id"], "kind": rows[global_index]["kind"],
                    "near_collision_pair_id": rows[global_index]["near_collision_pair_id"],
                    "majority_of_16_prompt_swap_success": paired_majority,
                    "individual_prior_sample_identity": scene_individual,
                    "best_of_16_query_a_success": query_best[0], "best_of_16_query_b_success": query_best[1],
                    "prompt_output_collapse_rate": scene_collapse,
                    "mean_requested_mse": float(torch.cat((mse_a_own[local], mse_b_own[local])).mean().cpu()),
                    "best_of_16_requested_mse": float(torch.stack((mse_a_own[local].min(), mse_b_own[local].min())).mean().cpu()),
                    "posterior_requested_mse": float(posterior_own[local].mean().cpu()),
                    "condition_c_requested_mse": float(condition_own[local].mean().cpu()),
                    "prompt_swap_consistency_mse": float(swap_consistency[local].mean().cpu()),
                    "median_requested_minus_alternate_mse": float(np.median(signed)),
                })
                individual_success.extend(np.concatenate((successes_a, successes_b)).tolist())
                best_success.extend(query_best)
                majority_success.append(paired_majority)
                collapse_values.extend((collapse_ratio[local].cpu().numpy() < 0.1).tolist())
                all_signed.extend(signed.tolist())
                prior_mse_values.extend(torch.cat((mse_a_own[local], mse_b_own[local])).cpu().numpy().tolist())
                prior_best_mse_values.extend([float(mse_a_own[local].min().cpu()), float(mse_b_own[local].min().cpu())])
                posterior_mse_values.extend(posterior_own[local].cpu().numpy().tolist())
                posterior_identity_values.extend((posterior_own[local] < posterior_alt[local]).cpu().numpy().tolist())
                condition_mse_values.extend(condition_own[local].cpu().numpy().tolist())
                swap_consistency_values.extend(swap_consistency[local].cpu().numpy().tolist())
                for band in range(3):
                    band_a_own = (requested_a[local, :, band] - source_a[local, band]).square().mean(dim=(1, 2))
                    band_a_alt = (requested_a[local, :, band] - source_b[local, band]).square().mean(dim=(1, 2))
                    band_b_own = (requested_b[local, :, band] - source_b[local, band]).square().mean(dim=(1, 2))
                    band_b_alt = (requested_b[local, :, band] - source_a[local, band]).square().mean(dim=(1, 2))
                    band_success[band].extend(torch.cat((band_a_own < band_a_alt, band_b_own < band_b_alt)).cpu().numpy().tolist())
                predicted_physical = torch.stack((requested_a[local].mean(dim=0), requested_b[local].mean(dim=0))).cpu().numpy() * scales[None, :, None, None]
                truth_physical = isolated_physical[global_index]
                predicted_flux = predicted_physical.sum(axis=(-2, -1), dtype=np.float64)
                truth_flux = truth_physical.sum(axis=(-2, -1), dtype=np.float64)
                flux_errors.extend((np.abs(predicted_flux - truth_flux) / np.maximum(np.abs(truth_flux), 1e-12)).ravel().tolist())
                centroid_errors.extend([centroid_error(predicted_physical[q], truth_physical[q]) for q in range(2)])
                if global_index < 5:
                    examples.append({
                        "index": global_index, "blend": blends_physical[global_index], "truth": truth_physical,
                        "prior_a": requested_a[local, :, :, :, :].cpu().numpy() * scales[None, :, None, None],
                        "prior_b": requested_b[local, :, :, :, :].cpu().numpy() * scales[None, :, None, None],
                        "posterior_a": posterior_a[local, :3].cpu().numpy() * scales[:, None, None],
                    })
                if global_index < 5:
                    for sample_index in range(K):
                        sample_hash_rows.append({
                            "scene_id": rows[global_index]["scene_id"], "sample_index": sample_index,
                            "prompt_a_requested_sha256": canonical_tensor_sha256((requested_a[local, sample_index].cpu().numpy() * scales[:, None, None]).astype(np.float32)),
                            "prompt_b_requested_sha256": canonical_tensor_sha256((requested_b[local, sample_index].cpu().numpy() * scales[:, None, None]).astype(np.float32)),
                        })
            print(json.dumps({"phase": "promptability", "completed": stop, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
    majority_rate = float(np.mean(majority_success))
    individual_rate = float(np.mean(individual_success))
    best_rate = float(np.mean(best_success))
    collapse_rate = float(np.mean(collapse_values))
    prior_mse = float(np.mean(prior_mse_values))
    condition_mse = float(np.mean(condition_mse_values))
    posterior_mse = float(np.mean(posterior_mse_values))
    prior_best_mse = float(np.mean(prior_best_mse_values))
    reconstruction_ratio = prior_mse / max(condition_mse, 1e-30)
    prior_posterior_gap = prior_best_mse / max(posterior_mse, 1e-30)
    posterior_identity = float(np.mean(posterior_identity_values))
    identity_gap = posterior_identity - individual_rate
    band_rates = [float(np.mean(values)) for values in band_success]
    gates = [
        {"gate": "majority_of_16_prompt_swap", "threshold": ">=0.80", "observed": majority_rate, "pass": majority_rate >= 0.80},
        {"gate": "individual_prior_sample_identity", "threshold": ">=0.70", "observed": individual_rate, "pass": individual_rate >= 0.70},
        {"gate": "best_of_16_requested_identity", "threshold": ">=0.90", "observed": best_rate, "pass": best_rate >= 0.90},
        {"gate": "prompt_output_collapse", "threshold": "<=0.10", "observed": collapse_rate, "pass": collapse_rate <= 0.10},
        {"gate": "median_identity_advantage", "threshold": "median requested-minus-alternate MSE <0", "observed": float(np.median(all_signed)), "pass": float(np.median(all_signed)) < 0},
        {"gate": "no_band_identity_inversion", "threshold": "each band identity rate >0.5", "observed": ";".join(f"{value:.6f}" for value in band_rates), "pass": all(value > 0.5 for value in band_rates)},
        {"gate": "reconstruction_factor_to_condition_c", "threshold": "<=3.0", "observed": reconstruction_ratio, "pass": reconstruction_ratio <= 3.0},
    ]
    passed = all(bool(row["pass"]) for row in gates)
    write_csv_fresh(run_dir / "tables/pre_atlas_promptability_gates.csv", gates)
    write_csv_fresh(run_dir / "tables/pre_atlas_promptability_per_scene.csv", scene_rows)
    write_csv_fresh(run_dir / "tables/prior_sample_hash_examples.csv", sample_hash_rows)
    summary = [{
        "validation_scenes": len(rows), "prior_samples_per_scene": K,
        "majority_of_16_prompt_swap_success": majority_rate,
        "individual_prior_sample_requested_success": individual_rate,
        "best_of_16_requested_success": best_rate,
        "prompt_output_collapse_rate": collapse_rate,
        "mean_prior_requested_mse_normalized": prior_mse,
        "mean_condition_c_requested_mse_normalized": condition_mse,
        "prior_to_condition_c_mse_ratio": reconstruction_ratio,
        "mean_posterior_requested_mse_normalized": posterior_mse,
        "prior_best_of_16_requested_mse_normalized": prior_best_mse,
        "prior_best_to_posterior_mse_ratio_diagnostic": prior_posterior_gap,
        "posterior_identity_success": posterior_identity,
        "posterior_minus_prior_identity_gap_diagnostic": identity_gap,
        "mean_prompt_swap_consistency_mse": float(np.mean(swap_consistency_values)),
        "mean_relative_per_band_flux_error": float(np.mean(flux_errors)),
        "mean_centroid_error_pixels": float(np.nanmean(centroid_errors)),
        "band_g_identity_rate": band_rates[0], "band_r_identity_rate": band_rates[1], "band_z_identity_rate": band_rates[2],
        "status": "PASS" if passed else "FAIL",
    }]
    write_csv_fresh(run_dir / "tables/pre_atlas_promptability_summary.csv", summary)
    report = f"""# Non-Atlas prior promptability gate

Status: **{'PASS' if passed else 'FAIL — STOP; ATLAS NOT AUTHORIZED'}**.

- Majority-of-16 paired prompt-swap success: {majority_rate:.6f} (gate >=0.80).
- Individual prior-sample requested identity: {individual_rate:.6f} (gate >=0.70).
- Best-of-16 requested identity: {best_rate:.6f} (gate >=0.90).
- Prompt-output collapse: {collapse_rate:.6f} (gate <=0.10).
- Prior/Condition-C normalized requested MSE: {prior_mse:.6g} / {condition_mse:.6g}; ratio {reconstruction_ratio:.6f} (gate <=3.0).
- Per-band identity rates g/r/z: {band_rates[0]:.6f} / {band_rates[1]:.6f} / {band_rates[2]:.6f}.
- Median requested-minus-alternate MSE: {float(np.median(all_signed)):.6g}.
- Mean prompt-swap decomposition consistency MSE: {float(np.mean(swap_consistency_values)):.6g}.
- Mean relative per-band flux error / centroid error: {float(np.mean(flux_errors)):.6g} / {float(np.nanmean(centroid_errors)):.6g} pixels.

Posterior diagnostic requested MSE is {posterior_mse:.6g}; prior best-of-16 is
{prior_best_mse:.6g} (diagnostic ratio {prior_posterior_gap:.6f}). Posterior identity
is {posterior_identity:.6f}; the posterior-minus-prior identity gap is {identity_gap:.6f}.
These posterior values are diagnostics only and do not rescue a failed prior.

Only the fresh Atlas-excluded validation partition was opened. Atlas evaluation,
forward-consistency calibration, control-concentration analysis, development, and
lockbox access remain blocked if any row in the gate table fails.
"""
    write_text_fresh(run_dir / "diagnostics/pre_atlas_promptability_report.md", report)
    if examples:
        figure, axes = plt.subplots(len(examples), 7, figsize=(14, 2.3 * len(examples)), constrained_layout=True)
        titles = ["truth A", "truth B", "blend", "prior A #1", "prior A #2", "prior B #1", "posterior A"]
        for column, title in enumerate(titles):
            axes[0, column].set_title(title, fontsize=8)
        for row_index, example in enumerate(examples):
            panels = [
                example["truth"][0, 1], example["truth"][1, 1], example["blend"][1],
                example["prior_a"][0, 1], example["prior_a"][1, 1], example["prior_b"][0, 1], example["posterior_a"][1],
            ]
            scale = max(float(np.max(np.abs(value))) for value in panels)
            for column, panel in enumerate(panels):
                axes[row_index, column].imshow(np.arcsinh(panel / max(scale, 1e-12) * 20), origin="lower", cmap="coolwarm")
                axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
        figure.suptitle("Non-Atlas Thayer-PU promptability examples (r band)")
        figure.savefig(run_dir / "example_grids/pre_atlas_promptability_grid.png", dpi=170)
        plt.close(figure)
    write_json_fresh(run_dir / "logs/pre_atlas_promptability_complete.json", {
        "status": "PASS" if passed else "FAIL", "next_gate_authorized": passed,
        "atlas_evaluation_authorized": False, "majority_of_16_prompt_swap_success": majority_rate,
        "individual_prior_sample_requested_success": individual_rate,
        "best_of_16_requested_success": best_rate, "prompt_output_collapse_rate": collapse_rate,
        "prior_to_condition_c_mse_ratio": reconstruction_ratio,
        "prior_best_to_posterior_mse_ratio_diagnostic": prior_posterior_gap,
        "runtime_seconds": time.time() - started, "device": "mps", "mps_fallback": False,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL", "gates": gates}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("latent", "promptability"), required=True)
    args = parser.parse_args()
    run_dir = require_run(args.run_dir, args.phase)
    if args.phase == "latent":
        latent_audit(run_dir)
    else:
        promptability_audit(run_dir)


if __name__ == "__main__":
    main()
