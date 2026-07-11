#!/usr/bin/env python3
"""Explicitly bounded Thayer-Select BTK engineering smoke run.

Modes deliberately separate BTK generation (the isolated BTK environment) from
MPS training (the pre-existing main environment).  This is not a scientific
trainer and has no uncertainty or recoverability head.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PILOT_SEED = 2026071101
SMOKE_GENERATION_SEED = 2026073101
SMOKE_NOISE_SEED = 2026073102
SMOKE_TRAINING_SEED = 2026073103
TRAIN_SCENES = 500
VALIDATION_SCENES = 100
TOTAL_SCENES = TRAIN_SCENES + VALIDATION_SCENES


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fit_scales(blends: np.ndarray) -> np.ndarray:
    """Fit positive per-band scales from the engineering training subset only."""

    scales = np.quantile(np.abs(blends), 0.995, axis=(0, 2, 3)).astype(np.float64)
    if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise RuntimeError(f"Invalid training-only normalization scales: {scales}")
    return scales


def normalize(images: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return images / scales[None, :, None, None]


def inverse_normalize(images: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return images * scales[None, :, None, None]


def normalization_check(run_dir: Path) -> None:
    paths = sorted((run_dir / "data/btk_engineering_pilot").glob("double_*.npz"))
    if len(paths) != 20:
        raise RuntimeError(f"Expected 20 pilot doubles, found {len(paths)}")
    training = np.stack([np.load(path, allow_pickle=False)["blend_noisy"] for path in paths[:16]])
    scales = fit_scales(training)
    restored = inverse_normalize(normalize(training, scales), scales)
    maximum = float(np.max(np.abs(restored - training)))
    relative = maximum / max(float(np.max(np.abs(training))), np.finfo(np.float64).tiny)
    result = {
        "status": "PASS" if maximum <= 1e-10 else "FAIL",
        "fit_subset": "first 16 deterministic engineering double scenes only",
        "validation_contribution": 0,
        "scales": scales.tolist(),
        "maximum_absolute_inversion_error": maximum,
        "maximum_relative_inversion_error": relative,
        "tolerance": 1e-10,
        "negative_pixels_preserved": True,
        "clipping": False,
    }
    path = run_dir / "diagnostics/smoke_normalization_check.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


def generate(run_dir: Path, catalog_path: Path) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from astropy.table import Table
    from btk.draw_blends import CatsimGenerator
    from btk.sampling_functions import SamplingFunction
    from btk_scene import (
        BAND_ORDER,
        STAMP_SIZE_ARCSEC,
        build_scene_specs,
        gaussian_prompt,
        load_catsim_catalog,
        validated_lsst_survey,
    )

    catalog, catalog_hash = load_catsim_catalog(catalog_path)
    pilot_specs = build_scene_specs(catalog.table, source_selection_seed=PILOT_SEED)
    engineering_pool = sorted({row for spec in pilot_specs for row in spec.catalog_rows})
    if len(engineering_pool) != 60:
        raise RuntimeError("Smoke generation must reuse exactly the 60 excluded pilot identities")

    class SmokeSampling(SamplingFunction):
        def __init__(self) -> None:
            super().__init__(stamp_size=int(STAMP_SIZE_ARCSEC), min_number=2, max_number=2, seed=0)
            self.stamp_size = STAMP_SIZE_ARCSEC
            self.rng = np.random.default_rng(SMOKE_GENERATION_SEED)
            self.scene_index = 0

        def __call__(self, table: Table) -> Table:
            rows = self.rng.choice(engineering_pool, size=2, replace=False)
            separation = float(self.rng.uniform(1.0, 2.8))
            angle = float(self.rng.uniform(0, 2 * np.pi))
            center = self.rng.uniform(-0.25, 0.25, size=2)
            delta = 0.5 * separation * np.asarray([np.cos(angle), np.sin(angle)])
            positions = np.stack([center - delta, center + delta])
            output = table[list(rows)].copy()
            output["ra"] = positions[:, 0]
            output["dec"] = positions[:, 1]
            output["smoke_scene_index"] = self.scene_index
            self.scene_index += 1
            return output

    survey = validated_lsst_survey()
    generator = CatsimGenerator(
        catalog,
        SmokeSampling(),
        survey,
        batch_size=TOTAL_SCENES,
        njobs=1,
        verbose=False,
        use_bar=False,
        add_noise="all",
        seed=SMOKE_NOISE_SEED,
        apply_shear=False,
        augment_data=False,
    )
    batch = next(generator)
    full_bands = tuple(batch.survey.available_filters)
    indices = [full_bands.index(band) for band in BAND_ORDER]
    blends = np.asarray(batch.blend_images[:, indices], dtype=np.float32)
    isolated = np.asarray(batch.isolated_images[:, :2][:, :, indices], dtype=np.float32)
    prompts = np.empty((TOTAL_SCENES, 2, blends.shape[-2], blends.shape[-1]), dtype=np.float32)
    catalog_rows = np.empty((TOTAL_SCENES, 2), dtype=np.int64)
    source_ids = []
    xy = np.empty((TOTAL_SCENES, 2, 2), dtype=np.float64)
    for index, scene in enumerate(batch.catalog_list):
        catalog_rows[index] = np.asarray(scene["catalog_row"], dtype=np.int64)
        source_ids.append([str(value) for value in scene["engineering_source_id"]])
        for source_index, source in enumerate(scene):
            xy[index, source_index] = [float(source["x_peak"]), float(source["y_peak"])]
            prompts[index, source_index] = gaussian_prompt(
                blends.shape[-2:], float(source["x_peak"]), float(source["y_peak"])
            ).astype(np.float32)
    query_indices = np.arange(TOTAL_SCENES, dtype=np.int64) % 2
    dataset_path = run_dir / "data/thayer_select_btk_smoke_dataset.npz"
    if dataset_path.exists():
        raise FileExistsError(dataset_path)
    np.savez_compressed(
        dataset_path,
        blends=blends,
        isolated=isolated,
        prompts=prompts,
        query_indices=query_indices,
        catalog_rows=catalog_rows,
        xy=xy,
    )
    allowed = set(engineering_pool)
    if not set(catalog_rows.ravel()).issubset(allowed):
        raise RuntimeError("Smoke dataset contains a non-engineering identity")
    manifest = {
        "description": "An explicitly authorized engineering smoke run; no full scientific Thayer-Select training has begun.",
        "catalog_path": str(catalog_path.relative_to(REPO)),
        "catalog_sha256": catalog_hash,
        "dataset_path": str(dataset_path.relative_to(run_dir)),
        "dataset_sha256": digest(dataset_path),
        "scene_count": TOTAL_SCENES,
        "training_scene_count": TRAIN_SCENES,
        "validation_scene_count": VALIDATION_SCENES,
        "source_count_per_scene": 2,
        "engineering_identity_pool_size": len(allowed),
        "all_ids_from_excluded_pilot_pool": True,
        "query_policy": "A/B alternating exactly; 300 each overall, 250 each in training",
        "band_order": list(BAND_ORDER),
        "array_shape": list(blends.shape),
        "image_units": "detected electrons per pixel",
        "generation_seed": SMOKE_GENERATION_SEED,
        "noise_seed": SMOKE_NOISE_SEED,
        "source_ids_sha256": sha256(json.dumps(source_ids, sort_keys=True).encode()).hexdigest(),
        "versions": {"btk": "1.0.9", "galsim": "2.8.4", "surveycodex": "1.2.0"},
    }
    manifest_path = run_dir / "manifests/thayer_select_btk_smoke_dataset.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


def train(run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; CPU fallback is prohibited")
    torch.manual_seed(SMOKE_TRAINING_SEED)
    np.random.seed(SMOKE_TRAINING_SEED)
    device = torch.device("mps")
    dataset_path = run_dir / "data/thayer_select_btk_smoke_dataset.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        blends = np.asarray(data["blends"], dtype=np.float64)
        isolated = np.asarray(data["isolated"], dtype=np.float64)
        prompts = np.asarray(data["prompts"], dtype=np.float64)
        query_indices = np.asarray(data["query_indices"], dtype=np.int64)
    if blends.shape[0] != TOTAL_SCENES or isolated.shape[:2] != (TOTAL_SCENES, 2):
        raise RuntimeError("Unexpected smoke dataset shape")
    scales = fit_scales(blends[:TRAIN_SCENES])
    blend_norm = normalize(blends, scales).astype(np.float32)
    isolated_flat = isolated.reshape(-1, 3, *isolated.shape[-2:])
    isolated_norm = normalize(isolated_flat, scales).reshape(isolated.shape).astype(np.float32)
    selected_target = isolated_norm[np.arange(TOTAL_SCENES), query_indices]
    selected_prompt = prompts[np.arange(TOTAL_SCENES), query_indices].astype(np.float32)
    inputs = np.concatenate([blend_norm, selected_prompt[:, None]], axis=1)

    class TinySelect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 3, 3, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    model = TinySelect().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(inputs[:TRAIN_SCENES]), torch.from_numpy(selected_target[:TRAIN_SCENES]))
    val_ds = TensorDataset(torch.from_numpy(inputs[TRAIN_SCENES:]), torch.from_numpy(selected_target[TRAIN_SCENES:]))
    generator = torch.Generator().manual_seed(SMOKE_TRAINING_SEED)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    metrics = []
    best_loss = float("inf")
    best_path = run_dir / "checkpoints/thayer_select_btk_smoke_best.pth"
    final_path = run_dir / "checkpoints/thayer_select_btk_smoke_final.pth"
    for path in (best_path, final_path):
        if path.exists():
            raise FileExistsError(path)
    for epoch in range(1, 4):
        model.train()
        train_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(x)
            train_count += len(x)
        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                val_sum += float(loss.detach().cpu()) * len(x)
                val_count += len(x)
        record = {"epoch": epoch, "training_loss": train_sum / train_count, "validation_loss": val_sum / val_count}
        metrics.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "config": "engineering-smoke-only"}, best_path)
    torch.save({"state_dict": model.state_dict(), "epoch": 3, "config": "engineering-smoke-only"}, final_path)

    model.eval()
    eval_indices = np.arange(TRAIN_SCENES, min(TOTAL_SCENES, TRAIN_SCENES + 20))
    both_predictions = []
    with torch.no_grad():
        for role in (0, 1):
            x = np.concatenate([blend_norm[eval_indices], prompts[eval_indices, role, None].astype(np.float32)], axis=1)
            both_predictions.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
        empty_x = np.concatenate([blend_norm[eval_indices], np.zeros((len(eval_indices), 1, *blends.shape[-2:]), dtype=np.float32)], axis=1)
        empty_predictions = model(torch.from_numpy(empty_x).to(device)).cpu().numpy()
        isolated_input = normalize(isolated[eval_indices, 0], scales).astype(np.float32)
        isolated_x = np.concatenate([isolated_input, prompts[eval_indices, 0, None].astype(np.float32)], axis=1)
        isolated_predictions = model(torch.from_numpy(isolated_x).to(device)).cpu().numpy()
    pred_a, pred_b = both_predictions
    target_a = isolated_norm[eval_indices, 0]
    target_b = isolated_norm[eval_indices, 1]
    mse_a_correct = float(np.mean((pred_a - target_a) ** 2))
    mse_a_swap = float(np.mean((pred_a - target_b) ** 2))
    mse_b_correct = float(np.mean((pred_b - target_b) ** 2))
    mse_b_swap = float(np.mean((pred_b - target_a) ** 2))
    finite = bool(np.all(np.isfinite(pred_a)) and np.all(np.isfinite(pred_b)) and np.all(np.isfinite(empty_predictions)))
    target_low, target_high = np.quantile(selected_target[:TRAIN_SCENES], [0.001, 0.999])
    in_range = float(np.mean((np.concatenate([pred_a, pred_b]) >= target_low) & (np.concatenate([pred_a, pred_b]) <= target_high)))
    inversion_probe = inverse_normalize(normalize(isolated[eval_indices, 0], scales), scales)
    inversion_error = float(np.max(np.abs(inversion_probe - isolated[eval_indices, 0])))
    summary = {
        "description": "An explicitly authorized engineering smoke run; no full scientific Thayer-Select training has begun.",
        "device": "mps",
        "epochs": 3,
        "training_scenes": TRAIN_SCENES,
        "validation_scenes": VALIDATION_SCENES,
        "training_seed": SMOKE_TRAINING_SEED,
        "normalization_scales_training_only": scales.tolist(),
        "training_loss_decreased": metrics[-1]["training_loss"] < metrics[0]["training_loss"],
        "validation_loss_finite": bool(np.isfinite(metrics[-1]["validation_loss"])),
        "prompt_outputs_differ_mean_abs": float(np.mean(np.abs(pred_a - pred_b))),
        "prompt_a_correct_mse": mse_a_correct,
        "prompt_a_swapped_mse": mse_a_swap,
        "prompt_b_correct_mse": mse_b_correct,
        "prompt_b_swapped_mse": mse_b_swap,
        "prompt_identity_correspondence_pass": bool(mse_a_correct < mse_a_swap and mse_b_correct < mse_b_swap),
        "empty_prompt_abs_flux_ratio_to_blend": float(np.sum(np.abs(empty_predictions)) / np.sum(np.abs(blend_norm[eval_indices]))),
        "outputs_finite": finite,
        "output_fraction_within_training_target_0p1_99p9_range": in_range,
        "flux_space_inversion_max_abs_error": inversion_error,
        "isolated_no_harm_normalized_mse": float(np.mean((isolated_predictions - target_a) ** 2)),
        "best_checkpoint": str(best_path.relative_to(run_dir)),
        "best_checkpoint_sha256": digest(best_path),
        "final_checkpoint": str(final_path.relative_to(run_dir)),
        "final_checkpoint_sha256": digest(final_path),
        "scientific_result": False,
    }
    metrics_path = run_dir / "tables/thayer_select_btk_smoke_epochs.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    summary_path = run_dir / "reports/thayer_select_btk_smoke_results.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    config = {
        "architecture": "Conv(4,16,3)-ReLU-Conv(16,16,3)-ReLU-Conv(16,3,3)",
        "optimizer": "Adam(lr=1e-3)",
        "loss": "normalized requested-isolated-source MSE",
        "batch_size": 32,
        "epochs": 3,
        "device": "mps; no CPU fallback",
        "seeds": {"generation": SMOKE_GENERATION_SEED, "noise": SMOKE_NOISE_SEED, "training": SMOKE_TRAINING_SEED},
    }
    config_path = run_dir / "manifests/thayer_select_btk_smoke_training_config.json"
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    scene = 0
    r = 1
    panels = [blend_norm[eval_indices[scene], r], target_a[scene, r], pred_a[scene, r], target_b[scene, r], pred_b[scene, r], empty_predictions[scene, r]]
    limit = max(float(np.max(np.abs(x))) for x in panels)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, panel, title in zip(axes.flat, panels, ["blend", "target A", "prediction A", "target B", "prediction B", "empty prompt prediction"]):
        image = axis.imshow(panel, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit); axis.set_title(title); fig.colorbar(image, ax=axis, fraction=.046)
    fig.suptitle("Engineering smoke only | normalized r band | shared symmetric linear scale")
    figure_path = run_dir / "figures/thayer_select_btk_smoke_qualitative.png"
    fig.savefig(figure_path, dpi=140); plt.close(fig)
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["normalization-check", "generate", "train"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode == "normalization-check":
        normalization_check(run_dir)
    elif args.mode == "generate":
        if args.catalog is None:
            raise SystemExit("--catalog is required for generate")
        generate(run_dir, args.catalog.resolve())
    else:
        train(run_dir)


if __name__ == "__main__":
    main()
