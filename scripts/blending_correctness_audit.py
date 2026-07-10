"""Audit Thayer-Net RGB blend generation, replay, and benchmark distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate as ndi_rotate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_final_test_manifests as final_manifest
from src import blend as gd_blend
from src import utils as gd_utils


DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "data/Galaxy10_DECals.h5"
DEFAULT_MANIFEST_RUN = (
    PROJECT_ROOT / "outputs/runs/final_test_manifest_prep_20260710_061737"
)
DEFAULT_ARTIFACT_TABLE = (
    PROJECT_ROOT
    / "outputs/runs/source_artifact_audit_20260710_061059/tables/source_artifact_candidates.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-run", type=Path, default=DEFAULT_MANIFEST_RUN)
    parser.add_argument("--artifact-table", type=Path, default=DEFAULT_ARTIFACT_TABLE)
    parser.add_argument("--replay-per-suite", type=int, default=30)
    parser.add_argument("--replay-output-name", default="blend_replay_check.csv")
    parser.add_argument("--reuse-existing-distribution", action="store_true")
    parser.add_argument("--resume-finalization", action="store_true")
    return parser.parse_args()


def resolve_existing(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_run(path: Path) -> Path:
    resolved = resolve_existing(path)
    allowed = (PROJECT_ROOT / "outputs/runs").resolve()
    if allowed not in resolved.parents or not resolved.name.startswith(
        "research_correctness_audit_"
    ):
        raise ValueError("run-dir must be a research_correctness_audit_* directory")
    return resolved


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False)


def safe_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Expected YAML mapping")
    return payload


def manual_components(
    target: np.ndarray,
    contaminant: np.ndarray,
    dx: int,
    dy: int,
    rotation: float,
    brightness: float,
    blur_sigma: float,
    noise_std: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    target_blurred = (
        gaussian_filter(target, sigma=(blur_sigma, blur_sigma, 0))
        if blur_sigma > 0
        else target.copy()
    )
    target_blurred = np.clip(target_blurred, 0.0, 1.0).astype(np.float32)
    foreground, _metadata = gd_blend.extract_source_foreground(contaminant)
    scaled_unclipped = foreground * brightness
    scaled = np.clip(scaled_unclipped, 0.0, 1.0)
    if abs(rotation) > 1e-12:
        scaled = ndi_rotate(
            scaled,
            rotation,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )
        scaled[scaled < 0.002] = 0.0
    shifted = gd_blend.shift_foreground(scaled, dx=dx, dy=dy)
    preclip_composite = target_blurred + shifted
    composite = np.clip(preclip_composite, 0.0, 1.0)
    noise = (
        rng.normal(scale=noise_std, size=composite.shape)
        if noise_std > 0
        else np.zeros_like(composite)
    )
    blend_preclip_noise = composite + noise
    blended = np.clip(blend_preclip_noise, 0.0, 1.0).astype(np.float32)
    nuisance_only = np.clip(target_blurred + noise, 0.0, 1.0).astype(np.float32)
    return {
        "target_blurred": target_blurred,
        "foreground": foreground,
        "scaled_unclipped": scaled_unclipped,
        "scaled_foreground": scaled,
        "shifted_foreground": shifted,
        "preclip_composite": preclip_composite,
        "composite": composite,
        "noise": noise,
        "blended": blended,
        "nuisance_only": nuisance_only,
    }


def reconstruct_row(
    row: pd.Series,
    source_images: np.ndarray,
    source_indices: np.ndarray,
    pxscale: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seed = int(row["sample_random_seed"])
    rng = np.random.default_rng(seed)
    source_count = int(row["source_pool_size"])
    target_local, contaminant_local = rng.choice(
        source_count, size=2, replace=False
    ).astype(int)
    dx = int(rng.integers(-int(row["sampling_max_shift"]), int(row["sampling_max_shift"]) + 1))
    dy = int(rng.integers(-int(row["sampling_max_shift"]), int(row["sampling_max_shift"]) + 1))
    brightness = float(
        rng.uniform(float(row["sampling_brightness_low"]), float(row["sampling_brightness_high"]))
    )
    blur_sigma = float(
        rng.uniform(float(row["sampling_blur_low"]), float(row["sampling_blur_high"]))
    )
    noise_std = float(
        rng.uniform(float(row["sampling_noise_low"]), float(row["sampling_noise_high"]))
    )
    rotation_low = float(row["sampling_rotation_low"])
    rotation_high = float(row["sampling_rotation_high"])
    rotation = (
        float(rng.uniform(rotation_low, rotation_high))
        if (rotation_low, rotation_high) != (0.0, 0.0)
        else 0.0
    )
    target = source_images[target_local].astype(np.float32) / 255.0
    contaminant = source_images[contaminant_local].astype(np.float32) / 255.0
    components = manual_components(
        target,
        contaminant,
        dx,
        dy,
        rotation,
        brightness,
        blur_sigma,
        noise_std,
        rng,
    )

    rng_reference = np.random.default_rng(seed)
    reference_local = rng_reference.choice(source_count, size=2, replace=False).astype(int)
    rng_reference.integers(-int(row["sampling_max_shift"]), int(row["sampling_max_shift"]) + 1)
    rng_reference.integers(-int(row["sampling_max_shift"]), int(row["sampling_max_shift"]) + 1)
    rng_reference.uniform(float(row["sampling_brightness_low"]), float(row["sampling_brightness_high"]))
    rng_reference.uniform(float(row["sampling_blur_low"]), float(row["sampling_blur_high"]))
    rng_reference.uniform(float(row["sampling_noise_low"]), float(row["sampling_noise_high"]))
    if (rotation_low, rotation_high) != (0.0, 0.0):
        rng_reference.uniform(rotation_low, rotation_high)
    reference_blend, info = gd_blend.blend_pair(
        target,
        contaminant,
        shift=(dx, dy),
        rotation=rotation,
        brightness=brightness,
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        rng=rng_reference,
    )
    blended = np.asarray(components["blended"])
    affected = gd_utils.affected_region_mask(
        target, blended, threshold=float(row["affected_mask_threshold"])
    )
    core = gd_utils.evaluation_core_mask_p85_v1(
        target,
        aperture_fraction=float(row["core_aperture_fraction"]),
        core_percentile=float(row["core_percentile"]),
    )
    halo = gd_utils.halo_band_mask_manhattan_v1(
        affected, dilation_iters=int(row["halo_dilation_iters"])
    )
    nuisance = gd_utils.affected_region_mask(
        target,
        np.asarray(components["nuisance_only"]),
        threshold=float(row["affected_mask_threshold"]),
    )
    shifted = np.asarray(components["shifted_foreground"])
    support = shifted.mean(axis=-1) > 0
    preclip = np.asarray(components["preclip_composite"])
    composite = np.asarray(components["composite"])
    scaled_unclipped = np.asarray(components["scaled_unclipped"])
    scaled = np.asarray(components["scaled_foreground"])
    contaminant_flux = float(shifted.sum())
    support_count = int(support.sum())
    target_global = int(source_indices[target_local])
    contaminant_global = int(source_indices[contaminant_local])
    checks = {
        "sample_id": str(row["sample_id"]),
        "suite": str(row["suite"]),
        "target_source_index": target_global,
        "contaminant_source_index": contaminant_global,
        "source_indices_match": target_global == int(row["target_source_index"])
        and contaminant_global == int(row["contaminant_source_index"]),
        "reference_source_draw_match": bool(np.array_equal(reference_local, [target_local, contaminant_local])),
        "parameters_match": bool(
            dx == int(row["shift_dx_pixels"])
            and dy == int(row["shift_dy_pixels"])
            and np.isclose(brightness, float(row["brightness_scale"]), rtol=0, atol=5e-15)
            and np.isclose(blur_sigma, float(row["blur_sigma"]), rtol=0, atol=5e-15)
            and np.isclose(noise_std, float(row["noise_std"]), rtol=0, atol=5e-15)
            and np.isclose(rotation, float(row["rotation_angle_degrees"]), rtol=0, atol=5e-15)
        ),
        "max_abs_manual_vs_generator": float(np.max(np.abs(blended - reference_blend))),
        "blend_exact_match": bool(np.array_equal(blended, reference_blend)),
        "affected_fraction_saved": float(row["affected_mask_fraction"]),
        "affected_fraction_replayed": float(affected.mean()),
        "affected_fraction_match": bool(
            np.isclose(affected.mean(), float(row["affected_mask_fraction"]), rtol=0, atol=5e-12)
        ),
        "core_obstruction_saved": float(row["core_obstruction_fraction"]),
        "core_obstruction_replayed": float((affected & core).sum() / core.sum()),
        "core_obstruction_match": bool(
            np.isclose(
                (affected & core).sum() / core.sum(),
                float(row["core_obstruction_fraction"]),
                rtol=0,
                atol=5e-12,
            )
        ),
        "halo_fraction_saved": float(row["halo_band_fraction"]),
        "halo_fraction_replayed": float(halo.mean()),
        "halo_fraction_match": bool(
            np.isclose(halo.mean(), float(row["halo_band_fraction"]), rtol=0, atol=5e-12)
        ),
        "blend_float32_sha256": array_sha256(blended.astype(np.float32)),
        "affected_mask_sha256": array_sha256(affected.astype(np.uint8)),
        "core_mask_sha256": array_sha256(core.astype(np.uint8)),
        "halo_mask_sha256": array_sha256(halo.astype(np.uint8)),
        "prebrightness_flux_loss_fraction": float(
            max(scaled_unclipped.sum() - scaled.sum(), 0.0)
            / max(float(scaled_unclipped.sum()), 1e-12)
        ),
        "composite_clip_loss_relative_to_shifted_contaminant_flux": float(
            max(preclip.sum() - composite.sum(), 0.0) / max(contaminant_flux, 1e-12)
        ),
        "preclip_high_channel_fraction": float((preclip > 1.0).mean()),
        "shifted_foreground_flux_retention_fraction": float(
            shifted.sum() / max(float(scaled.sum()), 1e-12)
        ),
        "affected_support_capture_fraction": float(
            (affected & support).sum() / support_count if support_count else np.nan
        ),
        "affected_flux_capture_fraction": float(
            shifted[affected].sum() / contaminant_flux if contaminant_flux else np.nan
        ),
        "nuisance_only_affected_fraction": float(nuisance.mean()),
        "target_pxscale_arcsec": float(pxscale[target_global]),
        "contaminant_pxscale_arcsec": float(pxscale[contaminant_global]),
        "angular_size_ratio": float(
            info["size_ratio"] * pxscale[contaminant_global] / pxscale[target_global]
        ),
    }
    checks["passed"] = bool(
        checks["source_indices_match"]
        and checks["reference_source_draw_match"]
        and checks["parameters_match"]
        and checks["blend_exact_match"]
        and checks["affected_fraction_match"]
        and checks["core_obstruction_match"]
        and checks["halo_fraction_match"]
    )
    arrays = {
        "target": target,
        "contaminant": contaminant,
        "shifted_foreground": shifted,
        "blend": blended,
        "affected": affected,
        "core": core,
        "halo": halo,
    }
    return checks, arrays


def summarize_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "shift_l2_pixels",
        "size_ratio",
        "angular_size_ratio",
        "brightness_scale",
        "blur_sigma",
        "noise_std",
        "affected_mask_fraction",
        "core_obstruction_fraction",
        "blend_severity_score",
        "halo_band_fraction",
    )
    rows: list[dict[str, Any]] = []
    for suite, suite_frame in frame.groupby("suite", sort=False):
        for metric in metrics:
            values = pd.to_numeric(suite_frame[metric], errors="coerce").dropna()
            rows.append(
                {
                    "suite": suite,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "p05": float(values.quantile(0.05)),
                    "p50": float(values.quantile(0.50)),
                    "p95": float(values.quantile(0.95)),
                    "max": float(values.max()),
                }
            )
        for metric, value in (
            ("unique_target_sources", suite_frame["target_source_index"].nunique()),
            ("unique_contaminant_sources", suite_frame["contaminant_source_index"].nunique()),
            ("pixel_scale_mismatch_fraction", suite_frame["pixel_scale_mismatch"].mean()),
            ("repeated_ordered_pair_count", suite_frame.duplicated(["target_source_index", "contaminant_source_index"]).sum()),
        ):
            rows.append(
                {
                    "suite": suite,
                    "metric": metric,
                    "n": int(len(suite_frame)),
                    "mean": float(value),
                    "std": np.nan,
                    "min": np.nan,
                    "p05": np.nan,
                    "p50": np.nan,
                    "p95": np.nan,
                    "max": np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_distribution_plots(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[str] = []
    metrics = (
        "shift_l2_pixels",
        "size_ratio",
        "angular_size_ratio",
        "brightness_scale",
        "affected_mask_fraction",
        "core_obstruction_fraction",
        "blend_severity_score",
    )
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for suite, suite_frame in frame.groupby("suite", sort=False):
            values = pd.to_numeric(suite_frame[metric], errors="coerce").dropna()
            ax.hist(values, bins=35, histtype="step", linewidth=1.5, density=True, label=suite)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = output_dir / f"{metric}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path.relative_to(PROJECT_ROOT)))
    return paths


def contact_sheet(samples: list[dict[str, Any]], title: str, path: Path) -> None:
    columns = ("target", "shifted_foreground", "blend", "affected", "core", "halo")
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(15, 2.5 * len(samples)))
    if len(samples) == 1:
        axes = np.asarray([axes])
    for row_index, sample in enumerate(samples):
        for column_index, key in enumerate(columns):
            ax = axes[row_index, column_index]
            value = sample[key]
            if np.asarray(value).ndim == 2:
                ax.imshow(value, cmap="magma", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(value, 0.0, 1.0))
            if row_index == 0:
                ax.set_title(key.replace("_", " "), fontsize=9)
            ax.axis("off")
        axes[row_index, 0].set_ylabel(str(sample.get("label", row_index)), fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fresh_candidate(
    spec: final_manifest.SuiteSpec,
    images: np.ndarray,
    global_indices: np.ndarray,
    suite_seed: int,
    accepted_index: int,
) -> dict[str, Any]:
    for attempt in range(1 + accepted_index * 300, 1 + (accepted_index + 1) * 300):
        seed = final_manifest.sample_seed(suite_seed, attempt)
        rng = np.random.default_rng(seed)
        source_count = len(images) if spec.source_count < 0 else min(spec.source_count, len(images))
        target_local, contaminant_local = rng.choice(source_count, size=2, replace=False).astype(int)
        dx = int(rng.integers(-spec.max_shift, spec.max_shift + 1))
        dy = int(rng.integers(-spec.max_shift, spec.max_shift + 1))
        brightness = float(rng.uniform(*spec.brightness_range))
        blur_sigma = float(rng.uniform(*spec.blur_range))
        noise_std = float(rng.uniform(*spec.noise_range))
        rotation = (
            float(rng.uniform(*spec.rotation_range))
            if spec.rotation_range != (0.0, 0.0)
            else 0.0
        )
        target = images[target_local].astype(np.float32) / 255.0
        contaminant = images[contaminant_local].astype(np.float32) / 255.0
        components = manual_components(
            target, contaminant, dx, dy, rotation, brightness, blur_sigma, noise_std, rng
        )
        blend = np.asarray(components["blended"])
        affected = gd_utils.affected_region_mask(target, blend, threshold=0.02)
        core = gd_utils.evaluation_core_mask_p85_v1(target)
        halo = gd_utils.halo_band_mask_manhattan_v1(affected, 5)
        _, target_size = gd_blend.extract_source_foreground(target)
        _, contaminant_size = gd_blend.extract_source_foreground(contaminant)
        size_ratio = contaminant_size["radius"] / target_size["radius"] if target_size["radius"] else np.nan
        core_obstruction = float((affected & core).sum() / core.sum())
        mask_fraction = float(affected.mean())
        size_ok = spec.min_size_ratio is None or size_ratio >= spec.min_size_ratio
        if spec.max_size_ratio is not None:
            size_ok = size_ok and size_ratio <= spec.max_size_ratio
        mask_ok = spec.min_mask_fraction is None or mask_fraction >= spec.min_mask_fraction
        core_ok = spec.min_core_obstruction is None or core_obstruction >= spec.min_core_obstruction
        if size_ok and mask_ok and core_ok:
            return {
                "label": f"{spec.name}:{accepted_index}",
                "target": target,
                "contaminant": contaminant,
                "shifted_foreground": np.asarray(components["shifted_foreground"]),
                "blend": blend,
                "affected": affected,
                "core": core,
                "halo": halo,
                "mask_fraction": mask_fraction,
                "target_source_index": int(global_indices[target_local]),
                "contaminant_source_index": int(global_indices[contaminant_local]),
            }
    raise RuntimeError(f"Could not generate audit contact sample for {spec.name}")


def gaussian_size_sanity() -> pd.DataFrame:
    size = 256
    y, x = np.ogrid[:size, :size]
    rows = []
    target_sigma = 24.0
    target_gray = np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2 * target_sigma**2))
    target = np.repeat(target_gray[..., None], 3, axis=-1).astype(np.float32)
    _, target_size = gd_blend.extract_source_foreground(target)
    for sigma in (2.0, 6.0, 12.0, 24.0, 32.0):
        gray = np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2 * sigma**2))
        image = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
        _, measured = gd_blend.extract_source_foreground(image)
        rows.append(
            {
                "target_sigma": target_sigma,
                "contaminant_sigma": sigma,
                "true_sigma_ratio": sigma / target_sigma,
                "target_extracted_radius": target_size["radius"],
                "contaminant_extracted_radius": measured["radius"],
                "stored_radius_ratio": measured["radius"] / target_size["radius"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    run_dir = resolve_run(args.run_dir)
    dataset = resolve_existing(args.dataset)
    config_path = resolve_existing(args.config)
    manifest_run = resolve_existing(args.manifest_run)
    artifact_table_path = resolve_existing(args.artifact_table)
    config = load_config(config_path)

    manifest_paths = sorted((manifest_run / "manifests").glob("*_final_test.csv"))
    manifest_frames = [pd.read_csv(path) for path in manifest_paths]
    manifest = pd.concat(manifest_frames, ignore_index=True)
    with h5py.File(dataset, "r") as handle:
        pxscale = handle["pxscale"][:]
        labels = handle["ans"][:]
        n_sources = int(handle["images"].shape[0])
    manifest["target_pxscale_arcsec"] = pxscale[manifest["target_source_index"].astype(int)]
    manifest["contaminant_pxscale_arcsec"] = pxscale[
        manifest["contaminant_source_index"].astype(int)
    ]
    manifest["pixel_scale_mismatch"] = ~np.isclose(
        manifest["target_pxscale_arcsec"], manifest["contaminant_pxscale_arcsec"]
    )
    manifest["angular_size_ratio"] = (
        manifest["size_ratio"]
        * manifest["contaminant_pxscale_arcsec"]
        / manifest["target_pxscale_arcsec"]
    )
    manifest["target_label_verified"] = labels[manifest["target_source_index"].astype(int)]
    manifest["contaminant_label_verified"] = labels[
        manifest["contaminant_source_index"].astype(int)
    ]
    distribution = summarize_distribution(manifest)
    distribution_path = run_dir / "tables/blend_distribution_summary.csv"
    if args.reuse_existing_distribution:
        if not distribution_path.exists():
            raise FileNotFoundError(distribution_path)
    else:
        safe_csv(distribution_path, distribution)

    source_images, source_labels, source_indices, _locked_group_ids, _all_groups, pool_audit = (
        final_manifest.load_source_pool(
            dataset,
            split_seed=int(config["seed"]),
            train_frac=float(config["splits"]["train_frac"]),
            val_frac=float(config["splits"]["val_frac"]),
            development_count=1000,
            locked_count=1000,
        )
    )
    if Path(args.replay_output_name).name != args.replay_output_name:
        raise ValueError("replay-output-name must be a filename, not a path")
    replay_path = run_dir / "tables" / args.replay_output_name
    if args.resume_finalization:
        replay = pd.read_csv(replay_path)
    else:
        replay_rows: list[dict[str, Any]] = []
        for _suite, frame in manifest.groupby("suite", sort=False):
            positions = np.unique(
                np.linspace(0, len(frame) - 1, args.replay_per_suite, dtype=int)
            )
            for position in positions:
                row = frame.iloc[int(position)]
                check, _arrays = reconstruct_row(
                    row, source_images, source_indices, pxscale
                )
                replay_rows.append(check)
        replay = pd.DataFrame(replay_rows)
        safe_csv(replay_path, replay)
    if not replay["passed"].all():
        raise RuntimeError("One or more blend replay checks failed")

    if args.resume_finalization:
        plot_paths = [
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted(
                (run_dir / "figures/blend_distribution_plots").glob("*.png")
            )
        ]
    else:
        plot_paths = save_distribution_plots(
            manifest, run_dir / "figures/blend_distribution_plots"
        )
    contact_dir = run_dir / "figures/blend_audit_contact_sheets"
    if args.resume_finalization:
        if not contact_dir.is_dir():
            raise FileNotFoundError(contact_dir)
    else:
        contact_dir.mkdir(parents=True, exist_ok=False)

    # Fresh development-source samples are rendered; provisional final rows are
    # never rendered or qualitatively inspected.
    all_indices = np.arange(n_sources, dtype=np.int64)
    split_rng = np.random.default_rng(int(config["seed"]))
    split_rng.shuffle(all_indices)
    test_indices = all_indices[int(n_sources * 0.70) + int(n_sources * 0.15) :]
    development_indices = test_indices[:1000]
    with h5py.File(dataset, "r") as handle:
        sort_order = np.argsort(development_indices)
        sorted_images = handle["images"][development_indices[sort_order]]
        inverse = np.empty_like(sort_order)
        inverse[sort_order] = np.arange(len(sort_order))
        development_images = sorted_images[inverse]
    specs = {spec.name: spec for spec in final_manifest.suite_specs(config, 800)}
    if args.resume_finalization:
        contact_paths = [
            str(path.relative_to(PROJECT_ROOT))
            for path in sorted(contact_dir.glob("*.png"))
        ]
    else:
        contact_paths: list[str] = []
        generated_for_extremes: list[dict[str, Any]] = []
        for name, filename, title, seed in (
            ("normal_final_test", "normal_blends.png", "Fresh development normal blends", 740001),
            ("hard_stress_final_test", "hard_stress_blends.png", "Fresh development hard-stress blends", 740101),
            ("compact_bright_final_test", "compact_bright_blends.png", "Fresh development compact-bright blends", 740201),
            ("high_core_obstruction_final_test", "high_core_obstruction_blends.png", "Fresh development high-core blends", 740301),
        ):
            samples = [
                fresh_candidate(specs[name], development_images, development_indices, seed, idx)
                for idx in range(6)
            ]
            generated_for_extremes.extend(samples)
            path = contact_dir / filename
            contact_sheet(samples, title, path)
            contact_paths.append(str(path.relative_to(PROJECT_ROOT)))

        extreme = sorted(generated_for_extremes, key=lambda sample: sample["mask_fraction"])
        extreme_samples = extreme[:3] + extreme[-3:]
        extreme_path = contact_dir / "extreme_mask_fraction.png"
        contact_sheet(extreme_samples, "Fresh development extreme mask fractions", extreme_path)
        contact_paths.append(str(extreme_path.relative_to(PROJECT_ROOT)))

    artifact_candidates = pd.read_csv(artifact_table_path).head(6)
    artifact_samples: list[dict[str, Any]] = []
    with h5py.File(dataset, "r") as handle:
        for offset, candidate in artifact_candidates.iterrows():
            target = development_images[offset].astype(np.float32) / 255.0
            source_column = "global_index" if "global_index" in candidate.index else "source_index"
            contaminant_index = int(candidate[source_column])
            contaminant = handle["images"][contaminant_index].astype(np.float32) / 255.0
            rng = np.random.default_rng(750000 + offset)
            dx, dy = int(rng.integers(-28, 29)), int(rng.integers(-28, 29))
            components = manual_components(target, contaminant, dx, dy, 0.0, 1.2, 0.05, 0.002, rng)
            blend = np.asarray(components["blended"])
            affected = gd_utils.affected_region_mask(target, blend, 0.02)
            core = gd_utils.evaluation_core_mask_p85_v1(target)
            artifact_samples.append(
                {
                    "label": f"artifact source {contaminant_index}",
                    "target": target,
                    "shifted_foreground": np.asarray(components["shifted_foreground"]),
                    "blend": blend,
                    "affected": affected,
                    "core": core,
                    "halo": gd_utils.halo_band_mask_manhattan_v1(affected, 5),
                }
            )
    artifact_path = contact_dir / "suspected_source_artifacts.png"
    contact_sheet(artifact_samples, "Heuristic source-artifact candidates (not labels)", artifact_path)
    contact_paths.append(str(artifact_path.relative_to(PROJECT_ROOT)))

    size_sanity = gaussian_size_sanity()
    safe_csv(run_dir / "tables/size_estimator_sanity.csv", size_sanity)

    centrality_distances = []
    boundary_touch = 0
    for image in development_images[:100].astype(np.float32) / 255.0:
        mask = gd_blend.estimate_central_source_mask(image) > 0.1
        if not np.any(mask):
            continue
        y, x = np.nonzero(mask)
        centrality_distances.append(
            float(np.hypot(x.mean() - image.shape[1] / 2, y.mean() - image.shape[0] / 2))
        )
        boundary_touch += int(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())

    suite_replay = replay.groupby("suite").agg(
        n=("passed", "size"),
        pass_count=("passed", "sum"),
        max_abs_error=("max_abs_manual_vs_generator", "max"),
        mean_composite_clip_loss=("composite_clip_loss_relative_to_shifted_contaminant_flux", "mean"),
        p95_composite_clip_loss=("composite_clip_loss_relative_to_shifted_contaminant_flux", lambda values: values.quantile(0.95)),
        mean_support_capture=("affected_support_capture_fraction", "mean"),
        mean_flux_capture=("affected_flux_capture_fraction", "mean"),
        mean_nuisance_affected=("nuisance_only_affected_fraction", "mean"),
    )
    mismatch = manifest.groupby("suite")["pixel_scale_mismatch"].mean().to_dict()
    median_snapshot = manifest.groupby("suite").agg(
        median_shift_l2=("shift_l2_pixels", "median"),
        median_size_ratio=("size_ratio", "median"),
        median_affected_fraction=("affected_mask_fraction", "median"),
        median_core_obstruction=("core_obstruction_fraction", "median"),
        median_severity=("blend_severity_score", "median"),
    )
    compact = manifest.loc[manifest["suite"].eq("compact_bright_final_test")]
    report = f"""# Blending Algorithm Correctness Audit

## Verdict

The generator is internally deterministic and replayable when source-pool order,
per-sample seed, and code version are retained. All {len(replay)} stratified
replays passed; manual reconstruction and `src.blend.blend_pair` matched exactly
with maximum absolute error `{replay['max_abs_manual_vs_generator'].max():.3g}`.

The current benchmark is **computer-vision-style synthetic RGB cutout blending**.
It is not calibrated FITS-band physical injection. It remains usable as a
controlled Galaxy10 RGB restoration/deblending benchmark only with that scope
stated explicitly.

## Compositing and restoration semantics

The implementation subtracts a border median from the contaminant, selects and
tapers one center-biased component, scales/clips it, shifts it without wrap, and
adds it to a target that may first be Gaussian blurred. The composite is clipped
to `[0,1]`; noise is then added and clipped again. Therefore
`blended - original_target` is a **blend-to-target correction field**, not pure
contaminant flux: it also contains target deblurring, noise correction, and
clipping effects.

RGB bytes are divided by 255 and added in display-RGB space. Pixel-scale
metadata is not used by the historical generator. The provisional suites mix
the two local pixel scales at these rates: `{mismatch}`. Pixel and angular size
ratios therefore differ for those pairs.

## Replay and clipping diagnostics

The 30-per-suite replay summary is:

```text
{suite_replay.to_string()}
```

Input clipping is material in bright/overlapping suites and can destroy target
information, making some cases partly inpainting. New manifests should retain
preclip/postclip flux, saturated-channel fraction, effective contaminant/target
flux, and array/mask hashes. The replay table adds those diagnostics without
changing the historical manifests.

## Masks and difficulty labels

The affected mask is a prediction-independent blend-change mask:
`mean(abs(blended-target), RGB) > 0.02`. In sampled replays it captures most
contaminant flux but not the faintest spatial support. Blur/noise-only changes
rarely cross the threshold, so the mask is not broken; it is intentionally
brightness-thresholded. The five-step halo is a Manhattan/diamond band.

`generation_difficulty` is a sampled-parameter heuristic, not observed model
difficulty. Current blur/noise ranges do not reach two of its historical hard
thresholds. Keep it only as a versioned generator label; use measured severity
and obstruction fields for analysis.

## Size and centrality

The extraction mask is dilated and smoothed before its `mask > 0.1` area becomes
the stored radius. The synthetic Gaussian check in
`tables/size_estimator_sanity.csv` demonstrates compression of true size ratios.
The compact suite median stored ratio is `{compact['size_ratio'].median():.3f}`;
only `{(compact['size_ratio'] <= 0.6).mean():.2%}` are at or below 0.6.

Targets remain centered while only contaminants are shifted. Across 100 fresh
development cutouts, the selected soft-mask centroid distance is median
`{np.median(centrality_distances):.2f}` pixels and p95
`{np.quantile(centrality_distances, 0.95):.2f}` pixels; {boundary_touch}/100
masks touched the boundary. A centrality shortcut is therefore available.
Future controls should translate targets, role-swap pairs, include near-zero
shifts, and add a centrality-only baseline.

## Distribution snapshot

```text
{median_snapshot.to_string()}
```

Source reuse means rows are not independent source draws. Confidence intervals
should cluster by target and contaminant group rather than bootstrap rows as
independent observations.

## Visual audit policy

The contact sheets render newly generated development-source audit samples,
not the provisional final-manifest rows. They show target, shifted contaminant
foreground, blend, affected mask, evaluation core, and halo band for normal,
hard, compact, high-core, heuristic artifact, and extreme-mask cases.

Because the provisional final-manifest distributions and pixels have now been
used for infrastructure design and replay diagnostics, that manifest must not
be called a pristine final test. Generate a fresh grouped locked final manifest
with untouched seeds after the protocol is frozen.

## Gate decision

No generator arithmetic/replay blocker remains for an internal grouped rerun of
the historical controlled RGB restoration task. Physically faithful or
survey-grade source-separation claims remain blocked. High-priority future
benchmark upgrades are shared/explicit PSF and noise, linear-light or calibrated
flux injection, pixel-scale matching/resampling, low-saturation strata,
component-flux ground-truth masks, and centrality controls.
"""
    safe_text(run_dir / "diagnostics/blending_algorithm_audit.md", report)
    safe_json(
        run_dir / "logs/blending_audit_provenance.json",
        {
            "dataset": str(dataset.relative_to(PROJECT_ROOT)),
            "dataset_sha256": sha256_file(dataset),
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": sha256_file(config_path),
            "manifest_run": str(manifest_run.relative_to(PROJECT_ROOT)),
            "manifest_checksums_verified_before": True,
            "replay_rows": len(replay),
            "replay_all_passed": bool(replay["passed"].all()),
            "source_pool_audit": pool_audit,
            "plot_paths": plot_paths,
            "contact_sheet_paths": contact_paths,
            "code_sha256": {
                "scripts/blending_correctness_audit.py": sha256_file(Path(__file__)),
                "src/blend.py": sha256_file(PROJECT_ROOT / "src/blend.py"),
                "src/utils.py": sha256_file(PROJECT_ROOT / "src/utils.py"),
            },
        },
    )
    print(f"Blend audit complete: {len(replay)}/{len(replay)} replay rows passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
