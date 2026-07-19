#!/usr/bin/env python3
"""Run bounded, catalog-parameter counterfactual optimization for Atlas v0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table
from btk.draw_blends import CatsimGenerator
from btk.sampling_functions import SamplingFunction
from surveycodex.utilities import mean_sky_level

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btk_scene import BAND_ORDER, STAMP_SIZE_ARCSEC, load_catsim_catalog, validated_lsst_survey  # noqa: E402
from src.competing_hypotheses import scientific_distance  # noqa: E402

CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
PAIR_COUNT = 25
IDENTITIES_PER_PAIR = 4
NUISANCE_TRIALS = 20
SEED = 2026071224
FLUX_COLUMNS = ("fluxnorm_bulge", "fluxnorm_disk", "fluxnorm_agn")
MAG_COLUMNS = ("u_ab", "g_ab", "r_ab", "i_ab", "z_ab", "y_ab")
ANGLE_COLUMNS = ("pa_bulge", "pa_disk")


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("refusing empty optimization table")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CounterfactualSampling(SamplingFunction):
    def __init__(
        self,
        target_row: int,
        contaminant_row: int,
        target_xy: tuple[float, float],
        contaminant_xy: tuple[float, float],
        flux_scale: float,
        orientation_delta_deg: float,
    ) -> None:
        super().__init__(stamp_size=int(STAMP_SIZE_ARCSEC), min_number=2, max_number=2, seed=0)
        self.stamp_size = STAMP_SIZE_ARCSEC
        self.target_row = target_row
        self.contaminant_row = contaminant_row
        self.target_xy = target_xy
        self.contaminant_xy = contaminant_xy
        self.flux_scale = flux_scale
        self.orientation_delta_deg = orientation_delta_deg

    def __call__(self, table: Table) -> Table:
        output = table[[self.target_row, self.contaminant_row]].copy()
        output["ra"] = [self.target_xy[0], self.contaminant_xy[0]]
        output["dec"] = [self.target_xy[1], self.contaminant_xy[1]]
        for column in FLUX_COLUMNS:
            output[column][1] = float(output[column][1]) * self.flux_scale
        magnitude_delta = -2.5 * np.log10(self.flux_scale)
        for column in MAG_COLUMNS:
            output[column][1] = float(output[column][1]) + magnitude_delta
        for column in ANGLE_COLUMNS:
            output[column][1] = (float(output[column][1]) + self.orientation_delta_deg) % 360.0
        return output


def render(
    catalog,
    survey,
    *,
    target_row: int,
    contaminant_row: int,
    target_xy: tuple[float, float],
    contaminant_xy: tuple[float, float],
    flux_scale: float,
    orientation_delta_deg: float,
    noise_seed: int,
):
    generator = CatsimGenerator(
        catalog,
        CounterfactualSampling(
            target_row,
            contaminant_row,
            target_xy,
            contaminant_xy,
            flux_scale,
            orientation_delta_deg,
        ),
        survey,
        batch_size=1,
        njobs=1,
        verbose=False,
        use_bar=False,
        add_noise="none",
        seed=noise_seed,
        apply_shear=False,
        augment_data=False,
    )
    batch = next(generator)
    band_indices = [tuple(survey.available_filters).index(band) for band in BAND_ORDER]
    blend = np.asarray(batch.blend_images[0, band_indices], dtype=np.float64)
    isolated = np.asarray(batch.isolated_images[0, :2][:, band_indices], dtype=np.float64)
    return blend, isolated


def nontrivial_rescaling(left: np.ndarray, right: np.ndarray) -> tuple[bool, float]:
    denominator = float(np.sum(right * right))
    if denominator <= np.finfo(float).tiny:
        return False, 0.0
    alpha = float(np.sum(left * right) / denominator)
    residual = float(np.linalg.norm(left - alpha * right) / (np.linalg.norm(left) + np.finfo(float).tiny))
    return residual > 0.01, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_ambiguity_atlas_v0_"):
        raise ValueError("unexpected run directory")
    if not (run_dir / "preregistration/freeze_record.json").exists():
        raise RuntimeError("missing preregistration freeze")
    optimization_dir = run_dir / "optimization"
    if not optimization_dir.exists():
        optimization_dir.mkdir()
    output_manifest = run_dir / "tables/targeted_optimization_pair_manifest.csv"
    if output_manifest.exists():
        raise FileExistsError(output_manifest)

    pairs = read_csv(run_dir / "tables/atlas_pair_manifest.csv")[:PAIR_COUNT]
    definitions = read_csv(run_dir / "manifests/atlas_pool_scene_definitions.csv")
    commitments = [
        row
        for row in read_csv(run_dir / "manifests/campaign_source_partition_commitments.csv")
        if row["campaign_role"] == "training"
    ]
    rng = np.random.default_rng(SEED)
    finite_pool_indices = rng.choice(len(commitments), size=64, replace=False)
    finite_pool = [commitments[int(index)] for index in finite_pool_indices]
    finite_pool_hash = sha256_json(
        [{"catalog_row": row["catalog_row"], "group": row["duplicate_group_id"]} for row in finite_pool]
    )
    catalog, _ = load_catsim_catalog(CATALOG)
    survey = validated_lsst_survey()
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER])
    mean_psf_pixel = float(np.mean([survey.get_filter(b).psf_fwhm.to_value("arcsec") for b in BAND_ORDER]) / 0.2)
    trajectories: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    started = time.time()
    for pair_index, pair in enumerate(pairs):
        left_def = definitions[int(pair["left_pool_index"])]
        right_def = definitions[int(pair["right_pool_index"])]
        excluded_groups = {
            left_def["target_group"],
            left_def["contaminant_group"],
            right_def["target_group"],
            right_def["contaminant_group"],
        }
        identity_options = [row for row in finite_pool if row["duplicate_group_id"] not in excluded_groups]
        identity_options = identity_options[: IDENTITIES_PER_PAIR - 1]
        identity_options.insert(
            0,
            {
                "catalog_row": right_def["contaminant_catalog_row"],
                "duplicate_group_id": right_def["contaminant_group"],
                "persistent_source_id": right_def["contaminant_source_id"],
            },
        )
        left_target_xy = (float(left_def["target_x_arcsec"]), float(left_def["target_y_arcsec"]))
        left_contaminant_xy = (float(left_def["contaminant_x_arcsec"]), float(left_def["contaminant_y_arcsec"]))
        right_target_xy = (float(right_def["target_x_arcsec"]), float(right_def["target_y_arcsec"]))
        right_original_xy = np.asarray(
            [float(right_def["contaminant_x_arcsec"]), float(right_def["contaminant_y_arcsec"])]
        )
        left_blend, left_isolated = render(
            catalog,
            survey,
            target_row=int(left_def["target_catalog_row"]),
            contaminant_row=int(left_def["contaminant_catalog_row"]),
            target_xy=left_target_xy,
            contaminant_xy=left_contaminant_xy,
            flux_scale=1.0,
            orientation_delta_deg=0.0,
            noise_seed=SEED + pair_index,
        )
        truth_distance = scientific_distance(
            left_isolated[0],
            np.load(run_dir / f"atlas/{pair['pair_id']}.npz", allow_pickle=False)["right_isolated"][0],
            mean_psf_fwhm_pixel=mean_psf_pixel,
        )
        trials: list[dict[str, object]] = []
        for identity_rank, identity in enumerate(identity_options):
            trials.append(
                {
                    "phase": "identity_search",
                    "trial_seed": SEED + pair_index * 1000 + identity_rank,
                    "contaminant_catalog_row": int(identity["catalog_row"]),
                    "contaminant_group": identity["duplicate_group_id"],
                    "x_arcsec": float(right_original_xy[0]),
                    "y_arcsec": float(right_original_xy[1]),
                    "flux_scale": 1.0,
                    "orientation_delta_deg": 0.0,
                }
            )
        for trial_index in range(NUISANCE_TRIALS):
            trial_rng = np.random.default_rng(SEED + pair_index * 1000 + 100 + trial_index)
            identity = identity_options[int(trial_rng.integers(0, len(identity_options)))]
            position = np.clip(right_original_xy + trial_rng.uniform(-0.6, 0.6, size=2), -4.5, 4.5)
            trials.append(
                {
                    "phase": "bounded_nuisance_search",
                    "trial_seed": SEED + pair_index * 1000 + 100 + trial_index,
                    "contaminant_catalog_row": int(identity["catalog_row"]),
                    "contaminant_group": identity["duplicate_group_id"],
                    "x_arcsec": float(position[0]),
                    "y_arcsec": float(position[1]),
                    "flux_scale": float(np.exp(trial_rng.uniform(np.log(0.5), np.log(2.0)))),
                    "orientation_delta_deg": float(trial_rng.uniform(0.0, 180.0)),
                }
            )
        best = None
        best_arrays = None
        for trial_index, trial in enumerate(trials):
            right_blend, right_isolated = render(
                catalog,
                survey,
                target_row=int(right_def["target_catalog_row"]),
                contaminant_row=int(trial["contaminant_catalog_row"]),
                target_xy=right_target_xy,
                contaminant_xy=(float(trial["x_arcsec"]), float(trial["y_arcsec"])),
                flux_scale=float(trial["flux_scale"]),
                orientation_delta_deg=float(trial["orientation_delta_deg"]),
                noise_seed=int(trial["trial_seed"]),
            )
            variance = np.maximum(0.5 * (left_blend + right_blend) + sky[:, None, None], 1.0)
            blend_distance = float(np.mean((left_blend - right_blend) ** 2 / variance))
            objective = blend_distance / (truth_distance.primary_normalized + 1e-12)
            row = {
                "optimized_pair_id": f"optimized_pair_{pair_index + 1:04d}",
                "source_pair_id": pair["pair_id"],
                "trial_index": trial_index,
                **trial,
                "blend_whitened_mse": blend_distance,
                "truth_primary_diameter": truth_distance.primary_normalized,
                "objective": objective,
                "right_blend_sha256": sha256_array(right_blend),
                "right_target_sha256": sha256_array(right_isolated[0]),
                "right_contaminant_sha256": sha256_array(right_isolated[1]),
            }
            trajectories.append(row)
            if best is None or objective < float(best["objective"]):
                best = row
                best_arrays = (right_blend, right_isolated)
        assert best is not None and best_arrays is not None
        replay_blend, replay_isolated = render(
            catalog,
            survey,
            target_row=int(right_def["target_catalog_row"]),
            contaminant_row=int(best["contaminant_catalog_row"]),
            target_xy=right_target_xy,
            contaminant_xy=(float(best["x_arcsec"]), float(best["y_arcsec"])),
            flux_scale=float(best["flux_scale"]),
            orientation_delta_deg=float(best["orientation_delta_deg"]),
            noise_seed=int(best["trial_seed"]),
        )
        replay_pass = sha256_array(replay_blend) == sha256_array(best_arrays[0]) and sha256_array(replay_isolated) == sha256_array(best_arrays[1])
        scaling_pass, scaling_residual = nontrivial_rescaling(left_blend, best_arrays[0])
        valid = bool(
            float(best["blend_whitened_mse"]) <= 0.25
            and truth_distance.primary_normalized > 1.0
            and replay_pass
            and scaling_pass
            and np.all(np.isfinite(best_arrays[0]))
            and np.all(np.isfinite(best_arrays[1]))
        )
        output_path = run_dir / f"optimization/optimized_pair_{pair_index + 1:04d}.npz"
        if output_path.exists():
            raise FileExistsError(output_path)
        np.savez_compressed(
            output_path,
            left_blend=left_blend.astype(np.float32),
            right_blend=best_arrays[0].astype(np.float32),
            left_isolated=left_isolated.astype(np.float32),
            right_isolated=best_arrays[1].astype(np.float32),
            sky_electrons=sky,
        )
        outputs.append(
            {
                "optimized_pair_id": f"optimized_pair_{pair_index + 1:04d}",
                "source_pair_id": pair["pair_id"],
                "construction_route": "CONTROLLED_COUNTERFACTUAL_OPTIMIZATION",
                "finite_contaminant_pool_sha256": finite_pool_hash,
                "optimizer_seed": SEED + pair_index * 1000,
                "trial_count": len(trials),
                "flux_scale_lower": 0.5,
                "flux_scale_upper": 2.0,
                "position_delta_bound_arcsec": 0.6,
                "orientation_lower_deg": 0.0,
                "orientation_upper_deg": 180.0,
                "selected_contaminant_catalog_row": best["contaminant_catalog_row"],
                "selected_contaminant_group": best["contaminant_group"],
                "selected_x_arcsec": best["x_arcsec"],
                "selected_y_arcsec": best["y_arcsec"],
                "selected_flux_scale": best["flux_scale"],
                "selected_orientation_delta_deg": best["orientation_delta_deg"],
                "blend_whitened_mse": best["blend_whitened_mse"],
                "truth_primary_diameter": truth_distance.primary_normalized,
                "global_rescaling_relative_residual": scaling_residual,
                "exact_replay_pass": replay_pass,
                "physical_parameter_bounds_pass": True,
                "different_requested_source_groups": left_def["target_group"] != right_def["target_group"],
                "valid_pair": valid,
                "array_path": str(output_path.relative_to(run_dir)),
                "array_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }
        )
        print(json.dumps({"phase": "optimize", "completed": pair_index + 1, "total": PAIR_COUNT, "valid": sum(bool(row["valid_pair"]) for row in outputs)}), flush=True)
    write_csv_fresh(run_dir / "optimization/counterfactual_optimization_trajectories.csv", trajectories)
    write_csv_fresh(output_manifest, outputs)
    valid_count = sum(bool(row["valid_pair"]) for row in outputs)
    report = f"""# Controlled counterfactual optimization audit

Status: **{'PASS' if valid_count > 0 else 'FAIL_NO_VALID_PAIR'}**.

Route 2 optimized only catalog-level, physically bounded contaminant identity,
position, flux scale, and orientation through exact BTK/GalSim rendering. It
never optimized source pixels. All {len(trajectories):,} trials, seeds, bounds,
objectives, and hashes are preserved. {valid_count}/{PAIR_COUNT} final pairs
passed the frozen observation-distance, truth-divergence, replay, finite-array,
and trivial-rescaling gates. These are route-feasibility pairs; the separately
frozen initial Atlas remains the visually reviewed Route-1 subset.
"""
    write_text_fresh(run_dir / "diagnostics/targeted_optimization_report.md", report)
    write_json_fresh(
        run_dir / "logs/targeted_optimization_complete.json",
        {
            "status": "PASS" if valid_count > 0 else "FAIL_NO_VALID_PAIR",
            "pair_count": PAIR_COUNT,
            "valid_pair_count": valid_count,
            "trial_count": len(trajectories),
            "runtime_seconds": time.time() - started,
            "development_scene_access_count": 0,
            "lockbox_scene_access_count": 0,
        },
    )


if __name__ == "__main__":
    main()
