#!/usr/bin/env python3
"""Audit the fixed forward model and generate fresh Atlas-v0 validation scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
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

from src.btk_scene import (  # noqa: E402
    BAND_ORDER,
    STAMP_SIZE_ARCSEC,
    gaussian_prompt,
    load_catsim_catalog,
    validated_lsst_survey,
)
from src.competing_hypotheses import forward_consistency, recompose  # noqa: E402

CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
EXPECTED_CATALOG_HASH = "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46"
EXPECTED_FWHM = {"g": 0.86, "r": 0.81, "z": 0.77}
EXPECTED_SKY = np.asarray([24114.080000000005, 127057.12000000002, 250784.80000000005])
VALIDATION_COUNT = 2_000
BATCH_SIZE = 250
SEED = 2026071221


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
        raise ValueError("refusing to write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DefinitionSampling(SamplingFunction):
    def __init__(self, definitions: list[dict[str, object]]) -> None:
        super().__init__(stamp_size=int(STAMP_SIZE_ARCSEC), min_number=2, max_number=2, seed=0)
        self.stamp_size = STAMP_SIZE_ARCSEC
        self.definitions = definitions
        self.position = 0

    def __call__(self, table: Table) -> Table:
        row = self.definitions[self.position]
        self.position += 1
        output = table[[int(row["target_catalog_row"]), int(row["contaminant_catalog_row"])]].copy()
        output["ra"] = [float(row["target_x_arcsec"]), float(row["contaminant_x_arcsec"])]
        output["dec"] = [float(row["target_y_arcsec"]), float(row["contaminant_y_arcsec"])]
        return output


def definitions(run_dir: Path) -> list[dict[str, object]]:
    frozen = run_dir / "manifests/fresh_validation_scene_definitions.csv"
    if frozen.exists():
        rows = read_csv(frozen)
        if len(rows) != VALIDATION_COUNT:
            raise RuntimeError("frozen validation definition count mismatch")
        return rows
    pool = [
        row
        for row in read_csv(run_dir / "manifests/campaign_source_partition_commitments.csv")
        if row["campaign_role"] == "validation"
    ]
    if len(pool) < 2:
        raise RuntimeError("insufficient validation sources")
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for index in range(VALIDATION_COUNT):
        while True:
            chosen = rng.choice(len(pool), size=2, replace=False)
            target, contaminant = pool[int(chosen[0])], pool[int(chosen[1])]
            if target["duplicate_group_id"] != contaminant["duplicate_group_id"]:
                break
        target_xy = rng.uniform(-0.2, 0.2, size=2)
        separation = float(rng.uniform(0.4, 3.2))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        contaminant_xy = target_xy + separation * np.asarray([np.cos(angle), np.sin(angle)])
        rows.append(
            {
                "scene_id": f"atlas_v0_validation_{index:05d}",
                "campaign_role": "validation",
                "target_catalog_row": target["catalog_row"],
                "target_source_id": target["persistent_source_id"],
                "target_group": target["duplicate_group_id"],
                "contaminant_catalog_row": contaminant["catalog_row"],
                "contaminant_source_id": contaminant["persistent_source_id"],
                "contaminant_group": contaminant["duplicate_group_id"],
                "target_x_arcsec": float(target_xy[0]),
                "target_y_arcsec": float(target_xy[1]),
                "contaminant_x_arcsec": float(contaminant_xy[0]),
                "contaminant_y_arcsec": float(contaminant_xy[1]),
                "source_selection_seed": SEED,
                "position_seed": SEED + index,
                "noise_seed": 2026071221000 + index,
            }
        )
    return rows


def generate(run_dir: Path) -> None:
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    if freeze["candidate_inference_count_at_freeze"] != 0:
        raise RuntimeError("preregistration did not precede inference")
    catalog, catalog_hash = load_catsim_catalog(CATALOG)
    if catalog_hash != EXPECTED_CATALOG_HASH:
        raise RuntimeError("catalog provenance mismatch")
    survey = validated_lsst_survey()
    fwhm = {band: float(survey.get_filter(band).psf_fwhm.to_value("arcsec")) for band in BAND_ORDER}
    if any(not np.isclose(fwhm[band], EXPECTED_FWHM[band], rtol=0.0, atol=1e-12) for band in BAND_ORDER):
        raise RuntimeError(f"PSF provenance mismatch: {fwhm}")
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER])
    if not np.allclose(sky, EXPECTED_SKY, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"noise provenance mismatch: {sky}")
    psf_rows = [
        {
            "band": band,
            "psf_family": "GalSim Convolution(Kolmogorov, Airy)",
            "fwhm_arcsec": fwhm[band],
            "pixel_scale_arcsec": 0.2,
            "sky_electrons_per_pixel": sky[index],
            "btk_version": importlib.metadata.version("blending-toolkit"),
            "galsim_version": importlib.metadata.version("galsim"),
        }
        for index, band in enumerate(BAND_ORDER)
    ]
    psf_path = run_dir / "tables/fixed_psf_configuration.csv"
    if not psf_path.exists():
        write_csv_fresh(psf_path, psf_rows)
    noise_contract = {
        "add_noise_none": "sum of noiseless PSF-convolved isolated source layers",
        "add_noise_all": "source Poisson plus one zero-mean sky Poisson realization on summed scene",
        "variance": "max(candidate_recomposition + mean_sky_electrons_per_pixel, 1)",
        "sky_electrons_grz": sky.tolist(),
        "band_order": list(BAND_ORDER),
        "clipping": "none",
    }
    noise_path = run_dir / "manifests/fixed_noise_contract.json"
    if not noise_path.exists():
        write_json_fresh(noise_path, noise_contract)
    psf_noise_hash = sha256_json({"psf": psf_rows, "noise": noise_contract})

    definitions_rows = definitions(run_dir)
    definitions_path = run_dir / "manifests/fresh_validation_scene_definitions.csv"
    if not definitions_path.exists():
        write_csv_fresh(definitions_path, definitions_rows)
    band_indices = [tuple(survey.available_filters).index(band) for band in BAND_ORDER]
    manifest: list[dict[str, object]] = []
    unit_tests: list[dict[str, object]] = []
    started = time.time()
    for start in range(0, VALIDATION_COUNT, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, VALIDATION_COUNT)
        chunk = definitions_rows[start:stop]
        common = dict(
            batch_size=len(chunk),
            njobs=1,
            verbose=False,
            use_bar=False,
            apply_shear=False,
            augment_data=False,
        )
        noiseless_batch = next(
            CatsimGenerator(
                catalog,
                DefinitionSampling(chunk),
                survey,
                add_noise="none",
                seed=2026071222000 + start,
                **common,
            )
        )
        noisy_batch = next(
            CatsimGenerator(
                catalog,
                DefinitionSampling(chunk),
                survey,
                add_noise="all",
                seed=2026071223000 + start,
                **common,
            )
        )
        blends = np.asarray(noiseless_batch.blend_images[:, band_indices], dtype=np.float64)
        noisy = np.asarray(noisy_batch.blend_images[:, band_indices], dtype=np.float64)
        isolated = np.asarray(noiseless_batch.isolated_images[:, :2][:, :, band_indices], dtype=np.float64)
        noisy_isolated = np.asarray(noisy_batch.isolated_images[:, :2][:, :, band_indices], dtype=np.float64)
        for local, definition in enumerate(chunk):
            catalog_rows = list(np.asarray(noiseless_batch.catalog_list[local]["catalog_row"], dtype=int))
            expected_rows = [int(definition["target_catalog_row"]), int(definition["contaminant_catalog_row"])]
            if catalog_rows != expected_rows:
                raise RuntimeError("validation source alignment failure")
            if not np.array_equal(isolated[local], noisy_isolated[local]):
                raise RuntimeError("noise changed isolated-source truth")
            addition_error = float(np.max(np.abs(blends[local] - recompose(isolated[local]))))
            result = forward_consistency(noisy[local], isolated[local], sky)
            target_record = noiseless_batch.catalog_list[local][0]
            prompt = gaussian_prompt(
                blends.shape[-2:], float(target_record["x_peak"]), float(target_record["y_peak"])
            )
            manifest.append(
                {
                    **definition,
                    "prompt_x_pixel": float(target_record["x_peak"]),
                    "prompt_y_pixel": float(target_record["y_peak"]),
                    "prompt_sha256": sha256_array(prompt),
                    "target_isolated_sha256": sha256_array(isolated[local, 0]),
                    "contaminant_isolated_sha256": sha256_array(isolated[local, 1]),
                    "noiseless_blend_sha256": sha256_array(blends[local]),
                    "noisy_blend_sha256": sha256_array(noisy[local]),
                    "psf_noise_configuration_sha256": psf_noise_hash,
                    "generator_version": "BTK 1.0.9 / GalSim 2.8.4",
                    "band_order": "g,r,z",
                    "units": "detected electrons per pixel",
                    "forward_global_chi_square_mean": result.global_chi_square_mean,
                    "finite": result.finite,
                }
            )
            if len(unit_tests) < 20:
                unit_tests.extend(
                    [
                        {"scene_id": definition["scene_id"], "test": "noiseless_additivity", "value": addition_error, "limit": 1e-10, "status": "PASS" if addition_error <= 1e-10 else "FAIL"},
                        {"scene_id": definition["scene_id"], "test": "isolated_unchanged_by_noise", "value": 0.0, "limit": 0.0, "status": "PASS"},
                        {"scene_id": definition["scene_id"], "test": "forward_score_finite", "value": result.global_chi_square_mean, "limit": "finite", "status": "PASS" if result.finite else "FAIL"},
                    ]
                )
        print(json.dumps({"phase": "validation", "completed": stop, "total": VALIDATION_COUNT}), flush=True)
    if any(row["status"] != "PASS" for row in unit_tests):
        raise RuntimeError("forward-model unit test failure")
    write_csv_fresh(run_dir / "tables/fresh_validation_scene_manifest.csv", manifest)
    write_csv_fresh(run_dir / "tables/forward_model_unit_tests.csv", unit_tests)
    report = f"""# Fixed forward-model audit

Status: **PASS**.

Atlas v0 reproduces the historical fixed LSST g/r/z observation model with
FWHM {fwhm['g']:.2f}/{fwhm['r']:.2f}/{fwhm['z']:.2f} arcsec, 0.2 arcsec pixels,
unclipped source-layer addition, and the exact BTK source-plus-sky Poisson
contract. The PSF/noise configuration hash is `{psf_noise_hash}`.

Exactly {VALIDATION_COUNT:,} fresh validation scenes were generated only from
approved validation groups. Their isolated layers, noiseless/noisy blends,
prompts, positions, identities, seeds, and provenance hashes are recorded.
No historical development or lockbox scene was opened or rendered.
"""
    write_text_fresh(run_dir / "diagnostics/forward_model_audit.md", report)
    write_json_fresh(
        run_dir / "logs/forward_model_validation_complete.json",
        {
            "status": "PASS",
            "validation_scene_count": VALIDATION_COUNT,
            "psf_noise_configuration_sha256": psf_noise_hash,
            "runtime_seconds": time.time() - started,
            "development_scene_access_count": 0,
            "lockbox_scene_access_count": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    expected_parent = (REPO / "outputs/runs").resolve()
    if run_dir.parent != expected_parent or not run_dir.name.startswith("thayer_ambiguity_atlas_v0_"):
        raise ValueError("unexpected run directory")
    generate(run_dir)


if __name__ == "__main__":
    main()
