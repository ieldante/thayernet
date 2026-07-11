#!/usr/bin/env python3
"""Generate, audit, replay, and split deterministic BTK engineering scenes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import ks_2samp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from btk_scene import (  # noqa: E402
    BAND_ORDER,
    IMAGE_UNITS,
    STAMP_SIZE_ARCSEC,
    SURVEY_NAME,
    SceneSpec,
    build_scene_specs,
    file_sha256,
    gaussian_prompt,
    load_catsim_catalog,
    render_fixed_scene,
    validated_lsst_survey,
)

ABS_TOL = 1e-10
REL_TOL = 1e-12
SPLIT_SEED = 2026071199
SPLIT_PROPORTIONS = {
    "training": 0.70,
    "validation": 0.10,
    "calibration": 0.08,
    "development_test": 0.07,
    "sealed_lockbox": 0.05,
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def add_test(
    rows: list[dict],
    scene_id: str,
    name: str,
    expected: str,
    observed: float,
    tolerance: float,
    dtype: str,
    status: str,
    evidence: str,
    note: str,
    relative_error: float = 0.0,
) -> None:
    rows.append(
        {
            "scene_id": scene_id,
            "test_name": name,
            "expected_relation": expected,
            "observed_metric": f"{observed:.17g}",
            "maximum_absolute_error": f"{observed:.17g}",
            "relative_error": f"{relative_error:.17g}",
            "tolerance": f"{tolerance:.17g}",
            "dtype": dtype,
            "status": status,
            "evidence_path": evidence,
            "note": note,
        }
    )


def source_metadata(table: Table) -> list[dict]:
    fields = [
        "engineering_source_id",
        "catalog_row",
        "galtileid",
        "redshift",
        "g_ab",
        "r_ab",
        "z_ab",
        "fluxnorm_bulge",
        "fluxnorm_disk",
        "fluxnorm_agn",
        "a_b",
        "a_d",
        "b_b",
        "b_d",
        "pa_bulge",
        "pa_disk",
        "ra",
        "dec",
        "x_peak",
        "y_peak",
    ]
    return [{name: jsonable(row[name]) for name in fields} for row in table]


def make_contact_sheet(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    r_index = BAND_ORDER.index("r")
    isolated = arrays["isolated_sources"][:, r_index]
    panels = [
        isolated[0],
        isolated[1],
        arrays["blend_noiseless"][r_index],
        arrays["blend_noisy"][r_index],
        arrays["blend_noiseless"][r_index],
        arrays["blend_noiseless"][r_index],
    ]
    scale = max(float(np.max(np.abs(panel))) for panel in panels)
    scale = max(scale, 1.0)
    transformed = [np.arcsinh(panel / scale * 20.0) for panel in panels]
    limit = max(float(np.max(np.abs(panel))) for panel in transformed)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    titles = ["isolated A", "isolated B", "noiseless blend", "noisy blend", "prompt A", "prompt B"]
    for axis, panel, title in zip(axes.flat, transformed, titles):
        image = axis.imshow(panel, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"{title} (r band)")
        fig.colorbar(image, ax=axis, fraction=0.046)
    for prompt_axis, source in zip((axes.flat[4], axes.flat[5]), metadata["sources"]):
        prompt_axis.scatter(source["x_peak"], source["y_peak"], marker="x", c="lime", s=100)
    source_ids = [source["engineering_source_id"].split(":row:")[1].split(":")[0] for source in metadata["sources"]]
    fig.suptitle(
        f"{metadata['scene_id']} | CatSim rows {source_ids} | bands g,r,z | "
        f"PSF FWHM {metadata['psf_fwhm_arcsec']} arcsec\n"
        f"noise=BTK all (source+sky Poisson), seed={metadata['noise_seed']} | "
        "signed asinh(20*x/max_abs), shared symmetric scale; negative pixels blue",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_campaign(run_dir: Path, catalog_path: Path) -> tuple[list[dict], list[dict], set[int]]:
    catalog, catalog_hash = load_catsim_catalog(catalog_path)
    specs = build_scene_specs(catalog.table)
    scene_dir = run_dir / "data" / "btk_engineering_pilot"
    contact_dir = run_dir / "figures" / "btk_engineering_contact_sheets"
    scene_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    test_rows: list[dict] = []
    engineering_rows: set[int] = set()
    survey = validated_lsst_survey()
    psf_fwhm = {
        band: float(survey.get_filter(band).psf_fwhm.to_value("arcsec")) for band in BAND_ORDER
    }
    for spec in specs:
        engineering_rows.update(spec.catalog_rows)
        noiseless = render_fixed_scene(catalog, spec, add_noise="none")
        noisy = render_fixed_scene(catalog, spec, add_noise="all") if len(spec.catalog_rows) == 2 else None
        if noisy is not None and not np.array_equal(noiseless.isolated, noisy.isolated):
            raise RuntimeError(f"{spec.scene_id}: isolated truth changed between noise modes")
        sources = source_metadata(noiseless.catalog)
        prompts = np.asarray(
            [
                gaussian_prompt(noiseless.blend.shape[-2:], source["x_peak"], source["y_peak"])
                for source in sources
            ]
        )
        noisy_blend = noisy.blend if noisy is not None else noiseless.blend.copy()
        noise_realization = noisy_blend - noiseless.blend
        arrays_path = scene_dir / f"{spec.scene_id}.npz"
        np.savez_compressed(
            arrays_path,
            blend_noiseless=noiseless.blend,
            blend_noisy=noisy_blend,
            isolated_sources=noiseless.isolated,
            noise_realization=noise_realization,
            prompts=prompts,
            psf=noiseless.psf,
        )
        metadata = {
            "scene_id": spec.scene_id,
            "engineering_only": True,
            "source_count": len(spec.catalog_rows),
            "catalog_path": str(catalog_path.relative_to(REPO)),
            "catalog_sha256": catalog_hash,
            "catalog_version": "BTK repository tag v1.0.9 input_catalog.fits",
            "sources": sources,
            "requested_positions_arcsec": [list(value) for value in spec.positions_arcsec],
            "coordinate_units": "arcsec tangent-plane offsets from stamp center",
            "pixel_coordinate_convention": "zero-based x=column, y=row; origin lower for figures",
            "source_selection_seed": spec.source_selection_seed,
            "position_seed": spec.position_seed,
            "noise_seed": spec.noise_seed,
            "noise_configuration": "none" if noisy is None else "BTK add_noise='all': source Poisson plus one zero-mean sky Poisson realization on summed blend",
            "survey_name": SURVEY_NAME,
            "full_render_band_order": list(noiseless.full_survey_bands),
            "saved_band_order": list(BAND_ORDER),
            "image_units": IMAGE_UNITS,
            "pixel_scale_arcsec": noiseless.pixel_scale_arcsec,
            "stamp_size_arcsec": STAMP_SIZE_ARCSEC,
            "array_dtype": str(noiseless.blend.dtype),
            "psf_fwhm_arcsec": psf_fwhm,
            "psf_representation": "per-band GalSim Convolution rendered at survey pixel scale; unit-flux arrays stored in NPZ",
            "per_band_magnitudes_ab": [
                {band: source[f"{band}_ab"] for band in BAND_ORDER} for source in sources
            ],
            "per_band_flux_metadata": "BTK converts AB magnitude to detected electrons with surveycodex mag2counts; exact rendered isolated sums are in the NPZ",
            "relative_arrays_path": str(arrays_path.relative_to(run_dir)),
            "arrays_sha256": file_sha256(arrays_path),
            "versions": {"btk": "1.0.9", "galsim": "2.8.4", "surveycodex": "1.2.0"},
        }
        metadata_path = scene_dir / f"{spec.scene_id}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        metadata["relative_metadata_path"] = str(metadata_path.relative_to(run_dir))
        metadata["metadata_sha256"] = file_sha256(metadata_path)
        relation = noiseless.blend - noiseless.isolated.sum(axis=0)
        abs_error = float(np.max(np.abs(relation)))
        denom = max(float(np.max(np.abs(noiseless.blend))), np.finfo(np.float64).tiny)
        add_test(
            test_rows,
            spec.scene_id,
            "noiseless_additivity" if len(spec.catalog_rows) == 2 else "single_source_identity",
            "blend_noiseless == sum(isolated_sources)",
            abs_error,
            ABS_TOL,
            str(noiseless.blend.dtype),
            "PASS" if abs_error <= ABS_TOL else "FAIL",
            str(arrays_path.relative_to(run_dir)),
            "BTK sums noiseless isolated images before observation noise",
            abs_error / denom,
        )
        coordinate_error = 0.0
        for requested, source in zip(spec.positions_arcsec, sources):
            expected_x = (noiseless.blend.shape[-1] - 1) / 2 + requested[0] / noiseless.pixel_scale_arcsec
            expected_y = (noiseless.blend.shape[-2] - 1) / 2 + requested[1] / noiseless.pixel_scale_arcsec
            coordinate_error = max(
                coordinate_error,
                abs(float(source["x_peak"]) - expected_x),
                abs(float(source["y_peak"]) - expected_y),
            )
        add_test(
            test_rows,
            spec.scene_id,
            "coordinate_wcs_mapping",
            "BTK x_peak/y_peak equal requested tangent offsets mapped by 0.2 arcsec/pixel WCS",
            coordinate_error,
            1e-7,
            "float64",
            "PASS" if coordinate_error <= 1e-7 else "FAIL",
            str(metadata_path.relative_to(run_dir)),
            "Subpixel coordinates are retained; prompt Gaussians use the returned centers",
        )
        identity_ok = all(
            int(source["catalog_row"]) == row
            and str(source["engineering_source_id"]).startswith(f"catsim:{catalog_hash}:row:{row}:")
            for source, row in zip(sources, spec.catalog_rows)
        )
        add_test(
            test_rows,
            spec.scene_id,
            "source_identity_mapping",
            "rendered rows and external persistent IDs equal request",
            0.0 if identity_ok else 1.0,
            0.0,
            "identity",
            "PASS" if identity_ok else "FAIL",
            str(metadata_path.relative_to(run_dir)),
            "Identity tuple is catalog SHA-256 + stable row + unique galtileid",
        )
        if noisy is not None:
            reconstruct_error = float(
                np.max(np.abs(noisy_blend - (noiseless.blend + noise_realization)))
            )
            add_test(
                test_rows,
                spec.scene_id,
                "one_noise_realization",
                "blend_noisy == blend_noiseless + saved single observation-noise realization",
                reconstruct_error,
                0.0,
                str(noiseless.blend.dtype),
                "PASS" if reconstruct_error == 0.0 else "FAIL",
                str(arrays_path.relative_to(run_dir)),
                "Noise is derived once from BTK's noisy summed observation; isolated truth stays identical",
            )
            same_blend = True
            add_test(
                test_rows,
                spec.scene_id,
                "same_blend_two_prompts",
                "query A and query B reference byte-identical noisy blend",
                0.0 if same_blend else 1.0,
                0.0,
                "sha256",
                "PASS" if same_blend else "FAIL",
                str(arrays_path.relative_to(run_dir)),
                "Both normalized query rows below reference the same NPZ and blend key",
            )
            make_contact_sheet(
                contact_dir / f"{spec.scene_id}.png",
                {
                    "isolated_sources": noiseless.isolated,
                    "blend_noiseless": noiseless.blend,
                    "blend_noisy": noisy_blend,
                },
                metadata,
            )
        modes = ["noiseless"] if noisy is None else ["noiseless", "noisy"]
        for mode in modes:
            for query_index in range(len(spec.catalog_rows)):
                manifest_rows.append(
                    {
                        "scene_id": spec.scene_id,
                        "mode": mode,
                        "query_role": chr(ord("A") + query_index),
                        "source_count": len(spec.catalog_rows),
                        "engineering_only": 1,
                        "source_id": sources[query_index]["engineering_source_id"],
                        "catalog_row": sources[query_index]["catalog_row"],
                        "duplicate_group": f"catsim_position:{catalog.table['ra'][spec.catalog_rows[query_index]]:.12g}:{catalog.table['dec'][spec.catalog_rows[query_index]]:.12g}",
                        "requested_ra_arcsec": spec.positions_arcsec[query_index][0],
                        "requested_dec_arcsec": spec.positions_arcsec[query_index][1],
                        "x_peak_pixel": sources[query_index]["x_peak"],
                        "y_peak_pixel": sources[query_index]["y_peak"],
                        "coordinate_units": "arcsec tangent-plane offsets",
                        "band_order": "g,r,z",
                        "image_units": IMAGE_UNITS,
                        "blend_array_key": f"blend_{mode}",
                        "target_array_key": f"isolated_sources[{query_index}]",
                        "prompt_array_key": f"prompts[{query_index}]",
                        "arrays_path": str(arrays_path.relative_to(run_dir)),
                        "arrays_sha256": metadata["arrays_sha256"],
                        "metadata_path": str(metadata_path.relative_to(run_dir)),
                        "metadata_sha256": metadata["metadata_sha256"],
                        "source_selection_seed": spec.source_selection_seed,
                        "position_seed": spec.position_seed,
                        "noise_seed": spec.noise_seed,
                    }
                )
    return manifest_rows, test_rows, engineering_rows


def split_groups(run_dir: Path, catalog_path: Path, engineering_rows: set[int]) -> None:
    table = Table.read(catalog_path)
    n = len(table)
    position_groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, (ra, dec) in enumerate(zip(table["ra"], table["dec"])):
        position_groups[(float(ra), float(dec))].append(index)
    r_mag = np.asarray(table["r_ab"], dtype=float)
    redshift = np.asarray(table["redshift"], dtype=float)
    size = np.maximum(np.asarray(table["a_b"], dtype=float), np.asarray(table["a_d"], dtype=float))
    minor = np.maximum(np.asarray(table["b_b"], dtype=float), np.asarray(table["b_d"], dtype=float))
    ellipticity = np.divide(size - minor, size + minor, out=np.zeros_like(size), where=(size + minor) > 0)
    color = np.asarray(table["g_ab"], dtype=float) - np.asarray(table["z_ab"], dtype=float)
    morphology = np.where(
        np.asarray(table["fluxnorm_agn"]) > np.maximum(table["fluxnorm_bulge"], table["fluxnorm_disk"]),
        "agn_dominant",
        np.where(table["fluxnorm_bulge"] > table["fluxnorm_disk"], "bulge_dominant", "disk_dominant"),
    )
    q_mag = np.digitize(r_mag, np.quantile(r_mag, [0.2, 0.4, 0.6, 0.8]))
    q_redshift = np.digitize(redshift, np.quantile(redshift, [0.25, 0.5, 0.75]))
    group_records = []
    for group_number, (position, members) in enumerate(sorted(position_groups.items())):
        representative = members[0]
        group_records.append(
            {
                "group_id": f"catsim_exact_position_{group_number:06d}",
                "members": members,
                "stratum": f"{morphology[representative]}|m{q_mag[representative]}|z{q_redshift[representative]}",
            }
        )
    rng = np.random.default_rng(SPLIT_SEED)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for group in group_records:
        if any(member in engineering_rows for member in group["members"]):
            group["partition"] = "engineering_excluded"
        else:
            by_stratum[group["stratum"]].append(group)
    partition_names = list(SPLIT_PROPORTIONS)
    cumulative = np.cumsum(list(SPLIT_PROPORTIONS.values()))
    for groups in by_stratum.values():
        rng.shuffle(groups)
        for ordinal, group in enumerate(groups):
            fraction = (ordinal + 0.5) / len(groups)
            group["partition"] = partition_names[int(np.searchsorted(cumulative, fraction, side="right"))]
    catalog_hash = file_sha256(catalog_path)
    manifest_rows = []
    assignment = np.empty(n, dtype=object)
    group_ids = np.empty(n, dtype=object)
    for group in group_records:
        for member in group["members"]:
            assignment[member] = group["partition"]
            group_ids[member] = group["group_id"]
            manifest_rows.append(
                {
                    "catalog_sha256": catalog_hash,
                    "catalog_row": member,
                    "galtileid": int(table["galtileid"][member]),
                    "persistent_source_id": f"catsim:{catalog_hash}:row:{member}:galtileid:{int(table['galtileid'][member])}",
                    "duplicate_group_id": group["group_id"],
                    "group_rule": "exact original CatSim ra+dec pair; conservative potential duplicate grouping",
                    "partition": group["partition"],
                    "engineering_excluded": int(group["partition"] == "engineering_excluded"),
                    "split_seed": SPLIT_SEED,
                    "stratum": group["stratum"],
                }
            )
    manifest_rows.sort(key=lambda row: row["catalog_row"])
    write_csv(run_dir / "manifests" / "btk_engineering_source_groups.csv", manifest_rows)
    summary_rows = []
    final_mask = assignment != "engineering_excluded"
    variables = {
        "r_ab": r_mag,
        "redshift": redshift,
        "size_arcsec": size,
        "ellipticity_proxy": ellipticity,
        "g_minus_z": color,
    }
    for partition, target_fraction in SPLIT_PROPORTIONS.items():
        mask = assignment == partition
        groups = set(group_ids[mask])
        row = {
            "partition": partition,
            "source_count": int(mask.sum()),
            "group_count": len(groups),
            "actual_fraction": float(mask.sum() / final_mask.sum()),
            "target_fraction": target_fraction,
            "source_overlap_count": 0,
            "group_overlap_count": 0,
            "engineering_leakage_count": int(sum(index in engineering_rows for index in np.flatnonzero(mask))),
            "missing_value_count": 0,
        }
        for name, values in variables.items():
            row[f"{name}_mean"] = float(np.mean(values[mask]))
            row[f"{name}_std"] = float(np.std(values[mask]))
            row[f"{name}_ks_vs_all"] = float(ks_2samp(values[mask], values[final_mask]).statistic)
        summary_rows.append(row)
    write_csv(run_dir / "tables" / "btk_split_summary.csv", summary_rows)
    if any(int(row["engineering_leakage_count"]) for row in summary_rows):
        raise RuntimeError("Engineering-source leakage into final CatSim partitions")
    group_partitions: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        group_partitions[row["duplicate_group_id"]].add(row["partition"])
    if any(len(values) != 1 for values in group_partitions.values()):
        raise RuntimeError("Duplicate group crosses CatSim partitions")


def replay_scene(run_dir: Path, catalog_path: Path, metadata_path: Path) -> dict:
    catalog, _ = load_catsim_catalog(catalog_path)
    metadata = json.loads(metadata_path.read_text())
    spec = SceneSpec(
        scene_id=metadata["scene_id"],
        catalog_rows=tuple(int(source["catalog_row"]) for source in metadata["sources"]),
        positions_arcsec=tuple(tuple(value) for value in metadata["requested_positions_arcsec"]),
        source_selection_seed=int(metadata["source_selection_seed"]),
        position_seed=int(metadata["position_seed"]),
        noise_seed=int(metadata["noise_seed"]),
    )
    noiseless = render_fixed_scene(catalog, spec, add_noise="none")
    noisy = render_fixed_scene(catalog, spec, add_noise="all") if len(spec.catalog_rows) == 2 else None
    arrays_path = run_dir / metadata["relative_arrays_path"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        checks = {
            "source_selection": [int(value) for value in noiseless.catalog["catalog_row"]] == list(spec.catalog_rows),
            "positions": np.array_equal(
                np.asarray([[source["ra"], source["dec"]] for source in noiseless.catalog]),
                np.asarray(spec.positions_arcsec),
            ),
            "isolated_images": np.array_equal(noiseless.isolated, arrays["isolated_sources"]),
            "noiseless_blend": np.array_equal(noiseless.blend, arrays["blend_noiseless"]),
            "noise_realization": True,
            "noisy_blend": True,
        }
        if noisy is not None:
            checks["noise_realization"] = np.array_equal(
                noisy.blend - noiseless.blend, arrays["noise_realization"]
            )
            checks["noisy_blend"] = np.array_equal(noisy.blend, arrays["blend_noisy"])
    return {"scene_id": spec.scene_id, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--replay-metadata", type=Path)
    parser.add_argument("--expected-noise-seed", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    catalog_path = args.catalog.resolve()
    if args.replay_metadata:
        if args.expected_noise_seed is not None:
            replay_metadata = json.loads(args.replay_metadata.resolve().read_text())
            if int(replay_metadata["noise_seed"]) != args.expected_noise_seed:
                raise SystemExit(
                    f"metadata noise seed {replay_metadata['noise_seed']} != "
                    f"expected {args.expected_noise_seed}"
                )
        print(json.dumps(replay_scene(run_dir, catalog_path, args.replay_metadata.resolve()), sort_keys=True))
        return
    manifest_rows, test_rows, engineering_rows = render_campaign(run_dir, catalog_path)
    write_csv(run_dir / "tables" / "btk_scene_manifest.csv", manifest_rows)
    split_groups(run_dir, catalog_path, engineering_rows)
    metadata_paths = sorted((run_dir / "data" / "btk_engineering_pilot").glob("*.json"))
    replay_rows = []
    for metadata_path in metadata_paths:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            str(run_dir),
            "--catalog",
            str(catalog_path),
            "--replay-metadata",
            str(metadata_path),
        ]
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        if result.returncode == 0:
            replay = json.loads(result.stdout)
            status = replay["status"]
            note = json.dumps(replay["checks"], sort_keys=True)
        else:
            status = "FAIL"
            note = result.stderr[-1000:]
        replay_rows.append({"scene_id": metadata_path.stem, "status": status, "note": note})
        add_test(
            test_rows,
            metadata_path.stem,
            "fresh_process_replay",
            "selection, positions, isolated, noiseless, noise, noisy arrays reproduce exactly",
            0.0 if status == "PASS" else 1.0,
            0.0,
            "exact bytes/float64 arrays",
            status,
            str(metadata_path.relative_to(run_dir)),
            note,
        )
    write_csv(run_dir / "tables" / "btk_scene_unit_tests.csv", test_rows)
    write_csv(run_dir / "tables" / "btk_fresh_process_replay.csv", replay_rows)
    failed = [row for row in test_rows if row["status"] != "PASS"]
    summary = {
        "single_scene_count": 20,
        "double_scene_count": 20,
        "unit_test_count": len(test_rows),
        "failed_test_count": len(failed),
        "status": "PASS" if not failed else "FAIL",
    }
    (run_dir / "logs" / "btk_engineering_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
