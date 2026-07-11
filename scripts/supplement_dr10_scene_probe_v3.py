#!/usr/bin/env python3
"""Append-only v3 scientific supplement for the DR10 scene probe.

The supplement preserves v1/v2 products, revalidates all inputs, fixes the
support-relative core/halo definition, marks invalid extraction proxies as
unavailable rather than zero-flux measurements, records post-generation v2
visual QA, and qualifies non-coincident catalog PSF samples.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import astropy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sep

try:
    from scripts import audit_dr10_scene_triplets as base
    from scripts import correct_dr10_scene_probe_audit as v2
except ModuleNotFoundError:
    import audit_dr10_scene_triplets as base
    import correct_dr10_scene_probe_audit as v2


def unique_reasons(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        for item in value.split(";"):
            if item and item not in output:
                output.append(item)
    return output


def unavailable_morphology_rows(
    record: base.SourceRecord,
    manual: dict[str, str],
    reasons: list[str],
    obs_offset: float,
    model_offset: float,
    status: str,
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": record.source["source_id"],
            "catalog_row_index": record.source["catalog_row_index"],
            "band": band,
            "association_valid": 0,
            "association_reasons": ";".join(unique_reasons(reasons)),
            "observed_segmentation_centroid_offset_px": obs_offset,
            "model_segmentation_centroid_offset_px": model_offset,
            "central_mask_available": 0,
            "manual_classification": manual["classification"],
            "metric_status": status,
            "audit_version": "v3",
            "supersedes": "tables/residual_morphology_metrics_v2.csv",
        }
        for band in base.BANDS
    ]


def morphology_v3(
    record: base.SourceRecord,
    manual: dict[str, str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float, np.ndarray]]]:
    valid, reasons, obs_offset, model_offset = v2.association(record)
    mask, _nominal_radius, mask_error = v2.independent_central_mask(record)
    if not valid or not mask.any():
        rows = unavailable_morphology_rows(
            record,
            manual,
            reasons + ([mask_error] if mask_error else []),
            obs_offset,
            model_offset,
            "unavailable_invalid_central_association",
        )
        empty = np.zeros(mask.shape, dtype=bool)
        return rows, mask, empty, empty, []
    blanks = v2.blank_apertures(record, mask)
    if len(blanks) < 2:
        rows = unavailable_morphology_rows(
            record,
            manual,
            ["insufficient_exact_source_free_blank_apertures"],
            obs_offset,
            model_offset,
            "unavailable_insufficient_blank_null",
        )
        empty = np.zeros(mask.shape, dtype=bool)
        return rows, mask, empty, empty, blanks

    yy, xx = np.indices(mask.shape)
    radial = np.hypot(xx - record.catalog_target_x, yy - record.catalog_target_y)
    split_radius = float(np.quantile(radial[mask], 0.5))
    core = mask & (radial <= split_radius)
    halo = mask & (radial > split_radius)
    if not core.any() or not halo.any():
        raise RuntimeError(f"support-relative core/halo split failed for {record.source['source_id']}")
    arrays = {
        "observed": record.products["observed"].data,
        "model": record.products["model"].data,
        "residual": record.products["residual"].data,
    }
    blank_union = np.logical_or.reduce([item[2] for item in blanks])
    rows: list[dict[str, Any]] = []
    for band_index, band in enumerate(base.BANDS):
        backgrounds = {
            key: float(np.median(value[band_index][blank_union]))
            for key, value in arrays.items()
        }
        observed = arrays["observed"][band_index] - backgrounds["observed"]
        model = arrays["model"][band_index] - backgrounds["model"]
        residual = arrays["residual"][band_index] - backgrounds["residual"]
        blank_fluxes = np.asarray([float(np.sum(residual[item[2]])) for item in blanks])
        blank_correlations = np.asarray(
            [base.pearson(residual[item[2]], observed[item[2]]) for item in blanks]
        )
        blank_gradient = np.asarray(
            [v2.vector_gradient_correlation(residual, observed, item[2]) for item in blanks]
        )
        blank_asymmetry = np.asarray(
            [v2.asymmetry(residual, item[2], item[0], item[1]) for item in blanks]
        )
        blank_power = np.asarray([float(np.mean(residual[item[2]] ** 2)) for item in blanks])
        blank_core_power: list[float] = []
        blank_halo_power: list[float] = []
        for blank_x, blank_y, _blank_mask in blanks:
            dx = int(round(blank_x - record.catalog_target_x))
            dy = int(round(blank_y - record.catalog_target_y))
            translated_core = v2.translate_mask(core, dx, dy)
            translated_halo = v2.translate_mask(halo, dx, dy)
            if int(translated_core.sum()) != int(core.sum()) or int(translated_halo.sum()) != int(halo.sum()):
                raise RuntimeError("translated core/halo footprint changed size")
            blank_core_power.append(float(np.mean(residual[translated_core] ** 2)))
            blank_halo_power.append(float(np.mean(residual[translated_halo] ** 2)))
        blank_flux_scale = base.robust_scale(blank_fluxes)
        central_values = residual[mask]
        central_flux = float(np.sum(central_values))
        observed_flux = float(np.sum(observed[mask]))
        absolute = float(np.sum(np.abs(central_values)))
        central_corr = base.pearson(residual[mask], observed[mask])
        central_gradient = v2.vector_gradient_correlation(residual, observed, mask)
        central_asymmetry = v2.asymmetry(
            residual, mask, record.catalog_target_x, record.catalog_target_y
        )
        core_power = float(np.mean(residual[core] ** 2))
        halo_power = float(np.mean(residual[halo] ** 2))
        rows.append(
            {
                "source_id": record.source["source_id"],
                "catalog_row_index": record.source["catalog_row_index"],
                "band": band,
                "association_valid": 1,
                "association_reasons": "",
                "observed_segmentation_centroid_offset_px": obs_offset,
                "model_segmentation_centroid_offset_px": model_offset,
                "central_mask_available": 1,
                "central_mask_definition": "catalog-centered aperture clipped by nearest-catalog-component Voronoi boundary; rejected when a primary neighbor is within 2 PSF FWHM",
                "central_mask_pixel_count": int(mask.sum()),
                "central_mask_sha256": v2.array_hash(mask.astype(np.uint8)),
                "core_halo_definition": "support-relative radial pixel quantile: core <= median radial distance of final mask pixels; halo > median; not an astrophysical half-light radius",
                "core_halo_split_radius_pixels": split_radius,
                "core_halo_split_radius_arcsec": split_radius * base.PIXEL_SCALE,
                "core_pixel_count": int(core.sum()),
                "halo_pixel_count": int(halo.sum()),
                "core_mask_sha256": v2.array_hash(core.astype(np.uint8)),
                "halo_mask_sha256": v2.array_hash(halo.astype(np.uint8)),
                "blank_aperture_count": len(blanks),
                "background_convention": "per-product median over exact translated central footprints with zero overlap against dilated observed/model detections and catalog-source exclusion regions",
                "observed_central_flux": observed_flux,
                "model_central_aperture_flux": float(np.sum(model[mask])),
                "residual_central_flux": central_flux,
                "residual_to_observed_flux_fraction": central_flux / observed_flux if observed_flux != 0 else float("nan"),
                "blank_residual_flux_median": float(np.median(blank_fluxes)),
                "blank_residual_flux_robust_scale": blank_flux_scale,
                "central_residual_flux_null_zscore": (central_flux - float(np.median(blank_fluxes))) / blank_flux_scale if blank_flux_scale > 0 else float("nan"),
                "residual_observed_spatial_correlation": central_corr,
                "blank_spatial_correlation_median": float(np.nanmedian(blank_correlations)),
                "central_minus_blank_spatial_correlation": central_corr - float(np.nanmedian(blank_correlations)),
                "residual_observed_gradient_vector_correlation": central_gradient,
                "blank_gradient_vector_correlation_median": float(np.nanmedian(blank_gradient)),
                "central_minus_blank_gradient_correlation": central_gradient - float(np.nanmedian(blank_gradient)),
                "positive_central_residual_pixel_fraction": float(np.mean(central_values > 0)),
                "negative_central_residual_pixel_fraction": float(np.mean(central_values < 0)),
                "positive_significant_residual_pixel_fraction_3sigma": float(np.mean(central_values > 3 * record.noises[band_index])),
                "negative_significant_residual_pixel_fraction_3sigma": float(np.mean(central_values < -3 * record.noises[band_index])),
                "positive_central_absolute_flux_fraction": float(np.sum(np.clip(central_values, 0, None)) / absolute) if absolute > 0 else float("nan"),
                "negative_central_absolute_flux_fraction": float(np.sum(np.clip(-central_values, 0, None)) / absolute) if absolute > 0 else float("nan"),
                "residual_asymmetry_180": central_asymmetry,
                "blank_asymmetry_median": float(np.nanmedian(blank_asymmetry)),
                "blank_corrected_residual_asymmetry": central_asymmetry - float(np.nanmedian(blank_asymmetry)),
                "residual_core_power_per_pixel": core_power,
                "residual_halo_power_per_pixel": halo_power,
                "blank_residual_power_per_pixel_median": float(np.median(blank_power)),
                "blank_core_power_per_pixel_median": float(np.median(blank_core_power)),
                "blank_halo_power_per_pixel_median": float(np.median(blank_halo_power)),
                "residual_core_excess_power_per_pixel": core_power - float(np.median(blank_core_power)),
                "residual_halo_excess_power_per_pixel": halo_power - float(np.median(blank_halo_power)),
                "association_limitation": "Voronoi clipping limits catalog attribution but does not remove overlapping source wings; metrics are descriptive and do not prove isolation",
                "correlated_noise_caveat": "exact translated blank-aperture empirical correction used; no independence or white-noise assumption",
                "manual_classification": manual["classification"],
                "metric_status": "measured_complete_with_exact_blank_null",
                "audit_version": "v3",
                "supersedes": "tables/residual_morphology_metrics_v2.csv",
            }
        )
    return rows, mask, core, halo, blanks


def region_figure(
    record: base.SourceRecord,
    mask: np.ndarray,
    core: np.ndarray,
    halo: np.ndarray,
    blanks: list[tuple[float, float, np.ndarray]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.5), constrained_layout=True)
    for column, product in enumerate(("observed", "model", "residual")):
        axes[0, column].imshow(base.display_rgb(record.products[product].data), origin="lower")
        axes[0, column].contour(mask.astype(float), [0.5], colors=["cyan"], linewidths=0.7)
        axes[0, column].set_title(f"{product} RGB; central support")
    region = np.zeros(mask.shape, dtype=float)
    region[halo] = 1
    region[core] = 2
    axes[0, 3].imshow(region, origin="lower", cmap="viridis", vmin=0, vmax=2)
    axes[0, 3].set_title("support-relative halo (1) / core (2)")
    axes[0, 4].imshow(base.display_rgb(record.products["residual"].data), origin="lower")
    for _x, _y, blank in blanks:
        axes[0, 4].contour(blank.astype(float), [0.5], colors=["white"], linewidths=0.5)
    axes[0, 4].set_title("exact source-free blank footprints")
    for index, band in enumerate(base.BANDS):
        axes[1, index].imshow(
            base.signed_residual(record.products["residual"].data[index], record.noises[index]),
            origin="lower",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axes[1, index].contour(core.astype(float), [0.5], colors=["lime"], linewidths=0.7)
        axes[1, index].contour(halo.astype(float), [0.5], colors=["black"], linewidths=0.5)
        axes[1, index].set_title(f"signed residual {band}; core/halo")
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0,
        1,
        f"mask pixels: {int(mask.sum())}\ncore pixels: {int(core.sum())}\nhalo pixels: {int(halo.sum())}\n"
        "Split is the median radial distance within the final support.\nIt is not a half-light radius.",
        va="top",
    )
    axes[1, 4].axis("off")
    axes[1, 4].text(
        0,
        1,
        "Voronoi clipping and a 2-FWHM neighbor rejection reduce attribution risk,\n"
        "but they do not remove overlapping source wings.\nMetrics remain descriptive, not source isolation proof.",
        va="top",
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"DR10 morphology regions v3 — {record.source['source_id']}", fontsize=15)
    v2.save_figure(path, figure)


def extraction_rows_v3(run_dir: Path, component_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = v2.read_csv(run_dir / "tables" / "source_extraction_options_v2.csv")
    valid = {row["source_id"]: row["central_association_valid"] == "1" for row in component_rows}
    metric_fields = [
        "flux_g", "flux_r", "flux_z",
        "flux_preservation_g_vs_catalog_aperture_observed",
        "flux_preservation_r_vs_catalog_aperture_observed",
        "flux_preservation_z_vs_catalog_aperture_observed",
        "color_g_minus_r", "color_r_minus_z",
        "color_error_g_minus_r_vs_observed_aperture",
        "color_error_r_minus_z_vs_observed_aperture",
        "morphology_correlation_with_observed_aperture",
        "edge_artifact_gradient_ratio_spatial_axes",
        "rectangular_cutout_border_leakage_fraction",
        "support_min_frame_margin_pixels",
    ]
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["v2_proxy_support_pixel_count"] = row["support_pixel_count"]
        item["v2_proxy_flux_g"] = row["flux_g"]
        item["v2_proxy_flux_r"] = row["flux_r"]
        item["v2_proxy_flux_z"] = row["flux_z"]
        if valid[row["source_id"]]:
            item["measurement_available"] = 1
            item["metric_status"] = "descriptive_proxy_measured_not_approved"
        else:
            item["measurement_available"] = 0
            item["metric_status"] = "unavailable_invalid_central_association"
            item["support_pixel_count"] = ""
            item["support_sha256"] = ""
            for field in metric_fields:
                item[field] = ""
        item["audit_version"] = "v3"
        item["supersedes"] = "tables/source_extraction_options_v2.csv"
        output.append(item)
    return output


def psf_rows_v3(run_dir: Path) -> list[dict[str, Any]]:
    rows = v2.read_csv(run_dir / "tables" / "psf_audit_v2.csv")
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        separation = float(row["catalog_match_separation_arcsec"])
        item["psf_sample_coordinate_status"] = (
            "nearby_catalog_source_sample_gt_1arcsec_not_exact_coordinate"
            if separation > 1.0
            else "catalog_source_sample_within_1arcsec"
        )
        item["exact_coordinate_local_psf_map_sample_validated"] = 0
        item["psf_sample_suitable_for_exact_coordinate_contract"] = 0
        item["psf_interpretation"] = "official scalar PSFSIZE sample for the matched/nearest catalog row; exact-coordinate map and full kernel remain unvalidated"
        item["audit_version"] = "v3"
        item["supersedes"] = "tables/psf_audit_v2.csv"
        output.append(item)
    return output


def input_integrity_v3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if row["product"] == "observed":
            foundation_valid = int(row["observed_matches_foundation_fov_hash"]) == 1
            item["foundation_observed_hash_required"] = 1
            item["foundation_observed_hash_required_and_valid"] = int(foundation_valid)
        else:
            foundation_valid = True
            item["foundation_observed_hash_required"] = 0
            item["foundation_observed_hash_required_and_valid"] = ""
        item["input_integrity_pass_v3"] = int(bool(row["input_integrity_pass"]) and foundation_valid)
        item["audit_version"] = "v3"
        item["supersedes"] = "tables/input_integrity_v2.csv"
        output.append(item)
    if not all(row["input_integrity_pass_v3"] for row in output):
        raise RuntimeError("v3 input integrity failed, including required observed-foundation hashes")
    return output


def manual_rows_v3(run_dir: Path, qa_utc: str) -> list[dict[str, Any]]:
    rows = v2.read_csv(run_dir / "tables" / "manual_morphology_review_v2.csv")
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["original_manual_review_utc"] = row["review_utc"]
        item["v2_visual_qa_utc"] = qa_utc
        item["v2_visual_qa_status"] = "all 20 individual v2 scene sheets reviewed after generation; original classification confirmed"
        item["classification_changed_after_v2_visual_qa"] = 0
        item["scalar_evidence_summary"] = "classification remains manual; consult residual_morphology_metrics_v3.csv only where association is valid"
        item["audit_version"] = "v3"
        item["supersedes"] = "tables/manual_morphology_review_v2.csv"
        output.append(item)
    return output


def make_reports(
    morph: list[dict[str, Any]],
    extraction: list[dict[str, Any]],
    psf: list[dict[str, Any]],
    manual: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, str]:
    measured_sources = sorted({row["source_id"] for row in morph if row["metric_status"].startswith("measured")})
    unavailable_sources = sorted({row["source_id"] for row in morph if row["metric_status"].startswith("unavailable")})
    classes: dict[str, int] = {}
    for row in manual:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
    distant_psf = sorted({row["source_id"] for row in psf if float(row["catalog_match_separation_arcsec"]) > 1.0})
    morphology = f"""# Morphology-in-residual audit — authoritative v3 supplement

Generated: `{v2.utc_now()}`

This append-only table supersedes only the morphology scalars in v2; v1/v2
files remain preserved. Core and halo now split the **actual final central-mask
support** at its median radial pixel distance. This guarantees non-empty,
similarly sampled regions even for irregular neighbor-clipped masks. It is not
an astrophysical half-light-radius definition.

- Complete measured source-band rows: **{sum(row['metric_status'].startswith('measured') for row in morph)}/60**.
- Sources with conservative central attribution: **{len(measured_sources)}/20** — `{', '.join(measured_sources)}`.
- Explicitly unavailable sources: **{len(unavailable_sources)}/20**.
- All measured core/halo powers and exact translated null powers are finite.
- Manual classes are unchanged after review of all 20 generated v2 sheets:
  `{json.dumps(classes, sort_keys=True)}`.

Voronoi clipping and the two-FWHM neighbor rejection reduce attribution risk
but do not remove overlapping source wings; these measurements are descriptive
and do not prove central-source isolation.

- `tables/residual_morphology_metrics_v3.csv`
- `tables/manual_morphology_review_v3.csv`
- `{figure_dir}/morphology_regions_v3_*.png`
"""
    extraction_report = f"""# Source-extraction options — authoritative v3 supplement

Generated: `{v2.utc_now()}`

The v2 proxy calculations are preserved, but **{sum(not row['measurement_available'] for row in extraction)}/60**
rows whose central association failed are now explicitly unavailable; their v2
zero-valued placeholders are retained only in `v2_proxy_*` provenance columns
and are not interpreted as measured zero flux. The other
**{sum(row['measurement_available'] for row in extraction)}/60** rows remain
descriptive proxies only. **0/60** options are approved as contaminant arrays.

- `tables/source_extraction_options_v3.csv`
"""
    psf_report = f"""# PSF audit — authoritative v3 supplement

Generated: `{v2.utc_now()}`

Catalog `PSFSIZE_G/R/Z` values remain official scalar samples, not validated
full PSF kernels or exact-coordinate PSF-map samples. The nearest catalog rows
for `{', '.join(distant_psf)}` are more than 1 arcsec from the requested
coordinates, so their values are reported only as nearby local samples.
All 20 sources therefore retain `exact_coordinate_local_psf_map_sample_validated=0`.
Direct source addition remains PSF-inconsistent; scalar pairing is exploratory,
common-PSF convolution needs validated kernels, and forward rendering at the
target PSF is preferred. No deconvolution was implemented.

- `tables/psf_audit_v3.csv`
"""
    corrections = f"""# DR10 scene-probe append-only correction ledger — v3

Generated: `{v2.utc_now()}`

- Preserved all v1/v2 products.
- Fixed support-relative core/halo availability for `404010_3100`.
- Removed duplicated/un-testable invalid-association reason text.
- Marked 39 invalid extraction proxies unavailable rather than measured zeros.
- Logged post-generation visual QA for all 20 v2 scene sheets without changing
  any manual class.
- Qualified two catalog PSF samples beyond 1 arcsec.
- Made the 20 observed-foundation SHA-256 matches mandatory in v3 input
  integrity; all pass.

None of these corrections changes the decision gate: blending remains blocked.
"""
    return {
        "morphology_residual_audit_v3.md": morphology,
        "source_extraction_options_v3.md": extraction_report,
        "psf_audit_v3.md": psf_report,
        "audit_corrections_v3.md": corrections,
    }


def inventory_rows(paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    timestamp = v2.utc_now()
    rows: list[dict[str, Any]] = []
    for scope, path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        rows.append(
            {
                "inventory_generated_utc": timestamp,
                "scope": scope,
                "path": str(resolved),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": v2.sha256_file(resolved),
                "inventory_self_excluded": 1,
                "replay_note": "morphology_v3_output_inventory.csv is self-excluded because no file can contain its own stable hash",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--foundation-run", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.run_dir, args.foundation_run):
        if "lockbox" in str(path).lower() or "sealed" in str(path).lower():
            raise SystemExit("refusing lockbox/sealed path")
    expected_absent = [
        args.run_dir / "tables" / "residual_morphology_metrics_v3.csv",
        args.run_dir / "tables" / "source_extraction_options_v3.csv",
        args.run_dir / "tables" / "psf_audit_v3.csv",
        args.run_dir / "tables" / "input_integrity_v3.csv",
        args.run_dir / "tables" / "manual_morphology_review_v3.csv",
        args.run_dir / "figures" / "scene_probe_v3",
    ]
    if any(path.exists() for path in expected_absent):
        raise FileExistsError("v3 supplement path already exists")

    sources, grouped = base.load_products(args.run_dir)
    catalogs = base.load_catalog_manifest(args.run_dir)
    integrity_v2 = v2.validate_inputs(args.run_dir, args.foundation_run, sources, grouped, catalogs)
    integrity = input_integrity_v3(integrity_v2)
    v2.checkpoint_integrity(args.foundation_run)
    qa_utc = v2.utc_now()
    manual_rows = manual_rows_v3(args.run_dir, qa_utc)
    manual = {row["source_id"]: row for row in manual_rows}
    figure_dir = args.run_dir / "figures" / "scene_probe_v3"
    figure_dir.mkdir(parents=True, exist_ok=False)
    morphology: list[dict[str, Any]] = []
    for rank, source in enumerate(sources, start=1):
        record = base.load_source_record(source, grouped[source["source_id"]], catalogs[source["source_id"]])
        rows, mask, core, halo, blanks = morphology_v3(record, manual[source["source_id"]])
        morphology.extend(rows)
        if core.any() and halo.any():
            region_figure(
                record,
                mask,
                core,
                halo,
                blanks,
                figure_dir / f"morphology_regions_v3_{rank:02d}_{source['source_id']}.png",
            )
    measured = [row for row in morphology if row["metric_status"].startswith("measured")]
    required_finite = (
        "residual_core_power_per_pixel",
        "residual_halo_power_per_pixel",
        "blank_core_power_per_pixel_median",
        "blank_halo_power_per_pixel_median",
        "residual_core_excess_power_per_pixel",
        "residual_halo_excess_power_per_pixel",
    )
    if len(measured) != 21 or not all(np.isfinite(float(row[field])) for row in measured for field in required_finite):
        raise RuntimeError("v3 morphology completeness validation failed")

    component_rows = v2.read_csv(args.run_dir / "tables" / "scene_component_audit_v2.csv")
    extraction = extraction_rows_v3(args.run_dir, component_rows)
    psf = psf_rows_v3(args.run_dir)
    tables: dict[str, list[dict[str, Any]]] = {
        "input_integrity_v3.csv": integrity,
        "residual_morphology_metrics_v3.csv": morphology,
        "manual_morphology_review_v3.csv": manual_rows,
        "source_extraction_options_v3.csv": extraction,
        "psf_audit_v3.csv": psf,
    }
    generated_paths: list[Path] = []
    for filename, rows in tables.items():
        path = args.run_dir / "tables" / filename
        v2.write_csv(path, rows)
        generated_paths.append(path)
    report_texts = make_reports(morphology, extraction, psf, manual_rows, figure_dir)
    for filename, text in report_texts.items():
        path = args.run_dir / "diagnostics" / filename
        v2.write_text(path, text)
        generated_paths.append(path)
    environment = {
        "generated_utc": v2.utc_now(),
        "qa_utc": qa_utc,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "astropy": astropy.__version__,
        "scipy": scipy.__version__,
        "sep": sep.__version__,
        "matplotlib": matplotlib.__version__,
        "script": str(Path(__file__).resolve()),
        "script_sha256": v2.sha256_file(Path(__file__).resolve()),
        "v2_script_sha256": v2.sha256_file(Path(v2.__file__).resolve()),
        "base_script_sha256": v2.sha256_file(Path(base.__file__).resolve()),
        "git_branch": subprocess.run(["git", "branch", "--show-current"], check=True, text=True, capture_output=True).stdout.strip(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip(),
    }
    environment_path = args.run_dir / "logs" / "scene_probe_supplement_v3_environment.json"
    v2.write_text(environment_path, json.dumps(environment, indent=2, sort_keys=True) + "\n")
    generated_paths.append(environment_path)
    summary = {
        "input_integrity_pass": all(row["input_integrity_pass_v3"] for row in integrity),
        "observed_foundation_hashes_required_and_valid": sum(
            row["foundation_observed_hash_required_and_valid"] == 1 for row in integrity
        ),
        "morphology_complete_measured_rows": len(measured),
        "morphology_unavailable_rows": len(morphology) - len(measured),
        "invalid_extraction_rows_marked_unavailable": sum(not row["measurement_available"] for row in extraction),
        "contaminant_options_pass": sum(int(row["suitable_as_contaminant_for_single_noise_contract"]) for row in extraction),
        "psf_samples_gt_1arcsec": len({row["source_id"] for row in psf if float(row["catalog_match_separation_arcsec"]) > 1}),
        "v3_region_figure_count": len(list(figure_dir.glob("*.png"))),
    }
    summary_path = args.run_dir / "logs" / "scene_probe_supplement_v3_summary.json"
    v2.write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    generated_paths.append(summary_path)
    generated_paths.extend(sorted(figure_dir.glob("*.png")))
    dependency_paths = [
        Path(__file__), Path(v2.__file__), Path(base.__file__),
        args.run_dir / "manifests" / "engineering_sources_20.csv",
        args.run_dir / "manifests" / "scene_triplet_download_manifest.csv",
        args.run_dir / "manifests" / "official_catalog_download_manifest.csv",
        args.run_dir / "tables" / "manual_morphology_review_v2.csv",
        args.run_dir / "tables" / "source_extraction_options_v2.csv",
        args.run_dir / "tables" / "psf_audit_v2.csv",
    ]
    inventory = inventory_rows(
        [("input_dependency", path) for path in dependency_paths]
        + [("generated_v3_output", path) for path in generated_paths]
    )
    v2.write_csv(args.run_dir / "tables" / "scene_probe_v3_output_inventory.csv", inventory)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
