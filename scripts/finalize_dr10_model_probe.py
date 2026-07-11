#!/usr/bin/env python3
"""Finalize the append-only DR10 model-probe report and decision gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts import correct_dr10_scene_probe_audit as audit
except ModuleNotFoundError:
    import correct_dr10_scene_probe_audit as audit


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], check=True, text=True, capture_output=True
    ).stdout.rstrip()


def morphology_inference_rows(
    morphology: list[dict[str, str]],
    closure: list[dict[str, str]],
) -> list[dict[str, Any]]:
    closure_by_key = {
        (row["source_id"], row["band"]): int(row["closure_valid_v2"])
        for row in closure
    }
    rows: list[dict[str, Any]] = []
    for row in morphology:
        key = (row["source_id"], row["band"])
        closure_valid = closure_by_key[key]
        measured = row["metric_status"].startswith("measured")
        if measured and closure_valid:
            status = "matched_additive_residual_descriptive_inference_allowed"
            allowed = 1
        elif measured:
            status = "downloaded_residual_product_only_closure_failed"
            allowed = 0
        else:
            status = "unavailable_invalid_central_association"
            allowed = 0
        rows.append(
            {
                "source_id": row["source_id"],
                "catalog_row_index": row["catalog_row_index"],
                "band": row["band"],
                "association_valid": row["association_valid"],
                "closure_valid": closure_valid,
                "morphology_metric_status_v3": row["metric_status"],
                "morphology_inference_status_v4": status,
                "may_interpret_product_as_matched_observed_minus_model": allowed,
                "central_structure_inference_allowed": allowed,
                "manual_classification": row["manual_classification"],
                "control_footprint_term": "catalog/detection-excluded exact translated control footprint; not guaranteed source-free",
                "inference_caveat": "residual/observed correlation is algebraically dependent; controls can contain undetected or diffuse structure; Voronoi clipping does not remove overlapping wings",
                "supersedes_for_inference": "diagnostics/morphology_residual_audit_v3.md wording only; v3 scalar values remain preserved",
            }
        )
    if len(rows) != 60:
        raise RuntimeError("expected 60 morphology inference rows")
    if sum(row["central_structure_inference_allowed"] for row in rows) != 15:
        raise RuntimeError("expected 15 closure-and-association-valid inference rows")
    return rows


def validate_inventory(path: Path) -> tuple[int, int]:
    rows = read_csv(path)
    valid = 0
    for row in rows:
        artifact = Path(row["path"])
        stat = artifact.stat()
        if (
            stat.st_size == int(row["size_bytes"])
            and stat.st_mtime_ns == int(row["mtime_ns"])
            and audit.sha256_file(artifact) == row["sha256"]
        ):
            valid += 1
    return valid, len(rows)


def final_inventory_rows(
    run_dir: Path,
    incident_dir: Path,
    foundation_run: Path,
    repo_paths: list[Path],
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Path]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "final_campaign_inventory.csv":
            candidates.append(("successful_probe_artifact", path))
    if incident_dir.exists():
        for path in sorted(incident_dir.rglob("*")):
            if path.is_file():
                candidates.append(("preserved_incident_artifact", path))
    for path in repo_paths:
        candidates.append(("campaign_code_or_contract", path))
    for row in read_csv(foundation_run / "tables" / "checkpoint_inventory_after.csv"):
        if "lockbox" in row["path"].lower() or "sealed" in row["path"].lower():
            raise RuntimeError("refusing protected checkpoint path")
        candidates.append(("unchanged_foundation_checkpoint", Path(row["path"])))
    timestamp = utc_now()
    output: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for scope, path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        stat = resolved.stat()
        output.append(
            {
                "inventory_generated_utc": timestamp,
                "scope": scope,
                "path": str(resolved),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": audit.sha256_file(resolved),
                "final_inventory_self_excluded": 1,
                "self_exclusion_note": "final_campaign_inventory.csv is excluded because no file can contain its own stable hash",
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--incident-dir", type=Path, required=True)
    parser.add_argument("--foundation-run", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.run_dir, args.incident_dir, args.foundation_run):
        lowered = str(path).lower()
        if "lockbox" in lowered or "sealed" in lowered:
            raise SystemExit("refusing lockbox/sealed path")
    outputs = [
        args.run_dir / "tables" / "morphology_inference_status_v4.csv",
        args.run_dir / "tables" / "decision_gate.csv",
        args.run_dir / "logs" / "repository_status_final.txt",
        args.run_dir / "logs" / "finalization_environment.json",
        args.run_dir / "reports" / "final_report.md",
        args.run_dir / "tables" / "final_campaign_inventory.csv",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("finalization output already exists")

    alignment = read_csv(args.run_dir / "tables" / "scene_triplet_alignment_v2.csv")
    closure = read_csv(args.run_dir / "tables" / "scene_triplet_closure_v2.csv")
    morphology = read_csv(args.run_dir / "tables" / "residual_morphology_metrics_v3.csv")
    components = read_csv(args.run_dir / "tables" / "scene_component_audit_v2.csv")
    extraction = read_csv(args.run_dir / "tables" / "source_extraction_options_v3.csv")
    psf = read_csv(args.run_dir / "tables" / "psf_audit_v3.csv")
    manual = read_csv(args.run_dir / "tables" / "manual_morphology_review_v3.csv")
    checkpoints = read_csv(args.run_dir / "tables" / "checkpoint_integrity_v2.csv")
    integrity = read_csv(args.run_dir / "tables" / "input_integrity_v3.csv")
    triplet_manifest = read_csv(args.run_dir / "manifests" / "scene_triplet_download_manifest.csv")
    catalog_manifest = read_csv(args.run_dir / "manifests" / "official_catalog_download_manifest.csv")

    by_source: dict[str, list[dict[str, str]]] = {}
    for row in closure:
        by_source.setdefault(row["source_id"], []).append(row)
    closure_pass_sources = sum(
        all(int(row["closure_valid_v2"]) for row in rows)
        for rows in by_source.values()
    )
    closure_failures = [
        source_id
        for source_id, rows in by_source.items()
        if not all(int(row["closure_valid_v2"]) for row in rows)
    ]
    alignment_pass = sum(int(row["alignment_pass_v2"]) for row in alignment)
    central_association = sum(int(row["central_association_valid"]) for row in components)
    central_isolation = sum(int(row["central_only_model_isolation_reliable"]) for row in components)
    extraction_approved = sum(
        int(row["suitable_as_contaminant_for_single_noise_contract"])
        for row in extraction
    )
    exact_psf = sum(int(row["exact_coordinate_local_psf_map_sample_validated"]) for row in psf)
    inference = morphology_inference_rows(morphology, closure)
    audit.write_csv(outputs[0], inference)

    gate_rows = [
        {
            "gate_id": "alignment",
            "criterion": "image/model/residual are pixel-aligned",
            "result": "PASS",
            "blocking": 0,
            "evidence": f"{alignment_pass}/20 exact shape, bands, CRPIX/CRVAL, matrix, scale, units, and full-grid WCS",
            "authoritative_path": "tables/scene_triplet_alignment_v2.csv",
        },
        {
            "gate_id": "closure",
            "criterion": "closure is numerically valid",
            "result": "FAIL",
            "blocking": 1,
            "evidence": f"{closure_pass_sources}/20 pass; failures={','.join(closure_failures)}",
            "authoritative_path": "tables/scene_triplet_closure_v2.csv",
        },
        {
            "gate_id": "central_source",
            "criterion": "central-only source extraction is reliable",
            "result": "FAIL",
            "blocking": 1,
            "evidence": f"{central_isolation}/20 reliable; summed scene has no per-source component planes",
            "authoritative_path": "tables/scene_component_audit_v2.csv",
        },
        {
            "gate_id": "morphology",
            "criterion": "important morphology is not systematically discarded or limitation is accepted",
            "result": "FAIL",
            "blocking": 1,
            "evidence": "only 5/20 sources have both valid association and closure; manual review finds 4 omissions and 5 unusable fits",
            "authoritative_path": "tables/manual_morphology_review_v3.csv;tables/morphology_inference_status_v4.csv",
        },
        {
            "gate_id": "single_noise_and_psf",
            "criterion": "exactly one noise realization and explicit consistent PSF",
            "result": "FAIL",
            "blocking": 1,
            "evidence": f"0/60 contaminant options approved; {exact_psf}/60 exact-coordinate PSF samples validated",
            "authoritative_path": "tables/source_extraction_options_v3.csv;tables/psf_audit_v3.csv",
        },
        {
            "gate_id": "no_whole_scene_paste",
            "criterion": "no whole-scene model cutout is pasted as one source",
            "result": "PASS",
            "blocking": 0,
            "evidence": "whole-scene model is explicitly prohibited; no blend was generated",
            "authoritative_path": "tables/scene_component_audit_v2.csv",
        },
        {
            "gate_id": "units_and_bands",
            "criterion": "flux units and band order are stable",
            "result": "PASS",
            "blocking": 0,
            "evidence": "20/20 grz; 0.262 arcsec/pixel; official unit nanomaggies per coadd pixel",
            "authoritative_path": "tables/scene_triplet_alignment_v2.csv",
        },
        {
            "gate_id": "deterministic_replay",
            "criterion": "extraction and blending replay deterministically",
            "result": "FAIL",
            "blocking": 1,
            "evidence": "audit inputs/outputs are hashed, but no valid source extraction or blend exists to replay",
            "authoritative_path": "tables/output_inventory_v2.csv;tables/scene_probe_v3_output_inventory.csv",
        },
    ]
    audit.write_csv(outputs[1], gate_rows)

    if not (
        alignment_pass == 20
        and closure_pass_sources == 15
        and central_association == 7
        and central_isolation == 0
        and extraction_approved == 0
        and exact_psf == 0
        and len(triplet_manifest) == 60
        and len(catalog_manifest) == 20
        and all(int(row["input_integrity_pass_v3"]) for row in integrity)
        and all(int(row["identity_unchanged"]) for row in checkpoints)
    ):
        raise RuntimeError("final decision evidence does not match preregistered expectations")

    v2_inventory_valid = validate_inventory(args.run_dir / "tables" / "output_inventory_v2.csv")
    v3_inventory_valid = validate_inventory(args.run_dir / "tables" / "scene_probe_v3_output_inventory.csv")
    branch = git(["branch", "--show-current"])
    head = git(["rev-parse", "HEAD"])
    status = git(["status", "--short", "--branch"])
    tracked_diff = git(["diff", "--name-only"])
    staged_diff = git(["diff", "--cached", "--name-only"])
    status_text = (
        f"captured_utc={utc_now()}\nbranch={branch}\nhead={head}\n"
        f"tracked_diff_names={tracked_diff or '<none>'}\n"
        f"staged_diff_names={staged_diff or '<none>'}\n\n{status}\n"
    )
    audit.write_text(outputs[2], status_text)
    environment = {
        "generated_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": audit.sha256_file(Path(__file__).resolve()),
        "branch": branch,
        "head": head,
        "v2_inventory_valid": list(v2_inventory_valid),
        "v3_inventory_valid": list(v3_inventory_valid),
        "lockbox_accessed": False,
        "training_run": False,
        "blend_manifest_generated": False,
    }
    audit.write_text(outputs[3], json.dumps(environment, indent=2, sort_keys=True) + "\n")

    classes: dict[str, list[str]] = {}
    for row in manual:
        classes.setdefault(row["classification"], []).append(row["source_id"])
    psf_stats: dict[str, tuple[float, float, float, float]] = {}
    for band in ("g", "r", "z"):
        values = np.asarray([float(row["psf_fwhm_arcsec"]) for row in psf if row["band"] == band])
        psf_stats[band] = (
            float(values.min()), float(values.max()), float(np.median(values)), float(values.std(ddof=1))
        )
    cross_band = np.asarray(
        [float(row["within_source_cross_band_range_arcsec"]) for row in psf if row["band"] == "g"]
    )
    component_observed = [int(row["observed_component_count"]) for row in components]
    component_model = [int(row["modeled_component_count"]) for row in components]
    neighbor_count = [int(row["modeled_neighbor_count_catalog_primary"]) for row in components]
    passing_closure_rows = [row for row in closure if int(row["closure_valid_v2"])]
    max_passing_closure = max(float(row["maximum_absolute_closure"]) for row in passing_closure_rows)
    max_failing_closure = max(float(row["maximum_absolute_closure"]) for row in closure)
    matched_sources = sorted(
        {row["source_id"] for row in inference if row["central_structure_inference_allowed"]}
    )
    measured_sources = sorted(
        {row["source_id"] for row in morphology if row["metric_status"].startswith("measured")}
    )
    core_all_three = 0
    halo_all_three = 0
    for source_id in matched_sources:
        source_rows = [row for row in morphology if row["source_id"] == source_id]
        core_all_three += all(float(row["residual_core_excess_power_per_pixel"]) > 0 for row in source_rows)
        halo_all_three += all(float(row["residual_halo_excess_power_per_pixel"]) > 0 for row in source_rows)
    observed_gt1 = sorted(
        {row["source_id"] for row in psf if float(row["catalog_match_separation_arcsec"]) > 1}
    )
    report = f"""# DR10 observed/model/residual engineering probe — final report

Run: `dr10_model_probe_20260711_160018`  
Foundation: `dr10_foundation_20260711_024415`  
Generated: `{utc_now()}`  
Branch/HEAD: `{branch}` / `{head}`

## Executive decision

**No. Matched DR10 observed/model/residual cutouts do not yet support the
requested source-only, single-noise-realization, PSF-consistent blending
procedure. FITS blending remains blocked.**

All 20 triplets are exactly aligned, but only **15/20** pass closure. The model
cutouts are summed scene predictions, not central-only planes; central-only
isolation is **0/20**. Options A and C import another coadd-noise realization,
while option B is an inseparable, already PSF-convolved segment of the summed
parametric scene. No exact-coordinate full PSF was validated. Consequently
**0/60** extraction options pass as contaminant arrays.

The selected contract is option 4: remain blocked and evaluate a separate
BTK/GalSim forward-rendering benchmark. The binding contract is
`docs/dr10_flux_blending_contract.md`.

## 1. Matched scene triplets

- Engineering sources: the same 20 foundation examples, in the same order.
- Products: `ls-dr10-south`, `ls-dr10-model`, and `ls-dr10-resid`.
- Geometry: `256x256`, ordered `g,r,z`, `0.262 arcsec/pixel`.
- Successful downloads: **60/60 FITS**, plus **20/20 official catalog boxes**;
  final failure tables are header-only.
- Immutable product directories:
  - `downloads/observed_20260711_160018/`
  - `downloads/model_20260711_160018/`
  - `downloads/residual_20260711_160018/`
  - `downloads/official_catalog_20260711_160018/`
- Request URLs, response/request headers, raw-response hashes, validated-file
  hashes, final URLs, and failures are recorded in `manifests/` and `logs/`.

An earlier attempt, `dr10_model_probe_20260711_155820`, stopped on an Astropy
pixel-scale API error. It is preserved as an incident record and excluded from
scientific results; nothing was overwritten or deleted.

## 2. Alignment and additivity

Alignment passes **20/20** for shape, band order, CRPIX/CRVAL, canonical CD/PC
matrix, full-grid celestial WCS, pixel scale, and normalized unit semantics.
Viewer cutouts omit `BUNIT`; the official DR10 data model establishes linear
nanomaggies per coadd pixel for these image-stack products.

Closure passes **15/20** under the joint finite-coverage, RMSE/noise, L1,
99.99th-percentile, and maximum/peak gates. Passing rows have 100% finite
coverage and maximum absolute closure no larger than `{max_passing_closure:.6g}`.
The campaign-wide maximum is `{max_failing_closure:.6g}`. Failures are:
`{', '.join(closure_failures)}`. Their source-dependent mechanism is unresolved;
no causal claim is made.

Authoritative evidence:

- `tables/scene_triplet_alignment_v2.csv`
- `tables/scene_triplet_closure_v2.csv`
- `diagnostics/scene_triplet_audit_v2.md`

The required unsuffixed v1 tables remain preserved, but their numerical audit
is superseded by v2 and the later inference qualifications.

## 3. Morphology in the residual

All 20 individual observed/model/residual sheets were manually reviewed. The
post-generation v2 QA was attested during the v3 supplement; that timestamp is
an attestation-recording time, not an independently captured review-event time.

| Manual class | Count | Source IDs |
|---|---:|---|
| model retains morphology sufficiently | {len(classes['model retains morphology sufficiently'])} | `{', '.join(classes['model retains morphology sufficiently'])}` |
| model moderately simplifies morphology | {len(classes['model moderately simplifies morphology'])} | `{', '.join(classes['model moderately simplifies morphology'])}` |
| model omits important target structure | {len(classes['model omits important target structure'])} | `{', '.join(classes['model omits important target structure'])}` |
| model fit is unusable | {len(classes['model fit is unusable'])} | `{', '.join(classes['model fit is unusable'])}` |

Conservative catalog/segmentation association is valid for **7/20** sources:
`{', '.join(measured_sources)}`. Two of those (`404010_3100`,
`457285_2761`) fail closure, so their scalar measurements describe only the
downloaded residual product. Only **5/20** sources, `{', '.join(matched_sources)}`,
support matched-additive residual interpretation.

Within those five, support-relative core excess power is positive in all three
bands for {core_all_three}/5 sources and halo excess power for {halo_all_three}/5.
The signed residuals and manual review therefore show that the residual is not
universally just noise/background; it can contain coherent central fit error
and omitted structure. This does not make the residual a source-only morphology
layer, nor prove attribution outside the five matched cases.

The v3 metrics use exact translated **catalog/detection-excluded control
footprints**. They are not guaranteed source-free: undetected or diffuse light
can remain. Core/halo regions split the actual final support by radial pixel
quantile and are not half-light radii.

Authoritative evidence:

- `tables/manual_morphology_review_v3.csv`
- `tables/residual_morphology_metrics_v3.csv`
- `tables/morphology_inference_status_v4.csv`
- `tables/radial_profiles_v2.csv`
- `diagnostics/morphology_residual_audit_v3.md`

## 4. Scene components

Observed detection finds `{min(component_observed)}–{max(component_observed)}`
components (median `{np.median(component_observed):.1f}`); the model finds
`{min(component_model)}–{max(component_model)}` (median
`{np.median(component_model):.1f}`). Official primary-catalog neighbor counts
range `{min(neighbor_count)}–{max(neighbor_count)}` (median
`{np.median(neighbor_count):.1f}`). Thus `ls-dr10-model` contains the central
prediction plus neighboring predictions, not only the requested source.

One scene contains an official G<13 bright-star component, seven contain at
least one G<16 GE/T2 component, and none contains an `REF_CAT=L3` large-galaxy
row. No explicit source-catalog background component exists; the raw model
outer pedestal was measured separately and reaches at most 0.673 residual-noise
units. This does not prove absence of diffuse/background modeling.

**0/20 central-only model components are reliably isolatable.** SEP segments
of a summed prediction do not bound overlapping Tractor-profile wings. The
whole model cutout is prohibited as a contaminant.

Evidence: `tables/scene_component_audit_v2.csv` and
`diagnostics/scene_component_audit_v2.md`.

## 5. PSF audit

Official catalog samples give:

| Band | Min–max FWHM (arcsec) | Median | Cross-source SD |
|---|---:|---:|---:|
| g | {psf_stats['g'][0]:.4f}–{psf_stats['g'][1]:.4f} | {psf_stats['g'][2]:.4f} | {psf_stats['g'][3]:.4f} |
| r | {psf_stats['r'][0]:.4f}–{psf_stats['r'][1]:.4f} | {psf_stats['r'][2]:.4f} | {psf_stats['r'][3]:.4f} |
| z | {psf_stats['z'][0]:.4f}–{psf_stats['z'][1]:.4f} | {psf_stats['z'][2]:.4f} | {psf_stats['z'][3]:.4f} |

Within-source cross-band ranges span `{cross_band.min():.4f}–{cross_band.max():.4f}`
arcsec (median `{np.median(cross_band):.4f}`). Direct addition would therefore
mix incompatible source- and band-dependent PSFs. Scalar-FWHM pairing is only
an exploratory screen. Convolution to a common broader PSF is feasible only
after full matching kernels and moment/encircled-energy checks; forward
rendering an intrinsic source model at the target PSF is preferred. No
deconvolution was implemented.

Exact-coordinate local maps/full kernels were validated for **0/60** band
samples. `{', '.join(observed_gt1)}` use nearest catalog samples more than 1
arcsec from the requested coordinate and are explicitly qualified. Official
local maps are available at
`south/coadd/<AAA>/<brick>/legacysurvey-<brick>-psfsize-<band>.fits.fz`.

Evidence: `tables/psf_audit_v3.csv` and `diagnostics/psf_audit_v3.md`.

## 6. Source-extraction comparison

| Option | Main benefit | Fatal issue in this probe | Decision |
|---|---|---|---|
| A — segmented observed | best observed morphology | neighbor/background leakage and a second coadd-noise realization | reject contaminant |
| B — segmented Tractor model | low-noise parametric estimate | summed-scene segment, inseparable wings, already convolved PSF | reject contaminant |
| C — model-assisted observed | retains some observed residual structure | coadd noise and fit errors remain; depends on invalid B proxy | reject contaminant |

Twenty-one option rows are descriptive proxies for seven associated sources;
39 invalid-association rows are explicitly unavailable, not zero-flux
measurements. **0/60** pass. Subpixel-shift and full-PSF consistency were not
validated.

Evidence: `tables/source_extraction_options_v3.csv`,
`diagnostics/source_extraction_options_v3.md`, and the comparative sheets.

## 7. Recommended scientific strategy

Choose **option 4: remain blocked and recommend BTK/GalSim**. The future
benchmark should forward-render target and contaminant source models at the
same declared per-band PSF, sum them in linear nanomaggies, then draw exactly
one final noise realization. Procedural injection parameters—not a Tractor fit
as astrophysical truth—are ground truth. One achromatic contaminant flux scalar
preserves color; every PSF, seed, shift, input, code, and output hash is part of
replay metadata.

The current environment is Python 3.14.6 and contains neither
`blending-toolkit` nor `GalSim`; BTK 1.0.9 requires Python below 3.13. The exact
next setup command, **after Python 3.12 is provisioned and a new environment is
explicitly authorized**, is:

```bash
python3.12 -m venv .venv-btk
```

This command creates only an isolated feasibility environment; it does not
authorize blend generation. No executable blending command is recommended yet.

## 8. Decision gate

| Gate | Result | Evidence |
|---|---|---|
| aligned products | PASS | 20/20 |
| numerical closure | **FAIL** | 15/20 |
| central-only source | **FAIL** | 0/20 reliable |
| acceptable morphology contract | **FAIL** | only 5/20 matched/attributable; 9 serious manual failures |
| one noise realization and consistent PSF | **FAIL** | 0/60 options; 0/60 exact PSF samples |
| no whole-scene paste | PASS | prohibited and no blends generated |
| stable units and bands | PASS | grz, nanomaggies/pixel, 20/20 |
| deterministic extraction/blend replay | **FAIL** | audit replays; no valid extraction/blend exists |

**Blending may not proceed.** Machine-readable evidence is
`tables/decision_gate.csv`.

## Contact sheets

- Complete v2 observed/model/residual, robust residual, RGB, mask, control
  footprint, and radial-profile sheets:
  `figures/scene_probe_v2/scene_triplet_v2_*.png`
- Complete v2 option-comparison sheets:
  `figures/scene_probe_v2/source_extraction_v2_*.png`
- Corrected support-relative core/halo diagnostics for the seven associated
  sources: `figures/scene_probe_v3/morphology_regions_v3_*.png`
- Four overview montage pages used in the original manual review remain at
  `figures/scene_probe_v1/manual_review_montage_page*.png`; their v1 scalar
  annotations are superseded.

## Unresolved blockers

1. Five source triplets fail closure; cause unresolved.
2. Summed model cutouts provide no official per-source component plane.
3. Only seven central associations are conservative, and only five also close.
4. Manual review finds four important omissions and five unusable fits.
5. A/C carry another coadd-noise realization; B is not isolated.
6. Exact-coordinate maps and full PSF kernels are unvalidated; scalar FWHM is
   insufficient.
7. Subpixel forward rendering, padding, edge-flux accounting, and common-PSF
   tests are absent.
8. Catalog/detection-excluded controls can retain undetected/diffuse light.
9. BTK/GalSim are absent and require a separate compatible Python environment.
10. No scientifically valid end-to-end blend exists to replay.

## Integrity, repository state, and files changed

- Input integrity: **80/80** triplet/catalog records pass v3, including all
  **20/20 required observed-foundation hashes**.
- Historical checkpoint integrity: **18/18** hash, size, and mtime identities
  unchanged. No sealed/lockbox path was opened or used.
- v2 inventory: **{v2_inventory_valid[0]}/{v2_inventory_valid[1]}** entries
  revalidated; v3 inventory: **{v3_inventory_valid[0]}/{v3_inventory_valid[1]}**.
- A final file-level SHA-256 inventory, including the previously omitted
  `scene_component_audit_v2.csv` dependency, is
  `tables/final_campaign_inventory.csv` (self-excluded by construction).
- Repository files created by this campaign:
  - `docs/dr10_flux_blending_contract.md`
  - `scripts/download_dr10_scene_triplets.py`
  - `scripts/download_dr10_probe_catalogs.py`
  - `scripts/audit_dr10_scene_triplets.py`
  - `scripts/correct_dr10_scene_probe_audit.py`
  - `scripts/supplement_dr10_scene_probe_v3.py`
  - `scripts/finalize_dr10_model_probe.py`
- Append-only artifact groups created: the preserved incident run
  `outputs/runs/dr10_model_probe_20260711_155820/` and the successful run
  `outputs/runs/dr10_model_probe_20260711_160018/`.
- No tracked file was modified; the index is empty. No commit, stage, push,
  merge, deletion, training run, or blend manifest occurred.

Final Git status snapshot:

```text
{status}
```
"""
    audit.write_text(outputs[4], report)

    repo_paths = [
        Path("docs/dr10_flux_blending_contract.md"),
        Path("scripts/download_dr10_scene_triplets.py"),
        Path("scripts/download_dr10_probe_catalogs.py"),
        Path("scripts/audit_dr10_scene_triplets.py"),
        Path("scripts/correct_dr10_scene_probe_audit.py"),
        Path("scripts/supplement_dr10_scene_probe_v3.py"),
        Path(__file__),
    ]
    inventory = final_inventory_rows(
        args.run_dir, args.incident_dir, args.foundation_run, repo_paths
    )
    audit.write_csv(outputs[5], inventory)
    print(
        json.dumps(
            {
                "alignment_pass": alignment_pass,
                "closure_pass_sources": closure_pass_sources,
                "matched_morphology_sources": len(matched_sources),
                "central_only_reliable": central_isolation,
                "extraction_options_approved": extraction_approved,
                "exact_psf_samples_validated": exact_psf,
                "blending_may_proceed": False,
                "final_inventory_entries": len(inventory),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
