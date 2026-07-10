"""Create the code/dataflow correctness audit and structured finding inventory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import train as gd_train


DEFAULT_DATASET = PROJECT_ROOT / "data/Galaxy10_DECals.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
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


def finding(
    finding_id: str,
    severity: str,
    category: str,
    file_or_function: str,
    description: str,
    evidence: str,
    risk: str,
    recommended_fix: str,
    fixed_now: str,
    retraining_must_wait: bool,
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "severity": severity,
        "category": category,
        "file_or_function": file_or_function,
        "description": description,
        "evidence": evidence,
        "risk": risk,
        "recommended_fix": recommended_fix,
        "fixed_now": fixed_now,
        "retraining_must_wait": retraining_must_wait,
    }


def findings() -> pd.DataFrame:
    rows = [
        finding("I001", "blocker", "source_split", "src/data.py:split_dataset", "Historical split assigns HDF5 rows, not duplicate/object groups.", "29 exact-pixel and 27 exact-coordinate pairs cross seed-42 splits; two exact train/test pairs reach the actual first-5000/first-1000 modeling pools.", "Validation selection and test estimates are not object-independent.", "Build and verify a union-find grouped split using exact pixels and exact coordinates before any new claim-bearing training.", "no; Part 5 is the required resolution", True),
        finding("I002", "high", "task_semantics", "src/blend.py:blend_pair and residual datasets", "The learned residual is a blend-to-target correction field, not pure contaminant flux.", "The target may be blurred before contamination; noise and clipping are applied; training target is blended minus original target.", "Pure source-separation wording overstates what is supervised and evaluated.", "Freeze correction-field/restoration terminology for this grouped reproduction; reserve contaminant-flux claims for a redesigned simulator.", "partially; audit terminology fixed, repository docs still require synchronization", True),
        finding("I003", "high", "reproducibility", "src/blend.py:generate_blends and historical CSVs", "Most historical samples omitted global source IDs and per-sample independent seeds.", "32 of 56 historical artifacts lack IDs; the original normal table required exact RNG-stream reconstruction.", "Historical samples cannot be independently reidentified from their result table alone.", "Require global source/group IDs, sample seed, parameters, generator/data/split hashes, and pixel/mask hashes in grouped manifests.", "historical omission preserved; src.blend now retains local indices; grouped manifests still required", True),
        finding("I004", "high", "evaluation_alignment", "scripts/run_stress_test.py:evaluate_samples", "A zip-based evaluator could silently truncate on prediction-count mismatch.", "The old loop used zip(samples,predictions) without a length assertion.", "A short prediction list could yield plausible but incomplete aggregate metrics.", "Assert equal counts and immutable sample-ID alignment before evaluating.", "yes; mismatch now raises and is unit-tested", False),
        finding("I005", "high", "metric_reporting", "aggregate_metrics in weighted/v0.3/ResUNet scripts", "Regional aggregate n historically counted all rows while pandas skipped NaN empty masks.", "Original v0.2 normal core MSE used 858 valid rows but reported n=1000.", "Regional coverage was overstated although paired method comparisons remained fair.", "Report n_total and valid counts for affected/core/non-core/halo metrics.", "yes for future outputs; historical files unchanged", False),
        finding("I006", "high", "metric_tests", "src/utils.py and scripts/metric_correctness_audit.py", "No canonical automated metric suite existed.", "Duplicated formulas had drifted in core/halo naming and edge-case handling.", "Future metric changes could silently alter claims.", "Run deterministic formula, mask, coverage, clipping, color, and alignment tests before grouped evaluation.", "yes; 29/29 audit checks pass", False),
        finding("I007", "high", "blending_physics", "src/data.py:normalise_images; src/blend.py:blend_pair", "Display RGB is added directly and the target alone may be blurred.", "RGB bytes are divided by 255; no linear-light/FITS calibration or shared PSF model is used.", "Results are not calibrated astronomical flux separation.", "Frame as controlled RGB cutout restoration; later add calibrated/linear injection and a shared PSF/noise model.", "scope fixed in audit; physical limitation unresolved", False),
        finding("I008", "high", "input_clipping", "src/blend.py:blend_pair", "Composite clipping is material in bright/overlap suites.", "Thirty-row-per-suite replay found mean contaminant-relative clip loss about 26% compact and 15-17% in hard/high-core/halo suites.", "Some examples become partly inpainting and brightness_scale is not effective flux ratio.", "Record pre/postclip flux and saturation; add low-saturation strata; preserve unclipped hashes.", "diagnostics added; generator behavior intentionally unchanged for historical reproduction", False),
        finding("I009", "high", "centrality_shortcut", "src/blend.py:estimate_central_source_mask and blend_pair", "Targets remain centered while a center-selected contaminant is shifted.", "Fresh development masks had median centroid distance 1.78 px and p95 12.94 px; only contaminants move.", "A network can exploit preserve-center/remove-offset shortcuts.", "Add role swaps, target translation, near-zero shifts, and a centrality-only baseline in a future benchmark.", "documented, not removed in historical reproduction", False),
        finding("I010", "high", "size_validity", "src/blend.py:extract_source_foreground", "Stored size ratio is based on a padded extraction mask and compresses true size differences.", "Synthetic sigma 2/24 maps to stored radius ratio 0.450; sigma 32/24 saturates at 1.0.", "Compact-suite interpretation is weaker than its name suggests.", "Use half-light radius or moments for measurement and a separate generous extraction mask.", "quantified; not changed for historical reproduction", False),
        finding("I011", "high", "metadata_use", "training split loaders", "Pixel scale is loaded but ignored in pairing and size constraints.", "6.3-8.2% of provisional suite pairs mix 0.262 and 0.524 arcsec/pixel sources.", "Pixel-size constraints need not be angular-size constraints.", "Record both scales/angular ratio; later match or resample pixel scale.", "diagnostics added; behavior unchanged", False),
        finding("I012", "high", "output_safety", "scripts/train_v03_color_structure_unet.py:make_run_paths", "Delta previously allowed arbitrary/reused run directories and normal-exit-only integrity finalization.", "run-dir was not contained under outputs/runs and child mkdir used exist_ok=True.", "Future runs could mix artifacts or omit failure integrity logs.", "Enforce containment/collision-safe paths and exit-time integrity finalization.", "yes", False),
        finding("I013", "medium", "device_safety", "full-run entry points", "CPU-fallback rejection was only partially deployed.", "Legacy weighted/residual/stress/size/evaluation scripts used permissive resolve_device.", "A future full run could silently execute on CPU.", "Use resolve_accelerator for every full training/evaluation entry point.", "yes; all detected full-run call sites now use resolve_accelerator", False),
        finding("I014", "medium", "core_semantics", "loss_core_mask_v02 vs evaluation_core_mask_p85_v1", "Training loss core and reported evaluation core are different masks.", "Loss uses >=55% aperture maximum centered at (H-1)/2; evaluation uses aperture p85 centered at H/2.", "Core-weighted training and core affected MSE must not be described as the same region.", "Version and test both definitions; store both in configs/manifests.", "yes", False),
        finding("I015", "medium", "halo_semantics", "v0.3 halo loss vs evaluation halo", "v0.3 square max-pool training halo differs from Manhattan evaluation halo.", "A one-pixel radius-five fixture gives 120 versus 60 ring pixels.", "Halo-loss claims can imply geometry equivalence that does not exist.", "Use one canonical geometry or explicitly name both.", "documented; unresolved for future v0.3 halo tuning", False),
        finding("I016", "medium", "difficulty_label", "src/blend.py:_compute_difficulty", "generation_difficulty is a parameter heuristic, not measured task/model difficulty.", "Current blur/noise ranges never hit two hard thresholds and label ordering overlaps measured severity.", "Difficulty-stratified claims can be misleading.", "Rename as generator_parameter_difficulty_v1 and use measured severity separately.", "pending grouped-manifest naming", True),
        finding("I017", "medium", "source_independence", "provisional suite manifests", "Rows reuse sources heavily and are not independent source observations.", "Across 5,000 rows, most sources occur in both roles and compact uses substantially fewer unique sources than rows.", "Naive per-row bootstrap intervals would be optimistic.", "Report reuse and use target/contaminant-group clustered intervals.", "diagnosed; grouped evaluator must retain group IDs", True),
        finding("I018", "medium", "replay_validation", "provisional final manifest", "Historical final-manifest validation did not hash generated pixels/masks and originally replayed only one row per suite.", "Metadata hashes do not prove numerical pixel replay.", "A consumer/code drift could change generated samples while metadata still looks valid.", "Hash float32 blends and masks; validate all grouped rows before inference.", "150 stratified rows now exact; full grouped-manifest replay still required", True),
        finding("I019", "medium", "comparator_pinning", "scripts/train_resunet_v04.py:discover_delta_checkpoint", "Newest-name comparator discovery is not identity/SHA pinned.", "A later matching *_best.pth could be selected by filename ordering.", "An incompatible or wrong experiment could silently become comparator.", "Require explicit paths, kind, semantics, architecture, and expected SHA.", "completed ResUNet config was correct; future risk unresolved", False),
        finding("I020", "medium", "determinism", "seed_everything and DataLoader generators", "Seeds are controlled but deterministic algorithm enforcement is not logged.", "No torch deterministic-algorithm setting or independent retraining exists.", "Bitwise reproducibility and training-seed robustness are not established.", "Log backend/determinism settings and run a second independent training seed only as preliminary evidence.", "environment/seeds logged; independent seed not yet run", False),
        finding("I021", "medium", "dependencies", "requirements.txt", "Dependency versions are unpinned.", "Generator masks and metrics depend on NumPy/SciPy/scikit-image behavior.", "Fresh environments may not reproduce exact masks or blends.", "Record full package versions and code/data hashes for every run; later freeze a lockfile.", "runtime package versions captured; requirements remain unpinned", False),
        finding("I022", "medium", "empty_masks", "historical boolean rates", "NaN masked values compare False and can bias win/worse denominators.", "Empty affected/core masks return NaN; ordinary boolean comparisons count them as non-wins/non-failures.", "Rates can have incorrect denominators.", "Filter finite aligned pairs and report valid/missing/tie counts.", "yes in canonical aligned_pair_outcomes; grouped evaluator must use it", True),
        finding("I023", "medium", "claim_safety", "README and docs", "Some final/current-best wording remains stronger than the grouped-validation status.", "Paper plan and briefing contain isolated final/clear-current-best wording; all 32x statements require development qualifiers.", "Readers may treat the old random-index result as final.", "Use original development-split wording and reserve final for a fresh locked grouped test.", "partially; claim inventory created, docs update pending", False),
        finding("I024", "medium", "architecture_ablation", "ResUNet v0.4 run", "ResUNet used 8k blends while v0.2 Moderate used 12k.", "The comparison is not a strict architecture-only matched-budget ablation.", "Niche gains/losses cannot be attributed solely to residual blocks.", "Describe as an 8k architecture candidate; use a matched control for causal claims.", "negative-result status correct; wording update pending", False),
        finding("I025", "medium", "channel_semantics", "Galaxy10 HDF5 schema", "The pipeline assumes channel order is RGB but the HDF5 has no channel-order attribute.", "images has shape N,H,W,3 and docs/dataset provenance identify RGB display cutouts; root attrs are empty.", "Color metrics rely on an external dataset convention rather than self-describing metadata.", "Record the RGB assumption and dataset provenance/hash in every manifest.", "documented in audit; grouped manifests must record it", True),
        finding("I026", "low", "background_estimation", "src/blend.py:estimate_background", "Border concatenation counts corner pixels twice.", "A 100-source audit found median zero difference and at most one uint8 step versus unique-border sampling.", "Small background-estimate bias is possible.", "Use a unique border mask in a future generator version, not during historical reproduction.", "quantified; unchanged", False),
        finding("I027", "low", "source_quality", "source artifact audit", "Clean-source labels are not validated.", "356 heuristic candidates remain pending manual review and include genuine morphology false positives.", "Calling unfiltered targets clean can imply artifact-free data.", "Use unblended/unfiltered terminology until blinded labels are frozen.", "mostly documented", False),
        finding("I028", "info", "model_input", "direct and residual Dataset/train loops", "No target, mask, source ID, or blend-parameter tensor is fed to the model.", "Model calls receive only blended RGB; target/core/affected values are used in supervised loss or metrics.", "No accidental model-input leakage found.", "Retain input-schema assertions in grouped trainer.", "pass", False),
        finding("I029", "info", "evaluation_fairness", "same-run prediction loops", "Comparators receive identical generated blends and use identical masks.", "All models are predicted from the same batch tensor and indexed into one per-sample row.", "Same-run comparisons are internally fair.", "Use manifest sample IDs to preserve this property across grouped runs.", "pass", False),
        finding("I030", "info", "checkpoint_selection", "v0.2/Delta/ResUNet train scripts", "Main evaluations use best-validation checkpoints and save final separately.", "v0.2 reloads best_state; Delta best epoch 20; ResUNet best epoch 19.", "No best/final checkpoint substitution found.", "Pin checkpoint kind and SHA in grouped evaluation.", "pass", False),
        finding("I031", "info", "affected_metric", "src/utils.py:affected_region_mask", "Affected mask is based only on blended versus target.", "Prediction is not a function argument; 29/29 metric tests pass.", "No model-dependent mask leakage found.", "Retain immutable per-sample mask hashes.", "pass", False),
        finding("I032", "info", "headline_arithmetic", "saved v0.2 per-sample tables", "Old 32x arithmetic remains plausible after metric and leak-severity checks.", "Macro ratio 32.3137x; micro ratio 36.2768x; clean-subset macro ratio 32.2150x.", "The issue is protocol validity, not a discovered arithmetic collapse.", "Run grouped old-checkpoint evaluation and grouped retraining before replacing claim status.", "diagnostic pass; not a final claim", True),
        finding("I033", "info", "dataset_variant", "local filesystem search", "No Galaxy10_DECals_NoDuplicated.h5-style file was found locally.", "Spotlight and targeted project/Documents/Downloads searches returned no match; prior audit also searched home.", "The pipeline necessarily uses the duplicated source file unless a new dataset is obtained.", "Group duplicates explicitly and audit any future no-duplicate file rather than trusting its name.", "pass/known limitation", False),
    ]
    order = {"blocker": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["severity"].map(order)
    return frame.sort_values(["_order", "finding_id"]).drop(columns="_order")


def claim_inventory() -> pd.DataFrame:
    patterns = {
        "final": re.compile(r"\bfinal\b", re.I),
        "held_out": re.compile(r"held[- ]out", re.I),
        "not_lucky": re.compile(r"not (?:just )?a lucky|not lucky", re.I),
        "survey": re.compile(r"survey[- ]ready|survey[- ]grade", re.I),
        "32x": re.compile(r"\b32(?:\.3)?x\b", re.I),
    }
    paths = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").rglob("*.md"))]
    rows: list[dict[str, Any]] = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            for term, pattern in patterns.items():
                if not pattern.search(line):
                    continue
                lower = line.lower()
                if term == "32x":
                    status = (
                        "qualified_on_line"
                        if "mse" in lower or "identity" in lower or "ratio" in lower
                        else "needs_development_context_review"
                    )
                elif term == "survey":
                    status = (
                        "limitation_or_negative_claim"
                        if any(token in lower for token in ("not ", "no ", "does not", "cannot"))
                        else "unsupported_positive_claim_review"
                    )
                elif term == "not_lucky":
                    status = "unsupported_training_seed_language_review"
                elif term == "held_out":
                    status = (
                        "qualified_as_development"
                        if "development" in lower
                        else "needs_development_qualifier_or_nearby_context"
                    )
                else:
                    status = (
                        "protocol_or_caveat_context"
                        if any(
                            token in lower
                            for token in (
                                "protocol",
                                "manifest",
                                "future",
                                "provisional",
                                "not final",
                                "needed",
                                "required",
                                "before",
                                "after",
                            )
                        )
                        else "manual_final_claim_review"
                    )
                rows.append(
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "line": line_number,
                        "term": term,
                        "text": line.strip(),
                        "review_status": status,
                    }
                )
    return pd.DataFrame(rows)


def package_versions() -> dict[str, Any]:
    names = (
        "torch",
        "numpy",
        "scipy",
        "scikit-image",
        "pandas",
        "h5py",
        "matplotlib",
        "PyYAML",
        "Pillow",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.replace("\n", " "),
        "packages": versions,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def main() -> int:
    args = parse_args()
    run_dir = resolve_run(args.run_dir)
    dataset = resolve_existing(args.dataset)
    finding_frame = findings()
    claims = claim_inventory()
    with h5py.File(dataset, "r") as handle:
        labels = handle["ans"][:]
        schema_rows = [
            {
                "dataset": name,
                "shape": "x".join(str(value) for value in value.shape),
                "dtype": str(value.dtype),
                "root_attribute_count": len(handle.attrs),
            }
            for name, value in handle.items()
        ]
    schema = pd.DataFrame(schema_rows)
    label_counts = {
        str(int(label)): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True))
    }

    cpu_rejected = False
    try:
        gd_train.resolve_accelerator("cpu")
    except RuntimeError:
        cpu_rejected = True
    resolve_calls = subprocess.run(
        ["rg", "-n", r"resolve_device\(", "scripts"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()

    safe_csv(run_dir / "tables/audit_findings.csv", finding_frame)
    safe_csv(run_dir / "tables/scientific_claim_inventory.csv", claims)
    safe_csv(run_dir / "tables/dataset_schema.csv", schema)
    safe_json(run_dir / "logs/package_versions.json", package_versions())
    code_paths = [
        "src/data.py",
        "src/blend.py",
        "src/utils.py",
        "src/models.py",
        "src/train.py",
        "scripts/train_weighted_residual_unet.py",
        "scripts/run_stress_test.py",
        "scripts/train_v03_color_structure_unet.py",
        "scripts/train_resunet_v04.py",
        "scripts/metric_correctness_audit.py",
        "scripts/blending_correctness_audit.py",
        "scripts/leak_severity_audit.py",
    ]
    safe_json(
        run_dir / "logs/infrastructure_code_hashes.json",
        {path: sha256_file(PROJECT_ROOT / path) for path in code_paths},
    )

    counts = finding_frame["severity"].value_counts().to_dict()
    manual_claims = int(
        claims["review_status"].isin(
            {
                "needs_development_context_review",
                "unsupported_positive_claim_review",
                "unsupported_training_seed_language_review",
                "needs_development_qualifier_or_nearby_context",
                "manual_final_claim_review",
            }
        ).sum()
    )
    report = f"""# Infrastructure Correctness Audit

## Executive gate

Retraining gate: **CLOSED at Part 1** because the historical row-index split is
not duplicate/object-group safe. No model-input target leakage, mask leakage,
blend-parameter leakage, comparator realignment, residual-sign error, or main
affected-MSE arithmetic bug was found.

Findings: `{counts}`. The source-group blocker is intentionally resolved only by
the verified Part 5 grouped split. High-severity blend-definition limitations
do not prevent reproducing the historical controlled RGB restoration task once
the task is labeled correctly; they do prevent calibrated/survey-grade or pure
contaminant-flux claims.

## 1. Dataset loading

- Active file: `data/Galaxy10_DECals.h5`, SHA-256
  `{sha256_file(dataset)}`.
- Images: `17736 x 256 x 256 x 3`, uint8, observed range 0--255.
- Code converts to float32 by division by 255; it treats the last dimension as
  RGB. The HDF5 itself has no channel-order attribute, so RGB is a documented
  external dataset convention.
- Labels: ten integer classes with counts `{label_counts}`.
- Metadata: RA, Dec, redshift, and pixel scale; no object/catalog ID and no HDF5
  root attributes. Redshift contains 92 NaNs; RA/Dec and pixel scale are finite.
- No local `Galaxy10_DECals_NoDuplicated.h5`-style file was found. The current
  pipeline uses the duplicated file.

## 2. Source splitting

`src.data.split_dataset` deterministically shuffles row indices with NumPy
PCG64 seed 42, then takes integer 70/15/15 boundaries. It returns shuffled
arrays, discards global indices, and regenerates the split rather than loading a
frozen manifest. Targets and contaminants are selected from the same assigned
row partition, so row-role containment is correct. Exact-content/object groups
can still cross roles indirectly; this is the blocker.

## 3. Blend generation

Target/contaminant selection is without replacement within a sample. Normal,
validation, test, and targeted generators use distinct documented seed streams,
but many historical samples share one sequential RNG and omit IDs. Foreground
extraction is source-only rather than full-rectangle pasting. Border median,
center-biased component selection, dilation/soft taper, shifts, target blur,
brightness clipping, composite clipping, and post-composite noise are detailed
in `diagnostics/blending_algorithm_audit.md`.

The generator is deterministic under frozen code/data/source-pool/seed state:
the corrected 150-row replay has exact float32 blend parity. The first replay
attempt exposed a float32/float64 percentile boundary mismatch in the newly
centralized core helper; that failure remains preserved and the corrected
append-only table passes.

## 4. Affected masks and metrics

Affected mask: `mean_channel(abs(blended-target)) > threshold`, independent of
prediction. Whole metrics, masked RGB channel normalization, PSNR/SSIM range,
clipped/unclipped state, Delta E, gradient proxy, macro/micro aggregation,
empty-mask coverage, improvement ratio, and aligned win rates were checked.
The metric gate passes 29/29 tests. Historical regional `n` coverage was
misreported, but the primary affected-MSE values were not.

## 5. Model input/output semantics

- Direct: blended RGB -> clipped target reconstruction.
- Residual/correction model: blended RGB -> unconstrained correction field;
  reconstruction is `blended - predicted_field`, then clipped for primary
  display-range metrics.
- The model receives only blended RGB. Targets, masks, and blend metadata are
  used only for supervised loss, diagnostics, or metrics.
- Training/evaluation residual sign is consistent.

## 6. Training loop

Claim-bearing scripts use `model.train()` for optimization, `model.eval()` and
`torch.no_grad()` for validation, zero gradients before backward, step once per
batch, shuffle training with seeded generators, and never mix validation/test
rows into training at the row level. v0.2 best state is selected by the actual
weighted validation loss and final state is saved separately. Loss weights are
background 1, affected extra 3, core-affected extra 2; the normalized optimized
loss is the logged validation selection value.

All detected full-run entry points now call `resolve_accelerator`; explicit CPU
is rejected: `{cpu_rejected}`. Remaining permissive resolve_device calls under
scripts after the fix: `{resolve_calls or 'none'}`.

## 7. Evaluation loop

Models are in eval/no-grad mode, main comparisons load best checkpoints, and
same-run comparators consume one shared sample list/batch tensor and one shared
mask. A silent zip-truncation risk was fixed. Historical local integer indices
are insufficient for cross-run joins; grouped evaluation must use unique
manifest sample IDs and assert exact one-to-one ordered alignment.

Same-generated-suite values must be labeled separately from the old headline;
the grouped old-checkpoint evaluation is diagnostic because the checkpoint was
trained on the historical split.

## 8. File/output safety

New audit outputs are inside one ignored timestamped master run and safe writers
refuse collisions. Historical runs/checkpoints remain untouched. Delta run-path
containment and exit-time checkpoint-integrity behavior were hardened. Generic
training checkpoint writes now refuse overwrite. Automatic public-doc updates
must stay disabled for grouped experiments until the audit decision is final.

## 9. Reproducibility

The environment, dataset fingerprint, package versions, code hashes, checkpoint
hashes, split seeds, and audit commands are captured. Historical manifests are
incomplete; grouped manifests must contain global/group IDs, independent sample
seeds, all parameters, source/split/generator hashes, and generated blend/mask
hashes. Requirements remain unpinned, so exact environment capture is required.

## 10. Scientific-claim safety

Every README/docs occurrence of `final`, `held-out`, `not lucky`, survey-ready/
survey-grade, and 32x is listed in
`tables/scientific_claim_inventory.csv`. `{manual_claims}` occurrences require
manual development/final-context review by the conservative scanner; many are
procedural or nearby-qualified rather than actual overclaims.

The supportable status before grouped evaluation is:

- v0.2 Moderate: current best **original development-split** model;
- old 32x: arithmetically plausible and not heavily driven by the known exact
  duplicates, but not a final object-independent estimate;
- Delta and ResUNet: same-run development ablations;
- no survey-grade, calibrated-flux, pure-contaminant, or training-seed-
  robustness claim.

## Part-1 decision

Do not retrain yet. Proceed to leakage severity and then a grouped source split
only after the metric/blending gates pass. Before training, resolve the blocker,
freeze correction-field terminology, build fully replayable grouped manifests,
and verify zero source/group/hash/coordinate overlap.
"""
    safe_text(run_dir / "diagnostics/infrastructure_correctness_audit.md", report)
    safe_json(
        run_dir / "logs/infrastructure_audit_summary.json",
        {
            "finding_counts": counts,
            "retraining_gate": "closed_source_group_blocker",
            "cpu_full_run_rejected": cpu_rejected,
            "claim_inventory_rows": len(claims),
            "claim_manual_review_rows": manual_claims,
            "dataset_sha256": sha256_file(dataset),
        },
    )
    print(f"Infrastructure audit saved: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
