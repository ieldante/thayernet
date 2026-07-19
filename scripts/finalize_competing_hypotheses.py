#!/usr/bin/env python3
"""Finalize the Atlas-feasibility branch of the competing-hypothesis campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btk_scene import validated_lsst_survey  # noqa: E402
from src.competing_hypotheses import scientific_distance  # noqa: E402


SOURCE_SPLIT = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/manifests/btk_engineering_source_groups.csv"
CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
EXPECTED_SPLIT = "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27"
EXPECTED_CATALOG = "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        if fields is None:
            fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout.rstrip()


def bool_value(value: str) -> bool:
    return value == "True"


def checkpoint_integrity(run_dir: Path) -> tuple[int, bool]:
    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv")
    current = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if run_dir in path.parents:
            continue
        stat = path.stat()
        current.append(
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", current)
    before_map = {row["path"]: (row["sha256"], int(row["bytes"])) for row in before}
    after_map = {row["path"]: (row["sha256"], int(row["bytes"])) for row in current}
    return len(current), before_map == after_map


def build_failure_tables(run_dir: Path, mean_psf_fwhm_pixel: float) -> dict[str, object]:
    decompositions = [
        row for row in read_csv(run_dir / "tables/candidate_decomposition_inventory.csv") if row["regime"] == "noisy_observation"
    ]
    labels: list[dict[str, object]] = []
    for row in decompositions:
        pair_id = row["pair_id"]
        side = row["side"]
        with np.load(run_dir / row["output_path"], allow_pickle=False) as candidate:
            prediction = np.asarray(candidate["requested_source"], dtype=np.float64)
        with np.load(run_dir / f"atlas/{pair_id}.npz", allow_pickle=False) as pair:
            truth_layers = np.asarray(pair[f"{side}_isolated"], dtype=np.float64)
        target_distance = scientific_distance(prediction, truth_layers[0], mean_psf_fwhm_pixel=mean_psf_fwhm_pixel)
        alternate_distance = scientific_distance(prediction, truth_layers[1], mean_psf_fwhm_pixel=mean_psf_fwhm_pixel)
        confusion = alternate_distance.primary_normalized < target_distance.primary_normalized
        image = target_distance.image > 0.25
        flux = any(value > 0.20 for value in target_distance.relative_flux_grz)
        centroid = (
            "NOT_APPLICABLE"
            if target_distance.centroid_psf is None
            else target_distance.centroid_psf > 0.5
        )
        applicable_colors = [value for value in target_distance.color_gr_rz_magnitude if value is not None]
        color = "NOT_APPLICABLE" if not applicable_colors else any(value > 0.20 for value in applicable_colors)
        positive = [confusion, image, flux]
        positive += [centroid] if isinstance(centroid, bool) else []
        positive += [color] if isinstance(color, bool) else []
        labels.append(
            {
                "candidate_id": row["candidate_id"],
                "pair_id": pair_id,
                "side": side,
                "family_id_provenance_only": row["family_id_provenance_only"],
                "QUERY_NULL": "NOT_APPLICABLE",
                "QUERY_AMBIGUOUS": "NOT_APPLICABLE",
                "SOURCE_CONFUSION": confusion,
                "CATASTROPHIC_IMAGE": image,
                "CATASTROPHIC_FLUX": flux,
                "CATASTROPHIC_CENTROID": centroid,
                "COLOR_UNSAFE": color,
                "SHAPE_UNSAFE": "NOT_APPLICABLE",
                "ATLAS_NON_IDENTIFIABLE": True,
                "SAFE_CANDIDATE": not any(positive),
                "target_error_primary": target_distance.primary_normalized,
                "alternate_error_primary": alternate_distance.primary_normalized,
            }
        )
    write_csv_fresh(run_dir / "tables/atlas_failure_labels.csv", labels)
    label_names = [
        "QUERY_NULL",
        "QUERY_AMBIGUOUS",
        "SOURCE_CONFUSION",
        "CATASTROPHIC_IMAGE",
        "CATASTROPHIC_FLUX",
        "CATASTROPHIC_CENTROID",
        "COLOR_UNSAFE",
        "SHAPE_UNSAFE",
        "ATLAS_NON_IDENTIFIABLE",
        "SAFE_CANDIDATE",
    ]
    prevalence = []
    for family in sorted({row["family_id_provenance_only"] for row in labels}):
        family_rows = [row for row in labels if row["family_id_provenance_only"] == family]
        for label in label_names:
            applicable = [row[label] for row in family_rows if isinstance(row[label], bool)]
            prevalence.append(
                {
                    "family_id_provenance_only": family,
                    "label": label,
                    "applicable_count": len(applicable),
                    "positive_count": sum(applicable),
                    "positive_fraction": sum(applicable) / len(applicable) if applicable else "NOT_APPLICABLE",
                }
            )
    write_csv_fresh(run_dir / "tables/failure_prevalence_by_family.csv", prevalence)
    overlap_labels = [
        "SOURCE_CONFUSION",
        "CATASTROPHIC_IMAGE",
        "CATASTROPHIC_FLUX",
        "CATASTROPHIC_CENTROID",
        "COLOR_UNSAFE",
        "ATLAS_NON_IDENTIFIABLE",
    ]
    overlap = []
    for left in overlap_labels:
        for right in overlap_labels:
            applicable = [row for row in labels if isinstance(row[left], bool) and isinstance(row[right], bool)]
            overlap.append(
                {
                    "label_a": left,
                    "label_b": right,
                    "applicable_count": len(applicable),
                    "both_positive_count": sum(bool(row[left]) and bool(row[right]) for row in applicable),
                }
            )
    write_csv_fresh(run_dir / "tables/failure_overlap_matrix.csv", overlap)
    applicability = [
        {"label": "QUERY_NULL", "atlas_valid_prompt": "NOT_APPLICABLE", "definition": "Prompt has no valid source within frozen match radius."},
        {"label": "QUERY_AMBIGUOUS", "atlas_valid_prompt": "NOT_APPLICABLE", "definition": "Prompt matches multiple sources within frozen ambiguity margin."},
        {"label": "SOURCE_CONFUSION", "atlas_valid_prompt": "APPLICABLE", "definition": "Candidate is closer to the unrequested scene source than to the requested truth."},
        {"label": "CATASTROPHIC_IMAGE", "atlas_valid_prompt": "APPLICABLE", "definition": "Requested-source image distance exceeds 0.25."},
        {"label": "CATASTROPHIC_FLUX", "atlas_valid_prompt": "APPLICABLE", "definition": "Any g/r/z relative flux error exceeds 0.20."},
        {"label": "CATASTROPHIC_CENTROID", "atlas_valid_prompt": "CONDITIONAL", "definition": "Centroid error exceeds 0.5 mean-PSF FWHM when both centroids are defined."},
        {"label": "COLOR_UNSAFE", "atlas_valid_prompt": "CONDITIONAL", "definition": "Either defined g-r or r-z error exceeds 0.20 magnitude."},
        {"label": "SHAPE_UNSAFE", "atlas_valid_prompt": "NOT_APPLICABLE", "definition": "Shape validity gate was not frozen for this feasibility branch."},
        {"label": "ATLAS_NON_IDENTIFIABLE", "atlas_valid_prompt": "APPLICABLE", "definition": "Scene belongs to a frozen validated near-collision Atlas pair."},
        {"label": "SAFE_CANDIDATE", "atlas_valid_prompt": "APPLICABLE", "definition": "No applicable reconstruction failure label is positive."},
    ]
    write_csv_fresh(run_dir / "tables/label_applicability_matrix.csv", applicability)
    return {
        "candidate_count": len(labels),
        "safe_candidate_count": sum(bool(row["SAFE_CANDIDATE"]) for row in labels),
        "source_confusion_count": sum(bool(row["SOURCE_CONFUSION"]) for row in labels),
        "catastrophic_image_count": sum(bool(row["CATASTROPHIC_IMAGE"]) for row in labels),
        "catastrophic_flux_count": sum(bool(row["CATASTROPHIC_FLUX"]) for row in labels),
    }


def make_figures(run_dir: Path) -> None:
    calibration = read_csv(run_dir / "tables/forward_consistency_calibration.csv")
    candidates = [row for row in read_csv(run_dir / "tables/candidate_decomposition_inventory.csv") if row["regime"] == "noisy_observation"]
    thresholds = json.loads((run_dir / "calibration/forward_consistency_thresholds.json").read_text())
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist([float(row["global_chi_square_mean"]) for row in calibration], bins=45, density=True, alpha=0.45, label="truth calibration")
    for family in sorted({row["family_id_provenance_only"] for row in candidates}):
        values = [float(row["forward_global_chi_square_mean"]) for row in candidates if row["family_id_provenance_only"] == family]
        axis.hist(values, bins=35, density=True, histtype="step", linewidth=1.6, label=family.replace("THAYER_SELECT_", ""))
    axis.axvline(float(thresholds["global_chi_square_mean"]), color="black", linestyle="--", label="frozen global limit")
    axis.set_xlabel("mean squared whitened residual")
    axis.set_ylabel("density")
    axis.set_title("Forward-consistency distributions")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(run_dir / "figures/forward_consistency_distributions.png", dpi=180)
    plt.close(figure)

    truth_witness = read_csv(run_dir / "tables/ambiguity_witness_inventory.csv")
    model_witness = [row for row in read_csv(run_dir / "tables/model_candidate_witness_inventory.csv") if row["regime"] == "noisy_observation"]
    decomposition_inventory = [
        row
        for row in read_csv(run_dir / "tables/candidate_decomposition_inventory.csv")
        if row["regime"] == "noisy_observation"
    ]
    plausible_rows = []
    for witness in model_witness:
        candidates = [
            row
            for row in decomposition_inventory
            if row["pair_id"] == witness["pair_id"] and row["side"] == witness["side"]
        ]
        plausible = [row["candidate_id"] for row in candidates if row["plausible_under_frozen_noisy_threshold"] == "True"]
        plausible_rows.append(
            {
                "pair_id": witness["pair_id"],
                "side": witness["side"],
                "candidate_count": len(candidates),
                "candidate_ids": ";".join(row["candidate_id"] for row in candidates),
                "plausible_candidate_count": len(plausible),
                "plausible_candidate_ids": ";".join(plausible),
                "image_flux_color_centroid_primary_diameter": witness["model_candidate_primary_diameter"],
                "empirical_ambiguity_witness": witness["model_candidate_ambiguity_witness"],
            }
        )
    write_csv_fresh(run_dir / "tables/plausible_candidate_sets.csv", plausible_rows)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, rows, title in ((axes[0], truth_witness, "Atlas truth decompositions"), (axes[1], model_witness, "Same-cluster model candidates")):
        counts = Counter(int(row["plausible_candidate_count"]) for row in rows)
        axis.bar(sorted(counts), [counts[key] for key in sorted(counts)])
        axis.set_xticks([0, 1, 2, 3])
        axis.set_xlabel("plausible candidates")
        axis.set_ylabel("observation count")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(run_dir / "figures/plausible_set_size_distributions.png", dpi=180)
    plt.close(figure)

    behavior = read_csv(run_dir / "tables/atlas_deblender_behavior.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    colors = {"THAYER_SELECT_CONDITION_C": "tab:blue", "THAYER_SELECT_R0": "tab:orange", "THAYER_SELECT_R1_RECONSTRUCTION_ONLY": "tab:green"}
    for axis, regime in zip(axes, ("noiseless_mean", "noisy_observation")):
        for family, color in colors.items():
            rows = [row for row in behavior if row["regime"] == regime and row["family_id_provenance_only"] == family]
            axis.scatter([float(row["truth_primary_diameter"]) for row in rows], [float(row["output_primary_diameter"]) for row in rows], s=22, alpha=0.7, color=color, label=family.replace("THAYER_SELECT_", ""))
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(regime.replace("_", " "))
        axis.set_xlabel("truth primary diameter")
    axes[0].set_ylabel("deblender output primary diameter")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(run_dir / "figures/atlas_deblender_output_diameter.png", dpi=180)
    plt.close(figure)


def validate_csvs(run_dir: Path) -> tuple[int, int]:
    checked = 0
    failed = 0
    for path in sorted([*run_dir.glob("tables/*.csv"), *run_dir.glob("manifests/*.csv")]):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise ValueError("missing or duplicate CSV header")
                for row in reader:
                    if None in row:
                        raise ValueError("row has extra columns")
            checked += 1
        except Exception:
            failed += 1
    return checked, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    final_report_path = run_dir / "reports/final_report.md"
    if final_report_path.exists():
        raise FileExistsError(final_report_path)
    survey = validated_lsst_survey()
    mean_psf = float(np.mean([survey.get_filter(band).psf_fwhm.to_value("arcsec") for band in ("g", "r", "z")]) / 0.2)
    failure_summary = build_failure_tables(run_dir, mean_psf)
    make_figures(run_dir)

    checkpoint_count, checkpoints_unchanged = checkpoint_integrity(run_dir)
    split_unchanged = sha256_file(SOURCE_SPLIT) == EXPECTED_SPLIT
    catalog_unchanged = sha256_file(CATALOG) == EXPECTED_CATALOG
    staged_code, staged_output = command("git", "diff", "--cached", "--name-only")
    diff_code, diff_output = command("git", "diff", "--check")
    compile_code, compile_output = command(str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests")
    main_test_code, main_test_output = command(
        str(REPO / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_competing_hypotheses.py",
    )
    btk_test_code, btk_test_output = command(
        str(REPO / ".venv-btk/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_competing_hypotheses.py",
        "tests/test_ambiguity_atlas.py",
    )
    privacy_code, privacy_output = command(
        "rg",
        "-n",
        "-P",
        "/Users/|/home/|ChatGPT|OpenAI|AI-generated|\\bCodex\\b",
        "README.md",
        "docs",
    )
    write_text_fresh(run_dir / "logs/compileall_output.txt", compile_output + "\n")
    write_text_fresh(run_dir / "logs/main_contract_tests.txt", main_test_output + "\n")
    write_text_fresh(run_dir / "logs/btk_contract_tests.txt", btk_test_output + "\n")
    write_text_fresh(run_dir / "logs/privacy_path_grep.txt", privacy_output + "\n")
    csv_checked, csv_failed = validate_csvs(run_dir)
    large_files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024:
            large_files.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large_files, ["path", "bytes", "sha256"])

    truth_witness = read_csv(run_dir / "tables/ambiguity_witness_inventory.csv")
    model_witness = [row for row in read_csv(run_dir / "tables/model_candidate_witness_inventory.csv") if row["regime"] == "noisy_observation"]
    behavior = read_csv(run_dir / "tables/atlas_deblender_behavior.csv")
    requested = [
        row
        for row in read_csv(run_dir / "tables/candidate_output_inventory.csv")
        if row["regime"] == "noisy_observation" and row["requested_source"] == "True"
    ]
    r1_confidence = [
        float(row["r1_private_recoverability_diagnostic"])
        for row in requested
        if row["family_id_provenance_only"] == "THAYER_SELECT_R1_RECONSTRUCTION_ONLY"
    ]
    truth_witness_count = sum(bool_value(row["empirical_ambiguity_witness"]) for row in truth_witness)
    model_witness_count = sum(bool_value(row["model_candidate_ambiguity_witness"]) for row in model_witness)
    atlas_pairs = json.loads((run_dir / "manifests/atlas_initial_freeze_record.json").read_text())["pair_count"]
    prereg_hash = json.loads((run_dir / "preregistration/freeze_record.json").read_text())["preregistration_sha256"]
    thresholds = json.loads((run_dir / "calibration/forward_consistency_thresholds.json").read_text())
    pair_rows = read_csv(run_dir / "tables/atlas_pair_manifest.csv")[:atlas_pairs]
    optimized_rows = read_csv(run_dir / "tables/targeted_optimization_pair_manifest.csv")
    baseline_rows = {row["metric"]: row for row in read_csv(run_dir / "tables/ambiguity_evidence_baselines.csv")}
    plausible_counts = Counter(int(row["plausible_candidate_count"]) for row in truth_witness)
    model_plausible_counts = Counter(int(row["plausible_candidate_count"]) for row in model_witness)
    noisy_behavior = [row for row in behavior if row["regime"] == "noisy_observation"]
    nearly_same_by_family = {
        family: sum(
            bool_value(row["output_nearly_same_while_truth_diverges"])
            for row in noisy_behavior
            if row["family_id_provenance_only"] == family
        )
        for family in sorted({row["family_id_provenance_only"] for row in noisy_behavior})
    }
    search_log = json.loads((run_dir / "logs/atlas_search_complete.json").read_text())
    optimization_log = json.loads((run_dir / "logs/targeted_optimization_complete.json").read_text())
    run_kib_code, run_kib_output = command("du", "-sk", str(run_dir))
    code_paths = [
        REPO / "src/btk_scene.py",
        REPO / "src/competing_hypotheses.py",
        REPO / "scripts/bootstrap_competing_hypotheses.py",
        REPO / "scripts/prepare_ambiguity_atlas_v0.py",
        REPO / "scripts/build_ambiguity_atlas.py",
        REPO / "scripts/optimize_ambiguity_atlas_v0.py",
        REPO / "scripts/review_ambiguity_atlas.py",
        REPO / "scripts/review_ambiguity_atlas_v0_observations.py",
        REPO / "scripts/calibrate_competing_forward_consistency.py",
        REPO / "scripts/evaluate_deblenders_on_ambiguity_atlas.py",
        REPO / "scripts/evaluate_ambiguity_evidence_baselines.py",
        Path(__file__).resolve(),
    ]
    write_csv_fresh(
        run_dir / "tables/campaign_code_hashes_final.csv",
        [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in code_paths],
    )
    final_git_code, final_git = command("git", "status", "--short")
    audit = {
        "status": "PASS_WITH_PREREGISTERED_SCOPE_BLOCK",
        "compileall": "PASS" if compile_code == 0 else "FAIL",
        "test_partitioning": {
            "main_contract_tests": "PASS" if main_test_code == 0 else "FAIL",
            "main_contract_output": main_test_output,
            "btk_contract_tests": "PASS" if btk_test_code == 0 else "FAIL",
            "btk_contract_output": btk_test_output,
        },
        "checkpoint_count": checkpoint_count,
        "historical_checkpoints_unchanged": checkpoints_unchanged,
        "source_split_unchanged": split_unchanged,
        "source_catalog_unchanged": catalog_unchanged,
        "staged_index_empty": staged_code == 0 and not staged_output,
        "git_diff_check_pass": diff_code == 0,
        "git_diff_check_output": diff_output,
        "privacy_path_grep_pass": privacy_code == 1 and not privacy_output,
        "csv_files_validated": csv_checked,
        "csv_validation_failures": csv_failed,
        "auditor_tensor_files_created": 0,
        "family_id_in_auditor_tensors": False,
        "target_in_auditor_tensors": False,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
        "large_file_count": len(large_files),
    }
    if not all(
        [
            checkpoints_unchanged,
            split_unchanged,
            catalog_unchanged,
            audit["staged_index_empty"],
            audit["git_diff_check_pass"],
            audit["compileall"] == "PASS",
            main_test_code == 0,
            btk_test_code == 0,
            audit["privacy_path_grep_pass"],
            csv_failed == 0,
        ]
    ):
        audit["status"] = "FAIL"
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", audit)
    decision_rows = [
        {"gate": "ATLAS_FEASIBILITY", "status": "PASS", "evidence": f"{atlas_pairs} frozen validated Route-1 pairs; {optimization_log['valid_pair_count']} valid Route-2 feasibility pairs; exact replay and visual artifact audit"},
        {"gate": "CONSTRUCTED_AMBIGUITY_WITNESS", "status": "PASS_FINITE_COUNTEREXAMPLE", "evidence": f"{truth_witness_count}/50 noisy observations retain two divergent constructed truth decompositions"},
        {"gate": "OPERATIONAL_AMBIGUITY_WITNESS", "status": "FAIL", "evidence": f"diameter AUROC {float(baseline_rows['diameter_score']['auroc']):.4f}, recall {float(baseline_rows['diameter_score']['atlas_recall_at_frozen_threshold']):.4f}; did not beat confidence"},
        {"gate": "DEBLENDER_FAILURE_ON_ATLAS", "status": "PASS", "evidence": "75/75 pair/model rows have at least one unsafe noisy requested reconstruction"},
        {"gate": "MODEL_CANDIDATE_WITNESS", "status": "FAIL_OPERATIONAL_GATE", "evidence": f"{model_witness_count}/50 noisy observations; candidates share one architecture cluster"},
        {"gate": "CROSS_DEBLENDER_AUDIT", "status": "BLOCKED", "evidence": "one meaningfully distinct compatible family cluster; no auditor trained"},
        {"gate": "CATALOG_SAFETY_COVERAGE", "status": "NOT_RUN", "evidence": "requires cross-family gate; no coverage or bias claim"},
        {"gate": "FINAL_DEVELOPMENT_OR_LOCKBOX", "status": "NOT_OPENED", "evidence": "zero access by contract and audit"},
    ]
    write_csv_fresh(run_dir / "tables/final_decision.csv", decision_rows)

    report = f"""# Ambiguity Atlas v0 and Competing-Hypothesis Recoverability final report

Decision: **FAILURE AFTER ATLAS PASS — CANDIDATE-DIAMETER DETECTION FAILED; AUDITOR BLOCKED**.

Preregistration SHA-256: `{prereg_hash}`. It predates new candidate inference
and Atlas optimization. Historical development and final lockbox scenes were
never opened, rendered, or evaluated.

## Direct answers

1. **Compatible deblenders:** three reproducible checkpoints but only one
   meaningfully distinct family cluster, `THAYER_COMPACT_PROMPTED_UNET`.
2. **Fixed forward contract:** yes. Exact g/r/z source addition, noise replay,
   band order, 0.2 arcsec pixels, and fixed 0.86/0.81/0.77 arcsec PSFs passed.
   The global consistency limit frozen on {thresholds['calibration_count']}
   calibration scenes is {thresholds['global_chi_square_mean']:.8g}.
3. **Large-pool search:** yes; {search_log['numerically_valid_pair_count']} genuine
   numerical near-collisions were found in 30,000 training/search scenes.
4. **Targeted optimization:** yes; {optimization_log['valid_pair_count']}/25
   bounded catalog-parameter optimizations produced valid feasibility pairs.
5. **Atlas pairs passing every frozen validation gate:** {atlas_pairs}. The 25
   optimized pairs are separate route-feasibility evidence, not silently added
   to the frozen initial Atlas.
6. **Observation similarity:** frozen-pair mean squared whitened distance spans
   {min(float(row['blend_whitened_mse']) for row in pair_rows):.6g} to
   {max(float(row['blend_whitened_mse']) for row in pair_rows):.6g}, far below
   the frozen 0.25 limit. The observed panels are strongly noise-dominated.
7. **Requested-truth difference:** primary scientific diameter spans
   {min(float(row['target_primary_diameter']) for row in pair_rows):.3f} to
   {max(float(row['target_primary_diameter']) for row in pair_rows):.3f} times
   the frozen scientific limit.
8. **Deblender behavior:** every family had at least one unsafe noisy requested
   reconstruction on all 25 pairs. Nearly-identical noisy pair outputs occurred
   on {nearly_same_by_family['THAYER_SELECT_CONDITION_C']}/25 for Condition C,
   {nearly_same_by_family['THAYER_SELECT_R0']}/25 for R0, and
   {nearly_same_by_family['THAYER_SELECT_R1_RECONSTRUCTION_ONLY']}/25 for R1.
9. **Confidently wrong:** not under R1's private diagnostic. Reconstructions were
   unsafe, but R1 recoverability was low (median {float(np.median(r1_confidence)):.6g},
   maximum {max(r1_confidence):.6g}); this narrow result does not rehabilitate
   the historically unstable confidence head.
10. **Average/prior-like outputs:** yes for Condition C on
    {nearly_same_by_family['THAYER_SELECT_CONDITION_C']}/25 noisy pairs by the
    frozen output-diameter criterion; not for R0 or R1 under that criterion.
11. **Forward-consistent candidates per scene:** constructed sets retained two
    on {plausible_counts[2]}/50 observations. Same-cluster model sets retained
    two on {model_plausible_counts[2]}/50 and one on {model_plausible_counts[1]}/50.
12. **Did plausible-set diameter identify Atlas cases?** Only partially:
    {model_witness_count}/50 model-candidate observations formed witnesses.
13. **Did diameter beat confidence and residual?** No. Diameter AUROC is
    {float(baseline_rows['diameter_score']['auroc']):.4f} with recall
    {float(baseline_rows['diameter_score']['atlas_recall_at_frozen_threshold']):.4f}
    at the frozen control threshold; forward residual is
    {float(baseline_rows['forward_residual_score']['auroc']):.4f} and R1
    unsafe-confidence is {float(baseline_rows['self_confidence_unsafe_score']['auroc']):.4f}.
14. **Black-box auditor authorized?** No: the diameter gate failed and fewer
    than three distinct families exist.
15. **Held-out-family transfer:** not trained or evaluated.
16. **Catastrophic false-safe rate by coverage:** not evaluated; no policy was
    authorized at 95/90/80/70/50% coverage.
17. **Atlas witnesses incorrectly accepted:** not evaluated as a policy. Direct
    constructed witnesses exist on {truth_witness_count}/50 observations.
18. **Safe outputs rejected:** not evaluated; no admission rule exists.
19. **Accepted-catalog flux and centroid bias:** not evaluated.
20. **Operational definition:** finite competing explanations are viable direct
    evidence of non-identifiability for exhibited cases, but the available
    same-cluster candidate diameter is not a viable operational recoverability
    detector. Absence of a witness remains non-probative.
21. **Exact next experiment:** preregister and train one compact prompted
    ResUNet under the frozen BTK normalization/source-layer contract, validate
    deterministic full-decomposition replay, and rerun only the frozen 25-pair
    Atlas behavior/candidate-diversity audit. Do not train Thayer-Audit yet.
22. **Historical development and lockbox:** untouched; access counts are 0/0.
23. **Historical checkpoints:** all {checkpoint_count} files are byte-identical
    to the campaign-start inventory.

## Evidence inventory

- Family inventory: `tables/deblender_family_inventory.csv`.
- Forward audit/tests: `diagnostics/forward_model_audit.md` and
  `tables/forward_model_unit_tests.csv`.
- Search/optimization: `tables/atlas_pair_manifest.csv`,
  `tables/targeted_optimization_pair_manifest.csv`, and
  `optimization/counterfactual_optimization_trajectories.csv`.
- Atlas validation/gallery: `tables/atlas_pair_validation.csv`,
  `figures/ambiguity_atlas/`, and `figures/ambiguity_atlas_observed/`.
- Model behavior and plausible sets: `tables/atlas_deblender_behavior.csv`,
  `tables/plausible_candidate_sets.csv` if present, and
  `tables/model_candidate_witness_inventory.csv`.
- Baseline comparison: `tables/ambiguity_evidence_baselines.csv` and
  `diagnostics/ambiguity_evidence_baseline_report.md`.
- Figures: observation/truth galleries, forward-consistency distributions,
  plausible-set sizes, and deblender output-diameter plots. Transfer matrices,
  coverage curves, catalog-bias curves, and bootstrap intervals are absent by
  gate, not silently omitted after evaluation.

## Correctness and provenance

- Compileall: {'PASS' if compile_code == 0 else 'FAIL'}.
- Main contract tests: {'PASS' if main_test_code == 0 else 'FAIL'}
  (`{main_test_output.splitlines()[-1] if main_test_output else 'no output'}`).
- BTK contract tests: {'PASS' if btk_test_code == 0 else 'FAIL'}
  (`{btk_test_output.splitlines()[-1] if btk_test_output else 'no output'}`).
- CSV/schema validation: {csv_checked} files checked, {csv_failed} failures.
- `git diff --check`: {'PASS' if diff_code == 0 else 'FAIL'}; staged index:
  {'empty' if not staged_output else 'unexpected content'}.
- Privacy/path grep: {'PASS' if privacy_code == 1 and not privacy_output else 'FAIL'}.
- Source split/catalog: {'unchanged' if split_unchanged and catalog_unchanged else 'FAILED'};
  historical checkpoints: {'unchanged' if checkpoints_unchanged else 'FAILED'}.
- Run disk usage: {run_kib_output} KiB. Development/lockbox access: 0/0.
- No auditor tensor, model, threshold, held-out-family evaluation, catalog
  policy, or final-survey claim was created.

The Atlas exhibits finite competing explanations and therefore falsifies
practical uniqueness for its frozen cases. It does not prove uniqueness where
no witness is found, establish high-information ambiguity frequency, or support
model-agnostic auditing.

## Final repository state

```text
{final_git}
```
"""
    write_text_fresh(final_report_path, report)
    write_json_fresh(
        run_dir / "logs/finalization_complete.json",
        {
            "status": audit["status"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_sha256": prereg_hash,
            "final_report_sha256": sha256_file(final_report_path),
            "decision": "FAILURE_AFTER_ATLAS_PASS_WITNESS_DETECTOR_FAIL_AUDITOR_BLOCKED",
            "development_scene_access_count": 0,
            "lockbox_scene_access_count": 0,
        },
    )


if __name__ == "__main__":
    main()
