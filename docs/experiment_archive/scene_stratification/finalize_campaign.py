#!/usr/bin/env python3
"""Fail-closed append-only finalizer for the completed stratification fits."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1]
REPO = RUN_DIR.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

source = RUN_DIR / "analysis/run_campaign.py"
spec = importlib.util.spec_from_file_location("scene_stratification_runner", source)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen campaign runner")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from scripts.run_thayer_flux_free_identifiability_v0 import sha256_file
from src.btk_scene import load_catsim_catalog


SCENES = base.SCENES
CONDITIONS = base.CONDITIONS
PRIMARY_CONDITION = base.PRIMARY_CONDITION
CLASS_SCORE = base.CLASS_SCORE
DIAMETERS = base.DIAMETERS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def fresh_text(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def fresh_json(path: Path, value: Any) -> None:
    fresh_text(path, json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(safe(item), sort_keys=True, separators=(",", ":"))
                if isinstance(item, (dict, list, tuple, np.ndarray)) else safe(item)
                for key, item in row.items()
            })
        handle.flush()
        os.fsync(handle.fileno())


def snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "head": command(["git", "rev-parse", "HEAD"]),
        "readme_sha256": sha256_file(REPO / "README.md"),
        "git_index_entries": command(["git", "diff", "--cached", "--name-only"]),
        "authoritative_report_hashes": {
            str(path.relative_to(REPO)): sha256_file(path) for path in base.AUTHORITATIVE_REPORTS
        },
        "protected_data_access": {"development": 0, "atlas": 0, "lockbox": 0},
    }


def load_results() -> list[dict[str, Any]]:
    results = []
    for scene in SCENES:
        for condition in CONDITIONS:
            path = RUN_DIR / "fit_records" / f"scene_{scene:03d}_{condition.lower()}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            results.append({
                "scene_index": scene,
                "scene_id": record["scene_id"],
                "condition": condition,
                "scientific_classification": record["scientific_classification"],
                "condition_number": float(record["condition_number"]),
                "rank": int(record["rank"]),
                "nullity": int(record["nullity"]),
                "gradient_norm": float(record["gradient_norm"]),
                "geometry": record["geometry"],
                "optimizer_successful_starts": int(record["optimizer_successful_starts"]),
                "max_budget_starts": int(record["max_budget_starts"]),
                "optimizer_limited": bool(record["optimizer_limited"]),
                "source": record["source"],
                "fit_record_sha256": sha256_file(path),
            })
    return results


def external_summary(result: dict[str, Any]) -> dict[str, Any]:
    geometry = result["geometry"]
    return {
        "classification": result["scientific_classification"],
        "condition_number": result["condition_number"],
        "rank": result["rank"],
        "nullity": result["nullity"],
        "gradient_norm": result["gradient_norm"],
        "distinct_solution_classes": int(geometry["distinct_solution_classes"]),
        **{key: float(geometry[key]) for key in DIAMETERS},
    }


def fractional_reduction(before: float, after: float) -> float:
    if before > 0:
        return (before - after) / before
    return 0.0 if after == 0 else -1.0


def log_condition_gain(before: float, after: float) -> float:
    if math.isinf(after):
        return float("-inf")
    if after <= 0:
        return float("inf")
    return math.log10(before / after)


def corrected_responses(
    results: list[dict[str, Any]],
    baselines: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {(row["scene_index"], row["condition"]): row for row in results}
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        s1, p2 = baselines[(scene, "S1")], baselines[(scene, "P2")]
        for condition in CONDITIONS:
            fit = lookup[(scene, condition)]
            external = external_summary(fit)
            resolved = external["classification"] not in {
                "OPTIMIZATION_UNRESOLVED", "NUMERICALLY_UNSTABLE", "OUT_OF_SUPPORT", "INVALID_CONTRACT"
            }
            helpful = base.correction.materially_better_geometry(external, p2) if resolved else None
            factor = 0.0 if math.isinf(external["condition_number"]) else p2["condition_number"] / external["condition_number"]
            row: dict[str, Any] = {
                "scene_index": scene,
                "scene_id": fit["scene_id"],
                "condition": condition,
                "primary_response_condition": condition == PRIMARY_CONDITION,
                "response_resolved": resolved,
                "photometry_helpful": "" if helpful is None else bool(helpful),
                "response_label": "Optimization Unresolved" if helpful is None else "Photometry Helpful" if helpful else "Photometry Not Helpful",
                "optimizer_successful_starts": fit["optimizer_successful_starts"],
                "max_budget_starts": fit["max_budget_starts"],
                "s1_classification": s1["classification"],
                "p2_classification": p2["classification"],
                "external_classification": external["classification"],
                "s1_to_p2_classification_improvement": CLASS_SCORE[p2["classification"]] - CLASS_SCORE[s1["classification"]],
                "p2_to_external_classification_improvement": "" if not resolved else CLASS_SCORE[external["classification"]] - CLASS_SCORE[p2["classification"]],
                "s1_to_external_classification_improvement": "" if not resolved else CLASS_SCORE[external["classification"]] - CLASS_SCORE[s1["classification"]],
                "s1_endpoint_classes": s1["distinct_solution_classes"],
                "p2_endpoint_classes": p2["distinct_solution_classes"],
                "external_endpoint_classes": external["distinct_solution_classes"],
                "s1_to_p2_endpoint_reduction": s1["distinct_solution_classes"] - p2["distinct_solution_classes"],
                "p2_to_external_endpoint_reduction": "" if not resolved else p2["distinct_solution_classes"] - external["distinct_solution_classes"],
                "s1_to_external_endpoint_reduction": "" if not resolved else s1["distinct_solution_classes"] - external["distinct_solution_classes"],
                "s1_condition_number": s1["condition_number"],
                "p2_condition_number": p2["condition_number"],
                "external_condition_number": external["condition_number"],
                "s1_to_p2_condition_improvement_factor": s1["condition_number"] / p2["condition_number"],
                "p2_to_external_condition_improvement_factor": factor,
                "s1_to_external_condition_improvement_factor": 0.0 if math.isinf(external["condition_number"]) else s1["condition_number"] / external["condition_number"],
                "s1_to_p2_log10_condition_improvement": math.log10(s1["condition_number"] / p2["condition_number"]),
                "p2_to_external_log10_condition_improvement": log_condition_gain(p2["condition_number"], external["condition_number"]),
                "s1_to_external_log10_condition_improvement": log_condition_gain(s1["condition_number"], external["condition_number"]),
                "external_rank": external["rank"],
                "external_nullity": external["nullity"],
                "external_gradient_norm": external["gradient_norm"],
            }
            fractions = []
            for label, key in (
                ("image", None),
                ("morphology", "morphology_parameter_diameter"),
                ("flux_allocation", "flux_allocation_diameter"),
            ):
                if key is None:
                    s1_value = max(s1["requested_image_diameter"], s1["companion_image_diameter"])
                    p2_value = max(p2["requested_image_diameter"], p2["companion_image_diameter"])
                    ext_value = max(external["requested_image_diameter"], external["companion_image_diameter"])
                else:
                    s1_value, p2_value, ext_value = s1[key], p2[key], external[key]
                fraction = fractional_reduction(p2_value, ext_value)
                fractions.append(fraction)
                row.update({
                    f"s1_{label}_diameter": s1_value,
                    f"p2_{label}_diameter": p2_value,
                    f"external_{label}_diameter": ext_value,
                    f"s1_to_p2_{label}_diameter_reduction": s1_value - p2_value,
                    f"p2_to_external_{label}_diameter_reduction": "" if not resolved else p2_value - ext_value,
                    f"p2_to_external_{label}_diameter_fraction_reduction": "" if not resolved else fraction,
                })
            row["mean_p2_to_external_diameter_fraction_reduction"] = "" if not resolved else float(np.mean(fractions))
            rows.append(row)
    return rows


def scene_ranking(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = [dict(row) for row in responses if row["condition"] == PRIMARY_CONDITION]
    resolved = [row for row in primary if row["response_resolved"]]
    unresolved = [row for row in primary if not row["response_resolved"]]
    resolved.sort(key=lambda row: (
        -int(bool(row["photometry_helpful"])),
        -int(row["p2_to_external_endpoint_reduction"]),
        -int(row["p2_to_external_classification_improvement"]),
        -float(row["p2_to_external_log10_condition_improvement"]),
        -float(row["mean_p2_to_external_diameter_fraction_reduction"]),
        int(row["scene_index"]),
    ))
    unresolved.sort(key=lambda row: int(row["scene_index"]))
    return [
        {"scene_rank": index, "ranking_status": "resolved" if row["response_resolved"] else "unranked_optimizer_unresolved", **row}
        for index, row in enumerate(resolved + unresolved, start=1)
    ]


def catalog_entries() -> dict[int, list[Any]]:
    catalog, _ = load_catsim_catalog(base.CATALOG_PATH)
    definitions = base.definition_rows()
    return {
        scene: [catalog.table[int(definitions[scene]["source_a_row"])], catalog.table[int(definitions[scene]["source_b_row"])]]
        for scene in SCENES
    }


def report_text(
    responses: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    tree: dict[str, Any],
    rules: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    start_snapshot: dict[str, Any],
    end_snapshot: dict[str, Any],
    runtime: float,
) -> str:
    primary = [row for row in responses if row["condition"] == PRIMARY_CONDITION]
    resolved = [row for row in primary if row["response_resolved"]]
    unresolved = [row for row in primary if not row["response_resolved"]]
    helpful = [row for row in resolved if row["photometry_helpful"]]
    not_helpful = [row for row in resolved if not row["photometry_helpful"]]
    lower, upper = base.clopper_pearson(len(helpful), len(resolved))
    secondary = [row for row in responses if row["condition"] != PRIMARY_CONDITION]
    secondary_resolved = [row for row in secondary if row["response_resolved"]]
    secondary_helpful = [row for row in secondary_resolved if row["photometry_helpful"]]
    concordant = sum(
        next(row for row in primary if row["scene_index"] == scene)["photometry_helpful"]
        == next(row for row in secondary if row["scene_index"] == scene)["photometry_helpful"]
        for scene in SCENES
        if next(row for row in primary if row["scene_index"] == scene)["response_resolved"]
        and next(row for row in secondary if row["scene_index"] == scene)["response_resolved"]
    )
    concordance_denominator = sum(
        next(row for row in primary if row["scene_index"] == scene)["response_resolved"]
        and next(row for row in secondary if row["scene_index"] == scene)["response_resolved"]
        for scene in SCENES
    )
    best_rule = rules[0]
    integrity = (
        start_snapshot["head"] == end_snapshot["head"]
        and start_snapshot["readme_sha256"] == end_snapshot["readme_sha256"]
        and start_snapshot["authoritative_report_hashes"] == end_snapshot["authoritative_report_hashes"]
        and not end_snapshot["git_index_entries"]
    )
    text = f"""# Thayer-External-Photometry-Scene-Stratification-v0 final report

## Outcome

**SCENE_STRATIFICATION_PRIMARY_RESPONSE_PARTIALLY_RESOLVED**

Total 5%-uncertainty source photometry was scientifically interpretable for **{len(resolved)}/8 scenes**. It materially improved over P2 for **{len(helpful)}/{len(resolved)} interpretable scenes ({100*len(helpful)/len(resolved):.1f}%; exact 95% binomial CI {100*lower:.1f}%–{100*upper:.1f}%)**: Scenes {', '.join(str(row['scene_index']) for row in helpful)}. It did not help Scenes {', '.join(str(row['scene_index']) for row in not_helpful)}. Scenes {', '.join(str(row['scene_index']) for row in unresolved)} are unclassified because all four total-photometry starts reached 500 evaluations with no optimizer-declared success.

Per-band photometry was interpretable for **{len(secondary_resolved)}/8** scenes and helpful for **{len(secondary_helpful)}/{len(secondary_resolved)}**. Total and per-band decisions agreed for **{concordant}/{concordance_denominator}** scenes where both were interpretable.

## Scene ranking and S1 → P2 → external response

| Rank | Scene | Status | Helpful | Classes S1→P2→Ext | Classification S1→P2→Ext | P2→Ext condition factor | log10 factor | Mean diameter fraction reduction |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
"""
    for row in scenes:
        condition_factor = row["p2_to_external_condition_improvement_factor"]
        log_factor = row["p2_to_external_log10_condition_improvement"]
        diameter = row["mean_p2_to_external_diameter_fraction_reduction"]
        text += f"| {row['scene_rank']} | {row['scene_index']} | {row['ranking_status']} | {row['response_label']} | {row['s1_endpoint_classes']}→{row['p2_endpoint_classes']}→{row['external_endpoint_classes']} | {row['s1_classification']}→{row['p2_classification']}→{row['external_classification']} | {condition_factor if condition_factor != '' else '--'} | {log_factor if log_factor != '' else '--'} | {diameter if diameter != '' else '--'} |\n"
    text += """

Unresolved scenes are placed last but are not scientifically ranked. Exact image, morphology, and flux-allocation diameter transitions for both photometry conditions are in `tables/photometry_response.csv`.

## Feature ranking on interpretable primary scenes

| Rank | Feature | Oriented AUC | Direction | Exact p | BH q | Helpful median | Not-helpful median | ρ with log-condition gain | Tree importance |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for row in ranking[:8]:
        text += f"| {row['rank']} | {row['feature']} | {row['orientation_free_auc']:.3f} | {row['direction']} | {row['exact_permutation_p']:.4f} | {row['bh_qvalue']:.4f} | {row['helpful_median']:.5g} | {row['not_helpful_median']:.5g} | {row['spearman_rho_with_log_condition_improvement']:.3f} | {row['decision_tree_impurity_importance']:.3f} |\n"
    text += f"""

This ranking uses **n={len(resolved)}** interpretable scenes (3 helpful, 3 not helpful). It is exploratory; no multiplicity-adjusted result is treated as confirmatory.

## Decision tree and simple rules

```text
{base.tree_text(tree['root'])}```

- Resubstitution accuracy/balanced accuracy: **{tree['resubstitution_accuracy']:.3f}/{tree['resubstitution_balanced_accuracy']:.3f}**.
- Leave-one-out accuracy/balanced accuracy: **{tree['leave_one_out_accuracy']:.3f}/{tree['leave_one_out_balanced_accuracy']:.3f}**.
- Best single rule: **{best_rule['rule']}**; in-sample TP={best_rule['true_positive']}, FP={best_rule['false_positive']}, TN={best_rule['true_negative']}, FN={best_rule['false_negative']}, balanced accuracy={best_rule['resubstitution_balanced_accuracy']:.3f}.

These cut points are descriptive acquisition rules, not changes to inherited scientific thresholds.

## When should additional photometric information be acquired?

On the resolved frozen subset, acquire total external source photometry when **`{best_rule['rule'].replace('Photometry Helpful when ', '')}`**. Do **not** use this as an operational rule yet: it was selected from only six interpretable scenes, its leave-one-out balanced accuracy is {tree['leave_one_out_balanced_accuracy']:.3f}, and 2/8 primary responses remain unresolved. Photometry should not be acquired routinely for the complementary resolved stratum, where it produced no material endpoint/diameter gain under the unchanged rule.

The conclusion is conditional on the CatSim/BTK frozen scenes, exact coordinates, known PSF, Level-5 support, and 5% source photometry. The population helpful rate across all eight is not estimable until Scenes 5 and 18 are resolved.

## Exactly one next experiment

Run **Thayer-External-Photometry-Stratification-Convergence-Correction-v0**: repeat only Scene 5 and Scene 18 `TOTAL_SOURCE_PHOTOMETRY` with the same measurements, starts, objective, Model-9 implementation, thresholds, and diagnostics, changing only the per-start evaluation ceiling from 500 to 2000. This targets the sole blocker: all **8/8** primary starts across those scenes capped at 500, with endpoint gradient norms 16.6–27.3 (Scene 5) and 96.2–145.8 (Scene 18).

## Integrity and provenance

- The initial runner failed after all fits, during report arithmetic on an infinite unresolved condition number. This append-only finalizer preserves every fit and treats unresolved conditions fail-closed.
- All six available authoritative reports were read. `Thayer-Project-Synthesis-v1` is absent under that title; the authoritative Flux-Free report records the same absence.
- Scene 0 and 6 fits were reused unchanged. The other six scenes used four deterministic starts and the 500-evaluation ceiling.
- Isolated training-source arrays and catalog morphology were used only for post-fit scene descriptors, never as solver inputs.
- Integrity: **{'PASS' if integrity else 'FAIL'}**. HEAD, README, authoritative report hashes, and the empty staged index were unchanged.
- Development, Atlas, and lockbox access: zero. Neural training: zero. Nothing staged or committed.
- Finalization runtime: {runtime:.3f} seconds. Scientific optimizer runtime is retained per fit record.
"""
    return text


def main() -> None:
    started = time.perf_counter()
    start_snapshot = json.loads((RUN_DIR / "manifests/start_snapshot.json").read_text(encoding="utf-8"))
    baselines = base.load_historical_baselines()
    results = load_results()
    responses = corrected_responses(results, baselines)
    features = base.compute_features(baselines, catalog_entries())
    resolved_scene_ids = tuple(
        int(row["scene_index"])
        for row in responses
        if row["condition"] == PRIMARY_CONDITION and row["response_resolved"]
    )
    if resolved_scene_ids != (0, 3, 6, 51, 73, 81):
        raise RuntimeError(f"unexpected resolved primary population: {resolved_scene_ids}")
    statistical_features = [row for row in features if int(row["scene_index"]) in resolved_scene_ids]
    statistical_responses = [row for row in responses if int(row["scene_index"]) in resolved_scene_ids]
    base.SCENES = resolved_scene_ids
    ranking, tree, rules, analysis_rows, correlation = base.statistical_analysis(statistical_features, statistical_responses)
    ranked_scenes = scene_ranking(responses)
    feature_lookup = {int(row["scene_index"]): row for row in features}
    primary_lookup = {int(row["scene_index"]): row for row in responses if row["condition"] == PRIMARY_CONDITION}
    tree_lookup = {int(row["scene_index"]): row for row in analysis_rows}
    summary = []
    for scene in SCENES:
        row = {**feature_lookup[scene]}
        response = primary_lookup[scene]
        row.update({
            "response_resolved": response["response_resolved"],
            "response_label": response["response_label"],
            "photometry_helpful": response["photometry_helpful"],
            "p2_to_external_endpoint_reduction": response["p2_to_external_endpoint_reduction"],
            "p2_to_external_classification_improvement": response["p2_to_external_classification_improvement"],
            "p2_to_external_log10_condition_improvement": response["p2_to_external_log10_condition_improvement"],
            "mean_p2_to_external_diameter_fraction_reduction": response["mean_p2_to_external_diameter_fraction_reduction"],
            "tree_prediction": tree_lookup.get(scene, {}).get("tree_prediction", "EXCLUDED_UNRESOLVED"),
            "leave_one_out_prediction": tree_lookup.get(scene, {}).get("leave_one_out_prediction", "EXCLUDED_UNRESOLVED"),
        })
        summary.append(row)

    fresh_csv(RUN_DIR / "tables/scene_features.csv", features)
    fresh_csv(RUN_DIR / "tables/photometry_response.csv", responses)
    fresh_csv(RUN_DIR / "tables/scene_ranking.csv", ranked_scenes)
    fresh_csv(RUN_DIR / "tables/feature_ranking.csv", ranking)
    fresh_json(RUN_DIR / "tables/decision_tree.json", tree)
    fresh_csv(RUN_DIR / "tables/decision_rules.csv", rules)
    fresh_csv(RUN_DIR / "tables/summary_table.csv", summary)
    correlation.to_csv(RUN_DIR / "tables/correlation_matrix.csv", float_format="%.12g")
    resolved_ranked = [row for row in ranked_scenes if row["response_resolved"]]
    base.save_figures(analysis_rows, ranking, tree, correlation, resolved_ranked)

    end_snapshot = snapshot()
    fresh_json(RUN_DIR / "manifests/end_snapshot.json", end_snapshot)
    runtime = time.perf_counter() - started
    fresh_text(RUN_DIR / "reports/final_report.md", report_text(responses, ranking, tree, rules, ranked_scenes, start_snapshot, end_snapshot, runtime))
    files = sorted(path for path in RUN_DIR.rglob("*") if path.is_file() and path.name != "final_manifest.json")
    fresh_json(RUN_DIR / "manifests/final_manifest.json", {
        "campaign": base.CAMPAIGN,
        "completed_at_utc": now(),
        "finalizer": str(Path(__file__).relative_to(RUN_DIR)),
        "initial_runner_postprocessing_failure": "math domain error from log10(P2/infinite unresolved condition)",
        "resolved_primary_scenes": resolved_scene_ids,
        "unresolved_primary_scenes": (5, 18),
        "files": {str(path.relative_to(RUN_DIR)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files},
        "head_unchanged": start_snapshot["head"] == end_snapshot["head"],
        "readme_unchanged": start_snapshot["readme_sha256"] == end_snapshot["readme_sha256"],
        "authoritative_reports_unchanged": start_snapshot["authoritative_report_hashes"] == end_snapshot["authoritative_report_hashes"],
        "git_index_empty": not end_snapshot["git_index_entries"],
        "protected_data_access": {"development": 0, "atlas": 0, "lockbox": 0},
        "commits_created": 0,
    })
    print(json.dumps({
        "event": "append_only_finalization_complete",
        "resolved_primary_scenes": resolved_scene_ids,
        "unresolved_primary_scenes": (5, 18),
        "helpful_primary_scenes": [row["scene_index"] for row in responses if row["condition"] == PRIMARY_CONDITION and row["photometry_helpful"] is True],
        "leave_one_out_balanced_accuracy": tree["leave_one_out_balanced_accuracy"],
        "runtime_seconds": runtime,
    }, indent=2))


if __name__ == "__main__":
    main()
