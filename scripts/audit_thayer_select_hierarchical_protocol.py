#!/usr/bin/env python3
"""Append-only corrective audit for the hierarchical-safety campaign.

This script never opens scene pixels, runs a neural model, fits a head, changes
thresholds, or evaluates development.  It reconstructs the persisted Phase-II
moderate composite label and records why a new blind preregistration cannot be
claimed after the historical hierarchical campaign has already completed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PHASE2 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
HISTORICAL = REPO / "outputs/runs/thayer_select_hierarchical_safety_20260711_225657"

LIMITS = {
    "image": 0.75,
    "flux": 0.30,
    "color": 0.30,
    "centroid": 2.0,
    "null_hallucination": 0.10,
}
UNIQUE_CLASSES = {"VALID_SOURCE", "PERTURBED_VALID"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def load_split(split: str) -> pd.DataFrame:
    manifest = pd.read_csv(PHASE2 / f"manifests/{split}_scene_manifest.csv", low_memory=False)
    if split in {"training", "validation"}:
        outcome = pd.read_csv(PHASE2 / f"tables/{split}_teacher_reliability_labels.csv", low_memory=False)
        actionable = pd.read_csv(PHASE2 / f"tables/{split}_actionable_acceptance_labels.csv", low_memory=False)
        outcome = outcome.merge(
            actionable.drop(columns=["partition", "query_class"]),
            on="scene_id",
            validate="one_to_one",
        )
        provenance = "frozen_condition_c_teacher"
    else:
        outcome = pd.read_csv(PHASE2 / "tables/calibration_per_sample.csv", low_memory=False)
        provenance = "phase2_r1_candidate"
    if manifest.scene_id.tolist() != outcome.scene_id.tolist():
        raise RuntimeError(f"Persisted row alignment failed for {split}")
    frame = manifest[["scene_id", "query_class"]].merge(
        outcome.drop(columns=["query_class", "partition"], errors="ignore"),
        on="scene_id",
        validate="one_to_one",
    )
    frame.insert(1, "partition", split)
    frame["reconstruction_provenance"] = provenance
    return frame


def finite_leq(values: np.ndarray, limit: float) -> np.ndarray:
    return np.isfinite(values) & (values <= limit)


def reconstruct_rows(frame: pd.DataFrame) -> pd.DataFrame:
    query = frame.query_class.astype(str).to_numpy()
    unique = np.isin(query, sorted(UNIQUE_CLASSES))
    null = query == "NULL_SOURCE"
    ambiguous = query == "AMBIGUOUS_SOURCE"
    evaluation_valid = bool_values(frame.evaluation_valid)
    confusion = bool_values(frame.source_confusion)
    catastrophic = bool_values(frame.catastrophic_failure)
    hallucination = bool_values(frame.hallucination)
    forced = bool_values(frame.forced_source_selection)

    values = {
        "image": pd.to_numeric(frame.normalized_rmse, errors="coerce").to_numpy(float),
        "flux": pd.to_numeric(frame.max_relative_flux_error, errors="coerce").to_numpy(float),
        "color": pd.to_numeric(frame.max_color_error_mag, errors="coerce").to_numpy(float),
        "centroid": pd.to_numeric(frame.centroid_error_pixels, errors="coerce").to_numpy(float),
        "null_hallucination": pd.to_numeric(frame.predicted_to_blend_flux_ratio, errors="coerce").to_numpy(float),
    }
    passes = {name: finite_leq(values[name], LIMITS[name]) for name in ("image", "flux", "color", "centroid")}
    unique_success = evaluation_valid & unique
    for name in ("image", "flux", "color", "centroid"):
        unique_success &= passes[name]
    unique_success &= ~confusion & ~catastrophic
    null_success = evaluation_valid & null & ~hallucination & ~catastrophic
    expected_contract = unique_success | null_success
    expected_actionable = unique_success

    original_contract = pd.to_numeric(frame.moderate_success, errors="raise").to_numpy(int).astype(bool)
    original_actionable = pd.to_numeric(frame.moderate_actionable_success, errors="raise").to_numpy(int).astype(bool)

    records: list[dict] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        applicable = {
            "image": bool(unique[index]),
            "flux": bool(unique[index]),
            "color": bool(unique[index]),
            "centroid": bool(unique[index]),
            "confusion": bool(unique[index]),
            "catastrophic": bool(unique[index] or null[index]),
            "null_hallucination": bool(null[index]),
            "forced_source": bool(ambiguous[index]),
        }
        component_pass = {
            "image": bool(passes["image"][index]) if applicable["image"] else None,
            "flux": bool(passes["flux"][index]) if applicable["flux"] else None,
            "color": bool(passes["color"][index]) if applicable["color"] else None,
            "centroid": bool(passes["centroid"][index]) if applicable["centroid"] else None,
            "confusion": bool(not confusion[index]) if applicable["confusion"] else None,
            "catastrophic": bool(not catastrophic[index]) if applicable["catastrophic"] else None,
            "null_hallucination": bool(not hallucination[index]) if applicable["null_hallucination"] else None,
            "forced_source": bool(not forced[index]) if applicable["forced_source"] else None,
        }
        failed = [name for name, value in component_pass.items() if value is False]
        if not evaluation_valid[index]:
            primary = "evaluation_invalid"
        elif ambiguous[index]:
            primary = "ambiguous_forced_negative"
        elif not failed:
            primary = "none"
        else:
            order = ("null_hallucination", "image", "flux", "color", "centroid", "confusion", "catastrophic", "forced_source")
            primary = next(name for name in order if name in failed)
        record = {
            "scene_id": row.scene_id,
            "partition": row.partition,
            "query_class": row.query_class,
            "query_state": "UNIQUE_VALID" if unique[index] else ("NULL" if null[index] else "AMBIGUOUS"),
            "reconstruction_provenance": row.reconstruction_provenance,
            "evaluation_valid": int(evaluation_valid[index]),
            "original_contract_success": int(original_contract[index]),
            "original_composite_label": int(original_actionable[index]),
            "recomputed_contract_success": int(expected_contract[index]),
            "recomputed_composite_label": int(expected_actionable[index]),
            "label_formula_match": int(original_actionable[index] == expected_actionable[index]),
            "contract_formula_match": int(original_contract[index] == expected_contract[index]),
            "composite_logically_meaningful_for_query": int(unique[index]),
            "failed_component_count": len(failed),
            "failure_signature": "+".join(failed) if failed else "none",
            "primary_failure_reason": primary,
        }
        for name in ("image", "flux", "color", "centroid", "confusion", "catastrophic", "null_hallucination", "forced_source"):
            record[f"{name}_applicable"] = int(applicable[name])
            record[f"{name}_pass"] = "NA" if component_pass[name] is None else int(component_pass[name])
        for name in ("image", "flux", "color", "centroid", "null_hallucination"):
            record[f"{name}_value"] = values[name][index] if applicable[name] else math.nan
            record[f"{name}_limit"] = LIMITS[name] if applicable[name] else math.nan
            record[f"{name}_distance_from_threshold"] = LIMITS[name] - values[name][index] if applicable[name] else math.nan
        record["confusion_value"] = int(confusion[index]) if applicable["confusion"] else math.nan
        record["catastrophic_value"] = int(catastrophic[index]) if applicable["catastrophic"] else math.nan
        record["forced_source_value"] = int(forced[index]) if applicable["forced_source"] else math.nan
        records.append(record)
    return pd.DataFrame(records)


def truth_table() -> pd.DataFrame:
    rows: list[dict] = []
    components = ("image", "flux", "color", "centroid", "confusion", "catastrophic")
    for valid in (0, 1):
        for mask in range(2 ** len(components)):
            passes = {name: bool(mask & (1 << bit)) for bit, name in enumerate(components)}
            contract = bool(valid and all(passes.values()))
            rows.append({
                "query_state": "UNIQUE_VALID", "evaluation_valid": valid,
                **{f"{name}_pass": int(value) for name, value in passes.items()},
                "null_hallucination_pass": "NA", "forced_source_pass": "NA",
                "original_contract_success": int(contract), "original_actionable_composite": int(contract),
            })
    for valid in (0, 1):
        for hallucination_pass in (0, 1):
            for catastrophic_pass in (0, 1):
                contract = bool(valid and hallucination_pass and catastrophic_pass)
                rows.append({
                    "query_state": "NULL", "evaluation_valid": valid,
                    **{f"{name}_pass": "NA" for name in ("image", "flux", "color", "centroid", "confusion")},
                    "catastrophic_pass": catastrophic_pass, "null_hallucination_pass": hallucination_pass,
                    "forced_source_pass": "NA", "original_contract_success": int(contract),
                    "original_actionable_composite": 0,
                })
    for valid in (0, 1):
        for forced_source_pass in (0, 1):
            rows.append({
                "query_state": "AMBIGUOUS", "evaluation_valid": valid,
                **{f"{name}_pass": "NA" for name in components},
                "null_hallucination_pass": "NA", "forced_source_pass": forced_source_pass,
                "original_contract_success": 0, "original_actionable_composite": 0,
            })
    return pd.DataFrame(rows)


def audit(run: Path) -> None:
    if not (run / "logs/bootstrap_complete.json").is_file():
        raise RuntimeError("Part A bootstrap gate is absent")
    bootstrap = json.loads((run / "logs/bootstrap_complete.json").read_text())
    if bootstrap.get("status") != "PASS":
        raise RuntimeError("Part A bootstrap did not pass")
    prereg = run / "preregistration"
    prereg.mkdir(exist_ok=False)

    frames = [load_split(split) for split in ("training", "validation", "calibration")]
    audited = pd.concat([reconstruct_rows(frame) for frame in frames], ignore_index=True)
    if audited.scene_id.duplicated().any():
        raise RuntimeError("Duplicate persisted scene IDs")
    write_csv_fresh(run / "tables/original_contract_truth_table.csv", truth_table())
    write_csv_fresh(run / "tables/original_contract_per_row.csv", audited)

    breakdown = (
        audited.groupby(["partition", "query_class", "primary_failure_reason", "failure_signature"], dropna=False)
        .size().rename("rows").reset_index()
    )
    breakdown["fraction_within_partition_query"] = breakdown["rows"] / breakdown.groupby(
        ["partition", "query_class"]
    )["rows"].transform("sum")
    write_csv_fresh(run / "tables/original_contract_failure_breakdown.csv", breakdown)

    unique = audited.query_state == "UNIQUE_VALID"
    null = audited.query_state == "NULL"
    ambiguous = audited.query_state == "AMBIGUOUS"
    near_boundary: dict[str, int] = {}
    for metric in ("image", "flux", "color", "centroid", "null_hallucination"):
        distance = pd.to_numeric(audited[f"{metric}_distance_from_threshold"], errors="coerce")
        near_boundary[metric] = int((distance.abs() <= 0.05 * LIMITS[metric]).sum())

    summary = {
        "rows": len(audited),
        "partition_rows": audited.partition.value_counts().sort_index().to_dict(),
        "query_class_rows": audited.query_class.value_counts().sort_index().to_dict(),
        "positive_rows": audited.groupby("partition").original_composite_label.sum().astype(int).to_dict(),
        "prevalence": audited.groupby("partition").original_composite_label.mean().to_dict(),
        "label_formula_mismatches": int((audited.label_formula_match == 0).sum()),
        "contract_formula_mismatches": int((audited.contract_formula_match == 0).sum()),
        "null_contract_success_but_composite_negative": int((null & (audited.original_contract_success == 1) & (audited.original_composite_label == 0)).sum()),
        "ambiguous_rows_forced_negative": int(ambiguous.sum()),
        "ambiguous_forced_source_candidates": int((ambiguous & (audited.forced_source_value == 1)).sum()),
        "unique_single_component_noncatastrophic_negatives": int((unique & (audited.failed_component_count == 1) & (audited.catastrophic_value == 0) & (audited.original_composite_label == 0)).sum()),
        "unique_catastrophic_negatives": int((unique & (audited.catastrophic_value == 1) & (audited.original_composite_label == 0)).sum()),
        "near_boundary_rows_by_metric": near_boundary,
        "reconstruction_provenance_by_partition": audited.groupby("partition").reconstruction_provenance.first().to_dict(),
    }

    required = {
        "preregistration/hierarchical_safety_preregistration.md",
        "docs/original_reliability_contract_postmortem.md",
        "tables/original_contract_truth_table.csv",
        "tables/original_contract_failure_breakdown.csv",
        "diagnostics/original_contract_audit.md",
        "tables/frozen_reconstructor_audit.csv",
        "diagnostics/frozen_reconstructor_report.md",
    }
    historical_missing = sorted(name for name in required if not (HISTORICAL / name).is_file() and not (REPO / name).is_file())
    protocol = {
        "historical_campaign": str(HISTORICAL.relative_to(REPO)),
        "historical_development_evaluation_complete": (HISTORICAL / "logs/development_evaluation_complete.json").is_file(),
        "historical_final_report_complete": (HISTORICAL / "reports/final_report.md").is_file(),
        "required_artifacts_absent_when_historical_campaign_completed": historical_missing,
        "blind_preregistration_now_possible": False,
        "new_neural_inference": False,
        "new_head_training": False,
        "new_calibration": False,
        "new_development_evaluation": False,
        "lockbox_scene_or_pixel_access": 0,
        "decision": "STOP_BEFORE_NEW_INFERENCE_OR_TRAINING",
    }

    prereg_text = f"""# Hierarchical safety preregistration status

**Status: RETROSPECTIVE AND INVALID AS A BLIND PREREGISTRATION.**

The historical hierarchical campaign at `{protocol['historical_campaign']}`
already fitted safety heads, froze a policy, and evaluated development before
this required file existed. Creating a document now cannot retroactively meet
the instruction to hash the preregistration before fitting any head.

The requested semantic definitions, risk formulas, moderate limits, training
and calibration populations, metrics, success gates, and analysis plan remain
recorded in the attached campaign request and the tracked hierarchical-safety
documents. They are not re-declared here as unseen choices because historical
development results are already known. This corrective campaign therefore
stops before any new inference, fitting, calibration, policy freeze, or
development evaluation. The sealed lockbox remains untouched.
"""
    prereg_path = prereg / "hierarchical_safety_preregistration.md"
    write_text_fresh(prereg_path, prereg_text)
    write_text_fresh(prereg / "hierarchical_safety_preregistration.sha256", sha256_file(prereg_path) + "\n")

    audit_text = f"""# Original moderate contract audit

Status: **BOOLEAN REAPPLICATION PASS; SCIENTIFIC COMPOSITE DEFECT CONFIRMED.**

The exact persisted `moderate_actionable_success` target was reconstructed for
all {summary['rows']:,} training, validation, and calibration rows. There were
{summary['label_formula_mismatches']} actionable-label and
{summary['contract_formula_mismatches']} underlying-contract reapplication
mismatches. Thus no implementation mismatch was found.

The target is nevertheless logically heterogeneous. UNIQUE_VALID rows pass
only when evaluation is valid, NRMSE <= 0.75, maximum band flux error <= 0.30,
maximum color error <= 0.30 mag, centroid error <= 2 pixels, and neither source
confusion nor catastrophic failure occurs. NULL rows can pass the underlying
null-outcome contract, but every NULL actionable label is forced to zero;
{summary['null_contract_success_but_composite_negative']:,} persisted NULL rows
show that distinction. Every one of the {summary['ambiguous_rows_forced_negative']:,}
AMBIGUOUS rows is also forced negative, including
{summary['ambiguous_forced_source_candidates']:,} internal forced-source
candidates. These ordinary negatives are mixed with mild one-component valid
failures ({summary['unique_single_component_noncatastrophic_negatives']:,}) and
catastrophic valid failures ({summary['unique_catastrophic_negatives']:,}).

Undefined valid-query metrics fail closed; undefined NULL/AMBIGUOUS source
metrics are explicitly not applicable in the corrective table. The persisted
formulas did not silently convert not-applicable values to ordinary metric
negatives, but the final binary target did collapse query invalidity, mild
quality failures, source confusion, and catastrophe into the same negative
class.

Training/validation outcomes came from the frozen Condition-C teacher, whereas
calibration outcomes came from the Phase-II R1 candidate. The Boolean formula
is identical, but the reconstruction provenance is not; this is target-domain
drift and prevents a clean interpretation of the earlier calibration collapse.

Because the required preregistration and this postmortem were absent before the
historical fitting and development evaluation, the corrective campaign stops
before all new neural inference and training.
"""
    write_text_fresh(run / "diagnostics/original_contract_audit.md", audit_text)
    write_json_fresh(run / "logs/original_contract_audit_summary.json", summary)
    write_json_fresh(run / "logs/historical_protocol_compliance_audit.json", protocol)

    report = f"""# Hierarchical safety corrective campaign report

## Outcome

**Stopped before new inference or training.** Part A provenance passed, and the
original moderate composite was reconstructed exactly for {summary['rows']:,}
persisted rows with zero Boolean reapplication mismatches. The historical
hierarchical campaign cannot be certified as compliant with the attached
sequence because its preregistration and original-contract postmortem were
missing before head fitting and one-time development evaluation.

The original composite label is scientifically defective as a direct learned
recoverability target: it mixes query invalidity, mild metric failures, source
confusion, and catastrophe, while not-applicable source metrics differ by query
class. Calibration also uses a different reconstruction provenance from
training/validation. These are design/provenance defects, not a newly detected
Boolean-code mismatch.

No model inference, feature extraction, head fitting, calibration, policy
change, development evaluation, or lockbox access occurred in this corrective
campaign. The historical complete-policy classification remains **FAILURE**;
it is not reinterpreted or retuned here.

## Exact corrective experiment

Begin a genuinely new, prospectively preregistered train/validation/calibration-
only feasibility campaign. Use one frozen reconstruction provenance for every
labelled split, perform the full original-contract and partition audits before
fitting, and require a nondegenerate calibration-only coverage gate before any
new development manifest is generated. Do not reuse the already evaluated
historical development set and do not access the lockbox.
"""
    write_text_fresh(run / "reports/final_report.md", report)
    write_json_fresh(run / "logs/corrective_campaign_complete.json", {
        "status": "STOPPED_PROTOCOL_NONCOMPLIANCE",
        "part_a_pass": True,
        "original_contract_boolean_reapplication_pass": True,
        "new_inference": False,
        "new_training": False,
        "new_development_evaluation": False,
        "lockbox_accessed": False,
        "completed_at_unix": time.time(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve():
        raise RuntimeError("Run path must be under outputs/runs")
    audit(run)
    print(run.relative_to(REPO))


if __name__ == "__main__":
    main()
