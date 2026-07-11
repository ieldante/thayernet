#!/usr/bin/env python3
"""Create append-only final finding and checkpoint-integrity tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FINAL_STATUS: dict[str, tuple[str, str, bool]] = {
    "I001": ("fixed", "Grouped exact-pixel/exact-coordinate split has zero verified cross-split leakage.", False),
    "I002": ("fixed_for_claims", "Correction-field semantics synchronized across core docs; physical separation remains out of scope.", False),
    "I003": ("fixed_for_future", "Grouped manifests contain global source/group IDs, seeds, parameters, hashes, and exact replay; historical omission is preserved.", False),
    "I004": ("fixed", "Prediction-count mismatch raises; aligned sample IDs are required.", False),
    "I005": ("fixed_for_future", "Grouped tables report total and regional valid counts; historical files are unchanged.", False),
    "I006": ("fixed", "29/29 deterministic metric audit checks pass.", False),
    "I007": ("documented_limitation", "Benchmark is explicitly framed as synthetic display-RGB restoration, not calibrated flux injection.", False),
    "I008": ("quantified_limitation", "Input clipping is quantified; grouped output metrics report clipped and unclipped values separately.", False),
    "I009": ("documented_limitation", "Target-centrality shortcut remains a future control.", False),
    "I010": ("documented_limitation", "Padded-mask size-ratio compression is quantified; estimator replacement is future work.", False),
    "I011": ("documented_limitation", "Both pixel scales and angular-size ratios are retained; matching/resampling is future work.", False),
    "I012": ("fixed", "Timestamp containment, collision refusal, and failure-time integrity are enforced.", False),
    "I013": ("fixed", "Full training/evaluation entry points reject silent CPU fallback.", False),
    "I014": ("fixed", "Loss-core and evaluation-core definitions are separately versioned/documented.", False),
    "I015": ("documented_limitation", "Training/evaluation halo geometries are distinguished; future halo tuning must unify them.", False),
    "I016": ("documented_limitation", "Legacy generation_difficulty is explicitly parameter metadata, not model difficulty.", False),
    "I017": ("partially_fixed", "Grouped rows retain target/contaminant group IDs; group-clustered uncertainty remains pending.", False),
    "I018": ("fixed", "All 13,000 grouped rows replay exactly, including blend and three mask hashes.", False),
    "I019": ("documented_future_risk", "Completed comparator was correct; future runs must pin explicit path, kind, semantics, and SHA.", False),
    "I020": ("partially_fixed", "Seeds and environment are logged; only one grouped training seed was run.", False),
    "I021": ("partially_fixed", "Full package versions/code hashes are saved; requirements remain unpinned.", False),
    "I022": ("fixed", "Finite aligned denominators and tie/missing counts are used in grouped evaluation.", False),
    "I023": ("fixed_for_development", "Claims now distinguish original, grouped-development, diagnostic exposure, and future final results.", False),
    "I024": ("documented", "ResUNet remains an 8k architecture ablation, not a matched-budget causal comparison.", False),
    "I025": ("documented", "RGB channel convention and dataset SHA are stored in grouped provenance.", False),
    "I026": ("accepted_low_risk", "Corner double-counting is quantified and retained for historical generator parity.", False),
    "I027": ("partially_fixed", "Sources are called unblended/unfiltered; 356 heuristic candidates await blinded review.", False),
    "I028": ("pass", "Grouped training/evaluation reconfirmed blended-RGB-only model input.", False),
    "I029": ("pass", "All grouped comparators share exact sample IDs, blends, targets, and masks.", False),
    "I030": ("pass", "Best-validation checkpoint kind and SHA are pinned; final is separate.", False),
    "I031": ("pass", "Affected masks remain prediction-independent and replay-hash verified.", False),
    "I032": ("resolved_as_development", "Grouped retrain remains strong at 28.81x normal; original 32.3x is historical context only.", False),
    "I033": ("pass_known_limitation", "No local no-duplicate dataset exists; explicit grouping is used.", False),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_checkpoints() -> pd.DataFrame:
    rows = []
    for path in sorted((PROJECT_ROOT / "outputs" / "checkpoints").glob("*.pth")):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def refuse(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final audit output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit_run_dir.resolve()

    findings_out = audit / "tables" / "audit_findings_final_status_corrected.csv"
    inventory_out = audit / "tables" / "checkpoint_inventory_after_corrected.csv"
    integrity_out = audit / "tables" / "checkpoint_integrity_final_corrected.csv"
    provenance_out = audit / "logs" / "finalization_provenance_corrected.json"
    refuse([findings_out, inventory_out, integrity_out, provenance_out])

    original = pd.read_csv(audit / "tables" / "audit_findings.csv", float_precision="round_trip")
    if set(original["finding_id"]) != set(FINAL_STATUS):
        raise ValueError("Final status mapping does not exactly cover original findings")
    original["final_status"] = original["finding_id"].map(lambda value: FINAL_STATUS[value][0])
    original["final_evidence"] = original["finding_id"].map(lambda value: FINAL_STATUS[value][1])
    original["retraining_must_wait_final"] = original["finding_id"].map(lambda value: FINAL_STATUS[value][2])
    additional = pd.DataFrame(
        [
            {
                "finding_id": "I034",
                "severity": "blocker",
                "category": "final_test_independence",
                "file_or_function": "outputs/runs/final_test_manifest_prep_20260710_061737",
                "description": "The earlier provisional final pool became exposed to the later grouped train/validation manifests.",
                "evidence": "Of 1,000 sources, 683 map to grouped train, 173 to validation, 144 to test; actual grouped blends use 499 in train and 91 in validation (590 union).",
                "risk": "Using this pool as final would leak model-development sources into the final estimate.",
                "recommended_fix": "After model/protocol freeze, allocate a fresh untouched group-disjoint final source partition and never tune on it.",
                "fixed_now": "no; old pool preserved and superseded",
                "retraining_must_wait": True,
                "final_status": "unresolved_final_claim_blocker",
                "final_evidence": "Overlap independently reproduced and saved; optional seed2 was not launched.",
                "retraining_must_wait_final": True,
            },
            {
                "finding_id": "I035",
                "severity": "medium",
                "category": "training_comparability",
                "file_or_function": "historical v0.2 versus grouped v0.2 retrain",
                "description": "The checkpoint comparison confounds split repair with training budget and seed.",
                "evidence": "Historical v0.2 used 12,000 blends; grouped retrain used the requested 8,000 and one training seed.",
                "risk": "The performance gap cannot be causally attributed only to duplicate-safe splitting.",
                "recommended_fix": "Use matched budgets and independent grouped training seeds before causal attribution.",
                "fixed_now": "documented; no extra training launched due final-test blocker",
                "retraining_must_wait": False,
                "final_status": "documented_limitation",
                "final_evidence": "Reports separate the confounds and avoid a leakage-only causal claim.",
                "retraining_must_wait_final": False,
            },
            {
                "finding_id": "I036",
                "severity": "low",
                "category": "configuration_provenance",
                "file_or_function": "grouped training logs/provenance.json",
                "description": "Embedded full_config retains defaults while effective CLI settings are recorded separately.",
                "evidence": "full_config contains 500/100/3 defaults; run_config and checkpoint metadata correctly contain 8000/1000/20 effective settings.",
                "risk": "A reader could mistake base config defaults for effective run settings.",
                "recommended_fix": "Label base config and resolved effective config explicitly in future runs.",
                "fixed_now": "documented in final report",
                "retraining_must_wait": False,
                "final_status": "documented_low_risk",
                "final_evidence": "Effective command, run config, checkpoint config, and history agree.",
                "retraining_must_wait_final": False,
            },
        ]
    )
    final_findings = pd.concat([original, additional], ignore_index=True)
    final_findings.to_csv(findings_out, index=False)

    before = pd.read_csv(audit / "tables" / "checkpoint_inventory_before.csv", float_precision="round_trip")
    before["path"] = before["path"].map(
        lambda raw: str(
            (Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / Path(raw)).resolve()
        )
    )
    after = inventory_checkpoints()
    after.to_csv(inventory_out, index=False)
    before_by_path = before.set_index("path")
    after_by_path = after.set_index("path")
    comparison_rows = []
    for path, row in before_by_path.iterrows():
        present = path in after_by_path.index
        current = after_by_path.loc[path] if present else None
        unchanged = bool(
            present
            and int(current["size_bytes"]) == int(row["size_bytes"])
            and int(current["mtime_ns"]) == int(row["mtime_ns"])
            and current["sha256"] == row["sha256"]
        )
        comparison_rows.append(
            {
                "path": path,
                "present_after": present,
                "size_unchanged": bool(present and int(current["size_bytes"]) == int(row["size_bytes"])),
                "mtime_ns_unchanged": bool(present and int(current["mtime_ns"]) == int(row["mtime_ns"])),
                "sha256_unchanged": bool(present and current["sha256"] == row["sha256"]),
                "all_unchanged": unchanged,
            }
        )
    integrity = pd.DataFrame(comparison_rows)
    integrity.to_csv(integrity_out, index=False)
    if not integrity["all_unchanged"].all():
        raise RuntimeError("A protected checkpoint changed during the audit")
    new_paths = sorted(set(after["path"]).difference(before["path"]))
    if len(new_paths) != 2 or not all("grouped_retrain_20260710_110917" in path for path in new_paths):
        raise RuntimeError(f"Unexpected new checkpoints: {new_paths}")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_findings": len(original),
        "final_findings": len(final_findings),
        "protected_checkpoint_count": len(before),
        "checkpoint_count_after": len(after),
        "all_protected_checkpoints_unchanged": True,
        "new_checkpoint_paths": new_paths,
        "status": "pass",
    }
    provenance_out.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
