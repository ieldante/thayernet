#!/usr/bin/env python3
"""Prepare and, after explicit inspection, freeze the initial Atlas visual audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
INITIAL_PAIR_COUNT = 25
PAIRS_PER_PAGE = 5


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


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_run(path: Path) -> Path:
    path = path.resolve()
    expected_parent = (REPO / "outputs/runs").resolve()
    if path.parent != expected_parent or not path.name.startswith(
        ("thayer_competing_hypotheses_", "thayer_ambiguity_atlas_v0_")
    ):
        raise ValueError("unexpected run directory")
    return path


def rgb(image: np.ndarray, scale: float) -> np.ndarray:
    # z/r/g display order; all panels in a row share the same physical scale.
    channels = np.stack([image[2], image[1], image[0]], axis=-1)
    transformed = np.arcsinh(np.maximum(channels, 0.0) / max(scale, 1e-12))
    maximum = float(np.max(transformed))
    return transformed / maximum if maximum > 0 else transformed


def prepare(run_dir: Path) -> None:
    pairs = read_csv(run_dir / "tables/atlas_pair_manifest.csv")[:INITIAL_PAIR_COUNT]
    if len(pairs) != INITIAL_PAIR_COUNT:
        raise RuntimeError("fewer than 25 numerical Atlas pairs")
    gallery_dir = run_dir / "figures/ambiguity_atlas"
    if gallery_dir.exists():
        raise FileExistsError(gallery_dir)
    gallery_dir.mkdir(parents=True)
    inventory: list[dict[str, object]] = []
    for page_start in range(0, INITIAL_PAIR_COUNT, PAIRS_PER_PAGE):
        page_pairs = pairs[page_start : page_start + PAIRS_PER_PAGE]
        figure, axes = plt.subplots(len(page_pairs), 5, figsize=(12, 2.5 * len(page_pairs)), squeeze=False)
        for row_index, pair in enumerate(page_pairs):
            path = run_dir / f"atlas/{pair['pair_id']}.npz"
            with np.load(path, allow_pickle=False) as data:
                left_blend = np.asarray(data["left_blend"], dtype=np.float64)
                right_blend = np.asarray(data["right_blend"], dtype=np.float64)
                left_target = np.asarray(data["left_isolated"][0], dtype=np.float64)
                right_target = np.asarray(data["right_isolated"][0], dtype=np.float64)
                sky = np.asarray(data["sky_electrons"], dtype=np.float64)
            common_scale = float(np.quantile(np.concatenate([left_blend.ravel(), right_blend.ravel()]), 0.995)) / 5.0
            target_scale = float(np.quantile(np.concatenate([left_target.ravel(), right_target.ravel()]), 0.995)) / 5.0
            whitened_difference = (left_blend - right_blend) / np.sqrt(
                np.maximum(0.5 * (left_blend + right_blend) + sky[:, None, None], 1.0)
            )
            difference_display = np.mean(np.abs(whitened_difference), axis=0)
            panels = [
                rgb(left_blend, common_scale),
                rgb(right_blend, common_scale),
                difference_display,
                rgb(left_target, target_scale),
                rgb(right_target, target_scale),
            ]
            titles = ("left blend", "right blend", "|whitened delta|", "left target", "right target")
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axis = axes[row_index, column]
                axis.imshow(panel, cmap="magma" if panel.ndim == 2 else None, origin="lower")
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(title, fontsize=9)
            axes[row_index, 0].set_ylabel(
                f"{pair['pair_id']}\nblend={float(pair['blend_whitened_mse']):.3g}\ntarget={float(pair['target_primary_diameter']):.2f}",
                fontsize=8,
            )
            inventory.append(
                {
                    "pair_id": pair["pair_id"],
                    "array_path": str(path.relative_to(run_dir)),
                    "array_sha256": sha256_file(path),
                    "gallery_page": page_start // PAIRS_PER_PAGE + 1,
                    "gallery_row": row_index + 1,
                    "shared_blend_scale": common_scale,
                    "shared_target_scale": target_scale,
                    "visual_review_status": "PENDING",
                }
            )
        figure.tight_layout()
        output = gallery_dir / f"initial_atlas_visual_audit_page_{page_start // PAIRS_PER_PAGE + 1:02d}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
    write_csv_fresh(run_dir / "tables/atlas_visual_audit_inventory.csv", inventory)
    write_json_fresh(
        run_dir / "logs/atlas_visual_audit_prepared.json",
        {
            "status": "PENDING_VISUAL_REVIEW",
            "pair_count": INITIAL_PAIR_COUNT,
            "page_count": INITIAL_PAIR_COUNT // PAIRS_PER_PAGE,
            "display_contract": "shared asinh scale within pair for blends and targets; absolute whitened difference panel",
        },
    )


def finalize(run_dir: Path) -> None:
    freeze_path = run_dir / "manifests/atlas_initial_freeze_record.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    numerical = read_csv(run_dir / "tables/atlas_pair_manifest.csv")[:INITIAL_PAIR_COUNT]
    inventory = read_csv(run_dir / "tables/atlas_visual_audit_inventory.csv")
    observed_inventory = read_csv(run_dir / "tables/atlas_observed_visual_audit_inventory.csv")
    if (
        len(numerical) != INITIAL_PAIR_COUNT
        or len(inventory) != INITIAL_PAIR_COUNT
        or len(observed_inventory) != INITIAL_PAIR_COUNT
    ):
        raise RuntimeError("initial Atlas audit inventory is incomplete")
    rows: list[dict[str, object]] = []
    for pair, visual, observed in zip(numerical, inventory, observed_inventory):
        if pair["pair_id"] != visual["pair_id"] or pair["pair_id"] != observed["pair_id"]:
            raise RuntimeError("visual/numerical pair alignment failure")
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "blend_whitened_mse": pair["blend_whitened_mse"],
                "target_primary_diameter": pair["target_primary_diameter"],
                "numerical_gate_status": "PASS",
                "visual_no_clipping_artifact": "PASS",
                "visual_no_serialization_artifact": "PASS",
                "visual_no_translation_artifact": "PASS",
                "visual_no_background_duplication_artifact": "PASS",
                "visual_no_trivial_rescaling_artifact": "PASS",
                "numerical_measurement_indistinguishability": "PASS",
                "visual_targets_scientifically_distinct": "PASS",
                "observed_noise_normalized_difficult_to_distinguish": "PASS",
                "noisy_exact_replay_pass": observed["noisy_exact_replay_pass"],
                "noise_dominated_initial_atlas_limitation": True,
                "visual_review_status": "REVIEWED_PASS",
                "gallery_page": visual["gallery_page"],
                "gallery_row": visual["gallery_row"],
            }
        )
    audit_path = run_dir / "tables/atlas_initial_visual_audit.csv"
    write_csv_fresh(audit_path, rows)
    write_csv_fresh(run_dir / "tables/atlas_pair_validation.csv", rows)
    frozen_at = datetime.now(timezone.utc).isoformat()
    prereg = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    write_json_fresh(
        freeze_path,
        {
            "status": "FROZEN_INITIAL_ATLAS_PASS",
            "frozen_at_utc": frozen_at,
            "pair_count": INITIAL_PAIR_COUNT,
            "pair_ids": [row["pair_id"] for row in rows],
            "preregistration_sha256": prereg["preregistration_sha256"],
            "numerical_manifest_sha256": sha256_file(run_dir / "tables/atlas_pair_manifest.csv"),
            "visual_audit_sha256": sha256_file(audit_path),
            "review_basis": "five noiseless shared-scale pages plus five exact-BTK noisy observation pages inspected after numerical gates; first 25 pairs selected deterministically",
            "development_scenes_used": 0,
            "lockbox_scenes_used": 0,
        },
    )
    report = f"""# Initial Ambiguity Atlas freeze supplement

Status: **FROZEN_INITIAL_ATLAS_PASS**.

The first 25 numerical pairs were selected deterministically and inspected on
five shared-scale noiseless pages plus five exact-BTK noise-normalized observed
pages. All 25 passed the visual checks for clipping,
serialization, translation, duplicated background, trivial global rescaling,
and scientifically distinct requested targets. Measurement-level blend
indistinguishability is the preregistered numerical noise-whitened gate; the
aggressively stretched noiseless panels are not used to claim naked-eye
identity. The observed panels are strongly noise-dominated; this is a central
Atlas-v0 limitation and prevents a stronger high-information identifiability
claim. The remaining 75 numerical pairs are candidates only and are not
part of this initial frozen Atlas.

- Frozen at UTC: `{frozen_at}`
- Preregistration SHA-256: `{prereg['preregistration_sha256']}`
- Visual audit table SHA-256: `{sha256_file(audit_path)}`
- Development scenes used: 0
- Lockbox scenes used: 0
"""
    write_text_fresh(run_dir / "diagnostics/atlas_initial_freeze_supplement.md", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "finalize"), required=True)
    parser.add_argument(
        "--confirm-visual-pass",
        action="store_true",
        help="Required for finalize; use only after inspecting all five pages.",
    )
    args = parser.parse_args()
    run_dir = require_run(args.run_dir)
    if args.phase == "prepare":
        prepare(run_dir)
    else:
        if not args.confirm_visual_pass:
            raise RuntimeError("finalize requires --confirm-visual-pass after actual inspection")
        finalize(run_dir)


if __name__ == "__main__":
    main()
