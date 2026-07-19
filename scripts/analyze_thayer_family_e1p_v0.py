#!/usr/bin/env python3
"""Post-fit source-role analysis for a completed Family-E1P micro run."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from run_thayer_family_e1p_v0 import (
    BAND_SCALES,
    MICRO_SPECS,
    REPO,
    FamilyE1UNet,
    load_paired_micro,
    require_mps,
    validate_run,
)


def fresh_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def template_coefficients(
    prediction: torch.Tensor,
    own: torch.Tensor,
    other: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_prediction = prediction.flatten(2).reshape(-1, prediction.shape[-2] * prediction.shape[-1])
    flat_own = own.flatten(2).reshape_as(flat_prediction)
    flat_other = other.flatten(2).reshape_as(flat_prediction)
    epsilon = torch.finfo(prediction.dtype).eps
    own_own = torch.sum(flat_own * flat_own, dim=1) + epsilon
    own_other = torch.sum(flat_own * flat_other, dim=1)
    other_other = torch.sum(flat_other * flat_other, dim=1) + epsilon
    own_prediction = torch.sum(flat_own * flat_prediction, dim=1)
    other_prediction = torch.sum(flat_other * flat_prediction, dim=1)
    determinant = torch.clamp(own_own * other_other - own_other.square(), min=epsilon)
    own_coefficient = torch.clamp(
        (other_other * own_prediction - own_other * other_prediction) / determinant,
        min=0.0,
    )
    other_coefficient = torch.clamp(
        (own_own * other_prediction - own_other * own_prediction) / determinant,
        min=0.0,
    )
    return own_coefficient, other_coefficient


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    validate_run(run)
    device = require_mps()
    scales = torch.tensor(BAND_SCALES, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    view_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    for condition, spec in MICRO_SPECS.items():
        indices = list(spec["indices"])
        paired, _ = load_paired_micro(indices)
        model_input = torch.from_numpy(paired["model_input"]).to(device)
        observed = torch.from_numpy(paired["observed"]).to(device)
        requested = torch.from_numpy(paired["requested"]).to(device) / scales
        companion = torch.from_numpy(paired["companion"]).to(device) / scales
        payload = torch.load(
            run / f"micro_overfit/{condition}_final_state.pth",
            map_location="cpu",
            weights_only=False,
        )
        model = FamilyE1UNet().to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            output = model(model_input, observed)
        predicted_requested = output.requested / scales
        predicted_companion = output.companion / scales
        count = len(indices)
        own_mse = torch.mean((predicted_requested - requested).square(), dim=(1, 2, 3))
        wrong_mse = torch.mean((predicted_requested - companion).square(), dim=(1, 2, 3))
        own_l1 = torch.mean(torch.abs(predicted_requested - requested), dim=(1, 2, 3))
        wrong_l1 = torch.mean(torch.abs(predicted_requested - companion), dim=(1, 2, 3))
        own_coefficient, other_coefficient = template_coefficients(
            predicted_requested,
            requested,
            companion,
        )
        own_coefficient = own_coefficient.reshape(2 * count, 3).mean(dim=1)
        other_coefficient = other_coefficient.reshape(2 * count, 3).mean(dim=1)
        for local, family_index in enumerate(indices):
            for view_index, view in ((local, "A"), (count + local, "B")):
                view_rows.append(
                    {
                        "condition": condition,
                        "family_e1_index": family_index,
                        "prompt_view": view,
                        "identity_correct": bool(own_mse[view_index] < wrong_mse[view_index]),
                        "own_mse": scalar(own_mse[view_index]),
                        "companion_mse": scalar(wrong_mse[view_index]),
                        "identity_margin_mse": scalar(wrong_mse[view_index] - own_mse[view_index]),
                        "own_l1": scalar(own_l1[view_index]),
                        "companion_l1": scalar(wrong_l1[view_index]),
                        "requested_template_coefficient": scalar(own_coefficient[view_index]),
                        "companion_template_coefficient": scalar(other_coefficient[view_index]),
                        "companion_template_fraction": scalar(
                            other_coefficient[view_index]
                            / (own_coefficient[view_index] + other_coefficient[view_index] + 1.0e-12)
                        ),
                    }
                )

        req_a = predicted_requested[:count]
        req_b = predicted_requested[count:]
        comp_a = predicted_companion[:count]
        comp_b = predicted_companion[count:]
        truth_a = requested[:count]
        truth_b = requested[count:]
        predicted_common = 0.5 * (req_a + req_b)
        truth_common = 0.5 * (truth_a + truth_b)
        predicted_contrast = 0.5 * (req_a - req_b)
        truth_contrast = 0.5 * (truth_a - truth_b)
        contrast_flat = predicted_contrast.flatten(1)
        truth_contrast_flat = truth_contrast.flatten(1)
        contrast_gain = torch.sum(contrast_flat * truth_contrast_flat, dim=1) / torch.clamp(
            torch.sum(truth_contrast_flat.square(), dim=1), min=1.0e-30
        )
        contrast_cosine = F.cosine_similarity(contrast_flat, truth_contrast_flat, dim=1, eps=1.0e-12)
        for local, family_index in enumerate(indices):
            truth_contrast_rms = torch.sqrt(torch.mean(truth_contrast[local].square()))
            decomposition_rows.append(
                {
                    "condition": condition,
                    "family_e1_index": family_index,
                    "requested_same_head_pair_l1": scalar(torch.mean(torch.abs(req_a[local] - req_b[local]))),
                    "requested_a_to_companion_b_l1": scalar(torch.mean(torch.abs(req_a[local] - comp_b[local]))),
                    "requested_b_to_companion_a_l1": scalar(torch.mean(torch.abs(req_b[local] - comp_a[local]))),
                    "common_mode_l1_error": scalar(torch.mean(torch.abs(predicted_common[local] - truth_common[local]))),
                    "contrast_mode_l1_error": scalar(torch.mean(torch.abs(predicted_contrast[local] - truth_contrast[local]))),
                    "contrast_gain": scalar(contrast_gain[local]),
                    "contrast_cosine": scalar(contrast_cosine[local]),
                    "predicted_contrast_rms": scalar(torch.sqrt(torch.mean(predicted_contrast[local].square()))),
                    "truth_contrast_rms": scalar(truth_contrast_rms),
                    "common_to_truth_contrast_rms_ratio": scalar(
                        torch.sqrt(torch.mean(predicted_common[local].square()))
                        / torch.clamp(truth_contrast_rms, min=1.0e-30)
                    ),
                }
            )
    fresh_csv(run / "tables/per_view_identity_diagnostics.csv", view_rows)
    fresh_csv(run / "tables/common_contrast_diagnostics.csv", decomposition_rows)
    print(f"wrote {len(view_rows)} view rows and {len(decomposition_rows)} decomposition rows")


if __name__ == "__main__":
    main()
