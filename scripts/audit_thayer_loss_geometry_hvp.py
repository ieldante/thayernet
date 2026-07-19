#!/usr/bin/env python3
"""Add a float64 Hessian-vector curvature check to the frozen Thayer-LG audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.audit_thayer_loss_geometry import load_inputs, verify_freeze, write_csv_fresh, write_json_fresh, write_text_fresh
from src.loss_geometry import canonical_configurations, scene_loss_terms


def normalize(direction: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(direction.flatten(1), dim=1).reshape((-1,) + (1,) * (direction.ndim - 1))
    return direction / torch.clamp(norms, min=1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    verify_freeze(run_dir)
    rows, _, arrays, _, trained = load_inputs()
    configs = canonical_configurations(arrays["targets"], arrays["counts"], trained, rows)
    targets_all = torch.from_numpy(np.ascontiguousarray(arrays["targets"])).double()
    counts_all = torch.from_numpy(np.ascontiguousarray(arrays["counts"]))
    blend_all = torch.from_numpy(np.ascontiguousarray(arrays["blend"])).double()
    records = []
    for config in ("O1_EXACT_TRUTH_DUPLICATED", "O2_TRAINED_EXPERT_OUTPUTS", "A1_EXACT_APPROVED_SET", "A3_TRAINED_EXPERT_OUTPUTS"):
        indices, output_np = configs[config]
        base = torch.from_numpy(np.ascontiguousarray(output_np)).double()
        targets, counts, blend = targets_all[indices], counts_all[indices], blend_all[indices]
        truth = targets.clone()
        ordinary = counts[:, 0] == 1
        truth[ordinary, :, 1] = truth[ordinary, :, 0]
        light = torch.zeros_like(base); light[..., :3, :, :] = truth[..., :3, :, :]; light[..., 3:, :, :] = -truth[..., :3, :, :]
        common = torch.ones_like(base)
        delta = truth[:, :, 0] - truth[:, :, 1]
        delta[ordinary] = torch.roll(truth[ordinary, :, 0], shifts=1, dims=-1) - truth[ordinary, :, 0]
        antisymmetric = torch.zeros_like(base); antisymmetric[:, :, 0] = delta; antisymmetric[:, :, 1] = -delta
        flux = base.clone()
        centroid = torch.roll(base, shifts=1, dims=-1) - base
        morphology = torch.zeros_like(base); morphology[..., :3, :, :] = torch.roll(base[..., :3, :, :], shifts=1, dims=-1) - base[..., :3, :, :]
        directions = {"source_light_exchange": light, "both_expert_common_mode": common, "expert_antisymmetric_separation": antisymmetric, "flux_scaling": flux, "centroid_shift": centroid, "morphology_perturbation": morphology}
        for name, raw_direction in directions.items():
            direction = normalize(raw_direction)
            variable = base.clone().requires_grad_(True)
            total = scene_loss_terms(variable, targets, counts, blend, rows, indices)["total"]
            gradient = torch.autograd.grad(total.sum(), variable, create_graph=True)[0]
            directional = (gradient * direction).sum()
            hvp = torch.autograd.grad(directional, variable)[0]
            first = (gradient.detach() * direction).flatten(1).sum(dim=1)
            curvature = (hvp.detach() * direction).flatten(1).sum(dim=1)
            for local, global_index in enumerate(indices):
                value = float(curvature[local])
                records.append({
                    "micro_index": int(global_index), "scene_id": rows[int(global_index)]["scene_id"], "kind": rows[int(global_index)]["kind"],
                    "configuration": config, "direction": name, "method": "float64_autograd_hessian_vector_product",
                    "directional_first_derivative": float(first[local]), "directional_curvature": value,
                    "flat_flag": abs(value) <= 1e-6, "weak_curvature_flag": abs(value) <= 1e-4, "negative_curvature": value < 0,
                })
    write_csv_fresh(run_dir / "tables/local_curvature_hvp.csv", records)
    write_text_fresh(run_dir / "diagnostics/local_curvature_hvp_report.md", """# Local curvature HVP validation

The preregistered float32 central differences were supplemented, without changing any frozen objective setting, by float64 automatic-differentiation Hessian-vector products. This avoids cancellation at the 1e-3 finite-difference step. Hard-assignment ties remain nonsmooth and are interpreted with the separate perturbation audit.
""")
    write_json_fresh(run_dir / "logs/local_curvature_hvp_complete.json", {"status": "PASS", "row_count": len(records), "model_parameter_gradient_count": 0, "model_optimizer_step_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    print(json.dumps({"status": "PASS", "row_count": len(records)}))


if __name__ == "__main__":
    main()
