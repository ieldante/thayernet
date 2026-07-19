#!/usr/bin/env python3
"""MPS-only Thayer-ME micro fit to frozen feasible projection targets."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_thayer_feasibility_projection import fresh_csv, fresh_json, fresh_text, sha256
from scripts.run_thayer_two_expert_micro_overfit import (
    BATCH_SIZE,
    CONDITION_C,
    EPOCHS,
    LEARNING_RATE,
    NORMALIZATION,
    evaluate,
    load_micro_arrays,
    require_mps,
    select_microset,
    training_batches,
)
from src.models_two_expert_decoder import (
    ThayerMixtureExperts,
    expert_parameter_distance,
    parameter_count,
    set_training_phase,
    warm_start_condition_c_encoder,
)


SEED = 2026071250


def direct_cost(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    requested = (predicted[..., :3, :, :] - target[..., :3, :, :]).square().mean(dim=(-3, -2, -1))
    companion = (predicted[..., 3:, :, :] - target[..., 3:, :, :]).square().mean(dim=(-3, -2, -1))
    return requested + companion


def hard_projected_set_loss(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c00 = direct_cost(outputs[:, 0], targets[:, 0])
    c10 = direct_cost(outputs[:, 1], targets[:, 0])
    c01 = direct_cost(outputs[:, 0], targets[:, 1])
    c11 = direct_cost(outputs[:, 1], targets[:, 1])
    identity = c00 + c11
    swapped = c01 + c10
    return torch.minimum(identity, swapped).mean(), identity <= swapped, swapped - identity


def model_outputs(model: ThayerMixtureExperts, arrays: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    output = np.empty((64, 2, 2, 6, 60, 60), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, 64, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, 64)
            blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"][start:stop])).to(device)
            pa = torch.from_numpy(np.ascontiguousarray(arrays["prompt_a"][start:stop])).to(device)
            pb = torch.from_numpy(np.ascontiguousarray(arrays["prompt_b"][start:stop])).to(device)
            joined = model(torch.cat((blend, blend)), torch.cat((pa, pb))).cpu().numpy()
            output[start:stop] = np.stack((joined[: stop - start], joined[stop - start :]), axis=1)
    return output


def target_fit_metrics(outputs: np.ndarray, projected: np.ndarray, exact: np.ndarray, blend: np.ndarray) -> dict[str, float]:
    def assignment_metrics(reference: np.ndarray) -> tuple[float, float, float]:
        tensor_output = torch.from_numpy(outputs)
        tensor_target = torch.from_numpy(reference)
        prompt_losses = []
        identities = []
        margins = []
        for prompt in (0, 1):
            loss, wins, margin = hard_projected_set_loss(tensor_output[:, prompt], tensor_target[:, prompt])
            prompt_losses.append(float(loss))
            identities.extend(wins.numpy().tolist())
            margins.extend(np.abs(margin.numpy()).tolist())
        return float(np.mean(prompt_losses)), float(np.mean(identities)), float(np.mean(margins))

    projection_loss, identity_fraction, assignment_margin = assignment_metrics(projected)
    truth_loss, _, _ = assignment_metrics(exact)
    recomposed = outputs[..., :3, :, :] + outputs[..., 3:, :, :]
    source_sum_mse = float(np.mean((recomposed - blend[:, None, None]) ** 2))
    return {
        "projection_reconstruction_loss": projection_loss,
        "exact_truth_reconstruction_loss": truth_loss,
        "identity_assignment_fraction": identity_fraction,
        "assignment_margin_mean": assignment_margin,
        "source_sum_mse_evaluation_only": source_sum_mse,
        "output_minimum_normalized": float(outputs.min()),
        "negative_output_fraction": float(np.mean(outputs < -1e-7)),
        "finite_output_fraction": float(np.mean(np.isfinite(outputs))),
    }


def success(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["ordinary_own_truth_coverage"] >= 0.90
        and metrics["ordinary_median_expert_diameter"] <= 1.0
        and metrics["ambiguous_own_truth_coverage"] >= 0.90
        and metrics["ambiguous_alternate_truth_coverage"] >= 0.90
        and metrics["ambiguous_both_mode_coverage"] >= 0.90
        and metrics["set_prompt_swap"] >= 0.90
        and metrics["ordinary_forward_consistency"] >= 0.90
        and metrics["ambiguous_forward_consistency"] >= 0.90
    )


def make_grid(path: Path, outputs_physical: np.ndarray, truth_physical: np.ndarray, indices: list[int], title: str) -> None:
    fig, axes = plt.subplots(len(indices), 4, figsize=(10, 2.5 * len(indices)))
    axes = np.atleast_2d(axes)
    for row, index in enumerate(indices):
        panels = (
            truth_physical[index, 0, 0, :3], truth_physical[index, 0, 1, :3],
            outputs_physical[index, 0, 0, :3], outputs_physical[index, 0, 1, :3],
        )
        for column, panel in enumerate(panels):
            image = np.moveaxis(panel, 0, -1)
            low, high = np.percentile(image, (1, 99))
            axes[row, column].imshow(np.clip((image - low) / max(high - low, 1e-12), 0, 1))
            axes[row, column].axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    gate = json.loads((run / "logs/projection_gate_complete_final.json").read_text())
    target_freeze = json.loads((run / "projection_targets/freeze_record_final.json").read_text())
    target_path = run / "projection_targets/projected_target_sets_final.h5"
    if gate["status"] != "PASS" or not gate["micro_training_authorized"] or gate["selected_method"] != "P0_HOMOTOPY_INTERIOR":
        raise RuntimeError("final projection gate did not authorize training")
    if sha256(target_path) != target_freeze["projected_target_file_sha256"]:
        raise RuntimeError("projected target hash mismatch")
    if any((run / "checkpoints").iterdir()) or any((run / "micro_overfit").iterdir()):
        raise RuntimeError("micro-training output collision")

    device = require_mps()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    rows, indices = select_microset()
    arrays = load_micro_arrays(indices, scales)
    with h5py.File(target_path, "r") as handle:
        if not bool(handle.attrs["complete"]) or int(handle.attrs["inference_input_fields_added"]) != 0:
            raise RuntimeError("projected target contract invalid")
        projected = np.asarray(handle["targets_normalized"], dtype=np.float32)
        projected_physical = np.asarray(handle["targets_physical"], dtype=np.float32)
    if projected.shape != (64, 2, 2, 6, 60, 60) or not np.all(np.isfinite(projected)) or np.min(projected_physical) < -1e-7:
        raise RuntimeError("projected target output contract violation")

    exact = arrays["targets"].copy()
    exact[:32, :, 1] = exact[:32, :, 0]
    exact_physical = exact * np.tile(scales, 2)[None, None, None, :, None, None]
    model = ThayerMixtureExperts()
    warm_rows = warm_start_condition_c_encoder(model, CONDITION_C)
    set_training_phase(model, 2)
    if parameter_count(model) != 165612:
        raise RuntimeError("Thayer-ME architecture parameter count changed")
    model = model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    epoch_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    dead_counts = [0, 0]
    stop_reason = "MAX_EPOCHS"
    passed_epoch = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        if sha256(target_path) != target_freeze["projected_target_file_sha256"]:
            raise RuntimeError("projected target hash changed during training")
        model.train()
        batch_losses = []
        identity_fractions = []
        margins = []
        grad1 = []
        grad2 = []
        for batch_indices in training_batches(rows, rng):
            blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"][batch_indices])).to(device)
            pa = torch.from_numpy(np.ascontiguousarray(arrays["prompt_a"][batch_indices])).to(device)
            pb = torch.from_numpy(np.ascontiguousarray(arrays["prompt_b"][batch_indices])).to(device)
            target = torch.from_numpy(np.ascontiguousarray(projected[batch_indices])).to(device)
            joined = model(torch.cat((blend, blend)), torch.cat((pa, pb)))
            oa, ob = joined[: len(blend)], joined[len(blend):]
            loss_a, wins_a, margin_a = hard_projected_set_loss(oa, target[:, 0])
            loss_b, wins_b, margin_b = hard_projected_set_loss(ob, target[:, 1])
            loss = 0.5 * (loss_a + loss_b)
            if not bool(torch.isfinite(loss).detach().cpu()):
                stop_reason = "NAN_INF"
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            g1 = float(torch.linalg.vector_norm(torch.stack([parameter.grad.detach().norm() for parameter in model.expert_1.parameters() if parameter.grad is not None])).cpu())
            g2 = float(torch.linalg.vector_norm(torch.stack([parameter.grad.detach().norm() for parameter in model.expert_2.parameters() if parameter.grad is not None])).cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            identity_fractions.extend([float(wins_a.float().mean().cpu()), float(wins_b.float().mean().cpu())])
            margins.extend([float(margin_a.abs().mean().cpu()), float(margin_b.abs().mean().cpu())])
            grad1.append(g1); grad2.append(g2)
        if stop_reason == "NAN_INF":
            break
        mean_loss = float(np.mean(batch_losses))
        mean_g1, mean_g2 = float(np.mean(grad1)), float(np.mean(grad2))
        dead_counts[0] = dead_counts[0] + 1 if mean_g1 <= 1e-12 else 0
        dead_counts[1] = dead_counts[1] + 1 if mean_g2 <= 1e-12 else 0
        if max(dead_counts) >= 5:
            stop_reason = "EXPERT_DEATH"
            break
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        epoch_rows.append({
            "epoch": epoch, "target_reconstruction_loss": mean_loss,
            "identity_assignment_fraction": float(np.mean(identity_fractions)), "assignment_margin_mean": float(np.mean(margins)),
            "expert_1_gradient_norm": mean_g1, "expert_2_gradient_norm": mean_g2,
            "expert_parameter_distance": float(expert_parameter_distance(model)), "device": "mps", "fallback": False,
            "elapsed_seconds": time.time() - started,
        })
        if epoch == 1 or epoch % 20 == 0:
            scientific, outputs_physical, per_scene = evaluate(model, arrays, rows, scales, device)
            outputs_normalized = outputs_physical / np.tile(scales, 2)[None, None, None, :, None, None]
            fit = target_fit_metrics(outputs_normalized, projected, exact, arrays["blend"])
            combined = {"epoch": epoch, **scientific, **fit}
            evaluation_rows.append(combined)
            print(json.dumps(combined, sort_keys=True), flush=True)
            if fit["finite_output_fraction"] != 1.0:
                stop_reason = "OUTPUT_CONTRACT_NONFINITE"
                break
            if epoch >= 100 and scientific["set_prompt_swap"] < 0.50:
                stop_reason = "PROMPT_COLLAPSE"
                break
            if success(scientific):
                stop_reason = "SUCCESS_GATES_MET"
                passed_epoch = epoch
                break

    if best_state is None:
        raise RuntimeError(f"micro training ended before a finite state: {stop_reason}")
    model.load_state_dict(best_state)
    final_scientific, final_outputs_physical, final_per_scene = evaluate(model, arrays, rows, scales, device)
    final_outputs_normalized = final_outputs_physical / np.tile(scales, 2)[None, None, None, :, None, None]
    final_fit = target_fit_metrics(final_outputs_normalized, projected, exact, arrays["blend"])
    final_metrics = {**final_scientific, **final_fit}
    coverage_nonzero = all(final_metrics[key] > 0 for key in (
        "ordinary_own_truth_coverage", "ambiguous_own_truth_coverage", "ambiguous_alternate_truth_coverage", "ambiguous_both_mode_coverage"
    ))
    if success(final_metrics):
        decision = "SUCCESS"
    elif coverage_nonzero:
        decision = "PARTIAL SUCCESS"
    else:
        decision = "FAILURE"
    output_contract_final = bool(final_fit["finite_output_fraction"] == 1.0 and final_fit["negative_output_fraction"] == 0.0)
    if not output_contract_final and decision == "SUCCESS":
        decision = "PARTIAL SUCCESS"
        stop_reason = "FINAL_NONNEGATIVITY_CONTRACT_FAILED"

    checkpoint_path = run / "checkpoints/thayer_fp_micro_best.pth"
    with checkpoint_path.open("xb") as handle:
        torch.save({
            "model": "Thayer-ME unchanged feasibility-projection micro fit", "state_dict": best_state,
            "epoch": best_epoch, "metrics": final_metrics, "seed": SEED, "parameter_count": parameter_count(model),
            "projected_target_sha256": target_freeze["projected_target_file_sha256"],
            "projection_method": "P0_HOMOTOPY_INTERIOR",
        }, handle)
    output_path = run / "micro_overfit/final_outputs.h5"
    with h5py.File(output_path, "x") as handle:
        handle.create_dataset("decompositions_physical", data=final_outputs_physical, compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.attrs["complete"] = True
        handle.attrs["epoch"] = best_epoch
        handle.attrs["device"] = "mps"
        handle.attrs["fallback"] = False
    fresh_csv(run / "micro_overfit/micro_epochs.csv", epoch_rows)
    fresh_csv(run / "micro_overfit/micro_gate_history.csv", evaluation_rows)
    fresh_csv(run / "micro_overfit/micro_per_scene.csv", final_per_scene)
    gates = [
        {"gate": "ordinary_own_truth_coverage", "observed": final_metrics["ordinary_own_truth_coverage"], "threshold": ">=0.90", "pass": final_metrics["ordinary_own_truth_coverage"] >= 0.90},
        {"gate": "ordinary_median_expert_diameter", "observed": final_metrics["ordinary_median_expert_diameter"], "threshold": "<=1.0", "pass": final_metrics["ordinary_median_expert_diameter"] <= 1.0},
        {"gate": "ambiguous_own_truth_coverage", "observed": final_metrics["ambiguous_own_truth_coverage"], "threshold": ">=0.90", "pass": final_metrics["ambiguous_own_truth_coverage"] >= 0.90},
        {"gate": "ambiguous_alternate_truth_coverage", "observed": final_metrics["ambiguous_alternate_truth_coverage"], "threshold": ">=0.90", "pass": final_metrics["ambiguous_alternate_truth_coverage"] >= 0.90},
        {"gate": "ambiguous_both_mode_coverage", "observed": final_metrics["ambiguous_both_mode_coverage"], "threshold": ">=0.90", "pass": final_metrics["ambiguous_both_mode_coverage"] >= 0.90},
        {"gate": "set_prompt_swap", "observed": final_metrics["set_prompt_swap"], "threshold": ">=0.90", "pass": final_metrics["set_prompt_swap"] >= 0.90},
        {"gate": "ordinary_forward_consistency", "observed": final_metrics["ordinary_forward_consistency"], "threshold": ">=0.90", "pass": final_metrics["ordinary_forward_consistency"] >= 0.90},
        {"gate": "ambiguous_forward_consistency", "observed": final_metrics["ambiguous_forward_consistency"], "threshold": ">=0.90", "pass": final_metrics["ambiguous_forward_consistency"] >= 0.90},
        {"gate": "final_nonnegative_output_contract", "observed": final_fit["negative_output_fraction"], "threshold": "0", "pass": output_contract_final},
    ]
    fresh_csv(run / "micro_overfit/micro_overfit_gates.csv", gates)
    fresh_csv(run / "tables/condition_c_warm_start_inventory.csv", warm_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([int(row["epoch"]) for row in epoch_rows], [float(row["target_reconstruction_loss"]) for row in epoch_rows])
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("direct projected-target loss")
    if evaluation_rows:
        x = [int(row["epoch"]) for row in evaluation_rows]
        for key in ("ordinary_own_truth_coverage", "ambiguous_own_truth_coverage", "ambiguous_alternate_truth_coverage", "ambiguous_both_mode_coverage"):
            axes[1].plot(x, [float(row[key]) for row in evaluation_rows], label=key.replace("_truth", "").replace("_coverage", ""))
        axes[1].axhline(0.90, color="black", linestyle="--"); axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=7); axes[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(run / "figures/micro_training_and_coverage.png", dpi=180); plt.close(fig)
    make_grid(run / "example_grids/ordinary_projected_micro_outputs.png", final_outputs_physical, projected_physical, [0, 8, 16, 24], "Ordinary projected targets and model outputs")
    make_grid(run / "example_grids/ambiguous_projected_micro_outputs.png", final_outputs_physical, projected_physical, [32, 40, 48, 56], "Ambiguous projected targets and model outputs")

    complete = {
        "status": decision, "stop_reason": stop_reason, "best_epoch": best_epoch, "passed_epoch": passed_epoch,
        "runtime_seconds": time.time() - started, "metrics": final_metrics, "output_contract_final": output_contract_final,
        "checkpoint_sha256": sha256(checkpoint_path), "outputs_sha256": sha256(output_path),
        "projected_target_sha256": target_freeze["projected_target_file_sha256"],
        "architecture_parameter_count": parameter_count(model), "mps_only": True, "fallback": False,
        "target_or_constraint_inference_input_count": 0, "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0,
    }
    fresh_json(run / "logs/micro_training_complete.json", complete)
    fresh_text(run / "micro_overfit/micro_overfit_report.md", f"""# Thayer-FP projected-target micro-overfit

Decision: **{decision}**. Best epoch `{best_epoch}`; stop reason `{stop_reason}`.

- Ordinary both-expert coverage / median diameter: `{final_metrics['ordinary_own_truth_coverage']:.6f}` / `{final_metrics['ordinary_median_expert_diameter']:.6f}`.
- Ambiguous own / alternate / both-mode: `{final_metrics['ambiguous_own_truth_coverage']:.6f}` / `{final_metrics['ambiguous_alternate_truth_coverage']:.6f}` / `{final_metrics['ambiguous_both_mode_coverage']:.6f}`.
- Set prompt swap: `{final_metrics['set_prompt_swap']:.6f}`.
- Ordinary / ambiguous forward consistency: `{final_metrics['ordinary_forward_consistency']:.6f}` / `{final_metrics['ambiguous_forward_consistency']:.6f}`.
- Projected-target / exact-truth direct reconstruction loss: `{final_metrics['projection_reconstruction_loss']:.9g}` / `{final_metrics['exact_truth_reconstruction_loss']:.9g}`.
- Final negative output fraction: `{final_metrics['negative_output_fraction']:.9g}`.
- Architecture: unchanged 165,612 parameters; MPS only; no fallback.
- Target or constraint fields added to inference: `0`.
- Atlas / development / lockbox accesses: `0 / 0 / 0`.
""")
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
