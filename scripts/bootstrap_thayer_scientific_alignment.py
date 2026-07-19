#!/usr/bin/env python3
"""Create the append-only Thayer-SA run and reproduce the frozen diagnosis."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LG = REPO / "outputs/runs/thayer_loss_geometry_20260712_205733"
ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
PROMPT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
ATLAS_CONTRACT = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"

SCENES = MH / "manifests/probabilistic_unet_training_scenes.h5"
TARGETS = MH / "target_sets/thayer_mh_training_target_sets.h5"
DEFINITIONS = MH / "manifests/probabilistic_unet_scene_definitions.csv"
MICRO_MANIFEST = MICRO / "tables/microset_manifest.csv"
TRAINED_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
ME_CHECKPOINT = MICRO / "checkpoints/thayer_me_micro_final.pth"
NORMALIZATION = PROMPT / "manifests/normalization.json"
FORWARD_THRESHOLDS = PU / "manifests/forward_consistency_thresholds.json"
NOISE_CONTRACT = ATLAS_CONTRACT / "manifests/fixed_noise_contract.json"
CONDITION_C = PROMPT / "checkpoints/c_randomized_coordinate_prompt_best.pth"

SUBDIRECTORIES = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "checkpoints", "objective_preflight", "micro_overfit", "gradients", "example_grids",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_inventory() -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted((REPO / "outputs/runs").rglob("*.pth"))
    ]


def source_inventory() -> list[dict[str, object]]:
    return [
        {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for root in ("src", "scripts", "tests")
        for path in sorted((REPO / root).rglob("*.py"))
    ]


def verify_frozen_inputs() -> dict[str, dict[str, str]]:
    if command(["git", "diff", "--cached", "--name-only"]):
        raise RuntimeError("staged index must remain empty")
    provenance = json.loads((LG / "logs/input_provenance.json").read_text())
    paths = {
        name: REPO / record["path"]
        for name, record in provenance["relevant_artifacts"].items()
        if name not in {"ambiguity_set_contract", "expert_specialization_contract", "forward_consistency_contract", "prompt_swap_contract", "source_layer_contract", "truth_coverage_contract"}
    }
    expected = {name: provenance["relevant_artifacts"][name]["sha256"] for name in paths}
    expected.update({
        "source_layer_contract": "dc3a78b65b2eda17b71887c7616189a24fbf1f367c8fb61014d6e291a2e02128",
        "truth_coverage_contract": "4d2d53ea7ef77c09b263ee90dec50b7138b0dbe07f2b16150113c0041e589d97",
    })
    paths.update({
        "source_layer_contract": REPO / "docs/multi_hypothesis_source_contract.md",
        "truth_coverage_contract": REPO / "docs/latent_truth_coverage.md",
    })
    result = {}
    for name, path in paths.items():
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected[name]:
            raise RuntimeError(f"frozen input mismatch: {name}: {path.relative_to(REPO)}")
        result[name] = {"path": str(path.relative_to(REPO)), "sha256": observed}
    if json.loads((LG / "diagnostics/final_correctness_audit.json").read_text())["failure_count"]:
        raise RuntimeError("authoritative Thayer-LG correctness audit failed")
    return result


def reproduce_loss_geometry() -> list[dict[str, object]]:
    ranking = read_csv(LG / "tables/objective_ranking_summary.csv")
    all_row = next(row for row in ranking if row["kind"] == "all")
    ambiguous_row = next(row for row in ranking if row["kind"] == "near_collision")
    compromises_all = round(float(all_row["fraction_compromise_beats_truth"]) * int(all_row["scene_count"]))
    compromises_ambiguous = round(float(ambiguous_row["fraction_compromise_beats_truth"]) * int(ambiguous_row["scene_count"]))

    trajectories = read_csv(LG / "tables/output_space_optimization_trajectories.csv")
    truth_path = [row for row in trajectories if row["protocol"] == "D0_FULL" and row["initialization"] == "exact_truth"]
    start = next(row for row in truth_path if row["step"] == "0")
    end = next(row for row in truth_path if row["step"] == "40")

    decomposition = read_csv(LG / "tables/canonical_loss_decomposition.csv")
    fractions = {}
    for kind, configuration in (("ordinary", "O2_TRAINED_EXPERT_OUTPUTS"), ("near_collision", "A3_TRAINED_EXPERT_OUTPUTS")):
        by_scene: dict[str, dict[str, float]] = defaultdict(dict)
        for row in decomposition:
            if row["kind"] == kind and row["configuration"] == configuration:
                by_scene[row["scene_id"]][row["term"]] = float(row["weighted_loss"])
        fractions[kind] = statistics.mean(values["forward"] / sum(values.values()) for values in by_scene.values())

    gradients = read_csv(LG / "tables/gradient_cosines.csv")
    conflicts = {}
    for kind in ("ordinary", "near_collision"):
        values = [
            float(row["cosine"])
            for row in gradients
            if row["kind"] == kind and row["left_gradient"] == "set_matching" and row["right_gradient"] == "forward"
        ]
        conflicts[kind] = sum(value < 0 for value in values) / len(values)

    gradient_norms = read_csv(LG / "tables/gradient_norms.csv")
    truth_fraction = {}
    for kind, configuration in (("ordinary", "O1_EXACT_TRUTH_DUPLICATED"), ("near_collision", "A1_EXACT_APPROVED_SET")):
        values = [float(row["fraction_sum_term_weighted_l2"]) for row in gradient_norms if row["kind"] == kind and row["configuration"] == configuration and row["term"] == "forward"]
        truth_fraction[kind] = statistics.mean(values)

    representability = read_csv(LG / "tables/truth_representability_audit.csv")
    exact_pass = sum(row["status"] == "PASS" for row in representability)
    exact_total = len(representability)
    checks = [
        ("compromises_beat_truth_all", 54, compromises_all, 0),
        ("compromises_beat_truth_ambiguous", 32, compromises_ambiguous, 0),
        ("truth_start_full_objective_start", 0.02937684, float(start["full_frozen_objective"]), 1e-7),
        ("truth_start_full_objective_end", 0.02899991, float(end["full_frozen_objective"]), 1e-7),
        ("truth_start_ordinary_coverage_end", 0.03125, float(end["ordinary_coverage"]), 1e-9),
        ("truth_start_ambiguous_both_mode_end", 0.0, float(end["ambiguous_both_mode_coverage"]), 1e-9),
        ("forward_fraction_trained_ordinary", 0.76495894, fractions["ordinary"], 1e-7),
        ("forward_fraction_trained_ambiguous", 0.86710765, fractions["near_collision"], 1e-7),
        ("forward_truth_gradient_fraction_ordinary", 1.0, truth_fraction["ordinary"], 1e-9),
        ("forward_truth_gradient_fraction_ambiguous", 0.981, truth_fraction["near_collision"], 0.002),
        ("gradient_conflict_ordinary", 0.6328125, conflicts["ordinary"], 1e-9),
        ("gradient_conflict_ambiguous", 0.515625, conflicts["near_collision"], 1e-9),
        ("exact_truth_representability_rows", exact_total, exact_pass, 0),
    ]
    rows = []
    for name, expected, observed, tolerance in checks:
        passed = abs(float(observed) - float(expected)) <= tolerance
        rows.append({"check": name, "expected": expected, "observed": observed, "absolute_tolerance": tolerance, "status": "PASS" if passed else "FAIL"})
    if not all(row["status"] == "PASS" for row in rows):
        raise RuntimeError("loss-geometry reproduction mismatch")
    return rows


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    frozen = verify_frozen_inputs()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_scientific_alignment_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    started = datetime.now(timezone.utc).isoformat()
    checkpoints = checkpoint_inventory()
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", checkpoints)
    write_csv_fresh(run_dir / "tables/source_code_hashes_before.csv", source_inventory())
    reproduction = reproduce_loss_geometry()
    write_csv_fresh(run_dir / "tables/loss_geometry_reproduction.csv", reproduction)

    import btk
    import galsim
    import h5py
    import matplotlib
    import numpy
    import scipy
    import torch

    disk = shutil.disk_usage(REPO)
    mps = {
        "built": bool(torch.backends.mps.is_built()),
        "available": bool(torch.backends.mps.is_available()),
        "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1",
    }
    if mps["fallback_enabled"]:
        raise RuntimeError("MPS fallback is prohibited")
    architecture_files = [REPO / "src/models_two_expert_decoder.py", REPO / "src/models_probabilistic_unet.py"]
    implementation_files = [REPO / "src/scientific_alignment.py", REPO / "scripts/bootstrap_thayer_scientific_alignment.py"]
    provenance = {
        "campaign": "Thayer Scientific Alignment micro-overfit correction",
        "working_experiment_name": "Thayer-SA",
        "campaign_started_utc": started,
        "run_dir": str(run_dir.relative_to(REPO)),
        "branch": command(["git", "branch", "--show-current"]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_v2": command(["git", "status", "--porcelain=v2", "--untracked-files=all"]).splitlines(),
        "staged_index": command(["git", "diff", "--cached", "--name-status"]).splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "torch", "btk", "galsim", "h5py", "scipy", "pandas", "matplotlib")},
        "mps": mps,
        "free_disk_bytes": disk.free,
        "frozen_inputs": frozen,
        "architecture": {"files": [str(path.relative_to(REPO)) for path in architecture_files], "combined_sha256": combined_hash(architecture_files), "parameter_count": 165612},
        "microset_manifest": {"path": str(MICRO_MANIFEST.relative_to(REPO)), "sha256": sha256_file(MICRO_MANIFEST)},
        "target_set_hashes": {
            Path(record["path"]).stem: record["sha256"]
            for record in json.loads((ME / "target_sets/reused_target_set_references.json").read_text())["files"]
        },
        "initialization_seeds": {"expert_1": 2026071201, "expert_2": 2026071202, "training": 2026071250},
        "source_layer_contract_sha256": sha256_file(REPO / "docs/multi_hypothesis_source_contract.md"),
        "truth_coverage_implementation_sha256": combined_hash([REPO / "src/competing_hypotheses.py", REPO / "scripts/run_thayer_two_expert_micro_overfit.py"]),
        "scientific_distance_implementation_sha256": sha256_file(REPO / "src/competing_hypotheses.py"),
        "scientific_alignment_implementation_sha256": combined_hash(implementation_files),
        "historical_checkpoint_count": len(checkpoints),
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)
    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", f"""# Thayer-SA environment snapshot

- Campaign start UTC: `{started}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python / Torch / BTK: `{sys.version.splitlines()[0]}` / `{torch.__version__}` / `{btk.__version__}`
- MPS built / available: `{mps['built']}` / `{mps['available']}`; fallback disabled.
- Free disk at start: `{disk.free}` bytes.
- Staged index: empty.
- Historical checkpoints: {len(checkpoints)} hashed before the campaign.
- Atlas / development / lockbox access: 0 / 0 / 0.
""")
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", """# Thayer-SA campaign contract

Thayer-SA is a micro-overfit-only correction of the frozen Thayer-ME objective. The architecture, 64-scene training microset, target sets, initialization seeds, source-layer semantics, scientific-distance implementation, thresholds, and evaluation metrics remain unchanged. Training may optimize only source-set reconstruction, the preregistered differentiable scientific surrogate, and ordinary expert concentration. Forward, source-sum, prompt-swap, and pair consistency are evaluation-only. Output-space and assignment preflights must pass before MPS-only neural fitting. Atlas, validation, calibration, development, and lockbox data remain sealed. Historical runs and checkpoints are immutable; every campaign artifact uses this collision-refusing run path.
""")
    write_json_fresh(run_dir / "logs/loss_geometry_reproduction.json", {
        "status": "PASS",
        "check_count": len(reproduction),
        "source_run": str(LG.relative_to(REPO)),
        "reproduced_before_preregistration": True,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    print(run_dir.relative_to(REPO))


if __name__ == "__main__":
    main()
