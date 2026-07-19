#!/usr/bin/env python3
"""Create the append-only Thayer-ME master run and verify immutable inputs."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
FLOW = REPO / "outputs/runs/thayer_flow_prior_20260712_182516"
MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
PROMPT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"

CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PROMPT / "manifests/source_split_manifest.csv"
NORMALIZATION = PROMPT / "manifests/normalization.json"
CONDITION_C = PROMPT / "checkpoints/c_randomized_coordinate_prompt_best.pth"
MH_BEST = MH / "checkpoints/thayer_mh_best.pth"
MH_FINAL = MH / "checkpoints/thayer_mh_final.pth"

EXPECTED = {
    CATALOG: "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46",
    SOURCE_SPLIT: "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27",
    NORMALIZATION: "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    CONDITION_C: "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
    MH_BEST: "4765c829a52193c056f2d0e1dea565e732f3fda977e5105497918beb58e25b32",
    MH_FINAL: "3511915059c972742f62f20fd748d1feb3c7e46fffc6e96c21015d33d606df67",
    ATLAS / "tables/atlas_pair_manifest.csv": "55c42584dd8521b7722d5d9b49a6e20cbc399977e5811c08df1b454ccd78d5fa",
    ATLAS / "tables/atlas_initial_visual_audit.csv": "1615d9f2b4941e032113db887bc9881727983c5495623105400ffc68929d21da",
    ATLAS / "manifests/fixed_noise_contract.json": "3ce4435330da83eace363ceee3856612e100f43b63d2493aed7441992494ec7b",
}

SUBDIRECTORIES = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "manifests", "checkpoints", "target_sets", "expert_outputs", "atlas_evaluation",
    "example_grids", "paper_figures",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_fixed_inputs() -> None:
    staged = command(["git", "diff", "--cached", "--name-only"]).splitlines()
    if staged:
        raise RuntimeError(f"staged files are prohibited: {staged}")
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen input altered: {path.relative_to(REPO)}")
    mh_audit = json.loads((MH / "diagnostics/final_correctness_audit.json").read_text())
    mh_stop = json.loads((MH / "logs/pre_atlas_evaluation_complete.json").read_text())
    if mh_audit["failure_count"] != 0 or mh_audit["lockbox_scene_access_count"] != 0:
        raise RuntimeError("authoritative Thayer-MH correctness or lockbox status unresolved")
    if mh_stop["status"] != "FAIL_SET_COVERAGE" or mh_stop["atlas_evaluation_count"] != 0:
        raise RuntimeError("unexpected Thayer-MH scientific boundary")
    source_summary = json.loads((ATLAS / "manifests/source_partition_summary.json").read_text())
    if source_summary["source_split_sha256"] != EXPECTED[SOURCE_SPLIT] or source_summary["sealed_lockbox_rows_committed"] != 0:
        raise RuntimeError("source-split or lockbox exclusion unresolved")
    atlas_hashes = read_csv(PU / "tables/frozen_atlas_artifact_hashes_after.csv")
    if len(atlas_hashes) != 25:
        raise RuntimeError("frozen Atlas artifact inventory incomplete")
    for row in atlas_hashes:
        if row["status"] != "PASS" or sha256_file(REPO / row["path"]) != row["observed_sha256"]:
            raise RuntimeError(f"frozen Atlas artifact altered: {row['path']}")


def checkpoint_inventory() -> list[dict[str, object]]:
    rows = []
    for path in sorted((REPO / "outputs/runs").rglob("*.pth")):
        rows.append({"path": str(path.relative_to(REPO)), "observed_sha256": sha256_file(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "status": "PASS"})
    if not rows or not any(row["path"] == str(MH_BEST.relative_to(REPO)) for row in rows):
        raise RuntimeError("historical checkpoint inventory incomplete")
    return rows


def source_inventory() -> list[dict[str, object]]:
    rows = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    verify_fixed_inputs()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_two_expert_decoder_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir(exist_ok=False)

    checkpoints = checkpoint_inventory()
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", checkpoints)
    write_csv_fresh(run_dir / "tables/source_code_hashes_before.csv", source_inventory())

    import btk
    import galsim
    import torch

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")

    target_paths = [
        MH / f"target_sets/thayer_mh_{partition}_target_sets.h5"
        for partition in ("training", "validation", "calibration")
    ]
    scene_paths = [
        MH / f"manifests/probabilistic_unet_{partition}_scenes.h5"
        for partition in ("training", "validation", "calibration")
    ]
    relevant = {
        "source_layer_contract": REPO / "docs/multi_hypothesis_source_contract.md",
        "prompt_contract": REPO / "docs/thayer_select_prompt_semantics.md",
        "forward_consistency_contract": REPO / "docs/forward_consistency_contract.md",
        "truth_coverage_contract": REPO / "docs/latent_truth_coverage.md",
        "forward_model": REPO / "src/btk_scene.py",
        "noise_contract": ATLAS / "manifests/fixed_noise_contract.json",
        "prompt_implementation": REPO / "scripts/thayer_select_prompt_ablation_common.py",
        "canonical_tensor_hash_contract": PU / "diagnostics/canonical_hash_contract.md",
        "normalization": NORMALIZATION,
        "target_inventory": MH / "tables/target_set_inventory.csv",
        "target_pair_validation": MH / "tables/target_set_pair_validation.csv",
        "scene_manifest": MH / "manifests/probabilistic_unet_scene_definitions.csv",
        "pair_manifest": MH / "tables/non_atlas_near_collision_pair_manifest.csv",
        "atlas_manifest": ATLAS / "tables/atlas_pair_manifest.csv",
        "atlas_pair_validation": ATLAS / "tables/atlas_pair_validation.csv",
        "atlas_metric": REPO / "src/competing_hypotheses.py",
        "atlas_witness_inventory": ATLAS / "tables/ambiguity_witness_inventory.csv",
    }
    artifact_hashes = {name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for name, path in relevant.items()}
    artifact_hashes["reused_target_sets"] = [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in target_paths]
    artifact_hashes["reused_scene_tensors"] = [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in scene_paths]
    disk = shutil.disk_usage(REPO)
    started = datetime.now(timezone.utc).isoformat()
    provenance = {
        "campaign": "Thayer-ME two-expert ambiguity decoder campaign",
        "working_model_name": "Thayer-ME",
        "campaign_started_utc": started,
        "run_dir": str(run_dir.relative_to(REPO)),
        "branch": command(["git", "branch", "--show-current"]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_v2": command(["git", "status", "--porcelain=v2", "--untracked-files=all"]).splitlines(),
        "staged_index": command(["git", "diff", "--cached", "--name-status"]).splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "torch": torch.__version__, "btk": btk.__version__, "galsim": galsim.__version__, "h5py": package_version("h5py"), "scipy": package_version("scipy")},
        "mps": {"built": torch.backends.mps.is_built(), "available": torch.backends.mps.is_available(), "probe": "PASS", "fallback_enabled": False},
        "free_disk_bytes": disk.free,
        "source_catalog": {"path": str(CATALOG.relative_to(REPO)), "sha256": sha256_file(CATALOG)},
        "source_split": {"path": str(SOURCE_SPLIT.relative_to(REPO)), "sha256": sha256_file(SOURCE_SPLIT)},
        "condition_c": {"path": str(CONDITION_C.relative_to(REPO)), "sha256": sha256_file(CONDITION_C)},
        "thayer_mh_checkpoints": {"best": sha256_file(MH_BEST), "final": sha256_file(MH_FINAL)},
        "relevant_artifacts": artifact_hashes,
        "historical_checkpoint_count": len(checkpoints),
        "atlas_evaluation_count_this_campaign": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)
    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", f"""# Thayer-ME environment snapshot

- Started UTC: `{started}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python: `{sys.version.splitlines()[0]}`
- Torch / BTK / GalSim: `{torch.__version__}` / `{btk.__version__}` / `{galsim.__version__}`
- MPS: built, available, execution-probed; fallback disabled.
- Free disk at start: `{disk.free}` bytes.
- Staged index: empty.
- Historical checkpoints: {len(checkpoints)} inventoried and hashed.
- Condition-C and Thayer-MH best/final checkpoints: byte-identical to frozen commitments.
- Frozen Atlas pair artifacts: 25 byte-identical files.
- New Atlas / development / lockbox access: 0 / 0 / 0.
""")
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", """# Thayer-ME campaign contract

This append-only campaign tests whether one Condition-C-compatible shared prompt encoder feeding two independently parameterized compact expert decoders can represent both approved decompositions of a non-Atlas ambiguity set while converging on ordinary scenes.

The exact Thayer-MH scene manifests and target sets are immutable reused inputs. Atlas sources, historical development, and the final lockbox are excluded from fitting, validation, calibration, debugging, and model selection. No source, pair, morphology, simulator, or expert identity enters the observed input. Target truth is training-only. Neural execution is MPS-only; CPU is limited to target matching, manifests, metrics, hashes, bootstrap, and reports. Baseline reproduction and target-set reaudit precede preregistration; preregistration precedes model implementation or fitting. A failed hard gate stops all later stages. Atlas inference is allowed once only after every non-Atlas gate passes and a full protocol is frozen and hashed. Historical artifacts are immutable.
""")
    write_json_fresh(run_dir / "logs/part_a_complete.json", {"status": "PASS", "run_dir": str(run_dir.relative_to(REPO)), "historical_checkpoint_count": len(checkpoints), "staged_index_empty": True, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    print(run_dir)


if __name__ == "__main__":
    main()
