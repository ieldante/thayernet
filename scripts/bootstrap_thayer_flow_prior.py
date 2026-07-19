#!/usr/bin/env python3
"""Create the append-only Thayer-PF run and enforce immutable-input gates."""

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
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
PROMPT_RUN = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"

CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PROMPT_RUN / "manifests/source_split_manifest.csv"
CONDITION_C = PROMPT_RUN / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PROMPT_RUN / "manifests/normalization.json"
PU_BEST = PU / "checkpoints/thayer_pu_best.pth"
PU_FINAL = PU / "checkpoints/thayer_pu_final.pth"

EXPECTED = {
    CATALOG: "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46",
    SOURCE_SPLIT: "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27",
    CONDITION_C: "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
    NORMALIZATION: "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    PU_BEST: "c1d17a3f67962cce2fec03d6b15da5f2e330ee97b31c270a7ff019a1373a557e",
    PU_FINAL: "351202703da907d429de41536d9172bbbf25259773029373ee831ba55f0b8e1a",
    ATLAS / "tables/atlas_pair_manifest.csv": "55c42584dd8521b7722d5d9b49a6e20cbc399977e5811c08df1b454ccd78d5fa",
    ATLAS / "tables/atlas_initial_visual_audit.csv": "1615d9f2b4941e032113db887bc9881727983c5495623105400ffc68929d21da",
    ATLAS / "manifests/fixed_noise_contract.json": "3ce4435330da83eace363ceee3856612e100f43b63d2493aed7441992494ec7b",
    ATLAS / "tables/fixed_psf_configuration.csv": "396d6f8cea74b6f906da1965a968330fe7c9ac45dabb7c9b24a81252ae69b15d",
}

SUBDIRECTORIES = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "manifests", "latent_samples", "flow_models", "checkpoints", "prior_samples",
    "posterior_samples", "candidate_sets", "atlas_evaluation", "example_grids",
    "paper_figures",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, cwd=REPO, check=True, text=True, capture_output=True)
    return result.stdout.strip()


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


def create_run() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_flow_prior_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir(exist_ok=False)
    return run_dir


def verify_fixed_inputs() -> None:
    staged = command(["git", "diff", "--cached", "--name-only"]).splitlines()
    if staged:
        raise RuntimeError(f"staged files are prohibited: {staged}")
    for path, expected in EXPECTED.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"frozen input altered: {path.relative_to(REPO)}")

    pu_audit = json.loads((PU / "diagnostics/final_correctness_audit.json").read_text())
    if pu_audit["failure_count"] != 0 or pu_audit["lockbox_scene_access_count"] != 0:
        raise RuntimeError("authoritative Thayer-PU correctness or lockbox status unresolved")
    if pu_audit["scientific_decision"] != "PARTIAL_SUCCESS_TRUTH_COVERAGE_FAILED":
        raise RuntimeError("unexpected authoritative Thayer-PU decision")

    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    if freeze["status"] != "FROZEN_INITIAL_ATLAS_PASS" or freeze["pair_count"] != 25:
        raise RuntimeError("Atlas freeze unresolved")
    if freeze["numerical_manifest_sha256"] != EXPECTED[ATLAS / "tables/atlas_pair_manifest.csv"]:
        raise RuntimeError("Atlas manifest commitment mismatch")
    if freeze["visual_audit_sha256"] != EXPECTED[ATLAS / "tables/atlas_initial_visual_audit.csv"]:
        raise RuntimeError("Atlas visual-audit commitment mismatch")
    split = json.loads((ATLAS / "manifests/source_partition_summary.json").read_text())
    if split["source_split_sha256"] != EXPECTED[SOURCE_SPLIT]:
        raise RuntimeError("source partition commitment mismatch")
    if split["sealed_lockbox_rows_committed"] != 0 or split["development_rows_committed"] != 0:
        raise RuntimeError("lockbox/development exclusion unresolved")

    atlas_hashes = read_csv(PU / "tables/frozen_atlas_artifact_hashes_after.csv")
    if len(atlas_hashes) != 25:
        raise RuntimeError("frozen Atlas artifact inventory is incomplete")
    for row in atlas_hashes:
        if row["status"] != "PASS" or sha256_file(REPO / row["path"]) != row["observed_sha256"]:
            raise RuntimeError(f"frozen Atlas artifact altered: {row['path']}")


def checkpoint_inventory() -> list[dict[str, object]]:
    reference = read_csv(PU / "tables/checkpoint_inventory_after.csv")
    expected_by_path = {row["path"]: row["observed_sha256"] for row in reference}
    expected_by_path[str(PU_BEST.relative_to(REPO))] = EXPECTED[PU_BEST]
    expected_by_path[str(PU_FINAL.relative_to(REPO))] = EXPECTED[PU_FINAL]
    rows: list[dict[str, object]] = []
    for relative, expected in sorted(expected_by_path.items()):
        path = REPO / relative
        observed = sha256_file(path)
        rows.append({
            "path": relative,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "status": "PASS" if observed == expected else "FAIL",
        })
    if len(rows) != 560 or any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("historical checkpoint inventory changed")
    return rows


def state_hashes() -> dict[str, str]:
    import torch

    payload = torch.load(PU_BEST, map_location="cpu", weights_only=False)
    state = payload["state_dict"]

    def digest_prefixes(prefixes: tuple[str, ...]) -> str:
        digest = hashlib.sha256()
        selected = [(name, tensor) for name, tensor in state.items() if name.startswith(prefixes)]
        if not selected:
            raise RuntimeError(f"empty state selection: {prefixes}")
        for name, tensor in sorted(selected):
            value = np.ascontiguousarray(tensor.detach().cpu().numpy())
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    return {
        "decoder": digest_prefixes(("enc1.", "enc2.", "bottleneck.", "dec2.", "dec1.", "decomposition_head.", "latent_injection.")),
        "posterior_network": digest_prefixes(("posterior.",)),
        "gaussian_prior_network": digest_prefixes(("prior.",)),
    }


def source_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            rows.append({
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return rows


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    verify_fixed_inputs()
    run_dir = create_run()
    started = datetime.now(timezone.utc).isoformat()
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

    state = state_hashes()
    disk = shutil.disk_usage(REPO)
    status = command(["git", "status", "--porcelain=v2", "--untracked-files=all"]).splitlines()
    relevant = {
        "source_layer_contract": REPO / "docs/multi_hypothesis_source_contract.md",
        "forward_consistency_contract": REPO / "docs/forward_consistency_contract.md",
        "forward_model": REPO / "src/btk_scene.py",
        "noise_contract": ATLAS / "manifests/fixed_noise_contract.json",
        "prompt_implementation": REPO / "scripts/thayer_select_prompt_ablation_common.py",
        "normalization": NORMALIZATION,
        "atlas_pair_manifest": ATLAS / "tables/atlas_pair_manifest.csv",
        "atlas_freeze_record": ATLAS / "manifests/atlas_initial_freeze_record.json",
    }
    provenance = {
        "campaign": "Thayer-PF conditional flow-prior truth-coverage campaign",
        "working_model_name": "Thayer-PF",
        "campaign_started_utc": started,
        "run_dir": str(run_dir.relative_to(REPO)),
        "branch": command(["git", "branch", "--show-current"]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_v2": status,
        "staged_index_empty": True,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__, "torch": torch.__version__, "btk": btk.__version__,
            "galsim": galsim.__version__, "astropy": package_version("astropy"),
            "h5py": package_version("h5py"), "scipy": package_version("scipy"),
            "scikit_image": package_version("scikit-image"), "surveycodex": package_version("surveycodex"),
        },
        "mps": {"built": torch.backends.mps.is_built(), "available": torch.backends.mps.is_available(), "probe": "PASS", "fallback_enabled": False},
        "free_disk_bytes": disk.free,
        "source_catalog": {"path": str(CATALOG.relative_to(REPO)), "sha256": sha256_file(CATALOG)},
        "source_split": {"path": str(SOURCE_SPLIT.relative_to(REPO)), "sha256": sha256_file(SOURCE_SPLIT)},
        "thayer_pu_best": {"path": str(PU_BEST.relative_to(REPO)), "sha256": sha256_file(PU_BEST)},
        "thayer_pu_final": {"path": str(PU_FINAL.relative_to(REPO)), "sha256": sha256_file(PU_FINAL)},
        "condition_c": {"path": str(CONDITION_C.relative_to(REPO)), "sha256": sha256_file(CONDITION_C)},
        "frozen_module_hashes": state,
        "relevant_artifacts": {name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for name, path in relevant.items()},
        "historical_checkpoint_count": len(checkpoints),
        "atlas_pair_artifact_count": 25,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
        "atlas_inference_count_this_campaign": 0,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)

    environment = f"""# Thayer-PF environment snapshot

- Started UTC: `{started}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python: `{sys.version.splitlines()[0]}`
- Torch / BTK / GalSim: `{torch.__version__}` / `{btk.__version__}` / `{galsim.__version__}`
- MPS: built, available, and execution-probed; fallback disabled.
- Free disk at start: `{disk.free}` bytes.
- Staged index: empty.
- Historical checkpoints inventoried: {len(checkpoints)}; all byte-identical.
- Frozen Atlas pair artifacts: 25; all byte-identical.
- Development / lockbox / new Atlas-inference access: 0 / 0 / 0.
"""
    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", environment)

    contract = """# Thayer-PF campaign contract

This append-only campaign tests whether a compact conditional flow prior can
place inference-time mass on truth modes already represented by the frozen
Thayer-PU posterior and decoder. The posterior/decoder sufficiency gate is
mandatory and precedes all flow implementation or fitting.

Frozen inputs are the authoritative Thayer-PU best/final checkpoints, Condition-C
checkpoint, source split, normalization, source-layer contract, forward/noise
contract, and frozen 25-pair Atlas. Atlas scenes and groups are excluded from
training, validation, calibration, and latent-teacher construction. The final
lockbox and unauthorized development data remain sealed.

Only MPS may execute neural inference or training; CPU is used for hashes,
metrics, tables, clustering, manifests, and reports. Posterior samples are
diagnostics or teacher targets only and are never deployable hypotheses. The
campaign stops before flow implementation if persisted baselines, the frozen
metric, or posterior/decoder sufficiency cannot be proven. The frozen Atlas is
not inferred on unless every non-Atlas gate passes and a one-time protocol is
hashed first. Historical runs and checkpoints are never modified.
"""
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", contract)
    write_json_fresh(run_dir / "logs/part_a_complete.json", {
        "status": "PASS", "run_dir": str(run_dir.relative_to(REPO)),
        "historical_checkpoint_count": len(checkpoints), "staged_index_empty": True,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    })
    print(run_dir)


if __name__ == "__main__":
    main()
