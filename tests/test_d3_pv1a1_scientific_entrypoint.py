"""Regression-first coverage for the frozen PV1-A1 scientific command."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest


REPO = Path(__file__).resolve().parents[1]
R2 = REPO / "outputs/runs/thayer_d3_pv1a1_readiness_r2_20260714_165947"
BUNDLE = R2 / "protocol_bundle/THAYER-D3-PV1-A1"
ENTRYPOINT = REPO / "scripts/run_thayer_d3_pv1a1_scientific.py"


def _template() -> dict[str, object]:
    return json.loads((BUNDLE / "scientific_command_template.json").read_text())


def _entrypoint_module():
    spec = importlib.util.spec_from_file_location("pv1a1_scientific_entrypoint", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pv1a1_future_command_references_exact_entrypoint() -> None:
    command = _template()["command"]
    assert command[1] == str(ENTRYPOINT)
    hashes = json.loads((BUNDLE / "protocol_hashes.json").read_text())
    assert _template()["preflight_protocol_content_tree_sha256"] == hashes["protocol_content_tree_before_command_sha256"]


def test_pv1a1_missing_scientific_entrypoint_reproduces_r2_failure() -> None:
    audit = json.loads((R2 / "audit/final_audit.json").read_text())
    assert audit["checks"]["future_command_entrypoint_exists"] is False
    assert audit["blockers"] == [{
        "classification": "IMPLEMENTATION_OR_CONTRACT_FAILURE",
        "message": f"frozen future command entrypoint does not exist: {ENTRYPOINT}",
    }]


def test_pv1a1_scientific_entrypoint_exists() -> None:
    assert ENTRYPOINT.is_file()


def test_pv1a1_scientific_entrypoint_imports() -> None:
    module = _entrypoint_module()
    assert callable(module.main)


def test_pv1a1_scientific_entrypoint_cli_matches_future_template() -> None:
    module = _entrypoint_module()
    command = _template()["command"]
    args = module.build_parser().parse_args(command[2:])
    assert args.protocol_bundle == BUNDLE
    assert args.protocol_hashes == BUNDLE / "protocol_hashes.json"
    assert args.readiness_root == R2
    assert args.create_fresh_timestamped_campaign_root is True
    assert args.execute_authoritative_science is True


def test_pv1a1_scientific_entrypoint_uses_protocol_operation_registry() -> None:
    module = _entrypoint_module()
    expected = tuple(_template()["required_operations"])
    assert module.load_operation_registry(BUNDLE) == expected


def test_pv1a1_scientific_entrypoint_declares_all_twenty_operations() -> None:
    """The 20 user obligations map onto the bundle's exact 15 machine stages."""

    module = _entrypoint_module()
    registry = module.load_operation_registry(BUNDLE)
    assert module.SCIENTIFIC_ORCHESTRATION_OBLIGATION_COUNT == 20
    assert registry == tuple(json.loads((BUNDLE / "execution_contract.json").read_text())["future_sequence"])
    assert len(registry) == 15  # Never invent five operations absent from the frozen bundle.


def test_pv1a1_scientific_entrypoint_operation_order_matches_protocol() -> None:
    module = _entrypoint_module()
    registry = list(module.load_operation_registry(BUNDLE))
    assert module.validate_operation_registry(BUNDLE, registry) == tuple(registry)
    with pytest.raises(module.EntrypointContractError, match="ordered registry"):
        module.validate_operation_registry(BUNDLE, registry[:2] + registry[3:4] + registry[2:3] + registry[4:])
    with pytest.raises(module.EntrypointContractError, match="missing operation"):
        module.validate_operation_registry(BUNDLE, registry[:-1])
    with pytest.raises(module.EntrypointContractError, match="duplicate operation"):
        module.validate_operation_registry(BUNDLE, registry + registry[-1:])
    with pytest.raises(module.EntrypointContractError, match="unknown operation"):
        module.validate_operation_registry(BUNDLE, registry[:-1] + ["UNKNOWN_OPERATION"])


def test_pv1a1_scientific_entrypoint_does_not_duplicate_scientific_constants() -> None:
    tree = ast.parse(ENTRYPOINT.read_text())
    forbidden_numbers = {2026071201, 2026071202, 611.9199829101562, 1805.8800048828125, 1854.199951171875}
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))}
    assert constants.isdisjoint(forbidden_numbers)
    assert "IMAGE_THRESHOLD" not in ENTRYPOINT.read_text()


def test_pv1a1_scientific_entrypoint_excludes_legacy_checkpoint_loading() -> None:
    module = _entrypoint_module()
    source = ENTRYPOINT.read_text()
    assert "reject_protected_checkpoint_use" in source
    assert module.legacy_checkpoint_load_attempts() == 0


def test_pv1a1_scientific_entrypoint_uses_fresh_seeded_initialization() -> None:
    source = ENTRYPOINT.read_text()
    assert "construct_fresh_initial_state" in source
    assert "load_state_dict" not in source


def test_pv1a1_scientific_entrypoint_preserves_audit_noninterference() -> None:
    module = _entrypoint_module()
    report = module.run_noninterference_probe(BUNDLE)
    assert report["status"] == "PASS"
    assert all(report["checks"].values())


def test_pv1a1_exact_future_command_reaches_pre_science_boundary(tmp_path: Path) -> None:
    command = list(_template()["command"])
    env = os.environ.copy()
    env["THAYER_PV1A1_PRE_SCIENCE_ONLY"] = "1"
    env["THAYER_PV1A1_OUTPUT_ROOT"] = str(tmp_path / "actual_entrypoint_traversal")
    completed = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "READY_TO_EXECUTE_PV1A1_SCIENCE" in completed.stdout
    traversal = json.loads((tmp_path / "actual_entrypoint_traversal/diagnostics/pre_science_traversal.json").read_text())
    assert traversal["operation_registry"] == _template()["required_operations"]
    assert traversal["scientific_target_dependent_losses"] == 0


def test_pv1a1_entrypoint_rejects_wrong_frozen_hashes(tmp_path: Path) -> None:
    module = _entrypoint_module()
    copied = tmp_path / "THAYER-D3-PV1-A1"
    shutil.copytree(BUNDLE, copied)
    hashes_path = copied / "protocol_hashes.json"
    hashes = json.loads(hashes_path.read_text())
    hashes["protocol_bundle_sha256"] = "0" * 64
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    with pytest.raises(module.EntrypointContractError, match="protocol-bundle hash"):
        module.validate_frozen_bundle(copied, hashes_path, R2)

    shutil.rmtree(copied)
    shutil.copytree(BUNDLE, copied)
    hashes_path = copied / "protocol_hashes.json"
    hashes = json.loads(hashes_path.read_text())
    hashes["source_freeze_sha256"] = "0" * 64
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    with pytest.raises(module.EntrypointContractError, match="source-freeze hash"):
        module.validate_frozen_bundle(copied, hashes_path, R2)


def test_pv1a1_entrypoint_help_uses_exact_cli() -> None:
    completed = subprocess.run([sys.executable, str(ENTRYPOINT), "--help"], cwd=REPO, text=True, capture_output=True)
    assert completed.returncode == 0
    for flag in ("--protocol-bundle", "--protocol-hashes", "--readiness-root", "--create-fresh-timestamped-campaign-root", "--execute-authoritative-science"):
        assert flag in completed.stdout


def test_pv1a1_worker_hashes_batched_tensors_without_rank_error() -> None:
    module = _entrypoint_module()
    import torch

    value = torch.zeros((1, 2, 2, 6, 60, 60), dtype=torch.float32)
    digest = module._tensor_observation_sha256(value, "N_PROMPT_EXPERT_CHANNEL_HEIGHT_WIDTH")
    assert len(digest) == 64


def test_pv1a1_assignment_flip_audit_normalizes_device() -> None:
    module = _entrypoint_module()
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    previous = torch.tensor([[True, False]], device="cpu")
    current = torch.tensor([[False, False]], device="mps")
    assert module._assignment_flip_count(current, previous) == 1


def test_pv1a1_worker_diagnostic_booleans_are_strict_json_native() -> None:
    module = _entrypoint_module()
    import numpy as np

    value = module._finite_positive(np.float64(1.0))
    assert type(value) is bool
    assert json.dumps({"value": value}, allow_nan=False) == '{"value": true}'
