"""Regression and static integration tests for Thayer-D3I41."""

from __future__ import annotations

from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
ORCHESTRATOR = REPO / "scripts/run_thayer_scientific_d3_v41.py"
WORKER = REPO / "scripts/run_thayer_scientific_d3_process_v41.py"
FROZEN_WORKER = REPO / "scripts/run_thayer_scientific_d3_process_v4.py"


class ThayerScientificD3V41Tests(unittest.TestCase):
    def source(self, path: Path) -> str:
        if not path.is_file():
            self.fail(f"required append-only v4.1 source is absent: {path}")
        return path.read_text(encoding="utf-8")

    def test_torch_utils_serialization_is_imported_during_bootstrap(self) -> None:
        source = self.source(WORKER)
        self.assertIn("import torch.utils.serialization as torch_utils_serialization", source)
        self.assertIn("import torch.utils.serialization.config", source)

    def test_torch_serialization_path_is_prewarmed_in_bootstrap_scratch(self) -> None:
        source = self.source(WORKER)
        self.assertIn("serialization_prewarm", source)
        self.assertIn("torch.save", source)
        self.assertIn("torch.load", source)
        self.assertIn("weights_only=True", source)

    def test_strict_phase_has_torch_utils_serialization_in_sys_modules(self) -> None:
        source = self.source(WORKER)
        self.assertIn("required_serialization_modules", source)
        self.assertIn("sys.modules", source)
        self.assertLess(source.index("required_serialization_modules"), source.index('guard.transition("strict")'))

    def test_strict_phase_checkpoint_path_causes_zero_external_pyc_reads(self) -> None:
        source = self.source(WORKER)
        self.assertIn("strict_external_pyc_reads", source)
        self.assertIn("D3I41-STRICT-EXTERNAL-PYC-READ", source)

    def test_strict_phase_does_not_broaden_package_read_allowlist(self) -> None:
        source = self.source(WORKER)
        self.assertIn("strict_write_roots=(output,)", source)
        self.assertNotIn("strict_pyc_read_roots", source)
        self.assertNotIn("allow_arbitrary_pyc", source)

    def test_scientific_worker_bytecode_writing_remains_disabled(self) -> None:
        orchestrator = self.source(ORCHESTRATOR)
        self.assertIn('"PYTHONDONTWRITEBYTECODE": "1"', orchestrator)
        self.assertIn('"-B"', orchestrator)

    def test_serialization_bootstrap_uses_no_scientific_checkpoint(self) -> None:
        source = self.source(WORKER)
        self.assertIn('"synthetic_probe"', source)
        self.assertIn("scientific_checkpoint_opened", source)
        self.assertIn("False", source)

    def test_v41_reloads_same_eight_containers_and_ninety_one_members(self) -> None:
        source = self.source(WORKER)
        self.assertIn("len(rows) != 8", source)
        self.assertIn("loaded_members != 91", source)

    def test_v41_dtype_validation_passes_previous_float32_member(self) -> None:
        source = self.source(WORKER)
        self.assertIn("numpy_dtype_contract_equal(d1[name], contracts", source)
        self.assertNotIn("str(d1[name].dtype) !=", source)

    def test_v41_reaches_model_construction_after_member_validation(self) -> None:
        adapter = self.source(WORKER)
        frozen_source = FROZEN_WORKER.read_text(encoding="utf-8")
        authoritative_source = frozen_source[frozen_source.index("def run_authoritative") :]
        self.assertIn("frozen.load_scientific_assets = load_scientific_assets_v41", adapter)
        self.assertLess(
            authoritative_source.index("assets = load_scientific_assets"),
            authoritative_source.index("experts = construct_experts"),
        )

    def test_v41_bridge_inherits_all_v4_scientific_authorities(self) -> None:
        source = self.source(ORCHESTRATOR)
        self.assertIn("FROZEN_BRIDGE_V4", source)
        self.assertIn('candidate["authorities"]', source)
        self.assertIn('candidate["scientific_contract"]', source)

    def test_v41_changes_only_two_authorized_implementation_contracts(self) -> None:
        source = self.source(ORCHESTRATOR)
        for assertion in (
            '"scientific_values_changed": False',
            '"artifact_values_changed": False',
            '"model_changed": False',
            '"optimizer_changed": False',
            '"policy_changed": False',
            '"runtime_strict_permissions_broadened": False',
            '"dtype_comparison_normalized": True',
            '"serialization_preimport_added": True',
        ):
            self.assertIn(assertion, source)

    def test_frozen_v4_worker_remains_literal_for_regression_proof(self) -> None:
        source = FROZEN_WORKER.read_text(encoding="utf-8")
        self.assertIn("str(d1[name].dtype) != contracts", source)


if __name__ == "__main__":
    unittest.main()
