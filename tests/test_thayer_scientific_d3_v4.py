"""Static CLI, propagation, regression, and isolation tests for launcher v4."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from scripts import run_thayer_scientific_d3_v4 as v4


REPO = Path(__file__).resolve().parents[1]
CURRENT = REPO / "scripts/run_thayer_scientific_d3.py"
ORCHESTRATOR = REPO / "scripts/run_thayer_scientific_d3_v4.py"
WORKER = REPO / "scripts/run_thayer_scientific_d3_process_v4.py"
POST = REPO / "scripts/run_thayer_d3_postprocess_v4.py"
V2 = REPO / "outputs/runs/thayer_d3_executable_contract_20260713_164320/future_d3_bundle/d3_executable_bundle_v2.json"


def arguments(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args and isinstance(node.args[0], ast.Constant):
            result.add(str(node.args[0].value))
    return result


class HistoricalRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = CURRENT.read_text(encoding="utf-8")

    def test_current_launcher_scientific_branch_drops_bundle_v3(self) -> None:
        self.assertNotIn("--bundle-v3", arguments(CURRENT))

    def test_current_launcher_hardcodes_or_resolves_bundle_v2(self) -> None:
        self.assertIn("thayer_d3_executable_contract_20260713_164320", self.source)

    def test_current_launcher_missing_scientific_worker(self) -> None:
        self.assertFalse((REPO / "scripts/run_thayer_scientific_d3_process.py").exists())

    def test_current_launcher_missing_postprocess_worker(self) -> None:
        self.assertFalse((REPO / "scripts/postprocess_thayer_scientific_d3.py").exists())

    def test_v3_policy_marker_does_not_reach_scientific_worker(self) -> None:
        self.assertNotIn("BUNDLE_V3_PROPAGATED_TO_SCIENTIFIC_WORKER", self.source)

    def test_v2_scope_metadata_incorrectly_blocks_v3_continuation(self) -> None:
        self.assertFalse(json.loads(V2.read_text(encoding="utf-8"))["scope"]["automatic_scientific_continuation"])


class V4StaticIntegrationTests(unittest.TestCase):
    def test_orchestrator_cli_is_exact(self) -> None:
        self.assertEqual(
            arguments(ORCHESTRATOR),
            {"--bundle-v3", "--bundle-v3-sha256", "--output-dir", "--governing-request", "--synthetic-integration-preflight-only"},
        )

    def test_worker_requires_bridge_and_has_no_bundle_v2_argument(self) -> None:
        worker_arguments = arguments(WORKER)
        self.assertEqual(worker_arguments, {"--bridge-v4", "--bridge-v4-sha256", "--output-root", "--strict-runtime-root", "--mode"})
        self.assertNotIn("--bundle", worker_arguments)
        self.assertNotIn("--bundle-v2", worker_arguments)

    def test_no_hardcoded_bundle_v2_path_in_scientific_branch(self) -> None:
        combined = ORCHESTRATOR.read_text(encoding="utf-8") + WORKER.read_text(encoding="utf-8")
        self.assertNotIn("thayer_d3_executable_contract_20260713_164320", combined)

    def test_workers_are_present_and_named_statically(self) -> None:
        self.assertTrue(WORKER.is_file())
        self.assertTrue(POST.is_file())
        source = ORCHESTRATOR.read_text(encoding="utf-8")
        self.assertIn("scripts/run_thayer_scientific_d3_process_v4.py", source)
        self.assertIn("scripts/run_thayer_d3_postprocess_v4.py", source)

    def test_bundle_v3_and_bridge_hash_propagate(self) -> None:
        source = ORCHESTRATOR.read_text(encoding="utf-8")
        self.assertIn("--bundle-v3-sha256", source)
        self.assertIn("--bridge-v4-sha256", source)
        self.assertIn("BUNDLE_V3_PROPAGATED_TO_SCIENTIFIC_WORKER", WORKER.read_text(encoding="utf-8"))

    def test_scientific_worker_has_no_matplotlib_import(self) -> None:
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(name == "matplotlib" or name.startswith("matplotlib.") for name in imported))

    def test_postprocessor_accepts_one_explicit_manifest(self) -> None:
        self.assertEqual(arguments(POST), {"--input-manifest"})
        source = POST.read_text(encoding="utf-8")
        self.assertIn("D3I-POST-ORIGINAL-SCIENTIFIC-INPUT", source)


class V4ProcessBoundaryTests(unittest.TestCase):
    def test_preflight_orchestrator_creates_candidate_log_parent_before_exclusive_open(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            preflight_root = (
                temporary_root
                / "missing_intermediate"
                / "candidate_002"
                / "preflight_root"
            )
            log_path = (
                preflight_root
                / "logs"
                / "actual_scientific_worker_pre_science.log"
            )

            self.assertFalse(preflight_root.exists())
            self.assertFalse(log_path.parent.exists())
            self.assertFalse((temporary_root / "logs").exists())

            v4.run_process(
                [sys.executable, "-c", "print('candidate-002-worker-launched')"],
                log_path,
                os.environ.copy(),
            )

            self.assertTrue(log_path.parent.is_dir())
            self.assertTrue(log_path.is_file())
            self.assertEqual(log_path.parent.parent, preflight_root)
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "candidate-002-worker-launched\n",
            )
            original_log = log_path.read_bytes()

            with self.assertRaises(FileExistsError):
                v4.run_process(
                    [sys.executable, "-c", "print('must-not-overwrite')"],
                    log_path,
                    os.environ.copy(),
                )

            self.assertEqual(log_path.read_bytes(), original_log)


if __name__ == "__main__":
    unittest.main()
