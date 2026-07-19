"""Static regression tests for the three-process Thayer-D3B boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
ORCHESTRATOR = REPO / "scripts/run_thayer_d3_readiness.py"
SCIENTIFIC = REPO / "scripts/run_thayer_d3_scientific_readiness.py"
POSTPROCESS = REPO / "scripts/run_thayer_d3_postprocess_readiness.py"


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class D3ReadinessProcessIsolationTest(unittest.TestCase):
    def test_three_entry_points_exist(self) -> None:
        self.assertTrue(ORCHESTRATOR.is_file())
        self.assertTrue(SCIENTIFIC.is_file())
        self.assertTrue(POSTPROCESS.is_file())

    def test_orchestrator_is_standard_library_only(self) -> None:
        forbidden = {"numpy", "torch", "matplotlib", "h5py", "src"}
        self.assertFalse(imported_roots(ORCHESTRATOR) & forbidden)

    def test_scientific_launcher_has_no_plotting_edge(self) -> None:
        source = SCIENTIFIC.read_text(encoding="utf-8")
        forbidden = {"matplotlib", "pyplot", "seaborn", "plotly"}
        self.assertFalse(imported_roots(SCIENTIFIC) & forbidden)
        self.assertNotIn("matplotlib", source.casefold())

    def test_postprocessor_is_a_distinct_launcher(self) -> None:
        self.assertNotEqual(SCIENTIFIC.resolve(), POSTPROCESS.resolve())
        self.assertIn("matplotlib", POSTPROCESS.read_text(encoding="utf-8").casefold())


if __name__ == "__main__":
    unittest.main()
