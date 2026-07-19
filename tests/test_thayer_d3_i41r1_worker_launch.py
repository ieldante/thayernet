"""Candidate-001 regression for direct R1 worker process startup."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPO = Path(__file__).resolve().parents[1]


class ThayerD3I41R1WorkerLaunchTests(unittest.TestCase):
    def test_r1_worker_direct_script_resolves_repository_modules(self) -> None:
        result = subprocess.run(
            [
                str(REPO / ".venv-btk/bin/python"),
                "-B",
                str(REPO / "scripts/run_thayer_scientific_d3_process_v41r1.py"),
                "--help",
            ],
            cwd=REPO,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
