"""Focused fail-closed tests for the reusable D3 scientific capsule tools."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/runs/thayer_d3_scientific_capsule_20260713_155637"
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_d3_scientific_capsule import validate_capsule, validate_files
from d3_scientific_capsule_guard import CapsuleAccessViolation, validate_small_payload


@unittest.skipUnless((RUN / "contract/d3_scientific_capsule_v1.json").is_file(), "authoritative Thayer-D3C run not present")
class D3ScientificCapsuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.capsule_path = RUN / "contract/d3_scientific_capsule_v1.json"
        cls.schema_path = RUN / "schema/d3_scientific_capsule_v1.schema.json"
        cls.manifest_path = RUN / "contract/d3_scientific_capsule_manifest.json"
        cls.chain_path = RUN / "contract/d3_scientific_capsule_hash_chain.json"
        cls.capsule = json.loads(cls.capsule_path.read_text(encoding="utf-8"))
        cls.schema = json.loads(cls.schema_path.read_text(encoding="utf-8"))

    def test_authoritative_capsule_and_hash_chain_validate(self) -> None:
        result = validate_files(
            repo=REPO,
            capsule_path=self.capsule_path,
            schema_path=self.schema_path,
            manifest_path=self.manifest_path,
            hash_chain_path=self.chain_path,
        )
        self.assertEqual(result["status"], "PASS", result["errors"])

    def test_sky_vector_removal_fails_closed(self) -> None:
        corrupted = copy.deepcopy(self.capsule)
        corrupted["observation_configuration"].pop("scientific_sky_vector")
        self.assertTrue(validate_capsule(corrupted, self.schema, repo=REPO, verify_files=False))

    def test_threshold_operator_drift_fails_closed(self) -> None:
        corrupted = copy.deepcopy(self.capsule)
        corrupted["forward_plausibility"]["comparison_operators"]["global_chi_square_mean"] = "<"
        self.assertIn(
            "PLAUSIBILITY_OPERATOR_MISMATCH",
            validate_capsule(corrupted, self.schema, repo=REPO, verify_files=False),
        )

    def test_artifact_hash_drift_fails_closed(self) -> None:
        corrupted = copy.deepcopy(self.capsule)
        corrupted["scientific_artifact_references"]["cached_features"]["sha256"] = "0" * 64
        self.assertIn(
            "ARTIFACT_HASH_MISMATCH:cached_features",
            validate_capsule(corrupted, self.schema, repo=REPO, verify_files=True),
        )

    def test_capsule_only_preflight_markers(self) -> None:
        command = [
            str(REPO / ".venv-btk/bin/python"),
            "-B",
            str(SCRIPTS / "bootstrap_thayer_authoritative_d3_from_capsule.py"),
            "--repo",
            str(REPO),
            "--capsule",
            str(self.capsule_path),
            "--schema",
            str(self.schema_path),
            "--manifest",
            str(self.manifest_path),
            "--hash-chain",
            str(self.chain_path),
        ]
        result = subprocess.run(command, cwd=REPO, check=True, text=True, capture_output=True)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED",
                "READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION",
            ],
        )

    def test_small_payload_rank_one_is_accepted(self) -> None:
        self.assertEqual(validate_small_payload({"vector": [1.0, 2.0, 3.0]}), (3, 1))

    def test_small_payload_rank_two_fails_closed(self) -> None:
        with self.assertRaises(CapsuleAccessViolation):
            validate_small_payload({"matrix": [[1.0, 2.0], [3.0, 4.0]]})

    def test_small_payload_scalar_limit_fails_closed(self) -> None:
        with self.assertRaises(CapsuleAccessViolation):
            validate_small_payload({"vector": list(range(65))})


if __name__ == "__main__":
    unittest.main()
