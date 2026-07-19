from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

import numpy as np

from scripts.run_thayer_pu_eligibility_v1 import (
    BATCH_SIZE,
    CHECKPOINT,
    EXPECTED_COUNTS,
    EXPECTED_HASHES,
    K,
    LATENT_SEEDS,
    deployed_mean,
    pairwise_candidate_diameter,
    sha256_file,
)


class ThayerPUEligibilityContractTests(unittest.TestCase):
    def test_frozen_rule_constants(self) -> None:
        self.assertEqual(K, 16)
        self.assertEqual(BATCH_SIZE, 8)
        self.assertEqual(LATENT_SEEDS, tuple(range(2026077600, 2026077616)))
        self.assertEqual(EXPECTED_COUNTS, {"training": 3998, "validation": 793, "calibration": 2800})

    def test_deployed_mean_uses_requested_half_and_float32(self) -> None:
        candidates = np.zeros((1, K, 6, 60, 60), dtype=np.float32)
        for index in range(K):
            candidates[0, index, :3] = index + 0.25
            candidates[0, index, 3:] = 1000 + index
        deployed = deployed_mean(candidates)
        expected = np.mean(np.arange(K, dtype=np.float64) + 0.25)
        self.assertEqual(deployed.shape, (1, 3, 60, 60))
        self.assertEqual(deployed.dtype, np.float32)
        self.assertTrue(np.all(deployed == np.float32(expected)))

    def test_candidate_diameter_is_maximum_pairwise_rms(self) -> None:
        candidates = np.zeros((K, 3, 60, 60), dtype=np.float32)
        candidates[1] = 1.0
        candidates[2] = -2.0
        self.assertAlmostEqual(pairwise_candidate_diameter(candidates), 3.0, places=12)

    def test_checkpoint_hash_is_frozen(self) -> None:
        self.assertEqual(sha256_file(CHECKPOINT), EXPECTED_HASHES[CHECKPOINT])

    def test_run_artifact_contract_when_requested(self) -> None:
        value = os.environ.get("THAYER_PU_ELIGIBILITY_RUN")
        if not value:
            self.skipTest("THAYER_PU_ELIGIBILITY_RUN not set")
        run = Path(value)
        deployment = json.loads((run / "deployment_rule/frozen_deployment_rule.json").read_text())
        preflight = json.loads((run / "logs/preflight_complete.json").read_text())
        self.assertFalse(deployment["selection_uses_truth"])
        self.assertFalse(deployment["selection_uses_atlas"])
        self.assertEqual(deployment["candidate_selection"], "none")
        self.assertEqual(preflight["scientific_classification"], "THAYER_PU_DEPLOYMENT_INELIGIBLE")
        self.assertFalse(preflight["full_inference_authorized"])
        self.assertFalse((run / "logs/inference_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
