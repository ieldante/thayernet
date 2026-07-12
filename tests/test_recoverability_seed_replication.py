"""Frozen choices for the Phase-II training-seed replication."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.run_thayer_select_seed_replication import (
    PARAMETER_COUNT,
    PRIMARY_CONTRACT,
    SEED_MAP,
    STABILITY_TOLERANCES,
)
from src.models_thayer_select import ThayerSelectNet


class SeedReplicationContractTests(unittest.TestCase):
    def test_seed_map_has_exactly_two_new_unique_pairs(self) -> None:
        self.assertEqual(list(SEED_MAP), ["R1_seed_2", "R1_seed_3"])
        pairs = [(value["initialization_seed"], value["minibatch_order_seed"]) for value in SEED_MAP.values()]
        self.assertEqual(len(set(pairs)), 2)
        self.assertNotIn(2026078101, {item for pair in pairs for item in pair})

    def test_architecture_and_contract_are_frozen(self) -> None:
        self.assertEqual(PRIMARY_CONTRACT, "permissive")
        self.assertEqual(sum(parameter.numel() for parameter in ThayerSelectNet(min_log_variance=-8.0, max_log_variance=2.0).parameters()), PARAMETER_COUNT)

    def test_stability_tolerances_are_positive_and_predeclared(self) -> None:
        self.assertEqual(set(STABILITY_TOLERANCES), {"auroc_range", "auprc_range", "brier_range", "risk_range_each_coverage", "null_hallucination_range"})
        self.assertTrue(np.all(np.asarray(list(STABILITY_TOLERANCES.values())) > 0))


if __name__ == "__main__":
    unittest.main()
