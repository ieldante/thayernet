import unittest

import numpy as np

from scripts.calibrate_hierarchical_safety import select_query_thresholds
from scripts.train_hierarchical_query_gate import macro_metrics


class QueryGateMetricTests(unittest.TestCase):
    def test_perfect_three_class_metrics(self):
        truth = np.asarray([0, 1, 2, 0, 1, 2])
        probability = np.eye(3)[truth] * 0.98 + 0.01
        metrics = macro_metrics(probability, truth)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["null_false_accept_rate"], 0.0)
        self.assertAlmostEqual(metrics["ambiguous_false_accept_rate"], 0.0)
        self.assertAlmostEqual(metrics["unique_false_reject_rate"], 0.0)

    def test_weighted_natural_summary_changes_support_only(self):
        truth = np.asarray([0, 1, 2])
        probability = np.eye(3) * 0.98 + 0.01
        metrics = macro_metrics(probability, truth, np.asarray([7.0, 2.0, 1.0]))
        self.assertEqual(metrics["support"], [7.0, 2.0, 1.0])
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)

    def test_threshold_selection_rejects_invalid_queries(self):
        probability = np.asarray([
            [0.95, 0.03, 0.02], [0.90, 0.05, 0.05],
            [0.02, 0.96, 0.02], [0.03, 0.95, 0.02],
            [0.20, 0.05, 0.75], [0.25, 0.05, 0.70],
        ])
        truth = np.asarray([0, 0, 1, 1, 2, 2])
        result = select_query_thresholds(probability, truth)
        self.assertTrue(result["nondegenerate"])
        self.assertLessEqual(result["null_false_accept_rate"], 0.05)
        self.assertLessEqual(result["ambiguous_false_accept_rate"], 0.10)


if __name__ == "__main__":
    unittest.main()
