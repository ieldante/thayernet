import unittest

import numpy as np

from scripts.run_frozen_head_ablation import (
    auprc,
    auroc,
    binary_metrics,
    isotonic_apply,
    isotonic_fit,
    temperature_apply,
    temperature_fit,
)


class FrozenHeadMetricTests(unittest.TestCase):
    def test_binary_ranking_metrics_are_exact_on_perfect_ordering(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(auroc(scores, labels), 1.0)
        self.assertEqual(auprc(scores, labels), 1.0)
        self.assertEqual(binary_metrics(scores, labels)["balanced_accuracy"], 1.0)

    def test_isotonic_is_monotone_and_finite(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        labels = np.array([0, 1, 0, 1, 1, 1])
        calibrated = isotonic_apply(scores, isotonic_fit(scores, labels))
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(np.all(np.diff(calibrated) >= 0))
        self.assertTrue(np.all((calibrated >= 0) & (calibrated <= 1)))

    def test_temperature_scaling_is_monotone(self):
        scores = np.array([0.05, 0.2, 0.4, 0.8, 0.95])
        labels = np.array([0, 0, 1, 1, 1])
        temperature = temperature_fit(scores, labels)
        calibrated = temperature_apply(scores, temperature)
        self.assertGreater(temperature, 0)
        self.assertTrue(np.all(np.diff(calibrated) > 0))


if __name__ == "__main__":
    unittest.main()
