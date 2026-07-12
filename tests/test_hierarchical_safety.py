import unittest

import numpy as np
import torch

from src.hierarchical_safety import (
    HierarchicalQuerySemantics,
    QueryState,
    associate_hierarchical_query,
    conformal_upper_bound,
    conformal_upper_offset,
    metric_specific_risks,
    pinball_loss,
)


class QuerySemanticTests(unittest.TestCase):
    def setUp(self):
        self.sources = np.asarray([[10.0, 10.0], [18.0, 10.0]])

    def test_exact_and_perturbed_are_unique(self):
        for prompt in ([10.0, 10.0], [12.0, 10.0]):
            result = associate_hierarchical_query(self.sources, prompt, image_shape=(60, 60))
            self.assertEqual(result.state, QueryState.UNIQUE_VALID)
            self.assertEqual(result.matched_index, 0)

    def test_alternate_source_is_valid_for_alternate(self):
        result = associate_hierarchical_query(self.sources, [18.0, 10.0], image_shape=(60, 60))
        self.assertEqual((result.state, result.matched_index), (QueryState.UNIQUE_VALID, 1))

    def test_null(self):
        result = associate_hierarchical_query(self.sources, [40.0, 40.0], image_shape=(60, 60))
        self.assertEqual(result.state, QueryState.NULL)
        self.assertIsNone(result.matched_index)

    def test_equal_and_near_ties_are_ambiguous(self):
        for prompt in ([14.0, 10.0], [13.75, 10.0]):
            result = associate_hierarchical_query(self.sources, prompt, image_shape=(60, 60))
            self.assertEqual(result.state, QueryState.AMBIGUOUS)
            self.assertIsNone(result.matched_index)

    def test_clear_nearest_with_two_candidates_is_unique(self):
        sources = np.asarray([[10.0, 10.0], [14.0, 10.0]])
        result = associate_hierarchical_query(sources, [10.5, 10.0], image_shape=(60, 60))
        self.assertEqual((result.state, result.matched_index), (QueryState.UNIQUE_VALID, 0))

    def test_edges_and_outside(self):
        result = associate_hierarchical_query(np.asarray([[1.0, 1.0]]), [0.0, 0.0], image_shape=(60, 60))
        self.assertEqual(result.state, QueryState.UNIQUE_VALID)
        with self.assertRaises(ValueError):
            associate_hierarchical_query(self.sources, [-0.01, 10.0], image_shape=(60, 60))


class RiskAndCalibrationTests(unittest.TestCase):
    def test_exact_reconstruction_has_zero_risk(self):
        truth = np.zeros((3, 5, 5), dtype=np.float32)
        truth[:, 2, 2] = [2.0, 3.0, 4.0]
        alternate = np.zeros_like(truth)
        alternate[:, 1, 1] = 1.0
        risks = metric_specific_risks(truth, truth, alternate, flux_floor_by_band=np.ones(3) * 0.01)
        self.assertAlmostEqual(risks["image_risk"], 0.0)
        self.assertAlmostEqual(risks["flux_risk_max"], 0.0)
        self.assertAlmostEqual(risks["centroid_risk_pixels"], 0.0)
        self.assertFalse(risks["confusion_risk"])

    def test_empty_truth_fails_closed(self):
        zero = np.zeros((3, 3, 3), dtype=np.float32)
        risks = metric_specific_risks(zero, zero, zero, flux_floor_by_band=np.ones(3))
        self.assertTrue(np.isinf(risks["image_risk"]))

    def test_pinball_tiny_array(self):
        pred = torch.tensor([0.0, 2.0])
        truth = torch.tensor([1.0, 1.0])
        self.assertAlmostEqual(float(pinball_loss(pred, truth, 0.9)), 0.5, places=6)

    def test_split_conformal_upper_bound(self):
        residuals = np.arange(10, dtype=float)
        offset = conformal_upper_offset(residuals, 0.2)
        self.assertEqual(offset, 8.0)
        np.testing.assert_array_equal(conformal_upper_bound(np.asarray([1.0, 2.0]), offset), [9.0, 10.0])


if __name__ == "__main__":
    unittest.main()
