"""Regression tests for metric, mask, and sample-alignment correctness.

These tests deliberately use tiny in-memory arrays.  They exercise no model,
dataset, checkpoint, accelerator, or output-directory path.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from scripts import run_stress_test
from src import utils


class WholeImageMetricTests(unittest.TestCase):
    def test_uint8_inputs_are_cast_before_subtraction(self) -> None:
        prediction = np.zeros((2, 2, 3), dtype=np.uint8)
        target = np.full((2, 2, 3), 255, dtype=np.uint8)

        self.assertEqual(utils.mse(prediction, target), 255.0**2)
        self.assertEqual(utils.mae(prediction, target), 255.0)

    def test_known_constant_rgb_metrics(self) -> None:
        target = np.zeros((8, 8, 3), dtype=np.float32)
        prediction = np.full_like(target, 0.5)

        result = utils.compute_metrics(prediction, target)

        self.assertAlmostEqual(result["mse"], 0.25, places=12)
        self.assertAlmostEqual(result["mae"], 0.5, places=12)
        self.assertAlmostEqual(result["psnr"], 20.0 * math.log10(2.0), places=12)
        # For two constant images, the contrast/structure terms cancel and
        # SSIM is C1 / (0.5**2 + C1), with C1=(0.01*data_range)**2.
        expected_ssim = 0.0001 / 0.2501
        self.assertAlmostEqual(result["ssim"], expected_ssim, places=12)

    def test_perfect_prediction_psnr_and_ssim(self) -> None:
        image = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float64).reshape(8, 8, 3)

        self.assertTrue(math.isinf(utils.psnr(image, image)))
        self.assertAlmostEqual(utils.ssim_metric(image, image), 1.0, places=12)


class AffectedAndMaskedMetricTests(unittest.TestCase):
    def test_affected_mask_threshold_is_strict_and_prediction_independent(self) -> None:
        target = np.zeros((2, 2, 3), dtype=np.float32)
        blend = target.copy()
        blend[0, 0] = 0.03
        blend[0, 1] = 0.02

        expected = np.array([[True, False], [False, False]])
        first_mask = utils.affected_region_mask(target, blend, threshold=0.02)

        prediction_a = np.zeros_like(target)
        prediction_b = np.ones_like(target)
        # Predictions affect the measured error, but never enter mask
        # construction.  Recomputing after changing them must be identical.
        np.testing.assert_array_equal(first_mask, expected)
        np.testing.assert_array_equal(
            utils.affected_region_mask(target, blend, threshold=0.02),
            first_mask,
        )
        self.assertNotEqual(
            utils.masked_mse(prediction_a, target, first_mask),
            utils.masked_mse(prediction_b, target, first_mask),
        )

    def test_uint8_affected_mask_does_not_wrap(self) -> None:
        target = np.zeros((1, 1, 3), dtype=np.uint8)
        blend = np.full((1, 1, 3), 255, dtype=np.uint8)

        self.assertTrue(utils.affected_region_mask(target, blend, threshold=254.0)[0, 0])

    def test_empty_mask_returns_nan_and_explicit_coverage(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.float32)
        empty = np.zeros((1, 1), dtype=bool)

        self.assertTrue(math.isnan(utils.masked_mse(image, image, empty)))
        self.assertTrue(math.isnan(utils.masked_mae(image, image, empty)))
        summary = utils.masked_mse_summary([image], [image], [empty])
        self.assertEqual(summary["n_total"], 1)
        self.assertEqual(summary["n_valid"], 0)
        self.assertEqual(summary["n_empty"], 1)
        self.assertTrue(math.isnan(float(summary["macro_mse"])))
        self.assertTrue(math.isnan(float(summary["micro_mse"])))

    def test_macro_and_micro_masked_mse_have_correct_counts(self) -> None:
        target_one = np.zeros((1, 1, 3), dtype=np.float32)
        prediction_one = np.ones_like(target_one)
        mask_one = np.ones((1, 1), dtype=bool)

        target_two = np.zeros((1, 2, 3), dtype=np.float32)
        prediction_two = np.full_like(target_two, 2.0)
        mask_two = np.ones((1, 2), dtype=bool)

        target_empty = np.zeros((1, 1, 3), dtype=np.float32)
        prediction_empty = np.full_like(target_empty, 100.0)
        mask_empty = np.zeros((1, 1), dtype=bool)

        summary = utils.masked_mse_summary(
            [prediction_one, prediction_two, prediction_empty],
            [target_one, target_two, target_empty],
            [mask_one, mask_two, mask_empty],
        )

        self.assertEqual(summary["n_total"], 3)
        self.assertEqual(summary["n_valid"], 2)
        self.assertEqual(summary["n_empty"], 1)
        self.assertAlmostEqual(float(summary["macro_mse"]), 2.5, places=12)
        # Three channel values at squared error 1 and six at squared error 4.
        self.assertAlmostEqual(float(summary["micro_mse"]), 3.0, places=12)


class RegionMaskTests(unittest.TestCase):
    def test_historical_eval_core_preserves_float32_tie_behavior(self) -> None:
        target = np.zeros((5, 5, 3), dtype=np.float64)
        target[2, 2] = 0.1
        target[2, 3] = 0.2
        # These distinct float64 values collapse to one float32 value.  The
        # historical implementation therefore includes both percentile ties.
        target[3, 2] = 1.00000006
        target[3, 3] = 1.00000007

        mask = utils.evaluation_core_mask_p85_v1(target)
        expected = np.zeros((5, 5), dtype=bool)
        expected[3, 2] = True
        expected[3, 3] = True

        np.testing.assert_array_equal(mask, expected)

    def test_v02_loss_core_uses_peak_fraction_not_percentile(self) -> None:
        target = np.zeros((5, 5, 3), dtype=np.float32)
        target[2, 2] = 1.0
        target[2, 3] = 0.60
        target[3, 2] = 0.54

        mask = utils.loss_core_mask_v02_numpy(
            target,
            aperture_fraction=0.30,
            brightness_fraction=0.55,
        )
        expected = np.zeros((5, 5), dtype=bool)
        expected[2, 2] = True
        expected[2, 3] = True

        np.testing.assert_array_equal(mask, expected)

    def test_halo_is_manhattan_dilation_minus_affected(self) -> None:
        affected = np.zeros((5, 5), dtype=bool)
        affected[2, 2] = True

        halo = utils.halo_band_mask_manhattan_v1(affected, dilation_iters=2)
        y_grid, x_grid = np.ogrid[:5, :5]
        expected = ((np.abs(y_grid - 2) + np.abs(x_grid - 2)) <= 2) & ~affected

        np.testing.assert_array_equal(halo, expected)
        self.assertEqual(int(halo.sum()), 12)
        self.assertFalse(utils.halo_band_mask_manhattan_v1(affected, 0).any())
        with self.assertRaises(ValueError):
            utils.halo_band_mask_manhattan_v1(affected, -1)


class SampleAlignmentTests(unittest.TestCase):
    def test_only_finite_aligned_pairs_enter_outcomes(self) -> None:
        result = utils.aligned_pair_outcomes(
            ["a", "b", "c", "d"],
            [1.0, np.nan, np.inf, 4.0],
            ["a", "b", "c", "d"],
            [2.0, 3.0, 4.0, 4.0],
        )

        self.assertEqual(result["n_total"], 4)
        self.assertEqual(result["n_valid_pairs"], 2)
        self.assertEqual(result["n_missing_pairs"], 2)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["ties"], 1)
        self.assertAlmostEqual(float(result["win_rate"]), 0.5, places=12)

    def test_reordered_or_missing_sample_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "identically ordered"):
            utils.aligned_pair_outcomes(
                ["a", "b"], [1.0, 2.0], ["b", "a"], [2.0, 1.0]
            )
        with self.assertRaisesRegex(ValueError, "identically ordered"):
            utils.aligned_pair_outcomes(
                ["a", "b"], [1.0, 2.0], ["a", "c"], [1.0, 2.0]
            )

    def test_duplicate_sample_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Candidate sample IDs must be unique"):
            utils.aligned_pair_outcomes(
                ["a", "a"], [1.0, 2.0], ["a", "b"], [1.0, 2.0]
            )
        with self.assertRaisesRegex(ValueError, "Reference sample IDs must be unique"):
            utils.aligned_pair_outcomes(
                ["a", "b"], [1.0, 2.0], ["a", "a"], [1.0, 2.0]
            )


class StressEvaluationGuardTests(unittest.TestCase):
    def test_sample_prediction_count_mismatch_refuses_zip_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing silent zip truncation"):
            run_stress_test.evaluate_samples([{}], [], affected_region_threshold=0.02)

    def test_improvement_ratio_perfect_denominator_policy_is_nan(self) -> None:
        self.assertAlmostEqual(run_stress_test.safe_ratio(2.0, 0.5), 4.0)
        self.assertTrue(math.isnan(run_stress_test.safe_ratio(1.0, 0.0)))
        self.assertTrue(math.isnan(run_stress_test.safe_ratio(1.0, -1.0)))
        self.assertTrue(math.isnan(run_stress_test.safe_ratio(np.nan, 1.0)))
        self.assertTrue(math.isnan(run_stress_test.safe_ratio(1.0, np.inf)))


if __name__ == "__main__":
    unittest.main()
