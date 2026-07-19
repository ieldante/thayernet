import unittest

import numpy as np

from src.psf_conditioning import (
    array_sha256,
    effective_configuration_count,
    kernel_distance,
    kernel_moments,
    meaningful_scene_variation,
    normalized_kernel,
)


class PSFConditioningTests(unittest.TestCase):
    def test_normalization_and_moments(self):
        yy, xx = np.indices((21, 21), dtype=float)
        kernel = np.exp(-((xx - 10.0) ** 2 + (yy - 10.0) ** 2) / 4.0)
        normalized = normalized_kernel(kernel)
        self.assertAlmostEqual(float(normalized.sum()), 1.0, places=14)
        moments = kernel_moments(kernel, 0.2)
        self.assertAlmostEqual(moments["centroid_x_pixel"], 10.0, places=12)
        self.assertAlmostEqual(moments["centroid_y_pixel"], 10.0, places=12)
        self.assertLess(moments["ellipticity_magnitude"], 1e-10)
        self.assertTrue(np.isnan(moments["orientation_radians"]))

    def test_hash_and_distance_are_deterministic(self):
        kernel = np.eye(5, dtype=np.float64) + 1.0
        first = normalized_kernel(kernel)
        second = normalized_kernel(kernel.copy())
        self.assertEqual(array_sha256(first), array_sha256(second))
        distances = kernel_distance(first, second)
        self.assertAlmostEqual(distances["l1"], 0.0, places=15)
        self.assertAlmostEqual(distances["l2"], 0.0, places=15)
        self.assertAlmostEqual(distances["cosine_distance"], 0.0, places=15)

    def test_effective_count_and_variation_gate(self):
        hashes = ["a", "a", "a"]
        self.assertAlmostEqual(effective_configuration_count(hashes), 1.0)
        self.assertFalse(meaningful_scene_variation(hashes, 0.0))
        self.assertTrue(meaningful_scene_variation(["a", "b"], 0.1))

    def test_invalid_kernel_fails_closed(self):
        with self.assertRaises(ValueError):
            normalized_kernel(np.asarray([[1.0, np.nan], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            normalized_kernel(np.zeros((5, 5)))


if __name__ == "__main__":
    unittest.main()
