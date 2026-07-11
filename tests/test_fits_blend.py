"""Scientific contract tests for the FITS flux-space blend generator."""

from __future__ import annotations

import json
import unittest

import numpy as np

from src.fits_blend import (
    BlendTransform,
    ScientificValidityError,
    array_sha256,
    audit_fits_blend,
    blend_fits_cutouts,
    replay_fits_blend,
    sample_blend_transform,
)


class FitsBlendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 9
        self.width = 10
        y, x = np.mgrid[: self.height, : self.width]
        self.target = np.stack(
            (
                -4.0 + 0.1 * x - 0.2 * y,
                3.0 + 0.03 * x + 0.04 * y,
                20.0 - 0.05 * x + 0.02 * y,
            )
        ).astype(np.float32)
        backgrounds = np.array([12.0, -7.0, 100.0], dtype=np.float32)
        self.contaminant = np.broadcast_to(
            backgrounds[:, np.newaxis, np.newaxis], self.target.shape
        ).copy()
        self.source_mask = np.zeros((self.height, self.width), dtype=bool)
        self.source_mask[3:6, 3:6] = True
        self.background_mask = ~self.source_mask
        self.core_mask = np.zeros_like(self.source_mask)
        self.core_mask[4, 4] = True
        self.contaminant[0, self.source_mask] += 2.0
        self.contaminant[1, self.source_mask] -= 3.0
        self.contaminant[2, self.source_mask] += 8.0

    def make_result(
        self,
        *,
        target: np.ndarray | None = None,
        transform: BlendTransform | None = None,
        psf_policy: str = "not_verified",
    ):
        if transform is None:
            transform = BlendTransform(
                sample_seed=17,
                shift_xy=(0.0, 0.0),
                flux_scales=(1.0, 1.0, 1.0),
            )
        return blend_fits_cutouts(
            self.target if target is None else target,
            self.contaminant,
            source_mask=self.source_mask,
            background_mask=self.background_mask,
            core_mask=self.core_mask,
            target_coordinate_xy=(4.5, 4.0),
            contaminant_coordinate_xy=(4.0, 4.0),
            target_source_id="target-001",
            target_group_id="group-target",
            contaminant_source_id="contaminant-002",
            contaminant_group_id="group-contaminant",
            transform=transform,
            psf_policy=psf_policy,
        )

    def test_exact_flux_addition_and_target_background_preservation(self) -> None:
        target_before = self.target.copy()
        contaminant_before = self.contaminant.copy()
        result = self.make_result()

        np.testing.assert_array_equal(self.target, target_before)
        np.testing.assert_array_equal(self.contaminant, contaminant_before)
        np.testing.assert_array_equal(
            result.blend, result.target + result.source_only_contaminant
        )
        np.testing.assert_array_equal(
            result.blend[:, ~result.affected_mask],
            result.target[:, ~result.affected_mask],
        )
        np.testing.assert_array_equal(
            result.source_only_contaminant[:, ~result.affected_mask], 0.0
        )
        self.assertTrue(np.any(result.target < 0.0))
        self.assertTrue(np.any(result.source_only_contaminant < 0.0))
        self.assertTrue(np.any(result.blend > 1.0))
        report = audit_fits_blend(result)
        self.assertTrue(report["flux_addition_exact"])
        self.assertTrue(report["target_background_preserved_outside_mask"])
        self.assertTrue(report["no_double_constant_background"])
        self.assertTrue(report["no_hidden_clipping"])
        self.assertEqual(result.target_coordinate_xy, (4.5, 4.0))
        self.assertEqual(result.contaminant_coordinate_xy, (4.0, 4.0))
        self.assertEqual(result.target_source_id, "target-001")
        self.assertEqual(result.target_group_id, "group-target")
        self.assertEqual(result.contaminant_source_id, "contaminant-002")
        self.assertEqual(result.contaminant_group_id, "group-contaminant")

    def test_constant_background_is_not_added_as_a_rectangle(self) -> None:
        result = self.make_result()
        expected_source_values = (2.0, -3.0, 8.0)
        for band, expected in enumerate(expected_source_values):
            np.testing.assert_array_equal(
                result.source_only_contaminant[band, self.source_mask], expected
            )
            np.testing.assert_array_equal(
                result.source_only_contaminant[band, ~self.source_mask], 0.0
            )

        policy = result.metadata["source_extraction"]
        self.assertTrue(policy["constant_contaminant_background_removed"])
        self.assertFalse(policy["spatial_background_residual_removed"])
        self.assertTrue(policy["contaminant_coadd_noise_inside_mask_retained"])
        self.assertFalse(result.metadata["optional_effects"]["synthetic_noise_added"])

    def test_band_order_and_per_band_flux_scaling_are_preserved(self) -> None:
        transform = BlendTransform(
            sample_seed=19,
            shift_xy=(0.0, 0.0),
            flux_scales=(0.5, 2.0, 3.0),
        )
        result = self.make_result(transform=transform)

        self.assertEqual(result.metadata["band_order"], ["g", "r", "z"])
        np.testing.assert_array_equal(
            result.source_only_contaminant[:, 4, 4],
            np.array([1.0, -6.0, 24.0], dtype=np.float32),
        )
        self.assertEqual(
            result.metadata["transform"]["flux_scales_by_band"],
            {"g": 0.5, "r": 2.0, "z": 3.0},
        )

    def test_subpixel_shift_is_flux_conserving_and_does_not_wrap(self) -> None:
        target = np.zeros((3, 7, 9), dtype=np.float32)
        contaminant = np.full_like(target, 5.0)
        source_mask = np.zeros((7, 9), dtype=bool)
        source_mask[3, 7] = True
        contaminant[:, 3, 7] += np.array([2.0, 4.0, 6.0], dtype=np.float32)
        transform = BlendTransform(
            sample_seed=23,
            shift_xy=(0.5, 0.0),
            flux_scales=(1.0, 1.0, 1.0),
        )
        result = blend_fits_cutouts(
            target,
            contaminant,
            source_mask=source_mask,
            background_mask=~source_mask,
            target_coordinate_xy=(4.0, 3.0),
            contaminant_coordinate_xy=(7.0, 3.0),
            target_source_id="target",
            target_group_id="target-group",
            contaminant_source_id="contaminant",
            contaminant_group_id="contaminant-group",
            transform=transform,
        )

        np.testing.assert_array_equal(
            result.source_only_contaminant[:, 3, 7], [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            result.source_only_contaminant[:, 3, 8], [1.0, 2.0, 3.0]
        )
        self.assertFalse(result.affected_mask[:, :7].any())
        self.assertFalse(result.affected_mask[:, 0].any())
        np.testing.assert_allclose(
            result.source_only_contaminant.sum(axis=(1, 2)), [2.0, 4.0, 6.0]
        )
        self.assertFalse(result.metadata["edge_truncation"]["edge_truncated"])
        self.assertEqual(
            result.metadata["edge_truncation"][
                "source_support_retained_fraction"
            ],
            1.0,
        )

    def test_affected_mask_does_not_depend_on_target_or_prediction(self) -> None:
        transform = BlendTransform(
            sample_seed=29,
            shift_xy=(1.25, -0.5),
            flux_scales=(1.0, 0.8, 1.2),
        )
        first = self.make_result(target=np.zeros_like(self.target), transform=transform)
        second = self.make_result(
            target=np.full_like(self.target, 1.0e6), transform=transform
        )

        np.testing.assert_array_equal(first.affected_mask, second.affected_mask)
        np.testing.assert_array_equal(first.core_mask, second.core_mask)
        self.assertFalse(
            first.metadata["mask_policy"]["affected_mask_uses_prediction"]
        )
        self.assertFalse(first.metadata["mask_policy"]["affected_mask_uses_target"])

    def test_seeded_transform_and_metadata_replay_are_exact(self) -> None:
        transform_a = sample_blend_transform(
            123456,
            shift_x_range=(-2.0, 2.0),
            shift_y_range=(-1.0, 1.0),
            flux_scale_ranges=((0.5, 1.5), (0.7, 1.2), (0.2, 2.0)),
        )
        transform_b = sample_blend_transform(
            123456,
            shift_x_range=(-2.0, 2.0),
            shift_y_range=(-1.0, 1.0),
            flux_scale_ranges=((0.5, 1.5), (0.7, 1.2), (0.2, 2.0)),
        )
        self.assertEqual(transform_a, transform_b)
        original = self.make_result(transform=transform_a)
        replayed = replay_fits_blend(
            self.target,
            self.contaminant,
            source_mask=self.source_mask,
            background_mask=self.background_mask,
            core_mask=self.core_mask,
            metadata=original.metadata,
        )

        np.testing.assert_array_equal(original.blend, replayed.blend)
        np.testing.assert_array_equal(
            original.source_only_contaminant, replayed.source_only_contaminant
        )
        np.testing.assert_array_equal(original.affected_mask, replayed.affected_mask)
        self.assertEqual(original.metadata["hashes"], replayed.metadata["hashes"])
        json.dumps(original.metadata, sort_keys=True)

        altered_target = self.target.copy()
        altered_target[0, 0, 0] += 1.0
        with self.assertRaisesRegex(ScientificValidityError, "target_sha256"):
            replay_fits_blend(
                altered_target,
                self.contaminant,
                source_mask=self.source_mask,
                background_mask=self.background_mask,
                core_mask=self.core_mask,
                metadata=original.metadata,
            )

    def test_100_randomized_exact_replay_cases(self) -> None:
        for sample_seed in range(100):
            transform = sample_blend_transform(
                sample_seed,
                shift_x_range=(-1.0, 1.0),
                shift_y_range=(-1.0, 1.0),
                flux_scale_ranges=((0.2, 1.8), (0.3, 1.7), (0.4, 1.6)),
            )
            original = self.make_result(transform=transform)
            replayed = replay_fits_blend(
                self.target,
                self.contaminant,
                source_mask=self.source_mask,
                background_mask=self.background_mask,
                core_mask=self.core_mask,
                metadata=original.metadata,
            )
            np.testing.assert_array_equal(original.blend, replayed.blend)
            np.testing.assert_array_equal(
                original.source_only_contaminant,
                replayed.source_only_contaminant,
            )
            np.testing.assert_array_equal(
                original.affected_mask, replayed.affected_mask
            )
            self.assertEqual(original.metadata["hashes"], replayed.metadata["hashes"])

    def test_hashes_cover_blend_and_both_output_masks(self) -> None:
        result = self.make_result()
        hashes = result.metadata["hashes"]
        self.assertEqual(hashes["blend_sha256"], array_sha256(result.blend))
        self.assertEqual(
            hashes["affected_mask_sha256"], array_sha256(result.affected_mask)
        )
        self.assertEqual(hashes["core_mask_sha256"], array_sha256(result.core_mask))
        with self.assertRaises(ValueError):
            result.blend[0, 0, 0] = 999.0

    def test_scientific_gate_exposes_coadd_noise_and_unverified_psf(self) -> None:
        result = self.make_result()
        self.assertFalse(
            result.metadata["source_extraction"]["noise_free_source_established"]
        )
        self.assertFalse(
            result.metadata["scientific_gate"][
                "manifest_ready_if_noise_free_source_required"
            ]
        )
        with self.assertRaisesRegex(ScientificValidityError, "noise-free"):
            audit_fits_blend(result, require_noise_free_source=True)
        with self.assertRaisesRegex(ScientificValidityError, "PSF compatibility"):
            audit_fits_blend(result, require_psf_compatibility=True)

        caller_verified = self.make_result(
            psf_policy="caller_verified_compatible"
        )
        ordinary_report = audit_fits_blend(caller_verified)
        self.assertTrue(ordinary_report["psf_compatibility_asserted"])
        self.assertFalse(
            caller_verified.metadata["psf"]["independently_verified_by_generator"]
        )
        with self.assertRaisesRegex(ScientificValidityError, "caller assertion"):
            audit_fits_blend(caller_verified, require_psf_compatibility=True)
        caller_verified.metadata["psf"]["independently_verified_by_generator"] = True
        with self.assertRaisesRegex(AssertionError, "unsupported independent PSF"):
            audit_fits_blend(caller_verified, require_psf_compatibility=True)

    def test_noncanonical_band_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical survey order"):
            blend_fits_cutouts(
                self.target,
                self.contaminant,
                source_mask=self.source_mask,
                background_mask=self.background_mask,
                core_mask=self.core_mask,
                target_coordinate_xy=(4.5, 4.0),
                contaminant_coordinate_xy=(4.0, 4.0),
                target_source_id="target",
                target_group_id="target-group",
                contaminant_source_id="contaminant",
                contaminant_group_id="contaminant-group",
                transform=BlendTransform(
                    sample_seed=41,
                    shift_xy=(0.0, 0.0),
                    flux_scales=(1.0, 1.0, 1.0),
                ),
                band_order=("r", "g", "z"),
            )

    def test_rejects_integer_arrays_overlapping_masks_and_border_sources(self) -> None:
        transform = BlendTransform(
            sample_seed=31,
            shift_xy=(0.0, 0.0),
            flux_scales=(1.0, 1.0, 1.0),
        )
        kwargs = {
            "source_mask": self.source_mask,
            "background_mask": self.background_mask,
            "target_coordinate_xy": (4.0, 4.0),
            "contaminant_coordinate_xy": (4.0, 4.0),
            "target_source_id": "target",
            "target_group_id": "target-group",
            "contaminant_source_id": "contaminant",
            "contaminant_group_id": "contaminant-group",
            "transform": transform,
        }
        with self.assertRaisesRegex(TypeError, "floating FITS-array"):
            blend_fits_cutouts(
                self.target.astype(np.int16), self.contaminant, **kwargs
            )

        overlapping = self.background_mask.copy()
        overlapping[4, 4] = True
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            blend_fits_cutouts(
                self.target,
                self.contaminant,
                **{**kwargs, "background_mask": overlapping},
            )

        border_source = self.source_mask.copy()
        border_source[0, 4] = True
        border_background = ~border_source
        with self.assertRaisesRegex(ValueError, "touches a cutout border"):
            blend_fits_cutouts(
                self.target,
                self.contaminant,
                **{
                    **kwargs,
                    "source_mask": border_source,
                    "background_mask": border_background,
                },
            )


if __name__ == "__main__":
    unittest.main()
