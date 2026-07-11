"""Focused tests for duplicate-safe splitting and invertible normalization."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from scripts import build_dr10_grouped_split as grouped
from scripts import study_dr10_normalization as normalization


class GroupLeakageTests(unittest.TestCase):
    def test_preallocation_rehash_detects_post_audit_header_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.fits"
            fits.PrimaryHDU(np.ones((3, 5, 5), dtype=np.float32)).writeto(path)
            record = {
                "_record_key": "source-a",
                "fits_path": str(path),
                "audited_file_sha256": grouped.file_sha256(path),
                "pixel_hash": grouped.current_fits_pixel_hash(path),
            }
            self.assertEqual(grouped.revalidate_accepted_fits([record]), 1)
            self.assertEqual(record["preallocation_hash_revalidation_pass"], 1)

            fits.setval(path, "DRIFT", value=1)
            with self.assertRaisesRegex(ValueError, "full-file SHA-256 changed"):
                grouped.revalidate_accepted_fits([record])

    def test_all_available_join_keys_must_resolve_to_the_same_row(self) -> None:
        source = {"catalog_row_index": "7", "source_id": "source-a"}
        audit_rows = [
            {"catalog_row_index": "7", "source_id": "source-b"},
            {"catalog_row_index": "8", "source_id": "source-a"},
        ]
        indices = grouped._index_rows(audit_rows)

        with self.assertRaisesRegex(ValueError, "Conflicting .* join keys"):
            grouped.unique_audit_match(source, indices, "test audit")

    def test_source_id_match_cannot_hide_dr8_id_conflict(self) -> None:
        source = {
            "catalog_row_index": "7",
            "source_id": "source-a",
            "dr8_id": "dr8-a",
        }
        audit_rows = [
            {
                "catalog_row_index": "7",
                "source_id": "source-a",
                "dr8_id": "dr8-b",
            }
        ]
        indices = grouped._index_rows(audit_rows)

        with self.assertRaisesRegex(ValueError, "Conflicting .* identifiers"):
            grouped.unique_audit_match(source, indices, "test audit")

    def test_mandatory_source_id_grouping_cannot_be_disabled(self) -> None:
        records = [
            {
                "_record_key": "a",
                "source_id": "same-source",
                "ra": "10.0",
                "dec": "-1.0",
                "pixel_hash": "a" * 64,
                "balance_stratum": "smooth|brightness_q0",
            },
            {
                "_record_key": "b",
                "source_id": "same-source",
                "ra": "11.0",
                "dec": "-2.0",
                "pixel_hash": "b" * 64,
                "balance_stratum": "featured|brightness_q1",
            },
        ]

        groups, evidence = grouped.construct_components(records, [], [])

        self.assertEqual(len(groups), 1)
        self.assertIn("stable_id:source_id", {item.evidence_type for item in evidence})

    def test_brightness_strata_use_one_flux_like_definition(self) -> None:
        records = [
            {
                "central_flux_r": "10",
                "mag_r": "19",
                "smooth_fraction": "0.9",
            },
            {
                "central_flux_r": "",
                "mag_r": "25",
                "smooth_fraction": "0.8",
            },
        ]

        _edges, definition = grouped.annotate_balance_strata(
            records, ["smooth_fraction"], 3
        )

        self.assertEqual(definition, "central_flux_r")
        self.assertEqual(
            {row["balance_brightness_source"] for row in records},
            {"central_flux_r"},
        )
        self.assertEqual(records[1]["balance_brightness_value"], "")

    def test_exact_evidence_is_unioned_before_role_assignment(self) -> None:
        records = [
            {
                "_record_key": "a",
                "source_id": "source-a",
                "ra": "10.0",
                "dec": "-1.0",
                "pixel_hash": "a" * 64,
                "balance_stratum": "smooth|brightness_q0",
            },
            {
                "_record_key": "b",
                "source_id": "source-b",
                "ra": "10.0",
                "dec": "-1.0",
                "pixel_hash": "b" * 64,
                "balance_stratum": "featured|brightness_q1",
            },
            {
                "_record_key": "c",
                "source_id": "source-c",
                "ra": "11.0",
                "dec": "-2.0",
                "pixel_hash": "a" * 64,
                "balance_stratum": "featured|brightness_q2",
            },
        ]

        groups, evidence = grouped.construct_components(records, ["source_id"], [])

        self.assertEqual(len(groups), 1)
        self.assertEqual({row["group_id"] for row in records}, {next(iter(groups))})
        self.assertEqual(
            {item.evidence_type for item in evidence},
            {"exact_coordinate", "exact_pixel_hash"},
        )

    def test_cross_role_group_leakage_fails_closed(self) -> None:
        rows = [
            {"_record_key": "a", "group_id": "group-1", "role": "train"},
            {"_record_key": "b", "group_id": "group-1", "role": "validation"},
        ]
        evidence = [grouped.EvidenceBucket("exact_pixel_hash", "hash", ("a", "b"))]
        group_checks, duplicate_checks = grouped.integrity_tables(rows, evidence)

        self.assertEqual(group_checks[0]["pass"], 0)
        self.assertEqual(duplicate_checks[0]["pass"], 0)
        with self.assertRaisesRegex(RuntimeError, "group leakage"):
            grouped.fail_closed_integrity(rows, group_checks, duplicate_checks)

    def test_balance_diagnostics_report_target_actual_deviation(self) -> None:
        rows = [
            {"role": "train", "balance_stratum": "a"},
            {"role": "train", "balance_stratum": "a"},
            {"role": "validation", "balance_stratum": "a"},
            {"role": "calibration", "balance_stratum": "b"},
            {"role": "development_test", "balance_stratum": "b"},
            {"role": "future_lockbox", "balance_stratum": "b"},
        ]
        fractions = {role: 0.2 for role in grouped.ROLES}

        diagnostics = grouped.balance_diagnostics(rows, fractions)

        self.assertEqual(
            diagnostics["source_role_balance"]["train"]["actual_source_count"],
            2,
        )
        self.assertGreater(
            diagnostics["max_absolute_source_fraction_deviation"], 0.0
        )


class NormalizationRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.array(
            [-100.0, -2.5, -0.01, 0.0, 0.02, 3.0, 250.0], dtype=np.float64
        )

    def test_all_global_transforms_round_trip_signed_flux(self) -> None:
        specs = [
            normalization.NormalizationSpec(
                "fixed_per_band_scale", "r", scale=17.0
            ),
            normalization.NormalizationSpec(
                "robust_signed_asinh",
                "r",
                softening=0.03,
                asinh_normalizer=4.2,
            ),
            normalization.NormalizationSpec(
                "global_percentile", "r", low=-1.5, high=20.0
            ),
        ]
        for spec in specs:
            with self.subTest(method=spec.method):
                transformed = normalization.transform_flux(self.values, spec)
                replayed = normalization.inverse_flux(transformed, spec)
                np.testing.assert_allclose(
                    replayed, self.values, rtol=2e-13, atol=2e-13
                )
                self.assertLess(transformed[0], transformed[3])
                self.assertLess(transformed[3], transformed[-1])

    def test_verified_ivar_round_trip_is_exact_on_positive_domain(self) -> None:
        ivar = np.array([0.25, 1.0, 4.0, 0.0, 2.0, 10.0, 0.5])
        spec = normalization.NormalizationSpec("variance_aware", "r")

        transformed = normalization.transform_flux(self.values, spec, ivar)
        replayed = normalization.inverse_flux(transformed, spec, ivar)

        valid = ivar > 0
        np.testing.assert_allclose(replayed[valid], self.values[valid], rtol=0, atol=0)
        self.assertTrue(np.isnan(transformed[~valid]).all())
        self.assertTrue(np.isnan(replayed[~valid]).all())

    def test_recommendation_gate_rejects_failed_roundtrip(self) -> None:
        samples = {
            band: np.linspace(-5.0, 10.0, 200, dtype=np.float64)
            for band in normalization.BANDS
        }
        specs = normalization.fit_specs(samples)
        rows = normalization.study_rows(specs, samples, {}, {}, "nanomaggies per pixel")
        normalization.validate_recommendation_gate(rows)
        rows[0]["roundtrip_pass"] = False

        with self.assertRaisesRegex(ValueError, "recommendation gate failed"):
            normalization.validate_recommendation_gate(rows)

    def test_sampling_token_uses_immutable_identity_not_path(self) -> None:
        token = normalization.immutable_sampling_token(
            "source-1", "group-1", "a" * 64, "g"
        )
        first, _ = normalization.deterministic_sample(self.values, 5, token)
        second, _ = normalization.deterministic_sample(self.values, 5, token)

        self.assertNotIn("/", token)
        np.testing.assert_array_equal(first, second)

    def test_train_quality_requires_frozen_request_and_file_hash_gates(self) -> None:
        path = str(Path("/tmp/train-source.fits").resolve())
        train = [
            {
                "_resolved_fits_path": path,
                "pixel_hash": "a" * 64,
                "preallocation_file_sha256": "b" * 64,
                "preallocation_pixel_hash": "a" * 64,
                "preallocation_hash_revalidation_pass": "1",
            }
        ]
        quality = [
            {
                "path": path,
                "frozen_fits_semantics_valid": "1",
                "manifest_request_semantics_valid": "1",
                "fits_structure_valid": "1",
                "band_order_valid": "1",
                "wcs_valid": "1",
                "pixel_hash": "a" * 64,
                "file_sha256": "b" * 64,
                "unit_source": "documented_official_product",
                "unit_value": "nanomaggies per pixel",
            }
        ]
        self.assertEqual(
            normalization.validate_train_quality(train, quality),
            "nanomaggies per pixel",
        )
        quality[0]["manifest_request_semantics_valid"] = "0"
        with self.assertRaisesRegex(ValueError, "manifest_request_semantics_valid"):
            normalization.validate_train_quality(train, quality)


class FitsAndIvarProvenanceTests(unittest.TestCase):
    @staticmethod
    def _wcs_header() -> fits.Header:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [3.0, 3.0]
        wcs.wcs.cdelt = np.array([-0.262 / 3600.0, 0.262 / 3600.0])
        wcs.wcs.crval = [190.0, 1.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs.to_header()

    def test_combined_image_invvar_hdus_are_selected_by_identity_and_hashed(self) -> None:
        image = np.arange(3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
        ivar = np.full_like(image, 7.0)
        image_header = self._wcs_header()
        image_header["IMAGETYP"] = "IMAGE"
        image_header["BANDS"] = "grz"
        ivar_header = self._wcs_header()
        ivar_header["IMAGETYP"] = "INVVAR"
        ivar_header["BANDS"] = "grz"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(image, header=image_header),
                    fits.ImageHDU(ivar, header=ivar_header),
                ]
            ).writeto(path)

            science = normalization.read_grz_cube(
                path, expected_image_type="IMAGE"
            )
            weights = normalization.read_grz_cube(
                path, expected_image_type="INVVAR"
            )

        self.assertEqual(science.hdu_index, 0)
        self.assertEqual(weights.hdu_index, 1)
        self.assertEqual(science.image_type, "IMAGE")
        self.assertEqual(weights.image_type, "INVVAR")
        np.testing.assert_array_equal(science.data, image)
        np.testing.assert_array_equal(weights.data, ivar)
        self.assertEqual(
            science.pixel_hash, normalization.semantic_pixel_hash(science.data)
        )
        self.assertEqual(
            weights.pixel_hash, normalization.semantic_pixel_hash(weights.data)
        )
        self.assertTrue(normalization.wcs_aligned(science, weights))
        normalization.verify_current_pixel_hash(science, science.pixel_hash, path)
        with self.assertRaisesRegex(ValueError, "differ from audited"):
            normalization.verify_current_pixel_hash(science, "0" * 64, path)

    def test_ivar_manifest_truth_flags_do_not_replace_required_hash_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            science_path = str((Path(directory) / "science.fits").resolve())
            manifest = Path(directory) / "ivar.csv"
            fields = [
                "science_fits_path",
                "ivar_fits_path",
                "semantics_verified",
                "wcs_alignment_verified",
                "science_flux_unit",
                "ivar_unit",
                "ivar_pixel_hash",
            ]
            row = {
                "science_fits_path": science_path,
                "ivar_fits_path": science_path,
                "semantics_verified": "1",
                "wcs_alignment_verified": "1",
                "science_flux_unit": "nanomaggies per pixel",
                "ivar_unit": "1/(nanomaggies)^2 per pixel",
                "ivar_pixel_hash": "",
            }
            with manifest.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "ivar_pixel_hash"):
                normalization.load_verified_ivar_index(
                    manifest,
                    allowed_science_paths={science_path},
                    science_flux_unit="nanomaggies per pixel",
                )

    def test_ivar_manifest_with_official_unit_and_hash_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            science_path = str((Path(directory) / "science.fits").resolve())
            ivar_path = str((Path(directory) / "invvar.fits").resolve())
            manifest = Path(directory) / "verified_ivar.csv"
            fields = [
                "science_fits_path",
                "ivar_fits_path",
                "semantics_verified",
                "wcs_alignment_verified",
                "science_flux_unit",
                "ivar_unit",
                "ivar_pixel_hash",
                "science_hdu",
                "ivar_hdu",
            ]
            row = {
                "science_fits_path": science_path,
                "ivar_fits_path": ivar_path,
                "semantics_verified": "1",
                "wcs_alignment_verified": "1",
                "science_flux_unit": "nanomaggies per pixel",
                "ivar_unit": "1/(nanomaggies)^2 per pixel",
                "ivar_pixel_hash": "a" * 64,
                "science_hdu": "0",
                "ivar_hdu": "1",
            }
            with manifest.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)

            index = normalization.load_verified_ivar_index(
                manifest,
                allowed_science_paths={science_path},
                science_flux_unit="nanomaggies per pixel",
            )

        self.assertEqual(index[science_path]["_ivar_pixel_hash"], "a" * 64)
        self.assertEqual(index[science_path]["_science_hdu"], 0)
        self.assertEqual(index[science_path]["_ivar_hdu"], 1)

    def test_conflicting_header_unit_fails_closed(self) -> None:
        cube = normalization.GrzCube(
            np.zeros((3, 2, 2)),
            fits.Header({"BUNIT": "electrons"}),
            WCS(naxis=2),
            0,
            "INVVAR",
            "a" * 64,
        )

        with self.assertRaisesRegex(ValueError, "BUNIT .* conflicts"):
            normalization.verify_header_unit(
                cube, "1/(nanomaggies)^2 per pixel", "INVVAR"
            )

    def test_full_grid_wcs_check_detects_off_diagonal_mismatch(self) -> None:
        class IdentityWcs:
            def pixel_to_world_values(self, x, y):
                return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

        class DiagonalOnlyMatchWcs:
            def pixel_to_world_values(self, x, y):
                x_array = np.asarray(x, dtype=float)
                y_array = np.asarray(y, dtype=float)
                return x_array + 1.0e-5 * (x_array - y_array), y_array

        data = np.zeros((3, 5, 5), dtype=np.float32)
        science = normalization.GrzCube(
            data, fits.Header(), IdentityWcs(), 0, "IMAGE", "a" * 64
        )
        ivar = normalization.GrzCube(
            data, fits.Header(), DiagonalOnlyMatchWcs(), 1, "INVVAR", "b" * 64
        )

        self.assertFalse(normalization.wcs_aligned(science, ivar))


if __name__ == "__main__":
    unittest.main()
