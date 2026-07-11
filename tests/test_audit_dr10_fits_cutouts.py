"""Focused, read-only tests for the DR10 FITS/source-isolation audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from scripts import audit_dr10_fits_cutouts as audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_FITS = (
    PROJECT_ROOT
    / "data/dr10_grz_cutouts/manual_smoke/ra190.1086_dec1.2005_grz_256.fits"
)


def write_synthetic_cutout(
    path: Path,
    *,
    size: int = 256,
    survey: str = "DECaLS",
    version: str = "DR10-south",
    include_band2: bool = True,
    pixel_scale: float = 0.262,
    data: np.ndarray | None = None,
) -> None:
    cube = (
        np.ones((3, size, size), dtype=np.float32)
        if data is None
        else np.asarray(data, dtype=np.float32)
    )
    hdu = fits.PrimaryHDU(cube)
    header = hdu.header
    header["BANDS"] = "grz"
    header["BAND0"] = "g"
    header["BAND1"] = "r"
    if include_band2:
        header["BAND2"] = "z"
    header["SURVEY"] = survey
    header["VERSION"] = version
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = -5.0
    header["CRPIX1"] = (size + 1) / 2.0
    header["CRPIX2"] = (size + 1) / 2.0
    degrees_per_pixel = pixel_scale / 3600.0
    header["CD1_1"] = -degrees_per_pixel
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = degrees_per_pixel
    hdu.writeto(path)


class PixelHashTests(unittest.TestCase):
    def test_hash_covers_band_order_dtype_and_values(self) -> None:
        array = np.arange(24, dtype=np.float32).reshape(3, 2, 4)

        self.assertEqual(audit.semantic_pixel_hash(array), audit.semantic_pixel_hash(array.copy()))
        self.assertNotEqual(
            audit.semantic_pixel_hash(array), audit.semantic_pixel_hash(array[::-1])
        )
        self.assertNotEqual(
            audit.semantic_pixel_hash(array), audit.semantic_pixel_hash(array.astype(np.float64))
        )


class DecisionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = audit.AuditSettings()
        self.quality = {
            "path": "/input/example.fits",
            "filename": "example.fits",
            "source_id": "source-1",
            "group_id": "group-1",
            "catalog_row_index": "1",
            "frozen_fits_semantics_valid": 1,
            "manifest_request_semantics_valid": 1,
            "fits_structure_valid": 1,
            "band_order_valid": 1,
            "wcs_valid": 1,
            "finite_fraction_all_bands": 1.0,
            "max_extreme_fraction": 0.0,
            "unit_source": "fits_header_BUNIT",
            "exact_duplicate_group_id": "",
            "audit_error": "",
        }
        self.isolation = {
            "blank_cutout": 0,
            "central_source_present": 1,
            "full_frame_object": 0,
            "central_mask_touches_border": 0,
            "central_centroid_offset_px": 0.5,
            "likely_preexisting_blend": 0,
            "nearest_neighbor_distance_px": 50.0,
            "neighbor_to_target_detection_flux_ratio": 0.01,
            "likely_stellar_candidate": 0,
            "any_source_mask_touches_border": 0,
        }

    def test_clean_source_is_accepted(self) -> None:
        decision = audit.decide_quality(self.quality, self.isolation, self.settings)
        self.assertEqual(decision["decision"], "accepted_clean_source")
        self.assertEqual(decision["all_reasons"], "all_fixed_rules_passed")

    def test_duplicate_is_reviewed_not_silently_merged(self) -> None:
        quality = {**self.quality, "exact_duplicate_group_id": "exact_pixels_000001"}
        decision = audit.decide_quality(quality, self.isolation, self.settings)
        self.assertEqual(decision["decision"], "manual_review")
        self.assertIn("exact_pixel_duplicate", decision["all_reasons"])

    def test_clear_close_blend_is_rejected(self) -> None:
        isolation = {
            **self.isolation,
            "likely_preexisting_blend": 1,
            "nearest_neighbor_distance_px": 8.0,
            "neighbor_to_target_detection_flux_ratio": 0.8,
        }
        decision = audit.decide_quality(self.quality, isolation, self.settings)
        self.assertEqual(decision["decision"], "rejected_for_source_library_use")
        self.assertIn("clear_preexisting_blend", decision["rejection_reasons"])

    def test_clear_non_nearest_neighbor_cannot_be_hidden(self) -> None:
        isolation = {
            **self.isolation,
            "likely_preexisting_blend": 1,
            "nearest_neighbor_distance_px": 4.0,
            "neighbor_to_target_detection_flux_ratio": 0.01,
            "clear_preexisting_blend": 1,
            "qualifying_clear_blend_neighbor_count": 1,
        }
        decision = audit.decide_quality(self.quality, isolation, self.settings)
        self.assertEqual(decision["decision"], "rejected_for_source_library_use")
        self.assertIn("clear_preexisting_blend", decision["rejection_reasons"])

    def test_peripheral_border_neighbor_is_diagnostic_not_review(self) -> None:
        isolation = {**self.isolation, "any_source_mask_touches_border": 1}
        decision = audit.decide_quality(self.quality, isolation, self.settings)
        self.assertEqual(decision["decision"], "accepted_clean_source")
        self.assertEqual(decision["all_reasons"], "all_fixed_rules_passed")


class ArtifactRowTests(unittest.TestCase):
    def test_artifact_row_retains_catalog_row_index(self) -> None:
        quality = {
            "path": "/input/example.fits",
            "filename": "example.fits",
            "source_id": "source-1",
            "group_id": "group-1",
            "catalog_row_index": "1234",
            "fits_structure_valid": 1,
            "frozen_fits_semantics_valid": 1,
            "manifest_request_semantics_valid": 1,
            "band_order_valid": 1,
            "wcs_valid": 1,
            "unit_source": "explicit_cli_documentation",
            "nan_inf_fraction_all_bands": 0.0,
            "max_extreme_fraction": 0.0,
            "exact_duplicate_group_id": "",
        }
        isolation = {
            "blank_cutout": 0,
            "full_frame_object": 0,
            "central_mask_touches_border": 1,
            "likely_preexisting_blend": 0,
            "likely_stellar_candidate": 0,
        }

        rows = audit.artifact_rows(
            [quality], {quality["path"]: isolation}, audit.AuditSettings()
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["catalog_row_index"], "1234")


class ManualSmokeFitsTests(unittest.TestCase):
    @unittest.skipUnless(SMOKE_FITS.is_file(), "manual smoke FITS is not installed")
    def test_smoke_fits_has_explicit_grz_wcs_and_expected_shape(self) -> None:
        payload = audit.read_fits_payload(SMOKE_FITS, ("g", "r", "z"))

        self.assertEqual(payload.cube_grz_order.shape, (3, 256, 256))
        self.assertEqual(payload.inferred_bands, ("g", "r", "z"))
        self.assertTrue(payload.wcs_valid)
        self.assertAlmostEqual(payload.pixel_scale_arcsec, 0.262, places=6)
        self.assertTrue(np.isfinite(payload.cube_grz_order).all())

    @unittest.skipUnless(SMOKE_FITS.is_file(), "manual smoke FITS is not installed")
    def test_visual_and_histogram_payloads_can_be_suppressed(self) -> None:
        quality, _bands, _isolation, visual, samples = audit.audit_one(
            SMOKE_FITS,
            {},
            audit.AuditSettings(),
            audit.OFFICIAL_IMAGE_UNIT,
            retain_visual=False,
            retain_histogram_samples=False,
        )

        self.assertEqual(quality["frozen_fits_semantics_valid"], 1)
        self.assertIsNone(visual.rgb)
        self.assertIsNone(visual.detection_image)
        self.assertTrue(all(sample.size == 0 for sample in samples))


class FrozenFitsSemanticsTests(unittest.TestCase):
    def test_wrong_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-size.fits"
            write_synthetic_cutout(path, size=128)

            with self.assertRaisesRegex(ValueError, "unexpected frozen cutout shape"):
                audit.read_fits_payload(path, ("g", "r", "z"))

    def test_wrong_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for survey, version in (("BASS", "DR10-south"), ("DECaLS", "DR9")):
                with self.subTest(survey=survey, version=version):
                    path = Path(directory) / f"{survey}-{version}.fits"
                    write_synthetic_cutout(path, survey=survey, version=version)
                    with self.assertRaisesRegex(ValueError, "frozen provenance mismatch"):
                        audit.read_fits_payload(path, ("g", "r", "z"))

    def test_wrong_pixel_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-scale.fits"
            write_synthetic_cutout(path, pixel_scale=0.30)

            with self.assertRaisesRegex(ValueError, "pixel scales"):
                audit.read_fits_payload(path, ("g", "r", "z"))

    def test_missing_indexed_band_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-band.fits"
            write_synthetic_cutout(path, include_band2=False)

            with self.assertRaisesRegex(ValueError, "frozen band metadata mismatch"):
                audit.read_fits_payload(path, ("g", "r", "z"))

    def test_per_band_finite_and_nonzero_gates_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nonfinite = np.ones((3, 256, 256), dtype=np.float32)
            nonfinite[0, :20, :] = np.nan
            nonfinite_path = Path(directory) / "nonfinite.fits"
            write_synthetic_cutout(nonfinite_path, data=nonfinite)
            with self.assertRaisesRegex(ValueError, "per-band finite fractions"):
                audit.read_fits_payload(nonfinite_path, ("g", "r", "z"))

            zero_band = np.ones((3, 256, 256), dtype=np.float32)
            zero_band[2] = 0.0
            zero_path = Path(directory) / "zero-band.fits"
            write_synthetic_cutout(zero_path, data=zero_band)
            with self.assertRaisesRegex(ValueError, "per-band nonzero fractions"):
                audit.read_fits_payload(zero_path, ("g", "r", "z"))


class ManifestPreflightTests(unittest.TestCase):
    def test_candidate_coverage_retains_terminal_rejection(self) -> None:
        sources = [
            {
                "source_id": "source-1",
                "catalog_row_index": "1",
                "provisional_group_id": "group-1",
                "ra": "10.0",
                "dec": "-5.0",
            },
            {
                "source_id": "source-2",
                "catalog_row_index": "2",
                "provisional_group_id": "group-2",
                "ra": "11.0",
                "dec": "-6.0",
            },
        ]
        outcomes = [
            {
                "run_id": "completed-run",
                "source_id": "source-1",
                "catalog_row_index": "1",
                "group_id": "group-1",
                "ra": "10.0",
                "dec": "-5.0",
                "status": "downloaded_valid",
            },
            {
                "run_id": "completed-run",
                "source_id": "source-2",
                "catalog_row_index": "2",
                "group_id": "group-2",
                "ra": "11.0",
                "dec": "-6.0",
                "status": "validation_rejected",
                "relative_path": "source-2.fits",
                "error": "blank or missing r band",
            },
        ]

        terminal, run_id = audit.validate_candidate_download_coverage(sources, outcomes)
        decision = audit.terminal_rejection_decision(terminal[0], Path("/tmp"))

        self.assertEqual(run_id, "completed-run")
        self.assertEqual(len(terminal), 1)
        self.assertEqual(decision["decision"], "rejected_for_source_library_use")
        self.assertEqual(
            decision["rejection_reasons"],
            "download_terminal_validation_rejected",
        )
        self.assertEqual(decision["download_error"], "blank or missing r band")

    def test_incomplete_or_mixed_latest_runs_fail_closed(self) -> None:
        source = {
            "source_id": "source-1",
            "catalog_row_index": "1",
            "provisional_group_id": "group-1",
            "ra": "10.0",
            "dec": "-5.0",
        }
        cancelled = {
            "run_id": "run-a",
            "source_id": "source-1",
            "catalog_row_index": "1",
            "group_id": "group-1",
            "ra": "10.0",
            "dec": "-5.0",
            "status": "cancelled",
            "error": "operator stop",
        }
        with self.assertRaisesRegex(ValueError, "incomplete downloader outcome"):
            audit.validate_candidate_download_coverage([source], [cancelled])
        exhausted_transient = {**cancelled, "status": "failed"}
        with self.assertRaisesRegex(ValueError, "incomplete downloader outcome"):
            audit.validate_candidate_download_coverage(
                [source], [exhausted_transient]
            )

        source_two = {
            **source,
            "source_id": "source-2",
            "catalog_row_index": "2",
            "provisional_group_id": "group-2",
        }
        successful = {**cancelled, "status": "downloaded_valid", "error": ""}
        mixed = {
            **successful,
            "run_id": "run-b",
            "source_id": "source-2",
            "catalog_row_index": "2",
            "group_id": "group-2",
        }
        with self.assertRaisesRegex(ValueError, "one completed downloader invocation"):
            audit.validate_candidate_download_coverage(
                [source, source_two], [successful, mixed]
            )

    def test_unmatched_fits_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unmatched.fits"
            path.write_bytes(b"SIMPLE")

            with self.assertRaisesRegex(ValueError, "no download-manifest row"):
                audit.validate_manifest_alignment(
                    [path],
                    {},
                    {},
                    [],
                    manifest_parent=Path(directory),
                    require_all_successful_rows=False,
                )

    def test_request_parameters_must_match_frozen_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = (Path(directory) / "example.fits").resolve()
            row = {
                "source_id": "source-1",
                "catalog_row_index": "1",
                "group_id": "group-1",
                "ra": "10.0",
                "dec": "-5.0",
                "fits_shape": "3x128x128",
                "bands_header": "grz",
                "request_parameters_json": json.dumps(
                    {
                        "ra": "10.0",
                        "dec": "-5.0",
                        "layer": "ls-dr10-south",
                        "bands": "grz",
                        "pixscale": "0.262",
                        "size": 128,
                    }
                ),
            }

            with self.assertRaisesRegex(ValueError, "request size mismatch"):
                audit.validate_manifest_request_semantics(
                    [path], {str(path): row}, {}, audit.AuditSettings()
                )


class BoundedRetentionTests(unittest.TestCase):
    def test_2000_file_retention_is_hard_bounded_and_deterministic(self) -> None:
        settings = audit.AuditSettings()
        first_visual, first_histogram = audit.retention_plan(2_000, settings)
        second_visual, second_histogram = audit.retention_plan(2_000, settings)

        self.assertEqual(first_visual, second_visual)
        self.assertEqual(first_histogram, second_histogram)
        self.assertEqual(len(first_visual), settings.max_contact_sheet_files)
        self.assertLessEqual(
            len(first_histogram) * settings.histogram_samples_per_file,
            settings.histogram_max_samples_per_band,
        )
        self.assertTrue(first_visual.issubset(set(range(2_000))))
        self.assertTrue(first_histogram.issubset(set(range(2_000))))


class OutputPreflightTests(unittest.TestCase):
    def test_output_preflight_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "not-created"
            planned = audit.preflight_output_paths(run_dir)

            self.assertFalse(run_dir.exists())
            self.assertTrue(all(not path.exists() for path in planned.values()))


class UnitProvenanceTests(unittest.TestCase):
    def test_production_unit_must_be_explicit_and_exact(self) -> None:
        settings = audit.AuditSettings()
        self.assertEqual(
            audit.require_documented_unit("  nanomaggies   per pixel  ", settings),
            audit.OFFICIAL_IMAGE_UNIT,
        )
        with self.assertRaisesRegex(ValueError, "requires explicit documented unit"):
            audit.require_documented_unit("", settings)


if __name__ == "__main__":
    unittest.main()
