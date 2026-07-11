"""Tests for append-only correction of the DR10 artifact table."""

from __future__ import annotations

import unittest

from scripts.create_dr10_audit_correction import corrected_rows


class ArtifactCorrectionTests(unittest.TestCase):
    def test_backfill_requires_exact_identity_match(self) -> None:
        artifact = {
            "path": "/data/a.fits",
            "filename": "a.fits",
            "source_id": "s1",
            "group_id": "g1",
            "catalog_row_index": "",
        }
        quality = {**artifact, "catalog_row_index": "42"}

        rows = corrected_rows([artifact], [quality])

        self.assertEqual(rows[0]["catalog_row_index"], "42")

    def test_nonblank_original_is_refused(self) -> None:
        artifact = {
            "path": "/data/a.fits",
            "filename": "a.fits",
            "source_id": "s1",
            "group_id": "g1",
            "catalog_row_index": "already-present",
        }
        quality = {**artifact, "catalog_row_index": "42"}

        with self.assertRaisesRegex(ValueError, "nonblank original"):
            corrected_rows([artifact], [quality])


if __name__ == "__main__":
    unittest.main()
