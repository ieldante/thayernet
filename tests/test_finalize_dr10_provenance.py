"""Focused, filesystem-free tests for the final DR10 provenance gate."""

from __future__ import annotations

import unittest

from scripts.finalize_dr10_provenance import (
    ProvenanceError,
    compare_checkpoint_inventories,
    parse_porcelain_v1_z,
    serialize_checkpoint_csv,
    validate_provenance_caveats,
)


def checkpoint_row(
    path: str = "outputs/checkpoints/model.pth",
    sha256: str = "a" * 64,
    size_bytes: int = 17,
    mtime_ns: int = 123456789,
) -> dict[str, object]:
    return {
        "path": path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
    }


class FinalizeProvenanceTests(unittest.TestCase):
    def test_parse_porcelain_preserves_spaces_and_rename_source(self) -> None:
        payload = (
            b"?? docs/new report.md\0"
            b" M tracked.py\0"
            b"R  scripts/new name.py\0scripts/old name.py\0"
        )
        self.assertEqual(
            parse_porcelain_v1_z(payload),
            [
                {"status": "??", "path": "docs/new report.md"},
                {"status": " M", "path": "tracked.py"},
                {
                    "status": "R ",
                    "path": "scripts/new name.py",
                    "original_path": "scripts/old name.py",
                },
            ],
        )

    def test_checkpoint_identity_generates_explicit_unchanged_row(self) -> None:
        before = checkpoint_row()
        result = compare_checkpoint_inventories([before], [dict(before)])
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["identity_unchanged"])
        self.assertEqual(result[0]["before_sha256"], "a" * 64)
        rendered = serialize_checkpoint_csv(result)
        self.assertIn("identity_unchanged", rendered.splitlines()[0])
        self.assertTrue(rendered.endswith("\n"))

    def test_checkpoint_comparison_fails_closed_on_every_integrity_axis(self) -> None:
        baseline = checkpoint_row()
        mutations = {
            "sha256": checkpoint_row(sha256="b" * 64),
            "size_bytes": checkpoint_row(size_bytes=18),
            "mtime_ns": checkpoint_row(mtime_ns=123456790),
        }
        for label, changed in mutations.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ProvenanceError, label):
                    compare_checkpoint_inventories([baseline], [changed])

    def test_checkpoint_comparison_rejects_path_set_or_duplicate_changes(self) -> None:
        baseline = checkpoint_row()
        extra = checkpoint_row(path="outputs/checkpoints/extra.pth")
        with self.assertRaisesRegex(ProvenanceError, "path set changed"):
            compare_checkpoint_inventories([baseline], [baseline, extra])
        with self.assertRaisesRegex(ProvenanceError, "duplicate checkpoint path"):
            compare_checkpoint_inventories([baseline, baseline], [baseline])

    def test_caveats_are_verbatim_and_empty_values_fail(self) -> None:
        caveat = "  Historical segment hash is unavailable; spacing is intentional.  "
        self.assertEqual(validate_provenance_caveats([caveat]), [caveat])
        for invalid in ("", "   ", "contains\x00nul"):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ProvenanceError):
                    validate_provenance_caveats([invalid])


if __name__ == "__main__":
    unittest.main()
