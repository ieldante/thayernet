from __future__ import annotations

import io
import unittest

import numpy as np
import torch

from src.canonical_tensor_hash import SCHEMA_VERSION, canonical_chw_float32, canonical_tensor_sha256


class CanonicalTensorHashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = np.arange(3 * 7 * 9, dtype=np.float64).reshape(3, 7, 9) / 11.0
        self.reference = canonical_tensor_sha256(self.sample)

    def test_batch_size_and_position_invariant(self) -> None:
        batch_one = self.sample[None]
        batch_many = np.stack((self.sample + 8, self.sample, self.sample - 3))
        self.assertEqual(self.reference, canonical_tensor_sha256(batch_one, layout="NCHW", sample_index=0))
        self.assertEqual(self.reference, canonical_tensor_sha256(batch_many, layout="NCHW", sample_index=1))

    def test_storage_clone_and_channel_last_invariant(self) -> None:
        padded = np.zeros((3, 7, 18), dtype=np.float64)
        padded[:, :, ::2] = self.sample
        noncontiguous = padded[:, :, ::2]
        self.assertFalse(noncontiguous.flags.c_contiguous)
        self.assertEqual(self.reference, canonical_tensor_sha256(noncontiguous))
        self.assertEqual(self.reference, canonical_tensor_sha256(self.sample.copy()))
        self.assertEqual(self.reference, canonical_tensor_sha256(np.moveaxis(self.sample, 0, -1), layout="HWC"))

    def test_serialization_reload_invariant(self) -> None:
        payload = io.BytesIO()
        np.save(payload, self.sample, allow_pickle=False)
        payload.seek(0)
        reloaded = np.load(payload, allow_pickle=False)
        self.assertEqual(self.reference, canonical_tensor_sha256(reloaded))

    def test_mps_cpu_transfer_invariant(self) -> None:
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        cpu = torch.from_numpy(self.sample.astype(np.float32))
        mps = cpu.to("mps")
        self.assertEqual(self.reference, canonical_tensor_sha256(cpu))
        self.assertEqual(canonical_tensor_sha256(cpu), canonical_tensor_sha256(mps))

    def test_one_pixel_channel_shape_and_value_sensitivity(self) -> None:
        changed = self.sample.copy()
        changed[1, 2, 3] += 0.25
        self.assertNotEqual(self.reference, canonical_tensor_sha256(changed))
        self.assertNotEqual(self.reference, canonical_tensor_sha256(self.sample[[1, 0, 2]]))
        self.assertNotEqual(self.reference, canonical_tensor_sha256(self.sample[:, :, :-1]))
        float64_changed = self.sample.copy()
        float64_changed[0, 0, 0] = np.float64(0.125)
        self.assertNotEqual(self.reference, canonical_tensor_sha256(float64_changed))

    def test_canonical_metadata_and_validation(self) -> None:
        value = canonical_chw_float32(self.sample)
        self.assertEqual(value.dtype, np.dtype("float32"))
        self.assertEqual(value.dtype.byteorder, "=")  # little-endian host representation of <f4
        self.assertTrue(value.flags.c_contiguous)
        with self.assertRaises(ValueError):
            canonical_tensor_sha256(self.sample[None], layout="NCHW")
        with self.assertRaises(ValueError):
            canonical_tensor_sha256(self.sample, schema_version=SCHEMA_VERSION + "-changed")


if __name__ == "__main__":
    unittest.main()
