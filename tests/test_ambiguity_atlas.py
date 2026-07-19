from __future__ import annotations

import numpy as np
import pytest

from scripts.build_ambiguity_atlas import downsample, nontrivial_rescaling


def test_atlas_embedding_downsample_is_fixed_block_mean() -> None:
    image = np.arange(3 * 60 * 60, dtype=float).reshape(3, 60, 60)
    reduced = downsample(image)
    assert reduced.shape == (3, 15, 15)
    assert reduced[0, 0, 0] == pytest.approx(image[0, :4, :4].mean())


def test_global_rescaling_is_rejected() -> None:
    right = np.arange(1, 101, dtype=float).reshape(10, 10)
    passed, residual = nontrivial_rescaling(2.5 * right, right)
    assert not passed
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_non_rescaling_difference_passes() -> None:
    right = np.ones((10, 10), dtype=float)
    left = right.copy()
    left[:5] = 3.0
    passed, residual = nontrivial_rescaling(left, right)
    assert passed
    assert residual > 0.01
