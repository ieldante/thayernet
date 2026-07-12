"""Frozen coordinate-query semantics for Thayer-Select Phase II.

The association routine is deliberately independent of model output and scene
difficulty.  It sees only the declared prompt coordinate, source centroids,
and frozen geometric thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class QueryClass(str, Enum):
    VALID_SOURCE = "VALID_SOURCE"
    PERTURBED_VALID = "PERTURBED_VALID"
    NULL_SOURCE = "NULL_SOURCE"
    AMBIGUOUS_SOURCE = "AMBIGUOUS_SOURCE"


@dataclass(frozen=True)
class PromptSemantics:
    """Frozen Phase-II matching geometry in image pixels."""

    version: str = "thayer-select-prompt-semantics-v1"
    matching_radius_pixels: float = 4.0
    psf_fwhm_pixels: float = 4.066666666666666
    ambiguity_margin_pixels: float = 1.0
    exact_source_radius_pixels: float = 0.5
    edge_policy: str = "finite in-frame prompts are valid; ownership is unchanged near edges"

    @property
    def matching_radius_psf(self) -> float:
        return self.matching_radius_pixels / self.psf_fwhm_pixels

    @property
    def ambiguity_margin_psf(self) -> float:
        return self.ambiguity_margin_pixels / self.psf_fwhm_pixels


@dataclass(frozen=True)
class Association:
    query_class: QueryClass
    matched_index: int | None
    nearest_distance_pixels: float
    second_distance_pixels: float
    coordinate_error_pixels: float
    candidate_count: int


def associate_prompt(
    source_xy: np.ndarray,
    prompt_xy: np.ndarray,
    *,
    image_shape: tuple[int, int],
    semantics: PromptSemantics | None = None,
) -> Association:
    """Associate a prompt without arbitrary tie breaking.

    A coordinate on the alternate real galaxy is simply a valid request for
    that galaxy.  Equal-distance and near-equal ownership are ambiguous.
    """

    policy = semantics or PromptSemantics()
    sources = np.asarray(source_xy, dtype=np.float64)
    prompt = np.asarray(prompt_xy, dtype=np.float64)
    if sources.ndim != 2 or sources.shape[1] != 2 or len(sources) < 1:
        raise ValueError("source_xy must have shape (N, 2) with N >= 1")
    if prompt.shape != (2,) or not np.isfinite(prompt).all() or not np.isfinite(sources).all():
        raise ValueError("source and prompt coordinates must be finite")
    height, width = image_shape
    if not (0.0 <= prompt[0] <= width - 1 and 0.0 <= prompt[1] <= height - 1):
        raise ValueError("prompt coordinate lies outside the image")
    distances = np.linalg.norm(sources - prompt[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    nearest = float(distances[order[0]])
    second = float(distances[order[1]]) if len(order) > 1 else float("inf")
    candidates = int(np.sum(distances <= policy.matching_radius_pixels))
    if candidates == 0:
        return Association(QueryClass.NULL_SOURCE, None, nearest, second, nearest, 0)
    if candidates > 1 and second - nearest <= policy.ambiguity_margin_pixels:
        return Association(QueryClass.AMBIGUOUS_SOURCE, None, nearest, second, nearest, candidates)
    matched = int(order[0])
    query_class = (
        QueryClass.VALID_SOURCE
        if nearest <= policy.exact_source_radius_pixels
        else QueryClass.PERTURBED_VALID
    )
    return Association(query_class, matched, nearest, second, nearest, candidates)
