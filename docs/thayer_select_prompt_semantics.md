# Thayer-Select prompt semantics

Version: `thayer-select-prompt-semantics-v1`, frozen before Phase II scene
generation. The mean `g,r,z` BTK PSF FWHM is 4.0667 pixels. The source-matching
radius is 4.0 pixels (0.984 mean-PSF FWHM), the ambiguity margin is 1.0 pixel
(0.246 mean-PSF FWHM), and an exact-source request lies within 0.5 pixel.

- `VALID_SOURCE`: exactly one source owns the coordinate and its distance is at
  most 0.5 pixel. That isolated source is the reconstruction target.
- `PERTURBED_VALID`: exactly one source owns the coordinate within 4.0 pixels,
  but the offset exceeds 0.5 pixel. The uniquely matched isolated source remains
  the target and the coordinate error is recorded.
- `NULL_SOURCE`: no source centroid is within 4.0 pixels. The exact target is a
  zero-flux image; desired behavior is no-source recognition and abstention.
- `AMBIGUOUS_SOURCE`: at least two sources lie within the matching radius and
  the two nearest distances differ by at most 1.0 pixel. No source is chosen by
  row order, source ID, or stable sort; no reconstruction target is assigned.

Association computes every Euclidean centroid distance, finds all candidates
inside the inclusive matching radius, and applies the ambiguity test before
unique nearest-source ownership. Exact equal-distance prompts are ambiguous.
A coordinate on the other real galaxy is a valid request for that galaxy, not
a wrong prompt. A finite in-frame coordinate near an edge follows the same
rules. Out-of-frame and nonfinite prompts are invalid rather than clipped.
Coordinates between sources are null when neither source owns them and
ambiguous when multiple sources are similarly compatible.

Every manifest stores query class, prompt coordinate, matching distances,
candidate count, matched index/ID/group when unique, null and ambiguity flags,
coordinate error, target-defined flag, and `ZERO_TARGET`, `ISOLATED_SOURCE`, or
`NO_ARBITRARY_TARGET`. Boundary, alternate-source, equal-distance, edge, radius,
and out-of-frame cases are unit-tested in `tests/test_recoverability_phase2.py`.
