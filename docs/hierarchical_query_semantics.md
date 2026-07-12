# Hierarchical query semantics

Status: preregistered before hierarchical scene generation or head training.

Thayer-Select uses exactly three primary query states. The rule sees only the
prompt coordinate, source centroids, image bounds, and fixed geometry. It does
not see reconstruction output, source difficulty, generator strata, or any
recoverability outcome.

## Frozen geometry

- Image geometry: 60 x 60 pixels at 0.2 arcsec/pixel.
- Mean PSF FWHM: 4.0666666667 pixels.
- Matching radius: 4.0 pixels, or 0.9836066 mean-PSF FWHM.
- Ambiguity margin: 1.0 pixel, or 0.2459016 mean-PSF FWHM.
- Maximum perturbed-valid displacement used by the generator: 3.5 pixels.
- Prompt centers must be finite and inside the closed pixel-center rectangle
  `[0, 59] x [0, 59]`. An outside prompt is invalid input, not a NULL query.
- Near an image edge, ownership follows the same distance rule; no edge-specific
  source ownership adjustment is made.

## Primary states

### UNIQUE_VALID

At least one source centroid is within 4.0 pixels, and the nearest source is
more than 1.0 pixel closer than the second-nearest source. The nearest source
is the uniquely matched request. Exact-center and perturbed-valid prompts share
this primary state; perturbation is retained only as analysis metadata.

A prompt landing uniquely on the alternate real source is a valid request for
that alternate source. It is never labeled a wrong query merely because a
generator initially chose the other source.

### NULL

No source centroid lies within 4.0 pixels. NULL has no matched source and no
valid-query reconstruction-risk targets.

### AMBIGUOUS

The nearest source is within the matching radius and the two nearest distances
differ by at most 1.0 pixel. This includes exact ties. A near tie remains
ambiguous when the second source lies just outside the hard radius, because an
infinitesimal boundary crossing cannot scientifically justify unique ownership.
No source is arbitrarily selected.

## Tie, perturbation, and source-count rules

- Stable sorting is used only for deterministic distance bookkeeping; it never
  breaks an ownership tie.
- Perturbations are rejection-sampled until they remain in-frame and
  UNIQUE_VALID for the intended source.
- NULL prompts are rejection-sampled until no source is within the matching
  radius.
- AMBIGUOUS prompts are generated near the two-source equidistance locus and
  verified by the same association function used for labels.
- For more than two sources, the nearest and second-nearest distances govern
  ambiguity; candidate count is preserved. Three-source scenes are excluded
  from the primary campaign unless support is validated before generation.

The executable contract is `src/hierarchical_safety.py`. Unit tests cover exact
requests, perturbed requests, alternate-source requests, NULL prompts, exact
and near ties, two-candidate clear ownership, image edges, and outside prompts.
