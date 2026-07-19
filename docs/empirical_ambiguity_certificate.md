# Empirical ambiguity certificate

An empirical witness exists for an observation and prompt when at least two
candidate full decompositions pass the frozen forward-consistency contract,
their requested-source layers differ beyond a frozen scientific limit, and
unit, clipping, translation, serialization, background, and trivial-rescaling
audits pass.

The primary scientific diameter is the maximum applicable normalized
component: image distance divided by 0.25, any-band relative flux distance
divided by 0.20, either color distance divided by 0.20 mag, or centroid distance
in mean-PSF units divided by 0.5. A diameter above 1 is scientifically
different.

## Current feasibility result

The frozen run is
`outputs/runs/thayer_competing_hypotheses_20260712_131111/`; preregistration
SHA-256 is
`692b4194da0486b8240fcda8227d36df9b1654187dd5c670d60c69b8c5fd5a4b`.
A model-independent search generated 30,000 approved training/validation
scenes, found 100 numerical near-collision candidates, and froze the first 25
after exact replay and five-page visual artifact review.

For each frozen pair, its two truth decompositions were tested against both
noisy observations. Both decompositions remained plausible and scientifically
divergent on 49 of 50 observations; every one of the 25 pairs had a witness on
at least one observation. The three available same-architecture-cluster model
candidates produced a witness on 18 of 50 observations.

## Claim boundary

The witness establishes only that the finite candidate family contains more
than one observation-consistent requested source under the frozen forward
model and tolerance. It can falsify uniqueness for a scene. It does not
enumerate all solutions, prove that a scene without a witness is unique, prove
formal identifiability, or establish model-agnostic transfer.

Historical development and lockbox scenes were not accessed.
