# Thayer-Identifiability-v1 scientific report

## Decision

Realistic hard morphology removes the unrestricted continuous allocation null
space for **7/8 scenes**.  Four scenes first become unique at Level 4
(single elliptical Sersic), three first become unique at Level 5 (bulge+disk),
and scene 51 has no truth-containing exact solution through Level 7.  Thus the
recoverability frontier is largely prior-limited, but it does not disappear
completely under the frozen realistic prior ladder.

The remainder is not evidence for a surviving continuous observation null in
the parametric families.  Every truth-containing structural solution has null
space zero and output-tangent condition between
`1.00639` and `4.76695`; the largest
condition times the frozen numerical tolerance is only
`4.546e-06`.  Scene 51 instead
fails model support: its nonzero tiny bulge has HLR about 0.015 arcsec, below
the globally frozen 0.03-arcsec Level-4/5 bound.  An empty admissible set is
not credited as uniqueness.

## Requested table

Diameter values prefixed by `>=` are certified prompt-consistent lower bounds;
`--` denotes an empty exact support, not zero diameter.

| Scene | Prior | Rank | Null space | Diameter | Classification |
| --- | --- | --- | --- | --- | --- |
| 0 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 0 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 0 | L2 Flux conservation | 10797 | 10797 | >=1.81739 | UNIDENTIFIABLE |
| 0 | L3 Smoothness | 10797 | 10797 | >=0.197491 | UNIDENTIFIABLE |
| 0 | L4 Elliptical Sersic | 8 | 0 | 0 | UNIQUE |
| 0 | L5 Bulge + disk | 12 | 0 | 0 | UNIQUE |
| 0 | L6 Shared color profile | 8 | 0 | 0 | UNIQUE |
| 0 | L7 Weak astrophysical morphology prior | 8 | 0 | 0 | UNIQUE |
| 3 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 3 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 3 | L2 Flux conservation | 10797 | 10797 | >=8.74189 | UNIDENTIFIABLE |
| 3 | L3 Smoothness | 10797 | 10797 | >=3.85874 | UNIDENTIFIABLE |
| 3 | L4 Elliptical Sersic | 8 | 0 | 0 | UNIQUE |
| 3 | L5 Bulge + disk | 12 | 0 | 0 | UNIQUE |
| 3 | L6 Shared color profile | 8 | 0 | 0 | UNIQUE |
| 3 | L7 Weak astrophysical morphology prior | 8 | 0 | 0 | UNIQUE |
| 5 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 5 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 5 | L2 Flux conservation | 10797 | 10797 | >=5.40968 | UNIDENTIFIABLE |
| 5 | L3 Smoothness | 10797 | 10797 | >=4.81931 | UNIDENTIFIABLE |
| 5 | L4 Elliptical Sersic | 8 | 0 | 0 | UNIQUE |
| 5 | L5 Bulge + disk | 12 | 0 | 0 | UNIQUE |
| 5 | L6 Shared color profile | 8 | 0 | 0 | UNIQUE |
| 5 | L7 Weak astrophysical morphology prior | 8 | 0 | 0 | UNIQUE |
| 6 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 6 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 6 | L2 Flux conservation | 10797 | 10797 | >=5.07051 | UNIDENTIFIABLE |
| 6 | L3 Smoothness | 10797 | 10797 | >=0.736835 | UNIDENTIFIABLE |
| 6 | L4 Elliptical Sersic | 8 | 0 | -- | UNIDENTIFIABLE |
| 6 | L5 Bulge + disk | 18 | 0 | 0 | UNIQUE |
| 6 | L6 Shared color profile | 14 | 0 | 0 | UNIQUE |
| 6 | L7 Weak astrophysical morphology prior | 14 | 0 | 0 | UNIQUE |
| 18 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 18 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 18 | L2 Flux conservation | 10797 | 10797 | >=2.70922 | UNIDENTIFIABLE |
| 18 | L3 Smoothness | 10797 | 10797 | >=0.307727 | UNIDENTIFIABLE |
| 18 | L4 Elliptical Sersic | 8 | 0 | -- | UNIDENTIFIABLE |
| 18 | L5 Bulge + disk | 18 | 0 | 0 | UNIQUE |
| 18 | L6 Shared color profile | 14 | 0 | 0 | UNIQUE |
| 18 | L7 Weak astrophysical morphology prior | 14 | 0 | 0 | UNIQUE |
| 51 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 51 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 51 | L2 Flux conservation | 10797 | 10797 | >=5.12575 | UNIDENTIFIABLE |
| 51 | L3 Smoothness | 10797 | 10797 | >=3.14211 | UNIDENTIFIABLE |
| 51 | L4 Elliptical Sersic | 8 | 0 | -- | UNIDENTIFIABLE |
| 51 | L5 Bulge + disk | 17 | 0 | -- | UNIDENTIFIABLE |
| 51 | L6 Shared color profile | 14 | 0 | -- | UNIDENTIFIABLE |
| 51 | L7 Weak astrophysical morphology prior | 14 | 0 | -- | UNIDENTIFIABLE |
| 73 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 73 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 73 | L2 Flux conservation | 10797 | 10797 | >=4.37389 | UNIDENTIFIABLE |
| 73 | L3 Smoothness | 10797 | 10797 | >=1.62954 | UNIDENTIFIABLE |
| 73 | L4 Elliptical Sersic | 8 | 0 | 0 | UNIQUE |
| 73 | L5 Bulge + disk | 12 | 0 | 0 | UNIQUE |
| 73 | L6 Shared color profile | 8 | 0 | 0 | UNIQUE |
| 73 | L7 Weak astrophysical morphology prior | 8 | 0 | 0 | UNIQUE |
| 81 | L0 No prior | 10800 | 10800 | infinity | UNIDENTIFIABLE |
| 81 | L1 Nonnegative flux | 10800 | 10800 | 10 | UNIDENTIFIABLE |
| 81 | L2 Flux conservation | 10797 | 10797 | >=5.62184 | UNIDENTIFIABLE |
| 81 | L3 Smoothness | 10797 | 10797 | >=5.41633 | UNIDENTIFIABLE |
| 81 | L4 Elliptical Sersic | 8 | 0 | -- | UNIDENTIFIABLE |
| 81 | L5 Bulge + disk | 18 | 0 | 0 | UNIQUE |
| 81 | L6 Shared color profile | 14 | 0 | 0 | UNIQUE |
| 81 | L7 Weak astrophysical morphology prior | 14 | 0 | 0 | UNIQUE |

## Minimum prior required for uniqueness

- Scene 0: Level 4
- Scene 3: Level 4
- Scene 5: Level 4
- Scene 6: Level 5
- Scene 18: Level 5
- Scene 51: none through Level 7
- Scene 73: Level 4
- Scene 81: Level 5

## Quantitative interpretation

- Levels 0 and 1 retain rank/null `10800/10800` in every scene and
  uncountably many exact solutions.  Nonnegativity alone has exact primary
  diameter 10.
- Exact per-source g/r/z fluxes reduce the tangent to rank/null
  `10797/10797`, but still leave uncountably many solutions.  The certified
  Level-2 diameter lower bounds span `1.81739`--`8.74189`.
- The hard TV smoothness prior leaves the same rank/null and an exact
  continuum.  Certified Level-3 diameter lower bounds span
  `0.197491`--`5.41633`; five of eight exceed the inherited
  scientific gate directly, while the remaining three still retain a
  nonunique prompt-conditioned continuum.
- All accepted structural singleton fits have full restricted rank, null zero,
  one exact observation-consistent output, diameter zero, unique prompt
  identity, and exact residuals from `5.280e-08` to
  `9.525e-07` against the fixed
  `9.537e-07` tolerance.
- Level 7 is a strictly positive soft density over Level-6 support.  As frozen,
  it changes posterior preference but cannot change structural rank, exact
  solution count, or diameter; its results therefore equal Level 6.

## Scientific answer

The prior-free Family-E1 frontier is **not universally fundamental**: explicit
prompt-centered galaxy structure collapses 10,797--10,800 dimensional exact
allocation null spaces to zero for seven scenes, with modest condition numbers.
However, the frontier **does not fully disappear under the allowed priors**:
scene 51 remains unidentifiable because every allowed structural family is
exactly misspecified at the frozen support boundary.  The evidence therefore
supports a mixed conclusion—mostly missing inductive structure, plus one
PSF-resolution/model-support limit—not a universal observation-information
limit and not universal recoverability.

## Exactly one recommended next experiment

Run **Thayer-Identifiability-PSF-Core-v1**: one training-free, preregistered
repeat on the same eight scenes that adds a single PSF-unresolved central
component branch to the Level-5 forward model, with all other priors, prompts,
noise convention, exactness tolerance, and scientific diameter gate frozen.
Its sole purpose is to determine whether scene 51's 0.015-arcsec component is
identifiable as unresolved flux or remains observation-limited.

## Integrity

The prior freeze SHA-256 is `5af9db12575fe8f7025149cf55685456a22b30e0e19b8d9d7738d65683cb3475`.  Eight authorized training
scene rows and sixteen corresponding training catalog rows were accessed.
Development, Atlas, lockbox, neural-model imports, and network-weight optimizer
steps were all zero.  README, Family-E1, D0/D1/D2/D3, thresholds, prompts,
historical checkpoints, and the staged index were unchanged.
