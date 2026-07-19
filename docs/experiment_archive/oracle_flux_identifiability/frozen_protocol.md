# Thayer-Identifiability-v1 prior and analysis freeze

Status: **FROZEN BEFORE NUMERICAL IDENTIFIABILITY EXECUTION**

Campaign: Prompt-Centered Sérsic / Bulge-Disk Identifiability Audit.

This is a training-free, append-only audit of the eight already frozen
Family-E1 training observations with Family-E1 indices
`[0, 3, 5, 6, 18, 51, 73, 81]`.  Index 6 is not duplicated when results are
counted.  Development, Atlas, and lockbox paths and arrays are prohibited.
Family-E1, D0/D1/D2/D3, prompts, scientific thresholds, README, historical
artifacts, checkpoints, and the Git index are read-only.  No neural model is
loaded and no network weight is optimized.

## Common inverse problem

For the frozen source images `A,B`, observed image `O`, and fixed exact signed
noise realization `R0 = O-A-B`, all levels analyze decompositions satisfying

`S_A + S_B + R0 = O`.

Thus the data target is the exact noiseless source sum `T=A+B`; noise is not
fitted.  The two prompt centers are the frozen exact source-coordinate prompts
from Family-E1P and are never optimized.  The requested source is the component
assigned to the requested prompt in the frozen manifest.

The only scientific distinction threshold is the inherited primary scientific
diameter gate `> 1.0`.  Its image, per-band flux, color, and centroid scales are
unchanged.  A storage-aware numerical equality tolerance is not a scientific
threshold: a parametric render is called exact when
`||render-T||_2/||T||_2 <= 8*eps(float32) = 9.5367431640625e-7`.

For nonnegative priors, diameter is the maximum inherited primary scientific
distance between requested-source images in the certified admissible set.
When only a constructive subfamily is optimized, the reported value is marked
as a certified lower bound.  Level 0 is unbounded in source-pair Euclidean
norm and is reported as infinite.  Empty admissible sets have undefined
diameter and may not be called unique.

## Frozen prior ladder

The levels are cumulative through Level 3.  Levels 4 and 5 retain Levels 1--3
and the Level-2 flux information but are alternative increasingly informative
structural families rather than mathematically nested supports.  Levels 6 and
7 refine Level 5.

### Level 0 — no prior

`S_A,S_B` are unrestricted real `3x60x60` arrays.  This is the completed
direct-output result with observation Jacobian `[I I]`.

### Level 1 — nonnegative flux

Require `S_A >= 0` and `S_B >= 0` elementwise.  No flux, color, shape, or
centroid is supplied.  The complete exact set is `S_A=X, S_B=T-X` for
`0 <= X <= T`.

### Level 2 — flux conservation

In addition to Level 1, fix each component's three stamp-integrated band
fluxes to the frozen isolated-source values.  This is deliberately an
optimistic oracle/external-photometry upper bound; exact component fluxes are
not claimed to be observation-only information.  Reporting this level makes
clear whether even perfect external photometry is sufficient.

### Level 3 — smoothness

In addition to Level 2, each component and band must obey the fixed normalized
anisotropic total-variation bound

`(sum |horizontal differences| + sum |vertical differences|) / band_flux <= 1 pixel^-1`.

This bound is fixed globally, not from any audited scene.  It excludes
pixel-scale spikes while retaining broad PSF-convolved galaxy profiles.  It is
a hard support condition, not a smoothing optimizer or MAP tie-breaker.

### Level 4 — prompt-centered elliptical Sérsic

Each component is one elliptical GalSim Sérsic profile, convolved with the
known frozen per-band LSST PSF and centered exactly on its assigned prompt.
Per-source g/r/z stamp fluxes remain fixed by Level 2 and one geometry is
shared across bands.  Free physical parameters per active source are Sérsic
index, half-light radius, axis ratio, and position angle, with global bounds
`0.5 <= n <= 6`, `0.03 <= HLR <= 3 arcsec`, `0.1 <= q <= 1`, and position
angle modulo pi.  A frozen truth outside this family yields zero exact
solutions; misspecification is not evidence of uniqueness.

### Level 5 — prompt-centered bulge + disk

Each source is a PSF-convolved sum of an `n=4` de Vaucouleurs bulge and an
`n=1` exponential disk at the assigned prompt.  Component HLR and q use the
Level-4 bounds; bulge and disk position angles are free modulo pi.  Geometry is
shared across bands.  Total source flux in each band is fixed, while the three
band-specific bulge fractions are free in `[0,1]`, allowing a physically
realistic bulge/disk color gradient.  Zero-flux component shape parameters are
output gauges and are quotiented out when rank and solution counts are
reported.

### Level 6 — shared color profile

Refine Level 5 by requiring one bulge fraction shared by g/r/z.  Equivalently,
the source has no internal bulge/disk color gradient after accounting for the
known band PSFs.  The two galaxies may have different total colors.  No color
value is learned or estimated from another split; the Level-2 per-source flux
vectors remain the sole color information.

### Level 7 — weak astrophysical morphology prior

Apply a continuous, strictly positive weak morphology density over the entire
Level-6 physical support, favoring ordinary finite sizes, non-extreme axis
ratios, and non-boundary bulge fractions but assigning zero probability to no
Level-6-admissible morphology.  This is intentionally weak and introduces no
new hard support, template, catalog lookup, learned prior, or scene-specific
bound.  Because identifiability concerns the exact support rather than a MAP
choice, Level 7 must have the same structural rank, exact-solution count, and
diameter as Level 6.  A soft preference alone is never credited with creating
data identifiability.

## Frozen diagnostics

For Levels 0--3, rank and nullity are those of the observation Jacobian
restricted to the equality-constrained tangent space.  Inequality constraints
that are locally inactive do not reduce tangent dimension.

For Levels 4--7, numerical profile derivatives are rendered with GalSim 2.8.4
and the BTK 1.0.9 LSST PSFs.  Zero-output parameter gauges are removed.  Each
source derivative space is orthonormalized, and the reported condition number
is that of the parameterization-invariant output-tangent observation map
`[Q_A Q_B]`.  Rank uses the standard float64 SVD tolerance
`max(m,n)*eps(float64)*s_max`.  If a prior has no exact solution, rank and
condition are evaluated at its deterministic closest projection and explicitly
marked as projection diagnostics.

Parametric exact-solution searches use 16 deterministic truth-free starts per
scene and family, seed `2026071507`, bounded least squares, at most 400 function
evaluations per start, and no generator-truth initialization.  Frozen catalog
parameters may be used only after fitting to verify model membership and to
compute an independent Jacobian replay.  Duplicate fitted outputs are merged
at the numerical exactness tolerance.  Analytic component identifiability and
the multi-start search are both required before reporting one global output
solution.

Level-2 diameter uses deterministic linear-program vertices under positivity,
band-flux, exact-sum, and prompt-half-plane constraints.  Level-3 diameter uses
the complete two-way convex normalized-morphology exchange path; that path
preserves the exact sum, both sources' band fluxes, nonnegativity, and the TV
bound.  Prompt consistency is checked at every reported endpoint.

## Frozen classifications

- `UNIDENTIFIABLE`: the truth-containing admissible set is empty, or it has an
  exact prompt-consistent diameter `>1`, or prompt/requested identity is not
  unique.
- `PARTIALLY IDENTIFIABLE`: more than one exact solution remains but every
  certified diameter is `<=1`, or only a strict subset of requested-source
  functionals is unique.
- `NEAR UNIQUE`: exactly one truth-containing output solution and zero exact
  nullity, but `condition_number * numerical_exactness_tolerance >= 1`.
- `UNIQUE`: exactly one truth-containing output solution, zero exact nullity,
  zero exact diameter, unique prompt identity, and
  `condition_number * numerical_exactness_tolerance < 1`.

The requested minimum prior is the first level classified `UNIQUE`, not the
first level producing a MAP estimate or an empty support.

## Required output and decision

The authoritative table has exactly the requested columns: Scene, Prior,
Rank, Null space, Diameter, Classification.  Supplementary machine-readable
columns record condition number, exact-solution count, prompt uniqueness,
requested-source identifiability, model membership, exact residual, and
diameter-bound status.  The final report answers whether realistic hard galaxy
structure removes exact ambiguity and separately whether poor conditioning
leaves a practical information frontier.  It recommends exactly one next
experiment and performs none beyond this frozen audit.
