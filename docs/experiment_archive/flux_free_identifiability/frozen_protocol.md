# Thayer-Flux-Free-Identifiability-v0 frozen protocol draft

Status: **FROZEN BY MODEL-9 PREPARATION BEFORE SCIENTIFIC SCENE ACCESS**

This is the single primary protocol for the next campaign. The preparation
campaign did not load or fit any of the eight frozen observations. The next
campaign must copy this file unchanged into its append-only run and verify its
SHA-256 before accessing a scientific observation.

## Scientific question and primary endpoint

The primary question is whether the requested and companion sources are
uniquely identifiable from the blended g/r/z observation under the frozen
Level-4 or Level-5 morphology support when individual source fluxes are free.
The primary endpoint is the number of the eight already frozen scenes that are
`UNIQUE` under at least one permitted family. Level-4, Level-5, and union counts
must be reported separately. The previous 7/8 result is conditional on oracle
per-source fluxes and is only a historical comparator.

## Data and access contract

Use exactly the eight observations and task identities already frozen by
Thayer-Identifiability-v1. Do not add, remove, replace, duplicate, or visually
select scenes. Scene 51 remains in scope. Before execution, record the scene
identifiers and observation hashes from historical metadata, then verify the
loaded observation hashes. Development, Atlas arrays, lockbox, fresh final
test data, and unrelated examples remain prohibited.

Permitted solver inputs are only:

- blended g/r/z observation;
- requested prompt coordinate;
- companion coordinate from the existing two-source task contract;
- known g/r/z PSFs;
- image dimensions and the fixed 0.2 arcsec/pixel scale;
- declared g/r/z noise sigma or sigma map from the frozen noise convention;
- Level-4 or Level-5 family identifier and the bounds below;
- observation/noise-derived flux scales used only for initialization and
  dimensionless diagnostics.

Prohibited inputs include isolated source images or masks, isolated-source
hash values as numerical features, true source fluxes, true source parameters,
catalog morphology, morphology labels, true B/T, truth initialization,
outcome-dependent family selection, and protected-split information. Every
solver invocation must persist `input_provenance_trace`; an unrecognized or
oracle-named input makes the scene/family `INVALID_CONTRACT`.

## Forward model

All calculations use float64 on CPU for optimization and local diagnostics.
The renderer in `src/model9_structured.py` evaluates pixel-integrated analytic
Sérsic densities with fixed 4x4 subpixel quadrature and normalizes each final
PSF-convolved stamp to its free physical band flux. Pixel centers use
zero-indexed `(x,y)` coordinates. Source centroids are fixed exactly at their
assigned prompts; no centroid offsets are permitted because they were not in
the authoritative Level-4/5 support.

Known GalSim PSFs are converted by `src/model9_galsim_adapter.py` to centered
31x31 g/r/z kernels. Each band is explicitly normalized to unit sum. GalSim
FFT ringing may be clipped only when the negative extremum is at most `1e-4`
of the positive peak and total negative mass is at most `5e-4` of positive
mass; larger negative values fail the contract. No unresolved PSF core is
added.

### Level 4: prompt-centered elliptical Sérsic

Each source has seven parameters:

`[flux_g, flux_r, flux_z, n, HLR, q, theta]`.

- each flux is direct, nonnegative, and has no hard upper bound;
- `0.5 <= n <= 6`;
- `0.03 <= HLR <= 3` arcsec;
- `0.1 <= q <= 1`;
- `theta` is modulo pi.

Geometry is shared across bands. The two sources have independent fluxes and
independent geometry.

### Level 5: prompt-centered bulge+disk

Each source has twelve parameters:

`[flux_g, flux_r, flux_z, disk_HLR, disk_q, disk_theta,
bulge_HLR, bulge_q, bulge_theta, BT_g, BT_r, BT_z]`.

The disk is Sérsic `n=1`; the bulge is Sérsic `n=4`. HLR, q, and angle use the
Level-4 bounds. Each band-specific B/T is in `[0,1]`. Total per-source band
fluxes are direct, nonnegative, unbounded-above likelihood parameters.

## Signed residual and objective

The astronomical layers are nonnegative renderer outputs `S_req` and
`S_comp`. The signed residual is derived, not independently optimized:

`R = O - S_req - S_comp`.

The sole primary objective is the fixed Gaussian signed-residual likelihood:

`NLL = 0.5 * sum((R / sigma)^2 + 2 log(sigma))`.

The optimizer minimizes the whitened residual vector. The constant log-sigma
term is still logged by band and in total. No truth loss, isolated-source flux
term, morphology label, neural loss, regularizer, or residual clipping is
permitted.

The nonnegative flux support is `[0, infinity)`. For starts and diagnostic
column scaling only, each band uses

`F_ref = max(sum(max(O_band, 0)), sqrt(sum(sigma_band^2)))`.

The two initial source fluxes sum to `F_ref` in every band. `F_ref` is not a
flux constraint and cannot be used as a prior or penalty.

## Optimizer and deterministic starts

- bounded SciPy trust-region reflective least squares;
- 16 starts per scene/family, retained in full whether favorable or not;
- seed `2026071519`;
- maximum 500 function evaluations per start;
- `ftol = xtol = gtol = 1e-10`;
- autograd float64 residual Jacobian supplied to the optimizer;
- `x_scale = "jac"`;
- no truth, catalog, isolated image, or previous fitted endpoint may initialize
  a start.

Scrambled Sobol morphology starts cover every bounded coordinate. Flux
allocation fractions cycle through
`[0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95]` with band offsets. Level-4
starts cycle through all source-wise `n=1/n=4` pairs. Level-5 begins with the
fixed B/T pairs `(0,0)`, `(1,1)`, `(0,1)`, `(1,0)`, `(0.1,0.1)`, `(0.5,0.5)`,
`(0.1,0.5)`, and `(0.5,0.1)`; remaining B/T starts are Sobol values. Do not
increase the budget for individual scenes.

## Symmetry quotient

Before endpoint comparison and local rank reporting:

- wrap angles modulo pi;
- set circular-ellipse angles to the canonical zero gauge;
- remove morphology columns of a zero-flux source;
- remove per-band B/T gauges when that source band has zero flux;
- remove bulge morphology at B/T=0 and disk morphology at B/T=1;
- when prompt coordinates coincide, sort source blocks canonically and record
  component-label symmetry.

Equivalent parameter vectors are one solution class. Genuinely different
source images or allocations remain distinct.

## Local diagnostics

Compute the float64 whitened-residual Jacobian at every accepted solution.
Flux columns are scaled by `F_ref`; bounded morphology columns are scaled by
their frozen support widths. Rank uses

`max(m,n) * eps(float64) * largest_singular_value`

after the symmetry quotient. Persist all singular values, rank, active
parameter count, null dimension and basis, condition number, gradient norm,
Gauss-Newton Hessian `J.T J`, Hessian eigenvalues, and Hessian condition.
Finite-difference and exact replay checks must pass before science.

## Acceptable endpoints and model support

An endpoint must have solver success and a finite objective. It is
likelihood-acceptable when chi-square is no greater than the 0.99 quantile for
`max(1, observation_pixels - active_parameters)` degrees of freedom. If every
converged endpoint fails this gate, the family is `OUT_OF_SUPPORT`; an empty
acceptable set is never unique.

Among acceptable endpoints, retain all solutions within

`best_NLL + 1e-8 + 1e-8 * max(abs(best_NLL), 1)`.

Numerical solution classes merge only when both requested and companion
relative L2 image distances are at most `1e-6`.

Report maximum pairwise:

- requested and companion relative L2 image diameter;
- source flux-allocation diameter scaled by `F_ref`;
- morphology-parameter diameter scaled by support width.

## Frozen classifications

Assign exactly one label per scene/family.

- `INVALID_CONTRACT`: provenance failure, oracle input, protected access, or
  post-freeze implementation/protocol change.
- `OPTIMIZATION_UNRESOLVED`: no successful endpoint or diagnostics cannot be
  produced.
- `NUMERICALLY_UNSTABLE`: nonfinite diagnostics or failure of the frozen
  deterministic numerical perturbation/replay checks.
- `OUT_OF_SUPPORT`: optimization resolves but no endpoint passes the fixed
  0.99 chi-square support gate.
- `NON_IDENTIFIABLE`: more than one class or nonzero local null space and at
  least one frozen image/flux/morphology diameter exceeds `1e-3`.
- `PARTIALLY_IDENTIFIABLE`: multiple classes or nonzero null space remain but
  all three diameter types are at most `1e-3`.
- `UNIQUE`: exactly one acceptable solution class; full active rank; null zero;
  condition number at most `1e6`; dimensionless gradient norm at most `1e-5`;
  requested and companion image diameter at most `1e-3`; flux-allocation and
  morphology diameter at most `1e-3`; stable prompt identity; no invalid
  zero-source collapse; and acceptable model support.
- `NEAR_UNIQUE`: one acceptable class with null zero that misses at least one
  strict UNIQUE numerical/diameter/conditioning rule without becoming
  numerically unstable.

Zero-source collapse is invalid for UNIQUE when either source total flux is at
most `1e-8` times the summed `F_ref`. Other bound contacts are reported and
must receive one-sided stability diagnostics; they are not silently discarded.

## Campaign-level outcome

Use the user-declared mapping without modification: union UNIQUE count 6-8 is
`FLUX_FREE_UNIQUENESS_LARGELY_SURVIVES`, 3-5 is
`FLUX_FREE_UNIQUENESS_PARTIALLY_SURVIVES`, and 0-2 caused by restored flux
ambiguity is `FLUX_FREE_UNIQUENESS_COLLAPSES`. Use support-limited,
optimization-unresolved, or invalid outcomes only under their declared causal
conditions. Do not select a better parameterization or family after results.

## Execution prohibition during preparation

No frozen scientific observation was loaded or fit while producing this
draft. No primary count, per-scene classification, or comparison outcome is
known. Neural training, PriorNet, and POST work remain prohibited.
