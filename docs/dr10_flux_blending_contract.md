# DR10 flux-space blending contract

Status: **blocked — choose option 4, remain blocked and evaluate BTK/GalSim**  
Campaign evidence: `outputs/runs/dr10_model_probe_20260711_160018/`  
Scope: engineering contract only; no training, blend manifest, or sealed data use

## Decision

The matched DR10 observed/model/residual cutouts do **not** currently provide a
validated source-only, single-noise-realization, PSF-consistent blending input.
The three products are pixel-aligned, but only 15/20 triplets pass the declared
additivity gate, no central-only component can be recovered reliably from any
summed model cutout, and no full or exact-coordinate PSF has been validated.

Therefore no current DR10 cutout array is authorized as a contaminant. The
recommended next route is a small, separate BTK/GalSim forward-rendering
feasibility study. BTK is designed to render blended and isolated objects with
a PSF and pixel noise, and accepts custom GalSim/FITS PSFs; GalSim supplies
explicit rendering and seeded noise machinery. Neither project's official
documentation establishes a ready-made Tractor integration, so that handoff
would be custom work and must be tested rather than assumed. See the official
[BTK user guide](https://lsstdesc.org/BlendingToolKit/user_guide.html),
[BTK drawing API](https://lsstdesc.org/BlendingToolKit/src/btk.draw_blends.html),
and [GalSim noise documentation](https://galsim-developers.github.io/GalSim/_build/html/random.html).

## Prohibited current constructions

The following are not scientifically valid contaminant arrays:

- a whole `ls-dr10-model` scene cutout;
- a segmentation of that summed scene model presented as an official
  per-source Tractor render;
- a segmented observed cutout, because it carries another coadd-noise and
  residual-background realization;
- a model-assisted observed extraction, because its residual term carries
  coadd noise and fit errors;
- any triplet that fails numerical closure, if the residual is interpreted as
  `observed - model`;
- direct addition based only on similar scalar PSF FWHM values.

Display RGB, signed-asinh, and robust-stretch PNG arrays are never scientific
inputs, targets, labels, or flux measurements.

## Proposed future simulator contract

This contract is a requirement for reopening blending; it is not implemented
by this campaign.

For band `b` in ordered `g,r,z`, define

`B_b = T_b + alpha * Shift(C_b, dx, dy) + N_b`.

### Target array

`T_b` is a **noise-free forward render** of one declared target-source model on
the output WCS and pixel grid, convolved with the declared output PSF for band
`b`. It contains no sky pedestal, detector/coadd background, unrelated source,
or random noise.

An observation-preserving hybrid target may be considered in a later contract,
but then the unchanged observed cutout already supplies the sole coadd-noise
realization and `N_b` must be identically zero. That hybrid is not approved by
this probe because a compatible source-only contaminant is still unavailable.

### Contaminant source array

`C_b` is a **noise-free, central-only forward render** of exactly one declared
contaminant source, on a padded grid large enough to contain its relevant
profile wings. It contains no neighboring source, sky, coadd noise, rectangular
cutout pedestal, or residual image. Cropping occurs only after a no-wrap shift,
and lost edge flux is measured and recorded.

An already PSF-convolved `ls-dr10-model` cutout is not `C_b`. If Tractor
parameters are used, they must be rendered as a single named component at the
output PSF. That result is labeled a **Tractor-parametric render**, not an
observed or astrophysical truth image.

### Background and noise

`N_b` is the one and only random background/noise realization. It is generated
after the noiseless source renders have been summed. Its model, covariance,
variance or exposure map, band coupling, seed, and software implementation are
explicit metadata. No source extraction may import an earlier coadd-noise
realization into `T_b` or `C_b` under this fully synthetic contract.

The initial simulator benchmark may omit source Poisson noise only if that
omission is named in the benchmark definition. If source Poisson noise is
included, it is drawn once from the combined expected scene, not once per
input cutout and then added together.

### Flux scaling and color

All scientific arrays remain linear `g,r,z` nanomaggies per output pixel.
Contaminant scaling uses one recorded nonnegative scalar `alpha` applied to all
three bands, preserving the contaminant's input colors. Per-band scaling is
forbidden unless the benchmark explicitly declares a color-augmentation model
and records the three factors separately. There is no clipping, renormalization
in display space, or hidden magnitude conversion.

The shift operator must conserve summed flux away from the output boundary.
Its kernel and padding are fixed and tested with point sources and PSF moments;
subpixel interpolation is not accepted merely because its scalar flux sum is
close.

### PSF handling

Both source renders use the same declared output PSF in each band. Acceptable
routes are:

1. forward-render each intrinsic/empirical source model through a validated
   target PSF; or
2. convolve already rendered inputs to a common, strictly broader PSF using
   validated matching kernels and moment/encircled-energy tests.

No sharpening or deconvolution is authorized. Scalar `PSFSIZE_G/R/Z` values
may screen candidate pairings but cannot validate ellipticity, wings, or
subpixel response. Official DR10 catalogs define these columns as weighted
average PSF FWHM, while official coadd products provide per-pixel PSF-size maps;
see the [DR10 catalog schema](https://www.legacysurvey.org/dr10/catalogs/) and
[DR10 file data model](https://www.legacysurvey.org/dr10/files/). Full kernels
or a documented simulator PSF remain required for rendering.

### Source-model semantics

The benchmark may use parametric or empirical source models, but each family is
named and stratified in evaluation. Tractor models are model predictions with
parametric assumptions; they are never promoted to astrophysical ground truth.
The DR10 file documentation itself describes the model stack as the Tractor's
prediction of the coadd scene.

### Ground truth

Ground truth is strictly **procedural injection truth**:

- source identity and model family;
- noiseless `T_b` and `alpha * Shift(C_b, dx,dy)` renders;
- their sum before noise;
- positions, fluxes, colors, transforms, and output PSFs;
- the single generated noise realization and its seed.

It does not mean that the source profile is the galaxy's true morphology, that
the Tractor fit is physically correct, or that a real blended observation has
a uniquely known decomposition.

### Replay and integrity

Every future blend must record, before generation:

- immutable source/catalog identifiers and hashes;
- input parameter-row hashes and source-model family/version;
- ordered bands, units, WCS, shape, pixel scale, and dtype;
- PSF source, per-band kernel hashes, and common-PSF rule;
- `alpha`, `dx`, `dy`, shift kernel, padding, crop, and lost-flux fraction;
- noise model, seed/seed sequence, and variance/covariance inputs;
- code commit or source-file hashes and complete dependency versions;
- noiseless component, noiseless blend, noise, and final-array hashes.

Replay is accepted only when a clean rerun reproduces the declared arrays under
the fixed software contract, or meets a preregistered numeric tolerance when
the rendering backend cannot promise bitwise portability. A filename or random
seed alone is not replay evidence.

## Gate for reopening blending

FITS blending stays closed until all conditions below pass on an engineering
set:

- matched products pass alignment and numerical closure;
- a true per-source render is available without summed-scene neighbors;
- the benchmark's accepted morphology limitation is written and stratified;
- exactly one intended noise realization is present;
- full PSF handling and subpixel rendering are validated;
- no whole scene is pasted as one source;
- units and `g,r,z` order are invariant;
- end-to-end generation replays from immutable inputs and recorded software;
- no protected evaluation data are inspected or used.

Until then, the DR10 observed/model/residual triplets are audit and diagnostic
products only.

## Known remaining limitations

- Only 20 engineering scenes were reviewed.
- Five triplets fail closure; the service-side cause is unresolved.
- Conservative central attribution is available for only seven sources, and
  only five of those also pass closure.
- Catalog/detection-excluded control apertures can still contain undetected or
  diffuse structure.
- The official catalog samples only scalar FWHM; exact-coordinate PSF maps and
  full kernels were not validated here.
- [BTK 1.0.9](https://lsstdesc.org/BlendingToolKit/install.html) currently
  requires Python below 3.13, while this repository's environment is Python
  3.14.6; a separate compatible environment is required.
- BTK/GalSim can make procedural image truth, not remove model-family bias or
  prove realism for the DR10 galaxy population.
