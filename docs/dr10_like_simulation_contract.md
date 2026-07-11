# Approximate DR10-conditioned simulation contract

Status: feasible as an approximation; not an exact DR10 instrument simulator.

## Scope and safe evidence

The sealed DR10 role is `future_lockbox` under
`data/manifests/dr10_grouped_source_split_20260711_024415/lockbox_policy.md`.
Only the split label, persistent identity, and group identity were loaded for
lockbox exclusion. No lockbox astrophysical column or linked source was opened.
The 20-source model-probe sample is a dedicated engineering manifest at
`outputs/runs/dr10_foundation_20260711_024415/manifests/dr10_engineering_sources.csv`;
it is absent from the five-way source manifest and was established by the
historical audit as non-lockbox engineering data.

Aggregate evidence is in
`outputs/runs/thayer_select_btk_foundation_20260711_152613/tables/dr10_like_simulation_parameters.csv`.
It establishes 0.262 arcsec/pixel, 256×256 acquisition stamps, per-band scalar
PSFSIZE distributions, and empirical source-excluded blank-region background
medians and robust noise scales. The companion figures are under
`figures/dr10_like_psf_noise_distributions/` in that run.

## Initial approximation

The initial custom survey may use:

- separate g,r,z GalSim PSFs, with each DR10 `PSFSIZE_band` interpreted only as
  the FWHM of a declared circular Moffat or Kolmogorov approximation;
- separate per-band observation-noise parameters fitted to the unsealed
  blank-region robust-scale distributions;
- 0.262 arcsec/pixel and an explicitly chosen stamp size (256×256 for direct
  geometry comparison, or a smaller declared engineering stamp for cost);
- CatSim intrinsic profiles and colors, rendered into detected electrons or a
  declared linear flux unit and converted to nanomaggies only through an
  explicit per-band zeropoint relation;
- one source-Poisson plus one sky/background observation realization after
  noiseless sources are summed.

BTK/surveycodex supports mutable survey/filter quantities and custom GalSim PSF
callables, so this approximation is technically feasible. Every band must keep
its own PSF and noise parameters. A scalar PSFSIZE is insufficient for exact
PSF matching and must never be described as a full DR10 PSF model.

## Flux, zeropoints, and omitted physics

BTK 1.0.9 renders detected electrons from AB magnitude, zeropoint, and exposure
through surveycodex. DR10 images are nanomaggies per coadd pixel. A DR10-like
configuration must either render directly into a documented nanomaggy scale or
record the per-band electron-to-nanomaggy conversion and its zeropoint. No
empirical flux/magnitude distribution is inferred from unsupported fields; the
current aggregate table therefore omits one rather than inventing it.

Omitted physics includes the spatially varying non-circular PSF, coadd kernels,
correlated resampling noise, exposure-level variation, masks, inverse variance,
detector defects, sky-subtraction systematics, deblender/Tractor modeling
errors, and calibration spatial variation. The DR10 residual is explicitly not
used as source-only noise truth.

More exact simulation would require exposure-level metadata, calibrated
per-position PSF images or models, throughput and zeropoints, gain/read-noise
and sky models, dither/coadd/resampling kernels, masks, and validation against
unsealed DR10 distributions. Until those exist, the correct label is
**approximate DR10-conditioned BTK/GalSim simulation**.
