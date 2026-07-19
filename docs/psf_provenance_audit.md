# PSF provenance audit

The authoritative audit is in
`outputs/runs/thayer_select_psf_conditioning_20260712_043442/`. It covers all
12,000 r-training scenes, 2,000 r-validation scenes, and 4,000 natural-
calibration scenes, for 54,000 scene-band provenance rows.

The renderer calls `get_surveys("LSST")` for every scene and passes no custom
PSF function. BTK therefore constructs the same deterministic SurveyCodex PSF
in each band on every call. The model family is a GalSim convolution of a
Kolmogorov profile and an Airy profile. Pixel scale is 0.2 arcsec/pixel.
There is no PSF seed; the persisted scene seed affecting this observation path
is the noise seed, which does not change the PSF.

Provenance is exact and uniform across partitions. All scenes have a PSF, but
they used BTK's implicit default rather than a scene-specific declared draw.
The audit makes that default explicit through its analytic representation,
package versions, configuration hash, per-band native-grid kernel hashes, and
sampled scene replay. Position angle is deliberately recorded as undefined:
the analytic profiles are axisymmetric and finite-grid anisotropy near 1e-6
does not define a stable physical orientation.

Key artifacts are `tables/psf_provenance_inventory.csv`,
`tables/psf_scene_alignment_replay.csv`,
`tables/psf_cross_band_distances.csv`,
`manifests/psf_kernel_bank.npz`, and `figures/psf_examples/` inside the run.

Three incomplete timestamped attempts are preserved and superseded. They
stopped before preregistration or fitting because of, respectively, a table-
column accessor error, an unstable orientation label caught before
finalization, and strict JSON handling of an undefined angle. The authoritative
run records them in `logs/superseded_attempts.json`; none was overwritten.
