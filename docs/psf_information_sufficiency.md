# PSF information sufficiency

Explicit PSF metadata is a legitimate deployable observation-process input,
but information sufficiency depends on variation. In the current benchmark,
the three image channels always use the same fixed g/r/z PSF triplet. Adding
the same three kernels or scalars to every scene cannot distinguish which
scene lies in the low-observability/high-obstruction regime.

The run `outputs/runs/thayer_select_psf_conditioning_20260712_043442/` found
one unique combined PSF configuration among 18,000 scenes and an effective
configuration count of 1.0. Within each band, FWHM, second-moment size, area,
ellipticity, concentration, and noise-equivalent area have zero variation up
to floating-point aggregation roundoff below 1.5e-14. Training, validation,
and calibration configuration distributions are identical.

Consequently, true PSF, within-partition shuffled PSF, and constant-median PSF
controls are identical by construction. No association test or fitted control
can establish independent PSF information from these scenes. The frozen gate
therefore prohibits claiming that explicit PSF improved or failed to improve
the A3 model empirically; the correct conclusion is that this dataset cannot
identify the effect.

Image and flux remain FAIL, centroid remains PASS, and no full-policy campaign
is authorized. Development and lockbox have no result and remained untouched.
The single next experiment is to prospectively generate scenes with realistic
varying PSFs, with the variation distribution and PSF availability frozen
before outcome-based fitting or selection.
