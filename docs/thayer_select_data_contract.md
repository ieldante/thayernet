# Initial Thayer-Select data contract

## Input and required output

For the current 12 arcsec LSST engineering stamp, the input is float data of
shape `(4, 60, 60)` in channel-first order:

1. normalized g-band blend;
2. normalized r-band blend;
3. normalized z-band blend;
4. one Gaussian requested-source coordinate channel.

The physical image planes are detected electrons per pixel in the CatSim pilot.
Band order is always `g,r,z`; unknown order or units are fatal. The first smoke
model receives no IVAR, exposure, mask, PSF, or noise channels. Those may be
added only after their inference-time availability and semantics are justified.

The required output is the requested noiseless isolated source with shape
`(3, 60, 60)`, same g,r,z order. Optional future outputs are a pixel
log-variance map and one global recoverability probability. Neither optional
head is part of the first coordinate-conditioned smoke run.

## Prompt convention

Coordinates are zero-based continuous pixels, `x=column`, `y=row`, obtained by
mapping BTK tangent-plane arcsecond offsets through the declared WCS. The
prompt is evaluated at pixel centers as a unit-peak Gaussian with fixed
`sigma=2.0` pixels; subpixel centers are retained without rounding. Values
outside the array are simply cropped by the boundary. A valid target must have
its declared center inside the stamp.

An empty prompt is an exactly zero channel. A wrong prompt is a finite Gaussian
at a seeded in-frame position at least a frozen exclusion radius from the
manifest target; variants point to the other source or to source-free sky.
Neither is silently recentered. No-source truth is an exact zero `(3,H,W)` map
for the source-selection output contract.

## Normalization and inversion

The first smoke contract uses one positive scale per band fitted from the
engineering training subset only (for example a robust high quantile of
absolute blend values). The transform is linear and unclipped:
`x_norm[b]=x[b]/scale[b]`; the isolated target uses the same scale. Inversion is
exact up to floating-point rounding: `x=scale*x_norm`. Negative pixels remain
negative. No per-scene normalization is allowed because it would hide absolute
flux and couple normalization to evaluation data.

For a real experiment, scales are fitted from the final training partition
only and frozen. Validation, calibration, development test, and sealed lockbox
never contribute. Approximate inversion error is reported as maximum absolute
and relative error in flux space; non-finite or non-positive scales fail.

## Scene/query coverage

The benchmark represents separate query rows for source A and source B against
the same unchanged blend, plus isolated-source/no-harm, empty-coordinate, and
wrong-coordinate queries. Future generators add two-, three-, and four-source
scenes without changing the channel contract. A target contains no unrelated
background, neighboring-source flux, or second observation-noise realization.

Oracle-only fields include the isolated target, every unprompted neighboring
source, full source list, exact identities, generator parameters not exposed as
input, PSF/noise truth not declared as input, and every future outcome label.
Uncertainty means predicted pixel dispersion; recoverability means a calibrated
probability of meeting a frozen outcome rule; accept/abstain is the downstream
decision made from that probability. They are not interchangeable.
