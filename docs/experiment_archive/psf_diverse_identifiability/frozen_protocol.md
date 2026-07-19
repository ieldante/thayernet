# Thayer-PSF-Diverse-Flux-Identifiability-v0 frozen protocol

Status: **FROZEN BEFORE SCIENTIFIC OBSERVATION ACCESS**

## Scope and primary endpoint

Exactly scenes `[0, 3, 5, 6, 18, 51, 73, 81]` are evaluated under Level 4 Sérsic and Level 5
bulge+disk. S1 is imported unchanged from the completed predecessor. New S2
and P2 fits use exactly 16 starts each, for 32 new fits and 512 retained
endpoints. The primary endpoint is P2 union `UNIQUE` count across eight scenes.

## Acquisition

Observation A is the authoritative original noisy BTK LSST g/r/z blend.
Observation A2 is regenerated from the same two CatSim rows and coordinates
with the original LSST PSF and noise seed `original_noise_seed +
100000000`. Observation B uses the same catalog rows,
coordinates, photometric calibration, exposure time, morphology, and intrinsic
flux, with that same paired second-seed rule and the preregistered PSF-B only.
The shared second seed couples the alternative S2/P2 controls fairly; both are
independent of Observation A. BTK `CatsimGenerator`, `add_noise='all'`, one
source-Poisson plus zero-mean sky-Poisson realization, 60x60 geometry, and 0.2
arcsec/pixel are frozen. No deconvolution or transformation of Observation A
is used. Simulator-generated isolated layers are discarded and never persisted
or passed to inference.

## Inference and objective

The solver receives exactly two blended g/r/z arrays, the frozen requested and
companion coordinates, both known normalized PSFs, image geometry/pixel scale,
the observation-only plug-in sigma maps, and frozen family/support metadata.
One shared source-parameter vector renders both observations. The joint
whitened residual is the concatenation of the two predecessor residual vectors;
the NLL is their exact sum. Starts and diagnostic parameter scales are computed
from Observation A only, making S2/P2 starts byte-identical. Per-observation and
per-band likelihood and chi-square terms are logged.

## Frozen solver

All bounds, transformations, symmetry gauges, endpoint tolerances, and
classification definitions are inherited unchanged from the predecessor:
`{"acceptable_gradient_norm": 1e-05, "angle_bounds_radians": [0.0, 3.141592653589793], "axis_ratio_bounds": [0.1, 1.0], "bulge_fraction_bounds": [0.0, 1.0], "endpoint_image_rtol": 1e-06, "flux_initialization_total_multiplier": 1.0, "ftol": 1e-10, "gtol": 1e-10, "half_light_radius_bounds_arcsec": [0.03, 3.0], "invalid_zero_flux_fraction": 1e-08, "max_nfev": 500, "maximum_condition_number": 1000000.0, "model_acceptance_quantile": 0.99, "objective_accept_atol": 1e-08, "objective_accept_rtol": 1e-08, "optimizer_seed": 2026071519, "oversample": 4, "pixel_scale_arcsec": 0.2, "psf_kernel_size": 31, "sersic_n_bounds": [0.5, 6.0], "starts_per_family": 16, "symmetry_tolerance": 1e-10, "unique_flux_allocation_diameter": 0.001, "unique_image_diameter": 0.001, "unique_morphology_diameter": 0.001, "xtol": 1e-10}`. The only multi-observation
extension is that objective comparison and clustering use the joint objective,
image equivalence must hold through both PSFs, diameters are the maximum over
both rendered observations, observation count is doubled for the fixed 0.99
chi-square support gate, and the Jacobian stacks both observation blocks.

## Causal and campaign rules

Classification priority is UNIQUE > NEAR_UNIQUE > PARTIALLY_IDENTIFIABLE >
NON_IDENTIFIABLE > OUT_OF_SUPPORT. At family level: S2 improves over S1 and P2
does not exceed S2 => ADDITIONAL_EXPOSURE_ONLY; P2 improves over S1 while S2
does not => PSF_DIVERSITY_SPECIFIC; both improve and P2 exceeds S2 =>
BOTH_EXPOSURE_AND_PSF_DIVERSITY; neither improves => NO_MEANINGFUL_GAIN; any
unresolved/unstable/invalid comparison => INCONCLUSIVE_OPTIMIZATION. A P2
campaign result materially exceeds S2 when its union UNIQUE count is at least
one larger. With P2 union 0-2, conditioning improvement requires at least half
of the 16 family fits to improve minimum nonzero singular value, condition,
endpoint-class count, or a frozen diameter by at least 5% relative to S1.
The user-declared campaign outcome mapping is then applied without alteration.

## Integrity

No isolated HDF5 dataset, development data, Atlas tensor, lockbox, neural
network, truth initialization, catalog morphology label, true source flux, or
per-source photometry may enter inference. Acquisition catalog rows are used
only inside the authoritative forward simulator and are destroyed before the
paired observation artifacts are sealed. Every result is written atomically
and every start, including failures and budget exhaustion, is retained.
