# Explicit PSF conditioning

The prospective run
`outputs/runs/thayer_select_psf_conditioning_20260712_043442/` stopped with
**PSF NON-INFORMATIVE BY CONSTRUCTION**. This is a provenance and variation
result, not a fitted PSF-conditioning result.

The historical Thayer-Select scenes use BTK's default SurveyCodex LSST PSF:
an axisymmetric GalSim convolution of Kolmogorov atmospheric seeing and an
Airy telescope profile. The exact analytic objects are replayable. Native
60x60 audit kernels reproduced deterministically in sampled scenes from the
training, validation, and natural-calibration partitions.

The PSF differs by band but not by scene. Fixed FWHM values are 0.86, 0.81,
and 0.77 arcsec in g, r, and z. Every one of the 18,000 audited scenes has the
same combined three-band configuration; its effective configuration count is
1.0. The PSF is not spatially varying, and both sources in a scene receive the
same per-band PSF. Scene noise seeds do not alter it.

The mandatory variation gate therefore stopped the campaign before
preregistration or fitting. No P0-P5 comparison, PSF embedding, shuffled
control, observability head, risk head, calibrator, GroupDRO model, information
ablation, policy, development evaluation, or lockbox evaluation exists.
Pixels-only A3 remains the authoritative observability result: strong ranking
but operational failure at high-precision recall, Brier, and ECE gates.

Exactly one next experiment is authorized for future preregistration:
**prospectively generate scenes with realistic varying PSFs**. It was not run
in this campaign.
