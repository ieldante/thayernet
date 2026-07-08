# Figure Captions

This directory contains reviewed, public-safe figures copied out of local
experiment outputs for README/report use. Raw `outputs/` directories and
checkpoints remain ignored.

## `normal_vs_stress_improvement_ratio.png`

Residual prediction improves affected-region MSE over direct reconstruction on
both normal held-out and hard stress tests.

## `affected_region_mse_bar.png`

Affected-region MSE highlights reconstruction quality only where the contaminant
altered the target image.

## `residual_success_over_direct.png`

Qualitative stress-test example where residual prediction preserves more target
structure than direct reconstruction. This is an illustrative case, not a claim
that residual wins on every sample.

## `direct_unet_success.png`

Legacy direct U-Net success example. The direct model removes a visually
significant contaminant while preserving the target galaxy structure.

## `direct_unet_partial_failure.png`

Legacy direct U-Net partial-failure example. The model suppresses some
contaminating structure but loses target detail in an ambiguous overlapping
region.

Earlier static figures may display the generator's legacy easy/medium/hard
metadata. These labels are retained for provenance but are not treated as final
model-failure categories. Current analysis separates generation difficulty,
measured blend severity, core obstruction, and model failure.
