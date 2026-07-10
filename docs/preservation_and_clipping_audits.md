# Preservation and Clipping Audits

These evaluation-only audits test two assumptions that affected-region scores
do not answer: whether a residual model preserves an unblended input, and
whether clipping to `[0, 1]` hides a material amount of error.

Runs:

- `outputs/runs/preservation_null_tests_20260710_063312`
- `outputs/runs/clipping_audit_20260710_063312`

The earlier `20260710_075442` directories are preserved as preliminary runs.
A read-only review found that their unaffected-region label was too strong and
their per-sample clipping rows were not saved. Append-only diagnostics in those
directories point to the corrected runs; no old result was deleted.

All model inference used MPS. CPU fallback was disabled. Thayer-BR v0.2
Moderate, the v0.3 Delta preservation/perceptual ablation, and the ResUNet v0.4
architecture ablation were evaluated on 1,000 unblended targets, 1,000
fixed-seed normal blends, and 1,000 fixed-seed hard-stress blends. All 16
pre-existing checkpoints were unchanged before and after inference.

These are development diagnostics. The historical random-index source split
contains duplicated objects across partitions, so none of these numbers is a
locked-final estimate.

## Unblended-Input Null Test

For an unblended input, the desired predicted residual is approximately zero.
The reconstruction is `unblended_input - predicted_residual`.

| Model | Mean null MSE | Unblended SSIM | p99 null MSE | Maximum null MSE | Cases with MSE > 0.001 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thayer-BR v0.2 Moderate | 0.00002646 | 0.998626 | 0.00026892 | 0.00134724 | 3/1,000 |
| Thayer-BR v0.3 Delta | 0.00000120 | 0.999933 | 0.00000735 | 0.00042650 | 0/1,000 |
| Thayer-ResUNet v0.4 | 0.00002144 | 0.998909 | 0.00006481 | 0.00030520 | 0/1,000 |

Delta's mean unblended-input MSE is about 22.1x lower than v0.2 Moderate.
ResUNet's mean unblended-input MSE is 19.0% lower than v0.2, but its unblended
core MSE is 53.5% higher and its mean absolute error is also higher.

These are unblended inputs, not artifact-filtered sources. The heuristic
source-artifact audit flags 23/1,000 inputs. Mean v0.2 null MSE is `0.0001562`
in that flagged stratum versus `0.0000234` among non-flagged inputs; Delta and
ResUNet show the same direction. The flags require manual review and can include
legitimate morphology, so this is stratification rather than automatic
exclusion.

The v0.2 mean remains small, but the distribution has a meaningful tail.
Manual review of the three highest combined-error examples shows false
subtraction of bright off-center sources and genuine target structure. That
failure must not be hidden by the aggregate mean. It also explains why an
artifact-screened benchmark should retain fields with companions rather than test
only visually simple central galaxies.

## Affected and Mask-Complement Regions

The affected mask uses mean absolute RGB blend change greater than `0.02`.
The mask complement still contains sub-threshold blend changes, blur, and
noise. The historical field name `unaffected_target_mse` therefore does not
measure pure model damage. The paired excess target error over identity is the
appropriate comparison for whether model modification helps or hurts in that
region; output-versus-blend MSE is reported separately.

| Suite | Model | Affected MSE | Mask-complement target MSE | Output-vs-blend MSE | Paired excess target MSE vs identity |
| --- | --- | ---: | ---: | ---: | ---: |
| Normal | Thayer-BR v0.2 Moderate | 0.002032 | 0.00007195 | 0.00004587 | 0.00003343 |
| Normal | Thayer-BR v0.3 Delta | 0.002261 | 0.00003608 | 0.00001958 | -0.00000244 |
| Normal | Thayer-ResUNet v0.4 | 0.002056 | 0.00005479 | 0.00002874 | 0.00001627 |
| Hard stress | Thayer-BR v0.2 Moderate | 0.003847 | 0.00005195 | 0.00004844 | 0.00003459 |
| Hard stress | Thayer-BR v0.3 Delta | 0.004205 | 0.00002659 | 0.00002439 | 0.00000923 |
| Hard stress | Thayer-ResUNet v0.4 | 0.003729 | 0.00003901 | 0.00003433 | 0.00002165 |

Delta has the lowest paired excess target error over identity in the
mask-complement region, but it remains worse than v0.2 on the primary
affected-region MSE in both suites. This is a preservation tradeoff, not
evidence that Delta should replace v0.2 as current best.

## Clipped Versus Unclipped Reconstruction

For every learned model, metrics were computed both on raw
`blended - predicted_residual` and after clipping to `[0, 1]`. Across the six
model/suite comparisons:

- clipping reduces whole-image MSE by at most 0.96%;
- clipping reduces affected-region MSE by at most 0.16%;
- the model ranking does not change;
- low-clipped channel fractions range from about 0.15% to 0.81%;
- high-clipped channel fractions range from about 0.03% to 0.12%.

The saved per-sample audit finds 10 of 6,000 model/sample comparisons with an
absolute affected-MSE reduction above `0.0001`, all in hard stress. None has a
relative reduction above 10%. The largest absolute reduction is `0.001126`,
while the largest group p99 is `0.0000701`.

There is no strong clipping dependence in this audit. The out-of-range and
residual-sign statistics should still be reported because they describe
unphysical subtraction or added-light behavior. Sign fractions alone are not
enough; the saved table also records the mean excursion magnitude.

## Saved Diagnostics

- `tables/null_preservation_metrics.csv`
- `tables/null_preservation_per_sample.csv`
- `tables/null_preservation_artifact_strata.csv`
- `tables/unaffected_region_metrics.csv`
- `tables/unaffected_region_per_sample.csv`
- `tables/clipped_vs_unclipped_metrics.csv`
- `tables/clipped_vs_unclipped_per_sample_metrics.csv`
- `tables/clipping_pixel_statistics.csv`
- `tables/clipping_pixel_statistics_per_sample.csv`
- `tables/clipping_effect_distribution_summary.csv`
- unblended reconstruction/error grid and false-residual heatmaps
- affected-versus-unaffected and clipped-versus-unclipped charts

The source-leakage finding and the unblended-input tail justified the grouped
benchmark repair. The grouped v0.2 retrain and grouped development evaluation
are now complete; these older random-index audits remain development
diagnostics and must not be presented as final-paper estimates.
