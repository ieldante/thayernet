# Figure Captions

This directory contains reviewed, public-safe figures copied out of local
experiment outputs for README and report use. Raw `outputs/` directories and
saved model checkpoint files remain ignored.

## Current Thayer-BR v0.2 Moderate Figures

### `v02_improvement_ratio.png`

Thayer-BR v0.2 Moderate improves affected-region MSE over identity on both
normal held-out and hard stress-test blends. Recommended README figure.

### `v02_affected_mse_bar.png`

Affected-region MSE comparison for identity, threshold, Thayer-Direct,
Thayer-Residual, Thayer-BR v0.1, and Thayer-BR v0.2 Moderate.

### `v02_core_mse.png`

Core affected MSE comparison. Use to show that v0.2 Moderate improves the
harder target-core overlap region relative to Thayer-BR v0.1.

### `v02_weighted_vs_v01_scatter.png`

Per-sample affected-region MSE comparison between Thayer-BR v0.2 Moderate and
Thayer-BR v0.1.

### `v02_multiseed_summary.png`

Multi-seed improvement ratio summary for Thayer-BR v0.2 Moderate on normal and
stress evaluations.

### `v02_weighted_improvement_example.png`

Qualitative example where Thayer-BR v0.2 Moderate improves over Thayer-BR
v0.1.

### `v02_counterexample.png`

Qualitative example where Thayer-BR v0.1 beats Thayer-BR v0.2 Moderate on an
individual sample. Use as a limitation or appendix figure, not as the leading
README figure.

## Historical Thayer-BR v0.1 Figures

### `balanced_normal_vs_stress_improvement_ratio.png`

Thayer-BR v0.1 improves affected-region MSE over identity on both
the current normal held-out evaluation and the hard stress test.

### `balanced_affected_region_mse_bar.png`

Affected-region MSE highlights reconstruction quality only where contaminant
light changed the target image.

### `balanced_direct_vs_balanced_scatter.png`

Per-sample affected-region MSE comparison between Thayer-Direct and
Thayer-BR v0.1 (balanced residual).

### `balanced_old_residual_vs_balanced_scatter.png`

Per-sample affected-region MSE comparison between Thayer-Residual and
Thayer-BR v0.1 (balanced residual).

### `balanced_stress_core_overlap.png`

Stress-test affected-region performance grouped by target-core overlap bin.

### `balanced_residual_improvement_example.png`

Qualitative example where Thayer-BR v0.1 improves over Thayer-Residual.

### `balanced_residual_failure_example.png`

Qualitative example where Thayer-BR v0.1 still fails in an
ambiguous overlap case. Use in limitations or appendix, not as a leading README
figure.

### `balanced_residual_counterexample.png`

Qualitative example where Thayer-Residual or Thayer-Direct beats Thayer-BR v0.1
on an individual sample.

## Legacy Direct/Residual Figures

### `normal_vs_stress_improvement_ratio.png`

Earlier direct-vs-residual improvement chart. Keep for provenance unless a
caption clearly states that it predates Thayer-BR v0.1.

### `affected_region_mse_bar.png`

Earlier direct-vs-residual affected-region MSE chart.

### `residual_success_over_direct.png`

Qualitative stress-test example where Thayer-Residual preserves more target
structure than Thayer-Direct. This is illustrative, not a claim that
Thayer-Residual wins on every sample.

### `direct_unet_success.png`

Legacy Thayer-Direct success example. The direct model removes a visually
significant contaminant while preserving target structure.

### `direct_unet_partial_failure.png`

Legacy Thayer-Direct partial-failure example. The model suppresses some
contaminating structure but loses target detail in an ambiguous overlapping
region.

Earlier static figures may display the generator's legacy easy/medium/hard
metadata. These labels are retained for provenance but are not treated as final
model-failure categories. Current analysis separates generation difficulty,
measured blend severity, core obstruction, and model failure.
