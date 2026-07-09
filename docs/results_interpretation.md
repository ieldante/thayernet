# Results Interpretation

## Why Threshold Can Be Worse Than Identity

The identity baseline leaves the blend unchanged. It fails to remove the
contaminant, but it also preserves all target light.

The threshold baseline tries to remove bright connected regions. That can delete
target structure, keep contaminant structure, or create holes where the hidden
target should be reconstructed. Because thresholding does not infer missing
target light, it often has higher affected-region error than identity.

## Why SSIM Changes Are Modest

SSIM is computed over the whole image. In these controlled cutouts, much of the
image is unchanged background or target light that the identity baseline already
preserves. A model can make a large improvement in the contaminant-altered area
while producing a smaller-looking change in whole-image SSIM.

## Why Affected-Region MSE Is Primary

Affected-region MSE evaluates only pixels where the blended image differs from
the target by more than the affected threshold. This isolates the actual
deblending problem. Whole-image metrics are still reported, but they can make
identity look deceptively strong because unchanged pixels dominate the average.

## Why Thayer-BR v0.1 Can Improve Normal Performance

Thayer-BR v0.1 is the current best research checkpoint. Balanced
hard-case training does not only teach rare stress cases. It exposes the model
to a wider range of overlap, brightness, size, and core-obstruction patterns.
That can improve the residual subtraction rule even on normal held-out blends,
especially when normal random sampling includes moderate overlap cases.

## Why Thayer-Direct Still Beats Residual Sometimes

Thayer-Residual preserves unchanged light through subtraction, but it can under-
or over-subtract in ambiguous regions. Thayer-Direct can sometimes redraw target
structure better, especially when the contaminant resembles a removable pattern
but the target structure is predictable from context. The aggregate result
favors Thayer-BR v0.1, but per-sample winners vary.

## Why Severe Blends Are Not Always Model-Hard

Blend severity measures image-level damage. Model failure measures model error.
Those are related but not identical.

A high-severity blend can be easy if the contaminant is bright, obvious, and
spatially separable. A low-severity blend can be hard if the affected pixels
land on the target core or erase structure that cannot be inferred from the
remaining image.

## Main Scientific Reading

- Thayer-Direct proves the benchmark is learnable.
- Hard stress testing reveals that Thayer-Direct loses robustness under
  stronger overlap and contaminant conditions.
- Thayer-Residual improves stress robustness by learning contaminant signal
  to subtract.
- Thayer-BR v0.1 improves the strongest aggregate metrics in the current
  same-run evaluation.
- The project remains a controlled synthetic study and should be written up with
  explicit limitations.

## Evaluation Robustness Audit

Audit run: `outputs/runs/evaluation_audit_20260708_220833`.

The audit found that affected-region masks are computed from
`abs(blended - target).mean(axis=-1) > threshold`, not from prediction error.
Across affected-mask thresholds `0.005`, `0.01`, `0.02`, and `0.04`,
Thayer-BR v0.1 remained the best learned model on both normal and stress sets.

Dilating the affected mask by `0`, `1`, `3`, `5`, and `9` pixels did not change
the aggregate ranking: Thayer-BR v0.1 remained best on normal and stress
sets. This supports the result against the concern that the default mask might
exclude faint halo contamination, although exact MSE values still depend on the
chosen mask definition.

The multi-seed audit used three independent 1,000-blend normal seeds and three
independent 1,000-blend stress seeds without retraining. Thayer-BR v0.1 won
all tested seeds. Mean improvement was `27.04 +/- 1.04x` on normal blends and
`15.76 +/- 0.07x` on stress blends.

Core-region metrics support the same broad result, but they also show the main
limitation: target-core affected pixels remain much harder than non-core
affected pixels. Thayer-BR v0.1 improves core affected MSE by `4.19x` on
normal blends and `6.15x` on stress blends, compared with larger non-core
improvements.

Residual logic was checked visually and numerically. The target residual is
`blended - target`, reconstruction is `blended - predicted_residual`, and the
Thayer-BR v0.1 predicted residual has high correlation with the true residual
in the audited samples. The headline `27.79x` normal and `16.47x` stress values are
therefore reasonable current same-run claims, with the caveat that they are
controlled synthetic results rather than survey-grade performance claims.
Thayer-BR v0.1 is an experimental research checkpoint name, not a public
production release or stable deployed model version.
