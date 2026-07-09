# Paper Plan

Working title: **Thayer-Net: Controlled Galaxy Deblending with Direct,
Residual, Balanced, and Weighted Residual U-Net Models**

The paper should remain honest about scope. The project is a controlled
synthetic deblending benchmark, not a deployment-ready survey pipeline.

## Abstract

We present Thayer-Net, a controlled synthetic benchmark for galaxy deblending
with Galaxy10 DECaLS cutouts. The benchmark constructs foreground-only blends
with known clean targets, avoiding rectangular cutout artifacts from naive image
addition. We compare identity and threshold baselines with compact U-Net models
trained for direct target reconstruction, residual contaminant prediction, and
balanced hard-case residual prediction. Thayer-Direct improves
affected-region MSE over identity on normal held-out blends, but stress testing
reveals weaker robustness under harder overlap conditions. Thayer-Residual
improves stress performance by learning what light to subtract. Thayer-BR v0.1
shows that balanced hard-case training improves robustness, and Thayer-BR v0.2 Moderate
further improves affected and core reconstruction with an
affected/core-weighted residual loss. The results support targeted synthetic
deblending as a useful controlled study, while remaining limited by simplified
sky, PSF, noise, source-environment, and apparent-size-normalization
assumptions.

## 1. Introduction

- Motivate astronomical source deblending: overlapping sources bias flux,
  morphology, color, and downstream inference.
- Explain why a controlled synthetic benchmark is useful before claiming real
  survey deployment.
- State the research question: can compact U-Nets recover target galaxies from
  synthetic blends better than simple baselines, and how does robustness change
  under harder blends?
- Preview the model progression: Thayer-Direct, Thayer-Residual, Thayer-BR
  v0.1, and Thayer-BR v0.2 Moderate.

## 2. Data and Synthetic Blend Construction

- Dataset: Galaxy10 DECaLS cutouts.
- Local HDF5 data are not included in the repository.
- Train/validation/test split is performed before blending to avoid
  source-image leakage.
- Blends are generated from target and contaminant images within the same split.
- Foreground-only contaminant extraction avoids rectangular cutout artifacts and
  double-background problems.
- Halo-aware masks preserve diffuse contaminant light.
- Limitations: synthetic foreground approximation, no full sky/PSF realism, and
  no physically correlated source environments.

## 3. Methods

- Identity baseline: return the blended image unchanged.
- Threshold baseline: simple threshold/connected-component removal.
- Thayer-Direct baseline: `blended -> target`.
- Thayer-Residual model-formulation upgrade: `blended -> residual`,
  reconstruction by subtraction.
- Thayer-BR v0.1 training-distribution upgrade: residual objective with 50%
  normal, 30%
  high-overlap/core-obstruction, and 20% brightness/size stress training blends.
- Thayer-BR v0.1 is not a new architecture; it is a residual U-Net trained on a
  more targeted blend distribution.
- Thayer-BR v0.2 Moderate objective upgrade: same residual U-Net family with
  normalized affected/core-weighted residual MSE.
- Thayer-BR v0.2 Strong as a weighting ablation, not the main model.
- Describe compact U-Net encoder-decoder structure with skip connections.

## 4. Evaluation

- Normal held-out synthetic evaluation.
- Hard stress-test distribution:
  - `n = 1000`
  - `max_shift = 18`
  - brightness range `0.8` to `1.4`
  - blur range `0.0` to `0.15`
  - noise range `0.0` to `0.006`
  - rotation off
  - `min_size_ratio = 0.75`
  - `min_mask_fraction = 0.01`
  - affected threshold `0.02`
- Metrics: whole-image MSE/MAE/PSNR/SSIM and affected-region MSE/MAE.
- Report model-improvement ratio and worse-than-identity counts.
- Analyze by generation difficulty, blend severity, core obstruction, and model
  failure without treating those concepts as interchangeable.
- Include the evaluation robustness audit:
  - affected-mask threshold sensitivity at `0.005`, `0.01`, `0.02`, and `0.04`
  - affected-mask dilation/halo sensitivity at `0`, `1`, `3`, `5`, and `9`
    pixels
  - three normal and three stress evaluation seeds, 1,000 blends per seed
  - residual sign/reconstruction audit
  - v0.2 multi-seed audit
  - size/visual audit, including apparent-size ratio, halo-band error, and
    visual-vs-metric disagreement examples
  - explicit caveat that standard normal blend outputs do not save global
    source indices

## 5. Results

### Thayer-Direct Baseline

- Earlier normal evaluation: Thayer-Direct affected-region MSE `0.004428` versus
  identity `0.062555`, about `14.13x` improvement.
- Current 1,000-blend comparable normal table: Thayer-Direct affected-region MSE
  `0.004236`, about `16.08x` improvement.

### Stress Degradation

- Hard stress identity affected-region MSE: `0.075541`.
- Thayer-Direct stress affected-region MSE: `0.009390`.
- Thayer-Direct stress improvement: `8.04x`.
- Interpretation: Thayer-Direct still works, but harder overlap reduces
  robustness.

### Thayer-Residual Improvement

- Thayer-Residual stress affected-region MSE: `0.007069`.
- Thayer-Residual stress improvement: `10.69x`.
- Worse-than-identity stress cases fall from `13/1000` for Thayer-Direct to
  `0/1000` for Thayer-Residual.
- Thayer-Direct remains better on some individual cases.

### Thayer-BR v0.1 Improvement

- Training: 8,000 blends, 1,000 validation blends, batch size 8, 20 epochs,
  best validation loss `0.000378` at epoch 18.
- Current normal affected-region MSE: `0.002451`, or `27.79x` improvement.
- Hard stress affected-region MSE: `0.004587`, or `16.47x` improvement.
- Worse-than-identity stress cases: `0/1000`.
- Thayer-BR v0.1 beats Thayer-Residual on `91.3%` of normal cases and `87.9%`
  of stress cases.
- Thayer-BR v0.1 beats Thayer-Direct on `76.1%` of normal cases and `93.1%` of
  stress cases.

### Thayer-BR v0.2 Moderate Current Best

- Training: 12,000 blends, 1,000 validation blends, batch size 8, 20 epochs,
  same 50/30/20 composition target as v0.1.
- Objective: normalized affected/core-weighted residual MSE.
- Moderate weights: affected extra weight `3`, core affected extra weight `2`.
- Normal affected-region MSE: `0.002108`, or about `32.3x` improvement.
- Hard stress affected-region MSE: `0.003847`, or about `19.6x` improvement.
- Stress core affected MSE: `0.009533`, improved from `0.013848` for v0.1.
- Worse-than-identity cases: `0/1000` normal and `0/1000` stress.
- Multi-seed improvement: `32.02 +/- 1.21x` normal and `19.55 +/- 0.30x`
  stress.

### Thayer-BR v0.2 Strong Ablation

- Strong weights: affected extra weight `5`, core affected extra weight `4`.
- Slightly improves stress core MSE relative to Moderate (`0.009344` versus
  `0.009533`).
- Worsens aggregate normal affected MSE, stress affected MSE, and stress
  non-core affected MSE relative to Moderate.
- Paper role: ablation showing that stronger weighting is not monotonically
  better.

### Evaluation Robustness Audit

- Audit run: `outputs/runs/evaluation_audit_20260708_220833`.
- Affected masks use `abs(blended - target)`, not prediction error.
- Balanced/weighted model ranking is robust across all tested affected-mask
  thresholds.
- The ranking is robust when affected masks are dilated by up to `9` pixels,
  supporting robustness to halo inclusion.
- Multi-seed evaluation supports Thayer-BR v0.2 Moderate:
  `32.02 +/- 1.21x` normal improvement and `19.55 +/- 0.30x` stress
  improvement.
- Core-obstructed pixels remain the hardest region, so qualitative figures
  should include core-overlap limitations and counterexamples.
- Residual logic audit confirms `residual = blended - target` and
  `reconstruction = blended - predicted_residual`.
- Size/visual audit finds wide apparent-size variation but weak learned-model
  affected-error dependence on size ratio.
- Halo-band audit improves in aggregate for v0.2 Moderate versus v0.1, while
  selected broad-error counterexamples remain important.

## 6. Discussion

- Thayer-Residual helps because unchanged target light can pass through the
  subtraction reconstruction path.
- Thayer-BR v0.1 helps because it makes core overlap and bright/similar-size
  contaminants more common during optimization through targeted training data.
- Thayer-BR v0.2 Moderate helps because the loss emphasizes the pixels most
  relevant to deblending: contaminated pixels and affected target-core pixels.
- Threshold is worse than identity because removing bright structures is not the
  same as reconstructing hidden target light.
- Some severe blends are easy if the contaminant is obvious; some low-severity
  blends are hard if they affect the target core.
- Remaining failures involve ambiguity, target-detail loss, over-smoothing,
  imperfect foreground extraction, and cases where different model objectives
  preserve different structure.
- Thayer-BR v0.2 Moderate shows that affected/core-weighted residual loss can
  improve the final controlled synthetic model, while stronger weighting is not
  monotonically better.
- The size/visual audit suggests apparent source size varies widely, but current
  learned-model performance is not strongly correlated with apparent size ratio.
  This supports a size-normalized benchmark as future work rather than an
  immediate invalidation of the current benchmark.
- Halo-band and visual-vs-metric examples should be used to show that lower MSE
  can still coexist with broad low-level residual artifacts in selected samples.
- Current results remain controlled synthetic results.

## 7. Future Work

- Save exact generated evaluation sets and global source indices for future
  reproducibility.
- Improve sky-background and noise realism.
- Add PSF realism and spatially varying seeing.
- Strengthen foreground extraction diagnostics.
- Evaluate current checkpoints on a size-normalized held-out benchmark that
  controls apparent target/contaminant radius ratio.
- Add explicit visual-vs-metric and halo-band diagnostics to future model
  comparisons.
- Test additional structure-aware losses only after separating aggregate gains
  from visually broad residual artifacts.
- Explore hybrid direct/residual or uncertainty-aware models.
- Evaluate on more realistic survey-style simulations before making broader
  astronomical claims.

## 8. Conclusion

Thayer-Net shows that compact U-Nets can substantially improve controlled
synthetic galaxy deblending over identity and threshold baselines. Stress
testing exposes robustness gaps in Thayer-Direct. Thayer-Residual improves
hard-case robustness, Thayer-BR v0.1 shows the value of balanced hard-case
sampling, and Thayer-BR v0.2 Moderate gives the strongest aggregate results in
the current evaluation. The main scientific lesson is that objective design,
loss weighting, and training distribution all matter, while the remaining scope
is still controlled and synthetic.

## Planned Figures

- Figure 1: pipeline diagram.
- Figure 2: Thayer-Direct vs Thayer-Residual schematic.
- Figure 3: v0.2 affected-region MSE bar chart.
- Figure 4: v0.2 normal vs stress improvement ratio.
- Figure 5: v0.2 versus v0.1 per-sample scatter.
- Figure 6: v0.2 core/non-core affected MSE comparison.
- Figure 7: multi-seed improvement summary.
- Figure 8: qualitative v0.2 success and counterexample.
- Appendix: residual/direct error-ratio histograms, blend-severity plots, and
  remaining failure cases.
- Appendix: apparent-size ratio audit, centrality/core-obstruction audit,
  halo-band error plots, and visual-vs-metric disagreement examples.

## Experiment 4 Addition

Experiment 4 adds Thayer-BR v0.2 Moderate, a weighted residual-loss run that emphasizes
affected pixels and affected target-core pixels while preserving the residual
prediction formulation. Primary run directory:
`outputs/runs/weighted_residual_20260709_030245`.

Paper role: final model improvement.

- Report the moderate `3/2` extra-weight setting as the current-best weighted
  residual model: normal affected MSE `0.002108`, stress affected MSE
  `0.003847`, stress core affected MSE `0.009533`, and no worse-than-identity
  cases.
- Include the stronger `5/4` extra-weight setting as an ablation from
  `outputs/runs/weighted_residual_20260709_043745`: it slightly improves stress
  core affected MSE to `0.009344`, but worsens normal and stress aggregate
  affected MSE relative to moderate.
- State that moderate weighting improves the final model, while stronger
  weighting shows that simply increasing the emphasis is not monotonically
  beneficial.
- Do not present the weighted model as a new architecture; it is an objective
  change on the same residual U-Net.

## Size/Visual Audit Addition

The size and visual audit is a diagnostic appendix/future-work bridge rather
than a training experiment. Primary run directory:
`outputs/runs/size_visual_audit_20260709_102251`.

Paper role: robustness audit and motivation for future size-normalized
evaluation.

- Report the apparent contaminant/target radius-ratio spread: approximately
  `0.49` at the 5th percentile, `1.06` at the median, and `2.37` at the 95th
  percentile.
- State that learned-model affected-MSE correlations with size ratio were weak
  in this audit, so the current headline is not obviously explained by a size
  shortcut.
- Note that aggregate halo-band MSE improves for Thayer-BR v0.2 Moderate versus
  Thayer-BR v0.1, while selected per-sample broad-error examples remain useful
  qualitative caveats.
- Recommend a future size-normalized held-out benchmark before making stronger
  claims about size-invariant deblending.
