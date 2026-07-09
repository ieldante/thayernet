# Paper Plan

Working title: **Thayer-Net: Controlled Galaxy Deblending with Direct,
Residual, and Balanced Residual U-Net Models**

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
improves stress performance by learning what light to subtract, and Thayer-BR
v0.1 further improves both normal and stress aggregate affected-region MSE by
changing the training distribution. The results support targeted synthetic
deblending as a useful controlled study, while remaining limited by simplified
sky, PSF, noise, and source-environment assumptions.

## 1. Introduction

- Motivate astronomical source deblending: overlapping sources bias flux,
  morphology, color, and downstream inference.
- Explain why a controlled synthetic benchmark is useful before claiming real
  survey deployment.
- State the research question: can compact U-Nets recover target galaxies from
  synthetic blends better than simple baselines, and how does robustness change
  under harder blends?
- Preview the three model experiments: Thayer-Direct, Thayer-Residual, and
  Thayer-BR v0.1.

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

### Evaluation Robustness Audit

- Audit run: `outputs/runs/evaluation_audit_20260708_220833`.
- Affected masks use `abs(blended - target)`, not prediction error.
- Thayer-BR v0.1 remains best across all tested affected-mask thresholds.
- Thayer-BR v0.1 remains best when affected masks are dilated by up to
  `9` pixels, supporting robustness to halo inclusion.
- Multi-seed evaluation keeps Thayer-BR v0.1 best on all tested seeds:
  `27.04 +/- 1.04x` normal improvement and `15.76 +/- 0.07x` stress
  improvement.
- Core-obstructed pixels remain the hardest region, so qualitative figures
  should include core-overlap limitations and counterexamples.
- Residual logic audit confirms `residual = blended - target` and
  `reconstruction = blended - predicted_residual`.

## 6. Discussion

- Thayer-Residual helps because unchanged target light can pass through the
  subtraction reconstruction path.
- Thayer-BR v0.1 helps because it makes core overlap and bright/similar-size
  contaminants more common during optimization through targeted training data.
- Threshold is worse than identity because removing bright structures is not the
  same as reconstructing hidden target light.
- Some severe blends are easy if the contaminant is obvious; some low-severity
  blends are hard if they affect the target core.
- Remaining failures involve ambiguity, target-detail loss, over-smoothing,
  imperfect foreground extraction, and cases where different model objectives
  preserve different structure.
- Current results remain controlled synthetic results.

## 7. Future Work

- Save exact generated evaluation sets and global source indices for future
  reproducibility.
- Improve sky-background and noise realism.
- Add PSF realism and spatially varying seeing.
- Strengthen foreground extraction diagnostics.
- Test affected-region-weighted or structure-aware losses.
- Explore hybrid direct/residual or uncertainty-aware models.
- Evaluate on more realistic survey-style simulations before making broader
  astronomical claims.

## 8. Conclusion

Thayer-Net shows that compact U-Nets can substantially improve controlled
synthetic galaxy deblending over identity and threshold baselines. Stress
testing exposes robustness gaps in Thayer-Direct. Thayer-Residual improves
hard-case robustness, and Thayer-BR v0.1 gives the strongest aggregate results
in the current evaluation. The main scientific lesson is that both objective
design and training distribution matter, while the remaining scope is still
controlled and synthetic.

## Planned Figures

- Figure 1: pipeline diagram.
- Figure 2: Thayer-Direct vs Thayer-Residual schematic.
- Figure 3: affected-region MSE bar chart.
- Figure 4: normal vs stress improvement ratio.
- Figure 5: Thayer-Direct vs Thayer-BR v0.1 scatter.
- Figure 6: Thayer-Residual vs Thayer-BR v0.1 scatter.
- Figure 7: stress performance by core overlap bin.
- Figure 8: qualitative success and failure examples.
- Appendix: residual/direct error-ratio histograms, blend-severity plots, and
  remaining failure cases.
