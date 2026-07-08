# Paper Plan

This plan captures the current paper narrative after the direct U-Net,
hard-case stress-test, and residual U-Net checkpoints. The eventual paper should
remain honest about scope: this is controlled synthetic deblending, not
deployment-ready survey deblending.

## Proposed Paper Arc

1. Introduce a controlled galaxy deblending benchmark built from Galaxy10 DECaLS
   cutouts with known targets and synthetic contaminant foregrounds.
2. Explain why foreground-only synthetic blending is used: it avoids rectangular
   artifacts from naive cutout addition and makes target/reconstruction metrics
   possible.
3. Train a compact direct U-Net (`blended -> target`) and show that it beats
   identity and threshold baselines on normal held-out blends.
4. Add hard-case stress testing to probe smaller shifts, brighter contaminants,
   similar-or-larger contaminant sizes, and target-core overlap.
5. Show that stress testing lowers direct-model improvement from about 14.13x to
   about 8.04x affected-region MSE improvement versus identity.
6. Train a residual U-Net (`blended -> residual`, reconstruction by subtraction)
   and show improved aggregate robustness, especially on the stress set.
7. Analyze remaining failures: ambiguity, core overlap, over-smoothing, and
   target-detail loss.
8. Discuss future work: balanced hard-case training, affected-region-weighted
   losses, improved foreground extraction checks, and more realistic sky
   simulation.

## Results to Report

| Model | Normal affected MSE | Normal improvement | Stress affected MSE | Stress improvement |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0.062555 | 1.00x | 0.075541 | 1.00x |
| Direct U-Net | 0.004428 | 14.13x | 0.009390 | 8.04x |
| Residual U-Net | 0.004039 | 15.49x | 0.007069 | 10.69x |

Important comparison notes:

- Residual prediction improves stress affected-region MSE from 0.009390 to
  0.007069.
- Residual prediction reduces stress worse-than-identity cases from 13/1000 to
  0/1000.
- Residual beats direct on 667/1000 stress cases and 310/800 normal cases.
- Direct still beats residual on some individual cases and has slightly better
  normal affected-region MAE.

## Planned Sections

### Abstract

Summarize the controlled benchmark, foreground-only blending, direct U-Net
baseline, stress-test degradation, residual-prediction improvement, and remaining
limitations.

### Introduction

Motivate astronomical deblending and explain why a controlled synthetic benchmark
is useful before moving toward more realistic survey simulations.

### Dataset

Describe Galaxy10 DECaLS, local data handling, normalization to `[0, 1]`, and
train/validation/test splitting before blend generation.

### Synthetic Blending Method

Explain foreground extraction, halo-aware masking, shifts, brightness scaling,
blur/noise, optional rotation, and why rectangular cutout artifacts are avoided.

### Baselines

Define identity and threshold/connected-component baselines. Explain why
affected-region metrics are needed to make identity a meaningful reference.

### Models

Describe the compact U-Net architecture and the two objectives:

- Direct reconstruction: `blended -> target`
- Residual prediction: `blended -> blended - target`; reconstruction by
  subtraction

### Evaluation Metrics

Describe whole-image MSE, MAE, PSNR, SSIM, affected-region masked MSE/MAE, and
improvement ratio versus identity.

### Terminology

Separate:

- legacy `generation_difficulty`
- measured `blend_severity_score` and `blend_severity_bin`
- `core_obstruction_fraction` and `core_overlap_bin`
- `model_failure_score`
- `model_improvement_ratio`

Make clear that high blend severity is not identical to model difficulty.

### Results

Report normal held-out results, stress-test results, and direct-vs-residual
comparison tables.

### Failure Analysis

Discuss qualitative examples and per-sample comparisons. Focus on ambiguity,
core obstruction, target-detail loss, direct wins over residual, and residual
stress robustness.

### Limitations

Discuss synthetic-blend scope, foreground-extraction assumptions, missing PSF and
sky realism, dataset limits, and model capacity.

### Conclusion

Summarize the current claim: compact U-Nets can substantially improve
controlled synthetic deblending over simple baselines, stress testing exposes
harder overlap cases, and residual prediction improves aggregate robustness
without solving every individual case.

## Planned Figures

- Pipeline diagram: data split, foreground extraction, blend generation,
  training, evaluation.
- Direct U-Net architecture schematic.
- Affected-region MSE bar chart.
- Normal vs stress improvement ratio chart.
- Direct vs residual per-sample affected-MSE scatter plot.
- Residual/direct error-ratio histogram.
- Qualitative residual success example.
- Qualitative direct-better example.
- Qualitative failure example focused on ambiguity/core overlap.
- Blend severity vs model affected-MSE plot.
- Core obstruction vs model-improvement-ratio plot.

## Future Work

- Balanced hard-case residual training.
- Improved preprocessing and foreground extraction diagnostics.
- Core-obstruction-balanced evaluation.
- Affected-region-weighted or structure-aware losses.
- More realistic sky background, PSF, noise, and source-crowding simulation.
- Evaluation across morphology labels and source-size regimes.
