# Experiment Log

Use this file to record experiment runs as the project progresses. Entries should include enough detail to make results interpretable without storing generated outputs in git.

## Entry Template

- Date:
- Code state:
- Dataset path:
- Split seed:
- Blend settings:
- Number of blends:
- Model settings:
- Training settings:
- Baseline metrics:
- Model metrics:
- Difficulty-bin analysis:
- Notes and failure cases:

## Current Status

The first direct-reconstruction checkpoint has been run and summarized below. Raw output tables, checkpoints, and generated intermediate files remain local under `outputs/` and are not committed.

## 2026-07-08 Direct U-Net Checkpoint

### Setup

- Task: reconstruct the clean target RGB image directly from the blended RGB image.
- Dataset: Galaxy10 DECaLS, loaded locally from `data/Galaxy10_DECals.h5`.
- Split seed: 42.
- Train blends: 5,000.
- Validation blends: 800.
- Held-out test blends: 800.
- Epochs: 20.
- Model: compact U-Net.
- Rotation: disabled for the main checkpoint.

### Headline Test Metrics

| Metric | Identity Baseline | Direct U-Net | Change |
| --- | ---: | ---: | ---: |
| Whole-image MSE | 0.005224 | 0.000566 | ~9.2x lower |
| Whole-image SSIM | 0.964264 | 0.976648 | higher |
| Affected-region MSE | 0.062555 | 0.004428 | ~14.1x lower |
| Affected-region MAE | 0.180238 | 0.041573 | ~4.3x lower |
| Mean affected-region fraction | 0.076684 | 0.076684 | mask definition |

Affected-region metrics use pixels where the blend differs from the target. They are more diagnostic for deblending than whole-image metrics because most pixels in each synthetic image remain unchanged.

### Original Difficulty Breakdown

| Original label | n | Identity affected MSE | Model affected MSE | Identity affected MAE | Model affected MAE | Mean mask fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 436 | 0.053210 | 0.003142 | 0.168328 | 0.035911 | 0.074178 |
| medium | 349 | 0.074438 | 0.005735 | 0.195235 | 0.047437 | 0.079736 |
| hard | 15 | 0.057729 | 0.011385 | 0.177522 | 0.069711 | 0.078518 |

The original easy/medium/hard labels are useful but crude. The hard bin is small in this run, and measured image-level severity appears more informative for later analysis.

### Measured Severity Notes

Measured severity labels split the 800 held-out examples almost evenly:

| Measured label | n |
| --- | ---: |
| easy | 267 |
| medium | 266 |
| hard | 267 |

The crosstab between original and measured labels shows substantial disagreement. For example, original `easy` examples split into 172 measured easy, 141 measured medium, and 123 measured hard cases. This supports using measured affected-region statistics in the final analysis rather than relying only on sampled blend parameters.

### Qualitative Notes

- The model removes visually separable contaminants well in many examples.
- Partial failures occur when contaminant light overlaps target structure and the model suppresses target detail along with the contaminant.
- Hard failures remain possible for heavily overlapping or visually ambiguous blends.
- Public-safe selected figures:
  - `reports/figures/direct_unet_success.png`
  - `reports/figures/direct_unet_partial_failure.png`

### Follow-Up

- Run a balanced hard-case stress test using measured severity bins.
- Compare the direct-reconstruction model with a residual-prediction variant.
- Add final tables and figure references to the formal report after the experimental set is complete.
