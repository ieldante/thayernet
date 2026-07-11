# Thayer-Select ambiguity benchmark plan

Ambiguity is evaluated after freezing the model, normalization, reconstruction
metrics, and calibration rule. It is not defined by generator difficulty.

## Required stress families

- Query source A and source B on the byte-identical blend; swapping the prompt
  should swap identity-consistent reconstruction rather than merely perturb it.
- Empty prompt and wrong-prompt cases measure no-source hallucination and false
  subtraction.
- Isolated-source/no-harm queries measure damage when no deblending is needed.
- Source-swap consistency checks whether A resembles B or vice versa under
  centroid, flux, color, and morphology metrics.
- Near-identical observable `(blend,prompt)` pairs with materially divergent
  requested-source truths probe irreducible ambiguity.
- Two-, three-, and four-source scenes span separation in PSF units, core
  obstruction, source visibility/count, flux/size ratios, morphology/color
  similarity, and SNR.

Explanatory scene variables are reported only as stratifiers. Empirical outcomes
are reconstruction, flux, color, shape, and centroid errors; catastrophic
failure; false subtraction; omission; source swap; no-source hallucination; and
isolated-source damage. Model predictions are pixel uncertainty, predicted
probability of acceptable reconstruction, and the resulting accept/abstain
score. Generator variables are never called recoverability.

## Calibration and leakage control

Only the calibration partition maps a raw score to probability and selects
predeclared risk/coverage thresholds. The development-test partition reports
those frozen choices. Neither it nor the sealed lockbox may revise thresholds.
Reports include risk–coverage curves, accepted-case confidence intervals,
catastrophic/false-subtraction rates, and subgroup coverage.

CatSim-to-COSMOS generator shift is evaluated only after a licensed/local
RealGalaxy sample exists and source groups are disjoint. DR10 has no isolated
truth, so it supports only output stability, artifact sensitivity,
hallucination/false-large-source behavior, distribution shift, abstention, and
qualitative failure modes—not supervised reconstruction accuracy.

Pixel uncertainty, probability of acceptable reconstruction, and an operational
accept/abstain decision must always be reported separately.
