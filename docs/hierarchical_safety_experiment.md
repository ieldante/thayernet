# Hierarchical safety experiment

This gated campaign asks whether a hierarchical, metric-specific safety policy
can reject invalid queries and rank the tail risk of uniquely valid frozen
Condition-C reconstructions without changing the successful reconstructor.

## Frozen scientific boundary

The overall Thayer-Select project does not change. Condition C is loaded from
its original best checkpoint, placed in evaluation mode, and has zero trainable
parameters. No encoder or decoder fine-tuning is permitted. Historical source
splits remain exact. The sealed lockbox remains unavailable for generation,
inspection, calibration, debugging, selection, or evaluation.

## Fresh populations

- Dataset Q training: 15,000 scenes, approximately 5,000 each UNIQUE_VALID,
  NULL, and AMBIGUOUS.
- Dataset Q validation: 2,000 balanced scenes, with both stratified and
  inverse-weighted natural-mixture summaries.
- Dataset R training: 15,000 UNIQUE_VALID scenes: 50% natural, 20% low SNR,
  15% high overlap/core obstruction, and 15% equal-flux/similar-size or
  confusion-prone.
- Dataset R validation: 2,000 fresh UNIQUE_VALID scenes from the natural valid
  distribution, with regime-stratified summaries.
- Natural calibration: 6,000 fresh scenes from the intended deployment mixture;
  it alone supplies operational calibration and thresholds.
- Stratified calibration diagnostic: 3,000 fresh balanced query/failure-regime
  scenes; it supplies plots only.
- Fresh development: 3,000 scenes, generated only after policy freeze and
  evaluated exactly once.

Targeted generator variables create training coverage but never enter a
deployable head. Every row stores scene and source IDs/groups, positions,
prompt, query state, match, seeds, PSF/noise settings, isolated/blend/prompt/
reconstruction hashes, continuous risks, confusion, stratum, and inverse
sampling weight. Deterministic replay and zero cross-partition source-group
leakage are hard gates.

## Feature families

- F_GLOBAL: pooled frozen bottleneck.
- F_PROMPT_LOCAL: Gaussian-weighted pooling around the prompt at encoder scales
  enc1, enc2, and bottleneck, with prompt coordinates scaled to each grid.
- F_RECON_SUMMARY: predicted per-band flux, concentration, centroid relative to
  prompt, output energy, and reconstruction/input local contrast.
- F_COMBINED: concatenation of the three families.

Truth images, oracle error, source identity, true SNR/separation/flux ratio,
generator stratum, and contract outcomes are forbidden deployable features.
MPS performs frozen neural inference; CPU performs head fitting, calibration,
statistics, tables, and figures. CPU neural fallback is an error.

## Gates and evaluation

The query gate must remove the ambiguity-over-valid inversion and show stable,
meaningful NULL and AMBIGUOUS recall across five lightweight head seeds. If it
fails, risk-policy evaluation stops. Valid-only risk heads report MAE, Spearman,
pinball loss, quantile coverage, top-risk recall, catastrophic AUPRC, tail
ranking, and seed stability against random, output-energy, monolithic R1, and
non-deployable oracle diagnostics.

After natural calibration and policy freeze, development compares reconstruction
only, original R1, query gate only, and the full hierarchy. Results remain
class-conditional and use all-source-group clustered bootstrap intervals.
Success, partial success, and failure retain the preregistered definitions in
the campaign contract; they cannot be redefined after development.
