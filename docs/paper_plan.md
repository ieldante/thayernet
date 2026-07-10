# Paper Plan

Working title: **Thayer-Net: Controlled Galaxy Deblending with Direct,
Residual, Balanced, and Weighted Residual U-Net Models**

The paper should remain honest about scope. The project is a controlled
synthetic deblending benchmark, not a deployment-ready survey pipeline.

## Abstract

We present Thayer-Net, a controlled synthetic benchmark for galaxy deblending
with Galaxy10 DECaLS cutouts. The benchmark constructs foreground-only blends
with known target references, reducing rectangular cutout artifacts from naive image
addition. We compare identity and threshold baselines with compact U-Net models
trained for direct target reconstruction, residual correction-field prediction, and
balanced hard-case residual prediction. Thayer-Direct improves
affected-region MSE over identity on original normal development blends, but stress testing
reveals weaker robustness under harder overlap conditions. Thayer-Residual
improves stress performance by learning a blend-to-target field to subtract. Thayer-BR v0.1
shows that balanced hard-case training improves robustness, and Thayer-BR v0.2 Moderate
further improves affected and core reconstruction with an
affected/core-weighted residual loss. The results support targeted synthetic
deblending as a useful controlled study, while remaining limited by simplified
sky, PSF, noise, source-environment, and apparent-size-normalization
assumptions. Current metrics come from development suites with confirmed
duplicate-object leakage across the historical random-index split. A
duplicate-safe grouped retrain remains strong, with `28.81x` and `15.80x`
lower affected-region MSE than identity on grouped normal and hard-stress
development suites, respectively, but a new untouched locked final test is
still required before reporting a final effect estimate.

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
- Historical train/validation/test arrays were created by random row-index
  splitting before blending. This prevents exact index reuse but does not avoid
  duplicate-object leakage; an RA/Dec audit found groups crossing splits.
- The grouped protocol unions exact-pixel and exact-coordinate matches before
  partition assignment, verifies zero source/group-role crossing, and saves
  source indices, group IDs, seeds, blend parameters, and replay/code hashes.
- Blends are generated from target and contaminant images within the same split.
- Foreground-only contaminant extraction reduces rectangular cutout artifacts and
  double-background problems.
- Halo-aware masks preserve diffuse contaminant light.
- Galaxy10 DECaLS inputs are RGB display cutouts, not calibrated FITS flux
  images.
- The generator adds normalized display RGB and clips the blend; it does not
  perform calibrated band-flux injection.
- Limitations: historical source duplication, synthetic foreground
  approximation, target-centrality cues, asymmetric apparent-size handling,
  no pixel-scale-aware angular normalization, no full sky/PSF realism, and no
  physically correlated source environments.

## 3. Methods

- Identity and threshold are sanity checks, not competitive astronomical
  deblenders.
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

- Grouped normal development evaluation; original row-split normal results are
  retained separately as historical context.
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
- Define the affected region as a prediction-independent blend-change or
  correction-field mask, not a pure contaminant-flux mask; it can include
  target-blur, input-clipping, and noise changes.
- Report macro per-sample masked means as primary, label micro/pooled values
  separately, and report nonempty-mask coverage for every regional metric.
- Report model-improvement ratio and worse-than-identity counts.
- Analyze by generation difficulty, blend severity, core obstruction, and model
  failure without treating those concepts as interchangeable.
- Include the evaluation robustness audit:
  - affected-mask threshold sensitivity at `0.005`, `0.01`, `0.02`, and `0.04`
  - affected-mask dilation/halo sensitivity at `0`, `1`, `3`, `5`, and `9`
    pixels
  - three normal and three stress blend-generation/evaluation seeds, 1,000
    blends per seed; these are not independent training seeds
  - residual sign/reconstruction audit
  - v0.2 multi-seed audit
  - size/visual audit, including apparent-size ratio, halo-band error, and
    visual-vs-metric disagreement examples
  - source-partition, exact-duplicate, and near-duplicate audit
  - clean-target preservation and clipped/unclipped reconstruction audits
- Treat all historical and grouped normal/stress suites as development
  benchmarks.
- Evaluate the final paper claim once on a frozen, duplicate-aware manifest
  that was not used for architecture, loss, or example selection.

## 5. Results

### Thayer-Direct Baseline

- Earlier normal evaluation: Thayer-Direct affected-region MSE `0.004428` versus
  identity `0.062555`, or 14.13x lower affected-region MSE.
- Current 1,000-blend comparable normal table: Thayer-Direct affected-region MSE
  `0.004236`, or 16.08x lower affected-region MSE versus identity.

### Stress Degradation

- Hard stress identity affected-region MSE: `0.075541`.
- Thayer-Direct stress affected-region MSE: `0.009390`.
- Thayer-Direct stress affected-MSE ratio versus identity: `8.04x` lower.
- Interpretation: Thayer-Direct still works, but harder overlap reduces
  robustness.

### Thayer-Residual Improvement

- Thayer-Residual stress affected-region MSE: `0.007069`.
- Thayer-Residual stress affected MSE is `10.69x` lower than identity.
- Worse-than-identity stress cases fall from `13/1000` for Thayer-Direct to
  `0/1000` for Thayer-Residual.
- Thayer-Direct remains better on some individual cases.

### Thayer-BR v0.1 Improvement

- Training: 8,000 blends, 1,000 validation blends, batch size 8, 20 epochs,
  best validation loss `0.000378` at epoch 18.
- Current normal affected-region MSE: `0.002451`, or `27.79x` lower than
  identity.
- Hard stress affected-region MSE: `0.004587`, or `16.47x` lower than identity.
- Worse-than-identity stress cases: `0/1000`.
- Thayer-BR v0.1 beats Thayer-Residual on `91.3%` of normal cases and `87.9%`
  of stress cases.
- Thayer-BR v0.1 beats Thayer-Direct on `76.1%` of normal cases and `93.1%` of
  stress cases.

### Thayer-BR v0.2 Moderate Historical Development Result

- Training: 12,000 blends, 1,000 validation blends, batch size 8, 20 epochs,
  same 50/30/20 composition target as v0.1.
- Objective: normalized affected/core-weighted residual MSE.
- Moderate weights: affected extra weight `3`, core affected extra weight `2`.
- Normal affected-region MSE: `0.002108`, 32.3x lower than identity MSE
  (about 5.7x lower RMSE).
- Hard stress affected-region MSE: `0.003847`, 19.6x lower than identity MSE
  (about 4.4x lower RMSE).
- Stress core affected MSE: `0.009533`, improved from `0.013848` for v0.1.
- Worse-than-identity cases: `0/1000` normal and `0/1000` stress.
- Evaluation-seed affected-MSE ratio: `32.02 +/- 1.21x` normal and
  `19.55 +/- 0.30x` stress; no training-seed robustness claim.
- Label every value in this subsection as an original development-split result,
  not the duplicate-safe or final estimate.

### Duplicate-Safe Grouped v0.2 Retrain

- Exact-pixel and exact-coordinate source groups remain wholly within one
  train, validation, or test partition; every grouped blend manifest row is
  replay-verified.
- Normal affected MSE: `0.00231890`, or `28.8127x` lower than identity, with
  `0/1000` worse-than-identity cases.
- Hard-stress affected MSE: `0.00458983`, or `15.8025x` lower than identity,
  with `3/1000` worse-than-identity cases.
- Compact-bright affected MSE: `0.00872771`, or `9.18304x` lower than identity,
  with `2/1000` worse-than-identity cases.
- High-core-obstruction affected MSE: `0.00491680`, or `15.8378x` lower than
  identity, with `1/1000` worse-than-identity case.
- Interpret this as strong duplicate-safe development performance that is lower
  than the original headline, not as a locked final-paper estimate.
- Note the training-budget confound: the grouped retrain uses 8,000 training
  blends, whereas the original v0.2 Moderate run used 12,000.

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
- Thayer-BR v0.1 remained best among methods in the earlier threshold and
  dilation audit. v0.2 was not included, and Direct/Residual ordering changes
  at some dilation radii; do not cite this as direct v0.2 mask robustness.
- Evaluation-seed variation supports the development ranking of Thayer-BR v0.2
  Moderate: `32.02 +/- 1.21x` lower normal affected MSE and
  `19.55 +/- 0.30x` lower stress affected MSE versus identity. It does not test
  independent retraining.
- Core-obstructed pixels remain the hardest region, so qualitative figures
  should include core-overlap limitations and counterexamples.
- Residual logic audit confirms `residual = blended - target` and
  `reconstruction = blended - predicted_residual`.
- Size/visual audit finds wide apparent-size variation but weak learned-model
  affected-error dependence on size ratio.
- Halo-band audit improves in aggregate for v0.2 Moderate versus v0.1, while
  selected broad-error counterexamples remain important.

### Benchmark-Defensibility Audits

- Report the major blocker: 29 raw-pixel-identical pairs and 27 exact-coordinate
  pairs cross the historical train/validation/test split.
- Separate the passing row-index/role-containment audit from the failing
  object-level duplicate audit.
- Quantify the known leak: the union implicates 57/17,736 sources (`0.321%`),
  13 historical normal rows, and 12 historical stress rows. Clean-subset
  affected-MSE ratios change by no more than `0.31%`, so its measured aggregate
  effect is minor even though the protocol defect is major.
- Treat all existing model tables as development results. Do not present them
  as the locked-final effect estimate.
- Mark the provisional five-suite final pool as superseded: 590 sources entered
  grouped train/validation (`499/91`). Select a fresh untouched group-disjoint
  final pool after model and protocol freeze.
- Explain that `54.575%` of the grouped evaluation rows expose the old
  checkpoint to a historical train/validation source group. The old checkpoint
  is therefore an exposure-confounded diagnostic even though its clean-neither
  ratios remain `31.53x`, `18.18x`, `11.68x`, and `18.27x` across the four
  grouped suites.
- Include the unblended preservation result: the Delta preservation ablation
  has substantially lower null MSE, while v0.2 has a three-case high-error tail
  with visible false subtraction.
- Report unaffected-region target error together with model output change
  versus the blend and paired excess error over identity; the mask complement
  is not pure model damage.
- Report clipped and unclipped output metrics. Aggregate affected MSE changes
  by at most 0.16%, and none of 6,000 model/sample rows changes by more than 10%
  relatively. Separately report that clipping the additive RGB blend input is
  material to task construction and can destroy component information.
- Present source-artifact flags as a 356-image manual-review pool, not a clean
  exclusion list or an estimate of artifact prevalence.

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
  improve the best original development-split model, while stronger weighting is not
  monotonically better.
- The size/visual audit suggests apparent source size varies widely, but current
  learned-model performance is not strongly correlated with apparent size ratio.
  This supports a size-normalized benchmark as future work rather than an
  immediate invalidation of the current benchmark.
- Halo-band and visual-vs-metric examples should be used to show that lower MSE
  can still coexist with broad low-level residual artifacts in selected samples.
- Current results remain controlled synthetic development results. Confirmed
  cross-split duplicated sources invalidate a final generalization claim from
  the historical suites. The grouped retrain repairs the demonstrated split
  mechanism and remains strong, but its evaluation suites also informed
  validation of the corrected pipeline and are not locked-final.

## 7. Future Work

- Extend the completed exact-pixel/coordinate grouping only for reviewed,
  high-confidence near-duplicates.
- After protocol and model freeze, select a fresh untouched group-disjoint
  final source pool; do not reuse the superseded provisional pool.
- Do not use locked-final examples or metrics for further model selection.
- Run independent grouped training seeds; two seeds would be preliminary, not
  a full training-seed robustness claim.
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

On the development suites, Thayer-Net shows that compact U-Nets lower
affected-region reconstruction error relative to identity and threshold sanity
checks. Stress
testing exposes robustness gaps in Thayer-Direct. Thayer-Residual improves
hard-case robustness, Thayer-BR v0.1 shows the value of balanced hard-case
sampling, and Thayer-BR v0.2 Moderate gives the strongest aggregate historical
results. The duplicate-safe grouped retrain preserves large gains (`28.81x`
normal and `15.80x` hard stress) while falling below the original development
headline. The main scientific lesson is that objective design, loss weighting,
training distribution, and split provenance all matter, while the remaining
scope is still controlled and synthetic. Final effect sizes should be withheld
until a fresh duplicate-aware locked test is run once.

## Planned Figures

- Figure 1: pipeline diagram.
- Figure 2: Thayer-Direct vs Thayer-Residual schematic.
- Figure 3: grouped-retrain normal/hard/compact/high-core affected-MSE ratios
  with explicit development-only status.
- Figure 4: historical original-development v0.2 affected-region MSE chart.
- Figure 5: historical original-development v0.2 normal/stress ratio.
- Figure 6: historical original-development v0.2 versus v0.1 diagnostics.
- Figure 7: evaluation-seed identity/model affected-MSE ratio summary; the
  checkpoints were not independently retrained.
- Figure 8: qualitative v0.2 success and counterexample.
- Figure 9: grouped-retrain affected/core/halo metrics with
  worse-than-identity counts.
- Figure 10: original development result, exposure-confounded old checkpoint,
  clean-neither diagnostic, and grouped retrain shown with status labels rather
  than as an unqualified ranking.
- Appendix: residual/direct error-ratio histograms, blend-severity plots, and
  remaining failure cases.
- Appendix: apparent-size ratio audit, centrality/core-obstruction audit,
  halo-band error plots, and visual-vs-metric disagreement examples.

## Experiment 4 Addition

Experiment 4 adds Thayer-BR v0.2 Moderate, a weighted residual-loss run that emphasizes
affected pixels and affected target-core pixels while preserving the residual
prediction formulation. Primary run directory:
`outputs/runs/weighted_residual_20260709_030245`.

Paper role: original development-model improvement, now accompanied by a
completed duplicate-safe grouped retrain but still pending a fresh locked final
evaluation.

- Report the moderate `3/2` extra-weight setting as the current-best weighted
  residual model: normal affected MSE `0.002108`, stress affected MSE
  `0.003847`, stress core affected MSE `0.009533`, and no worse-than-identity
  cases.
- Include the stronger `5/4` extra-weight setting as an ablation from
  `outputs/runs/weighted_residual_20260709_043745`: it slightly improves stress
  core affected MSE to `0.009344`, but worsens normal and stress aggregate
  affected MSE relative to moderate.
- State that moderate weighting improves the original development model, while stronger
  weighting shows that simply increasing the emphasis is not monotonically
  beneficial.
- Do not present the weighted model as a new architecture; it is an objective
  change on the same residual U-Net.

## Research-Correctness Audit and Grouped Retrain Addition

Paper role: infrastructure validation and corrected development estimate.

- Preserve the 32.3x/19.6x result as historical development evidence and label
  the exact duplicate/source-group limitation beside it.
- Lead the corrected estimate with the grouped retrain's `28.8127x` normal,
  `15.8025x` hard, `9.18304x` compact-bright, and `15.8378x` high-core ratios.
- State that the known duplicate set has a minor measured aggregate effect but
  represents a major protocol error.
- Do not use the old checkpoint's better grouped-suite score as the corrected
  result because 54.575% of rows carry historical source-group exposure.
- Describe exact manifest replay, checkpoint/code hashes, MPS device logging,
  macro/micro distinction, and regional coverage as reproducibility controls.
- Require a fresh group-disjoint final pool after freeze; neither the grouped
  development suites nor the superseded provisional pool is final.

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
  in this audit, so the historical development headline is not obviously
  explained by a size shortcut.
- Note that aggregate halo-band MSE improves for Thayer-BR v0.2 Moderate versus
  Thayer-BR v0.1, while selected per-sample broad-error examples remain useful
  qualitative caveats.
- Recommend a future size-normalized held-out benchmark before making stronger
  claims about size-invariant deblending.

## Experiment 5 Addition

Experiment 5 adds Thayer-BR v0.3 Color/Structure Candidate. Primary run
directory: `outputs/runs/br_v03_delta_color_20260709_185630`.

Paper role: ablation/tradeoff, not current best.

- State that v0.3 keeps the same residual U-Net architecture as v0.2 Moderate
  and changes the objective with low-weight reconstruction, affected/core,
  gradient, differentiable RGB chroma/color-direction, and halo-band terms.
- Report that CIEDE2000 was implemented with scikit-image for evaluation only;
  the training loss used differentiable RGB color proxies instead of exact
  Delta E. Note that Galaxy10 DECaLS RGB images are survey composites, so
  Delta E is visual-quality evidence rather than the primary scientific metric.
- Main result: v0.3 worsens normal affected MSE (`0.002590` versus `0.002025`)
  and hard-stress affected MSE (`0.004772` versus `0.003648`) relative to
  Thayer-BR v0.2 Moderate, so it should not replace the current best model.
- Useful targeted result: compact-bright affected MSE improves slightly
  (`0.006325` versus `0.006514`) and compact-bright gradient error improves,
  but color/Delta E is worse there. Present this as a compact-contaminant
  ablation, not as a general improvement.
- Include v0.3 qualitative grids only as diagnostic examples: success over
  v0.2, failure/tradeoff, v0.2 counterexample, direct counterexample, compact
  contaminant case, muted-color case, and halo-band comparison.
- Do not update README or current-best claims unless a future repeated-seed
  variant improves primary affected/core metrics and visual/color metrics.

### Delta Follow-up Ablation/Tradeoff

The stronger-color Delta follow-up is in
`outputs/runs/br_v03_delta_candidate_20260710_031425`. Paper role:
ablation/tradeoff, not current best.

- Report that Delta increased the color-proxy weight from `0.05` to `0.10` and
  improved on the first v0.3 candidate in aggregate affected MSE and Delta E
  2000 on all shared suites.
- Relative to v0.2 Moderate, report the primary regressions: normal affected
  MSE `0.002275` versus `0.002025`, hard-stress affected MSE `0.004035` versus
  `0.003648`, and one versus zero stress worse-than-identity cases.
- Report the targeted gains: compact-bright affected MSE `0.005428` versus
  `0.006514`, improved Lab chroma and halo-band MSE across all six suites, and
  improved Delta E on normal and compact-bright samples.
- Pair the muted-color improvement grid with the Delta color-artifact grid and
  retain the v0.2/direct counterexamples. These figures support a visual
  tradeoff rather than a general ranking change.
- Keep Thayer-BR v0.2 Moderate as current best.

## Experiment 6 Addition

Experiment 6 adds Thayer-ResUNet v0.4 Candidate. Primary run directory:
`outputs/runs/resunet_v04_candidate_20260710_043109`.

Paper role: architecture ablation with targeted compact/halo/color gains, not
current best.

- Describe the residual-block U-Net as a controlled small architecture change:
  `2,014,595` parameters versus `1,927,075` for the standard v0.2 U-Net, or
  `+4.54%`, with the same residual target and v0.2 Moderate weighted loss.
- Report the training scale and outcome: 8,000/1,000 train/validation blends,
  20 epochs, best validation loss `0.001076` at epoch 19, and final
  train/validation loss `0.000792 / 0.001082`.
- Lead the result with the strict tradeoff: normal affected MSE was essentially
  tied and slightly better (`0.002118` versus `0.002132`), but stress affected
  MSE was slightly worse (`0.003950` versus `0.003929`), stress core MSE was
  2.7% worse, and stress added one worse-than-identity case.
- Report targeted gains without promoting them: compact-bright affected MSE
  improved by about 20.7%; halo-band MSE improved across all six suites; and
  affected Delta E improved across all suites despite no color loss.
- Report the core limitations: high-core-obstruction core MSE worsened by about
  5.5%, and color-saturation core MSE worsened by about 11.3%.
- Include the same-run affected/core/halo charts, scatter and ratio histogram,
  plus the ResUNet improvement, failure/tradeoff, and direct-still-wins grids.
- State explicitly that the verdict is `architecture_ablation` and that
  Thayer-BR v0.2 Moderate remains current best.
- State that Core+ and Halo-safe were not run because the baseline architecture
  failed the preregistered Part B2 promotion gate; this was the controlled
  stopping decision, not a missing result.

## Clean Benchmark Plan Addition

Clean-source-filtered and artifact-heavy suites were unavailable in Experiments
5 and 6 because the dataset has no validated source-quality flags. Treat
`docs/clean_benchmark_plan.md` as a separate non-training future benchmark:
it proposes blinded source-quality filtering, artifact-stress pools,
size-matched evaluation, and a stratified compact-contaminant suite without
altering historical results or the current-best claim.
