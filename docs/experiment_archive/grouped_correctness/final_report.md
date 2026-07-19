# Thayer-Net Research Correctness Audit: Final Report

Date: 2026-07-10  
Branch: `thayer-br-0.3`  
Master audit run: `outputs/runs/research_correctness_audit_20260710_092241/`

## Executive verdict

The corrected pipeline is trustworthy for a controlled, source-group-disjoint
**development benchmark**. The audit found no target/mask/parameter tensor
leakage into model inputs, no residual-sign inversion, no affected-MSE
arithmetic failure, no comparator realignment error, and no silent CPU fallback
in the grouped training or evaluations. Metric unit checks, grouped role
containment, and exact blend/mask replay all pass.

The historical row-index split was not scientifically valid as a
source-independent split: 29 pixel-identical and 27 exact-coordinate pairs
crossed partitions. This implicated 57/17,736 sources (0.321%). Its measured
aggregate effect was minor—excluding implicated historical evaluation rows
changed the reported improvement ratios by at most about 0.31%—but the protocol
defect was major.

The old 32.3x normal result is still numerically plausible, not abandoned.
However, it remains an original development-split result. A grouped v0.2
Moderate retrain completed and remains strong at 28.81x lower normal
affected-region MSE than identity. It is the defensible current development
estimate, not final-paper performance.

One final-claim blocker remains. The earlier provisional 1,000-source “final”
pool is superseded because the later grouped resplit/retrain reused 590 of its
sources in train or validation. There is no untouched final source pool left in
the current protocol. The optional second training seed was therefore not
launched.

## Headline grouped-development result

| Suite | Identity affected MSE | Grouped v0.2 affected MSE | Identity/model MSE ratio | Core affected MSE | Halo-band MSE | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 0.0668139 | 0.00231890 | 28.8127x | 0.00497364 | 0.000435626 | 0/1000 |
| Hard stress | 0.0725308 | 0.00458983 | 15.8025x | 0.0115079 | 0.000640123 | 3/1000 |
| Compact bright | 0.0801469 | 0.00872771 | 9.18304x | 0.0118618 | 0.000778985 | 2/1000 |
| High core obstruction | 0.0778714 | 0.00491680 | 15.8378x | 0.0123239 | 0.000548833 | 1/1000 |

These are clipped macro per-sample values. All clipped/unclipped, macro/micro,
valid-count, whole-image, SSIM, PSNR, sign, and clipping statistics remain in
`tables/grouped_retrain_suite_metrics.csv`. Output clipping changes grouped
affected MSE by only about 0.22%–0.35%, so it does not materially create the
conclusion. Composite/input clipping in bright synthetic blends remains a
separate, material generator limitation.

The grouped normal MSE ratio is about 5.37x in RMSE, not 28.81x in RMSE.

## Environment and preservation

- Dataset: `data/Galaxy10_DECals.h5`
- Shape/dtype: 17,736 x 256 x 256 x 3, `uint8`
- Dataset SHA-256:
  `19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571`
- Metadata: class label, RA, Dec, redshift, pixel scale; no stable object ID and
  no HDF5 channel-order attribute.
- No local `Galaxy10_DECals_NoDuplicated.h5` variant was found.
- Python: 3.14.6
- PyTorch: 2.12.1
- MPS: built and available; CUDA unavailable.
- Full grouped training and all tensor inference selected `mps`; CPU was used
  only for metadata, hashing, plotting, CSV aggregation, and tiny analytic
  checks.
- All 16 checkpoints present before this campaign retain exact size,
  nanosecond mtime, and SHA-256.
- Exactly two new timestamped grouped checkpoints were added. No historical
  checkpoint or run was overwritten or deleted.

## Audit summary

### Infrastructure and data flow

The full dataflow audit verified dataset normalization, source selection,
blend construction, masks, model input schemas, residual reconstruction,
training/evaluation modes, checkpoint selection, aligned comparators, and
output containment.

The model sees only blended RGB. Targets, masks, source indices, group IDs, and
blend parameters are used only for supervision or post-inference evaluation.
The optimized residual is `blended - target`; reconstruction is
`blended - predicted_residual`. Because the generator may blur the target,
add noise, and clip the composite, this is a blend-to-target correction field,
not necessarily pure contaminant flux.

### Blending algorithm

The generator is deterministic and exactly replayable when round-trip CSV float
parsing is used. A first replay attempt exposed ordinary decimal-parser rounding
of `noise_std`; that failure is preserved, and the corrected parser reproduces
all hashes.

The algorithm is internally suitable for a controlled computer-vision
restoration benchmark, but it is not calibrated astronomical injection.
Limitations retained and documented include:

- nonlinear display-RGB addition rather than calibrated FITS-band flux;
- target-only blur and simplified noise/PSF behavior;
- material input clipping in bright/overlap suites;
- centered targets and shifted contaminants, permitting a centrality shortcut;
- padded extraction masks that compress the reported size ratio;
- pixel-scale mismatch not used for pairing/resampling;
- possible source artifacts and foreground-mask edge effects.

### Metric correctness

Twenty-nine independent metric-audit checks and 15 permanent standard-library
unit tests pass. They cover whole and regional MSE/MAE, PSNR, SSIM, core and
non-core masks, Manhattan halo bands, Delta E/color range handling,
gradient/edge metrics, empty-mask coverage, macro versus micro aggregation,
clipped versus unclipped reconstruction, prediction-independent affected
masks, and sample-ID-aligned win/worse rates.

Historical regional tables sometimes reported `n=1000` while empty core masks
were skipped (for example, 858 valid normal core rows). New tables retain both
total and valid counts. Historical values were not rewritten.

### Leakage severity

- Cross-split exact-pixel pairs: 29
- Cross-split exact-coordinate pairs: 27
- Unique implicated sources: 57/17,736 (0.321%)
- Historical implicated normal rows: 13/1,000
- Historical implicated stress rows: 12/1,000
- Maximum clean-subset change in historical improvement ratio: about 0.31%

Verdict: **minor measured aggregate effect, major protocol defect**. Grouped
retraining was still required.

### Duplicate-safe grouped source split

`data/manifests/grouped_source_split_20260710_100907/` contains:

- 12,417 train sources;
- 2,660 validation sources;
- 2,659 grouped development-test sources;
- 17,675 source-identity groups;
- zero source/group overlap across partitions;
- zero cross-split exact-pixel duplicates;
- zero cross-split exact-coordinate duplicates.

Grouping uses the union of exact pixel hashes and exact coordinates. It does
not prove that all perceptual near-duplicates are the same physical source.

### Grouped blend manifests

`data/manifests/grouped_blends_20260710_103233/` contains:

- 8,000 balanced training rows (50% normal, 30% high overlap, 20%
  brightness/size stress);
- 1,000 validation rows;
- 1,000 normal test rows;
- 1,000 hard-stress rows;
- 1,000 compact-bright rows;
- 1,000 high-core-obstruction rows.

Both target and contaminant remain in their assigned partition. All 71 manifest
integrity checks and all 13,000 blend/affected/core/halo hash replays pass.
These are grouped **development** manifests, not an untouched final benchmark.

### Existing historical v0.2 on grouped tests

The old checkpoint achieved 32.33x normal, 18.15x hard, 11.75x compact-bright,
and 18.43x high-core affected-MSE ratios on the new grouped suites. This was a
fast diagnostic, not a source-independent evaluation: 2,183/4,000 rows (54.575%)
contained a source group exposed to its historical training or validation
pools.

On the 1,817 clean-neither rows, the old checkpoint still achieved 31.53x
normal, 18.18x hard, 11.68x compact-bright, and 18.27x high-core. This supports
the plausibility of the old result but cannot promote that checkpoint as a
leakage-cleared result.

### Grouped v0.2 Moderate retrain

Run:
`outputs/runs/br_v02_moderate_grouped_retrain_20260710_110917/`

Settings:

- historical 1,927,075-parameter v0.2 U-Net;
- residual correction-field prediction;
- affected/core extra weights 3/2;
- 8,000 train and 1,000 validation blends;
- 20 epochs, batch size 8;
- training seed 3042;
- MPS device.

Epoch 20 was both best and final:

- train weighted loss: 0.0010825181
- validation weighted loss: 0.0011635236
- validation affected MSE: 0.0033365143

Best checkpoint:
`outputs/checkpoints/unet_br_v02_moderate_grouped_retrain_20260710_110917_best.pth`

Best SHA-256:
`eea442ff21bdfbdd74815d7b292e786f187dc9a63fea73d4adde98a4b082802b`

Final checkpoint:
`outputs/checkpoints/unet_br_v02_moderate_grouped_retrain_20260710_110917_final.pth`

Final SHA-256:
`c67b67ffd19f52f46e91b96f99207255853548332bd1e9949631be7e72d3051f`

The best/final state tensors are identical because epoch 20 was also the
best-validation epoch; checkpoint-kind metadata remains distinct.

The grouped retrain is weaker than the old checkpoint on all identical grouped
suites. That gap cannot be attributed solely to leakage: the historical model
used 12,000 training blends, the requested grouped run used 8,000, and only one
grouped seed was run.

### Final-test independence blocker

The earlier provisional source pool is not final-eligible after the grouped
resplit:

- mapped by the grouped split: 683 train, 173 validation, 144 test;
- actually used in grouped training blends: 499;
- actually used in grouped validation blends: 91;
- used in grouped train or validation: 590/1,000.

Those historical files remain preserved and are explicitly superseded. A future
final evaluation needs either new independent data or a new four-way
group-disjoint train/validation/development/final split followed by retraining.
The final partition must be allocated before training and must remain
uninspected until the protocol is frozen.

## Findings status

The final table contains 36 findings: 2 blocker, 11 high, 14 medium, 3 low, and
6 informational.

| Finding class | Final status |
| --- | --- |
| Historical row-index source split | Fixed for grouped development |
| Final-test independence | Unresolved blocker; old provisional pool superseded |
| Model-input/target leakage | No leakage found |
| Metric arithmetic/alignment | Fixed and tested |
| Manifest replay/provenance | Fixed for grouped manifests |
| Display-RGB physical realism | Documented high limitation |
| Composite clipping | Quantified high limitation |
| Target centrality | Documented high limitation |
| Size-ratio and pixel-scale semantics | Documented high limitations |
| Empty-mask and regional coverage | Fixed for future outputs |
| Statistical source reuse | Group IDs retained; clustered uncertainty pending |
| Training-seed robustness | Not established |
| Dependency locking | Versions captured; lockfile still pending |
| Source-artifact labels | Heuristic only; blinded review pending |
| Historical versus grouped training budget | Documented comparison confound |

The authoritative row-level disposition is
`tables/audit_findings_final_status_corrected.csv`. The preliminary
`audit_findings_final_status.csv` and `checkpoint_integrity_final.csv` are
preserved from a stopped finalization attempt that compared relative paths with
absolute paths. The corrected integrity table normalizes paths and passes; no
checkpoint changed.

## Final model status

- **Thayer-BR v0.2 Moderate, original checkpoint:** historical
  development-split leader; preserve its 32.3x/19.6x values as historical
  context only.
- **Thayer-BR v0.2 Moderate, grouped retrain:** preferred duplicate-safe
  development reference; 28.81x normal and 15.80x hard, with compact/high-core
  stress values reported above.
- **Thayer-BR v0.3 Delta:** compact/perceptual tradeoff ablation; not current
  best.
- **Thayer-ResUNet v0.4:** architecture ablation; not current best.

The v0.2 Moderate **model family** remains current best. No checkpoint should be
called a final current-best paper model until a fresh untouched final benchmark
is run. No second grouped training seed was run, so training-seed robustness is
not claimed.

## Claim recommendation

**Soften and replace the headline.**

- Preserve “32.3x lower normal affected-region MSE” and “19.6x lower hard-stress
  affected-region MSE” only as original row-index development results.
- Lead current scientific discussion with “28.81x lower affected-region MSE
  than identity on exact-pixel/exact-coordinate group-disjoint normal
  development data,” plus the three grouped stress results.
- State that 28.81x is an MSE ratio (about 5.37x in RMSE).
- State that the measured exact-leak aggregate effect was minor, while the split
  protocol itself was invalid.
- Do not claim final-paper, survey-ready, calibrated-flux, or independent
  training-seed performance.

## Verification and safety status

- `python -m compileall src scripts tests`: pass.
- Permanent metric tests via `python -m unittest discover -s tests -v`:
  15/15 pass.
- `pytest` was not installed in the private virtual environment; its attempted
  invocation failed before test collection and is not presented as a test pass.
- Independent metric audit: 29/29 checks pass.
- All master/run CSVs: imported successfully with the artifact CSV validator.
- `git diff --check`: pass.
- Git branch: `thayer-br-0.3`.
- Git index: empty.
- Privacy grep: no matches outside excluded `.git`, `.venv`, `data`,
  `outputs`, and cache directories.
- Protected checkpoint integrity: pass, 16/16 unchanged.
- New checkpoints: exactly the two timestamped grouped best/final files.
- No files were committed, staged, pushed, merged, or deleted.

## Exact next steps

1. Freeze the generator, metric definitions, analysis code, and model-selection
   policy.
2. Obtain independent source groups or create a four-way grouped
   train/validation/development/final split before retraining. Keep the final
   partition completely untouched and unrendered.
3. Repeat the v0.2 Moderate grouped training under that final-safe split. Match
   the historical 12,000-blend budget if the goal is a causal split comparison.
4. Run at least two additional independent training seeds eventually; describe
   two seeds only as preliminary, not full robustness.
5. Add group-clustered bootstrap intervals using target and contaminant group
   IDs.
6. Complete blinded review of high-confidence near-duplicate and source-artifact
   candidates; do not automatically merge morphology lookalikes without
   identity evidence.
7. Add target-translation/role-swap, pixel-scale-matched, size-normalized,
   low-clipping, and calibrated or linear-light injection controls.
8. Evaluate the frozen best checkpoint once on the untouched final manifests,
   report clipped and unclipped metrics, and do not tune afterward.

## Primary artifacts

- `diagnostics/infrastructure_correctness_audit.md`
- `diagnostics/blending_algorithm_audit.md`
- `diagnostics/metric_correctness_audit.md`
- `diagnostics/leak_severity_report.md`
- `diagnostics/grouped_source_split_report.md`
- `diagnostics/grouped_blend_manifest_report.md`
- `reports/existing_v02_grouped_eval.md`
- `diagnostics/grouped_existing_v02_historical_exposure_report.md`
- `diagnostics/grouped_retrain_v02_report.md`
- `diagnostics/provisional_final_pool_superseded.md`
- `tables/grouped_retrain_suite_metrics.csv`
- `tables/grouped_retrain_per_sample_metrics.csv`
- `tables/grouped_retrain_comparison_summary.csv`
- `tables/audit_findings_final_status_corrected.csv`
- `tables/checkpoint_integrity_final_corrected.csv`
- `figures/grouped_retrain_training_history.png`
- `figures/grouped_existing_vs_retrain.png`

