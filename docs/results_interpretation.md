# Results Interpretation

## High-Level Takeaway

Thayer-BR v0.2 Moderate remains the strongest Thayer-Net formulation, but its
results now have two distinct statuses. The original checkpoint is the best
model on the historical development split: affected-region MSE changes from
`0.068122` to `0.002108` on normal blends, 32.3x lower than identity, and from
`0.075541` to `0.003847` on hard stress, 19.6x lower. These MSE ratios
correspond to about 5.7x and 4.4x lower RMSE, respectively. They are preserved
as historical development results, not treated as a leakage-cleared estimate.

A duplicate-safe grouped retrain of the same v0.2 Moderate formulation remains
strong on newly generated grouped development suites, although it does not
reproduce the historical headline: affected-region MSE is `0.00231890`
(`28.8127x` lower than identity) on normal blends, `0.00458983` (`15.8025x`)
on hard stress, `0.00872771` (`9.18304x`) on compact-bright stress, and
`0.00491680` (`15.8378x`) on high-core obstruction. Worse-than-identity counts
are `0/1000`, `3/1000`, `2/1000`, and `1/1000`, respectively. These results
support the method under stricter infrastructure, but the grouped suites are
still development benchmarks rather than a locked final test.

The evaluation-seed audit gives `32.02 +/- 1.21x` lower normal affected MSE and
`19.55 +/- 0.30x` lower stress affected MSE versus identity. It varies
blend-generation/evaluation seeds, not independent training seeds.

## Development-Benchmark Status

The current normal/stress suites were used during model development. A later
audit found duplicate RA/Dec source groups across the historical random-index
train/validation/test partitions, including train-to-test crossings. Thus,
these results support a development ranking but not a leakage-cleared final
paper estimate. The grouped development partition now resolves the known
exact-pixel/coordinate split defect; a separate untouched final source pool and
newly frozen final-test manifest are still required, and that final set must
not guide more model choices.

The correctness audit then built a split that keeps exact-pixel and
exact-coordinate groups wholly within one partition. Its train, validation,
normal, hard, compact-bright, and high-core manifests passed source/group-role
containment checks and exact replay for every generated row. This resolves the
known split defect for the grouped experiments, but those manifests were used
to validate the infrastructure and model and therefore remain development
benchmarks.

Galaxy10 DECaLS images are RGB display cutouts, not calibrated FITS flux
images. Results concern synthetic restoration of Galaxy10 RGB cutouts, not
survey-grade source separation or calibrated photometry. Additive RGB
compositing, blend-input clipping, target centrality, asymmetric apparent-size
handling, and the absence of pixel-scale-aware angular normalization further
limit physical interpretation.

## Grouped Retrain Interpretation

The grouped retrain is the correct comparison for asking whether the v0.2
Moderate training recipe survives duplicate-safe source partitioning. It does:
all four grouped suites retain large reductions in affected-region MSE versus
identity. The reduction relative to the original 32.3x/19.6x development
headline is real and should not be hidden. It can reflect the stricter source
partition, newly generated suites, and the grouped retrain's 8,000-blend
training budget versus 12,000 blends for the historical v0.2 run; the current
experiment does not isolate those factors.

The old v0.2 checkpoint performs better than the grouped retrain on the same
grouped test manifests, but that is not an unbiased old-versus-new training
comparison. `54.575%` of those evaluation rows contain a source group seen by
the old checkpoint in its historical train or validation pool. Restricting the
old-checkpoint diagnostic to the `45.425%` clean-neither rows still gives
affected-MSE ratios of `31.53x`, `18.18x`, `11.68x`, and `18.27x` on normal,
hard, compact-bright, and high-core suites. This makes strong generalization
plausible, but only the grouped retrain has source-group-safe training and
evaluation by construction.

Headline masked values are macro means of per-sample metrics. Micro/pooled
metrics should remain supplementary and labeled explicitly, because variable
mask area changes their weighting. Core and other regional metrics must report
the number of samples with a valid nonempty region; they must not silently
average missing regions as zeros.

## Model Progression

Thayer-Direct establishes that learned reconstruction is useful in the
controlled benchmark. It maps `blended -> target` and strongly beats identity
and threshold baselines, but its stress performance drops under harder overlap
and contaminant conditions.

Identity and threshold are sanity checks, not competitive astronomical
deblenders.

Thayer-Residual changes the task to residual prediction. It predicts
`blended - target` and reconstructs with `blended - predicted_residual`. This
helps stress robustness because the model learns what contaminant light to
subtract rather than redrawing the whole galaxy.

Thayer-BR v0.1 keeps residual prediction but changes the training distribution
to include more high-overlap/core-obstruction and brightness/size stress cases.
It was the previous best model and showed that targeted hard-case sampling
matters.

Thayer-BR v0.2 Moderate keeps the same residual U-Net family and adds moderate
affected/core-weighted residual loss. It is the current-best formulation; its
historical and grouped-retrain estimates must remain separately labeled.

Thayer-BR v0.2 Strong is a stronger weighting ablation. It slightly improves
stress core MSE relative to Moderate, but worsens aggregate normal affected MSE,
aggregate stress affected MSE, and stress non-core affected MSE. It should not
be treated as the main model.

## Why Affected-Region MSE Matters

Affected-region MSE evaluates only pixels where the blend differs from the
target above the affected threshold. This isolates the pixels where contaminant
light changed the clean target.

Whole-image metrics are still useful, but they can make identity look
deceptively strong because most pixels in each synthetic blend are unchanged.
Affected-region MSE is therefore the primary metric for deblending quality in
this benchmark.

## Why SSIM Improves More Modestly

SSIM is computed over the whole image. In these cutouts, large regions are
unchanged background or target light that the identity baseline already
preserves. A model can substantially improve the contaminated region while
showing a smaller whole-image SSIM change.

This does not make SSIM useless; it means SSIM should be interpreted as a global
image-quality metric, not as the main measure of contaminant removal.

## Why Residual Prediction Helps

Residual prediction asks the model to learn contaminant signal:

```text
true_residual = blended - target
reconstruction = blended - predicted_residual
```

This gives the model a direct path for preserving already-correct target light.
It can focus on subtracting the extra contaminant contribution. That is why
Thayer-Residual improves stress performance relative to Thayer-Direct even
though direct reconstruction still wins on some individual samples.

## Why Balanced Training Helps

Random blend sampling under-represents some of the cases most relevant to
deblending failure, especially core overlap and bright/similar-size
contaminants. Thayer-BR v0.1 makes those cases common during optimization
through a 50/30/20 mix of normal, high-overlap/core-obstruction, and
brightness/size stress blends.

The improvement from Thayer-Residual to Thayer-BR v0.1 shows that training
distribution matters even when architecture and residual formulation are held
fixed.

## Why Weighted Loss Helps

Thayer-BR v0.2 Moderate changes the loss so contaminated pixels and affected
target-core pixels receive more emphasis than unchanged background pixels. The
moderate setting uses affected/core extra weights `3/2`.

This better matches the metric and scientific objective. The model is not
rewarded primarily for background pixels that were already easy; more training
pressure is placed on the regions where contaminant light changed the target.

Compared with Thayer-BR v0.1, v0.2 Moderate lowers:

- normal affected MSE from `0.002451` to `0.002108`;
- stress affected MSE from `0.004587` to `0.003847`;
- stress core MSE from `0.013848` to `0.009533`.

## Why Strong Weighting Is Not Best

Thayer-BR v0.2 Strong uses affected/core extra weights `5/4`. It slightly
improves stress core MSE relative to Moderate (`0.009344` versus `0.009533`),
but worsens normal affected MSE (`0.002306` versus `0.002108`), stress affected
MSE (`0.004030` versus `0.003847`), and stress non-core affected MSE.

This shows that the weighting objective has a tradeoff. More core emphasis can
over-focus the model on core pixels and reduce broader affected-region quality.
Moderate weighting is the best current balance.

## What the Size-Ratio Audit Found

The size/visual audit found that apparent contaminant/target radius ratio varies
substantially: approximately `0.49` at the 5th percentile, `1.06` at the median,
and `2.37` at the 95th percentile. That is enough variation to justify a future
size-normalized benchmark.

Learned-model affected-error dependence on apparent size ratio was
weak in the audit (`-0.17` direct, `0.05` residual, `-0.14` BR v0.1, `-0.12`
BR v0.2). This does not rule out a size shortcut; it motivates a controlled,
size-normalized test after duplicate-aware source filtering.

The right conclusion is cautious: keep the current controlled benchmark result,
and add a size-normalized held-out benchmark before making stronger
size-invariance claims.

## Why Some Visual Examples Still Look Weird

Lower affected-region MSE does not always match visual preference. A model can
reduce large core errors while introducing broad low-level residual patterns
that are visible in error maps. Conversely, a direct model can sometimes look
cleaner while having worse affected-region MSE.

The halo-band audit did not show an aggregate v0.2 halo penalty. Normal
halo-band MSE improved from `0.000300` for BR v0.1 to `0.000250` for v0.2
Moderate; stress halo-band MSE improved from `0.000359` to `0.000320`.

Selected individual examples still show broad low-level artifacts and
visual-vs-metric disagreements. These examples should appear in the paper as
limitations and diagnostic figures.

## Why v0.3 Color/Structure Is an Ablation

Experiment 5 tested Thayer-BR v0.3 Color/Structure Candidate in
`outputs/runs/br_v03_delta_color_20260709_185630`. It kept the same residual
U-Net architecture as v0.2 Moderate but added reconstruction L1, affected/core
reconstruction loss, finite-difference gradient loss, differentiable RGB
chroma/color-direction loss, and halo-band regularization.

The result is a tradeoff rather than a new best model. Compared with
Thayer-BR v0.2 Moderate, v0.3 worsened the primary affected-region MSE on the
main suites:

- normal affected MSE: `0.002025` -> `0.002590`;
- hard-stress affected MSE: `0.003648` -> `0.004772`;
- high-core-obstruction affected MSE: `0.004312` -> `0.005443`;
- color-saturation affected MSE: `0.005227` -> `0.006341`.

It did produce targeted secondary gains. Compact-bright affected MSE improved
slightly (`0.006514` -> `0.006325`), compact-bright gradient error improved
(`0.012529` -> `0.011820`), normal halo-band MSE improved marginally
(`0.000340` -> `0.000333`), and the color-saturation chroma proxy improved
slightly. These gains are not broad enough to outweigh the normal/stress MSE
regression.

Delta E 2000 was implemented with scikit-image for evaluation only. It should
be interpreted as perceptual visual evidence under a standard RGB/sRGB-like
assumption, not as an astronomical truth-color metric. Galaxy10 DECaLS RGB
cutouts are survey composites, so affected-region MSE remains the primary
scientific metric.

## Why v0.3 Delta Remains a Tradeoff

The stronger-color Delta follow-up ran in
`outputs/runs/br_v03_delta_candidate_20260710_031425` with color-proxy weight
`0.10`. It improved on the first v0.3 Color/Structure candidate in both
aggregate affected MSE and Delta E 2000 on all six shared suites. It also
improved Lab chroma error and halo-band MSE relative to v0.2 Moderate on all
six suites, and compact-bright affected MSE improved from `0.006514` to
`0.005428`.

Those targeted gains did not transfer to the main reconstruction criteria.
Relative to v0.2 Moderate on the same samples, normal affected MSE worsened
from `0.002025` to `0.002275`, hard-stress affected MSE worsened from `0.003648`
to `0.004035`, and the stress worse-than-identity count increased from zero to
one. Delta E improved on normal and compact-bright blends but worsened slightly
on stress, high-core, halo-band, and color-saturation aggregates. Qualitative
examples include both muted-color improvement and a Delta-specific color
artifact. Delta is therefore a better color/compact/halo ablation than the
first v0.3 candidate, but it is still a `visual_tradeoff`, not a replacement
for v0.2 Moderate.

## Why ResUNet v0.4 Is an Architecture Ablation

Experiment 6 tested a small residual-block U-Net in
`outputs/runs/resunet_v04_candidate_20260710_043109`. It has `2,014,595`
parameters versus `1,927,075` for the standard model (`+4.54%`) and used the
same residual task, 50/30/20 balanced distribution, and v0.2 Moderate weighted
loss without color or halo auxiliaries.

The result was close but did not clear the strict gate. Same-run normal
affected MSE changed from `0.002132` to `0.002118`, while hard-stress affected
MSE changed from `0.003929` to `0.003950` and added one worse-than-identity
case. Compact-bright affected MSE improved substantially (`0.006840` to
`0.005427`), and halo-band MSE improved on every suite. Affected Delta E 2000
also improved on every suite despite the absence of a color loss.

The main counterweight is core reconstruction. Hard-stress core MSE worsened
by about 2.7%, high-core-obstruction core MSE worsened by about 5.5%, and
color-saturation core MSE worsened by about 11.3%. The qualitative set contains
both improvements and failures, including a case where Thayer-Direct remains
better. This is a targeted compact/halo/color tradeoff, not evidence that the
architecture is generally superior.

The Core+ and Halo-safe tuning variants were not run because Part B2 was
conditional on the baseline ResUNet clearly beating v0.2 Moderate. Running
them after the strict gate failed would have violated the controlled stopping
rule. Thayer-BR v0.2 Moderate remains current best.

## What the Source-Leakage Audit Changes

The historical source arrays are disjoint by HDF5 row index, and no auditable
target/contaminant role violation was found. They are not object-disjoint:
there are 29 pixel-identical pairs and 27 exact-coordinate pairs crossing
train/validation/test. This is a protocol blocker, not evidence
that model outputs or recorded metrics were fabricated.

The duplicate union implicates 57 of 17,736 sources (`0.321%`). Historical
source-index reconstruction identifies 13/1,000 normal and 12/1,000 stress
samples with an implicated target or contaminant. Excluding those rows changes
the historical v0.2 affected-MSE ratios by no more than `0.31%`. Thus, the
measured aggregate effect of the known duplicates is minor, while the protocol
defect is major: random row disjointness did not justify an object-independent
generalization claim, and unrecognized near-duplicates remain an audit
limitation.

The exact-pixel/coordinate grouped split and fully replayed grouped manifests
remove the demonstrated cross-partition leakage mechanism. The grouped
retrain is therefore the preferred correctness result for this model family,
but it is not a final-paper estimate. The earlier provisional final pool is
superseded: 590 of its sources were subsequently assigned to grouped training
or validation (`499` train and `91` validation). A fresh untouched,
group-disjoint final source pool must be selected after model and protocol
freeze and evaluated once without using its metrics or examples for further
decisions.

## What the Preservation and Clipping Audits Add

On 1,000 unblended inputs, mean reconstruction MSE is `0.00002646` for v0.2,
`0.00000120` for the Delta preservation ablation, and `0.00002144` for the
ResUNet architecture ablation. Delta therefore preserves unblended cutouts much
better on average, even though it is worse on the primary blended
affected-region MSE. ResUNet lowers mean null MSE relative to v0.2 but raises
evaluation-core null MSE by about 53.5%.

The v0.2 average is small, but 3/1,000 null inputs exceed MSE `0.001`; the saved
grid shows false subtraction of bright off-center sources and target structure.
Heuristic artifact candidates make up 23/1,000 inputs and have higher mean null
error for every model. These are unblended tests, not a certified clean-source
benchmark.

Outside the `>0.02` affected mask, v0.2 has more paired excess target MSE over
identity than Delta or ResUNet on both normal and stress suites. This complement
also contains sub-threshold blend changes, so the audit separately reports
model output change versus the blended input rather than labeling all error as
damage.

Post-reconstruction clipping changes macro affected MSE by at most 0.16% and
does not alter model rankings. Ten of 6,000 model/sample comparisons have an
absolute clipping gain above `0.0001`, but none exceeds a 10% relative gain.
Thus, output clipping is not a major aggregate dependency, while rare
output-range behavior remains worth reporting. This result does not make
clipping irrelevant to blend construction: clipping the synthetic blend input
after additive RGB compositing can saturate information and materially define
the learning problem. Input- and output-clipping effects must remain separate.

## What the Source-Artifact Audit Adds

The streaming heuristic audit flags 356/17,736 sources (2.01%) for manual
review, including 178 saturation, 104 color-streak, 89 large-edge-mask, 52
edge-touching, 10 axis-line, and 3 blank flags. Categories overlap. These are
candidate flags, not measured artifact prevalence: contact sheets contain both
clear stripes/patches and expected genuine-morphology false positives. No
source was removed tonight.

## Limitations and Future Work

The current result is limited to controlled synthetic Galaxy10 DECaLS-style
blends with known clean targets. It does not prove performance on arbitrary
survey imagery, crowded real fields, physically correlated sources, spatially
varying PSFs, or realistic sky-background conditions.

Recommended next steps:

- review and, where justified, extend grouping to high-confidence
  near-duplicates without using final-test outcomes;
- create a fresh untouched final source pool after protocol/model freeze; the
  superseded provisional pool must not be reused as the final test;
- report independent grouped-training seeds before making a robustness claim;
- run a size-normalized held-out benchmark;
- execute the separately documented clean, artifact-stress, size-normalized,
  and compact-contaminant benchmark plan in `docs/clean_benchmark_plan.md`;
- keep halo-band and visual-vs-metric diagnostics in future comparisons;
- manually label the source-artifact review pool before defining clean and
  artifact-stress subsets;
- improve sky, PSF, and noise realism;
- treat v0.3 color/structure loss as an ablation unless a repeated-seed variant
  improves primary affected/core MSE as well as visual metrics;
- report counterexamples alongside headline metrics.

The completed Delta and ResUNet runs did not include clean-source-filtered or
artifact-heavy suites because the local source metadata has no validated
artifact-quality flags. The clean benchmark document is a non-training plan;
it does not change historical metrics or current-best status.
