# Limitations and Next Steps

## Thayer-Select Phase II boundary — 2026-07-11

The promptability baseline is frozen and the recoverability campaign completed
with partial success. Its largest known failure is not ordinary reconstruction
error but request validity: the Phase-I reconstruction-only model hallucinated
on every exact empty prompt under the declared criterion and often selected a
real alternate source when queried there. Phase II fixes the semantics (the
alternate coordinate is a valid request for that alternate galaxy), introduces
explicit null and ambiguous classes, and tests calibrated abstention.

This remains controlled BTK development work. It does not authorize a lockbox
evaluation, a final-paper claim, calibrated per-pixel uncertainty language, or
survey readiness. Optional independent training seeds are secondary to
finishing the primary calibration and one-time frozen development evaluation.
The selected PERMISSIVE actionable label remained only 3.12% positive, making
head learning and calibration fragile. R1's calibrated score achieved AUROC
0.875 but assigned ambiguous queries higher mean confidence than clear valid
queries. Overall selective risk declined modestly, while catastrophic failure
rose at 80% and 70% coverage. R1 null hallucination (8.25%) did not beat frozen
Phase-I C (7.5%) or R0 (2.25%) on identical new null-coordinate scenes. The
one-time development pass retained scalar uncertainty aggregates but not full
maps; those maps were not regenerated.

The next scientific gate is consistent valid-only selective-risk improvement
across two fixed-protocol seed replications. Feasibility mining found zero
candidate ambiguity pairs meeting both provisional cutoffs, so a full Ambiguity
Atlas is not justified. Lockbox evaluation remains separately unauthorized.

## Current decision

The grouped v0.2 Moderate retrain and grouped development evaluation are
complete. The retrained checkpoint has 28.8x lower normal and 15.8x lower
hard-stress affected-region MSE than identity on the grouped suites. This is
strong duplicate-safe development evidence, not a locked final-paper result.
The original v0.2 result remains a historical development result; Delta is a
compact/color/preservation tradeoff, and ResUNet v0.4 is a compact/halo
architecture ablation.

The original random-index protocol is not eligible for new claims. The source
audit found 29 pixel-identical pairs crossing train/validation/test, including
27 same-coordinate duplicated objects. The authorized grouped retrain corrected
the observed exact-pixel and exact-coordinate leakage before training. It did
not establish exhaustive near-duplicate identity resolution or final-test
independence.

## Highest-priority corrections

1. Freeze the model/checkpoint list, generator, masks, metrics, clipping policy,
   and reporting rules.
2. Create a fresh untouched final source pool after that freeze. It must be
   group-disjoint from every source group used for grouped training, validation,
   and development testing.
3. Audit exact pixels, exact coordinates, and high-confidence perceptual
   candidates for that final pool. Exact-group disjointness is not proof of an
   exhaustive near-duplicate audit.
4. Manually review the 356-source artifact candidate pool without model scores,
   then freeze versioned artifact-screened-source and artifact-stress flags.
5. Run the predeclared final comparison once and report all suites. Do not infer
   training-seed robustness from evaluation-seed variation or one grouped
   retrain.

The earlier provisional 1,000-source final pool is superseded and not
final-eligible: under the grouped split it maps to 683 train, 173 validation,
and 144 test sources, and the actual grouped train/validation blend manifests
use 499/91 of those sources (590 total). The grouped blend infrastructure itself
contains 8,000 train, 1,000 validation, and four 1,000-row test manifests, with
71/71 integrity checks and 13,000/13,000 exact replays. It is development
infrastructure, not the untouched final pool.

## Model-behavior limitations

- v0.2 has a small aggregate unblended-input error but a meaningful tail:
  3/1,000 null inputs exceed MSE `0.001`, with false subtraction visible around
  bright off-center sources and target structure.
- Delta reduces mean unblended-input MSE by about 22.1x relative to v0.2 and
  lowers paired excess target error over identity in the mask-complement region,
  but worsens normal/stress affected MSE. This is a preservation/perceptual
  tradeoff, not a new best model.
- ResUNet improves compact-bright and halo-band aggregates but does not improve
  the main stress/core gate consistently.
- Clipping has little aggregate effect and does not change rankings, but
  per-sample out-of-range statistics should remain visible.
- Source-artifact heuristics have expected false positives and must not become
  automatic exclusions without review.

## Scope limitations

Galaxy10 DECaLS inputs are RGB display cutouts, not calibrated FITS flux images.
The work studies controlled synthetic restoration of RGB cutouts. It does not
establish survey-grade deblending, calibrated photometry, or source separation
in crowded real fields. Identity and threshold are sanity checks, not strong
astronomical deblenders.

Additional realism work should follow benchmark repair: apparent-size-matched
evaluation, PSF variation, sky/background mismatch, detector artifacts,
correlated environments, and calibrated-data validation.

## Claim boundaries

Safe current wording separates the two development protocols:

> On the original random-index development suites, Thayer-BR v0.2 Moderate has
> 32.3x lower normal and 19.6x lower stress affected-region MSE than identity,
> corresponding to about 5.7x and 4.4x lower RMSE. These are development results
> from a source split with confirmed duplicate leakage. After exact-pixel and
> exact-coordinate grouping and retraining, the corresponding grouped
> development ratios are 28.8x and 15.8x. Neither protocol is an untouched
> final test; a fresh group-disjoint final pool is required for a paper claim.

Do not claim survey readiness, independent-training-seed robustness, a
leakage-cleared final result, or that heuristic artifact flags are ground truth.

## Frozen-head ablation limitations and next step

The moderate recoverability label has only five validation positives. Although
balanced frozen heads achieved high validation AUROC and much higher AUPRC than
the unweighted baseline, the paired head differences are uncertain and the
validation-selected MLP degraded sharply on calibration. Ambiguity inversion
persisted, catastrophic rejection remained weak, and both isotonic and the
selected MLP's temperature-scaled scores had operationally important ties.

The H4 centroid result is inconclusive: its AUPRC interval includes large gains
and losses. The generator-metadata oracle is explanatory only and cannot be
used at inference. Boundary proximity and contract sensitivity indicate target
noise/heterogeneity but do not authorize post hoc relabeling.

Do not claim that recoverability is nonlinearly solved, that cross-band
centroids add independent value, or that selective abstention now succeeds.
The single next experiment is to redesign and preregister the moderate
reliability target with separate failure reasons. Do not begin a new head or
backbone experiment until that target protocol is frozen. Development and the
future lockbox remain unavailable for this design work.

## Hierarchical-policy limitations and next step

The hierarchical experiment establishes that query validity is learnable from
frozen model-accessible features: NULL rejection was essentially perfect and
ambiguity inversion disappeared. It also establishes that valid-only metric
risks contain substantial rank signal. Neither result establishes a useful
operating policy.

The dominant limitation is feasibility under the frozen scientific limits.
Condition-C development means were image NRMSE `2.257`, maximum per-band flux
risk `11.735`, and centroid error `4.057` pixels, while the moderate policy
limits were `0.75`, `0.50`, and `2.0`. After conformal calibration, only one
natural-calibration valid scene and one development valid scene passed every
gate. This is operational abstention collapse. Query-gate confidence alone was
also anti-correlated with valid reconstruction safety at lower coverages, so it
must not be used as a surrogate risk score.

The five-seed risk heads were stable in log-space ranking but not uniformly
stable after exponential inversion; several seeds produced very wide raw
intervals. Marginal 90% conformal coverage does not guarantee conditional tail
coverage, class-conditional coverage, or useful interval width. The historical
R1 fresh-scene ranking remained at least as effective as the hierarchy at
useful coverage.

Do not retune this campaign after its one-time development result. The next
experiment should use only train/validation/calibration artifacts and
preregister: (1) an aperture-flux measurement audit; (2) log-space conditional
conformal diagnostics; (3) a fixed catastrophic-risk budget; and (4) a minimum
70% valid calibration coverage gate. Keep Condition C frozen and compare
against R1 before authorizing any new development set. A targeted ambiguity
pilot may use simulator optimization, matched source pairs, and multi-hypothesis
truth sets, but the full Atlas and lockbox remain deferred.

## Hierarchical protocol limitation

The historical hierarchical campaign omitted the required preregistration file
and full original-contract postmortem before fitting. Its development result
must therefore remain historical evidence, not a fully preregistered policy
claim. The corrective audit found no Boolean label-code mismatch, but confirmed
that the learned binary target mixed inapplicable query states, mild failures,
and catastrophes and that calibration labels came from a different
reconstruction provenance than training/validation labels.

The next experiment must be prospective and train/validation/calibration-only:
hash the complete protocol before fitting, use a single frozen reconstructor for
all empirical outcomes, complete row-level applicability and drift audits, and
require nondegenerate calibration coverage before creating a new development
manifest. Never reuse the historical development scenes for tuning and keep the
lockbox sealed.

## Prospective feasibility limitation and one next experiment

The new prospective evidence removes the earlier mixed-provenance and
applicability defects. It does not establish an operational hierarchy. The
catastrophic-valid model ranked extremely well, but the preregistered AUPRC gate
was mathematically unattainable at the realized prevalence and therefore fails
without post-hoc repair. Image and flux conformal bounds achieved 90% marginal
coverage while falling to 69–76% in the weakest frozen subgroups; rare extreme
predictions also dominated mean interval width.

Do not build a policy, generate development data, or access the lockbox. Run
one separately preregistered train/validation/calibration-only conditional-
calibration correction: preflight gate attainability, use a bounded
prevalence-adjusted AP lift, keep Condition C and all heads frozen, calibrate
log residuals with partial pooling over fixed SNR/overlap groups, and require
85–95% subgroup coverage plus a bounded 95th-percentile width.
