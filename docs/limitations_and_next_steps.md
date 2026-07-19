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

## Conditional-calibration failure and one next experiment

The correction campaign repaired the defective gate design and preserved
strong rank transfer, but it did not repair image or flux conditional coverage.
The low-SNR/high-obstruction intersection contained 193 calibration rows and
was adequately source-group supported; selected image/flux coverage was 0.637
and 0.684. The failure is therefore not attributable to an underpowered group.
Normalized conformal helped centroid, but deployable Mondrian-normalized
correction did not rescue image or flux. Rare raw-space width outliers also
remained despite bounded medians and 95th percentiles.

Do not build a full policy, generate development data, or access the lockbox.
Run exactly one train/validation/calibration-only corrective experiment: a
preregistered partially pooled deployable scale head with robust heavy-tail
loss and frozen shrinkage over model-accessible features. Retain the frozen
reconstruction and risk heads and reuse the exact subgroup contract.

## Scale-correction failure and one next experiment

The partially pooled correction did not solve residual-scale misspecification.
Although validation residual-scale ranking improved and width inflation stayed
bounded, image difficult-subgroup coverage fell from `0.637` to `0.549` and
flux remained essentially unchanged at `0.679`. Bootstrap intervals exclude
the required stability margin. The oracle contrast shows exploitable regime
structure, but success requiring physical subgroup identity is non-deployable.

Do not build a full policy, generate development data, or access the lockbox.
Run exactly one separately preregistered train/validation/calibration-only
experiment: a monotone q=0.90 additive scale model over the same four
deployable proxies, with shape constraints fixed from proxy meaning and the
same OOF targets, gates, and natural-calibration audit. Do not broaden the
architecture or feature search.

## Shape-constrained quantile failure and one next experiment

Convex main effects represented the training-tail reversals and passed every
shape and centering test, but they did not transfer conditional coverage. The
single positive z0-by-z1 interaction was
effectively zero and changed neither worst validation-cell coverage nor final
subgroup coverage. Selected image/flux worst supported coverage fell to
`0.544`/`0.591`; bootstrap uncertainty does not approach the frozen gate.

Do not build a full policy, generate development data, or access the lockbox.
Run exactly one separately preregistered train/validation/calibration-only
experiment: replace the single separable hinge product with one small convex
tensor-product quantile surface over z0 and z1, keep z2/z3 additive, and retain
the exact proxies, OOF targets, partitions, gates, and audit. Do not run a
broader feature or architecture search.

## Observable-regime information limit and one next experiment

The richer spatial hypothesis produced real ranking signal: A3 validation
joint-hard AUROC was `0.901`, far above the exact four-proxy A0 at `0.711`, and
calibration AUROC remained `0.880`. This does not establish useful regime
observability. Recall at the frozen 0.70 precision point was only `0.083`, and
natural-calibration probability transfer failed both Brier and ECE gates.
Obstruction was much less rank-predictable than SNR, and continuous magnitudes
did not transfer despite strong SNR rank order.

Do not repair probability calibration post hoc, run GroupDRO, fit another
quantile model, construct predicted groups, generate development scenes, or
access lockbox. Run exactly one separately preregistered train/validation/
natural-calibration data-level experiment: provide an explicit observed PSF
representation to the otherwise frozen observability pipeline. Do not combine
it with IVAR, multiple epochs, or other new inputs in the same experiment.

## Fixed-PSF construction limit and one next experiment

The explicit-PSF audit established exact provenance but also established that
the current benchmark cannot answer the conditioning question. Every scene
uses the same g/r/z PSF triplet. The triplet differs by band but has no
between-scene or spatial variation, so it cannot explain which scenes are
unusually difficult. True, shuffled, and constant-median PSF controls would be
identical.

Do not fit a PSF-conditioned observability head, risk head, calibrator,
GroupDRO model, or policy on these fixed-PSF scenes. Do not generate
development scenes or access the lockbox. Run exactly one future experiment:
prospectively generate train/validation/natural-calibration scenes with a
preregistered realistic distribution of varying, inference-available PSFs.
Do not combine that experiment with IVAR, multiple observations, or other new
information sources.

## Competing-hypothesis limitations and next step

The frozen Atlas establishes finite near-collision examples under the current
BTK forward model and noise tolerance; it does not establish the frequency of
such ambiguity in a survey population. The 25 frozen pairs are deliberately
targeted stress cases, and all three tested reconstructors share one compact
prompted-U-Net architecture cluster. Their 18/50 model-candidate witnesses are
therefore not evidence of cross-family coverage, while the 32 single-candidate
cases are not evidence of uniqueness.

No black-box auditor, leave-one-family-out rotation, safety-coverage curve, or
accepted-catalog bias analysis was run. Shape safety remains not applicable.
The exact next experiment is one prospectively preregistered compact prompted
ResUNet, trained once on MPS with the frozen BTK normalization and source-layer
contract, followed only by deterministic replay and the frozen 25-pair Atlas
behavior audit. Reassess family diversity before any Thayer-Audit training.

## Atlas v0 limitation and one next experiment

The frozen Atlas v0 observations are strongly noise-dominated. They are valid
finite counterexamples under the controlled model, but they do not establish
high-information ambiguity or survey prevalence. More importantly, the
same-cluster candidate diameter failed the frozen detector comparison and was
worse than the narrow R1 confidence baseline. Do not train Thayer-Audit from
these candidates. Run exactly one next experiment: add one preregistered compact
prompted ResUNet under the frozen source-layer contract, then rerun only the
frozen Atlas behavior and candidate-diversity audit.

## Prompted-ResUNet promptability failure and one next experiment

The residual topology was structurally distinct and reconstructed at a similar
whole-image scale to Condition C, but it did not reliably honor the requested
coordinate. Low output collapse does not rescue the result: only 39.47% of
paired A/B queries selected both requested sources correctly, and individual
query success was 69.5%. The Atlas was therefore not opened for this model.

Do not tune the residual architecture on Atlas, rerun Atlas inference, add
another deterministic U-Net variant, admit a second family, or train an auditor.
Run exactly one future experiment: preregister a coordinate-conditioned
conditional VAE that generates multiple requested-source hypotheses under the
same normalization, source-layer contract, Atlas exclusions, and MPS policy.
Require promptability and forward-consistent non-Atlas multi-sample diversity
before any one-pass Atlas authorization.

## Thayer-PU limitation and one next experiment

Thayer-PU demonstrates that a compact conditional latent model can retain
prompt identity, generate forward-consistent candidates, concentrate more on
ordinary controls than on non-Atlas near-collisions, and materially improve
Atlas candidate-diameter discrimination. This does not establish a calibrated
posterior. On the frozen Atlas, none of the retained prior samples covered the
own requested truth or paired alternate truth, and the witness count stopped at
24/50 rather than the preregistered 30/50.

Do not resample Atlas, tune thresholds, increase VAE size, train Thayer-Audit,
or admit catalogs. Run exactly one future experiment: preregister a conditional
normalizing-flow prior correction on the frozen Thayer-PU representation and
repeat every current non-Atlas gate before any new one-pass Atlas authorization.

## Thayer-PF sufficiency failure and one next experiment

The prior-correction premise failed prospectively. Posterior samples did not
cover own truths on ordinary or non-Atlas near-collision scenes, and transferred
posterior latents did not recover the paired alternate truth. High forward
consistency only shows that these decompositions can recompose to the noisy
observation within tolerance; it does not show that the decoder contains the
known scientific modes.

Do not fit a flow, deepen the prior, pool latent teachers, rerun Atlas, train an
auditor, or access the final lockbox. Run exactly one future experiment:
preregister ambiguity-set decoder training that exposes both decompositions
under each approved non-Atlas near-collision condition and requires prompt-
specific own/alternate truth coverage plus forward consistency before revisiting
any prior correction.

## Thayer-MH shared-decoder limitation and one next experiment

Explicit two-target supervision did not solve truth representation. The shared
token decoder remained prompt-faithful and forward-consistent, yet covered no
approved own or alternate truth on ordinary or near-collision validation. The
result rules out promoting this K=2 shared decoder; it does not rule out the
target sets or prove uniqueness.

Do not change the frozen coverage metric, add generic visual diversity, tune on
Atlas, train an auditor, or access the final lockbox. Run exactly one future
experiment: preregister a K=2 model with the same shared prompted encoder but
two compact expert decoders, using permutation-invariant approved-target
matching and unchanged concentration, exclusion, and forward-consistency gates.

## Thayer-ME capacity limitation and one next experiment

Independent decoder parameters did not make the approved modes reachable under
the frozen training protocol. The microset result rules out full training of
this exact model: prompt identity and recomposition passed, while every truth-
coverage rate was zero and ordinary expert separation exceeded the scientific
limit. It does not show that more capacity would solve the problem.

Do not enlarge the model, rerun the micro fit, weaken truth coverage, tune on
validation or Atlas, train an auditor, or access the lockbox. Run exactly one
future experiment: a training-free frozen loss-geometry audit on the persisted
micro targets and outputs, decomposing the normalized objective against image,
flux, color, centroid, and primary scientific distance to determine whether the
current loss ranks exact truth proximity correctly.

## Thayer-LG limitation and next step

Exact truth is representable and the frozen metric accepts it, so neither the
output contract nor coverage implementation explains the micro-overfit
failure. The full objective instead favors lower forward-to-observed error even
when scientific source identity degrades. Hard assignment is also unstable at
collapsed means, and the frozen coverage boundary is narrow along several
directions. The audit does not establish that one replacement objective will
solve neural training.

The one next experiment is a prospective micro-overfit-only rerun using
source-set reconstruction plus ordinary concentration and a preregistered
differentiable surrogate of the unchanged scientific distance, with forward
consistency retained solely as a gate.

## Thayer-SA output-optimization limitation and one next experiment

The corrected surrogate accurately tracks the frozen metric and makes exact
truth stationary, but those properties were not enough for robust detached
optimization. Compromise losses decreased without consistently entering the
narrow coverage region, while random bounded outputs barely moved. This
failure occurs before hard-assignment and neural optimization can be diagnosed.

Do not train the model, change thresholds, add capacity, restore forward loss,
or open protected data. Run exactly one training-free output-space conditioning
experiment: prospectively compare a near-truth smooth component geometry under
the same target sets, thresholds, hard assignment, and initializations, with
coverage entry as the unchanged gate.

## Thayer-OC conditioning limitation and one next experiment

Conditioning improved some fixed starts but did not identify one global method
that passed all frozen gates. Raw L-BFGS approached ambiguous coverage from the
persisted Thayer-ME output but retained poor ordinary coverage. Physical T/D
Adam variants were numerically nonstationary at exact truth. The frozen modal
curvature diagnostic was unresolved, so no quantitative condition-number claim
is justified.

Do not select an optimizer per scene, transfer T/D or Jacobian conditioning to
neural training, weaken thresholds, change targets or assignment, or open
protected data. Run exactly one separate experiment: a preregistered direct
feasibility-learning micro-audit that projects into the unchanged frozen
scientific region.

## After Thayer-FP

Direct feasibility projection succeeded completely on the microset, but the
unchanged model did not learn the projected targets. All scientific coverage
rates stayed at zero despite strong prompt swap and forward consistency, and
the final outputs violated nonnegativity. Do not revisit scalar-loss
conditioning, relax thresholds, change target sets, select projection methods
per scene, or open Atlas/development/lockbox data.

Run exactly one next experiment: a separately preregistered decoder-capacity
ladder on the same 64 scenes and frozen P0 targets, varying only expert-decoder
capacity while preserving encoder inputs, hard assignment, thresholds, and
evaluation gates.

## After the Thayer-CL contract preflight

The capacity ladder did not begin. The historical identity head permits
negative physical source contributions, while ReLU, square, and absolute-value
heads are distinct prospective replacements and no frozen contract chooses
among them. The exact first violating historical batch is unavailable because
batch-level output checks were not persisted. Decoder capacity, the L0-L3
threshold, and seed stability remain unresolved.

Run exactly one next experiment: a separate preregistered output-
parameterization campaign at fixed L0 capacity. Compare the three named
nonnegative mappings under identical head-only representability tests and
fixed one/eight-scene P0 micro gates, select one global mapping prospectively,
and do not run the width ladder in that campaign.

## After the Thayer-OP fixed-L0 mapping campaign

Output parameterization solved physical nonnegativity but did not solve
scientific one-scene memorization. All three mappings represented every P0
target and passed synthetic fitting, yet each finished with zero ordinary
coverage and zero ambiguous both-mode coverage under the identical L0 system.
The square and absolute-value heads achieved ordinary expert diameter below
1.0, but neither put both experts in the scientific target region. ReLU also
showed a material dead-region fraction. No eight-scene aggregation claim and
no decoder-capacity claim can be made.

Run exactly one next diagnostic: a fixed-feature L0 expert-decoder
optimization audit on the frozen ambiguous scene. Retain the same hard
assignment and output mapping while comparing the neural decoder trajectory
with direct cached-feature output optimization. Do not test larger decoders,
open the remaining microset rows, or access Atlas, development, or lockbox data.

## Repository-integrity audit limitation and next experiment

The exact-path retry resolved direct reachability only through D2. Square D0
and D1 succeeded; square D2 failed; D3 and tangent diagnostics were blocked by
the frozen progression rule. The result therefore does not distinguish whether
the full L0 decoder can reshape frozen encoder features successfully, and it
does not support a decoder-capacity claim.

Run exactly one next experiment: preregister a square-only, one-scene full-L0
fixed-feature diagnostic that keeps D2 as the failed control but prospectively
authorizes D3 after the demonstrated D1 pass. Reuse the exact cache, endpoint,
loss, hard assignment, evaluator, and thresholds. Do not add capacity, change
scientific definitions, or open ordinary, eight-scene, remaining-microset,
Atlas, development, or lockbox data.

## After the Thayer-D3 frozen-input stop

Thayer-D3 could not perform its preregistered feature-trajectory comparison.
The successful D1 artifact stores raw, mapped, and physical outputs but omits
the optimized penultimate tensors. Inferring an endpoint from those outputs is
not equivalent to recovering the actual optimized feature state, and D1 is not
assumed unique. D3 therefore stopped before the one-step trace and optimizer
construction.

Do not classify this stop as decoder optimization, capacity, assignment, or
square-mapping evidence. Do not run D3, a tangent diagnostic, eight-scene
fitting, or a capacity ladder yet. Run exactly one next experiment: a separately
preregistered square-only D1 endpoint-persistence replay on the identical
one-scene cache and frozen heads, saving both penultimate tensors and verifying
their mapped outputs. Do not open broader data.

## After the Thayer-D1R endpoint replay

The exact square-D1 result is reproducible and its missing feature endpoint is
now complete. All 54 trajectory hashes and the 100/100/100 scientific result
matched; four semantic prompt/expert tensors replay through unchanged frozen
heads in a restricted fresh process; and batch, position, layout,
serialization, canonicalization, and device-transfer invariance passed.

This does not establish that the existing full L0 decoder can reach the
endpoint from cached encoder features, and it does not justify more capacity or
broader data. Run exactly one next experiment: a separately preregistered
one-scene square-only D3 campaign using this persisted endpoint only as an
evaluation reference. Do not run eight-scene fitting or a capacity ladder in
that campaign unless its frozen decision rules authorize a later experiment.

## After the Thayer-D3R runtime stop

The complete D1R artifact did not unblock execution readiness. A guarded import
attempted prohibited deletion and the temporary-directory bootstrap failed
before optimizer construction. This supplies no decoder mechanism or capacity
evidence.

Run exactly one next experiment: a metadata-only D3 readiness audit that proves
a deletion-free import/tempfile path and persists the frozen forward evaluator
metadata in an isolated non-Atlas-path contract. It must load no scientific
scene tensor and construct no optimizer. Do not retry D3, open broader data, or
scale capacity until that readiness audit passes.

## After the Thayer-D3B readiness pass

The runtime lifecycle blocker is resolved operationally. Bootstrap and shutdown
cleanup are confined to fresh scratch, the strict scientific path has no
Matplotlib or package-cache activity, and the pure forward evaluator is
path-independent on the synthetic reference suite. All metadata prerequisites
remain intact. The authoritative evidence is
`outputs/runs/thayer_d3_runtime_readiness_20260713_135017/`; all process-phase
inventories and the separate postprocessor lifecycle audit passed.

This does not establish that the L0 decoder can reach the D1 feature endpoint,
does not supply a D3 trajectory, and does not support a capacity or broader-data
claim. Run exactly one next experiment: a separately preregistered square-only,
one-scene authoritative D3 campaign that freezes the validated runtime, guard,
scientific launcher, postprocessor, and evaluator hashes. Do not run eight-scene
fitting, add capacity, or access Atlas, development, or lockbox data.

## After the Thayer-D3A preregistration stop

Runtime readiness alone did not provide every scientific value required by the
downstream gate. The pure evaluator needs an explicit sky vector and frozen
global, per-band, and relative-flux plausibility thresholds. D3A found no
isolated non-Atlas artifact containing those values and stopped before the
scientific interpreter.

Run exactly one next experiment: a metadata-only forward-gate
contract-isolation audit that persists those exact values, hashes, and
provenance in a non-Atlas artifact. Load no scene tensor, construct no model or
optimizer, and do not run D3 in that audit. Eight-scene work and the capacity
ladder remain unauthorized.

## After the Thayer-D3C capsule pass

The incomplete scientific-contract blocker is resolved. The exact sky vector,
plausibility and truth-coverage thresholds, tolerances, units, semantics, code
hashes, and immutable artifact references are now self-contained in one
validated capsule. This is a contract result, not a D3 scientific result: no
scientific tensor or model was loaded, and L0 reachability remains unknown.

Run exactly one next experiment: a separately preregistered square-only,
one-scene authoritative D3 campaign that freezes the exact Thayer-D3C capsule,
schema, manifest, hash chain, runtime manifest, four scientific containers, and
code hashes. It must construct all evaluator settings from the capsule alone
and must not query historical scientific configuration, open broader data, run
eight-scene fitting, or add capacity.

## After the Thayer-D3E executable-contract pass

Capsule-consumer drift, exact architecture serialization, member schemas, and
the optimizer/checkpoint execution path are now closed as readiness blockers.
The result came entirely from metadata inspection and deterministic synthetic
execution. It supplies no evidence about whether the L0 decoder reaches the D1
endpoint or passes a scientific D3 gate.

Run exactly one next experiment: a new separately preregistered square-only,
one-scene authoritative D3 campaign freezing executable bundle SHA-256
`884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045`.
Do not perform another capsule/readiness campaign unless that bundle's validator
finds a concrete defect. Do not open broader data, run eight-scene fitting, or
start the capacity ladder before the frozen D3 decision rules authorize it.

## Thayer-D3S bundle regression

The proposed scientific run cannot currently be preregistered from the bundle
alone. The registry lacks an executable expert-activity/death gate and other
required stop/outcome definitions. Do not infer thresholds from previous code
or results and do not run D3 with bundle v2.

Run exactly one next experiment: a metadata-only executable-contract v3
campaign that freezes every missing definition, adds a fail-closed corruption
test for each, and emits a new hashed bundle. It must not load scientific
tensors, fit a decoder, open broader data, or start the capacity ladder.

## After the Thayer-D3P policy-contract pass

The five missing policy families and the complete 16-policy launcher surface
are now executable, branch-complete, and frozen in bundle v3. This remains a
control-contract result: no scientific trajectory exists, and L0 success,
optimization failure, and capacity remain unknown.

Run exactly one next experiment: a separately preregistered square one-scene
scientific D3 campaign freezing bundle-v3 SHA-256
`30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c`.
It must run policy preflight and then continue directly into scientific D3.
Do not add another metadata, readiness, capsule, or policy campaign unless the
frozen bundle-v3 validator identifies a concrete defect.

## Superseding Thayer-D3I limitation and next campaign

The v4 integration path is executable, but its frozen scientific worker used a
literal dtype-string comparison (`float32` versus `<f4`) and did not preimport
two torch serialization modules before the strict phase. It stopped before any
D3 model execution. Exactly one next campaign is allowed: append-only v4.1
contract-token normalization plus serialization preimport, followed by the
same full gates and one-scene D3 retry. No broader data is authorized.

## Thayer-D3I41 limitation and one next experiment

V4.1 proved that canonical dtype comparison and serialization prewarm solve the
two v4 defects, but its added complete-member inventory used a CHW-only hash on
a higher-rank tensor. The frozen candidate stopped before model construction,
so no L0 or capacity claim is available. Exactly one next experiment is an
append-only v4.2 rank-aware inventory-hash correction followed by the same
one-scene retry after all v4.1 gates re-pass; broader data remain unauthorized.

R1's prerequisite components passed, but candidate 002 never reached
independent eligibility because its synthetic log path collided with candidate
001. Exactly one next experiment is an append-only R2 candidate-log-isolation
campaign; broader data and capacity work remain unauthorized.
## Thayer-Audit v0 limitations

The direct auditor did not obtain a usable POST comparison domain. Every
eligible valid Condition-C reconstruction was unsafe under the frozen
scientific and physical source-output contract, so AUROC was undefined and a
zero-coverage fail-closed policy was the only honest threshold result. AUPRC of
1.0 in a one-class all-positive domain is not evidence of discrimination.

PRE query detection was useful but failed the formal calibration macro-F1 gate
(`0.7980 < 0.82`). Condition C was the only eligible aligned family, so
deblender-family generalization is unproven. Atlas v0 abstention saturated at
100% for both Atlas and controls and is non-discriminative.

The one next experiment is a prospective physically compliant frozen-deblender
family-diversity audit. It must establish complete aligned source-group-safe
train/validation/calibration outputs for at least one structurally distinct
family before another catalog policy is fitted. Do not change scientific
thresholds, use truth-only inference features, restart D3, or access development
or final-lockbox outcomes.

## Thayer-PU Eligibility v1 limitation

Thayer-PU did not reach label-support evaluation. Its prompt identity was
strong, but MPS execution at batch size 1 changed every preflight candidate and
deployed canonical hash relative to batch size 8. The exact frozen replay gate
therefore failed. This result does not measure safe prevalence or invalidate
the checkpoint's earlier stochastic-candidate findings.

Exactly one next experiment is **Thayer-Audit Family-D v0 — One New Physically
Compliant Frozen Family Eligibility Audit**. It must freeze the family and
truth-free deployment rule before labels and preserve exact replay, physical
nonnegativity, OOF partitions, safety thresholds, and privacy boundaries.

## Family-E v0 limitation and next step

The preferred nonnegative exact simplex construction assumes a nonnegative
observed allocation budget, but the authoritative inputs are signed
zero-background noisy images. This is a contract-level incompatibility, not a
trained-model quality result. Family-E v0 supplies no evidence about the
preregistered U-Net's capacity and no safety-label support.

Run exactly one separately preregistered, training-free
**signed-noise-residual physical-contract preflight** next. Requested and
companion source layers must remain nonnegative; the residual/noise term alone
may be signed and must close the exact observed identity. Require complete
10,000/2,000/2,000 target representability before model construction. Do not
train Family-E, tune thresholds, add auditor complexity, or access development,
Atlas selection, or the lockbox until that preflight passes.

## Signed-noise-residual preflight result

The physical contradiction identified by Family-E v0 is resolved by treating
the observational residual as signed noise rather than a third nonnegative
source. Full target representability passed, but this is not evidence that a
coordinate U-Net can learn the mapping or produce safe catalog outputs.

Run exactly one separately preregistered Family-E1 model-eligibility campaign
next. Preserve the ReLU source map, signed closure residual, data selectors,
OOF groups, thresholds, prompt semantics, MPS requirement, and privacy
boundary. Require objective alignment and micro-overfit before full training.
Do not authorize Thayer-Audit v1 until actual safe/unsafe label support exists.

## Family-E1 v0 limitation and next step

The signed physical output space and frozen objective are viable, and the
network can memorize one ordinary paired-prompt scene. It did not preserve
requested/companion identity across the difficult and mixed-eight microsets,
despite large reconstruction-loss reductions. This is a reconstruction
prompt-ordering failure, not evidence about safe-label prevalence.

Run exactly one separately preregistered micro-only **Family-E1P
Paired-Prompt Identity Intervention** on the same scenes. Retain the
architecture, ReLU source map, signed residual, thresholds, and privacy
boundary; add one explicit paired-prompt source-ordering term and require the
unchanged 0.90 identity gate before full training. Do not generate OOF outputs,
safety labels, or train an auditor unless that separate micro gate passes.
