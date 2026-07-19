# Current project status

Phase I promptability is complete and frozen. Coordinate prompting removed the
centered-target shortcut, halved mean requested-source MSE relative to the
randomized unprompted control, and achieved 98% prompt-swap success. It did not
learn no-source semantics: empty prompts hallucinated under the declared
criterion in 100% of cases.

Phase II recoverability and selective abstention completed under the frozen,
append-only protocol with **PARTIAL SUCCESS**. R0 and R1 completed 20 MPS
epochs; R1 added 4,277 parameters, kept log variance inside `[-8, 2]`, and used
isotonic calibration selected by calibration-only cross-validation. Calibration
AUROC/AUPRC were `0.8746`/`0.2475`; Brier improved from `0.1010` raw to `0.0456`
calibrated. All-query permissive-contract risk declined from `0.9560` at full
coverage to `0.9379` at 70%, but catastrophic failure increased at 80% and 70%
coverage, and ambiguous queries received a higher mean score (`0.1017`) than
clear valid queries (`0.0614`). R1 null hallucination was `8.25%`, versus
`7.5%` for frozen Phase-I C and `2.25%` for R0 on identical new null-coordinate
scenes. This does not meet the success gates.

The source split and lockbox policy are unchanged; the lockbox remained sealed.
Current evidence is controlled BTK development evidence. DR10 is still a
real-sky out-of-distribution benchmark and is not evidence of calibrated survey
performance. The next authorized experiment is two fixed-protocol R1 training-
seed replications; a full Ambiguity Atlas and lockbox evaluation are not
authorized.

## Conditional-calibration status — 2026-07-12

The prospective correction run
`outputs/runs/thayer_select_conditional_calibration_20260712_021556/` is
complete with **FAILURE**. It reproduced the original `0.691429` sag and found
all frozen physical subgroups adequately supported. Marginal coverage remained
near 90%, ranking transferred strongly, and centroid passed at worst supported
coverage `0.888`. Image and flux failed: low-SNR/high-obstruction coverage was
only `0.637` and `0.684`, respectively. The attainable catastrophic sanity
gate passed.

Condition C was unchanged and no reconstruction inference was needed. No
development data or policy was created, and the lockbox remained sealed. A
full policy campaign is still not authorized. The only next experiment is a
preregistered train/validation/calibration-only partially pooled deployable
scale-model correction.

## Superseding status — frozen-representation diagnostic (2026-07-11)

The fixed-protocol seed replication and root-cause analysis are complete. The
subsequent frozen-head ablation is recorded in
`outputs/runs/thayer_select_frozen_head_ablation_20260711_220756/` and is
classified **NO CLEAR IMPROVEMENT**. Balanced heads improved validation AUPRC,
but the validation-selected H2 head degraded to calibration AUROC/AUPRC
0.514/0.032, kept ambiguous prompts above valid prompts, and did not reject
catastrophic source failures reliably. Temperature scaling retained more
resolution than isotonic but still produced 100% realized coverage at several
nominal operating points because the selected MLP scores saturated.

The encoder remained fully frozen, no reconstruction backbone was retrained,
development was not evaluated, and the lockbox remained sealed. The only next
recommended experiment is a preregistered redesign of the moderate
reliability-contract target and its failure-specific labels.

## Superseding status — hierarchical safety policy (2026-07-11)

The hierarchical campaign is complete in
`outputs/runs/thayer_select_hierarchical_safety_20260711_225657/` and is
classified **FAILURE under the frozen gates**. This classification does not
erase successful subcomponents: the five-seed UNIQUE_VALID/NULL/AMBIGUOUS gate
reached balanced validation macro F1/AUPRC `0.881`/`0.923`, recalled NULL at
`99.85%` and AMBIGUOUS at `88.89%`, and removed ambiguity-over-valid inversion
in every seed. Valid-only image, flux, and centroid heads also ranked the tail,
and split conformal upper bounds reached approximately 90% natural-calibration
coverage.

The complete policy was not operational. It accepted only 1/4,200 natural-
calibration valid scenes, 0/1,000 stratified-diagnostic valid scenes, and
1/2,000 fresh development valid scenes. On development, query-gate-only false
acceptance was `0%` for NULL and `9.2%` for AMBIGUOUS, but the full policy's
near-zero coverage made its zero invalid acceptance unusable. At 95%, 90%,
80%, and 70% diagnostic valid coverage, hierarchical catastrophic rates were
`0.825`, `0.816`, `0.793`, and `0.764`; these were not materially better than
the historical R1 ranking.

Condition C remained byte-identical and fully frozen. The fresh 3,000-scene
development manifest was generated after policy freeze and evaluated exactly
once. The lockbox remained sealed. Recoverability is now documented as a
derived policy, not a monolithic training label, but that policy is not ready
for deployment or lockbox evaluation.

## Protocol correction — 2026-07-12

An append-only compliance audit found that the historical hierarchical run did
not contain the required pre-fit preregistration or full original-contract
postmortem. Those omissions cannot be repaired retrospectively after its
one-time development evaluation. The exact persisted moderate composite was
subsequently reconstructed for all 13,500 Phase-II rows with zero Boolean
reapplication mismatches, but it confirmed a heterogeneous target and a
training/validation-versus-calibration reconstruction-provenance change. No new
inference or training was performed. The historical complete-policy result
remains FAILURE, and a new prospectively preregistered train/validation/calibration-only
feasibility campaign is required before another development set is authorized.

## Prospective feasibility status — 2026-07-12

The prospective train/validation/calibration-only run
`outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/` is
complete with **PARTIAL SUCCESS**. Its preregistration was hashed before every
fit, Condition C was the only reconstructor across all 32,000 scenes, and the
applicability audit found zero prior-style logical defects. Query validity and
all continuous valid risks were strongly learnable; score calibration remained
noncollapsed.

The frozen catastrophic AUPRC gate was impossible at the observed 81.65%
prevalence, and image/flux conditional calibration remained uneven despite
90% marginal coverage. Those gates were not changed. No full policy or
development manifest was created, the lockbox remained sealed, and the
historical hierarchy remains FAILURE. A future full-policy campaign is not yet
authorized.

## Superseding scale-correction status — 2026-07-12

The prospective partially pooled run
`outputs/runs/thayer_select_scale_correction_20260712_024957/` completed with
**FAILURE**. It reproduced the prior image/flux/centroid results exactly and
used genuinely out-of-fold training residual targets. Partial pooling retained
ranking and bounded widths but produced image/flux worst supported coverage
`0.549`/`0.679`; bootstrap lower bounds were `0.477`/`0.614`. Only the
non-deployable physical-group oracle exceeded 0.90 subgroup coverage.

Condition C and deployed heads remained frozen. No full policy exists,
development and lockbox remained untouched, and a hierarchical-policy
campaign is not authorized. The one next experiment is a preregistered
train/validation/calibration-only monotone quantile scale model over the same
four deployable proxies.

## Shape-constrained quantile status — 2026-07-12

The prospective shape-constrained run
`outputs/runs/thayer_select_shape_constrained_quantile_20260712_033406/`
completed with **FAILURE**. The training-only OOF audit reproduced the strong
proxy-tail reversals and rejected global monotonicity. All convexity,
upper-half monotonicity, and interaction-sign constraints passed.

Validation selected Q1 for both risks; the positive z0-by-z1
interaction did not improve worst supported validation-cell coverage. Selected
natural-calibration marginal/worst coverage was `0.9221`/`0.5440` for image
and `0.9221`/`0.5907` for flux. Bootstrap lower bounds were `0.4730`/`0.5222`.
Both risks fail, centroid remains PASS, and no hierarchical-policy campaign is
authorized. Development and lockbox remained untouched.

## Observable-regime distillation status — 2026-07-12

The prospective run
`outputs/runs/thayer_select_observability_distillation_20260712_035843/`
completed with **OBSERVATIONAL INFORMATION LIMIT — FAILURE**. Rich spatial
features substantially improved joint-hard ranking over the same-proxy A0
baseline: A3 five-seed validation AUROC was `0.901 ± 0.004` versus `0.711`,
and natural-calibration AUROC was `0.880`. SNR was strongly rank-predictable,
but obstruction was weaker.

The frozen gate still failed. Recall at precision 0.70 was `0.083`, natural-
calibration Brier `0.140` exceeded the `0.064` prevalence reference, and ECE
`0.219` exceeded `0.15`. GroupDRO, new quantile models, and predicted or
multigroup calibration were therefore not run. Image/flux remain FAIL,
centroid remains PASS, and no policy campaign is authorized. The one next
experiment is a separately preregistered observability study with explicit PSF
input. Development and lockbox remain untouched.

## Explicit-PSF information status — 2026-07-12

The prospective audit
`outputs/runs/thayer_select_psf_conditioning_20260712_043442/` stopped with
**PSF NON-INFORMATIVE BY CONSTRUCTION**. Exact PSF provenance is available:
BTK used fixed axisymmetric LSST Kolmogorov-plus-Airy profiles with g/r/z FWHM
0.86/0.81/0.77 arcsec. However, all 18,000 training, validation, and natural-
calibration scenes share one combined PSF configuration, with no spatial or
between-scene variation.

The mandatory variation gate stopped the campaign before preregistration,
association analysis, model fitting, controls, risk continuation, calibration,
or policy work. Pixels-only A3 remains authoritative; image and flux remain
FAIL, centroid remains PASS, and no development or lockbox result exists. The
single next experiment is to prospectively generate scenes with realistic
varying PSFs.

## Competing-hypothesis recoverability status — 2026-07-12

The prospective run
`outputs/runs/thayer_competing_hypotheses_20260712_131111/` completed with
**PARTIAL SUCCESS — ATLAS AND FINITE AMBIGUITY WITNESSES WORK; CROSS-FAMILY
AUDITING IS BLOCKED**. A 30,000-scene approved training/validation search found
100 numerical near-collision candidates and froze the first 25 after exact
replay and visual artifact review. The superseding Atlas v0 audit records
constructed truth witnesses on 50/50 noisy observations and same-cluster model
candidates on 19/50.

Condition C, R0, and reconstruction-only R1 share one compact prompted-U-Net
family cluster. No Thayer-Audit model, leave-one-family-out evaluation,
catalog policy, development result, or lockbox result was produced. The exact
next experiment is one prospective prompted ResUNet addition followed only by
the frozen Atlas behavior audit and a renewed compatible-family count.

## Ambiguity Atlas v0 status — 2026-07-12

The authoritative v0 run
`outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/` passes Atlas
construction but fails the operational ambiguity-witness detector gate. Route
1 found 100 numerical pairs from 30,000 training/search scenes and froze 25;
Route 2 produced 25/25 bounded optimization-feasibility pairs. Constructed
witnesses pass on 50/50 frozen observations, but same-cluster candidate
diameter has AUROC 0.4712 and zero recall at the frozen 4% control false-positive
rate. No auditor, catalog policy, development access, lockbox access, or
cross-family claim is authorized.

## Prompted ResUNet candidate-diversity status — 2026-07-12

The prospective run
`outputs/runs/thayer_prompted_resunet_diversity_20260712_154122/` completed with
**PROMPTABILITY FAILURE — ATLAS EVALUATION NOT AUTHORIZED**. The compact
residual model had 199,219 parameters and was freshly trained on 10,000 scenes;
all 59 Atlas-related source groups were excluded, and all 11,500 train/validation
definitions replayed exactly.

On 1,500 fresh validation scenes, prompt-swap success was `0.3947` and
individual requested-source success was `0.695`, below the frozen `0.80` and
`0.75` gates. Whole-image MSE was `1.1205x` Condition C and output collapse was
`0.00067`. The frozen Atlas was not evaluated, so no candidate-diversity,
witness, AUROC, or operating-point result changed. No second family, third
family, auditor, or catalog policy is authorized.

## Thayer-PU multi-hypothesis status — 2026-07-12

The authoritative run
`outputs/runs/thayer_probabilistic_unet_20260712_163340/` completed with
**PARTIAL SUCCESS — STOCHASTIC DIAMETER IMPROVED; TRUTH COVERAGE FAILED**.
The 170,278-parameter model completed 30 MPS epochs after all 59 Atlas-related
groups were excluded and 20,000/20,000 scenes replayed exactly.

All non-Atlas gates passed. The one-time Atlas pass produced 24/50 witnesses,
AUROC 0.856 with bootstrap interval 0.751–0.942, and 0.32 recall at 4% control
false positives. Own and alternate truth coverage were both zero, so the 30/50
witness and coverage gates failed. No Thayer-Audit, catalog policy, development
result, or lockbox result is authorized.

## Thayer-PF truth-coverage status — 2026-07-12

The append-only run `outputs/runs/thayer_flow_prior_20260712_182516/` stopped at
the preregistered posterior/decoder sufficiency gate. The frozen metric audit
and persisted Thayer-PU reproduction passed. K=32 posterior own-truth coverage
was 0% on both ordinary and non-Atlas near-collision scenes; cross-decoded
alternate coverage was 0%. No flow was implemented or trained, Atlas evaluation
count is zero, and development/lockbox access counts remain zero. The exact next
experiment is one ambiguity-set decoder-training campaign.

## Thayer-MH ambiguity-set decoder status — 2026-07-12

The append-only run
`outputs/runs/thayer_multiple_hypotheses_20260712_190701/` completed with
**FAILURE — NON-ATLAS AMBIGUITY-SET TRUTH COVERAGE FAILED; ATLAS PROHIBITED**.
All 19,000 scenes replayed exactly, 2,000/2,000 prospective pairs passed target
construction, and the 120,022-parameter shared K=2 decoder completed 30 MPS
epochs. Prompt-swap success was 0.992 for both tokens and set level; requested
MSE was 0.864 times Condition C. Ordinary own, near-own, near-alternate, and
both-mode coverage were all zero. Atlas/development/lockbox access counts remain
0/0/0, and no auditor or catalog policy is authorized.

## Thayer-ME two-expert capacity status — 2026-07-12

The append-only run
`outputs/runs/thayer_two_expert_decoder_20260712_203121/` stopped with
**REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE** at the preregistered
micro-overfit gate. Thayer-MH baselines and all reused target hashes reproduced
before implementation. The 165,612-parameter model had two disjoint 46,470-
parameter decoders and one shared 72,672-parameter prompted encoder.

After 400 MPS epochs on 32 ordinary and 32 ambiguous training-only observations,
expert/set prompt swap and forward consistency passed, but ordinary, own,
alternate, and both-mode truth coverage remained zero. Full training,
calibration, Atlas, development, and lockbox access counts are 0/0/0/0/0. No
auditor or catalog policy is authorized.

## Thayer-LG frozen loss-geometry result (2026-07-12)

The authoritative training-free run is
`outputs/runs/thayer_loss_geometry_20260712_205733/`. It reproduced the
Thayer-ME micro-overfit failure, proved exact-truth representability and metric
coverage, and found direct objective misalignment: compromise beat truth on
84.375% of rows and full-objective optimization from truth destroyed coverage
while lowering loss. The primary classification is `MIXED CAUSE`, led by
objective misalignment, forward-term scale dominance, and gradient conflict.
Atlas, development, and lockbox access remained zero.

## Thayer-SA scientific-alignment result (2026-07-12)

The append-only run
`outputs/runs/thayer_scientific_alignment_20260712_220315/` stopped with
**FAILURE — CORRECTED OBJECTIVE STILL MISALIGNED** at the preregistered detached
output-space gate. The surrogate passed exact-metric alignment with Spearman
0.990679, Kendall 0.957683, and 100% threshold-side agreement. Exact truth was
stationary and fully covered. Compromise starts lowered loss but did not reach
the required coverage, and random bounded outputs did not materially improve.
Assignment auditing and MPS neural training were not reached. Full non-Atlas
training remains unauthorized; Atlas, development, and lockbox access are zero.
An append-only protocol addendum records that detached-output optimizer smoke
checks occurred before the formal freeze, so the strict superseding correctness
status is FAIL even though the official persisted preflight was frozen first.

## Thayer-OC output-conditioning result (2026-07-12)

The preregistered training-free run is
`outputs/runs/thayer_output_conditioning_20260712_225459/`. All authoritative
baselines reproduced and 593 historical checkpoints remained unchanged. No
global conditioning method passed all frozen 90% gates. Raw L-BFGS produced the
strongest ambiguous own/alternate endpoints from Thayer-ME outputs, while raw
Adam produced the strongest both-mode endpoint from the persisted Thayer-SA
compromise; neither supplied a deployable global result. Adam-based T/D methods
failed truth stationarity. Scientific status is **PARTIAL SUCCESS — SCIENTIFIC-
BASIN EXTREMITY**. Strict correctness status is **FAIL** because the frozen
actual-objective HVP/finite-difference condition estimate was unresolved. No
neural training, Atlas, development, or lockbox access occurred.

## Thayer-FP direct feasibility projection

The preregistered run is
`outputs/runs/thayer_feasibility_projection_20260712_234216/`. Exact truths and
all authoritative baselines reproduced. P0 projected every target pairing into
the strict training interior and achieved complete ordinary and ambiguous
target-set coverage without changing thresholds. The unchanged Thayer-ME then
failed after 400 MPS-only epochs: every truth-coverage category remained zero,
ordinary diameter was 3.564, and final outputs violated nonnegativity. Prompt
swap and forward consistency stayed strong. Status: **FAILURE — PROJECTED
TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT MEMORIZE THEM**. Atlas,
development, and lockbox access remained zero.

## Thayer-CL output-contract preflight

The preregistered run is
`outputs/runs/thayer_capacity_ladder_20260713_013132/`. All 24 Thayer-FP checks
reproduced. The prior negatives originated in the raw linear decoder head and
were present in physical source layers; the first persisted violation was at
epoch 1, while the exact batch was not recorded. No frozen contract selects a
unique nonnegative replacement among multiple admissible mappings. Status:
**FAIL-CLOSED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING**. No model was
constructed, no optimizer step occurred, and capacity remains unresolved.
Strict correctness is **PASS** with 27/27 checks; the pre-load and closure
inventories confirm all 594 historical checkpoints are unchanged. The earlier
`20260713_005215` attempt remains preserved as a superseded strict-correctness
failure for its incomplete initial checkpoint inventory. Atlas, development,
and lockbox access remained zero.

## Thayer-OP fixed-L0 output parameterization

The authoritative preregistered run is
`outputs/runs/thayer_output_parameterization_20260713_023120/`. ReLU, square,
and absolute value all represented the frozen P0 targets, passed stop-rule and
synthetic MPS preflights, and maintained finite nonnegative physical outputs.
The frozen encoder remained byte-identical and only the output mapping differed.
All three mappings nevertheless failed the frozen final ordinary and ambiguous
one-scene coverage gates. Status: **NO MAPPING PASSES**. The eight-scene gate
was not opened, no mapping was selected, and the decoder-capacity ladder is not
authorized. Atlas, development, and lockbox access remained zero.

## Thayer-RI fixed-feature closure — 2026-07-13

The strict exact-path repository audit closed with no result-changing
production defect. Independent references, seven golden cases, exact lineage,
differential truth injection, and one-step gradient/optimizer traces passed.

The fixed-feature ladder produced a **FROZEN-FEATURE CONDITIONING BARRIER**.
Square passed D0 and D1 at 100% own, alternate, and both-mode coverage. ReLU
and absolute value failed D0. Square failed D2 with zero scientific coverage,
so D3 and tangent diagnostics were not authorized. Decoder capacity remains
unresolved; a capacity ladder is not authorized. Ordinary, eight-scene,
remaining-microset, Atlas, development, and lockbox access remained zero, and
all 600 inventoried historical checkpoints remained byte-identical.

## Thayer-D3 full-L0 fixed-feature diagnostic — 2026-07-13

The square-only D3 campaign froze preregistration before all tensor loads and
reproduced persisted D0/D1/D2, all frozen hashes, the immutable joined prompt
cache, target hashes, and the exact square initial state. It then stopped
fail-closed before optimizer construction because the successful D1 endpoint
artifact contained output tensors and metrics but not its optimized
penultimate tensors.

Status: **FROZEN-INPUT PROVENANCE FAILURE — D3 NOT RUN**. Decoder optimization
and capacity remain unresolved. Neither eight-scene fitting nor a capacity
ladder is authorized. Ordinary, broader microset, Atlas, development, and
lockbox data remained untouched.

## Thayer-D1R square endpoint persistence — 2026-07-13

The exact square-D1 free-feature optimization was replayed on the same frozen
ambiguous scene. All 54 scheduled physical hashes, the final objective, raw,
mapped and physical outputs, hard assignment, and 100% own/alternate/both-mode
coverage reproduced exactly. Four named prompt/expert penultimate tensors are
now persisted under a documented schema.

Fresh-process replay reproduced all output hashes and scientific metrics. All
13 batch, position, layout, serialization, canonicalization, and MPS-to-CPU
checks passed with zero difference. Status: **SUCCESS — D1 ENDPOINT PERSISTED
AND REPLAYED**. D3 was not run; decoder capacity remains unresolved. Exactly
one separate square-only D3 campaign is now authorized. Eight-scene fitting
and a capacity ladder remain unauthorized, and all broader/protected data
remained untouched.

## Thayer-D3R authoritative retry — 2026-07-13

The fresh retry preregistered the complete D1R endpoint, passed the exact-path
guard self-test, and matched all 600 historical checkpoints before scientific
execution. It then stopped fail-closed before optimizer construction: a
Matplotlib dependency attempted guard-prohibited cache deletion, and PyTorch's
temporary-directory probe could not complete under the no-delete policy. The
frozen ignored-guard-event rule prohibited an in-campaign retry.

Status: **EXECUTION-READINESS FAILURE — D3 NOT RUN**. No L0 sufficiency,
capacity, assignment, or square-optimization conclusion follows. Eight-scene
fitting and the capacity ladder remain unauthorized. Broader scenes, Atlas
arrays, development, and lockbox remained untouched.

## Thayer-D3B runtime readiness — 2026-07-13

The authoritative metadata-only readiness record is
`outputs/runs/thayer_d3_runtime_readiness_20260713_135017/`. Its scientific
interpreters isolated NumPy and PyTorch bootstrap behavior in disposable
scratch and never imported Matplotlib. A separate Matplotlib/Agg interpreter
handled only a synthetic figure. The primary, both cold, warm-cache, and
shutdown-audited scientific processes passed and emitted the exact readiness
marker. The pure forward evaluator matched an independent reference on twelve
synthetic cases with zero filesystem access.

All 21 D1R prerequisites and all 600 historical checkpoint hashes still
matched. No scientific tensor was deserialized, no model or optimizer was
constructed, and D3 was not run. All process-phase inventories and the
postprocessor lifecycle audit passed. Status: **READINESS PASS — D3 SCIENTIFIC
STATUS UNKNOWN**. One separately preregistered authoritative D3 campaign is
operationally permitted. Eight-scene fitting, the capacity ladder, Atlas,
development, and lockbox remain closed.

## Thayer-D3A authoritative preregistration — 2026-07-13

The fresh append-only record is
`outputs/runs/thayer_authoritative_d3_20260713_145040/`. Its standard-library
bootstrap matched 27/27 runtime hashes, 11/11 scientific container hashes, and
600/600 historical checkpoint hashes before any third-party import or tensor
deserialization.

Status: **PREREGISTRATION INCOMPLETE — D3 NOT RUN**. The isolated allowed
evidence does not persist the exact scientific sky vector and plausibility
thresholds required by the forward and truth-coverage gates for novel D3
outputs. No model, optimizer, gradient, decoder forward, or D3 trajectory
exists. Exactly one metadata-only forward-gate contract-isolation audit is
next. Eight-scene fitting and the capacity ladder remain unauthorized; broader
scenes, Atlas, development, and lockbox were untouched.

## Thayer-D3C scientific contract capsule — 2026-07-13

The fresh metadata-only record is
`outputs/runs/thayer_d3_scientific_capsule_20260713_155637/`. It froze
preregistration before extracting the exact sky or threshold values, then
resolved all 97 scientific dependencies into one canonical capsule.

Status: **SCIENTIFIC CONTRACT CAPSULE PASS**. The schema, manifest, hash chain,
16 corruption tests, 12 synthetic evaluator comparisons, zero-I/O audit, and
four cwd/environment process modes passed. The capsule-only launcher emitted
both required readiness markers. No scientific tensor, model, optimizer,
decoder forward, gradient, or D3 step occurred; Atlas scenes, development, and
lockbox remained untouched, and all 600 historical checkpoints remained
unchanged.

One new separately preregistered authoritative D3 campaign is contractually
authorized using this exact capsule. D3 remains scientifically unknown;
eight-scene fitting and the capacity ladder remain unauthorized.

The earlier append-only preclosure run
`outputs/runs/thayer_d3_scientific_capsule_20260713_153815/` is preserved but
non-authoritative. Closure review found that its generic small-JSON guard
counted selected values but did not recursively enforce rank inside a mapping.
The corrected run added direct rank-1 acceptance, rank-2 rejection, and
65-scalar rejection proofs before rebuilding the capsule.

## 2026-07-13 — Thayer-D3E executable contract

Status: **EXECUTABLE D3 CONTRACT PASS — SCIENTIFIC D3 NOT RUN**.

The prior capsule-driven attempt stopped because capsule v1's self-consistent
97-entry schema omitted nine requirements enforced by the actual D3 consumer.
Thayer-D3E reproduced those nine identifiers exactly and built append-only
capsule v2 from one canonical 180-requirement registry. Builder, validator,
preflight, consumer, and actual runtime-access sets were identical.

All scientific container headers passed. The exact two-expert square L0 model
instantiated at 46,470 parameters per expert, loaded both initial states
strictly, and completed a production-shape synthetic MPS forward, production
and reference assignment/loss/evaluator comparisons, one AdamW step, checkpoint
save/reload, and fresh-process replay. All 25 corrupted capsules failed before
model execution. The actual consumer emitted both required readiness markers.

Scientific array loads, scientific D3 steps, Atlas, development, and lockbox
access were all zero. Authoritative D3 is now executable and authorized only in
a new separately preregistered campaign freezing bundle SHA-256
`884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045`.
D3 remains scientifically unknown; broader data and the capacity ladder remain
closed.

## Thayer-D3S bundle regression — 2026-07-13

The append-only campaign
`outputs/runs/thayer_scientific_d3_20260713_170508/` stopped with
**EXECUTABLE BUNDLE REGRESSION — D3 NOT RUN**. The bundle hash matched and the
registry contained 180 entries, but it did not freeze the required
expert-activity/death gate, prompt-collapse stop, tangent protocol, complete
outcome mapping, or semantic-state rules. No setting was inferred.

Preregistration was deliberately not frozen. Scientific tensor loads, model
constructions, decoder forwards, optimizer steps, D3 steps, and protected-data
access were all zero. All 600 historical checkpoints remained unchanged. D3
and L0 capacity remain unknown; eight-scene fitting and the capacity ladder are
not authorized.

## 2026-07-13 — Thayer-D3P policy closure

Status: **D3 POLICY CONTRACT PASS — SCIENTIFIC D3 NOT RUN**.

The exact bundle-v2 regression reproduced, then a preregistered 16-policy
registry and pure engine closed the scientific launcher's complete control
surface. Seventy-six fixtures executed all 106 branches, all 256 outcome
vectors mapped exactly once, all 11 semantic states passed persistence and
fresh-process replay, and 30/30 bundle corruptions were rejected. Declared,
defined, accessed, tested, persisted, and launcher policy sets matched.

The actual launcher emitted all four policy readiness markers for bundle-v3
SHA-256
`30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c`.
Scientific tensor, model, optimizer, decoder-forward, D3-step, Atlas,
development, and lockbox counts were zero. The next experiment is one
separately preregistered scientific D3 run freezing this exact hash.

## Thayer-D3I — v4 integration pass, frozen scientific contract stop

Bridge-v4 SHA-256
`3ab6e4a525297f48cc7fd9428651c604aa1236ed0a4425f9953c5b5772345dc5`
passed the actual synthetic worker/replay/postprocess path and 25 corruption
tests. Scientific continuation loaded eight containers / 91 members, then
stopped before model construction on a dtype-token validation defect and two
blocked cache reads. Outcome is `IMPLEMENTATION_OR_CONTRACT_FAILURE`;
authorization is `none`; broader and protected data remained untouched.

## Thayer-D3I41 status

Candidate 002 passed all v4/v4.1 integration gates and corrected the prior
dtype and serialization failures. The same eight containers / 91 members and
four D1 dtype contracts validated, but added member-inventory hashing failed
before model construction. Outcome is `IMPLEMENTATION_OR_CONTRACT_FAILURE`;
authorization is `none`; L0 sufficiency remains unknown.

Thayer-D3I41R1 passed the 19 exact contract tests and production-adapter probes,
then stopped fail-closed when candidate 002 collision-refused candidate 001's
synthetic log path. No scientific payload was loaded; authorization is `none`.
## 2026-07-14 — Thayer-Audit v0 direct catalog-safety audit

Run: `outputs/runs/thayer_audit_v0_20260714_154655/`.

- Preregistration SHA-256:
  `3ca55b23997c8bfb0d6be2d395096020ab04df1d730f043d04a0b7c6d6a9f1c2`.
- D3 remains a valid negative result for one frozen two-expert decoder setup;
  it is not a prerequisite for the external auditor and was not retrained or
  repaired.
- PRE-AUDIT used only blend g/r/z plus prompt. Validation/calibration macro-F1
  was `0.8947/0.7980`; null recall `1.0000/0.9988`; ambiguity recall
  `0.9009/0.9100`.
- POST-AUDIT used only blend, prompt, frozen reconstruction, residual, and 25
  deployable scalars. No truth-only inference feature entered either network.
- Every eligible valid Condition-C reconstruction was unsafe under the frozen
  safety OR. AUROC was undefined, the AUPRC-lift gate was unattainable, and the
  fail-closed policy accepted zero valid requests.
- One eligible family left held-family generalization unresolved. Atlas v0 was
  post-freeze development-only and produced 100% abstention on both Atlas and
  matched controls.
- Outcome: **DIRECT_AUDITOR_PARTIAL**. No prospective Audit/Atlas v1 is
  authorized. The one next experiment is a prospective physically compliant
  frozen-deblender family-diversity audit.
- Development outcomes and the final lockbox were untouched; no reconstruction
  model or historical checkpoint changed.

## 2026-07-14 — Thayer-PU Eligibility v1

Run: `outputs/runs/thayer_pu_eligibility_v1_20260714_213113/`. Condition C's
12,493 unsafe / 0 safe labels reproduced. The frozen unchanged Thayer-PU
mean-of-16 rule passed promptability and repeated/batch-4 replay but failed
exact single-scene versus batched hashes for all 24 preflight scenes. Outcome:
**THAYER_PU_DEPLOYMENT_INELIGIBLE**. Full inference, safety labeling, bootstrap,
family comparison, and auditor training did not run. Development, Atlas
selection, and final lockbox remained untouched; Thayer-Audit v1 is not
authorized.

## 2026-07-14 — Thayer-Audit Family-E v0

Run: `outputs/runs/thayer_family_e_v0_20260714_195256/`.
Preregistration SHA-256:
`256bffe3bc53b572b7596bba844f0afdbf4abf3c4cb1d8906fc0ad08663d8881`.
The nonnegative exact simplex construction passed synthetic MPS checks but
failed frozen-target representability: roughly 48% of observed pixels are
negative under zero-background noise, and every required partition contains
target sums above the observation. Outcome:
**DATA_OR_IMPLEMENTATION_FAILURE**. No model, checkpoint, output, safety label,
or auditor was produced. Development, Atlas selection, and lockbox access
remain zero; Thayer-Audit v1 is not authorized.

## 2026-07-14 — Family-E1 signed-noise-residual preflight

Run:
`outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340/`.
Outcome: **SIGNED_NOISE_RESIDUAL_CONTRACT_PASS**. All 14,000 frozen target
pairs were representable with nonnegative requested/companion ReLU source
layers and a signed algebraic residual. Maximum float32 closure error was
`0.015625`; mapped-source negative count was zero. No model, optimizer,
checkpoint, reconstruction, label, or auditor was produced. A separately
preregistered Family-E1 model-eligibility campaign is authorized;
Thayer-Audit v1 is not.

## 2026-07-14 — Thayer-Family-E1-v0

Run: `outputs/runs/thayer_family_e1_v0_20260714_214715/`. Outcome:
**FAMILY_E1_RECONSTRUCTION_FAILURE**. The 1,162,662-parameter compact
coordinate U-Net preserved in-forward ReLU nonnegative requested/companion
sources and exact signed-residual closure. Objective alignment passed, no
compromise beat truth, and ordinary one-scene micro-overfit passed. Difficult
and mandatory mixed-eight prompt identity failed at `0.50` and `0.5625`
against `0.90`, so full training and every downstream output/label stage were
prohibited. Development, Atlas selection, lockbox, and auditor-training counts
remain zero. Thayer-Audit v1 is not authorized.
