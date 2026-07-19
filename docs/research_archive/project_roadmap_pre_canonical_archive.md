# Thayer-Select project roadmap

1. **Complete — promptability.** Freeze the group-safe CatSim partitions,
   validate explicit-seed replay, compare centered/unprompted/randomized/
   prompted controls, and establish prompt-swap behavior.
2. **Complete with partial success — recoverability.** R0 and R1 completed,
   calibration used calibration only, and the newly frozen development manifest
   was evaluated once. Discrimination and risk–coverage improved, but ambiguity
   ranking and catastrophic-error gates failed.
3. **Next authorized — replication.** Repeat R1 with two independent initialization
   and minibatch-order seeds without changing architecture, manifests, losses,
   contracts, calibration protocol, or metrics.
4. **Not yet authorized — ambiguity benchmark.** Feasibility mining checked
   77,671 filtered candidate edges and found zero pairs meeting both provisional
   cutoffs. Do not build a full Ambiguity Atlas yet.
5. **Separately authorized — final/real-sky evaluation.** Keep the lockbox
   sealed until the full protocol is frozen. Treat DR10 as a real-sky OOD
   benchmark with its independent source-only/PSF/unit gates, not as a shortcut
   around controlled validation.

## Roadmap update after frozen-head ablation

6. **Complete — seed replication and root-cause analysis.** Phase-II instability,
   ambiguity inversion, isotonic collapse, low-SNR failure concentration, and
   unused frozen-latent information are documented.
7. **Complete with no clear improvement — frozen-head diagnostic.** H0-H4,
   calibration comparisons, the centroid augmentation, and the non-deployable
   oracle used only train/validation/calibration evidence. No development or
   lockbox evaluation occurred.
8. **Exactly one next experiment — target redesign.** Redesign and preregister
   the moderate reliability contract with failure-specific labels before any
   further head, backbone, representation, or ambiguity-construction change.
9. **Still sealed — final lockbox.** Do not use the lockbox for contract design,
   target selection, calibration, debugging, visual review, or threshold tuning.

## Roadmap update after hierarchical safety campaign

10. **Complete — hierarchical policy experiment.** Query validity, separate
    valid-only image/flux/centroid risks, confusion risk, vector scaling,
    split-conformal upper bounds, and one frozen accept/abstain rule were tested
    without changing Condition C.
11. **Successful component — query gate.** The three-state gate removed the
    ambiguity inversion, rejected all fresh development NULL queries, and cut
    AMBIGUOUS false acceptance to 9.2% at 66.65% valid-query coverage.
12. **Failed system gate — operational coverage.** The complete policy accepted
    1/2,000 development valid scenes and did not beat the historical R1 ranking
    at useful diagnostic coverage. Lockbox evaluation is not authorized.
13. **Next experiment — risk-limit feasibility and conditional conformal.** Use
    train/validation/calibration artifacts only. Audit aperture flux scaling and
    log-tail stability, preregister a fixed catastrophic-risk budget plus at
    least 70% valid calibration coverage, and compare with R1 before creating
    another development set. Keep Condition C frozen.
14. **Ambiguity benchmark — targeted pilot only.** A later pilot may combine
    simulator optimization, matched source pairs, and multi-hypothesis truth
    sets. Do not build the full Atlas and do not use development or lockbox
    scenes for ambiguity engineering.
15. **Protocol correction — complete.** Preserve the 2026-07-11 hierarchical
    result as historical evidence, but do not certify its sequence as fully
    preregistered. The 2026-07-12 corrective audit reconstructed every original
    composite label and stopped before new inference or fitting.
16. **Next authorization gate — prospective feasibility only.** Before another
    development manifest, freeze and hash a new preregistration, use one
    reconstruction provenance across train/validation/calibration, complete the
    row-level contract and drift audits, and pass the calibration-only minimum-
    coverage gate. The lockbox remains sealed.

## Roadmap update after prospective feasibility

17. **Complete with partial success — prospective component feasibility.** The
    query gate and image/flux/centroid/confusion rankers passed under uniform
    Condition-C provenance. Marginal calibration retained resolution.
18. **Frozen-gate failure — catastrophic AUPRC criterion.** The observed
    prevalence made the preregistered `1.25 × prevalence` AUPRC threshold exceed
    1.0. Preserve the failure; do not reinterpret the excellent 0.997 AUPRC as
    a formal pass.
19. **Exactly one next experiment — conditional-calibration correction.** Keep
    the reconstructor and heads frozen, preflight every gate for attainability,
    replace the unbounded AP ratio with a bounded prevalence-adjusted lift, and
    require 85–95% coverage plus bounded 95th-percentile width in each frozen
    SNR/overlap subgroup.
20. **Still prohibited — development and lockbox.** Do not build a development
    manifest or full hierarchical policy until the corrective feasibility
    experiment passes under a separately hashed preregistration.

## Roadmap update after conditional calibration

21. **Complete with failure — conditional-calibration correction.** The run
    reproduced the historical sag, passed attainability and provenance checks,
    and found every frozen subgroup supported. Centroid passed, but image and
    flux low-SNR/high-obstruction coverage was only 0.637 and 0.684.
22. **Full policy remains blocked.** Near-90% marginal coverage and strong rank
    transfer do not compensate for supported conditional failures. Do not
    generate development scenes or access the lockbox.
23. **Exactly one next experiment — partially pooled deployable scale model.**
    Keep Condition C and the selected risk heads frozen. Preregister a robust
    heavy-tail scale loss plus pooling/shrinkage over model-accessible features,
    fit with group-safe train/validation separation, and repeat the same
    natural-calibration subgroup audit. Do not add oracle inputs.

## Roadmap update after scale correction

24. **Complete with failure — partially pooled scale correction.** Exact
    baseline replay and strict OOF targets passed, but deployable image/flux
    worst-subgroup coverage was `0.549`/`0.679`. Width and ranking gates did not
    compensate for failed coverage transfer.
25. **Full policy remains blocked.** The physical-group oracle was informative
    but non-deployable and broke marginal calibration; it cannot authorize a
    policy campaign. Development and lockbox remain prohibited.
26. **Exactly one next experiment — monotone quantile scale model.** Freeze the
    same four continuous deployable proxies and fit a shape-constrained
    additive q=0.90 scale model with the same OOF targets, gates, and natural
    calibration audit. Do not run a broader search.

## Roadmap update after shape-constrained quantile correction

27. **Complete with failure — convex additive quantile scale.** The OOF proxy
    audit, preregistration, constraint checks, and prior-result replay passed.
    The single positive upper-half interaction shrank nearly to zero and did
    not improve validation-cell coverage.
28. **Full policy remains blocked.** Selected image/flux worst supported
    coverage was `0.544`/`0.591`, below both the prior baseline and frozen
    gates. Development and lockbox remain prohibited.
29. **Exactly one next experiment — convex z0-by-z1 tensor product.** Retain
    the same proxies, OOF targets, partitions, and gates. Preregister one small
    convex tensor-product quantile surface for z0 and z1 with the other two
    effects additive. Do not broaden features or run a general architecture
    search.

## Roadmap update after observable-regime distillation

30. **Superseded route — same-proxy tensor product.** The new campaign did not
    run the proposed tensor-product correction as its primary experiment. The
    exact same-proxy failures reproduced, and richer observability was tested
    first as required.
31. **Complete with information-limit failure — spatial observability.** A3
    materially improved joint-hard AUROC and AP lift, but failed frozen
    fixed-precision recall and natural-calibration Brier/ECE. Ranking signal is
    not sufficient deployable regime identification.
32. **Continuation blocked.** Do not run GroupDRO, direct upper quantiles,
    predicted-group calibration, multigroup calibration, development, or
    lockbox from this result.
33. **Exactly one next experiment — explicit PSF input.** Add an observed PSF
    representation to one separately preregistered train/validation/natural-
    calibration observability experiment. Keep Condition C frozen and do not
    combine this with other new information sources.

## Roadmap update after explicit-PSF provenance audit

34. **Complete with construction-level failure — fixed PSF.** Exact BTK/
    SurveyCodex/GalSim provenance is replayable, but all 18,000 audited scenes
    use one fixed combined g/r/z PSF configuration. Chromatic differences do
    not create scene-level information.
35. **Conditioning and policy continuation blocked.** Do not fit P0-P5,
    shuffled controls, observability heads, risk heads, calibrators, GroupDRO,
    or a full policy from the fixed-PSF benchmark. Development and lockbox
    remain prohibited.
36. **Exactly one next experiment — prospective realistic PSF variation.**
    Generate a new train/validation/natural-calibration scene campaign with a
    preregistered realistic distribution of varying, inference-available PSFs.
    Do not add IVAR, multiple observations, or other new information in the
    same experiment.

## Roadmap update after competing-hypothesis feasibility

37. **Complete with partial success — Ambiguity Atlas.** Freeze 25 validated
    near-collision pairs from the 30,000-scene prospective pool. Preserve the
    other 75 numerical pairs as candidates, not as frozen Atlas members.
38. **Superseded by Atlas v0 — finite empirical witnesses.** The authoritative
    Atlas audit retains two scientifically divergent constructed decompositions
    on 50/50 noisy observations and 19/50 same-cluster model witnesses.
39. **Cross-family continuation blocked.** Condition C, R0, and R1 form one
    compatible architecture cluster. Do not train Thayer-Audit, run catalog
    coverage, or make model-agnostic claims.
40. **Exactly one next experiment — prompted ResUNet family.** Preregister and
    train one compact prompted ResUNet under the frozen BTK source-layer and
    normalization contracts, validate full-decomposition replay, then rerun
    only the frozen Atlas behavior audit and reassess family compatibility.

## Roadmap update after Ambiguity Atlas v0

41. **Complete — direct Atlas feasibility.** Freeze 25 noise-dominated but
    replayable Route-1 pairs; preserve 25 valid Route-2 optimization pairs as
    feasibility evidence.
42. **Failed — operational candidate-diameter gate.** Diameter did not beat
    self-confidence or forward residual and has zero recall at the frozen
    control threshold. Do not train Thayer-Audit or produce catalog policies.
43. **Exactly one next experiment — prompted ResUNet family.** Add one
    preregistered compatible residual architecture, replay the frozen Atlas,
    and reassess candidate diversity before any auditor training.

## Roadmap update after prompted ResUNet promptability failure

44. **Complete with pre-Atlas failure — prompted ResUNet.** The architecture,
    source exclusions, full manifest replay, and MPS fit passed, but prompt-swap
    and requested-identity gates failed on fresh validation scenes.
45. **Atlas and auditor continuation blocked.** Do not run ResUNet Atlas
    inference, recompute candidate diameter, admit a second family, add a third
    family, or train Thayer-Audit from this run.
46. **Exactly one next experiment — explicit multi-hypothesis generation.**
    Preregister one coordinate-conditioned conditional VAE under the same
    source-layer contract and Atlas exclusions. Require non-Atlas promptability
    plus forward-consistent multi-sample diversity before any Atlas evaluation;
    do not train another deterministic U-Net variant.

## Roadmap update after Thayer-PU partial success

47. **Complete — prompt-faithful stochastic family.** Canonical hashing, source
    isolation, 20,000-scene replay, MPS training, latent use, promptability,
    prior quality, forward consistency, and control concentration passed.
48. **Partial — Atlas discrimination without truth coverage.** Witnesses rose
    to 24/50, AUROC to 0.856, and 4%-FPR recall to 0.32, but the 30/50 witness
    target and both truth-coverage gates failed.
49. **Auditor continuation blocked.** Do not train Thayer-Audit, admit catalogs,
    rerun Atlas, or tune the current prior from Atlas outcomes.
50. **Exactly one next experiment — prior correction.** Preregister a compact
    conditional normalizing-flow prior on the frozen Thayer-PU representation,
    retaining every current non-Atlas and one-pass Atlas gate.

## Roadmap update after Thayer-PF sufficiency failure

51. **Complete with pre-fit failure — posterior/decoder sufficiency.** Persisted
    baselines and the frozen metric reproduced, but posterior own-truth and
    cross-decoded alternate-truth coverage were all zero.
52. **Flow and Atlas continuation blocked.** Do not implement or fit a flow,
    construct latent teachers, resample Atlas, train an auditor, or access the
    final lockbox from this campaign.
53. **Exactly one next experiment — ambiguity-set decoder training.** Present
    both approved near-collision decompositions under each observationally
    equivalent non-Atlas condition while preserving prompt identity and forward
    consistency; require truth coverage before any prior correction.

## Roadmap update after Thayer-MH coverage failure

54. **Complete — prospective ambiguity-set construction.** Expanded Atlas-pool
    exclusion, 2,000 approved pairs, 19,000 exact replays, and K=2 loss tests
    passed.
55. **Failure — shared token decoder representation.** Promptability and forward
    consistency passed, but ordinary, own, alternate, and both-mode truth
    coverage were all zero.
56. **Atlas and auditor continuation blocked.** Do not freeze an Atlas protocol,
    rerun Atlas, tune thresholds, train Thayer-Audit, or access the lockbox.
57. **Exactly one next experiment — separate experts.** Preregister one K=2
    shared-prompt-encoder model with two compact expert decoders, retaining
    permutation-invariant approved-target matching, ordinary concentration, and
    every current source-exclusion and forward-consistency gate.

## Roadmap update after Thayer-ME micro-overfit failure

58. **Complete — independent-expert implementation and isolation audit.** The
    165,612-parameter architecture, exact target reuse, expert independence,
    initialization, prompt, decomposition, and MPS-only checks passed.
59. **Failure — training-only representational-capacity gate.** Prompt swap and
    forward consistency passed, but ordinary, own, alternate, and both-mode
    truth coverage stayed at zero after the frozen 400-epoch micro protocol.
60. **Full training, Atlas, and auditor continuation blocked.** Do not enlarge
    the model, rerun the micro gate, fit on validation, access Atlas, train an
    auditor, or access the lockbox.
61. **Exactly one next experiment — frozen loss-geometry audit.** On persisted
    micro targets and outputs only, decompose normalized reconstruction,
    source-sum, flux, color, centroid, and frozen scientific distances to test
    whether the training objective ranks exact truth proximity correctly;
    perform no neural fitting and do not change the coverage metric.

62. **Exactly one next experiment — scientific-gradient objective microgate.**
    Run one prospective micro-overfit-only Thayer-ME experiment on the same
    frozen 64 rows using source-set reconstruction plus ordinary concentration
    and a preregistered differentiable surrogate of the unchanged scientific
    distance. Retain forward consistency as an evaluation gate rather than an
    optimized forward-to-observed term. Do not open Atlas, development, or
    lockbox data unless that future campaign independently passes every
    non-Atlas gate.

## Roadmap update after Thayer-SA preflight failure

63. **Complete — scientific surrogate alignment.** The physical g/r/z
    differentiable surrogate passed rank, threshold-side, perturbation, and
    exact-truth stationary tests without changing thresholds.
64. **Failure — detached output-space optimization.** Loss decreased from
    several compromise starts, but ordinary and both-mode truth coverage did
    not approach the preregistered 90% target; random bounded outputs barely
    improved.
65. **Blocked — assignment and neural stages.** Do not audit assignment as a
    continuation, fit Thayer-ME, open full manifests, or access protected data
    after the failed prerequisite.
66. **Exactly one next experiment — output-space conditioning.** Preregister a
    training-free comparison of near-truth smooth scientific component
    geometry under the same targets, thresholds, hard assignment, and detached
    output initializations. Require coverage entry before neural fitting.

## Roadmap update after Thayer-OC

67. **Complete — preregistered conditioning comparison.** Six fixed methods,
    five fixed initializations, exact-truth controls, and unchanged scientific
    gates were run without neural parameters or protected-data access.
68. **Partial — coverage entry without a global pass.** Selected endpoints
    materially exceeded raw-space baselines, but no method cleared every 90%
    gate across the fixed starts. Adam-based T/D methods were ineligible after
    exact-truth drift.
69. **Blocked — neural conditioning transfer.** Do not fit a T/D head, use the
    Jacobian preconditioner in training, select optimizers per scene, or open
    Atlas, development, or lockbox data.
70. **Exactly one next experiment — direct feasibility learning.** Run one
    separately preregistered micro-audit that projects or learns feasibility in
    the unchanged frozen scientific region. Preserve targets and thresholds.
71. **Complete — direct feasible projection.** P0 placed all 64 microset target
    sets inside the unchanged scientific region with strict 0.95 target slack.
72. **Failed — unchanged neural micro capacity.** Direct projected-target
    learning retained zero coverage in every category and failed output
    nonnegativity after 400 MPS-only epochs.
73. **Exactly one next experiment — controlled decoder-capacity ladder.** Keep
    the P0 targets, microset, assignment, prompt contract, thresholds, and
    protected-data boundary fixed; vary only preregistered decoder capacity.
74. **Stopped — capacity-ladder output-contract prerequisite.** Thayer-CL
    reproduced Thayer-FP but found no unique frozen nonnegative neural output
    mapping. No decoder condition was constructed or fitted.
75. **Exactly one next experiment — output parameterization at fixed L0.**
    Prospectively compare the preregistered nonnegative head mappings under
    identical L0 architecture, P0 targets, optimizer, and micro sanity gates;
    freeze one global mapping before returning to the width ladder.

## Roadmap update after Thayer-OP

76. **Complete — fixed-L0 physical-mapping audit.** ReLU, square, and absolute
    value each represented all P0 targets, passed stop self-tests and synthetic
    fitting, and eliminated physical negative values under matched compute.
77. **Failed — one-scene scientific memorization.** No mapping passed the final
    ordinary gate or recovered both ambiguous truth modes. Eight-scene fitting
    stopped by preregistration, and no output mapping was selected.
78. **Blocked — decoder-capacity ladder.** Thayer-CL measured no capacity
    result, and Thayer-OP does not authorize L1-L3 construction or fitting.
79. **Exactly one next diagnostic — fixed-feature decoder optimization.** On
    the frozen ambiguous scene, retain the same hard assignment and mapping and
    compare the L0 neural decoder trajectory with direct cached-feature output
    optimization.
80. **Closed — repository integrity and D0-D2 reachability.** The exact-path
    audit found no result-changing production defect. Square passed D0 and D1;
    ReLU and absolute value failed D0; square failed D2. The result is a
    frozen-feature conditioning barrier. D3, tangent diagnostics, L1-L3, and
    broader data remained closed.
81. **Exactly one next experiment — prospectively authorized square D3.**
    Preregister a one-scene square-mapping full-L0 fixed-feature diagnostic
    that retains D2 as the failed control but authorizes D3 after the proven D1
    pass. Reuse the same cache, endpoint, loss, assignment, evaluation, and
    access restrictions; do not add capacity or broader data.
82. **Stopped — D1 endpoint reference incomplete.** Thayer-D3 reproduced the
    controls and exact square initial state but found that the successful D1
    artifact omitted both optimized penultimate tensors. No optimizer was
    constructed and no D3 or tangent result was produced.
83. **Exactly one next experiment — D1 endpoint-persistence replay.** Re-run
    only the square D1 free-feature condition under its exact frozen scene,
    cache, heads, loss, assignment, optimizer, and budget. Save both optimized
    penultimate tensors and verify their frozen-head outputs. Do not run D3,
    add capacity, or open broader data in that experiment.
84. **Complete — exact D1 endpoint persistence and replay.** Thayer-D1R
    reproduced all 54 trajectory hashes, 100/100/100 coverage, assignment,
    objective, and outputs; persisted four semantic prompt/expert tensors; and
    passed restricted fresh-process and batch/serialization replay.
85. **Exactly one next experiment — separate square-only D3.** Start a new
    preregistered one-scene full-L0 fixed-feature campaign using the complete
    D1 reference. Do not run eight-scene fitting, add capacity, or access
    broader/protected data unless that future D3 result authorizes it.
86. **Stopped — Thayer-D3R runtime readiness.** The complete D1R reference,
    guard self-test, frozen-input hashes, and 600 historical checkpoints passed,
    but a guarded import attempted prohibited deletion and temporary bootstrap
    failed before optimizer construction. No D3 result exists.
87. **Exactly one next experiment — metadata-only D3 readiness audit.** Prove a
    deletion-free import/tempfile path and persist the frozen forward evaluator
    metadata in an isolated non-Atlas-path contract. Load no scene tensors and
    construct no optimizer.
88. **Complete — Thayer-D3B runtime readiness.** Bootstrap cleanup was confined
    to disposable scratch; strict phases had zero deletion, cache write,
    Matplotlib import, protected access, or blocked read. Cold, warm, and
    post-shutdown processes passed, the pure evaluator matched its independent
    reference on all twelve cases, every process-phase inventory was frozen,
    the postprocessor lifecycle stayed confined, and all metadata prerequisites
    remained valid. The authoritative record is
    `outputs/runs/thayer_d3_runtime_readiness_20260713_135017/`.
89. **Exactly one next experiment — authoritative square-only D3.** Start one
    separately preregistered one-scene full-L0 campaign with the frozen D3B
    runtime, guard, scientific launcher, postprocessor, and evaluator hashes.
    Do not open broader data, run eight-scene fitting, or add capacity.
90. **Stopped — Thayer-D3A preregistration completeness.** Runtime hashes,
    scientific container hashes, and 600 historical checkpoints matched, but
    the isolated evidence did not persist the scientific sky vector and
    plausibility thresholds required by the forward gate. No scientific
    process, tensor load, model, optimizer, or decoder forward ran.
91. **Exactly one next experiment — forward-gate contract isolation.** Persist
    the exact scientific sky vector and global, per-band, and relative-flux
    plausibility thresholds with hashes and provenance in a non-Atlas artifact.
    Load no scene tensor and do not run D3 in that experiment.
92. **Complete — Thayer-D3C scientific contract capsule.** Preregistration
    preceded value extraction; all 97 dependencies resolved; schema, hash
    chain, corruption, evaluator, zero-I/O, cwd/environment, and checkpoint
    gates passed. No scientific tensor, model, optimizer, or D3 step occurred.
93. **Exactly one next experiment — capsule-only authoritative D3.** Start one
    separately preregistered square-only one-scene D3 campaign that freezes the
    exact capsule, schema, manifest, hash chain, runtime manifest, four tensor
    containers, and code hashes. It must not query historical configuration,
    open broader data, run eight-scene fitting, or add capacity.
94. **Stopped — capsule-v1 consumer contract drift.** The capsule-v1 producer,
    schema, and validators passed, but the actual consumer required nine
    undeclared evidence, architecture, initialization, and member-schema
    entries. No scientific tensor, model, optimizer, or D3 step ran.
95. **Complete — Thayer-D3E executable contract.** One canonical 180-item
    registry now drives capsule-v2 builder, validator, preflight, and consumer.
    Exact artifact headers, L0 construction, strict state loading, synthetic
    MPS forward/backward/update, checkpoint replay, 25 corruption tests, and
    runtime requirement closure passed with no scientific data access.
96. **Exactly one next experiment — bundle-frozen authoritative D3.** Freeze
    executable bundle SHA-256
    `884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045`
    in a new preregistration, then run only the authorized square one-scene L0
    D3. Do not query historical configuration, open broader data, or run the
    capacity ladder.
97. **Fail closed on the D3S bundle regression.** The bundle-driven
    preregistration audit found no executable expert-activity/death gate and
    additional required trajectory definitions. D3 did not run. The only next
    experiment is a metadata-only executable-contract v3 campaign that freezes
    the missing settings and tests them without scientific tensor access.
98. **Complete — Thayer-D3P policy closure.** Sixteen canonical policies now
    drive one pure engine used by the actual launcher. Seventy-six fixtures,
    106 branch assertions, 256 outcome combinations, all semantic states,
    exact set equality, and 30 bundle corruptions passed with zero scientific
    work.
99. **Exactly one next experiment — bundle-v3 scientific D3.** Freeze bundle-v3
    SHA-256
    `30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c`,
    run policy preflight, then continue directly into the authorized square
    one-scene scientific D3 trajectory. Do not introduce another readiness or
    policy campaign unless bundle-v3 validation fails concretely.

100. **Exactly one next campaign — v4.1 contract-token normalization.** Add an
     append-only bridge/worker version that compares NumPy dtype identity or a
     normalized token and preimports `torch.utils.serialization` before the
     strict phase. Repeat regression, synthetic, 25-corruption, flow, and
     source-freeze gates before retrying the same one-scene D3. Do not open
     broader data.

14. **Thayer-D3I41 closed fail-closed.** Both planned v4 corrections and every
    integration gate passed, but the scientific retry stopped before model
    construction on a higher-rank member-inventory hash-domain error. The one
    next experiment is an append-only v4.2 inventory-hash correction followed
    by the same one-scene retry after all existing gates re-pass.

- Thayer-D3I41R1 closed the audited dtype, adapter, and exact-test prerequisites
  but stopped before eligibility on a candidate-log collision. The single next
  experiment is a separately authorized append-only R2 log-isolation rerun.
## Thayer-Audit v0 disposition

Thayer-Audit v0 is complete with **DIRECT_AUDITOR_PARTIAL** at
`outputs/runs/thayer_audit_v0_20260714_154655/`. PRE query detection was useful
but missed the frozen calibration macro-F1 gate. POST had unsafe prevalence
1.0 under the unchanged scientific and physical contract, so no two-class
ranking or nonzero-coverage safe policy was possible. Held-family transfer is
unresolved and Atlas v0 remained diagnostic-only.

Next: run exactly one separately preregistered prospective physically compliant
frozen-deblender family-diversity audit. Do not restart D3, run a capacity
ladder solely because this audit failed, train on development/lockbox outcomes,
or authorize Audit/Atlas v1 yet.

## Thayer-PU Eligibility v1 disposition

The unchanged Thayer-PU family stopped before full inference because its
single-scene and batched canonical hashes differed under the frozen rule.
Promptability passed, but deployment eligibility did not. Thayer-Audit v1
remains unauthorized. Run exactly one next experiment: **Thayer-Audit Family-D
v0 — One New Physically Compliant Frozen Family Eligibility Audit**. Do not
post-hoc select another Thayer-PU sampling rule from label outcomes.

## Family-E v0 disposition

Family-E v0 stopped at the preregistered Part-E gate before model construction.
Three nonnegative requested/companion/residual contributions cannot conserve
the signed zero-background observation, and the frozen targets are not
representable. Thayer-Audit v1 remains blocked.

Exactly one next experiment is authorized for recommendation, not execution:
a separately preregistered training-free signed-noise-residual
physical-contract preflight. It must keep requested/companion source outputs
nonnegative, allow only the residual/noise layer to be signed, conserve the raw
observation exactly, and prove full target representability before any
architecture is built.

## Family-E1 signed-residual preflight disposition

The training-free physical-contract correction passed all frozen provenance,
MPS, target-representability, source-nonnegativity, residual-sign, and
conservation gates. The prior all-nonnegative simplex failure remains
authoritative for that construction.

Next, run exactly one separately preregistered
**Thayer-Family-E1-v0 — Nonnegative-Source Signed-Residual Model Eligibility**
campaign. Do not train an auditor or access development, Atlas selection, or
the lockbox. The model campaign must independently pass every learning,
deployment, unchanged-safety, label-support, and distinctness gate.

## Family-E1 v0 disposition

Family-E1 passed the signed physical contract, architecture count, objective
alignment, and ordinary one-scene micro gate, then failed the mandatory
mixed-eight prompt-identity gate. Full seeds, OOF folds, labels, comparison,
bootstrap, and auditor training remain closed. Exactly one next experiment is
a separately preregistered micro-only **Family-E1P Paired-Prompt Identity
Intervention** on the identical frozen micro scenes; no full training is
authorized unless it clears the unchanged 0.90 identity gate.
