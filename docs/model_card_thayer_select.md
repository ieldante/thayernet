# Model card: Thayer-Select

Thayer-Select is a compact U-Net family for selecting and reconstructing a
requested isolated galaxy from a controlled two-source BTK blend using three
normalized `g,r,z` channels and one Gaussian coordinate prompt. It is a
research model, not a survey production deblender.

Phase-I Condition C has 119,091 parameters and a reconstruction head only.
Phase-II R0 keeps that reconstruction-only architecture but trains on valid,
perturbed-valid, null, and ambiguous queries. Phase-II R1 shares the same
backbone and adds a finite bounded pixel log-variance map, a global empirical
contract-success probability, and a no-source probability. Morphology, source
identity, split labels, simulator difficulty, clean targets, evaluation errors,
and masks never enter the forward pass.

The completed R0 has 119,091 parameters. R1 has 123,368 parameters (+4,277),
bounded log variance `[-8, 2]`, a global actionable-success head, and a
no-source head. R1 completed 20 epochs on MPS after fail-closed loss-design
corrections. Its selected calibrator is isotonic and its primary development
contract is PERMISSIVE because every predeclared actionable contract was highly
imbalanced on training/validation labels.

Intended use is controlled investigation of promptability, recoverability, and
selective abstention under frozen BTK manifests. Known limitations include
heavy-tailed reconstruction errors, simulator dependence, one primary training
seed unless replication completes, uncertain transfer to real data, and the
fact that a calibrated global score does not automatically calibrate each pixel
uncertainty. Null and ambiguous queries require abstention-aware evaluation.
The current R1 is not a successful safety model: ambiguous queries rank above
clear valid queries on average, catastrophic failure does not fall at 80%
coverage, and null hallucination is not better than R0 or frozen Phase-I C on
identical new null-coordinate scenes. Full uncertainty maps were not persisted
for the one-time development pass. The lockbox remains unavailable for model
selection, debugging, calibration, threshold tuning, figures, or qualitative
inspection.

## Hierarchical safety-policy addendum

The reconstruction model remains Phase-I Condition C at SHA-256
`e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`.
The hierarchical campaign did not add parameters to, fine-tune, or alter that
model. It adds external lightweight CPU heads over frozen model-accessible
features:

- a five-seed F_COMBINED small-MLP ensemble for UNIQUE_VALID/NULL/AMBIGUOUS;
- separate five-seed image-, flux-, and centroid-risk median/q=0.90 heads;
- a separate five-seed confusion-risk ensemble;
- vector scaling, temperature scaling, and split-conformal upper residuals fit
  on natural calibration only.

These heads are a research diagnostic, not a deployable release. The query gate
worked, but the complete moderate-limit policy accepted only 0.05% of valid
development scenes. The full-policy zero false acceptance for invalid queries
therefore reflects abstention collapse, not useful safety. Raw risk-head outputs
also showed log-to-linear tail instability, although log-space rankings and
conformal marginal coverage were strong. Do not expose a reconstruction for a
NULL or AMBIGUOUS query; do not treat marginal conformal coverage as
class-conditional or deployment coverage; do not use generator variables as
head inputs.

The fresh development manifest was evaluated once after freeze. No lockbox
evaluation exists. Current classification is **FAILURE** for the complete
hierarchical policy.

## Protocol-status correction

The 2026-07-11 hierarchical result remains valid historical evidence but is not
certified as a fully preregistered sequence: the required preregistration and
full original-composite postmortem were absent before head fitting and the
one-time development evaluation. A later append-only audit reproduced every
persisted Phase-II moderate label exactly and made no model or policy change.
Do not interpret that retrospective audit as preregistration, and do not
authorize lockbox evaluation from the historical result.

## Prospective feasibility addendum

The 2026-07-12 feasibility campaign did not change Thayer-Select or produce a
new deployable model. It reused byte-identical Phase-I Condition C for all
training, validation, and calibration outcomes and fitted external CPU-only
query/risk heads over frozen model-accessible features.

The query gate and all risk rankers showed strong prospective signal, but the
campaign is only PARTIAL SUCCESS: its catastrophic AUPRC gate was unattainable
at the observed prevalence, and image/flux calibration was only marginally—not
conditionally—reliable. No operational threshold or combined policy exists.
There was zero development and lockbox access. Intended use remains controlled
promptable selective galaxy deblending research; survey deployment, final
selective-risk claims, and lockbox evaluation remain prohibited.

## Conditional-calibration addendum

The prospective conditional-calibration run did not alter the Thayer-Select
reconstruction model. It trained only external CPU risk and scale heads over
the frozen F_COMBINED representation and performed no reconstruction
inference. Selected heads retained natural-calibration Spearman `0.870`,
`0.862`, and `0.952` for image, flux, and centroid.

This did not establish reliable conditional bounds. Image and flux coverage in
the adequately supported low-SNR/high-obstruction intersection was `0.637` and
`0.684`, while centroid's lowest supported coverage was `0.888`. The overall
campaign is **FAILURE**. Physical source variables were used only for auditing,
not as deployable features. No operational policy, development result, or
lockbox result exists.

## Partially pooled scale-correction addendum

The scale-correction run did not modify or execute the Thayer-Select
reconstruction backbone. It used persisted frozen features and trained only
small external CPU scale models from source-group-held-out residual targets.
All deployable inputs were available from observed blends, prompts, and frozen
model outputs; physical simulator groups remained audit-only.

The primary partially pooled model retained image/flux calibration Spearman
`0.877`/`0.866` with bounded median-width inflation `1.336x`/`1.055x`, but
worst supported coverage was only `0.549`/`0.679`. IMAGE_RISK and FLUX_RISK
therefore fail; CENTROID_RISK remains a reproduced PASS. No operational use,
full policy, development claim, lockbox claim, or end-to-end safety claim is
authorized.

## Shape-constrained quantile addendum

This campaign added no reconstruction capability and did not change
Thayer-Select weights. It fitted external CPU-only q=0.90 scale models from
four frozen deployable proxies and source-group-held-out residual targets.
Convexity, upper-half monotonicity, and nonnegative interaction constraints all
held, but the interaction did not improve validation coverage and is not a
promoted component.

Validation selected the additive convex model for both risks. Their worst
supported natural-calibration coverage was only `0.544` and `0.591`, despite
strong retained ranking. IMAGE_RISK and FLUX_RISK fail;
CENTROID_RISK remains a reproduced PASS. No operational policy, development
result, lockbox result, or full-policy authorization exists.

## Observable-regime distillation addendum

The observability campaign did not alter or fine-tune Thayer-Select. It used
frozen prompt-local encoder patches, observed blend patches, persisted
candidate/residual patches, and frozen risk outputs to train small external
CPU heads. Simulator SNR and obstruction were supervision/evaluation labels
only; no physical oracle value entered deployable inference arrays.

Spatial features improved the joint-hard AUROC from `0.711` for the historical
four-proxy baseline to `0.901` across five A3 seeds, transferring at `0.880` on
natural calibration. The head is not a deployable regime detector: fixed-
precision recall, Brier, and ECE gates failed, and continuous magnitudes did
not transfer safely. The campaign stopped before GroupDRO, new risk quantiles,
or predicted/multigroup calibration. IMAGE_RISK and FLUX_RISK remain FAIL;
CENTROID_RISK remains PASS. No full policy, development result, lockbox result,
or end-to-end safety authorization exists.

## Explicit-PSF audit addendum

The PSF campaign did not alter, execute, or fine-tune Thayer-Select and fitted
no external model. It audited the exact observation-process configuration used
by the historical BTK scenes. The LSST g/r/z PSFs are deterministic,
axisymmetric Kolmogorov-plus-Airy profiles with fixed 0.86/0.81/0.77 arcsec
FWHM. All 18,000 audited scenes share one combined configuration.

The result is **PSF NON-INFORMATIVE BY CONSTRUCTION**, not evidence that PSF
conditioning is generally useless. The current data lack the scene-level PSF
variation needed to identify an incremental effect. IMAGE_RISK and FLUX_RISK
remain FAIL; CENTROID_RISK remains PASS. No PSF-conditioned component,
operational policy, development result, lockbox result, or end-to-end safety
authorization exists.

## Competing-hypothesis Atlas addendum

Condition C, reconstruction-only R0, and reconstruction-only R1 were evaluated
without fine-tuning on the frozen 25-pair Ambiguity Atlas. All three can be
queried for every declared source and mapped back to detected electrons without
clipping, but they share one compact prompted-U-Net family cluster and do not
constitute cross-family evidence.

All 75 pair/model combinations contained at least one unsafe noisy requested
reconstruction. Condition C produced nearly the same mean-scene answer on all
25 pairs despite divergent truths; R0 did so on 16 and R1 on 1. R1's private
recoverability diagnostic was low on these cases and was excluded from every
finite-candidate witness. This is an Atlas stress result, not a population
failure rate or catalog policy. No auditor, development evaluation, lockbox
evaluation, or model-agnostic transfer claim exists.

## Ambiguity Atlas v0 addendum

On the new frozen Atlas, all three same-cluster checkpoints again produced at
least one unsafe requested reconstruction on every pair. Constructed competing
truths demonstrate ambiguity on 50/50 observations, but model-candidate
diameter does not operationalize that fact: AUROC 0.4712 with zero recall at the
frozen control threshold. This is a failed safety-gate result, not permission
to use R1 confidence as a catalog policy. Development and lockbox remain
untouched.

## Prompted ResUNet feasibility addendum

The separately trained 199,219-parameter prompted ResUNet is not a promoted
Thayer-Select component. It used the same input/output normalization contract
and excluded every Atlas-related group, but failed the frozen pre-Atlas
promptability gate: 39.47% prompt-swap success and 69.5% individual requested-
source success. It was never evaluated on the Atlas, development set, or
lockbox. No second candidate family, uncertainty estimate, auditor, catalog
policy, or model-agnostic claim follows from this experiment.

## Thayer-PU stochastic candidate addendum

Thayer-PU is a 170,278-parameter experimental stochastic candidate family, not
a promoted Thayer-Select reconstructor or calibrated posterior. Its truth-free
prior generates six-channel requested/companion decompositions; its posterior
uses truth only during training. All Atlas-related groups were excluded from
fit, validation, calibration, and non-Atlas collision construction.

The model passed non-Atlas promptability and forward-consistency gates and
improved frozen Atlas candidate-diameter evidence to 24/50 witnesses, AUROC
0.856, and 0.32 recall at 4% control false positives. It achieved zero own and
alternate truth coverage on Atlas, so no auditor, catalog policy, production
claim, formal posterior-correctness claim, development result, or lockbox result
is authorized.

## Thayer-PF blocked addendum

Thayer-PF is not a model artifact. Its required pre-fit evaluation found zero
posterior own-truth and cross-decoded alternate-truth coverage on protected
non-Atlas data. No flow prior, mixture base, checkpoint, deployable hypothesis,
or Atlas result was produced. The frozen Thayer-PU decoder and posterior remain
diagnostic research components, not truth-covering posterior machinery.

## Thayer-MH failed experimental addendum

Thayer-MH is a 120,022-parameter experimental K=2 set decoder, not a promoted
Thayer-Select model. It passed 0.992 prompt-swap and high forward-consistency
gates but achieved zero ordinary, own, alternate, and both-mode scientific
coverage. No Atlas evaluation, auditor input, catalog policy, development claim,
production claim, or lockbox result follows from this failed non-Atlas campaign.

## Thayer-ME failed capacity addendum

Thayer-ME is a 165,612-parameter experimental two-expert decoder, not a promoted
Thayer-Select model. Its independent expert parameters and training-only
microset isolation passed audit. The model remained prompt-faithful and forward-
consistent but achieved zero ordinary, own, alternate, and both-mode scientific
coverage. Full training was prohibited. No Atlas result, auditor input, catalog
policy, development claim, production claim, or lockbox result exists.

## Thayer-LG objective-geometry limitation

The frozen Thayer-ME objective is not a reliable proxy for scientific source
truth. Exact truths pass the output and coverage contracts, yet compromise
configurations often receive lower loss and full-objective output optimization
can reduce loss while leaving every ambiguity coverage region. Forward
consistency must not be reported as source-identification correctness. No
Atlas, development, or lockbox evaluation was performed in this audit.

## Thayer-SA failed objective-correction addendum

Thayer-SA is an objective preflight, not a promoted model. Its differentiable
scientific distance tracked the frozen metric and assigned zero loss to truth,
but the official detached optimizer did not recover high truth coverage from
compromise or random starts. The campaign stopped before assignment auditing or
neural fitting, created no checkpoint, and does not authorize full training,
Atlas evaluation, an auditor, a catalog policy, development claims, production
claims, or lockbox access.

## Thayer-OC conditioning addendum

Thayer-OC is a detached-output audit, not a model or promoted training method.
No globally fixed condition passed every truth-coverage gate, and three
Adam-based T/D conditions failed exact-truth stationarity. Selected endpoints
show partial coverage gains only. The run created no model checkpoint and does
not authorize neural training, Atlas evaluation, an auditor, a catalog policy,
development claims, production claims, or lockbox access.

## Thayer-FP feasibility-projection addendum

P0 offline targets inside the unchanged scientific region are feasible
training-only representatives, not new truth and not inference features. The
unchanged Thayer-ME failed to memorize them under direct reconstruction: every
scientific coverage rate remained zero and the final output failed
nonnegativity. Current decoder capacity, encoder conditioning, or output
parameterization is directly implicated on the microset. No full training,
Atlas evaluation, auditor, catalog policy, development result, production
claim, or lockbox result is authorized.

## Thayer-CL contract-preflight addendum

Thayer-CL is not a trained model or capacity result. It reproduced Thayer-FP,
confirmed that the unconstrained head emitted negative physical source layers,
and stopped because no unique compliant replacement mapping was defined by the
frozen contracts. No L0-L3 model, capacity comparison, full training, Atlas
evaluation, auditor, catalog policy, development result, production claim, or
lockbox result is authorized.

## Thayer-OP output-parameterization addendum

Thayer-OP is a fixed-L0 micro-overfit audit, not a promoted model. ReLU, square,
and absolute value were each used identically by loss and evaluation and each
enforced finite nonnegative physical source layers. All three passed target
representability and synthetic fitting, but none passed the final ordinary or
ambiguous one-scene scientific gate. No mapping, capacity ladder, full
training, Atlas evaluation, auditor, catalog policy, development result,
production claim, or lockbox result is authorized.

## Repository-integrity fixed-feature evidence

The audited one-scene path agrees with independent scientific references and
contains no proven result-changing implementation defect. Square reached both
approved truth modes in D0 and D1, while its frozen-penultimate final-head-only
D2 condition failed. ReLU and absolute value stopped at D0. D3 was not opened,
so this is a frozen-feature conditioning diagnosis rather than evidence for or
against L0 decoder capacity. It does not authorize model promotion, broader
training, a capacity ladder, Atlas evaluation, development use, or lockbox use.

## Thayer-D3 frozen-input addendum

Thayer-D3 is not a trained model and produced no decoder-optimization result.
Its exact square controls, cache, targets, and initial state reproduced, but the
successful D1 artifact did not persist the optimized penultimate tensors needed
by the prospective feature-trajectory contract. The campaign stopped before an
optimizer or gradient trace existed.

No L0 sufficiency, decoder-capacity, mapping practicality, eight-scene,
promotion, Atlas, development, production, or lockbox claim follows. The only
authorized next evidence step is the separate D1 endpoint-persistence replay.

## Thayer-D3R execution-readiness addendum

The complete D1R endpoint was available, but the authoritative retry stopped
before optimizer construction after a guarded import attempted prohibited
deletion. Thayer-D3R is not a trained model and supplies no L0 decoder,
capacity, mapping-practicality, promotion, Atlas, development, production, or
lockbox claim. Broader fitting remains unauthorized.

## Thayer-D3A preregistration addendum

Thayer-D3A is not a trained model. Its frozen runtime, scientific containers,
and checkpoints matched, but the permitted evidence lacked the scientific sky
and plausibility-threshold values required by the forward gate. The campaign
stopped before a tensor load, decoder, optimizer, gradient, or D3 step. No L0
sufficiency, capacity, mapping-practicality, promotion, Atlas, development,
production, or lockbox claim follows.

## Thayer-D3S bundle-regression addendum

Thayer-D3S is not a trained model or scientific evaluation. Its bundle hash
matched, but the registry omitted required expert-activity/death and related
trajectory definitions. The campaign stopped before preregistration, tensor
loading, model construction, gradients, or D3. No sufficiency, capacity,
promotion, production, Atlas, development, or lockbox claim follows.

## Thayer-D3I update

V4 launcher integration passed synthetically, but authoritative science
stopped after eight allowlisted container loads and before model construction.
The outcome is `IMPLEMENTATION_OR_CONTRACT_FAILURE`, not evidence about model
quality or L0 capacity. No promotion, eight-scene, capacity-ladder, Atlas,
development, lockbox, or production authorization follows.

## Thayer-D3I41 evidence boundary

V4.1 cleared the earlier dtype and serialization defects, then stopped in
member-inventory reporting before model construction. This result is not
evidence about Thayer-Select quality, L0 optimization, or decoder capacity.
There is no promotion, eight-scene, capacity-ladder, Atlas, development,
lockbox, or production authorization.

R1 did not change this status. It stopped before candidate eligibility or
scientific payload access on an append-only orchestration-log collision. No
model-quality, optimization, generalization, or capacity claim is supported.
## Thayer-Audit v0 external safety layer

Thayer-Audit v0 is not a promoted model or catalog policy. Its PRE network sees
only blend g/r/z and the Gaussian prompt; its POST network sees only blend,
prompt, frozen reconstruction, residual, and deployable diagnostics. Truth is
used only for supervision/evaluation. PRE may abstain before reconstruction;
POST may reject a proposal afterward.

The result is **DIRECT_AUDITOR_PARTIAL**. PRE query detection was informative
but missed its calibration macro-F1 gate. All eligible Condition-C valid
outputs were unsafe under the frozen scientific/physical contract, leaving no
safe POST class and forcing zero accepted coverage. Held-family generalization
is unresolved. Atlas v0 was development-only; development outcomes and the
final lockbox were untouched.

D3 remains a valid narrow negative result for one multi-hypothesis decoder
setup, but D3 success is not required in principle for an external auditor.
No prospective Audit/Atlas v1 is authorized by v0.

## Thayer-PU eligibility boundary

The unchanged epoch-27 Thayer-PU checkpoint remains a structurally distinct,
promptable stochastic candidate, not an eligible audit family. Under the
preregistered mean-of-16 rule, repeated batch-8 and batch-4 execution matched,
but single-scene execution changed all 24 preflight canonical hashes. The
campaign stopped before full inference and produced no new safety prevalence,
family-comparison, calibration, auditor, development, Atlas-selection, or
lockbox result. Thayer-Audit v1 is not authorized.

## Family-E physical-contract boundary

Family-E v0 did not instantiate or train its preregistered model. The
nonnegative exact simplex allocation was valid only for nonnegative observed
budgets, whereas the frozen zero-background observations preserve signed
noise. Target representability therefore failed before architecture
construction. No Family-E output, safety prevalence, family generalization, or
auditor result exists. Condition C and repaired Thayer-PU remain unsafe-only;
Thayer-Audit v1 remains unauthorized.

## Signed-noise-residual contract update

A training-free Family-E1 preflight established that nonnegative
requested/companion source layers plus a signed noise residual can represent
all frozen targets and conserve signed zero-background observations. This
removes the earlier physical output-space contradiction only. No trained
Family-E1 model or safety result exists, so Condition C and Thayer-PU remain
the only evaluated unsafe-only families and Thayer-Audit v1 remains
unauthorized.

## Family-E1 model boundary

Family-E1 v0 instantiated one 1,162,662-parameter compact coordinate U-Net.
Its requested and companion catalog sources remained nonnegative under
in-forward ReLU, its signed residual conserved the raw observation, and its
objective-alignment audit passed. It is not a frozen eligible reconstruction
family: the mandatory mixed-eight prompt-identity gate failed at 0.5625 before
full training. No Family-E1 checkpoint, OOF output, safety label, family
comparison, or auditor result exists. Development, Atlas selection, and the
final lockbox remain untouched.
