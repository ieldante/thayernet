# Thayer-Audit v0

The authoritative run is
`outputs/runs/thayer_audit_v0_20260714_154655/`. Its preregistration SHA-256 is
`3ca55b23997c8bfb0d6be2d395096020ab04df1d730f043d04a0b7c6d6a9f1c2`.
The outcome is **DIRECT_AUDITOR_PARTIAL** under the append-only outcome-mapping
correction preserved in the run.

D3 remains a valid negative result for one frozen two-expert decoder setup: two
fresh 46,470-parameter expert decoders did not learn the two approved hidden
modes under the square mapping, hard assignment, direct objective, optimizer,
and 5,000-step budget. That result is not a prerequisite for an external audit
layer and was not retrained, repaired, extended, or reinterpreted here.

The PRE-AUDIT network used only the observed normalized g/r/z blend and the
Gaussian coordinate prompt. It achieved validation/calibration macro-F1
`0.8947/0.7980`, null recall `1.0000/0.9988`, and ambiguous recall
`0.9009/0.9100`. Query detection was useful, but the calibration macro-F1
missed the frozen `0.82` gate.

The POST-AUDIT network used only the blend, prompt, frozen Condition-C
reconstruction, observation-minus-reconstruction residual, and 25 deployable
scalar diagnostics. Truth was used only to form supervision and evaluation
labels. Every eligible valid reconstruction was unsafe under the frozen OR of
scientific image/flux/color/centroid, confusion, physical nonnegativity,
false-subtraction, and worse-than-baseline conditions. POST unsafe prevalence
was therefore `1.0`; AUROC was undefined rather than invented, and the
prevalence-plus-0.15 AUPRC gate was mathematically unattainable.

No calibration threshold satisfied the joint constraints. The frozen policy
failed closed below the minimum POST score, accepted zero valid requests, and
therefore failed the 50% coverage gate. Its apparent 100% risk reduction is the
trivial consequence of accepting nothing, not a usable catalog result.

Condition C was the only core-eligible frozen family. R0/R1 share its
architecture cluster, Thayer-PU lacked complete aligned out-of-fit outputs
under one deployment sampling rule, and prompted ResUNet failed promptability.
Held-family generalization is unresolved and no deblender-agnostic claim is
supported.

Atlas v0 remained development-only and was evaluated only after model,
calibrator, and threshold freeze. The fail-closed policy abstained on all 50
Atlas observations and all 25 matched controls, a saturated non-discriminative
diagnostic. Development outcomes and the final lockbox were untouched.

No prospective Audit/Atlas v1 is authorized. The one recommended next
experiment is a prospective physically compliant frozen-deblender
family-diversity audit before another catalog-policy attempt.

## Thayer-PU eligibility follow-up

The follow-up run
`outputs/runs/thayer_pu_eligibility_v1_20260714_213113/` reproduced Condition
C's 12,493/12,493 unsafe labels with zero substantive mismatch. Unchanged
Thayer-PU promptability passed, but exact single-scene versus batched output
hashes failed on all 24 preflight scenes. Outcome:
`THAYER_PU_DEPLOYMENT_INELIGIBLE`. Full inference and labeling did not run, so
the one-family limitation and Thayer-Audit v1 prohibition remain.

## Family-E v0 follow-up

The repaired fixed-batch Thayer-PU continuation ultimately produced 0 safe /
7,591 unsafe outputs. Family-E v0 then froze a nonnegative exact simplex
allocation but failed target representability because the authoritative
zero-background observations contain signed noise. Outcome:
`DATA_OR_IMPLEMENTATION_FAILURE`; no model or auditor was trained, and
Thayer-Audit v1 remains unauthorized.

## Signed-noise-residual preflight update

The Family-E1 training-free contract correction passed physical
representability on all 14,000 frozen episodes. It did not generate labels or
train an auditor. Thayer-Audit v1 remains unauthorized; only the separate
Family-E1 model-eligibility campaign is authorized next.

## Family-E1 eligibility update

Family-E1 did not supply a safe-support family. Its signed physical contract
and objective audit passed, but the mandatory mixed-eight prompt-identity gate
failed before full training and OOF output generation. Thayer-Audit v1 remains
unauthorized, and no auditor was trained in the Family-E1 campaign.
