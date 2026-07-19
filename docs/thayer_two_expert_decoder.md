# Thayer-ME two-expert ambiguity decoder

Thayer-ME tested whether decoder parameter sharing caused Thayer-MH to settle
on low-loss compromise decompositions. The model retained one prompted
Condition-C-compatible encoder and replaced the shared token decoder with two
independently initialized, independently parameterized compact decoders.

The authoritative run is
`outputs/runs/thayer_two_expert_decoder_20260712_203121/`. Its preregistration
hash is `c5e0c4bb80ccf58346b9c5053d4ac607f7316d2e823aac629537f09742ed4c62`.
The 165,612-parameter model contains 72,672 shared encoder parameters and
46,470 parameters in each expert. Expert parameter storage is disjoint and the
initial parameter distance is nonzero. Only compatible encoder tensors were
loaded from Condition C.

The campaign reused the exact Thayer-MH scene tensors, manifests, and target
sets without regeneration. Thayer-MH promptability, reconstruction, coverage,
forward-consistency, and zero-Atlas-access results reproduced exactly before
implementation. All 2,000 pair gates, Atlas-source exclusions, partition
boundaries, and target hashes passed.

The isolated micro-overfit gate failed after the frozen 400-epoch MPS protocol.
Both expert and set prompt-swap rates passed, and all ambiguous observations
were forward-consistent, but ordinary, own, alternate, and both-mode truth
coverage remained zero. The ordinary median expert diameter was 5.166 frozen
scientific units, above the 1.0 concentration limit. This is classified as
`REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE`.

Full training, calibration, Atlas evaluation, auditing, and lockbox access were
therefore prohibited. The result does not invalidate the approved target sets
or prove source uniqueness. It shows that independent compact decoders trained
under the current normalized reconstruction and consistency objective still do
not satisfy the frozen scientific truth-coverage geometry, even on 64 isolated
training-only observations.

Multiple-choice and mixture-of-experts learning are established prior work.
The tested contribution is the prompted astronomical ambiguity-set contract,
full requested-plus-companion decomposition, protected source boundary, and
fail-closed truth-coverage gate.

## Frozen loss-geometry audit

The training-free Thayer-LG audit reproduced the persisted micro-overfit
failure and passed every exact-truth representability and coverage sanity
check. Exact truth was objective-optimal on only 15.625% of microset rows;
compromise beat truth on 84.375%, including all ambiguous rows. Detached
full-objective optimization from exact truth lowered loss while destroying
coverage. The failure is therefore not explained solely by network capacity or
decoder sharing; the frozen objective directly favors scientifically incorrect
directions.

## Scientific-alignment correction status

Thayer-SA kept this architecture and microset unchanged, removed forward and
source-sum terms from optimization, and aligned a differentiable scientific
surrogate with the frozen metric. Exact truth became a zero-loss stationary
point, but detached optimization from compromises did not reach the required
coverage. Neural retraining of Thayer-ME was therefore not authorized.

## Output-conditioning status

Thayer-OC did not train or modify Thayer-ME. Raw L-BFGS moved the persisted
expert outputs to 0.844 own, 0.875 alternate, and 0.750 both-mode ambiguous
coverage, but ordinary coverage remained 0.125 and the same method did not pass
other fixed starts. No conditioning method cleared all global gates. The model
remains an experimental failed micro-overfit artifact; no full training,
Atlas, auditor, development, production, or lockbox claim is authorized.

## Feasibility-projection micro result

Thayer-FP constructed completely feasible P0 target sets without changing
truths or thresholds, then trained this exact architecture directly against
them for 400 MPS-only epochs. Ordinary and ambiguous own/alternate/both-mode
coverage all remained zero, ordinary diameter ended at 3.564, and the final
output violated nonnegativity. Prompt swap and forward consistency remained
strong. Existing expert capacity, shared-encoder conditioning, or output
parameterization is now directly implicated; a controlled capacity ladder is
the only authorized next experiment.

## Capacity-ladder contract preflight

Thayer-CL reproduced Thayer-FP and traced the negative physical values to the
unconstrained linear decoder head. Because the historical identity head is
invalid under the new nonnegative contract and several distinct replacement
mappings are possible, no unique mapping was frozen. The campaign stopped
before decoder construction, so decoder capacity remains unresolved. A
separate output-parameterization campaign at unchanged L0 width must precede
any capacity ladder.
