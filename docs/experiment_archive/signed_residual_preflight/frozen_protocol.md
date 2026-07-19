# Thayer Family-E1 signed-noise-residual physical-contract preflight v0

Frozen UTC: `2026-07-15T00:24:57Z`

Status: **FROZEN BEFORE NEW SOURCE-TENSOR LOAD OR PHYSICAL EVALUATION**

## Purpose

This is the single training-free contract correction recommended by
Thayer-Audit Family-E v0. It tests whether nonnegative requested and companion
source layers can coexist with exact conservation of the signed
zero-background observation when the residual/noise layer is explicitly
allowed to be signed.

This is not neural training, architecture construction, optimizer
construction, reconstruction-quality evaluation, safety labeling, threshold
tuning, an auditor campaign, Atlas selection, development evaluation, or
lockbox access.

## Frozen evidence and sample

The authoritative antecedent is
`outputs/runs/thayer_family_e_v0_20260714_195256/reports/final_report.md`,
SHA-256
`1ad5495a729fe898359439590a5ace8a196697c70fa9514172ce02094985114c`.

Reuse exactly the prior compact selectors:

- training: 10,000 rows, selector SHA-256
  `4a8768eaa70e1d3f5f7a29fd4035e994c9c6f1494d3553e6ac0f805c8e911bc1`;
- validation: 2,000 rows, selector SHA-256
  `bc5c65ffab19baea38e37edcb4d5dabd15bae1c0266b7dfdaa749eba5c6c464d`;
- calibration: 2,000 rows, selector SHA-256
  `70326c1835726677e5d98c50323329f919bcd405f0f379420987fcd97e20fa0c`.

No row, source group, prompt subtype, or partition may be changed. No invalid
query is introduced. Prior checks established zero source-group/source-pair
overlap and five connected-source-group training folds with zero cross-fold
group overlap.

## Source and observation semantics

- Band order: g/r/z.
- Grid: 60 by 60.
- Requested and companion outputs: finite nonnegative detected-electron source
  contributions with zero-background semantics.
- Observation `O`: the unaltered signed zero-background detected-electron
  blend.
- Residual `P_noise`: a signed noise/background-closure layer, explicitly
  not a physical astronomical source layer.
- Frozen positive g/r/z normalization scales:
  `[611.9199829101562, 1805.8800048828125, 1854.199951171875]`.

No sky offset, observed clipping, absolute value, square mapping, truth-based
deployment rescaling, or post-hoc output repair is allowed.

## Sole physical construction

The future head coordinate convention is six normalized channels in exact
order requested g/r/z then companion g/r/z. Let these raw normalized values be
`L_req` and `L_comp`; let positive band scale be `S`.

The sole in-forward source mapping is:

- `P_req = S * ReLU(L_req)`;
- `P_comp = S * ReLU(L_comp)`;
- `P_noise = O - P_req - P_comp`.

The same `P_req` and `P_comp` must be consumed by any future loss, metric,
hash, persistence, safety label, and deployment path. ReLU is therefore the
physical forward parameterization, not evaluation-only or post-hoc clipping.

By construction:

- `P_req >= 0`;
- `P_comp >= 0`;
- `P_req + P_comp + P_noise = O` up to frozen floating-point tolerance;
- `P_noise` may have either sign;
- zero source is represented exactly by zero or negative normalized logits;
- there is no forced positive source floor.

At inference the construction receives only model logits, frozen scales, and
the observed blend. Clean isolated targets do not enter inference.

## Target-representability witness

Truth is permitted only for this offline representability proof. For each
nonnegative target source layer define:

- `L_req_star = T_req / S`;
- `L_comp_star = T_comp / S`.

Apply the exact deployed ReLU-and-scale mapping and define
`P_noise_star = O - P_req_star - P_comp_star`.

This witness passes only if every target is finite/nonnegative and the mapped
requested/companion sources recover their targets within the frozen float32
physical tolerance. The witness is not an inference rule and may not be
persisted as a model output or used for model selection.

## Frozen numerical tolerances

For each partition define:

- source round-trip tolerance:
  `1e-6 * max(1, max(abs(T_req)), max(abs(T_comp)))`;
- float32 conservation tolerance:
  `1e-5 * max(1, max(abs(O)), max(abs(P_req)), max(abs(P_comp)), max(abs(P_noise)))`;
- float64 reference conservation tolerance:
  `1e-10 * max(1, max(abs(O)), max(abs(P_req)), max(abs(P_comp)), max(abs(P_noise)))`.

No tolerance may be changed after tensor inspection.

## Synthetic MPS gates

On MPS, require:

1. requested and companion negative fractions exactly zero;
2. finite requested, companion, residual, and gradients;
3. float32 conservation within the frozen rule;
4. exact zero-source output for zero and negative logits;
5. low positive flux representable;
6. both positive and negative signed residual values representable;
7. changing the observation with fixed logits changes only residual;
8. changing requested logits with fixed observation changes requested and the
   closure residual but not companion;
9. correct g/r/z scale and six-channel ordering;
10. no CPU neural fallback.

## Full frozen-target gates

Audit all 10,000/2,000/2,000 episodes sequentially on CPU. Require in every
partition:

1. finite observed and isolated tensors;
2. requested and companion targets nonnegative;
3. mapped source round-trip maximum error within tolerance;
4. mapped requested/companion negative fraction exactly zero;
5. signed residual finite;
6. both signs actually present in the residual population;
7. float32 and float64 conservation within their frozen tolerances;
8. exact selector/prompt/source provenance retained;
9. no target value used except to construct the offline inverse witness and
   evaluate representability.

Report residual minimum, maximum, mean, RMS, negative/positive/zero fractions,
per-band RMS, source round-trip errors, and conservation errors. These are
contract diagnostics, not reconstruction performance.

## Outcome categories

Assign exactly one:

1. `SIGNED_NOISE_RESIDUAL_CONTRACT_PASS` — all provenance, synthetic MPS,
   target representability, source nonnegativity, signed-residual, and
   conservation gates pass.
2. `SIGNED_NOISE_RESIDUAL_CONTRACT_FAIL` — the construction is internally
   valid but any frozen target or numerical gate fails.
3. `SIGNED_NOISE_RESIDUAL_DATA_OR_IMPLEMENTATION_FAILURE` — provenance,
   schema, device, or audit execution cannot be validated.

Only `SIGNED_NOISE_RESIDUAL_CONTRACT_PASS` authorizes one separately
preregistered next campaign:
`Thayer-Family-E1-v0 — Nonnegative-Source Signed-Residual Model Eligibility`.
That future campaign must still pass objective alignment, architecture count,
micro-overfit, full training, OOF generation, replay, unchanged safety gates,
label support, family distinctness, and bootstrap. This preflight alone cannot
authorize Thayer-Audit v1 or claim scientific reconstruction success.

## Integrity

No model or optimizer may be constructed. No checkpoint or reconstruction may
be written. Condition C, Thayer-PU, Family-E v0, thresholds, prompt semantics,
source-layer semantics, historical checkpoints, and README remain unchanged.
Development, Atlas selection, final lockbox, and auditor training access counts
remain zero. Nothing may be staged or committed.
