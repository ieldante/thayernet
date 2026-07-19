# Family-E1 signed-noise-residual physical-contract preflight final report

## Outcome

**SIGNED_NOISE_RESIDUAL_CONTRACT_PASS**.

Preregistration SHA-256:
`be546f7f1aa2ec04f1a76f84bc5305c87521d5b89331c681dc3cdf18a5293d3b`.

The corrected physical output space represented every frozen requested and
companion target in all 14,000 episodes while preserving nonnegative physical
source layers and exact signed-observation closure within the preregistered
float32 and float64 tolerances.

This is a physical representability result, not a trained reconstruction or
catalog-safety result.

## Frozen construction

The six normalized future head coordinates are ordered requested g/r/z then
companion g/r/z. With positive frozen scale `S` and signed zero-background
observation `O`:

- `P_req = S * ReLU(L_req)`;
- `P_comp = S * ReLU(L_comp)`;
- `P_noise = O - P_req - P_comp`.

Requested and companion are the only astronomical source layers. They are
nonnegative by construction. `P_noise` is a signed observational
noise/background-closure layer, not a source.

ReLU is the sole in-forward physical mapping. It is not evaluation-only
clipping or a post-hoc repair. No sky offset, observed clipping, absolute
value, square mapping, or truth-based deployment rescaling was used.

## Synthetic MPS preflight

All gates passed on MPS without CPU fallback:

- requested/companion negative fraction: exactly zero;
- signed residual: both positive and negative values represented;
- outputs and gradients: finite;
- zero and negative logits: exact zero source output;
- low positive flux: represented;
- conservation error: `0.0`, tolerance `0.0110452001953125`;
- changing `O` with fixed logits changed only the residual;
- changing requested logits left companion output exact;
- channel order and g/r/z scale semantics: correct.

## Full frozen-target representability

Truth was used only to create the offline inverse-coordinate witness
`L_star = T / S`. It is not an inference rule.

| Partition | Episodes | Requested max error | Companion max error | Source tolerance | Mapped negative count | Float32 closure error / tolerance | Float64 closure error / tolerance |
|---|---:|---:|---:|---:|---:|---:|---:|
| training | 10,000 | 0.015625 | 0.0078125 | 0.28313296875 | 0 | 0.015625 / 2.8313296875 | 3.637978807091713e-12 / 2.8313296875e-05 |
| validation | 2,000 | 0.03125 | 0.125 | 2.0357505 | 0 | 0.015625 / 20.3608425 | 3.637978807091713e-12 / 0.000203608425 |
| calibration | 2,000 | 0.5 | 0.03125 | 12.83203 | 0 | 0.0078125 / 128.32092 | 7.275957614183426e-12 / 0.0012832092 |

All observed and source tensors were finite. All source targets were
nonnegative. Every mapped requested and companion value was nonnegative.

## Signed residual diagnostics

| Partition | Minimum | Maximum | Mean | RMS | Negative fraction | Positive fraction | g/r/z RMS |
|---|---:|---:|---:|---:|---:|---:|---|
| training | -2599.130371 | 2805.069336 | -0.048349 | 366.112369 | 0.500543 | 0.499457 | 155.357193 / 356.497777 / 500.887489 |
| validation | -2745.992432 | 2489.622314 | 0.066563 | 366.001331 | 0.500289 | 0.499711 | 155.401198 / 356.521468 / 500.613458 |
| calibration | -8789.496094 | 5671.662109 | -0.054062 | 366.097474 | 0.500334 | 0.499666 | 155.387662 / 356.528951 / 500.823186 |

The roughly balanced signs confirm that the closure layer captures the signed
noise semantics that made the original all-nonnegative simplex impossible.

## Scientific interpretation

Family-E v0 correctly failed because it required requested, companion, and
residual all to be nonnegative while summing to signed observations. This
preflight changes exactly one contract element: the residual/noise layer may
be signed. The prior failure is preserved and not reinterpreted.

The corrected output space now has all required physical target support.
However, this preflight does not establish:

- learnability by the frozen coordinate-conditioned architecture;
- objective alignment;
- one-scene or eight-scene micro-overfit;
- full three-seed training;
- promptability;
- source-group-safe OOF output generation;
- deterministic replay or batch consistency;
- unchanged catalog-safety performance;
- safe/unsafe label support;
- family distinctness;
- bootstrap intervals;
- auditor viability.

No model, optimizer, checkpoint, reconstruction, safety label, family
comparison, bootstrap, or auditor was created.

## Authorization

Exactly one separately preregistered campaign is authorized next:

**Thayer-Family-E1-v0 — Nonnegative-Source Signed-Residual Model Eligibility**.

That campaign must preserve this exact physical map and independently pass
objective alignment, architecture-count, MPS, micro-overfit, full training,
OOF provenance, replay, unchanged safety gates, label-support, family
distinctness, and bootstrap gates.

Thayer-Audit v1 remains unauthorized until actual source-group-safe Family-E1
outputs supply adequate safe and unsafe labels.

## Integrity

- Preregistration preceded every new source-tensor load in this campaign.
- Compileall: PASS.
- Focused tests: 27 passed, 0 failed.
- Gate table: 23/23 PASS.
- CSV/schema validation: 2 files, 0 failures.
- Historical checkpoints: 743 present, 0 missing, 0 mismatched.
- Family-E v0 final report: unchanged.
- Condition-C checkpoint: unchanged at `e9176dc5…e382`.
- Thayer-PU checkpoint: unchanged at `c1d17a3f…557e`.
- Threshold, prompt, and safety implementation hashes: unchanged.
- README: unchanged at `67f66f35…0116a1`.
- Development / Atlas-selection / lockbox access: `0 / 0 / 0`.
- Staged index: empty.
- `git diff --check`: PASS.
- Branch / Git HEAD: `thayer-select` /
  `74b8ff7efbbf7e9891cc8fd8095a9931e3b63174`.
- Runtime through report assembly: approximately `518` seconds.
- Run tree at report assembly: 17 files, 69,849 bytes.
- Final Git status: `diagnostics/final_git_status.txt`, SHA-256
  `a9936ea51468be7516d3f4fbe552b2495009148134e13a2ca78a135969cb62eb`.

## Reusable and generated artifacts

Reusable code/tests for later review:

- `src/family_e_signed_residual.py`;
- `scripts/audit_family_e1_signed_noise_residual_preflight.py`;
- `tests/test_family_e_signed_residual.py`;
- `tests/test_family_e1_signed_residual_preflight_artifacts.py`.

The entire
`outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340/`
tree should remain ignored. Nothing was staged, committed, pushed, merged,
deleted, moved, or overwritten.
