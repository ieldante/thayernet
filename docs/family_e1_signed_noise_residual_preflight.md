# Family-E1 signed-noise-residual preflight

The authoritative run is
`outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340/`.
Preregistration SHA-256:
`be546f7f1aa2ec04f1a76f84bc5305c87521d5b89331c681dc3cdf18a5293d3b`.

Outcome: **SIGNED_NOISE_RESIDUAL_CONTRACT_PASS**.

The sole frozen construction maps requested and companion g/r/z logits through
in-forward ReLU and the positive training-only band scales. It defines the
signed closure residual as
`P_noise = O - P_req - P_comp`. Requested and companion layers remain finite
and nonnegative; the residual is not an astronomical source layer and may
carry either sign.

All 10,000 training, 2,000 validation, and 2,000 calibration target pairs were
representable. Mapped-source negative count was zero. Maximum requested /
companion round-trip errors were:

- training: `0.015625 / 0.0078125`, tolerance `0.28313296875`;
- validation: `0.03125 / 0.125`, tolerance `2.0357505`;
- calibration: `0.5 / 0.03125`, tolerance `12.83203`.

Float32 conservation errors were at most `0.015625`; float64 errors were at
most `7.275957614183426e-12`. Residual values were finite and approximately
half negative and half positive in every partition. Synthetic MPS tests passed
nonnegativity, gradients, zero/low-flux representation, isolation, and closure
without CPU fallback.

This is a physical representability result only. No model, optimizer,
checkpoint, reconstruction, safety label, family comparison, bootstrap, or
auditor was produced. Thayer-Audit v1 remains unauthorized.

Exactly one next campaign is authorized:
**Thayer-Family-E1-v0 — Nonnegative-Source Signed-Residual Model Eligibility**.
It must be separately preregistered and still pass objective alignment,
micro-overfit, full training, source-group OOF generation, replay, unchanged
safety gates, label support, family distinctness, and bootstrap.

## Model-campaign disposition — 2026-07-14

The authorized Family-E1 v0 campaign preserved this physical contract and
passed objective alignment and ordinary one-scene micro-overfit. It stopped at
the mandatory mixed-eight prompt-identity gate (`0.5625 < 0.90`). This does not
change the preflight result: nonnegative source layers plus a signed residual
remain physically valid. It means no eligible trained Family-E1 checkpoint,
OOF output, or safety label was produced.
