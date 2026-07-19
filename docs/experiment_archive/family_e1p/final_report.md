# Thayer-Family-E1P-v0 scientific report

## Outcome

**FAIL — prompt identity remains below the frozen 0.90 gate. Family-E1 full training is not authorized.**

The difficult scene reached prompt identity `0.5000` and prompt-swap pair success `0.0000`. The mixed-eight set reached `0.5625` and `0.1250`. These exactly reproduce the prior Family-E1 paired micro results: all `28` compared scientific values match numerically with zero difference.

The required single failure classification is **Prompt too weak**, qualified precisely as: **the source-identity-aligned prompt component is too weak even though generic prompt modulation survives throughout the network**. This is not a reconstruction-capacity failure and not literal prompt disappearance.

## Critical protocol finding

The proposed intervention was already present in Family-E1. The prior runner duplicated each blend into A/B prompt views and swapped requested/companion targets, and its frozen preregistration explicitly specified that pairing. This campaign therefore changed no training tensor semantics; it is an instrumented deterministic replication. Paired examples alone do not repair Family-E1 identity.

## Required measurements

| Condition | Prompt identity | Prompt swap | Requested-source error | Companion leakage | Gate |
|---|---:|---:|---:|---:|---|
| difficult | 0.5000 | 0.0000 | 0.018574 | 0.463642 | FAIL |
| mixed eight | 0.5625 | 0.1250 | 0.013217 | 0.418998 | FAIL |

Prompt swap is the stricter scene-level event that both prompt views select their requested source. The difficult pair had exactly one correct view. In mixed eight, `7/8` scenes had exactly one correct view and only `1/8` had both correct; the sole full success was Family-E1 index `81`.

For each identical observation, prediction A/B differences were:

| Condition | L1 diff / truth / ratio | Flux diff / truth / ratio | Centroid px / truth / ratio | Color diff / truth / ratio | Prediction cosine / truth cosine |
|---|---|---|---|---|---|
| difficult | 0.002308 / 0.004524 / 0.510 | 7.373 / 7.753 / 0.951 | 0.871 / 1.249 / 0.698 | 0.319 / 0.323 / 0.987 | 0.959 / 0.881 |
| mixed eight | 0.027165 / 0.028348 / 0.958 | 31.172 / 31.245 / 0.998 | 9.530 / 10.644 / 0.895 | 0.540 / 0.551 / 0.980 | 0.540 / 0.225 |

The mixed aggregate L1 ratio `0.958` is not evidence of correct identity: its per-scene median is `0.748`, while the median cosine between the predicted A-B contrast and the true source A-B contrast is only `0.000316` and the median signed contrast gain is `0.000518`. Index 81 alone has contrast cosine `0.999487` and gain `1.011353`. Thus most prompt-driven output changes are nearly orthogonal to requested-source identity.

## Prompt influence trace

No encoder, decoder, head, or source output met the frozen numerical-indistinguishability diagnostic in either condition. Every per-layer gradient with respect to the prompt input was nonzero.

| Condition/layer | Feature modulation | Cross-correlation | MI proxy (nats) | Prompt-gradient RMS |
|---|---:|---:|---:|---:|
| difficult enc0_second | 0.449193 | 0.895242 | 0.091900 | 2.499e-04 |
| difficult enc3 | 0.043556 | 0.999001 | 0.000948 | 6.705e-07 |
| difficult requested output | 0.287467 | 0.957484 | 0.039694 | 1.416e-05 |
| mixed enc0_second | 0.543564 | 0.842666 | 0.129433 | 1.536e-05 |
| mixed enc3 | 0.783484 | 0.673380 | 0.239310 | 1.273e-05 |
| mixed requested output | 1.386050 | 0.531071 | 0.535254 | 4.669e-06 |

The difficult prompt is diluted internally: modulation falls `90.3%` from enc0_second to enc3 and the layer-energy prompt gradient attenuates by `372.7x`; decoder skips restore visible prompt dependence by dec0. That cannot be the campaign-wide primary cause because mixed-eight retains strong bottleneck modulation (`0.783`) and stronger requested-output modulation (`1.386`) yet still fails identity. The failure is semantic strength/alignment, not the first layer going numerically blind.

Final objective prompt-gradient RMS was `5.703e-04` difficult and `1.599e-04` mixed-eight.

## Why conditioning, not reconstruction capacity, is the remaining bottleneck

- Total-objective reduction was `0.994415` difficult and `0.999458` mixed; unchanged requested/companion reduction gates passed in both.
- The same architecture achieves both-view identity on index 81 and previously on the ordinary scene, demonstrating representational reach without changing capacity.
- Requested/companion outputs stayed nonnegative and finite, and the signed residual conserved the observation. Maximum closure error/tolerance was `0.000854492/0.187986` difficult and `0.00292969/0.425762` mixed.
- What fails is role assignment: companion leakage remains `0.464` / `0.419`, identity margins are near zero for seven mixed scenes, and their predicted prompt contrasts are not aligned with the true source contrasts.

Therefore reconstruction capacity is not the gate that remains. Family-E1 learns prompt-responsive features and scalar changes, but the identity-aligned component is too weak to override scene-specific source ordering/mixing.

## Classification and authorization

Primary classification: **Prompt too weak** (identity-aligned component). Prompt ignored, forgotten, overwritten, skip-dominated, and decoder-ignored are rejected by the nonzero layerwise modulation/gradients. Prompt dilution is a real secondary difficult-scene observation but cannot explain the mixed-eight failure.

The success clause is not activated. **Do not resume Family-E1 full training.** No full training, validation, calibration, OOF generation, safety labeling, auditor work, or alternate experiment was authorized or run.

## Integrity

- Exact architecture/parameter count: unchanged FamilyE1UNet, `1,162,662`; no new model modules or loss terms.
- MPS only; CPU fallback false; same optimizer, weights, thresholds, seeds, and update budgets.
- Focused tests: `..................                                                       [100%]`; compileall, CSV schemas, and git diff checks passed.
- README SHA-256 remains `67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1`.
- `767` historical checkpoints were rehashed with zero mismatches; the two new files are explicitly micro-only states inside this run.
- Git index empty; nothing staged or committed. Authoritative inputs unchanged. Validation/calibration/development/Atlas/lockbox access and OOF/label/auditor counts are all zero.

## Evidence inventory

- Frozen contract: `preregistration/family_e1p_paired_prompt_identity_intervention.md`.
- Aggregate and per-scene metrics: `tables/micro_overfit_results.csv`, `tables/per_scene_pair_metrics.csv`.
- Per-view role and contrast diagnosis: `tables/per_view_identity_diagnostics.csv`, `tables/common_contrast_diagnostics.csv`.
- Full layer trace and gradients: `tables/layerwise_prompt_trace.csv`, `tables/prompt_gradient_metrics.csv`.
- Failure taxonomy: `tables/failure_taxonomy.csv`, `diagnostics/failure_classification.json`.
- Exact prior replication: `tables/prior_family_e1_replication.csv`.
- Visual summary: `figures/prompt_conditioning_diagnosis.png`.
- Integrity: `tables/integrity_checks.csv` and checkpoint inventories.
