# Thayer-Family-E1P-v0 preregistration

Frozen UTC: `2026-07-15T02:52:28.073830Z`

Status: **FROZEN BEFORE MODEL CONSTRUCTION, OPTIMIZER CONSTRUCTION, TRAINING-TENSOR LOAD, OR FITTING**.

## Sole question

Can the unchanged Family-E1 architecture learn requested-source identity when every microset observation is presented exactly twice, once with the stored prompt and requested/companion targets and once with the companion-coordinate prompt and exchanged targets? This is a prompt-conditioning intervention only.

The prior Family-E1 runner already constructed exactly this paired tensor in `augmented_prompt_views` and its frozen preregistration explicitly required both prompt views. Therefore this campaign is a deterministic paired-prompt replication plus internal influence tracing, not a new optimization treatment. This fact must be part of the scientific disposition.

## Frozen model, map, objective, optimizer, and prompt

- Model source SHA-256: `b40a5467c3f16ce94ab0860e43a5d9937b8a86ef0c01afc0f4c4da844e8201e7`; exact expected trainable parameters: `1,162,662`.
- Input remains normalized observed g/r/z plus the unchanged unit-peak sigma-2 Gaussian coordinate prompt.
- Output remains six raw channels mapped in `forward` to nonnegative requested and companion sources by ReLU and the signed residual `O-requested-companion`.
- Objective remains requested L1 `1.0`, companion L1 `1.0`, relative flux `0.25`, centroid `0.10`, and color `0.10`; no paired, contrastive, ordering, safety, or other new loss is added.
- MPS AdamW remains learning rate `3e-3`, weight decay `1e-4`, gradient clip `5.0`, full augmented micro-batch, and no scheduler.
- Prompt encoding, band scales, target semantics, conservation rule, architecture, parameter count, and all scientific thresholds remain unchanged.

## Frozen microsets and budgets

Only training-selector difficult index `6` for `2,000` updates with seed `2026071512`, and mixed indices `[0,3,5,6,18,51,73,81]` for `3,000` updates with seed `2026071513`. The ordinary scene, full training, validation, calibration, OOF, safety labels, auditor, development, Atlas selection, and lockbox are prohibited.

Each selected blend appears in ordered views `[A scenes, B scenes]`. The observed three-channel tensor must be byte-identical across a pair. Only the prompt channel changes. Requested and companion targets exchange exactly.

## Frozen measures

- Prompt identity is the unchanged strict rate at which normalized requested-prediction MSE is smaller to its requested target than to its companion target. Gate: `>=0.90` separately for difficult and mixed-eight.
- Prompt swap is the scene rate for which both A and B views pass identity.
- Requested-source error is mean band-normalized pixel L1.
- Companion leakage is the mean nonnegative companion coefficient fraction from a two-template least-squares decomposition of the requested prediction against requested and companion truths.
- Same-observation A/B comparisons are normalized-pixel L1 difference, normalized integrated-flux difference, soft-centroid distance in pixels, log-color difference, and cosine similarity. L1/flux/centroid/color response ratios use the corresponding true A/B difference as denominator.
- Encoder and decoder tracing reports paired prompt-activation RMS, feature modulation (prompt RMS / mean feature RMS), centered cross-correlation, pair cosine, and `0.5*log1p(prompt-difference power / feature power)` as a Gaussian-channel mutual-information proxy. The gradient measure is RMS of the gradient of each layer's mean-square activation with respect to the prompt input.
- Numerical indistinguishability is diagnostic only, not a new scientific gate: feature modulation `<=1e-6` and cross-correlation `>=0.999999`.

## Outcome

Success requires prompt identity `>=0.90` on both frozen microsets while the unchanged physical contract holds. Success authorizes exactly one experiment: resume Family-E1 full training with no change to architecture, parameter count, physical contract, signed residual, or objective. Failure authorizes no full training and must receive exactly one quantitative cause label from: Prompt ignored, Prompt diluted, Prompt forgotten, Prompt overwritten, Prompt too weak, Skip connections dominate, Decoder ignores prompt, or Other.

All campaign artifacts are collision-refusing and append-only. README, historical checkpoints, git index, and all prior run files remain unchanged; nothing is staged or committed.
