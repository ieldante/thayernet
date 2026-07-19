# Thayer-Family-E1-v0 preregistration

Frozen UTC: `2026-07-15T01:47:17.426854Z`

Status: **FROZEN BEFORE MODEL CONSTRUCTION, MODEL-SOURCE IMPORT, TRAINING-TENSOR LOAD, OR FITTING**

## Hypothesis and boundaries

The sole hypothesis is that one compact coordinate-conditioned reconstruction model with nonnegative requested and companion source outputs and a derived signed residual can create meaningful safe and unsafe support under the unchanged Thayer-Audit v0 gates. This campaign trains exactly one architecture family. It does not train a POST auditor, tune from safety labels, access development, use Atlas for selection, or access the final lockbox.

The authoritative signed-residual preflight outcome must remain exactly `SIGNED_NOISE_RESIDUAL_CONTRACT_PASS`. Requested and companion are catalog-source layers; the signed residual is observational noise/background closure and is not a catalog source.

## Frozen data

Use the exact Family-E selectors: training 10,000 (`4a8768eaa70e1d3f5f7a29fd4035e994c9c6f1494d3553e6ac0f805c8e911bc1`), validation 2,000 (`bc5c65ffab19baea38e37edcb4d5dabd15bae1c0266b7dfdaa749eba5c6c464d`), and calibration 2,000 (`70326c1835726677e5d98c50323329f919bcd405f0f379420987fcd97e20fa0c`). The upstream manifests and HDF5 tensors retain their authoritative hashes. Cross-partition source-group overlap, source-pair overlap, and duplicate source pairs are zero. Prompt support is training 8,515 `UNIQUE_VALID` plus 1,485 `PERTURBED_VALID`, validation 1,429 plus 571, and calibration 1,400 plus 600. No source/group ID or target-derived difficulty variable is an input.

Five connected-source-group folds from the immutable training selector are used. Each contains 2,000 episodes; no source group crosses folds. Fold models exclude both source groups of every held episode and use identical settings. OOF fold seed is `2026071501`.

## Inputs and normalization

Input is exactly normalized observed g/r/z followed by the stored unit-peak sigma-2-pixel Gaussian coordinate prompt: four channels. Frozen positive g/r/z scales are `611.9199829101562, 1805.8800048828125, 1854.199951171875`. Observations and targets are never clipped or offset. Targets are used only for supervised fitting and evaluation.

## Sole architecture

One compact coordinate U-Net is frozen. Encoder widths are `24, 48, 96, 128`. Each stage has two bias-free 3x3 convolutions, GroupNorm(8), and SiLU. Between stages a bias-free stride-2 3x3 convolution changes width and is followed by GroupNorm(8) and SiLU. The mirrored decoder bilinearly upsamples to each skip shape, applies a bias-free 3x3 width-changing convolution plus GroupNorm(8)/SiLU, concatenates the skip, then applies two bias-free 3x3 convolutions with GroupNorm(8)/SiLU. The biased 1x1 head maps 24 channels to six normalized logits ordered requested g/r/z then companion g/r/z. Head bias initializes to `0.01`; other convolutions use Kaiming-normal initialization, GroupNorm scale one/bias zero. BatchNorm, attention, transformers, recurrence, latent variables, stochastic sampling, experts, and variants are prohibited. The hard ceiling is 3,000,000 trainable parameters; expected exact count is 1,162,662.

## Sole physical mapping

Inside `forward`, and nowhere afterward:

- `P_req = S * ReLU(R_req)`;
- `P_comp = S * ReLU(R_comp)`;
- `P_noise = O - P_req - P_comp`.

The same mapped source tensors enter loss, metrics, persistence, hashing, labels, and deployment. There is no post-forward clipping, positive floor, observation transformation, truth-based rescaling, softplus, square, absolute-value, or simplex alternative. Float32 conservation tolerance is `1e-5 * max(1, max(abs(O)), max(abs(P_req)), max(abs(P_comp)), max(abs(P_noise)))`; float64 reference tolerance is the same scale times `1e-10`.

## Frozen objective

All sources are divided by the positive band scales for objective evaluation. The total is the nonnegative weighted sum:

1. requested normalized-pixel L1 mean, weight `1.0`;
2. companion normalized-pixel L1 mean, weight `1.0`;
3. mean per-source/per-band absolute relative flux error with denominator `abs(truth flux)+1e-6`, weight `0.25`;
4. mean per-source/per-band soft-centroid Euclidean error divided by 60, with prediction and truth centroids each using its own `flux+1e-6` denominator, weight `0.10`;
5. mean per-source absolute g/r and r/z log-flux-ratio error with `log(flux+1e-6)`, weight `0.10`.

No residual-target, source-sum, adversarial, perceptual, uncertainty, auditor, safety-label, D1/D3 endpoint, or Atlas loss is used. Exact truth must be a stationary global minimum. Equal allocation, swapping, 0.8 requested scaling, and averaged source fixtures must not beat truth. Gradients must be finite and nonzero away from optimum; ReLU inactive-gradient fractions are measured; exact zero target pixels remain representable.

## Objective and micro gates

Objective alignment runs on synthetic fixtures and the frozen ordinary/difficult/mixed crops before fitting. Any compromise below truth stops the campaign.

Micro fitting uses MPS AdamW with learning rate `3e-3`, weight decay `1e-4`, batch equal to the augmented microset, gradient clip `5.0`, and no scheduler. Selectors are ordinary index 16, difficult index 6, and mixed indices `[0,3,5,6,18,51,73,81]`. Each scene is trained with its stored requested prompt and a deterministically reconstructed companion-coordinate sigma-2 prompt with swapped targets. Budgets are 1,500 / 2,000 / 3,000 updates. Pass requires at least 95% total-objective reduction, at least 80% reduction in each requested and companion L1, requested identity closer than inversion on at least 90% of prompt views, exact zero source-negative/nonfinite fractions, conservation within tolerance, and updates to the head and first decoder up-convolution. Ordinary and mixed-eight must pass; difficult-only failure is reported but is not a stop.

## Full optimization and checkpoint selection

Seeds are `2026071501, 2026071502, 2026071503`; `2026071501` is the preregistered primary seed. MPS only. AdamW learning rate `3e-4`, weight decay `1e-4`; batch 16; maximum 40 epochs; patience 8; gradient clip 5.0; CosineAnnealingLR with `T_max=40, eta_min=0`, stepped after each epoch. Training order is seeded by seed and epoch.

Checkpoint selection is lexicographic and validation-only: lowest requested-source validation L1, then companion-source validation L1, then validation flux surrogate, then validation centroid surrogate. Safety labels, safe prevalence, calibration outcomes, Atlas, development, and lockbox never select a checkpoint or primary seed. Persist every requested loss/diagnostic per epoch.

## Freeze, OOF, deployment, and replay

Freeze one selected checkpoint per seed. Freeze architecture, physical mapping, inference source, prompt constructor, fixed-batch executor, normalization, and canonical hash implementation. Training labels use only genuine five-fold OOF outputs. Validation and calibration use the frozen full-fit primary checkpoint.

Deployment uses fixed neural batch size 16. Short chunks receive explicit zero dummy rows which are discarded. Canonical hashes use `thayer-per-sample-tensor-sha256-v1`, CHW little-endian contiguous float32 with its versioned header. Replay at least 100 OOF training, 100 validation, 100 calibration episodes and every physical edge fixture. Exact tensor/hash replay, batch consistency, prompt/source order, shapes/dtypes, zero source negatives, and conservation are mandatory.

## Unchanged labels and support gates

After output freeze, apply `src/direct_catalog_safety_auditor.py` at hash `9efe750a60d746cd6cd496c6843a9a4f62500016ed599d1a1a84e4edac199df4` with unchanged image, flux, color, centroid, confusion, catastrophic, false-subtraction, worse-than-baseline, and source-output rules. Requested and companion source layers must both be finite/nonnegative; the signed residual is excluded from catalog-source nonnegativity. Invalid queries retain label `-1` (none are selected here).

Training gates: safe >=500, unsafe >=500, safe prevalence in [0.05,0.95], >=100 distinct safe and unsafe source groups, and >=100 safe `UNIQUE_VALID`. Validation/calibration gates: safe and unsafe each >=150, prevalence in [0.05,0.95], >=50 distinct safe and unsafe groups, >=100 safe `UNIQUE_VALID`, and >=50 safe `PERTURBED_VALID`. Scientific gates additionally require 100% source-output contract pass, >=10% catastrophic pass in validation/calibration, >=5% joint safe, prompt-swap >=0.90, and no systematic inversion.

## Family comparison and bootstrap

Compare aligned non-development Family-E1 outputs with frozen Condition C and repaired Thayer-PU. Distinctness requires safety disagreement >=0.10, reconstruction-error Spearman <=0.90, or a materially different failure profile. Family identity is never a future auditor input.

Run exactly 300 deterministic connected-source-group bootstrap replicates with seed `2026071599` for safe prevalence, catastrophic-pass, flux-pass, output-contract-pass, joint-safe, false-subtraction, safe-source-group count, and family disagreement. Do not claim subgroup-conditional guarantees.

## Outcomes and authorization

Assign exactly one: `FAMILY_E1_ELIGIBLE_WITH_LABEL_SUPPORT`, `FAMILY_E1_PHYSICALLY_VALID_BUT_LABEL_COLLAPSED`, `FAMILY_E1_SCIENTIFIC_PARTIAL`, `FAMILY_E1_RECONSTRUCTION_FAILURE`, or `DATA_OR_IMPLEMENTATION_FAILURE`. Only the eligible outcome authorizes one separately preregistered `Thayer-Audit v1 — Multi-Family POST Auditor`. No auditor is trained here. Every other outcome receives exactly one next experiment.

## Integrity

One architecture, one mapping, no safety selection, no clipping, no truth deployment, genuine OOF, leak-free source groups, and zero development/Atlas-selection/lockbox access are mandatory. The preflight, Condition C, Thayer-PU, historical checkpoints, README, and staged index remain unchanged. Nothing is staged, committed, pushed, merged, deleted, moved, renamed, or historically overwritten.
