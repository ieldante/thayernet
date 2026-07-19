# Thayer-Family-E v0 preregistration

Frozen UTC: `2026-07-14T23:57:27Z`

Status: **FROZEN BEFORE MODEL CONSTRUCTION, TRAINING-TENSOR LOAD, OR FITTING**

## Scientific hypothesis and scope

The sole hypothesis is that one compact coordinate-conditioned reconstruction family whose requested, companion, and residual outputs are nonnegative and conserve the observed detected-electron tensor by construction can create a meaningful mixture of safe and unsafe outputs under the unchanged Thayer-Audit v0 gates.

This campaign builds only `Family-E`. It does not train an auditor, search architectures or hyperparameters, continue D3, tune thresholds, select on Atlas or development, or access the final lockbox.

## Authoritative contracts

- Scientific thresholds: `docs/d3_threshold_contract.md`, SHA-256 `ac6c4585d214008c03b19b6b61b69dee999242d02d7e1cb724caf5fffa7320e3`.
- Prompt constructor: `scripts/thayer_select_prompt_ablation_common.py`, SHA-256 `449079faf20a29a1c65cd9c5916d1cffe641b4ef0ac5293ca9987cf2c3904fb7`; unit-peak Gaussian, sigma 2 pixels.
- Source-group partition: `outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/source_split_manifest.csv`, SHA-256 `98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27`.
- Source-layer contract: `docs/physical_source_output_contract.md`, SHA-256 `3fbd8c019a0489106ec0be8efc1cbe0a152c36fd022e928673813c9bab74303f`.
- Safety implementation: `src/direct_catalog_safety_auditor.py`, SHA-256 `9efe750a60d746cd6cd496c6843a9a4f62500016ed599d1a1a84e4edac199df4`.
- Input normalization: `outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json`, SHA-256 `940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a`; g/r/z scales `611.9199829101562 / 1805.8800048828125 / 1854.199951171875`; clipping is false and negative observed pixels are preserved.
- Band order is g/r/z. Physical layers use finite detected electrons on the common 60 by 60 grid with zero-background semantics.

No additive background or sky offset is authorized. The physical allocation base `O` is exactly the raw observed `blend` tensor. No clipping, ReLU, absolute value, square, softplus, offset, truth rescaling, or other transformation of `O` is allowed.

## Exact source-safe manifests

The compact manifests are immutable row selectors into hashed upstream manifests; each `upstream_index` resolves the exact scene ID, prompt hash, source rows, source IDs, source groups, coordinates, and tensor path.

| Partition | Frozen selector | Rows | Selector SHA-256 | Upstream manifest SHA-256 | Upstream HDF5 SHA-256 |
|---|---|---:|---|---|---|
| training | `manifests/training_manifest.csv` | 10,000 | `4a8768eaa70e1d3f5f7a29fd4035e994c9c6f1494d3553e6ac0f805c8e911bc1` | `6c20d846709987c96c3d27c586756f1f48d75904a9e285ffd48d9b0a7b047ac3` | `a9efead2293b47afca61c1a156ac0fed9cdd4bc1c5920e197a581022c5fa0f22` |
| validation | `manifests/validation_manifest.csv` | 2,000 | `bc5c65ffab19baea38e37edcb4d5dabd15bae1c0266b7dfdaa749eba5c6c464d` | `acdb4071cb0c3b2eb67e9d9f26f0dd43f0ea76872efd137c2e067386cdf82413` | `5a29100a96a1c01d657e91e68430809a68794fe647fee20012ca4d542933ab17` |
| calibration | `manifests/calibration_manifest.csv` | 2,000 | `70326c1835726677e5d98c50323329f919bcd405f0f379420987fcd97e20fa0c` | `7fbfa02ce5d73ceefd4ce6478b5c6ea8b87de8745536ca4c6d4aff9ac348c74f` | `99392093cc096b467bcee840e9af88f8600d620130422a536893a8f35a705b10` |

Selection is deterministic: first 10,000 valid `v2_r_training` rows, all 2,000 valid `v2_r_validation` rows, and first 2,000 valid natural-calibration rows. Cross-partition source-group overlap, source-pair overlap, and duplicate pairs are all exactly zero.

Prompt support is frozen as:

- training: 8,515 `UNIQUE_VALID`, 1,485 `PERTURBED_VALID`;
- validation: 1,429 `UNIQUE_VALID`, 571 `PERTURBED_VALID`;
- calibration: 1,400 `UNIQUE_VALID`, 600 `PERTURBED_VALID`.

Invalid-query rows are excluded from reconstruction training and POST labels.

## OOF policy

Five connected-source-group folds are frozen in `manifests/training_manifest.csv`. Each fold has exactly 2,000 episodes and no source group appears in more than one fold. If full execution is reached, fold models use the exact architecture, objective, optimizer, scheduler, budget, and primary seed `2026071501`; each held fold is absent from that fold model's fit. Only these cross-fitted outputs are eligible as future training-auditor episodes. Full-fit training outputs are ineligible.

## Input and task

Input has exactly four channels:

1. normalized observed g;
2. normalized observed r;
3. normalized observed z;
4. unit-peak sigma-2-pixel Gaussian coordinate prompt.

The deployed output is an explicit requested-source reconstruction in physical detected electrons. Companion allocation is also explicit. Source IDs, truth-derived difficulty variables, clean targets, source groups, and family identity are not model inputs. Clean isolated targets are used only for supervised loss and evaluation.

## Physical output parameterization

Exactly one construction is frozen: per-pixel, per-band simplex source allocation.

The output head emits 9 logits, reshaped to `(3 bands, 3 allocations, 60, 60)`. Softmax is applied over allocation index in the exact order requested / companion / residual:

`a_req + a_comp + a_res = 1`.

For raw observed tensor `O`:

- `P_req = a_req * O`;
- `P_comp = a_comp * O`;
- `P_res = a_res * O`.

The same tensors feed loss, metrics, hashes, persistence, and deployment. There is no secondary evaluation mapping.

Frozen physical tolerances:

- negative fraction must be exactly 0;
- nonfinite fraction must be exactly 0;
- conservation maximum absolute error must be at most `1e-6 * max(1, max(abs(O)))` in float64 reference and `1e-5 * max(1, max(abs(O)))` in float32/MPS execution;
- target representability requires finite nonnegative requested and companion targets and `T_req + T_comp <= O + 1e-6 * max(1, max(abs(O)))` at every pixel and band;
- softmax zero/low-flux representability must reach absolute output error at most `1e-7 * max(1, max(abs(O)))` on the frozen synthetic fixture;
- no fixed positive allocation floor is allowed.

Preflight must verify analytic conservation, synthetic nonnegativity, finite gradients, low-flux and zero-source limits, correct g/r/z order, detected-electron units, and full frozen-target representability. Because the authoritative input contract preserves negative pixels and uses zero-background semantics, any negative `O` or any target-sum exceedance is a physical incompatibility and triggers the Part-E stop. No background offset may be added after observing this result.

## Fixed architecture

One U-Net-like network is frozen.

Encoder widths: `24, 48, 96, 128`.

- Stage 0: bias-free 3x3 convolution 4 to 24, GroupNorm(8), SiLU; then bias-free 3x3 convolution 24 to 24, GroupNorm(8), SiLU.
- Down 0: bias-free stride-2 3x3 convolution 24 to 48, GroupNorm(8), SiLU.
- Stage 1: two bias-free 3x3 convolutions 48 to 48, each GroupNorm(8), SiLU.
- Down 1: bias-free stride-2 3x3 convolution 48 to 96, GroupNorm(8), SiLU.
- Stage 2: two bias-free 3x3 convolutions 96 to 96, each GroupNorm(8), SiLU.
- Down 2: bias-free stride-2 3x3 convolution 96 to 128, GroupNorm(8), SiLU.
- Stage 3: two bias-free 3x3 convolutions 128 to 128, each GroupNorm(8), SiLU.
- Decoder 2: bilinear upsample to the stage-2 spatial shape; bias-free 3x3 convolution 128 to 96, GroupNorm(8), SiLU; concatenate the 96-channel skip; bias-free 3x3 convolutions 192 to 96 and 96 to 96, each GroupNorm(8), SiLU.
- Decoder 1: bilinear upsample; 3x3 convolution 96 to 48, GroupNorm(8), SiLU; concatenate skip; 3x3 convolutions 96 to 48 and 48 to 48, each GroupNorm(8), SiLU.
- Decoder 0: bilinear upsample; 3x3 convolution 48 to 24, GroupNorm(8), SiLU; concatenate skip; 3x3 convolutions 48 to 24 and 24 to 24, each GroupNorm(8), SiLU.
- Output head: biased 1x1 convolution 24 to 9 allocation logits.

All decoder convolutions except the final head are bias-free. Expected trainable parameter count is exactly `1,162,737`; construction must reproduce it and remain below 3,000,000. BatchNorm, attention, transformers, recurrent units, latent variables, sampling, experts, and alternate model conditions are prohibited.

## Objective

The exact physical outputs above feed the full supervised objective.

1. Requested-source physical-pixel L1 mean, weight `1.0`.
2. Companion-source physical-pixel L1 mean, weight `1.0`.
3. Per-source, per-band absolute relative flux error, denominator `abs(truth_flux) + 1e-6`, mean, weight `0.25`.
4. Per-source, per-band differentiable centroid Euclidean error divided by 60 pixels, with flux denominator `truth_or_prediction_flux + 1e-6`, mean, weight `0.10`.
5. Per-source g-r and r-z log-flux-ratio absolute error using `log(flux + 1e-6)`, mean, weight `0.10`.

No source-sum loss is used. No adversarial, perceptual, uncertainty, auditor, Atlas, D3 endpoint, safety-label, or post-hoc loss is used. Requested/companion ordering follows the matched prompt; no permutation matching is needed for this approved requested-source task.

Before fitting, exact truth must be zero or the numerical minimum for every component and full objective. Frozen compromise fixtures are: equal requested/companion split, requested/companion swap, and 0.8 requested-flux scaling with residual reallocation. Each must have objective no lower than exact truth. Any lower compromise triggers STOP.

## Optimization and checkpoint selection

Seeds: `2026071501, 2026071502, 2026071503`. The primary seed is `2026071501`; calibration safety prevalence cannot select a seed.

- MPS only; no fallback.
- AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Batch size `16`.
- Maximum `40` epochs.
- CosineAnnealingLR stepped once per completed epoch, `T_max=40`, `eta_min=0`.
- Validation patience `8`, monitored on requested-source reconstruction L1.
- Global gradient norm clipping `5.0`.
- Data order is seeded per seed and epoch.
- Checkpoint comparison is lexicographic: lowest validation requested-source L1, then lowest validation flux error, then lowest validation centroid error.
- No safety label, calibration result, development, Atlas, or lockbox information enters checkpoint selection.
- Every checkpoint is fresh and collision-refusing.

## Micro-overfit and sanity gates

Frozen training selector indices:

- ordinary scene: `16` (`v2_r_training_00016`);
- difficult scene: `6` (`v2_r_training_00006`, confusion-prone);
- mixed eight-scene set: `[0, 3, 5, 6, 18, 51, 73, 81]`, spanning natural, low-SNR, confusion-prone, high-overlap, equal-flux/similar-size, and both prompt subtypes.

Each ordinary one-scene and eight-scene fit must reduce full loss by at least 95%, visibly and numerically improve requested reconstruction, maintain zero negative/nonfinite fractions and conservation tolerance from step zero, pass prompt swap without source inversion, update both final and non-final decoder blocks, and remain entirely on MPS. Ordinary one-scene failure stops full training. Difficult-only failure is documented but does not stop if the other two pass.

## Full training, freeze, replay, and labels

If authorized by every earlier gate, train all three full-data seeds identically and persist component losses, gradient/update norms, extrema, negative/nonfinite fractions, conservation error, and prompt-swap metrics. Freeze one validation-selected checkpoint per seed and hash architecture, parameterization, inference code, fixed-batch executor, prompt construction, and output schema.

Generate:

- 10,000 primary-seed five-fold OOF training outputs;
- 2,000 primary full-fit validation outputs;
- 2,000 primary full-fit calibration outputs;
- other full-fit seeds only for stability analysis.

Deployment uses fixed batch size 16 with zero dummy rows for the last short chunk and strips dummy outputs. There is no batch-size-one branch. Replay at least 100 episodes per partition plus every physical edge case; candidate, prompt, physical-output, and source-sum hashes must match exactly. Any unresolved mismatch stops.

Apply the unchanged Thayer-Audit v0 label implementation and thresholds to unaltered Family-E outputs. Invalid queries, if retained outside these manifests, have label -1.

## Label-support and scientific gates

Training requires at least 500 safe and 500 unsafe episodes, prevalence in `[0.05,0.95]`, at least 100 distinct safe and unsafe source groups, and at least 100 safe `UNIQUE_VALID` episodes.

Validation and calibration each require at least 150 safe and 150 unsafe episodes, prevalence in `[0.05,0.95]`, at least 50 distinct safe and unsafe source groups, at least 100 safe `UNIQUE_VALID`, and at least 50 safe `PERTURBED_VALID`.

Scientific eligibility additionally requires 100% physical output-contract pass, at least 10% catastrophic pass in validation/calibration, at least 5% joint safe in validation/calibration, prompt-swap success at least 0.90, and no systematic source inversion.

## Family distinctness and bootstrap

Compare aligned non-development Family-E episodes with Condition C and Thayer-PU. Structural distinctness is fixed by allocation parameterization, exact conservation, nonnegative construction, and architecture scale. Behavioral distinctness passes if safety-label disagreement with either prior family is at least 0.10, reconstruction-error rank correlation is at most 0.90, or the gate-failure profile is materially different.

Use exactly 300 deterministic connected-source-group bootstrap replicates with seed `2026071599` for safe prevalence, catastrophic pass, flux pass, output-contract pass, joint safe, false subtraction, safe-source-group count, and family disagreement. No subgroup-conditional guarantee is authorized.

## Stop conditions and outcome precedence

Stop before model construction if physical target representability fails. Stop before training if objective alignment, architecture count, MPS, or physical synthetic preflight fails. Stop before full training if ordinary one-scene or eight-scene micro-overfit fails. Stop before labels on leakage, incomplete OOF provenance, or unresolved replay failure.

Assign exactly one outcome:

1. `FAMILY_E_ELIGIBLE_WITH_LABEL_SUPPORT`;
2. `FAMILY_E_PHYSICALLY_VALID_BUT_LABEL_COLLAPSED`;
3. `FAMILY_E_SCIENTIFIC_PARTIAL`;
4. `FAMILY_E_RECONSTRUCTION_FAILURE`;
5. `DATA_OR_IMPLEMENTATION_FAILURE`.

A pre-model physical/data-contract incompatibility maps to `DATA_OR_IMPLEMENTATION_FAILURE`. Physical validity with failed label support maps to the physically-valid collapsed category. Micro-overfit failure maps to reconstruction failure.

Only `FAMILY_E_ELIGIBLE_WITH_LABEL_SUPPORT` authorizes a separately preregistered Thayer-Audit v1. No auditor is trained here under any outcome.

## Privacy and integrity

Development scene access, Atlas selection access, and final-lockbox access must remain exactly zero. Condition C, Thayer-PU, all historical checkpoints, README, and the staged index remain unchanged. No stage, commit, push, merge, delete, move, rename, or historical overwrite is permitted.
