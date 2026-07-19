# Thayer-PU Batch-R1 batch-invariance repair preregistration

Frozen UTC: `2026-07-14T22:43:23.165863000Z`, before loading the checkpoint,
executing any neural inference, creating corrected inference source, or inspecting
or computing any Thayer-PU safety label. Working experiment: `Thayer-PU-Batch-R1`.

## Authoritative inputs and scientific non-change contract

The authoritative failed campaign is
`outputs/runs/thayer_pu_eligibility_v1_20260714_213113`. Its scientific outcome
was `THAYER_PU_DEPLOYMENT_INELIGIBLE` solely because all 24 batch-1 candidate and
deployed tensors differed from batch 8; repeated batch 8 and batch 4 versus 8
were exact. The integrity addendum changed no scientific result.

- Checkpoint: `outputs/runs/thayer_probabilistic_unet_20260712_163340/checkpoints/thayer_pu_best.pth`, epoch 27, SHA-256 `c1d17a3f67962cce2fec03d6b15da5f2e330ee97b31c270a7ff019a1373a557e`.
- Constructor: `src.models_probabilistic_unet.ThayerProbabilisticUNet`, SHA-256 `b86de449ba0524c5675ea300e87ff753c4d18b974ca18e26fbae74a760ed8b1e`, latent dimension 8.
- Eligibility launcher currently referenced by the authoritative campaign: `scripts/run_thayer_pu_eligibility_v1.py`, SHA-256 `5997f61303b958b7b46f0c908bbdbfe45b863cc0be010da388b89646f7e0974a`. The authoritative run separately preserves preregistration-time and finalization-time hashes because diagnostic/finalization phases were appended after the scientific preflight.
- Flattened candidate sampling helper: `scripts/evaluate_probabilistic_unet_pre_atlas.py::sample_outputs`, whole-file SHA-256 `9f066fbd0eefc61a40c27fab0b101842993a64c8999df54064a0724233f4eea1`.
- Prompt constructor: `scripts/thayer_select_prompt_ablation_common.py::gaussian_prompt_numpy`, whole-file SHA-256 `449079faf20a29a1c65cd9c5916d1cffe641b4ef0ac5293ca9987cf2c3904fb7`; unit-peak Gaussian, sigma 2 pixels.
- Canonical hasher: `src/canonical_tensor_hash.py`, SHA-256 `65566c01c5e6a76bc35e638423562180f370edb7b5b8bc5a3931ae2ca994bb6e`; schema `thayer-per-sample-tensor-sha256-v1`.
- Frozen deployment JSON: `outputs/runs/thayer_pu_eligibility_v1_20260714_213113/deployment_rule/frozen_deployment_rule.json`, SHA-256 `0c484b8a33dcf2d4a32a44835a7898872705e030a31963294a2574c96df8cf6d`.
- Frozen latent manifest: `outputs/runs/thayer_pu_eligibility_v1_20260714_213113/deployment_rule/latent_seed_manifest.csv`, SHA-256 `03c2c319a790f8bc7b357315f91516fe6c2ff8843ee57110fd4e82933996ddbf`.
- Original preregistration: SHA-256 `6f5cd5de57e7810aab947c9c59e955bb09215abb5d32251dff663cbf753d578c`.
- Original 24-row preflight audit: SHA-256 `d57606cf444ea61cba63b660d015dc1b8bf902196d683e4f56c684e24c870a35`.
- Historical 743-checkpoint reference inventory: SHA-256 `83323c01abeecabc8d76d7bb25082c58ac3aeb2d8256db2cde932ff27c7beea3`.

The checkpoint bytes, architecture, latent dimension, prompt semantics, K=16,
latent seed rule, deployment aggregation and mapping, source-layer semantics,
scene manifests, source partitions, thresholds, output contract, safety labels,
and auditor architecture are immutable. No retraining, clipping, rounding,
truth-based selection, threshold relaxation, CPU fallback for authoritative
inference, development access, Atlas selection access, or lockbox access is
permitted.

## Exact frozen episodes and data hashes

The preflight set is eligibility positions 0 through 7 in each frozen source
manifest, in the exact order below.

| order | partition | position | scene ID | prompt subtype |
|---:|---|---:|---|---|
| 0 | training | 0 | `v2_r_training_00000` | `PERTURBED_VALID` |
| 1 | training | 1 | `v2_r_training_00003` | `UNIQUE_VALID` |
| 2 | training | 2 | `v2_r_training_00008` | `UNIQUE_VALID` |
| 3 | training | 3 | `v2_r_training_00011` | `UNIQUE_VALID` |
| 4 | training | 4 | `v2_r_training_00013` | `UNIQUE_VALID` |
| 5 | training | 5 | `v2_r_training_00015` | `UNIQUE_VALID` |
| 6 | training | 6 | `v2_r_training_00016` | `UNIQUE_VALID` |
| 7 | training | 7 | `v2_r_training_00023` | `UNIQUE_VALID` |
| 8 | validation | 0 | `v2_r_validation_00000` | `UNIQUE_VALID` |
| 9 | validation | 1 | `v2_r_validation_00003` | `UNIQUE_VALID` |
| 10 | validation | 2 | `v2_r_validation_00005` | `PERTURBED_VALID` |
| 11 | validation | 3 | `v2_r_validation_00007` | `PERTURBED_VALID` |
| 12 | validation | 4 | `v2_r_validation_00008` | `PERTURBED_VALID` |
| 13 | validation | 5 | `v2_r_validation_00011` | `UNIQUE_VALID` |
| 14 | validation | 6 | `v2_r_validation_00012` | `PERTURBED_VALID` |
| 15 | validation | 7 | `v2_r_validation_00013` | `UNIQUE_VALID` |
| 16 | calibration | 0 | `v2_natural_calibration_00000` | `PERTURBED_VALID` |
| 17 | calibration | 1 | `v2_natural_calibration_00001` | `UNIQUE_VALID` |
| 18 | calibration | 2 | `v2_natural_calibration_00002` | `UNIQUE_VALID` |
| 19 | calibration | 3 | `v2_natural_calibration_00003` | `UNIQUE_VALID` |
| 20 | calibration | 4 | `v2_natural_calibration_00005` | `UNIQUE_VALID` |
| 21 | calibration | 5 | `v2_natural_calibration_00011` | `PERTURBED_VALID` |
| 22 | calibration | 6 | `v2_natural_calibration_00012` | `PERTURBED_VALID` |
| 23 | calibration | 7 | `v2_natural_calibration_00013` | `UNIQUE_VALID` |

Frozen derived source-manifest SHA-256 values are training
`93cdb687d83f73acf7990a7cd04da8431f0562cd882c2edbf9840ecad9cbe789`,
validation `b92a5d272f0b6a7e125d55e25f9d214638e6109c1506af3081fd1bca481f4ab5`,
and calibration `212e144498c0e7af6438358e3776006fe10c3c250cf4c5be3470fce0dc4a39d9`.
Frozen upstream scene-HDF5 SHA-256 values are training
`a9efead2293b47afca61c1a156ac0fed9cdd4bc1c5920e197a581022c5fa0f22`,
validation `5a29100a96a1c01d657e91e68430809a68794fe647fee20012ca4d542933ab17`,
and calibration `99392093cc096b467bcee840e9af88f8600d620130422a536893a8f35a705b10`.
Any mismatch stops the campaign as `DATA_OR_IMPLEMENTATION_FAILURE`.

## Frozen seed, candidate, and deployment contract

Candidate indices 0 through 15 map one-to-one and in order to NumPy PCG64 seeds
2026077600 through 2026077615. For a partition of N episodes, each seed generates
`standard_normal((N,8)).astype(float32)`; the 16 arrays stack on axis 1 to
`(N,16,8)`. All epsilon tensors are generated once in canonical episode-major,
candidate-minor order outside the model and reused unchanged for every batch
condition. Candidate output is `(N,16,6,60,60)` in that same order. Requested
channels are 0:3. Deployment is the CPU float64 mean over candidate axis 1 and a
single contiguous float32 cast, with no candidate selection, clipping, or rounding.

## Reproduction and exact comparisons

MPS is authoritative; CPU float32 is diagnostic only. Run batch sizes 1, 1
repeated, 2, 4, 8, and 8 repeated. Compare B1/B1-repeat, B8/B8-repeat, B1/B2,
B1/B4, B1/B8, B2/B4, and B4/B8 for all 24 x 16 candidates and all 24 deployed
outputs. Compare tensors before hashes.

For every comparison record scene ID, prompt ID/subtype, candidate index, seed,
shape, dtype, device, min, max, mean, population standard deviation, exact
canonical hash, `array_equal`, maximum and mean absolute difference,
differing-element count and fraction, and the first C-order differing coordinate.
For finite same-dtype float32 values, ULP distance is computed by mapping IEEE-754
bit patterns to monotone signed integer order; report maximum ULP at differing
elements. Also report raw/canonical equality, raw/canonical hash equality,
contiguity, strides, byte order, and exact `.npy` serialization equality without
rounding or truncation.

Classify batch 1 as deterministic/systematic, stochastic, seed-misaligned,
candidate-order-misaligned, numerical-only, hash/serialization-only, or another
exact mechanism. Stop without repair if the discrepancy does not reproduce.

## Model-mode and mutable-state audit

Call `model.eval()` before every inference entry point and require every recursive
submodule's `training` flag to be false. Inventory qualified name, class,
training state, direct parameter/buffer names, buffer persistence and values,
normalization type, running-stat behavior, dropout probability, stochastic-depth
behavior, and any custom batch-dependent logic. Explicitly audit BatchNorm,
InstanceNorm, GroupNorm, LayerNorm, Dropout, stochastic depth, custom
normalization, and batch-dimension code. Hash every named parameter and buffer
before inference, after B1, and after B8. Any mutation blocks repair until its
exact cause is proven.

## Seed/order and shape/branch audits

Verify exact scene, prompt, candidate index, seed, epsilon tensor, epsilon hash,
candidate order, and deployment order across every batch condition. Compare the
historical path with the canonical precomputed-epsilon path. Trace shapes,
dtypes, devices, strides, and contiguity at inputs, prompt embeddings, prior
encoder blocks and statistics, latent reparameterization/injection, encoder and
decoder blocks, concatenations/skips, raw normalized output, physical candidate
output, and deployed output for B1/B2/B4/B8. Statically and dynamically audit
`squeeze`, `unsqueeze`, `view`, `reshape`, `flatten`, `[0]`, N==1 branches,
broadcasts, reductions over dimension 0, padding/crops, and noncontiguous inputs.

## Earliest-divergence audit

For the identical first frozen episode and candidate-0 epsilon, capture each major
block when run alone, first in a batch of four valid distinct episodes, repeated
four times, and paired with three different valid neighbors. Repeat on CPU
float32 and MPS float32. Record exact equality, maximum/mean absolute difference,
shape, dtype, strides, contiguity, and device at every boundary. Determine whether
neighbors, batch count alone, normalization, padding, reduction order, or MPS
kernel selection causes the earliest divergence.

## Regression-first and permitted repair

Before corrected source is created, add and execute the 17 named regression tests
specified in the campaign request, including reproduction, repeatability,
recursive eval, immutable buffers, latent/order invariance, absence of B1 squeeze
or special branches, B1-versus-B2/B4/B8 equality, prompt/deployment/checkpoint
preservation, no truth, and no rounding. Preserve the pre-fix result log.

Only a proven missing eval call, normalization-buffer mutation, active stochastic
module, batch-constructed seed/order defect, N=1 squeeze/indexing defect,
broadcast/shape defect, equal-value hash/serialization defect, or MPS
batch-kernel defect may be repaired. Each append-only candidate makes the
smallest correction and preserves prior candidates. An MPS fixed executor is
eligible only with one frozen batch size, explicitly marked dummy padding,
proof that dummy and real neighbors cannot affect real rows, dummy removal before
deployment, and exact rerun hashes. Batch 8 is not privileged by label outcomes.

Every candidate must rerun the new and inherited tests, promptability, seed/order,
state, B1/B2/B4/B8, CPU diagnostics, canonical hashes, and all 24 episodes. It
must emit `MODEL_STATE_IMMUTABLE`, `LATENT_SEQUENCE_IDENTICAL`,
`CANDIDATE_ORDER_IDENTICAL`, `PROMPT_IDENTITY_PRESERVED`,
`DEPLOYED_OUTPUT_REPLAY_EXACT`, and `BATCH_DEPLOYMENT_CONTRACT_PASS`, plus exactly
one applicable executor marker: `ARBITRARY_BATCH_INVARIANCE_PASS` or
`FIXED_BATCH_EXECUTOR_PASS`.

## Source freeze and eligibility continuation

The first fully passing candidate is authoritative. Freeze exact hashes for the
launcher, wrapper, seed generator, batching executor, deployment rule, prompt
constructor, hasher, tests, and validation evidence. Require unchanged checkpoint
and architecture, no clipping/rounding, and no truth. After freeze, no source
repair or retry is allowed.

Only after source freeze and a passing corrected 24-episode preflight may complete
outputs and Thayer-Audit safety labels be generated. Then rerun the exact original
source-group-safe training/validation/calibration manifests, full replay,
unchanged safety gates, label-support gates, Condition-C comparison, and family
eligibility. The original gates remain: training at least 500 safe and 500 unsafe;
validation/calibration at least 150 each; safe prevalence in [0.05,0.95]; at least
100 safe UNIQUE_VALID rows per partition; at least 50 safe PERTURBED_VALID rows
and 100 distinct safe source groups in validation and calibration. No POST auditor
is trained. Use original outcome precedence among
`THAYER_PU_ELIGIBLE_WITH_LABEL_SUPPORT`,
`THAYER_PU_ELIGIBLE_BUT_LABEL_COLLAPSED`,
`THAYER_PU_DEPLOYMENT_INELIGIBLE`, and `DATA_OR_IMPLEMENTATION_FAILURE`.

Development-scene, Atlas-selection, and lockbox access counters begin at zero and
must remain zero. Safety-label inspection is prohibited until repair source is
frozen and the corrected preflight passes.
