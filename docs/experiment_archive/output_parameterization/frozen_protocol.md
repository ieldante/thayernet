# Fixed-L0 Output-Parameterization preregistration

Working name: **Thayer-OP (Thayer Output Parameterization)**  
Frozen at UTC: `2026-07-13T06:31:20.195028+00:00`  
Microset manifest SHA-256: `9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085`  
Frozen P0 target-file SHA-256: `d58ef71e988de8584a78865f00747b931c1e65f6e406e437cebdca60a049b181`  
Frozen P0 canonical-hash table SHA-256: `b45e26f95f7ecf7fc117f3b4660901224060fd2a9ef9eecd1a408ba5e693a65b`  
Frozen row-selection table SHA-256: `97d11b2ef3b2f32c8f979ce4bf5b0a1505c1048575240c1a567295e61c0bc979`  
Output-mapping implementation SHA-256: `a47c322ffa3fda58a84a45c0a15891f60cef2455215ec99a229c6200f8edf1ae`

## Scope and protected boundary

This is a fixed-L0, training-only output-parameterization campaign. It compares exactly ReLU, square, and absolute value inside the model forward path. It is not a decoder-capacity ladder, 64-row fit, full-data campaign, Atlas evaluation, development evaluation, auditor campaign, or lockbox evaluation. The remaining 56 microset scene inputs are prohibited. Historical runs and checkpoints are read-only. New artifacts are timestamped, campaign-local, append-only, and collision-refusing.

Atlas, validation, calibration, development, and lockbox arrays are forbidden. Truth, target indices, pair IDs, projection metadata, and scientific constraints may not enter the model input. The only inference tensors are normalized g/r/z blends and the frozen Condition-C Gaussian coordinate prompt.

## Frozen rows and targets

The one-scene ordinary row is micro index `0`, `pu_training_ordinary_00000`, source HDF5 index `0`. The one-scene ambiguous observation is micro index `32`, `pu_training_near_00000`, pair `pu_training_pair_00001`, source HDF5 index `12000`. The eight-scene set is micro indices `[0, 8, 16, 24, 32, 40, 48, 56]`: ordinary `0,8,16,24` and ambiguous `32,40,48,56`, in that exact order. The exact selection is hashed in `tables/frozen_row_selection.csv`.

Every fit uses the exact final P0 tensor and per-sample canonical hashes. No target, threshold, source-layer semantic, prompt rule, hard assignment, or truth-coverage definition may change. The P0 tensor may be read in full only for the mapping representability audit. Scene arrays may be loaded only for the eight frozen source indices.

## Fixed L0 architecture and encoder isolation

Every mapping uses the exact Condition-C shared encoder `4->16->32->64`, the exact L0 decoder blocks `96->32` and `48->16`, and two independent 46,470-parameter expert decoders seeded `2026071201` and `2026071202`. The encoder checkpoint hash is `e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`. Every encoder parameter has `requires_grad=False`, the encoder remains in `eval()` throughout, no encoder tensor enters an optimizer, and encoder state is byte-hashed before and after every condition. Only the two L0 expert decoders train. Decoder topology, width, depth, GroupNorm, SiLU, bilinear skips, 1x1 head, expert count, and parameter count are identical across mappings.

## Exact output mappings and physical path

- M0 ReLU: `physical_normalized = relu(raw)`.
- M1 Square: `physical_normalized = raw ** 2`.
- M2 Absolute: `physical_normalized = abs(raw)`.

The mapped normalized tensor passes through the single frozen positive-scale multiplication to become the physical detected-electron tensor. Training loss, hard assignment, truth coverage, prompt swap, forward consistency, source sum, hashes, and saved outputs consume that exact physical tensor. The reconstruction loss weights physical residuals by the frozen positive per-band scales, preserving the prior dimensionless direct requested-plus-companion MSE without creating a second source-value path. Training on raw output, evaluation-only clamping, detached value-changing postprocessing, or a different train/evaluation mapping is prohibited.

The dtype is float32; channel order is requested g/r/z then companion g/r/z; source order is requested then companion; spatial shape is 60x60; zero background is exact. Numerical-zero tolerance is `1e-07` normalized source units, physical negative tolerance is `0.0` detected electrons, nonfinite tolerance is zero values, and frozen physical round-trip tolerance is `0.00390625` detected electrons. Any physical value below zero or any nonfinite value triggers synchronous termination before another optimizer step.

## Matched initialization

`initial_physical_epsilon` is `9.999999406318238e-08` in the normalized source layer, derived from the frozen `1e-7` exact-arithmetic numerical-zero contract. It maps to per-band physical values g/r/z `0.0000611920 / 0.0001805880 / 0.0001854200` detected electrons. Every final 1x1 convolution weight is initialized to zero. ReLU and absolute biases are `9.999999406318238e-08`; square bias is `sqrt(9.999999406318238e-08)`. Earlier decoder tensors use the same two frozen expert seeds for every mapping. This makes the initial mapped tensor byte-identical across mappings.

## Representability, gradient, and stop-rule gates

For every P0 target, ReLU and absolute use raw witness `target`; square uses `sqrt(target)`. Each witness must map to finite, nonnegative, correctly shaped output within `0.00390625` physical electrons, reproduce deterministically under the canonical hash, and show no positive-target saturation. Exact zero, near-zero, sparse positive, constant positive, high-value, and z-extrema cases are mandatory.

Gradient audits cover initialization, numerical zero, low positive output `1e-6`, median positive P0 value, high P0 value, and z-band extrema. Framework subgradient zero at the exact ReLU boundary or absolute-value cusp is reported but does not disqualify a mapping. A mapping is ineligible only if the derivative is nonfinite or unusable over a material portion of strictly positive P0 support.

Before fitting, isolated sentinels must prove synchronous stopping for negative physical output, NaN, Inf, target-hash mismatch, and simulated MPS fallback. Each expected incident must be written before the injected path terminates; its local status must be failed; optimizer-step count may not advance; and no checkpoint may be promoted. Every self-test must pass.

## Synthetic output-head preflight

With the encoder bypassed, train only a zero-weight L0 16-to-6 1x1 head on MPS using a deterministic 4x4 one-hot spatial basis. Cases are zero, constant positive, sparse positive, the central 4x4 crop of P0 row 0/prompt 0/slot 0, and a 4x4 crop centered on the globally maximal z-channel P0 value under a deterministic first-index tie rule. Each case uses AdamW, learning rate `0.03`, weight decay zero, and exactly `500` steps. A pixel receives one independent weight and the common bias, so the approximate two-parameter Adam displacement bound is `2 * 0.03 * 500 = 30` normalized units, above the frozen P0 maximum near 18.41. Nonzero cases require at least 95% loss reduction; the zero case must remain within `1e-12` normalized MSE. Gradients and outputs must remain finite and physical negatives must remain zero.

## Common neural compute contract

Every real fit uses MPS-only AdamW, learning rate `0.001`, weight decay `0`, no scheduler, gradient-norm clipping `5.0`, microbatch `8`, effective batch `8`, accumulation `1`, exactly `3200` optimizer steps, and exactly `25600` scene presentations. One-scene batches repeat only the one frozen observation eight times. Eight-scene batches present the eight frozen observations once in the fixed row order. There is no shuffle, augmentation, early success stop, extra seed, mapping-specific schedule, or extra optimization. Evaluate at steps 0, 1, every 100 steps, and step 3200. MPS memory probes must pass for all eligible mappings before a real fit.

## One-scene gates

Each gate starts from a fresh matched initialization. The ordinary gate requires zero physical negatives at every used forward, both experts covering the approved ordinary scientific region on both prompts, set-level coverage 100%, median expert diameter <=1.0, set prompt identity 100%, finite forward evaluation, finite gradients, and no dead expert. A failure remains reportable but cannot be selected.

The ambiguous gate uses the same single ambiguous observation and requires zero physical negatives, own coverage 100%, alternate coverage 100%, both-mode coverage 100%, both experts active, no mode collapse, set prompt identity 100%, and finite forward evaluation for both decompositions. If every mapping fails, the campaign stops before eight-scene fitting and concludes that mapping alone is insufficient.

## Eight-scene gate and selection

Only mappings passing both one-scene gates may fit the frozen eight-scene set. The remaining 56 scene rows stay unopened. Report ordinary, own, alternate, both-mode coverage; ordinary expert diameter; prompt swap; forward and source-sum consistency; dimensionless projected-target loss; z-channel projected-target MSE; zero/stagnant derivative fraction; raw magnitude; and mapping boundary/cusp activation.

A mapping is selectable only if representability, self-tests, synthetic fits, both one-scene gates, prompt fidelity, finite behavior, and zero negatives all pass. Among selectable mappings: (1) maximize the minimum of eight-scene ordinary/own/alternate/both-mode coverage; (2) minimize physical projected-target loss; (3) minimize ordinary expert diameter; (4) minimize derivative-stagnation fraction; (5) minimize z-band target error; (6) prefer ReLU only on an exact remaining tie. Forward consistency is evaluation-only and never the first selection metric. No mapping may be selected if all remain at zero both-mode coverage.

## Decisions and authorization

The primary outcome is exactly one of RELU SELECTED, SQUARE SELECTED, ABSOLUTE SELECTED, MULTIPLE MAPPINGS EQUIVALENT, or NO MAPPING PASSES. A practical tie means all preceding lexicographic quantities are exactly equal at their recorded float64 values; the ReLU rule still freezes one later-ladder mapping. One selected mapping authorizes only a separate decoder-capacity-ladder campaign. No width ladder runs here. If no mapping passes, recommend exactly one isolated diagnostic based on dead gradients, hard assignment, frozen encoder representation, or expert-decoder optimization. If one-scene passes but eight-scene fails, aggregation within the microset is the blocker and only the smallest isolated follow-up may be recommended.
