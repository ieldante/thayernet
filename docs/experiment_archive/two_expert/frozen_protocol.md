# Preregistration: Thayer-ME two-expert ambiguity decoder

Frozen at UTC `2026-07-13T00:31:32.388577+00:00` after exact Thayer-MH reproduction and target-set reuse audit, and before model implementation, microset fitting, full fitting, checkpoint selection, or Atlas inference.

## Scientific hypothesis and scope

A Condition-C-compatible shared prompt-sensitive encoder followed by two independently parameterized compact expert decoders can represent both approved decompositions of an ambiguous observation, while both experts converge to the same answer on ordinary uniquely recoverable scenes. This is a decoder-capacity and specialization feasibility campaign, not a posterior, auditor, catalog policy, final-lockbox campaign, or proof of uniqueness.

The exact read-only Thayer-MH training, validation, and calibration scene tensors, manifests, and target sets are reused. Target hashes are `{"calibration": "9f660292c957ff72cd00356b82ccf3461a2e99f8a0fdb819a6e5d20084140910", "training": "7fc92222ff2d980c4beb787b961fa7bdaf3130c055ce842dc8fd5f600c29c19a", "validation": "a73477ab54f8c95ee6c14a9b13574e6f65e185e9dcebdc6f158dc564e573a55e"}`. Scene hashes are `{"calibration": "f86c7da62272c8c895a0be3e54020211cce86907fd86d53ae7478d5c672117f4", "training": "d6ca6f1cbcb136a075f0216460e5f6b2dcd5fefbb63894803b86069df4e5f48d", "validation": "cbc6db971d4e78d85e572227e2b63a4d8789717a557d3ac2a3111896a1007699"}`. No Atlas group, development row, final-lockbox row, source ID, pair ID, morphology label, simulator difficulty, or generator parameter may enter inference.

## Architecture and initialization

The shared encoder is the Condition-C-compatible `enc1 4->16`, `enc2 16->32`, `bottleneck 32->64` path receiving only normalized g/r/z plus the fixed Gaussian coordinate prompt. Every compatible encoder tensor is loaded exactly from Condition C. No hypothesis token or expert identity enters the encoder or observed input. `enc1` and `enc2` remain frozen; `bottleneck` is frozen for phase 1 and may train only in phase 2.

Expert 1 and Expert 2 each independently contain `dec2 96->32`, `dec1 48->16`, and a `16->6` output head. They share no convolution, normalization, output-head, or late assignment parameters. Each emits requested g/r/z plus companion g/r/z with the unchanged unclipped, zero-background source-layer semantics. The decoders are initialized independently with frozen seeds 2026071201 and 2026071202; neither is copied from Condition C or from the other expert. Expected parameters are 72,672 shared encoder + 46,470 per expert = 165,612 total, below the frozen 250,000 ceiling.

## Loss and assignment

For ambiguous targets `{Y_A,Y_B}`, compute both expert-to-target assignments and minimize the sum of requested, companion, and source-sum reconstruction costs. The winning assignment is per scene and no global expert semantics exist. Ordinary scenes supervise both experts to the one approved decomposition and add a 0.10 concentration loss. The fixed full objective also includes 0.50 observed-blend forward/recomposition loss, 0.25 unordered prompt-swap loss, and 0.05 pair-set consistency. There is no generic diversity reward and no target-aware separation term.

## Isolated micro-overfit gate

The microset is frozen as the first 32 training ordinary scenes plus both members of the first 16 sorted training ambiguity pairs, for 32 ambiguous observations. Both prompts are used. Validation, calibration, Atlas, development, and lockbox rows are prohibited. The fit uses MPS only, AdamW, batch size 8, learning rate 1e-3, weight decay 0, seed 2026071250, at most 400 epochs, and early stopping only after all micro gates pass. The exact frozen scientific-distance and forward-consistency metrics apply. Required rates are >=0.90 for ordinary own-truth, ambiguous own-truth, alternate-truth, both-mode, prompt swap, and ordinary/ambiguous forward consistency; median ordinary expert diameter must be <=1.0. Failure is `REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE` and prohibits full training. Capacity and gates cannot change afterward.

## Full training and specialization audit

Only after micro pass: MPS-only AdamW, seed 2026071260, 30 epochs, batch size 8, learning rate 3e-4, weight decay 1e-4, and the exact Thayer-MH 6 ordinary + 2 ambiguity observations per batch schedule. Epochs 1-5 freeze the full encoder; epochs 6-30 unfreeze only the bottleneck. Select one best checkpoint by lowest protected validation objective; save best and final separately. Track per-expert assignment frequency/entropy, output distance, gradient norm, parameter distance, ordinary concentration, ambiguous separation, identity, forward consistency, coverage, assignment flips, and flux-scale-only differences. Stop on NaN/Inf, MPS fallback, hash/source exposure, collision, dead expert, sustained ambiguity collapse, uncontrolled ordinary divergence, or source-sum/forward instability.

## Non-Atlas and Atlas gates

Gate ranges and thresholds are frozen in `tables/preregistered_gate_attainability.csv`. Promptability is checked before truth coverage; truth coverage before control concentration. Own, alternate, and both-mode near-collision coverage must each be at least 0.05, rather than merely one example. Any failed stage stops before Atlas. Only after every non-Atlas gate passes may the selected checkpoint, matching, forward threshold, truth metrics, diameter metrics, controls, low-FPR threshold, and success gates be frozen and hashed. Atlas inference is one pass over the 50 frozen observations, with no retraining, recalibration, threshold change, seed addition, or post-Atlas tuning. Overall success requires every preceding gate; partial success requires non-Atlas specialization and nonzero Atlas truth coverage but insufficient operational low-FPR performance.

Final-lockbox and unauthorized development access remain zero under every outcome. Historical artifacts remain immutable.
