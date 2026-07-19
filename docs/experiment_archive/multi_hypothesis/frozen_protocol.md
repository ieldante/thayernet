# Preregistration: ambiguity-set multiple-hypothesis decoder

Frozen at UTC `2026-07-12T23:07:07.449562+00:00` after exact baseline reproduction and expanded Atlas-source exclusion, and before Thayer-MH model implementation, target rendering, fitting, checkpoint selection, or new Atlas inference.

## Hypothesis and boundaries

A compact coordinate-conditioned K=2 shared decoder trained with explicit set-valued supervision can represent both scientifically approved decompositions of an observationally equivalent non-Atlas pair while collapsing to one solution on ordinary controls. It must preserve prompt identity and forward consistency. This is decoder-representation feasibility, not posterior calibration, posterior completeness, black-box auditing, or proof of uniqueness.

Excluded Atlas-related groups: 36,288; exclusion-set SHA-256 `821a060c402746ce17b72af42ff474d9943b2cb391c8ec294099d22c4cfc0140`. Approved-source commitment SHA-256 `74a82655e2e47a39ae7be3872cfa313a89d62e97be7fd8a268301b5e999c9cfe`. Final lockbox, historical development, all Atlas observations, and all Atlas-source groups are prohibited during training, validation, calibration, target construction, debugging, and model selection.

## Prospective data and equivalence contract

- Training: 12,000 ordinary observations and 3,000 ambiguous observations from 1,500 pairs.
- Validation: 1,500 ordinary and 500 ambiguous observations from 250 pairs.
- Calibration: 1,500 ordinary and 500 ambiguous observations from 250 pairs.
- Fresh near-collision search pools: 16,000 / 4,000 / 4,000 scenes for training / validation / calibration, with seeds 2026079101 / 2026079102 / 2026079103.
- Ordinary seeds: 2026079201 / 2026079202 / 2026079203. Noise bases: 2026079300 / 2026079400 / 2026079500 multiplied by 10,000 plus scene index.
- Each pair uses four distinct source groups from exactly one allowed partition and two unique pool scenes. No group crosses partitions.
- A pair is approved only when exact replay/additivity/finite/hash checks pass, mean whitened observation distance <=1.0, requested-source primary scientific distance >1.0, and global-rescaling relative residual >0.01. Candidate selection is rank by observation-embedding distance divided by target-embedding distance, with 32 nearest neighbors and no scene reuse. Counts are hard requirements; weak pairs are forbidden.
- Ordinary target set is the one canonical full decomposition. For either member of an approved pair, the target set contains the left and right full decompositions. Pair provenance and group IDs are stored only in manifests, never inference tensors.

## Full decomposition and prompt semantics

Each hypothesis outputs six unclipped normalized channels: requested g/r/z followed by companion g/r/z. Both are zero-background PSF-convolved source layers; their sum is the noiseless two-source scene. Prompt A maps canonical `[A,B]` and prompt B maps `[B,A]`. For an alternate decomposition, the source nearest the requested coordinate is requested and the other source is companion; the association must be unambiguous and is frozen in the target manifest. Band order g/r/z, training-only normalization, no activation, no clipping, source ordering, and prompt-swap channel exchange are frozen.

## Architecture and warm start

Thayer-MH uses the Condition-C-compatible prompted encoder (`enc1 4->16`, `enc2 16->32`, `bottleneck 32->64`), shared decoder (`dec2 96->32`, `dec1 48->16`), and one shared 16->6 head. K=2 learned 8-dimensional hypothesis tokens are injected through learned linear maps into the 64-channel bottleneck and 32-channel late decoder stage. Decoder weights are shared; only the token distinguishes slots. Matching Condition-C encoder/decoder tensors load exactly, and the historical 3-channel head initializes both output halves. The exact parameter count must be <=300,000. No stochastic sampling, family ID, source ID, catalog metadata, target truth, or generator difficulty enters inference.

## Loss and training

For each prompt, component losses are normalized MSE with fixed weights: requested 1.0, companion 1.0, source-sum 0.5, forward/noiseless-sum 0.5, prompt-swap 0.25, ordinary concentration 0.10, ambiguous pair-equivalence set consistency 0.05. For a two-target set, compute identity and swapped total target assignment and use the smaller assignment per scene without preserving a global slot identity. For an ordinary one-target set, supervise both hypotheses to the one target and apply hypothesis concentration. There is no generic diversity reward.

Training is MPS-only with `PYTORCH_ENABLE_MPS_FALLBACK` disabled: seed 2026079601, 30 epochs, batch size 8, AdamW learning rate 3e-4 and weight decay 1e-4, fixed ordinary:ambiguous batch ratio 3:1. Epochs 1-5 freeze enc1, enc2, and bottleneck while training tokens, injections, decoders, and head. Epochs 6-30 keep enc1 and enc2 frozen and unfreeze bottleneck. Select the best checkpoint by the lowest validation objective only; save best and final separately. Stop on NaN/Inf, MPS fallback, manifest/hash/source exposure, checkpoint collision, prompt collapse sustained for 3 validations, source-sum instability, ambiguous slot collapse sustained for 3 validations, or uncontrolled ordinary divergence.

## Gates and one-time Atlas boundary

All gate definitions and attainable ranges are frozen in `tables/preregistered_gate_attainability.csv`. Promptability is evaluated first; then non-Atlas set coverage, control concentration, pair consistency, and forward consistency. Own and alternate near-collision coverage and both-mode coverage must each be nonzero. Ordinary false witnesses must be <=0.10, near/control diameter ratio must be >=1.25 with pair-bootstrap lower endpoint >1, and non-Atlas recall at the calibration-control 95th-percentile diameter must be nonzero. Any failure stops before Atlas.

Only after every non-Atlas gate passes may one selected checkpoint, threshold, ordering rule, deterministic inference protocol, metrics, controls, and success gates be hashed. Atlas evaluation is exactly one pass over 50 frozen observations; no retraining, recalibration, threshold changes, or post-Atlas tuning is allowed. Atlas own and alternate coverage must become nonzero, AUROC must be >=0.806, recall at the frozen 4% control FPR >=0.32, and safe-control false witnesses <=0.10. Final lockbox access remains zero under every outcome.
