# Frozen Loss-Geometry Audit preregistration

Working name: **Thayer-LG (Thayer Loss Geometry)**  
Frozen at UTC: `2026-07-13T00:57:35.272851+00:00`  
Audited microset manifest SHA-256: `9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085`

## Scientific boundary

This is a training-free audit of persisted Thayer-ME microset inputs, approved target sets, and trained outputs. Neural parameters are immutable and may not receive gradients or optimizer steps. Automatic differentiation is allowed only with respect to detached output tensors. Atlas inference, validation/calibration evaluation, historical development evaluation, and final-lockbox access are forbidden. Results cannot be used to alter a loss weight, target set, source-layer semantic, or coverage threshold in this campaign.

## Frozen rows and contracts

The audited rows are exactly the 64 rows in the immutable Thayer-ME microset manifest: the first 32 training ordinary rows selected by the original selector and both observations of the first 16 lexicographically sorted approved training pairs. Both prompts are retained. Outputs are float32 N x 2-expert x 6-channel x 60 x 60 tensors; channels 0:3 are requested g/r/z and 3:6 are companion g/r/z. Values are unclipped normalized source layers, inverted with the frozen training-only g/r/z scales. Background is exactly zero.

The frozen objective is the implementation in `src/models_two_expert_decoder.py` as invoked by `scripts/run_thayer_two_expert_micro_overfit.py`: decomposition cost = requested MSE + companion MSE + 0.5 source-sum MSE; ordinary scenes supervise both experts to target zero and add 0.10 expert concentration; ambiguous scenes use hard min(identity, swap); prompt losses are averaged across A/B; total = target + 0.5 forward + 0.25 prompt-swap + 0.05 paired-observation set equivalence. No additional regularizer or target-aware separation term exists. Weights are immutable.

For scene-level accounting, paired-observation equivalence is assigned to both members of each ambiguous pair with effective coefficient 0.10; its mean over all 64 rows therefore equals the original 0.05 pair-average contribution. Aggregate reproduction also evaluates the original batch-equivalent formula directly.

## Canonical configurations

Ordinary: O1 exact target duplicated; O2 persisted trained experts; O3 trained expert mean duplicated; O4 all-zero; O5 exact target duplicated after transferring 25% of requested light into the companion layer while preserving the source sum. Ambiguous: A1 exact approved set in stored order; A2 the same set with expert slots reversed; A3 persisted trained experts; A4 both experts equal the pixelwise mean of the approved set; A5 both equal the persisted expert mean; A6 own target duplicated; A7 alternate target duplicated; A8 each exact decomposition is replaced by a 50/50 requested/companion allocation at fixed source sum. The corresponding prompt-specific target tensors define prompt A/B mappings.

## Distances and coverage

The primary scientific distance is the frozen `scientific_distance` maximum of image NRMSE/0.25, per-band relative flux/0.20, valid g-r and r-z color differences/0.20, and centroid displacement in mean-PSF units/0.5. Coverage requires primary distance <= 1.0 and frozen forward plausibility. Image, flux, color, centroid, and forward metrics are reported separately. The differentiable scientific surrogate uses the maximum of the same image, relative-flux, valid-color, and centroid components with epsilons fixed by the implementation; at nondifferentiable ties PyTorch's deterministic subgradient is accepted.

## Paths and perturbations

All interpolation grids are frozen to 21 equally spaced points from 0 through 1. Truth-to-trained is affine. Truth-set-to-collapsed moves each expert toward the approved-set mean. Source-light transfer uses 21 fractions from -0.5 through 0.5; positive values move requested light to companion and negative values move companion light to requested, always preserving the sum. Expert separation moves duplicated set mean toward the two exact targets. Flux-preserving morphology mixes each requested source with its one-pixel positive-x roll and rescales every band to its original total when the denominator magnitude exceeds 1e-12.

Assignment perturbations use deterministic Gaussian directions with seed 2026071301 and scales 1e-7, 1e-6, 1e-5, and 1e-4 normalized units. Finite-difference gradient checks use central step 1e-4. Directional curvature uses central steps 1e-3 and reports `(L(x+h d)-2L(x)+L(x-h d))/h^2` for unit-L2 directions. A flat-direction flag is absolute curvature <= 1e-6; weak curvature is <= 1e-4. Gradient cosine uses denominator floor 1e-20; two zero gradients are reported as undefined, not aligned.

Potential dominance is flagged when one weighted term supplies >= 0.75 of the sum of individual weighted gradient L2 norms, or when one band/layer supplies >= 0.75. A gradient conflict is negative cosine; severe conflict is <= -0.5. Assignment instability is any slot flip under perturbation <= 1e-5 or any interpolation assignment margin <= 1e-7.

## Output-space optimization

All neural state is absent from the optimization graph. Free output tensors use CPU float32 Adam, seed 2026071302, learning rate 0.01, 40 updates, logging every 5 updates, and elementwise diagnostic bounds [-8, 8] applied after every step. D0 runs from exact truth, persisted trained outputs, collapsed/duplicated mean, uniform random [0,1], and the source-sum-preserving compromise. D1 target reconstruction/set matching only; D2 target plus ordinary concentration (identical to the exact implemented target loss, reported separately for clarity); D3 target plus 0.5 forward; and D4 removes each of target, forward, prompt-swap, and pair-equivalence once. D1-D4 start from persisted trained outputs. These are diagnostics, never model fitting or future-loss selection.

## Decision gates

Any frozen-input mismatch stops the campaign. Any exact truth that cannot be constructed, hashed, mapped across prompts, pass its named coverage check, satisfy ordinary duplication concentration, or satisfy frozen forward plausibility stops interpretation and is classified as OUTPUT-CONTRACT DEFECT or COVERAGE-METRIC DEFECT. Otherwise the primary category is selected exactly from the ten user-specified categories. OBJECTIVE MISALIGNMENT requires direct lower frozen objective for a compromise/trained output than approved truth or a truth-started D0 trajectory that leaves coverage while lowering objective. LOSS-SCALE DOMINANCE, GRADIENT CONFLICT, and PERMUTATION-MATCHING PATHOLOGY require the thresholds above. OPTIMIZATION/NETWORK BOTTLENECK requires truth representability, truth objective optimality, and direct optimization reaching truth while the neural microfit failed. MIXED CAUSE is used only when multiple categories have direct evidence. Exactly one future experiment will be recommended and not run.

## Numerical and access guarantees

CSV floats retain at least 10 significant digits. Equality reproduction tolerances are absolute 1e-7 for persisted aggregate rates and 1e-6 for persisted expert diameter; tensor reconstruction tolerance is max absolute 1e-6 physical electrons and 1e-7 normalized units where exact arithmetic is expected. SHA-256 is used for files and canonical little-endian float32 CHW tensors. No loss weight or threshold may be tuned after inspecting results. Access counters for Atlas, development, and lockbox remain zero.
