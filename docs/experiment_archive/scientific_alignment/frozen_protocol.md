# Preregistration: Thayer-SA scientific-alignment micro-overfit

Frozen at UTC `2026-07-13T02:20:09.591056+00:00` after the persisted Thayer-LG diagnosis reproduced and before the official surrogate audit, official output-space preflight, assignment audit, or neural fitting.

## Scope and immutable inputs

Thayer-SA tests only whether the unchanged 165,612-parameter Thayer-ME architecture can memorize the frozen 64-row training microset under a corrected scientific objective. Architecture, shared prompted encoder, independent decoders, expert seeds `2026071201/2026071202`, training seed `2026071250`, microset manifest `9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085`, target sets, prompt mapping, source-layer semantics, normalization, truth-coverage implementation, scientific thresholds, and forward evaluation contract are immutable. Atlas, validation, calibration, development, and lockbox access are prohibited. No full-data training follows a pass in this run.

## Differentiable scientific distance

All quantities are computed after inverse normalization into physical g/r/z detected-electron source layers. The image component is the frozen symmetric L2 image distance divided by `0.25`, using floor `1e-12`. Each g/r/z total-flux relative error uses the frozen symmetric denominator and floor `1e-12`, then divides by `0.20`. Physical flux-derived g-r and r-z magnitudes use positivity floor `1e-12` and divide absolute color error by `0.20 mag`; display RGB is forbidden. The centroid uses nonnegative summed-band weights, floor `1e-12`, and divides displacement by the fixed mean PSF FWHM and `0.50`. Target tensors are detached; gradients through target values are forbidden. Prediction-dependent portions of the frozen symmetric denominators remain differentiable by design.

The seven normalized components are combined by the zero-anchored log-mean-exp smooth maximum `tau * (logsumexp(v/tau) - log(7))`, with temperature `tau=0.005`. Exact truth must have value at most `1e-6` and gradient norm at most `1e-5`.

## Corrected objective and balancing

For each expert/target pair, `C = L_requested + L_companion + L_science`. Requested and companion reconstruction are squared frozen image-normalized distances; `lambda_science=1.0`. Ambiguous rows use the exact hard minimum over identity and swap, averaging across the two experts. Ordinary rows supervise both experts to the one truth and add weight `1.0` times their threshold-normalized requested-source scientific distance. Prompts are averaged, experts are averaged, and the 32 ordinary and 32 ambiguous rows enter with equal per-row weight. There is no factor from expert, prompt, or target-set cardinality.

Forward loss, source-sum loss, prompt-swap loss, pair-equivalence loss, generic diversity, adversarial, perceptual, uncertainty, likelihood, target-aware separation, and scene-recomposition regularizers are absent. Forward, source sum, prompt swap, and pair consistency are evaluation-only.

## Official surrogate and weight gates

Canonical truth, trained Thayer-ME output, collapsed mean, and source-sum-preserving wrong allocation are compared with the exact metric. Required Spearman is at least `0.95`, Kendall at least `0.90`, and threshold-side agreement at least `0.98`; truth must rank best and approved sets must outrank every compromise. Flux, translation, color, and morphology perturbations must activate their intended components. A one-threshold component violation must contribute between `0.90` and `1.05`. At exact truth no term may have a nonzero gradient above `1e-5`; a zero total gradient is stationary and has component share zero. At compromise, raw/weighted terms, band contributions, and ordinary/ambiguous weights are descriptive and cannot change weights.

## Official CPU output-space preflight

Free output tensors are detached from all model parameters and optimized on CPU float32 with Adam, seed `2026071303`, learning rate `1e-4`, no weight decay, 400 updates, and no clipping-based gate relaxation. Initializations are exact truth, persisted trained Thayer-ME output, collapsed truth mean, source-sum-preserving wrong allocation, and deterministic random bounded within the frozen target minimum/maximum. Exact truth must remain within `1e-6` loss, `1e-5` tensor RMS, and full frozen coverage. From trained/compromise/collapsed starts, corrected loss and mean scientific distance must each fall by at least 10%, and ordinary/own/alternate/both-mode coverage must each enter at least `0.90`. Random output must reduce loss and scientific distance by at least 20%. Any failure is `CORRECTED OBJECTIVE STILL MISALIGNED` and stops the campaign before assignment or neural fitting.

## Assignment audit if preflight passes

The hard two-permutation rule remains unchanged. Audit exact-truth and truth-to-compromise paths, deterministic Gaussian perturbations at `1e-7` through `1e-4`, identity/swap margins, assignment flips, and gradient jumps near ties. Exact-truth median absolute margin must exceed `0.10`; perturbations through `1e-5` may flip at most 5% of non-tied exact-truth assignments. Ties at deliberately collapsed outputs are reported but do not alone fail; instability along covered truth neighborhoods does. Failure stops before neural fitting and recommends one separate smooth-assignment campaign.

## Neural micro-overfit if authorized

Use the exact Thayer-ME architecture and Condition-C encoder warm start, independent expert initialization seeds, full phase-2 trainability contract, AdamW, batch size 8, learning rate `1e-3`, weight decay 0, seed `2026071250`, no augmentation, and at most 400 epochs. MPS is mandatory and CPU fallback is forbidden. Checkpoints are selected by lowest corrected microset objective, then truth-coverage hierarchy, without post-hoc gate changes.

Ordinary both-expert own-truth coverage, ambiguous own coverage, alternate coverage, both-mode coverage, expert-1 prompt identity, expert-2 prompt identity, set prompt identity, ordinary forward consistency, ambiguous forward consistency, and source-sum consistency must each be at least `0.90`; median ordinary expert diameter must be at most `1.0`. Exact truth proves every gate attainable: all rate gates can attain `1.0` and diameter can attain `0.0`; on 32-row groups, `0.90` corresponds to at least 29/32. A nonzero substantial improvement with one remaining gate is PARTIAL SUCCESS. Zero truth coverage, divergent ordinary experts, unusable forward behavior, or prompt collapse is FAILURE.

## Decisions

SUCCESS authorizes only a separate preregistered full non-Atlas campaign. PARTIAL recommends exactly one focused correction. FAILURE distinguishes corrected-objective/output optimization from assignment geometry or neural optimization without adding capacity or loosening thresholds. Atlas, development, and lockbox accesses remain zero under every outcome; all historical checkpoints remain immutable.
