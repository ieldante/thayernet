# Thayer-Select Phase II final report

## Executive summary

- **Headline scientific result:** calibrated R1 achieved `PARTIAL SUCCESS` under the frozen gates; null-prompt hallucination on the new development set changed from 7.5% for frozen Phase-I C to 8.2% for R1, and the sampled calibrated risk–coverage curve had area 0.4697.
- **Headline limitation:** this is a controlled BTK development result, not a final-paper or real-sky calibration claim; optional R1 seed replications were not run.
- **Current status:** Phase I remains frozen. Phase II primary training, calibration, and one-time development evaluation are complete. The sealed lockbox remained untouched.
- **Next phase authorization:** The full Ambiguity Atlas and lockbox evaluation are not authorized by this result.

## Answers to the predeclared questions

1. **Provenance and split gates:** passed. Source and duplicate-group crossings were zero; checkpoint and manifest hashes were retained.
2. **Prompt semantics:** implemented and boundary-tested for valid, perturbed-valid, null, ambiguous, edge, equal-distance, and alternate-galaxy requests.
3. **Empirical recoverability:** yes. Labels came from frozen-teacher reconstruction outcomes and fixed scientific contracts, never generator difficulty.
4. **Primary contract:** `permissive`. Selection used only training/validation actionable-label balance under the predeclared imbalance rule; null and ambiguous queries remained abstention targets rather than positive global-acceptance examples.
5. **Full MPS training:** R0 and R1 each completed 20 epochs on MPS, batch size 8.
6. **Uncertainty stability:** bounded log variance [-8.0, 2.0] remained finite and within bounds.
7. **Contract-success prediction:** calibration AUROC was 0.8746 and AUPRC was 0.2475.
8. **Calibration:** `isotonic` was selected by calibration-only five-fold Brier score. Raw Brier 0.1010; calibrated Brier 0.0456.
9. **Risk–coverage:** the frozen decision gate reported selective risk decreased/non-increased as coverage fell.
10. **Coverage points:** 95% risk 0.9537, 90% 0.9511, 80% 0.9456, 70% 0.9379. Corresponding catastrophic rates were 0.5189, 0.5189, 0.5325, and 0.5443.
11. **Null hallucination:** from the prior declared empty-prompt 100% criterion to 8.2% for Phase-II R1 null queries; on identical new null scenes, frozen Phase-I C was 7.5%.
12. **Ambiguous prompts:** mean score separation and forced-selection behavior are in `tables/development_metrics_macro.csv`; the ambiguity decision gate was `False`.
13. **Valid-source cost:** Phase-I C valid normalized RMSE 1.6557; R1 valid normalized RMSE 0.9887.
14. **Seed persistence:** not established; the two optional replications were deferred so calibration and frozen development evaluation would complete.
15. **Within-regime value:** measured in `tables/uncertainty_validity_correlations.csv`; conclusions remain development-only.
16. **Freeze order:** verified. Architecture, checkpoints, contracts, calibrator, score, thresholds, and metric code were hashed before development generation; evaluation occurred exactly once.
17. **Lockbox:** untouched; zero lockbox scenes were generated, opened, rendered, calibrated, or evaluated.
18. **Campaign classification:** **PARTIAL SUCCESS** under the predeclared gates.
19. **Ambiguity Atlas:** not yet justified by the provisional candidate yield.
20. **Exact next experiment:** run two frozen R1 seed replications on the same train/validation/calibration manifests, repeat calibration without changing contracts, and require consistent valid-only risk–coverage improvement before any separately authorized lockbox evaluation or full Ambiguity Atlas.

## Core artifacts

- R0 parameters: 119091; R1 parameters: 123368 (+4277).
- R0 best/final: `e8007205452a77df084caab309fc6c91d23898bd0cbd1f58f7ff6de911b30a6a` / `bc2858045cf3fbc234b57b0b71d9ad9434d16c201cc29d49a5a3b97c67217e5d`.
- R1 best/final: `6637c10fd940b7a853a9e2abd1aef2c371988f31f264c1bf433ec3b161a51750` / `f90d4435e6c07e5d0b4b9e8497809d4d08596b803d08e488d06d84fa36d63c63`.
- Development manifest: `ce0609471311913dc305023e6a46a25d984ce93a7c3d96f447d82752e380179e`.
- R0/R1 runtimes: 586.1s / 1047.7s.
- Git HEAD at start: `9aacc0cb0e819c3296d8da40e049182b8fca5771`; branch `thayer-select`.
- Final correctness audit: all 16 checks passed, including compileall, relevant Thayer-Select unit tests, CSV schemas, checkpoint integrity, bounded variance, sample alignment, calibration isolation, one-time development evaluation, and zero lockbox-scene hits. Full repository discovery was not the campaign gate because the Python 3.9 BTK environment lacks `requests` and pre-existing DR10-only code uses Python >=3.10 `zip(strict=...)`.

Training curves, calibration diagrams, risk–coverage plots, per-query and per-sample tables, failure/null/ambiguous/accepted-rejected galleries, uncertainty diagnostics, manifest hashes, checkpoint hashes, runtime, disk inventory, git status, and old-checkpoint integrity are stored under this timestamped run. These are controlled BTK development results. DR10 remains a real-sky OOD benchmark.

The one-time development pass retained per-sample scalar pixel-uncertainty aggregates but did not persist full uncertainty-map arrays. The maps were not regenerated after the reporting failure because doing so would require a prohibited second development inference pass. This is a documented deliverable omission and should be corrected prospectively in the next frozen campaign.
