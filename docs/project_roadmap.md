# Thayer-Select project roadmap

1. **Complete — promptability.** Freeze the group-safe CatSim partitions,
   validate explicit-seed replay, compare centered/unprompted/randomized/
   prompted controls, and establish prompt-swap behavior.
2. **Complete with partial success — recoverability.** R0 and R1 completed,
   calibration used calibration only, and the newly frozen development manifest
   was evaluated once. Discrimination and risk–coverage improved, but ambiguity
   ranking and catastrophic-error gates failed.
3. **Next authorized — replication.** Repeat R1 with two independent initialization
   and minibatch-order seeds without changing architecture, manifests, losses,
   contracts, calibration protocol, or metrics.
4. **Not yet authorized — ambiguity benchmark.** Feasibility mining checked
   77,671 filtered candidate edges and found zero pairs meeting both provisional
   cutoffs. Do not build a full Ambiguity Atlas yet.
5. **Separately authorized — final/real-sky evaluation.** Keep the lockbox
   sealed until the full protocol is frozen. Treat DR10 as a real-sky OOD
   benchmark with its independent source-only/PSF/unit gates, not as a shortcut
   around controlled validation.

## Roadmap update after frozen-head ablation

6. **Complete — seed replication and root-cause analysis.** Phase-II instability,
   ambiguity inversion, isotonic collapse, low-SNR failure concentration, and
   unused frozen-latent information are documented.
7. **Complete with no clear improvement — frozen-head diagnostic.** H0-H4,
   calibration comparisons, the centroid augmentation, and the non-deployable
   oracle used only train/validation/calibration evidence. No development or
   lockbox evaluation occurred.
8. **Exactly one next experiment — target redesign.** Redesign and preregister
   the moderate reliability contract with failure-specific labels before any
   further head, backbone, representation, or ambiguity-construction change.
9. **Still sealed — final lockbox.** Do not use the lockbox for contract design,
   target selection, calibration, debugging, visual review, or threshold tuning.

## Roadmap update after hierarchical safety campaign

10. **Complete — hierarchical policy experiment.** Query validity, separate
    valid-only image/flux/centroid risks, confusion risk, vector scaling,
    split-conformal upper bounds, and one frozen accept/abstain rule were tested
    without changing Condition C.
11. **Successful component — query gate.** The three-state gate removed the
    ambiguity inversion, rejected all fresh development NULL queries, and cut
    AMBIGUOUS false acceptance to 9.2% at 66.65% valid-query coverage.
12. **Failed system gate — operational coverage.** The complete policy accepted
    1/2,000 development valid scenes and did not beat the historical R1 ranking
    at useful diagnostic coverage. Lockbox evaluation is not authorized.
13. **Next experiment — risk-limit feasibility and conditional conformal.** Use
    train/validation/calibration artifacts only. Audit aperture flux scaling and
    log-tail stability, preregister a fixed catastrophic-risk budget plus at
    least 70% valid calibration coverage, and compare with R1 before creating
    another development set. Keep Condition C frozen.
14. **Ambiguity benchmark — targeted pilot only.** A later pilot may combine
    simulator optimization, matched source pairs, and multi-hypothesis truth
    sets. Do not build the full Atlas and do not use development or lockbox
    scenes for ambiguity engineering.
15. **Protocol correction — complete.** Preserve the 2026-07-11 hierarchical
    result as historical evidence, but do not certify its sequence as fully
    preregistered. The 2026-07-12 corrective audit reconstructed every original
    composite label and stopped before new inference or fitting.
16. **Next authorization gate — prospective feasibility only.** Before another
    development manifest, freeze and hash a new preregistration, use one
    reconstruction provenance across train/validation/calibration, complete the
    row-level contract and drift audits, and pass the calibration-only minimum-
    coverage gate. The lockbox remains sealed.

## Roadmap update after prospective feasibility

17. **Complete with partial success — prospective component feasibility.** The
    query gate and image/flux/centroid/confusion rankers passed under uniform
    Condition-C provenance. Marginal calibration retained resolution.
18. **Frozen-gate failure — catastrophic AUPRC criterion.** The observed
    prevalence made the preregistered `1.25 × prevalence` AUPRC threshold exceed
    1.0. Preserve the failure; do not reinterpret the excellent 0.997 AUPRC as
    a formal pass.
19. **Exactly one next experiment — conditional-calibration correction.** Keep
    the reconstructor and heads frozen, preflight every gate for attainability,
    replace the unbounded AP ratio with a bounded prevalence-adjusted lift, and
    require 85–95% coverage plus bounded 95th-percentile width in each frozen
    SNR/overlap subgroup.
20. **Still prohibited — development and lockbox.** Do not build a development
    manifest or full hierarchical policy until the corrective feasibility
    experiment passes under a separately hashed preregistration.
