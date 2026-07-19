# Atlas stochastic hypotheses

Thayer-PU was evaluated once on the 50 frozen Atlas v0 observations after every
non-Atlas gate passed and the checkpoint, K=32 sample seeds, forward tolerance,
matched controls, clustering rule, scientific metrics, and operating thresholds
were hashed and frozen.

The result is **PARTIAL SUCCESS**:

- model-generated witnesses: 24/50, versus 19/50;
- candidate-diameter AUROC: 0.856, versus 0.4712;
- pair-cluster bootstrap 95% AUROC interval: 0.751–0.942;
- recall at 4% control false positives: 0.32, versus zero;
- safe-control false witnesses: 0.08;
- Atlas forward-consistency rate: 1.0;
- own-truth coverage: 0.0;
- paired alternate-truth coverage: 0.0.

The candidate family exposes more operationally discriminative, observation-
consistent disagreement, but its prior samples do not cover the frozen Atlas
truths. It misses the preregistered 30/50 witness target and both coverage gates.
No formal posterior-correctness, black-box-auditor, catalog-admission, or survey-
population claim follows. No witness still does not prove uniqueness.

The exact next experiment is one preregistered conditional normalizing-flow
prior correction on the frozen Thayer-PU representation. It must retain every
current source exclusion, non-Atlas gate, one-pass Atlas rule, and lockbox
protection. Increasing VAE size or tuning on Atlas is not authorized.

