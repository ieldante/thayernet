# Thayer-PU Batch-R1 final report

## Outcome

**THAYER_PU_ELIGIBLE_BUT_LABEL_COLLAPSED**. The fixed eight-row padded MPS executor passed exact deployment
replay and the complete original eligibility continuation. No POST auditor was
trained.

## Required answers

1. **Why did the prior eligibility run stop?** All 24 batch-1 preflight episodes differed in every candidate and deployed hash from batch 8, violating the frozen exact deployment gate.
2. **Was batch-1 inference deterministic?** Yes. Repeated batch 1 was exact.
3. **What was the first divergent layer?** On authoritative MPS, `enc2.block.1`, the GroupNorm immediately after the first `enc2` convolution.
4. **Did model mode or normalization cause it?** The model was recursively in eval mode. GroupNorm's batch-geometry-specific MPS kernel numerics were the first divergence, but there were no batch statistics or running-state semantics.
5. **Were model buffers mutated?** No; all parameters and buffers were bitwise immutable after B1 and B8.
6. **Were latent tensors identical?** Yes on MPS: epsilon, prior statistics, and derived latent z were exact across batch sizes.
7. **Was candidate order identical?** Yes, candidate indices 0–15 mapped exactly to seeds 2026077600–2026077615 in every condition.
8. **Did batch-size-one shape handling cause it?** No. No squeeze, N=1 branch, indexing collapse, bad broadcast, crop, or padding existed in the frozen forward path.
9. **Was the mismatch tensor-valued or hash-only?** Tensor-valued. Raw/canonical equality, hashes, and exact serialization changed together.
10. **Did CPU show the same behavior?** CPU diagnostic inference also depended on batch geometry, first diverging later at `prior.statistics`.
11. **Was MPS responsible?** Yes for the authoritative MPS discrepancy: MPS first diverged at `enc2.block.1`; fixed-geometry neighbor changes were exact. The phenomenon was not uniquely MPS-only because CPU had an analogous effect.
12. **What exact correction was made?** Every MPS neural call now uses exactly eight episode rows; short chunks receive explicit zero dummy rows, real rows retain order, and dummy outputs are removed before scaling and deployment.
13. **Was a fixed executor required?** Yes: `FIXED_BATCH_EXECUTOR_PASS`. Arbitrary native batch geometries remain numerically different.
14. **Did the correction preserve prompt identity?** Yes: majority and individual identity were 1.0, with band rates 1.0/1.0/0.9505208333.
15. **Did it preserve checkpoint and architecture?** Yes; checkpoint SHA-256 remained `c1d17a3f67962cce2fec03d6b15da5f2e330ee97b31c270a7ff019a1373a557e`, and no model source changed.
16. **Did all batch-deployment tests pass?** Yes: 17/17 new tests, 9 inherited tests passed with 1 inherited skip, and all 2,040 candidate/deployed candidate-validation comparisons were exact.
17. **Did the original eligibility preflight pass?** Yes under the frozen fixed executor.
18. **Were complete outputs generated?** Yes: 7591 outputs across 3,998 training, 793 validation, and 2,800 calibration rows.
19. **What were safe and unsafe counts by partition?** training: 0 safe / 3998 unsafe; validation: 0 safe / 793 unsafe; calibration: 0 safe / 2800 unsafe.
20. **Did Thayer-PU supply safe support?** No; label-support gates did not all pass.
21. **Is it an eligible second family?** No for combined-family POST-auditor training because label support collapsed, although deployment consistency and structural family distinctness passed.
22. **What outcome was assigned?** `THAYER_PU_ELIGIBLE_BUT_LABEL_COLLAPSED`.
23. **Is Thayer-Audit v1 authorized?** No.
24. **What exactly happens next?** Recommend exactly one new family: Thayer-Audit Family-E v0 — One Nonnegative Flux-Conserving Frozen Family Eligibility Audit.
25. **Were development, Atlas selection, and lockbox untouched?** Yes; access counts remained 0/0/0.
26. **Were all historical checkpoints unchanged?** Yes; 743/743 historical entries plus the frozen Thayer-PU checkpoint matched.
27. **What reusable code/tests should eventually be committed?** Review the fixed-geometry executor, the 17 regression tests, the tensor/state/shape/divergence audit harness, and the eligibility-continuation launcher; nothing was staged or committed here.
28. **What generated artifacts should remain ignored?** The entire `outputs/runs/thayer_pu_batch_r1_20260714_224244` tree, including preregistration, diagnostic tables, candidate evidence, source freeze, complete reconstructions, labels, bootstrap results, reports, and logs.

## Label-support detail

`tables/thayer_pu_safety_prevalence.csv` and `tables/label_support_gates.csv`
report valid-query, safe/unsafe, prevalence, source-group, prompt-subtype,
catastrophic-pass, output-contract-pass, and joint-safe counts. Family metrics are
in `tables/family_distinctness_gates.csv`; 300 connected-source-group replicates
are in `bootstrap/`.

## Authorization

Thayer-Audit v1 is not authorized.
