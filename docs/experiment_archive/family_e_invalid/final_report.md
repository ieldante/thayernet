# Thayer-Audit Family-E v0 final report

## Outcome

**DATA_OR_IMPLEMENTATION_FAILURE**.

The preregistered simplex construction is mathematically valid for a
nonnegative observed allocation budget, and its synthetic MPS physical checks
passed. The authoritative BTK observations use zero-background semantics and
preserve signed noise, however. Frozen-target representability therefore
failed in training, validation, and calibration, triggering the mandatory
Part-E stop before architecture construction.

Preregistration SHA-256:
`256bffe3bc53b572b7596bba844f0afdbf4abf3c4cb1d8906fc0ad08663d8881`.

Thayer-Audit v1 is **not authorized**. No POST auditor was trained.

## Decisive physical-contract result

Family-E froze nine per-pixel logits: requested, companion, and residual for
each g/r/z band. Softmax over the three allocations gives fractions summing to
one, and the physical outputs are:

- `P_req = a_req * O`;
- `P_comp = a_comp * O`;
- `P_res = a_res * O`.

For nonnegative `O`, this guarantees nonnegative layers and exact source-sum
conservation. On the synthetic MPS fixture, the maximum conservation error was
`4.76837158203125e-07`; gradients were finite, zero-source error was `0.0`,
and low flux was representable.

The raw frozen observations are signed:

| Partition | Episodes | Negative observed values | Negative fraction | Episodes with negative observations | Target-sum exceedances | Episodes with exceedance |
|---|---:|---:|---:|---:|---:|---:|
| training | 10,000 | 52,582,711 | 0.486877 | 9,999 | 53,963,971 | 10,000 |
| validation | 2,000 | 10,406,743 | 0.481794 | 2,000 | 10,741,143 | 2,000 |
| calibration | 2,000 | 10,419,031 | 0.482363 | 2,000 | 10,392,281 | 2,000 |

All requested and companion targets were finite and nonnegative. The
incompatibility is structural: nonnegative requested, companion, and residual
layers cannot sum to a negative zero-background observation, and the frozen
targets cannot fit where their sum exceeds that observation. No clipping,
offset, additive sky/background term, softplus transformation, or truth-based
rescaling was applied.

Frozen stop token:
`SIGNED_ZERO_BACKGROUND_OBSERVATIONS_INCOMPATIBLE_WITH_NONNEGATIVE_EXACT_SIMPLEX_CONSERVATION`.

## Required answers

1. **Why was POST-auditor training previously impossible?** Condition C and repaired Thayer-PU supplied only unsafe labeled outputs, so no safe class existed for binary POST learning.
2. **What did Condition C produce?** 0 safe / 12,493 unsafe episodes; catastrophic failure 99.9039% and physical output-contract failure 99.8319%.
3. **What did Thayer-PU produce?** After fixed-batch repair: training 0 safe / 3,998 unsafe, validation 0 / 793, calibration 0 / 2,800; total 0 / 7,591.
4. **What exact Family-E physical parameterization was frozen?** A g/r/z per-pixel requested/companion/residual softmax simplex with nine logits and physical layers `a_i * O`, where `O` is the unaltered raw observed zero-background tensor.
5. **How was nonnegativity guaranteed?** For nonnegative `O`, softmax fractions and their products with `O` are nonnegative. The synthetic construction passed. The real `O` is signed, so the construction cannot guarantee nonnegative deployed layers and failed before model construction.
6. **How was flux/source-sum conservation guaranteed?** Softmax fractions sum to one, so the three products sum to `O`; synthetic maximum error was `4.76837158203125e-07`.
7. **Could the parameterization represent the target source layers?** No. Every partition had signed observed pixels and target-sum exceedances; every episode had at least one exceedance.
8. **Did the objective preserve exact truth as a minimum?** Not evaluated. Target representability is a prerequisite; exact truth is outside the frozen physical output space, so the objective could not authorize training.
9. **What was the exact architecture and parameter count?** The preregistered 24/48/96/128 compact U-Net specification has an analytically verified `1,162,737` trainable parameters, below the 3,000,000 ceiling. No neural module was instantiated, so no constructed-model count is claimed.
10. **Did one-scene micro-overfit pass?** Not run; Part E stopped first.
11. **Did eight-scene micro-overfit pass?** Not run; Part E stopped first.
12. **Did full training complete for all seeds?** No; zero seeds were trained.
13. **Were outputs physically nonnegative at every step?** No learned outputs exist. Synthetic nonnegative-input allocations passed from step zero; real signed-input compatibility failed before any step.
14. **Was source-sum conservation exact?** Yes within the frozen MPS numerical tolerance on the synthetic construction. No model outputs were generated on real episodes.
15. **Did promptability pass?** Not evaluated; no model existed.
16. **Did deterministic replay pass?** Not evaluated; no checkpoint or output existed.
17. **Were training auditor-eligibility outputs truly OOF?** No outputs were generated. The frozen five-fold connected-source-group plan is valid—2,000 rows per fold and zero cross-fold group overlap—but it was not executed.
18. **Were source groups leak-free?** Yes at the manifest boundary: zero train/validation/calibration source-group or source-pair overlap, zero duplicate pairs, and zero OOF cross-fold group overlap.
19. **How many training, validation, and calibration outputs were generated?** 0 / 0 / 0.
20. **What were safe and unsafe counts by partition?** Not defined; no Family-E safety labels were generated.
21. **What was safe prevalence by partition?** Not defined.
22. **Which scientific gates dominated remaining failures?** Scientific gates were not reached. Frozen physical target representability was the decisive prerequisite failure.
23. **Did all outputs pass the physical output contract?** No Family-E outputs exist; a 100% output-contract claim is therefore not made.
24. **What was catastrophic-pass rate?** Not evaluated.
25. **What was joint-safe rate?** Not evaluated.
26. **Were safe examples present in every partition?** No safety examples were generated.
27. **Did label-support gates pass?** No; they were not evaluable and Family-E is ineligible.
28. **Was Family-E distinct from Condition C and Thayer-PU?** Structurally preregistered, but not behaviorally evaluable because no Family-E reconstruction exists. Family distinctness did not pass.
29. **What authoritative outcome was assigned?** `DATA_OR_IMPLEMENTATION_FAILURE`.
30. **Is Thayer-Audit v1 authorized?** No.
31. **What exactly happens next?** Run exactly one separately preregistered, training-free signed-noise-residual physical-contract preflight. Keep requested and companion layers nonnegative, permit only the residual/noise layer to be signed, conserve the raw observation exactly, and prove full 10,000/2,000/2,000 target representability before constructing a model.
32. **Were development, Atlas selection, and lockbox untouched?** Yes; access counts are 0 / 0 / 0.
33. **Were all historical checkpoints unchanged?** Yes; 743/743 were present and hash-identical. Condition-C and Thayer-PU checkpoint hashes also matched.
34. **What reusable code/tests should eventually be committed?** Review `src/family_e.py`, `scripts/audit_family_e_physical_preflight.py`, `tests/test_family_e_physical_parameterization.py`, and `tests/test_family_e_campaign_artifacts.py`. They implement the frozen simplex primitive, sequential representability audit, static parameter-count proof, and provenance/integrity tests.
35. **What generated artifacts should remain ignored?** The entire `outputs/runs/thayer_family_e_v0_20260714_195256/` tree: preregistration, provenance, compact manifests, diagnostics, representability evidence, stop records, tables, reports, and empty reserved stage directories.

## Artifact inventory

- Preregistration: `preregistration/family_e_nonnegative_flux_conserving_eligibility.md` and its SHA-256 record.
- Architecture manifest: `architecture/architecture_manifest.json` — frozen but not constructed.
- Physical proof: `diagnostics/physical_output_contract.md`, `physical_contract/target_representability.json`, and `tables/physical_contract_preflight.csv`.
- Objective alignment: `physical_contract/objective_alignment_audit.json` — not run due to the prerequisite stop.
- Micro-overfit: `tables/micro_overfit_results.csv` — all not run.
- Training curves: none; no optimizer step occurred.
- Checkpoints: `tables/family_e_checkpoint_inventory.csv` — none created.
- OOF provenance: `manifests/training_manifest.csv`, `manifests/manifest_provenance.json`, and `inference/oof_provenance.json`.
- Replay: `replay_verification/status.json` — not run.
- Safety-label inventory: `tables/family_e_safety_prevalence.csv` — no labels.
- Gate prevalence: `tables/gate_prevalence.csv` — not evaluated.
- Family comparison: `family_comparison/status.json` — not run.
- Bootstrap: `bootstrap/status.json` — 0/300 replicates because no labels exist.
- Core decision: `reports/frozen_core_decision.json`.
- Integrity: `diagnostics/integrity_audit.json`, `tables/integrity_checks.csv`, and before/after checkpoint inventories.

## Integrity and runtime

- Compileall: PASS.
- Focused tests: 30 passed, 1 skipped, 0 failed.
- CSV/schema validation: 10 files, 0 failures.
- Historical checkpoints: 743 present, 0 missing, 0 mismatched.
- Prior authoritative run artifacts: 0 mismatches.
- Condition-C checkpoint: `e9176dc5…e382`, unchanged.
- Thayer-PU checkpoint: `c1d17a3f…557e`, unchanged.
- README SHA-256: `67f66f35…0116a1`, unchanged.
- Staged index: empty.
- `git diff --check`: PASS.
- Branch / Git HEAD: `thayer-select` / `74b8ff7efbbf7e9891cc8fd8095a9931e3b63174`.
- Runtime: approximately `942` seconds from master-run creation through report assembly.
- Run tree at report assembly: 34 files, 625,199 bytes; allocated size 696 KiB.
- Filesystem free at report assembly: 439,714,116 KiB.
- Final Git status is preserved in `diagnostics/final_git_status.txt`, SHA-256 `ff6d2120480f208b8bb64231b27db4c9a1bfecfc0cb38b67263d09874ece652f`. The staged index is empty; the worktree retains extensive pre-existing unstaged/untracked work plus the Family-E source, tests, and requested documentation additions.

## Authorization

No Family-E training continuation and no Thayer-Audit v1 POST auditor are
authorized by this run. The sole next recommendation is the training-free
signed-noise-residual physical-contract preflight described above.
