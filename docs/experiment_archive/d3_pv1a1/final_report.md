# Final PV1-A1 scientific entrypoint implementation and authoritative D3 execution

Status: **engineering/readiness PASS; authoritative scientific outcome `MIXED_CAUSE`; downstream authorization `none`.**

The exact frozen R3 command executed once. Primary L0 ran from the fresh step-zero state through its 5,000-step budget and did not satisfy the full frozen success predicate. Runtime and independent replay both selected branch `NONE`; eight-scene science and the capacity ladder therefore did not run. The scientific branch status remains `UNKNOWN`, while the frozen outcome mapper classified the observed informative-no-success optimization diagnostic plus repeated hard-assignment flips as `MIXED_CAUSE`.

Important registry clarification: both machine-readable registries in the authoritative R2 bundle contain the same **15**, not 20, ordered stages. R3 loaded and executed those exact 15 stages. The 20 numbered implementation requirements were enforced as orchestration obligations; five operation names were not invented.

## Authoritative identities

- Protocol: `THAYER-D3-PV1-A1`
- R3 root: `outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200`
- Entrypoint SHA-256: `4529f311e759829b7385bab2f1b1deb6e5564c7334571cc4739fd8b632d7835e`
- Source-freeze SHA-256: `78444d641953b5df8cc6779762f935e0d3dd61635a66cdccc3c38934597ac119`
- Event-chain root SHA-256: `155891532556a24bebfe97be89e142f74843dadea266e81f43445cc73573eefc`
- Protocol-bundle SHA-256: `3ee5127b6c37b554c8e136dff66e77954d5aedbe905cf809101966bf08c24fc0`
- Scientific run: `scientific_run/authoritative_pv1a1_20260714_182005_552915`

## Final scientific result

- L0 scientific result: `FAIL` at `BUDGET_EXHAUSTED`, step 5,000.
- Final L0 own coverage: pass; maximum own distance `0.7153765825510313`.
- Final alternate coverage: fail; maximum alternate distance `3.6737798207730687`.
- Final both-mode coverage: fail.
- Prompt identity: pass.
- Forward consistency: pass for all four prompt/expert candidates.
- Assignment flips: 5.
- Optimization diagnostic: `INFORMATIVE_NO_SUCCESS`.
- Hard-assignment diagnostic: `REPEATED_FLIPS`.
- Square-mapping diagnostic: `USABLE_DERIVATIVES`.
- Validated observed capture: `1.0`, not below the frozen capacity threshold.
- Selected branch: `NONE`.
- Eight-scene scientific result: not produced.
- Capacity-ladder result: not produced.
- Frozen outcome category: `MIXED_CAUSE`.
- Downstream authorization: `none`.

Scientific counts were 10,108 decoder forwards, 5,054 target-dependent loss evaluations, 5,000 backward passes, 5,000 optimizer steps, one branch decision, one model construction, and 54 append-only checkpoints.

## Gate and integrity result

- Primary gate: 483 passing XML test cases (460 ordinary tests plus 23 subtests), 5 justified platform skips, 0 failures, 0 errors.
- BTK gate: 6 passed, 0 failed, 0 errors.
- Union: 489 passed, 5 skipped, 0 failed, 0 errors. This is the R2 union plus 18 new entrypoint/regression cases.
- Audit corruption assertions: 50/50 pass.
- Runtime/independent synthetic replay: 16,832/16,832 exact.
- Fresh capacity construction: 12/12 exact.
- Independent readiness audit: 30/30 PASS.
- Scientific checkpoint replay: PASS at terminal step 5,000.
- Source unchanged after first scientific loss: yes.
- Protected legacy checkpoint model/optimizer loads: 0.
- Historical checkpoint integrity: 600/600 unchanged.
- README unchanged; staged index empty; `git diff --check` and cached check pass.
- Atlas, development, lockbox, and broader-scene access counts: 0.

## Required questions

1. **Why did R2 stop?** Its independent audit found one blocker: `future_command_entrypoint_exists = false`.

2. **What exact entrypoint was absent?** `<REPOSITORY_ROOT>/scripts/run_thayer_d3_pv1a1_scientific.py`.

3. **What regression test reproduced the failure?** `test_pv1a1_missing_scientific_entrypoint_reproduces_r2_failure`; the pre-fix run preserved 2 passes and 11 expected failures caused by the absent file.

4. **Was the exact future-command CLI implemented?** Yes. It accepts the five exact frozen arguments and flags, and the command array was loaded directly from the R3 template.

5. **Were all 20 operations loaded from the frozen registry?** The frozen bundle has no 20-entry machine registry. Both authoritative registry sources contain 15 identical stages; all 15 were loaded and executed, while all 20 implementation obligations were enforced without inventing operations.

6. **Did operation order match exactly?** Yes. The scientific operation replay's expected and actual 15-stage sequences are identical.

7. **Was scientific logic reused rather than duplicated?** Yes. Initialization, optimizer/loss helpers, metric calculators, policy functions, audit observers, runtime producer, independent replayer, cache generator, capacity constructors, and checkpoint primitives were reused. The entrypoint contains orchestration and validation.

8. **Did fresh seeded initialization reproduce?** Yes, with canonical state `b011ecfd8478c5d4f5656483f4e25f5b776fefb42150cd51133dc42d31706e9b`.

9. **Was the legacy checkpoint scientifically excluded?** Yes. It was byte-hashed only; model loads, optimizer loads, transfers, and scientific evidence references were all zero.

10. **Did audit noninterference pass?** Yes: RNG, model, optimizer, tensor, target, and gradient checks passed.

11. **Did the actual entrypoint reach the pre-scientific boundary?** Yes, for both the R2 and frozen R3 command templates, with `READY_TO_EXECUTE_PV1A1_SCIENCE` and zero pre-science loss/update/decision counts.

12. **What normal implementation defects were repaired before science?** Source-manifest hash-core validation, cache-promotion-rule interpretation, batched-tensor observation, MPS/CPU assignment-audit normalization, and strict JSON conversion of NumPy booleans. Each had preserved failing evidence before its minimal fix.

13. **What is the new source-freeze hash?** `78444d641953b5df8cc6779762f935e0d3dd61635a66cdccc3c38934597ac119`.

14. **What is the new protocol-bundle hash?** `3ee5127b6c37b554c8e136dff66e77954d5aedbe905cf809101966bf08c24fc0`.

15. **Did the complete gate matrix pass?** Yes: 489 passed, 5 skipped, 0 failed, 0 errors.

16. **Did the independent audit reach 30/30?** Yes, before scientific execution.

17. **Did the exact future scientific command execute?** Yes, once, directly from the frozen R3 command array; exit code 0.

18. **How many scientific decoder forwards occurred?** 10,108.

19. **How many backward passes occurred?** 5,000.

20. **How many optimizer steps occurred?** 5,000.

21. **What scientific branch was selected?** `NONE`; the branch-level D3 status is `UNKNOWN`.

22. **What authoritative scientific outcome was produced?** `MIXED_CAUSE`, after L0 budget exhaustion with informative optimization and repeated assignment flips.

23. **Is L0 capacity sufficient?** No. L0 did not satisfy the frozen full scientific success predicate.

24. **Is eight-scene work authorized?** No. The L0 success branch was not selected, and no eight-scene scientific result was produced.

25. **Is a capacity ladder authorized?** No. Optimization and hard-assignment alternatives were supported, and observed capture was not strictly below the capacity threshold.

26. **What exact experiment happens next?** None. Downstream authorization is `none`; no follow-up experiment is approved by this result.

27. **Were broader scenes, Atlas, development, and lockbox untouched?** Yes. Only the eight authorized source indices were read from the frozen training-scene container; broader-scene, Atlas, development, and lockbox counts are zero.

28. **Were all historical checkpoints unchanged?** Yes, 600/600 exact after science.

29. **What reusable source/tests should eventually be committed?** `scripts/run_thayer_d3_pv1a1_scientific.py` and `tests/test_d3_pv1a1_scientific_entrypoint.py`, after normal review. Nothing was staged or committed here.

30. **What generated artifacts should remain ignored?** The entire R3 output tree: diagnostics, logs, preregistration, regression results, pre-science traversals, source/protocol freezes, event/audit records, generated caches, test XML/logs, scientific checkpoints, semantic states, replay artifacts, reports, figures, and tables.

No retry, repair, threshold change, policy change, or follow-up scientific run was performed after the first target-dependent loss.
