# Thayer-Family-E1-v0 final report

## Outcome

**FAMILY_E1_RECONSTRUCTION_FAILURE**.

The physical contract and objective-alignment audit passed, and ordinary one-scene micro-overfit passed. The difficult and mandatory mixed-eight tests failed the frozen prompt-identity gate at `0.5000` and `0.5625` versus `0.90`. Full training, checkpoint selection, OOF inference, replay, safety labeling, family comparison, bootstrap, and auditor training therefore did not run.

Preregistration SHA-256: `33c65102ec946cb980709fe66ca3728e85e0066c844354932af49d31c2aa65d5`. Architecture manifest SHA-256: `d51d6cff367d4f794b718f9c9a1864ee88aecabef27387eb6ea6fb0a955f1664`. Exact parameter count: `1,162,662`.

## Required answers

1. **Why did the original all-nonnegative Family-E contract fail?** It required requested, companion, and residual all to be nonnegative while summing to signed zero-background observations; every partition contained negative observed pixels and target sums above the observation.
2. **What signed-residual contract was authorized?** `P_req = S*ReLU(R_req)`, `P_comp = S*ReLU(R_comp)`, and `P_noise = O-P_req-P_comp`; the residual may be signed and is not a catalog source.
3. **What exact architecture was trained?** Only micro-fit: one four-input-channel compact U-Net with widths 24/48/96/128, two Conv-GroupNorm-SiLU blocks per stage, stride-2 downsamples, bilinear skip decoder, and one six-channel head.
4. **Parameter count?** `1,162,662`.
5. **Was ReLU inside forward?** Yes.
6. **Were requested and companion outputs always nonnegative?** Yes; every evaluated negative fraction was exactly zero.
7. **Was conservation maintained?** Yes; maximum micro closure error was `0.00469970703125`, below the frozen evaluated tolerance `0.78902109375`.
8. **Did objective alignment pass?** Yes.
9. **Was exact truth stationary?** Yes, at zero objective with zero gradient under the frozen subgradient convention.
10. **Did any compromise beat truth?** No.
11. **Did ordinary one-scene micro-overfit pass?** Yes: objective reduction `0.998960`, requested/companion reductions `0.977846/0.963738`, identity `1.0`.
12. **Did difficult one-scene pass?** No: reconstruction reductions passed, but identity was `0.5`.
13. **Did eight-scene pass?** No: reconstruction reductions passed, but identity was `0.5625`.
14. **Did all three seeds complete?** No; the mandatory micro stop prohibited full training.
15. **What checkpoints were selected?** None.
16. **Were selection decisions validation-only?** No selection occurred; the frozen rule was validation-only and no safety/calibration result was accessed.
17. **Were OOF training outputs genuine?** Not generated after the mandatory stop.
18. **Were source groups leak-free?** Yes in all frozen partition and five-fold audits; maximum overlap was zero.
19. **Did deterministic replay pass?** Not run because no eligible checkpoint/output existed.
20. **Did batch consistency pass?** Not run for the same reason.
21. **Episodes labeled per partition?** `0 / 0 / 0`.
22. **Safe/unsafe counts?** Not measured; no labels were constructed.
23. **Safe prevalence?** Not measured.
24. **Did the source-output contract pass 100%?** It passed every physical and micro output evaluated; full-partition prevalence was not measured.
25. **Catastrophic-pass rate?** Not measured.
26. **Joint-safe rate?** Not measured.
27. **Which gate dominated?** Prompt identity/source ordering, not objective reduction, source nonnegativity, finiteness, or conservation.
28. **Were safe examples present in every partition?** Unknown; labels were prohibited.
29. **Did all label-support gates pass?** No; they were not reached after reconstruction failure.
30. **Was Family-E1 distinct from prior families?** Not scientifically evaluated because no frozen family outputs existed.
31. **Authoritative outcome?** `FAMILY_E1_RECONSTRUCTION_FAILURE`.
32. **Is Thayer-Audit v1 authorized?** No.
33. **What happens next?** Exactly one separately preregistered micro-only **Family-E1P Paired-Prompt Identity Intervention** on the same ordinary/difficult/eight scenes: retain the signed physical map and safety boundary, add one explicit paired-prompt source-ordering term, and require the unchanged 0.90 identity gate before any full training.
34. **Were development, Atlas selection, and lockbox untouched?** Yes: `0 / 0 / 0`.
35. **Were historical checkpoints unchanged?** Yes: `743` checked, zero mismatches.
36. **Reusable source/tests to review later?** `src/family_e1.py`, the bootstrap/run/finalize launchers, `tests/test_family_e1.py`, and `tests/test_thayer_family_e1_v0_artifacts.py`.
37. **Generated artifacts to remain ignored?** Both `outputs/runs/thayer_family_e1_v0_20260714_214638/` (failed bootstrap marker) and this entire `outputs/runs/thayer_family_e1_v0_20260714_214715/` tree.

## Evidence inventory

- Physical proof: `physical_contract/mps_physical_preflight.json` and the unchanged authoritative preflight.
- Objective audit: `tables/objective_alignment_audit.csv`, `objective_audit/objective_alignment_summary.json`, and `diagnostics/objective_alignment.md`.
- Architecture: `architecture/architecture_manifest.json`.
- Micro results/curves: `tables/micro_overfit_results.csv`, `micro_overfit/*_trace.csv`, and `figures/micro_overfit_curves.png`.
- Checkpoint/OOF/replay/label/comparison/bootstrap status artifacts explicitly record the mandatory stop and zero downstream payloads.
- Integrity: `tables/integrity_checks_r1.csv`, `tables/checkpoint_inventory_after_r1.csv`, and `diagnostics/final_git_status.txt`.

## Integrity and runtime

- Focused tests: `31 passed, 1 warning in 2.50s`.
- Compileall / CSV / git diff / staged / README / checkpoint audit: PASS.
- One architecture and one in-forward ReLU mapping only; no post-hoc clipping, truth deployment, safety-based selection, CPU fallback, or auditor training.
- Preflight run files unchanged; Condition C and Thayer-PU checkpoints unchanged.
- Runtime through report assembly: `993.9` seconds.
- Run disk usage / filesystem free: `1056768` / `450128904192` bytes.
- Final Git status: `diagnostics/final_git_status.txt`.
