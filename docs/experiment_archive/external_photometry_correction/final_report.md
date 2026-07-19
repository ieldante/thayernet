# Thayer-External-Photometry-Convergence-Correction-v0 final report

## Outcome

**EXTERNAL_PHOTOMETRY_TARGETED_CAMPAIGN_JUSTIFIED**

The only authorized scientific-execution change was the per-start maximum function-evaluation budget: **150 -> 500**. Scenes, Level-5 bulge+disk structure, the two photometry conditions, exact noisy measurements, 5% uncertainties, four deterministic physical starts, optimizer, objective, parameterization, supports, PSF, observation, tolerances, gradient diagnostic, endpoint acceptance/clustering, ranks, diameters, classification logic, and replay procedure were unchanged. Measurements were reused from the frozen preflight table and were not regenerated.

## Four-fit convergence result

| Scene | Condition | Fit classification | Successful starts | 500-ceiling starts | Best nfev | Best total objective | Gradient norm | Classes | Replay |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | TOTAL_SOURCE_PHOTOMETRY | CONVERGED_MODERATE_PROMISE | 4/4 | 0/4 | 42 | 67026.0173657 | 9.78734990799 | 1 | exact |
| 0 | PER_BAND_SOURCE_PHOTOMETRY | CONVERGED_MODERATE_PROMISE | 2/4 | 2/4 | 227 | 67056.928454 | 7.96807216727 | 1 | exact |
| 6 | TOTAL_SOURCE_PHOTOMETRY | CONVERGED_NO_CLEAR_GAIN | 4/4 | 0/4 | 94 | 67106.3718394 | 22.1290387123 | 1 | exact |
| 6 | PER_BAND_SOURCE_PHOTOMETRY | CONVERGED_NO_CLEAR_GAIN | 4/4 | 0/4 | 36 | 67134.8204297 | 52.0262347295 | 1 | exact |


All four fits were converged and interpretable under the frozen preflight optimization-limitation rule: **yes**. Scene 0's one-class result survived at 500 evaluations: **yes**. Scene 6 gained materially over P2: **no**.

Per-band photometry materially exceeded total photometry: **Scene 0 no; Scene 6 no** under the same endpoint-class/diameter improvement rule. Photometry materially exceeded PSF diversity: **Scene 0 TOTAL_SOURCE_PHOTOMETRY=yes; Scene 0 PER_BAND_SOURCE_PHOTOMETRY=yes; Scene 6 TOTAL_SOURCE_PHOTOMETRY=no; Scene 6 PER_BAND_SOURCE_PHOTOMETRY=no**. This evidence authorizes **a targeted scene-stratified follow-up**.

The direct bounded solver has no unconstrained initialization variables. Every physical starting vector and initialization hash matched the preflight exactly; no preflight endpoint or truth parameter was used as an initialization.

## Preflight comparison

Objective, gradient, endpoint-class, diameter, ceiling-resolution, and classification changes are recorded fit-by-fit in `tables/comparison_to_preflight.csv`. All objective components, local rank/nullity/condition diagnostics, boundary contacts, fitted g/r/z source fluxes, and exact replay hashes are retained in the summary, endpoint table, fitted-flux table, and four atomic fit records.

## Max-budget and optimizer-declared failures

- Scene 0 / PER_BAND_SOURCE_PHOTOMETRY / start 1: success=False, status=0, nfev=500, message=`The maximum number of function evaluations is exceeded.`
- Scene 0 / PER_BAND_SOURCE_PHOTOMETRY / start 2: success=False, status=0, nfev=500, message=`The maximum number of function evaluations is exceeded.`

## Integrity and runtime

Authorization, oracle-information, and final integrity gates passed: **yes**. All 600 historical checkpoints matched before and after; README and HEAD were unchanged; historical reports and preflight artifacts were unchanged; protected development, Atlas-tensor, and lockbox access were zero; the Git index remained empty; nothing was staged or committed.

Exact campaign runtime: **854.857279 seconds** (14.248 minutes), from 2026-07-19T00:56:38.345517+00:00 to 2026-07-19T01:10:53.205404+00:00. The preregistered linear estimate was 1237.082 seconds, with no wall-clock cutoff.

## Exactly one next experiment

**Thayer-External-Photometry-Scene-Stratification-v0** — Run one targeted scene-stratified photometry experiment to isolate why Scene 0 benefits while Scene 6 does not.
