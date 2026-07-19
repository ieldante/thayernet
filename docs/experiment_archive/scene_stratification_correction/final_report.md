# Thayer-External-Photometry-Stratification-Convergence-Correction-v0 final report

## Outcome

**STRATIFICATION_DATASET_COMPLETE**

The only scientific-execution change was **`max_nfev: 500 -> 2000`**. The exact predecessor measurements, 5% uncertainties, four direct-bounded physical starts, optimizer, parameterization, Level-5 bulge+disk family, observation and photometry likelihoods, objective, PSF, coordinates, morphology support, all other tolerances, gradient gate, endpoint acceptance and clustering, rank/nullity and condition rules, scientific diameters, replay, and classification logic were unchanged. No resolved scene was rerun.

- Scene 5: resolved; classification `HELPFUL`; response `Photometry Helpful`.
- Scene 18: resolved; classification `NOT_HELPFUL`; response `Photometry Not Helpful`.

Campaign primary answer: **2/2 previously unresolved scenes became scientifically interpretable.**

## Fit diagnostics

| Scene | Scene label | Fit class | Successful starts | 2000-ceiling starts | Best nfev | Total objective | Observation objective | Photometry objective | Gradient | Rank | Nullity | Condition | Classes | Requested diameter | Companion diameter | Morphology diameter | Flux-allocation diameter | Replay | Perturbation |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 5 | HELPFUL | NEAR_UNIQUE | 1/4 | 3/4 | 2000 | 67067.3016504 | 67053.6672567 | 13.6343937218 | 19.0109919777 | 24 | 0 | 55825.11825341288 | 1 | 0 | 0 | 0 | 0 | exact | stable |
| 18 | NOT_HELPFUL | NON_IDENTIFIABLE | 4/4 | 0/4 | 1822 | 67223.3073213 | 67209.4086288 | 13.8986925373 | 146.008934652 | 22 | 0 | 24432279.649236135 | 2 | 4.54057e-05 | 9.1317e-05 | 0.924684 | 2.75143e-06 | exact | stable |

Fitted source g/r/z fluxes and complete boundary-contact structures are in `tables/fitted_source_fluxes.csv`, `tables/convergence_correction_summary.csv`, and the atomic fit records. Every residual-function objective evaluation is retained in each fit record.

## Comparison with the 500-evaluation result

| Scene | Objective reduction | Gradient reduction | Endpoint classes | Starts now converged | Classification change |
| ---: | ---: | ---: | --- | ---: | --- |
| 5 | 0.0836993379489 | 0.183932440004 | 1→1 | 1 | OPTIMIZATION_UNRESOLVED -> NEAR_UNIQUE |
| 18 | 0.00568947357533 | -9.45522085523 | 1→2 | 4 | OPTIMIZATION_UNRESOLVED -> NON_IDENTIFIABLE |

All requested scientific-diameter changes and direct P2 comparisons are in `tables/comparison_to_500eval.csv` and `tables/comparison_to_p2.csv`.

## Corrected stratification

The descriptive helpful rate changed from 3/6 (50.0%) to 4/8 (50.0%; exact 95% CI 15.7%–84.3%).

The bulge-fraction hypothesis **strengthened**. Its corrected oriented AUC is 1.000, exact p=0.0286, BH q=0.4286; corrected leave-one-out balanced accuracy is 0.750. It remains exploratory and is not a validated predictor.

Corrected outputs, when authorized by at least one resolved scene, are separately labeled and do not overwrite the frozen six-scene analysis. No validated predictor is claimed from at most eight scenes.

## Optimizer failures

- Scene 5 / start 1: success=False, status=0, nfev=2000, message=`The maximum number of function evaluations is exceeded.`
- Scene 5 / start 2: success=False, status=0, nfev=2000, message=`The maximum number of function evaluations is exceeded.`
- Scene 5 / start 3: success=False, status=0, nfev=2000, message=`The maximum number of function evaluations is exceeded.`

## Starts at the 2000-evaluation ceiling

- Scene 5 / start 1: nfev=2000.
- Scene 5 / start 2: nfev=2000.
- Scene 5 / start 3: nfev=2000.

## Integrity and runtime

Authorization and final integrity status: **PASS**. All 600 historical checkpoints matched before and after; README and HEAD were unchanged; predecessor and historical reports were unchanged; protected development, Atlas-tensor, lockbox, and isolated-source access were zero; the Git index remained empty; nothing was staged or committed.

Exact campaign runtime: **4357.357412 seconds** (72.622624 minutes), from `2026-07-19T07:21:02.957622+00:00` to `2026-07-19T08:33:40.351099+00:00`.

## Exactly one next experiment

**Thayer-External-Photometry-Stratification-Independent-Scene-Validation-v0** — Run a preregistered independent-scene validation of the exploratory low-|ΔB/T| candidate stratification rule.
