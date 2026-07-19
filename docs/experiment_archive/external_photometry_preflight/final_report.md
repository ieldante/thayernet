# Thayer-External-Photometry-Preflight-v0 final report

## Outcome

**PREFLIGHT_OPTIMIZATION_LIMITED**

This was a two-scene, reduced-budget preflight, not a population-level campaign. It used only Scenes 0 and 6, only Level 5 bulge+disk, exactly four deterministic starts, and at most 150 function evaluations per start. Authoritative S1 and P2 results were reused without rerunning them.

## External information supplied

`TOTAL_SOURCE_PHOTOMETRY` supplied one noisy per-source scalar equal to the frozen `g+r+z` detected-electron combination. `PER_BAND_SOURCE_PHOTOMETRY` supplied noisy per-source g, r, and z measurements. Every supplied measurement used 5% relative Gaussian uncertainty and frozen deterministic seed 2026071805. Measurements entered as an explicit likelihood; no source flux was fixed to truth. Measured values and uncertainties are in `tables/external_photometry_measurements.csv`.

## Scene-level evidence

| Scene | Total photometry | Per-band photometry | S1 classes | P2 classes | Total classes | Per-band classes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | MODERATE_PROMISE | OPTIMIZATION_LIMITED | 1 | 2 | 1 | 1 |
| 6 | NO_CLEAR_GAIN | NO_CLEAR_GAIN | 1 | 1 | 1 | 1 |


Per-band photometry reduced multiplicity more than P2 on both scenes: **no** under the frozen endpoint-class rule. Total broadband photometry was sufficient to show preflight promise: **yes**. The exact campaign-level decision is `PREFLIGHT_OPTIMIZATION_LIMITED`; therefore the full eight-scene campaign is **not justified by this preflight**.

The literal `STRONG_PROMISE` rule is demanding because both S1 baselines already have one class and zero reported diameters; equality at zero is not counted as improvement. This preflight does not relabel any result `UNIQUE` and makes no population claim.

## Optimization limitations

Optimizer-declared successful endpoints by fit were: Scene 0 TOTAL_SOURCE_PHOTOMETRY=3/4, Scene 0 PER_BAND_SOURCE_PHOTOMETRY=0/4, Scene 6 TOTAL_SOURCE_PHOTOMETRY=4/4, Scene 6 PER_BAND_SOURCE_PHOTOMETRY=4/4. Starts hitting the 150-evaluation ceiling: 5/16. Every endpoint was retained, and each best endpoint received an exact deterministic replay. The reduced start/evaluation budget limits basin discovery relative to the authoritative campaigns.

## Integrity and runtime

No isolated-source image, morphology truth, mask, or truth initialization entered inference. Catalog photometry was used only to generate noisy external measurements and was discarded before fitting. Protected development, Atlas-tensor, and lockbox accesses were zero. Nothing was staged or committed; README and HEAD were unchanged subject to the final integrity manifest.

Exact runtime: **562.870292 seconds** (9.381 minutes), from 2026-07-18T19:48:52.362682+00:00 to 2026-07-18T19:58:15.236739+00:00.

## Exactly one next experiment

**Thayer-External-Photometry-Convergence-Correction-v0: repeat only these four fits with the authoritative 500-evaluation budget before any eight-scene expansion.**
