# Thayer-External-Photometry-Scene-Stratification-v0 final report

## Outcome

**SCENE_STRATIFICATION_PRIMARY_RESPONSE_PARTIALLY_RESOLVED**

Total 5%-uncertainty source photometry was scientifically interpretable for **6/8 scenes**. It materially improved over P2 for **3/6 interpretable scenes (50.0%; exact 95% binomial CI 11.8%–88.2%)**: Scenes 0, 51, 73. It did not help Scenes 3, 6, 81. Scenes 5, 18 are unclassified because all four total-photometry starts reached 500 evaluations with no optimizer-declared success.

Per-band photometry was interpretable for **7/8** scenes and helpful for **3/7**. Total and per-band decisions agreed for **6/6** scenes where both were interpretable.

## Scene ranking and S1 → P2 → external response

| Rank | Scene | Status | Helpful | Classes S1→P2→Ext | Classification S1→P2→Ext | P2→Ext condition factor | log10 factor | Mean diameter fraction reduction |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | 73 | resolved | Photometry Helpful | 1→3→1 | NEAR_UNIQUE→NON_IDENTIFIABLE→NEAR_UNIQUE | 0.20283463802728421 | -0.6928578786306809 | 1.0 |
| 2 | 51 | resolved | Photometry Helpful | 2→3→1 | PARTIALLY_IDENTIFIABLE→NON_IDENTIFIABLE→NEAR_UNIQUE | 0.07960047305237909 | -1.0990843513147501 | 1.0 |
| 3 | 0 | resolved | Photometry Helpful | 1→2→1 | NEAR_UNIQUE→NON_IDENTIFIABLE→NEAR_UNIQUE | 0.42504372709717253 | -0.37156638886700283 | 1.0 |
| 4 | 3 | resolved | Photometry Not Helpful | 3→1→1 | NON_IDENTIFIABLE→NEAR_UNIQUE→NEAR_UNIQUE | 0.39321756269430036 | -0.40536709305026686 | 0.0 |
| 5 | 6 | resolved | Photometry Not Helpful | 1→1→1 | NEAR_UNIQUE→NEAR_UNIQUE→NEAR_UNIQUE | 0.12285061316430683 | -0.910622671555 | 0.0 |
| 6 | 81 | resolved | Photometry Not Helpful | 5→1→2 | PARTIALLY_IDENTIFIABLE→NEAR_UNIQUE→PARTIALLY_IDENTIFIABLE | 3.0401681734950663 | 0.48289760821427263 | -1.0 |
| 7 | 5 | unranked_optimizer_unresolved | Optimization Unresolved | 1→2→1 | NEAR_UNIQUE→NON_IDENTIFIABLE→OPTIMIZATION_UNRESOLVED | 0.007774290714780812 | -2.1093392232282335 | -- |
| 8 | 18 | unranked_optimizer_unresolved | Optimization Unresolved | 1→1→1 | NEAR_UNIQUE→NEAR_UNIQUE→OPTIMIZATION_UNRESOLVED | 0.0 | -inf | -- |


Unresolved scenes are placed last but are not scientifically ranked. Exact image, morphology, and flux-allocation diameter transitions for both photometry conditions are in `tables/photometry_response.csv`.

## Feature ranking on interpretable primary scenes

| Rank | Feature | Oriented AUC | Direction | Exact p | BH q | Helpful median | Not-helpful median | ρ with log-condition gain | Tree importance |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bulge_fraction_difference | 1.000 | lower predicts helpful | 0.1000 | 0.8750 | 0 | 0.068356 | -0.029 | 1.000 |
| 2 | log10_total_brightness_electrons | 0.778 | lower predicts helpful | 0.4000 | 0.8750 | 5.2469 | 5.5296 | 0.943 | 0.000 |
| 3 | morphology_similarity | 0.778 | higher predicts helpful | 0.4000 | 0.8750 | 0.99257 | 0.9679 | -0.829 | 0.000 |
| 4 | sersic_parameter_similarity | 0.778 | lower predicts helpful | 0.4000 | 0.8750 | 0.71822 | 0.7618 | -0.543 | 0.000 |
| 5 | s1_endpoint_multiplicity | 0.778 | lower predicts helpful | 0.4000 | 0.8750 | 1 | 3 | 0.395 | 0.000 |
| 6 | effective_radius_ratio | 0.778 | higher predicts helpful | 0.4000 | 0.8750 | 0.81367 | 0.63082 | -0.143 | 0.000 |
| 7 | psf_sensitivity_log10_condition_gain | 0.667 | lower predicts helpful | 0.7000 | 0.8750 | -0.0043405 | 0.014587 | 0.429 | 0.000 |
| 8 | log10_s1_condition_number | 0.667 | lower predicts helpful | 0.7000 | 0.8750 | 2.6567 | 2.7329 | -0.371 | 0.000 |


This ranking uses **n=6** interpretable scenes (3 helpful, 3 not helpful). It is exploratory; no multiplicity-adjusted result is treated as confirmatory.

## Decision tree and simple rules

```text
if bulge_fraction_difference <= 0.00491334:
  predict Photometry Helpful (n=3, helpful=3)
else:
  predict Photometry Not Helpful (n=3, helpful=0)
```

- Resubstitution accuracy/balanced accuracy: **1.000/1.000**.
- Leave-one-out accuracy/balanced accuracy: **0.333/0.333**.
- Best single rule: **Photometry Helpful when bulge_fraction_difference <= 0.00491334**; in-sample TP=3, FP=0, TN=3, FN=0, balanced accuracy=1.000.

These cut points are descriptive acquisition rules, not changes to inherited scientific thresholds.

## When should additional photometric information be acquired?

On the resolved frozen subset, acquire total external source photometry when **`bulge_fraction_difference <= 0.00491334`**. Do **not** use this as an operational rule yet: it was selected from only six interpretable scenes, its leave-one-out balanced accuracy is 0.333, and 2/8 primary responses remain unresolved. Photometry should not be acquired routinely for the complementary resolved stratum, where it produced no material endpoint/diameter gain under the unchanged rule.

The conclusion is conditional on the CatSim/BTK frozen scenes, exact coordinates, known PSF, Level-5 support, and 5% source photometry. The population helpful rate across all eight is not estimable until Scenes 5 and 18 are resolved.

## Exactly one next experiment

Run **Thayer-External-Photometry-Stratification-Convergence-Correction-v0**: repeat only Scene 5 and Scene 18 `TOTAL_SOURCE_PHOTOMETRY` with the same measurements, starts, objective, Model-9 implementation, thresholds, and diagnostics, changing only the per-start evaluation ceiling from 500 to 2000. This targets the sole blocker: all **8/8** primary starts across those scenes capped at 500, with endpoint gradient norms 16.6–27.3 (Scene 5) and 96.2–145.8 (Scene 18).

## Integrity and provenance

- The initial runner failed after all fits, during report arithmetic on an infinite unresolved condition number. This append-only finalizer preserves every fit and treats unresolved conditions fail-closed.
- All six available authoritative reports were read. `Thayer-Project-Synthesis-v1` is absent under that title; the authoritative Flux-Free report records the same absence.
- Scene 0 and 6 fits were reused unchanged. The other six scenes used four deterministic starts and the 500-evaluation ceiling.
- Isolated training-source arrays and catalog morphology were used only for post-fit scene descriptors, never as solver inputs.
- Integrity: **PASS**. HEAD, README, authoritative report hashes, and the empty staged index were unchanged.
- Development, Atlas, and lockbox access: zero. Neural training: zero. Nothing staged or committed.
- Finalization runtime: 1.167 seconds. Scientific optimizer runtime is retained per fit record.
