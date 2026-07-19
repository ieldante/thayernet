# Thayer-External-Photometry-Scene-Stratification-v0 frozen protocol

Frozen before generating external measurements or running any scientific fit for Scenes 3, 5, 18, 51, 73, or 81.

## Scientific scope

- Population: all eight frozen Family-E1 training scenes, indices `0, 3, 5, 6, 18, 51, 73, 81`.
- Structural family: frozen Model-9 Level 5 bulge+disk only.
- Historical conditions: authoritative S1 `FLUX_FREE_SINGLE` and P2 `PSF_DIVERSE`, reused without rerunning.
- External conditions: `TOTAL_SOURCE_PHOTOMETRY` and `PER_BAND_SOURCE_PHOTOMETRY` on the unchanged single observation.
- Scene 0 and Scene 6 external fits and measurements are reused byte-for-byte from the authoritative convergence-correction run. They are not rerun or regenerated.
- The other six scenes use exactly four deterministic physical Model-9 starts, the frozen bounded-TRF optimizer, float64 CPU execution, and at most 500 function evaluations per start. Fits execute sequentially and every endpoint is retained.
- Renderer, parameterization, morphology support, PSF, observation, noise convention, objective, tolerances, gradient diagnostic, rank/nullity analysis, endpoint acceptance and clustering, diameter definitions, classification logic, and replay logic are unchanged.

## External information

- Total photometry is one noisy `g+r+z` detected-electron scalar per source.
- Per-band photometry is one noisy g, r, and z detected-electron value per source.
- Both use independent Gaussian uncertainty equal to 5% of the latent catalog value and the existing deterministic seed rule `2026071805 + 100*scene_index + condition_index`.
- Catalog photometry is used only by the deterministic measurement generator. Only noisy measurements and declared uncertainties enter fitting.
- No isolated-source image, morphology truth, catalog morphology parameter, mask, or truth initialization enters a fit.

## Primary response and unchanged scientific rules

- `TOTAL_SOURCE_PHOTOMETRY` is the primary acquisition response because the corrected predecessor found no material per-band advantage in either tested scene. Per-band response is a prespecified sensitivity analysis.
- A condition is `Photometry Helpful` exactly when it materially improves over P2 using the predecessor's frozen rule: fewer endpoint classes; or, with no diameter worsening, at least 50% reduction of every nonzero P2 scientific diameter. Otherwise it is `Photometry Not Helpful`.
- Endpoint reduction is `P2 classes - photometry classes`.
- Condition improvement is reported as both `P2 condition / photometry condition` and `log10(P2 condition / photometry condition)`; values above one or above zero respectively are improvements.
- Image diameter is the maximum of requested and companion image diameter. Image, morphology, and flux-allocation diameter reductions are reported in absolute units and fractionally when the P2 denominator is nonzero.
- Scientific classifications use the existing ordered states `NON_IDENTIFIABLE < PARTIALLY_IDENTIFIABLE < NEAR_UNIQUE < UNIQUE` and the frozen Model-9 thresholds. No previous threshold is modified.

## Scene properties

The following fifteen prespecified predictors are computed for every scene:

1. overlap fraction, using the authoritative Recoverability definition;
2. centroid separation in mean-PSF FWHM units;
3. symmetric source flux ratio from rendered g/r/z totals;
4. log10 total g+r+z scene brightness in detected electrons;
5. morphology similarity: cosine of per-band flux-normalized isolated templates after centroid alignment;
6. color similarity: cosine of the two rendered g/r/z flux vectors;
7. Sérsic-parameter similarity: one minus the mean frozen-range-normalized distance in effective Sérsic index, log effective radius, axis ratio, and 180-degree-periodic position angle;
8. absolute catalog bulge-fraction difference;
9. symmetric effective-radius ratio;
10. PSF sensitivity: `log10(S2 Level-5 condition / P2 Level-5 condition)`, isolating PSF diversity from repeated exposure;
11. log10 S1 Level-5 condition number;
12. S1 Level-5 endpoint multiplicity;
13. S1 Level-5 image diameter;
14. S1 Level-5 morphology diameter;
15. S1 Level-5 flux-allocation diameter.

Effective catalog morphology uses `B/T = fluxnorm_bulge/(fluxnorm_bulge+fluxnorm_disk)`, `n_eff = 1+3 B/T`, and flux-fraction-weighted component radius, axis ratio, and circular position angle. The radius is `sqrt(a*b)` for each active catalog component. These aggregate truth-derived scene descriptors are used only after inference to explain response.

## Transparent statistical analysis

- Sample size is fixed at eight; all inference is exploratory and reports this limitation.
- Feature ranking uses orientation-free univariate ROC AUC for the primary binary response, exact label-permutation p-values preserving the observed class count, Benjamini-Hochberg q-values, helpful-minus-not-helpful median effects, and Spearman correlation with the continuous log-condition response.
- The decision tree is implemented transparently with exhaustive midpoint splits, Gini impurity, maximum depth two, and no hidden model or learned representation. Resubstitution accuracy, balanced accuracy, leave-one-out predictions, and leave-one-out balanced accuracy are reported.
- Simple rules include the best one-split rule and all terminal tree paths. Predictor cut points are descriptive acquisition rules, not changes to any inherited scientific gate.
- Scene ranking is lexicographic: primary helpful status, endpoint reduction, classification improvement, log-condition improvement, and mean finite P2-to-photometry diameter reduction.
- The correlation matrix is Spearman rank correlation across the fifteen predictors plus primary response metrics.

## Integrity

- No neural network is trained or loaded.
- Model-9, PriorNet, prior experiments, thresholds, reports, checkpoints, and README are not modified.
- Development, Atlas, and lockbox are not accessed.
- Outputs are append-only inside this fresh run directory.
- Nothing is staged or committed.
- The requested `Thayer-Project-Synthesis-v1` artifact is absent under that title; the authoritative Flux-Free report explicitly records that absence. This campaign uses the available authoritative reports and records the missing-source limitation.
