# Thayer-Select Phase II root-cause analysis

Date: 2026-07-11  
Scope: frozen Promptability, Recoverability, and three-seed Recoverability outputs only  
Decision: **Phase II failed because its recoverability target was too sparse and heterogeneous for the shared head, and isotonic calibration then converted the already-wrong raw ordering into unusable threshold plateaus.**

## Executive Summary

Thayer-Select does not rank ambiguous scenes above valid scenes because isotonic calibration reversed them. The inversion already exists in the raw neural score and is almost identical in all three seeds: mean raw ambiguous-minus-valid score was **+0.1381, +0.1473, and +0.1394**. Isotonic calibration is monotone, so it cannot create that inversion. It does, however, destroy operating-point resolution: 2,000 unique raw calibration scores became only **11, 7, and 7** calibrated values, with **40.9%, 40.2%, and 63.2%** of calibration samples mapped to exactly zero. The requested 80%-coverage threshold is the lower 20th percentile; because more than 20% of every calibration set is zero, every threshold became zero. Applying `score >= 0` accepts all 2,000 development scenes.

The deeper cause is the learning target. The selected global actionable label is positive only for a valid or perturbed-valid reconstruction passing the permissive oracle contract. Null and ambiguous queries are always global negatives. Among the 11,500 training and validation scenes, only **359 (3.12%)** were positive; even among the 8,050 source queries, only **4.46%** were positive. This combines several very different negative mechanisms—low-SNR reconstruction failure, source confusion, null requests, and semantic ambiguity—into one overwhelmingly negative class. The teacher labels themselves are outcome-grounded rather than generator-defined: max reconstruction flux error predicts them with AUROC **0.990**, versus **0.898** for the strongest generator variable, SNR. But the deployed head cannot observe the oracle flux/color/centroid errors at inference, so it learns generator-like shortcuts, especially SNR and brightness imbalance, instead of the full conjunction that defines success.

The representation is not missing all relevant information. A five-fold, class-balanced linear probe on the exact frozen pooled encoder features separated successful from catastrophic source reconstructions with AUROC **0.968** and balanced accuracy **0.913**, compared with AUROC **0.919** for the shipped raw head on the same subset. A four-group probe (successful, catastrophic, ambiguous, null) reached macro AUROC **0.910**. Ambiguity alone was only moderately separable from source queries (AUROC **0.711**), so the encoder knows some—but not all—of the required relation. This points to label/head/shared-objective conflict rather than complete representation collapse.

The physical failure audit also rejects the proposed “smooth fused galaxy” explanation. Catastrophic source scenes were **less** radially smooth, less concentrated, more asymmetric, and higher in edge/high-frequency structure than successful scenes. Simple integrated color gaps were near chance, but cross-band centroid shift was strongly larger in failures (median **1.50 px** versus **0.59 px**, AUROC **0.727**), indicating potentially useful spatial chromatic information that the current global score does not fully exploit.

Overall confidence in this diagnosis: **high (90%)**. Confidence is highest for the label sparsity and calibration mechanisms, and moderate for the precise division between head optimization and shared-latent conflict.

## Top 5 findings

1. **The raw head, not the calibrator, creates the ambiguous-over-valid inversion.** The raw gap is stable at approximately +0.14 across all three seeds.
2. **The global label is scientifically defined but statistically ill-conditioned.** Only 3.12% of training/validation scenes are actionable positives, while null and ambiguous examples are forced into the same negative class as failed source reconstructions.
3. **Isotonic calibration destroys threshold resolution.** It compresses 2,000 distinct calibration scores to 7–11 values and puts 40–63% of samples on the zero plateau, making every 80%-coverage threshold zero and every resulting development decision “accept.”
4. **The encoder contains substantially more failure information than the shipped head uses.** The frozen-feature source-outcome probe reaches AUROC 0.968 versus 0.919 for the existing head; the four-group probe reaches macro AUROC 0.910.
5. **Catastrophic failures are primarily low-SNR, obstructed, nearly equal-source problems—not unusually smooth or unusually color-similar scenes.** SNR is the dominant univariate predictor across seeds (mean AUROC 0.917). Integrated color differences are near chance, while spatial chromatic centroid shift is informative.

## 1. Label Audit

The table below uses the original frozen R1 development evaluation. “Loss” is the unweighted per-sample binary cross-entropy of the raw recoverability head against the permissive actionable label. The full composite Phase-II training loss cannot be reconstructed per original-development sample because the original full reconstruction and uncertainty maps were intentionally not persisted, and development inference was not regenerated.

| Query class | Samples | Positive | Negative | Success rate | Catastrophic rate | Mean normalized reconstruction error | Mean raw recoverability | Mean calibrated probability | Mean uncertainty | Mean recoverability BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VALID | 900 | 65 | 835 | 7.22% | 73.33% | 0.9887 | 0.1275 | 0.0614 | 64.03 | 0.4104 |
| NULL | 400 | 0 | 400 | 0.00% | 0.00% | 0.0687 | 0.0024 | 0.0049 | 60.44 | 0.0027 |
| AMBIGUOUS | 400 | 0 | 400 | 0.00% | 35.75% | 0.1168 | **0.2656** | **0.1017** | 62.27 | **0.6764** |
| PERTURBED | 300 | 23 | 277 | 7.67% | 75.33% | 1.2884 | 0.1159 | 0.0596 | 70.85 | 0.4039 |

The ambiguous class has no positive labels by construction yet receives the highest mean raw probability and the highest classification loss. Null is learned well because it has a dedicated no-source task and extremely distinctive prompt geometry. Ambiguous requests lack a dedicated ambiguity target; they compete only through the sparse global actionable BCE while sharing the pooled latent with reconstruction, uncertainty, and no-source objectives.

### Could the labels themselves cause the failure?

**Yes, as a learning target—not because the oracle outcomes were mislabeled.** The target has three problems:

- **Extreme sparsity:** 359/11,500 total training/validation positives (3.12%); 359/8,050 positives among eligible source queries (4.46%).
- **Heterogeneous negatives:** ambiguous, null, low-SNR catastrophic, confused, and merely contract-missing scenes all share label zero.
- **Inference mismatch:** the label is a conjunction of reconstruction flux, color, centroid, confusion, and catastrophic criteria, while the head sees only pooled encoder appearance features before the outcome is known.

This encourages the head to learn shortcuts correlated with rare success—most strongly SNR—rather than semantic ambiguity or realized reconstruction validity.

![Label audit](outputs/runs/thayer_select_root_cause_analysis_20260711/figures/label_audit.png)

## 2. Calibration Audit

### Calibration-set collapse

| Seed | Positive rate | Unique raw | Unique calibrated | Tie compression | Largest plateau | Zero plateau | Raw AUROC | Calibrated AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 original | 5.50% | 2,000 | 11 | 99.45% | 818 (40.9%) | 40.9% | 0.861 | 0.875 |
| R1 seed 2 | 1.70% | 2,000 | 7 | 99.65% | 803 (40.2%) | 40.2% | 0.876 | 0.908 |
| R1 seed 3 | 4.70% | 2,000 | 7 | 99.65% | 1,265 (63.2%) | 63.2% | 0.888 | 0.898 |

“Tie compression” is `1 - unique calibrated values / N`. Isotonic maps each monotone block to its empirical positive frequency. With only 34–110 positives per 2,000-scene calibration set and long lower-score regions containing no positives, the pool-adjacent-violators solution assigns those regions exactly zero. The in-sample ECE of approximately zero is therefore mechanical: each fitted plateau equals its own empirical frequency. It is not evidence that threshold selection will have adequate resolution.

### Why every 80% threshold is zero

The frozen code defines an 80%-coverage threshold as the lower empirical 20th percentile of calibrated scores. Every zero plateau contains more than 20% of the calibration set, so the 20th percentile is exactly zero. The selection rule is inclusive (`score >= threshold`), hence every zero-score sample is accepted. All three nominal 80%-coverage thresholds therefore realize **100% development coverage (2,000/2,000)**.

### Did isotonic destroy ranking?

**It destroyed fine ranking and attainable coverage, but it did not create the semantic inversion.** Raw-versus-calibrated Spearman correlation remains 0.862–0.961, and AUROC is usually similar because isotonic is monotone. However, 2,000 raw ranks become only 7–11 plateaus; seed 3 development AUROC even falls from 0.896 raw to 0.886 calibrated. Most importantly, ambiguous raw means already exceed valid raw means in all seeds:

| Seed | Mean raw valid | Mean raw ambiguous | Ambiguous minus valid |
|---|---:|---:|---:|
| R1 original | 0.1275 | 0.2656 | +0.1381 |
| R1 seed 2 | 0.1298 | 0.2771 | +0.1473 |
| R1 seed 3 | 0.1272 | 0.2666 | +0.1394 |

![Calibration collapse](outputs/runs/thayer_select_root_cause_analysis_20260711/figures/calibration_collapse.png)

## 3. Catastrophic-Failure Correlation

This audit is restricted to valid and perturbed-valid source queries, where all requested variables have scientific meaning. Predictive power is directionless univariate AUROC for catastrophic failure; no model was trained. Flux and size ratios are expressed as absolute log-ratio imbalance. Values below summarize the mean and range over the three frozen R1 seeds.

| Rank | Variable | Mean AUROC | Seed range | Failure direction |
|---:|---|---:|---:|---|
| 1 | SNR | **0.917** | 0.905–0.931 | lower SNR |
| 2 | Core obstruction | **0.711** | 0.699–0.729 | more obstruction |
| 3 | Flux ratio | **0.707** | 0.644–0.784 | more equal-flux sources |
| 4 | Morphology difference | 0.630 | 0.587–0.680 | more similar catalog B/T |
| 5 | Image overlap | 0.617 | 0.530–0.695 | inconsistent; usually lower overlap |
| 6 | Separation | 0.535 | 0.526–0.543 | inconsistent / near chance |
| 7 | Separation / PSF | 0.535 | 0.526–0.543 | inconsistent / near chance |
| 8 | Size ratio | 0.524 | 0.509–0.548 | near chance |
| 9 | Integrated color difference | 0.520 | 0.506–0.528 | near chance |

For the original seed, SNR AUROC is 0.931 with a stratified bootstrap 95% interval of 0.914–0.946. Separation and separation/PSF are not material predictors once this audit is restricted to source requests. This means “the galaxies are simply too close” is not an adequate explanation of catastrophic source reconstruction. Low visibility, core interference, and roughly equal source strength are much more important.

![Failure predictor ranking](outputs/runs/thayer_select_root_cause_analysis_20260711/figures/failure_predictive_power.png)

## 4. Color Analysis

The comparison uses 88 permissive-successful versus 886 catastrophic original-seed source reconstructions.

| Variable | Successful median | Catastrophic median | AUROC | Two-sided Mann–Whitney p | Result |
|---|---:|---:|---:|---:|---|
| `|delta(g-r)|` | 0.376 mag | 0.451 mag | 0.531 | 0.344 | no evidence of a difference |
| `|delta(r-z)|` | 0.477 mag | 0.514 mag | 0.516 | 0.620 | no evidence of a difference |
| Cross-band centroid shift | 0.587 px | **1.502 px** | **0.727** | **1.83e-12** | much larger in failures |

The simple “larger color gap causes failure” hypothesis is unsupported. Integrated color differences are near chance across seeds. However, spatial color structure is informative: failure scenes have substantially larger g/r/z centroid displacement in the frozen noiseless combined-source images. Color is therefore not absent from the data, but the potentially useful signal is **where band-dependent light is located**, not merely the global `g-r` or `r-z` difference.

## 5. Morphology Analysis

Metrics were measured on the frozen noiseless sum of the two isolated sources, with bands flux-normalized before combination. This avoids letting pixel noise trivially turn the SNR result into an apparent morphology result. It characterizes the scene morphology, not the morphology of the model prediction.

| Metric | Successful median | Catastrophic median | Failure direction | AUROC | p |
|---|---:|---:|---|---:|---:|
| Radial smoothness | **0.663** | 0.295 | lower / less smooth | 0.745 | 3.20e-14 |
| Concentration | **2.583** | 2.188 | lower | 0.729 | 1.35e-12 |
| High-frequency energy | 0.0104 | **0.0153** | higher | 0.687 | 6.90e-09 |
| Edge density | 0.529 | **0.586** | higher | 0.659 | 9.06e-07 |
| Asymmetry | 0.200 | **0.321** | higher | 0.635 | 2.91e-05 |

**Failures are not smoother.** They are intrinsically less radially smooth, less concentrated, more asymmetric, and richer in edges/high-frequency structure. The hypothesis that Thayer-Select mainly mistakes a fused pair for one smooth galaxy is rejected for catastrophic source scenes. A more defensible interpretation is that the reconstruction and score struggle with low-SNR, spatially complex, chromatically shifted scenes.

![Color and morphology audit](outputs/runs/thayer_select_root_cause_analysis_20260711/figures/color_morphology_audit.png)

## 6. Frozen Latent Investigation

The original best R1 checkpoint was loaded unchanged. Only `enc1`, `enc2`, `bottleneck`, and global average pooling were executed on the frozen development tensors using MPS. The decoder, reconstruction head, variance head, recoverability head, and no-source head were not executed; no development reconstruction or checkpoint was generated. Five-fold linear probes were class-balanced within each training fold and evaluated out of fold.

| Probe | Samples | Balanced accuracy | Macro AUROC | Existing raw-head AUROC |
|---|---:|---:|---:|---:|
| Successful vs catastrophic source | 974 | **0.913** | **0.968** | 0.919 |
| Four groups: success/catastrophic/ambiguous/null | 1,774 | 0.751 | **0.910** | — |
| Ambiguous vs all source queries | 1,600 | 0.673 | 0.711 | — |
| Null vs all source queries | 1,600 | 0.995 | 0.997 | — |

The four-group per-class AUROCs were 0.937 successful, 0.911 catastrophic, 0.794 ambiguous, and 0.999 null. Ambiguous recall was only 0.473 in the four-way probe, so ambiguity is not perfectly represented. Nevertheless, the representation contains enough information to separate source outcomes and the four broad groups far better than chance.

The 64-dimensional pooled latent has an effective covariance rank of only **2.07**, although no dimension is constant. This is strong compression, but not complete collapse: the remaining axes preserve highly predictive source-outcome and null information. The evidence therefore supports **“the encoder partly knows; the trained head/objective underuses it”** more strongly than “the representation contains no information.”

![Latent probes](outputs/runs/thayer_select_root_cause_analysis_20260711/figures/latent_probe.png)

## 7. Teacher Audit

This audit uses the frozen Phase-I teacher labels on all training and validation source queries. It asks how well each scalar, by itself, predicts the permissive actionable label.

| Predictor | Family | AUROC | Direction for success |
|---|---|---:|---|
| Max flux error | reconstruction outcome | **0.990** | lower |
| Normalized RMSE | reconstruction outcome | **0.935** | lower |
| Max color error | reconstruction outcome | **0.900** | lower |
| SNR | generator | **0.898** | higher |
| Centroid error | reconstruction outcome | 0.869 | lower |
| Flux-ratio imbalance | generator | 0.697 | higher |
| Core obstruction | generator | 0.676 | lower |
| Separation / PSF | generator | 0.535 | weak |
| Source color distance | generator | 0.534 | weak |
| Size ratio | generator | 0.533 | weak |
| Prompt offset | generator | 0.506 | chance |

### Which is stronger?

**Actual reconstruction quality is stronger.** The best reconstruction metric reaches AUROC 0.990, compared with 0.898 for the best generator variable. This is expected because the label is explicitly defined from the reconstruction outcome. Generator difficulty can still predict the label well—especially SNR—because it predicts whether the teacher will meet those thresholds.

The operational problem is not that generator parameters were used to create the label; they were not. The problem is that at inference time the recoverability head must predict a rare, multi-metric realized-outcome label from pre-outcome appearance features. SNR is the easiest available proxy and becomes a shortcut. That shortcut does not encode “this prompt is equidistant from two sources,” so bright or otherwise favorable ambiguous scenes can be ranked above valid but difficult scenes.

![Teacher audit](outputs/runs/thayer_select_root_cause_analysis_20260711/figures/teacher_audit.png)

## 8. Root-Cause Ranking

Confidence values are confidence that the factor materially contributes to the observed Phase-II failure; they are not probabilities required to sum to one.

| Rank | Candidate | Confidence | Evidence-based judgment |
|---:|---|---:|---|
| 1 | **Bad labels / target design** | **90%** | The labels are not factually wrong, but the global target is only 3.12% positive and merges four different negative mechanisms. It asks a pre-outcome head to predict a rare conjunction of oracle outcome metrics. |
| 2 | **Shared latent conflict / head underuse** | **85%** | The same pooled bottleneck serves reconstruction, bounded uncertainty, recoverability, and no-source tasks. A balanced linear probe on that exact representation outperforms the shipped head and separates all four groups with macro AUROC 0.910. |
| 3 | **Isotonic calibration** | **80% overall; 99% for threshold collapse** | It does not create the raw inversion, but it compresses 2,000 ranks to 7–11 levels and directly causes zero 80%-coverage thresholds and 100% acceptance. |
| 4 | **Optimization instability** | **70%** | Reconstruction failure and calibrated positive rates vary substantially by seed, so Phase II is unstable. However, the raw ambiguous-valid gap is nearly identical across seeds, making instability secondary for the systematic inversion. |
| 5 | **Missing color physics** | **60%** | Integrated color gaps are near chance, but cross-band centroid shifts are strongly predictive. Spatial chromatic information exists and is underexploited, but it is not the dominant scalar correlate. |
| 6 | **Representation collapse** | **45%** | Effective pooled-latent rank is only 2.07/64, which is concerning, but high probe AUROCs show that decisive information survives. Complete representational failure is contradicted. |
| 7 | **Uncertainty head** | **40%** | Pixel uncertainty is only a moderate and inconsistent failure discriminator and may add shared-gradient pressure, but the frozen evidence does not isolate it as the cause of the semantic score inversion. |
| 8 | **True physical ambiguity** | **20%** | It explains why ambiguous queries should abstain, not why the model confidently ranks them above valid queries. Separation itself is near chance for catastrophic source failure, and the encoder can partially identify ambiguity. |

## 9. One Recommended Next Experiment

### Frozen-backbone, class-balanced head-only ranking experiment

Freeze the original R1 encoder, decoder, reconstruction head, variance head, and no-source head. Using only the already frozen **training** teacher outcomes, fit one new linear recoverability head on the pooled encoder features with class-balanced sampling across: actionable source successes, catastrophic source failures, ambiguous queries, and null queries. Optimize a single pairwise ranking objective that requires actionable source successes to rank above each negative stratum. Select once on frozen validation and report raw ordering once on calibration; do not apply isotonic calibration in this experiment and do not use development or lockbox data for selection.

This is one causal experiment: it changes only the recoverability head and its imbalance handling. Reconstruction remains bit-for-bit unchanged by construction. It directly tests whether the Phase-II failure is label/head conditioning rather than missing encoder information.

### Expected impact

If the diagnosis is correct, the raw ambiguous-minus-valid gap should reverse from approximately **+0.14** to below zero, and source success-versus-catastrophic AUROC should move toward the frozen-feature probe ceiling of **0.968** from the existing **0.919**. A realistic expected gain is **0.03–0.05 AUROC in raw selective ranking**, not improved reconstruction quality. The experiment should also restore enough score resolution to make later coverage calibration meaningful, although calibration itself is outside this experiment.

### Confidence level

**High-moderate (80%)** that this is the highest-value next experiment. It is narrowly targeted, uses evidence already present in the frozen representation, and cleanly falsifies the leading explanation without changing the scientific reconstructor.

## Supporting artifacts and reproducibility

- Standalone analysis: `scripts/analyze_phase2_root_causes.py`
- Derived evidence bundle: `outputs/runs/thayer_select_root_cause_analysis_20260711/evidence.json`
- Complete tables: `outputs/runs/thayer_select_root_cause_analysis_20260711/tables/`
- Supporting plots: `outputs/runs/thayer_select_root_cause_analysis_20260711/figures/`

The analysis read only the frozen Promptability, Recoverability, seed-replication, development-scene, calibration, checkpoint, and source-catalog artifacts. It did not read or enumerate future-lockbox samples; did not train or alter a neural model; did not create a checkpoint; and did not regenerate development reconstructions.
