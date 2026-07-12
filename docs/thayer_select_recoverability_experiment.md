# Thayer-Select Phase II recoverability experiment

The campaign asks whether a compact coordinate-conditioned model can predict
whether its requested-source reconstruction will satisfy a predeclared
scientific contract, and whether calibrated abstention reduces catastrophic
error, hallucination, and source confusion at useful coverage.

Fresh group-safe manifests contain 10,000 training, 1,500 validation, and 2,000
calibration scenes with 55% valid, 15% perturbed-valid, 20% null, and 10%
ambiguous queries. They reuse the immutable CatSim source partitions but not
the earlier development scenes. Null targets are exact zero arrays; ambiguous
rows are masked out of pixel supervision and never receive an arbitrary source
truth. Full manifests preserve seeds, source IDs/groups, geometry, scene
variables, PSF/noise metadata, and array hashes with stratified exact replay.

R0 is a reconstruction-only safety control trained on the mixed query
distribution. R1 retains the same compact prompted backbone and adds a bounded
log-variance map, global contract-success head, and no-source head. Its log
variance is sigmoid-mapped to `[-8, 2]`; training stops on nonfinite values,
fallback, bound violation, manifest/label mismatch, or leakage. R1 uses masked
bounded Gaussian NLL, fixed-weight global and no-source BCE terms, and a small
uncertainty-saturation penalty with an explicit 25% validation saturation stop
threshold. Whole-image MSE retains reconstruction supervision for valid and
null targets; bounded heteroscedastic NLL is restricted to the oracle
requested-source support during training/evaluation loss computation so empty
background cannot force variance to its lower cap. The support is never a
model input. The global head receives encoder features only,
never oracle error or generator difficulty.

The frozen Phase-I C model supplies separate training-only and validation-only
empirical outcome labels under all three contracts. R0 and R1 use matched
manifests, 20 epochs, batch size 8, Adam, cosine scheduling, a fixed primary
seed, and MPS. Best and final checkpoints are separate. After checkpoint and
contract freeze, temperature scaling and isotonic regression are compared by
five-fold calibration-only Brier score. The selected calibrator, score, and
coverage/probability thresholds are frozen before a new 2,000-scene development
manifest is rendered read-only and evaluated exactly once.

Primary reporting includes per-query reconstruction/photometry/centroid and
failure metrics, macro and per-sample tables, calibration, all-query and
valid-only risk–coverage, oracle/random/baseline references, within-severity
uncertainty correlations, no-harm gates, and feasibility-only ambiguity-pair
mining that excludes development and lockbox scenes.

## Frozen outcome

The authoritative run is
`outputs/runs/thayer_select_recoverability_20260711_191518`. R0 and R1 completed
20 MPS epochs with 119,091 and 123,368 parameters. The intended MODERATE
actionable label was only 0.4% positive; the predeclared closest-to-balanced
fallback selected PERMISSIVE at 3.12%, still highly imbalanced. R1 used a
training-only positive weight of 20. Two fail-closed uncertainty stops exposed
whole-image NLL background saturation before the final source-supported NLL
form completed stably; every incident and superseding code hash is preserved.

Isotonic calibration won calibration-only five-fold Brier comparison. AUROC was
0.8746, AUPRC 0.2475, and Brier changed from 0.1010 raw to 0.0456 calibrated.
On the one-time development set, permissive actionable risk fell from 0.9560 at
100% coverage to 0.9537/0.9511/0.9456/0.9379 at 95/90/80/70%. This is weak
selective improvement: catastrophic failure did not decline at useful coverage.
Ambiguous queries were misranked above valid queries and R1 null hallucination
was 8.25%, slightly worse than frozen Phase-I C on identical new null scenes.
The predeclared classification is PARTIAL SUCCESS. No development inference was
rerun after a reporting-only serialization failure. Full pixel-uncertainty maps
were not persisted and were not regenerated because the development pass was
one-time. The lockbox remained completely untouched.
