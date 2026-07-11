# DR10 direct source-extraction decision

## Authoritative decision — 2026-07-11

DR10 FITS acquisition succeeded, but the direct isolated-source extraction
contract failed. The acquisition produced 60/60 valid observed/model/residual
FITS products and 20/20 valid catalog boxes. All 20 scene triplets are aligned
in shape, g,r,z ordering, WCS, 0.262 arcsec/pixel scale, and linear
nanomaggies-per-coadd-pixel semantics. Numerical closure passes 15/20 scenes.

Those successes do not establish isolated-source truth. The whole Tractor
model is a model of the full scene, not a central-source stamp. The Tractor
residual is not a source-only noise layer: it may contain coherent fit error,
omitted morphology, and unresolved source attribution. A segmented observed
source retains its original coadd noise and background. A segmented model is
an inseparable piece of an already PSF-convolved summed scene, and a
model-assisted observed proxy retains coadd noise and fit error. None of the
60 tested option/source combinations satisfies the intended one-source,
one-noise-realization, PSF-consistent supervised-training contract.

Direct supervised blending from DR10 observed/model/residual products therefore
remains **blocked**. The rejected route must not be reopened without new source
products that supply official per-source forward renders, exact local PSFs,
and a valid one-observation-noise construction. DR10 must not be represented as
isolated-source supervised truth.

DR10 remains valuable as an unlabelled real-sky distribution-shift set for
output stability, hallucination, false large-source production, artifact
sensitivity, qualitative failure analysis, and abstention behavior. It cannot
support supervised reconstruction-accuracy claims without isolated truth.

## Versioned evidence

The immutable authoritative run is
`outputs/runs/dr10_model_probe_20260711_160018/`.

- v1 is preserved but superseded where later corrected audits exist:
  `tables/scene_triplet_alignment.csv`, `tables/scene_triplet_closure.csv`,
  `tables/manual_morphology_review.csv`, `tables/residual_morphology_metrics.csv`,
  `tables/radial_profiles.csv`, `tables/psf_audit.csv`,
  `tables/scene_component_audit.csv`, and
  `tables/source_extraction_options.csv`.
- v2 is authoritative for alignment, closure, scene components, and radial
  profiles, but superseded for the final morphology, PSF, and extraction
  conclusions where a v3/v4 artifact exists:
  `tables/scene_triplet_alignment_v2.csv`,
  `tables/scene_triplet_closure_v2.csv`,
  `tables/scene_component_audit_v2.csv`, `tables/radial_profiles_v2.csv`,
  `tables/manual_morphology_review_v2.csv`,
  `tables/residual_morphology_metrics_v2.csv`, `tables/psf_audit_v2.csv`, and
  `tables/source_extraction_options_v2.csv`.
- v3 is authoritative for manual morphology, residual morphology, PSF, and
  source-extraction options: `tables/manual_morphology_review_v3.csv`,
  `tables/residual_morphology_metrics_v3.csv`,
  `tables/morphology_inference_status_v4.csv`, `tables/psf_audit_v3.csv`, and
  `tables/source_extraction_options_v3.csv`.
- The binding machine-readable decision is `tables/decision_gate.csv`; the
  narrative authority is `reports/final_report.md`.

Historical reports, figures, and tables remain unchanged. This decision adds a
stable repository-level pointer to them and does not authorize another DR10
extraction experiment.
