# Differentiable scientific distance

The Thayer-SA training-only surrogate operates on inverse-normalized physical
g/r/z source layers. It mirrors the frozen requested-source metric with image
distance divided by 0.25, each band flux error divided by 0.20, physical
flux-derived g-r and r-z errors divided by 0.20 mag, and nonnegative soft-
centroid displacement divided by 0.5 mean-PSF FWHM. Frozen image, flux,
positivity, and centroid floors remain in force. Display RGB is never used.

Seven normalized components are combined with a zero-anchored log-mean-exp
smooth maximum at temperature 0.005. Exact truth has zero surrogate value and
zero gradient. On the frozen canonical outputs, the surrogate achieved
Spearman 0.990679, Kendall 0.957683, and 100% threshold-side agreement with the
exact nondifferentiable metric. Flux, color, translation, and morphology unit
perturbations activated their intended components.

Passing this alignment audit did not establish favorable optimization
geometry. The detached output-space gate failed, so the surrogate is not yet a
validated training objective.
