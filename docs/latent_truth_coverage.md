# Latent truth coverage

Truth coverage uses the frozen primary scientific distance. A candidate covers
a named truth only when at least one forward-consistent requested-source sample
has primary normalized distance at most 1.0. Physical inverse-normalized g/r/z
source layers, fixed zero-background semantics, common pixel alignment, and the
unchanged image, flux, color, and centroid thresholds are required.

The metric audit passed exact-truth, floating-tolerance, flux-only, translation-
only, exact-alternate, between-truth, band-order, background, and alignment
cases. The frozen Atlas threshold was not changed.

Under K=32 posterior sampling, coverage was 0/512 ordinary prompt evaluations,
0/500 near-collision own-posterior evaluations, and 0/500 cross-posterior
alternate evaluations. High forward consistency and low normalized pixel MSE on
some near-collision scenes do not imply scientific truth coverage because the
metric is sensitive to relative flux, color, and centroid errors as well as
image norm. Posterior samples remain training/evaluation diagnostics and are not
deployable hypotheses.
