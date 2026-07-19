# Scientific-region projection contract

The frozen region is defined in physical detected-electron source layers with
requested g/r/z followed by companion g/r/z, exact 60 x 60 dimensions, fixed
band order, zero background, finite values, and nonnegative layers. Requested
sources must satisfy componentwise image/0.25, per-band flux/0.20, applicable
color/0.20 mag, and centroid/(0.5 mean-PSF-FWHM) ratios at or below 1.0.

P0 follows `X(alpha)=(1-alpha) candidate+alpha exact_truth` on 1,025 fixed
alpha points, finds the earliest feasible interval, refines its boundary with
40 deterministic bisection steps, and enters the separate 0.95 training
interior. The 0.95 value is target-construction slack; scientific evaluation
remains at 1.0. Forward consistency, source-sum error, and prompt swap are
evaluation-only.

Ordinary rows retain two projected expert representatives of the same approved
truth region. Ambiguous rows preserve the canonical unordered own/alternate
set under the unchanged identity/swap assignment. Canonical hashes cover each
physical six-channel target. Projection optimizers contain no neural
parameter, and target provenance is stored outside inference tensors.
