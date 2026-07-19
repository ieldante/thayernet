# Source-allocation null-space audit

Source-sum preservation alone does not preserve scientific identity. Exact
light-transfer paths kept the requested-plus-companion image fixed and remained
forward-consistent, but crossed the frozen source-truth boundary. Positive
requested-to-companion transfer retained coverage through 15% and lost it at
20%; the reverse direction lost coverage more quickly because relative flux
and color errors were asymmetric at low source flux.

The audit did not find a true local flat source-allocation null space under the
complete objective. Float64 Hessian-vector products gave positive curvature in
all audited directions. Source-light exchange was consistently the weakest
direction, with median curvature approximately 1.16e-4 on ordinary rows and
1.26e-4 on ambiguous rows. The preregistered float32 finite differences showed
cancellation near this scale and are retained as numerical evidence; the
float64 HVP table is the authoritative curvature supplement.

The important failure is therefore not zero curvature. It is a cheap,
scientifically destructive direction relative to the forward-to-observed
gradient and the narrow coverage boundary. A decomposition can preserve the
source sum, remain forward-consistent, and still assign the light to the wrong
scientific source.
