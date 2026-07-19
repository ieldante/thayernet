# Scientific basin geometry

Thayer-SA established that the corrected surrogate tracks the frozen metric
and gives exact truth zero loss and zero gradient. Thayer-OC shows that those
properties do not create a broad practical basin under the tested global
optimizers.

At the persisted Thayer-SA compromise, the raw gradient L2 was 2.193. The
common-mode and allocation-mode projections were 1.286 and 1.777, respectively,
so allocation gradients were not weaker; the common/allocation ratio was
0.724. The mean hard-assignment margin was 4.829, although some prompt/row
margins were exactly tied. These observations do not support a simple global
weak-allocation explanation.

Coverage entry depended strongly on initialization. Raw L-BFGS brought the
persisted Thayer-ME output to 0.844 own, 0.875 alternate, and 0.750 both-mode
coverage but only 0.125 ordinary coverage. No method cleared all four 90%
requirements across every frozen non-truth start. The best ordinary and best
both-mode endpoints came from different method/initialization combinations,
which is not a deployable global rule.

The local curvature magnitude is unresolved. The actual-objective HVP values
were nonfinite at both audited configurations, and the frozen float32 central
difference quantized to zero at the compromise. No dense Hessian was formed,
and no finite condition-number claim is made.

The primary scientific category is `SCIENTIFIC-BASIN EXTREMITY`, with partial
optimization success but no passing method. The one next experiment is direct
feasibility learning or projection into the unchanged frozen scientific
region, not threshold relaxation or per-scene optimizer selection.
