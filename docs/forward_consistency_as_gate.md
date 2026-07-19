# Forward consistency as an evaluation gate

Forward consistency tests whether requested and companion layers plausibly
recompose the observed noisy blend under the frozen source-plus-sky Poisson
contract. It is necessary but not sufficient for scientific source identity.
Thayer-LG showed that its optimized loss term dominated truth-point gradients
and favored source-sum-preserving compromises.

Thayer-SA therefore removes forward reconstruction from the optimized
objective. Global and per-band whitened residuals, source-sum error, relative
flux residual, and flux conservation remain mandatory evaluation diagnostics.
They cannot change source-recovery thresholds or rescue a candidate outside
truth coverage.

The Thayer-SA campaign stopped before neural fitting, so it does not yet show
that forward consistency remains strong after training without a forward loss.
That question remains gated behind a corrected objective that first passes
detached output-space optimization.
