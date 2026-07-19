# Thayer-Audit Family-E v0

The authoritative run is
`outputs/runs/thayer_family_e_v0_20260714_195256/`. The preregistration
SHA-256 is
`256bffe3bc53b572b7596bba844f0afdbf4abf3c4cb1d8906fc0ad08663d8881`.
The authoritative outcome is **DATA_OR_IMPLEMENTATION_FAILURE**.

Condition C produced 0 safe / 12,493 unsafe outputs. Thayer-PU after its
fixed-batch repair produced 0 safe / 7,591 unsafe outputs across training,
validation, and calibration. Binary POST-auditor learning therefore remains
scientifically impossible, and no auditor was trained here.

Family-E froze one nine-logit per-pixel, per-band requested/companion/residual
softmax allocation. Multiplying those fractions by the raw observed tensor
would conserve it exactly. The construction passed synthetic MPS
nonnegativity, conservation, finite-gradient, low-flux, and zero-source tests
when the observation was nonnegative.

The authoritative BTK observations instead use zero-background semantics and
preserve signed noise. Negative observed fractions were 0.486877 training,
0.481794 validation, and 0.482363 calibration. At least 9,999/10,000 training
episodes and every validation/calibration episode contained a negative
observed pixel; every episode also contained pixels where the two nonnegative
target sources exceeded the observed value.

Thus three nonnegative contributions cannot sum to the signed observation, and
the frozen targets are not representable by this simplex construction. The
Part-E stop rule fired before architecture construction, objective testing,
micro-overfit, training, checkpoints, inference, OOF outputs, replay, safety
labels, family comparison, bootstrap, or auditor training. Physical compliance
was not treated as scientific success.

Development, Atlas selection, and final-lockbox access remained zero.
Condition C, Thayer-PU, historical checkpoints, thresholds, prompt/source-layer
semantics, and README were not modified.

Thayer-Audit v1 remains unauthorized. The single recommended next experiment
is a separately preregistered, training-free signed-noise-residual
physical-contract preflight: keep requested and companion source layers
nonnegative, permit the residual/noise term to be signed, and require their sum
to equal the zero-background observation exactly.

## Signed-noise-residual preflight follow-up

The separately preregistered training-free correction passed. Nonnegative
requested/companion layers with a signed algebraic residual represented all
10,000/2,000/2,000 frozen targets and conserved the observation within frozen
float32/float64 tolerances. No model or labels were produced. Exactly one
separate Family-E1 model-eligibility campaign is now authorized; Thayer-Audit
v1 remains unauthorized.

## Family-E1 continuation result

The signed-residual correction passed its physical preflight, but the ensuing
Family-E1 model campaign stopped at prompt identity before full training.
Ordinary one-scene micro-overfit passed; difficult and mixed-eight identity
failed at `0.50` and `0.5625`. The original all-nonnegative-simplex failure
remains authoritative for Family-E v0, while Family-E1 establishes only that
the corrected physical space and objective are valid—not that the frozen
network is eligible.
