# D3 Sky-Vector Contract

The scientific sky vector is not an image background estimate to subtract. It
is the per-band additive sky-electron expectation in the frozen Poisson
variance:

`variance = maximum(recomposed_noiseless + sky[:, None, None], 1.0)`

The exact g/r/z values are
`[24114.080000000005, 127057.12000000002, 250784.80000000005]` detected
electrons per pixel. They are finite, nonnegative, rank one, and have no scene
or spatial axis. They are not squared, inverted, or normalized before use.
Residuals are divided by the square root of the resulting variance.

The numeric vector has one frozen authoritative machine-readable source. Its
meaning, order, operation, and source hash are independently confirmed by the
production and reference evaluators. This is reported as a single-source
numeric limitation, not as a double-source numeric match.

