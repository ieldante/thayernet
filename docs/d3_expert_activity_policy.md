# D3 Expert-Activity and Expert-Death Policy

Activity is evaluated at every declared evaluation using only finite-state,
optimizer-membership, norm, learning-rate, and frozen-parameter records.
Frozen parameters are excluded. Numerical zero is `1e-7`.

At positive learning rate, an expert is active only when gradient,
parameter-update, and physical-output-change norms are each greater than
`1e-7`. Zero gradient, nonzero gradient with zero update, and nonzero update
with zero output change are distinct temporary inactivity modes. Active status
resets the inactivity streak. Zero learning rate exempts update and output-
change checks and also resets the streak.

Three consecutive inactive evaluations mark an expert dead. One dead expert is
terminal because both experts are required by the frozen scientific semantics.
Nonfinite expert state and omission from the optimizer are immediate terminal
failures. No source truth or scientific outcome enters this policy.
