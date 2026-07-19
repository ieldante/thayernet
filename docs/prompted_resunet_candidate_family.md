# Prompted ResUNet candidate-family experiment

The prospective run
`outputs/runs/thayer_prompted_resunet_diversity_20260712_154122/` tested one
freshly initialized 199,219-parameter residual encoder-decoder under the frozen
g/r/z plus Gaussian-prompt source-layer contract. All 59 source groups appearing
in the frozen Atlas or targeted feasibility pairs were excluded from its 10,000
training and 1,500 validation scenes. All 11,500 scene definitions replayed
exactly. Training used MPS for 20 epochs; historical development and lockbox
access remained zero.

The model failed the mandatory pre-Atlas promptability gate. Prompt-swap success
was 39.47% against an 80% minimum, and individual requested-source success was
69.5% against a 75% minimum. Whole-image MSE was 1.1205 times Condition C on the
same validation queries, while source-region MSE was worse. Output collapse was
only 0.067%, so the failure is not a constant-output explanation.

The frozen Atlas was not evaluated. The experiment therefore does not establish
a second scientifically useful candidate family, added ambiguity witnesses, or
cross-family generalization. It does not authorize an auditor or catalog policy.

