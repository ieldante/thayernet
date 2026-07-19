# Prior and posterior gap

Thayer-PU separates inference-time prior evidence from training-only posterior
diagnostics. The prior is `p(z | observed blend)` and cannot accept prompts or
truth layers. The posterior is `q(z | observed blend, source A, source B)` and
is used only during fitting and diagnostic comparison.

On the fresh Atlas-excluded validation set, posterior requested-source MSE was
0.3481 in normalized units. Prior best-of-16 MSE was 0.3183, a ratio of 0.9143
against the frozen maximum of 2.0. Posterior requested-identity success was
0.99375; individual prior-sample identity was 0.99384. The prior/posterior gap
gate therefore passed.

This does not establish posterior calibration. On Atlas v0, no retained prior
sample reached either the own requested truth or paired alternate truth under
the frozen scientific-distance criterion. Posterior samples were not used to
rescue that result, select samples, or tune the prior after Atlas access.

