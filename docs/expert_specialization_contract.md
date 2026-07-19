# Expert specialization contract

Thayer-ME has one shared prompt-sensitive encoder and two compact expert
decoders. The experts receive the same encoder and skip features. They do not
share decoder convolutions, normalization parameters, output heads, or late
source-assignment blocks. No expert identifier, source identifier, pair
identifier, morphology label, simulator field, or target truth enters the
observed input.

Each expert emits six unclipped normalized channels: requested g/r/z followed
by companion g/r/z. Expert identity has no global scientific meaning. On
two-target scenes, the lower-cost of the two expert-to-target assignments is
used per scene. On ordinary scenes, both experts receive the same approved
target and an explicit concentration penalty. Generic diversity is forbidden.

The specialization audit records assignment frequency, assignment entropy,
output and parameter distance, gradient norms, ordinary concentration,
ambiguous separation, prompt identity, forward consistency, and scientific
mode coverage. Unequal global assignment frequency is allowed; dead experts,
flux-scale-only separation, uncontrolled ordinary splitting, and disagreement
without approved-mode coverage are failures.

The micro-overfit result did not establish valid specialization. Both experts
remained active and parameter-distinct, but their output separation did not
cover approved truths and ordinary outputs did not concentrate. Forward
consistency and prompt identity alone were insufficient.
