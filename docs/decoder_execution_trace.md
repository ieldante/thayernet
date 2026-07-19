# Decoder Execution Trace

The audited forward and optimization path is:

1. Load the exact prompted Condition-C encoder and freeze every encoder parameter.
2. Encode the same blend with prompt A and prompt B in one joined batch.
3. Persist every decoder-consumed feature tensor and canonical per-sample hash.
4. Restore the mapping-specific ambiguous one-scene L0 endpoint.
5. Decode two experts to complete six-channel raw requested/companion logits.
6. Apply ReLU, square, or absolute value inside the differentiable graph.
7. Multiply once by the frozen repeated three-band physical scale.
8. Compute requested-plus-companion costs for both target permutations.
9. Select the lower per-sample permutation and average only after assignment.
10. Feed the identical mapped physical tensor to loss, assignment, coverage,
    prompt, forward-consistency, z-band, and serialization consumers.

One traced AdamW step for every mapping showed nonzero gradients in both experts
and both final heads, updates in both experts, exact membership of all 92,940
decoder parameters, no encoder parameter in the optimizer, and an unchanged
encoder tensor hash. The trace found no hidden detach or duplicate target
reuse.

MPS feature values were not bitwise invariant to a dummy larger batch.
Propagating the discrepancy through all restored decoders bounded the maximum
physical difference at 0.00103759765625 detected electrons, below the frozen
0.00390625 one-ULP tolerance, with identical assignment and coverage decisions.
Every D0-D3 condition therefore used one persisted joined prompt-A/prompt-B
batch-size-two cache.
