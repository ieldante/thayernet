# D3 Prompt-Collapse Policy

Prompt A orders source A as requested and source B as companion; prompt B
reverses those roles. Collapse is measured in physical mapped output space,
separately per expert, using normalized RMS
`rms(x-y) / max(rms(x), rms(y), 1e-12)`.

The exact tolerance is `1e-7`. Both requested and companion same-slot
distances must be within tolerance for an expert to be collapsed. One collapsed
expert is a nonterminal partial-collapse record. Both collapsed experts for
three consecutive evaluations produce terminal `PROMPT_COLLAPSE`; any other
evaluation resets the streak. A missing prompt or semantic layer is immediate
terminal failure.

Expert permutation is not allowed for the terminal test. Set-level permutation
is diagnostic only. A valid source swap is checked after prompt B is reordered
to canonical source order. The scientific D3 campaign fixes the ordinary-scene
concentration exemption to false. No collapse threshold was tuned from a
scientific trajectory.
