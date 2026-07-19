# Thayer-MH multiple-hypothesis decoder

Thayer-MH is a 120,022-parameter coordinate-conditioned K=2 set decoder. It
warm-starts Condition C, shares every decoder weight between hypotheses, and
uses two learned eight-dimensional tokens. Each output is a six-channel
requested g/r/z plus companion g/r/z decomposition.

The authoritative run is
`outputs/runs/thayer_multiple_hypotheses_20260712_190701/`. It excluded 36,288
groups appearing in Atlas pairs, feasibility pairs, controls, or the historical
Atlas candidate pool. A fresh search produced 1,500/250/250 approved
training/validation/calibration near-collision pairs, and all 19,000 final
scenes replayed exactly.

Promptability passed after 30 MPS epochs: both token-specific and set-level
prompt-swap success were 0.992, and requested reconstruction MSE was 0.864 times
Condition C. The representation gate failed. Ordinary own-truth, near-own,
paired alternate-truth, and both-mode coverage were all zero, despite forward-
consistent fractions of 0.933 and 1.000. Atlas was not evaluated and the final
lockbox remains untouched.

Multiple-hypothesis prediction is established prior work. The experiment's
specific contribution is its ambiguity-set source-decomposition contract and
fail-closed astronomical benchmark boundary. Outputs are not a complete
posterior, and absence of a covered second hypothesis does not prove uniqueness.

## Independent-expert follow-up

Thayer-ME replaced the shared token decoder with two independent compact
decoders while retaining the same encoder, targets, and permutation-invariant
contract. It failed the training-only micro-overfit gate: prompt swap and
forward consistency passed, but ordinary, own, alternate, and both-mode truth
coverage remained zero. Full training and Atlas evaluation were not authorized.

## Frozen loss-geometry follow-up

Thayer-LG verified that exact approved truths are representable and pass the
frozen coverage metric, but the inherited objective is not truth-optimal on the
isolated microset. Compromise configurations beat truth on all ambiguous rows,
and full-objective optimization from exact truth left coverage while lowering
loss. This rejects a shared-decoder-only explanation and identifies objective
geometry as a direct contributor.
