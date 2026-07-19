# Absolute-value physical source head

The audited mapping is `physical = abs(raw)` inside the model forward pass.
Its final convolution weights were initialized to zero and its final bias to
the frozen positive epsilon, so its initial physical tensor matched the ReLU
and square conditions exactly. Its frozen implementation SHA-256 is
`a47c322ffa3fda58a84a45c0a15891f60cef2455215ec99a229c6200f8edf1ae`.

Absolute value represented all 256 frozen P0 target entries with zero
round-trip error. Its derivative has unit magnitude away from zero, while the
framework supplies a finite zero subgradient at the cusp. A small negative raw
perturbation reverses the local sign of the derivative rather than creating a
negative physical value. The preflight found usable gradients throughout
material target support; final cusp activation was 8.10185e-5 in both real
one-scene conditions.

All five synthetic MPS fits passed with finite gradients and zero physical
negatives. In the frozen L0 fits, final ordinary coverage was 0% despite an
ordinary expert diameter of 0.8475. Ambiguous own, alternate, and both-mode
coverage were all 0%. Prompt swap and forward-consistency evaluation passed,
but those metrics do not establish truth-mode recovery. Absolute value is not
selected, and it was not advanced to eight-scene fitting.
