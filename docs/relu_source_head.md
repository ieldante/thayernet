# ReLU physical source head

The audited mapping is `physical = relu(raw)` inside the model forward pass.
Its final convolution weights were initialized to zero and its final bias to
the frozen positive epsilon, so its initial physical tensor matched the square
and absolute-value conditions exactly. Its frozen implementation SHA-256 is
`a47c322ffa3fda58a84a45c0a15891f60cef2455215ec99a229c6200f8edf1ae`.

ReLU represented all 256 frozen P0 target entries with zero round-trip error
and passed every special-case witness. Its derivative is one on positive raw
support and zero at exactly zero under the framework subgradient. A negative
raw perturbation is mapped to zero, so the nonpositive raw half-line is a dead
region. This risk was material in the real fits: the final zero-gradient or
stagnation fraction was 95.95% for the ordinary scene and 92.65% for the
ambiguous scene.

All five synthetic MPS fits passed with finite gradients and zero physical
negatives. In the frozen L0 fits, however, final ordinary coverage was 0% with
expert diameter 10.0000, and ambiguous own, alternate, and both-mode coverage
were all 0%. Prompt swap and forward-consistency evaluation passed, but those
metrics do not replace supervised truth-mode coverage. ReLU is not selected,
and it was not advanced to eight-scene fitting.
