# Square physical source head

The audited mapping is `physical = raw ** 2` inside the model forward pass.
Its final convolution weights were initialized to zero and its final bias to
the square root of the frozen positive epsilon, giving the same initial
physical tensor as ReLU and absolute value. Its frozen implementation SHA-256
is `a47c322ffa3fda58a84a45c0a15891f60cef2455215ec99a229c6200f8edf1ae`.

Square represented all 256 frozen P0 target entries. Float32 square-root then
square round-trip reached a maximum error of 0.00390625 detected electrons,
exactly the frozen physical tolerance. The mapping has sign symmetry and
derivative `2 * raw`; its exact-zero derivative is zero and its gradient is
small near zero, but the preflight found usable gradients throughout material
positive target support. The final real-fit stagnation fraction was
1.15741e-5 in both one-scene conditions.

All five synthetic MPS fits passed with finite gradients and zero physical
negatives. In the frozen L0 fits, final ordinary coverage was 0% despite an
ordinary expert diameter of 0.9160. Ambiguous own, alternate, and both-mode
coverage were all 0% at the frozen final endpoint. Prompt swap and
forward-consistency evaluation passed, but those metrics do not establish
truth-mode recovery. Square is not selected, and it was not advanced to
eight-scene fitting.
