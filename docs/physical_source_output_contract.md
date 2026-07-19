# Physical source output contract

Physical requested and companion g/r/z source layers must be finite,
nonnegative detected-electron contributions in the exact six-channel order on
the common 60 x 60 grid with zero-background semantics. Frozen inverse
normalization multiplies normalized channels by positive training-only band
scales; it cannot repair a negative value.

Thayer-FP used an unconstrained linear six-channel head. Its raw normalized
output, post-head output, and pre-scale output were identical. Negative values
therefore entered at the decoder head and remained negative after inverse
normalization. The first persisted violation was the end-of-epoch-1
evaluation, with normalized minimum -2.050540 and negative fraction 0.365785.
The historical loop did not persist batch-level output checks, so the first
violating batch is unresolved. At the stored best epoch, the physical minimum
was -842.390 detected electrons.

The frozen P0 targets are finite and nonnegative, and their stored normalized
and physical tensors round-trip exactly under the positive scale mapping. That
validates the target domain but does not define a neural output head. The
Thayer-OC clamp is a detached audit projection and must not be reinterpreted as
a frozen model activation or used as post-hoc coverage clipping.

Thayer-CL found no unique frozen compliant mapping. Any future mapping must be
selected prospectively in a separate preregistration, used identically across
all later capacity conditions, represent zero and the full target range, and
enforce synchronous stopping on any physical contract violation.

## Thayer-OP fixed-L0 result

Thayer-OP prospectively tested ReLU, square, and absolute value as in-forward
physical mappings under identical L0 capacity. The exact mapped tensor was
used by training loss, hard assignment, truth coverage, prompt swap, forward
consistency, source sums, hashes, and saved outputs. No evaluation-only
clipping or second physical path was used.

All three mappings represented every P0 target and produced zero physical
negative values in synthetic and real fitting. None passed the frozen final
ordinary or ambiguous one-scene scientific gate, so no mapping was selected
and the physical source-head contract remains unresolved. The capacity ladder
is not authorized.

## Integrity-audited one-scene clarification

Independent references and differential truth injection confirmed that loss
and evaluation consume the same mapped physical output, normalization is
inverted exactly once, assignment is per sample and permutation-correct, and
requested/companion and band order are correct. Square can reach both approved
truth modes when raw logits or penultimate tensors are free, while its frozen
penultimate final-head-only readout cannot. The physical contract is therefore
implemented consistently on the audited path, but the usable neural readout
remains unresolved and no mapping is promoted.
