# Frozen loss-geometry audit

Thayer-LG is the training-free audit of the exact objective used by the
Thayer-ME micro-overfit experiment. The authoritative run is
`outputs/runs/thayer_loss_geometry_20260712_205733/`. Its preregistration was
hashed before any per-scene loss inspection. No model inference, neural-weight
update, Atlas evaluation, development evaluation, or lockbox access occurred.

The persisted Thayer-ME failure reproduced exactly. All exact six-channel
truths were representable, prompt-mapped correctly, forward-plausible, and
covered by the frozen scientific metric. Output contract and coverage-metric
defects were therefore rejected.

The objective itself was not truth-optimal. An approved truth was the
scene-level objective optimum on only 15.625% of the 64 microset rows, and a
preregistered compromise beat truth on 84.375%. This occurred on every
ambiguous row. Detached optimization from exact truth lowered the full
objective from 0.029377 to 0.029000 while ordinary coverage fell from 1.0 to
0.03125 and ambiguous both-mode coverage fell from 1.0 to zero.

The primary diagnosis is `MIXED CAUSE`, with directly supported secondary
categories `OBJECTIVE MISALIGNMENT`, `LOSS-SCALE DOMINANCE`, `GRADIENT
CONFLICT`, `PERMUTATION-MATCHING PATHOLOGY`, and descriptive
`SCIENTIFIC-THRESHOLD EXTREMITY`. The optimized normalized forward-to-observed
term supplies the only nonzero truth-point gradient in most configurations and
pulls away from exact source targets. Hard assignment is additionally tied and
perturbation-sensitive at collapsed means. Float64 Hessian-vector products did
not support a global flat source-allocation null space; source-light exchange
was the weakest audited direction but had positive curvature.

The one recommended future experiment is a prospective micro-overfit-only
Thayer-ME rerun on the same 64 rows using source-set reconstruction plus
ordinary concentration and a preregistered differentiable surrogate of the
unchanged scientific distance, while retaining forward consistency only as an
evaluation gate. That experiment was not run here.

## Thayer-SA follow-up

The prospective Thayer-SA run reproduced this diagnosis and produced a
surrogate with strong exact-metric alignment, but its preregistered detached
output optimizer did not reliably enter truth coverage from compromise or
random starts. The campaign stopped before assignment auditing or neural
fitting. This preserves the Thayer-LG conclusion while narrowing the next
barrier to corrected-objective optimization geometry.

## Thayer-OC follow-up

The preregistered conditioning audit preserved the corrected objective and
found partial, initialization-dependent coverage gains but no globally passing
method. Allocation gradients were stronger than common-mode gradients at the
persisted compromise, rejecting a simple weak-allocation explanation. Raw
L-BFGS improved selected starts; Adam-based physical T/D methods failed truth
stationarity. The practical basin remains too narrow for neural authorization.
The modal condition number is unresolved because actual-objective HVPs were
nonfinite under the frozen diagnostic.
