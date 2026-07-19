# Output-space conditioning audit

Thayer-OC is the preregistered, training-free conditioning audit in
`outputs/runs/thayer_output_conditioning_20260712_225459/`. Its preregistration
SHA-256 is
`4202c5ddc9b9733138168b2acc650334e1ef10b002f7799071a3a12bc827e484`.
The freeze preceded every per-scene array load, detached gradient, curvature
calculation, and optimization. All Thayer-SA and Thayer-ME baselines
reproduced, and all 593 historical checkpoints remained byte-identical.

The audit compared the historical raw Adam protocol, projected raw L-BFGS,
total/allocation Adam, total/allocation L-BFGS, alternating total/allocation
updates, and threshold-Jacobian-preconditioned total/allocation updates. Every
condition optimized the unchanged Thayer-SA scalar objective with the unchanged
hard two-permutation assignment. Targets, thresholds, architecture, and
coverage gates did not change.

No method passed all frozen gates. Raw L-BFGS reached 0.125 ordinary, 0.844
ambiguous-own, 0.875 alternate, and 0.750 both-mode coverage from the persisted
Thayer-ME outputs, but it did not generalize across the other fixed
initializations. Historical raw Adam reached its strongest both-mode result,
0.8125, from the persisted Thayer-SA compromise while ordinary coverage was
only 0.28125. T/D Adam materially improved several compromise starts, but T/D
Adam, alternating T/D, and Jacobian-preconditioned T/D moved the exact-truth
control outside full coverage and are ineligible.

The scientific result is **PARTIAL SUCCESS — SCIENTIFIC-BASIN EXTREMITY**:
conditioning materially increased coverage in selected global
method/initialization combinations, but no globally fixed method entered all
required truth regions and the result remained strongly initialization-
dependent. No neural experiment is authorized by a passing conditioning
method.

Strict correctness status is **FAIL** with one diagnostic failure. The
actual-objective Hessian-vector values were nonfinite, and the frozen float32
finite difference at `h=1e-3` quantized to zero at the persisted compromise.
The modal condition number is therefore unresolved, not zero. This limitation
does not change the directly observed optimization trajectories or gate
failures.

Exactly one next experiment is recommended and was not run: a separately
preregistered direct feasibility-learning micro-audit that projects into the
unchanged frozen scientific region. Neural training, Atlas, development, and
lockbox access remained zero.

## Thayer-FP follow-up

Thayer-FP executed that direct projection audit. The globally frozen P0 method
placed every microset target set inside the unchanged scientific region, so
the former scalar-optimization barrier is not a target-feasibility barrier.
The unchanged Thayer-ME still produced zero truth coverage after direct target
learning. The active microset question has therefore moved to capacity,
encoder conditioning, or neural output parameterization. The unresolved
Thayer-OC HVP status remains unchanged.
