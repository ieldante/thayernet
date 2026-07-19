# Micro-overfit capacity gate

Before full Thayer-ME training, a frozen training-only microset selected 32
ordinary observations and both members of 16 approved training ambiguity pairs.
No validation, calibration, Atlas, development, or lockbox row entered the
microset. Both prompts and full six-channel decompositions were retained.

The preregistered gate required at least 0.90 ordinary both-expert truth
coverage, ambiguous own coverage, alternate coverage, both-mode coverage,
expert and set prompt swap, and ordinary and ambiguous forward consistency.
Median ordinary expert diameter had to be at most 1.0 frozen scientific unit.

After 400 MPS epochs, expert prompt-swap rates were 0.969 each, set prompt swap
was 0.953, and ordinary and ambiguous forward consistency were 0.969 and 1.000.
Ordinary, ambiguous own, alternate, and both-mode coverage were all zero.
Median ordinary diameter was 5.166. The gate therefore failed and full training
was not run.

Low normalized reconstruction loss, correct prompt behavior, and valid
observation recomposition did not imply proximity under the frozen scientific
image, flux, color, and centroid thresholds. The next experiment must audit
that loss-to-coverage geometry before any model-capacity change.
