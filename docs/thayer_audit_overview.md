# Thayer-Audit overview

Thayer-Audit is the proposed independent black-box layer for identifying unsafe
candidate reconstructions. It may use the observed blend, coordinate prompt,
candidate requested source, candidate full decomposition, forward residual,
forward-consistency score, candidate measurements, plausible-set diameter, and
legitimate observational metadata.

It may not receive target truth, true error, source identity, simulator
difficulty, true SNR or obstruction, deblender family/checkpoint/path/
architecture identity, private activations, gradients, or training loss.
Failure heads retain positive, negative, and not-applicable semantics.

## Current status

No Thayer-Audit model, tensor, calibrator, threshold, or catalog policy was
trained in the competing-hypothesis campaign. The preregistered cross-family
gate was unattainable because Condition C, R0, and reconstruction-only R1 share
one compact prompted-U-Net family cluster. Training an auditor on these
controls would not justify a model-agnostic claim.

The campaign instead validated the upstream prerequisites: a frozen
forward-consistency score, a finite ambiguity-witness definition, a 25-pair
Atlas, and explicit same-cluster candidate limitations. The auditor itself can
fail and remains prospective.

Atlas v0 confirms that the auditor can fail before training: candidate-set
diameter did not beat confidence or forward residual on the frozen
Atlas-versus-control comparison. Thayer-Audit remains untrained and blocked.
