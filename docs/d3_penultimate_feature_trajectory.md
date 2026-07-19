# D3 Penultimate Feature Trajectory

Thayer-D3R produced no learned penultimate-feature trajectory. The campaign
stopped during guarded import bootstrap, before the one-step trace, optimizer
construction, or decoder forward pass.

The four authoritative Thayer-D1R endpoint tensors remain complete and
replayable, but they were not used as training targets, initialization, losses,
or checkpoint-selection signals. No statement can be made about D3 motion
toward or away from that endpoint, z-band feature stagnation, or a different
successful endpoint.

Run artifacts contain explicit `not_run` sentinels so absence of trajectory
files cannot be mistaken for missing persistence after a completed fit.

## Thayer-D3A preregistration stop

Thayer-D3A also produced no learned feature trajectory. Its runtime, scientific
container, and checkpoint hashes matched, but preregistration could not freeze
the scientific sky and plausibility-threshold values required by the forward
and truth-coverage gates. The scientific process was not launched, so no
one-step movement, D1 distance, z-band trajectory, or alternative feature
endpoint exists. D1 remained an evaluation-only artifact and was never loaded.
