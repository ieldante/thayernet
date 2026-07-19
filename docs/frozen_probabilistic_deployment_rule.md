# Frozen probabilistic deployment rule

Thayer-PU Eligibility v1 froze one rule before inference: K=16 explicit
standard-normal epsilon samples in seed order 2026077600–2026077615, decoded
through the truth-free prior, requested channels 0:3 averaged on CPU in
float64, and one final float32 cast. Neural inference was MPS-only at batch size
8. No truth, posterior encoder, Atlas outcome, safety label, best-of-K choice,
medoid comparison, clipping, or physical correction entered deployment.

The rule passed prompt identity, repeated batch-8 replay, and exact batch-4
versus batch-8 canonical hashes. It failed the separately frozen requirement
that a scene executed alone equal the same scene in a batch: all 24 preflight
scenes changed candidate and deployed hashes. The deployment rule therefore
cannot be used for complete audit inference under this contract. No alternate
rule or tolerance was selected after observing the failure.

