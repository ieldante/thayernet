# Thayer-PU probabilistic hypothesis generator

Thayer-PU is a compact prompted probabilistic U-Net tested as a stochastic
candidate family for Thayer-Select. The authoritative run is
`outputs/runs/thayer_probabilistic_unet_20260712_163340/`; its preregistration
hash is `eb62db24da7c77f35f56d1187f561f88a2e63e2acd89c01c859c1fd2213b2b09`.

The model has 170,278 parameters and an eight-dimensional scene-level latent.
It warm-starts the frozen Condition-C encoder and decoder, keeps the earliest
prompt-sensitive blocks frozen, and adds separate conditional prior and
training-only posterior networks. The prior receives only the observed g/r/z
blend. The posterior may receive the two true source layers during training,
but posterior samples are not inference-time evidence.

All 59 Atlas-related source groups were excluded from training, validation,
calibration, and non-Atlas near-collision construction. The campaign built
20,000 replayable two-source scenes supporting both prompts. Training completed
30 MPS epochs without fallback; epoch 27 was selected by the frozen validation
objective. Canonical per-sample hashing, promptability, latent use, the
prior/posterior gap, forward consistency, and selective control concentration
all passed before Atlas access.

The one-time Atlas result is a partial success. Model-generated witnesses rose
from 19/50 to 24/50. Candidate-diameter AUROC rose from 0.4712 to 0.856 with a
pair-cluster bootstrap 95% interval of 0.751–0.942, and recall at the frozen 4%
control false-positive rate rose from zero to 0.32. Safe-control false witnesses
were 0.08.

The sampled prior did not cover the actual requested truth or the paired
alternate truth on any Atlas observation under the frozen scientific-distance
criterion. The witness target of 30/50 also failed. Thayer-PU is therefore
retained as a stochastic candidate family, not promoted as a calibrated
posterior, production deblender, auditor, or catalog policy.

Variational and probabilistic galaxy deblending are established prior work.
The contribution tested here is not the VAE architecture itself; it is the
combination of prompt-specific full decompositions, truth-free prior sampling,
forward filtering, protected source partitions, and one-pass evaluation on a
frozen empirical ambiguity benchmark.

## Posterior/decoder truth-coverage follow-up

The prospective Thayer-PF campaign stopped before flow implementation. Under
the unchanged coverage metric, K=32 posterior samples achieved 0% own-truth
coverage on 256 ordinary scenes and 250 non-Atlas near-collision pairs; cross-
decoded paired alternate coverage was also 0%. Forward consistency remained
high, so it does not establish truth representability. The evidence does not
support a prior-only correction. Thayer-PU's authoritative Atlas result remains
unchanged and no new Atlas inference was run.

## Ambiguity-set decoder follow-up

Thayer-MH trained a deterministic K=2 shared decoder on approved non-Atlas
target sets. It retained 0.992 prompt-swap success and improved requested MSE
relative to Condition C, but ordinary, near-own, near-alternate, and both-mode
scientific coverage were all zero. Prompt identity and forward consistency
again did not establish truth-mode representation. Atlas was not reopened.
