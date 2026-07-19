# Multi-hypothesis source contract

Each Thayer-PU sample is a six-channel zero-background decomposition. Channels
0:3 are the requested g/r/z source contribution and channels 3:6 are the
companion contribution. Their unclipped sum is the candidate noiseless scene.
Inverse normalization uses the frozen training-only g/r/z scales.

For one scene and one matched latent sample, prompt A means `[source A, source
B]` and prompt B means `[source B, source A]`. The scene-level prior depends
only on the observed blend, so the same latent sample can be queried under both
prompts. Target truth is forbidden from prior inference. The training-only
posterior receives truth layers in canonical manifest A/B order, never in
prompt order.

A candidate becomes plausible only after recomposition through the frozen
source-plus-sky Poisson observation contract and calibration-only filtering.
Scientific diversity is measured only among retained candidates. Diversity
that fails forward consistency is not ambiguity. A witness requires at least
two retained requested layers separated beyond the frozen image, flux, color,
or centroid limits. Failure to find a witness does not prove uniqueness.

Candidate hashes use `thayer-per-sample-tensor-sha256-v1`: one CHW sample,
explicit little-endian float32, contiguous C-order bytes, and a header carrying
schema version, shape, dtype, byte order, and dimension order. Batch position,
batch size, device, strides, and storage layout do not enter the digest.
Historical candidate hashes remain unchanged.

