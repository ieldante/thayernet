# Cached Encoder Feature Contract

The one-scene fixed-feature campaigns use the immutable joined prompt-A/B cache
from the repository-integrity run. Prompt A and prompt B each contain three
finite float32 levels with shapes `1x16x60x60`, `1x32x30x30`, and
`1x64x15x15`. Their six canonical hashes must match the frozen inventory and
the prompt views must remain distinct.

Using the cache means that no executable encoder graph or encoder parameter is
needed for D1 or D3 optimization. The encoder may not be recomputed, and cache
hashes must be compared before and after every campaign that consumes it.

Thayer-D1R verified the exact cache and its encoder hash but optimized only
detached penultimate tensors. The cache remained unchanged. Ordinary,
eight-scene, remaining-microset, Atlas, development, and lockbox inputs are not
part of this contract.
