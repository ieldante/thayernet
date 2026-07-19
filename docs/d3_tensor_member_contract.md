# D3 Tensor Member Contract

Thayer-D3E validated scientific containers by reading only container metadata
and tensor headers. It did not deserialize scientific array or storage values.

| Container | Required members | Header contract |
| --- | ---: | --- |
| P0 target set | 11 | exact names, shapes, dtypes, endianness, roles, hashes |
| D1 endpoint | 4 | four prompt/expert penultimate tensors, each `[16,60,60]` float32 little-endian |
| Cached encoder features | 6 | prompt A/B `enc1`, `enc2`, and `bottleneck` tensors |
| Initial decoder state | 36 | 18 float32 tensors for each of two experts |

Cached-feature shapes are `[1,16,60,60]` for `enc1`, `[1,32,30,30]` for
`enc2`, and `[1,64,15,15]` for `bottleneck`, independently for both prompts.
The initial-state contract fixes every state-dict key, shape, dtype, member
hash, and expert-level canonical state hash. Every expected member was present,
no unexpected member was accepted, and all validation rows passed.

The capsule binds each scientific container by exact repository-relative path,
byte size, file SHA-256, member inventory, semantic role, member shape, dtype,
endianness, and canonical member hash. Corrupting or removing a required member
causes the actual consumer to stop before model execution.
