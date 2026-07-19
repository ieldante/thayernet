# Square Decoder Optimization

## Established controls

Under the exact one-scene square mapping, persisted D0 direct-logit and D1
free-penultimate optimization each reached 100% own, alternate, and both-mode
coverage. Persisted D2 final-head-only optimization reached zero coverage in
all three categories. These controls reproduced from their existing tables and
endpoints; none was rerun.

## D3 status

Full-L0 decoder optimization was not started. The prerequisite D1 artifact
retained successful outputs and metrics but omitted both optimized penultimate
tensors required for the frozen evaluation-only reference. Thayer-D3 therefore
constructed no optimizer, ran no autograd trace, updated no decoder layer, and
created no training checkpoint.

There is consequently no evidence about square derivative attenuation during
decoder training, assignment flips, layer-gradient behavior, dead channels,
expert dominance, z-band stagnation, tangent capture, or movement toward a
feasible feature region. The square mapping remains directly navigable in D0
and feature-reachable in D1; its practicality for full existing L0 decoder
training is unresolved.

Neither eight-scene fitting nor a capacity ladder is authorized. The exact next
experiment is a square-only D1 endpoint-persistence replay, not a D3 retry,
optimizer change, assignment change, or capacity increase.
